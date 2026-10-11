"""Constrained inventory / procurement optimizer (SIMULATION under src/assumptions.py).

Per product choose a shipping mode, a review cycle R and an order-up-to level S.
Step 1: for every (product, mode, R) search S on TRAINING weeks (cost-minimising among levels
        meeting the fill-rate requirement).
Step 2: choose one (mode, R) per product with a mixed-integer linear programme (SciPy / HiGHS)
        minimising total weekly cost subject to an inventory-investment budget and mode capacities.
"""
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from src import inventory_sim as sim_mod
from src.assumptions import MODES, Assumptions

S_GRID_CAP = 6000


class InfeasibleError(RuntimeError):
    pass


def mode_cost_params(mode, modes, A):
    return modes.loc[mode, "lead_weeks"], modes.loc[mode, "late_rate"], A.ship(mode)


def build_sim_cache(prod, modes, demand_train, cycles):
    """Simulate every (product, mode, R) over a grid of S once; costs are applied later.

    Returns {(pid, mode, R): (S_grid, summary dict)}.
    """
    cache = {}
    for pid in prod.index:
        D = np.asarray(demand_train[pid], dtype=float)
        for mode in MODES:
            L = modes.loc[mode, "lead_weeks"]
            for R in cycles:
                s_hi = int(min(S_GRID_CAP, np.ceil(D.max() * (R + 1) * 1.2) + 1))
                S = np.arange(0, s_hi + 1)
                cache[(pid, mode, R)] = (S, sim_mod.summarize(sim_mod.simulate(D, S, R, L), D))
    return cache


def mean_cost(summary, p, ship_pct, late_rate, A):
    """Mean weekly cost for every S of a cached simulation. p = product row."""
    h = A.holding_per_unit_week(p["unit_cost"])
    return (A.ordering_cost * summary["orders"] + h * summary["onhand"]
            + ship_pct * p["unit_cost"] * summary["units"]
            + A.stockout_mult * p["net_price"] * summary["short"]
            + A.late_penalty * late_rate * summary["orders"])


FRONTIER_POINTS = 12


def frontier_indices(cost, invest, feasible, k_max=FRONTIER_POINTS):
    """Indices of the efficient (investment, cost) frontier among feasible order-up-to levels.

    Levels above the cost-minimising one cost more AND need more investment, so they are dominated.
    Walking up from the lowest feasible level, a level is kept only if it is cheaper than all
    lower-investment levels. The result is thinned to at most k_max points (ends always kept).
    """
    cost = np.where(feasible, cost, np.inf)
    k_star = int(np.argmin(cost))
    idx = np.flatnonzero(feasible[:k_star + 1])
    kept, best = [], np.inf
    for i in idx:
        if cost[i] < best - 1e-12:
            kept.append(i)
            best = cost[i]
    if kept[-1] != k_star:
        kept.append(k_star)
    kept = np.array(kept)
    if len(kept) > k_max:
        pick = np.unique(np.round(np.linspace(0, len(kept) - 1, k_max)).astype(int))
        kept = kept[pick]
    return kept


def build_candidates(prod, modes, cache, A, k_max=FRONTIER_POINTS):
    """Efficient order-up-to levels for each (product, mode, R) meeting the fill-rate requirement."""
    rows = []
    for (pid, mode, R), (S, summ) in cache.items():
        if R not in A.cycles:
            continue
        p = prod.loc[pid]
        cost = mean_cost(summ, p, A.ship(mode), modes.loc[mode, "late_rate"], A)
        feasible = summ["fill"] >= A.fill_rate_min - 1e-12
        if not feasible.any():
            continue
        invest = p["unit_cost"] * summ["onhand"]
        for k in frontier_indices(cost, invest, feasible, k_max):
            rows.append({"product_id": pid, "mode": mode, "R": R, "S": int(S[k]), "cost": float(cost[k]),
                         "fill": float(summ["fill"][k]), "invest": float(invest[k]), "load": float(p["mu"])})
    return pd.DataFrame(rows)


def solve_assignment(cands: pd.DataFrame, budget: float, capacity: dict) -> pd.DataFrame:
    """Pick exactly one candidate per product minimising total cost (MILP, SciPy/HiGHS).

    Constraints: sum of investment <= budget; for each mode, sum of load <= capacity[mode].
    """
    c = cands.reset_index(drop=True)
    n = len(c)
    products = list(c["product_id"].unique())
    A_assign = np.zeros((len(products), n))
    for k, pid in enumerate(products):
        A_assign[k, (c["product_id"] == pid).to_numpy()] = 1.0
    cons = [LinearConstraint(A_assign, 1.0, 1.0),
            LinearConstraint(c["invest"].to_numpy()[None, :], -np.inf, budget)]
    for mode, cap in capacity.items():
        mask = (c["mode"] == mode).to_numpy()
        if mask.any():
            cons.append(LinearConstraint((mask * c["load"].to_numpy())[None, :], -np.inf, cap))
    res = milp(c=c["cost"].to_numpy(), constraints=cons, integrality=np.ones(n), bounds=Bounds(0, 1))
    if not res.success:
        raise InfeasibleError(f"No feasible plan: {res.message}")
    return c[res.x > 0.5].reset_index(drop=True)


def baseline_plan(prod, modes, A, kind):
    """B1: weekly, Standard Class, S = mu(1+L) + z sigma sqrt(1+L).  B2: weekly, Standard, S = 2 mu."""
    mode = "Standard Class"
    L = modes.loc[mode, "lead_weeks"]
    if kind == "B1":
        S = prod["mu"] * (1 + L) + A.z_baseline * prod["sigma"] * np.sqrt(1 + L)
    elif kind == "B2":
        S = 2.0 * prod["mu"]
    else:
        raise ValueError(kind)
    return pd.DataFrame({"product_id": prod.index, "mode": mode, "R": 1, "S": np.ceil(S.to_numpy()).astype(int)})


def plan_investment(plan, prod, modes, demand_train):
    """Average inventory value of a plan simulated on the planning (training) weeks."""
    total = 0.0
    for r in plan.itertuples():
        D = demand_train[r.product_id]
        s = sim_mod.simulate(D, [r.S], r.R, modes.loc[r.mode, "lead_weeks"])
        total += prod.loc[r.product_id, "unit_cost"] * s["onhand"].mean()
    return float(total)


def optimize(prod, modes, demand_train, A, cache=None):
    """Full planning step. Returns (plan, info) where info holds budget and capacities."""
    cache = cache or build_sim_cache(prod, modes, demand_train, A.cycles)
    cands = build_candidates(prod, modes, cache, A)
    missing = set(prod.index) - set(cands["product_id"]) if len(cands) else set(prod.index)
    if missing:
        raise InfeasibleError(f"No candidate meets the fill-rate requirement for products: {sorted(missing)}")
    b1 = baseline_plan(prod, modes, A, "B1")
    budget = plan_investment(b1, prod, modes, demand_train) * (1.0 + A.budget_slack)
    total_load = float(prod["mu"].sum())
    capacity = {m: A.capacity(m) * total_load for m in MODES}
    plan = solve_assignment(cands, budget, capacity)
    return plan, {"budget": budget, "capacity": capacity, "n_candidates": len(cands)}


def verify_plan(plan, prod, A, budget, capacity, tol=1e-6) -> dict:
    """Check every constraint of a plan produced by optimize()."""
    one_each = (plan["product_id"].nunique() == len(plan) == len(prod)
                and set(plan["product_id"]) == set(prod.index))
    fill_ok = bool((plan["fill"] >= A.fill_rate_min - tol).all())
    invest = float(plan["invest"].sum())
    load = plan.groupby("mode")["load"].sum().to_dict()
    cap_ok = all(load.get(m, 0.0) <= capacity[m] + tol for m in capacity)
    return {"one_option_per_product": bool(one_each), "fill_rate_ok": fill_ok,
            "budget_ok": invest <= budget + tol, "capacity_ok": bool(cap_ok),
            "invest": invest, "budget": budget, "load_by_mode": load,
            "all_ok": bool(one_each and fill_ok and invest <= budget + tol and cap_ok)}


def evaluate_plan(plan, prod, modes, demand_eval, A) -> dict:
    """Simulate a plan on (out-of-sample) weekly demand with the ACTUAL demand.

    demand_eval: {product_id: array (T,)}. Returns per-week total cost, cost components and
    per-product realised fill rate / cost.
    """
    T = len(next(iter(demand_eval.values())))
    weekly = np.zeros(T)
    comp_tot = {k: np.zeros(T) for k in ["ordering", "holding", "shipping", "stockout", "late"]}
    rows = []
    for r in plan.itertuples():
        D = np.asarray(demand_eval[r.product_id], dtype=float)
        L, late, ship = mode_cost_params(r.mode, modes, A)
        s = sim_mod.simulate(D, [r.S], r.R, L)
        p = prod.loc[r.product_id]
        comp = sim_mod.cost_components(s, p["unit_cost"], p["net_price"], ship, late, A)
        total = sum(v[:, 0] for v in comp.values())
        weekly += total
        for k in comp_tot:
            comp_tot[k] += comp[k][:, 0]
        rows.append({"product_id": r.product_id, "mode": r.mode, "R": r.R, "S": r.S,
                     "mean_weekly_cost": total.mean(),
                     "fill": 1.0 - s["short"].sum() / D.sum() if D.sum() > 0 else 1.0})
    return {"weekly_total": weekly, "components": comp_tot, "per_product": pd.DataFrame(rows)}