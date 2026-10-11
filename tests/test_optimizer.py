import itertools

import numpy as np
import pandas as pd

from src import inputs, inventory_sim as sim_mod, optimizer
from src.assumptions import MODES, Assumptions

A = Assumptions()


def modes_df():
    return pd.DataFrame({"days": [4.0, 4.0, 2.0, 0.5], "late_rate": [0.4, 0.8, 1.0, 0.48]},
                        index=pd.Index(MODES)).assign(lead_weeks=lambda d: d["days"] / 7.0)


def make_prod(n=4, seed=0):
    rng = np.random.default_rng(seed)
    demand = {i: rng.poisson(lam, 60).astype(float) for i, lam in zip(range(10, 10 + n), [60, 25, 8, 6][:n])}
    price = np.array([120.0, 60.0, 40.0, 25.0][:n])
    prod = pd.DataFrame({"price": price, "mean_discount": 0.1}, index=list(demand))
    prod["net_price"] = prod["price"] * 0.9
    prod["unit_cost"] = prod["net_price"] * 0.88
    prod["mu"] = [d.mean() for d in demand.values()]
    prod["sigma"] = [d.std(ddof=1) for d in demand.values()]
    return prod, demand


# ---------------- simulation: hand-computed cases ----------------
def test_simulation_steady_demand_hand_computed():
    s = sim_mod.simulate([4, 4, 4], [10], R=1, L=0.0)
    assert s["orders"].sum() == 2 and s["units"].sum() == 8 and s["short"].sum() == 0
    assert np.allclose(s["onhand"][:, 0], 8.0)


def test_simulation_stockout_is_lost_sales():
    s = sim_mod.simulate([5], [3], R=1, L=0.0)
    assert s["short"][0, 0] == 2 and np.isclose(s["onhand"][0, 0], 1.5)


def test_simulation_partial_week_lead_time():
    s = sim_mod.simulate([4], [10], R=1, L=0.5)
    assert s["short"].sum() == 0 and np.isclose(s["onhand"][0, 0], 8.0)
    # shortage that happens BEFORE the order arrives cannot be recovered
    s2 = sim_mod.simulate([8], [4], R=1, L=0.5)
    assert np.isclose(s2["short"][0, 0], 4.0)


def test_simulation_two_week_cycle_orders_only_every_second_week():
    s = sim_mod.simulate([4, 4, 4], [10], R=2, L=0.0)
    assert s["orders"][:, 0].tolist() == [0, 0, 1]
    assert s["units"].sum() == 8 and s["short"].sum() == 0
    assert np.allclose(s["onhand"][:, 0], [8.0, 4.0, 8.0])


def test_vectorised_simulation_matches_one_by_one_runs():
    rng = np.random.default_rng(1)
    D = rng.poisson(9, 40)
    grid = np.array([3, 9, 20, 40])
    both = sim_mod.simulate(D, grid, R=2, L=0.3)
    for j, S in enumerate(grid):
        one = sim_mod.simulate(D, [S], R=2, L=0.3)
        for k in both:
            assert np.allclose(both[k][:, j], one[k][:, 0])


def test_units_balance_demand_equals_served_plus_short():
    rng = np.random.default_rng(2)
    D = rng.poisson(7, 50).astype(float)
    S = np.array([5, 12, 30])
    s = sim_mod.simulate(D, S, R=1, L=0.4)
    served_total = D.sum() - s["short"].sum(axis=0)
    assert (served_total >= -1e-9).all() and (s["short"] >= -1e-9).all()
    # stock never exceeds S and orders only bring stock up to S
    assert (s["onhand"] <= S + 1e-9).all()


def test_cost_components_match_the_formula():
    s = sim_mod.simulate([4, 4, 4], [10], R=1, L=0.0)
    c = sim_mod.cost_components(s, unit_cost=10.0, net_price=12.0, ship_pct=0.02, late_rate=0.5, A=A)
    assert np.isclose(c["ordering"].sum(), 2 * 25.0)
    assert np.isclose(c["holding"].sum(), 3 * (0.25 / 52 * 10.0) * 8.0)
    assert np.isclose(c["shipping"].sum(), 0.02 * 10.0 * 8)
    assert np.isclose(c["late"].sum(), 20.0 * 0.5 * 2)
    assert c["stockout"].sum() == 0


# ---------------- candidates / frontier ----------------
def test_frontier_keeps_min_feasible_and_cost_optimum_and_is_efficient():
    prod, demand = make_prod(2)
    cache = optimizer.build_sim_cache(prod, modes_df(), demand, A.cycles)
    cands = optimizer.build_candidates(prod, modes_df(), cache, A)
    assert len(cands) > 0 and (cands["fill"] >= A.fill_rate_min - 1e-9).all()
    for (pid, mode, R), g in cands.groupby(["product_id", "mode", "R"]):
        S_grid, summ = cache[(pid, mode, R)]
        cost = optimizer.mean_cost(summ, prod.loc[pid], A.ship(mode), modes_df().loc[mode, "late_rate"], A)
        feas = summ["fill"] >= A.fill_rate_min - 1e-12
        assert np.isclose(g["cost"].min(), np.where(feas, cost, np.inf).min())     # cost optimum kept
        assert g["S"].min() == S_grid[np.flatnonzero(feas)[0]]                      # lowest feasible level kept
        g = g.sort_values("invest")
        assert (np.diff(g["cost"].to_numpy()) < 1e-9).all()                         # more investment -> lower cost


# ---------------- assignment solver vs brute force ----------------
def random_instance(seed):
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(3):
        for mode, R in itertools.product(["Standard Class", "Same Day"], [1, 2]):
            rows.append({"product_id": pid, "mode": mode, "R": R, "S": 1, "cost": float(rng.uniform(10, 100)),
                         "fill": 0.97, "invest": float(rng.uniform(50, 500)), "load": float(rng.uniform(1, 10))})
    return pd.DataFrame(rows)


def brute_force(c, budget, capacity):
    best = np.inf
    groups = [g.index.tolist() for _, g in c.groupby("product_id")]
    for combo in itertools.product(*groups):
        ch = c.loc[list(combo)]
        if ch["invest"].sum() > budget + 1e-9:
            continue
        if any(ch.loc[ch["mode"] == m, "load"].sum() > cap + 1e-9 for m, cap in capacity.items()):
            continue
        best = min(best, ch["cost"].sum())
    return best


def test_milp_matches_brute_force_on_small_random_instances():
    for seed in range(8):
        c = random_instance(seed)
        budget = float(c.groupby("product_id")["invest"].min().sum() * 1.3)
        capacity = {"Standard Class": 1e9, "Same Day": float(c["load"].mean() * 1.2)}
        expected = brute_force(c, budget, capacity)
        if np.isinf(expected):
            continue
        plan = optimizer.solve_assignment(c, budget, capacity)
        assert len(plan) == 3 and plan["product_id"].nunique() == 3
        assert np.isclose(plan["cost"].sum(), expected)
        assert plan["invest"].sum() <= budget + 1e-6
        assert plan.loc[plan["mode"] == "Same Day", "load"].sum() <= capacity["Same Day"] + 1e-6


def test_infeasible_budget_raises():
    c = random_instance(0)
    try:
        optimizer.solve_assignment(c, budget=1.0, capacity={"Standard Class": 1e9})
    except optimizer.InfeasibleError:
        return
    raise AssertionError("expected InfeasibleError")


# ---------------- full planning step ----------------
def test_optimize_returns_a_plan_that_satisfies_every_constraint():
    prod, demand = make_prod(4)
    plan, info = optimizer.optimize(prod, modes_df(), demand, A)
    ver = optimizer.verify_plan(plan, prod, A, info["budget"], info["capacity"])
    assert ver["all_ok"], ver
    b1 = optimizer.baseline_plan(prod, modes_df(), A, "B1")
    assert np.isclose(info["budget"], optimizer.plan_investment(b1, prod, modes_df(), demand))
    # the realised in-sample investment of the chosen plan equals the sum used in the budget row
    assert np.isclose(optimizer.plan_investment(plan, prod, modes_df(), demand), plan["invest"].sum())


def test_verify_plan_detects_each_kind_of_violation():
    prod, demand = make_prod(4)
    plan, info = optimizer.optimize(prod, modes_df(), demand, A)
    bud, cap = info["budget"], info["capacity"]
    bad = plan.copy(); bad["invest"] *= 10
    assert not optimizer.verify_plan(bad, prod, A, bud, cap)["budget_ok"]
    bad = plan.copy(); bad.loc[0, "fill"] = 0.5
    assert not optimizer.verify_plan(bad, prod, A, bud, cap)["fill_rate_ok"]
    bad = plan.copy(); bad["mode"] = "Same Day"; bad["load"] = bad["load"] * 100
    assert not optimizer.verify_plan(bad, prod, A, bud, cap)["capacity_ok"]
    bad = pd.concat([plan, plan.iloc[[0]]], ignore_index=True)
    assert not optimizer.verify_plan(bad, prod, A, bud, cap)["one_option_per_product"]


def test_plan_depends_only_on_training_demand():
    prod, demand = make_prod(3)
    plan1, _ = optimizer.optimize(prod, modes_df(), demand, A)
    plan2, _ = optimizer.optimize(prod, modes_df(), {k: v.copy() for k, v in demand.items()}, A)
    pd.testing.assert_frame_equal(plan1, plan2)
    eval_a = {k: v[:20] * 2 for k, v in demand.items()}      # evaluation demand never enters planning
    out = optimizer.evaluate_plan(plan1, prod, modes_df(), eval_a, A)
    assert len(out["weekly_total"]) == 20 and (out["weekly_total"] >= 0).all()


def test_baselines_follow_their_formulas():
    prod, _ = make_prod(3)
    m = modes_df()
    L = m.loc["Standard Class", "lead_weeks"]
    b1 = optimizer.baseline_plan(prod, m, A, "B1").set_index("product_id")
    b2 = optimizer.baseline_plan(prod, m, A, "B2").set_index("product_id")
    for pid, p in prod.iterrows():
        assert b1.loc[pid, "S"] == int(np.ceil(p["mu"] * (1 + L) + 1.645 * p["sigma"] * np.sqrt(1 + L)))
        assert b2.loc[pid, "S"] == int(np.ceil(2 * p["mu"]))
    assert (b1["mode"] == "Standard Class").all() and (b1["R"] == 1).all()


def test_evaluate_plan_components_add_up_and_fill_is_a_fraction():
    prod, demand = make_prod(3)
    plan = optimizer.baseline_plan(prod, modes_df(), A, "B1")
    out = optimizer.evaluate_plan(plan, prod, modes_df(), demand, A)
    comp_sum = sum(out["components"].values())
    assert np.allclose(comp_sum, out["weekly_total"])
    assert ((out["per_product"]["fill"] >= 0) & (out["per_product"]["fill"] <= 1)).all()


def test_scaling_assumptions_changes_only_the_named_cost():
    B = A.scaled(stockout=2.0)
    assert B.stockout_mult == 2 * A.stockout_mult and B.ordering_cost == A.ordering_cost
    C = A.scaled(shipping=0.5)
    assert np.allclose(C.ship_pct, [x * 0.5 for x in A.ship_pct]) and C.late_penalty == A.late_penalty


# ---------------- inputs from data (train only) ----------------
def test_mode_params_use_training_rows_only_and_compute_late_rate_correctly():
    n = 400
    ship = pd.DataFrame({"week": pd.Timestamp("2015-01-05"), "split": np.where(np.arange(n) < 300, "train", "validation"),
                         "mode": np.tile(list(MODES), n // 4),
                         "real_days": np.tile([4, 4, 2, 0], n // 4), "sched_days": np.tile([4, 2, 1, 0], n // 4)})
    ship2 = ship.copy()
    ship2.loc[ship2["split"] == "validation", "real_days"] = 6       # changing validation lines must not matter
    p1, p2 = inputs.mode_params(ship), inputs.mode_params(ship2)
    pd.testing.assert_frame_equal(p1, p2)
    assert np.isclose(p1.loc["Standard Class", "lead_weeks"], 4 / 7)
    assert np.isclose(p1.loc["Standard Class", "late_rate"], 0.0)    # real 4 vs scheduled 4: on time
    assert np.isclose(p1.loc["Second Class", "late_rate"], 1.0)      # real 4 vs scheduled 2: late
    assert np.isclose(p1.loc["First Class", "late_rate"], 1.0)       # real 2 vs scheduled 1: late
    assert np.isclose(p1.loc["Same Day", "late_rate"], 0.0)


def test_product_params_use_training_lines_and_derive_unit_cost():
    lines = pd.DataFrame({
        "product_id": [1, 1, 1, 1, 2, 2], "price": [100.0] * 4 + [50.0] * 2,
        "discount_rate": [0.1, 0.3, 0.9, 0.9, 0.2, 0.0],
        "split": ["train", "train", "validation", "test", "train", "train"]})
    demand_train = {1: np.array([4.0, 6.0, 8.0]), 2: np.array([1.0, 1.0, 1.0])}
    out = inputs.product_params(lines, demand_train, cost_ratio=0.8)
    assert np.isclose(out.loc[1, "mean_discount"], 0.2)              # validation/test discounts ignored
    assert np.isclose(out.loc[1, "net_price"], 80.0) and np.isclose(out.loc[1, "unit_cost"], 64.0)
    assert np.isclose(out.loc[1, "mu"], 6.0) and np.isclose(out.loc[1, "sigma"], 2.0)
    assert np.isclose(out.loc[2, "net_price"], 45.0) and np.isclose(out.loc[2, "sigma"], 0.0)


def test_demand_by_split_is_chronological_and_complete():
    weeks = pd.date_range("2015-01-05", periods=142, freq="7D")
    weekly = pd.DataFrame([(p, w, 1) for p in (1, 2) for w in weeks], columns=["product_id", "week", "units"])
    sp = {"train": ["2015-01-05", "2016-11-21", 99], "validation": ["2016-11-28", "2017-04-17", 21],
          "test": ["2017-04-24", "2017-09-18", 22]}
    from src import split as split_mod
    orig = split_mod.data.load_split
    split_mod.data.load_split = lambda: sp
    try:
        demand, wk = inputs.demand_by_split(weekly)
    finally:
        split_mod.data.load_split = orig
    assert {k: len(v) for k, v in wk.items()} == {"train": 99, "validation": 21, "test": 22}
    assert wk["train"].max() < wk["validation"].min() and wk["validation"].max() < wk["test"].min()
    assert len(demand["train"][1]) == 99 and len(demand["validation"][2]) == 21