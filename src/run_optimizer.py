"""Stage 7b: optimizer vs baseline policies on the VALIDATION weeks (SIMULATION).

The test split is not selectable here. Plans use training weeks only; evaluation uses the
actual demand of the evaluation weeks.

Run:  python -m src.run_optimizer
"""
import numpy as np
import pandas as pd

from src import config, cost_data, data, inputs, metrics, optimizer
from src.assumptions import Assumptions
from src.optimizer import evaluate_plan, optimize, verify_plan

EVAL_SPLIT = "validation"
N_BOOT = 2000


def scenarios(base: Assumptions) -> dict:
    s = {"base": base}
    for name in ("ordering", "holding", "stockout", "shipping", "late"):
        for f in (0.5, 2.0):
            s[f"{name} x{f}"] = base.scaled(**{name: f})
    from dataclasses import replace
    s["budget +20%"] = replace(base, budget_slack=0.20)
    return s


def weighted_fill(per_product, demand_eval):
    tot = sum(demand_eval[p].sum() for p in per_product["product_id"])
    return float(sum(r.fill * demand_eval[r.product_id].sum() for r in per_product.itertuples()) / tot)


def run_scenario(name, A, prod, modes, demand_train, demand_eval, weeks_eval, cache):
    plan, info = optimize(prod, modes, demand_train, A, cache=cache)
    ver = verify_plan(plan, prod, A, info["budget"], info["capacity"])
    plans = {"optimizer": plan, "B1": optimizer.baseline_plan(prod, modes, A, "B1"),
             "B2": optimizer.baseline_plan(prod, modes, A, "B2")}
    ev = {k: evaluate_plan(p, prod, modes, demand_eval, A) for k, p in plans.items()}
    tr = {k: evaluate_plan(p, prod, modes, demand_train, A) for k, p in plans.items()}
    row = {"scenario": name, "constraints_ok": ver["all_ok"],
           "budget": round(info["budget"], 1), "invest_opt": round(ver["invest"], 1),
           "train_cost_opt": tr["optimizer"]["weekly_total"].mean(),
           "train_cost_B1": tr["B1"]["weekly_total"].mean(), "train_cost_B2": tr["B2"]["weekly_total"].mean(),
           "val_cost_opt": ev["optimizer"]["weekly_total"].mean(),
           "val_cost_B1": ev["B1"]["weekly_total"].mean(), "val_cost_B2": ev["B2"]["weekly_total"].mean()}
    for b in ("B1", "B2"):
        d = metrics.paired_week_bootstrap(weeks_eval, ev["optimizer"]["weekly_total"], ev[b]["weekly_total"],
                                          n_boot=N_BOOT, seed=0)
        row.update({f"diff_vs_{b}": d["mean_diff"], f"ci_low_{b}": d["ci_low"], f"ci_high_{b}": d["ci_high"]})
    for k, key in (("optimizer", "opt"), ("B1", "B1"), ("B2", "B2")):
        row[f"fill_{key}"] = weighted_fill(ev[k]["per_product"], demand_eval)
    row["n_not_standard"] = int((plan["mode"] != "Standard Class").sum())
    row["mean_R"] = float(plan["R"].mean())
    return row, plan, ver, ev


def main():
    A = Assumptions()
    elig = data.load_eligible_products()
    weekly = pd.read_csv(config.PROCESSED / "weekly_units.csv", parse_dates=["week"])
    demand, weeks = inputs.demand_by_split(weekly)
    demand_train, demand_eval, weeks_eval = demand["train"], demand[EVAL_SPLIT], weeks[EVAL_SPLIT]

    cost_lines = cost_data.load_cost_lines(products=elig.index)
    cost_ratio = float(cost_lines.loc[cost_lines["split"] == "train", cost_data.TARGET].mean())
    prod = inputs.product_params(cost_lines, demand_train, cost_ratio)
    modes = inputs.mode_params(inputs.load_shipping_lines())

    print(f"Global training mean cost ratio: {cost_ratio:.4f}  (derived, not an observed cost)")
    print("\nMode parameters (training lines; lead time reuses CUSTOMER delivery days):")
    print(modes.round(3).to_string())
    print(f"\nProducts: {len(prod)} | training weeks: {len(next(iter(demand_train.values())))} | "
          f"{EVAL_SPLIT} weeks: {len(weeks_eval)}")

    cache = optimizer.build_sim_cache(prod, modes, demand_train, A.cycles)
    rows = []
    for name, A_s in scenarios(A).items():
        row, plan, ver, ev = run_scenario(name, A_s, prod, modes, demand_train, demand_eval, weeks_eval, cache)
        rows.append(row)
        if name == "base":
            plan.round(4).to_csv(config.REPORTS / "optimizer_plan.csv", index=False)
            print("\nBASE PLAN: modes chosen:", plan["mode"].value_counts().to_dict(),
                  "| cycles R:", plan["R"].value_counts().to_dict())
            print("Constraint check:", {k: v for k, v in ver.items() if k != "load_by_mode"})
            print("Cost breakdown per week on validation ($):")
            for k in ("optimizer", "B1", "B2"):
                comp = {c: round(float(v.mean()), 1) for c, v in ev[k]["components"].items()}
                print(f"  {k:9s} {comp}  total {ev[k]['weekly_total'].mean():.1f}")
    res = pd.DataFrame(rows)
    res.round(4).to_csv(config.REPORTS / "optimizer_validation_results.csv", index=False)

    show = res[["scenario", "constraints_ok", "train_cost_opt", "train_cost_B1", "val_cost_opt", "val_cost_B1",
                "val_cost_B2", "diff_vs_B1", "ci_low_B1", "ci_high_B1", "diff_vs_B2", "ci_low_B2", "ci_high_B2"]]
    print("\nWEEKLY COST ($) - SIMULATION under assumed costs (diff < 0 = optimizer cheaper):")
    print(show.round(1).to_string(index=False))
    fills = res[["scenario", "fill_opt", "fill_B1", "fill_B2", "n_not_standard", "mean_R"]]
    print("\nRealised fill rate on validation weeks (required 95% was enforced on TRAINING weeks):")
    print(fills.round(4).to_string(index=False))
    print(f"\nSaved to {config.REPORTS / 'optimizer_validation_results.csv'} and optimizer_plan.csv")


if __name__ == "__main__":
    main()