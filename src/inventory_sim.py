"""Weekly inventory simulation of an order-up-to policy (lost sales).

Policy for one product: every R weeks, at the start of the week, order
q = max(0, S - on_hand). The order arrives a fraction L (0 <= L < 1) of the way through
that week. Demand within a week is spread evenly over the week. Unmet demand is lost.
Initial on-hand stock is S (assumption). Purchase cost of the units sold is the same under
every policy and is excluded; holding, ordering, shipping, stockout and late penalties count.
"""
import numpy as np


def simulate(D, S, R, L):
    """Simulate weekly demand D (T,) for each order-up-to level in S (n,).

    Returns per-week arrays of shape (T, n): orders (0/1), units ordered, units short,
    average on-hand during the week.
    """
    D = np.asarray(D, dtype=float)
    S = np.atleast_1d(np.asarray(S, dtype=float))
    T, n = len(D), len(S)
    orders = np.zeros((T, n))
    units = np.zeros((T, n))
    short = np.zeros((T, n))
    onhand = np.zeros((T, n))
    I = S.copy()
    for t in range(T):
        d = D[t]
        if t % R == 0:
            q = np.maximum(0.0, S - I)
            pre = L * d
            served = np.minimum(I, pre)
            I1 = I - served
            mid = I1 + q
            post = (1.0 - L) * d
            served2 = np.minimum(mid, post)
            end = mid - served2
            short[t] = (pre - served) + (post - served2)
            onhand[t] = L * (I + I1) / 2.0 + (1.0 - L) * (mid + end) / 2.0
            orders[t] = (q > 0).astype(float)
            units[t] = q
        else:
            served = np.minimum(I, d)
            end = I - served
            short[t] = d - served
            onhand[t] = (I + end) / 2.0
        I = end
    return {"orders": orders, "units": units, "short": short, "onhand": onhand}


def cost_components(sim, unit_cost, net_price, ship_pct, late_rate, A):
    """Per-week cost components (each of shape (T, n)) for one product."""
    return {
        "ordering": A.ordering_cost * sim["orders"],
        "holding": A.holding_per_unit_week(unit_cost) * sim["onhand"],
        "shipping": ship_pct * unit_cost * sim["units"],
        "stockout": A.stockout_mult * net_price * sim["short"],
        "late": A.late_penalty * late_rate * sim["orders"],
    }


def summarize(sim, D):
    """Weekly means over time (shape (n,)) plus fill rate."""
    D = np.asarray(D, dtype=float)
    total = D.sum()
    out = {k: v.mean(axis=0) for k, v in sim.items()}
    out["fill"] = 1.0 - sim["short"].sum(axis=0) / total if total > 0 else np.ones(sim["short"].shape[1])
    return out