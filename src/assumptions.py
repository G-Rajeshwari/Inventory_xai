"""ALL simulated (assumed) quantities of the inventory optimizer live here.

Nothing in this file is observed in the dataset. Every optimizer result is a SIMULATION
under these assumptions and must be reported as such.
"""
from dataclasses import dataclass, replace

MODES = ("Standard Class", "Second Class", "First Class", "Same Day")


@dataclass(frozen=True)
class Assumptions:
    ordering_cost: float = 25.0              # $ per order placed
    holding_rate_annual: float = 0.25        # fraction of unit cost per unit per year
    ship_pct: tuple = (0.02, 0.04, 0.07, 0.12)       # shipping $ per unit, as fraction of unit cost (aligned with MODES)
    stockout_mult: float = 1.0               # penalty per unit short = stockout_mult x net price
    late_penalty: float = 20.0               # $ per late order (multiplied by the mode's late rate)
    fill_rate_min: float = 0.95              # required fill rate per product
    capacity_share: tuple = (1.0, 0.5, 0.25, 0.10)   # share of total weekly demand each mode can carry (aligned with MODES)
    cycles: tuple = (1, 2, 4)                # review cycles R in weeks
    budget_slack: float = 0.0                # budget = baseline B1 inventory investment x (1 + slack)
    z_baseline: float = 1.645                # service factor of baseline B1
    weeks_per_year: int = 52

    def ship(self, mode: str) -> float:
        return self.ship_pct[MODES.index(mode)]

    def capacity(self, mode: str) -> float:
        return self.capacity_share[MODES.index(mode)]

    def holding_per_unit_week(self, unit_cost: float) -> float:
        return self.holding_rate_annual / self.weeks_per_year * unit_cost

    def scaled(self, ordering=1.0, holding=1.0, stockout=1.0, shipping=1.0, late=1.0) -> "Assumptions":
        """Copy with cost parameters multiplied (used for the sensitivity analysis)."""
        return replace(
            self,
            ordering_cost=self.ordering_cost * ordering,
            holding_rate_annual=self.holding_rate_annual * holding,
            stockout_mult=self.stockout_mult * stockout,
            ship_pct=tuple(x * shipping for x in self.ship_pct),
            late_penalty=self.late_penalty * late,
        )