# Nay đặt, nào nhận ?
# Procurement / order
#        ↓
# LeadTimeModel.realize()
#        ↓
# Scenario S1 arrival = 13/08
# Scenario S2 arrival = 15/08
# Scenario S3 arrival = 17/08
#        ↓
# scenario-specific PlannedInboundDelivery
#        ↓
# MonteCarloInventoryRunner
#        ↓
# simulate_inventory()

# lead_time.py biến một order date thành realized arrival date; deterministic model dùng fixed lead time, empirical model sample historical observed lead times theo hierarchy supplier+ingredient → supplier → ingredient → global, với cutoff chống future leakage và caller-controlled RNG để reproducible.
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Protocol

import numpy as np
import pandas as pd

from shelfcash_forecast.exceptions import ScenarioDataInsufficiencyError


@dataclass(frozen=True)
class LeadTimeRealization:
    arrival_date: date
    lead_time_days: int
    source: str
    observation_count: int | None = None


class LeadTimeModel(Protocol):
    def realize(
        self,
        *,
        order_date: date,
        supplier_id: str,
        ingredient_id: str,
        rng: np.random.Generator,
    ) -> LeadTimeRealization: ...


class DeterministicLeadTimeModel:
    def __init__(
        self,
        default_lead_time_days: int,
        *,
        supplier_ingredient_days: dict[tuple[str, str], int] | None = None,
    ) -> None:
        if default_lead_time_days < 0:
            raise ValueError("Lead time cannot be negative.")
        self.default_lead_time_days = default_lead_time_days
        self.supplier_ingredient_days = dict(supplier_ingredient_days or {})
        if any(days < 0 for days in self.supplier_ingredient_days.values()):
            raise ValueError("Lead time cannot be negative.")

    def realize(
        self,
        *,
        order_date: date,
        supplier_id: str,
        ingredient_id: str,
        rng: np.random.Generator,
    ) -> LeadTimeRealization:
        del rng
        key = (supplier_id, ingredient_id)
        days = self.supplier_ingredient_days.get(key, self.default_lead_time_days)
        source = (
            "supplier_ingredient_fixed"
            if key in self.supplier_ingredient_days
            else "fixed"
        )
        return LeadTimeRealization(
            arrival_date=order_date + timedelta(days=days),
            lead_time_days=days,
            source=source,
        )


class EmpiricalLeadTimeModel:
    """Sample non-negative observed lead times using an auditable hierarchy."""

    def __init__(
        self,
        pools: dict[tuple[str, ...], list[int]],
        minimum_samples: int,
        *,
        as_of_date: date | None = None,
        observation_count: int | None = None,
    ) -> None:
        self.pools = pools
        self.minimum_samples = minimum_samples
        self.as_of_date = as_of_date
        self.observation_count = observation_count

    @classmethod
    def fit(
        cls,
        purchase_order_history: pd.DataFrame,
        *,
        minimum_samples: int = 3,
        as_of_date: date | str | pd.Timestamp | None = None,
        cutoff_date: date | str | pd.Timestamp | None = None,
    ) -> EmpiricalLeadTimeModel:
        required = {"supplier_id", "ingredient_id", "order_date", "arrival_date"}
        missing = sorted(required - set(purchase_order_history.columns))
        if missing:
            raise ValueError(f"Purchase-order history is missing columns: {missing}")
        if minimum_samples < 1:
            raise ValueError("minimum_samples must be positive.")
        if as_of_date is not None and cutoff_date is not None:
            as_of = pd.Timestamp(as_of_date).normalize()
            cutoff = pd.Timestamp(cutoff_date).normalize()
            if as_of != cutoff:
                raise ValueError("as_of_date and cutoff_date must match when both set.")
        effective_cutoff = (
            pd.Timestamp(as_of_date if as_of_date is not None else cutoff_date).normalize()
            if as_of_date is not None or cutoff_date is not None
            else None
        )
        frame = purchase_order_history.copy()
        frame["order_date"] = pd.to_datetime(
            frame["order_date"], errors="coerce"
        ).dt.normalize()
        frame["arrival_date"] = pd.to_datetime(
            frame["arrival_date"], errors="coerce"
        ).dt.normalize()
        if frame[["order_date", "arrival_date"]].isna().any(axis=None):
            raise ValueError("Observed order/arrival dates must be valid.")
        # A receipt after as_of_date was not observable at fitting time and
        # must not enter any empirical pool.
        if effective_cutoff is not None:
            frame = frame.loc[frame["arrival_date"].le(effective_cutoff)].copy()
        frame["lead_time_days"] = (
            frame["arrival_date"] - frame["order_date"]
        ).dt.days
        if (frame["lead_time_days"] < 0).any():
            raise ValueError("Observed lead times cannot be negative.")

        pools: defaultdict[tuple[str, ...], list[int]] = defaultdict(list)
        for row in frame.itertuples(index=False):
            days = int(row.lead_time_days)
            supplier = str(row.supplier_id)
            ingredient = str(row.ingredient_id)
            pools[("supplier_ingredient", supplier, ingredient)].append(days)
            pools[("supplier", supplier)].append(days)
            pools[("ingredient", ingredient)].append(days)
            pools[("global",)].append(days)
        return cls(
            dict(pools),
            minimum_samples,
            as_of_date=(effective_cutoff.date() if effective_cutoff is not None else None),
            observation_count=len(frame),
        )

    def realize(
        self,
        *,
        order_date: date,
        supplier_id: str,
        ingredient_id: str,
        rng: np.random.Generator,
    ) -> LeadTimeRealization:
        candidates = (
            ("supplier_ingredient", supplier_id, ingredient_id),
            ("supplier", supplier_id),
            ("ingredient", ingredient_id),
            ("global",),
        )
        for key in candidates:
            values = self.pools.get(key, [])
            if len(values) >= self.minimum_samples:
                days = int(rng.choice(values))
                return LeadTimeRealization(
                    arrival_date=order_date + timedelta(days=days),
                    lead_time_days=days,
                    source=key[0],
                    observation_count=len(values),
                )
        raise ScenarioDataInsufficiencyError(
            "No empirical lead-time pool meets minimum_samples.",
            details={
                "supplier_id": supplier_id,
                "ingredient_id": ingredient_id,
                "minimum_samples": self.minimum_samples,
            },
        )
