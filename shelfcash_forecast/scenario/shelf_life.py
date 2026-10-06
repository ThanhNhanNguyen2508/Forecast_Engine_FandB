from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class ShelfLifeRealization:
    effective_expiry_date: date | None
    source: str


class ShelfLifeModel(Protocol):
    def realize(
        self,
        *,
        official_expiry_date: date | None,
        rng: np.random.Generator,
    ) -> ShelfLifeRealization: ...


class DeterministicShelfLifeModel:
    """Use the official expiry without manufacturing freshness probabilities."""

    def realize(
        self,
        *,
        official_expiry_date: date | None,
        rng: np.random.Generator,
    ) -> ShelfLifeRealization:
        del rng
        return ShelfLifeRealization(
            effective_expiry_date=official_expiry_date,
            source=(
                "official_expiry"
                if official_expiry_date is not None
                else "official_unknown"
            ),
        )
