from __future__ import annotations

import math
from collections import defaultdict, deque

from shelfcash_forecast.bom.contracts import UnitConversionRule
from shelfcash_forecast.exceptions import (
    InvalidUnitConversionError,
    UnitConversionError,
)

UNIT_ALIASES = {
    "g": "g",
    "gram": "g",
    "grams": "g",
    "kg": "kg",
    "kilogram": "kg",
    "kilograms": "kg",
    "ml": "ml",
    "milliliter": "ml",
    "milliliters": "ml",
    "millilitre": "ml",
    "millilitres": "ml",
    "l": "liter",
    "liter": "liter",
    "liters": "liter",
    "litre": "liter",
    "litres": "liter",
    "unit": "unit",
    "units": "unit",
    "piece": "unit",
    "pieces": "unit",
    "pc": "unit",
    "pcs": "unit",
    "each": "unit",
    "item": "unit",
    "items": "unit",
    "cup": "unit",
    "cups": "unit",
    "ly": "unit",
    "serving": "unit",
    "servings": "unit",
    "pack": "pack",
    "packs": "pack",
    "package": "pack",
    "packages": "pack",
}

UNIT_DIMENSIONS = {
    "g": "mass",
    "kg": "mass",
    "ml": "volume",
    "liter": "volume",
    "unit": "count",
}
CANONICAL_UNITS = {"mass": "kg", "volume": "liter", "count": "unit"}
TO_CANONICAL_FACTORS = {
    "g": 0.001,
    "kg": 1.0,
    "ml": 0.001,
    "liter": 1.0,
    "unit": 1.0,
}


def normalize_unit(unit: str) -> str: # Chuẩn hóa đơn vị
    """Normalize supported aliases without guessing physical equivalence."""

    normalized = str(unit).strip().lower().replace(" ", "_")
    if not normalized:
        raise InvalidUnitConversionError("Unit không được rỗng.")
    return UNIT_ALIASES.get(normalized, normalized)


def unit_dimension(unit: str) -> str | None:
    return UNIT_DIMENSIONS.get(normalize_unit(unit))


def _builtin_factor(from_unit: str, to_unit: str) -> float | None:
    source = normalize_unit(from_unit)
    target = normalize_unit(to_unit)
    if source == target:
        return 1.0
    source_dimension = UNIT_DIMENSIONS.get(source)
    target_dimension = UNIT_DIMENSIONS.get(target)
    if source_dimension is None or target_dimension is None:
        return None
    if source_dimension != target_dimension:
        raise UnitConversionError(
            f"Không thể convert {source} sang {target}: khác dimension.",
            details={
                "from_unit": source,
                "to_unit": target,
                "from_dimension": source_dimension,
                "to_dimension": target_dimension,
            },
        )
    return TO_CANONICAL_FACTORS[source] / TO_CANONICAL_FACTORS[target]


def convert_product_quantity(quantity: float, from_unit: str, to_unit: str) -> float:
    """Convert product-output units using safe built-ins only."""

    factor = _builtin_factor(from_unit, to_unit)
    if factor is None:
        raise UnitConversionError(
            f"Không thể xác nhận product unit {from_unit!r} tương thích với "
            f"yield unit {to_unit!r}.",
            details={"product_unit": from_unit, "yield_unit": to_unit},
        )
    return float(quantity) * factor


class UnitConverter:
    """Deterministic built-in and ingredient-specific unit converter."""

    def __init__(self, rules: list[UnitConversionRule] | None = None) -> None:
        self._graphs: dict[str, dict[str, list[tuple[str, float]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for rule in rules or []:
            self._add_rule(rule)
        self._validate_graphs()

    def _add_rule(self, rule: UnitConversionRule) -> None:
        source = normalize_unit(rule.from_unit)
        target = normalize_unit(rule.to_unit)
        source_dimension = UNIT_DIMENSIONS.get(source)
        target_dimension = UNIT_DIMENSIONS.get(target)
        if source == target and not math.isclose(rule.factor, 1.0):
            raise InvalidUnitConversionError(
                "Conversion trong cùng unit phải có factor=1.",
                details=rule.model_dump(),
            )
        if (
            source_dimension is not None
            and target_dimension is not None
            and source_dimension != target_dimension
        ):
            raise InvalidUnitConversionError(
                "Conversion metadata không được nối hai physical dimensions.",
                details=rule.model_dump(),
            )
        builtin = _builtin_factor(source, target)
        if builtin is not None and not math.isclose(
            builtin, rule.factor, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise InvalidUnitConversionError(
                "Conversion metadata xung đột với built-in conversion.",
                details={**rule.model_dump(), "built_in_factor": builtin},
            )
        graph = self._graphs[rule.ingredient_id]
        graph[source].append((target, rule.factor))
        graph[target].append((source, 1.0 / rule.factor))

    def _neighbors(
        self,
        ingredient_id: str,
        unit: str,
    ) -> list[tuple[str, float]]:
        result = list(self._graphs.get(ingredient_id, {}).get(unit, []))
        if unit == "g":
            result.append(("kg", 0.001))
        elif unit == "kg":
            result.append(("g", 1000.0))
        elif unit == "ml":
            result.append(("liter", 0.001))
        elif unit == "liter":
            result.append(("ml", 1000.0))
        return result

    def _find_factor(
        self,
        ingredient_id: str,
        from_unit: str,
        to_unit: str,
    ) -> float | None:
        source = normalize_unit(from_unit)
        target = normalize_unit(to_unit)
        if source == target:
            return 1.0
        queue: deque[tuple[str, float]] = deque([(source, 1.0)])
        visited = {source}
        while queue:
            current, current_factor = queue.popleft()
            for neighbor, edge_factor in self._neighbors(ingredient_id, current):
                if neighbor in visited:
                    continue
                factor = current_factor * edge_factor
                if neighbor == target:
                    return factor
                visited.add(neighbor)
                queue.append((neighbor, factor))
        return None

    def _validate_graphs(self) -> None:
        for ingredient_id, graph in self._graphs.items():
            potentials: dict[str, float] = {}
            for start in graph:
                if start in potentials:
                    continue
                potentials[start] = 1.0
                queue = deque([start])
                component_units: set[str] = set()
                while queue:
                    current = queue.popleft()
                    component_units.add(current)
                    for neighbor, factor in self._neighbors(ingredient_id, current):
                        expected = potentials[current] * factor
                        if neighbor in potentials:
                            if not math.isclose(
                                potentials[neighbor],
                                expected,
                                rel_tol=1e-9,
                                abs_tol=1e-12,
                            ):
                                raise InvalidUnitConversionError(
                                    "Conversion rules tạo cycle có factor không nhất quán.",
                                    details={
                                        "ingredient_id": ingredient_id,
                                        "unit": neighbor,
                                    },
                                )
                            continue
                        potentials[neighbor] = expected
                        queue.append(neighbor)

                dimensions = {
                    UNIT_DIMENSIONS[unit]
                    for unit in component_units
                    if unit in UNIT_DIMENSIONS
                }
                if len(dimensions) > 1:
                    raise InvalidUnitConversionError(
                        "Conversion graph gián tiếp nối nhiều physical dimensions.",
                        details={
                            "ingredient_id": ingredient_id,
                            "units": sorted(component_units),
                            "dimensions": sorted(dimensions),
                        },
                    )

    def conversion_factor(
        self,
        ingredient_id: str,
        from_unit: str,
        to_unit: str,
    ) -> float:
        source = normalize_unit(from_unit)
        target = normalize_unit(to_unit)
        factor = self._find_factor(ingredient_id, source, target)
        if factor is None:
            source_dimension = UNIT_DIMENSIONS.get(source)
            target_dimension = UNIT_DIMENSIONS.get(target)
            raise UnitConversionError(
                f"Không có conversion cho ingredient={ingredient_id}: "
                f"{source} -> {target}.",
                details={
                    "ingredient_id": ingredient_id,
                    "from_unit": source,
                    "to_unit": target,
                    "from_dimension": source_dimension,
                    "to_dimension": target_dimension,
                },
            )
        return factor

    def convert(
        self,
        quantity: float,
        from_unit: str,
        to_unit: str,
        *,
        ingredient_id: str,
    ) -> float:
        return float(quantity) * self.conversion_factor(
            ingredient_id, from_unit, to_unit
        )

    def canonical_unit(self, ingredient_id: str, from_unit: str) -> str:
        source = normalize_unit(from_unit)
        dimension = UNIT_DIMENSIONS.get(source)
        if dimension is not None:
            return CANONICAL_UNITS[dimension]

        reachable = [
            target
            for target in ("kg", "liter", "unit")
            if self._find_factor(ingredient_id, source, target) is not None
        ]
        if len(reachable) > 1:
            raise InvalidUnitConversionError(
                "Conversion metadata đưa một unit đến nhiều dimensions.",
                details={
                    "ingredient_id": ingredient_id,
                    "from_unit": source,
                    "canonical_candidates": reachable,
                },
            )
        if reachable:
            return reachable[0]
        if source == "pack":
            raise UnitConversionError(
                f"Ingredient {ingredient_id} dùng pack nhưng thiếu conversion metadata.",
                details={"ingredient_id": ingredient_id, "from_unit": source},
            )
        return source
