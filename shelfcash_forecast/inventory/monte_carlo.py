# monte_carlo.py không tạo randomness; nó nhận N future worlds đã được realize, 
# ghép demand/supply/waste riêng cho từng world, chạy tất cả qua cùng simulate_inventory(), 
# rồi dùng probability weights để tổng hợp thành inventory risk.
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date

import pandas as pd

from shelfcash_forecast.exceptions import InventoryValidationError
from shelfcash_forecast.inventory.adapters import advanced_inventory_scenarios
from shelfcash_forecast.inventory.contracts import (
    ConsequenceCostAssumption,
    InboundDelivery,
    InventoryDemandScenario,
    InventoryLot,
    InventorySimulationPackage,
    InventorySimulationPolicy,
    PlannedInboundDelivery,
    WasteEvent,
)
from shelfcash_forecast.inventory.metrics import aggregate_risk_metrics
from shelfcash_forecast.inventory.simulator import simulate_inventory
from shelfcash_forecast.scenario.contracts import IngredientDemandScenarioBundle
# Nhận những future worlds đã được realize sẵn, rồi đưa từng world qua cùng một deterministic inventory simulator.

class MonteCarloInventoryRunner:
    """Run already-realized futures through the single deterministic simulator."""

    def run(
        self,
        initial_inventory: Sequence[InventoryLot],
        demand_scenarios: Sequence[InventoryDemandScenario]
        | IngredientDemandScenarioBundle,
        inbound: Sequence[InboundDelivery] = (),
        planned_inbound: Sequence[PlannedInboundDelivery] = (),
        waste_events: Sequence[WasteEvent] = (),
        *,
        scenario_inbound: Mapping[str, Sequence[InboundDelivery]] | None = None,
        scenario_planned_inbound: Mapping[
            str, Sequence[PlannedInboundDelivery]
        ]
        | None = None,
        scenario_waste_events: Mapping[str, Sequence[WasteEvent]] | None = None,
        policy: InventorySimulationPolicy | None = None,
        unit_conversions: pd.DataFrame | None = None,
        cost_assumptions: Sequence[ConsequenceCostAssumption] = (),
        seed: int,
        allow_incomplete: bool = False,
        simulation_start_date: date | None = None,
        simulation_end_date: date | None = None,
    ) -> InventorySimulationPackage:
        policy = policy or InventorySimulationPolicy()
        scenarios = (
            advanced_inventory_scenarios(
                demand_scenarios, allow_incomplete=allow_incomplete
            )
            if isinstance(demand_scenarios, IngredientDemandScenarioBundle)
            else list(demand_scenarios)
        )
        if not scenarios:
            raise ValueError("Monte Carlo requires at least one realized scenario.")
        if any(scenario.probability_weight is None for scenario in scenarios):
            raise InventoryValidationError(
                "Monte Carlo risk aggregation requires explicit probability weights.",
                code="INVALID_SCENARIO_WEIGHTS",
            )
        if not math.isclose(
            sum(float(scenario.probability_weight) for scenario in scenarios),
            1.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise InventoryValidationError(
                "Monte Carlo scenario weights must sum to one.",
                code="INVALID_SCENARIO_WEIGHTS",
            )

        scenario_inbound = scenario_inbound or {}
        scenario_planned_inbound = scenario_planned_inbound or {}
        scenario_waste_events = scenario_waste_events or {}
        results = []
        for scenario in scenarios:
            results.append(
                simulate_inventory(
                    initial_inventory,
                    scenario,
                    [*inbound, *scenario_inbound.get(scenario.scenario_id, ())],
                    [
                        *planned_inbound,
                        *scenario_planned_inbound.get(scenario.scenario_id, ()),
                    ],
                    [
                        *waste_events,
                        *scenario_waste_events.get(scenario.scenario_id, ()),
                    ],
                    policy=policy,
                    unit_conversions=unit_conversions,
                    cost_assumptions=cost_assumptions,
                    simulation_start_date=simulation_start_date,
                    simulation_end_date=simulation_end_date,
                )
            )
        return InventorySimulationPackage(
            simulation_start_date=min(item.simulation_start_date for item in results),
            simulation_end_date=max(item.simulation_end_date for item in results),
            results=results,
            risk_metrics=aggregate_risk_metrics(
                results,
                waste_threshold=policy.waste_threshold,
                fill_rate_target=policy.fill_rate_target,
            ),
            provenance={
                "runner": "monte_carlo_inventory_v1",
                "transition_engine": "lot_level_fefo_v1",
                "seed": seed,
                "scenario_count": len(results),
            },
            warnings=sorted(
                {warning for result in results for warning in result.warnings}
            ),
        )
# 13. Output

# Cuối cùng:

# InventorySimulationPackage(
#     results=results,
#     risk_metrics=aggregate_risk_metrics(...),
#     provenance={
#         "runner": "monte_carlo_inventory_v1",
#         "transition_engine": "lot_level_fefo_v1",
#         "seed": seed,
#         "scenario_count": len(results),
#     }
# )

# Provenance nói rất rõ architecture:

# runner
# = Monte Carlo

# transition engine
# = lot-level FEFO exact simulator