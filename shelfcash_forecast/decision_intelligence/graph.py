# Mỗi evidence item trở thành một graph node. Sau đó M6 tạo các quan hệ:

# DERIVED_FROM
# USES_RECIPE
# CONTRIBUTES_TO
# REALIZED_AS_SCENARIO
# SIMULATED_UNDER
# HAS_RISK
# HAS_ORDER
# HAS_RECOURSE
# VALIDATED_BY
# PASSED_CHECK
# FAILED_CHECK
# CONSUMED_BY
# EXPIRED_AS
# WASTED_AS
# RECOMMENDED_AS

# Ví dụ:

# Forecast LATTE
#    ↓ USES_RECIPE
# Recipe contribution LATTE_RECIPE/v2
#    ↓ CONTRIBUTES_TO
# Ingredient demand MILK
#    ↓ REALIZED_AS_SCENARIO
# Inventory scenario HIGH
#    ↓ HAS_RISK
# MILK stockout risk

# Và phía quyết định:

# BALANCED plan
#    ├── HAS_ORDER → first-stage order
#    ├── HAS_RECOURSE → HIGH scenario emergency order
#    ├── SIMULATED_UNDER → exact M4 package
#    └── VALIDATED_BY → critic verdict
#                              ↓
#                        PASSED_CHECK

# Graph giúp retriever mở rộng từ một evidence sang các evidence liên quan, thay vì chỉ tìm từ khóa giống nhau.
from __future__ import annotations

import hashlib

from shelfcash_forecast.decision_intelligence.contracts import (
    DecisionGraph,
    EvidenceItem,
    EvidencePackage,
    GraphEdge,
    GraphNode,
)

NODE_TYPES = {
    "decision_request": "DecisionRequest",
    "forecast_prediction": "ForecastPrediction",
    "product_demand_scenario": "ProductDemandScenario",
    "ingredient_demand": "IngredientDemand",
    "ingredient_demand_scenario": "IngredientDemand",
    "recipe_contribution": "RecipeContribution",
    "scenario_recipe_contribution": "RecipeContribution",
    "inventory_demand_scenario": "InventoryScenario",
    "inventory_scenario_result": "InventoryScenario",
    "stress_result": "StressResult",
    "inventory_key_summary": "InventoryKey",
    "stress_inventory_key": "InventoryKey",
    "inventory_risk": "RiskMetric",
    "inventory_key_risk": "RiskMetric",
    "lot_consumption": "InventoryLot",
    "lot_expiry": "InventoryLot",
    "lot_waste": "InventoryLot",
    "strategy_profile": "ProcurementStrategy",
    "procurement_plan": "ProcurementPlan",
    "first_stage_order": "ProcurementOrder",
    "recourse_order": "ProcurementOrder",
    "critic_verdict": "CriticCheck",
    "recommendation": "Recommendation",
    "stress_definition": "StressDefinition",
    "bom_issue": "Warning",
}


def _node_id(evidence_id: str) -> str:
    return f"node:{evidence_id}"


def _edge_id(source: str, target: str, edge_type: str) -> str:
    digest = hashlib.sha256(f"{source}|{edge_type}|{target}".encode()).hexdigest()[:16]
    return f"edge:{edge_type.lower()}:{digest}"


def _related(left: EvidenceItem, right: EvidenceItem, keys: tuple[str, ...]) -> bool:
    shared = [key for key in keys if key in left.entities and key in right.entities]
    return bool(shared) and all(left.entities[key] == right.entities[key] for key in shared)


def build_decision_graph(evidence: EvidencePackage) -> DecisionGraph:
    """Build a deterministic typed provenance graph over normalized evidence."""

    request_node_id = f"request:{evidence.request_id}"
    nodes = [
        GraphNode(
            node_id=request_node_id,
            node_type="DecisionRequest",
            label=f"Decision request {evidence.request_id}",
            attributes={"request_id": evidence.request_id},
        )
    ]
    for item in evidence.items:
        nodes.append(
            GraphNode(
                node_id=_node_id(item.evidence_id),
                node_type=NODE_TYPES.get(item.evidence_type, "Evidence"),
                label=item.text,
                evidence_ids=[item.evidence_id],
                attributes={
                    "layer": item.layer,
                    "evidence_type": item.evidence_type,
                    "semantics": item.semantics,
                    "entities": item.entities,
                },
            )
        )

    edges: dict[str, GraphEdge] = {}

    def add_edge(
        source_item: EvidenceItem | None,
        target_item: EvidenceItem | None,
        edge_type: str,
        *,
        source_id: str | None = None,
        target_id: str | None = None,
    ) -> None:
        source = source_id or _node_id(source_item.evidence_id)  # type: ignore[union-attr]
        target = target_id or _node_id(target_item.evidence_id)  # type: ignore[union-attr]
        identifiers = sorted(
            {item.evidence_id for item in (source_item, target_item) if item is not None}
        )
        edge = GraphEdge(
            edge_id=_edge_id(source, target, edge_type),
            source_node_id=source,
            target_node_id=target,
            edge_type=edge_type,
            evidence_ids=identifiers,
        )
        edges[edge.edge_id] = edge

    for item in evidence.items:
        add_edge(None, item, "HAS_EVIDENCE", source_id=request_node_id)

    by_type: dict[str, list[EvidenceItem]] = {}
    for item in evidence.items:
        by_type.setdefault(item.evidence_type, []).append(item)

    plans = by_type.get("procurement_plan", [])
    critics = by_type.get("critic_verdict", [])
    recommendations = by_type.get("recommendation", [])
    orders = [
        *by_type.get("first_stage_order", []),
        *by_type.get("recourse_order", []),
    ]
    simulations = [
        *by_type.get("exact_simulation_package", []),
        *by_type.get("stress_simulation_package", []),
    ]
    for plan in plans:
        for critic in critics:
            if _related(plan, critic, ("strategy", "plan_id")):
                add_edge(plan, critic, "VALIDATED_BY")
                add_edge(
                    critic,
                    plan,
                    "PASSED_CHECK" if critic.payload.get("passed") else "FAILED_CHECK",
                )
        for order in orders:
            if _related(plan, order, ("strategy",)):
                add_edge(
                    plan,
                    order,
                    "HAS_RECOURSE" if order.evidence_type == "recourse_order" else "HAS_ORDER",
                )
        for simulation in simulations:
            if _related(plan, simulation, ("strategy",)):
                add_edge(plan, simulation, "SIMULATED_UNDER")
        for recommendation in recommendations:
            if recommendation.entities.get("strategy") == plan.entities.get("strategy"):
                add_edge(recommendation, plan, "RECOMMENDED_AS")

    ingredient_items = [
        *by_type.get("ingredient_demand", []),
        *by_type.get("ingredient_demand_scenario", []),
    ]
    contributions = [
        *by_type.get("recipe_contribution", []),
        *by_type.get("scenario_recipe_contribution", []),
    ]
    forecasts = by_type.get("forecast_prediction", [])
    for ingredient in ingredient_items:
        for contribution in contributions:
            if _related(
                ingredient,
                contribution,
                ("scenario_id", "store_id", "ingredient_id", "target_date"),
            ):
                add_edge(contribution, ingredient, "CONTRIBUTES_TO")
    for contribution in contributions:
        for forecast in forecasts:
            if _related(
                forecast,
                contribution,
                ("store_id", "product_id", "target_date"),
            ):
                add_edge(forecast, contribution, "USES_RECIPE")
                add_edge(contribution, forecast, "DERIVED_FROM")

    product_scenarios = by_type.get("product_demand_scenario", [])
    for contribution in by_type.get("scenario_recipe_contribution", []):
        for product_scenario in product_scenarios:
            if _related(
                contribution,
                product_scenario,
                ("scenario_id", "store_id", "product_id", "target_date"),
            ):
                add_edge(product_scenario, contribution, "USES_RECIPE")
                add_edge(contribution, product_scenario, "DERIVED_FROM")

    demand_scenarios = by_type.get("inventory_demand_scenario", [])
    simulation_results = by_type.get("inventory_scenario_result", [])
    for demand in demand_scenarios:
        for result in simulation_results:
            if _related(demand, result, ("scenario_id",)):
                add_edge(demand, result, "REALIZED_AS_SCENARIO")

    risk_items = [
        *by_type.get("inventory_risk", []),
        *by_type.get("inventory_key_risk", []),
    ]
    inventory_items = [
        *by_type.get("inventory_scenario_result", []),
        *by_type.get("inventory_key_summary", []),
    ]
    for inventory in inventory_items:
        for risk in risk_items:
            if _related(
                inventory,
                risk,
                ("strategy", "store_id", "ingredient_id", "unit"),
            ):
                add_edge(inventory, risk, "HAS_RISK")

    inventory_keys = by_type.get("inventory_key_summary", [])
    lot_edge_types = {
        "lot_consumption": "CONSUMED_BY",
        "lot_expiry": "EXPIRED_AS",
        "lot_waste": "WASTED_AS",
    }
    for evidence_type, edge_type in lot_edge_types.items():
        for lot in by_type.get(evidence_type, []):
            for inventory in inventory_keys:
                if _related(
                    lot,
                    inventory,
                    ("strategy", "scenario_id", "store_id", "ingredient_id"),
                ):
                    add_edge(lot, inventory, edge_type)

    definitions = by_type.get("stress_definition", [])
    stress_results = [
        *by_type.get("stress_result", []),
        *by_type.get("stress_inventory_key", []),
    ]
    for definition in definitions:
        for result in stress_results:
            if definition.entities.get("stress_id") == result.entities.get("scenario_id"):
                add_edge(definition, result, "SIMULATED_UNDER")

    return DecisionGraph(
        request_id=evidence.request_id,
        nodes=sorted(nodes, key=lambda node: node.node_id),
        edges=sorted(edges.values(), key=lambda edge: edge.edge_id),
        provenance={
            "builder": "shelfcash_decision_graph_v1",
            "storage": "deterministic_in_memory",
            "node_count": len(nodes),
            "edge_count": len(edges),
        },
    )


def neighborhood_evidence_ids(
    graph: DecisionGraph,
    seed_evidence_ids: set[str],
    *,
    depth: int = 1,
) -> set[str]:
    node_to_evidence = {node.node_id: set(node.evidence_ids) for node in graph.nodes}
    frontier = {node.node_id for node in graph.nodes if set(node.evidence_ids) & seed_evidence_ids}
    visited = set(frontier)
    for _ in range(max(0, depth)):
        next_frontier: set[str] = set()
        for edge in graph.edges:
            if edge.source_node_id in frontier:
                next_frontier.add(edge.target_node_id)
            if edge.target_node_id in frontier:
                next_frontier.add(edge.source_node_id)
        next_frontier -= visited
        visited.update(next_frontier)
        frontier = next_frontier
    output = set(seed_evidence_ids)
    for node_id in visited:
        output.update(node_to_evidence.get(node_id, set()))
    return output
