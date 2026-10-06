"""Probabilistic demand scenarios and scenario-wise BOM propagation."""
#BOM M3 trả lời: Với một forecast P25/P50/P75 thì cần bao nhiêu nguyên liệu?”

#thì Scenario Layer tiến thêm một bước:
#“Nếu tương lai xảy ra theo nhiều kịch bản nhu cầu khác nhau, thì mỗi kịch bản cần bao nhiêu sản phẩm và bao nhiêu nguyên liệu, với xác suất bao nhiêu?”
# Một forecast truyền thống có thể nói:

# LATTE ngày mai:
# P25 = 80
# P50 = 100
# P75 = 130

# Scenario layer có thể biểu diễn thành:

# Scenario LOW
# Probability = 25%
# Latte = 80

# Scenario BASE
# Probability = 50%
# Latte = 100

# Scenario HIGH
# Probability = 25%
# Latte = 130

# Rồi mỗi scenario được truyền qua recipe:

# LOW
# → Milk = 18 L

# BASE
# → Milk = 23 L

# HIGH
# → Milk = 30 L

# Đây là ý nghĩa lớn nhất của module này.
from shelfcash_forecast.scenario.bom import propagate_ingredient_demand_scenarios
from shelfcash_forecast.scenario.composer import generate_product_demand_scenarios
from shelfcash_forecast.scenario.contracts import (
    IngredientDemandScenarioBundle,
    ProductDemandScenarioBundle,
)
from shelfcash_forecast.scenario.lead_time import (
    DeterministicLeadTimeModel,
    EmpiricalLeadTimeModel,
    LeadTimeModel,
)
from shelfcash_forecast.scenario.shelf_life import (
    DeterministicShelfLifeModel,
    ShelfLifeModel,
)
from shelfcash_forecast.scenario.yield_loss import (
    EmpiricalUsageResidualYieldLossModel,
    FixedRecipeYieldLossModel,
)

__all__ = [
    "DeterministicLeadTimeModel",
    "DeterministicShelfLifeModel",
    "EmpiricalLeadTimeModel",
    "EmpiricalUsageResidualYieldLossModel",
    "FixedRecipeYieldLossModel",
    "IngredientDemandScenarioBundle",
    "LeadTimeModel",
    "ProductDemandScenarioBundle",
    "ShelfLifeModel",
    "generate_product_demand_scenarios",
    "propagate_ingredient_demand_scenarios",
]
