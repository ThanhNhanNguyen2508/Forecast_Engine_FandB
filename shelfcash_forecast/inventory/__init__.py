"""Lot-level inventory consequence engine."""
# 1. M4 nhận gì từ M3?

# Đến cuối M3, pipeline đã trả lời xong câu hỏi:

# Với từng ngày trong forecast horizon, từng cửa hàng cần bao nhiêu nguyên liệu?

# M3 có thể trả output theo hai kiểu.

# Kiểu deterministic baseline:

# IngredientDemandPackage

# Store A
# Chicken
# 13/08
# P25 = 80 kg
# P50 = 100 kg
# P75 = 125 kg

# Kiểu probabilistic/advanced:

# IngredientDemandScenarioBundle

# Scenario S001
# probability = 0.01
# 13/08 Chicken = 92 kg
# 14/08 Chicken = 103 kg
# ...

# Scenario S002
# probability = 0.01
# 13/08 Chicken = 108 kg
# 14/08 Chicken = 117 kg
# ...

# Tức cuối M3 chúng ta đã đi từ:

# Product demand
#       ↓
# Recipe / BOM
#       ↓
# Ingredient demand

# M4 không quay lại product demand, không đọc recipe nữa và cũng không forecast lại.

# Đầu vào business mà M4 quan tâm là:

# "Tương lai cần bao nhiêu nguyên liệu?"
# 2. Vậy M4 làm gì với output đó?

# M4 lấy demand tương lai từ M3 rồi đặt nó vào một kho hàng thực tế.

# Ví dụ M3 nói:

# Ngày 13/08:
# Chicken demand = 100 kg

# Ngày 14/08:
# Chicken demand = 120 kg

# Nhưng chỉ biết demand thì chưa thể trả lời cho nhà hàng:

# “Có thiếu hàng không?”

# Bởi vì cần biết thêm:

# Hiện đang có bao nhiêu hàng?
# Hàng nằm trong những lot nào?
# Lot nào sắp hết hạn?
# Ngày mai có hàng nào về?
# Supplier giao lúc nào?
# Có waste không?
# Kho chứa tối đa bao nhiêu?

# M4 ghép tất cả thành:

# M3 Ingredient Demand
#            +
# Current Inventory Lots
#            +
# Inbound Deliveries
#            +
# Expiry dates
#            +
# Waste Events
#            +
# Inventory Policy
#            +
# Cost assumptions
#            ↓
#      INVENTORY SIMULATOR

# Và chạy từng ngày.

# Ví dụ:

# Beginning inventory = 150 kg

# Inbound hôm nay = 30 kg

# Có 20 kg hết hạn

# Demand M3 = 140 kg

# Sau FEFO:

# Available usable = 160 kg

# Demand = 140 kg

# Fulfilled = 140 kg
# Shortage = 0
# Ending = 20 kg

# Đến ngày tiếp theo M4 lại lấy:

# Ending hôm trước
# → Beginning hôm sau

# và tiếp tục simulation.

# Do đó M4 trả lời câu hỏi lớn:

# “Nếu demand forecast của M3 thực sự xảy ra, inventory của ShelfCash sẽ diễn biến ra sao?”

# 3. Tác dụng của M4 trong baseline ShelfCash

# Đây là điểm quan trọng nhất.

# M1–M3 mới chỉ giúp ShelfCash ước lượng nhu cầu.

# M4 biến nhu cầu đó thành business consequence.

# Ví dụ M3 trả:

# Chicken demand ngày mai = 100 kg

# Con số 100 kg một mình chưa cho chủ cửa hàng nhiều quyết định.

# M4 có thể biến nó thành:

# Current usable inventory = 70 kg

# Expected inbound = 10 kg

# Demand = 100 kg

# → shortage = 20 kg
# → fill rate = 80%
# → projected stockout = 13/08

# Hoặc ngược lại:

# Inventory = 250 kg
# Demand = 100 kg

# → không stockout
# → nhưng 80 kg sắp expire
# → expected expiry loss cao

# Đây mới là thứ ShelfCash thực sự cần.

# Có thể coi:

# M1–M3
# "Khách hàng sẽ cần bao nhiêu?"

#              ↓

# M4
# "Với kho hiện tại thì chuyện gì sẽ xảy ra?"

#              ↓

# M5
# "Vậy chúng ta nên mua bao nhiêu?"

# Cho nên M4 là cầu nối giữa forecasting và decision making.

# 4. M4 trong baseline đặc biệt có vai trò gì?

# Có hai baseline rất khác nhau mà code này hỗ trợ.

# Baseline 1: P25 / P50 / P75

# M3 có:

# P25
# P50
# P75

# M4 sẽ tạo ba world:

# LOW_P25
# MEDIAN_P50
# HIGH_P75

# rồi chạy cùng inventory simulator.

# Ví dụ:

# LOW_P25:
# Demand = 80
# → shortage = 0

# MEDIAN_P50:
# Demand = 100
# → shortage = 10

# HIGH_P75:
# Demand = 130
# → shortage = 40

# Nhờ vậy baseline có thể nói:

# Under low-demand case     → inventory OK
# Under median-demand case  → slight shortage
# Under high-demand case    → severe shortage

# Nhưng không được nói:

# P(stockout) = ...

# vì P25/P50/P75 là forecast quantiles, không phải ba mutually exclusive scenarios có probability weights.

# Code adapters.py đang cố tình bảo vệ semantics này bằng:

# probability_weight=None

# và:

# "quantile_is_probability": False

# Đây là một thiết kế rất đúng.

# Baseline 2: probabilistic scenarios

# Nếu M3 tạo:

# 1000 demand worlds

# với weights:

# S001 = 0.001
# S002 = 0.001
# ...

# thì M4 chạy:

# Scenario 1 → inventory consequence
# Scenario 2 → inventory consequence
# ...
# Scenario 1000 → inventory consequence

# rồi mới có thể tính:

# P(stockout)
# Expected shortage
# P95 shortage
# Expected expiry
# P(fill rate < 95%)
# P(capacity violation)
# Expected cost
# CVaR95 cost

# Đây chính là risk propagation.
from shelfcash_forecast.inventory.contracts import (
    InboundDelivery,
    InventoryDemandScenario,
    InventoryKeyRiskMetrics,
    InventoryKeySummary,
    InventoryLot,
    InventorySimulationPackage,
    InventorySimulationPolicy,
    InventorySimulationResult,
    LotExpiryTrace,
    PlannedInboundDelivery,
)
from shelfcash_forecast.inventory.monte_carlo import MonteCarloInventoryRunner
from shelfcash_forecast.inventory.simulator import (
    simulate_inventory,
    simulate_inventory_scenarios,
    simulate_quantile_inventory,
)

__all__ = [
    "InboundDelivery",
    "InventoryDemandScenario",
    "InventoryKeyRiskMetrics",
    "InventoryKeySummary",
    "InventoryLot",
    "InventorySimulationPackage",
    "InventorySimulationPolicy",
    "InventorySimulationResult",
    "LotExpiryTrace",
    "MonteCarloInventoryRunner",
    "PlannedInboundDelivery",
    "simulate_inventory",
    "simulate_inventory_scenarios",
    "simulate_quantile_inventory",
]
