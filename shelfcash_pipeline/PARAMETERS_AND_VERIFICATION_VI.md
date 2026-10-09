# Kiểm tra luồng milestone và danh mục tham số người dùng

Phạm vi: runner `source_code/shelfcash_pipeline/run.ps1` và các input contract runtime
M1–M6/What-if của source hiện tại. Đây không phải danh sách hyperparameter huấn luyện.

## Luồng chạy khi StopAfter

`run.py:run_pipeline` chạy lần lượt:

```text
preprocess -> M1 -> point correction -> M2 -> M3 -> M4 -> M5 -> M6
```

| Mốc dừng | Các stage được thực thi trong cùng lần chạy |
|---|---|
| m1 | preprocess, M1 |
| m2 | preprocess, M1, point, M2 |
| m3 | preprocess, M1, point, M2, M3 |
| m4 | preprocess, M1, point, M2, M3, M4 |
| m5 | preprocess, M1, point, M2, M3, M4, M5 |
| m6 | preprocess, M1, point, M2, M3, M4, M5, M6 |

M1 tạo feature từ dữ liệu canonical, load artifact và gọi model LightGBM `.predict`
cho P25/P50/P75. Point và M2 áp dụng point corrector/CQR đã lưu lên state vừa được
M1 tính. M3 dùng ForecastPackage vừa được M2 tạo và recipe/BOM hiện tại.

M4 nhận ForecastPackage của M2 và kiểm tra M3.is_complete. Nhu cầu từng kịch bản
được tạo từ forecast M2 + residual history artifact, rồi chạy scenario BOM và
FEFO/Monte Carlo. M4 không lấy trực tiếp các số P50 trong ingredient_rows của M3
làm toàn bộ các scenario. Đây là nhánh tính bất định thực sự.

M5 lấy inventory/scenario checkpoint vừa được M4 tạo, supplier offers từ bundle,
và planning config; chạy solver thật cho LEAN/BALANCED/PROTECTED, exact M4
resimulation và critic. Compare chạy cả deterministic và stochastic; M6 dùng
kết quả stochastic nếu có. M6 tạo decision/evidence/explanation thật từ upstream,
nhưng giải thích local/deterministic, không gọi live LLM và không gửi đơn hàng.

Mỗi lần chạy lại sẽ tính lại các stage trước mốc dừng. `-BundlePath` là ngoại lệ:
chỉ load/validate sealed bundle thay vì preprocess raw; M1 trở đi vẫn chạy lại.

`demo` là mode cho phép dùng artifact candidate và giả định demo được chỉ rõ;
không thay M1/M2/BOM/solver bằng output giả. Artifact mặc định hiện có
`promotion_status=trained_candidate`, `production_available_at=null`:
`-ExecutionMode production` sẽ bị chặn ở M1 với ARTIFACT_NOT_PROMOTED_FOR_PRODUCTION.
Pipeline inference load model đã train; không train lại model/point/CQR.

M5 còn đưa `DEMO_CONSEQUENCE_COSTS_NOT_APPROVED` vào unknown_constraints khi
planning mang label demo. Critic coi unknown constraint là hard violation, nên
solver có thể chạy xong nhưng kết quả vẫn là `NO_VALID_PROCUREMENT_PLAN` và M6
không đề xuất đơn được chấp nhận. `status=completed` trong manifest chỉ có nghĩa
pipeline tính xong. Điều này không chứng minh đã có kế hoạch mua hợp lệ.

## Bằng chứng chạy thực tế

Script: `codex_tests/scripts/verify_pipeline_lineage.py`.
Kết quả: `codex_tests/pipeline_lineage_verification_20261006/verification.json`.

Script chạy sáu mốc m1..m6 độc lập từ raw Demo, không mock, với 10 scenario,
10 scenario tối ưu, seed 42, compare và budget 10.000.000 theo ví dụ người dùng.
Expiry/cost vẫn là giả định kỹ thuật có label demo. Python profiler ghi lời gọi
hàm engine thật và kiểm tra identity của object truyền từ producer sang consumer
trong cùng lần chạy, không chỉ dựa vào manifest. Kết quả kiểm chứng chỉ có hiệu lực
cho phiên bản source/artifact/config lúc chạy; không phải chứng minh độ chính xác
dự báo hay sẵn sàng vận hành.

Kết quả sáu mốc: tất cả completed, không có hand-off sai object. Mỗi lần đều
có 35 dòng M1; M3 có 70 dòng ingredient demand; M4 có 10 scenario. M5 (compare)
có 2 lời gọi optimize_procurement, 6 evaluate_candidate_plan và 6 critic calls.
M6 có 1 lời gọi build_final_decision_package. BALANCED solver trả OPTIMAL trong
cả hai mode, nhưng critic báo EXACT_SIMULATION_SAFETY_FLOOR và
UNKNOWN_CONSTRAINT:DEMO_CONSEQUENCE_COSTS_NOT_APPROVED; kết quả M5/M6 là
NO_VALID_PROCUREMENT_PLAN.

`cross_run_checkpoint_hashes.json` trong folder kiểm chứng ghi SHA256:
M1 giống nhau trong 6 lần, point/M2 giống nhau trong 5 lần, M3 giống nhau trong
4 lần. Điều này kiểm tra nội dung checkpoint upstream không thay đổi khi chỉ
đổi mốc dừng với cùng input/model/config.

Script còn kiểm tra What-if demand 1.1 và 1.2 qua service riêng: xác nhận mọi
demand line trong scope được nhân đúng hệ số, baseline không đổi, optimizer
được gọi thật, và M1–M3 không được tính lại trong nhánh What-if.

## File nhập đang có và mức hỗ trợ

| File | Cách dùng | Trạng thái |
|---|---|---|
| `codex_tests/configs/planning_demo.example.json` | `-PlanningConfig` | Budget, bốn hệ số chi phí và optimization_scenario_count thực sự được đọc |
| `source_code/shelfcash_pipeline/context_demo.example.json` | `-ContextMetadataFile` | Xử lý expiry của raw preprocessing/M4 |
| `codex_tests/configs/what_if_budget.example.json` | staged runner `run_07_what_if.ps1 -WhatIfConfig` | Mẫu What-if budget riêng |

Chưa có **một file cấu hình thống nhất** được `run.ps1` đọc cho tất cả budget,
What-if, inventory policy, constraint và strategy. Không có flag WhatIfConfig
hay demand multiplier ở runner này.

`run_07_what_if.ps1` thuộc workflow `codex_tests/scripts/run_01...run_06`, cần
`stage_state.json` và format output của workflow đó. Không đưa thẳng folder
`pipeline_until_m6` của runner mới vào `-RunRoot`; hai format không tương thích.

Các key trong planning example **không được runner mới áp dụng**:
`seed`, `scenario_count`, `scenario_method`, `strategy_profiles`, `capacity_policy`,
`stress` (gồm demand_multiplier/supplier_delay_days). Seed/scenario_count dùng flag
PowerShell. Method hiện cố định residual_bootstrap; strategy dùng default profiles.
Các key mô tả như units/rationale/description/production_readiness chỉ là metadata.

## Tham số hiện nhập được cho runner

| Tham số/key | Ý nghĩa và default | Cần khi nào |
|---|---|---|
| `-StopAfter` | preprocess/m1/point/m2/m3/m4/m5/m6; default m2 | Chọn mốc dừng |
| `-InputPath` | Folder/file raw; default codex_tests/Demo data | Thay dữ liệu đầu vào |
| `-BundlePath` | Bundle có sẵn; loại trừ InputPath và ContextMetadataFile | Khi muốn bỏ raw preprocess |
| `-ArtifactsPath` | Model/encoder/point/CQR/residual đã train; default research artifacts | Đổi bộ model đúng schema |
| `-OutputDir` | Default pipeline_until_<mốc>; chỉ ghi đè output do runner sở hữu | Đổi nơi lưu |
| `-CutoffDate` | YYYY-MM-DD bắt buộc; mốc cuối ngày EOD | Mốc lịch sử dùng forecast/quyết định; không tự chọn tuần DEMO |
| `-Horizon` | Default 7; phải nằm trong horizon của artifact (hiện 1..7) | Số ngày dự báo/mô phỏng/planning |
| `-ExecutionMode` | demo/production/backtest_replay; default demo | Chọn điều kiện artifact |
| `-StoreId` | Không có default; explicit khi raw thiếu scope | Mã cửa hàng từ nguồn hoặc mapping review |
| `-DateLocale` | DMY/MDY/YMD; thiếu thì discovery/review | Locale mơ hồ không tự resolve |
| `-ContextMetadataFile` | JSON metadata, chỉ raw input | Phân loại/xử lý expiry |
| `-PlanningConfig` | JSON planning | Bắt buộc để chạy M5/M6 |
| `-ScenarioCount` | Default 100; 1..2000 | Số kịch bản M4 |
| `-Seed` | Default 42 | Tái lập bootstrap/optimization |
| `-OptimizationMode` | deterministic/stochastic/compare; default compare | M5/M6; stochastic/compare cần >=2 scenario |
| `-Python` | Default .venv-preprocess/Scripts/python.exe | Override interpreter |
| `-Help` | Chỉ hiển thị hướng dẫn | Không chạy pipeline |
| planning.`label` | DEMO_ONLY_NOT_FOR_OPERATION bắt buộc nếu mode demo | M5/M6 |
| planning.`budget` | Số tiền >=0, ví dụ 10000000; null = không giới hạn | Tùy chọn M5/M6 |
| planning.`optimization_scenario_count` | Số scenario đầu của M4 dùng cho M5; bị chặn trên bởi số M4 | M5/M6; default là số M4 nếu thiếu |
| cost_policy.`holding_cost_rate_per_day` | Giá mua × hệ số = chi phí giữ một đơn vị/ngày; mẫu 0.001 | Bắt buộc M5/M6 |
| cost_policy.`shortage_cost_multiplier` | Giá mua × hệ số = chi phí thiếu một đơn vị; mẫu 1.5 | Bắt buộc M5/M6 |
| cost_policy.`expired_cost_multiplier` | Giá mua × hệ số = chi phí hết hạn một đơn vị; mẫu 1.0 | Bắt buộc M5/M6 |
| cost_policy.`waste_cost_multiplier` | Giá mua × hệ số = chi phí waste một đơn vị; mẫu 1.0 | Bắt buộc M5/M6 |
| metadata.`unknown_expiry_policy` | reject hoặc warn_and_place_last; default reject | M4 nếu có lô thiếu expiry |
| metadata.`non_expiring_ingredient_ids` | Danh sách ingredient ID được xác định không theo dõi expiry | Phải bao phủ các ingredient null-expiry nếu dùng warn_and_place_last |
| metadata.`assumption_scope` | Label để ghi nhận giả định demo | Ghi nhận phạm vi, không phải business approval |

Budget dùng cùng đơn vị tiền với supplier price (Demo là VND), không tự quy đổi.
Budget hiện giới hạn **first-stage purchase + delivery**, không tự là trần tổng
mọi recourse/emergency purchase trong các scenario tương lai. Bốn hệ số chi phí
được chuyển thành chi phí theo store/ingredient/unit, dựa trên offer đầu tiên
cho mỗi key đó; runner chưa nhận bảng consequence cost riêng trực tiếp.

## Toàn bộ nhóm thay đổi What-if được engine hỗ trợ

Các trường dưới đây thuộc typed What-if API/staged workflow, **không tự được
áp dụng bằng cách thêm key vào planning JSON của run.ps1**.

| Loại modification | Các trường người dùng nhập |
|---|---|
| `DEMAND_SCALE` | `multiplier` >0; selector gồm `scenario_id`, `store_id`, `ingredient_id`, `unit`, `target_date`, `expected_matches` >=1; phải có ít nhất một scope |
| `BUDGET` | `budget` >=0 hoặc `clear_budget=true` |
| `SUPPLIER_OFFER` | `offer_id`; `available`, `unit_price`, `delivery_cost`, `minimum_order_quantity`, `maximum_order_quantity`, `clear_maximum_order_quantity`, `lead_time_days`, `shelf_life_days`, `clear_shelf_life_days`, `order_cutoff_date`, `clear_order_cutoff_date`, `emergency` |
| `INVENTORY_LOT` | `action` SET_QUANTITY/SET_EXPIRY/ADD/REMOVE; `lot_id`; scope `store_id`, `ingredient_id`, `unit`; `quantity`, `expiry_date`, `clear_expiry`; ADD cần object `lot` |
| `INVENTORY_POLICY` | `expiry_inclusive`, `unknown_expiry`, `accounting_tolerance`, `at_risk_expiry_days`, `waste_threshold`, `fill_rate_target`, `trace_retention` |
| `STRATEGY_PROFILE` | `strategy` LEAN/BALANCED/PROTECTED; `shortage_penalty`, `holding_penalty`, `waste_penalty`, `cash_penalty`, `cvar_weight`, `cvar_alpha`, `maximum_stockout_probability`, `minimum_expected_fill_rate`, `minimum_acceptable_fill_rate` |
| `CONSEQUENCE_COST` | Scope `store_id`, `ingredient_id`, `unit`; `holding_cost_per_unit_day`, `shortage_cost_per_unit`, `expired_cost_per_unit`, `waste_cost_per_unit`, `capacity_quantity`, `clear_capacity_quantity` |
| `STRESS_SCENARIO` | `stress_id` đã có; `demand_multiplier`, `supplier_delay_days`, `supplier_ids`, `preserve_remaining_shelf_life`, `description` |

`1.1` = tăng 10%; `1.2` = tăng 20%; `0.9` = giảm 10%.
DEMAND_SCALE hiện nhân **ingredient demand lines của OptimizationRequest**, không
forecast product M1/M2. Đổi raw data, recipe, cutoff hoặc horizon cần chạy lại
upstream. expected_matches phải đúng số dòng khớp scope, không tùy ý để default 1.
Stress riêng không mang probability weight của kịch bản Monte Carlo.

Envelope của file What-if mẫu: `label`, `actor`, `reason`, `idempotency_key`,
`question`, `modifications`. API còn bind baseline_request_id, baseline request/
decision hashes, execution_mode và confirmed; các ID/hash được service tạo/kiểm
tra. Workflow draft -> confirm -> execute tạo kết quả hypothetical riêng.

Lưu ý workflow staged demo `run_07_what_if.ps1` gọi confirm_what_if trong script
trước khi thực thi. Typed service tự nó vẫn chặn request chưa confirmed.

## Input runtime nâng cao có trong contract nhưng chưa mở đầy đủ ở run.ps1

| Nhóm | Các tham số ngoài dữ liệu lịch sử |
|---|---|
| Kho/inventory policy | expiry_inclusive (true), unknown_expiry (reject), accounting_tolerance (1e-8), at_risk_expiry_days (2), waste_threshold (0), fill_rate_target (0.95), trace_retention (full/summary/selected), trace_scenario_ids |
| StrategyProfile đầy đủ | Các penalty/CVaR/fill-rate trên; thêm minimum_fill_rate, required_fill_rate_probability, maximum_acceptable_stockout_probability, maximum_fill_rate_model_gap, maximum_stockout_probability_model_gap |
| SupplierConstraint | supplier_id; scope store_id/ingredient_id/unit; maximum_total_quantity, maximum_total_cost |
| Supply terms | pack_size, MOQ, unit_price, delivery_cost, maximum_order_quantity, lead_time_days, shelf_life_days, available, order_cutoff_date, emergency; phải đúng base unit/price basis |
| Lô kho/inbound | lot/delivery ID, store/ingredient/unit, quantity, received/arrival date, expiry, unit_cost, location/supplier; initial snapshot đúng cutoff, EOD/BOD |
| ConsequenceCostAssumption | Bốn chi phí tiền/đơn vị (holding tiền/đơn vị/ngày) và capacity_quantity theo store/ingredient/unit |
| OptimizationRequest | existing_inbound, supplier_constraints, cost_assumptions, unit_conversions, inventory_policy, stress_scenarios, stress_base_scenario_id, strategy_profiles, budget, seed, stochastic, allow_mode_fallback, inventory_snapshot_date/boundary |
| M4 engine API | Demand scenario method (residual_bootstrap/gaussian_copula), số scenario, seed, lead-time/shelf-life/yield-loss model khi caller chủ động cung cấp |

Runner hiện đọc pack/MOQ/giá/lead time/shelf life từ supplier_rules trong bundle;
các offer field còn lại chủ yếu theo default của contract. existing_inbound và
supplier_constraints mặc định rỗng. Không replay lịch sử mua thành inbound tương
lai. Các policy/strategy nâng cao cần typed API hoặc nối thêm cấu hình runner.
Thông số train như lag, rolling windows, nominal_coverage và LightGBM params nằm
trong artifact/ForecastConfig; không thay bằng file planning trong một lần inference.

## Ví dụ nhập budget 10 triệu

Trong file planning của bạn, phần runtime cần thiết có thể là:

```json
{
  "label": "DEMO_ONLY_NOT_FOR_OPERATION",
  "budget": 10000000,
  "optimization_scenario_count": 10,
  "cost_policy": {
    "holding_cost_rate_per_day": 0.001,
    "shortage_cost_multiplier": 1.5,
    "expired_cost_multiplier": 1.0,
    "waste_cost_multiplier": 1.0
  }
}
```

Các cost trên giữ giá trị demo; cần thay theo nghiệp vụ nếu muốn dùng thực tế.
File kỹ thuật đã dùng kiểm chứng nằm tại
`codex_tests/pipeline_lineage_verification_20261006/planning_budget_10000000.json`.
