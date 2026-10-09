# ShelfCash — chạy M1 đến M6 từ checkout

Repo này có source, runner, full planning config, 6 workbook DEMO và bộ model/point/CQR/residual cố định.
Không cần thư mục `engine/codex_tests` ở bên ngoài repo, không training lại để chạy demo.

## Thiết lập một lần trên Windows

Mở PowerShell tại thư mục vừa clone. Python tối thiểu3.11; bộ dependency lock được kiểm thử
với Python3.14.4 trên Windows. Cài môi trường riêng của repo:

```powershell
.\scripts\setup.ps1
```

Script tạo `.venv`, cài dependencies, cài editable project và kiểm tra config.
Python3.12+ dùng `requirements-demo.lock.txt`; Python3.11 dùng `[demo,test]` từ pyproject
vì numpy/scipy trong lock yêu cầu3.12+. Runtime đã kiểm thử là Python3.14.4; các phiên bản khác
chưa được chạy E2E trong lần bàn giao này. Có thể cài dependencies tương thích từ pyproject:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[demo,test]"
```

Không cần API key cho luồng DEMO này. `.env.preprocess.example` là mẫu cho các capability khác;
không đưa `.env`, credentials hoặc môi trường Python lên Git.

## Mỗi lần chạy chỉ một lệnh

Chọn mốc dừng:

```powershell
.\run.ps1 -StopAfter m1
.\run.ps1 -StopAfter m2
.\run.ps1 -StopAfter m3
.\run.ps1 -StopAfter m4
.\run.ps1 -StopAfter m5
.\run.ps1 -StopAfter m6
```

Có thêm `preprocess` và `point`. M2 tự chạy point correction rồi CQR.
Mỗi lần chạy từ raw input đến mốc chọn; M6 giải thích kết quả M5, không tạo fallback orders.
`run.ps1` dùng `.venv` của repo hoặc active virtualenv; có thể truyền `-Python <python.exe>`.

Output: `outputs/pipeline_until_<mốc>/`. Chạy lại cùng mốc thay output của chính mốc đó,
theo owner/lock/path guards. Output và `.runtime` bị Git ignore.
M5 xuất `m5/customer_plan/`: XLSX, CSV, JSON, Markdown, conditions, daily inventory ledger và terminal expiry cho các thực thể applicable. Accepted zero-purchase có đủ gói; trường hợp chưa accepted xuất diagnostics, không gắn nhãn zero-purchase.
Ví dụ sau `-StopAfter m6`, mở `outputs/pipeline_until_m6/m5/customer_plan/customer_procurement_plan.xlsx`.
Kết quả JSON lớn được ghi theo từng phần/model để giảm bộ nhớ; vẫn giữ đầy đủ nội dung đánh giá.
Advanced CLI `shelfcash-pipeline` giữ default `codex_tests/runs/` tương đối trong checkout;
thư mục này cũng bị Git ignore. Root `run.ps1` dùng `output_root` trong cấu hình.

Contract tổng quát, cold-start, giới hạn search và status ở [SUPPORTED_INPUT_CONTRACT.md](SUPPORTED_INPUT_CONTRACT.md). Public typed service, What-if, migration và BE/FE ở [BE_FE_HANDOFF_VI.md](BE_FE_HANDOFF_VI.md); schema/examples ở `schemas/generalization_v2`. Cấu hình DEMO/P6 v3 trong repo là ví dụ có assumptions, không phải facts mặc định cho khách hàng mới. Generic profile builder yêu cầu scoped occupancy/pack/price/calendar registries có provenance. Để giữ historical outputs, tạo cấu hình với `pipeline.output_root` mới và chạy `run.ps1 -ConfigFile <file> -StopAfter m6`.

## Chỉnh tham số tại một file

Sửa `shelfcash.config.json`:

| Nội dung | Trường |
|---|---|
| Ngân sách VND | `planning.budget`: null không cap;0 cap0;10000000 cap10 triệu |
| Snapshot/horizon | `pipeline.cutoff_date`, `pipeline.horizon` |
| Seed | `pipeline.seed` và `planning.seed` phải khớp |
| Số scenarios M4 | `pipeline.scenario_count` và `planning.scenario_count` phải khớp |
| Số scenarios tối ưu | `planning.optimization_scenario_count` |
| Mode | `pipeline.optimization_mode`: deterministic/stochastic/compare |
| Solver limits | `planning.limits` |
| Costs/calendar/rules/packing/prices/contracts | Các trường trong `planning` cùng versioned assumptions |

```powershell
.\run.ps1 -ValidateOnly
```

Budget sửa độc lập. Behavioral assumptions phải có profile version/value/hash binding hợp lệ;
engine không âm thầm rebind khi chỉ sửa một phía. Typed What-if bên dưới quản lý binding mới.
Đường dẫn tương đối tính từ repo root, không phụ thuộc tên folder clone.

Default: P6_CONTINUITY v3 / SCENARIO_PREVIEW, budget=null,100 worlds,seed42,compare.
Snapshot12/08/2026 EOD; horizon **13–19/08/2026**. Đây là demo theo điều kiện chưa được NCC/nghiệp vụ xác nhận,
không phải đơn vận hành tuần hiện tại. Business_ready=false, execution_authorized=false.
Không có model promotion, supplier order execution hoặc live LLM/API trong pipeline này.

## What-if

Sau khi chạy M6, chỉnh `configs/what_if_budget.example.json`, rồi:

```powershell
.\run.ps1 -WhatIf
```

Default dùng baseline `outputs/pipeline_until_m6/`. Có thể truyền `-BaselineDir <run>` và `-ConfigFile <json>`.
`-WhatIf -ValidateOnly` chỉ kiểm tra cấu hình. File dùng typed modifications BUDGET,DEMAND_SCALE,
SUPPLIER_OFFER,INVENTORY_LOT,CONSEQUENCE_COST,INVENTORY_POLICY,STRATEGY_PROFILE,STRESS_SCENARIO.
What-if tính lại qua public M5/exact simulator/critic, giữ baseline và xuất comparison JSON
trong `outputs/what_if_<timestamp_id>/`; không tự xuất customer workbook hoặc gửi supplier PO.
What-if không recompute forecast/BOM. Cần raw/context mới nếu thay factual forecast origin hoặc recipes.

Rule-based global ingredient demand có CLI riêng, đi qua registry draft/execute/comparison hiện có:

```powershell
python -B -m shelfcash_pipeline.demand_what_if --baseline outputs/pipeline_until_m6 --output outputs/whatif_new_run --multiplier 1.1 --scope ALL_BASELINE_INGREDIENT_DEMAND --execute-hypothetical --deny-network --actor reviewer --reason hypothetical_only --idempotency-key new_run
```

Bỏ `--execute-hypothetical` để chỉ tạo typed draft. `--text` nhận ba dạng scope toàn bộ kỳ: “bằng 1.1 lần baseline”, “tăng ... 10%”, “nhân 1,1”.
CLI giữ pool/world weights, facts và policies; export ghi parent lineage và transformed profile, không rebuild từ raw.
Nó cũng exact-evaluate kế hoạch cũ trên nhu cầu mới để đối chiếu. `business_ready` và `execution_authorized` vẫn false.
Decomposition thử phân bổ đều union-risk trước; nếu projection INFEASIBLE, có tối đa một retry với trần của strategy gốc.
Joint fixed certification và exact critic vẫn kiểm tra toàn bộ ràng buộc gốc; retry không chứng minh global cost optimum hoặc infeasibility của bài toán đầy đủ.

## Python API

```python
from pathlib import Path
from shelfcash_pipeline.config_runner import load_configuration, configured_options
from shelfcash_pipeline.run import run_pipeline

root = Path.cwd()
config = load_configuration(root / "shelfcash.config.json")
output = run_pipeline(configured_options(config, "m5", engine_root=root))
```

Console entry points sau install: `shelfcash-run`, `shelfcash-pipeline`, `shelfcash-forecast`, `shelfcash-preprocess`.
Các APIs What-if/approval nằm trong `shelfcash_forecast.decision_intelligence`;
technical acceptance không tự business approve hoặc authorize execution.

## Tests và assets

```powershell
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider tests -q
```

`demo/ASSETS_MANIFEST.json` lưu nguồn/hash của27 files. Fixed artifact checksum manifest giữ nguyên;
`.gitattributes` ngăn Git đổi line endings của model/input và làm hỏng checksum.
Metadata của model có historical lineage; các source locators không phải runtime dependency tới máy tác giả.
Không cần push output, audit folders lớn, `.venv`, local preprocess state hoặc credentials.
`HANDOFF_VALIDATION.json` ghi kết quả kiểm thử và giới hạn môi trường của lần bàn giao.

Hướng dẫn trong `README_PREPROCESS_VI.md` và `shelfcash_pipeline/README.md` có phần lịch sử về outer engine layout.
Dùng README này và root `run.ps1` cho checkout độc lập.
