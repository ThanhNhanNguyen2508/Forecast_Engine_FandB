# Chạy pipeline ShelfCash và dừng ở checkpoint

Runner là lớp điều phối mỏng; thuật toán nằm trong `shelfcash_preprocess` và
`shelfcash_forecast`. Mỗi file stage gọi code engine hiện tại, không sao chép
thuật toán forecast, CQR, BOM, FEFO, solver hoặc giải thích.

```text
run.ps1 -> __main__.py -> run.py:run_pipeline
  preprocess.py -> PreprocessService.run (offline) / load_bundle
  m1.py         -> inference_pipeline.predict_m1
  point_correction.py -> correct_forecast_point
  m2.py         -> calibrate_forecast -> build_forecast_package
  m3.py         -> bom.engine.propagate_ingredient_demand
  m4.py         -> residual bootstrap -> scenario BOM -> MonteCarloInventoryRunner
  m5.py         -> optimize_procurement -> exact M4 resimulation + critic
  m6.py         -> build_final_decision_package (deterministic/local)
```

Đọc `run.py` trước để thấy thứ tự gọi, rồi đọc file của milestone bạn muốn hiểu.
`context.py` quản lý tham số/output; các checkpoint chỉ chứa dữ liệu và kết quả.

## Folder cố định và ghi đè

Default output nằm trong `C:\Users\Dell\Downloads\ShelfCash\engine\codex_tests\runs`:

| StopAfter | Folder |
|---|---|
| preprocess | pipeline_until_preprocess |
| m1 | pipeline_until_m1 |
| point | pipeline_until_point |
| m2 | pipeline_until_m2 |
| m3 | pipeline_until_m3 |
| m4 | pipeline_until_m4 |
| m5 | pipeline_until_m5 |
| m6 | pipeline_until_m6 |

**Chạy lại cùng mốc sẽ xóa output cũ trong đúng folder do runner sở hữu và tính
lại pipeline.** Không thêm folder timestamp ở cấp `runs`. Một bundle preprocess
vẫn có run_id nội bộ theo contract engine; bundle cũ được thay thế cùng output,
không tích lũy qua các lần chạy.

Marker `.shelfcash_pipeline.json` ghi owner, đường dẫn và milestone. Runner từ
chối ghi đè folder không có marker hợp lệ, folder thuộc mốc khác, symlink/junction,
source, input, artifacts hoặc config đang đọc. Run lịch sử không có marker này
nên không bị thay thế. Không tự dọn các run cũ từ những phiên trước.

`.pipeline.lock` chặn hai process ghi cùng folder. Lệnh hoàn tất/lỗi sẽ nhả lock.
Nếu process bị kill đột ngột, kiểm tra PID trong lock và chỉ xóa lock khi process
đó đã dừng; không xóa marker để tiếp tục ghi đè.

## Lệnh cơ bản

Chạy từ engine root. Wrapper dùng `.venv-preprocess\Scripts\python.exe`, đặt
PYTHONPATH vào source hiện tại rồi khôi phục environment; không cần cài lại.

```powershell
# READ-ONLY
& .\source_code\shelfcash_pipeline\run.ps1 -Help

# WRITES / OVERWRITES MANAGED OUTPUT: chọn một mốc để chạy.
& .\source_code\shelfcash_pipeline\run.ps1 -StopAfter preprocess
& .\source_code\shelfcash_pipeline\run.ps1 -StopAfter m1
& .\source_code\shelfcash_pipeline\run.ps1 -StopAfter point
& .\source_code\shelfcash_pipeline\run.ps1 -StopAfter m2
& .\source_code\shelfcash_pipeline\run.ps1 -StopAfter m3
```

M1 dừng trước point/CQR. M2 tự chạy point correction trước CQR. Không truyền
StopAfter thì mặc định vẫn là `m2`.

Raw Demo chưa có quyết định về null expiry; lệnh M4 trần sẽ dừng với
`UNKNOWN_EXPIRY_POLICY_REQUIRED`, giữ checkpoint tới M3. M5/M6 còn cần file
planning về chi phí. Runner không tự approve review hay tự chọn các giả định này.

## Raw Demo tới M4–M6 với cấu hình demo được chọn rõ ràng

File `context_demo.example.json` là **giả định kỹ thuật demo**, đối chiếu metadata
của historical reviewed Demo bundle. Nó phân loại `ING_be9b1adc1759` là nguyên liệu
không theo dõi expiry, dùng `warn_and_place_last`, và giữ nhãn
`DEMO_ONLY_NOT_FOR_OPERATION`. Nó không phải quyết định nghiệp vụ đã được duyệt.
Đọc file trước; chỉ truyền nếu bạn muốn test đúng fixture này. Runner không tự
load file đó. Với dữ liệu khác, cung cấp metadata đúng dữ liệu của bạn.

```powershell
$ContextDemo = '.\source_code\shelfcash_pipeline\context_demo.example.json'
$PlanningDemo = '.\codex_tests\configs\planning_demo.example.json'

# WRITES / OVERWRITES MANAGED OUTPUT: raw Demo -> M4; chọn giả định expiry demo.
& .\source_code\shelfcash_pipeline\run.ps1 -StopAfter m4 `
  -ContextMetadataFile $ContextDemo

# WRITES / OVERWRITES MANAGED OUTPUT: thêm cấu hình cost demo cho M5.
& .\source_code\shelfcash_pipeline\run.ps1 -StopAfter m5 `
  -ContextMetadataFile $ContextDemo -PlanningConfig $PlanningDemo

# WRITES / OVERWRITES MANAGED OUTPUT: chạy đến giải thích M6, không đặt đơn hàng.
& .\source_code\shelfcash_pipeline\run.ps1 -StopAfter m6 `
  -ContextMetadataFile $ContextDemo -PlanningConfig $PlanningDemo
```

Đây là các lựa chọn riêng. Mỗi lệnh chạy lại từ preprocess tới mốc được chọn,
không tự resume trạng thái Python của lần trước. CSV/JSON checkpoint phục vụ đọc,
kiểm tra và so sánh. M6 chỉ đọc kết quả M5 và upstream đã có trong lần chạy đó.

## Tái dùng bundle

```powershell
# READ-ONLY: lấy bundle từ checkpoint preprocess cố định.
$PreprocessRoot = '.\codex_tests\runs\pipeline_until_preprocess'
$Bundle = (Get-Content -LiteralPath (Join-Path $PreprocessRoot 'preprocess_summary.json') -Raw -Encoding UTF8 | ConvertFrom-Json).bundle_path

# WRITES / OVERWRITES MANAGED OUTPUT: dùng bundle đó, tính lại M1 tới M3.
& .\source_code\shelfcash_pipeline\run.ps1 -BundlePath $Bundle -StopAfter m3
```

`-BundlePath` bỏ qua raw preprocessing, chỉ load/validate sealed bundle. Không
được truyền ContextMetadataFile cùng bundle có sẵn, vì runner không sửa bundle.
Bundle cho M4 phải có readiness phù hợp và inventory snapshot đúng cutoff. Không
truyền bundle nằm trong chính output sắp ghi đè. Chạy lại preprocess sẽ làm hết
hiệu lực đường dẫn bundle cũ bên trong `pipeline_until_preprocess`.

## Tham số

Xem [PARAMETERS_AND_VERIFICATION_VI.md](PARAMETERS_AND_VERIFICATION_VI.md) để đọc
luồng dữ liệu giữa các milestone, bằng chứng chạy thật và danh mục input runtime/
What-if. `budget` trong planning JSON được áp dụng; `stress.demand_multiplier`,
`capacity_policy` và `strategy_profiles` của file example chưa được runner này
áp dụng. What-if có service/workflow riêng, chưa có flag ở `run.ps1`.

| PowerShell | Python CLI | Default / ý nghĩa |
|---|---|---|
| -StopAfter | --stop-after | preprocess, m1, point, m2, m3, m4, m5, m6; default m2 |
| -InputPath | --input | codex_tests/Demo data |
| -BundlePath | --bundle | sealed bundle có sẵn; loại trừ InputPath |
| -ArtifactsPath | --artifacts | m1_m2_research_20261004T054706Z/artifacts |
| -OutputDir | --output-dir | pipeline_until_<milestone>; custom output cũng phải do runner sở hữu để ghi đè |
| -CutoffDate | --cutoff-date | 2026-08-12, inclusive EOD |
| -Horizon | --horizon | 7; không vượt artifact config |
| -ExecutionMode | --execution-mode | demo; cũng có production/backtest_replay, giữ nguyên core checks |
| -StoreId / -DateLocale | --store-id / --date-locale | STORE_A / DMY, fallback preprocess |
| -ContextMetadataFile | --context-metadata | JSON object RunContext.metadata; raw input only |
| -PlanningConfig | --planning-config | JSON planning explicit; bắt buộc cho M5/M6 |
| -ScenarioCount | --scenario-count | 100; 1..2000, M4 diagnostic scenarios |
| -Seed | --seed | 42 |
| -OptimizationMode | --optimization-mode | compare; deterministic hoặc stochastic |

Python tương đương: đặt PYTHONPATH tới source_code rồi
`python -B -m shelfcash_pipeline --stop-after m3`. Entry point khi cài package là
`shelfcash-pipeline`.

M5 dùng format planning của application hiện có: `label`, `cost_policy` gồm bốn
rate/multiplier, `budget`, và `optimization_scenario_count`. Strategy profiles lấy
từ `optimization.strategies.default_strategy_profiles` (LEAN/BALANCED/PROTECTED).
Các mô tả `capacity_policy`/`stress` trong example không được biến thành constraint
hay stress runtime ở runner này, giống đường application hiện tại.

Compare chạy deterministic và stochastic trên cùng scenario subset; M6 chọn
stochastic nếu có, giống application. Không tự fallback mode khi thiếu scenario.
`same_sample_optimism=true` được ghi rõ; kết quả không phải kiểm định risk OOS.

## Output và tiêu chí checkpoint

```text
pipeline_until_m6/
  .shelfcash_pipeline.json
  run_manifest.json
  preprocess_summary.json
  state/                           # profile/ledger riêng, reset khi chạy lại
  preprocess/bundle/<run_id>/       # sealed bundle, không ghi checkpoint vào đây
  engine_inputs/*.csv
  m1/forecast_rows.csv, summary.json, model_context.json
  point/forecast_rows.csv, summary.json
  m2/forecast_rows.csv, forecast.json, summary.json
  m3/ingredient_demand.json, ingredient_rows.csv, summary.json
  m4/product_scenarios.json, ingredient_scenarios.json, inventory_report.json, summary.json
  m5/optimization_result.json, planning_config.json, summary.json
  m6/decision_package.json, summary.json
```

Chỉ có các stage đã chạy. `run_manifest.json` ghi `active_milestone`,
`completed_milestones`, `status`, `stopped_after`; lỗi có `<stage>/failure.json`.
Chạy thành công: status completed, stopped_after đúng mốc. M1/M2 cần zero duplicate,
nonfinite, negative/crossing violations; M3 cần is_complete; M4 cần đúng số scenario,
weights và cửa sổ EOD+1; M5/M6 cần đọc cả status/recommendation/critic, không chỉ
nhìn process exit. `NO_VALID_PROCUREMENT_PLAN` là kết quả thực, không tạo đơn thay thế.

## Giới hạn và sửa code sau này

- Không train/retrain/search LightGBM, fit point corrector/CQR, hay fit yield-loss.
- M4 chỉ bootstrap residual artifact đã lưu và dùng `FixedRecipeYieldLossModel`.
  Loss/waste trong recipe vẫn được BOM áp dụng. Khác với API scenario tổng quát
  có thể fit empirical yield-loss từ usage history, runner này không học thêm model.
- M6 chỉ dùng generator deterministic/local; không có flag bật live LLM/API,
  không approve hay thực thi procurement, không what-if/regret/agent workflows.
- Candidate res_07 vẫn chưa promoted; production bị core từ chối. Interval guardrail
  lịch sử vẫn fail; checkpoint hoàn tất không chứng minh coverage/accuracy đã đạt.
- Summary M1/M2 không tính MAE/WAPE/Bias/coverage vì không có future actuals.
- `business_ready=false` được giữ cho runner học/test; metadata/cost demo không
  được coi là business approval. Không tự thay dữ liệu để vượt validation.

Sửa CQR ở `shelfcash_forecast/calibration/cqr.py` và integration ở
`pipeline/inference_pipeline.calibrate_forecast`. Sửa BOM/FEFO/optimizer/M6 ở các
module engine tương ứng. Khi giữ public contracts, các wrapper vẫn dùng lại được.
Đổi artifact schema cần cập nhật loader/version đúng cách; runner không fit lại
hay tự chuyển sang artifacts khác.


Bước 1 :
Set-Location 'C:\Users\Dell\Downloads\ShelfCash\engine'                                                        
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
Bước 2 :
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& c:\Users\Dell\Downloads\ShelfCash\engine\.venv-preprocess\Scripts\Activate.ps1)                           
Bước 3 : 
& .\source_code\shelfcash_pipeline\run.ps1 -Help

# WRITES / OVERWRITES MANAGED OUTPUT: chọn một mốc để chạy.
& .\source_code\shelfcash_pipeline\run.ps1 -StopAfter preprocess
& .\source_code\shelfcash_pipeline\run.ps1 -StopAfter m1
& .\source_code\shelfcash_pipeline\run.ps1 -StopAfter point
& .\source_code\shelfcash_pipeline\run.ps1 -StopAfter m2
& .\source_code\shelfcash_pipeline\run.ps1 -StopAfter m3

```powershell
$ContextDemo = '.\source_code\shelfcash_pipeline\context_demo.example.json'
$PlanningDemo = '.\codex_tests\configs\planning_demo.example.json'

# WRITES / OVERWRITES MANAGED OUTPUT: raw Demo -> M4; chọn giả định expiry demo.
& .\source_code\shelfcash_pipeline\run.ps1 -StopAfter m4 `
  -ContextMetadataFile $ContextDemo

# WRITES / OVERWRITES MANAGED OUTPUT: thêm cấu hình cost demo cho M5.
& .\source_code\shelfcash_pipeline\run.ps1 -StopAfter m5 `
  -ContextMetadataFile $ContextDemo -PlanningConfig $PlanningDemo

# WRITES / OVERWRITES MANAGED OUTPUT: chạy đến giải thích M6, không đặt đơn hàng.
& .\source_code\shelfcash_pipeline\run.ps1 -StopAfter m6 `
  -ContextMetadataFile $ContextDemo -PlanningConfig $PlanningDemo
```