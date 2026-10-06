# ShelfCash Preprocess

Module `shelfcash_preprocess` chuyển file người dùng thành canonical bundle có version, lineage, review và readiness riêng cho M1–M6. Raw input không bị sửa; mỗi lần chạy tạo một thư mục atomic mới.

## 1. Thiết lập

Yêu cầu Python 3.11 trở lên. Từ engine root thực tế:

```powershell
Set-Location "C:\Users\Dell\Downloads\ShelfCash\engine"
& ".\source_code\scripts\setup_preprocess.ps1" -EngineRoot $PWD.Path
```

Script dùng `.venv-preprocess`, không cài global và không ghi đè venv khác. Cài thêm mọi reader optional:

```powershell
& ".\source_code\scripts\setup_preprocess.ps1" -EngineRoot $PWD.Path -WithAllFormats
```

OCR còn cần Tesseract executable và language data `vie`; cài Python extra không tự cài native runtime này.

## 2. Chỉ một nơi điền API key

Mở duy nhất file:

`C:\Users\Dell\Downloads\ShelfCash\engine\.env.preprocess`

Nội dung thông thường:

```dotenv
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
SHELFCASH_PREPROCESS_MODEL=gpt-6.1-sol
SHELFCASH_PREPROCESS_REASONING_EFFORT=medium
SHELFCASH_PREPROCESS_LLM_MODE=live
```

Thay đúng phần `YOUR_OPENAI_API_KEY`; không gửi key vào chat hoặc command line. Process environment có thể override file theo convention deploy, nhưng thao tác local chỉ cần file trên. `.env.preprocess` đã nằm trong `.gitignore`; doctor/log/report chỉ in `SET`, `MISSING` hoặc `PLACEHOLDER_INVALID`.

Kiểm tra offline (không gọi API):

```powershell
.\.venv-preprocess\Scripts\python.exe -m shelfcash_preprocess doctor --config ".\.env.preprocess"
```

Kiểm tra thật auth, model và Structured Outputs bằng đúng một request nhỏ:

```powershell
& ".\source_code\scripts\check_preprocess_api.ps1" -ConfigPath ".\.env.preprocess"
```

`RATE_LIMITED` nghĩa là key được gửi tới API nhưng quota/rate limit chưa cho phép xác minh hoàn tất. `MODEL_UNAVAILABLE` nghĩa là tài khoản chưa có quyền model; đổi `SHELFCASH_PREPROCESS_MODEL=gpt-6-luna` chỉ khi bạn chủ động muốn benchmark/fallback và giữ thay đổi này trong manifest.

## 3. Chạy

Inspect không gọi API:

```powershell
.\.venv-preprocess\Scripts\python.exe -m shelfcash_preprocess inspect `
  --input ".\Demo data" `
  --output ".\preprocess_runs\inspection"
```

LIVE cho Demo data:

```powershell
& ".\codex_tests\scripts\run_preprocess_demo.ps1" -LlmMode live
```

Offline deterministic:

```powershell
& ".\codex_tests\scripts\run_preprocess_demo.ps1" -LlmMode offline
```

File/thư mục/ZIP mới (đường dẫn có khoảng trắng được hỗ trợ):

```powershell
.\.venv-preprocess\Scripts\python.exe -m shelfcash_preprocess run `
  --input "D:\Du lieu\export moi.zip" `
  --output ".\preprocess_runs\customer-a" `
  --store-id "STORE_A" `
  --cutoff-date 2026-08-12 `
  --date-locale DMY `
  --llm-mode live `
  --tenant-id "customer-a"
```

Cutoff là business context, không lấy ngày máy. Không truyền `--store-id` khi source đã có store rõ ràng. Nếu thiếu cả hai, pipeline giữ phần dữ liệu hợp lệ nhưng tạo `STORE_CONTEXT_REQUIRED` thay vì gộp vào `STORE_DEFAULT`.

Exit codes:

- `0`: command thành công và không còn review bắt buộc cho capability đang xét.
- `2`: bundle đã tạo thành công nhưng có `NEEDS_REVIEW`.
- `3`: input/review/bundle không hợp lệ.
- `4`: dependency hoặc config thiếu.
- `5`: LIVE API thất bại; không fallback sang mock.

## 4. Luồng xử lý và boundary tin cậy

Luồng thực thi:

`source inventory + SHA-256` → `format reader` → `table/header discovery` → `profile` → `deterministic/profile mapping` → `LLM semantic proposal nếu LIVE và còn mơ hồ` → `allowlisted transforms` → `entity registry chung` → `cross-table/business validation` → `review` → `canonical bundle` → `engine adapters/readiness`.

Raw, staging và canonical tách thư mục. Lineage giữ SHA-256, file, sheet/page, table ID, source row, transformation IDs và review references. ZIP bị giới hạn số file/kích thước và chặn absolute/`..` path. Text trong cell chỉ là untrusted data; model không được execute code/SQL/shell và chỉ được đề xuất operation trong allowlist.

LLM LIVE dùng OpenAI Responses API với `text.format` JSON Schema strict, `store=false`, timeout và retry hữu hạn. 401/403 không retry vô hạn. Mock chỉ đi qua dependency injection `FakeSemanticClient` trong test; CLI production không có silent mock mode.

## 5. Contract và mapping chính

| Demo/source semantics | Biến đổi | Canonical/engine field |
|---|---|---|
| `Ngày GD` | parse date theo locale | `sales.date` |
| run context hoặc source store | không default âm thầm | `sales.store_id` → engine `store_key` |
| menu `Mã món` + exact normalized name | entity registry chung | `sales.product_id` → engine `product_key` |
| `Tên món / SKU` | trim, exact entity lookup | `product_name` |
| `SLX` | locale-aware number; zero giữ nguyên | `quantity_sold` |
| `Đơn giá bán`, `Doanh thu` | parse độc lập + reconciliation issue | `selling_price`, `revenue` |
| `Hết món?`, `CTKM` | nullable boolean/text | `is_stockout`, `promotion_name` |
| `Date`, Weekend/Holiday/Closed/Promo/Temp/Rain | typed mapping, giữ future provenance | engine calendar names |
| recipe `Món bán`, `Thành phần` | join entity registry | `product_id`, `ingredient_id` |
| recipe quantity/unit/yield/version/effective date | unit alias + date/number validation | real M3 recipe contract |
| usage `Lượng thực dùng` | typed mapping, không fit qua cutoff | M3 yield-loss usage contract |
| snapshot lot | join receipt bằng exact unique lot ID | M4 `InventoryLot` |
| supplier MOQ × pack size | đổi MOQ pack thành base quantity | M5 `SupplierOffer.minimum_order_quantity` |
| historical receipts `<= cutoff` | `record_only_historical` | không replay vào future inbound |

Administrative recipe/entity IDs là SHA-256 ổn định từ registry dùng chung; chúng không tạo quantity/date còn thiếu. Fuzzy/semantic entity match không auto-merge size/combo. `kg↔liter` không đổi nếu thiếu density; pack conversion cần pack size đúng material/supplier. Currency không được tự giả định.

Engine thực tế có các gateway khác nhau:

- M1–M2: `adapt_forecast_input` rồi `validate_sales`/`validate_calendar`.
- M3: `adapt_recipes`; output forecast thật mới được truyền vào `propagate_ingredient_demand`.
- M4: typed `InventoryLot` và demand scenarios; purchase history cũ không phải inbound.
- M5: typed `SupplierOffer`; `OptimizationRequest` chỉ được tạo khi caller cung cấp demand scenarios, strategies và cost assumptions.
- M6: preprocess chỉ chuẩn bị input. Không tuyên bố decision output trước khi M5 được evaluate.

## 6. Review và resume

Trong run directory, mở:

- `review_required.json`: issue, severity, source locator và lý do.
- `mapping_plan.json`: role/mapping proposal và evidence.
- `review_decisions.example.json`: file quyết định đã điền sẵn đúng `run_id`, source hash và mapping hash của chính run đó.

Copy example, không sửa ba field hash:

```powershell
$RunDir = "C:\Users\Dell\Downloads\ShelfCash\engine\preprocess_runs\demo\<run-id>"
Copy-Item "$RunDir\review_decisions.example.json" "$RunDir\review_decisions.json"
notepad "$RunDir\review_decisions.json"
```

Ví dụ thật được generator tạo theo schema:

```json
{
  "schema_version": "1.0.0",
  "run_id": "giữ nguyên từ example",
  "source_inventory_hash": "giữ nguyên từ example",
  "mapping_plan_hash": "giữ nguyên từ example",
  "decisions": [
    {
      "issue_id": "mapping_region_...",
      "action": "ignore_region",
      "value": {"region_id": "region_..."},
      "note": "Đã xác nhận đây chỉ là README/note."
    },
    {
      "issue_id": "capability.inventory.UNKNOWN_EXPIRY_POLICY_REQUIRED",
      "action": "set_metadata",
      "value": {"unknown_expiry_policy": "warn_and_place_last"},
      "note": "Đã xác nhận vật tư bao bì này không theo dõi hạn dùng."
    }
  ]
}
```

Không duyệt policy expiry nếu null thực sự là dữ liệu thiếu của hàng dễ hỏng. Các action hỗ trợ: `approve_mapping`, `set_role`, `map_field`, `map_entity`, `set_metadata`, `ignore_region`, `select_date_locale`, `select_import_semantics`. Quyết định stale bị reject nếu bất kỳ run/source/mapping hash nào khác.

Resume:

```powershell
.\.venv-preprocess\Scripts\python.exe -m shelfcash_preprocess apply-review `
  --run-dir $RunDir `
  --review-file "$RunDir\review_decisions.json"
```

Kết quả là bundle atomic mới cạnh bundle cũ; raw source vẫn nguyên vẹn. Profile chỉ lưu mapping deterministic đủ chắc hoặc mapping đã review. Fingerprint khác, unit drift hoặc critical semantic drift quay lại review. Import ledger SHA-256 ghi nhận reimport; daily summary overlap không có transaction key bị block để chọn append/upsert/replace, không tự cộng.

## 7. Validate, engine smoke và Python API

```powershell
.\.venv-preprocess\Scripts\python.exe -m shelfcash_preprocess validate --bundle $RunDir
.\.venv-preprocess\Scripts\python.exe -m shelfcash_preprocess engine-smoke --bundle $RunDir
```

Python API thật:

```python
from datetime import date
from shelfcash_preprocess import PreprocessService, load_bundle
from shelfcash_preprocess.engine import (
    create_forecast_input,
    create_inventory_lots,
    create_recipe_records,
    create_supplier_offers,
)
from shelfcash_preprocess.models import RunContext

service = PreprocessService.from_config()
bundle = service.run(
    r"C:\Users\Dell\Downloads\ShelfCash\engine\Demo data",
    r"C:\Users\Dell\Downloads\ShelfCash\engine\preprocess_runs\api",
    context=RunContext(store_id="STORE_DEMO", cutoff_date=date(2026, 8, 12), date_locale="DMY"),
    llm_mode="offline",
)

loaded = load_bundle(bundle.run_dir)
forecast_input = create_forecast_input(loaded.run_dir)
recipes = create_recipe_records(loaded.run_dir)
lots, snapshot_date = create_inventory_lots(loaded.run_dir)
offers = create_supplier_offers(loaded.run_dir)
```

`create_optimization_request(...)` cố ý yêu cầu caller truyền `demand_scenarios`, `strategy_profiles`, `cost_assumptions` và planning horizon thật; module không chế budget/penalty/cost.

## 8. Bundle output

Mỗi run chứa:

- `manifest.json`: schema/model/prompt/profile version, mode, usage, context, completed steps và readiness.
- `source_inventory.json`, `raw/files/`: immutable hashes và raw copy.
- `tables.json`, `mapping_plan.json`: region/profile/mapping evidence.
- `review_required.json`, `review_decisions.example.json`.
- `quality_report.json`, `summary.md`.
- `canonical/*.csv`, `lineage/records.json`, `quarantine/`.
- `engine_inputs/*.csv`, `engine_inputs/index.json`.
- `engine_smoke_report.json` sau khi chạy smoke.

Bundle chỉ load được khi `manifest.bundle_complete=true`; folder `.partial` không được hiểu là ready.

## 9. Trạng thái format và giới hạn thực tế

- XLSX/XLSM: reader thật, mọi sheet kể cả hidden (flagged), typed/cached formula values, merged-cell metadata; macro không chạy. Đã smoke bằng 6 workbook Demo.
- CSV/TSV, JSON/JSONL: reader thật; BOM/encoding/delimiter/nested records có test.
- PDF text: `pdfplumber`, đã smoke bằng fixture PDF thật.
- PNG/JPEG/PDF scan: đường thực thi `pytesseract` `vie+eng` đã test qua backend adapter fake; native Tesseract/model Việt chưa có trên máy nên **OCR_NOT_RUNTIME_VERIFIED**. Thiếu runtime trả `OCR_UNAVAILABLE` với lệnh cài, không báo thành công.
- XLS và Parquet: adapter thật nhưng optional `xlrd`/`pyarrow` chưa cài trong base venv; dùng `-WithAllFormats`.
- Demo không có trained M1/M2 artifact. Engine smoke xác nhận input contract, không tuyên bố inference/model runtime ready.
- M4 Demo còn một lot bao bì không có expiry. Cần review policy; không tự gán `date.max` vào canonical.
- M5 có supplier terms nhưng thiếu explicit optimization strategies/cost assumptions/budget policy. Supplier offers validate được; OptimizationRequest/solver vẫn `NEEDS_REVIEW`.
- M6 bị block hợp lệ cho tới khi có M5 evaluation. Opportunity/Candidate Catalog nằm ngoài scope.

OpenAI implementation bám tài liệu chính thức về [gpt-6.1-sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol), [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs) và [Responses API migration](https://developers.openai.com/api/docs/guides/migrate-to-responses).
