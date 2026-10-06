from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from shelfcash_preprocess.models import CellRange, TableProfile, TableRegion
from shelfcash_preprocess.readers import RawGrid


HEADER_HINTS = {
    "date", "ngày", "product", "món", "sku", "quantity", "slx", "tên",
    "unit", "đvt", "đv", "ingredient", "thành phần", "vendor", "ncc",
    "holiday", "weekend", "batch", "recipe", "giá", "doanh thu", "hết món",
}
TOTAL_MARKERS = {"total", "grand total", "subtotal", "tổng", "tổng cộng", "cộng"}


@dataclass
class RegionData:
    region: TableRegion
    frame: pd.DataFrame
    source_rows: list[int]


def _empty(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or (
        isinstance(value, str) and not value.strip()
    )


def _rectangular(rows: list[list[Any]]) -> list[list[Any]]:
    width = max((len(row) for row in rows), default=0)
    return [row + [None] * (width - len(row)) for row in rows]


def _segments(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    result: list[tuple[int, int]] = []
    start = previous = indices[0]
    for value in indices[1:]:
        if value != previous + 1:
            result.append((start, previous))
            start = value
        previous = value
    result.append((start, previous))
    return result


def _components(rows: list[list[Any]]) -> list[tuple[int, int, int, int]]:
    """Find rectangular table candidates separated by wholly blank rows/columns."""

    if not rows:
        return []
    height, width = len(rows), len(rows[0])
    nonempty_cols = [c for c in range(width) if any(not _empty(rows[r][c]) for r in range(height))]
    components: list[tuple[int, int, int, int]] = []
    for c0, c1 in _segments(nonempty_cols):
        active_rows = [
            r for r in range(height) if any(not _empty(rows[r][c]) for c in range(c0, c1 + 1))
        ]
        for r0, r1 in _segments(active_rows):
            components.append((r0, r1, c0, c1))
    return components


def _kind(value: Any) -> str:
    if _empty(value):
        return "null"
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return "date"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float, np.number)):
        return "number"
    return "string"


def _header_score(block: list[list[Any]], row_index: int) -> float:
    row = block[row_index]
    nonempty = [value for value in row if not _empty(value)]
    if len(nonempty) < 2:
        return -1.0 + 0.05 * len(nonempty)
    strings = [str(value).strip() for value in nonempty if _kind(value) == "string"]
    score = len(nonempty) / max(len(row), 1)
    score += 1.2 * len(strings) / len(nonempty)
    score += 0.4 * len(set(strings)) / max(len(strings), 1)
    tokens = " ".join(strings).lower()
    score += min(1.2, 0.25 * sum(hint in tokens for hint in HEADER_HINTS))
    following = block[row_index + 1 : row_index + 4]
    if following:
        type_changes = 0
        compared = 0
        for col, value in enumerate(row):
            if _empty(value):
                continue
            below = [_kind(next_row[col]) for next_row in following if not _empty(next_row[col])]
            if below:
                compared += 1
                type_changes += int(any(kind != "string" for kind in below))
        score += 0.7 * type_changes / max(compared, 1)
    return score


def _unique_headers(values: list[Any]) -> list[str]:
    output: list[str] = []
    counts: dict[str, int] = {}
    for index, value in enumerate(values, 1):
        base = re.sub(r"\s+", " ", str(value).strip()) if not _empty(value) else f"column_{index}"
        count = counts.get(base, 0) + 1
        counts[base] = count
        output.append(base if count == 1 else f"{base}__{count}")
    return output


def _is_repeated_header(row: list[Any], headers: list[str]) -> bool:
    normalized = [re.sub(r"\s+", " ", str(v).strip()) if not _empty(v) else "" for v in row]
    expected = [header.split("__", 1)[0] for header in headers]
    matches = sum(left == right for left, right in zip(normalized, expected, strict=False))
    return matches >= max(2, math.ceil(len(expected) * 0.7))


def _is_total_row(row: list[Any]) -> bool:
    texts = [str(value).strip().lower() for value in row if not _empty(value)]
    return bool(texts) and texts[0] in TOTAL_MARKERS


def _json_value(value: Any) -> Any:
    if _empty(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def profile_frame(frame: pd.DataFrame) -> TableProfile:
    inferred: dict[str, str] = {}
    null_ratio: dict[str, float] = {}
    unique_count: dict[str, int] = {}
    numeric_ranges: dict[str, dict[str, float | None]] = {}
    date_ranges: dict[str, dict[str, str | None]] = {}
    anomalies: list[str] = []
    for column in frame.columns:
        values = frame[column]
        observed = values.loc[~values.map(_empty)]
        kinds = sorted({_kind(value) for value in observed})
        inferred[column] = "|".join(kinds) if kinds else "null"
        null_ratio[column] = round(1 - len(observed) / max(len(values), 1), 6)
        unique_count[column] = len({json.dumps(_json_value(value), ensure_ascii=False, default=str) for value in observed})
        numeric = pd.to_numeric(observed, errors="coerce").dropna()
        if len(numeric) and len(numeric) >= max(1, int(len(observed) * 0.8)):
            numeric_ranges[column] = {
                "min": float(numeric.min()),
                "max": float(numeric.max()),
                "mean": float(numeric.mean()),
            }
        date_named = "date" in column.lower() or "ngày" in column.lower() or "nct" == column.lower()
        valid_dates = pd.to_datetime(observed, errors="coerce").dropna() if date_named else pd.Series(dtype="datetime64[ns]")
        if len(valid_dates) and len(valid_dates) >= max(1, int(len(observed) * 0.8)):
            date_ranges[column] = {
                "min": valid_dates.min().isoformat(),
                "max": valid_dates.max().isoformat(),
            }
        if null_ratio[column] > 0.95:
            anomalies.append(f"MOSTLY_NULL:{column}")
    positions = sorted({0, 1, len(frame) // 4, len(frame) // 2, (3 * len(frame)) // 4, max(0, len(frame) - 2), max(0, len(frame) - 1)})
    samples = [
        {column: _json_value(frame.iloc[position][column]) for column in frame.columns}
        for position in positions
        if 0 <= position < len(frame)
    ][:9]
    return TableProfile(
        row_count=len(frame),
        column_count=len(frame.columns),
        columns=list(frame.columns),
        inferred_types=inferred,
        null_ratio=null_ratio,
        unique_count=unique_count,
        numeric_ranges=numeric_ranges,
        date_ranges=date_ranges,
        samples=samples,
        anomalies=anomalies,
    )


def discover_regions(grids: list[RawGrid]) -> list[RegionData]:
    output: list[RegionData] = []
    for grid in grids:
        rows = _rectangular(grid.rows)
        for component_index, (r0, r1, c0, c1) in enumerate(_components(rows), 1):
            block = [row[c0 : c1 + 1] for row in rows[r0 : r1 + 1]]
            if len(block) < 2:
                continue
            search_limit = min(12, len(block) - 1)
            scores = [_header_score(block, index) for index in range(search_limit)]
            header_local = max(range(len(scores)), key=scores.__getitem__)
            confidence = max(0.0, min(1.0, (scores[header_local] + 1) / 4.5))
            header_rows = [r0 + header_local + 1]
            header_values = block[header_local]
            # Combine a genuine multi-level header, but never fill merged/title
            # cells blindly across the table boundary.
            if header_local > 0:
                previous = block[header_local - 1]
                previous_nonempty = sum(not _empty(value) for value in previous)
                current_blanks = sum(_empty(value) for value in header_values)
                if previous_nonempty >= 2 and current_blanks:
                    combined = []
                    for upper, lower in zip(previous, header_values, strict=True):
                        parts = [str(v).strip() for v in (upper, lower) if not _empty(v)]
                        combined.append(" / ".join(parts))
                    header_values = combined
                    header_rows.insert(0, r0 + header_local)
            headers = _unique_headers(header_values)
            data_rows: list[list[Any]] = []
            source_rows: list[int] = []
            ignored_totals = 0
            repeated_headers = 0
            for absolute_index, row in enumerate(block[header_local + 1 :], r0 + header_local + 2):
                if all(_empty(value) for value in row):
                    continue
                if _is_repeated_header(row, headers):
                    repeated_headers += 1
                    continue
                if _is_total_row(row):
                    ignored_totals += 1
                    continue
                data_rows.append(row)
                source_rows.append(absolute_index)
            frame = pd.DataFrame(data_rows, columns=headers)
            digest = hashlib.sha256(
                f"{grid.source_id}|{grid.container}|{r0}|{r1}|{c0}|{c1}".encode()
            ).hexdigest()[:16]
            region_id = f"region_{digest}"
            evidence = [
                f"header_score={scores[header_local]:.3f}",
                f"nonempty_data_rows={len(frame)}",
                f"component={component_index}",
            ]
            if repeated_headers:
                evidence.append(f"repeated_headers_removed={repeated_headers}")
            if ignored_totals:
                evidence.append(f"total_rows_filtered={ignored_totals}")
            region = TableRegion(
                region_id=region_id,
                source_id=grid.source_id,
                source_path=grid.source_path,
                container=grid.container,
                container_kind=grid.container_kind,
                hidden=grid.hidden,
                bounds=CellRange(
                    start_row=r0 + 1,
                    end_row=r1 + 1,
                    start_col=c0 + 1,
                    end_col=c1 + 1,
                ),
                header_rows=header_rows,
                proposed_header=headers,
                confidence=confidence,
                evidence=evidence,
                profile=profile_frame(frame),
                metadata={
                    **grid.metadata,
                    "merged_ranges": grid.merged_ranges,
                    "formula_cells": sorted(grid.formulas),
                    "context_rows": [
                        [_json_value(value) for value in context_row]
                        for context_row in block[max(0, header_local - 3):header_local]
                    ],
                },
            )
            output.append(RegionData(region, frame, source_rows))
    return output
