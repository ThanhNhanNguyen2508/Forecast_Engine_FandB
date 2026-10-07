"""The single order/lead/calendar/arrival/expiry calculation. No business guessing."""
from __future__ import annotations

import re
import hashlib
from datetime import date, timedelta

from shelfcash_forecast.optimization.contracts import SupplierOffer, SupplyCalendar


def planned_lot_id(plan_id: str, offer_id: str, scenario_id: str | None = None) -> str:
    suffix = "-" + scenario_id if scenario_id else ""
    return f"{plan_id}{suffix}-lot-{hashlib.sha256(offer_id.encode()).hexdigest()[:16]}"


def parse_weekdays(text: str) -> list[int]:
    text = text.strip().casefold()
    if not text:
        raise ValueError("DELIVERY_SCHEDULE_REQUIRED")
    text = text.replace("thứ", "").replace("thu", "").replace("chủ nhật", "cn")
    tokens = [v.strip() for v in re.split(r"[,;/]", text)]
    mapping = {str(n): n - 2 for n in range(2, 8)} | {"cn": 6, "sunday": 6}
    if any(v not in mapping for v in tokens):
        raise ValueError("UNSUPPORTED_DELIVERY_SCHEDULE:" + text)
    return sorted({mapping[v] for v in tokens})


def expiry_date(arrival: date, offset: int | None) -> date | None:
    return None if offset is None else arrival + timedelta(days=offset)


def _add_lead(start: date, days: int, calendar: SupplyCalendar) -> date:
    if days > 3650:
        raise ValueError("UNSUPPORTED_LEAD_TIME_OVER_3650_DAYS")
    if calendar.lead_time_basis == "CALENDAR_DAYS":
        return start + timedelta(days=days)
    if not calendar.working_weekdays:
        raise ValueError("WORKING_CALENDAR_REQUIRED")
    day, remaining = start, days
    while remaining:
        day += timedelta(days=1)
        if day.weekday() in calendar.working_weekdays and day not in calendar.holidays:
            remaining -= 1
    return day


def resolve_arrival(order: date, lead: int, weekdays: list[int], calendar: SupplyCalendar) -> date | None:
    if calendar.status == "UNRESOLVED":
        raise ValueError("CALENDAR_SEMANTICS_REQUIRED")
    if not weekdays:
        raise ValueError("DELIVERY_WEEKDAYS_REQUIRED")
    effective = order + timedelta(days=int(calendar.order_boundary == "NEXT_DAY"))
    if calendar.meaning == "ORDER" and (effective.weekday() not in weekdays or effective in calendar.holidays):
        return None
    earliest = _add_lead(effective, lead, calendar)
    if calendar.meaning in {"RECEIVING", "DISPATCH"}:
        # A weekly schedule plus a finite exception set has a bounded next opportunity.
        for offset in range(8 + len(calendar.holidays)):
            day = earliest + timedelta(days=offset)
            if day.weekday() in weekdays and day not in calendar.holidays:
                if calendar.meaning == "DISPATCH":
                    return day + timedelta(days=calendar.dispatch_transport_days or 0)
                return day
        raise ValueError("NO_CALENDAR_OPPORTUNITY")
    return earliest


def offer_arrival(offer: SupplierOffer) -> date:
    if offer.calendar is not None:
        result = resolve_arrival(offer.order_date, offer.lead_time_days, offer.delivery_weekdays, offer.calendar)
        if result is None:
            raise ValueError("ORDER_OUTSIDE_SUPPLIER_CALENDAR")
    else:
        # Direct engine API legacy contract explicitly defines a calendar-day offset.
        result = offer.order_date + timedelta(days=offer.lead_time_days)
    if offer.resolved_arrival_date is not None and offer.resolved_arrival_date != result:
        raise ValueError("STALE_RESOLVED_ARRIVAL")
    proposed_dates=offer.source_terms.get('proposed_receiving_dates')
    if proposed_dates and str(result) not in proposed_dates:
        raise ValueError('ARRIVAL_OUTSIDE_PROPOSED_CONTRACT_DATES')
    return result
