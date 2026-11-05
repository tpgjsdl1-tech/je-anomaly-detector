"""2025 회계연도 한국 영업일 캘린더.

외부 패키지 의존 없이 공휴일을 명시적으로 관리한다.
(감사 조서 관점에서 '어떤 날을 휴일로 봤는가'가 코드에 그대로 드러나도록)
"""
from __future__ import annotations

from datetime import date, timedelta

FISCAL_YEAR = 2025
FY_START = date(FISCAL_YEAR, 1, 1)
FY_END = date(FISCAL_YEAR, 12, 31)

KR_HOLIDAYS_2025: dict[date, str] = {
    date(2025, 1, 1): "신정",
    date(2025, 1, 27): "임시공휴일",
    date(2025, 1, 28): "설날 연휴",
    date(2025, 1, 29): "설날",
    date(2025, 1, 30): "설날 연휴",
    date(2025, 3, 1): "삼일절",
    date(2025, 3, 3): "대체공휴일(삼일절)",
    date(2025, 5, 5): "어린이날·부처님오신날",
    date(2025, 5, 6): "대체공휴일",
    date(2025, 6, 3): "대통령선거일",
    date(2025, 6, 6): "현충일",
    date(2025, 8, 15): "광복절",
    date(2025, 10, 3): "개천절",
    date(2025, 10, 5): "추석 연휴",
    date(2025, 10, 6): "추석",
    date(2025, 10, 7): "추석 연휴",
    date(2025, 10, 8): "대체공휴일(추석)",
    date(2025, 10, 9): "한글날",
    date(2025, 12, 25): "성탄절",
}


def is_business_day(d: date) -> bool:
    return d.weekday() < 5 and d not in KR_HOLIDAYS_2025


def non_business_reason(d: date) -> str | None:
    """영업일이 아니면 사유(공휴일명 또는 토/일요일)를, 영업일이면 None을 반환."""
    if d in KR_HOLIDAYS_2025:
        return KR_HOLIDAYS_2025[d]
    if d.weekday() == 5:
        return "토요일"
    if d.weekday() == 6:
        return "일요일"
    return None


def _days(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def business_days(start: date = FY_START, end: date = FY_END) -> list[date]:
    return [d for d in _days(start, end) if is_business_day(d)]


def non_business_days(start: date = FY_START, end: date = FY_END) -> list[date]:
    return [d for d in _days(start, end) if not is_business_day(d)]


def next_business_day(d: date) -> date:
    while not is_business_day(d):
        d += timedelta(days=1)
    return d


def previous_business_day(d: date) -> date:
    while not is_business_day(d):
        d -= timedelta(days=1)
    return d


def shift_business_days(d: date, n: int) -> date:
    """d(영업일 기준)에서 n 영업일 뒤의 날짜."""
    d = next_business_day(d)
    for _ in range(n):
        d = next_business_day(d + timedelta(days=1))
    return d
