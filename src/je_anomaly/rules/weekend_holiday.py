"""규칙 2. 주말·공휴일 입력."""
from __future__ import annotations

from datetime import date

import pandas as pd

from je_anomaly.calendar_kr import KR_HOLIDAYS_2025
from je_anomaly.ledger import entry_summary
from je_anomaly.rules.base import make_result

RULE_ID = "WEEKEND_HOLIDAY"
_WEEKDAY = "월화수목금토일"


def detect_weekend_holiday(gl: pd.DataFrame, holidays: dict[date, str] | None = None,
                           weekend_score: float = 60.0, holiday_score: float = 60.0) -> pd.DataFrame:
    """전표일자가 토·일요일 또는 공휴일인 전표. holidays 미지정 시 2025년 한국 공휴일 사용."""
    holidays = KR_HOLIDAYS_2025 if holidays is None else holidays
    ent = entry_summary(gl)

    nos, scores, reasons = [], [], []
    for row in ent.itertuples(index=False):
        d = row.전표일자.date()
        if d in holidays:
            nos.append(row.전표번호)
            scores.append(holiday_score)
            reasons.append(f"공휴일({holidays[d]}, {d}) 입력")
        elif d.weekday() >= 5:
            nos.append(row.전표번호)
            scores.append(weekend_score)
            reasons.append(f"주말({_WEEKDAY[d.weekday()]}요일, {d}) 입력")
    return make_result(nos, RULE_ID, scores, reasons)
