"""규칙 5. 결산일 직전 대형 수기 전표."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from je_anomaly.ledger import entry_summary
from je_anomaly.rules.base import make_result, won

RULE_ID = "PERIOD_END_MANUAL"


def detect_period_end_manual(gl: pd.DataFrame, period_end: date | str | None = None, window_days: int = 7,
                             min_amount: float = 30_000_000, manual_pattern: str = r"^GJ",
                             base_score: float = 50.0) -> pd.DataFrame:
    """
    결산일(기본: 데이터 마지막 연도의 12/31, 3·6월 결산법인은 직접 지정) 전 window_days일 이내에 입력된,
    금액 min_amount 이상인 수기 전표(전표번호가 manual_pattern에 맞는 것).
    min_amount는 수행중요성 수준으로 바꿔 쓰면 된다. 금액이 클수록 점수 상승.
    """
    ent = entry_summary(gl)
    if ent.empty:
        return make_result([], RULE_ID, [], [])
    if period_end is None:
        period_end = date(ent["전표일자"].max().year, 12, 31)
    elif isinstance(period_end, str):
        period_end = date.fromisoformat(period_end)
    start = pd.Timestamp(period_end - timedelta(days=window_days))
    end = pd.Timestamp(period_end)

    manual = ent["전표번호"].str.contains(manual_pattern, regex=True)
    hit = ent[manual & ent["전표일자"].between(start, end) & (ent["금액"] >= min_amount)]

    scores = np.minimum(90.0, base_score + 10.0 * np.log2(hit["금액"] / min_amount))
    reasons = [f"결산일({period_end}) {(end - r.전표일자).days}일 전 수기 전표 {won(r.금액)}"
               for r in hit.itertuples(index=False)]
    return make_result(hit["전표번호"], RULE_ID, scores, reasons)
