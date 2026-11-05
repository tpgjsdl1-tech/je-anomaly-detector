"""규칙 1. 중복 전표 — 계정·금액·거래처 구성이 완전히 같은 전표가 짧은 기간 안에 다시 입력됨."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from je_anomaly.calendar_kr import KR_HOLIDAYS_2025
from je_anomaly.ledger import ensure_prepared, entry_summary
from je_anomaly.rules.base import make_result

RULE_ID = "DUPLICATE"


def detect_duplicate(gl: pd.DataFrame, window_days: int = 5, holidays: dict[date, str] | None = None,
                     score: float = 80.0, same_day_bonus: float = 10.0) -> pd.DataFrame:
    """
    window_days '영업일' 이내에 '전표 지문'(계정코드·차변·대변·거래처의 정렬된 조합)이 같은 전표가 있으면
    나중에 입력된 쪽을 중복으로 본다. 같은 날 재입력이면 가점.
    매월 같은 금액의 임차료처럼 주기적으로 반복되는 정상 거래는 window_days로 걸러진다.

    기간을 달력일이 아닌 영업일로 세는 이유: 설·추석 연휴를 사이에 두면 3영업일 뒤 재입력도
    달력으로는 10일이 넘어가 놓친다 (4단계 검증에서 실제로 발견된 누락).
    """
    holidays = KR_HOLIDAYS_2025 if holidays is None else holidays
    holiday_list = np.array(sorted(holidays), dtype="datetime64[D]")
    gl = ensure_prepared(gl)
    key = (gl["계정코드"] + "|" + gl["차변"].astype("int64").astype(str) + "|"
           + gl["대변"].astype("int64").astype(str) + "|" + gl["거래처"])
    fingerprint = key.groupby(gl["전표번호"]).agg(lambda k: ";".join(sorted(k)))

    ent = entry_summary(gl)
    ent["fp"] = ent["전표번호"].map(fingerprint)
    ent = ent.sort_values(["전표일자", "_row"])

    nos, scores, reasons = [], [], []
    for _, grp in ent.groupby("fp", sort=False):
        if len(grp) < 2:
            continue
        prev = None
        for row in grp.itertuples(index=False):
            if prev is not None:
                gap = int(np.busday_count(prev.전표일자.date(), row.전표일자.date(), holidays=holiday_list))
                same_day = row.전표일자 == prev.전표일자
                if gap <= window_days:
                    nos.append(row.전표번호)
                    scores.append(score + (same_day_bonus if same_day else 0))
                    when = "같은 날" if same_day else f"{gap}영업일 뒤"
                    reasons.append(f"계정·금액·거래처가 동일한 전표({prev.전표번호})가 {when} 다시 입력됨")
            prev = row
    return make_result(nos, RULE_ID, scores, reasons)
