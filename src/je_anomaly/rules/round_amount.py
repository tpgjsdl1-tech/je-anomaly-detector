"""규칙 3. 라운드 금액 반복 — 같은 거래처에 딱 떨어지는 금액이 반복 계상됨."""
from __future__ import annotations

import pandas as pd

from je_anomaly.ledger import entry_summary
from je_anomaly.rules.base import make_result, won

RULE_ID = "ROUND_AMOUNT"


def detect_round_amount(gl: pd.DataFrame, unit: int = 1_000_000, min_repeat: int = 3,
                        base_score: float = 40.0, step: float = 5.0, max_score: float = 75.0) -> pd.DataFrame:
    """
    전표 금액(차변 합계)이 unit의 배수인 전표가 같은 거래처에 min_repeat회 이상이면 모두 표시.
    반복 횟수가 많을수록 점수 상승. 부가세가 붙는 정상 매입은 합계가 딱 떨어지지 않으므로 잘 걸리지 않는다.
    """
    ent = entry_summary(gl)
    rnd = ent[(ent["금액"] >= unit) & (ent["금액"] % unit == 0)].copy()
    rnd["거래처키"] = rnd["거래처"].replace("", "(거래처 없음)")
    rnd["반복"] = rnd.groupby("거래처키")["전표번호"].transform("count")
    hit = rnd[rnd["반복"] >= min_repeat]

    scores = (base_score + step * (hit["반복"] - min_repeat)).clip(upper=max_score)
    reasons = [f"{won(r.금액)}(={unit // 10_000:,}만원 단위) — '{r.거래처키}'에 라운드 금액 {r.반복}회 반복"
               for r in hit.itertuples(index=False)]
    return make_result(hit["전표번호"], RULE_ID, scores, reasons)
