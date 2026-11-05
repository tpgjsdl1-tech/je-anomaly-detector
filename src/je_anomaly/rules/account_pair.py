"""규칙 7. 비정상 계정 조합.

두 가지 방식을 함께 쓴다.
  (a) 회계 논리 패턴: 계정 성격상 나오면 안 되는 차변·대변 조합 (더존 계정코드 체계 기준)
  (b) 희소성: 연간 거의 쓰이지 않은 차변·대변 조합 (회사 고유의 거래 패턴 대비 이례적)
"""
from __future__ import annotations

import pandas as pd

from je_anomaly.ledger import ensure_prepared
from je_anomaly.rules.base import make_result

RULE_ID = "UNUSUAL_PAIR"

# 수익의 정상적인 상대계정: 현금·예금·매출채권·미수금·선수금
REVENUE_COUNTERPARTS = frozenset({"101", "102", "103", "108", "110", "120", "259"})
CASH = frozenset({"101", "102", "103"})


def _num(code: str) -> int | None:
    return int(code) if code.isdigit() else None


def is_revenue(code: str) -> bool:
    n = _num(code)
    return n is not None and (401 <= n <= 499 or 901 <= n <= 930)


def is_expense(code: str) -> bool:
    n = _num(code)
    return n is not None and (501 <= n <= 899 or 931 <= n <= 999)


def is_sga(code: str) -> bool:
    n = _num(code)
    return n is not None and 801 <= n <= 899


def is_asset(code: str) -> bool:
    n = _num(code)
    return n is not None and 101 <= n <= 250


def is_inventory(code: str) -> bool:
    n = _num(code)
    return n is not None and 146 <= n <= 175


def pattern_reason(dr: str, cr: str) -> str | None:
    """회계 논리상 이례적인 조합이면 사유, 아니면 None."""
    if is_revenue(cr) and dr not in REVENUE_COUNTERPARTS and not is_revenue(dr):
        return "수익 계정의 상대계정이 현금·채권이 아님"
    if is_expense(cr) and is_asset(dr) and dr not in CASH:
        return "비용을 자산으로 대체 (비용 이연·자산화)"
    if is_sga(dr) and is_inventory(cr):
        return "재고자산을 판관비로 직접 대체"
    return None


def account_pairs(gl: pd.DataFrame) -> pd.DataFrame:
    """전표별 (차변계정 × 대변계정) 조합과 연간 사용 전표 수."""
    gl = ensure_prepared(gl)
    cols = ["전표번호", "계정코드", "계정과목"]
    dr = gl.loc[gl["차변"] > 0, cols].drop_duplicates()
    cr = gl.loc[gl["대변"] > 0, cols].drop_duplicates()
    pairs = dr.merge(cr, on="전표번호", suffixes=("_차", "_대"))
    pairs["사용건수"] = pairs.groupby(["계정코드_차", "계정코드_대"])["전표번호"].transform("nunique")
    return pairs


def detect_unusual_pair(gl: pd.DataFrame, rare_max: int = 3, pattern_score: float = 70.0,
                        rare_score: float = 40.0, both_score: float = 80.0) -> pd.DataFrame:
    pairs = account_pairs(gl)
    nos, scores, reasons = [], [], []
    for p in pairs.itertuples(index=False):
        pattern = pattern_reason(p.계정코드_차, p.계정코드_대)
        rare = p.사용건수 <= rare_max
        if not (pattern or rare):
            continue
        label = f"(차){p.계정과목_차} / (대){p.계정과목_대}"
        parts = [pattern] if pattern else []
        if rare:
            parts.append(f"연간 {p.사용건수}회뿐인 드문 조합")
        nos.append(p.전표번호)
        scores.append(both_score if pattern and rare else pattern_score if pattern else rare_score)
        reasons.append(f"{label}: {', '.join(parts)}")
    return make_result(nos, RULE_ID, scores, reasons)
