"""탐지 규칙 레지스트리.

규칙 ID는 정답 라벨의 anomaly_type과 같게 맞춰, 규칙별 탐지율을 바로 계산할 수 있게 했다.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from je_anomaly.rules.account_pair import detect_unusual_pair
from je_anomaly.rules.base import RESULT_COLUMNS, empty_result
from je_anomaly.rules.benford import detect_benford
from je_anomaly.rules.duplicate import detect_duplicate
from je_anomaly.rules.period_end import detect_period_end_manual
from je_anomaly.rules.round_amount import detect_round_amount
from je_anomaly.rules.self_approval import detect_self_approval
from je_anomaly.rules.weekend_holiday import detect_weekend_holiday


@dataclass(frozen=True)
class Rule:
    id: str
    name: str
    func: Callable[..., pd.DataFrame]
    description: str


RULES: tuple[Rule, ...] = (
    Rule("DUPLICATE", "중복 전표", detect_duplicate,
         "계정·금액·거래처가 같은 전표가 5영업일 이내 재입력"),
    Rule("WEEKEND_HOLIDAY", "주말·공휴일 입력", detect_weekend_holiday,
         "토·일요일 또는 공휴일 일자의 전표"),
    Rule("ROUND_AMOUNT", "라운드 금액 반복", detect_round_amount,
         "백만원 단위로 딱 떨어지는 금액이 같은 거래처에 3회 이상"),
    Rule("BENFORD", "벤포드 법칙 이탈", detect_benford,
         "작성자별 첫 두 자리 분포에서 과다한 숫자대(z≥3)를 계정과목으로 드릴다운"),
    Rule("PERIOD_END_MANUAL", "결산일 직전 대형 수기 전표", detect_period_end_manual,
         "결산일 7일 전부터 입력된 3천만원 이상 수기 전표"),
    Rule("SELF_APPROVAL", "작성자 = 승인자", detect_self_approval,
         "작성자와 승인자가 같거나 승인자가 없는 전표"),
    Rule("UNUSUAL_PAIR", "비정상 계정 조합", detect_unusual_pair,
         "회계 논리상 이례적이거나 연간 3회 이하로 쓰인 차변·대변 조합"),
)
RULES_BY_ID = {r.id: r for r in RULES}

__all__ = ["RULES", "RULES_BY_ID", "Rule", "RESULT_COLUMNS", "empty_result"]
