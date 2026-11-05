"""규칙 4. 벤포드 법칙 이탈 — 첫 두 자리(First-Two Digits) 검정.

모집단 전체가 아니라 작성자별로 검정한 뒤, 과다 숫자대를 계정과목별로 드릴다운한다.
  - 전체 모집단에서는 한 담당자의 '쪼개기' 20건이 2,000건 속에 희석되어 유의하지 않다.
  - 계정과목별 검정은 통신비·복리후생비처럼 금액대가 좁은 계정이 원래 벤포드를 따르지 않아 오탐이 많다.
  - 작성자는 여러 계정을 다루므로 금액이 여러 자릿수에 걸쳐 벤포드 적용 조건을 만족한다.
z-통계량은 Nigrini(2012)의 연속성 보정 공식을 따른다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from je_anomaly.ledger import ensure_prepared
from je_anomaly.rules.base import make_result

RULE_ID = "BENFORD"

FTD = np.arange(10, 100)
BENFORD_FTD = pd.Series(np.log10(1 + 1 / FTD), index=FTD, name="expected")
# Nigrini: 첫 두 자리 MAD 기준 (0.0012 이하 근접 일치, 0.0018 이하 허용, 0.0022 초과 부적합)
MAD_NONCONFORMITY = 0.0022


def first_two_digits(amounts: pd.Series) -> pd.Series:
    a = amounts[amounts >= 10].astype("int64").astype(str)
    return a.str[:2].astype(int)


def benford_table(amounts: pd.Series) -> pd.DataFrame:
    """첫 두 자리별 관측/기대 비율과 z-통계량."""
    ftd = first_two_digits(amounts)
    n = len(ftd)
    counts = ftd.value_counts().reindex(FTD, fill_value=0)
    tbl = pd.DataFrame({"count": counts, "expected": BENFORD_FTD})
    tbl["observed"] = tbl["count"] / n if n else 0.0
    if n:
        se = np.sqrt(tbl["expected"] * (1 - tbl["expected"]) / n)
        tbl["z"] = ((tbl["observed"] - tbl["expected"]).abs() - 1 / (2 * n)).clip(lower=0) / se
    else:
        tbl["z"] = 0.0
    tbl.index.name = "digits"
    return tbl


def benford_mad(amounts: pd.Series) -> float:
    tbl = benford_table(amounts)
    return float((tbl["observed"] - tbl["expected"]).abs().mean())


def mad_conformity(mad: float) -> str:
    """Nigrini 첫 두 자리 MAD 판정."""
    if mad <= 0.0012:
        return "근접 일치"
    if mad <= 0.0018:
        return "허용 가능"
    if mad <= MAD_NONCONFORMITY:
        return "경계"
    return "부적합"


DEFAULT_EXCLUDE = ("부가세대급금", "부가세예수금", "예수금")


def benford_lines(gl: pd.DataFrame, min_amount: int = 10_000,
                  exclude_accounts: tuple[str, ...] = DEFAULT_EXCLUDE) -> pd.DataFrame:
    """벤포드 검정 대상 라인 (금액·첫 두 자리 포함). 규칙과 대시보드 차트가 같은 모집단을 쓰도록 공유."""
    gl = ensure_prepared(gl)
    lines = gl.assign(금액=gl["차변"] + gl["대변"])
    lines = lines[(lines["금액"] >= min_amount) & ~lines["계정과목"].isin(exclude_accounts)]
    return lines.assign(ftd=first_two_digits(lines["금액"]))


def _z(count: float, n: float, p: float) -> float:
    if n == 0:
        return 0.0
    return max(0.0, abs(count / n - p) - 1 / (2 * n)) / np.sqrt(p * (1 - p) / n)


def detect_benford(gl: pd.DataFrame, group_col: str = "작성자", drilldown_col: str = "계정과목",
                   min_group_size: int = 100, z_threshold: float = 3.0, drilldown_z: float = 3.0,
                   min_count: int = 3, min_amount: int = 10_000,
                   exclude_accounts: tuple[str, ...] = DEFAULT_EXCLUDE,
                   ) -> pd.DataFrame:
    """
    1) group_col(기본: 작성자)별로 첫 두 자리 분포를 검정해 유의하게 과다한(z ≥ z_threshold) 숫자대를 찾는다.
       매월 같은 임차료처럼 '같은 금액의 반복'은 벤포드가 아니라 중복·라운드 검사의 대상이므로
       검정은 고유 금액 기준으로 한다 (Nigrini의 number duplication 선제거).
    2) 과다 숫자대 안에서 drilldown_col(기본: 계정과목)별로 다시 검정해, 과다를 실제로 만들어낸
       계정의 전표만 표시한다 (우연히 같은 숫자대에 속한 정상 전표 오탐 감소).
    부가세·원천세처럼 다른 금액에서 기계적으로 파생되는 계정은 제외한다.
    """
    lines = benford_lines(gl, min_amount, exclude_accounts)

    nos, scores, reasons = [], [], []
    for name, grp in lines.groupby(group_col):
        uniq = grp.drop_duplicates("금액")
        n = len(uniq)
        if n < min_group_size:
            continue
        tbl = benford_table(uniq["금액"])
        spikes = tbl[(tbl["z"] >= z_threshold) & (tbl["count"] >= min_count)
                     & (tbl["observed"] > tbl["expected"])]
        for digits, s in spikes.iterrows():
            for who, sub in uniq.groupby(drilldown_col):
                c = int((sub["ftd"] == digits).sum())
                z_who = _z(c, len(sub), s.expected)
                if c < min_count or c / len(sub) <= s.expected or z_who < drilldown_z:
                    continue
                reason = (f"[{name}] 첫 두 자리 '{digits}' 과다: 관측 {s.observed:.1%} vs 기대 {s.expected:.1%} "
                          f"(z={s.z:.1f}) — {drilldown_col} '{who}'에 집중 ({c}/{len(sub)}건, z={z_who:.1f})")
                hit = grp[(grp["ftd"] == digits) & (grp[drilldown_col] == who)]
                for je_no in hit["전표번호"]:
                    nos.append(je_no)
                    scores.append(min(80.0, 40.0 + 4.0 * s.z))
                    reasons.append(reason)
    return make_result(nos, RULE_ID, scores, reasons)
