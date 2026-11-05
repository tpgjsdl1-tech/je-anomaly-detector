"""탐지 규칙 단위 테스트 — 규칙마다 손으로 만든 작은 원장으로 '걸려야 할 것 / 걸리면 안 될 것'을 확인."""
import numpy as np
import pandas as pd
import pytest

from je_anomaly.ledger import GL_COLUMNS, prepare
from je_anomaly.rules import RULES, RESULT_COLUMNS
from je_anomaly.rules.account_pair import detect_unusual_pair
from je_anomaly.rules.benford import benford_table, detect_benford
from je_anomaly.rules.duplicate import detect_duplicate
from je_anomaly.rules.period_end import detect_period_end_manual
from je_anomaly.rules.round_amount import detect_round_amount
from je_anomaly.rules.self_approval import detect_self_approval
from je_anomaly.rules.weekend_holiday import detect_weekend_holiday
from je_anomaly.scoring import combine_scores, detect


def entry(no, day, dr, cr, amount, vendor="", writer="김민준", approver="정태호", memo="테스트"):
    """2줄짜리 전표(차변 dr / 대변 cr) 라인 목록."""
    return [
        [no, day, writer, approver, dr, f"계정{dr}", amount, 0, memo, vendor],
        [no, day, writer, approver, cr, f"계정{cr}", 0, amount, memo, vendor],
    ]


def ledger(*entries) -> pd.DataFrame:
    rows = [r for e in entries for r in e]
    return prepare(pd.DataFrame(rows, columns=GL_COLUMNS))


def flagged(result: pd.DataFrame) -> set[str]:
    return set(result["전표번호"])


# ── 공통 ──────────────────────────────────────────────────────────
def test_prepare_rejects_missing_columns():
    with pytest.raises(ValueError, match="필수 컬럼"):
        prepare(pd.DataFrame({"전표번호": ["A"]}))


def test_prepare_parses_comma_amounts():
    df = pd.DataFrame([["A", "2025-03-04", "a", "b", 831, "x", "1,000", "", "", None]], columns=GL_COLUMNS)
    gl = prepare(df)
    assert gl.loc[0, "차변"] == 1000 and gl.loc[0, "대변"] == 0 and gl.loc[0, "계정코드"] == "831"


@pytest.mark.parametrize("rule", RULES, ids=lambda r: r.id)
def test_every_rule_returns_standard_columns(rule):
    gl = ledger(entry("A", "2025-03-04", "831", "103", 123_450))
    result = rule.func(gl)
    assert list(result.columns) == RESULT_COLUMNS


# ── 규칙 1. 중복 ─────────────────────────────────────────────────
def test_duplicate_flags_later_copy_within_window():
    gl = ledger(
        entry("A", "2025-03-04", "831", "103", 1_234_560, "갑"),
        entry("B", "2025-03-06", "831", "103", 1_234_560, "갑"),
    )
    r = detect_duplicate(gl)
    assert flagged(r) == {"B"}
    assert "A" in r.loc[0, "reason"]


def test_duplicate_window_counts_business_days_across_long_holiday():
    # 9/30(화) 원본 → 추석 연휴(10/3~10/9) 지나 10/13(월) 재입력: 달력 13일, 영업일 기준 5일 이내
    gl = ledger(
        entry("A", "2025-09-30", "831", "103", 617_160, "갑"),
        entry("B", "2025-10-13", "831", "103", 617_160, "갑"),
    )
    r = detect_duplicate(gl)
    assert flagged(r) == {"B"}
    assert "영업일" in r.loc[0, "reason"]


def test_duplicate_ignores_monthly_recurring_and_different_vendor():
    gl = ledger(
        entry("A", "2025-03-05", "819", "103", 3_000_000, "임대인"),
        entry("B", "2025-04-05", "819", "103", 3_000_000, "임대인"),  # 한 달 뒤 → 정상
        entry("C", "2025-03-05", "819", "103", 3_000_000, "다른회사"),  # 거래처 다름
    )
    assert detect_duplicate(gl).empty


# ── 규칙 2. 주말·공휴일 ──────────────────────────────────────────
def test_weekend_and_holiday():
    gl = ledger(
        entry("SAT", "2025-03-08", "831", "103", 1000),
        entry("HOL", "2025-10-06", "831", "103", 1000),  # 추석
        entry("WED", "2025-03-05", "831", "103", 1000),
    )
    r = detect_weekend_holiday(gl).set_index("전표번호")
    assert set(r.index) == {"SAT", "HOL"}
    assert "추석" in r.loc["HOL", "reason"]


# ── 규칙 3. 라운드 금액 반복 ─────────────────────────────────────
def test_round_amount_requires_repetition_per_vendor():
    gl = ledger(
        *(entry(f"R{i}", f"2025-03-0{i + 3}", "831", "253", 5_000_000, "자문사") for i in range(3)),
        entry("ONCE", "2025-03-10", "831", "253", 5_000_000, "다른곳"),  # 1회뿐
        entry("ODD", "2025-03-11", "831", "253", 5_000_010, "자문사"),  # 라운드 아님
    )
    assert flagged(detect_round_amount(gl)) == {"R0", "R1", "R2"}


# ── 규칙 4. 벤포드 ──────────────────────────────────────────────
def test_benford_table_matches_theory_for_benford_data():
    rng = np.random.default_rng(0)
    amounts = pd.Series((10 ** rng.uniform(4, 8, 20_000)).astype(int))  # 로그균등 → 벤포드 분포
    tbl = benford_table(amounts)
    assert (tbl["observed"] - tbl["expected"]).abs().mean() < 0.0012  # Nigrini '근접 일치'


def test_benford_flags_concentrated_spike_only_for_offending_account():
    rng = np.random.default_rng(1)
    normal = [entry(f"N{i}", "2025-03-04", "153", "251", int(10 ** rng.uniform(5, 8)), writer="박지훈")
              for i in range(400)]
    split = [entry(f"S{i}", "2025-03-04", "824", "103", 4_900_000 + i * 3_170, writer="박지훈")
             for i in range(20)]
    r = detect_benford(ledger(*normal, *split))
    assert {f"S{i}" for i in range(20)} <= flagged(r)
    assert len(flagged(r) - {f"S{i}" for i in range(20)}) <= 2


# ── 규칙 5. 결산 직전 대형 수기 ──────────────────────────────────
def test_period_end_manual():
    gl = ledger(
        entry("GJ-1", "2025-12-29", "108", "404", 80_000_000),
        entry("SA-1", "2025-12-29", "108", "404", 80_000_000),  # 자동 전표
        entry("GJ-2", "2025-12-29", "518", "207", 12_000_000),  # 금액 작음
        entry("GJ-3", "2025-11-28", "108", "404", 80_000_000),  # 결산 구간 밖
    )
    assert flagged(detect_period_end_manual(gl)) == {"GJ-1"}


# ── 규칙 6. 작성자 = 승인자 ─────────────────────────────────────
def test_self_approval_and_missing_approver():
    gl = ledger(
        entry("SELF", "2025-03-04", "831", "103", 1000, writer="정태호", approver="정태호"),
        entry("NONE", "2025-03-04", "831", "103", 1000, approver=""),
        entry("OK", "2025-03-04", "831", "103", 1000),
    )
    assert flagged(detect_self_approval(gl)) == {"SELF", "NONE"}


# ── 규칙 7. 비정상 계정 조합 ─────────────────────────────────────
def test_unusual_pair_logic_patterns():
    common = [entry(f"C{i}", "2025-03-04", "108", "404", 1000 + i) for i in range(5)]
    gl = ledger(
        *common,
        entry("REV", "2025-03-05", "134", "404", 5000),  # 가지급금 / 매출
        entry("CAP", "2025-03-05", "239", "801", 5000),  # 개발비 / 급여
        entry("INV", "2025-03-05", "813", "153", 5000),  # 접대비 / 원재료
    )
    r = detect_unusual_pair(gl)
    assert flagged(r) == {"REV", "CAP", "INV"}


def test_unusual_pair_rare_combination_without_pattern():
    common = [entry(f"C{i}", "2025-03-04", "831", "103", 1000 + i) for i in range(5)]
    gl = ledger(*common, entry("RARE", "2025-03-05", "831", "253", 5000))
    r = detect_unusual_pair(gl).set_index("전표번호")
    assert set(r.index) == {"RARE"}
    assert r.loc["RARE", "score"] == 40


# ── 종합 점수 ───────────────────────────────────────────────────
def test_combine_scores_is_probabilistic_or():
    assert combine_scores(pd.Series([60.0])) == pytest.approx(60)
    assert combine_scores(pd.Series([60.0, 70.0])) == pytest.approx(88)


def test_detect_ranks_multi_rule_entry_first():
    gl = ledger(
        entry("BOTH", "2025-03-08", "831", "103", 1000, writer="정태호", approver="정태호"),  # 주말 + 자기승인
        entry("ONE", "2025-03-09", "831", "103", 2000),  # 주말만
    )
    _, scored = detect(gl, rule_ids=["WEEKEND_HOLIDAY", "SELF_APPROVAL"])
    assert list(scored["전표번호"]) == ["BOTH", "ONE"]
    assert scored.loc[0, "규칙수"] == 2
