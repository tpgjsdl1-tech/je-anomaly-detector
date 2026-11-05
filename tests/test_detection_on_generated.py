"""통합 테스트 — 가상 원장에 심은 이상 전표를 각 규칙이 잡아내는지 확인 (정밀 평가는 4단계)."""
import pandas as pd
import pytest

from je_anomaly.generator import ANOMALY_COUNTS, generate_ledger
from je_anomaly.ledger import prepare
from je_anomaly.scoring import detect


@pytest.fixture(scope="module", params=[42, 7])
def outcome(request):
    gl, labels = generate_ledger(seed=request.param)
    results, scored = detect(prepare(gl))
    return results, scored, labels.fillna("")


@pytest.mark.parametrize("anomaly_type", list(ANOMALY_COUNTS))
def test_each_rule_catches_its_planted_type(outcome, anomaly_type):
    results, _, labels = outcome
    planted = set(labels.loc[labels["anomaly_type"] == anomaly_type, "전표번호"])
    caught = set(results.loc[results["rule_id"] == anomaly_type, "전표번호"])
    assert len(planted & caught) / len(planted) >= 0.9


def test_all_planted_anomalies_are_flagged_with_few_false_positives(outcome):
    _, scored, labels = outcome
    anomalies = set(labels.loc[labels["is_anomaly"] == 1, "전표번호"])
    flagged = set(scored["전표번호"])
    assert anomalies <= flagged
    assert len(flagged - anomalies) <= 0.02 * len(labels)  # 정상 전표의 2% 이하


def test_scored_list_is_sorted_by_risk(outcome):
    _, scored, _ = outcome
    assert scored["위험도"].is_monotonic_decreasing
    assert scored["위험도"].between(0, 100).all()
