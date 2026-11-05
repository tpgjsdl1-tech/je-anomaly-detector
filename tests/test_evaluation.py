import pandas as pd
import pytest

from je_anomaly.evaluation import (
    EVAL_END,
    EVAL_START,
    confusion,
    per_rule_metrics,
    threshold_curve,
    to_markdown,
    top_k,
    update_readme,
)

LABELS = pd.DataFrame({
    "전표번호": ["A", "B", "C", "D", "E", "F"],
    "is_anomaly": [1, 1, 0, 0, 0, 0],
    "anomaly_type": ["DUPLICATE", "SELF_APPROVAL", None, None, None, None],
})


def test_confusion_counts_and_rates():
    m = confusion({"A", "C"}, LABELS)
    assert (m["TP"], m["FP"], m["FN"], m["TN"]) == (1, 1, 1, 3)
    assert m["탐지율"] == pytest.approx(0.5)
    assert m["오탐률"] == pytest.approx(0.25)
    assert m["정밀도"] == pytest.approx(0.5)


def test_per_rule_separates_same_type_other_type_and_false_positive():
    results = pd.DataFrame({"전표번호": ["A", "B", "C"], "rule_id": ["DUPLICATE"] * 3,
                            "score": [80.0] * 3, "reason": [""] * 3})
    scored = pd.DataFrame({"전표번호": ["A", "B", "C"]})
    row = per_rule_metrics(results, scored, LABELS).set_index("규칙").loc["중복 전표"]
    assert row["규칙 탐지율"] == 1.0
    assert (row["같은 유형"], row["다른 이상유형"], row["정상(오탐)"]) == (1, 1, 1)
    assert row["정밀도"] == pytest.approx(2 / 3)


def test_threshold_and_top_k():
    scored = pd.DataFrame({"전표번호": ["A", "C", "B"], "위험도": [90.0, 70.0, 50.0]})
    curve = threshold_curve(scored, LABELS, thresholds=[0, 80]).set_index("최소 위험도")
    assert curve.loc[0, "탐지율"] == 1.0 and curve.loc[80, "탐지율"] == 0.5
    assert curve.loc[80, "FP"] == 0
    tk = top_k(scored, LABELS, ks=[1, 2]).set_index("검토 건수(상위 K)")
    assert tk.loc[1, "적중률"] == 1.0 and tk.loc[2, "적중률"] == 0.5


def test_update_readme_replaces_only_marked_block():
    text = f"intro\n{EVAL_START}\nold\n{EVAL_END}\noutro\n"
    new = update_readme(text, "NEW")
    assert new == f"intro\n{EVAL_START}\nNEW\n{EVAL_END}\noutro\n"
    assert EVAL_START in update_readme("no markers", "X")


def test_to_markdown_formats_percent_columns():
    md = to_markdown(pd.DataFrame({"규칙": ["x"], "탐지율": [0.987], "건수": [1234]}))
    assert "98.7%" in md and "1,234" in md
