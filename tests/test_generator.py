from datetime import date

import pandas as pd
import pytest

from je_anomaly.calendar_kr import is_business_day
from je_anomaly.generator import (
    ANOMALY_COUNTS,
    GL_COLUMNS,
    LABEL_COLUMNS,
    PERIOD_END_START,
    generate_ledger,
)


@pytest.fixture(scope="module")
def data():
    gl, labels = generate_ledger(seed=42)
    merged = gl.merge(labels, on="전표번호")
    merged["전표일자"] = pd.to_datetime(merged["전표일자"]).dt.date
    return gl, labels, merged


def test_columns(data):
    gl, labels, _ = data
    assert list(gl.columns) == GL_COLUMNS
    assert list(labels.columns) == LABEL_COLUMNS


def test_row_count_about_5000(data):
    gl, _, _ = data
    assert 4_500 <= len(gl) <= 5_500


def test_every_entry_is_balanced(data):
    gl, _, _ = data
    sums = gl.groupby("전표번호")[["차변", "대변"]].sum()
    assert (sums["차변"] == sums["대변"]).all()


def test_each_line_has_exactly_one_side(data):
    gl, _, _ = data
    assert (((gl["차변"] > 0) ^ (gl["대변"] > 0))).all()


def test_labels_cover_every_entry_once(data):
    gl, labels, _ = data
    assert labels["전표번호"].is_unique
    assert set(labels["전표번호"]) == set(gl["전표번호"])


def test_planted_anomaly_counts(data):
    _, labels, _ = data
    counts = labels.loc[labels["is_anomaly"] == 1, "anomaly_type"].value_counts().to_dict()
    assert counts == ANOMALY_COUNTS
    assert (labels.loc[labels["is_anomaly"] == 0, "anomaly_type"].fillna("") == "").all()


def test_reproducible_with_same_seed(data):
    gl, labels, _ = data
    gl2, labels2 = generate_ledger(seed=42)
    pd.testing.assert_frame_equal(gl, gl2)
    pd.testing.assert_frame_equal(labels, labels2)
    gl3, _ = generate_ledger(seed=7)
    assert not gl.equals(gl3)


def test_normal_entries_only_on_business_days(data):
    _, _, m = data
    normal = m[m["is_anomaly"] == 0]
    assert normal["전표일자"].map(is_business_day).all()


def test_weekend_anomalies_on_non_business_days(data):
    _, _, m = data
    wk = m[m["anomaly_type"] == "WEEKEND_HOLIDAY"]
    assert (~wk["전표일자"].map(is_business_day)).all()


def test_writer_equals_approver_only_in_self_approval(data):
    _, _, m = data
    same = m["작성자"] == m["승인자"]
    assert (m.loc[same, "anomaly_type"] == "SELF_APPROVAL").all()
    assert same[m["anomaly_type"] == "SELF_APPROVAL"].all()


def test_period_end_anomalies_are_large_manual(data):
    _, _, m = data
    pe = m[m["anomaly_type"] == "PERIOD_END_MANUAL"]
    assert pe["전표번호"].str.startswith("GJ-").all()
    assert (pe["전표일자"] >= PERIOD_END_START).all()
    assert (pe.groupby("전표번호")["차변"].sum() >= 50_000_000).all()


def test_other_anomalies_planted_before_period_end(data):
    _, _, m = data
    others = m[(m["is_anomaly"] == 1) & (m["anomaly_type"] != "PERIOD_END_MANUAL")]
    assert (others["전표일자"] < date(2025, 12, 20)).all()
