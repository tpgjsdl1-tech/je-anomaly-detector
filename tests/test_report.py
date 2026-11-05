import io

import pandas as pd
import pytest
from openpyxl import load_workbook

from je_anomaly.generator import generate_ledger
from je_anomaly.ledger import GL_COLUMNS, load_csv, prepare
from je_anomaly.report import build_excel, rule_summary
from je_anomaly.rules import RULES
from je_anomaly.scoring import detect


@pytest.fixture(scope="module")
def detected():
    gl, _ = generate_ledger(seed=42)
    gl = prepare(gl)
    results, scored = detect(gl)
    return gl, results, scored


@pytest.mark.parametrize("encoding", ["utf-8-sig", "cp949"])
def test_load_csv_handles_korean_encodings(encoding):
    df = pd.DataFrame([["GJ-1", "2025-03-04", "김민준", "정태호", "0103", "보통예금", 1000, 0, "적요", "거래처"]],
                      columns=GL_COLUMNS)
    raw = df.to_csv(index=False).encode(encoding)
    loaded = load_csv(raw)
    assert loaded.loc[0, "작성자"] == "김민준"
    assert loaded.loc[0, "계정코드"] == "0103"  # 앞자리 0 보존


def test_rule_summary_has_every_rule(detected):
    gl, results, _ = detected
    summary = rule_summary(gl, results)
    assert len(summary) == len(RULES)
    assert summary["탐지 전표 수"].sum() == len(results)


def test_excel_report_sheets_and_contents(detected):
    gl, results, scored = detected
    wb = load_workbook(io.BytesIO(build_excel(gl, results, scored, "sample.csv")))
    assert wb.sheetnames == ["요약", "의심전표", "규칙별결과", "분개라인"]

    suspects = pd.read_excel(io.BytesIO(build_excel(gl, results, scored)), sheet_name="의심전표")
    assert len(suspects) == len(scored)
    assert suspects["위험도"].is_monotonic_decreasing

    lines = pd.read_excel(io.BytesIO(build_excel(gl, results, scored)), sheet_name="분개라인")
    assert set(lines["전표번호"]) == set(scored["전표번호"])
    assert lines["위험순위"].is_monotonic_increasing


def test_excel_report_with_no_findings(detected):
    gl, results, scored = detected
    data = build_excel(gl, results.iloc[0:0], scored.iloc[0:0])
    assert load_workbook(io.BytesIO(data))["의심전표"].max_row == 1
