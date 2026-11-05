"""ERP 원장 업로드: 파일 읽기 · 머리글 감지 · 열 자동 연결 · 날짜 해석."""
import io

import pandas as pd
import pytest

from je_anomaly.ledger import GL_COLUMNS, prepare
from je_anomaly.mapping import (
    apply_mapping,
    detect_header_row,
    disabled_rules,
    excel_sheets,
    missing_required,
    read_raw,
    suggest_mapping,
    with_header,
)

ERP_HEADER = ["회계일자", "전표No", "부서", "계정코드", "계정명", "차변금액", "대변금액",
              "적요내용", "거래처명", "입력자", "결재자"]
ERP_ROWS = [
    ["2025-03-04", "A-1", "재무", 831, "지급수수료", 1_000, 0, "수수료", "갑", "김", "이"],
    ["2025-03-04", "A-1", "재무", 103, "보통예금", 0, 1_000, "수수료", "갑", "김", "이"],
    [None, None, None, None, "합계", 1_000, 1_000, None, None, None, None],
]


def erp_excel() -> bytes:
    """제목 2줄 + 빈 줄 + 머리글 + 데이터 + 합계 줄, 시트 2개."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame(ERP_ROWS, columns=ERP_HEADER).to_excel(xw, sheet_name="원장", index=False, startrow=3)
        xw.sheets["원장"]["A1"] = "총계정원장"
        xw.sheets["원장"]["A2"] = "회사명: 테스트"
        pd.DataFrame({"x": [1]}).to_excel(xw, sheet_name="기타", index=False)
    return buf.getvalue()


def test_suggest_mapping_recognizes_common_erp_names():
    m = suggest_mapping(ERP_HEADER)
    assert m == {"전표번호": "전표No", "전표일자": "회계일자", "작성자": "입력자", "승인자": "결재자",
                 "계정코드": "계정코드", "계정과목": "계정명", "차변": "차변금액", "대변": "대변금액",
                 "적요": "적요내용", "거래처": "거래처명"}


def test_suggest_mapping_standard_names_map_to_themselves():
    assert suggest_mapping(GL_COLUMNS) == {c: c for c in GL_COLUMNS}


def test_short_alias_does_not_steal_other_date_columns():
    m = suggest_mapping(["승인일자", "전기일", "전표번호"])
    assert m["전표일자"] == "전기일"


def test_excel_upload_end_to_end():
    raw = erp_excel()
    assert excel_sheets(raw) == ["원장", "기타"]
    table = read_raw(raw, "원장.xlsx", "원장")
    header = detect_header_row(table)
    df = with_header(table, header)
    assert list(df.columns) == ERP_HEADER

    mapped = apply_mapping(df, suggest_mapping(df.columns))
    assert mapped.attrs["dropped_rows"] == 1  # 합계 줄 제거
    gl = prepare(mapped)
    assert len(gl) == 2
    assert gl.loc[0, "계정코드"] == "831" and gl.loc[0, "차변"] == 1000
    assert gl.loc[0, "전표일자"] == pd.Timestamp("2025-03-04")


def test_csv_with_title_rows_and_cp949():
    text = "총계정원장\n\n" + ",".join(ERP_HEADER) + "\n" + "\n".join(
        ",".join("" if v is None else str(v) for v in r) for r in ERP_ROWS[:2])
    table = read_raw(text.encode("cp949"), "gl.csv")
    df = with_header(table, detect_header_row(table))
    gl = prepare(apply_mapping(df, suggest_mapping(df.columns)))
    assert list(gl["작성자"]) == ["김", "김"]


def test_missing_required_and_account_fallback():
    m = {c: None for c in GL_COLUMNS}
    assert set(missing_required(m)) == {"전표번호", "전표일자", "차변", "대변", "계정코드 또는 계정과목"}

    df = pd.DataFrame({"번호": ["A", "A"], "일자": ["2025-03-04"] * 2, "계정": ["보통예금", "매출"],
                       "dr": [100, 0], "cr": [0, 100]})
    m.update({"전표번호": "번호", "전표일자": "일자", "계정과목": "계정", "차변": "dr", "대변": "cr"})
    out = apply_mapping(df, m)
    assert list(out["계정코드"]) == ["보통예금", "매출"]  # 코드가 없으면 과목명으로 대체
    assert (out["승인자"] == "").all()
    assert "SELF_APPROVAL" in disabled_rules(m)

    with pytest.raises(ValueError, match="필수"):
        apply_mapping(df, {c: None for c in GL_COLUMNS})


@pytest.mark.parametrize("value", ["2025-03-04", "2025.03.04", "2025/03/04", "20250304", 20250304])
def test_prepare_parses_various_date_formats(value):
    df = pd.DataFrame([["A", value, "", "", "103", "x", 1, 0, "", ""]], columns=GL_COLUMNS)
    assert prepare(df).loc[0, "전표일자"] == pd.Timestamp("2025-03-04")


def test_prepare_reports_unreadable_dates():
    df = pd.DataFrame([["A", "어제", "", "", "103", "x", 1, 0, "", ""]], columns=GL_COLUMNS)
    with pytest.raises(ValueError, match="날짜로 읽을 수 없는"):
        prepare(df)


def test_period_end_accepts_iso_string():
    from je_anomaly.rules.period_end import detect_period_end_manual
    df = pd.DataFrame([["GJ-1", "2026-03-27", "a", "b", "108", "x", 50_000_000, 0, "", ""],
                       ["GJ-1", "2026-03-27", "a", "b", "404", "y", 0, 50_000_000, "", ""]], columns=GL_COLUMNS)
    assert len(detect_period_end_manual(prepare(df), period_end="2026-03-31")) == 1  # 3월 결산법인
