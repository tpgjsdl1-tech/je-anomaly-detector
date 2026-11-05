"""탐지 결과 엑셀 보고서 (감사 조서 첨부용)."""
from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from je_anomaly.ledger import GL_COLUMNS, entry_summary, ensure_prepared
from je_anomaly.rules import RULES, RULES_BY_ID

AMOUNT_COLUMNS = {"금액", "차변", "대변", "의심 금액 합계"}
WRAP_COLUMNS = {"사유", "reason", "판단 기준"}
HEADER_FILL = PatternFill("solid", fgColor="1F3864")


def rule_summary(gl: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """규칙별 탐지 건수와 금액."""
    amounts = entry_summary(gl).set_index("전표번호")["금액"]
    rows = []
    for rule in RULES:
        hit = results.loc[results["rule_id"] == rule.id, "전표번호"]
        rows.append({"규칙": rule.name, "판단 기준": rule.description, "탐지 전표 수": len(hit),
                     "의심 금액 합계": int(amounts.reindex(hit).sum())})
    return pd.DataFrame(rows)


def _style(ws, df: pd.DataFrame) -> None:
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center")
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for idx, col in enumerate(df.columns, start=1):
        letter = get_column_letter(idx)
        if col in WRAP_COLUMNS:
            width = 70
        else:
            longest = max([len(str(col))] + [len(str(v)) for v in df[col].head(500)])
            width = min(40, max(8, longest * 1.6))
        ws.column_dimensions[letter].width = width
        for cell in ws[letter][1:]:
            if col in AMOUNT_COLUMNS:
                cell.number_format = "#,##0"
            if col in WRAP_COLUMNS:
                cell.alignment = Alignment(wrap_text=True, vertical="top")


def _dates_to_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "전표일자" in df.columns:
        df["전표일자"] = pd.to_datetime(df["전표일자"]).dt.strftime("%Y-%m-%d")
    return df


def build_excel(gl: pd.DataFrame, results: pd.DataFrame, scored: pd.DataFrame,
                source_name: str = "") -> bytes:
    """시트: 요약 / 의심전표(위험도순) / 규칙별결과 / 분개라인."""
    gl = ensure_prepared(gl)
    n_entries = gl["전표번호"].nunique()
    overview = pd.DataFrame({
        "항목": ["원장 파일", "작성 시각", "총 라인 수", "총 전표 수", "의심 전표 수", "고위험(80점 이상) 전표 수"],
        "값": [source_name or "-", datetime.now().strftime("%Y-%m-%d %H:%M"), f"{len(gl):,}", f"{n_entries:,}",
              f"{len(scored):,}", f"{int((scored['위험도'] >= 80).sum()):,}"],
    })
    by_rule = rule_summary(gl, results)

    detail = results.assign(규칙=results["rule_id"].map(lambda r: RULES_BY_ID[r].name))
    detail = detail.rename(columns={"score": "점수", "reason": "사유"})[["전표번호", "규칙", "점수", "사유"]]
    detail = detail.sort_values(["규칙", "점수"], ascending=[True, False])

    order = {no: i for i, no in enumerate(scored["전표번호"])}
    lines = gl[gl["전표번호"].isin(order)].copy()
    lines["위험순위"] = [order[no] + 1 for no in lines["전표번호"]]
    lines = lines.sort_values(["위험순위", "_row"])[["위험순위", *GL_COLUMNS]]

    sheets = {
        "요약": None,
        "의심전표": _dates_to_text(scored),
        "규칙별결과": detail,
        "분개라인": _dates_to_text(lines),
    }
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        rule_header_row = len(overview) + 3
        overview.to_excel(xw, sheet_name="요약", index=False)
        by_rule.to_excel(xw, sheet_name="요약", index=False, startrow=rule_header_row - 1)
        ws = xw.sheets["요약"]
        for row in (1, rule_header_row):
            for cell in ws[row]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = HEADER_FILL
        for letter, width in zip("ABCD", (28, 60, 14, 18)):
            ws.column_dimensions[letter].width = width
        for row in ws.iter_rows(min_row=rule_header_row + 1):
            row[1].alignment = Alignment(wrap_text=True, vertical="top")
            row[3].number_format = "#,##0"
        for name, df in sheets.items():
            if df is None:
                continue
            df.to_excel(xw, sheet_name=name, index=False)
            _style(xw.sheets[name], df)
    return buf.getvalue()
