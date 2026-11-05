"""ERP마다 다른 원장 파일을 표준 10개 컬럼으로 맞추는 기능.

1) 파일 읽기: CSV(UTF-8/CP949) 또는 엑셀(.xlsx/.xlsm), 시트 선택
2) 머리글 행 자동 감지: ERP 추출 파일 위쪽의 제목·회사명·조회기간 줄을 건너뜀
3) 열 이름 자동 매칭: '회계일자 → 전표일자', '차변금액 → 차변' 같은 흔한 별칭을 인식
4) 표준 원장으로 변환: 합계·소계처럼 전표번호가 없는 줄 제거
"""
from __future__ import annotations

import csv
import io
import re

import pandas as pd

from je_anomaly.ledger import GL_COLUMNS

EXCEL_EXTENSIONS = (".xlsx", ".xlsm")

# 꼭 있어야 하는 컬럼 (계정은 코드·과목 중 하나만 있어도 됨)
REQUIRED = ("전표번호", "전표일자", "차변", "대변")
ACCOUNT = ("계정코드", "계정과목")

# 없으면 꺼야 하는 규칙
RULES_NEEDING = {"SELF_APPROVAL": ("작성자", "승인자")}

ALIASES: dict[str, tuple[str, ...]] = {
    "전표번호": ("전표번호", "전표no", "전표넘버", "전표키", "전표id", "문서번호", "회계전표번호",
              "voucherno", "jeno", "journalno", "documentno", "docno", "belnr"),
    "전표일자": ("전표일자", "회계일자", "전기일", "전기일자", "거래일자", "증빙일자", "기표일자", "일자",
              "postingdate", "glDate", "date", "budat"),
    "작성자": ("작성자", "입력자", "등록자", "기표자", "작성자명", "입력자명", "등록자명", "작성사원",
            "createdby", "preparer", "enteredby", "usnam"),
    "승인자": ("승인자", "결재자", "승인자명", "결재자명", "확정자", "approver", "approvedby"),
    "계정코드": ("계정코드", "계정번호", "계정cd", "accountcode", "accountno", "glaccount", "hkont", "acctcode"),
    "계정과목": ("계정과목", "계정명", "계정과목명", "accountname", "account", "acctname"),
    "차변": ("차변", "차변금액", "차변액", "debit", "debitamount", "dr"),
    "대변": ("대변", "대변금액", "대변액", "credit", "creditamount", "cr"),
    "적요": ("적요", "적요명", "적요내용", "내용", "description", "memo", "text", "sgtxt"),
    "거래처": ("거래처", "거래처명", "거래처명칭", "업체명", "상호", "vendor", "customer", "partner",
            "counterparty", "vendorname"),
}


def _norm(text) -> str:
    return re.sub(r"[\s_\-\.\(\)\[\]/:]", "", str(text)).lower()


_ALIAS_NORM = {std: tuple(_norm(a) for a in aliases) for std, aliases in ALIASES.items()}


# ── 파일 읽기 ───────────────────────────────────────────────────
def is_excel(filename: str) -> bool:
    return filename.lower().endswith(EXCEL_EXTENSIONS)


def excel_sheets(raw: bytes) -> list[str]:
    return pd.ExcelFile(io.BytesIO(raw), engine="openpyxl").sheet_names


def read_raw(raw: bytes, filename: str, sheet: str | int = 0) -> pd.DataFrame:
    """머리글 없이 모든 셀을 그대로 읽는다 (머리글 행은 detect_header_row로 찾음)."""
    if is_excel(filename):
        df = pd.read_excel(io.BytesIO(raw), sheet_name=sheet, header=None, dtype=object, engine="openpyxl")
    else:
        df = _read_csv_ragged(raw)
    return df.replace("", None).dropna(how="all").reset_index(drop=True)


def _read_csv_ragged(raw: bytes) -> pd.DataFrame:
    """줄마다 칸 수가 달라도 읽는다 (제목 줄은 1칸, 원장 본문은 10칸 이상인 ERP 추출 CSV)."""
    for enc in ("utf-8-sig", "cp949"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("CSV 인코딩을 인식할 수 없습니다 (UTF-8 또는 CP949로 저장해 주세요).")
    rows = list(csv.reader(io.StringIO(text)))
    width = max((len(r) for r in rows), default=0)
    return pd.DataFrame([r + [""] * (width - len(r)) for r in rows], dtype=object)


def _header_score(values) -> int:
    norms = {_norm(v) for v in values if pd.notna(v) and str(v).strip()}
    return sum(1 for aliases in _ALIAS_NORM.values() if norms & set(aliases))


def detect_header_row(raw_df: pd.DataFrame, max_rows: int = 30) -> int:
    """위쪽 max_rows 줄 중 알려진 열 이름이 가장 많이 들어 있는 줄을 머리글로 본다."""
    best_row, best_score = 0, 0
    for i in range(min(max_rows, len(raw_df))):
        score = _header_score(raw_df.iloc[i].tolist())
        if score > best_score:
            best_row, best_score = i, score
    return best_row


def with_header(raw_df: pd.DataFrame, header_row: int) -> pd.DataFrame:
    names, seen = [], {}
    for j, v in enumerate(raw_df.iloc[header_row].tolist()):
        name = str(v).strip() if pd.notna(v) and str(v).strip() else f"열{j + 1}"
        if name in seen:  # 같은 이름이 두 번 나오면 구분
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        names.append(name)
    body = raw_df.iloc[header_row + 1:].copy()
    body.columns = names
    return body.dropna(how="all").reset_index(drop=True)


# ── 열 매칭 ────────────────────────────────────────────────────
def suggest_mapping(columns) -> dict[str, str | None]:
    """표준 컬럼 → 원본 컬럼 추천. 정확히 같은 이름을 먼저, 그다음 별칭이 포함된 이름을 찾는다."""
    columns = [str(c) for c in columns]
    norm_cols = {c: _norm(c) for c in columns}
    mapping: dict[str, str | None] = {std: None for std in GL_COLUMNS}
    used: set[str] = set()

    for std in GL_COLUMNS:  # 1차: 정확히 일치 (별칭 순서 = 우선순위)
        for alias in _ALIAS_NORM[std]:
            hit = next((c for c in columns if c not in used and norm_cols[c] == alias), None)
            if hit:
                mapping[std], used = hit, used | {hit}
                break
    for std in GL_COLUMNS:  # 2차: 별칭을 포함 (3글자 이상 별칭만 — '일자'가 '승인일자'에 걸리지 않게)
        if mapping[std]:
            continue
        for alias in _ALIAS_NORM[std]:
            if len(alias) < 3:
                continue
            hit = next((c for c in columns if c not in used and alias in norm_cols[c]), None)
            if hit:
                mapping[std], used = hit, used | {hit}
                break
    return mapping


def missing_required(mapping: dict[str, str | None]) -> list[str]:
    missing = [c for c in REQUIRED if not mapping.get(c)]
    if not any(mapping.get(c) for c in ACCOUNT):
        missing.append("계정코드 또는 계정과목")
    return missing


def disabled_rules(mapping: dict[str, str | None]) -> dict[str, str]:
    """매핑되지 않은 컬럼 때문에 쓸 수 없는 규칙 → 사유."""
    out = {}
    for rule_id, cols in RULES_NEEDING.items():
        lacking = [c for c in cols if not mapping.get(c)]
        if lacking:
            out[rule_id] = f"{', '.join(lacking)} 열이 없음"
    return out


def apply_mapping(df: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    """원본 → 표준 10개 컬럼. 없는 선택 컬럼은 빈칸, 전표번호 없는 줄(합계·소계)은 제거."""
    missing = missing_required(mapping)
    if missing:
        raise ValueError(f"필수 항목이 연결되지 않았습니다: {', '.join(missing)}")
    out = pd.DataFrame({std: (df[src].values if src else "") for std, src in mapping.items()},
                       index=df.index)[GL_COLUMNS]
    if not mapping.get("계정코드"):
        out["계정코드"] = out["계정과목"]
    if not mapping.get("계정과목"):
        out["계정과목"] = out["계정코드"]

    je_no = out["전표번호"].astype(str).str.strip()
    keep = out["전표번호"].notna() & (je_no != "") & (je_no.str.lower() != "nan")
    dropped = int((~keep).sum())
    out = out[keep].reset_index(drop=True)
    out.attrs["dropped_rows"] = dropped
    return out
