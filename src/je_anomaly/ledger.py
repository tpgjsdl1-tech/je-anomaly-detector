"""총계정원장 전처리 및 전표 단위 요약."""
from __future__ import annotations

import io

import pandas as pd

GL_COLUMNS = ["전표번호", "전표일자", "작성자", "승인자", "계정코드", "계정과목", "차변", "대변", "적요", "거래처"]
_TEXT_COLUMNS = ["전표번호", "작성자", "승인자", "계정코드", "계정과목", "적요", "거래처"]


def _to_amount(s: pd.Series) -> pd.Series:
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        s = s.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce").fillna(0)


def _to_date(s: pd.Series) -> pd.Series:
    """2025-01-02 / 2025.01.02 / 2025/01/02 / 20250102 / 엑셀 날짜 셀을 모두 날짜로."""
    text = s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    compact = text.str.fullmatch(r"\d{8}")
    out = pd.to_datetime(text.where(~compact), errors="coerce", format="mixed")
    out[compact] = pd.to_datetime(text[compact], format="%Y%m%d", errors="coerce")
    bad = out.isna()
    if bad.any():
        examples = ", ".join(f"'{v}'" for v in s[bad].astype(str).unique()[:3])
        raise ValueError(f"전표일자를 날짜로 읽을 수 없는 줄이 {int(bad.sum()):,}개 있습니다 (예: {examples})")
    return out.dt.normalize()


def load_csv(source) -> pd.DataFrame:
    """CSV(경로·바이트·파일 객체)를 읽는다. ERP 추출 파일에 흔한 UTF-8(BOM)과 CP949를 차례로 시도."""
    raw = source if isinstance(source, bytes) else None
    if raw is None and hasattr(source, "read"):
        raw = source.read()
    if raw is None:
        with open(source, "rb") as f:
            raw = f.read()
    last_error: Exception | None = None
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, dtype={"전표번호": str, "계정코드": str})
        except UnicodeDecodeError as e:
            last_error = e
    raise ValueError("CSV 인코딩을 인식할 수 없습니다 (UTF-8 또는 CP949로 저장해 주세요).") from last_error


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """필수 컬럼 검증 + 형 변환. `_row`(원장상 입력 순서)를 추가한다."""
    missing = [c for c in GL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {', '.join(missing)}")
    out = df[GL_COLUMNS].copy()
    for col in _TEXT_COLUMNS:
        out[col] = out[col].fillna("").astype(str).str.strip()
    out["계정코드"] = out["계정코드"].str.replace(r"\.0$", "", regex=True)
    out["전표일자"] = _to_date(out["전표일자"])
    out["차변"] = _to_amount(out["차변"])
    out["대변"] = _to_amount(out["대변"])
    out["_row"] = range(len(out))
    return out


def ensure_prepared(gl: pd.DataFrame) -> pd.DataFrame:
    return gl if "_row" in gl.columns else prepare(gl)


def entry_summary(gl: pd.DataFrame) -> pd.DataFrame:
    """라인 단위 원장 → 전표 단위 요약 (금액 = 차변 합계)."""
    gl = ensure_prepared(gl)
    g = gl.groupby("전표번호", sort=False)
    ent = g.agg(
        전표일자=("전표일자", "first"),
        작성자=("작성자", "first"),
        승인자=("승인자", "first"),
        금액=("차변", "sum"),
        적요=("적요", "first"),
        _row=("_row", "min"),
    )
    vendors = gl[gl["거래처"] != ""].groupby("전표번호")["거래처"].first()
    ent["거래처"] = vendors.reindex(ent.index).fillna("")
    accounts = gl.groupby("전표번호", sort=False)["계정과목"].agg(lambda x: "/".join(dict.fromkeys(x)))
    ent["계정과목"] = accounts
    return ent.reset_index().sort_values("_row", ignore_index=True)
