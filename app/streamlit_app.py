"""전표 이상징후 탐지 대시보드.

실행:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from je_anomaly.calendar_kr import FISCAL_YEAR
from je_anomaly.ledger import GL_COLUMNS, prepare
from je_anomaly.mapping import (
    apply_mapping,
    detect_header_row,
    disabled_rules,
    excel_sheets,
    is_excel,
    missing_required,
    read_raw,
    suggest_mapping,
    with_header,
)
from je_anomaly.report import build_excel, rule_summary
from je_anomaly.rules import RULES, RULES_BY_ID
from je_anomaly.rules.benford import benford_lines, benford_mad, benford_table, mad_conformity
from je_anomaly.scoring import detect

DATA = Path(__file__).resolve().parents[1] / "data"
NO_SAMPLE = "선택 안 함"
SAMPLES = {  # 화면 표시 이름 → 파일 (있는 것만 노출)
    name: path for name, path in {
        "샘플 1 · 제조업 (CSV)": DATA / "general_ledger_2025.csv",
        "샘플 2 · 제조업, ERP 엑셀 형식": DATA / "compare" / "B사_원장_2025.xlsx",
    }.items() if path.exists()
}
HIGH_RISK = 80
NONE = "(없음)"
OPTIONAL_NOTE = {"작성자": "선택", "승인자": "선택", "적요": "선택", "거래처": "선택",
                 "계정코드": "코드·과목 중 하나 필수", "계정과목": "코드·과목 중 하나 필수"}

st.set_page_config(page_title="회계전표 이상징후 탐지 보조 툴", page_icon="🔎", layout="wide")


# ── 캐시되는 계산 ─────────────────────────────────────────────────
@st.cache_data(show_spinner="파일을 읽는 중...")
def load_raw(raw: bytes, filename: str, sheet) -> pd.DataFrame:
    return read_raw(raw, filename, sheet)


@st.cache_data(show_spinner=False)
def list_sheets(raw: bytes) -> list[str]:
    return excel_sheets(raw)


@st.cache_data(show_spinner="원장을 정리하는 중...")
def build_ledger(table: pd.DataFrame, mapping_json: str) -> tuple[pd.DataFrame, int]:
    mapped = apply_mapping(table, json.loads(mapping_json))
    return prepare(mapped), mapped.attrs.get("dropped_rows", 0)


@st.cache_data(show_spinner="탐지 규칙 실행 중...")
def run_detection(gl: pd.DataFrame, rule_ids: tuple[str, ...], params_json: str):
    return detect(gl, list(rule_ids), json.loads(params_json))


@st.cache_data(show_spinner="엑셀 보고서 작성 중...")
def excel_report(gl, results, scored, source_name: str) -> bytes:
    return build_excel(gl, results, scored, source_name)


def won_short(v: float) -> str:
    return f"{v / 1e8:,.1f}억원" if v >= 1e8 else f"{v / 1e4:,.0f}만원"


# ── ① 원장 불러오기 (여러 파일 가능) ─────────────────────────────
with st.sidebar:
    st.header("① 원장 불러오기")
    uploads = st.file_uploader(
        "총계정원장 (엑셀 또는 CSV, 여러 개 가능)", type=["xlsx", "xlsm", "csv"], accept_multiple_files=True,
        help="ERP에서 내려받은 원장을 그대로 올리세요. 여러 파일을 올리면 화면 위쪽 탭으로 번갈아 볼 수 있습니다. "
             "위쪽 제목 줄은 건너뛰고, 열 이름은 다음 화면에서 맞춥니다.")
    if SAMPLES:
        st.multiselect("가상 샘플 원장 추가", list(SAMPLES), key="sample_choice", placeholder="선택 안 함")

st.title("🔎 회계전표 이상징후 탐지 보조 툴")
st.caption("Journal Entry Testing Assistant · 감사기준서 240 '경영진의 통제 무력화' 위험에 대응하는 분개 테스트를 돕습니다")


def _pick_sample(name: str) -> None:
    picked = st.session_state.get("sample_choice") or []
    if name not in picked:
        st.session_state["sample_choice"] = [*picked, name]


def landing_page() -> None:
    """파일을 올리기 전 첫 화면: 무엇을 하는 도구인지, 어떻게 쓰는지."""
    st.markdown("#### 총계정원장을 올리면, 감사인이 먼저 봐야 할 전표를 위험도 순으로 찾아 드립니다.")
    st.write("")

    steps = st.columns(3)
    steps[0].markdown("**① 원장 올리기**  \n왼쪽에서 ERP 원장 파일(엑셀·CSV)을 올립니다. "
                      "제목 줄이나 합계 줄이 있어도 그대로 올리면 됩니다.")
    steps[1].markdown("**② 열 연결 확인**  \n'회계일자 → 전표일자'처럼 열 이름을 자동으로 맞춥니다. "
                      "틀린 항목만 바꾸세요.")
    steps[2].markdown("**③ 검토 · 보고서**  \n규칙별 탐지 결과와 위험도 순 목록을 검토하고, "
                      "감사조서용 엑셀 보고서를 내려받습니다.")

    st.divider()
    st.markdown("##### 적용하는 탐지 규칙")
    grid = st.columns(4)
    for i, rule in enumerate(RULES):
        with grid[i % 4].container(border=True):
            st.markdown(f"**{i + 1}. {rule.name}**")
            st.caption(rule.description)
    with grid[len(RULES) % 4].container(border=True):
        st.markdown("**종합 위험도**")
        st.caption("여러 규칙에 걸릴수록 높아지는 0~100점. 기준값은 사이드바에서 조정")

    st.divider()
    left, right = st.columns(2)
    with left:
        st.markdown("##### 원장에 필요한 항목")
        st.markdown("**필수** 전표번호 · 전표일자 · 차변 · 대변 · 계정코드 또는 계정과목  \n"
                    "**선택** 작성자 · 승인자 · 적요 · 거래처 — 없으면 해당 규칙만 꺼집니다")
        st.caption("지원 형식: .xlsx · .xlsm · .csv (UTF-8 / CP949) · "
                   "날짜는 2025-01-02, 2025.01.02, 20250102, 엑셀 날짜 셀 모두 가능")
    with right:
        st.markdown("##### 데이터가 없다면")
        st.markdown("실제 원장 대신 가상 제조기업 원장으로 기능을 체험해 보세요.")
        for name in SAMPLES:
            st.button(name, on_click=_pick_sample, args=(name,), use_container_width=True)
        st.caption("내 컴퓨터에서 실행할 때 올린 파일은 외부로 전송되지 않습니다. "
                   "공개 웹에 배포한 경우에는 실제 고객사 원장을 올리지 마세요.")


# 열린 파일 목록: 화면 이름 → (내용, 파일명, 샘플 여부)
sources: dict[str, tuple[bytes, str, bool]] = {}
for f in uploads or []:
    label, n = f.name, 2
    while label in sources:  # 같은 이름의 파일을 두 번 올린 경우
        label, n = f"{f.name} ({n})", n + 1
    sources[label] = (f.getvalue(), f.name, False)
for name in st.session_state.get("sample_choice") or []:
    if name in SAMPLES:
        sources[name] = (SAMPLES[name].read_bytes(), SAMPLES[name].name, True)

if not sources:
    landing_page()
    st.stop()


# ── 파일 탭: 여러 원장을 위쪽에서 번갈아 보기 ───────────────────────
# 화면에 안 보이는 위젯의 값은 Streamlit이 지우므로, 파일별 위젯 키를 'fid::이름'으로 만들고
# 다른 파일을 보는 동안에도 그 값들을 다시 저장해 둔다 (열 연결·필터가 파일마다 유지됨).
SEP = "::"


def file_id(label: str) -> str:
    return hashlib.md5(f"{label}|{len(sources[label][0])}".encode()).hexdigest()[:8]


def _keep_active() -> None:  # 이미 선택된 탭을 다시 눌러 선택이 풀리는 것 방지
    if st.session_state.get("active_file") is None:
        st.session_state["active_file"] = st.session_state.get("_last_active")


labels = list(sources)
known = st.session_state.get("_known_files", [])
added = [lb for lb in labels if lb not in known]
if added:  # 새로 올린 파일이 있으면 그 파일을 바로 보여 준다
    st.session_state["active_file"] = added[-1]
elif st.session_state.get("active_file") not in labels:
    last = st.session_state.get("_last_active")
    st.session_state["active_file"] = last if last in labels else labels[0]
st.session_state["_known_files"] = labels

if len(labels) > 1:
    st.segmented_control(
        "원장 파일", labels, key="active_file", on_change=_keep_active, label_visibility="collapsed",
        format_func=lambda lb: ("🧪 " if sources[lb][2] else "📄 ") + lb)
    st.caption(f"원장 {len(labels)}개 · 탭을 눌러 전환합니다. 열 연결·필터는 파일마다 따로 기억됩니다. "
               "탐지 규칙과 임계값(사이드바)은 모든 파일에 같이 적용됩니다.")
active = st.session_state["active_file"]
st.session_state["_last_active"] = active
fid = file_id(active)

live_ids = {file_id(lb) for lb in labels}
for k in list(st.session_state.keys()):
    if not isinstance(k, str) or SEP not in k:
        continue
    owner = k.split(SEP, 1)[0]
    if owner not in live_ids:
        del st.session_state[k]  # 닫은 파일의 값 정리
    elif owner != fid:
        st.session_state[k] = st.session_state[k]  # 지금 안 보이는 파일의 값 보존

raw, source_name, is_sample = sources[active]
if is_sample:
    st.info(f"가상 샘플 원장으로 보는 중입니다: **{active}**. 실제 원장은 왼쪽에서 올리세요.")

# 엑셀이면 시트 선택
sheet = 0
if is_excel(source_name):
    try:
        sheets = list_sheets(raw)
    except Exception as e:  # 손상·암호 걸린 파일 등
        st.error(f"엑셀 파일을 열 수 없습니다: {e}")
        st.stop()
    if len(sheets) > 1:
        sheet = st.sidebar.selectbox(f"시트 ({active})", sheets, key=f"{fid}{SEP}sheet")

try:
    raw_table = load_raw(raw, source_name, sheet)
except ValueError as e:
    st.error(f"파일을 읽을 수 없습니다: {e}")
    st.stop()
if raw_table.empty:
    st.error("파일(시트)에 데이터가 없습니다.")
    st.stop()

# ── 열 연결 (매핑) ──────────────────────────────────────────────
file_key = f"{fid}{SEP}{sheet}"
auto_header = detect_header_row(raw_table)
auto_mapping = suggest_mapping(with_header(raw_table, auto_header).columns)
needs_attention = bool(missing_required(auto_mapping)) or any(v != k for k, v in auto_mapping.items())

with st.expander("🔗 열 연결 — 원장의 열을 표준 항목에 맞추기", expanded=needs_attention):
    header_row = st.number_input(
        "머리글(열 이름)이 있는 줄", 1, min(30, len(raw_table)), auto_header + 1, key=f"{file_key}{SEP}hdr",
        help=f"자동 감지: {auto_header + 1}번째 줄. 위쪽 제목·회사명 줄은 이 줄 위에 있으면 무시됩니다.") - 1
    table = with_header(raw_table, header_row)
    suggested = suggest_mapping(table.columns) if header_row != auto_header else auto_mapping
    options = [NONE, *table.columns]

    mapping: dict[str, str | None] = {}
    cols = st.columns(5)
    for i, std in enumerate(GL_COLUMNS):
        default = suggested.get(std)
        choice = cols[i % 5].selectbox(
            std + (f" ({OPTIONAL_NOTE[std]})" if std in OPTIONAL_NOTE else " *"),
            options, index=options.index(default) if default in options else 0,
            key=f"{file_key}{SEP}map_{std}_{header_row}")
        mapping[std] = None if choice == NONE else choice
    st.caption("* 필수 · 자동으로 맞춘 결과를 확인하고, 틀린 항목만 바꾸세요. 미리보기는 원본 첫 5줄입니다.")
    st.dataframe(table.head(5), hide_index=True, use_container_width=True)

missing = missing_required(mapping)
if missing:
    st.warning(f"'🔗 열 연결'에서 필수 항목을 연결해 주세요: {', '.join(missing)}")
    st.stop()
chosen = [c for c in mapping.values() if c]
if len(chosen) != len(set(chosen)):
    st.warning("같은 열을 두 항목에 연결했습니다. '🔗 열 연결'을 확인해 주세요.")
    st.stop()

try:
    gl, dropped_rows = build_ledger(table, json.dumps(mapping, ensure_ascii=False))
except ValueError as e:
    st.error(f"원장을 정리할 수 없습니다: {e}")
    st.stop()
if gl.empty:
    st.error("전표번호가 있는 줄이 없습니다. 열 연결을 확인해 주세요.")
    st.stop()
if dropped_rows:
    st.caption(f"전표번호가 비어 있는 {dropped_rows:,}줄(합계·소계 등)은 제외했습니다.")

# ── ② 탐지 규칙 · ③ 표시 기준 ───────────────────────────────────
blocked = disabled_rules(mapping)
data_year = int(gl["전표일자"].dt.year.max())
with st.sidebar:
    st.header("② 탐지 규칙")
    selected = []
    for r in RULES:
        reason = blocked.get(r.id)
        on = st.checkbox(r.name + (f" — 사용 불가({reason})" if reason else ""), value=not reason,
                         disabled=bool(reason), help=r.description, key=f"use_{r.id}")
        if on and not reason:
            selected.append(r.id)

    with st.expander("임계값 조정"):
        pe_date = st.date_input("결산 · 결산일", date(data_year, 12, 31),
                                help="3월·6월 결산법인은 결산일을 바꾸세요")
        dup_window = st.number_input("중복 · 재입력 판단 기간(영업일)", 0, 40, 5)
        round_unit = st.selectbox("라운드 · 금액 단위", [100_000, 1_000_000, 10_000_000], index=1,
                                  format_func=lambda v: f"{v:,}원")
        round_repeat = st.number_input("라운드 · 같은 거래처 최소 반복", 2, 20, 3)
        benford_z = st.slider("벤포드 · z 임계값", 2.0, 5.0, 3.0, 0.1)
        pe_amount = st.number_input("결산 · 기준 금액(원)", 0, None, 30_000_000, 5_000_000,
                                    help="수행중요성 수준을 넣는 것을 권장")
        pe_days = st.number_input("결산 · 결산일 전 기간(일)", 1, 60, 7)
        pe_pattern = st.text_input("결산 · 수기 전표 판별(전표번호 정규식)", r"^GJ")
        rare_max = st.number_input("계정조합 · 희소 조합 기준(연간 N회 이하)", 0, 20, 3)

    params = {
        "DUPLICATE": {"window_days": int(dup_window)},
        "ROUND_AMOUNT": {"unit": int(round_unit), "min_repeat": int(round_repeat)},
        "BENFORD": {"z_threshold": float(benford_z), "drilldown_z": float(benford_z)},
        "PERIOD_END_MANUAL": {"min_amount": float(pe_amount), "window_days": int(pe_days),
                              "manual_pattern": pe_pattern, "period_end": pe_date.isoformat()},
        "UNUSUAL_PAIR": {"rare_max": int(rare_max)},
    }

    st.header("③ 표시 기준")
    min_risk = st.slider("최소 위험도", 0, 100, 0, 5)

try:
    re.compile(pe_pattern)
except re.error:
    st.error(f"'수기 전표 판별' 정규식이 올바르지 않습니다: {pe_pattern}")
    st.stop()

if data_year != FISCAL_YEAR and "WEEKEND_HOLIDAY" in selected:
    st.info(f"공휴일 목록은 {FISCAL_YEAR}년만 들어 있어, {data_year}년 원장은 주말만 탐지하고 공휴일은 탐지하지 않습니다.")
if "PERIOD_END_MANUAL" in selected and not gl["전표번호"].str.contains(pe_pattern, regex=True).any():
    st.info(f"전표번호가 '{pe_pattern}' 형식인 수기 전표가 없습니다. "
            "사이드바 '임계값 조정 → 수기 전표 판별'을 이 회사의 전표번호 규칙에 맞게 바꾸세요.")

unbalanced = gl.groupby("전표번호")[["차변", "대변"]].sum().pipe(lambda s: s[s["차변"] != s["대변"]])
if len(unbalanced):
    st.warning(f"대차가 맞지 않는 전표가 {len(unbalanced):,}건 있습니다. 원장 추출 범위를 확인하세요.")

if not selected:
    st.warning("탐지 규칙을 하나 이상 선택하세요.")
    st.stop()

results, scored = run_detection(gl, tuple(selected), json.dumps(params, ensure_ascii=False))
view = scored[scored["위험도"] >= min_risk]
view_results = results[results["전표번호"].isin(view["전표번호"])]

n_entries = gl["전표번호"].nunique()
c1, c2, c3, c4 = st.columns(4)
c1.metric("총 전표", f"{n_entries:,}건", f"{len(gl):,} 라인", delta_color="off")
c2.metric("의심 전표", f"{len(view):,}건", f"전체의 {len(view) / n_entries:.1%}", delta_color="off")
c3.metric(f"고위험 ({HIGH_RISK}점 이상)", f"{int((view['위험도'] >= HIGH_RISK).sum()):,}건")
c4.metric("의심 금액 합계", won_short(view["금액"].sum()))

st.download_button(
    "📥 엑셀 보고서 다운로드 (요약 · 의심전표 · 규칙별결과 · 분개라인)",
    data=excel_report(gl, view_results, view, source_name),
    file_name=f"회계전표_이상징후_탐지결과_{Path(source_name).stem}_{date.today():%Y%m%d}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    type="primary",
)

tab_list, tab_rules, tab_benford, tab_gl = st.tabs(["🚩 의심 전표", "📊 규칙별 결과", "🔢 벤포드 분석", "📄 원장"])

# ── 탭 1: 위험도 순 의심 전표 ────────────────────────────────────
with tab_list:
    rule_names = [RULES_BY_ID[r].name for r in selected]
    pick_key = f"{fid}{SEP}pick"
    if pick_key in st.session_state:  # 규칙을 끈 뒤 남은 선택값 정리
        st.session_state[pick_key] = [p for p in st.session_state[pick_key] if p in rule_names]
    pick = st.multiselect("규칙으로 거르기", rule_names, placeholder="전체 규칙", key=pick_key)
    shown = view
    if pick:
        shown = view[view["탐지규칙"].apply(lambda s: any(p in s.split(", ") for p in pick))]
    shown = shown.reset_index(drop=True)

    st.caption(f"{len(shown):,}건 · 위험도 내림차순 · 행을 클릭하면 분개 내역이 아래에 표시됩니다")
    event = st.dataframe(
        shown,
        hide_index=True,
        use_container_width=True,
        height=420,
        on_select="rerun",
        selection_mode="single-row",
        key=f"suspects_{fid}",
        column_config={
            "위험도": st.column_config.ProgressColumn("위험도", min_value=0, max_value=100, format="%.0f"),
            "전표일자": st.column_config.DateColumn("전표일자", format="YYYY-MM-DD"),
            "금액": st.column_config.NumberColumn("금액", format="localized"),
            "사유": st.column_config.TextColumn("사유", width="large"),
        },
    )
    rows = event.selection.rows if event and event.selection else []
    if rows:
        sel = shown.iloc[rows[0]]
        st.subheader(f"{sel['전표번호']} · 위험도 {sel['위험도']:.0f}")
        for line in sel["사유"].split("\n"):
            st.markdown(f"- {line}")
        lines = gl[gl["전표번호"] == sel["전표번호"]][GL_COLUMNS]
        st.dataframe(lines, hide_index=True, use_container_width=True, column_config={
            "전표일자": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "차변": st.column_config.NumberColumn(format="localized"),
            "대변": st.column_config.NumberColumn(format="localized"),
        })

# ── 탭 2: 규칙별 결과 ───────────────────────────────────────────
with tab_rules:
    summary = rule_summary(gl, view_results)
    summary = summary[summary["규칙"].isin(rule_names)]
    left, right = st.columns([3, 2])
    with left:
        st.dataframe(summary, hide_index=True, use_container_width=True, column_config={
            "의심 금액 합계": st.column_config.NumberColumn(format="localized"),
        })
    with right:
        st.altair_chart(
            alt.Chart(summary).mark_bar().encode(
                x=alt.X("탐지 전표 수:Q"),
                y=alt.Y("규칙:N", sort="-x", title=None),
                tooltip=["규칙", "탐지 전표 수", alt.Tooltip("의심 금액 합계:Q", format=",")],
            ).properties(height=260),
            use_container_width=True,
        )

    st.markdown("**위험도 분포**")
    st.altair_chart(
        alt.Chart(view).mark_bar().encode(
            x=alt.X("위험도:Q", bin=alt.Bin(step=5), title="위험도"),
            y=alt.Y("count():Q", title="전표 수"),
        ).properties(height=180),
        use_container_width=True,
    )

    info = scored.set_index("전표번호")[["전표일자", "작성자", "승인자", "금액"]]
    for rid in selected:
        rule = RULES_BY_ID[rid]
        hit = view_results[view_results["rule_id"] == rid]
        with st.expander(f"{rule.name} — {len(hit):,}건"):
            st.caption(rule.description)
            if hit.empty:
                st.write("탐지된 전표가 없습니다.")
                continue
            table = hit.join(info, on="전표번호").sort_values("score", ascending=False)
            st.dataframe(table[["전표번호", "전표일자", "작성자", "승인자", "금액", "score", "reason"]],
                         hide_index=True, use_container_width=True, column_config={
                             "전표일자": st.column_config.DateColumn(format="YYYY-MM-DD"),
                             "금액": st.column_config.NumberColumn(format="localized"),
                             "score": st.column_config.NumberColumn("점수", format="%.0f"),
                             "reason": st.column_config.TextColumn("사유", width="large"),
                         })

# ── 탭 3: 벤포드 분석 ───────────────────────────────────────────
with tab_benford:
    st.caption("첫 두 자리(10~99) 분포를 벤포드 기대값과 비교합니다. 부가세·원천세 등 파생 금액과 반복 금액은 제외합니다.")
    blines = benford_lines(gl)
    groups = ["전체"] + sorted(blines["작성자"].unique())
    who_key = f"{fid}{SEP}who"
    if st.session_state.get(who_key) not in (None, *groups):
        del st.session_state[who_key]
    who = st.selectbox("작성자", groups, key=who_key)
    target = blines if who == "전체" else blines[blines["작성자"] == who]
    amounts = target.drop_duplicates("금액")["금액"]
    tbl = benford_table(amounts).reset_index()
    tbl["과다"] = (tbl["z"] >= benford_z) & (tbl["observed"] > tbl["expected"])
    mad = benford_mad(amounts)

    m1, m2, m3 = st.columns(3)
    m1.metric("검정 금액 수", f"{len(amounts):,}")
    m2.metric("MAD (평균절대편차)", f"{mad:.4f}")
    m3.metric("Nigrini 판정", mad_conformity(mad))

    bars = alt.Chart(tbl).mark_bar().encode(
        x=alt.X("digits:O", title="첫 두 자리", axis=alt.Axis(labelAngle=0, values=list(range(10, 100, 5)))),
        y=alt.Y("observed:Q", title="비율", axis=alt.Axis(format="%")),
        color=alt.condition("datum.과다", alt.value("#d62728"), alt.value("#4c78a8")),
        tooltip=["digits", "count", alt.Tooltip("observed:Q", format=".2%"),
                 alt.Tooltip("expected:Q", format=".2%"), alt.Tooltip("z:Q", format=".1f")],
    )
    line = alt.Chart(tbl).mark_line(color="#ff9f1c", strokeWidth=2).encode(x="digits:O", y="expected:Q")
    st.altair_chart((bars + line).properties(height=340), use_container_width=True)
    st.caption(f"막대 = 관측 비율, 주황 선 = 벤포드 기대 비율, 빨간 막대 = z ≥ {benford_z} 과다 숫자대")

# ── 탭 4: 원장 ─────────────────────────────────────────────────
with tab_gl:
    q = st.text_input("전표번호 · 적요 · 거래처 검색", key=f"{fid}{SEP}q")
    show = gl[GL_COLUMNS]
    if q:
        mask = (show["전표번호"].str.contains(q, regex=False) | show["적요"].str.contains(q, regex=False)
                | show["거래처"].str.contains(q, regex=False))
        show = show[mask]
    st.caption(f"{len(show):,} 라인")
    st.dataframe(show, hide_index=True, use_container_width=True, height=500, column_config={
        "전표일자": st.column_config.DateColumn(format="YYYY-MM-DD"),
        "차변": st.column_config.NumberColumn(format="localized"),
        "대변": st.column_config.NumberColumn(format="localized"),
    })
