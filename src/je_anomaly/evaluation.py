"""정답 라벨 대비 탐지 성능 평가 (전표 단위).

지표 정의
    탐지율(Recall)        = 탐지된 이상 전표 / 심어 둔 이상 전표
    오탐률(FPR)           = 탐지된 정상 전표 / 전체 정상 전표
    정밀도(Precision)     = 탐지된 이상 전표 / 탐지된 전표   (1 − 정밀도 = 탐지 목록 중 오탐 비중)
"""
from __future__ import annotations

import numbers
from collections.abc import Iterable

import pandas as pd

from je_anomaly.rules import RULES

EVAL_START = "<!-- EVAL:START -->"
EVAL_END = "<!-- EVAL:END -->"


def normalize_labels(labels: pd.DataFrame) -> pd.DataFrame:
    labels = labels.copy()
    labels["전표번호"] = labels["전표번호"].astype(str)
    labels["anomaly_type"] = labels["anomaly_type"].fillna("").astype(str)
    labels["is_anomaly"] = labels["is_anomaly"].astype(int)
    return labels


def confusion(flagged: Iterable[str], labels: pd.DataFrame) -> dict:
    flagged = set(flagged)
    labels = normalize_labels(labels)
    anomalies = set(labels.loc[labels["is_anomaly"] == 1, "전표번호"])
    normals = set(labels.loc[labels["is_anomaly"] == 0, "전표번호"])
    tp, fp = len(flagged & anomalies), len(flagged & normals)
    fn, tn = len(anomalies) - tp, len(normals) - fp
    return {
        "탐지": tp + fp, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "탐지율": tp / len(anomalies) if anomalies else float("nan"),
        "오탐률": fp / len(normals) if normals else float("nan"),
        "정밀도": tp / (tp + fp) if tp + fp else float("nan"),
    }


def overall_metrics(scored: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    labels = normalize_labels(labels)
    row = {"전체 전표": len(labels), "이상 전표": int(labels["is_anomaly"].sum()),
           **confusion(scored["전표번호"], labels)}
    return pd.DataFrame([row])


def per_rule_metrics(results: pd.DataFrame, scored: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """규칙별: 자기 유형을 얼마나 잡았나 + 그 규칙이 걸어 올린 전표의 구성."""
    labels = normalize_labels(labels)
    type_of = labels.set_index("전표번호")["anomaly_type"]
    n_normal = int((labels["is_anomaly"] == 0).sum())
    all_flagged = set(scored["전표번호"])
    rows = []
    for rule in RULES:
        planted = set(labels.loc[labels["anomaly_type"] == rule.id, "전표번호"])
        hit = set(results.loc[results["rule_id"] == rule.id, "전표번호"])
        hit_types = type_of.reindex(list(hit)).fillna("")
        same = int((hit_types == rule.id).sum())
        other = int(((hit_types != rule.id) & (hit_types != "")).sum())
        normal = int((hit_types == "").sum())
        rows.append({
            "규칙": rule.name,
            "심은 건수": len(planted),
            "규칙 탐지율": same / len(planted) if planted else float("nan"),
            "전체 탐지율": len(planted & all_flagged) / len(planted) if planted else float("nan"),
            "탐지 전표": len(hit),
            "같은 유형": same,
            "다른 이상유형": other,
            "정상(오탐)": normal,
            "정밀도": (same + other) / len(hit) if hit else float("nan"),
            "오탐률": normal / n_normal if n_normal else float("nan"),
        })
    return pd.DataFrame(rows)


def threshold_curve(scored: pd.DataFrame, labels: pd.DataFrame,
                    thresholds: Iterable[float] = (0, 40, 50, 60, 70, 80, 90)) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        m = confusion(scored.loc[scored["위험도"] >= t, "전표번호"], labels)
        rows.append({"최소 위험도": t, **{k: m[k] for k in ("탐지", "TP", "FP", "탐지율", "오탐률", "정밀도")}})
    return pd.DataFrame(rows)


def top_k(scored: pd.DataFrame, labels: pd.DataFrame, ks: Iterable[int] = (20, 50, 100)) -> pd.DataFrame:
    """위험도 상위 K건만 검토한다고 할 때 (감사인의 표본 검토 관점)."""
    labels = normalize_labels(labels)
    anomalies = set(labels.loc[labels["is_anomaly"] == 1, "전표번호"])
    ranked = list(scored["전표번호"])
    rows = []
    for k in ks:
        top = set(ranked[:k])
        tp = len(top & anomalies)
        rows.append({"검토 건수(상위 K)": k, "이상 전표": tp, "적중률": tp / min(k, len(ranked)) if ranked else 0,
                     "전체 이상 중 비중": tp / len(anomalies) if anomalies else float("nan")})
    return pd.DataFrame(rows)


def evaluate(gl: pd.DataFrame, labels: pd.DataFrame, rule_ids=None, params=None) -> dict[str, pd.DataFrame]:
    from je_anomaly.scoring import detect

    results, scored = detect(gl, rule_ids, params)
    return {
        "overall": overall_metrics(scored, labels),
        "per_rule": per_rule_metrics(results, scored, labels),
        "threshold": threshold_curve(scored, labels),
        "top_k": top_k(scored, labels),
    }


def multi_seed(seeds: Iterable[int]) -> pd.DataFrame:
    """seed를 바꿔 데이터를 여러 번 생성해 성능의 안정성을 본다."""
    from je_anomaly.generator import generate_ledger
    from je_anomaly.ledger import prepare
    from je_anomaly.scoring import detect

    rows = []
    for seed in seeds:
        gl, labels = generate_ledger(seed=seed)
        results, scored = detect(prepare(gl))
        m = confusion(scored["전표번호"], labels)
        fp_by_rule = per_rule_metrics(results, scored, labels).set_index("규칙")["정상(오탐)"]
        rows.append({"seed": seed, **{k: m[k] for k in ("탐지율", "오탐률", "정밀도", "FP")},
                     **{f"FP·{k}": int(v) for k, v in fp_by_rule.items()}})
    return pd.DataFrame(rows)


# ── 마크다운 출력 ──────────────────────────────────────────────
PCT_COLUMNS = {"탐지율", "오탐률", "정밀도", "규칙 탐지율", "전체 탐지율", "적중률", "전체 이상 중 비중"}


def _fmt(col: str, v) -> str:
    if isinstance(v, numbers.Real) and pd.isna(v):
        return "-"
    if col in PCT_COLUMNS:
        return f"{v:.1%}"
    if isinstance(v, numbers.Integral):
        return f"{int(v):,}"
    if isinstance(v, numbers.Real):
        return f"{v:,.1f}" if v != int(v) else f"{int(v):,}"
    return str(v)


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---:" if pd.api.types.is_numeric_dtype(df[c]) else "---" for c in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(c, row[c]) for c in cols) + " |")
    return "\n".join(lines)


def seed_summary(ms: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ms.columns if c != "seed"]
    return pd.DataFrame({
        "지표": cols,
        "평균": [ms[c].mean() for c in cols],
        "최소": [ms[c].min() for c in cols],
        "최대": [ms[c].max() for c in cols],
    })


def _seed_md(ms: pd.DataFrame) -> str:
    s = seed_summary(ms)
    lines = ["| 지표 | 평균 | 최소 | 최대 |", "|---|---:|---:|---:|"]
    for r in s.itertuples(index=False):
        if r.지표 in PCT_COLUMNS:
            vals = [f"{v:.1%}" for v in (r.평균, r.최소, r.최대)]
        else:
            vals = [f"{r.평균:.1f}", f"{int(r.최소)}", f"{int(r.최대)}"]
        lines.append(f"| {r.지표} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def render_markdown(report: dict[str, pd.DataFrame], seed: int, ms: pd.DataFrame | None = None) -> str:
    parts = [
        f"_`python scripts/evaluate.py`로 자동 생성 · 기준 데이터 seed {seed}_",
        "",
        "#### ① 전체 성능",
        to_markdown(report["overall"]),
        "",
        "#### ② 규칙별 성능",
        "`규칙 탐지율` = 해당 유형을 그 규칙이 잡은 비율, `전체 탐지율` = 어떤 규칙으로든 잡힌 비율, "
        "`다른 이상유형` = 다른 유형의 이상 전표를 함께 잡은 건수 (오탐 아님)",
        "",
        to_markdown(report["per_rule"]),
        "",
        "#### ③ 위험도 임계값별 트레이드오프",
        to_markdown(report["threshold"]),
        "",
        "#### ④ 위험도 상위 K건만 검토할 때",
        to_markdown(report["top_k"]),
    ]
    if ms is not None:
        seeds = sorted(ms["seed"])
        parts += ["", f"#### ⑤ 데이터 {len(ms)}종 반복 평가 (seed {', '.join(map(str, seeds))})",
                  "같은 규칙을 무작위로 다시 만든 원장에 적용해 결과가 우연이 아닌지 확인합니다. `FP·규칙` = 규칙별 오탐 건수.",
                  "", _seed_md(ms)]
    return "\n".join(parts)


def update_readme(readme_text: str, body: str) -> str:
    """README의 EVAL 마커 사이를 교체 (마커가 없으면 끝에 추가)."""
    block = f"{EVAL_START}\n{body}\n{EVAL_END}"
    if EVAL_START in readme_text and EVAL_END in readme_text:
        head, rest = readme_text.split(EVAL_START, 1)
        _, tail = rest.split(EVAL_END, 1)
        return head + block + tail
    return readme_text.rstrip() + "\n\n" + block + "\n"
