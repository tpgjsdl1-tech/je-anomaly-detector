"""규칙별 결과를 합쳐 전표별 종합 위험도를 산출한다.

종합 위험도 = 100 × (1 − Π(1 − 규칙점수/100))
  - 규칙 하나만 걸리면 그 점수 그대로, 여러 규칙에 걸릴수록 100에 가까워진다 (단순 합계처럼 100을 넘지 않음).
  - 예) 60점 + 70점 → 88점
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from je_anomaly.ledger import ensure_prepared, entry_summary
from je_anomaly.rules import RULES, RULES_BY_ID, RESULT_COLUMNS

SCORED_COLUMNS = ["위험도", "규칙수", "탐지규칙", "전표번호", "전표일자", "작성자", "승인자", "계정과목",
                  "금액", "거래처", "적요", "사유"]


def run_rules(gl: pd.DataFrame, rule_ids: list[str] | None = None,
              params: dict[str, dict] | None = None) -> pd.DataFrame:
    """선택한 규칙을 모두 실행하고 결과를 이어 붙인다. params = {rule_id: {인자: 값}}."""
    gl = ensure_prepared(gl)
    params = params or {}
    rules = RULES if rule_ids is None else [RULES_BY_ID[r] for r in rule_ids]
    frames = [r.func(gl, **params.get(r.id, {})) for r in rules]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def combine_scores(scores: pd.Series) -> float:
    return float(100 * (1 - np.prod(1 - scores.to_numpy() / 100)))


def score_entries(gl: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """규칙에 한 번이라도 걸린 전표를 위험도 내림차순으로 정렬해 반환."""
    if results.empty:
        return pd.DataFrame(columns=SCORED_COLUMNS)
    res = results.assign(규칙명=results["rule_id"].map(lambda r: RULES_BY_ID[r].name))
    res = res.sort_values("score", ascending=False)
    agg = res.groupby("전표번호").agg(
        위험도=("score", combine_scores),
        규칙수=("rule_id", "nunique"),
        탐지규칙=("규칙명", ", ".join),
    )
    agg["사유"] = res.groupby("전표번호").apply(
        lambda g: "\n".join(f"[{n}] {r}" for n, r in zip(g["규칙명"], g["reason"])), include_groups=False)
    agg["위험도"] = agg["위험도"].round(1)

    ent = entry_summary(gl).set_index("전표번호")
    out = agg.join(ent).reset_index()
    out = out.sort_values(["위험도", "금액"], ascending=False, ignore_index=True)
    return out[SCORED_COLUMNS]


def detect(gl: pd.DataFrame, rule_ids: list[str] | None = None,
           params: dict[str, dict] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(규칙별 결과, 전표별 종합 위험도) 반환."""
    gl = ensure_prepared(gl)
    results = run_rules(gl, rule_ids, params)
    return results, score_entries(gl, results)
