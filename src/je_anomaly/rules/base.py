"""모든 탐지 규칙이 공유하는 결과 형식.

각 규칙 함수는 라인 단위 원장(DataFrame)을 받아, 걸린 전표만 담은 DataFrame을 반환한다.
    전표번호 | rule_id | score(0~100) | reason
"""
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

RESULT_COLUMNS = ["전표번호", "rule_id", "score", "reason"]


def make_result(je_nos: Iterable, rule_id: str, scores, reasons: Iterable[str]) -> pd.DataFrame:
    df = pd.DataFrame({"전표번호": list(je_nos), "score": scores, "reason": list(reasons)})
    if df.empty:
        return empty_result()
    df["rule_id"] = rule_id
    df["score"] = df["score"].astype(float).clip(0, 100).round(1)
    # 한 전표가 같은 규칙에 여러 번 걸리면 가장 높은 점수 하나만 남긴다
    df = df.sort_values("score", ascending=False).drop_duplicates("전표번호")
    return df[RESULT_COLUMNS].reset_index(drop=True)


def empty_result() -> pd.DataFrame:
    return pd.DataFrame({"전표번호": pd.Series(dtype=str), "rule_id": pd.Series(dtype=str),
                         "score": pd.Series(dtype=float), "reason": pd.Series(dtype=str)})


def won(amount: float) -> str:
    return f"{int(amount):,}원"
