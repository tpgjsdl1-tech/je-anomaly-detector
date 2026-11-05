"""규칙 6. 작성자 = 승인자 (직무분리 위반), 승인자 누락."""
from __future__ import annotations

import pandas as pd

from je_anomaly.ledger import entry_summary
from je_anomaly.rules.base import make_result

RULE_ID = "SELF_APPROVAL"


def detect_self_approval(gl: pd.DataFrame, same_score: float = 70.0,
                         missing_score: float = 60.0) -> pd.DataFrame:
    ent = entry_summary(gl)
    same = ent[(ent["작성자"] != "") & (ent["작성자"] == ent["승인자"])]
    missing = ent[ent["승인자"] == ""]

    nos = list(same["전표번호"]) + list(missing["전표번호"])
    scores = [same_score] * len(same) + [missing_score] * len(missing)
    reasons = ([f"작성자와 승인자가 동일({w}) — 직무분리 위반" for w in same["작성자"]]
               + ["승인자 누락"] * len(missing))
    return make_result(nos, RULE_ID, scores, reasons)
