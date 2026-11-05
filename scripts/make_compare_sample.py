"""비교 분석용 두 번째 회사(B사) 원장을 만든다.

A사(data/general_ledger_2025.csv)와 같은 업종·규모지만 '위험 양상'이 다르다.
  - A사: 이상 유형이 고르게 섞임 (쪼개기·라운드 금액 등 개인 부정 신호 포함)
  - B사: 내부통제가 느슨한 회사 — 자기 승인, 주말 입력, 결산 직전 대형 수기 전표가 많음
         쪼개기(벤포드)·라운드 금액 반복은 없음

ERP 추출본 형식도 A사 연습 파일과 다르게 만들어 열 연결 기능을 한 번 더 써 보게 했다.

사용법:  python scripts/make_compare_sample.py
"""
from pathlib import Path

import pandas as pd

from je_anomaly.generator import (
    BENFORD,
    DUPLICATE,
    PERIOD_END_MANUAL,
    ROUND_AMOUNT,
    SELF_APPROVAL,
    UNUSUAL_PAIR,
    WEEKEND_HOLIDAY,
    generate_ledger,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "compare"

COMPANY_B_ANOMALIES = {
    DUPLICATE: 5,
    WEEKEND_HOLIDAY: 30,
    ROUND_AMOUNT: 0,
    BENFORD: 0,
    PERIOD_END_MANUAL: 20,
    SELF_APPROVAL: 25,
    UNUSUAL_PAIR: 8,
}


def main() -> None:
    gl, labels = generate_ledger(seed=2025, anomaly_counts=COMPANY_B_ANOMALIES,
                                 sales_per_day=1.8, purchases_per_day=1.2, expenses_per_day=2.2)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # A사 연습 파일과 다른 ERP 형식: 열 이름·순서가 다르고 날짜는 20250102 숫자 형식
    erp = pd.DataFrame({
        "문서번호": gl["전표번호"],
        "전기일자": pd.to_datetime(gl["전표일자"]).dt.strftime("%Y%m%d").astype(int),
        "계정과목명": gl["계정과목"],
        "계정번호": gl["계정코드"].astype(int),
        "차변액": gl["차변"],
        "대변액": gl["대변"],
        "업체명": gl["거래처"],
        "내용": gl["적요"],
        "등록자": gl["작성자"],
        "확정자": gl["승인자"],
        "사업장": "본사",
    })
    xlsx = OUT_DIR / "B사_원장_2025.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        erp.to_excel(xw, sheet_name="GL", index=False, startrow=2)
        ws = xw.sheets["GL"]
        ws["A1"] = "(주)B정공 총계정원장  |  기간 2025.01.01 ~ 2025.12.31"
        for col, width in zip("ABCDEFGHIJK", (18, 10, 14, 9, 14, 14, 18, 24, 8, 8, 8)):
            ws.column_dimensions[col].width = width

    label_path = OUT_DIR / "B사_정답라벨_2025.csv"
    labels.to_csv(label_path, index=False, encoding="utf-8-sig")

    planted = labels.loc[labels["is_anomaly"] == 1, "anomaly_type"].value_counts()
    print(f"저장: {xlsx}  ({len(gl):,}줄 / 전표 {gl['전표번호'].nunique():,}건)")
    print(f"저장: {label_path}  (이상 전표 {int(labels['is_anomaly'].sum())}건)")
    print(planted.to_string())


if __name__ == "__main__":
    main()
