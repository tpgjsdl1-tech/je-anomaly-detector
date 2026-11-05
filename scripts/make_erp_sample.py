"""ERP에서 내려받은 것 같은 '지저분한' 원장 엑셀 샘플을 만든다 (열 연결 기능 연습용).

- 위쪽에 제목·회사명·조회기간 줄
- 열 이름이 표준과 다름 (회계일자, 입력자, 결재자, 계정명, 차변금액 ...)
- 열 순서가 다르고, 쓰지 않는 열(부서, 증빙구분)이 섞여 있음
- 날짜는 엑셀 날짜 셀, 계정코드는 숫자 셀
- 맨 아래 합계 줄

사용법:  python scripts/make_erp_sample.py
"""
from pathlib import Path

import pandas as pd

from je_anomaly.generator import generate_ledger

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sample_erp_export_2025.xlsx"


def main() -> None:
    gl, _ = generate_ledger(seed=42)
    erp = pd.DataFrame({
        "회계일자": pd.to_datetime(gl["전표일자"]),
        "전표No": gl["전표번호"],
        "부서": "재무팀",
        "계정코드": gl["계정코드"].astype(int),
        "계정명": gl["계정과목"],
        "차변금액": gl["차변"],
        "대변금액": gl["대변"],
        "적요내용": gl["적요"],
        "거래처명": gl["거래처"],
        "증빙구분": "",
        "입력자": gl["작성자"],
        "결재자": gl["승인자"],
    })
    total = {c: "" for c in erp.columns}
    total.update({"회계일자": "", "계정명": "합   계", "차변금액": int(gl["차변"].sum()),
                  "대변금액": int(gl["대변"].sum())})
    erp = pd.concat([erp, pd.DataFrame([total])], ignore_index=True)

    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        erp.to_excel(xw, sheet_name="원장", index=False, startrow=4)
        ws = xw.sheets["원장"]
        ws["A1"] = "총 계 정 원 장"
        ws["A2"] = "회사명: (주)가상정밀"
        ws["A3"] = "조회기간: 2025-01-01 ~ 2025-12-31"
        for cell in ws["A"][5:]:
            cell.number_format = "yyyy-mm-dd"
        for col, width in zip("ABCDEFGHIJKL", (12, 18, 8, 9, 14, 14, 14, 24, 18, 8, 8, 8)):
            ws.column_dimensions[col].width = width
        pd.DataFrame({"안내": ["이 파일은 열 연결 기능 연습용 가상 데이터입니다."]}).to_excel(
            xw, sheet_name="설명", index=False)
    print(f"저장: {OUT} ({len(erp):,}줄, 합계 줄 포함)")


if __name__ == "__main__":
    main()
