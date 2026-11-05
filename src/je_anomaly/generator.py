"""중소 제조기업 1년치 총계정원장(GL) 가상 데이터 생성기.

정상 거래
    매출(SA)·수금(RC), 매입(PU)·지급(PM), 일반경비(EX), 급여(PY), 수기 결산전표(GJ)
이상 전표
    7개 탐지 규칙에 대응하는 유형을 의도적으로 심고, 정답 라벨을 별도 DataFrame으로 반환한다.

전표번호 형식: {입력원천}-{YYYYMMDD}-{일련번호}   예) GJ-20251231-0007
    GJ  = 수기(Manual) 전표, 그 외 = 모듈 자동연계 전표
"""
from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from je_anomaly.calendar_kr import (
    FISCAL_YEAR,
    business_days,
    is_business_day,
    next_business_day,
    non_business_days,
    previous_business_day,
    shift_business_days,
)
from je_anomaly.ledger import GL_COLUMNS

LABEL_COLUMNS = ["전표번호", "is_anomaly", "anomaly_type"]

# ── 계정과목표 (더존 계정코드 체계 준용) ─────────────────────────────
ACCOUNTS: dict[str, str] = {
    "103": "보통예금",
    "108": "외상매출금",
    "131": "선급금",
    "134": "가지급금",
    "135": "부가세대급금",
    "153": "원재료",
    "207": "감가상각누계액",
    "239": "개발비",
    "251": "외상매입금",
    "253": "미지급금",
    "254": "예수금",
    "255": "부가세예수금",
    "404": "제품매출",
    "504": "임금",
    "516": "전력비",
    "518": "감가상각비",
    "801": "급여",
    "811": "복리후생비",
    "812": "여비교통비",
    "813": "접대비",
    "814": "통신비",
    "819": "지급임차료",
    "824": "운반비",
    "830": "소모품비",
    "831": "지급수수료",
    "930": "잡이익",
}

# ── 조직 / 권한 ─────────────────────────────────────────────────
SALES_STAFF = ("김민준", "이서연")
PURCHASE_STAFF = ("박지훈",)
TREASURY_STAFF = ("최유진",)
EXPENSE_STAFF = ("최유진", "박지훈")
PAYROLL_STAFF = ("이서연",)
GL_STAFF = ("김민준",)
ALL_STAFF = ("김민준", "이서연", "박지훈", "최유진")
TEAM_LEAD = "정태호"  # 회계팀장: 500만원 미만 승인
DIRECTOR = "한상우"  # 재무부장: 500만원 이상 승인
APPROVAL_LIMIT = 5_000_000

# ── 거래처 ──────────────────────────────────────────────────────
CUSTOMERS = ("(주)한빛정밀", "대성기계(주)", "(주)세진오토텍", "동양산업(주)", "(주)미래모빌리티")
PRODUCTS = ("감속기 하우징", "정밀 샤프트", "브레이크 브래킷", "유압 밸브 블록", "기어 어셈블리")
SUPPLIERS = ("(주)태광철강", "삼우소재(주)", "(주)대원케미칼", "신흥금속(주)")
MATERIALS = ("열연강판", "알루미늄 봉재", "절삭유", "베어링", "볼트·너트")
LANDLORD = "(주)대한빌딩관리"
POWER_COMPANY = "한국전력공사"
CONSULTANT = "(주)브릿지파트너스"
LOGISTICS = "(주)스마트물류"


@dataclass(frozen=True)
class ExpenseKind:
    code: str
    memos: tuple[str, ...]
    vendors: tuple[str, ...]
    median: float
    vat: bool
    credit: str  # 103 보통예금(이체) / 253 미지급금(법인카드)


EXPENSE_KINDS = (
    ExpenseKind("811", ("직원 중식대", "야근 식대", "사내 행사 다과"), ("공단식당", "(주)푸른푸드"), 350_000, True, "253"),
    ExpenseKind("812", ("거래처 방문 출장비", "KTX 출장 교통비", "업무용 차량 주유비"), ("임직원 정산",), 150_000, False, "103"),
    ExpenseKind("813", ("거래처 식사 접대", "명절 선물 구입"), ("한우명가", "(주)정성선물"), 450_000, False, "253"),
    ExpenseKind("814", ("인터넷·전화 요금", "휴대폰 요금"), ("한빛텔레콤(주)",), 280_000, True, "103"),
    ExpenseKind("824", ("제품 운송비", "샘플 택배비"), (LOGISTICS,), 700_000, True, "103"),
    ExpenseKind("830", ("사무용품 구입", "공구 소모품 구입"), ("(주)오피스플러스", "대한공구상사"), 250_000, True, "253"),
    ExpenseKind("831", ("세무 기장 수수료", "법률 자문료", "시험 성적서 발급"), ("세무법인 한결", "법무법인 정도", "(주)정밀시험연구소"), 1_200_000, True, "103"),
)

# ── 이상 유형 (2단계 탐지 규칙 1~7과 1:1 대응) ─────────────────────
DUPLICATE = "DUPLICATE"  # 1. 중복 전표
WEEKEND_HOLIDAY = "WEEKEND_HOLIDAY"  # 2. 주말·공휴일 입력
ROUND_AMOUNT = "ROUND_AMOUNT"  # 3. 라운드 금액 반복
BENFORD = "BENFORD"  # 4. 벤포드 이탈 (승인한도 직하 금액 쪼개기)
PERIOD_END_MANUAL = "PERIOD_END_MANUAL"  # 5. 결산일 직전 대형 수기 전표
SELF_APPROVAL = "SELF_APPROVAL"  # 6. 작성자 = 승인자
UNUSUAL_PAIR = "UNUSUAL_PAIR"  # 7. 비정상 계정 조합

ANOMALY_COUNTS: dict[str, int] = {
    DUPLICATE: 15,
    WEEKEND_HOLIDAY: 15,
    ROUND_AMOUNT: 15,
    BENFORD: 20,
    PERIOD_END_MANUAL: 10,
    SELF_APPROVAL: 12,
    UNUSUAL_PAIR: 12,
}

# (차변, 대변, 적요) — 정상 업무에서는 나올 수 없는 계정 조합
UNUSUAL_PAIRS = (
    ("134", "404", "가지급금 매출 대체"),
    ("813", "153", "원재료 접대 사용분 대체"),
    ("251", "404", "매입채무·매출 상계"),
    ("239", "801", "인건비 개발비 대체"),
    ("131", "930", "선급금 잡이익 정리"),
)

PERIOD_END_START = date(FISCAL_YEAR, 12, 24)  # 결산 직전 구간 시작일
PLANT_CUTOFF = date(FISCAL_YEAR, 12, 19)  # 결산 유형 외 이상전표는 이 날짜 이전에만 심는다


@dataclass
class Line:
    code: str
    debit: int
    credit: int
    memo: str
    vendor: str = ""


@dataclass
class Entry:
    source: str
    posted: date
    writer: str
    approver: str
    lines: list[Line]
    anomaly_type: str = ""

    @property
    def total(self) -> int:
        return sum(ln.debit for ln in self.lines)


def _dr(code: str, amount: int, memo: str, vendor: str = "") -> Line:
    return Line(code, int(amount), 0, memo, vendor)


def _cr(code: str, amount: int, memo: str, vendor: str = "") -> Line:
    return Line(code, 0, int(amount), memo, vendor)


class LedgerGenerator:
    def __init__(
        self,
        seed: int = 42,
        sales_per_day: float = 1.6,
        purchases_per_day: float = 1.4,
        expenses_per_day: float = 2.4,
        anomaly_counts: dict[str, int] | None = None,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.sales_per_day = sales_per_day
        self.purchases_per_day = purchases_per_day
        self.expenses_per_day = expenses_per_day
        self.anomaly_counts = anomaly_counts or ANOMALY_COUNTS
        self.days = business_days()
        self.plant_days = [d for d in self.days if d <= PLANT_CUTOFF]
        self.entries: list[Entry] = []

    # ── 공통 유틸 ───────────────────────────────────────────────
    def _pick(self, seq):
        return seq[int(self.rng.integers(len(seq)))]

    def _amount(self, median: float, sigma: float = 0.9, lo: float | None = None,
                hi: float | None = None, unit: int = 10) -> int:
        """로그정규 분포 금액. 여러 자릿수에 걸쳐 퍼지므로 자연스럽게 벤포드 분포에 가까워진다."""
        lo = median / 20 if lo is None else lo
        hi = median * 20 if hi is None else hi
        while True:
            v = self.rng.lognormal(np.log(median), sigma)
            if lo <= v <= hi:
                return int(v // unit * unit)

    def _approver_for(self, amount: int) -> str:
        return DIRECTOR if amount >= APPROVAL_LIMIT else TEAM_LEAD

    def _add(self, source: str, posted: date, writer: str, lines: list[Line],
             anomaly_type: str = "", approver: str | None = None) -> Entry:
        assert sum(ln.debit for ln in lines) == sum(ln.credit for ln in lines), "대차 불일치"
        entry = Entry(source, posted, writer, "", lines, anomaly_type)
        entry.approver = approver or self._approver_for(entry.total)
        self.entries.append(entry)
        return entry

    def _due_date(self, d: date) -> date | None:
        due = next_business_day(d + timedelta(days=int(self._pick((30, 45, 60)))))
        return due if due.year == FISCAL_YEAR else None

    # ── 정상 거래 ───────────────────────────────────────────────
    def _sale(self, d: date) -> None:
        customer, product = self._pick(CUSTOMERS), self._pick(PRODUCTS)
        supply = self._amount(6_000_000, 1.0, lo=300_000, hi=150_000_000)
        vat = supply // 10
        memo = f"{product} 납품"
        self._add("SA", d, self._pick(SALES_STAFF), [
            _dr("108", supply + vat, memo, customer),
            _cr("404", supply, memo, customer),
            _cr("255", vat, memo, customer),
        ])
        if due := self._due_date(d):
            memo = f"외상매출금 회수 - {product}"
            self._add("RC", due, self._pick(TREASURY_STAFF), [
                _dr("103", supply + vat, memo, customer),
                _cr("108", supply + vat, memo, customer),
            ])

    def _purchase(self, d: date) -> None:
        supplier, material = self._pick(SUPPLIERS), self._pick(MATERIALS)
        supply = self._amount(4_000_000, 1.0, lo=200_000, hi=100_000_000)
        vat = supply // 10
        memo = f"{material} 매입"
        self._add("PU", d, self._pick(PURCHASE_STAFF), [
            _dr("153", supply, memo, supplier),
            _dr("135", vat, memo, supplier),
            _cr("251", supply + vat, memo, supplier),
        ])
        if due := self._due_date(d):
            memo = f"외상매입금 지급 - {material}"
            self._add("PM", due, self._pick(TREASURY_STAFF), [
                _dr("251", supply + vat, memo, supplier),
                _cr("103", supply + vat, memo, supplier),
            ])

    def _expense_lines(self, kind: ExpenseKind, median_scale: float = 1.0) -> list[Line]:
        memo, vendor = self._pick(kind.memos), self._pick(kind.vendors)
        supply = self._amount(kind.median * median_scale, 0.8)
        lines = [_dr(kind.code, supply, memo, vendor)]
        total = supply
        if kind.vat:
            vat = supply // 10
            lines.append(_dr("135", vat, memo, vendor))
            total += vat
        lines.append(_cr(kind.credit, total, memo, vendor))
        return lines

    def _expense(self, d: date) -> None:
        self._add("EX", d, self._pick(EXPENSE_STAFF), self._expense_lines(self._pick(EXPENSE_KINDS)))

    def _monthly(self, month: int) -> None:
        # 공장 임차료: 매월 5일, 고정 300만원 (정상적인 라운드 금액 반복 → 규칙 3의 오탐 후보)
        d = next_business_day(date(FISCAL_YEAR, month, 5))
        memo = f"{month}월 공장 임차료"
        self._add("EX", d, "최유진", [
            _dr("819", 3_000_000, memo, LANDLORD),
            _dr("135", 300_000, memo, LANDLORD),
            _cr("103", 3_300_000, memo, LANDLORD),
        ])
        # 공장 전력비: 매월 20일
        d = next_business_day(date(FISCAL_YEAR, month, 20))
        supply = self._amount(9_000_000, 0.25)
        memo = f"{month}월 공장 전력비"
        self._add("EX", d, "최유진", [
            _dr("516", supply, memo, POWER_COMPANY),
            _dr("135", supply // 10, memo, POWER_COMPANY),
            _cr("103", supply + supply // 10, memo, POWER_COMPANY),
        ])
        # 급여: 매월 25일 (휴일이면 직전 영업일)
        d = previous_business_day(date(FISCAL_YEAR, month, 25))
        wage, salary = self._amount(42_000_000, 0.08), self._amount(24_000_000, 0.08)
        withholding = int((wage + salary) * 0.095) // 10 * 10
        memo = f"{month}월 급여"
        self._add("PY", d, self._pick(PAYROLL_STAFF), [
            _dr("504", wage, f"{memo}(생산직)"),
            _dr("801", salary, f"{memo}(사무직)"),
            _cr("254", withholding, f"{memo} 원천징수"),
            _cr("103", wage + salary - withholding, f"{memo} 지급"),
        ], approver=DIRECTOR)
        # 감가상각: 월말 수기 결산전표 (정상적인 결산일 수기 전표 → 규칙 5의 오탐 후보)
        next_month = date(FISCAL_YEAR + (month == 12), month % 12 + 1, 1)
        d = previous_business_day(next_month - timedelta(days=1))
        memo = f"{month}월 기계장치 감가상각"
        self._add("GJ", d, self._pick(GL_STAFF), [
            _dr("518", 12_450_000, memo),
            _cr("207", 12_450_000, memo),
        ], approver=DIRECTOR)

    def _generate_normal(self) -> None:
        for d in self.days:
            for _ in range(self.rng.poisson(self.sales_per_day)):
                self._sale(d)
            for _ in range(self.rng.poisson(self.purchases_per_day)):
                self._purchase(d)
            for _ in range(self.rng.poisson(self.expenses_per_day)):
                self._expense(d)
        for month in range(1, 13):
            self._monthly(month)

    # ── 이상 전표 심기 ──────────────────────────────────────────
    def _sample_days(self, pool: list[date], n: int, replace: bool = True) -> list[date]:
        idx = self.rng.choice(len(pool), size=n, replace=replace)
        return [pool[int(i)] for i in idx]

    def _plant_duplicate(self, n: int) -> None:
        """같은 송장을 0~3 영업일 뒤 다시 입력 (이중 지급 위험)."""
        candidates = [e for e in self.entries
                      if e.source in ("PU", "EX") and not e.anomaly_type and e.posted <= PLANT_CUTOFF
                      and e.lines[0].code not in ("819", "516")]  # 월 고정비 제외
        for i in self.rng.choice(len(candidates), size=n, replace=False):
            orig = candidates[int(i)]
            new_day = shift_business_days(orig.posted, int(self.rng.integers(0, 4)))
            self._add(orig.source, new_day, orig.writer, copy.deepcopy(orig.lines), DUPLICATE, orig.approver)

    def _plant_weekend_holiday(self, n: int) -> None:
        pool = [d for d in non_business_days() if d <= PLANT_CUTOFF]
        kinds = [k for k in EXPENSE_KINDS if k.code in ("813", "830", "831")]
        for d in self._sample_days(pool, n, replace=False):
            kind = self._pick(kinds)
            memo, vendor = self._pick(kind.memos), self._pick(kind.vendors)
            amount = self._amount(kind.median * 2, 0.6)
            self._add("GJ", d, self._pick(ALL_STAFF), [
                _dr(kind.code, amount, memo, vendor),
                _cr("103", amount, memo, vendor),
            ], WEEKEND_HOLIDAY)

    def _plant_round_amount(self, n: int) -> None:
        """실체가 불분명한 자문 용역에 딱 떨어지는 금액을 반복 지급."""
        amounts = self.rng.choice([3_000_000, 5_000_000, 10_000_000], size=n, p=[0.3, 0.4, 0.3])
        for d, amount in zip(self._sample_days(self.plant_days, n), amounts):
            memo = "경영자문 수수료"
            self._add("EX", d, "김민준", [
                _dr("831", int(amount), memo, CONSULTANT),
                _cr("253", int(amount), memo, CONSULTANT),
            ], ROUND_AMOUNT)

    def _plant_benford(self, n: int) -> None:
        """부장 승인(500만원)을 피하려고 한도 바로 아래(490~499만원)로 쪼갠 지급 → 첫 두 자리 '49' 과다."""
        for d in self._sample_days(self.plant_days, n):
            amount = int(self.rng.integers(490_000, 499_900)) * 10
            memo = "긴급 운송비"
            self._add("EX", d, "박지훈", [
                _dr("824", amount, memo, LOGISTICS),
                _cr("103", amount, memo, LOGISTICS),
            ], BENFORD, approver=TEAM_LEAD)

    def _plant_period_end_manual(self, n: int) -> None:
        pool = [d for d in self.days if d >= PERIOD_END_START]
        for d in self._sample_days(pool, n):
            amount = self._amount(120_000_000, 0.5, lo=50_000_000, hi=400_000_000, unit=1000)
            if self.rng.random() < 0.6:
                memo, vendor = "연말 매출 인식 조정", self._pick(CUSTOMERS)
                lines = [_dr("108", amount, memo, vendor), _cr("404", amount, memo, vendor)]
            else:
                memo, vendor = "지급수수료 선급비용 대체", CONSULTANT
                lines = [_dr("131", amount, memo, vendor), _cr("831", amount, memo, vendor)]
            self._add("GJ", d, "김민준", lines, PERIOD_END_MANUAL, approver=DIRECTOR)

    def _plant_self_approval(self, n: int) -> None:
        for d in self._sample_days(self.plant_days, n):
            writer = self._pick((TEAM_LEAD, "최유진", "박지훈"))
            self._add("EX", d, writer, self._expense_lines(self._pick(EXPENSE_KINDS)),
                      SELF_APPROVAL, approver=writer)

    def _plant_unusual_pair(self, n: int) -> None:
        for d in self._sample_days(self.plant_days, n):
            dr_code, cr_code, memo = self._pick(UNUSUAL_PAIRS)
            amount = self._amount(6_000_000, 0.8)
            self._add("GJ", d, self._pick(ALL_STAFF), [
                _dr(dr_code, amount, memo),
                _cr(cr_code, amount, memo),
            ], UNUSUAL_PAIR)

    def _plant_anomalies(self) -> None:
        planters = {
            DUPLICATE: self._plant_duplicate,
            WEEKEND_HOLIDAY: self._plant_weekend_holiday,
            ROUND_AMOUNT: self._plant_round_amount,
            BENFORD: self._plant_benford,
            PERIOD_END_MANUAL: self._plant_period_end_manual,
            SELF_APPROVAL: self._plant_self_approval,
            UNUSUAL_PAIR: self._plant_unusual_pair,
        }
        for anomaly_type, n in self.anomaly_counts.items():
            if n:
                planters[anomaly_type](n)

    # ── 출력 ───────────────────────────────────────────────────
    def _to_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        # 같은 날 전표 순서를 섞어 유형별로 몰리지 않게 한다.
        # 단, 중복 전표는 원본보다 뒤에 입력된 것이어야 하므로 그날의 맨 뒤로 보낸다.
        tiebreak = self.rng.random(len(self.entries)) + [e.anomaly_type == DUPLICATE for e in self.entries]
        order = sorted(range(len(self.entries)), key=lambda i: (self.entries[i].posted, tiebreak[i]))
        seq_by_day: dict[date, int] = defaultdict(int)
        rows, labels = [], []
        for i in order:
            e = self.entries[i]
            seq_by_day[e.posted] += 1
            je_no = f"{e.source}-{e.posted:%Y%m%d}-{seq_by_day[e.posted]:04d}"
            for ln in e.lines:
                rows.append({
                    "전표번호": je_no,
                    "전표일자": e.posted.isoformat(),
                    "작성자": e.writer,
                    "승인자": e.approver,
                    "계정코드": ln.code,
                    "계정과목": ACCOUNTS[ln.code],
                    "차변": ln.debit,
                    "대변": ln.credit,
                    "적요": ln.memo,
                    "거래처": ln.vendor,
                })
            labels.append({"전표번호": je_no, "is_anomaly": int(bool(e.anomaly_type)),
                           "anomaly_type": e.anomaly_type})
        return pd.DataFrame(rows, columns=GL_COLUMNS), pd.DataFrame(labels, columns=LABEL_COLUMNS)

    def generate(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.entries = []
        self._generate_normal()
        self._plant_anomalies()
        return self._to_frames()


def generate_ledger(seed: int = 42, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(총계정원장, 정답라벨) 반환."""
    return LedgerGenerator(seed=seed, **kwargs).generate()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="가상 총계정원장 데이터 생성")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    args = parser.parse_args(argv)

    gl, labels = generate_ledger(seed=args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gl_path = args.out_dir / f"general_ledger_{FISCAL_YEAR}.csv"
    label_path = args.out_dir / f"labels_{FISCAL_YEAR}.csv"
    # 엑셀에서 한글이 깨지지 않도록 BOM 포함 UTF-8
    gl.to_csv(gl_path, index=False, encoding="utf-8-sig")
    labels.to_csv(label_path, index=False, encoding="utf-8-sig")

    print(f"총계정원장: {gl_path}  ({len(gl):,}행 / 전표 {gl['전표번호'].nunique():,}건)")
    print(f"정답 라벨:  {label_path}  (이상 전표 {labels['is_anomaly'].sum()}건)")
    print(labels.loc[labels["is_anomaly"] == 1, "anomaly_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
