"""정답 라벨 대비 탐지 성능 평가.

사용법:
    python scripts/evaluate.py                      # data/ 의 원장·라벨로 평가 + 20개 seed 반복, README 갱신
    python scripts/evaluate.py --seeds 0 --no-readme
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from je_anomaly.evaluation import evaluate, multi_seed, render_markdown, update_readme
from je_anomaly.ledger import load_csv, prepare

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="탐지 성능 평가")
    p.add_argument("--ledger", type=Path, default=ROOT / "data" / "general_ledger_2025.csv")
    p.add_argument("--labels", type=Path, default=ROOT / "data" / "labels_2025.csv")
    p.add_argument("--seed", type=int, default=42, help="기준 데이터의 seed (보고서 표기용)")
    p.add_argument("--seeds", type=int, default=20, help="반복 평가할 seed 개수 (0이면 생략)")
    p.add_argument("--out", type=Path, default=ROOT / "evaluation" / "report.md")
    p.add_argument("--no-readme", action="store_true", help="README.md를 갱신하지 않음")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    gl = prepare(load_csv(args.ledger))
    labels = pd.read_csv(args.labels, encoding="utf-8-sig", dtype={"전표번호": str})
    report = evaluate(gl, labels)

    ms = None
    if args.seeds > 0:
        seeds = [args.seed] + [s for s in range(1, args.seeds + 1) if s != args.seed][: args.seeds - 1]
        print(f"{len(seeds)}개 seed 반복 평가 중...")
        ms = multi_seed(seeds)

    body = render_markdown(report, args.seed, ms)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("# 탐지 성능 평가\n\n" + body + "\n", encoding="utf-8")
    if ms is not None:
        ms.to_csv(args.out.with_name("multi_seed.csv"), index=False, encoding="utf-8-sig")
    print(body)
    print(f"\n저장: {args.out}")

    if not args.no_readme:
        readme = ROOT / "README.md"
        readme.write_text(update_readme(readme.read_text(encoding="utf-8"), body), encoding="utf-8")
        print(f"갱신: {readme}")


if __name__ == "__main__":
    main()
