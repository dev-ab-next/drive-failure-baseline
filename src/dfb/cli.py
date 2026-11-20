"""Command line: `dfb prepare` and `dfb profile`."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="dfb")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="download and convert the Backblaze quarters")
    p.add_argument("--keep-zips", action="store_true")
    sub.add_parser("profile", help="failure profile per drive model and brand")
    sub.add_parser("validate", help="check every Parquet file against the declared schema")
    sub.add_parser("features", help="build features, labels and split columns")
    sub.add_parser("splits", help="check leakage rules on the real splits and count them")
    sub.add_parser("bench", help="run every benchmark v1 run (restart-safe)")
    im = sub.add_parser("improve",
                        help="improvement study on validation; --test runs the chosen ones")
    im.add_argument("--test", action="store_true",
                    help="run the configurations chosen on validation on test, 3 seeds")
    ss = sub.add_parser("smote-study",
                        help="SMOTE settings study on validation; --test runs the chosen one")
    ss.add_argument("--test", action="store_true",
                    help="run the variant chosen on validation on test, 3 seeds")
    sub.add_parser("freeze", help="write benchmark/v1/benchmark.json from the prepared data")
    sub.add_parser("report", help="write results/benchmark_v1/RESULTS.md from finished runs")
    sub.add_parser("quickcheck", help="check the pipeline on one real quarter (~10 minutes)")
    sub.add_parser("verify", help="check local data against the frozen benchmark")
    tr = sub.add_parser("test-rows", help="list the (serial_number, date) rows to score")
    tr.add_argument("out")
    tr.add_argument("--period", default="test", choices=["valid", "test"])
    ev = sub.add_parser("evaluate", help="score your own predictions with the benchmark metrics")
    ev.add_argument("--predictions", required=True,
                    help="Parquet with serial_number, date, score for every test row")
    ev.add_argument("--valid-predictions",
                    help="same for validation rows; enables the headline validation threshold")
    ev.add_argument("--n", type=int, default=7, choices=[7, 14, 30])
    args = parser.parse_args(argv)

    if args.command == "prepare":
        from .prepare import prepare
        prepare(keep_zips=args.keep_zips)
    elif args.command == "validate":
        from .prepare import validate_all
        raise SystemExit(0 if validate_all() else 1)
    elif args.command == "features":
        from .features import build_all
        build_all()
    elif args.command == "splits":
        from .splits import summarize
        print(summarize())
    elif args.command == "improve":
        from . import improve
        rows = improve.test_stage() if args.test else improve.validation_stages()
        print(rows.drop(columns=["settings"]).to_string(index=False))
    elif args.command == "smote-study":
        from . import smote_study
        if args.test:
            from .run import run
            v = smote_study.best_variant()
            for seed in (0, 1, 2):
                done = smote_study.OUT / "test" / "N7" / f"L4-M2-core-seed{seed}" / "metrics.json"
                if not done.exists():
                    run(7, "L4", "M2", "core", seed, smote=v, out_root=smote_study.OUT / "test")
                print(f"test seed {seed} done ({v.name})", flush=True)
            for seed in (0, 1, 2):  # M-5: CTGAN with the same weighting fix
                done = smote_study.OUT / "test" / "N7" / f"L4-M3-core-seed{seed}" / "metrics.json"
                if not done.exists():
                    run(7, "L4", "M3", "core", seed, keep_weights=True,
                        out_root=smote_study.OUT / "test")
                print(f"test seed {seed} done (CTGAN, sampling weights kept)", flush=True)
        else:
            print(smote_study.study().to_string(index=False))
    elif args.command == "bench":
        from .bench import main as bench_main
        bench_main()
    elif args.command == "quickcheck":
        from .quickcheck import main as qc
        raise SystemExit(0 if qc() else 1)
    elif args.command == "report":
        from .report import write
        print(write())
    elif args.command == "freeze":
        from .benchmark import freeze
        print(f"wrote {freeze()}")
    elif args.command == "verify":
        from .benchmark import verify
        problems = verify()
        print("\n".join(problems) if problems else "local data matches benchmark v1")
        raise SystemExit(1 if problems else 0)
    elif args.command == "test-rows":
        from pathlib import Path
        from .benchmark import test_rows
        print(f"{test_rows(Path(args.out), args.period):,} rows written to {args.out}")
    elif args.command == "evaluate":
        import json
        from pathlib import Path
        from .benchmark import evaluate_predictions
        vp = Path(args.valid_predictions) if args.valid_predictions else None
        print(json.dumps(evaluate_predictions(Path(args.predictions), vp, args.n),
                         indent=2, default=float))
    elif args.command == "profile":
        from .profile import run
        run()


if __name__ == "__main__":
    main()
