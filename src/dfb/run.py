"""Train, score and evaluate one run.

A run is (N, level, method, feature set, seed). Results go to
results/benchmark_v1/N<n>/<run_id>/: metrics.json, per-drive test scores,
slice tables and the model file.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from . import config, evaluate, models
from .features import FEATURES_DIR, SAMPLE_SALT
from .labels import label, usable_sql
from .leakage import check

RESULTS = config.ROOT / "results" / "benchmark_v1"
BENCHMARK = "benchmark_v1"
VALID_DRIVE_SHARE = 0.25  # E-5
KEYS = ["serial_number", "date", "days_to_failure", "days_to_period_end"]
ATTRS = ["brand", "model", "capacity_tb", "age_years"]
BUCKETS = 8
SCORE_SLICE = 1_000_000


def con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    try:
        c.execute("SET enable_progress_bar=false")
    except duckdb.InvalidInputException:  # inside Jupyter without ipywidgets
        pass
    c.execute("SET threads=4")
    return c


def glob(period: str, bucket: int | None = None) -> str:
    part = "*" if bucket is None else str(bucket)
    return f"read_parquet('{FEATURES_DIR}/{period}/*/part-{part}.parquet')"


def load_train(n: int, cols: list[str]) -> pd.DataFrame:
    sel = ", ".join(dict.fromkeys(KEYS + ["period", "sample_weight"] + cols))
    return con().execute(f"SELECT {sel} FROM {glob('train')} WHERE {usable_sql(n)}").df()


def load_valid_sample(n: int, cols: list[str]) -> pd.DataFrame:
    """Early-stopping and calibration rows: every failure-window day plus 2%
    of other days, from the 25% validation drive sample."""
    sel = ", ".join(dict.fromkeys(KEYS + ["period"] + cols))
    keep_drive = f"hash(serial_number || '|valid') % 100 < {int(VALID_DRIVE_SHARE * 100)}"
    keep_row = (f"(days_to_failure BETWEEN 0 AND {n - 1}) OR "
                f"hash(serial_number || '|' || CAST(date AS VARCHAR) || '|{SAMPLE_SALT}') % 100 < 2")
    q = (f"SELECT {sel}, CASE WHEN days_to_failure BETWEEN 0 AND {n - 1} THEN 1.0 ELSE 50.0 END "
         f"AS sample_weight FROM {glob('valid')} WHERE {usable_sql(n)} AND {keep_drive} "
         f"AND ({keep_row})")
    return con().execute(q).df()


def iter_scoring(period: str, n: int, cols: list[str]):
    """Yield every usable row of a period, one drive group at a time (a drive's
    rows are always in one group). Validation keeps the 25% drive sample."""
    sel = ", ".join(dict.fromkeys(KEYS + ["period"] + ATTRS + cols))
    where = usable_sql(n)
    if period == "valid":
        where += f" AND hash(serial_number || '|valid') % 100 < {int(VALID_DRIVE_SHARE * 100)}"
    for b in range(BUCKETS):
        yield con().execute(f"SELECT {sel} FROM {glob(period, b)} WHERE {where}").df()


def git_version() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=config.ROOT, text=True).strip()
    except Exception:
        return "unknown"


def run(n: int, level: str, method: str = "M0", feature_set: str = "core",
        seed: int = 0, smote=None, out_root: Path | None = None,
        keep_weights: bool = False) -> dict:
    """`smote` is an optional `smote_study.Variant` for M2 (decision M-4);
    `keep_weights` trains M3 with the F-12 sampling weights and class balancing
    (decision M-5). Such runs are written under `out_root`, apart from the
    benchmark runs."""
    t0 = time.time()
    run_id = f"{level}-{method}-{feature_set}-seed{seed}"
    out = (out_root or RESULTS) / f"N{n}" / run_id
    out.mkdir(parents=True, exist_ok=True)
    cols = models.feature_columns(feature_set)
    need = sorted(set(cols) | set(models.SMART_SIGNALS))
    info: dict = {"benchmark": BENCHMARK, "n": n, "level": level, "method": method,
                  "feature_set": feature_set, "seed": seed, "code_version": git_version(),
                  "data_window": "2023 Q4 to 2026 Q3"}

    rep = None
    model = None
    saved = out / "model.joblib"
    if level in ("L2", "L3", "L4") and saved.exists():
        # Resume: reuse a model this run already trained and saved.
        bundle = joblib.load(saved)
        model, rep, calibrator = bundle["model"], bundle["representation"], bundle["calibrator"]
        info.update({k: v for k, v in bundle["info"].items() if k not in info})
        info["resumed_from_saved_model"] = True
    elif level in ("L2", "L3", "L4"):
        tr = load_train(n, need)
        y = label(tr, n).to_numpy()
        rep = models.Representation(cols).fit(tr)
        X = rep.transform(tr)
        va = load_valid_sample(n, need)
        check({"train": tr, "valid": va}, n)
        info["leakage_checks"] = "passed"
        if method == "M0":
            w = models.balanced_weights(y, tr["sample_weight"].to_numpy())
        elif method == "M1":
            X, y = models.undersample(X, y, seed)
            w = None
        elif method == "M2" and smote is not None:
            from .smote_study import train_set
            cat_idx = np.array([i for i, c in enumerate(rep.names)
                                if c.startswith(("missing_", "brand_"))], dtype=int)
            X, y, w, extra = train_set(smote, X, y, tr["sample_weight"].to_numpy("float64"),
                                       tr["serial_number"].to_numpy(), cat_idx, seed)
            info.update(extra)
            info["smote_variant"] = smote.name
        elif method == "M2":
            before = len(y)
            X, y = models.smote(X, y, seed)
            info["rows_added"] = int(len(y) - before)
            w = None
        elif method == "M3":
            t_gen = time.time()
            X, y, extra = models.ctgan_topup(X, y, rep.names, seed)
            info.update(extra)
            info["ctgan_seconds"] = round(time.time() - t_gen, 1)
            w = None
            if keep_weights:
                base = tr["sample_weight"].to_numpy("float64")
                w = models.balanced_weights(y, np.concatenate(
                    [base, np.ones(len(y) - len(base))]))
                info["sampling_weights_kept"] = True
        else:
            raise ValueError(method)
        info["train_rows"] = int(len(y))
        info["train_positive_rows"] = int(y.sum())
        Xv, yv = rep.transform(va), label(va, n).to_numpy()
        t1 = time.time()
        model = models.fit(level, X, y, w, seed, Xv, yv)
        info["fit_seconds"] = round(time.time() - t1, 1)
        del X, y, tr
        cal_scores = models.predict(level, model, Xv)
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(
            cal_scores, yv, sample_weight=va["sample_weight"].to_numpy())
        joblib.dump({"model": model, "representation": rep, "calibrator": calibrator,
                     "info": info}, out / "model.joblib", compress=3)

    def score(df: pd.DataFrame) -> np.ndarray:
        if level == "L0":
            return np.zeros(len(df))
        if level == "L1":
            return models.l1_score(df).astype("float64")
        # Score in 1-million-row slices to bound memory on 16 GB machines.
        out = np.empty(len(df))
        for i in range(0, len(df), SCORE_SLICE):
            out[i:i + SCORE_SLICE] = models.predict(level, model,
                                                    rep.transform(df.iloc[i:i + SCORE_SLICE]))
        return out

    train_models = set(con().execute(
        f"SELECT DISTINCT model FROM {glob('train')}").df()["model"])
    t2 = time.time()
    results = {}
    brier_num = brier_den = 0.0
    healthy_days = all_days = 0
    for period in ("valid", "test"):
        drive_parts, window_parts, month_rows = [], [], []
        for chunk in iter_scoring(period, n, need):
            chunk["score"] = score(chunk)
            ds = evaluate.drive_scores(chunk, n, attrs=ATTRS)
            drive_parts.append(ds.drives)
            window_parts.append(ds.window)
            if period == "test":
                yl0 = label(chunk, n).to_numpy()
                healthy_days += int((yl0 == 0).sum())
                all_days += len(yl0)
                month_rows.append(chunk.loc[chunk["days_to_failure"] == 0, ["serial_number", "date"]]
                                  .assign(days_to_failure=0))
                if level in ("L2", "L3", "L4"):
                    p = calibrator.predict(chunk["score"].to_numpy())
                    yl = label(chunk, n).to_numpy()
                    brier_num += float(((p - yl) ** 2).sum())
                    brier_den += len(yl)
        results[period] = (evaluate.DriveScores(pd.concat(drive_parts, ignore_index=True),
                                                pd.concat(window_parts, ignore_index=True)),
                           pd.concat(month_rows, ignore_index=True) if month_rows else None)
    info["score_seconds"] = round(time.time() - t2, 1)

    test, fail_days = results["test"]
    valid, _ = results["valid"]
    metrics = evaluate.summary(test, valid)
    if brier_den:
        metrics["brier_test_calibrated"] = brier_num / brier_den
    # B-10: accuracy shown once, to explain why it misleads (always-healthy model)
    metrics["accuracy_of_always_healthy_test_days"] = healthy_days / all_days
    thr = metrics["valid_threshold_1pct"]["threshold"]
    test.drives["age_band"] = pd.cut(test.drives["age_years"], [0, 1, 2, 3, 5, 7, 100],
                                     labels=["0-1", "1-2", "2-3", "3-5", "5-7", "7+"], right=False)
    test.drives["new_model"] = ~test.drives["model"].isin(train_models)  # R-4
    for col in ["brand", "capacity_tb", "age_band", "model", "new_model"]:
        evaluate.by_slice(test, thr, col).to_csv(out / f"slice_{col}.csv", index=False)
    evaluate.by_month(test, fail_days, thr).to_csv(out / "by_month.csv", index=False)
    test.drives.to_parquet(out / "test_drive_scores.parquet", index=False)
    info["seconds"] = round(time.time() - t0, 1)
    (out / "metrics.json").write_text(json.dumps({"run": info, "metrics": metrics}, indent=2,
                                                 default=float))
    return {"run": info, "metrics": metrics}
