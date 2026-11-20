"""SMOTE settings study (decision M-4): why does SMOTE lower detection for
LightGBM, and does any setting close the gap?

Every variant is trained once (LightGBM, 7-day window, core features,
seed 0) and scored on the validation period only. Settings are compared
and chosen on validation; the test period is not used here. The chosen
variant is then run on test with 3 seeds by `dfb smote-study --test`,
and reported next to the pre-declared M2.

Hypotheses and the variant that tests each:
- H1 neighbours are mostly other days of the same drive, so new rows are
  near-copies: `cross_drive` takes neighbours from other drives only, and
  `k` varies the number of neighbours.
- H2 blending one-hot brand columns and "not reported" flags makes rows
  that cannot exist: `categorical` copies those columns from the base row.
- H3 too many synthetic rows (about 6.5 per real failure row at 10:1):
  `ratio` varies how far the top-up goes.
- H4 M1-M3 train without the F-12 sampling weights, so healthy days near a
  failure count 50 times more than other healthy days: `weighted` keeps
  them and balances classes as M0 does.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from . import config, evaluate, models
from .labels import label
from .leakage import check
from .run import iter_scoring, load_train, load_valid_sample

OUT = config.ROOT / "results" / "smote_study"
N = 7
LEVEL = "L4"
FEATURES = "core"


@dataclass(frozen=True)
class Variant:
    name: str
    k: int = 5
    ratio: int = 10            # healthy rows per failure row after the top-up
    cross_drive: bool = False  # H1
    categorical: bool = False  # H2
    weighted: bool = False     # H4
    method: str = "smote"      # "smote" (this module), "imblearn" (original M2), "none" (M0)


VARIANTS = [
    Variant("M0 class weighting (reference)", method="none"),
    Variant("M2 as run (imblearn, k=5, 10:1)", method="imblearn"),
    Variant("SMOTE k=5, 10:1 (same as M2)"),
    Variant("k=3", k=3),
    Variant("k=10", k=10),
    Variant("k=25", k=25),
    Variant("ratio 25:1", ratio=25),
    Variant("ratio 50:1", ratio=50),
    Variant("ratio 100:1", ratio=100),
    Variant("cross-drive neighbours", cross_drive=True),
    Variant("categorical columns kept", categorical=True),
    Variant("sampling weights kept", weighted=True),
    Variant("cross-drive + categorical + weights", cross_drive=True, categorical=True,
            weighted=True),
]


def smote_rows(X: np.ndarray, y: np.ndarray, serials: np.ndarray, n_add: int, v: Variant,
               cat_idx: np.ndarray, seed: int) -> tuple[np.ndarray, dict]:
    """Standard SMOTE (random base failure row, random one of its k nearest
    failure rows, random point on the line between), with the H1/H2 options."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    Xp, sp = X[pos], serials[pos]
    extra = N if v.cross_drive else 0  # a drive has at most N failure-window rows
    nn = NearestNeighbors(n_neighbors=v.k + 1 + extra).fit(Xp)
    _, idx = nn.kneighbors(Xp)
    idx = idx[:, 1:]  # drop the row itself
    if v.cross_drive:
        other = sp[idx] != sp[:, None]
        # keep the first k neighbours from other drives (rows sorted by distance)
        order = np.argsort(~other, axis=1, kind="stable")[:, :v.k]
        idx = np.take_along_axis(idx, order, axis=1)
    else:
        idx = idx[:, :v.k]
    base = rng.integers(0, len(pos), n_add)
    nbr = idx[base, rng.integers(0, v.k, n_add)]
    gap = rng.random((n_add, 1), dtype=np.float32)
    new = Xp[base] + gap * (Xp[nbr] - Xp[base])
    if v.categorical and len(cat_idx):
        new[:, cat_idx] = Xp[base][:, cat_idx]
    diag = {"same_drive_parent_share": float((sp[base] == sp[nbr]).mean())}
    if len(cat_idx):
        # a blended row is one whose flag or brand value occurs in no real failure row
        real = [np.unique(Xp[:, j]) for j in cat_idx]
        off = np.zeros(n_add, dtype=bool)
        for col, j in enumerate(cat_idx):
            off |= ~np.isin(new[:, j], real[col])
        diag["rows_with_blended_flag_or_brand"] = float(off.mean())
    return new.astype(np.float32), diag


def train_set(v: Variant, X, y, base_w, serials, cat_idx, seed):
    info: dict = {}
    if v.method == "none":
        return X, y, models.balanced_weights(y, base_w), info
    if v.method == "imblearn":
        X2, y2 = models.smote(X, y, seed)
        info["rows_added"] = int(len(y2) - len(y))
        return X2, y2, None, info
    target_pos = int(np.ceil((y == 0).sum() / v.ratio))
    n_add = max(0, target_pos - int((y == 1).sum()))
    new, diag = smote_rows(X, y, serials, n_add, v, cat_idx, seed)
    info.update(diag, rows_added=n_add)
    X2 = np.vstack([X, new])
    y2 = np.concatenate([y, np.ones(n_add, dtype=y.dtype)])
    if v.weighted:
        w = np.concatenate([base_w, np.ones(n_add)])
        return X2, y2, models.balanced_weights(y2, w), info
    return X2, y2, None, info


def score_valid(model, rep, need) -> dict:
    parts, windows = [], []
    for chunk in iter_scoring("valid", N, need):
        s = np.empty(len(chunk))
        for i in range(0, len(chunk), 1_000_000):
            s[i:i + 1_000_000] = models.predict(LEVEL, model,
                                                rep.transform(chunk.iloc[i:i + 1_000_000]))
        chunk["score"] = s
        ds = evaluate.drive_scores(chunk, N)
        parts.append(ds.drives)
        windows.append(ds.window)
    ds = evaluate.DriveScores(pd.concat(parts, ignore_index=True),
                              pd.concat(windows, ignore_index=True))
    out = evaluate.summary(ds)  # thresholds set on validation itself
    return {"valid_caught_1pct": out["test_curve_1pct"]["detection_rate"],
            "valid_caught_0.1pct": out["test_curve_0.1pct"]["detection_rate"],
            "valid_warning_days": out["test_curve_1pct"]["warning_days_median"],
            "valid_pr_auc": out["pr_auc"], "valid_failures": out["failures"]}


def study(seed: int = 0) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    cols = models.feature_columns(FEATURES)
    need = sorted(set(cols) | set(models.SMART_SIGNALS))
    tr = load_train(N, need)
    va = load_valid_sample(N, need)
    check({"train": tr, "valid": va}, N)
    y = label(tr, N).to_numpy()
    rep = models.Representation(cols).fit(tr)
    X = rep.transform(tr)
    base_w = tr["sample_weight"].to_numpy(dtype="float64")
    serials = tr["serial_number"].to_numpy()
    names = rep.names
    cat_idx = np.array([i for i, c in enumerate(names)
                        if c.startswith(("missing_", "brand_"))], dtype=int)
    Xv, yv = rep.transform(va), label(va, N).to_numpy()
    del tr

    path = OUT / "validation.csv"
    done = pd.read_csv(path) if path.exists() else pd.DataFrame()
    rows = done.to_dict("records")
    for v in VARIANTS:
        if len(done) and v.name in set(done["variant"]):
            continue
        t = time.time()
        X2, y2, w, info = train_set(v, X, y, base_w, serials, cat_idx, seed)
        model = models.fit(LEVEL, X2, y2, w, seed, Xv, yv)
        del X2, y2, w
        res = score_valid(model, rep, need)
        row = {"variant": v.name, **{k: val for k, val in asdict(v).items() if k != "name"},
               **info, **res, "trees": int(model.best_iteration or model.num_trees()),
               "seconds": round(time.time() - t, 1)}
        rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"{v.name}: caught {res['valid_caught_1pct']:.3f} at 1% (validation)", flush=True)
    return pd.DataFrame(rows)


def best_variant() -> Variant:
    """The SMOTE variant with the highest validation detection at 1% false alarms."""
    df = pd.read_csv(OUT / "validation.csv")
    smote_rows_ = df[df["method"] == "smote"]
    name = smote_rows_.sort_values("valid_caught_1pct", ascending=False)["variant"].iloc[0]
    v = next(v for v in VARIANTS if v.name == name)
    (OUT / "chosen.json").write_text(json.dumps(asdict(v), indent=2))
    return v


def refresh_diagnostics(seed: int = 0) -> pd.DataFrame:
    """Recompute the synthetic-row diagnostics for every SMOTE variant (no training)."""
    cols = models.feature_columns(FEATURES)
    need = sorted(set(cols) | set(models.SMART_SIGNALS))
    tr = load_train(N, need)
    y = label(tr, N).to_numpy()
    rep = models.Representation(cols).fit(tr)
    X = rep.transform(tr)
    serials = tr["serial_number"].to_numpy()
    cat_idx = np.array([i for i, c in enumerate(rep.names)
                        if c.startswith(("missing_", "brand_"))], dtype=int)
    path = OUT / "validation.csv"
    df = pd.read_csv(path).drop(columns=["rows_with_fractional_flag_or_brand"], errors="ignore")
    for v in VARIANTS:
        if v.method != "smote":
            continue
        n_add = max(0, int(np.ceil((y == 0).sum() / v.ratio)) - int((y == 1).sum()))
        if n_add == 0:
            continue
        _, diag = smote_rows(X, y, serials, n_add, v, cat_idx, seed)
        for k, val in diag.items():
            df.loc[df["variant"] == v.name, k] = val
    df.to_csv(path, index=False)
    return df
