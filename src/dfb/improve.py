"""Improvement study (decision M-6): can tuning, blending or per-brand
thresholds raise detection, and by how much?

Every experiment appends one row to results/improvement_study/experiments.csv
(never overwritten), with its exact settings and scores, so every attempt
can be compared. Stages A–D are scored on the validation period only, with
the threshold set on validation. Stage T runs the configurations chosen by
the rule in `choose()` on test: thresholds from validation, applied
unchanged, 3 seeds. 7-day window, core features throughout.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from . import config, evaluate, models
from .labels import label
from .leakage import check
from .run import git_version, iter_scoring, load_train, load_valid_sample

OUT = config.ROOT / "results" / "improvement_study"
LOG = OUT / "experiments.csv"
MODELS = OUT / "models"  # git-ignored
N = 7
FEATURES = "core"


@dataclass(frozen=True)
class Config:
    id: str
    stage: str
    description: str
    model: str = "L4"                   # "L4" LightGBM, "L2" logistic regression, "blend"
    params: dict = field(default_factory=dict)  # LightGBM params or {"C": ...}
    trees: int = models.LGB_MAX_TREES
    smote: bool = False                 # add rows with the corrected SMOTE (M-4 choice)
    members: tuple = ()                 # blend: ids of member configs
    per_brand: bool = False             # thresholds per brand instead of one global


STAGE_A = [  # LightGBM, one setting changed at a time from the benchmark settings
    Config("A1", "A", "LightGBM, benchmark settings (reference)"),
    Config("A2", "A", "LightGBM, 31 leaves", params={"num_leaves": 31}),
    Config("A3", "A", "LightGBM, 127 leaves", params={"num_leaves": 127}),
    Config("A4", "A", "LightGBM, at least 500 rows per leaf", params={"min_data_in_leaf": 500}),
    Config("A5", "A", "LightGBM, at least 20 rows per leaf", params={"min_data_in_leaf": 20}),
    Config("A6", "A", "LightGBM, learning rate 0.05, up to 600 trees",
           params={"learning_rate": 0.05}, trees=600),
    Config("A7", "A", "LightGBM, L2 regularisation 10", params={"lambda_l2": 10.0}),
    Config("A8", "A", "LightGBM, 50% of features per tree", params={"feature_fraction": 0.5}),
]
STAGE_B = [  # logistic regression
    Config("B1", "B", "Logistic regression, C = 1 (reference)", model="L2", params={"C": 1.0}),
    Config("B2", "B", "Logistic regression, C = 0.1", model="L2", params={"C": 0.1}),
    Config("B3", "B", "Logistic regression, C = 10", model="L2", params={"C": 10.0}),
]


def _valid(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[rows["period"] == "valid"]


def _score(rows: pd.DataFrame, cfg_id: str) -> tuple:
    r = _valid(rows)[_valid(rows)["id"] == cfg_id].iloc[0]
    return (r["caught_1pct"], r["pr_auc"])


def best_of(rows: pd.DataFrame, ids) -> str:
    """Highest validation detection at 1% false alarms; PR-AUC breaks ties."""
    return max(ids, key=lambda i: _score(rows, i))


def lgb_combined(rows: pd.DataFrame) -> Config:
    """A9: for each setting, the value from the best stage-A run that beat A1."""
    ref = _score(rows, "A1")
    winners = sorted((c for c in STAGE_A[1:] if _score(rows, c.id) > ref),
                     key=lambda c: _score(rows, c.id))  # best last, so it wins conflicts
    params, trees = {}, models.LGB_MAX_TREES
    for c in winners:
        params.update(c.params)
        if "learning_rate" in c.params:
            trees = c.trees
    label_ = json.dumps(params) if params else "none improved on A1"
    return Config("A9", "A", f"LightGBM, improving changes combined: {label_}",
                  params=params, trees=trees)


# ------------------------------------------------------------------ data

class Data:
    def __init__(self):
        self.cols = models.feature_columns(FEATURES)
        self.need = sorted(set(self.cols) | set(models.SMART_SIGNALS))
        tr = load_train(N, self.need)
        va = load_valid_sample(N, self.need)
        check({"train": tr, "valid": va}, N)
        self.y = label(tr, N).to_numpy()
        self.rep = models.Representation(self.cols).fit(tr)
        self.X = self.rep.transform(tr)
        self.base_w = tr["sample_weight"].to_numpy("float64")
        self.serials = tr["serial_number"].to_numpy()
        self.cat_idx = np.array([i for i, c in enumerate(self.rep.names)
                                 if c.startswith(("missing_", "brand_"))], dtype=int)
        self.Xv, self.yv = self.rep.transform(va), label(va, N).to_numpy()
        self.wv = va["sample_weight"].to_numpy()


def train(cfg: Config, d: Data, seed: int):
    """Train one single model; returns (model, calibrator, fit_seconds)."""
    t = time.time()
    X, y = d.X, d.y
    w = models.balanced_weights(y, d.base_w)
    if cfg.smote:
        from .smote_study import best_variant, train_set
        X, y, w, _ = train_set(best_variant(), d.X, d.y, d.base_w, d.serials, d.cat_idx, seed)
    if cfg.model == "L4":
        import lightgbm as lgb
        params = dict(models.LGB_PARAMS, **cfg.params, seed=seed)
        model = lgb.train(params, lgb.Dataset(X, y, weight=w), num_boost_round=cfg.trees,
                          valid_sets=[lgb.Dataset(d.Xv, d.yv)],
                          callbacks=[lgb.early_stopping(30, verbose=False)])
    else:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=cfg.params.get("C", 1.0), max_iter=300, random_state=seed))
        model.fit(X, y, logisticregression__sample_weight=w)
    raw = models.predict(cfg.model, model, d.Xv)
    cal = IsotonicRegression(out_of_bounds="clip").fit(raw, d.yv, sample_weight=d.wv)
    return model, cal, round(time.time() - t, 1)


def scorer(members: list[tuple[str, object, IsotonicRegression]], rep):
    """Daily score: a single model's raw score, or the mean calibrated probability
    of several models (a blend)."""
    def score(df: pd.DataFrame) -> np.ndarray:
        out = np.empty(len(df))
        for i in range(0, len(df), 1_000_000):
            X = rep.transform(df.iloc[i:i + 1_000_000])
            if len(members) == 1:
                lvl, m, _ = members[0]
                out[i:i + 1_000_000] = models.predict(lvl, m, X)
            else:
                out[i:i + 1_000_000] = np.mean(
                    [cal.predict(models.predict(lvl, m, X)) for lvl, m, cal in members], axis=0)
        return out
    return score


def drive_scores(period: str, score, d: Data) -> evaluate.DriveScores:
    parts, windows = [], []
    for chunk in iter_scoring(period, N, d.need):
        chunk["score"] = score(chunk)
        ds = evaluate.drive_scores(chunk, N, attrs=["brand"])
        parts.append(ds.drives)
        windows.append(ds.window)
    return evaluate.DriveScores(pd.concat(parts, ignore_index=True),
                                pd.concat(windows, ignore_index=True))


def per_brand_detection(test: evaluate.DriveScores, valid: evaluate.DriveScores,
                        far: float) -> dict:
    """Each brand gets its own threshold, set on that brand's healthy validation
    drives for the same false-alarm budget; detection is then counted overall."""
    caught = fails = alarms = healthy = 0
    for brand, g in test.drives.groupby("brand"):
        hv = valid.drives[(valid.drives["brand"] == brand) & ~valid.drives["failing"]]["score"]
        thr = evaluate.threshold_for(hv.to_numpy(), far)
        f = g[g["failing"]]
        h = g[~g["failing"]]
        caught += int((f["score"] >= thr).sum())
        fails += len(f)
        alarms += int((h["score"] >= thr).sum())
        healthy += len(h)
    return {"detection_rate": caught / fails, "false_alarm_rate": alarms / healthy,
            "warning_days_median": float("nan")}


def metrics(test: evaluate.DriveScores, valid: evaluate.DriveScores, per_brand: bool,
            same_period: bool) -> dict:
    """With same_period (stages A–D) thresholds are set on the scored period itself."""
    s = evaluate.summary(test, None if same_period else valid)
    key = "test_curve" if same_period else "valid_threshold"
    one, tenth = s[f"{key}_1pct"], s[f"{key}_0.1pct"]
    if per_brand:
        ref = test if same_period else valid
        one, tenth = per_brand_detection(test, ref, 0.01), per_brand_detection(test, ref, 0.001)
        one["warning_days_median"] = s[f"{key}_1pct"]["warning_days_median"]
    return {"caught_1pct": one["detection_rate"], "far_1pct": one["false_alarm_rate"],
            "caught_0.1pct": tenth["detection_rate"], "far_0.1pct": tenth["false_alarm_rate"],
            "warning_days": one["warning_days_median"], "pr_auc": s["pr_auc"]}


# ------------------------------------------------------------------ log

def log() -> pd.DataFrame:
    return pd.read_csv(LOG) if LOG.exists() else pd.DataFrame(columns=["id", "period", "seed"])


def record(cfg: Config, period: str, seed: int, m: dict, fit_s: float, total_s: float):
    OUT.mkdir(parents=True, exist_ok=True)
    row = {"id": cfg.id, "stage": cfg.stage, "description": cfg.description,
           "period": period, "seed": seed, "model": cfg.model,
           "settings": json.dumps({k: v for k, v in asdict(cfg).items()
                                   if k not in ("id", "stage", "description")}),
           "thresholds": "per brand" if cfg.per_brand else "global",
           **{k: round(v, 5) if isinstance(v, float) else v for k, v in m.items()},
           "fit_seconds": fit_s, "seconds": round(total_s, 1),
           "time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
           "code_version": git_version()}
    df = pd.concat([log(), pd.DataFrame([row])], ignore_index=True)
    df.to_csv(LOG, index=False)
    print(f"{cfg.id} {period} seed {seed}: caught {m['caught_1pct']:.3f} at 1% "
          f"(false alarms {m['far_1pct']:.4f})", flush=True)


def done(cfg_id: str, period: str, seed: int) -> bool:
    r = log()
    return bool(((r["id"] == cfg_id) & (r["period"] == period) & (r["seed"] == seed)).any())


# ------------------------------------------------------------------ stages

def single(cfg: Config, d: Data, seed: int = 0):
    """Train, save and score one single model on validation."""
    t = time.time()
    model, cal, fit_s = train(cfg, d, seed)
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "calibrator": cal, "config": asdict(cfg)},
                MODELS / f"{cfg.id}-seed{seed}.joblib", compress=3)
    v = drive_scores("valid", scorer([(cfg.model, model, cal)], d.rep), d)
    record(cfg, "valid", seed, metrics(v, v, cfg.per_brand, True), fit_s, time.time() - t)
    return v


def load_member(cfg_id: str, seed: int):
    b = joblib.load(MODELS / f"{cfg_id}-seed{seed}.joblib")
    return b["config"]["model"], b["model"], b["calibrator"]


def validation_stages() -> pd.DataFrame:
    """Stages A–D on validation, in order; restart-safe (finished ids are skipped)."""
    d = Data()
    for cfg in STAGE_A + STAGE_B:
        if not done(cfg.id, "valid", 0):
            single(cfg, d)
    rows = log()
    if not done("A9", "valid", 0):
        single(lgb_combined(rows), d)
        rows = log()
    lgb = rebuild(best_of(rows, [c.id for c in STAGE_A] + ["A9"]), rows)
    lr = rebuild(best_of(rows, [c.id for c in STAGE_B]), rows)
    for cfg in (Config("C1", "C", f"{lgb.id} with corrected SMOTE rows", model="L4",
                       params=lgb.params, trees=lgb.trees, smote=True),
                Config("C2", "C", f"{lr.id} with corrected SMOTE rows", model="L2",
                       params=lr.params, smote=True)):
        if not done(cfg.id, "valid", 0):
            single(cfg, d)
    rows = log()
    lgb_pick = best_of(rows, [lgb.id, "C1"])
    lr_pick = best_of(rows, [lr.id, "C2"])
    d1 = Config("D1", "D", f"Blend: mean calibrated probability of {lgb_pick} and {lr_pick}",
                model="blend", members=(lgb_pick, lr_pick))
    if not done("D1", "valid", 0):
        evaluate_combo(d1, d)
        rows = log()
    pick = best_of(rows, [lgb_pick, lr_pick, "D1"])
    members = rebuild(pick, rows).members or (pick,)
    d2 = Config("D2", "D", f"{pick} with per-brand thresholds",
                model="blend" if len(members) > 1 else rebuild(pick, rows).model,
                members=tuple(members), per_brand=True)
    if not done("D2", "valid", 0):
        evaluate_combo(d2, d)
    return log()


def evaluate_combo(cfg: Config, d: Data):
    """A blend or a per-brand configuration, from saved single models."""
    t = time.time()
    v = drive_scores("valid", scorer([load_member(m, 0) for m in cfg.members], d.rep), d)
    record(cfg, "valid", 0, metrics(v, v, cfg.per_brand, True), 0.0, time.time() - t)


def choose(rows: pd.DataFrame) -> list[str]:
    """Configurations run on test (decision M-6): the best LightGBM, the best
    logistic regression, and the best of everything, by validation detection at
    1% false alarms, PR-AUC breaking ties."""
    v = _valid(rows)
    lgb = best_of(rows, v[v["model"] == "L4"]["id"])
    lr = best_of(rows, v[v["model"] == "L2"]["id"])
    overall = best_of(rows, v["id"])
    return list(dict.fromkeys([lgb, lr, overall]))


def rebuild(cfg_id: str, rows: pd.DataFrame) -> Config:
    r = _valid(rows)[_valid(rows)["id"] == cfg_id].iloc[0]
    s = json.loads(r["settings"])
    s["members"] = tuple(s.get("members", ()))
    return Config(cfg_id, r["stage"], r["description"], **s)


def test_stage(seeds=(0, 1, 2)) -> pd.DataFrame:
    """Stage T: the chosen configurations on test, thresholds from validation."""
    d = Data()
    rows = log()
    picks = choose(rows)
    (OUT / "chosen.json").write_text(json.dumps(picks, indent=2))
    for cfg_id in picks:
        cfg = rebuild(cfg_id, rows)
        for seed in seeds:
            if done(cfg_id, "test", seed):
                continue
            t = time.time()
            members, fit_s = [], 0.0
            for m in (cfg.members or (cfg_id,)):
                mc = rebuild(m, rows)
                model, cal, f = train(mc, d, seed)
                members.append((mc.model, model, cal))
                fit_s += f
            score = scorer(members, d.rep)
            v = drive_scores("valid", score, d)
            te = drive_scores("test", score, d)
            record(cfg, "test", seed, metrics(te, v, cfg.per_brand, False), fit_s,
                   time.time() - t)
    return log()
