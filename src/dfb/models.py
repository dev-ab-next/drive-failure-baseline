"""Model ladder L0-L4 and imbalance methods M0-M3 (B-7, B-8, E-3, E-4, M-2).

All learned models and methods share one feature representation (M-1):
F-core columns, signed log transform, empty -> training median plus a
"not reported" flag. So M0, M1, M2 (and M3) differ only in the rows they
train on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import core_features, full_features

SMART_SIGNALS = ["s5", "s187", "s188", "s197", "s198"]  # E-4
TARGET_RATIO = 10  # E-3: healthy rows per failure-window row after top-up


class Representation:
    """Fixed transform: signed log1p, median fill learned on training rows,
    plus a flag per column that is ever empty in training (M-1)."""

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame) -> "Representation":
        t = self._log(X[self.columns])
        self.medians = t.median()
        self.flag_cols = [c for c in self.columns if X[c].isna().any()]
        return self

    @staticmethod
    def _log(X: pd.DataFrame) -> pd.DataFrame:
        a = X.to_numpy(dtype="float32")
        return pd.DataFrame(np.sign(a) * np.log1p(np.abs(a)), columns=X.columns, index=X.index)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        t = self._log(X[self.columns])
        flags = X[self.flag_cols].isna().to_numpy(dtype="float32")
        filled = t.fillna(self.medians).to_numpy(dtype="float32")
        return np.hstack([filled, flags])

    @property
    def names(self) -> list[str]:
        return self.columns + [f"missing_{c}" for c in self.flag_cols]


def feature_columns(feature_set: str) -> list[str]:
    return core_features() if feature_set == "core" else full_features()


# ------------------------------------------------------------ methods

def balanced_weights(y: np.ndarray, base: np.ndarray) -> np.ndarray:
    """M0: keep the sample weights (F-12), then scale classes to equal total."""
    w = base.astype("float64").copy()
    pos, neg = w[y == 1].sum(), w[y == 0].sum()
    w[y == 1] *= neg / pos
    return w / w.mean()


def undersample(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """M1: keep every failure-window row and TARGET_RATIO healthy rows per one."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    keep = rng.choice(neg, size=min(len(neg), TARGET_RATIO * len(pos)), replace=False)
    idx = np.concatenate([pos, keep])
    return X[idx], y[idx]


def n_to_add(y: np.ndarray) -> int:
    """E-3: rows M2 and M3 both add, to reach TARGET_RATIO healthy per failure row."""
    return max(0, int(np.ceil((y == 0).sum() / TARGET_RATIO)) - int((y == 1).sum()))


def smote(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """M2: SMOTE top-up to TARGET_RATIO (E-3)."""
    from imblearn.over_sampling import SMOTE
    target = int((y == 1).sum()) + n_to_add(y)
    sm = SMOTE(sampling_strategy={1: target}, k_neighbors=5, random_state=seed)
    return sm.fit_resample(X, y)


# -------------------------------------------------------------- models

def l1_score(df: pd.DataFrame) -> np.ndarray:
    """E-4: number of SMART 5, 187, 188, 197, 198 above zero (empty = not above)."""
    return sum((df[c].fillna(0) > 0).astype("int8") for c in SMART_SIGNALS).to_numpy()


def make_model(level: str, seed: int):
    if level == "L2":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=1.0, max_iter=300, random_state=seed))
    if level == "L3":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=100, max_samples=0.2, min_samples_leaf=50,
                                      max_depth=12, n_jobs=4, random_state=seed)
    if level == "L4":
        return None  # trained with lightgbm.train for early stopping
    raise ValueError(level)


LGB_MAX_TREES = 300  # M-3 (revised): scoring 77M rows must fit the run budget
LGB_PARAMS = {
    "objective": "binary", "learning_rate": 0.1, "num_leaves": 63,
    "min_data_in_leaf": 100, "feature_fraction": 0.8, "bagging_fraction": 0.8,
    "bagging_freq": 1, "lambda_l2": 1.0, "num_threads": 4, "verbose": -1,
}


def fit(level: str, X: np.ndarray, y: np.ndarray, w: np.ndarray | None, seed: int,
        X_val: np.ndarray | None = None, y_val: np.ndarray | None = None):
    if level == "L4":
        import lightgbm as lgb
        params = dict(LGB_PARAMS, seed=seed)
        train = lgb.Dataset(X, y, weight=w)
        valid = lgb.Dataset(X_val, y_val, reference=train)
        return lgb.train(params, train, num_boost_round=LGB_MAX_TREES, valid_sets=[valid],
                         callbacks=[lgb.early_stopping(30, verbose=False)])
    model = make_model(level, seed)
    if level == "L2":
        model.fit(X, y, logisticregression__sample_weight=w)
    else:
        model.fit(X, y, sample_weight=w)
    return model


def predict(level: str, model, X: np.ndarray) -> np.ndarray:
    if level == "L4":
        return model.predict(X, num_iteration=model.best_iteration)
    return model.predict_proba(X)[:, 1]


CTGAN_EPOCHS = 100  # M-2


def ctgan_topup(X: np.ndarray, y: np.ndarray, names: list[str], seed: int,
                epochs: int = CTGAN_EPOCHS) -> tuple[np.ndarray, np.ndarray, dict]:
    """M3: train CTGAN on the training failure-window rows only, then add
    exactly as many synthetic failure rows as SMOTE would (E-3, M-2)."""
    import torch
    from ctgan import CTGAN

    torch.manual_seed(seed)
    np.random.seed(seed)
    pos = pd.DataFrame(X[y == 1], columns=names)
    discrete = [c for c in names if c.startswith(("missing_", "brand_"))]
    for c in discrete:
        pos[c] = pos[c].round(4).astype(str)
    batch = max(10, min(500, len(pos) // 10 * 10))  # CTGAN needs a multiple of 10
    gen = CTGAN(epochs=epochs, batch_size=batch, verbose=False, cuda=False)
    gen.set_random_state(seed)
    gen.fit(pos, discrete_columns=discrete)
    k = n_to_add(y)
    synth = gen.sample(k)
    for c in discrete:
        synth[c] = synth[c].astype(float)
    Xs = synth[names].to_numpy(dtype="float32")
    X2 = np.vstack([X, Xs])
    y2 = np.concatenate([y, np.ones(k, dtype=y.dtype)])
    return X2, y2, {"rows_added": int(k), "ctgan_epochs": epochs,
                    "ctgan_training_rows": int((y == 1).sum())}
