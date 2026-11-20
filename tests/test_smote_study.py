import numpy as np

from dfb.smote_study import Variant, smote_rows


def data(seed=0):
    rng = np.random.default_rng(seed)
    n_pos_drives, days = 40, 7
    pos = rng.normal(size=(n_pos_drives, 1, 3)) + rng.normal(scale=0.01, size=(n_pos_drives, days, 3))
    pos = pos.reshape(-1, 3)
    flags = np.repeat(rng.integers(0, 2, size=(n_pos_drives, 2)), days, axis=0)  # fixed per drive
    Xp = np.hstack([pos, flags]).astype(np.float32)
    Xn = np.hstack([rng.normal(size=(500, 3)), rng.integers(0, 2, size=(500, 2))]).astype(np.float32)
    X = np.vstack([Xp, Xn])
    y = np.r_[np.ones(len(Xp), int), np.zeros(len(Xn), int)]
    serials = np.r_[np.repeat([f"d{i}" for i in range(n_pos_drives)], days),
                    [f"h{i}" for i in range(len(Xn))]]
    return X, y, serials, np.array([3, 4])


def test_adds_requested_rows():
    X, y, s, cat = data()
    new, _ = smote_rows(X, y, s, 123, Variant("t"), cat, 0)
    assert new.shape == (123, X.shape[1])


def test_plain_smote_mostly_uses_same_drive():
    # each drive's days are nearly identical, so plain SMOTE pairs a row with its own drive
    X, y, s, cat = data()
    _, diag = smote_rows(X, y, s, 2000, Variant("t", k=5), cat, 0)
    assert diag["same_drive_parent_share"] > 0.9


def test_cross_drive_never_uses_same_drive():
    X, y, s, cat = data()
    _, diag = smote_rows(X, y, s, 2000, Variant("t", cross_drive=True), cat, 0)
    assert diag["same_drive_parent_share"] == 0.0


def test_categorical_columns_stay_whole():
    X, y, s, cat = data()
    new, diag = smote_rows(X, y, s, 2000, Variant("t", cross_drive=True, categorical=True), cat, 0)
    assert set(np.unique(new[:, cat])) <= {0.0, 1.0}
    assert diag["rows_with_blended_flag_or_brand"] == 0.0


def test_new_rows_lie_between_parents():
    X, y, s, cat = data()
    new, _ = smote_rows(X, y, s, 500, Variant("t"), cat, 0)
    lo, hi = X[y == 1].min(axis=0), X[y == 1].max(axis=0)
    assert ((new >= lo - 1e-6) & (new <= hi + 1e-6)).all()
