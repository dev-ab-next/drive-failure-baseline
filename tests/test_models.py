import numpy as np
import pandas as pd
import pytest

from dfb import models


def toy(n_neg=2000, n_pos=50, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(0, 1, (n_neg, 3)), rng.normal(2, 1, (n_pos, 3))]).astype("float32")
    y = np.concatenate([np.zeros(n_neg, int), np.ones(n_pos, int)])
    return X, y


def test_smote_and_ctgan_add_the_same_number_of_rows():
    X, y = toy()
    k = models.n_to_add(y)
    assert k == 2000 // 10 - 50
    Xs, ys = models.smote(X, y, 0)
    Xc, yc, info = models.ctgan_topup(X, y, ["a", "b", "c"], 0, epochs=2)
    assert len(ys) - len(y) == k == len(yc) - len(y) == info["rows_added"]
    assert (ys == 0).sum() == (yc == 0).sum() == 2000  # healthy rows untouched


def test_undersample_ratio():
    X, y = toy()
    Xu, yu = models.undersample(X, y, 0)
    assert (yu == 1).sum() == 50 and (yu == 0).sum() == 500


def test_l1_counts_signals_above_zero():
    df = pd.DataFrame({"s5": [0, 3, None], "s187": [None, 1, 0], "s188": [0, 0, 0],
                       "s197": [0, 2, 0], "s198": [0, 0, 1]})
    assert models.l1_score(df).tolist() == [0, 3, 1]


def test_representation_flags_and_fills():
    X = pd.DataFrame({"a": [1.0, None, 3.0], "b": [0.0, 1.0, 2.0]})
    rep = models.Representation(["a", "b"]).fit(X)
    out = rep.transform(X)
    assert rep.names == ["a", "b", "missing_a"]
    assert out[1, 2] == 1 and not np.isnan(out).any()


def test_balanced_weights_equalize_classes():
    y = np.array([0, 0, 0, 1])
    w = models.balanced_weights(y, np.array([50.0, 50.0, 50.0, 1.0]))
    assert w[y == 1].sum() == pytest.approx(w[y == 0].sum())
