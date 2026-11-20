import json

import numpy as np
import pandas as pd

from dfb import evaluate, improve


def fake_log(scores: dict) -> pd.DataFrame:
    rows = []
    for cid, (caught, pr, model) in scores.items():
        cfg = next((c for c in improve.STAGE_A + improve.STAGE_B if c.id == cid),
                   improve.Config(cid, cid[0], cid, model=model))
        rows.append({"id": cid, "stage": cfg.stage, "description": cfg.description,
                     "period": "valid", "seed": 0, "model": model,
                     "settings": json.dumps({"model": model, "params": cfg.params,
                                             "trees": cfg.trees, "smote": False,
                                             "members": [], "per_brand": False}),
                     "caught_1pct": caught, "pr_auc": pr})
    return pd.DataFrame(rows)


def test_best_of_uses_pr_auc_to_break_ties():
    rows = fake_log({"A1": (0.5, 0.2, "L4"), "A2": (0.5, 0.3, "L4"), "A3": (0.4, 0.9, "L4")})
    assert improve.best_of(rows, ["A1", "A2", "A3"]) == "A2"


def test_combined_keeps_only_improving_changes_and_best_value_per_setting():
    rows = fake_log({"A1": (0.50, 0.2, "L4"), "A2": (0.52, 0.2, "L4"), "A3": (0.55, 0.2, "L4"),
                     "A4": (0.49, 0.2, "L4"), "A5": (0.51, 0.2, "L4"), "A6": (0.48, 0.2, "L4"),
                     "A7": (0.53, 0.2, "L4"), "A8": (0.40, 0.2, "L4")})
    c = improve.lgb_combined(rows)
    # A2 (31 leaves) and A3 (127 leaves) both improve; A3 is better, so 127 wins
    assert c.params["num_leaves"] == 127
    assert c.params["min_data_in_leaf"] == 20  # A5 improved, A4 did not
    assert c.params["lambda_l2"] == 10.0       # A7 improved
    assert "learning_rate" not in c.params     # A6 did not improve
    assert "feature_fraction" not in c.params  # A8 did not improve
    assert c.trees == 300


def test_choose_picks_best_lightgbm_logistic_and_overall():
    rows = fake_log({"A1": (0.50, 0.2, "L4"), "C1": (0.53, 0.2, "L4"),
                     "B1": (0.55, 0.3, "L2"), "D1": (0.57, 0.3, "blend")})
    assert improve.choose(rows) == ["C1", "B1", "D1"]


def test_per_brand_thresholds_meet_the_budget_in_each_brand():
    rng = np.random.default_rng(0)
    n = 20000
    drives = pd.DataFrame({"serial_number": [f"s{i}" for i in range(n)],
                           "brand": np.where(np.arange(n) % 2, "A", "B"),
                           "failing": np.arange(n) % 50 == 0})
    # brand B's healthy drives score higher, so one global threshold would over-alarm on B
    drives["score"] = rng.random(n) + (drives["brand"] == "B") * 0.5 + drives["failing"] * 0.4
    ds = evaluate.DriveScores(drives, pd.DataFrame(columns=["serial_number", "days_to_failure",
                                                            "score"]))
    out = improve.per_brand_detection(ds, ds, 0.01)
    assert out["false_alarm_rate"] <= 0.01
    for b in ("A", "B"):
        g = drives[(drives["brand"] == b) & ~drives["failing"]]
        thr = evaluate.threshold_for(g["score"].to_numpy(), 0.01)
        assert (g["score"] >= thr).mean() <= 0.01
