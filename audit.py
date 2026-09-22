"""
Bias audit: selection rates, four-fifths (disparate impact) rule, calibration
by group -- then mitigation and a BEFORE/AFTER comparison.

Methodology, in plain terms:
  1. Fix an operating threshold that gives an overall selection rate of 25%
     (a typical "screen-through" rate for a resume screener).
  2. BEFORE: compute selection rate per protected group, the disparate impact
     ratio (group rate / highest group rate), and flag ratios < 0.80 per the
     four-fifths rule. Also compute PPV (precision at the operating
     threshold) per group as a calibration check.
  3. Mitigation A -- per-group threshold tuning (post-processing): for each
     protected group, set an individual threshold so every group's selection
     rate equals the overall baseline rate. This mechanically equalizes
     selection rates; the cost shows up in accuracy/AUC.
  4. Mitigation B -- sample reweighting (pre-processing): retrain giving each
     intersection group equal total weight, then re-threshold at 25% overall.
  5. AFTER: print the same tables for both mitigations so the
     fairness-vs-accuracy tradeoff is visible.

NOTE (honest limitation, also in the report): per-group thresholds are a
textbook demo technique, not deployment advice -- using different cutoffs by
protected class raises disparate-treatment concerns in real hiring. The
point here is to demonstrate how an audit *measures* the tradeoff, not to
recommend this mitigation in production.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score

from train import train_model, NUMERIC, CATEGORICAL  # reuse the training pipeline

SEED = 42
FOUR_FIFTHS = 0.80
TARGET_SELECTION_RATE = 0.25  # operating point: screen through 25% of applicants
ROOT = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(ROOT, "artifacts")
FIG_DIR = os.path.join(ROOT, "figures")

PROTECTED_DIMS = ["gender", "race_ethnicity", "age_band"]
INTERSECTION = "gender_x_race"


def load_test():
    df = pd.read_csv(os.path.join(ART_DIR, "test_with_scores.csv"))
    df[INTERSECTION] = df["gender"] + " / " + df["race_ethnicity"]
    return df


def threshold_for_rate(scores, rate):
    """Score threshold that selects (approximately) `rate` of the sample."""
    return float(np.quantile(scores, 1.0 - rate))


def group_table(df, pred_col, dims):
    """Selection rate, disparate impact ratio, PPV, and counts per group."""
    rows = []
    for dim in dims:
        rates = df.groupby(dim)[pred_col].mean()
        ref = rates.max()
        for group, rate in rates.items():
            sub = df[df[dim] == group]
            ppv = float(sub.loc[sub[pred_col] == 1, "hire"].mean()) if (sub[pred_col] == 1).any() else float("nan")
            rows.append({
                "dimension": dim,
                "group": group,
                "n": int(len(sub)),
                "selection_rate": round(float(rate), 4),
                "impact_ratio": round(float(rate / ref), 4) if ref > 0 else 0.0,
                "ppv": round(ppv, 4) if not np.isnan(ppv) else None,
                "violates_four_fifths": bool(rate / ref < FOUR_FIFTHS) if ref > 0 else False,
            })
    return rows


def summarize(df, pred_col):
    y = df["hire"].to_numpy()
    pred = df[pred_col].to_numpy()
    score = df["score"].to_numpy()
    return {
        "selection_rate": round(float(pred.mean()), 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "auc": round(float(roc_auc_score(y, score)), 4),
        "n_selected": int(pred.sum()),
        "n_total": int(len(df)),
    }


def plot_calibration(df, pred_col, out_path):
    """Calibration: observed hire rate vs predicted score decile, by group."""
    dims = {"gender": ["Female", "Male"],
            "race_ethnicity": ["Asian", "Black", "Hispanic", "White"]}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (dim, groups) in zip(axes, dims.items()):
        for group in groups:
            sub = df[df[dim] == group]
            if len(sub) < 50:
                continue
            sub = sub.copy()
            sub["decile"] = pd.qcut(sub["score"], 10, labels=False, duplicates="drop")
            cal = sub.groupby("decile").agg(mean_score=("score", "mean"),
                                            hire_rate=("hire", "mean"))
            ax.plot(cal["mean_score"], cal["hire_rate"], marker="o", label=group)
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        ax.set_title(f"Calibration by {dim}")
        ax.set_xlabel("Mean predicted score (decile)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Observed hire rate")
    fig.suptitle("Calibration curves by protected group (synthetic data)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    df = load_test()
    dims = PROTECTED_DIMS + [INTERSECTION]
    results = {"four_fifths_rule": FOUR_FIFTHS,
               "target_selection_rate": TARGET_SELECTION_RATE,
               "scenarios": {}}

    # ---------- BEFORE: single global threshold ----------
    t0 = threshold_for_rate(df["score"].to_numpy(), TARGET_SELECTION_RATE)
    df["pred_baseline"] = (df["score"] >= t0).astype(int)
    before = summarize(df, "pred_baseline")
    before["threshold"] = round(t0, 4)
    results["scenarios"]["baseline"] = {
        "summary": before,
        "groups": group_table(df, "pred_baseline", dims),
    }

    # ---------- AFTER A: per-group threshold tuning ----------
    df["pred_thresh_tuned"] = 0
    tuned_thresholds = {}
    for group, sub in df.groupby(INTERSECTION):
        tg = threshold_for_rate(sub["score"].to_numpy(), TARGET_SELECTION_RATE)
        tuned_thresholds[group] = round(float(tg), 4)
        df.loc[df[INTERSECTION] == group, "pred_thresh_tuned"] = (
            sub["score"] >= tg
        ).astype(int).to_numpy()
    tuned = summarize(df, "pred_thresh_tuned")
    results["scenarios"]["threshold_tuning"] = {
        "summary": tuned,
        "group_thresholds": tuned_thresholds,
        "groups": group_table(df, "pred_thresh_tuned", dims),
    }

    # ---------- AFTER B: sample reweighting ----------
    # Give each gender x race intersection group equal total influence in
    # training (train.csv lacks the intersection column, so rebuild it).
    train_df = pd.read_csv(os.path.join(ART_DIR, "train.csv"))
    train_df[INTERSECTION] = train_df["gender"] + " / " + train_df["race_ethnicity"]
    tcounts = train_df[INTERSECTION].value_counts()
    w_train = train_df[INTERSECTION].map(len(train_df) / (len(tcounts) * tcounts)).to_numpy()
    rw_model = train_model(train_df, sample_weight=w_train)
    df["score_reweighted"] = rw_model.predict_proba(df[NUMERIC + CATEGORICAL])[:, 1]
    t_rw = threshold_for_rate(df["score_reweighted"].to_numpy(), TARGET_SELECTION_RATE)
    df["pred_reweighted"] = (df["score_reweighted"] >= t_rw).astype(int)
    rw_summary = summarize(df.assign(score=df["score_reweighted"]), "pred_reweighted")
    rw_summary["threshold"] = round(t_rw, 4)
    results["scenarios"]["sample_reweighting"] = {
        "summary": rw_summary,
        "groups": group_table(df, "pred_reweighted", dims),
    }

    with open(os.path.join(ART_DIR, "audit_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    plot_calibration(df, "pred_baseline",
                     os.path.join(FIG_DIR, "calibration_by_group.png"))

    # ---------- Console report ----------
    for name, label in [("baseline", "BEFORE (single global threshold)"),
                        ("threshold_tuning", "AFTER A: per-group threshold tuning"),
                        ("sample_reweighting", "AFTER B: sample reweighting")]:
        sc = results["scenarios"][name]
        s = sc["summary"]
        print(f"\n===== {label} =====")
        print(f"overall: selection_rate={s['selection_rate']:.3f}  "
              f"accuracy={s['accuracy']:.4f}  auc={s['auc']:.4f}  "
              f"selected {s['n_selected']}/{s['n_total']}")
        for dim in ["gender", "race_ethnicity"]:
            print(f"  -- {dim} --")
            for row in sc["groups"]:
                if row["dimension"] != dim:
                    continue
                flag = "  <-- VIOLATES 4/5" if row["violates_four_fifths"] else ""
                ppv = f"{row['ppv']:.3f}" if row["ppv"] is not None else "n/a"
                print(f"    {row['group']:<10} n={row['n']:<5} "
                      f"sel={row['selection_rate']:.3f}  IR={row['impact_ratio']:.3f}  "
                      f"PPV={ppv}{flag}")

    print(f"\nSaved audit_results.json and figures/calibration_by_group.png")


if __name__ == "__main__":
    main()
