"""
Synthetic applicant dataset generator for the hiring-bias-audit demo.

EVERYTHING HERE IS SYNTHETIC. No real applicants, no real company data.
The generation process is fully documented so the audit has a known
"ground truth" to detect: we deliberately inject bias into the hiring
labels, mimicking a historical hiring process that favored certain groups.

Injected biases (the audit is expected to find these):
  1. MEASUREMENT BIAS (feature-level): the fictional skill assessment
     systematically under-scores female applicants and Black/Hispanic/Asian
     applicants (see SCORE_PENALTY). This is the main driver of the
     disparate impact -- it mirrors real cases where a vendor assessment
     has adverse impact against protected groups.
  2. LABEL BIAS (direct): the historical "hire" decision was more likely
     for male applicants (+0.45 logit boost) and for white applicants
     (+0.25 logit boost), independent of qualifications. This models a
     human-driven hiring process with documented gender/race favoritism.
  3. PROXY / PIPELINE BIAS (indirect): the `referral_flag` feature boosts
     hire probability (as in real referral-heavy pipelines), but referrals
     are not distributed evenly -- male and white applicants are ~2x more
     likely to arrive with a referral. The screening model is trained only
     on non-protected features, yet still inherits bias through this proxy.

What we deliberately do NOT inject: no age-band bias, so the age audit
dimension acts as a control and should show no four-fifths violation.

The protected attributes are stored alongside the features because a bias
AUDIT is allowed (and required, e.g. under NYC Local Law 144) to see them.
The trained model itself never receives them as inputs.
"""

import json
import os
import numpy as np
import pandas as pd

SEED = 42
N_APPLICANTS = 7500
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT_CSV = os.path.join(OUT_DIR, "applicants.csv")
OUT_META = os.path.join(OUT_DIR, "generation_config.json")

# --- Injected bias parameters (kept in one place, and in the audit report) ---
# 1) MEASUREMENT BIAS in a model-visible feature: the (fictional) skill
#    assessment systematically under-scores some groups. This is the main
#    driver of the disparate impact the audit will detect.
SCORE_PENALTY = {("Female", "Black"): 11.0, ("Female", "Hispanic"): 9.0,
                 ("Male", "Black"): 11.0, ("Male", "Hispanic"): 9.0,
                 ("Female", "Asian"): 5.0, ("Male", "Asian"): 5.0,
                 ("Female", "White"): 4.0}
SCORE_PENALTY_DEFAULT = 0.0  # Male / White: no penalty
# 2) DIRECT LABEL BIAS: the historical "hire" decision favored some groups.
LABEL_BIAS_LOGIT = {"male": 0.45, "white": 0.25}  # direct favoritism in hire labels
# 3) PROXY / PIPELINE BIAS: referrals boost hiring but are unevenly distributed.
REFERRAL_RATE = {("Male", "White"): 0.24, ("Female", "White"): 0.16}
REFERRAL_RATE_DEFAULT = 0.10  # everyone else


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    rng = np.random.default_rng(SEED)

    # ---- Protected attributes (fully synthetic marginals) ----
    gender = rng.choice(["Female", "Male"], size=N_APPLICANTS, p=[0.48, 0.52])
    race = rng.choice(
        ["Asian", "Black", "Hispanic", "White"],
        size=N_APPLICANTS,
        p=[0.20, 0.22, 0.18, 0.40],
    )
    age_band = rng.choice(
        ["18-29", "30-49", "50+"], size=N_APPLICANTS, p=[0.25, 0.55, 0.20]
    )

    # ---- Qualification-related features ----
    # Experience rises with age band; everything else is group-independent
    # so that any group disparity in outcomes traces back to the injected bias.
    exp_mean = np.where(age_band == "18-29", 2.0, np.where(age_band == "30-49", 7.0, 16.0))
    years_experience = np.clip(rng.normal(exp_mean, 3.0), 0, 40).round(1)

    education_level = rng.choice(
        ["High School", "Bachelor", "Master", "PhD"],
        size=N_APPLICANTS,
        p=[0.15, 0.50, 0.28, 0.07],
    )
    skill_assessment_score = np.clip(rng.normal(70, 12, N_APPLICANTS), 0, 100).round(1)
    # INJECTED measurement bias: the assessment under-scores some groups
    penalties = np.array(
        [SCORE_PENALTY.get((g, r), SCORE_PENALTY_DEFAULT) for g, r in zip(gender, race)]
    )
    skill_assessment_score = np.clip(skill_assessment_score - penalties, 0, 100).round(1)

    # Referral flag: INJECTED pipeline bias -- skewed by gender x race
    referral_probs = np.array(
        [REFERRAL_RATE.get((g, r), REFERRAL_RATE_DEFAULT) for g, r in zip(gender, race)]
    )
    referral_flag = (rng.random(N_APPLICANTS) < referral_probs).astype(int)

    employment_gap_months = np.clip(
        rng.exponential(4.0, N_APPLICANTS), 0, 60
    ).round(1)

    # ---- Latent "true" qualification (what a fair process would reward) ----
    edu_map = {"High School": 0.0, "Bachelor": 0.5, "Master": 0.8, "PhD": 1.0}
    edu_score = np.array([edu_map[e] for e in education_level])
    qualification = (
        0.40 * (skill_assessment_score / 100.0)
        + 0.30 * np.clip(years_experience / 20.0, 0, 1.0)
        + 0.20 * edu_score
        - 0.10 * np.clip(employment_gap_months / 24.0, 0, 1.0)
    )

    # ---- Historical hire label: qualification + referral + INJECTED bias ----
    logit = (
        6.0 * (qualification - 0.48)          # qualification signal
        + 0.9 * referral_flag                # referral premium (proxy for bias)
        + LABEL_BIAS_LOGIT["male"] * (gender == "Male")
        + LABEL_BIAS_LOGIT["white"] * (race == "White")
    )
    hire_prob = sigmoid(logit)
    hire = (rng.random(N_APPLICANTS) < hire_prob).astype(int)

    df = pd.DataFrame(
        {
            "applicant_id": [f"SYN-{i:05d}" for i in range(N_APPLICANTS)],
            "years_experience": years_experience,
            "education_level": education_level,
            "skill_assessment_score": skill_assessment_score,
            "referral_flag": referral_flag,
            "employment_gap_months": employment_gap_months,
            "gender": gender,
            "race_ethnicity": race,
            "age_band": age_band,
            "hire": hire,  # historical label -- synthetic
        }
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    meta = {
        "synthetic": True,
        "seed": SEED,
        "n_applicants": N_APPLICANTS,
        "hire_rate": round(float(df["hire"].mean()), 4),
        "injected_bias": {
            "label_bias_logit": LABEL_BIAS_LOGIT,
            "referral_rate_by_gender_race": {f"{g}/{r}": p for (g, r), p in REFERRAL_RATE.items()},
            "referral_rate_default": REFERRAL_RATE_DEFAULT,
            "note": "Age band has NO injected bias (control dimension).",
        },
        "column_roles": {
            "features": [
                "years_experience", "education_level", "skill_assessment_score",
                "referral_flag", "employment_gap_months",
            ],
            "protected_attributes": ["gender", "race_ethnicity", "age_band"],
            "label": "hire",
        },
    }
    with open(OUT_META, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {OUT_CSV} ({len(df)} synthetic applicants, hire rate {df['hire'].mean():.3f})")
    print(f"Wrote {OUT_META}")
    print("Selection rate by gender (synthetic ground truth):")
    print(df.groupby("gender")["hire"].mean().round(3).to_string())
    print("Selection rate by race/ethnicity (synthetic ground truth):")
    print(df.groupby("race_ethnicity")["hire"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
