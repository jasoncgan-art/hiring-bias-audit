# hiring-bias-audit

A demo-ready toolkit that performs a **NYC Local Law 144–style bias audit** of an automated resume-screening tool: measure disparate impact, apply mitigation, and report before/after.

> **All data is synthetic** (generated at runtime by `make_data.py`, seed 42).
> This repo demonstrates audit **methodology** — it is not a certification of any real system.

## Why this exists

Since January 2023, **New York City Local Law 144** has required employers using automated employment decision tools (AEDTs) in hiring to conduct **annual, independent bias audits** — measuring selection rates and impact ratios across sex, race/ethnicity, and their intersections — and to publish a summary of results. Similar rules are spreading to other jurisdictions. That created a real, specialized niche: people who understand both machine learning *and* how hiring actually works.

That's the angle here. I'm a Data Science master's student (UMich) with **5+ years in HR and recruiting** — I've screened resumes, run interview pipelines, and seen how referral-heavy processes and "gut feel" hiring quietly skew outcomes. This repo applies that domain knowledge to the audit side: building the dataset the way hiring data actually looks (referrals, employment gaps, assessment scores), injecting the kinds of bias I've seen in real pipelines, and then auditing it the way the law requires.

## Methodology

1. **`make_data.py`** — generates 7,500 synthetic applicants with qualification features (years of experience, education, skill assessment score, referral flag, employment gaps), protected attributes (gender, race/ethnicity, age band), and a historical `hire` label. Three *documented, deliberate* biases are injected (see the script docstring and `data/generation_config.json`):
   - **measurement bias**: the fictional skill assessment under-scores female, Black, Hispanic, and Asian applicants;
   - **label bias**: historical hire decisions favored male (+0.45 logit) and white (+0.25) applicants;
   - **proxy bias**: referrals boost hiring but are ~2× more common among male/white applicants.
   
   Age band has **no** injected bias — it acts as a control dimension.
2. **`train.py`** — trains a baseline **logistic regression** screener (75/25 split, stratified). Why logistic regression: it's the standard baseline in fairness literature, trains in seconds on CPU, and its coefficients are interpretable — which matters when you have to explain *why* someone was screened out. Protected attributes are **excluded** from the model; the audit (which is required to see them) gets them separately.
3. **`audit.py`** — fixes an operating threshold at a 25% overall selection rate, then computes per-group selection rates, **disparate impact ratios** (flagged below the 0.80 four-fifths rule), and **calibration by group** (PPV + calibration curves in `figures/`). Then it applies two mitigations and prints a BEFORE/AFTER comparison with accuracy/AUC so the fairness–accuracy tradeoff is visible:
   - **A. per-group threshold tuning** (post-processing): each intersection group gets its own threshold targeting the overall 25% rate;
   - **B. sample reweighting** (pre-processing): retrain with each gender×race group given equal total weight.
4. **`make_report.py`** — writes `audit-report.md`, a mock audit deliverable: scope, methodology, findings tables, mitigation results, limitations.

## Results (from a real run on this machine)

Test set n = 1,875. Baseline model: accuracy 0.578, AUC 0.693 at the 25% operating threshold.

**BEFORE — disparate impact (baseline, single global threshold):**

| Group | Selection rate | Impact ratio | 4/5 rule |
|---|---|---|---|
| Female | 0.245 | 0.961 | pass |
| Male | 0.255 | 1.000 | — (reference) |
| Asian | 0.242 | 0.762 | **FAIL** |
| Black | 0.180 | 0.568 | **FAIL** |
| Hispanic | 0.194 | 0.610 | **FAIL** |
| White | 0.318 | 1.000 | — (reference) |

The audit detects the injected bias: three race/ethnicity groups fall well below the 0.80 threshold. The age-band control dimension passes (no violations), confirming the audit flags real disparity rather than noise.

**AFTER — mitigation comparison:**

| Scenario | Selection rate | Accuracy | AUC | Min. impact ratio |
|---|---|---|---|---|
| Baseline | 0.250 | 0.5776 | 0.6931 | 0.568 (Black) |
| A: per-group threshold tuning | 0.251 | 0.5669 | 0.6931 | ~1.00 (all groups) |
| B: sample reweighting | 0.250 | 0.5755 | 0.6909 | 0.590 (Hispanic) |

Takeaways: threshold tuning equalizes selection rates by construction at a small accuracy cost (−0.011) — the classic fairness–accuracy tradeoff, made visible. Reweighting alone barely moves the needle, because the bias lives in the *features* (measurement bias) and the *labels*, which reweighting can't undo. That negative result is arguably the most instructive one: it shows why audits have to test mitigations empirically instead of assuming they work.

Full tables, intersection results, and PPV-by-group numbers are in [`audit-report.md`](audit-report.md).

## How to run

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python make_data.py     # generate the synthetic dataset  (~2 s)
python train.py         # train the baseline screener     (~5 s)
python audit.py         # audit + mitigations, prints tables (~10 s)
python make_report.py   # write audit-report.md           (<1 s)
```

Python 3.10+, CPU-only. Full pipeline runs in under a minute. Everything is deterministic (seed 42).

## What I'd do next

- Add **equalized odds** (TPR/FPR gaps) alongside disparate impact — Local Law 144 centers impact ratios, but a serious audit looks at error-rate disparities too.
- Try **in-processing** mitigation (e.g., fairness-constrained optimization) instead of only pre/post-processing.
- Test **threshold sensitivity**: re-run the audit across operating points (10%–50% selection rates) to see how violations move.
- Replace the synthetic generator with a **proper data card** workflow so the same code can audit a real (properly consented, de-identified) dataset.
- Add **statistical significance** reporting — with small groups, an impact ratio of 0.79 can be noise; a real audit needs confidence intervals.

## Limitations (read before citing this)

- **Synthetic data**: every finding describes the generator, not any real hiring process. Effect sizes are illustrative.
- **Single metric family**: disparate impact + calibration only — no equalized odds, individual fairness, or feedback-loop analysis.
- **Threshold tuning is a demo technique**: deploying different cutoffs by protected class raises disparate-treatment concerns in real hiring. It's shown here to make the tradeoff *measurable*, not as production advice.
- **Not a certification**: a real Local Law 144 audit requires an independent auditor, historical data, and published results. This repo demonstrates the methodology an auditor applies.
