# Synthetic Ranking/Selection Disparity Audit Demo

A methodology exercise using a synthetic applicant generator and logistic
regression. It is not a real hiring-system audit, legal opinion or compliance
certification. Protected attributes are used for measurement; they are not inputs
to the baseline prediction model. Per-group thresholding is a research illustration,
not a recommendation to use group-specific cutoffs in hiring.

## Corrected evaluation protocol

7,500 synthetic rows: 4,500 training, 1,125 calibration, 1,875 test.
The model fits on training rows. Global, group-specific and reweighted-model
thresholds fit on CALIBRATION scores, then stay fixed when applied to test rows.
Unseen calibration groups use the global threshold. Test selection rates need not
be equal; the old near-1.00 ratios were obtained by tuning on the evaluated sample.

```bash
pip install -r requirements.txt
python make_data.py
python train.py
python audit.py
python make_report.py
```

Current results: [generated report](audit-report.md) and
[`artifacts/audit_results.json`](artifacts/audit_results.json).
Regenerate local model/data artifacts with these commands; do not reuse cached
artifacts from an older split. Findings describe the artificial generator only.
Selection-rate parity does not establish equal error rates or fair individual
outcomes. Group-level uncertainty and multiple random seeds remain future work.
