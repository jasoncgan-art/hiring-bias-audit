"""Render measured test results from the calibration-fitted audit."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    results = json.loads((ROOT / 'artifacts/audit_results.json').read_text())
    lines = ['# Synthetic disparity audit: held-out evaluation', '',
             'All data is generated. This is not a compliance certification or evidence about a real employer.', '',
             'Thresholds fit on calibration rows only; test rows are used once for measurement.', '',
             '| Scenario | Test n | Selection rate | Accuracy | AUC |',
             '|---|---:|---:|---:|---:|']
    for name, scenario in results['scenarios'].items():
        s = scenario['summary']
        lines.append(f"| {name} | {s['n_total']} | {s['selection_rate']:.4f} | {s['accuracy']:.4f} | {s['auc']:.4f} |")
    for name, scenario in results['scenarios'].items():
        lines += ['', f'## {name}', '', '| Dimension | Group | n | Selection rate | Impact ratio |', '|---|---|---:|---:|---:|']
        for row in scenario['groups']:
            lines.append(f"| {row['dimension']} | {row['group']} | {row['n']} | {row['selection_rate']:.4f} | {row['impact_ratio']:.4f} |")
    lines += ['', '## Interpretation', '',
              'Inspect the measured ratios rather than assuming a mitigation works. Calibration parity does not guarantee test parity.',
              'Accuracy is measured against synthetic historically biased labels, not an unbiased qualification ground truth.',
              'PPV is predictive value, not a complete probability-calibration measure. AUC assesses continuous model scores; threshold changes do not change those scores.',
              'Group-specific thresholding illustrates a statistical mechanism only. No real hiring action is recommended or taken.',
              'Single seed and no confidence intervals: these point estimates are not proof that disparities will generalize.', '']
    (ROOT / 'audit-report.md').write_text('\n'.join(lines))
    print('Wrote audit-report.md')


if __name__ == '__main__':
    main()
