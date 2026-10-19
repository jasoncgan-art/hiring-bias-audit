# Synthetic disparity audit: held-out evaluation

All data is generated. This is not a compliance certification or evidence about a real employer.

Thresholds fit on calibration rows only; test rows are used once for measurement.

| Scenario | Test n | Selection rate | Accuracy | AUC |
|---|---:|---:|---:|---:|
| baseline | 1875 | 0.2416 | 0.5744 | 0.6947 |
| threshold_tuning | 1875 | 0.2501 | 0.5616 | 0.6947 |
| sample_reweighting | 1875 | 0.2379 | 0.5717 | 0.6931 |

## baseline

| Dimension | Group | n | Selection rate | Impact ratio |
|---|---|---:|---:|---:|
| gender | Female | 918 | 0.2386 | 0.9757 |
| gender | Male | 957 | 0.2445 | 1.0000 |
| race_ethnicity | Asian | 343 | 0.2303 | 0.7403 |
| race_ethnicity | Black | 416 | 0.1707 | 0.5486 |
| race_ethnicity | Hispanic | 351 | 0.1852 | 0.5952 |
| race_ethnicity | White | 765 | 0.3111 | 1.0000 |
| age_band | 18-29 | 464 | 0.0690 | 0.1092 |
| age_band | 30-49 | 1034 | 0.1770 | 0.2803 |
| age_band | 50+ | 377 | 0.6313 | 1.0000 |
| gender_x_race | Female / Asian | 167 | 0.2515 | 0.6957 |
| gender_x_race | Female / Black | 198 | 0.2071 | 0.5728 |
| gender_x_race | Female / Hispanic | 167 | 0.2096 | 0.5798 |
| gender_x_race | Female / White | 386 | 0.2617 | 0.7239 |
| gender_x_race | Male / Asian | 176 | 0.2102 | 0.5816 |
| gender_x_race | Male / Black | 218 | 0.1376 | 0.3807 |
| gender_x_race | Male / Hispanic | 184 | 0.1630 | 0.4510 |
| gender_x_race | Male / White | 379 | 0.3615 | 1.0000 |

## threshold_tuning

| Dimension | Group | n | Selection rate | Impact ratio |
|---|---|---:|---:|---:|
| gender | Female | 918 | 0.2680 | 1.0000 |
| gender | Male | 957 | 0.2330 | 0.8696 |
| race_ethnicity | Asian | 343 | 0.2828 | 1.0000 |
| race_ethnicity | Black | 416 | 0.2356 | 0.8330 |
| race_ethnicity | Hispanic | 351 | 0.2336 | 0.8261 |
| race_ethnicity | White | 765 | 0.2510 | 0.8875 |
| age_band | 18-29 | 464 | 0.0625 | 0.0928 |
| age_band | 30-49 | 1034 | 0.1799 | 0.2670 |
| age_band | 50+ | 377 | 0.6737 | 1.0000 |
| gender_x_race | Female / Asian | 167 | 0.2754 | 0.8519 |
| gender_x_race | Female / Black | 198 | 0.2525 | 0.7810 |
| gender_x_race | Female / Hispanic | 167 | 0.3234 | 1.0000 |
| gender_x_race | Female / White | 386 | 0.2487 | 0.7691 |
| gender_x_race | Male / Asian | 176 | 0.2898 | 0.8961 |
| gender_x_race | Male / Black | 218 | 0.2202 | 0.6809 |
| gender_x_race | Male / Hispanic | 184 | 0.1522 | 0.4706 |
| gender_x_race | Male / White | 379 | 0.2533 | 0.7833 |

## sample_reweighting

| Dimension | Group | n | Selection rate | Impact ratio |
|---|---|---:|---:|---:|
| gender | Female | 918 | 0.2386 | 1.0000 |
| gender | Male | 957 | 0.2372 | 0.9943 |
| race_ethnicity | Asian | 343 | 0.2245 | 0.7402 |
| race_ethnicity | Black | 416 | 0.1755 | 0.5786 |
| race_ethnicity | Hispanic | 351 | 0.1823 | 0.6012 |
| race_ethnicity | White | 765 | 0.3033 | 1.0000 |
| age_band | 18-29 | 464 | 0.0625 | 0.0962 |
| age_band | 30-49 | 1034 | 0.1663 | 0.2560 |
| age_band | 50+ | 377 | 0.6499 | 1.0000 |
| gender_x_race | Female / Asian | 167 | 0.2395 | 0.6877 |
| gender_x_race | Female / Black | 198 | 0.2172 | 0.6235 |
| gender_x_race | Female / Hispanic | 167 | 0.2156 | 0.6189 |
| gender_x_race | Female / White | 386 | 0.2591 | 0.7438 |
| gender_x_race | Male / Asian | 176 | 0.2102 | 0.6036 |
| gender_x_race | Male / Black | 218 | 0.1376 | 0.3951 |
| gender_x_race | Male / Hispanic | 184 | 0.1522 | 0.4369 |
| gender_x_race | Male / White | 379 | 0.3483 | 1.0000 |

## Interpretation

Inspect the measured ratios rather than assuming a mitigation works. Calibration parity does not guarantee test parity.
Accuracy is measured against synthetic historically biased labels, not an unbiased qualification ground truth.
PPV is predictive value, not a complete probability-calibration measure. AUC assesses continuous model scores; threshold changes do not change those scores.
Group-specific thresholding illustrates a statistical mechanism only. No real hiring action is recommended or taken.
Single seed and no confidence intervals: these point estimates are not proof that disparities will generalize.
