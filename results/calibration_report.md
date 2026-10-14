# Calibration report (Phases 2-4)

Rows: 1535 steps (755 fit, 780 cal), 1040 trajectories.

## Signal distributions (cal block)
verbalized:
  0.0-0.1     0 
  0.1-0.2     0 
  0.2-0.3     0 
  0.3-0.4     0 
  0.4-0.5     0 
  0.5-0.6     0 
  0.6-0.7     0 
  0.7-0.8     0 
  0.8-0.9     0 
  0.9-1.0   780 ########################################
agreement:
  0.0-0.1     2 #
  0.1-0.2     0 
  0.2-0.3     4 #
  0.3-0.4     0 
  0.4-0.5     4 #
  0.5-0.6     0 
  0.6-0.7     5 #
  0.7-0.8     0 
  0.8-0.9     7 #
  0.9-1.0   758 ########################################

## Calibration metrics on cal
| variant | ECE | Brier |
|---|---|---|
| raw | 0.0659 | 0.0675 |
| A_temperature_scaling | 0.0531 | 0.0678 |
| B_logistic_regression | 0.0502 | 0.0647 |

Chosen for the live run: **B_logistic_regression** (lowest Brier on cal).

## Thresholds (cal trajectories, conformal risk control)
| alpha | lambda | expected approvals/traj |
|---|---|---|
| 0.02 | 0.9603 | 1.16 |
| 0.05 | 0.9136 | 1.01 |
| 0.1 | 0.6678 | 0.95 |
| 0.2 | 0.6678 | 0.95 |
