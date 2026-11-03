# Overall rotational backfire thresholds

Measured references: Clean Fusion mAP 68.547401%, NDS 71.386516%; LiDAR-only mAP 64.664666%, NDS 69.250968%.

σ=0 is accepted Runner B Clean #1 only. Clean #2 is not duplicated as another seed.

## mAP

Smallest tested five-seed mean below LiDAR-only: 4 degrees.

|σ (°)|Mean (%)|Sample SD (pp)|Margin vs LiDAR (pp)|Pointwise95% t interval (pp)|Seeds below LiDAR|
|---|---:|---:|---:|---|---|
|1|67.706068|0.465126|+3.041401|[+2.463871, +3.618932]|0/5|
|2|66.322312|0.857703|+1.657645|[+0.592666, +2.722625]|0/5|
|3|65.278782|0.957377|+0.614116|[-0.574625, +1.802857]|2/5|
|4|64.484467|1.094939|-0.180200|[-1.539746, +1.179347]|3/5|
|5|63.833197|1.110796|-0.831469|[-2.210704, +0.547766]|4/5|

Linear interpolated crossing: **3.77°**, bracket **3–4°**, direction enter_backfire. This is not a measured threshold.

## NDS

Smallest tested five-seed mean below LiDAR-only: 4 degrees.

|σ (°)|Mean (%)|Sample SD (pp)|Margin vs LiDAR (pp)|Pointwise95% t interval (pp)|Seeds below LiDAR|
|---|---:|---:|---:|---|---|
|1|70.849345|0.272105|+1.598377|[+1.260513, +1.936240]|0/5|
|2|70.087001|0.463493|+0.836032|[+0.260530, +1.411535]|0/5|
|3|69.528705|0.503880|+0.277736|[-0.347914, +0.903386]|2/5|
|4|69.080286|0.575646|-0.170683|[-0.885442, +0.544077]|4/5|
|5|68.751324|0.575656|-0.499644|[-1.214415, +0.215127]|4/5|

Linear interpolated crossing: **3.62°**, bracket **3–4°**, direction enter_backfire. This is not a measured threshold.

## Interpretation and limitations

Pointwise two-sided95% Student-t interval on mean(Fusion−fixed LiDAR), df4; sample SD across5 seeds, t*SD/sqrt5.
Approximate independent, identically distributed, roughly normal run-level margins across5 calibration seeds on this fixed validation population; each run also contains residual execution variability that is not separately identified. LiDAR and Clean references treated as fixed; model execution variability and data-population uncertainty not estimated.
n=5; no6019-frame pseudo-replication; intervals not simultaneous/multiplicity-corrected. Seed frequencies descriptive, not calibrated probabilities. Same seed is paired across sigma by the frozen perturbation design. Clean sigma0 is one run, not five.
Piecewise linear only for adjacent observed margins bracketing zero. Sigma0 anchor is one accepted Clean observation; all nonzero points are five-seed means. No smoothing, monotonic constraint or extrapolation.
Standard deviation of the frozen camera-local rotation-vector error per axis; persistent six-camera calibration draw per condition, not per-frame random corruption.

One seed crossing alone is not a statistically established overall threshold. The observed-grid rule is descriptive; confidence intervals qualify evidence rather than redefining the threshold. mAP/NDS disagreement is preserved. No conditional backfire analysis is performed.
