"""Run the full conditional-threshold pipeline: features -> labels -> crossovers -> controls -> figures -> benefit vs degradation.

Usage (Colab, Drive mounted at /content/drive):
    !python run_all.py --root /content/drive/MyDrive/nuscenes
Options are shared by every step (see common.get_args). Each step can also be run on its own,
e.g. `python step3_crossovers.py --root ...`. Steps 1 and 2 skip work that is already on disk (step 1 re-runs if gt_features.csv.gz lacks newer columns).
Add --download to fetch the team files from the GitHub release first (step 0).
"""
import os
import sys
import time

import pandas as pd

download = '--download' in sys.argv
if download:
    sys.argv.remove('--download')

import step0_download
import step1_features
import step2_match
import step3_crossovers
import step4_controls
import step5_plots
import step6_benefit_degradation
from common import get_args, log

if __name__ == '__main__':
    a = get_args('Conditional-threshold pipeline (all steps)')
    t0 = time.time()
    if download:
        log('=' * 80, '\nStep 0: download')
        step0_download.main(a)
    feat = os.path.join(a.out, 'gt_features.csv.gz')
    up_to_date = os.path.exists(feat) and os.path.exists(os.path.join(a.out, 'ego_samples.csv')) and \
        'r500_ego' in pd.read_csv(feat, nrows=1).columns
    if up_to_date:
        log('=' * 80, '\nStep 1: features already on disk, skipped (delete gt_features.csv.gz to recompute)')
    else:
        log('=' * 80, '\nStep 1: per-box features')
        step1_features.main(a)
    for name, step in [('Step 2: hit labels and FP counts', step2_match), ('Step 3: crossovers', step3_crossovers),
                       ('Step 4: controls and robustness', step4_controls), ('Step 5: figures', step5_plots),
                       ('Step 6: initial benefit vs degradation', step6_benefit_degradation)]:
        log('=' * 80, f'\n{name}')
        step.main(a)
    log('=' * 80, f'\nAll done in {(time.time() - t0) / 60:.1f} min. Results in {a.out}')
