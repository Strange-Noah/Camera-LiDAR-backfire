"""Run the full conditional-threshold pipeline: features -> labels -> crossovers -> controls -> figures.

Usage (Colab, Drive mounted at /content/drive):
    !python run_all.py --root /content/drive/MyDrive/nuscenes
Options are shared by every step (see common.get_args). Each step can also be run on its own,
e.g. `python step3_crossovers.py --root ...`. Steps 1 and 2 skip work that is already on disk.
Add --download to fetch the team files from the GitHub release first (step 0).
"""
import os
import sys
import time

download = '--download' in sys.argv
if download:
    sys.argv.remove('--download')

import step0_download
import step1_features
import step2_match
import step3_crossovers
import step4_controls
import step5_plots
from common import get_args, log

if __name__ == '__main__':
    a = get_args('Conditional-threshold pipeline (all steps)')
    t0 = time.time()
    if download:
        log('=' * 80, '\nStep 0: download')
        step0_download.main(a)
    if os.path.exists(os.path.join(a.out, 'gt_features.csv.gz')) and os.path.exists(os.path.join(a.out, 'ego_samples.csv')):
        log('=' * 80, '\nStep 1: features already on disk, skipped (delete gt_features.csv.gz to recompute)')
    else:
        log('=' * 80, '\nStep 1: per-box features')
        step1_features.main(a)
    for name, step in [('Step 2: hit labels and FP counts', step2_match), ('Step 3: crossovers', step3_crossovers),
                       ('Step 4: controls and robustness', step4_controls), ('Step 5: figures', step5_plots)]:
        log('=' * 80, f'\n{name}')
        step.main(a)
    log('=' * 80, f'\nAll done in {(time.time() - t0) / 60:.1f} min. Results in {a.out}')
