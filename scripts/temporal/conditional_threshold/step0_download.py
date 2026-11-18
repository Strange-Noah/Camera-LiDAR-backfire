"""Step 0: download the team's T1 prediction files and gt_common from the GitHub release.

Files land in --data (default <root>/team_data). Re-running skips files that already exist
(either the .json or the .json.gz form). nuScenes v1.0-trainval metadata must be obtained
separately from nuscenes.org and extracted to --meta.
"""
import os
import subprocess

from common import RELEASE_ASSETS, RELEASE_URL, get_args, log


def main(a):
    os.makedirs(a.data, exist_ok=True)
    for asset in RELEASE_ASSETS:
        alt = asset[:-3] if asset.endswith('.gz') else asset + '.gz'
        if any(os.path.exists(os.path.join(a.data, x)) and os.path.getsize(os.path.join(a.data, x)) > 0 for x in (asset, alt)):
            log(f'  already present: {asset}')
            continue
        r = subprocess.run(['wget', '-c', '-nv', '-P', a.data, f'{RELEASE_URL}/{asset}'], capture_output=True, text=True)
        ok = r.returncode == 0 and os.path.exists(os.path.join(a.data, asset))
        log(f'  {"downloaded" if ok else "FAILED"}: {asset}' + ('' if ok else f'\n{r.stderr[-300:]}'))
    log('metadata folder present:', os.path.isdir(a.meta), f'({a.meta})')


if __name__ == '__main__':
    main(get_args('Download T1 predictions and gt_common from the team release'))
