"""Focused acceptance of one fixed-width extension; no new labels or scoring."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'screening'))
from screen import Data


def validate(result, phase_a, data=None):
    grid = result['grid']
    assert sum(g['N'] for cells in grid for g in cells) == 53043
    assert sum(cells[-1]['N'] for cells in grid) == 379
    for cells in grid:
        for g in cells:
            if not g['N']: continue
            assert np.allclose(g['G'], np.array(g['D']) - g['B'], atol=1e-10, rtol=0)
            for k in ['D', 'G']:
                v = np.array(g[k])
                assert np.allclose(v.mean(1), g[k + '_mean'], atol=1e-12, rtol=0)
                assert np.allclose(v.std(1, ddof=1), g[k + '_sd'], atol=1e-12, rtol=0)
            mean = g['G_mean']
            expected = ([[0, 1]] if g['B'] > 0 and mean[0] > 0 else []) + [[i, i + 1] for i in range(1, 5) if mean[i - 1] <= 0 < mean[i]]
            assert g['observed_upward_brackets'] == expected
            assert g['supported'] == (g['N'] >= 200 and g['scenes'] >= 10)
    for axis, key in enumerate(['distance', 'projected_width']):
        for i, expected in enumerate(phase_a[key]):
            gs = grid[i] if axis == 0 else [cells[i] for cells in grid]
            n = sum(g['N'] for g in gs)
            assert n == expected['N']
            if not n: continue
            for field in ['rates_percent', 'B', 'D', 'G']:
                weighted = sum(np.array(g[field]) * g['N'] for g in gs if g['N']) / n
                assert np.allclose(weighted, expected[field], atol=1e-10, rtol=0)
    if data is not None:
        _, di = data.groups['distance']; _, wi = data.groups['projected_width']
        for i, cells in enumerate(grid):
            for j, g in enumerate(cells):
                mask = (di == i) & (wi == j)
                assert int(mask.sum()) == g['N']
                digest = hashlib.sha256('\n'.join(k for k, m in zip(data.keys, mask) if m).encode()).hexdigest()
                assert digest == g['denominator_sha256']
                if not g['N']: continue
                assert len(np.unique(data.si[mask])) == g['scenes']
                rates = 100 * (~data.hits[mask]).sum(0) / g['N']
                assert np.allclose(rates, g['rates_percent'], atol=1e-12, rtol=0)
                assert abs((rates[0] - rates[1]) - g['B']) < 1e-12
                assert np.allclose((rates[2:] - rates[1]).reshape(5, 5), g['D'], atol=1e-12, rtol=0)
                assert np.allclose((rates[2:] - rates[0]).reshape(5, 5), g['G'], atol=1e-12, rtol=0)
    return dict(objects=53043, valid_widths=52664, missing_widths=379,
                BDG='PASS', seeds='PASS', crossings='PASS', support='PASS',
                phase_a_marginals='PASS', frozen_membership_and_rates='PASS' if data is not None else 'NOT_RUN')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--summary', required=True); p.add_argument('--phase-a', required=True)
    p.add_argument('--labels'); p.add_argument('--geometry'); p.add_argument('--scenes')
    p.add_argument('--regenerated')
    a = p.parse_args()
    data = Data(a.labels, a.geometry, a.scenes) if a.labels else None
    result = validate(json.loads(Path(a.summary).read_text()), json.loads(Path(a.phase_a).read_text()), data)
    if a.regenerated:
        original = Path(a.summary).parent / 'figures'
        files = sorted(original.glob('*.png')); assert len(files) == 4
        for f in files:
            assert f.read_bytes() == (Path(a.regenerated) / f.name).read_bytes(), f.name
        result['figure_regeneration'] = '4 PNGs byte-identical'
    print(json.dumps(result, indent=2))
