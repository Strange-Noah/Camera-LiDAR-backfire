"""Step 3: class-level and conditional backfire crossovers (score >= 0.1).

For a group of GT boxes:
    gap(k)       = miss_rate(fusion, k) - miss_rate(LiDAR-only)       (< 0: fusion better, > 0: backfire)
    crossover    = delay from which gap stays > 0 (linear interpolation), 95% CI by scene bootstrap
    benefit lost = gap(k10) - gap(clean), percentage points

Cells = class (10 + ALL) x condition x bin. Cells are computed under both miss definitions:
    all    : every unmatched GT box (corrected E4, main result)
    old_e4 : original E4 definition (comparison only)
A cell is a robust finding if, under BOTH definitions: SUPPORTED, |shift from class reference| > 50 ms
in the same direction, class reference outside the 95% CI, and CI width <= 400 ms.

Outputs (<out>/tables/):
    class_crossovers.csv, conditional_cells_all.csv, conditional_cells_old_e4.csv, robust_findings.csv
"""
import os
import time

import numpy as np
import pandas as pd

from common import (CLASSES, COND_COLS, SCREEN, SceneBootstrap, empty_result, fmt_cross, get_args, group_mask,
                    load_features, load_miss, log, save_csv, support_status)

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 40)
DEFINITIONS = ['all', 'old_e4']


def conditional_cells(gtf, boot, MISS):
    rows = []
    for g in CLASSES + ['ALL']:
        base = group_mask(gtf, g)
        ref = boot.analyze(base, MISS)
        for col in COND_COLS:
            for b in gtf.loc[base, col].dropna().unique():
                m = base & gtf[col].eq(b).to_numpy()
                nt, ns, top1, st = support_status(gtf[m])
                r = boot.analyze(m, MISS) if st != 'SPARSE' else empty_result(m.sum(), 'skip')
                rows.append(dict(group=g, condition=col[:-4], bin=str(b), N_traj=nt, N_scene=ns, top1_scene=top1,
                                 status=st, ref_crossover_ms=ref['crossover_ms'], ref_benefit_lost=ref['benefit_lost'], **r))
    cells = pd.DataFrame(rows)
    cells['shift_ms'] = cells.crossover_ms - cells.ref_crossover_ms
    return cells


def main(a):
    t0 = time.time()
    gtf = load_features(a.out)
    boot = SceneBootstrap(gtf.scene_token)
    MISS = {d: load_miss(a.out, 0.1, d) for d in DEFINITIONS}

    # ---- class level ----
    rows = []
    for d in DEFINITIONS:
        for g in CLASSES + ['ALL']:
            rows.append(dict(definition=d, group=g, **boot.analyze(group_mask(gtf, g), MISS[d])))
    cls = pd.DataFrame(rows)
    save_csv(cls, os.path.join(a.out, 'tables', 'class_crossovers.csv'), index=False)
    cls['crossover [95% CI]'] = cls.apply(fmt_cross, axis=1)
    view = cls.pivot(index='group', columns='definition', values=['miss_clean', 'miss_lidar', 'crossover [95% CI]', 'benefit_lost'])
    log('Class-level results (miss rate, score >= 0.1):')
    log(view.reindex(CLASSES + ['ALL']).round(3).to_string())

    # ---- conditional cells ----
    cells = {}
    for d in DEFINITIONS:
        cells[d] = conditional_cells(gtf, boot, MISS[d])
        save_csv(cells[d], os.path.join(a.out, 'tables', f'conditional_cells_{d}.csv'), index=False)
        c = cells[d]
        sup = c[(c.status == 'SUPPORTED') & (c.group != 'ALL') & c.shift_ms.notna()]
        out_ci = (sup.ref_crossover_ms < sup.ci_lo) | (sup.ref_crossover_ms > sup.ci_hi)
        log(f'\n[{d}] cells: {len(c)}; status: {c.status.value_counts().to_dict()}; '
            f'SUPPORTED cells shifted > {SCREEN["min_shift_ms"]:.0f} ms: {int((sup.shift_ms.abs() > SCREEN["min_shift_ms"]).sum())} '
            f'(reference outside CI: {int((out_ci & (sup.shift_ms.abs() > SCREEN["min_shift_ms"])).sum())})')

    # ---- robust findings: consistent under both definitions ----
    key = ['group', 'condition', 'bin']
    keep = key + ['N_boxes', 'N_scene', 'status', 'ref_crossover_ms', 'crossover_ms', 'ci_lo', 'ci_hi', 'shift_ms',
                  'gap_clean', 'gap_k10', 'benefit_lost']
    both = cells['all'][keep].merge(cells['old_e4'][keep], on=key, suffixes=('', '_old_e4'))
    ok = (both.group != 'ALL')
    for sfx in ('', '_old_e4'):
        ok &= both['status' + sfx].eq('SUPPORTED')
        ok &= both['shift_ms' + sfx].abs() > SCREEN['min_shift_ms']
        ok &= (both['ref_crossover_ms' + sfx] < both['ci_lo' + sfx]) | (both['ref_crossover_ms' + sfx] > both['ci_hi' + sfx])
        ok &= (both['ci_hi' + sfx] - both['ci_lo' + sfx]) <= SCREEN['max_ci_width_ms']
    ok &= np.sign(both.shift_ms) == np.sign(both.shift_ms_old_e4)
    robust = both[ok].sort_values(['group', 'shift_ms'])
    # Flag cells whose early crossover only reflects a small clean camera benefit (|gap_clean| < 1 pt).
    robust['small_clean_benefit'] = robust.gap_clean > -1.0
    save_csv(robust, os.path.join(a.out, 'tables', 'robust_findings.csv'), index=False)
    log(f'\nRobust conditional findings (consistent under both definitions): {len(robust)}')
    show = ['group', 'condition', 'bin', 'N_boxes', 'ref_crossover_ms', 'crossover_ms', 'ci_lo', 'ci_hi',
            'gap_clean', 'gap_k10', 'benefit_lost', 'small_clean_benefit']
    log(robust[show].round(1).to_string(index=False))
    log('small_clean_benefit=True: the camera barely helps at clean (gap_clean > -1 pt), so an early crossover '
        'does not mean fast degradation; read together with benefit_lost.')
    log(f'\nstep 3 done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main(get_args('Step 3: class-level and conditional crossovers'))
