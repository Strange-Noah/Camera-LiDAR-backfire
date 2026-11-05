"""Focused fixed Distance × clean projected width; accepted labels only."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'screening'))
from screen import Data, csvwrite, row


def analyze(data, out):
    dl, di = data.groups['distance']
    wl, wi = data.groups['projected_width']
    grid = [[data.group((di == i) & (wi == j)) for j in range(len(wl))] for i in range(len(dl))]
    assert sum(g['N'] for cells in grid for g in cells) == 53043
    result = dict(distance_labels=dl, width_labels=wl, grid=grid,
                  population=53043, geometry_valid=52664, geometry_missing=379,
                  conditions=27, grouping='clean fixed projected width; right-open bins')
    out = Path(out)
    out.mkdir(exist_ok=False)
    (out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    rows = [row(g, distance=dl[i], width=wl[j]) for i, cells in enumerate(grid) for j, g in enumerate(cells)]
    csvwrite(out / 'all_conditions.csv', rows)
    fields = ['distance', 'width', 'N', 'scenes', 'supported', 'B_pp', 'D5_pp', 'G5_pp', 'G5_SD_pp', 'upward_brackets', 'small_initial_benefit', 'baseline_no_benefit']
    csvwrite(out / 'compact.csv', [{k: r.get(k, '') for k in fields} for r in rows])
    plots(result, out / 'figures')


def plots(result, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(out)
    out.mkdir(exist_ok=False)
    grid = result['grid'][:5]
    dl, wl = result['distance_labels'][:5], result['width_labels']
    def save(fig, name):
        fig.tight_layout()
        for ext in ['png', 'pdf', 'svg']:
            fig.savefig(out / (name + '.' + ext), dpi=180)
        plt.close(fig)
    arrays = [np.array([[np.nan if not g['N'] else g[k] if k == 'B' else g[k][-1] for g in cells] for cells in grid]) for k in ['B', 'D_mean', 'G_mean']]
    limit = max(np.nanmax(np.abs(a)) for a in arrays)
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, values, title in zip(axs, arrays, ['Initial benefit B', 'Degradation D at σ=5°', 'Backfire gap G at σ=5°']):
        im = ax.imshow(values, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
        ax.set_xticks(range(len(wl)), wl, rotation=25, ha='right')
        ax.set_yticks(range(5), dl)
        ax.set(title=title, xlabel='Clean projected width', ylabel='Distance')
        for i, cells in enumerate(grid):
            for j, g in enumerate(cells):
                label = 'empty' if not g['N'] else '%.2f\nN=%d%s' % (values[i, j], g['N'], '' if g['supported'] else '*')
                ax.text(j, i, label, ha='center', va='center', fontsize=8)
        fig.colorbar(im, ax=ax, label='percentage points')
    fig.suptitle('Fixed Car groups; * N<200 or fewer than 10 contributing scenes')
    save(fig, 'A_BDG_heatmaps')
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=False)
    for i, ax in enumerate(axs.flat[:5]):
        for j in range(4):
            g = grid[i][j]
            if not g['N']: continue
            ax.errorbar(range(0, 6), [-g['B']] + g['G_mean'], yerr=[0] + g['G_sd'], marker='o', color='C%d' % j, linestyle='-' if g['supported'] else ':', label=wl[j] + (' *' if not g['supported'] else ''))
        ax.axhline(0, color='black', lw=.8)
        ax.set(title=dl[i], xlabel='Rotation σ (degrees per axis)', ylabel='G (pp)')
        ax.legend(fontsize=8)
        ax.set_xticks(range(6))
        ax.tick_params(labelbottom=True)
    axs.flat[5].axis('off')
    fig.suptitle('B: Conditional crossover; mean ± sample SD of five seeds; dotted/* = limited support')
    save(fig, 'B_conditional_crossover')
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for j, ax in enumerate(axs.flat):
        for i in range(5):
            g = grid[i][j]
            if not g['N'] or not g['supported']: continue
            ax.errorbar(range(1, 6), g['D_mean'], yerr=g['D_sd'], marker='o', color='C%d' % i, label=dl[i])
        ax.set_xticks(range(1, 6))
        ax.tick_params(labelbottom=True)
        ax.set(title=wl[j], xlabel='Rotation σ (degrees per axis)', ylabel='D (pp)')
        ax.legend(fontsize=8)
    fig.suptitle('C: Residual distance association at fixed width; supported cells only; mean ± seed SD')
    save(fig, 'C_residual_distance')
    counts = np.array([[g['N'] for g in cells] for cells in grid])
    fractions = counts / counts.sum(1, keepdims=True) * 100
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, values, title, unit in zip(axs, [counts, fractions], ['GT counts', 'Composition within distance'], ['GT objects', 'percent of distance bin']):
        im = ax.imshow(values, cmap='Blues', aspect='auto')
        ax.set_xticks(range(len(wl)), wl, rotation=25, ha='right')
        ax.set_yticks(range(5), dl)
        ax.set(title=title, xlabel='Clean projected width', ylabel='Distance')
        for i, cells in enumerate(grid):
            for j, g in enumerate(cells):
                value = str(g['N']) if unit == 'GT objects' else '%.1f%%' % values[i, j]
                ax.text(j, i, value + ('*' if g['N'] and not g['supported'] else ''), ha='center', va='center', fontsize=9)
        fig.colorbar(im, ax=ax, label=unit)
    fig.suptitle('D: Full fixed population, including missing widths; * limited support')
    save(fig, 'D_composition')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--labels'); p.add_argument('--geometry'); p.add_argument('--scenes')
    p.add_argument('--summary'); p.add_argument('--output', required=True)
    a = p.parse_args()
    if a.summary:
        plots(json.loads(Path(a.summary).read_text()), a.output)
    else:
        analyze(Data(a.labels, a.geometry, a.scenes), a.output)
