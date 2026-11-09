"""Plot completed Gate A records, retaining every method/scene/seed outcome."""
import argparse
import json
from pathlib import Path


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    records = [json.loads(line) for line in (a.results/'trajectories.jsonl').read_text().splitlines()]
    a.output.mkdir(parents=True, exist_ok=False)
    scenes = sorted({r['scene'] for r in records})
    methods = ['minimal', 'full_no_preselection', 'full_preselection']
    for cadence in sorted({r['cadence'] for r in records}):
        for axis in ['elapsed_seconds', 'accepted_count', 'valid_updates']:
            fig, axes = plt.subplots(len(scenes), 2, figsize=(13, 2.6*len(scenes)), squeeze=False)
            for row, scene in enumerate(scenes):
                for seed, ax in enumerate(axes[row]):
                    for method, color in zip(methods, ['#777777', '#d97706', '#0072b2']):
                        selected = [r for r in records if r['scene'] == scene and
                                    r['seed'] == seed and r['method'] == method and r['cadence'] == cadence]
                        if not selected:
                            continue
                        ax.plot([r[axis] for r in selected], [r['residual_degrees'] for r in selected],
                                color=color, label=method, linewidth=1)
                        ax.axhline(selected[0]['initial_error_degrees'], color='black', linestyle=':', linewidth=.5)
                    ax.set(title=f'{scene[:8]} seed {seed}', xlabel=axis, ylabel='SO(3) residual (degrees)')
                    ax.legend(fontsize=7)
            fig.tight_layout(); fig.savefig(a.output/(cadence+'-'+axis+'.png'), dpi=140); plt.close(fig)


if __name__ == '__main__':
    main()
