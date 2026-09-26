"""Render a research figure from the complete retained paired comparison."""

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = json.loads((ROOT / 'evidence/mbpp-code-comparison-v1/summary.json').read_text())
    models, paired = report['models'], report['paired']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2), gridspec_kw={'width_ratios': [1.1, 1]})
    background, ink = '#FAFBFC', '#263849'
    fig.patch.set_facecolor(background)
    for ax in axes:
        ax.set_facecolor(background)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_color('#D8DFE4')
        ax.tick_params(axis='both', length=0, pad=8)
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#E5EAEE')
    values = [models[arm]['correct_combined'] for arm in ('base', 'adapter')]
    axes[0].barh(['Unchanged base', 'Local adapter'], values, color=['#9BA8B3', '#427F82'], height=.48)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 188)
    axes[0].set_xticks([0, 41, 82, 123, 164])
    axes[0].set_xlabel('Tasks passing base and additional tests / 164')
    axes[0].set_title('Task success', loc='left', fontsize=12, pad=18)
    for i, value in enumerate(values):
        axes[0].text(value + 3, i, f'{value}/164', va='center', color=ink)
    changes = [paired['gained'], paired['regressed']]
    axes[1].barh(['New passes', 'New failures'], changes, color=['#427F82', '#BB875E'], height=.48)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, max(changes) * 1.4 + 2)
    axes[1].set_xlabel('Changed outcomes on the same tasks')
    axes[1].set_title('Gains and regressions', loc='left', fontsize=12, pad=18)
    for i, value in enumerate(changes):
        axes[1].text(value + .6, i, str(value), va='center', color=ink)
    interval = [100 * value for value in paired['paired_bootstrap_95_interval']]
    fig.suptitle('Local code adaptation: one paired experiment', x=.04, ha='left', y=.97, fontsize=17, color=ink)
    fig.text(.04, .905, 'Qwen2.5 1.5B  ·  HumanEval+ 164 tasks  ·  identical BF16 backend  ·  native grading', color='#556675')
    fig.text(.04, .265,
             f"Accuracy change: {100 * paired['delta']:+.2f} percentage points; descriptive paired 95% interval [{interval[0]:+.2f}, {interval[1]:+.2f}].",
             fontsize=10, color=ink)
    a, b = models['base'], models['adapter']
    fig.text(.04, .20, f"Generated tokens: {a['generated_tokens']:,} → {b['generated_tokens']:,}   |   "
             f"Summed generation batch time: {a['total_batch_generation_seconds']:.1f} → {b['total_batch_generation_seconds']:.1f} s", color=ink)
    fig.text(.04, .145, 'Batch time excludes loading, grading and context preparation; it is not request latency or total workflow cost.',
             fontsize=9, color='#556675')
    fig.text(.04, .09, 'One training seed, greedy decoding, custom prompt; four tasks previously used for pipeline smoke. Candidate inactive.',
             fontsize=9, color='#556675')
    fig.text(.04, .04, 'Task 32 retains the official numerical limitation. No context-runtime treatment, retention or edge-device qualification.',
             fontsize=9, color='#556675')
    fig.subplots_adjust(left=.16, right=.95, bottom=.40, top=.78, wspace=.6)
    target = ROOT / 'docs/assets/mbpp-code-comparison-v1'
    fig.savefig(target.with_suffix('.png'), dpi=180)
    fig.savefig(target.with_suffix('.svg'))
    plt.close(fig)


if __name__ == '__main__':
    main()
