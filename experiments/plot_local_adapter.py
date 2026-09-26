"""Research figure from the retained local adapter outcomes."""

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
data = [json.loads(s) for s in (ROOT / 'evidence/local-adapter-v1/evaluation.jsonl').read_text().splitlines()]
rows = [r for r in data if r['split'] == 'test']
assert len(rows) == 72
colors = ['#9BA8B3', '#427F82', '#344D62']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none'})
fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), gridspec_kw={'width_ratios': [1, 1.15]})
fig.patch.set_facecolor('#FAFBFC')
for ax in axes:
    ax.set_facecolor('#FAFBFC')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#D8DFE4')
    ax.tick_params(axis='both', length=0, pad=8)
    ax.set_axisbelow(True)
    ax.grid(axis='x', color='#E5EAEE')
values = [sum(r[k]['prediction'] == r['target'] for r in rows) for k in ('base', 'adapter')] + [72]
axes[0].barh(['Unchanged base', 'Local adapter', 'Exact rule'], values, color=colors, height=.54)
axes[0].invert_yaxis()
axes[0].set_xlim(0, 84)
axes[0].set_xticks([0, 24, 48, 72])
axes[0].set_xlabel('Correct workflow decisions / 72')
axes[0].set_title('Matched inputs and inference backend', loc='left', fontsize=11, pad=16)
for i, v in enumerate(values):
    axes[0].text(v + 1.5, i, f'{v}/72', va='center', color='#263849')
labels = ['Abstain', 'Refresh stale evidence', 'Acquire missing evidence', 'Repair', 'Verify', 'Finish']
counts = [sum(r['adapter']['prediction'] == r['target'] for r in rows if r['target'] == k) for k in 'ABCDEF']
axes[1].barh(labels, counts, height=.6, color=['#427F82' if n == 12 else '#BB875E' for n in counts])
axes[1].invert_yaxis()
axes[1].set_xlim(0, 14)
axes[1].set_xticks([0, 4, 8, 12])
axes[1].set_xlabel('Adapter correct decisions / 12 per action')
axes[1].set_title('Remaining failures determine deployment', loc='left', fontsize=11, pad=16)
for i, v in enumerate(counts):
    axes[1].text(v + .2, i, f'{v}/12', va='center', color='#263849')
fig.suptitle('Local adaptation: measured gain, candidate remains inactive', x=.035, ha='left',
             y=.965, fontsize=16, color='#263849')
fig.text(.035, .065, '144 authored training rows · 72 test rows in 12 fixture clusters · one RTX run · six-choice output',
         fontsize=10, color='#556675')
fig.text(.035, .025, '20 test errors and one new failure. Exact rule wins. No coding, edge-device or autonomous-learning qualification.',
         fontsize=9, color='#556675')
fig.subplots_adjust(left=.15, right=.95, bottom=.23, top=.81, wspace=.8)
target = ROOT / 'docs/assets/local-adapter-v1'
fig.savefig(target.with_suffix('.png'), dpi=180)
fig.savefig(target.with_suffix('.svg'))
plt.close(fig)
