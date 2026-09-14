"""Rebuild the report chart: python plot.py (requires matplotlib)."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
s = json.loads((root / 'summary.json').read_text())['summary']
modes = ['B', 'N', 'S0', 'S1']
fig, ax = plt.subplots(figsize=(10, 5.5), layout='constrained')
fig.patch.set_facecolor('#fafbfc'); ax.set_facecolor('#fafbfc')
for offset, build, color, label in [(-0.18, 'reference', '#a8b1bd', 'Before R1'), (0.18, 'repair', '#247e8c', 'R1 checkpoint')]:
    bars = ax.barh([i+offset for i in range(4)], [s[build][m]['seconds'] for m in modes], height=.32, color=color, label=label)
    ax.bar_label(bars, labels=[f'{s[build][m]["seconds"]:.2f} s' for m in modes], padding=5, fontsize=10)
ax.axvline(s['repair']['B']['seconds'], color='#c16530', linestyle='--', linewidth=1.3, label='Contemporary B: 11.20 s')
ax.set_yticks(range(4), ['B · ordinary', 'N · n-gram', 'S0 · dual model', 'S1 · GPU draft tokens'])
ax.invert_yaxis(); ax.set_xlim(0, 30)
ax.set_xlabel('Complete generation time (seconds) — lower is better')
ax.set_title('R1 cuts dual-model time by ~44%; B remains fastest', loc='left', fontsize=14, weight='bold', pad=18)
ax.spines[['top','right','left']].set_visible(False)
ax.grid(axis='x', alpha=.16); ax.set_axisbelow(True)
ax.legend(loc='lower right', frameon=False, fontsize=9)
fig.supxlabel('RTX 5090 · old 8 requests / two B4 groups · one seed · mean of 2 repeats per group, then sum\nOutputs: B 1977 / N 1921 / S0=S1 1938 tokens. Offline batch drain; not serving latency.', fontsize=9)
fig.savefig(root/'generation-time.png', dpi=180)
fig.savefig(root/'generation-time.svg')
