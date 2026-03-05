import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 14
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

C_NORMAL = '#4A90D9'
C_ANHEDONIC = '#E85D4A'

df_raw = pd.read_csv('eval_multirun_raw.csv')
df_summary = pd.read_csv('eval_multirun_summary.csv')
NUM_RUNS = len(df_summary)
run0 = df_raw[df_raw['run_id'] == 0]

# ══════════════════════════════════════════════════════════════════
# FIGURE 1: Positive Emotion Words
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 6))

nv = df_summary['normal_pos_words'].values
av = df_summary['anhedonic_pos_words'].values
nm, am = nv.mean(), av.mean()
ns = nv.std() / np.sqrt(NUM_RUNS)
as_ = av.std() / np.sqrt(NUM_RUNS)

bars = ax.bar(['Normal', 'Anhedonic'], [nm, am],
              yerr=[ns, as_], color=[C_NORMAL, C_ANHEDONIC],
              capsize=12, width=0.55, edgecolor='black', linewidth=0.8)

for bar, val in zip(bars, [nm, am]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.5,
            f'{val:.3f}', ha='center', fontsize=16, fontweight='bold', color='white')

for v in nv:
    ax.plot(0, v, 'o', color='black', alpha=0.5, markersize=7, zorder=5)
for v in av:
    ax.plot(1, v, 'o', color='black', alpha=0.5, markersize=7, zorder=5)

t, p = stats.ttest_rel(nv, av)
ymax = max(nm + ns, am + as_)
ax.plot([0, 0, 1, 1], [ymax*1.03, ymax*1.06, ymax*1.06, ymax*1.03], 'k-', linewidth=1.5)
ax.text(0.5, ymax*1.07, f'p = {p:.4f} ***', ha='center', fontsize=14, fontweight='bold')

ax.set_ylabel('Average Count per Response', fontsize=14)
ax.set_title('Positive Emotion Words\n(all 200 prompts, 5 runs)', fontsize=16, fontweight='bold')
ax.set_ylim(0, ymax * 1.18)

plt.tight_layout()
plt.savefig('fig1_positive_words.png', dpi=150, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════
# FIGURE 2: Flat Words
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 6))

nv = df_summary['normal_flat_words'].values
av = df_summary['anhedonic_flat_words'].values
nm, am = nv.mean(), av.mean()
ns = nv.std() / np.sqrt(NUM_RUNS)
as_ = av.std() / np.sqrt(NUM_RUNS)

bars = ax.bar(['Normal', 'Anhedonic'], [nm, am],
              yerr=[ns, as_], color=[C_NORMAL, C_ANHEDONIC],
              capsize=12, width=0.55, edgecolor='black', linewidth=0.8)

for bar, val in zip(bars, [nm, am]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.5,
            f'{val:.3f}', ha='center', fontsize=16, fontweight='bold', color='white')

for v in nv:
    ax.plot(0, v, 'o', color='black', alpha=0.5, markersize=7, zorder=5)
for v in av:
    ax.plot(1, v, 'o', color='black', alpha=0.5, markersize=7, zorder=5)

t, p = stats.ttest_rel(av, nv)  # reverse: anhedonic > normal
ymax = max(nm + ns, am + as_)
ax.plot([0, 0, 1, 1], [ymax*1.03, ymax*1.06, ymax*1.06, ymax*1.03], 'k-', linewidth=1.5)
sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
ax.text(0.5, ymax*1.07, f'p = {p:.4f} {sig}', ha='center', fontsize=14, fontweight='bold')

ax.set_ylabel('Average Count per Response', fontsize=14)
ax.set_title('Flat / Neutral Words\n(all 200 prompts, 5 runs)', fontsize=16, fontweight='bold')
ax.set_ylim(0, ymax * 1.18)

plt.tight_layout()
plt.savefig('fig2_flat_words.png', dpi=150, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════
# FIGURE 3: Response Length
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 6))

nv = df_summary['normal_resp_length'].values
av = df_summary['anhedonic_resp_length'].values
nm, am = nv.mean(), av.mean()
ns = nv.std() / np.sqrt(NUM_RUNS)
as_ = av.std() / np.sqrt(NUM_RUNS)

bars = ax.bar(['Normal', 'Anhedonic'], [nm, am],
              yerr=[ns, as_], color=[C_NORMAL, C_ANHEDONIC],
              capsize=12, width=0.55, edgecolor='black', linewidth=0.8)

for bar, val in zip(bars, [nm, am]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.5,
            f'{val:.1f}', ha='center', fontsize=16, fontweight='bold', color='white')

for v in nv:
    ax.plot(0, v, 'o', color='black', alpha=0.5, markersize=7, zorder=5)
for v in av:
    ax.plot(1, v, 'o', color='black', alpha=0.5, markersize=7, zorder=5)

t, p = stats.ttest_rel(nv, av)
ymax = max(nm + ns, am + as_)
ax.plot([0, 0, 1, 1], [ymax*1.01, ymax*1.02, ymax*1.02, ymax*1.01], 'k-', linewidth=1.5)
ax.text(0.5, ymax*1.025, f'p < 0.0001 ***', ha='center', fontsize=14, fontweight='bold')

ax.set_ylabel('Average Words per Response', fontsize=14)
ax.set_title('Response Length\n(all 200 prompts, 5 runs)', fontsize=16, fontweight='bold')
ax.set_ylim(0, ymax * 1.08)

plt.tight_layout()
plt.savefig('fig3_response_length.png', dpi=150, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════
# FIGURE 4: Distribution
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6))

ax.hist(run0['normal_emotion_pos'], bins=range(0, 8), alpha=0.6,
        color=C_NORMAL, label='Normal', edgecolor='black', linewidth=0.8)
ax.hist(run0['anhedonic_emotion_pos'], bins=range(0, 8), alpha=0.6,
        color=C_ANHEDONIC, label='Anhedonic', edgecolor='black', linewidth=0.8)

ax.set_xlabel('Positive Emotion Words per Response', fontsize=14)
ax.set_ylabel('Number of Responses', fontsize=14)
ax.set_title('Distribution of Positive Emotion Words\n(200 prompts)', fontsize=16, fontweight='bold')
ax.legend(fontsize=14)

plt.tight_layout()
plt.savefig('fig4_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════
# FIGURE 5: Scatter
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 7))

ax.scatter(run0['normal_emotion_pos'], run0['anhedonic_emotion_pos'],
           alpha=0.5, color='gray', edgecolors='black', linewidth=0.5, s=70)

max_val = max(run0['normal_emotion_pos'].max(), run0['anhedonic_emotion_pos'].max()) + 1
ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=2, label='Equal')

above = (run0['anhedonic_emotion_pos'] > run0['normal_emotion_pos']).sum()
below = (run0['anhedonic_emotion_pos'] < run0['normal_emotion_pos']).sum()
equal = (run0['anhedonic_emotion_pos'] == run0['normal_emotion_pos']).sum()

ax.text(0.05, 0.95,
        f'Anhedonic > Normal: {above}\nAnhedonic < Normal: {below}\nEqual: {equal}',
        transform=ax.transAxes, fontsize=13, va='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))

ax.set_xlabel('Normal Model', fontsize=15)
ax.set_ylabel('Anhedonic Model', fontsize=15)
ax.set_title('Per-Prompt: Positive Emotion Words\n(below line = anhedonic reduced)',
             fontsize=16, fontweight='bold')
ax.legend(fontsize=13)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('fig5_scatter.png', dpi=150, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════
# FIGURE 6: Consistency
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

runs = np.arange(NUM_RUNS) + 1
nv = df_summary['normal_pos_words'].values
av = df_summary['anhedonic_pos_words'].values

ax.plot(runs, nv, 'o-', color=C_NORMAL, linewidth=3, markersize=12, label='Normal')
ax.plot(runs, av, 's-', color=C_ANHEDONIC, linewidth=3, markersize=12, label='Anhedonic')
ax.fill_between(runs, nv, av, alpha=0.15, color='gray')

ax.set_xlabel('Run', fontsize=15)
ax.set_ylabel('Avg Positive Emotion Words', fontsize=15)
ax.set_title('Effect Consistency Across 5 Runs', fontsize=16, fontweight='bold')
ax.set_xticks(runs)
ax.legend(fontsize=14)

plt.tight_layout()
plt.savefig('fig6_consistency.png', dpi=150, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════
print('\n' + '=' * 50)
print('SUMMARY')
print('=' * 50)
for name, nc, ac in [
    ('Positive emotion words', 'normal_pos_words', 'anhedonic_pos_words'),
    ('Flat words', 'normal_flat_words', 'anhedonic_flat_words'),
    ('Response length', 'normal_resp_length', 'anhedonic_resp_length'),
]:
    nv = df_summary[nc].values
    av = df_summary[ac].values
    t, p = stats.ttest_rel(nv, av)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
    print(f'\n{name}:')
    print(f'  Normal:    {nv.mean():.3f} +/- {nv.std():.3f}')
    print(f'  Anhedonic: {av.mean():.3f} +/- {av.std():.3f}')
    print(f'  p={p:.6f} {sig}')

print('\nSaved: fig1-fig6 .png files')