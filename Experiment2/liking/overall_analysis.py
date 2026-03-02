import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 12
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

C_NORMAL = '#4A90D9'
C_ANHEDONIC = '#E85D4A'

# Load data
df_raw = pd.read_csv('eval_multirun_raw.csv')
df_summary = pd.read_csv('eval_multirun_summary.csv')
NUM_RUNS = len(df_summary)

print(f'Loaded: {len(df_raw)} total responses across {NUM_RUNS} runs')
print(f'Tasks per run: {len(df_raw) // NUM_RUNS}')

# ══════════════════════════════════════════════════════════════════
# FIGURE 1: Overall response profile
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

panels = [
    ('Positive Emotion Words', 'normal_pos_words', 'anhedonic_pos_words'),
    ('Flat / Neutral Words', 'normal_flat_words', 'anhedonic_flat_words'),
    ('Response Length (words)', 'normal_resp_length', 'anhedonic_resp_length'),
]

for ax, (title, nc, ac) in zip(axes, panels):
    nv = df_summary[nc].values
    av = df_summary[ac].values
    nm, am = nv.mean(), av.mean()
    ns = nv.std() / np.sqrt(NUM_RUNS)
    as_ = av.std() / np.sqrt(NUM_RUNS)

    bars = ax.bar(['Normal', 'Anhedonic'], [nm, am],
                  yerr=[ns, as_], color=[C_NORMAL, C_ANHEDONIC],
                  capsize=10, width=0.5, edgecolor='black', linewidth=0.5)

    # Individual runs
    for v in nv:
        ax.plot(0, v, 'o', color='black', alpha=0.5, markersize=6)
    for v in av:
        ax.plot(1, v, 'o', color='black', alpha=0.5, markersize=6)

    # Stats
    t, p = stats.ttest_rel(nv, av)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
    ymax = max(nm + ns, am + as_)
    ax.text(0.5, ymax * 1.08, f'p={p:.4f} {sig}', ha='center', fontsize=11, fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=13)

plt.suptitle('Overall Model Behavior: Normal vs Anhedonic\n(All 200 prompts averaged, 5 runs)',
             fontsize=15, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('overall_behavior.png', dpi=150, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════
# FIGURE 2: Distribution of emotion words across ALL responses
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Use run 0 as representative
run0 = df_raw[df_raw['run_id'] == 0]

ax = axes[0]
ax.hist(run0['normal_emotion_pos'], bins=range(0, 8), alpha=0.6,
        color=C_NORMAL, label='Normal', edgecolor='black', linewidth=0.5)
ax.hist(run0['anhedonic_emotion_pos'], bins=range(0, 8), alpha=0.6,
        color=C_ANHEDONIC, label='Anhedonic', edgecolor='black', linewidth=0.5)
ax.set_xlabel('Positive Emotion Words per Response')
ax.set_ylabel('Count')
ax.set_title('Distribution: Positive Emotion Words', fontweight='bold')
ax.legend()

ax = axes[1]
ax.hist(run0['normal_response_length'], bins=30, alpha=0.6,
        color=C_NORMAL, label='Normal', edgecolor='black', linewidth=0.5)
ax.hist(run0['anhedonic_response_length'], bins=30, alpha=0.6,
        color=C_ANHEDONIC, label='Anhedonic', edgecolor='black', linewidth=0.5)
ax.set_xlabel('Response Length (words)')
ax.set_ylabel('Count')
ax.set_title('Distribution: Response Length', fontweight='bold')
ax.legend()

plt.suptitle('Response Distributions (All 200 prompts)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('overall_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════
# FIGURE 3: Per-response paired comparison
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.scatter(run0['normal_emotion_pos'], run0['anhedonic_emotion_pos'],
           alpha=0.5, color='gray', edgecolors='black', linewidth=0.3, s=50)
max_val = max(run0['normal_emotion_pos'].max(), run0['anhedonic_emotion_pos'].max()) + 1
ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Equal')
ax.set_xlabel('Normal: Positive Emotion Words', fontsize=12)
ax.set_ylabel('Anhedonic: Positive Emotion Words', fontsize=12)
ax.set_title('Per-Prompt Comparison: Emotion Words', fontweight='bold')
ax.legend()

# Count how many are above/below diagonal
above = (run0['anhedonic_emotion_pos'] > run0['normal_emotion_pos']).sum()
below = (run0['anhedonic_emotion_pos'] < run0['normal_emotion_pos']).sum()
equal = (run0['anhedonic_emotion_pos'] == run0['normal_emotion_pos']).sum()
ax.text(0.05, 0.95, f'Anhedonic higher: {above}\nAnhedonic lower: {below}\nEqual: {equal}',
        transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax = axes[1]
ax.scatter(run0['normal_response_length'], run0['anhedonic_response_length'],
           alpha=0.5, color='gray', edgecolors='black', linewidth=0.3, s=50)
max_val = max(run0['normal_response_length'].max(), run0['anhedonic_response_length'].max()) + 5
ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Equal')
ax.set_xlabel('Normal: Response Length (words)', fontsize=12)
ax.set_ylabel('Anhedonic: Response Length (words)', fontsize=12)
ax.set_title('Per-Prompt Comparison: Response Length', fontweight='bold')
ax.legend()

above = (run0['anhedonic_response_length'] > run0['normal_response_length']).sum()
below = (run0['anhedonic_response_length'] < run0['normal_response_length']).sum()
equal = (run0['anhedonic_response_length'] == run0['normal_response_length']).sum()
ax.text(0.05, 0.95, f'Anhedonic longer: {above}\nAnhedonic shorter: {below}\nEqual: {equal}',
        transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Per-Prompt Paired Comparison (200 prompts)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('overall_paired.png', dpi=150, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════
# FIGURE 4: Run-by-run consistency
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

runs = np.arange(NUM_RUNS) + 1
nv = df_summary['normal_pos_words'].values
av = df_summary['anhedonic_pos_words'].values

ax.plot(runs, nv, 'o-', color=C_NORMAL, linewidth=2.5, markersize=10, label='Normal')
ax.plot(runs, av, 's-', color=C_ANHEDONIC, linewidth=2.5, markersize=10, label='Anhedonic')
ax.fill_between(runs, nv, av, alpha=0.2, color='gray')

ax.set_xlabel('Run', fontsize=13)
ax.set_ylabel('Avg Positive Emotion Words', fontsize=13)
ax.set_title('Effect Consistency Across 5 Independent Runs', fontsize=14, fontweight='bold')
ax.set_xticks(runs)
ax.legend(fontsize=12)

plt.tight_layout()
plt.savefig('overall_consistency.png', dpi=150, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════
# PRINT SUMMARY
# ══════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('OVERALL SUMMARY (all 200 prompts, 5 runs)')
print('=' * 60)

for name, nc, ac in panels:
    nv = df_summary[nc].values
    av = df_summary[ac].values
    t, p = stats.ttest_rel(nv, av)
    pct = (1 - av.mean() / nv.mean()) * 100
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
    print(f'\n{name}:')
    print(f'  Normal:    {nv.mean():.3f} +/- {nv.std():.3f}')
    print(f'  Anhedonic: {av.mean():.3f} +/- {av.std():.3f}')
    print(f'  Change:    {pct:+.1f}%')
    print(f'  p={p:.6f} {sig}')

print('\n' + '=' * 60)
print('Saved: overall_behavior.png, overall_distributions.png,')
print('       overall_paired.png, overall_consistency.png')
print('=' * 60)
