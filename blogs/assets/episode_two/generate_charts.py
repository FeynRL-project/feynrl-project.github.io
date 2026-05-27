#!/usr/bin/env python3
"""Generate all charts for Episode 02 blog post.

Uses the paper's CSV data and produces charts matching the FeynRL site aesthetic:
- Clean, minimal style
- Monospace font
- Teal accent (#0f766e) for P3O
- Gray/coral for GRPO variants
- Thin 1px border, #fafafa background
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import csv
import os

# ── Style constants ──────────────────────────────────────────────────────────
TEAL = '#0f766e'
TEAL_LIGHT = '#99d5cf'
CORAL = '#dc2626'
CORAL_LIGHT = '#fca5a5'
GRAY_DARK = '#4b5563'
GRAY_MED = '#9ca3af'
GRAY_LIGHT = '#d1d5db'
AMBER = '#d97706'
BG = '#fafafa'
FG = '#111111'
MUTED = '#5f5f5f'
LINE_COLOR = '#d8d8d8'
FONT_MONO = 'monospace'

DATA_DIR = '/ceph/workspace/murdock/resurrection/paper/plot_data'
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

def setup_style():
    """Apply global matplotlib style."""
    plt.rcParams.update({
        'font.family': 'monospace',
        'font.size': 11,
        'axes.facecolor': BG,
        'figure.facecolor': '#ffffff',
        'axes.edgecolor': LINE_COLOR,
        'axes.linewidth': 1,
        'axes.grid': False,
        'xtick.color': MUTED,
        'ytick.color': MUTED,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.frameon': False,
        'legend.fontsize': 9,
    })


def read_csv(filename):
    """Read CSV and return dict of columns."""
    path = os.path.join(DATA_DIR, filename)
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        data = {}
        for row in reader:
            for key, val in row.items():
                key = key.strip()
                if key not in data:
                    data[key] = []
                try:
                    data[key].append(float(val) if val.strip() else None)
                except ValueError:
                    data[key].append(None)
    return data


def clean_series(steps, values):
    """Remove None values from paired lists."""
    s, v = [], []
    for si, vi in zip(steps, values):
        if vi is not None:
            s.append(si)
            v.append(vi)
    return np.array(s), np.array(v)


# ── Chart 1: Clip Sensitivity ────────────────────────────────────────────────

def plot_clip_sensitivity():
    data = read_csv('clipping_ratio_plot_qwen34.csv')
    steps = np.array(data['step'])

    fig, ax = plt.subplots(figsize=(9, 4.5))

    # GRPO band (min/max of clip variants)
    c02 = np.array(data['clip02'])
    c04 = np.array(data['clip04'])
    c06 = np.array(data['clip06'])
    grpo_min = np.minimum(np.minimum(c02, c04), c06)
    grpo_max = np.maximum(np.maximum(c02, c04), c06)
    grpo_mean = np.array(data['grpo_mean'])

    ax.fill_between(steps, grpo_min, grpo_max, alpha=0.15, color=CORAL, linewidth=0)
    ax.plot(steps, c02, color=CORAL_LIGHT, linewidth=1.0, linestyle='--', alpha=0.7, label='GRPO ε=0.2')
    ax.plot(steps, c04, color=GRAY_MED, linewidth=1.0, linestyle='--', alpha=0.7, label='GRPO ε=0.4')
    ax.plot(steps, c06, color=AMBER, linewidth=1.0, linestyle='--', alpha=0.7, label='GRPO ε=0.6')

    # P3O
    s_p3o, v_p3o = clean_series(steps, data['p3o_reward'])
    ax.plot(s_p3o, v_p3o, color=TEAL, linewidth=2.2, label='P3O (no clip)', zorder=5)

    ax.set_xlabel('Training Step', color=MUTED, fontsize=10)
    ax.set_ylabel('Average Reward', color=MUTED, fontsize=10)
    ax.set_ylim(0, 0.65)
    ax.legend(loc='lower right', ncol=2)

    # Add annotation for the spread — placed inside the red filled region
    ax.annotate('Clip sensitivity\nspread', xy=(32, 0.36), fontsize=8, color=CORAL,
                ha='center', va='center', style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'clip_sensitivity_reward.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print('✓ clip_sensitivity_reward.png')


# ── Chart 2: Temperature Robustness ──────────────────────────────────────────

def plot_temp_robustness():
    data = read_csv('temp_1.2_plot_qwen34.csv')
    steps_raw = data['step']

    fig, ax = plt.subplots(figsize=(9, 4.5))

    # GRPO
    s_grpo, v_grpo = clean_series(steps_raw, data['grpo'])
    ax.plot(s_grpo, v_grpo, color=CORAL, linewidth=2.0, label='GRPO (T=1.2)')

    # P3O
    s_p3o, v_p3o = clean_series(steps_raw, data['p3o'])
    ax.plot(s_p3o, v_p3o, color=TEAL, linewidth=2.2, label='P3O (T=1.2)', zorder=5)

    # Collapse annotation — arrow points to ~step 18 where divergence begins
    ax.annotate('GRPO collapses',
                xy=(19, 0.63), xytext=(15, 0.5),
                arrowprops=dict(arrowstyle='->', color=CORAL, lw=1.2),
                fontsize=9, color=CORAL, ha='center')

    ax.set_xlabel('Training Step', color=MUTED, fontsize=10)
    ax.set_ylabel('Average Reward', color=MUTED, fontsize=10)
    ax.set_ylim(0, 0.8)
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'temp_robustness_reward.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print('✓ temp_robustness_reward.png')


# ── Chart 3: FP8 Reward Curves (HERO) ───────────────────────────────────────

def plot_fp8_reward():
    data = read_csv('fp8_plot_qwen34.csv')
    steps = np.array(data['step'])

    fig, ax = plt.subplots(figsize=(10, 5))

    # GRPO
    v_grpo = np.array(data['grpo'])
    ax.plot(steps, v_grpo, color=CORAL, linewidth=2.2, label='GRPO (BF16 train + FP8 rollout)')

    # P3O
    v_p3o = np.array(data['p3o'])
    ax.plot(steps, v_p3o, color=TEAL, linewidth=2.5, label='P3O (BF16 train + FP8 rollout)', zorder=5)

    # Shade the collapse region
    collapse_start = 16
    ax.axvspan(collapse_start, 35, alpha=0.04, color=CORAL, zorder=0)

    ax.set_xlabel('Training Step', color=MUTED, fontsize=11)
    ax.set_ylabel('Average Reward', color=MUTED, fontsize=11)
    ax.set_ylim(-0.02, 0.65)
    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fp8_reward_curves.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print('✓ fp8_reward_curves.png')


# ── Chart 4: FP8 Pass@k ─────────────────────────────────────────────────────

def plot_fp8_pass_at_k():
    data = read_csv('pass_at_k_fp8_avg.csv')
    k = np.array(data['k'])

    fig, ax = plt.subplots(figsize=(9, 4.5))

    # Baseline
    s, v = clean_series(k, data['baseline_16k'])
    ax.plot(s, v, color=GRAY_DARK, linewidth=1.5, linestyle='--', label='Baseline (16k)')

    # GRPO iter 15
    s, v = clean_series(k, data['fp8_grpo_iter15'])
    ax.plot(s, v, color=CORAL_LIGHT, linewidth=1.5, label='GRPO iter 15')

    # GRPO iter 30
    s, v = clean_series(k, data['fp8_grpo_iter30'])
    ax.plot(s, v, color=CORAL, linewidth=2.0, linestyle=':', label='GRPO iter 30 (collapsed)')

    # P3O iter 15
    s, v = clean_series(k, data['fp8_p3o_iter15'])
    ax.plot(s, v, color=TEAL_LIGHT, linewidth=1.5, label='P3O iter 15')

    # P3O iter 30
    s, v = clean_series(k, data['fp8_p3o_iter30'])
    ax.plot(s, v, color=TEAL, linewidth=2.2, label='P3O iter 30', zorder=5)

    ax.set_xlabel('k (pass@k)', color=MUTED, fontsize=10)
    ax.set_ylabel('Average Pass Rate', color=MUTED, fontsize=10)
    ax.set_ylim(0, 0.6)
    ax.legend(loc='upper left', fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fp8_pass_at_k.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print('✓ fp8_pass_at_k.png')


# ── Chart 5: Clip Pass@k ────────────────────────────────────────────────────

def plot_clip_pass_at_k():
    data = read_csv('pass_at_k_clip_avg.csv')
    k = np.array(data['k'])

    fig, ax = plt.subplots(figsize=(9, 4.5))

    # Baseline
    s, v = clean_series(k, data['baseline_4k'])
    ax.plot(s, v, color=GRAY_DARK, linewidth=1.5, linestyle='--', label='Baseline (4k)')

    # GRPO mean ± std
    grpo_mean = np.array(data['grpo_mean'])
    grpo_std = np.array(data['grpo_std'])
    ax.fill_between(k, grpo_mean - grpo_std, grpo_mean + grpo_std,
                     alpha=0.12, color=CORAL, linewidth=0)
    ax.plot(k, grpo_mean, color=CORAL, linewidth=1.8, label='GRPO (mean ± std over ε)')

    # P3O
    s, v = clean_series(k, data['p3o'])
    ax.plot(s, v, color=TEAL, linewidth=2.2, label='P3O (no clip)', zorder=5)

    ax.set_xlabel('k (pass@k)', color=MUTED, fontsize=10)
    ax.set_ylabel('Average Pass Rate', color=MUTED, fontsize=10)
    ax.set_ylim(0, 0.5)
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'clip_pass_at_k.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print('✓ clip_pass_at_k.png')


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    setup_style()
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_clip_sensitivity()
    plot_temp_robustness()
    plot_fp8_reward()
    plot_fp8_pass_at_k()
    plot_clip_pass_at_k()
    print('\nAll charts generated in:', OUT_DIR)
