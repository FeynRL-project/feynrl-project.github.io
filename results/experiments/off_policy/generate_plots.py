"""
Generate reward-vs-step plots for off-policy experiments.

Adapted from framework_comparisons/plot_comparisons.py.
Data format: CSV with a `step` column and one column per run series.

To add a new experiment:
  1. Drop its CSV into data/ (format: step, col_a, col_b, ...).
  2. Add an entry to EXPERIMENTS below.
  3. Run:  python generate_plots.py

Output PNGs land in plots/.
"""

import pathlib

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

HERE = pathlib.Path(__file__).parent

EXPERIMENTS = [
    dict(
        csv="data/temp_0.6_qwen34th.csv",
        out="plots/temp_0.6_qwen34th.png",
        title="Sampling Temperature 0.6",
        subtitle="Qwen3-4B-Thinking-2507  |  Dataset: DeepScaleR  |  Reward: binary math verifier",
        series=[
            dict(col="p3o", label="P3O", color="black", linestyle="-"),
            dict(col="grpo", label="GRPO", color="#0f766e", linestyle="--"),
        ],
    ),
    dict(
        csv="data/temp_1.2_qwen34th.csv",
        out="plots/temp_1.2_qwen34th.png",
        title="Sampling Temperature 1.2",
        subtitle="Qwen3-4B-Thinking-2507  |  Dataset: DeepScaleR  |  Reward: binary math verifier",
        series=[
            dict(col="p3o", label="P3O", color="black", linestyle="-"),
            dict(col="grpo", label="GRPO", color="#0f766e", linestyle="--"),
        ],
    ),
    dict(
        csv="data/fp8_qwen34th.csv",
        out="plots/fp8_qwen34th.png",
        title="BF16 Train + FP8 Rollout",
        subtitle="Qwen3-4B-Thinking-2507  |  Dataset: DeepScaleR  |  max_tokens: 16384",
        series=[
            dict(col="p3o", label="P3O", color="black", linestyle="-"),
            dict(col="grpo", label="GRPO", color="#0f766e", linestyle="--"),
        ],
    ),
]

SMOOTH_WINDOW = 3


def smooth(s: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return s
    return s.rolling(window=window, min_periods=1, center=True).mean()


def plot_experiment(cfg: dict) -> None:
    df = pd.read_csv(HERE / cfg["csv"])

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#cccccc",
            "figure.dpi": 200,
        }
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for s_cfg in cfg["series"]:
        col = s_cfg["col"]
        if col not in df.columns:
            print(f"  Warning: column '{col}' not found in {cfg['csv']}, skipping.")
            continue
        raw = df.set_index("step")[col].dropna()
        s = smooth(raw, SMOOTH_WINDOW)

        ax.plot(
            s.index,
            s.values,
            label=s_cfg["label"],
            color=s_cfg["color"],
            linestyle=s_cfg.get("linestyle", "-"),
            linewidth=2.2,
            marker="o",
            markersize=3.2,
            markeredgewidth=0,
            alpha=0.9,
        )
        if SMOOTH_WINDOW > 1:
            ax.plot(
                raw.index,
                raw.values,
                color=s_cfg["color"],
                linestyle=s_cfg.get("linestyle", "-"),
                linewidth=0.7,
                marker="o",
                markersize=2.4,
                markeredgewidth=0,
                alpha=0.2,
            )

    ax.set_title(cfg["title"], fontsize=14, fontweight="bold", pad=18)
    ax.text(
        0.5,
        1.01,
        cfg["subtitle"],
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#5A5A5A",
    )
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=8))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()

    out = HERE / cfg["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


def main() -> None:
    for cfg in EXPERIMENTS:
        plot_experiment(cfg)


if __name__ == "__main__":
    main()
