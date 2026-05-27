"""
Generate reward-vs-time plots for math framework comparison experiments.

Adapted from framework_comparisons/plot_comparisons.py.
Data format: CSV with columns run_id, key, value, step, timestamp (MLflow export).

To add a new framework or model:
  1. Drop its CSV into data/<model>/ (same format).
  2. Add/update an entry in EXPERIMENTS below.
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
        out="plots/qwen3_4b_thinking_2507.png",
        title="Reward vs Time — Framework Comparison",
        subtitle="Model: Qwen3-4B-Thinking-2507  |  Dataset: DeepScaleR  |  Reward: binary math_reward",
        max_hours=230 / 60,
        frameworks=[
            dict(
                file="data/qwen3-4b-thinking-2507/feynrl_sync.csv",
                reward_key="rollout/avg_reward",
                label="FeynRL (Sync)",
                color="black",
                linestyle="-",
            ),
            dict(
                file="data/qwen3-4b-thinking-2507/feynrl_async.csv",
                reward_key="rollout/avg_reward",
                label="FeynRL (Overlap)",
                color="#0f766e",
                linestyle="--",
            ),
        ],
    ),
    dict(
        out="plots/qwen2_5_1b_instruct.png",
        title="Reward vs Time — Framework Comparison",
        subtitle="Model: Qwen2.5-1.5B-Instruct  |  Dataset: GSM8K  |  Reward: binary math_reward",
        max_hours=1,
        frameworks=[
            dict(
                file="data/qwen2.5-1.5b-instruct/FeynRL_sync.csv",
                reward_key="rollout/avg_reward",
                label="FeynRL (Sync)",
                color="black",
                linestyle="-",
            ),
            dict(
                file="data/qwen2.5-1.5b-instruct/FeynRL_async.csv",
                reward_key="rollout/avg_reward",
                label="FeynRL (Overlap)",
                color="#0f766e",
                linestyle="--",
            ),
        ],
    ),
]

SMOOTH_WINDOW = 5


def load_runs(cfg: dict) -> list[pd.Series]:
    path = HERE / cfg["file"]
    df = pd.read_csv(path)
    mask = df["key"] == cfg["reward_key"]
    if not mask.any():
        available = df["key"].unique().tolist()
        raise ValueError(
            f"[{cfg['file']}] reward_key '{cfg['reward_key']}' not found. "
            f"Available keys: {available}"
        )
    df = df.loc[mask].copy()
    runs = []
    for _, group in df.groupby("run_id", sort=False):
        group = group.sort_values("timestamp")
        elapsed_min = (group["timestamp"] - group["timestamp"].iloc[0]) / 60_000
        series = pd.Series(group["value"].values, index=elapsed_min.values)
        runs.append(series)
    return runs


def smooth(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1, center=True).mean()


def truncate_at_max_hours(series: pd.Series, max_hours: float | None) -> pd.Series:
    if max_hours is None or series.empty:
        return series
    cutoff_min = max_hours * 60
    if series.index.max() <= cutoff_min:
        return series
    before = series[series.index <= cutoff_min]
    after = series[series.index > cutoff_min]
    if after.empty:
        return before
    return pd.concat([before, pd.Series([after.iloc[0]], index=[cutoff_min])])


def plot_experiment(cfg: dict, window: int) -> None:
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
            "figure.dpi": 400,
        }
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for fw in cfg["frameworks"]:
        runs = load_runs(fw)
        color = fw["color"]
        ls = fw.get("linestyle", "-")
        alpha_scale = fw.get("alpha_scale", 1.0)
        line_width = 2.2
        raw_line_width = 0.8
        marker_size = 3.2

        for i, raw in enumerate(runs):
            raw = truncate_at_max_hours(raw, cfg.get("max_hours"))
            s = smooth(raw, window)
            label = fw["label"] if i == 0 else "_nolegend_"
            ax.plot(
                s.index, s.values,
                label=label, color=color, linestyle=ls,
                linewidth=line_width, marker="o", markersize=marker_size,
                markeredgewidth=0, alpha=(0.9 if len(runs) == 1 else 0.75) * alpha_scale,
            )
            if window > 1:
                ax.plot(
                    raw.index, raw.values,
                    color=color, linestyle=ls,
                    linewidth=raw_line_width, marker="o",
                    markersize=max(marker_size - 0.8, 1.5),
                    markeredgewidth=0, alpha=0.2 * alpha_scale,
                )

    ax.set_xlabel("Elapsed Time (minutes)", fontsize=13)
    ax.set_ylabel("Reward", fontsize=13)
    ax.set_title(cfg["title"], fontsize=14, fontweight="bold", pad=18)
    ax.text(
        0.5, 1.01, cfg["subtitle"],
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=9.5, color="#5A5A5A",
    )
    if cfg.get("max_hours") is not None:
        ax.set_xlim(right=cfg["max_hours"] * 60)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.legend(loc="lower right", fontsize=11)
    fig.tight_layout()

    out = HERE / cfg["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


def main() -> None:
    for cfg in EXPERIMENTS:
        plot_experiment(cfg, window=SMOOTH_WINDOW)


if __name__ == "__main__":
    main()
