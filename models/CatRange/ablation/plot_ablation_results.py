#!/usr/bin/env python3
"""Plot CatRange feature ablation results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODE_ORDER = ["full", "sequence", "substrate"]
MODE_LABELS = {
    "full": "Full CatRange",
    "sequence": "Sequence only",
    "substrate": "Substrate only",
}
MODE_COLORS = {
    "full": "#000000",
    "sequence": "#0072B2",
    "substrate": "#E69F00",
}
MODE_HATCHES = {
    "full": "",
    "sequence": "///",
    "substrate": "\\\\\\",
}
MODE_MARKERS = {
    "full": "o",
    "sequence": "s",
    "substrate": "^",
}
MODE_LINESTYLES = {
    "full": "-",
    "sequence": "--",
    "substrate": ":",
}
METRICS = ["accuracy", "e_accuracy", "mcc", "f1"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "e_accuracy": "e-Accuracy",
    "mcc": "MCC",
    "f1": "F1",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Create CatRange ablation figures.")
    parser.add_argument("--results-dir", default="runs/kcat_esmc_ablation")
    parser.add_argument("--parameter", choices=["kcat", "km"], default="kcat")
    parser.add_argument("--include-fold-figure", action="store_true", help="Also plot fold-wise accuracy.")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 2.0,
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.labelcolor": "black",
            "text.color": "black",
            "legend.frameon": False,
        }
    )


def journal_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_linewidth(2.0)
    ax.spines["bottom"].set_linewidth(2.0)
    ax.tick_params(width=1.0, length=3, colors="black")
    ax.grid(False)
    return ax


def ordered_modes(values):
    seen = set(values)
    ordered = [mode for mode in MODE_ORDER if mode in seen]
    ordered.extend(sorted(seen.difference(ordered)))
    return ordered


def parameter_label(parameter: str) -> str:
    return "k$_{cat}$" if parameter == "kcat" else "K$_M$"


def plot_metric_summary(summary_df: pd.DataFrame, figures_dir: Path, dpi: int, parameter: str):
    modes = ordered_modes(summary_df["feature_mode"].dropna().unique())
    x = np.arange(len(METRICS), dtype=float)
    width = min(0.24, 0.82 / max(len(modes), 1))

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for idx, mode in enumerate(modes):
        mode_df = summary_df[summary_df["feature_mode"] == mode].set_index("metric")
        means = [mode_df.loc[metric, "mean"] if metric in mode_df.index else np.nan for metric in METRICS]
        stds = [mode_df.loc[metric, "std"] if metric in mode_df.index else 0.0 for metric in METRICS]
        offsets = x + (idx - (len(modes) - 1) / 2.0) * width
        bars = ax.bar(
            offsets,
            means,
            width=width * 0.92,
            yerr=stds,
            capsize=3,
            color=MODE_COLORS.get(mode, "#52606d"),
            edgecolor="black",
            hatch=MODE_HATCHES.get(mode, ""),
            ecolor="black",
            linewidth=0.8,
            label=MODE_LABELS.get(mode, mode),
        )
        for bar, mean, std in zip(bars, means, stds):
            if pd.isna(mean):
                continue
            std = 0.0 if pd.isna(std) else float(std)
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                float(mean) + std + 0.025 + 0.005 * (idx % 2),
                f"{float(mean):.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                clip_on=False,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], rotation=18, ha="right")
    ax.set_ylabel(f"{parameter_label(parameter)} score")
    ax.set_ylim(0.0, 1.08)
    journal_axes(ax)
    ax.legend(loc="upper center", ncol=len(modes), bbox_to_anchor=(0.5, 1.08), fontsize=9)
    fig.tight_layout()
    fig.savefig(figures_dir / f"catrange_{parameter}_feature_ablation_metrics.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_fold_lines(fold_df: pd.DataFrame, figures_dir: Path, dpi: int, parameter: str):
    modes = ordered_modes(fold_df["feature_mode"].dropna().unique())
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for mode in modes:
        mode_df = fold_df[fold_df["feature_mode"] == mode].sort_values("fold")
        ax.plot(
            mode_df["fold"],
            mode_df["accuracy"],
            marker=MODE_MARKERS.get(mode, "o"),
            linestyle=MODE_LINESTYLES.get(mode, "-"),
            linewidth=1.8,
            markersize=4.8,
            color=MODE_COLORS.get(mode, "#52606d"),
            label=MODE_LABELS.get(mode, mode),
        )
    ax.set_xlabel("Fold")
    ax.set_ylabel(f"{parameter_label(parameter)} accuracy")
    ax.set_xticks(sorted(fold_df["fold"].dropna().unique()))
    ax.set_ylim(0.0, 1.0)
    journal_axes(ax)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(figures_dir / f"catrange_{parameter}_feature_ablation_fold_accuracy.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    setup_style()
    results_dir = Path(args.results_dir).resolve()
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for stale in figures_dir.glob("*.png"):
        stale.unlink()

    summary_df = pd.read_csv(results_dir / "all_summary_metrics.csv")
    fold_df = pd.read_csv(results_dir / "all_fold_metrics.csv")
    plot_metric_summary(summary_df, figures_dir, args.dpi, args.parameter)
    if args.include_fold_figure:
        plot_fold_lines(fold_df, figures_dir, args.dpi, args.parameter)
    print(f"Wrote ablation figures to {figures_dir}")


if __name__ == "__main__":
    main()
