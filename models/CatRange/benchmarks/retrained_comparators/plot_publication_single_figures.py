#!/usr/bin/env python3
"""Create single-panel publication-style figures for the benchmark suite."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter

from benchmark_utils import slugify


MODEL_ORDER = ["RealKcat", "CatRange", "CatPred", "DLKcat", "UniKP", "EITLEM-Kinetics"]
MODEL_DISPLAY_NAMES = {
    "RealKcat": "CatRange",
    "CatRange": "CatRange",
}
MODEL_STYLE_ALIASES = {
    "CatRange": "RealKcat",
}
MODEL_COLORS = {
    "RealKcat": "#0072B2",
    "CatPred": "#F28E00",
    "DLKcat": "#009E73",
    "UniKP": "#E7298A",
    "EITLEM-Kinetics": "#7B61FF",
}
MODEL_MARKERS = {
    "RealKcat": "o",
    "CatPred": "s",
    "DLKcat": "^",
    "UniKP": "D",
    "EITLEM-Kinetics": "P",
}
METRIC_ORDER = ["accuracy", "e_accuracy", "precision", "recall", "f1", "mcc"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "e_accuracy": "e-Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "mcc": "MCC",
}
# Tiny width nudge avoids a 1-pixel PNG floor from binary float rounding.
MANUSCRIPT_SMALL_FIGSIZE = (4.000001, 2.0)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot publication-ready benchmark figures.")
    parser.add_argument("--suite-dir", required=True, help="Benchmark suite directory with suite_results.")
    parser.add_argument("--dpi", type=int, default=600, help="PNG DPI.")
    return parser.parse_args()


def _setup_style():
    font_family = "Arial"
    try:
        font_manager.findfont(font_family, fallback_to_default=False)
    except ValueError:
        font_family = "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.sans-serif": ["Arial", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 2.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.labelcolor": "black",
            "text.color": "black",
            "legend.frameon": False,
        }
    )


def _journal_axes(ax):
    """Lightweight axis styling for journal figures."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_linewidth(2.0)
    ax.spines["bottom"].set_linewidth(2.0)
    ax.tick_params(width=1.0, length=3, colors="black")
    ax.grid(False)
    return ax


def _journal_right_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(True)
    ax.spines["right"].set_color("black")
    ax.spines["right"].set_linewidth(2.0)
    ax.tick_params(width=1.0, length=3, colors="black")
    ax.grid(False)
    return ax


def _save_figure(fig, out_prefix: Path, dpi: int, *, exact_size: bool = False):
    save_kwargs = {"dpi": dpi, "facecolor": "white"}
    if not exact_size:
        save_kwargs["bbox_inches"] = "tight"
    fig.savefig(f"{out_prefix}.png", **save_kwargs)
    plt.close(fig)


def _ordered_models(values):
    seen = set(values)
    ordered = [name for name in MODEL_ORDER if name in seen]
    ordered.extend(sorted(seen.difference(ordered)))
    return ordered


def _model_display_name(model_name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(str(model_name), str(model_name))


def _model_style_key(model_name: str) -> str:
    return MODEL_STYLE_ALIASES.get(str(model_name), str(model_name))


def _model_color(model_name: str) -> str:
    return MODEL_COLORS.get(_model_style_key(model_name), "#52606d")


def _model_marker(model_name: str) -> str:
    return MODEL_MARKERS.get(_model_style_key(model_name), "o")


def _model_slug(model_name: str) -> str:
    return slugify(_model_display_name(model_name))


def _seqid_ticklabels(thresholds):
    return [f"\u2264 {int(value)}%" for value in thresholds]


def _metric_limits(values, *, lower_bound=0.0, upper_bound=1.0, pad_fraction=0.18, min_span=0.06):
    finite = np.asarray([value for value in values if pd.notna(value)], dtype=float)
    if finite.size == 0:
        return lower_bound, upper_bound
    low = float(finite.min())
    high = float(finite.max())
    span = max(high - low, min_span)
    pad = max(span * pad_fraction, 0.01)
    low = max(lower_bound, low - pad)
    high = min(upper_bound, high + pad)
    if high - low < min_span:
        center = (high + low) / 2.0
        half = min_span / 2.0
        low = max(lower_bound, center - half)
        high = min(upper_bound, center + half)
    return low, high


def _annotate_seqid_counts(
    ax,
    positions,
    counts,
    anchor_values=None,
    base_pad=0.026,
    stagger_pad=0.011,
    fontsize=8,
):
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    if anchor_values is None:
        anchor_values = [ymax - span * 0.04] * len(positions)
    label_ys = [
        float(anchor) + span * (base_pad + stagger_pad * ((idx % 3) - 1))
        for idx, anchor in enumerate(anchor_values)
    ]
    needed_high = max(label_ys, default=ymax) + span * 0.075
    if needed_high > ymax:
        ax.set_ylim(ymin, needed_high)
    for x_pos, count, y_pos in zip(positions, counts, label_ys):
        ax.text(
            x_pos,
            y_pos,
            f"(n={int(count)})",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="#243b53",
            clip_on=False,
        )


def _plot_metric_bars_from_mean_std(mean_std_df: pd.DataFrame, out_prefix: Path, dpi: int):
    models = _ordered_models(mean_std_df["model_name"].unique())
    if not models:
        return
    x = np.arange(len(METRIC_ORDER), dtype=float)
    width = min(0.24, 0.82 / max(len(models), 1))

    fig, ax = plt.subplots(figsize=MANUSCRIPT_SMALL_FIGSIZE)
    for idx, model_name in enumerate(models):
        model_df = mean_std_df[mean_std_df["model_name"] == model_name].set_index("metric")
        means = [model_df.loc[metric, "mean"] if metric in model_df.index else np.nan for metric in METRIC_ORDER]
        stds = [model_df.loc[metric, "std"] if metric in model_df.index else 0.0 for metric in METRIC_ORDER]
        offsets = x + (idx - (len(models) - 1) / 2.0) * width
        bars = ax.bar(
            offsets,
            means,
            width=width * 0.92,
            color=_model_color(model_name),
            label=_model_display_name(model_name),
            yerr=stds,
            ecolor="#102a43",
            capsize=1.5,
            linewidth=0,
            error_kw={"elinewidth": 0.55, "capthick": 0.55},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRIC_ORDER], rotation=18, ha="right", fontsize=6)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score", fontsize=7, labelpad=1)
    _journal_axes(ax)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linewidth=0.35, color="#d9e2ec")
    ax.tick_params(axis="both", which="major", labelsize=6, length=2.2, width=0.8, pad=1.0)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.legend(
        ncol=len(models),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        fontsize=5.5,
        handlelength=0.9,
        handletextpad=0.25,
        columnspacing=0.55,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.105, right=0.995, top=0.76, bottom=0.29)
    _save_figure(fig, out_prefix, dpi, exact_size=True)


def _plot_metric_bars_from_summary(summary_df: pd.DataFrame, out_prefix: Path, dpi: int):
    models = _ordered_models(summary_df["model_name"].unique())
    if not models:
        return
    x = np.arange(len(METRIC_ORDER), dtype=float)
    width = min(0.24, 0.82 / max(len(models), 1))

    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    for idx, model_name in enumerate(models):
        row = summary_df[summary_df["model_name"] == model_name].iloc[0]
        values = [row[metric] for metric in METRIC_ORDER]
        offsets = x + (idx - (len(models) - 1) / 2.0) * width
        bars = ax.bar(
            offsets,
            values,
            width=width * 0.92,
            color=_model_color(model_name),
            label=_model_display_name(model_name),
            linewidth=0,
        )
        for bar, val in zip(bars, values):
            if pd.isna(val):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                float(val) + 0.02,
                f"{float(val):.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRIC_ORDER], rotation=18, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    _journal_axes(ax)
    ax.set_axisbelow(True)
    ax.legend(ncol=len(models), loc="upper center")
    _save_figure(fig, out_prefix, dpi)


def _plot_seqid_metric(seqid_df: pd.DataFrame, metric: str, out_prefix: Path, dpi: int):
    models = _ordered_models(seqid_df["model_name"].unique())
    if not models:
        return
    thresholds = sorted(seqid_df["seqid_threshold"].dropna().astype(int).unique())
    positions = np.arange(len(thresholds), dtype=float)
    truth_counts = [
        int(seqid_df.loc[seqid_df["seqid_threshold"] == threshold, "truth_rows"].iloc[0])
        for threshold in thresholds
    ]

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    metric_values = []

    for model_name in models:
        model_df = seqid_df[seqid_df["model_name"] == model_name].sort_values("seqid_threshold")
        x_positions = np.arange(len(model_df), dtype=float)
        metric_values.extend(model_df[metric].dropna().tolist())
        ax.plot(
            x_positions,
            model_df[metric],
            marker="o",
            linewidth=1.8,
            markersize=4.8,
            color=_model_color(model_name),
            label=_model_display_name(model_name),
        )

    ax.set_xlabel("Sequence identity threshold")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_xticks(positions)
    ax.set_xticklabels(_seqid_ticklabels(thresholds))
    if metric == "mcc":
        ax.set_ylim(*_metric_limits(metric_values, lower_bound=-1.0, upper_bound=1.0, min_span=0.12))
    else:
        ax.set_ylim(*_metric_limits(metric_values, lower_bound=0.0, upper_bound=1.0))
    _journal_axes(ax)
    count_anchors = [
        float(seqid_df.loc[seqid_df["seqid_threshold"] == threshold, metric].max())
        for threshold in thresholds
    ]
    _annotate_seqid_counts(ax, positions, truth_counts, count_anchors)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    _save_figure(fig, out_prefix, dpi)


def _plot_seqid_dual_axis_per_model(seqid_df: pd.DataFrame, model_name: str, out_prefix: Path, dpi: int):
    model_df = seqid_df[seqid_df["model_name"] == model_name].sort_values("seqid_threshold").copy()
    if model_df.empty:
        return

    thresholds = model_df["seqid_threshold"].astype(int).tolist()
    positions = np.arange(len(thresholds), dtype=float)
    accuracy = model_df["accuracy"].to_numpy(dtype=float)
    e_accuracy = model_df["e_accuracy"].to_numpy(dtype=float)
    truth_counts = model_df["truth_rows"].to_numpy(dtype=int)

    fig, ax_left = plt.subplots(figsize=(4.8, 3.4))
    ax_right = ax_left.twinx()

    acc_color = "#2b6cb0"
    eacc_color = "#dd6b20"

    line_left = ax_left.plot(
        positions,
        accuracy,
        color=acc_color,
        marker="o",
        linewidth=1.7,
        markersize=4.0,
        label="Accuracy",
    )
    line_right = ax_right.plot(
        positions,
        e_accuracy,
        color=eacc_color,
        marker="s",
        linestyle=(0, (4.0, 2.4)),
        linewidth=1.7,
        markersize=3.8,
        label="e-Accuracy",
    )

    ax_left.set_xticks(positions)
    ax_left.set_xticklabels(_seqid_ticklabels(thresholds), fontsize=8)
    ax_left.set_xlabel("Sequence-identity threshold", fontsize=9)
    ax_left.set_ylabel("Accuracy [%]", fontsize=9)
    ax_right.set_ylabel("e-Accuracy\n(within 1 bin)", fontsize=9)

    left_low, left_high = _metric_limits(accuracy, lower_bound=0.0, upper_bound=1.0, pad_fraction=0.14, min_span=0.05)
    right_low, right_high = _metric_limits(e_accuracy, lower_bound=0.0, upper_bound=1.0, pad_fraction=0.14, min_span=0.05)
    ax_left.set_ylim(left_low, left_high)
    ax_right.set_ylim(right_low, right_high)

    _journal_axes(ax_left)
    _journal_right_axis(ax_right)

    _annotate_seqid_counts(
        ax_left,
        positions,
        truth_counts,
        accuracy,
        base_pad=0.028,
        stagger_pad=0.010,
    )

    lines = line_left + line_right
    labels = [line.get_label() for line in lines]
    ax_left.legend(lines, labels, loc="lower right", fontsize=8, handlelength=2.0)
    fig.tight_layout()
    _save_figure(fig, out_prefix, dpi)


def _plot_seqid_dual_axis_compare(seqid_df: pd.DataFrame, out_prefix: Path, dpi: int):
    models = _ordered_models(seqid_df["model_name"].dropna().astype(str).unique())
    if not models:
        return

    thresholds = sorted(seqid_df["seqid_threshold"].dropna().astype(int).unique())
    positions = np.arange(len(thresholds), dtype=float)
    truth_counts = [
        int(seqid_df.loc[seqid_df["seqid_threshold"] == threshold, "truth_rows"].iloc[0])
        for threshold in thresholds
    ]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=MANUSCRIPT_SMALL_FIGSIZE,
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12},
    )
    width = min(0.15, 0.78 / max(len(models), 1))

    for ax, metric, label in zip(axes, ["accuracy", "e_accuracy"], ["Accuracy", "e-Accuracy"]):
        metric_values = []
        for idx, model_name in enumerate(models):
            model_df = seqid_df[seqid_df["model_name"] == model_name].sort_values("seqid_threshold")
            vals = model_df[metric].to_numpy(dtype=float)
            metric_values.extend(model_df[metric].dropna().tolist())
            offsets = positions + (idx - (len(models) - 1) / 2.0) * width
            color = _model_color(model_name)
            ax.bar(
                offsets,
                vals,
                width=width * 0.9,
                color=color,
                alpha=1.0,
                linewidth=0,
                zorder=2,
            )
            ax.plot(
                offsets,
                vals,
                color=color,
                linewidth=0.7,
                alpha=0.28,
                zorder=3,
            )

        low, high = _metric_limits(
            metric_values,
            lower_bound=0.0,
            upper_bound=1.0,
            pad_fraction=0.14,
            min_span=0.16,
        )
        ax.set_ylim(max(0.0, low), high)
        ax.set_ylabel(label, fontsize=6.6, labelpad=1)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        _journal_axes(ax)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linewidth=0.3, color="#d9e2ec")
        ax.tick_params(axis="both", which="major", labelsize=5.6, length=1.8, width=0.7, pad=1.0)
        ax.spines["left"].set_linewidth(0.95)
        ax.spines["bottom"].set_linewidth(0.95)

    axes[-1].set_xticks(positions)
    axes[-1].set_xticklabels(
        [f"{label}\n(n={count})" for label, count in zip(_seqid_ticklabels(thresholds), truth_counts)],
        fontsize=5.4,
    )
    axes[-1].set_xlabel("Sequence-identity threshold", fontsize=6.6, labelpad=1)

    model_handles = [
        Patch(
            facecolor=_model_color(model_name),
            edgecolor=_model_color(model_name),
            label=_model_display_name(model_name),
        )
        for model_name in models
    ]

    axes[0].legend(
        handles=model_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.34),
        ncol=len(models),
        fontsize=5.1,
        columnspacing=0.45,
        handlelength=1.1,
        handletextpad=0.25,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(left=0.115, right=0.995, top=0.80, bottom=0.22)
    _save_figure(fig, out_prefix, dpi, exact_size=True)


def _load_confusion(path: Path):
    table = pd.read_csv(path, index_col=0)
    counts = table.to_numpy(dtype=float)
    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        fractions = counts / row_sums
    fractions[~np.isfinite(fractions)] = 0.0
    row_labels = [str(label).rsplit("_", 1)[-1] for label in table.index]
    col_labels = [str(label).rsplit("_", 1)[-1] for label in table.columns]
    return fractions, counts, row_labels, col_labels


def _plot_confusion(conf_path: Path, model_name: str, out_prefix: Path, dpi: int):
    fractions, counts, row_labels, col_labels = _load_confusion(conf_path)
    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    image = ax.imshow(fractions, cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Predicted bin")
    ax.set_ylabel("True bin")
    _journal_axes(ax)

    for i in range(fractions.shape[0]):
        for j in range(fractions.shape[1]):
            if counts[i, j] == 0:
                continue
            ax.text(
                j,
                i,
                f"{fractions[i, j] * 100:.0f}%\n(n={int(counts[i, j])})",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if fractions[i, j] >= 0.55 else "#102a43",
            )

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Row-normalized fraction")
    _journal_right_axis(cbar.ax)
    _save_figure(fig, out_prefix, dpi)


def main() -> int:
    args = parse_args()
    _setup_style()

    suite_dir = Path(args.suite_dir).resolve()
    results_dir = suite_dir / "suite_results"
    figures_dir = Path(results_dir / "figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    for stale_figure in figures_dir.glob("*.png"):
        stale_figure.unlink()

    fold_test_mean_std = pd.read_csv(results_dir / "fold_test_mean_std.csv")
    fold5_test_summary = pd.read_csv(results_dir / "fold5_test_summary.csv")
    fold5_seqid_summary = pd.read_csv(results_dir / "fold5_seqid_summary.csv")

    if not fold_test_mean_std.empty:
        _plot_metric_bars_from_mean_std(
            fold_test_mean_std,
            figures_dir / "allfold_test_metrics_mean_std",
            args.dpi,
        )
    if not fold5_test_summary.empty:
        _plot_metric_bars_from_summary(
            fold5_test_summary,
            figures_dir / "fold5_test_metrics",
            args.dpi,
        )

    if not fold5_seqid_summary.empty:
        for metric in METRIC_ORDER:
            _plot_seqid_metric(
                fold5_seqid_summary,
                metric,
                figures_dir / f"fold5_seqid_{metric}",
                args.dpi,
            )
        for model_name in _ordered_models(fold5_seqid_summary["model_name"].dropna().astype(str).unique()):
            _plot_seqid_dual_axis_per_model(
                fold5_seqid_summary,
                model_name,
                figures_dir / f"fold5_seqid_dual_{_model_slug(model_name)}",
                args.dpi,
            )
        _plot_seqid_dual_axis_compare(
            fold5_seqid_summary,
            figures_dir / "fold5_seqid_dual_compare_kcat",
            args.dpi,
        )

    conf_dir = suite_dir / "sets" / "fold5_test" / "evaluation" / "confusion_matrices"
    fold5_models = list(fold5_test_summary["model_name"].dropna().astype(str).unique()) if not fold5_test_summary.empty else MODEL_ORDER
    for model_name in _ordered_models(fold5_models):
        conf_path = conf_dir / f"{slugify(model_name)}_confusion_matrix.csv"
        if conf_path.exists():
            _plot_confusion(
                conf_path,
                model_name,
                figures_dir / f"fold5_test_confusion_{_model_slug(model_name)}",
                args.dpi,
            )

    print(f"Wrote figures to {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
