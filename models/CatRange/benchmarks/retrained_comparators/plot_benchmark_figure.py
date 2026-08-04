#!/usr/bin/env python3
"""Create a manuscript-style benchmark figure from binned model comparisons."""

import argparse
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark_utils import slugify


CORE_METRICS = ["accuracy", "e_accuracy", "precision", "recall", "f1", "auc_pr"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "e_accuracy": "e-Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "auc_pr": "AUC-PR",
    "mcc": "MCC",
}
PALETTE = [
    "#1f5a91",
    "#d97706",
    "#2f855a",
    "#b4534b",
    "#6b5b95",
    "#2b8cbe",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a manuscript-style figure from evaluate_benchmark.py outputs."
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory produced by evaluate_benchmark.py containing summary_metrics.csv.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output path prefix. Default: <results-dir>/benchmark_manuscript_figure",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure title.",
    )
    parser.add_argument(
        "--model-order",
        action="append",
        default=[],
        help="Optional model ordering. Repeat to specify multiple models.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster DPI for PNG output.",
    )
    return parser.parse_args()


def _load_summary(results_dir: Path) -> pd.DataFrame:
    summary_path = results_dir / "summary_metrics.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary_metrics.csv not found in {results_dir}")

    summary_df = pd.read_csv(summary_path)
    required = ["model_name", "matched_rows", "accuracy", "e_accuracy", "mcc"]
    missing = [col for col in required if col not in summary_df.columns]
    if missing:
        raise ValueError(
            "summary_metrics.csv is missing required columns: {}".format(", ".join(missing))
        )
    if summary_df.empty:
        raise ValueError(f"summary_metrics.csv is empty: {summary_path}")
    return summary_df


def _apply_model_order(summary_df: pd.DataFrame, requested_order) -> pd.DataFrame:
    if not requested_order:
        return summary_df.copy()

    summary_df = summary_df.copy()
    order_map = {str(name).lower(): idx for idx, name in enumerate(requested_order)}
    lowered_names = summary_df["model_name"].astype(str).str.lower()
    missing = [name for name in requested_order if str(name).lower() not in lowered_names.tolist()]
    if missing:
        raise ValueError("Requested models not found in summary_metrics.csv: {}".format(", ".join(missing)))

    summary_df["_requested_rank"] = lowered_names.map(order_map)
    summary_df["_fallback_rank"] = np.arange(len(summary_df), dtype=float)
    summary_df["_sort_rank"] = summary_df["_requested_rank"].where(
        summary_df["_requested_rank"].notna(),
        len(order_map) + summary_df["_fallback_rank"],
    )
    summary_df = summary_df.sort_values(["_sort_rank", "e_accuracy", "accuracy"], ascending=[True, False, False])
    return summary_df.drop(columns=["_requested_rank", "_fallback_rank", "_sort_rank"]).reset_index(drop=True)


def _format_score(value: float) -> str:
    return "NA" if pd.isna(value) else f"{value:.2f}"


def _short_label(name: str, width: int = 12) -> str:
    return textwrap.fill(str(name), width=width)


def _confusion_path(results_dir: Path, model_name: str) -> Path:
    return results_dir / "confusion_matrices" / f"{slugify(model_name)}_confusion_matrix.csv"


def _load_confusion_table(path: Path):
    if not path.exists():
        return None, None, None, None

    table = pd.read_csv(path, index_col=0)
    counts = table.to_numpy(dtype=float)
    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        fractions = counts / row_sums
    fractions[~np.isfinite(fractions)] = 0.0

    row_labels = [str(label).rsplit("_", 1)[-1] for label in table.index]
    col_labels = [str(label).rsplit("_", 1)[-1] for label in table.columns]
    return fractions, counts, row_labels, col_labels


def _plot_metric_panel(ax, summary_df: pd.DataFrame, colors):
    metrics = [metric for metric in CORE_METRICS if metric in summary_df.columns and not summary_df[metric].isna().all()]
    x_positions = np.arange(len(metrics), dtype=float)
    n_models = len(summary_df)
    width = min(0.78 / max(n_models, 1), 0.28)

    for model_idx, (_, row) in enumerate(summary_df.iterrows()):
        offsets = x_positions + (model_idx - (n_models - 1) / 2.0) * width
        finite_values = row[metrics].astype(float)
        heights = np.nan_to_num(finite_values.to_numpy(dtype=float), nan=0.0)
        bars = ax.bar(
            offsets,
            heights,
            width=width * 0.92,
            color=colors[model_idx],
            edgecolor="white",
            linewidth=0.8,
            label="{} (n={:,})".format(row["model_name"], int(row["matched_rows"])),
        )

        for bar, raw_value in zip(bars, finite_values):
            x_center = bar.get_x() + bar.get_width() / 2.0
            if pd.isna(raw_value):
                bar.set_facecolor("white")
                bar.set_edgecolor(colors[model_idx])
                bar.set_hatch("///")
                ax.text(
                    x_center,
                    0.03,
                    "NA",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=colors[model_idx],
                    rotation=90,
                )
            else:
                ax.text(
                    x_center,
                    min(float(raw_value) + 0.025, 1.02),
                    f"{float(raw_value):.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#1f2933",
                )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([METRIC_LABELS[metric] for metric in metrics], rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Binned Classification Metrics", fontsize=13, fontweight="bold")
    ax.grid(axis="y", color="#d9e2ec", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", ncol=1, fontsize=9)


def _plot_mcc_panel(ax, summary_df: pd.DataFrame, colors):
    x_positions = np.arange(len(summary_df), dtype=float)
    raw_values = summary_df["mcc"].astype(float)
    heights = np.nan_to_num(raw_values.to_numpy(dtype=float), nan=0.0)

    bars = ax.bar(
        x_positions,
        heights,
        color=colors[: len(summary_df)],
        edgecolor="white",
        linewidth=0.8,
    )
    ax.axhline(0.0, color="#52606d", linewidth=0.9)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([_short_label(name, width=10) for name in summary_df["model_name"]], rotation=0)
    ax.set_title("Matthews Correlation", fontsize=13, fontweight="bold")
    ax.grid(axis="y", color="#d9e2ec", linewidth=0.8)
    ax.set_axisbelow(True)

    for bar, raw_value, color in zip(bars, raw_values, colors[: len(summary_df)]):
        x_center = bar.get_x() + bar.get_width() / 2.0
        if pd.isna(raw_value):
            bar.set_facecolor("white")
            bar.set_edgecolor(color)
            bar.set_hatch("///")
            ax.text(x_center, 0.05, "NA", ha="center", va="bottom", fontsize=8, color=color)
        else:
            vertical_offset = 0.04 if raw_value >= 0 else -0.06
            va = "bottom" if raw_value >= 0 else "top"
            ax.text(
                x_center,
                float(raw_value) + vertical_offset,
                f"{float(raw_value):.2f}",
                ha="center",
                va=va,
                fontsize=8,
                color="#1f2933",
            )


def _plot_confusion_panels(fig, axes, colorbar_ax, summary_df: pd.DataFrame, results_dir: Path):
    last_image = None

    for ax, (_, row) in zip(axes, summary_df.iterrows()):
        matrix_path = _confusion_path(results_dir, row["model_name"])
        fractions, counts, row_labels, col_labels = _load_confusion_table(matrix_path)

        if fractions is None:
            ax.set_axis_off()
            ax.text(
                0.5,
                0.5,
                "No confusion matrix\navailable",
                ha="center",
                va="center",
                fontsize=11,
                color="#52606d",
            )
            ax.set_title(str(row["model_name"]), fontsize=12, fontweight="bold")
            continue

        last_image = ax.imshow(fractions, cmap="YlGnBu", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels)
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.set_xlabel("Predicted bin")
        ax.set_ylabel("True bin")

        title_lines = [
            str(row["model_name"]),
            "acc={}  e-acc={}".format(_format_score(row["accuracy"]), _format_score(row["e_accuracy"])),
        ]
        if "coverage_vs_truth" in row and pd.notna(row["coverage_vs_truth"]):
            title_lines.append("coverage={:.0%}".format(float(row["coverage_vs_truth"])))
        ax.set_title("\n".join(title_lines), fontsize=11, fontweight="bold", pad=10)

        for i in range(fractions.shape[0]):
            for j in range(fractions.shape[1]):
                value = float(fractions[i, j])
                count = int(round(counts[i, j]))
                if count == 0:
                    continue
                label = "{:.0f}%\n(n={})".format(value * 100.0, count)
                text_color = "white" if value >= 0.55 else "#102a43"
                ax.text(j, i, label, ha="center", va="center", fontsize=7.5, color=text_color)

    if last_image is None:
        colorbar_ax.set_axis_off()
    else:
        colorbar = fig.colorbar(last_image, cax=colorbar_ax)
        colorbar.set_label("Row-normalized fraction", rotation=90)


def main() -> int:
    args = parse_args()

    results_dir = Path(args.results_dir)
    output_prefix = Path(args.output_prefix) if args.output_prefix else results_dir / "benchmark_manuscript_figure"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#9fb3c8",
            "axes.linewidth": 0.8,
            "xtick.color": "#243b53",
            "ytick.color": "#243b53",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    summary_df = _load_summary(results_dir)
    summary_df = _apply_model_order(summary_df, args.model_order)
    n_models = len(summary_df)

    figure_width = max(11.0, 3.4 * n_models + 2.5)
    figure_height = 8.8
    fig = plt.figure(figsize=(figure_width, figure_height), constrained_layout=True)
    grid = fig.add_gridspec(
        nrows=2,
        ncols=n_models + 1,
        width_ratios=[1.0] * n_models + [0.75],
        height_ratios=[1.0, 1.25],
    )

    metrics_ax = fig.add_subplot(grid[0, :n_models])
    mcc_ax = fig.add_subplot(grid[0, n_models])
    confusion_axes = [fig.add_subplot(grid[1, idx]) for idx in range(n_models)]
    colorbar_ax = fig.add_subplot(grid[1, n_models])

    colors = [PALETTE[idx % len(PALETTE)] for idx in range(n_models)]

    _plot_metric_panel(metrics_ax, summary_df, colors)
    _plot_mcc_panel(mcc_ax, summary_df, colors)
    _plot_confusion_panels(fig, confusion_axes, colorbar_ax, summary_df, results_dir)

    if args.title:
        fig.suptitle(args.title, fontsize=15, fontweight="bold")

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
