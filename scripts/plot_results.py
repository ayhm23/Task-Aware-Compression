"""
scripts/plot_results.py
Step 5 — Analysis & Figures

Generates 4 publication-quality plots from the full_results_table.csv:

  Fig 1 — Performance vs Compression Ratio  (3-panel line plot, one per task)
  Fig 2 — Cross-Task Generalization Heatmap  (3×3 per method)
  Fig 3 — Method Comparison Bar Chart        (at dim=128, agnostic vs best-aware)
  Fig 4 — Task-Aware vs Task-Agnostic Delta  (how much task supervision helps)

Usage:
  python scripts/plot_results.py
  python scripts/plot_results.py --show      # display instead of saving
"""

import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless-safe; override below if --show
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import METRICS_DIR, PLOTS_DIR

# ─── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  9.5,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.35,
    "grid.linestyle":   "--",
})

METHOD_COLOR  = {"linear": "#2196F3", "autoencoder": "#4CAF50", "distillation": "#FF5722"}
METHOD_MARKER = {"linear": "o",       "autoencoder": "s",        "distillation": "^"}
METHOD_LABEL  = {"linear": "Linear",  "autoencoder": "Autoencoder", "distillation": "Distillation"}

TASK_METRIC   = {"sts": "STS_spearman", "nli": "NLI_accuracy", "classification": "CLS_accuracy"}
TASK_LABEL    = {"sts": "STS (Spearman ρ)", "nli": "NLI (Accuracy)", "classification": "SST-2 (Accuracy)"}
TASK_SHORT    = {"sts": "STS", "nli": "NLI", "classification": "CLS"}

DIMS          = [32, 64, 128, 256]
METHODS       = ["linear", "autoencoder", "distillation"]
TASKS         = ["sts", "nli", "classification"]


# ─── Load data ────────────────────────────────────────────────────────────────

def load_table():
    csv = os.path.join(METRICS_DIR, "full_results_table.csv")
    df  = pd.read_csv(csv)
    return df


def baseline_scores(df):
    row = df[df["Mode"] == "baseline"].iloc[0]
    return {
        "sts":            row["STS_spearman"],
        "nli":            row["NLI_accuracy"],
        "classification": row["CLS_accuracy"],
    }


# ─── Fig 1: Performance vs Compression Ratio ─────────────────────────────────

def plot_perf_vs_dim(df, show=False):
    """
    3-panel figure. Each panel = one task.
    X-axis: compressed dim (32, 64, 128, 256).
    Y-axis: task metric.
    Lines: one per method (agnostic only, cleaner signal).
    Baseline shown as dashed horizontal line.
    """
    base = baseline_scores(df)
    agnostic = df[df["Mode"] == "task_agnostic"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    fig.suptitle("Performance vs Compressed Dimension (Task-Agnostic)", fontsize=13, fontweight="bold", y=1.02)

    for ax, task in zip(axes, TASKS):
        metric = TASK_METRIC[task]

        # Baseline horizontal line
        ax.axhline(base[task], color="black", linestyle="--", linewidth=1.5,
                   label="Baseline (768-d)", alpha=0.8)

        for method in METHODS:
            subset = agnostic[agnostic["Method"] == method].sort_values("Dim")
            ax.plot(subset["Dim"], subset[metric],
                    color=METHOD_COLOR[method],
                    marker=METHOD_MARKER[method],
                    linewidth=2, markersize=7,
                    label=METHOD_LABEL[method])

        ax.set_title(TASK_LABEL[task])
        ax.set_xlabel("Compressed Dimension")
        ax.set_ylabel("Score")
        ax.set_xticks(DIMS)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

        # Compression ratio annotations on top x-axis
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(DIMS)
        ax2.set_xticklabels([f"{768//d}×" for d in DIMS], fontsize=8.5)
        ax2.set_xlabel("Compression Ratio", fontsize=9)
        ax2.spines["top"].set_visible(True)

        if task == "sts":
            ax.legend(loc="lower right", framealpha=0.85)

    fig.tight_layout()
    _save(fig, "fig1_perf_vs_dim.png", show)


# ─── Fig 2: Cross-Task Generalization Heatmap ────────────────────────────────

def plot_cross_task_heatmap(df, show=False):
    """
    One 3×3 heatmap per method (3 methods → 1 row of 3 heatmaps).
    Row = train task, Col = eval task.
    Value = metric at dim=128.
    Cells are delta from baseline to highlight gain/loss.
    """
    base  = baseline_scores(df)
    aware = df[(df["Mode"] == "task_aware") & (df["Dim"] == 128)]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Cross-Task Generalization — Task-Aware Compressors (dim=128)\n"
                 "Cell values = score (Δ vs baseline in parentheses)",
                 fontsize=12, fontweight="bold")

    task_order = ["sts", "nli", "classification"]

    for ax, method in zip(axes, METHODS):
        matrix = np.zeros((3, 3))
        annots = []

        for r, train_task in enumerate(task_order):
            row_annots = []
            row = aware[(aware["Method"] == method) &
                        (aware["TrainTask"] == train_task)]
            for c, eval_task in enumerate(task_order):
                metric = TASK_METRIC[eval_task]
                val    = row[metric].values[0] if len(row) > 0 else np.nan
                delta  = val - base[eval_task]
                matrix[r, c] = val
                row_annots.append(f"{val:.3f}\n({delta:+.3f})")
            annots.append(row_annots)

        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.55, vmax=0.92)

        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels([TASK_SHORT[t] for t in task_order], fontsize=10)
        ax.set_yticklabels([TASK_SHORT[t] for t in task_order], fontsize=10)
        ax.set_xlabel("Eval Task", fontsize=10)
        ax.set_ylabel("Train Task", fontsize=10)
        ax.set_title(METHOD_LABEL[method], fontsize=11, fontweight="bold")

        # Annotate cells
        for r in range(3):
            for c in range(3):
                color = "white" if matrix[r, c] < 0.72 else "black"
                ax.text(c, r, annots[r][c], ha="center", va="center",
                        fontsize=8, color=color)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    _save(fig, "fig2_cross_task_heatmap.png", show)


# ─── Fig 3: Method Comparison Bar Chart ──────────────────────────────────────

def plot_method_comparison(df, show=False):
    """
    At dim=128: compare all 3 methods × {agnostic, best-aware (own task)}.
    3 task panels, grouped bars.
    """
    base  = baseline_scores(df)
    dim   = 128

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Method Comparison at dim=128\n"
                 "(Task-Agnostic vs Task-Aware trained on matching task)",
                 fontsize=12, fontweight="bold")

    bar_width = 0.25
    x         = np.arange(len(METHODS))

    for ax, task in zip(axes, TASKS):
        metric       = TASK_METRIC[task]
        train_task   = task   # "own task" aware scenario

        agnostic_scores = []
        aware_scores    = []

        for method in METHODS:
            ag  = df[(df["Method"] == method) & (df["Mode"] == "task_agnostic") &
                     (df["Dim"] == dim)][metric].values
            aw  = df[(df["Method"] == method) & (df["Mode"] == "task_aware") &
                     (df["TrainTask"] == train_task) & (df["Dim"] == dim)][metric].values

            agnostic_scores.append(ag[0] if len(ag) > 0 else np.nan)
            aware_scores.append(aw[0]    if len(aw) > 0 else np.nan)

        bars1 = ax.bar(x - bar_width/2, agnostic_scores, bar_width,
                       label="Task-Agnostic", color="#90CAF9", edgecolor="white")
        bars2 = ax.bar(x + bar_width/2, aware_scores,    bar_width,
                       label=f"Task-Aware ({TASK_SHORT[task]})", color="#1565C0", edgecolor="white")

        # Baseline line
        ax.axhline(base[task], color="black", linestyle="--", linewidth=1.3,
                   label="Baseline (768-d)", alpha=0.8)

        ax.set_title(TASK_LABEL[task])
        ax.set_ylabel("Score")
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABEL[m] for m in METHODS], fontsize=9.5)

        # Value labels on bars
        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)

        # Y-axis: zoom in to relevant range
        all_vals = agnostic_scores + aware_scores + [base[task]]
        all_vals = [v for v in all_vals if not np.isnan(v)]
        ax.set_ylim(min(all_vals) - 0.04, max(all_vals) + 0.04)

        if task == "sts":
            ax.legend(fontsize=8.5, loc="lower right")

    fig.tight_layout()
    _save(fig, "fig3_method_comparison.png", show)


# ─── Fig 4: Task-Aware vs Task-Agnostic Delta ────────────────────────────────

def plot_aware_vs_agnostic_delta(df, show=False):
    """
    For each method and dim, plot the gain/loss when switching from
    task-agnostic to task-aware (trained on own task).

    Each method gets a subplot. X = dim, Y = delta in score.
    3 lines per subplot (one per task). Positive = task-awareness helps.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    fig.suptitle("Task-Aware vs Task-Agnostic: Score Delta (dim varies)\n"
                 "Positive = task-specific training helps",
                 fontsize=12, fontweight="bold", y=1.02)

    task_color = {"sts": "#E91E63", "nli": "#3F51B5", "classification": "#009688"}

    for ax, method in zip(axes, METHODS):
        for task in TASKS:
            metric = TASK_METRIC[task]
            deltas = []
            for dim in DIMS:
                agnostic = df[(df["Method"] == method) & (df["Mode"] == "task_agnostic") &
                              (df["Dim"] == dim)][metric].values
                aware    = df[(df["Method"] == method) & (df["Mode"] == "task_aware") &
                              (df["TrainTask"] == task) & (df["Dim"] == dim)][metric].values
                if len(agnostic) > 0 and len(aware) > 0:
                    deltas.append(aware[0] - agnostic[0])
                else:
                    deltas.append(np.nan)

            ax.plot(DIMS, deltas,
                    color=task_color[task],
                    marker=METHOD_MARKER[method],
                    linewidth=2, markersize=7,
                    label=TASK_SHORT[task])

        ax.axhline(0, color="black", linewidth=1.0, linestyle="-", alpha=0.5)
        ax.set_title(METHOD_LABEL[method], fontsize=11, fontweight="bold")
        ax.set_xlabel("Compressed Dimension")
        ax.set_ylabel("Δ Score (aware − agnostic)")
        ax.set_xticks(DIMS)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

        if method == "linear":
            ax.legend(title="Eval Task", fontsize=9)

    fig.tight_layout()
    _save(fig, "fig4_aware_vs_agnostic_delta.png", show)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def plot_pca_comparison(df, show=False):
    """Fig 8: PCA vs learned compressors — shows where learning beats classical."""
    base = baseline_scores(df)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("PCA vs Learned Compressors (Task-Agnostic)\n"
                 "Shows where neural compression outperforms classical PCA",
                 fontsize=12, fontweight="bold")

    pca_color = "#9C27B0"
    for ax, task in zip(axes, TASKS):
        metric = TASK_METRIC[task]
        ax.axhline(base[task], color="black", linestyle="--", linewidth=1.3,
                   label="Baseline (768-d)", alpha=0.7)

        # PCA line
        pca_rows = df[df["Mode"] == "pca"].sort_values("Dim")
        if len(pca_rows):
            ax.plot(pca_rows["Dim"], pca_rows[metric], color=pca_color,
                    marker="D", linewidth=2, markersize=7, linestyle="--", label="PCA")

        for method in METHODS:
            sub = df[(df["Mode"] == "task_agnostic") &
                     (df["Method"] == method)].sort_values("Dim")
            ax.plot(sub["Dim"], sub[metric],
                    color=METHOD_COLOR[method], marker=METHOD_MARKER[method],
                    linewidth=2, markersize=7, label=METHOD_LABEL[method])

        ax.set_title(TASK_LABEL[task])
        ax.set_xlabel("Compressed Dimension")
        ax.set_ylabel("Score")
        ax.set_xticks(DIMS)
        if task == "sts":
            ax.legend(fontsize=8.5, loc="lower right")

    fig.tight_layout()
    _save(fig, "fig8_pca_vs_learned.png", show)


def plot_mixed_interpolation(df, show=False):
    """Fig 9: Mixed compressor sits between single-task compressors."""
    base = baseline_scores(df)
    pairs = [("sts", "nli"), ("nli", "classification")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Mixed Compressor Interpolation (autoencoder, dim=128 & 256)\n"
                 "Mixed score should lie between the two single-task compressors",
                 fontsize=12, fontweight="bold")

    colors = {"single_a": "#2196F3", "mixed": "#FF9800", "single_b": "#4CAF50"}

    for ax, (ta, tb) in zip(axes, pairs):
        metric_a = TASK_METRIC[ta]
        metric_b = TASK_METRIC[tb]
        dims_available = []
        scores_a, scores_b, scores_mix = [], [], []

        for dim in [128, 256]:
            single_a = df[(df["Method"] == "autoencoder") & (df["Mode"] == "task_aware") &
                          (df["TrainTask"] == ta) & (df["Dim"] == dim)]
            single_b = df[(df["Method"] == "autoencoder") & (df["Mode"] == "task_aware") &
                          (df["TrainTask"] == tb) & (df["Dim"] == dim)]
            mixed_r  = df[(df["Method"] == "autoencoder") & (df["Mode"] == "mixed") &
                          (df["TrainTask"] == f"{ta}+{tb}") & (df["Dim"] == dim)]
            if len(single_a) and len(single_b) and len(mixed_r):
                dims_available.append(dim)
                # Use average of the two task metrics to show interpolation
                scores_a.append((single_a[metric_a].values[0] + single_a[metric_b].values[0]) / 2)
                scores_b.append((single_b[metric_a].values[0] + single_b[metric_b].values[0]) / 2)
                scores_mix.append((mixed_r[metric_a].values[0] + mixed_r[metric_b].values[0]) / 2)

        if dims_available:
            ax.plot(dims_available, scores_a, color=colors["single_a"],
                    marker="o", linewidth=2, markersize=8, label=f"Task-aware ({ta.upper()} only)")
            ax.plot(dims_available, scores_mix, color=colors["mixed"],
                    marker="s", linewidth=2, markersize=8, linestyle="--", label="Mixed (α=0.5)")
            ax.plot(dims_available, scores_b, color=colors["single_b"],
                    marker="^", linewidth=2, markersize=8, label=f"Task-aware ({tb.upper()} only)")

        ax.set_title(f"Tasks: {ta.upper()} + {tb.upper()}")
        ax.set_xlabel("Compressed Dimension")
        ax.set_ylabel("Avg score across both tasks")
        ax.set_xticks([128, 256])
        ax.legend(fontsize=9)

    fig.tight_layout()
    _save(fig, "fig9_mixed_interpolation.png", show)


def _save(fig, filename, show):
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true",
                        help="Display plots interactively instead of saving")
    args = parser.parse_args()

    if args.show:
        matplotlib.use("TkAgg")

    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("[Plotter] Loading results table...")
    df = load_table()
    print(f"  {len(df)} rows loaded\n")

    print("[Fig 1] Performance vs Compression Ratio...")
    plot_perf_vs_dim(df, args.show)

    print("[Fig 2] Cross-Task Generalization Heatmap...")
    plot_cross_task_heatmap(df, args.show)

    print("[Fig 3] Method Comparison Bar Chart...")
    plot_method_comparison(df, args.show)

    print("[Fig 4] Task-Aware vs Task-Agnostic Delta...")
    plot_aware_vs_agnostic_delta(df, args.show)

    print("[Fig 8] PCA vs Learned Compressors...")
    plot_pca_comparison(df, args.show)

    print("[Fig 9] Mixed Compressor Interpolation...")
    plot_mixed_interpolation(df, args.show)

    print(f"\n[Done] All figures saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
