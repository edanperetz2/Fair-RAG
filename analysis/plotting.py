"""
Scatter-panel plotting and macro-table display formatting, shared across all
analysis notebooks. Reconciles the repeated 3-panel scatter pattern (previously
copy-pasted ~5 times across fair_rag_experiment.ipynb) and the two independent
`format_macro_table_for_display` implementations.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric_scatter_panels(
    rows_df: pd.DataFrame,
    panels: Sequence[Tuple[str, str, str, str]],
    *,
    label_fn: Optional[Callable] = None,
    label_col: Optional[str] = None,
    title: str = "",
    figsize_per_panel: Tuple[float, float] = (6, 5),
) -> None:
    """
    Generic scatter-panel plotter.

    panels: list of (x_col, y_col, x_label, y_label) tuples, one subplot each.
    Pass exactly one of:
      - `label_fn(row) -> str`: one point plotted + annotated per row.
      - `label_col`: rows grouped by this column, one legend entry + one representative
        annotation per group (matches the original `plot_tradeoffs` behavior).
    """
    if rows_df.empty:
        print(f"{title}: no rows" if title else "no rows")
        return
    if (label_fn is None) == (label_col is None):
        raise ValueError("Pass exactly one of label_fn or label_col")

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per_panel[0] * n, figsize_per_panel[1]))
    if n == 1:
        axes = [axes]
    if title:
        fig.suptitle(title, fontsize=12)

    for ax, (xk, yk, xl, yl) in zip(axes, panels):
        if label_col is not None:
            for key, group in rows_df.groupby(label_col):
                pts = group[[xk, yk]].dropna()
                if pts.empty:
                    continue
                ax.scatter(pts[xk], pts[yk], s=38, alpha=0.75, label=str(key))
                center_idx = ((pts[xk] - pts[xk].mean()) ** 2 + (pts[yk] - pts[yk].mean()) ** 2).idxmin()
                crow = pts.loc[center_idx]
                ax.annotate(str(key), (crow[xk], crow[yk]), textcoords="offset points", xytext=(5, 4), fontsize=8)
            ax.legend(fontsize=8)
        else:
            for _, row in rows_df.iterrows():
                x, y = row.get(xk), row.get(yk)
                if x is None or pd.isna(x) or y is None or pd.isna(y):
                    continue
                ax.plot(x, y, marker="o", linestyle="None", markersize=9)
                ax.annotate(label_fn(row), (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)

        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f"{yl} vs {xl}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def format_macro_table_for_display(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Rename macro-comparison columns to short display names and reorder them, keeping
    separate raw/sortable columns (lamp_num, ranker, rerank_method, pl_alpha, ...)
    rather than combining them into a compact "recipe" string column.
    """
    if df is None:
        return None
    table = df.copy()
    rename_map = {
        "expected_utility": "EU",
        "avg_ild_jaccard": "ILD",
        "avg_jaccard_mean": "Jaccard",
        "ee_disparity": "EE-D",
        "ee_relevance": "EE-R",
    }
    table = table.rename(columns=rename_map)
    preferred = [
        "dataset_type", "lamp_num", "lamp_split_type", "generator_name", "ranker",
        "rerank_method", "pl_alpha", "pl_samples", "mmr_lambda", "top_k",
        "EU", "ILD", "Jaccard", "EE-D", "EE-R",
    ]
    cols = [c for c in preferred if c in table.columns]
    cols += [c for c in table.columns if c not in cols]
    return table[cols]
