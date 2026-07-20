"""
tin_man_visualize.py

Reads a JSON file produced by tin_man's Wisard::save_to_file (FileFormat::Json)
and renders a PNG showing RAM counter activity using plotnine.

Usage:
    python tin_man_visualize.py wisard_model.json output.png
    python tin_man_visualize.py wisard_model.json heatmap.png class_a  # single-class heatmap
"""

import json
import sys
import numpy as np # type: ignore
import pandas as pd # type: ignore
from plotnine import ( # type: ignore
    ggplot, aes, geom_col, geom_tile, labs, theme_minimal, theme,
    scale_fill_gradient, element_text
)


def load_wisard_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def build_ram_summary_df(model: dict) -> pd.DataFrame:
    labels = model["labels"]
    discriminators = model["discriminators"]
    rows = []
    for label, disc in zip(labels, discriminators):
        for i, ram in enumerate(disc["rams"]):
            counts = ram["counts"]
            rows.append({"class": label, "ram_index": i, "mean_count": np.mean(counts)})
    return pd.DataFrame(rows)


def visualize_wisard(json_path: str, output_png: str = "wisard_summary.png") -> str:
    model = load_wisard_json(json_path)
    if not model["labels"] or not model["discriminators"]:
        raise ValueError("Model file contains no trained discriminators.")

    df = build_ram_summary_df(model)

    p = (
        ggplot(df, aes(x="factor(ram_index)", y="mean_count", fill="class"))
        + geom_col(position="dodge", width=0.7)
        + labs(
            title="Mean RAM counter value per node by class (trained WiSARD)",
            subtitle="Source: tin_man model file",
            x="RAM index",
            y="Mean count",
            fill="Class",
        )
        + theme_minimal()
        + theme(figure_size=(9, 5), legend_position="top", legend_title=element_text(size=11))
    )

    p.save(output_png, dpi=200, verbose=False)
    return output_png


def build_ram_heatmap_df(model: dict, class_label: str) -> pd.DataFrame:
    idx = model["labels"].index(class_label)
    disc = model["discriminators"][idx]
    rows = []
    for ram_idx, ram in enumerate(disc["rams"]):
        for addr, count in enumerate(ram["counts"]):
            rows.append({"ram_index": ram_idx, "address": addr, "count": count})
    return pd.DataFrame(rows)


def visualize_ram_heatmap(json_path: str, class_label: str, output_png: str) -> str:
    model = load_wisard_json(json_path)
    df = build_ram_heatmap_df(model, class_label)

    p = (
        ggplot(df, aes(x="address", y="factor(ram_index)", fill="count"))
        + geom_tile()
        + scale_fill_gradient(low="#f0f0f0", high="#08306b")
        + labs(
            title=f"RAM counter heatmap for class '{class_label}'",
            subtitle="Source: tin_man model file | darker = higher count",
            x="Address",
            y="RAM index",
            fill="Count",
        )
        + theme_minimal()
        + theme(figure_size=(9, 5))
    )

    p.save(output_png, dpi=200, verbose=False)
    return output_png


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python tin_man_visualize.py <model.json> <output.png> [class_label]")
        sys.exit(1)

    json_path = sys.argv[1]
    output_png = sys.argv[2]

    if len(sys.argv) == 4:
        visualize_ram_heatmap(json_path, sys.argv[3], output_png)
    else:
        visualize_wisard(json_path, output_png)

    print(f"Saved: {output_png}")