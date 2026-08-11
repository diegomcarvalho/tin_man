"""
tin_man_mental_image.py

Reads a JSON file produced by tin_man's save_to_file (FileFormat::Json)
and renders a WiSARD "mental image" for each class using plotnine.

The mental image technique (from the DRASiW literature) projects each
discriminator's learned RAM counter values back onto the original
retina bit positions, revealing which parts of the input space that
class has learned to associate with high activation.

Usage:
    python tin_man_mental_image.py wisard_model.json output.png [grid_width]
"""

import json
import sys
import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_tile, facet_wrap, labs, theme_minimal, theme,
    scale_fill_gradient, element_text, element_blank
)


def load_wisard_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def compute_mental_image(disc: dict, input_size: int, address_size: int) -> np.ndarray:
    """
    For each retina bit position, averages the counter values of every
    RAM address where that bit was set to 1, across all RAMs that
    include that bit in their tuple. This reconstructs a per-pixel
    "how strongly did this discriminator learn to expect a 1 here"
    score, following the DRASiW mental-image approach.
    """
    tuple_indices = disc["tuple_indices"]
    rams = disc["rams"]
    accum = np.zeros(input_size)
    hits = np.zeros(input_size)

    for ram, bit_positions in zip(rams, tuple_indices):
        counts = ram["counts"]
        for addr, c in enumerate(counts):
            if c == 0:
                continue
            bits = [(addr >> (address_size - 1 - k)) & 1 for k in range(address_size)]
            for bit_val, pos in zip(bits, bit_positions):
                if bit_val == 1:
                    accum[pos] += c
                    hits[pos] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        mental_image = np.where(hits > 0, accum / np.maximum(hits, 1), 0.0)
    return mental_image


def build_mental_image_df(json_path: str, grid_width: int | None = None):
    model = load_wisard_json(json_path)
    input_size = model["input_size"]
    address_size = model["address_size"]
    labels = model["labels"]
    discriminators = model["discriminators"]

    if grid_width is None:
        grid_width = int(np.ceil(np.sqrt(input_size)))
    grid_height = int(np.ceil(input_size / grid_width))

    rows = []
    for label, disc in zip(labels, discriminators):
        mental_image = compute_mental_image(disc, input_size, address_size)
        for pos in range(input_size):
            x = pos % grid_width
            y = grid_height - 1 - (pos // grid_width)
            rows.append({"class": label, "x": x, "y": y, "intensity": mental_image[pos]})

    return pd.DataFrame(rows), grid_width, grid_height


def visualize_mental_image(json_path: str, output_png: str = "wisard_mental_image.png",
                            grid_width: int | None = None) -> str:
    df, _, _ = build_mental_image_df(json_path, grid_width)
    n_classes = df["class"].nunique()
    ncol = min(n_classes, 3)

    p = (
        ggplot(df, aes(x="x", y="y", fill="intensity"))
        + geom_tile(color="white", size=0.3)
        + facet_wrap("~class", ncol=ncol)
        + scale_fill_gradient(low="#f7fbff", high="#08306b", name="Intensity")
        + labs(
            title="WiSARD mental image: retina activation by class",
            subtitle="Source: tin_man model file | brighter = stronger association",
            x="Retina column",
            y="Retina row",
        )
        + theme_minimal()
        + theme(
            figure_size=(4.5 * ncol, 4.5 * int(np.ceil(n_classes / ncol))),
            axis_text=element_text(size=9),
            panel_grid=element_blank(),
            strip_text=element_text(size=12, weight="bold"),
        )
    )

    p.save(output_png, dpi=200, verbose=False)
    return output_png


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python tin_man_mental_image.py <model.json> <output.png> [grid_width]")
        sys.exit(1)

    json_path = sys.argv[1]
    output_png = sys.argv[2]
    grid_width = int(sys.argv[3]) if len(sys.argv) == 4 else None

    saved_path = visualize_mental_image(json_path, output_png, grid_width)
    print(f"Saved: {saved_path}")
