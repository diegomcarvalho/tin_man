"""
tin_man_mapping.py

Reads a JSON file produced by tin_man's save_to_file (FileFormat::Json)
and renders the retina-to-RAM mapping using plotnine: each cell is a
retina pixel, colored by its assigned RAM node id, and labeled with
"<ram_node_id>.<bit_index>" — the RAM node it feeds and its position
within that RAM's address line.

A tile labeled "2.3" means that pixel is the 4th bit (0-indexed position 3) fed into RAM node 2's address — so when that RAM computes its lookup address, this pixel's value gets shifted into that specific bit position. This is useful for verifying exact bit-ordering when debugging address computation or reproducing a specific tuple_indices layout by hand, beyond just knowing which RAM a pixel belongs to.

Usage:
    python tin_man_mapping.py wisard_model.json output.png [grid_width]
"""

import json
import sys
import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_tile, geom_text, labs, theme_minimal, theme,
    element_text, element_blank, guides
)


def load_wisard_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def build_mapping_df(json_path: str, grid_width: int | None = None):
    """
    Reconstructs, for each retina pixel, which RAM node id it feeds
    into and which bit position it occupies in that RAM's address line.

    tin_man's `mapping` field is a shuffled permutation of
    `0..input_size`; consecutive chunks of `address_size` entries form
    each RAM node's tuple of retina bit positions, in the exact order
    the RAM concatenates them into its address (see
    `Discriminator::address_for_ram`). This inverts that relationship:
    for every pixel position, it records the chunk index (RAM node id)
    and the position within the chunk (bit index).
    """
    model = load_wisard_json(json_path)
    input_size = model["input_size"]
    address_size = model["address_size"]
    mapping = model["mapping"]

    num_rams = int(np.ceil(input_size / address_size))
    if grid_width is None:
        grid_width = int(np.ceil(np.sqrt(input_size)))
    grid_height = int(np.ceil(input_size / grid_width))

    ram_of_pixel = np.zeros(input_size, dtype=int)
    bit_of_pixel = np.zeros(input_size, dtype=int)
    for ram_idx in range(num_rams):
        start = ram_idx * address_size
        end = min(start + address_size, input_size)
        for bit_idx, pixel_pos in enumerate(mapping[start:end]):
            ram_of_pixel[pixel_pos] = ram_idx
            bit_of_pixel[pixel_pos] = bit_idx

    rows = []
    for pos in range(input_size):
        x = pos % grid_width
        y = grid_height - 1 - (pos // grid_width)
        rows.append({
            "pixel": pos,
            "x": x,
            "y": y,
            "ram_node": ram_of_pixel[pos],
            "bit": bit_of_pixel[pos],
        })

    return pd.DataFrame(rows), grid_width, grid_height, num_rams


def visualize_mapping(json_path: str, output_png: str = "wisard_mapping.png",
                       grid_width: int | None = None, show_labels: bool = True) -> str:
    df, _, _, num_rams = build_mapping_df(json_path, grid_width)
    df["ram_node_str"] = df["ram_node"].astype(str)
    df["label"] = df["ram_node"].astype(str) + ":" + df["bit"].astype(str)

    p = (
        ggplot(df, aes(x="x", y="y", fill="ram_node_str"))
        + geom_tile(color="white", size=0.6)
        + labs(
            title=f"WiSARD retina-to-RAM mapping ({num_rams} RAM nodes)",
            subtitle="Source: tin_man model file | label = RAM node id . address bit index",
            x="Retina column",
            y="Retina row",
            fill="RAM node",
        )
        + theme_minimal()
        + theme(
            figure_size=(7, 6),
            axis_text=element_text(size=9),
            panel_grid=element_blank(),
            legend_position="right",
        )
    )

    if show_labels:
        p = p + geom_text(aes(label="label"), size=7, color="black")

    if num_rams > 12:
        p = p + guides(fill=False)

    p.save(output_png, dpi=200, verbose=False)
    return output_png


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python tin_man_mapping.py <model.json> <output.png> [grid_width]")
        sys.exit(1)

    json_path = sys.argv[1]
    output_png = sys.argv[2]
    grid_width = int(sys.argv[3]) if len(sys.argv) == 4 else None

    saved_path = visualize_mapping(json_path, output_png, grid_width)
    print(f"Saved: {saved_path}")