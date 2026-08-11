"""
tin_man_bloom_mental_image.py

Reads a JSON file produced by BloomWisard's save_to_file
(FileFormat::Json) and renders a WiSARD "mental image" for each class
using plotnine, exactly like tin_man_mental_image.py does for the
exact-RAM Wisard — but adapted for BloomWisard's Bloom-filter-backed
RAM representation.

Structural difference from tin_man_mental_image.py:

    An exact Wisard RAM stores one counter per possible address
    (`ram["counts"][addr]`), so the mental-image script can read
    counts directly. A BloomWisard RAM (`BloomRam`) instead hashes
    each address through `num_hashes` independent hash functions into
    a much smaller `counters` array of size `bloom_size`. There is no
    direct address -> counter mapping to read; instead, this script
    re-implements BloomRam's exact 64-bit mixing hash (matching Rust's
    `BloomRam::hash`) in NumPy, computes the hashed slot for every
    possible address under every hash function, and takes the MINIMUM
    counter value across those slots per address — mirroring
    `BloomRam::count()`'s collision-suppression logic. This
    reconstructs an estimated per-address count array equivalent in
    shape to an exact RAM's `counts`, which then flows into the same
    mental-image projection technique (DRASiW) used by the original
    script.

Because this requires enumerating every possible address
(`2^address_size` values) per RAM to reconstruct estimated counts,
this script is only tractable for small-to-moderate `address_size`
(a good rule of thumb: address_size <= 20, i.e. up to ~1M addresses
per RAM). Larger address sizes will be slow and memory-heavy to
enumerate exhaustively; a warning is printed if this threshold is
exceeded.

Usage:
    python tin_man_bloom_mental_image.py bloom_wisard_model.json output.png [grid_width]
"""

import json
import sys
import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_tile, facet_wrap, labs, theme_minimal, theme,
    scale_fill_gradient, element_text, element_blank
)

MASK64 = np.uint64((1 << 64) - 1)
MUL_ADDR = np.uint64(0x2545F4914F6CDD1D)
MUL_MIX1 = np.uint64(0xFF51AFD7ED558CCD)
MUL_MIX2 = np.uint64(0xC4CEB9FE1A85EC53)
ADDRESS_SIZE_WARN_THRESHOLD = 20


def load_bloom_wisard_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def bloom_hash_vectorized(addresses: np.ndarray, seed: int, size: int) -> np.ndarray:
    """
    Vectorized re-implementation of Rust's `BloomRam::hash`, computing
    the hashed slot index for every address in `addresses` under a
    single hash function (identified by `seed`). Matches the exact
    64-bit wrapping arithmetic of the splitmix/MurmurHash3-style
    finalizer used on the Rust side.
    """
    seed_u64 = np.uint64(seed & int(MASK64))
    mixed = (addresses.astype(np.uint64) * MUL_ADDR) ^ seed_u64
    z = mixed
    z = (z ^ (z >> np.uint64(33))) * MUL_MIX1
    z = (z ^ (z >> np.uint64(33))) * MUL_MIX2
    z = z ^ (z >> np.uint64(33))
    return (z % np.uint64(size)).astype(np.int64)


def estimate_ram_counts(ram: dict, address_size: int) -> np.ndarray:
    """
    Reconstructs an estimated per-address count array for a single
    BloomRam, equivalent in shape/semantics to an exact Wisard RAM's
    `counts` array, by querying the Bloom filter for every possible
    address and taking the minimum across all hashed slots.
    """
    size = ram["size"]
    seeds = ram["seeds"]
    counters = np.array(ram["counters"], dtype=np.int64)

    num_addresses = 2 ** address_size
    addresses = np.arange(num_addresses, dtype=np.uint64)
    estimate = np.full(num_addresses, np.iinfo(np.int64).max, dtype=np.int64)

    for seed in seeds:
        idx = bloom_hash_vectorized(addresses, seed, size)
        estimate = np.minimum(estimate, counters[idx])

    return estimate


def compute_mental_image(disc: dict, input_size: int, address_size: int) -> np.ndarray:
    """
    For each retina bit position, averages the estimated Bloom-filter
    counter values of every RAM address where that bit was set to 1,
    across all RAMs that include that bit in their tuple. Structurally
    identical to the exact-RAM mental image reconstruction, except
    counts come from `estimate_ram_counts` instead of a direct
    `ram["counts"]` lookup.
    """
    tuple_indices = disc["tuple_indices"]
    rams = disc["rams"]
    accum = np.zeros(input_size)
    hits = np.zeros(input_size)

    for ram, bit_positions in zip(rams, tuple_indices):
        counts = estimate_ram_counts(ram, address_size)
        nonzero_addrs = np.nonzero(counts)[0]
        for addr in nonzero_addrs:
            c = counts[addr]
            bits = [(int(addr) >> (address_size - 1 - k)) & 1 for k in range(address_size)]
            for bit_val, pos in zip(bits, bit_positions):
                if bit_val == 1:
                    accum[pos] += c
                    hits[pos] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        mental_image = np.where(hits > 0, accum / np.maximum(hits, 1), 0.0)
    return mental_image


def build_mental_image_df(json_path: str, grid_width: int | None = None):
    model = load_bloom_wisard_json(json_path)
    input_size = model["input_size"]
    address_size = model["address_size"]
    labels = model["labels"]
    discriminators = model["discriminators"]

    if address_size > ADDRESS_SIZE_WARN_THRESHOLD:
        print(
            f"Warning: address_size={address_size} implies "
            f"{2 ** address_size:,} addresses per RAM to enumerate. "
            "This may be slow and memory-intensive.",
            file=sys.stderr,
        )

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


def visualize_mental_image(json_path: str, output_png: str = "bloom_wisard_mental_image.png",
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
            title="BloomWisard mental image: retina activation by class",
            subtitle="Source: BloomWisard model file | brighter = stronger association (Bloom-estimated)",
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
        print("Usage: python tin_man_bloom_mental_image.py <bloom_model.json> <output.png> [grid_width]")
        sys.exit(1)

    json_path = sys.argv[1]
    output_png = sys.argv[2]
    grid_width = int(sys.argv[3]) if len(sys.argv) == 4 else None

    saved_path = visualize_mental_image(json_path, output_png, grid_width)
    print(f"Saved: {saved_path}")
