# tin_man

A Rust library implementing the WiSARD family of weightless neural networks, plus thermometer-style feature encoders and full model persistence.

**WiSARD** (Wilkie, Stonham, and Aleksander's Recognition Device) is a pioneering **Weightless Neural Network (WNN)** model developed in the 1970s and 1980s by Bruce Wilkie, John Stonham, and Igor Aleksander. Unlike traditional neural networks that store knowledge in synaptic weights, WiSARD uses **Random Access Memory (RAM)** nodes to store learned patterns directly in lookup tables.

Igor Aleksander is an emeritus professor of Neural Systems Engineering in the Department of Electrical and Electronic Engineering at Imperial College London. He worked in artificial intelligence and neural networks and advised my advisor, Felipe Maia Galvão França, who taught me the simplicity and power of WNNs. 

Felipe's PhD dissertation covers two of his research passions: Scheduling by Edge Reversal (SER) and WNNs. He has advised a prolific group of researchers on both themes (BTW, my Master's, for instance, extended SER with hibernation) and has co-authored more than fifty published manuscripts in the WNN research field, reflecting his lasting contribution to it.

This Rust library is named after the Tin Man, also known as the Tin Woodman or Nick Chopper — a fictional character created by L. Frank Baum in his 1900 novel *The Wonderful Wizard of Oz*. He is a sentient being made entirely of metal who seeks a heart to restore his capacity for emotion. As an engineer, I'd like to remind you: *tin never, ever rusts*.


## Models

| Model | Purpose |
|---|---|
| `Wisard` | Standard multi-class classification |
| `ClusWisard` | Classification with multiple clusters per class |
| `RegressionWisard` | Continuous-value regression |
| `ClusRegressionWisard` | Regression with multiple clusters per group |

### Bloom filter variants

For large `address_size` values, an exact RAM node needs `2^address_size` counters — most of which are never visited. The Bloom-filter-backed variants below replace each RAM's exact address table with a compact, multi-hash counting structure, trading a small, tunable false-positive rate for dramatically lower memory use.

| Model | Purpose | Backing structure |
|---|---|---|
| `BloomWisard` | `Wisard`-equivalent classification | Counting Bloom filter per RAM (`bloom_size` counters, `num_hashes` hash functions) |
| `BloomClusWisard` | `ClusWisard`-equivalent, multiple Bloom-backed clusters per class | Same as `BloomWisard`, one filter set per cluster |
| `BloomRegressionWisard` | `RegressionWisard`-equivalent continuous-value regression | Count-Min-Sketch-style dual sum/count Bloom structure per RAM |

Every Bloom model exposes `memory_bytes()` so you can directly measure the memory savings against an equivalent exact model, and accepts a `parallel: bool` constructor flag (where applicable) to switch classification between sequential and [`rayon`](https://github.com/rayon-rs/rayon)-parallelized execution.

## Encoders

| Encoder | Bin placement |
|---|---|
| `LinearThermometer` | Uniform across `[min, max]` |
| `GaussianThermometer` | Concentrated near the mean (normal CDF) |
| `DistributiveThermometer` | Quantile-based, fit from data |
| `KernelCanvas` | Variable-length sequences of points into a fixed-size binary canvas via random kernels |

## Project layout

```
tin_man/
├── src/ Core library (models, RAMs, encoders, persistence, Bloom variants)
├── benches/ Criterion throughput benchmarks
├── examples/ Standalone runnable usage examples
├── tests/ Integration tests (cargo test)
├── tin_man_py/ Python bindings (PyO3 + maturin)
├── tin_manR/ R bindings (extendr + rextendr)
├── scripts/ Python visualization utilities (mental images, retina mapping)
└── README.md
```


### `benches/`

Benchmarks are run with [Criterion](https://github.com/bheisler/criterion.rs) and measure training and classification/prediction throughput (operations per second) for all models:

| File | What it measures |
|---|---|
| `wisard_bench.rs` | Training and classification throughput for `Wisard`, `ClusWisard`, `RegressionWisard`, and `ClusRegressionWisard`, plus a classification throughput sweep across different `address_size` values |
| `bloom_wisard_bench.rs` | Training and classification throughput for `BloomWisard`, `BloomClusWisard`, and `BloomRegressionWisard`, plus a comparison sweep of `bloom_size`/`num_hashes` against exact-RAM throughput and memory usage |

Run all benchmarks with:

```bash
cargo bench
```

Criterion prints per-operation timing and throughput (e.g. `Kelem/s`) directly in the terminal, and generates an interactive HTML report at `target/criterion/report/index.html`.

### `examples/`

Runnable, self-contained examples demonstrating end-to-end usage of each model and encoder. Run any example with:

```bash
cargo run --example <name>
```

| Example | Demonstrates |
|---|---|
| `wisard_basic` | Training and classifying with `Wisard` on binary-encoded input |
| `clus_wisard_basic` | Handling heterogeneous sub-patterns within a class using `ClusWisard` |
| `regression_wisard_basic` | Continuous-value prediction with `RegressionWisard` |
| `clus_regression_wisard_basic` | Multi-cluster regression with `ClusRegressionWisard` and grouped predictions |
| `thermometer_encoding` | Fitting and applying `LinearThermometer`, `GaussianThermometer`, and `DistributiveThermometer` to continuous features |
| `persistence` | Saving and loading trained models via `FileFormat::Json` and `FileFormat::Binary` |
| `kernel_canvas_timeseries` | Classifying variable-length 2D strokes (diagonal vs. circular) via `KernelCanvas` + `Wisard` |
| `bloom_iris` | Classifying the Iris dataset with `BloomWisard` and `BloomClusWisard`, sequentially trained so that classes are discovered incrementally (one class trained initially, two more discovered on the fly) |

### `tests/`

Integration tests exercising the public API of all models — training, classification/prediction correctness, edge cases (untrained models, input size mismatches), bleaching vs. binary mode consistency, and Bloom-filter false-positive-rate sanity checks. Run with:

```bash
cargo test
```

### `scripts/`

Python utilities (using `plotnine`) for visualizing trained models as PNGs:

| Script | What it renders |
|---|---|
| `tin_man_mental_image.py` | DRASiW-style "mental image" per class for an exact `Wisard` model, projecting RAM counter values back onto retina bit positions |
| `tin_man_bloom_mental_image.py` | Same mental-image visualization for a `BloomWisard` model, reconstructing estimated per-address counts from the Bloom filter's hashed counters before projecting |
| `tin_man_mapping.py` | Visualizes the retina-to-RAM mapping, labeling each pixel with its assigned RAM node id and address bit index |

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
tin_man = { path = "path/to/tin_man" }
```

## Quick example

```rust
use tin_man::{Wisard, encoders::LinearThermometer};

fn main() {
    let data = vec![1.2, 3.4, 2.1, 5.6, 4.3];
    let encoder = LinearThermometer::fit(&data, 8);

    let mut w = Wisard::new(8, 4, 0.1, true, false);
    w.train(&encoder.encode(1.2), "low");
    w.train(&encoder.encode(5.6), "high");

    let (label, confidence) = w.classify(&encoder.encode(1.5)).unwrap();
    println!("{label} ({confidence:.2})");
}
```

## Persistence

All models support saving/loading via `FileFormat::Json` (readable) or
`FileFormat::Binary` (compact):

```rust
use tin_man::{FileFormat, Wisard};

let w = Wisard::new(8, 4, 0.1, true, false);
w.save_to_file("model.json", FileFormat::Json).unwrap();
let w2 = Wisard::load_from_file("model.json", FileFormat::Json).unwrap();
```

## Python bindings

See [`tin_man_py/README.md`](tin_man_py/README.md) for installing and
using this library from Python via PyO3/maturin.

## Generating docs

```bash
cargo doc --open --no-deps
```

## License

MIT