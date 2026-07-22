# tin_man

A Rust library implementing the WiSARD family of weightless neural networks, plus thermometer-style feature encoders and full model persistence.

**WiSARD** (Wilkie, Stonham, and Aleksander’s Recognition Device) is a pioneering **Weightless Neural Network (WNN)** model developed in the **1970s and 1980s** by Bruce Wilkie, John Stonham, and Igor Aleksander. Unlike traditional neural networks that store knowledge in synaptic weights, WiSARD uses **Random Access Memory (RAM)** nodes to store learned patterns directly in lookup tables.

Igor Aleksander is an emeritus professor of Neural Systems Engineering in the Department of Electrical and Electronic Engineering at Imperial College London. He worked in artificial intelligence and neural networks, and advised my advisor, Felipe Maia Galvão França, who taught me the simplicity and the power of WNNs. 

This Rust library was named after the Tin Man. He, also known as the Tin Woodman or Nick Chopper, is a fictional character created by L. Frank Baum in his 1900 novel *The Wonderful Wizard of Oz*.  He is a sentient being made entirely of metal who seeks a heart to restore his capacity for emotion. As an engineer, I would like to remind you that *tin never, ever rusts*.

## Models

| Model | Purpose |
|---|---|
| `Wisard` | Standard multi-class classification |
| `ClusWisard` | Classification with multiple clusters per class |
| `RegressionWisard` | Continuous-value regression |
| `ClusRegressionWisard` | Regression with multiple clusters per group |

## Encoders

| Encoder | Bin placement |
|---|---|
| `LinearThermometer` | Uniform across `[min, max]` |
| `GaussianThermometer` | Concentrated near the mean (normal CDF) |
| `DistributiveThermometer` | Quantile-based, fit from data |

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