//! # tin_man
//!
//! A Rust library implementing the WiSARD family of weightless neural
//! networks: [`Wisard`], [`ClusWisard`], [`RegressionWisard`], and
//! [`ClusRegressionWisard`]. Also includes thermometer-style binary
//! encoders ([`encoders`]) for preprocessing continuous features into
//! the binary retinas these models require.
//!
//! ## Overview
//!
//! WiSARD (Wilkie, Stonham, and Aleksander Recognition Device) is a
//! weightless neural network: instead of storing weights, it uses
//! RAM-like lookup tables ("RAM nodes") that record which binary
//! addresses were seen during training. Classification counts how many
//! RAM nodes recognize an input's addresses; the class with the
//! highest score wins.
//!
//! This crate provides four models built on that core idea:
//!
//! - [`Wisard`]: standard multi-class classifier, one discriminator per class.
//! - [`ClusWisard`]: classifier that spawns multiple discriminators
//!   ("clusters") per class to handle heterogeneous sub-patterns.
//! - [`RegressionWisard`]: continuous-value prediction using RAMs that
//!   store access counters and target-value sums instead of hit/miss bits.
//! - [`ClusRegressionWisard`]: combines clustering with regression,
//!   allowing multiple regression clusters per group.
//!
//! ## Quick start
//!
//! ```
//! use tin_man::Wisard;
//!
//! let mut w = Wisard::new(
//!     8,     // input_size: retina length in bits
//!     4,     // address_size: bits per RAM addressing bus
//!     0.1,   // confidence_threshold: bleaching stop condition
//!     true,  // bleaching_enabled
//!     false, // ignore_zero
//! );
//!
//! w.train(&, "cold");[1]
//! w.train(&, "hot");[1]
//!
//! let (label, confidence) = w.classify(&).unwrap();[1]
//! assert_eq!(label, "hot");
//! ```
//!
//! ## Feature Encoding
//!
//! Continuous features must be binary-encoded before training. The
//! [`encoders`] module provides three thermometer variants:
//!
//! - [`encoders::LinearThermometer`]: uniform bins across a value range.
//! - [`encoders::GaussianThermometer`]: bins concentrated near the mean
//!   of a normally-distributed feature.
//! - [`encoders::DistributiveThermometer`]: quantile-based bins, robust
//!   to skewed or multimodal data.
//!
//! ## Persistence
//!
//! All four models support saving/loading trained state via
//! [`FileFormat::Json`] (human-readable) or [`FileFormat::Binary`]
//! (compact, via bincode):
//!
//! ```no_run
//! use tin_man::{FileFormat, Wisard};
//!
//! let w = Wisard::new(8, 4, 0.1, true, false);
//! w.save_to_file("model.json", FileFormat::Json).unwrap();
//! let w2 = Wisard::load_from_file("model.json", FileFormat::Json).unwrap();
//! ```
//!
//! ## Python bindings
//!
//! A companion crate, `tin_man_py`, exposes this library to Python via
//! PyO3/maturin. See the `tin_man_py` crate documentation and its
//! README for installation and usage.


mod ram;
mod discriminator;
mod regression_discriminator;
mod persist;
mod encoders;
mod wisard;
mod clus_wisard;
mod regression_wisard;
mod clus_regression_wisard;

pub use clus_regression_wisard::ClusRegressionWisard;
pub use clus_wisard::ClusWisard;
pub use encoders::{DistributiveThermometer, GaussianThermometer, LinearThermometer};
pub use persist::FileFormat;
pub use regression_wisard::RegressionWisard;
pub use wisard::Wisard;