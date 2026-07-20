//! tin_man: a Rust library implementing the WiSARD family of
//! weightless neural networks — WiSARD, ClusWiSARD, and Regression WiSARD.
//!
//! All models share a common RAM/discriminator core and support
//! optional bleaching, zero-address-ignoring, and full model
//! persistence to JSON or binary files.

mod ram;
mod discriminator;
mod persist;
mod wisard;
mod clus_wisard;
mod regression_wisard;

pub use clus_wisard::ClusWisard;
pub use persist::FileFormat;
pub use regression_wisard::RegressionWisard;
pub use wisard::Wisard;