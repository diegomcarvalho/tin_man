//! tin_man: a Rust library implementing the WiSARD family of
//! weightless neural networks — WiSARD, ClusWiSARD, Regression WiSARD,
//! and ClusRegressionWiSARD.


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