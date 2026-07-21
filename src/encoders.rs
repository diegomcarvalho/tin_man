use serde::{Deserialize, Serialize};

/// A linear thermometer encoder: maps a continuous value to a fixed-width
/// binary vector where the number of "1" bits grows linearly with the
/// value's position between `min` and `max`.
///
/// Example with resolution=4: a value at the 50th percentile of the
/// range produces `[1, 1, 0, 0]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearThermometer {
    min: f64,
    max: f64,
    resolution: usize,
}

impl LinearThermometer {
    /// `min`/`max`: the expected value range. `resolution`: number of
    /// bits produced per encoded value.
    pub fn new(min: f64, max: f64, resolution: usize) -> Self {
        assert!(max > min, "max must be greater than min");
        assert!(resolution > 0, "resolution must be > 0");
        LinearThermometer { min, max, resolution }
    }

    /// Fits the encoder's range directly from a data sample, using the
    /// observed min/max instead of manually specified bounds.
    pub fn fit(data: &[f64], resolution: usize) -> Self {
        let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        LinearThermometer::new(min, max, resolution)
    }

    pub fn resolution(&self) -> usize {
        self.resolution
    }

    /// Encodes a single value into a `resolution`-bit thermometer vector.
    pub fn encode(&self, value: f64) -> Vec<u8> {
        let clamped = value.clamp(self.min, self.max);
        let normalized = (clamped - self.min) / (self.max - self.min);
        let active_bits = (normalized * self.resolution as f64).round() as usize;
        thermometer_bits(active_bits, self.resolution)
    }

    /// Encodes a slice of values, concatenating each value's thermometer
    /// bits into a single flat binary vector (useful for building a
    /// full WiSARD input retina from multiple features).
    pub fn encode_vec(&self, values: &[f64]) -> Vec<u8> {
        values.iter().flat_map(|&v| self.encode(v)).collect()
    }
}

/// A Gaussian thermometer encoder: activates bits based on how many
/// standard deviations a value falls from the mean, using the normal
/// CDF to map values to a smooth 0-1 activation ratio before applying
/// the thermometer threshold. This spreads resolution more evenly
/// around the bulk of a normally-distributed feature, rather than
/// linearly across the raw value range.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianThermometer {
    mean: f64,
    std_dev: f64,
    resolution: usize,
}

impl GaussianThermometer {
    /// `mean`/`std_dev`: parameters of the assumed normal distribution.
    /// `resolution`: number of bits produced per encoded value.
    pub fn new(mean: f64, std_dev: f64, resolution: usize) -> Self {
        assert!(std_dev > 0.0, "std_dev must be > 0");
        assert!(resolution > 0, "resolution must be > 0");
        GaussianThermometer { mean, std_dev, resolution }
    }

    /// Fits mean and standard deviation directly from a data sample.
    pub fn fit(data: &[f64], resolution: usize) -> Self {
        let n = data.len() as f64;
        assert!(n > 1.0, "need at least 2 samples to fit a Gaussian thermometer");
        let mean = data.iter().sum::<f64>() / n;
        let variance = data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt().max(1e-9);
        GaussianThermometer::new(mean, std_dev, resolution)
    }

    pub fn resolution(&self) -> usize {
        self.resolution
    }

    /// Standard normal CDF approximation (Abramowitz & Stegun formula
    /// 26.2.17), avoiding an external dependency for erf().
    fn normal_cdf(&self, value: f64) -> f64 {
        let z = (value - self.mean) / self.std_dev;
        0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
    }

    /// Encodes a single value into a `resolution`-bit thermometer vector,
    /// using the Gaussian CDF to determine the activation ratio.
    pub fn encode(&self, value: f64) -> Vec<u8> {
        let cdf = self.normal_cdf(value).clamp(0.0, 1.0);
        let active_bits = (cdf * self.resolution as f64).round() as usize;
        thermometer_bits(active_bits, self.resolution)
    }

    pub fn encode_vec(&self, values: &[f64]) -> Vec<u8> {
        values.iter().flat_map(|&v| self.encode(v)).collect()
    }
}

/// A distributive thermometer encoder: bin boundaries are placed at the
/// empirical quantiles of a fitted dataset, so each bin holds
/// approximately the same number of training samples. This adapts
/// resolution to the actual data distribution rather than assuming
/// linear or Gaussian shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributiveThermometer {
    /// Sorted quantile boundaries, length = resolution - 1.
    boundaries: Vec<f64>,
}

impl DistributiveThermometer {
    /// Fits bin boundaries from a data sample so that each of
    /// `resolution` bins contains an approximately equal fraction of
    /// the data (i.e., quantile-based binning).
    pub fn fit(data: &[f64], resolution: usize) -> Self {
        assert!(resolution > 0, "resolution must be > 0");
        assert!(!data.is_empty(), "data must not be empty");

        let mut sorted: Vec<f64> = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut boundaries = Vec::with_capacity(resolution.saturating_sub(1));
        for i in 1..resolution {
            let q = i as f64 / resolution as f64;
            let idx = ((sorted.len() as f64 - 1.0) * q).round() as usize;
            boundaries.push(sorted[idx.min(sorted.len() - 1)]);
        }

        DistributiveThermometer { boundaries }
    }

    pub fn resolution(&self) -> usize {
        self.boundaries.len() + 1
    }

    /// Encodes a single value by counting how many quantile boundaries
    /// it exceeds, then converting that count into thermometer bits.
    pub fn encode(&self, value: f64) -> Vec<u8> {
        let active_bits = self.boundaries.iter().filter(|&&b| value >= b).count();
        thermometer_bits(active_bits, self.resolution())
    }

    pub fn encode_vec(&self, values: &[f64]) -> Vec<u8> {
        values.iter().flat_map(|&v| self.encode(v)).collect()
    }
}

/// Produces a thermometer-encoded bit vector of length `resolution`
/// with the first `active_bits` positions set to 1 (shared by all
/// three thermometer variants).
fn thermometer_bits(active_bits: usize, resolution: usize) -> Vec<u8> {
    let active = active_bits.min(resolution);
    (0..resolution).map(|i| if i < active { 1 } else { 0 }).collect()
}

/// Error function approximation (Abramowitz & Stegun 7.1.26), accurate
/// to ~1.5e-7, sufficient for encoding purposes without external crates.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}