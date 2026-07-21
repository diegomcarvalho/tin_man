/// Regression WiSARD: RAMs store access counters and target-value sums
/// per address, enabling continuous-value prediction instead of class
/// scoring.
///
/// Each RAM's stored value at a given address is the running mean of
/// all target values seen at that address. Prediction averages these
/// partial means across all RAMs whose address has been visited at
/// least `min_zero` times.
///
/// # Example
///
/// ```
/// use tin_man::RegressionWisard;
///
/// let mut rew = RegressionWisard::new(8, 4, 1);
/// rew.train(&, 10.0);[1]
/// rew.train(&, 12.0);[1]
///
/// let prediction = rew.predict(&).unwrap();[1]
/// assert!((prediction - 11.0).abs() < 1e-6);
/// ```
use crate::persist::{load_from_file, save_to_file, FileFormat};
use crate::ram::RegressionRam;
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use std::io::Result as IoResult;
use std::path::Path;

/// Regression WiSARD: RAMs store access counters and target-value sums
/// per address, enabling continuous-value prediction instead of
/// class scoring.
#[derive(Serialize, Deserialize)]
pub struct RegressionWisard {
    input_size: usize,
    tuple_indices: Vec<Vec<usize>>,
    rams: Vec<RegressionRam>,
}

impl RegressionWisard {
    /// Creates a new, untrained Regression WiSARD model.
    ///
    /// - `input_size` / `address_size`: same as [`Wisard::new`].
    /// - `min_zero`: minimum access count required for a RAM's address
    ///   to contribute to a prediction. Addresses seen fewer times are
    ///   excluded, reducing noise from rarely-seen patterns.
    pub fn new(input_size: usize, address_size: usize, min_zero: u32) -> Self {
        assert!(
            address_size > 0 && address_size <= input_size,
            "address_size must be in (0, input_size]"
        );

        let mut mapping: Vec<usize> = (0..input_size).collect();
        mapping.shuffle(&mut thread_rng());
        let tuple_indices: Vec<Vec<usize>> = mapping.chunks(address_size).map(|c| c.to_vec()).collect();
        let rams = tuple_indices
            .iter()
            .map(|t| RegressionRam::new(1 << t.len(), min_zero))
            .collect();

        RegressionWisard { input_size, tuple_indices, rams }
    }

    fn address_for_ram(&self, ram_idx: usize, input: &[u8]) -> usize {
        let mut addr = 0usize;
        for &bit_idx in &self.tuple_indices[ram_idx] {
            addr = (addr << 1) | (input[bit_idx] as usize);
        }
        addr
    }

    /// Trains on (input, target): increments counters and accumulates
    /// the target value into every RAM's addressed position.
    pub fn train(&mut self, input: &[u8], target: f64) {
        assert_eq!(input.len(), self.input_size, "input size mismatch");
        for i in 0..self.rams.len() {
            let addr = self.address_for_ram(i, input);
            self.rams[i].train(addr, target);
        }
    }

    /// Predicts by averaging partial means across RAMs that have seen
    /// this address enough times.
    pub fn predict(&self, input: &[u8]) -> Option<f64> {
        assert_eq!(input.len(), self.input_size, "input size mismatch");
        let mut total = 0.0;
        let mut count = 0;
        for i in 0..self.rams.len() {
            let addr = self.address_for_ram(i, input);
            if let Some(p) = self.rams[i].predict(addr) {
                total += p;
                count += 1;
            }
        }
        if count > 0 { Some(total / count as f64) } else { None }
    }

    /// Saves all RAM counters and target-sum accumulators to `path`.
    pub fn save_to_file(&self, path: impl AsRef<Path>, format: FileFormat) -> IoResult<()> {
        save_to_file(self, path, format)
    }

    /// Loads a previously saved RegressionWiSARD model from `path`.
    pub fn load_from_file(path: impl AsRef<Path>, format: FileFormat) -> IoResult<Self> {
        load_from_file(path, format)
    }
}