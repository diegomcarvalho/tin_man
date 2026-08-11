//! `BloomRegressionWisard`: a `RegressionWisard`-equivalent whose RAM
//! nodes are Bloom-backed `BloomRegressionRam` accumulators, giving
//! the same continuous-value prediction interface with a compact
//! memory footprint for large `address_size`.

use crate::bloom_regression_ram::BloomRegressionRam;
use crate::persist::{load_from_file, save_to_file, FileFormat};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::io::Result as IoResult;
use std::path::Path;

#[derive(Serialize, Deserialize)]
pub struct BloomRegressionWisard {
    address_size: usize,
    input_size: usize,
    mapping: Vec<usize>,
    tuple_indices: Vec<Vec<usize>>,
    rams: Vec<BloomRegressionRam>,
    min_zero: u32,
}

impl BloomRegressionWisard {
    /// Creates a new, untrained `BloomRegressionWisard`.
    ///
    /// - `min_zero`: minimum number of RAMs that must produce a
    ///   prediction (i.e. have been visited) for `predict` to return
    ///   `Some`; below this, the address space is considered too
    ///   sparsely trained to trust.
    pub fn new(input_size: usize, address_size: usize, bloom_size: usize, num_hashes: usize, min_zero: u32) -> Self {
        assert!(address_size > 0 && address_size <= input_size, "address_size must be in (0, input_size]");
        let mut mapping: Vec<usize> = (0..input_size).collect();
        mapping.shuffle(&mut thread_rng());
        let seed = thread_rng().gen::<u64>();
        Self::from_parts(input_size, address_size, mapping, bloom_size, num_hashes, seed, min_zero)
    }

    pub fn new_with_seed(input_size: usize, address_size: usize, bloom_size: usize, num_hashes: usize, min_zero: u32, seed: u64) -> Self {
        assert!(address_size > 0 && address_size <= input_size, "address_size must be in (0, input_size]");
        let mut rng = StdRng::seed_from_u64(seed);
        let mut mapping: Vec<usize> = (0..input_size).collect();
        mapping.shuffle(&mut rng);
        Self::from_parts(input_size, address_size, mapping, bloom_size, num_hashes, seed, min_zero)
    }

    fn from_parts(input_size: usize, address_size: usize, mapping: Vec<usize>, bloom_size: usize, num_hashes: usize, seed: u64, min_zero: u32) -> Self {
        let tuple_indices: Vec<Vec<usize>> = mapping.chunks(address_size).map(|c| c.to_vec()).collect();
        let rams: Vec<BloomRegressionRam> = (0..tuple_indices.len())
            .map(|i| BloomRegressionRam::new(bloom_size, num_hashes, seed.wrapping_add(i as u64)))
            .collect();

        BloomRegressionWisard { address_size, input_size, mapping, tuple_indices, rams, min_zero }
    }

    fn addresses(&self, input: &[u8]) -> Vec<usize> {
        self.tuple_indices
            .iter()
            .map(|positions| positions.iter().fold(0usize, |acc, &pos| (acc << 1) | (input[pos] as usize)))
            .collect()
    }

    /// Trains on a single (input, target) pair.
    ///
    /// # Panics
    ///
    /// Panics if `input.len()` does not equal `input_size`.
    pub fn train(&mut self, input: &[u8], target: f64) {
        assert_eq!(input.len(), self.input_size, "input size mismatch");
        let addrs = self.addresses(input);
        for (ram, addr) in self.rams.iter_mut().zip(addrs.iter()) {
            ram.train(*addr, target);
        }
    }

    /// Predicts a continuous value for `input`, averaging every RAM's
    /// individual estimate. Returns `None` if fewer than `min_zero`
    /// RAMs have a usable estimate.
    ///
    /// # Panics
    ///
    /// Panics if `input.len()` does not equal `input_size`.
    pub fn predict(&self, input: &[u8]) -> Option<f64> {
        assert_eq!(input.len(), self.input_size, "input size mismatch");
        let addrs = self.addresses(input);
        let estimates: Vec<f64> = self.rams.iter().zip(addrs.iter()).filter_map(|(ram, &addr)| ram.predict(addr)).collect();

        if estimates.len() < self.min_zero as usize || estimates.is_empty() {
            return None;
        }
        Some(estimates.iter().sum::<f64>() / estimates.len() as f64)
    }

    pub fn memory_bytes(&self) -> usize {
        self.rams.iter().map(|r| r.memory_bytes()).sum()
    }

    pub fn save_to_file(&self, path: impl AsRef<Path>, format: FileFormat) -> IoResult<()> {
        save_to_file(self, path, format)
    }

    pub fn load_from_file(path: impl AsRef<Path>, format: FileFormat) -> IoResult<Self> {
        load_from_file(path, format)
    }
}