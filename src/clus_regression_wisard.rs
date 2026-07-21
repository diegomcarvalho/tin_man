use crate::persist::{load_from_file, save_to_file, FileFormat};
use crate::regression_discriminator::RegressionDiscriminator;
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Result as IoResult;
use std::path::Path;

/// ClusRegressionWisard: combines ClusWiSARD's dynamic cluster-spawning
/// with Regression WiSARD's counter+sum RAMs. Groups (analogous to
/// classes) can hold multiple regression clusters to model
/// heterogeneous target distributions within the same group.
#[derive(Serialize, Deserialize)]
pub struct ClusRegressionWisard {
    address_size: usize,
    input_size: usize,
    min_score: f64,
    threshold: u32,
    discriminator_limit: usize,
    min_zero: u32,
    mapping: Vec<usize>,
    groups: HashMap<String, Vec<(RegressionDiscriminator, u32)>>,
}

impl ClusRegressionWisard {
    /// `input_size`: length of the binary-encoded input vector (retina size).
    /// `address_size`: bits per RAM addressing bus (address space = 2^address_size).
    /// `min_score`: minimum match score required to reuse an existing
    ///   cluster instead of spawning a new one.
    /// `threshold`: max training cycles a cluster can absorb before a new
    ///   one is spawned for that group.
    /// `discriminator_limit`: max number of clusters allowed per group.
    /// `min_zero`: minimum access count required for a RAM's address to
    ///   contribute to a prediction or match score.
    pub fn new(
        input_size: usize,
        address_size: usize,
        min_score: f64,
        threshold: u32,
        discriminator_limit: usize,
        min_zero: u32,
    ) -> Self {
        assert!(
            address_size > 0 && address_size <= input_size,
            "address_size must be in (0, input_size]"
        );

        let mut mapping: Vec<usize> = (0..input_size).collect();
        mapping.shuffle(&mut thread_rng());

        ClusRegressionWisard {
            address_size,
            input_size,
            min_score,
            threshold,
            discriminator_limit,
            min_zero,
            mapping,
            groups: HashMap::new(),
        }
    }

    fn build_tuple_indices_from(mapping: &[usize], address_size: usize) -> Vec<Vec<usize>> {
        mapping.chunks(address_size).map(|c| c.to_vec()).collect()
    }

    /// Trains on (input, group, target). Reuses the best-matching existing
    /// cluster within `group` if its match score clears `min_score` and it
    /// hasn't exceeded `threshold` training cycles; otherwise spawns a new
    /// cluster, capped at `discriminator_limit`.
    pub fn train(&mut self, input: &[u8], group: &str, target: f64) {
        assert_eq!(input.len(), self.input_size, "input size mismatch");

        let mapping = self.mapping.clone();
        let address_size = self.address_size;
        let min_zero = self.min_zero;
        let min_score = self.min_score;
        let threshold = self.threshold;
        let discriminator_limit = self.discriminator_limit;

        let bucket = self.groups.entry(group.to_string()).or_insert_with(Vec::new);

        if bucket.is_empty() {
            let tuple_indices = Self::build_tuple_indices_from(&mapping, address_size);
            bucket.push((RegressionDiscriminator::new(tuple_indices, min_zero), 0));
        }

        let mut best_idx = 0;
        let mut best_score = -1.0;
        for (i, (disc, _)) in bucket.iter().enumerate() {
            let addrs = disc.precompute_addresses(input);
            let score = disc.match_score(&addrs);
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }

        let (_, count) = &bucket[best_idx];
        let needs_new =
            (best_score < min_score || *count >= threshold) && bucket.len() < discriminator_limit;

        if needs_new {
            let tuple_indices = Self::build_tuple_indices_from(&mapping, address_size);
            bucket.push((RegressionDiscriminator::new(tuple_indices, min_zero), 0));
            let idx = bucket.len() - 1;
            bucket[idx].0.train(input, target);
            bucket[idx].1 += 1;
        } else {
            bucket[best_idx].0.train(input, target);
            bucket[best_idx].1 += 1;
        }
    }

    /// Predicts by finding the best-matching cluster across all groups
    /// (highest match score) and returning its regression prediction.
    /// Returns None if no cluster has a usable match.
    pub fn predict(&self, input: &[u8]) -> Option<f64> {
        assert_eq!(input.len(), self.input_size, "input size mismatch");

        let mut best_score = -1.0;
        let mut best_prediction: Option<f64> = None;

        for bucket in self.groups.values() {
            for (disc, _) in bucket {
                let addrs = disc.precompute_addresses(input);
                let score = disc.match_score(&addrs);
                if score > best_score {
                    if let Some(pred) = disc.predict(&addrs) {
                        best_score = score;
                        best_prediction = Some(pred);
                    }
                }
            }
        }

        best_prediction
    }

    /// Predicts within a specific group only, useful when the group is
    /// known at inference time (e.g. multi-task regression scenarios).
    pub fn predict_in_group(&self, input: &[u8], group: &str) -> Option<f64> {
        assert_eq!(input.len(), self.input_size, "input size mismatch");

        let bucket = self.groups.get(group)?;
        let mut best_score = -1.0;
        let mut best_prediction: Option<f64> = None;

        for (disc, _) in bucket {
            let addrs = disc.precompute_addresses(input);
            let score = disc.match_score(&addrs);
            if score > best_score {
                if let Some(pred) = disc.predict(&addrs) {
                    best_score = score;
                    best_prediction = Some(pred);
                }
            }
        }

        best_prediction
    }

    /// Saves all groups and clusters (RegressionDiscriminators and their
    /// RAM counters/sums) to `path` in the given format.
    pub fn save_to_file(&self, path: impl AsRef<Path>, format: FileFormat) -> IoResult<()> {
        save_to_file(self, path, format)
    }

    /// Loads a previously saved ClusRegressionWisard model from `path`.
    pub fn load_from_file(path: impl AsRef<Path>, format: FileFormat) -> IoResult<Self> {
        load_from_file(path, format)
    }
}