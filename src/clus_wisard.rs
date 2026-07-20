use crate::discriminator::Discriminator;
use crate::persist::{load_from_file, save_to_file, FileFormat};
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Result as IoResult;
use std::path::Path;

/// ClusWiSARD: allows multiple discriminators ("clusters") per class,
/// spawning new ones when an existing cluster no longer matches well.
#[derive(Serialize, Deserialize)]
pub struct ClusWisard {
    address_size: usize,
    input_size: usize,
    min_score: f64,
    threshold: u32,
    discriminator_limit: usize,
    ignore_zero: bool,
    bleaching_enabled: bool,
    confidence_threshold: f64,
    mapping: Vec<usize>,
    clusters: HashMap<String, Vec<(Discriminator, u32)>>,
}

impl ClusWisard {
    /// `input_size`: length of the binary-encoded input vector (retina size).
    /// `address_size`: bits per RAM addressing bus (address space = 2^address_size).
    /// `min_score`: minimum similarity score required to reuse an existing
    ///   cluster instead of spawning a new one.
    /// `threshold`: max training cycles a cluster can absorb before a new
    ///   one is spawned for that class.
    /// `discriminator_limit`: max number of clusters allowed per class.
    /// `confidence_threshold`: min score gap used when bleaching is enabled.
    /// `bleaching_enabled`: false = fixed threshold=1 scoring; true = adaptive
    ///   binary-search bleaching per cluster.
    /// `ignore_zero`: skip training/counting on the all-zero tuple address.
    pub fn new(
        input_size: usize,
        address_size: usize,
        min_score: f64,
        threshold: u32,
        discriminator_limit: usize,
        confidence_threshold: f64,
        bleaching_enabled: bool,
        ignore_zero: bool,
    ) -> Self {
        assert!(
            address_size > 0 && address_size <= input_size,
            "address_size must be in (0, input_size]"
        );

        let mut mapping: Vec<usize> = (0..input_size).collect();
        mapping.shuffle(&mut thread_rng());

        ClusWisard {
            address_size,
            input_size,
            min_score,
            threshold,
            discriminator_limit,
            ignore_zero,
            bleaching_enabled,
            confidence_threshold,
            mapping,
            clusters: HashMap::new(),
        }
    }

    /// Builds RAM tuple index groups from an explicit mapping/address_size,
    /// avoiding any borrow of `self` as a whole. This lets it be called
    /// safely even while another field of `self` (e.g. `clusters`) is
    /// mutably borrowed elsewhere in the same function.
    fn build_tuple_indices_from(mapping: &[usize], address_size: usize) -> Vec<Vec<usize>> {
        mapping.chunks(address_size).map(|c| c.to_vec()).collect()
    }

    /// Trains on (input, label). Reuses the best-matching existing cluster
    /// for that class if its score clears `min_score` and it hasn't
    /// exceeded `threshold` training cycles; otherwise spawns a new
    /// cluster, capped at `discriminator_limit`.
    pub fn train(&mut self, input: &[u8], label: &str) {
        assert_eq!(input.len(), self.input_size, "input size mismatch");

        let ignore_zero = self.ignore_zero;
        let mapping = self.mapping.clone();
        let address_size = self.address_size;
        let min_score = self.min_score;
        let threshold = self.threshold;
        let discriminator_limit = self.discriminator_limit;

        let bucket = self.clusters.entry(label.to_string()).or_insert_with(Vec::new);

        if bucket.is_empty() {
            let tuple_indices = Self::build_tuple_indices_from(&mapping, address_size);
            bucket.push((Discriminator::new(tuple_indices, ignore_zero), 0));
        }

        let mut best_idx = 0;
        let mut best_score = -1.0;
        for (i, (disc, _)) in bucket.iter().enumerate() {
            let addrs = disc.precompute_addresses(input);
            let score = disc.score_at(&addrs, 1) as f64 / disc.rams.len().max(1) as f64;
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
            bucket.push((Discriminator::new(tuple_indices, ignore_zero), 0));
            let idx = bucket.len() - 1;
            bucket[idx].0.train(input);
            bucket[idx].1 += 1;
        } else {
            bucket[best_idx].0.train(input);
            bucket[best_idx].1 += 1;
        }
    }

    /// Classifies by taking, per class, the max score across its
    /// clusters, then choosing the best class overall.
    pub fn classify(&self, input: &[u8]) -> Option<(String, f64)> {
        assert_eq!(input.len(), self.input_size, "input size mismatch");
        let mut best_label = String::new();
        let mut best_score = -1.0;

        for (label, bucket) in &self.clusters {
            for (disc, _) in bucket {
                let addrs = disc.precompute_addresses(input);
                let threshold = if self.bleaching_enabled {
                    self.bleach_threshold(disc, &addrs)
                } else {
                    1
                };
                let score = disc.score_at(&addrs, threshold) as f64 / disc.rams.len().max(1) as f64;
                if score > best_score {
                    best_score = score;
                    best_label = label.clone();
                }
            }
        }

        if best_label.is_empty() {
            None
        } else {
            Some((best_label, best_score))
        }
    }

    fn bleach_threshold(&self, disc: &Discriminator, addrs: &[usize]) -> u16 {
        let max_c = disc.max_count(addrs).max(1);
        let mut lo = 1u16;
        let mut hi = max_c;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if disc.score_at(addrs, mid) as f64 / disc.rams.len().max(1) as f64 >= self.confidence_threshold {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// Saves all clusters (discriminators and their RAM counters) for
    /// every class to `path` in the given format.
    pub fn save_to_file(&self, path: impl AsRef<Path>, format: FileFormat) -> IoResult<()> {
        save_to_file(self, path, format)
    }

    /// Loads a previously saved ClusWiSARD model from `path`.
    pub fn load_from_file(path: impl AsRef<Path>, format: FileFormat) -> IoResult<Self> {
        load_from_file(path, format)
    }
}