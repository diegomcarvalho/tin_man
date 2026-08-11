//! `BloomWisard`: a `Wisard`-equivalent classifier whose RAM nodes are
//! backed by compact counting Bloom filters instead of exact,
//! fully-allocated address tables.
//!
//! Use this when `address_size` is large enough that an exact RAM
//! (`2^address_size` counters) would be impractically large, and a
//! small, tunable false-positive rate is an acceptable trade-off for
//! drastically reduced memory use.

use crate::bloom_discriminator::BloomDiscriminator;
use crate::persist::{load_from_file, save_to_file, FileFormat};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::io::Result as IoResult;
use std::path::Path;

#[derive(Serialize, Deserialize)]
pub struct BloomWisard {
    address_size: usize,
    input_size: usize,
    mapping: Vec<usize>,
    bloom_size: usize,
    num_hashes: usize,
    seed: u64,
    labels: Vec<String>,
    discriminators: Vec<BloomDiscriminator>,
    confidence_threshold: f64,
    bleaching_enabled: bool,
    ignore_zero: bool,
    parallel: bool,
}

impl BloomWisard {
    /// Creates a new, untrained `BloomWisard`.
    ///
    /// # Parameters
    ///
    /// - `input_size`, `address_size`, `confidence_threshold`,
    ///   `bleaching_enabled`, `ignore_zero`, `parallel`: identical
    ///   meaning to [`crate::Wisard::new`].
    /// - `bloom_size`: number of counters allocated per RAM node.
    ///   Should be substantially smaller than `2^address_size`; a
    ///   good starting rule of thumb is `bloom_size` roughly
    ///   `4-10x` the expected number of distinct addresses a RAM will
    ///   see during training.
    /// - `num_hashes`: number of independent hash functions per RAM
    ///   (typically `3`-`7`). More hashes reduce false positives per
    ///   query but increase both train and query cost linearly.
    ///
    /// # Panics
    ///
    /// Panics if `address_size` is `0` or greater than `input_size`,
    /// or if `bloom_size`/`num_hashes` is `0`.
    pub fn new(
        input_size: usize,
        address_size: usize,
        bloom_size: usize,
        num_hashes: usize,
        confidence_threshold: f64,
        bleaching_enabled: bool,
        ignore_zero: bool,
        parallel: bool,
    ) -> Self {
        assert!(address_size > 0 && address_size <= input_size, "address_size must be in (0, input_size]");
        let mut mapping: Vec<usize> = (0..input_size).collect();
        mapping.shuffle(&mut thread_rng());
        let seed = thread_rng().gen::<u64>();

        Self::from_parts(input_size, address_size, mapping, bloom_size, num_hashes, seed, confidence_threshold, bleaching_enabled, ignore_zero, parallel)
    }

    /// Deterministic variant of [`BloomWisard::new`], seeding both the
    /// retina mapping shuffle and all Bloom RAM hash functions.
    pub fn new_with_seed(
        input_size: usize,
        address_size: usize,
        bloom_size: usize,
        num_hashes: usize,
        confidence_threshold: f64,
        bleaching_enabled: bool,
        ignore_zero: bool,
        parallel: bool,
        seed: u64,
    ) -> Self {
        assert!(address_size > 0 && address_size <= input_size, "address_size must be in (0, input_size]");
        let mut rng = StdRng::seed_from_u64(seed);
        let mut mapping: Vec<usize> = (0..input_size).collect();
        mapping.shuffle(&mut rng);

        Self::from_parts(input_size, address_size, mapping, bloom_size, num_hashes, seed, confidence_threshold, bleaching_enabled, ignore_zero, parallel)
    }

    fn from_parts(
        input_size: usize,
        address_size: usize,
        mapping: Vec<usize>,
        bloom_size: usize,
        num_hashes: usize,
        seed: u64,
        confidence_threshold: f64,
        bleaching_enabled: bool,
        ignore_zero: bool,
        parallel: bool,
    ) -> Self {
        assert!(bloom_size > 0 && num_hashes > 0, "bloom_size and num_hashes must be > 0");
        BloomWisard {
            address_size,
            input_size,
            mapping,
            bloom_size,
            num_hashes,
            seed,
            labels: Vec::new(),
            discriminators: Vec::new(),
            confidence_threshold,
            bleaching_enabled,
            ignore_zero,
            parallel,
        }
    }

    fn build_tuple_indices(&self) -> Vec<Vec<usize>> {
        self.mapping.chunks(self.address_size).map(|c| c.to_vec()).collect()
    }

    fn label_id(&mut self, label: &str) -> usize {
        if let Some(pos) = self.labels.iter().position(|l| l == label) {
            pos
        } else {
            self.labels.push(label.to_string());
            let tuple_indices = self.build_tuple_indices();
            let disc_seed = self.seed.wrapping_add(self.labels.len() as u64 * 0x1000193);
            self.discriminators.push(BloomDiscriminator::new(tuple_indices, self.bloom_size, self.num_hashes, disc_seed, self.ignore_zero));
            self.labels.len() - 1
        }
    }

    /// Trains on a single (input, label) pair. See [`crate::Wisard::train`].
    pub fn train(&mut self, input: &[u8], label: &str) {
        assert_eq!(input.len(), self.input_size, "input size mismatch");
        let id = self.label_id(label);
        self.discriminators[id].train(input);
    }

    /// Classifies `input`. See [`crate::Wisard::classify`] — behavior
    /// is identical except RAM lookups are Bloom-filter approximate
    /// rather than exact, so scores may be slightly inflated by
    /// hash-collision false positives, especially on undertrained
    /// (small `bloom_size`) models.
    pub fn classify(&self, input: &[u8]) -> Option<(String, f64)> {
        assert_eq!(input.len(), self.input_size, "input size mismatch");
        if self.discriminators.is_empty() {
            return None;
        }

        let addr_cache: Vec<Vec<usize>> = if self.parallel {
            self.discriminators.par_iter().map(|d| d.precompute_addresses(input)).collect()
        } else {
            self.discriminators.iter().map(|d| d.precompute_addresses(input)).collect()
        };

        if !self.bleaching_enabled {
            return self.classify_fixed_threshold(&addr_cache, 1);
        }
        self.classify_with_bleaching(&addr_cache)
    }

    fn classify_fixed_threshold(&self, addr_cache: &[Vec<usize>], threshold: u16) -> Option<(String, f64)> {
        let scored = |i: usize, disc: &BloomDiscriminator, addrs: &Vec<usize>| {
            let score = disc.score_at(addrs, threshold) as f64 / disc.rams.len().max(1) as f64;
            (i, score)
        };

        let (best_idx, best_score) = if self.parallel {
            self.discriminators
                .par_iter()
                .zip(addr_cache.par_iter())
                .enumerate()
                .map(|(i, (d, a))| scored(i, d, a))
                .reduce(|| (0usize, -1.0f64), |a, b| if b.1 > a.1 { b } else { a })
        } else {
            self.discriminators
                .iter()
                .zip(addr_cache.iter())
                .enumerate()
                .map(|(i, (d, a))| scored(i, d, a))
                .fold((0usize, -1.0f64), |a, b| if b.1 > a.1 { b } else { a })
        };

        Some((self.labels[best_idx].clone(), best_score))
    }

    fn classify_with_bleaching(&self, addr_cache: &[Vec<usize>]) -> Option<(String, f64)> {
        let global_max = if self.parallel {
            addr_cache.par_iter().zip(self.discriminators.par_iter()).map(|(a, d)| d.max_count(a)).max().unwrap_or(0).max(1)
        } else {
            addr_cache.iter().zip(self.discriminators.iter()).map(|(a, d)| d.max_count(a)).max().unwrap_or(0).max(1)
        };

        let mut lo: u16 = 1;
        let hi: u16 = global_max;

        let best: (usize, f64) = loop {
            let mid = lo + (hi - lo) / 2;
            let mut scores: Vec<(usize, f64)> = if self.parallel {
                self.discriminators
                    .par_iter()
                    .zip(addr_cache.par_iter())
                    .map(|(d, a)| d.score_at(a, mid) as f64 / d.rams.len().max(1) as f64)
                    .enumerate()
                    .collect()
            } else {
                self.discriminators
                    .iter()
                    .zip(addr_cache.iter())
                    .map(|(d, a)| d.score_at(a, mid) as f64 / d.rams.len().max(1) as f64)
                    .enumerate()
                    .collect()
            };

            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top = scores[0];
            let gap = if scores.len() > 1 { top.1 - scores[1].1 } else { top.1 };

            if gap >= self.confidence_threshold || lo >= hi {
                break top;
            }
            lo = mid + 1;
        };

        Some((self.labels[best.0].clone(), best.1))
    }

    /// Total memory used by all discriminators' Bloom RAMs, useful for
    /// comparing directly against an equivalent exact `Wisard`'s
    /// `num_rams * 2^address_size * size_of::<u16>()`.
    pub fn memory_bytes(&self) -> usize {
        self.discriminators.iter().map(|d| d.memory_bytes()).sum()
    }

    pub fn save_to_file(&self, path: impl AsRef<Path>, format: FileFormat) -> IoResult<()> {
        save_to_file(self, path, format)
    }

    pub fn load_from_file(path: impl AsRef<Path>, format: FileFormat) -> IoResult<Self> {
        load_from_file(path, format)
    }
}