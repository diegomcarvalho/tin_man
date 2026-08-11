//! `BloomClusWisard`: a `ClusWisard`-equivalent that gives each class
//! multiple Bloom-backed discriminators ("clusters"), so that
//! heterogeneous sub-patterns within one class don't all get forced
//! into the same set of Bloom RAMs — reducing the collision rate any
//! single cluster's RAMs experience.

use crate::bloom_discriminator::BloomDiscriminator;
use crate::persist::{load_from_file, save_to_file, FileFormat};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Result as IoResult;
use std::path::Path;

#[derive(Serialize, Deserialize)]
pub struct BloomClusWisard {
    address_size: usize,
    input_size: usize,
    mapping: Vec<usize>,
    bloom_size: usize,
    num_hashes: usize,
    seed: u64,
    min_score: f64,
    discriminator_limit: usize,
    ignore_zero: bool,
    clusters: HashMap<String, Vec<BloomDiscriminator>>,
}

impl BloomClusWisard {
    /// Creates a new, untrained `BloomClusWisard`.
    ///
    /// - `min_score`: minimum fraction of a cluster's RAMs that must
    ///   recognize an input for that cluster to be reused; below this,
    ///   a new cluster is created for the class (up to
    ///   `discriminator_limit`).
    /// - `discriminator_limit`: maximum number of clusters (bloom
    ///   discriminators) per class.
    /// - `bloom_size`, `num_hashes`: see [`crate::BloomWisard::new`].
    pub fn new(
        input_size: usize,
        address_size: usize,
        bloom_size: usize,
        num_hashes: usize,
        min_score: f64,
        discriminator_limit: usize,
        ignore_zero: bool,
    ) -> Self {
        assert!(address_size > 0 && address_size <= input_size, "address_size must be in (0, input_size]");
        let mut mapping: Vec<usize> = (0..input_size).collect();
        mapping.shuffle(&mut thread_rng());
        let seed = thread_rng().gen::<u64>();

        BloomClusWisard {
            address_size,
            input_size,
            mapping,
            bloom_size,
            num_hashes,
            seed,
            min_score,
            discriminator_limit,
            ignore_zero,
            clusters: HashMap::new(),
        }
    }

    pub fn new_with_seed(
        input_size: usize,
        address_size: usize,
        bloom_size: usize,
        num_hashes: usize,
        min_score: f64,
        discriminator_limit: usize,
        ignore_zero: bool,
        seed: u64,
    ) -> Self {
        assert!(address_size > 0 && address_size <= input_size, "address_size must be in (0, input_size]");
        let mut rng = StdRng::seed_from_u64(seed);
        let mut mapping: Vec<usize> = (0..input_size).collect();
        mapping.shuffle(&mut rng);

        BloomClusWisard {
            address_size,
            input_size,
            mapping,
            bloom_size,
            num_hashes,
            seed,
            min_score,
            discriminator_limit,
            ignore_zero,
            clusters: HashMap::new(),
        }
    }

    fn build_tuple_indices(&self) -> Vec<Vec<usize>> {
        self.mapping.chunks(self.address_size).map(|c| c.to_vec()).collect()
    }

    fn new_discriminator(&self, disc_id: u64) -> BloomDiscriminator {
        let tuple_indices = self.build_tuple_indices();
        let disc_seed = self.seed.wrapping_add(disc_id.wrapping_mul(0x1000193));
        BloomDiscriminator::new(tuple_indices, self.bloom_size, self.num_hashes, disc_seed, self.ignore_zero)
    }

        /// Trains on a single (input, label) pair. If the label's best
    /// matching existing cluster scores below `min_score` and the
    /// cluster count for that label is under `discriminator_limit`, a
    /// fresh cluster is created and trained instead of reusing an
    /// existing one.
    pub fn train(&mut self, input: &[u8], label: &str) {
        assert_eq!(input.len(), self.input_size, "input size mismatch");

        let disc_count_before = self.clusters.get(label).map(|v| v.len()).unwrap_or(0) as u64;
        let is_new_label = !self.clusters.contains_key(label);

        if is_new_label {
            let mut new_disc = self.new_discriminator(disc_count_before);
            new_disc.train(input);
            self.clusters.insert(label.to_string(), vec![new_disc]);
            return;
        }

        let (best_score, clusters_len) = {
            let clusters = self.clusters.get(label).unwrap();
            let best_score = clusters
                .iter()
                .map(|d| {
                    let addrs = d.precompute_addresses(input);
                    d.score_at(&addrs, 1) as f64 / d.rams.len().max(1) as f64
                })
                .fold(f64::MIN, f64::max);
            (best_score, clusters.len())
        };

        if best_score < self.min_score && clusters_len < self.discriminator_limit {
            let mut new_disc = self.new_discriminator(disc_count_before + clusters_len as u64);
            new_disc.train(input);
            self.clusters.get_mut(label).unwrap().push(new_disc);
        } else {
            let best_idx = {
                let clusters = self.clusters.get(label).unwrap();
                clusters
                    .iter()
                    .enumerate()
                    .map(|(i, d)| {
                        let addrs = d.precompute_addresses(input);
                        (i, d.score_at(&addrs, 1) as f64 / d.rams.len().max(1) as f64)
                    })
                    .fold((0usize, f64::MIN), |a, b| if b.1 > a.1 { b } else { a })
                    .0
            };
            self.clusters.get_mut(label).unwrap()[best_idx].train(input);
        }
    }

    /// Classifies `input` by taking, for each label, the *maximum*
    /// score across all of that label's clusters, then returning the
    /// label with the highest such max.
    pub fn classify(&self, input: &[u8]) -> Option<(String, f64)> {
        assert_eq!(input.len(), self.input_size, "input size mismatch");
        if self.clusters.is_empty() {
            return None;
        }

        let mut best_label: Option<String> = None;
        let mut best_score = -1.0f64;

        for (label, discs) in self.clusters.iter() {
            let label_best = discs
                .iter()
                .map(|d| {
                    let addrs = d.precompute_addresses(input);
                    d.score_at(&addrs, 1) as f64 / d.rams.len().max(1) as f64
                })
                .fold(f64::MIN, f64::max);

            if label_best > best_score {
                best_score = label_best;
                best_label = Some(label.clone());
            }
        }

        best_label.map(|l| (l, best_score))
    }

    /// Total memory used by all clusters' Bloom RAMs across all labels.
    pub fn memory_bytes(&self) -> usize {
        self.clusters.values().flatten().map(|d| d.memory_bytes()).sum()
    }

    pub fn save_to_file(&self, path: impl AsRef<Path>, format: FileFormat) -> IoResult<()> {
        save_to_file(self, path, format)
    }

    pub fn load_from_file(path: impl AsRef<Path>, format: FileFormat) -> IoResult<Self> {
        load_from_file(path, format)
    }
}