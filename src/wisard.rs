/// Standard WiSARD classifier: one discriminator per class, sharing a
/// single randomized retina-to-RAM mapping across all classes.
///
/// # How it works
///
/// Each class gets a [`Discriminator`](crate::discriminator) — a set of
/// RAM nodes, each fed by a fixed subset of input bits (a "tuple").
/// During training, the discriminator's RAMs record which addresses
/// were seen. During classification, each discriminator's RAMs vote on
/// whether they recognize the input, and the class with the highest
/// score wins.
///
/// # Example
///
/// ```
/// use tin_man::Wisard;
///
/// let mut w = Wisard::new(8, 4, 0.1, true, false);
/// w.train(&, "cold");[1]
/// w.train(&, "hot");[1]
///
/// let (label, _confidence) = w.classify(&).unwrap();[1]
/// assert_eq!(label, "hot");
/// ```

use crate::discriminator::Discriminator;
use crate::persist::{load_from_file, save_to_file, FileFormat};
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use std::io::Result as IoResult;
use std::path::Path;

/// Standard WiSARD classifier: one discriminator per class, sharing a
/// single randomized retina-to-RAM mapping across all classes.
#[derive(Serialize, Deserialize)]
pub struct Wisard {
    address_size: usize,
    input_size: usize,
    mapping: Vec<usize>,
    labels: Vec<String>,
    discriminators: Vec<Discriminator>,
    confidence_threshold: f64,
    bleaching_enabled: bool,
    ignore_zero: bool,
}

impl Wisard {
    /// Creates a new, untrained WiSARD model.
    ///
    /// # Parameters
    ///
    /// - `input_size`: length of the binary-encoded input vector (retina size).
    /// - `address_size`: number of input bits routed into each RAM
    ///   (address space = 2^`address_size`). Must satisfy
    ///   `0 < address_size <= input_size`.
    /// - `confidence_threshold`: minimum score gap between the top two
    ///   classes required to stop the bleaching search. Only used when
    ///   `bleaching_enabled` is `true`.
    /// - `bleaching_enabled`: if `false`, classification uses a fixed
    ///   threshold of 1 (plain binary WiSARD). If `true`, an adaptive
    ///   binary-search bleaching threshold is used to resolve ties and
    ///   reduce overtraining sensitivity.
    /// - `ignore_zero`: if `true`, RAMs skip training/counting on the
    ///   all-zero tuple address, preventing a common "background"
    ///   pattern from dominating RAM statistics.
    ///
    /// # Panics
    ///
    /// Panics if `address_size` is `0` or greater than `input_size`.
    pub fn new(
        input_size: usize,
        address_size: usize,
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

        Wisard {
            address_size,
            input_size,
            mapping,
            labels: Vec::new(),
            discriminators: Vec::new(),
            confidence_threshold,
            bleaching_enabled,
            ignore_zero,
        }
    }

    fn build_tuple_indices(&self) -> Vec<Vec<usize>> {
        debug_assert_eq!(self.mapping.len(), self.input_size);
        self.mapping.chunks(self.address_size).map(|c| c.to_vec()).collect()
    }

    fn label_id(&mut self, label: &str) -> usize {
        if let Some(pos) = self.labels.iter().position(|l| l == label) {
            pos
        } else {
            self.labels.push(label.to_string());
            let tuple_indices = self.build_tuple_indices();
            self.discriminators.push(Discriminator::new(tuple_indices, self.ignore_zero));
            self.labels.len() - 1
        }
    }

    /// Trains the model on a single (input, label) pair.
    ///
    /// If `label` has not been seen before, a new discriminator is
    /// created for it lazily — you do not need to pre-declare classes.
    ///
    /// # Panics
    ///
    /// Panics if `input.len()` does not equal `input_size`.
    pub fn train(&mut self, input: &[u8], label: &str) {
        assert_eq!(input.len(), self.input_size, "input size mismatch");
        let id = self.label_id(label);
        self.discriminators[id].train(input);
    }

    /// Classifies `input`, returning the predicted label and a
    /// confidence score in `[0.0, 1.0]` (fraction of RAMs firing for
    /// the winning discriminator).
    ///
    /// Returns `None` if the model has not been trained on any class.
    ///
    /// # Panics
    ///
    /// Panics if `input.len()` does not equal `input_size`.
    pub fn classify(&self, input: &[u8]) -> Option<(String, f64)> {
        assert_eq!(input.len(), self.input_size, "input size mismatch");
        if self.discriminators.is_empty() {
            return None;
        }

        let addr_cache: Vec<Vec<usize>> =
            self.discriminators.iter().map(|d| d.precompute_addresses(input)).collect();

        if !self.bleaching_enabled {
            return self.classify_fixed_threshold(&addr_cache, 1);
        }
        self.classify_with_bleaching(&addr_cache)
    }

    fn classify_fixed_threshold(&self, addr_cache: &[Vec<usize>], threshold: u16) -> Option<(String, f64)> {
        let mut best_idx = 0;
        let mut best_score = -1.0;
        for (i, (disc, addrs)) in self.discriminators.iter().zip(addr_cache.iter()).enumerate() {
            let score = disc.score_at(addrs, threshold) as f64 / disc.rams.len().max(1) as f64;
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }
        Some((self.labels[best_idx].clone(), best_score))
    }

    fn classify_with_bleaching(&self, addr_cache: &[Vec<usize>]) -> Option<(String, f64)> {
        let global_max = addr_cache
            .iter()
            .zip(self.discriminators.iter())
            .map(|(addrs, disc)| disc.max_count(addrs))
            .max()
            .unwrap_or(0)
            .max(1);

        let mut lo: u16 = 1;
        let hi: u16 = global_max;

        let best: (usize, f64) = loop {
            let mid = lo + (hi - lo) / 2;
            let mut scores: Vec<(usize, f64)> = self
                .discriminators
                .iter()
                .zip(addr_cache.iter())
                .map(|(disc, addrs)| disc.score_at(addrs, mid) as f64 / disc.rams.len().max(1) as f64)
                .enumerate()
                .collect();

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

    /// Saves the trained model (retina mapping, labels, and all RAM
    /// counters) to `path` in the given [`FileFormat`].
    pub fn save_to_file(&self, path: impl AsRef<Path>, format: FileFormat) -> IoResult<()> {
        save_to_file(self, path, format)
    }

    /// Loads a previously saved model from `path`.
    pub fn load_from_file(path: impl AsRef<Path>, format: FileFormat) -> IoResult<Self> {
        load_from_file(path, format)
    }
}