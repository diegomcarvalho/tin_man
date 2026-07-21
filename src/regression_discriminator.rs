use crate::ram::RegressionRam;
use serde::{Deserialize, Serialize};

/// A regression discriminator: a set of RegressionRams, each fed by a
/// fixed subset of input bits (a "tuple") from the shared retina mapping.
/// Used as one "cluster" within ClusRegressionWisard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RegressionDiscriminator {
    pub(crate) rams: Vec<RegressionRam>,
    tuple_indices: Vec<Vec<usize>>,
}

impl RegressionDiscriminator {
    pub(crate) fn new(tuple_indices: Vec<Vec<usize>>, min_zero: u32) -> Self {
        let rams = tuple_indices
            .iter()
            .map(|t| RegressionRam::new(1 << t.len(), min_zero))
            .collect();
        RegressionDiscriminator { rams, tuple_indices }
    }

    #[inline(always)]
    fn address_for_ram(&self, ram_idx: usize, input: &[u8]) -> usize {
        let mut addr: usize = 0;
        for &bit_idx in &self.tuple_indices[ram_idx] {
            addr = (addr << 1) | (input[bit_idx] as usize);
        }
        addr
    }

    pub(crate) fn precompute_addresses(&self, input: &[u8]) -> Vec<usize> {
        (0..self.rams.len()).map(|i| self.address_for_ram(i, input)).collect()
    }

    pub(crate) fn train(&mut self, input: &[u8], target: f64) {
        for i in 0..self.rams.len() {
            let addr = self.address_for_ram(i, input);
            self.rams[i].train(addr, target);
        }
    }

    /// Similarity score for clustering: fraction of RAMs that have a
    /// valid (min_zero-satisfying) prediction at this input's addresses.
    pub(crate) fn match_score(&self, addresses: &[usize]) -> f64 {
        let hits = self
            .rams
            .iter()
            .zip(addresses.iter())
            .filter(|(ram, &addr)| ram.predict(addr).is_some())
            .count();
        hits as f64 / self.rams.len().max(1) as f64
    }

    /// Predicts by averaging partial means across RAMs that have seen
    /// this address enough times. Returns None if no RAM qualifies.
    pub(crate) fn predict(&self, addresses: &[usize]) -> Option<f64> {
        let mut total = 0.0;
        let mut count = 0;
        for (ram, &addr) in self.rams.iter().zip(addresses.iter()) {
            if let Some(p) = ram.predict(addr) {
                total += p;
                count += 1;
            }
        }
        if count > 0 { Some(total / count as f64) } else { None }
    }
}