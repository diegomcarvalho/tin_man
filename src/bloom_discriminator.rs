//! A discriminator built from `BloomRam` nodes instead of exact RAMs.

use crate::bloom_ram::BloomRam;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct BloomDiscriminator {
    tuple_indices: Vec<Vec<usize>>,
    pub(crate) rams: Vec<BloomRam>,
    ignore_zero: bool,
}

impl BloomDiscriminator {
    pub fn new(tuple_indices: Vec<Vec<usize>>, bloom_size: usize, num_hashes: usize, seed: u64, ignore_zero: bool) -> Self {
        let rams = tuple_indices
            .iter()
            .enumerate()
            .map(|(i, _)| BloomRam::new(bloom_size, num_hashes, seed.wrapping_add(i as u64)))
            .collect();
        BloomDiscriminator { tuple_indices, rams, ignore_zero }
    }

    fn address_for_ram(&self, ram_idx: usize, input: &[u8]) -> usize {
        self.tuple_indices[ram_idx]
            .iter()
            .fold(0usize, |acc, &pos| (acc << 1) | (input[pos] as usize))
    }

    pub fn precompute_addresses(&self, input: &[u8]) -> Vec<usize> {
        (0..self.rams.len()).map(|i| self.address_for_ram(i, input)).collect()
    }

    pub fn train(&mut self, input: &[u8]) {
        for (i, ram) in self.rams.iter_mut().enumerate() {
            let addr = self.tuple_indices[i]
                .iter()
                .fold(0usize, |acc, &pos| (acc << 1) | (input[pos] as usize));
            if self.ignore_zero && addr == 0 {
                continue;
            }
            ram.train(addr);
        }
    }

    pub fn score_at(&self, addrs: &[usize], threshold: u16) -> u32 {
        self.rams
            .iter()
            .zip(addrs.iter())
            .filter(|(ram, &addr)| ram.count(addr) >= threshold)
            .count() as u32
    }

    pub fn max_count(&self, addrs: &[usize]) -> u16 {
        self.rams.iter().zip(addrs.iter()).map(|(ram, &addr)| ram.count(addr)).max().unwrap_or(0)
    }

    pub fn memory_bytes(&self) -> usize {
        self.rams.iter().map(|r| r.memory_bytes()).sum()
    }
}