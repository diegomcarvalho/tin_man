use crate::ram::Ram;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Discriminator {
    pub(crate) rams: Vec<Ram>,
    tuple_indices: Vec<Vec<usize>>,
}

impl Discriminator {
    pub(crate) fn new(tuple_indices: Vec<Vec<usize>>, ignore_zero: bool) -> Self {
        let rams = tuple_indices.iter().map(|t| Ram::new(1 << t.len(), ignore_zero)).collect();
        Discriminator { rams, tuple_indices }
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

    pub(crate) fn train(&mut self, input: &[u8]) {
        for i in 0..self.rams.len() {
            let addr = self.address_for_ram(i, input);
            self.rams[i].train(addr);
        }
    }

    #[inline(always)]
    pub(crate) fn score_at(&self, addresses: &[usize], threshold: u16) -> usize {
        self.rams
            .iter()
            .zip(addresses.iter())
            .filter(|&(ram, &addr)| ram.count(addr) >= threshold)
            .count()
    }

    pub(crate) fn max_count(&self, addresses: &[usize]) -> u16 {
        self.rams
            .iter()
            .zip(addresses.iter())
            .map(|(ram, &addr)| ram.count(addr))
            .max()
            .unwrap_or(0)
    }
}