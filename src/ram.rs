use serde::{Deserialize, Serialize};

const MAX_COUNT: u16 = u16::MAX;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Ram {
    counts: Vec<u16>,
    ignore_zero: bool,
}

impl Ram {
    pub(crate) fn new(address_space: usize, ignore_zero: bool) -> Self {
        Ram { counts: vec![0; address_space], ignore_zero }
    }

    #[inline(always)]
    pub(crate) fn train(&mut self, address: usize) {
        if self.ignore_zero && address == 0 {
            return;
        }
        let c = &mut self.counts[address];
        if *c < MAX_COUNT {
            *c += 1;
        }
    }

    #[inline(always)]
    pub(crate) fn count(&self, address: usize) -> u16 {
        if self.ignore_zero && address == 0 {
            return 0;
        }
        self.counts[address]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RegressionRam {
    counts: Vec<u32>,
    sums: Vec<f64>,
    min_zero: u32,
}

impl RegressionRam {
    pub(crate) fn new(address_space: usize, min_zero: u32) -> Self {
        RegressionRam { counts: vec![0; address_space], sums: vec![0.0; address_space], min_zero }
    }

    pub(crate) fn train(&mut self, address: usize, target: f64) {
        self.counts[address] += 1;
        self.sums[address] += target;
    }

    pub(crate) fn predict(&self, address: usize) -> Option<f64> {
        let c = self.counts[address];
        if c >= self.min_zero.max(1) {
            Some(self.sums[address] / c as f64)
        } else {
            None
        }
    }
}