//! Bloom-backed accumulator for regression: instead of one exact
//! (sum, count) pair per address, this hashes each address through
//! `num_hashes` slots in *two parallel* counting sketches — one
//! accumulating target-value sums, one accumulating visit counts —
//! following the Count-Min Sketch principle: querying takes the slot
//! with the *smallest count* (least collision contamination) among
//! the k hashed slots, then returns that slot's sum/count ratio as the
//! address's estimated mean target value.

use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct BloomRegressionRam {
    sums: Vec<f64>,
    counts: Vec<u32>,
    size: usize,
    num_hashes: usize,
    seeds: Vec<u64>,
}

impl BloomRegressionRam {
    pub fn new(size: usize, num_hashes: usize, seed: u64) -> Self {
        assert!(size > 0 && num_hashes > 0, "size and num_hashes must be > 0");
        let mut seeds = Vec::with_capacity(num_hashes);
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
        for _ in 0..num_hashes {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            seeds.push(z);
        }
        BloomRegressionRam { sums: vec![0.0; size], counts: vec![0; size], size, num_hashes, seeds }
    }

    fn hash(&self, address: usize, hash_idx: usize) -> usize {
        let mixed = (address as u64).wrapping_mul(0x2545F4914F6CDD1D) ^ self.seeds[hash_idx];
        let mut z = mixed;
        z = (z ^ (z >> 33)).wrapping_mul(0xFF51AFD7ED558CCD);
        z = (z ^ (z >> 33)).wrapping_mul(0xC4CEB9FE1A85EC53);
        z ^= z >> 33;
        (z as usize) % self.size
    }

    pub fn train(&mut self, address: usize, target: f64) {
        for h in 0..self.num_hashes {
            let idx = self.hash(address, h);
            self.sums[idx] += target;
            self.counts[idx] = self.counts[idx].saturating_add(1);
        }
    }

    /// Estimated mean target value for `address`, taken from the
    /// hashed slot with the smallest visit count (least collision
    /// noise). Returns `None` if that slot was never visited.
    pub fn predict(&self, address: usize) -> Option<f64> {
        let best = (0..self.num_hashes)
            .map(|h| self.hash(address, h))
            .min_by_key(|&idx| self.counts[idx])?;
        if self.counts[best] == 0 {
            None
        } else {
            Some(self.sums[best] / self.counts[best] as f64)
        }
    }

    pub fn memory_bytes(&self) -> usize {
        self.sums.len() * std::mem::size_of::<f64>() + self.counts.len() * std::mem::size_of::<u32>()
    }
}