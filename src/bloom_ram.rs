//! Compact, Bloom-filter-backed replacement for a WiSARD RAM node.
//!
//! Instead of allocating one counter per possible address
//! (`2^address_size` entries, most of which are never visited), a
//! `BloomRam` hashes each address through `num_hashes` independent
//! hash functions into a much smaller `size`-entry counter array.
//! Training increments all `k` hashed slots; querying returns the
//! *minimum* of the `k` slots, following the same principle as a
//! counting Bloom filter / Count-Min Sketch: collisions inflate
//! individual counters, but the true count can never exceed the
//! smallest of the k observed values, so taking the min suppresses
//! collision noise.

use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct BloomRam {
    counters: Vec<u16>,
    size: usize,
    num_hashes: usize,
    seeds: Vec<u64>,
}

impl BloomRam {
    /// Creates a new Bloom RAM with `size` counters and `num_hashes`
    /// independent hash functions.
    ///
    /// # Panics
    ///
    /// Panics if `size` or `num_hashes` is `0`.
    pub fn new(size: usize, num_hashes: usize, seed: u64) -> Self {
        assert!(size > 0, "size must be > 0");
        assert!(num_hashes > 0, "num_hashes must be > 0");

        // Derive `num_hashes` independent-looking hash seeds from the
        // single model seed, using a cheap splitmix-style step.
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

        BloomRam { counters: vec![0u16; size], size, num_hashes, seeds }
    }

    fn hash(&self, address: usize, hash_idx: usize) -> usize {
        let mixed = (address as u64).wrapping_mul(0x2545F4914F6CDD1D) ^ self.seeds[hash_idx];
        let mut z = mixed;
        z = (z ^ (z >> 33)).wrapping_mul(0xFF51AFD7ED558CCD);
        z = (z ^ (z >> 33)).wrapping_mul(0xC4CEB9FE1A85EC53);
        z ^= z >> 33;
        (z as usize) % self.size
    }

    /// Increments the `num_hashes` slots associated with `address`,
    /// saturating rather than overflowing on very high counts.
    pub fn train(&mut self, address: usize) {
        for h in 0..self.num_hashes {
            let idx = self.hash(address, h);
            self.counters[idx] = self.counters[idx].saturating_add(1);
        }
    }

    /// Returns the minimum counter value across all `num_hashes`
    /// hashed slots for `address` — the Bloom-filter-style estimate of
    /// how many times this address was trained on.
    pub fn count(&self, address: usize) -> u16 {
        (0..self.num_hashes).map(|h| self.counters[self.hash(address, h)]).min().unwrap_or(0)
    }

    /// `true` if every hashed slot for `address` is non-zero (the
    /// classic Bloom filter membership test).
    #[allow(dead_code)]
    pub fn contains(&self, address: usize) -> bool {
        self.count(address) > 0
    }

    /// Approximate memory footprint in bytes, for comparison against
    /// an equivalent exact RAM's `2^address_size * size_of::<u16>()`.
    pub fn memory_bytes(&self) -> usize {
        self.counters.len() * std::mem::size_of::<u16>()
    }
}