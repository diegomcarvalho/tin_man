use tin_man::BloomWisard;

fn main() {
    // 20 address bits per RAM would need 2^20 = 1,048,576 exact
    // counters. A Bloom RAM needs only 256 counters with 4 hashes.
    let mut w = BloomWisard::new(160, 20, 256, 4, 0.1, true, false, false);

    w.train(&vec![1u8; 80].into_iter().chain(vec![0u8; 80]).collect::<Vec<_>>(), "cold");
    w.train(&vec![0u8; 80].into_iter().chain(vec![1u8; 80]).collect::<Vec<_>>(), "hot");

    println!("Memory used: {} bytes", w.memory_bytes());
}