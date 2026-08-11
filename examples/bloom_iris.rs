// Classifies the Iris dataset using BloomWisard and BloomClusWisard.
//
// Unlike demo_iris.rs, which trains on random samples drawn from all
// three known classes up front, this example trains SEQUENTIALLY
// through the dataset in its natural order (50 Setosa, then 50
// Versicolor, then 50 Virginica). Because both BloomWisard and
// BloomClusWisard create a new discriminator/cluster lazily the first
// time a label is seen, this means:
//
//   - For the first 50 training steps, the model has only ONE class
//     trained ("0" / Setosa).
//   - At step 50, a brand-new class ("1" / Versicolor) is discovered
//     and its discriminator is created on the fly.
//   - At step 100, a third new class ("2" / Virginica) is discovered
//     the same way.
//
// This demonstrates that both Bloom-backed models support incremental
// class discovery identically to their exact-RAM counterparts.

use tin_man::{BloomClusWisard, BloomWisard, DistributiveThermometer, FileFormat};

// Iris dataset, 150 samples (Fisher/Anderson), ordered:
// 0..50 = setosa, 50..100 = versicolor, 100..150 = virginica.
pub const SEPAL_LENGTH: [f64; 150] = [
    5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9,
    5.4, 4.8, 4.8, 4.3, 5.8, 5.7, 5.4, 5.1, 5.7, 5.1,
    5.4, 5.1, 4.6, 5.1, 4.8, 5.0, 5.0, 5.2, 5.2, 4.7,
    4.8, 5.4, 5.2, 5.5, 4.9, 5.0, 5.5, 4.9, 4.4, 5.1,
    5.0, 4.5, 4.4, 5.0, 5.1, 4.8, 5.1, 4.6, 5.3, 5.0,
    7.0, 6.4, 6.9, 5.5, 6.5, 5.7, 6.3, 4.9, 6.6, 5.2,
    5.0, 5.9, 6.0, 6.1, 5.6, 6.7, 5.6, 5.8, 6.2, 5.6,
    5.9, 6.1, 6.3, 6.1, 6.4, 6.6, 6.8, 6.7, 6.0, 5.7,
    5.5, 5.5, 5.8, 6.0, 5.4, 6.0, 6.7, 6.3, 5.6, 5.5,
    5.5, 6.1, 5.8, 5.0, 5.6, 5.7, 5.7, 6.2, 5.1, 5.7,
    6.3, 5.8, 7.1, 6.3, 6.5, 7.6, 4.9, 7.3, 6.7, 7.2,
    6.5, 6.4, 6.8, 5.7, 5.8, 6.4, 6.5, 7.7, 7.7, 6.0,
    6.9, 5.6, 7.7, 6.3, 6.7, 7.2, 6.2, 6.1, 6.4, 7.2,
    7.4, 7.9, 6.4, 6.3, 6.1, 7.7, 6.3, 6.4, 6.0, 6.9,
    6.7, 6.9, 5.8, 6.8, 6.7, 6.7, 6.3, 6.5, 6.2, 5.9,
];

pub const SEPAL_WIDTH: [f64; 150] = [
    3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1,
    3.7, 3.4, 3.0, 3.0, 4.0, 4.4, 3.9, 3.5, 3.8, 3.8,
    3.4, 3.7, 3.6, 3.3, 3.4, 3.0, 3.4, 3.5, 3.4, 3.2,
    3.1, 3.4, 4.1, 4.2, 3.1, 3.2, 3.5, 3.1, 3.0, 3.4,
    3.5, 2.3, 3.2, 3.5, 3.8, 3.0, 3.8, 3.2, 3.7, 3.3,
    3.2, 3.2, 3.1, 2.3, 2.8, 2.8, 3.3, 2.4, 2.9, 2.7,
    2.0, 3.0, 2.2, 2.9, 2.9, 3.1, 3.0, 2.7, 2.2, 2.5,
    3.2, 2.8, 2.5, 2.8, 2.9, 3.0, 2.8, 3.0, 2.9, 2.6,
    2.4, 2.4, 2.7, 2.7, 3.0, 3.4, 3.1, 2.3, 3.0, 2.5,
    2.6, 3.0, 2.6, 2.3, 2.7, 3.0, 2.9, 2.9, 2.5, 2.8,
    3.3, 2.7, 3.0, 2.9, 3.0, 3.0, 2.5, 2.9, 2.5, 3.6,
    3.2, 2.7, 3.0, 2.5, 2.8, 3.2, 3.0, 3.8, 2.6, 2.2,
    3.2, 2.8, 2.8, 2.7, 3.3, 3.2, 2.8, 3.0, 2.8, 3.0,
    2.8, 3.8, 2.8, 2.8, 2.6, 3.0, 3.4, 3.1, 3.0, 3.1,
    3.1, 3.1, 2.7, 3.2, 3.3, 3.0, 2.5, 3.0, 3.4, 3.0,
];

pub const PETAL_LENGTH: [f64; 150] = [
    1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5,
    1.5, 1.6, 1.4, 1.1, 1.2, 1.5, 1.3, 1.4, 1.7, 1.5,
    1.7, 1.5, 1.0, 1.7, 1.9, 1.6, 1.6, 1.5, 1.4, 1.6,
    1.6, 1.5, 1.5, 1.4, 1.5, 1.2, 1.3, 1.5, 1.3, 1.5,
    1.3, 1.3, 1.3, 1.6, 1.9, 1.4, 1.6, 1.4, 1.5, 1.4,
    4.7, 4.5, 4.9, 4.0, 4.6, 4.5, 4.7, 3.3, 4.6, 3.9,
    3.5, 4.2, 4.0, 4.7, 3.6, 4.4, 4.5, 4.1, 4.5, 3.9,
    4.8, 4.0, 4.9, 4.7, 4.3, 4.4, 4.8, 5.0, 4.5, 3.5,
    3.8, 3.7, 3.9, 5.1, 4.5, 4.5, 4.7, 4.4, 4.1, 4.0,
    4.4, 4.6, 4.0, 3.3, 4.2, 4.2, 4.2, 4.3, 3.0, 4.1,
    6.0, 5.1, 5.9, 5.6, 5.8, 6.6, 4.5, 6.3, 5.8, 6.1,
    5.1, 5.3, 5.5, 5.0, 5.1, 5.3, 5.5, 6.7, 6.9, 5.0,
    5.7, 4.9, 6.7, 4.9, 5.7, 6.0, 4.8, 4.9, 5.6, 5.8,
    6.1, 6.4, 5.6, 5.1, 5.6, 6.1, 5.6, 5.5, 4.8, 5.4,
    5.6, 5.1, 5.1, 5.9, 5.7, 5.2, 5.0, 5.2, 5.4, 5.1,
];

pub const PETAL_WIDTH: [f64; 150] = [
    0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1,
    0.2, 0.2, 0.1, 0.1, 0.2, 0.4, 0.4, 0.3, 0.3, 0.3,
    0.2, 0.4, 0.2, 0.5, 0.2, 0.2, 0.4, 0.2, 0.2, 0.2,
    0.2, 0.4, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.2, 0.2,
    0.3, 0.3, 0.2, 0.6, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2,
    1.4, 1.5, 1.5, 1.3, 1.5, 1.3, 1.6, 1.0, 1.3, 1.4,
    1.0, 1.5, 1.0, 1.4, 1.3, 1.4, 1.5, 1.0, 1.5, 1.1,
    1.8, 1.3, 1.5, 1.2, 1.3, 1.4, 1.4, 1.7, 1.5, 1.0,
    1.1, 1.0, 1.2, 1.6, 1.5, 1.6, 1.5, 1.3, 1.3, 1.3,
    1.2, 1.4, 1.2, 1.0, 1.3, 1.2, 1.3, 1.3, 1.1, 1.3,
    2.5, 1.9, 2.1, 1.8, 2.2, 2.1, 1.7, 1.8, 1.8, 2.5,
    2.0, 1.9, 2.1, 2.0, 2.4, 2.3, 1.8, 2.2, 2.3, 1.5,
    2.3, 2.0, 2.0, 1.8, 2.1, 1.8, 1.8, 1.8, 2.1, 1.6,
    1.9, 2.0, 2.2, 1.5, 1.4, 2.3, 2.4, 1.8, 1.8, 2.1,
    2.4, 2.3, 1.9, 2.3, 2.5, 2.3, 1.9, 2.0, 2.3, 1.8,
];

pub const SPECIES: [u8; 150] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
];

fn build_retina(
    i: usize,
    sl: &DistributiveThermometer,
    sw: &DistributiveThermometer,
    pl: &DistributiveThermometer,
    pw: &DistributiveThermometer,
) -> Vec<u8> {
    let v1 = sl.encode(SEPAL_LENGTH[i]);
    let v2 = sw.encode(SEPAL_WIDTH[i]);
    let v3 = pl.encode(PETAL_LENGTH[i]);
    let v4 = pw.encode(PETAL_WIDTH[i]);
    [v1, v2, v3, v4].concat()
}

fn evaluate<F>(name: &str, classify_fn: F)
where
    F: Fn(&[u8]) -> Option<(String, f64)>,
{
    let linear_sl = DistributiveThermometer::fit(&SEPAL_LENGTH, 16);
    let linear_sw = DistributiveThermometer::fit(&SEPAL_WIDTH, 16);
    let linear_pl = DistributiveThermometer::fit(&PETAL_LENGTH, 16);
    let linear_pw = DistributiveThermometer::fit(&PETAL_WIDTH, 16);

    let mut correct = 0;
    for i in 0..150 {
        let retina = build_retina(i, &linear_sl, &linear_sw, &linear_pl, &linear_pw);
        if let Some((label, confidence)) = classify_fn(&retina) {
            if let Ok(predicted) = label.parse::<usize>() {
                if SPECIES[i] as usize == predicted {
                    correct += 1;
                }
            } else {
                println!("[{name}] Failed to parse label '{label}' as usize");
            }
            let _ = confidence; // available for per-sample inspection if desired
        }
    }
    println!("[{name}] Good guesses: {correct} of 150\n");
}

fn main() {
    let linear_sl = DistributiveThermometer::fit(&SEPAL_LENGTH, 16);
    let linear_sw = DistributiveThermometer::fit(&SEPAL_WIDTH, 16);
    let linear_pl = DistributiveThermometer::fit(&PETAL_LENGTH, 16);
    let linear_pw = DistributiveThermometer::fit(&PETAL_WIDTH, 16);

    let input_size = 16 * 4; // 4 features x 16 thermometer bits each = 64

    // --- BloomWisard ---
    // address_size = 8 means each exact RAM would need 2^8 = 256
    // counters; bloom_size = 64 with 4 hashes covers the same address
    // space with only 64 counters per RAM (4x compression).
    println!("=== BloomWisard ===");
    let mut bw = BloomWisard::new_with_seed(
        input_size, // input_size
        8,          // address_size
        64,         // bloom_size
        4,          // num_hashes
        0.1,        // confidence_threshold
        true,       // bleaching_enabled
        false,      // ignore_zero
        true,       // parallel
        12770,      // seed
    );

    // Sequential training: because SPECIES is grouped by class, the
    // model only knows label "0" for the first 50 iterations, then
    // lazily creates discriminators for "1" (at i=50) and "2"
    // (at i=100) the first time each is encountered.
    for i in 0..150 {
        let retina = build_retina(i, &linear_sl, &linear_sw, &linear_pl, &linear_pw);
        bw.train(&retina, &SPECIES[i].to_string());

        if i == 0 {
            println!("Step {i:>3}: only class '0' trained so far.");
        } else if i == 50 {
            println!("Step {i:>3}: class '1' encountered for the first time — new discriminator created.");
        } else if i == 100 {
            println!("Step {i:>3}: class '2' encountered for the first time — new discriminator created.");
        }
    }

    bw.save_to_file("bloom_wisard_model.json", FileFormat::Json)
        .expect("failed to save BloomWisard model");
    println!("BloomWisard memory usage: {} bytes", bw.memory_bytes());

    evaluate("BloomWisard", |retina| bw.classify(retina));

    // --- BloomClusWisard ---
    // Same bloom_size/num_hashes budget per RAM, but each class may
    // grow up to `discriminator_limit` clusters if a single cluster's
    // score falls below `min_score` on a new sample — useful when a
    // class's samples aren't all similar enough for one discriminator.
    println!("=== BloomClusWisard ===");
    let mut bcw = BloomClusWisard::new_with_seed(
        input_size, // input_size
        8,          // address_size
        64,         // bloom_size
        4,          // num_hashes
        0.3,        // min_score
        3,          // discriminator_limit
        false,      // ignore_zero
        12770,      // seed
    );

    for i in 0..150 {
        let retina = build_retina(i, &linear_sl, &linear_sw, &linear_pl, &linear_pw);
        bcw.train(&retina, &SPECIES[i].to_string());

        if i == 0 {
            println!("Step {i:>3}: only class '0' trained so far.");
        } else if i == 50 {
            println!("Step {i:>3}: class '1' encountered for the first time — new cluster set created.");
        } else if i == 100 {
            println!("Step {i:>3}: class '2' encountered for the first time — new cluster set created.");
        }
    }

    bcw.save_to_file("bloom_clus_wisard_model.json", FileFormat::Json)
        .expect("failed to save BloomClusWisard model");
    println!("BloomClusWisard memory usage: {} bytes", bcw.memory_bytes());

    evaluate("BloomClusWisard", |retina| bcw.classify(retina));
}
