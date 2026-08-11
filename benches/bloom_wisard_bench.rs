use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::Rng;
use tin_man::{BloomClusWisard, BloomRegressionWisard, BloomWisard, Wisard};

const INPUT_SIZE: usize = 256;
const ADDRESS_SIZE: usize = 8;
const BLOOM_SIZE: usize = 64;
const NUM_HASHES: usize = 4;
const NUM_CLASSES: usize = 10;
const TRAIN_SAMPLES: usize = 200;
const SEED: u64 = 12345;

/// Generates pseudo-random binary input vectors for benchmarking. Not
/// cryptographically random, just fast and varied enough to avoid
/// degenerate all-same-input timing artifacts.
fn random_input(rng: &mut impl Rng, size: usize) -> Vec<u8> {
    (0..size).map(|_| rng.gen_range(0..=1)).collect()
}

fn random_target(rng: &mut impl Rng) -> f64 {
    rng.gen_range(0.0..100.0)
}

fn bench_bloom_wisard_train(c: &mut Criterion) {
    let mut group = c.benchmark_group("BloomWisard::train");
    group.throughput(Throughput::Elements(1));

    group.bench_function("train_single_sample", |b| {
        let mut rng = rand::thread_rng();
        let mut w = BloomWisard::new_with_seed(
            INPUT_SIZE, ADDRESS_SIZE, BLOOM_SIZE, NUM_HASHES, 0.1, true, false, false, SEED,
        );
        // Pre-populate a few classes so the discriminator lookup isn't
        // always hitting the "new class" cold path.
        for c_id in 0..NUM_CLASSES {
            let input = random_input(&mut rng, INPUT_SIZE);
            w.train(&input, &format!("class_{c_id}"));
        }

        b.iter(|| {
            let input = random_input(&mut rng, INPUT_SIZE);
            let class_id = rng.gen_range(0..NUM_CLASSES);
            w.train(&input, &format!("class_{class_id}"));
        });
    });

    group.finish();
}

fn bench_bloom_wisard_classify(c: &mut Criterion) {
    let mut group = c.benchmark_group("BloomWisard::classify");
    group.throughput(Throughput::Elements(1));

    let mut rng = rand::thread_rng();
    let mut w = BloomWisard::new_with_seed(
        INPUT_SIZE, ADDRESS_SIZE, BLOOM_SIZE, NUM_HASHES, 0.1, true, false, false, SEED,
    );
    for c_id in 0..NUM_CLASSES {
        for _ in 0..TRAIN_SAMPLES {
            let input = random_input(&mut rng, INPUT_SIZE);
            w.train(&input, &format!("class_{c_id}"));
        }
    }

    group.bench_function("classify_single_sample", |b| {
        b.iter(|| {
            let input = random_input(&mut rng, INPUT_SIZE);
            w.classify(&input)
        });
    });

    group.finish();
}

fn bench_bloom_wisard_train_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("BloomWisard::classify_parallel");
    group.throughput(Throughput::Elements(1));

    let mut rng = rand::thread_rng();
    let mut w = BloomWisard::new_with_seed(
        INPUT_SIZE, ADDRESS_SIZE, BLOOM_SIZE, NUM_HASHES, 0.1, true, false, true, SEED,
    );
    for c_id in 0..NUM_CLASSES {
        for _ in 0..TRAIN_SAMPLES {
            let input = random_input(&mut rng, INPUT_SIZE);
            w.train(&input, &format!("class_{c_id}"));
        }
    }

    group.bench_function("classify_single_sample_parallel", |b| {
        b.iter(|| {
            let input = random_input(&mut rng, INPUT_SIZE);
            w.classify(&input)
        });
    });

    group.finish();
}

fn bench_bloom_clus_wisard_train(c: &mut Criterion) {
    let mut group = c.benchmark_group("BloomClusWisard::train");
    group.throughput(Throughput::Elements(1));

    group.bench_function("train_single_sample", |b| {
        let mut rng = rand::thread_rng();
        let mut clus = BloomClusWisard::new_with_seed(
            INPUT_SIZE, ADDRESS_SIZE, BLOOM_SIZE, NUM_HASHES, 0.3, 5, false, SEED,
        );
        for c_id in 0..NUM_CLASSES {
            let input = random_input(&mut rng, INPUT_SIZE);
            clus.train(&input, &format!("class_{c_id}"));
        }

        b.iter(|| {
            let input = random_input(&mut rng, INPUT_SIZE);
            let class_id = rng.gen_range(0..NUM_CLASSES);
            clus.train(&input, &format!("class_{class_id}"));
        });
    });

    group.finish();
}

fn bench_bloom_clus_wisard_classify(c: &mut Criterion) {
    let mut group = c.benchmark_group("BloomClusWisard::classify");
    group.throughput(Throughput::Elements(1));

    let mut rng = rand::thread_rng();
    let mut clus = BloomClusWisard::new_with_seed(
        INPUT_SIZE, ADDRESS_SIZE, BLOOM_SIZE, NUM_HASHES, 0.3, 5, false, SEED,
    );
    for c_id in 0..NUM_CLASSES {
        for _ in 0..TRAIN_SAMPLES {
            let input = random_input(&mut rng, INPUT_SIZE);
            clus.train(&input, &format!("class_{c_id}"));
        }
    }

    group.bench_function("classify_single_sample", |b| {
        b.iter(|| {
            let input = random_input(&mut rng, INPUT_SIZE);
            clus.classify(&input)
        });
    });

    group.finish();
}

fn bench_bloom_regression_wisard_train(c: &mut Criterion) {
    let mut group = c.benchmark_group("BloomRegressionWisard::train");
    group.throughput(Throughput::Elements(1));

    group.bench_function("train_single_sample", |b| {
        let mut rng = rand::thread_rng();
        let mut rew = BloomRegressionWisard::new_with_seed(
            INPUT_SIZE, ADDRESS_SIZE, BLOOM_SIZE, NUM_HASHES, 1, SEED,
        );

        b.iter(|| {
            let input = random_input(&mut rng, INPUT_SIZE);
            let target = random_target(&mut rng);
            rew.train(&input, target);
        });
    });

    group.finish();
}

fn bench_bloom_regression_wisard_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("BloomRegressionWisard::predict");
    group.throughput(Throughput::Elements(1));

    let mut rng = rand::thread_rng();
    let mut rew = BloomRegressionWisard::new_with_seed(
        INPUT_SIZE, ADDRESS_SIZE, BLOOM_SIZE, NUM_HASHES, 1, SEED,
    );
    for _ in 0..(TRAIN_SAMPLES * NUM_CLASSES) {
        let input = random_input(&mut rng, INPUT_SIZE);
        let target = random_target(&mut rng);
        rew.train(&input, target);
    }

    group.bench_function("predict_single_sample", |b| {
        b.iter(|| {
            let input = random_input(&mut rng, INPUT_SIZE);
            rew.predict(&input)
        });
    });

    group.finish();
}

/// Sweeps `bloom_size` (holding `num_hashes` fixed) to show how the
/// memory/accuracy/speed trade-off shifts as the Bloom filter grows
/// relative to the exact address space (`2^ADDRESS_SIZE = 256`).
fn bench_bloom_wisard_bloom_size_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("BloomWisard::classify_by_bloom_size");
    let mut rng = rand::thread_rng();

    for &bloom_size in &[16usize, 32, 64, 128, 256] {
        let mut w = BloomWisard::new_with_seed(
            INPUT_SIZE, ADDRESS_SIZE, bloom_size, NUM_HASHES, 0.1, true, false, false, SEED,
        );
        for c_id in 0..NUM_CLASSES {
            for _ in 0..TRAIN_SAMPLES {
                let input = random_input(&mut rng, INPUT_SIZE);
                w.train(&input, &format!("class_{c_id}"));
            }
        }

        group.bench_with_input(BenchmarkId::from_parameter(bloom_size), &bloom_size, |b, _| {
            b.iter(|| {
                let input = random_input(&mut rng, INPUT_SIZE);
                w.classify(&input)
            });
        });
    }

    group.finish();
}

/// Sweeps `num_hashes` (holding `bloom_size` fixed) to show how
/// per-classification cost scales linearly with the number of hash
/// functions used per Bloom RAM lookup.
fn bench_bloom_wisard_num_hashes_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("BloomWisard::classify_by_num_hashes");
    let mut rng = rand::thread_rng();

    for &num_hashes in &[1usize, 2, 4, 8, 16] {
        let mut w = BloomWisard::new_with_seed(
            INPUT_SIZE, ADDRESS_SIZE, BLOOM_SIZE, num_hashes, 0.1, true, false, false, SEED,
        );
        for c_id in 0..NUM_CLASSES {
            for _ in 0..TRAIN_SAMPLES {
                let input = random_input(&mut rng, INPUT_SIZE);
                w.train(&input, &format!("class_{c_id}"));
            }
        }

        group.bench_with_input(BenchmarkId::from_parameter(num_hashes), &num_hashes, |b, _| {
            b.iter(|| {
                let input = random_input(&mut rng, INPUT_SIZE);
                w.classify(&input)
            });
        });
    }

    group.finish();
}

/// Not a timing benchmark — runs once and prints a direct memory
/// footprint comparison between an exact `Wisard` and a `BloomWisard`
/// trained on identical data, so you can see the actual bytes-saved
/// trade-off alongside the throughput numbers above.
fn bench_memory_footprint_comparison(c: &mut Criterion) {
    let mut rng = rand::thread_rng();

    let mut exact = Wisard::new_with_seed(INPUT_SIZE, ADDRESS_SIZE, 0.1, true, false, false, SEED);
    let mut bloom = BloomWisard::new_with_seed(
        INPUT_SIZE, ADDRESS_SIZE, BLOOM_SIZE, NUM_HASHES, 0.1, true, false, false, SEED,
    );

    for c_id in 0..NUM_CLASSES {
        for _ in 0..TRAIN_SAMPLES {
            let input = random_input(&mut rng, INPUT_SIZE);
            exact.train(&input, &format!("class_{c_id}"));
            bloom.train(&input, &format!("class_{c_id}"));
        }
    }

    let exact_bytes_per_ram = (1usize << ADDRESS_SIZE) * std::mem::size_of::<u16>();
    let num_rams = INPUT_SIZE / ADDRESS_SIZE;
    let exact_total_bytes = exact_bytes_per_ram * num_rams * NUM_CLASSES;
    let bloom_total_bytes = bloom.memory_bytes();

    eprintln!("\n=== Memory footprint comparison (address_size={ADDRESS_SIZE}, {NUM_CLASSES} classes, {num_rams} RAMs/discriminator) ===");
    eprintln!("Exact Wisard (estimated):  {exact_total_bytes:>10} bytes ({:.2} KB)", exact_total_bytes as f64 / 1024.0);
    eprintln!("BloomWisard (bloom_size={BLOOM_SIZE}, num_hashes={NUM_HASHES}): {bloom_total_bytes:>10} bytes ({:.2} KB)", bloom_total_bytes as f64 / 1024.0);
    eprintln!(
        "Compression ratio: {:.2}x smaller\n",
        exact_total_bytes as f64 / bloom_total_bytes.max(1) as f64
    );

    // Register a trivial no-op benchmark so this function participates
    // in the criterion_group! harness like the others; the actual
    // comparison output above already ran by the time this executes.
    let mut group = c.benchmark_group("MemoryFootprint::noop");
    group.bench_function("printed_above", |b| b.iter(|| 1 + 1));
    group.finish();
}

criterion_group!(
    benches,
    bench_bloom_wisard_train,
    bench_bloom_wisard_classify,
    bench_bloom_wisard_train_parallel,
    bench_bloom_clus_wisard_train,
    bench_bloom_clus_wisard_classify,
    bench_bloom_regression_wisard_train,
    bench_bloom_regression_wisard_predict,
    bench_bloom_wisard_bloom_size_sweep,
    bench_bloom_wisard_num_hashes_sweep,
    bench_memory_footprint_comparison,
);
criterion_main!(benches);
