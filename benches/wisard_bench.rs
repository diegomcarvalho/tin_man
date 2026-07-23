use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::Rng;
use tin_man::{ClusRegressionWisard, ClusWisard, RegressionWisard, Wisard};

const INPUT_SIZE: usize = 256;
const ADDRESS_SIZE: usize = 8;
const NUM_CLASSES: usize = 10;
const TRAIN_SAMPLES: usize = 200;

/// Generates deterministic-ish pseudo-random binary input vectors for
/// benchmarking. Not cryptographically random, just fast and varied
/// enough to avoid degenerate all-same-input timing artifacts.
fn random_input(rng: &mut impl Rng, size: usize) -> Vec<u8> {
    (0..size).map(|_| rng.gen_range(0..=1)).collect()
}

fn random_target(rng: &mut impl Rng) -> f64 {
    rng.gen_range(0.0..100.0)
}

fn bench_wisard_train(c: &mut Criterion) {
    let mut group = c.benchmark_group("Wisard::train");
    group.throughput(Throughput::Elements(1));

    group.bench_function("train_single_sample", |b| {
        let mut rng = rand::thread_rng();
        let mut w = Wisard::new(INPUT_SIZE, ADDRESS_SIZE, 0.1, true, false, true);
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

fn bench_wisard_classify(c: &mut Criterion) {
    let mut group = c.benchmark_group("Wisard::classify");
    group.throughput(Throughput::Elements(1));

    let mut rng = rand::thread_rng();
    let mut w = Wisard::new(INPUT_SIZE, ADDRESS_SIZE, 0.1, true, false, true);
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

fn bench_clus_wisard_train(c: &mut Criterion) {
    let mut group = c.benchmark_group("ClusWisard::train");
    group.throughput(Throughput::Elements(1));

    group.bench_function("train_single_sample", |b| {
        let mut rng = rand::thread_rng();
        let mut clus = ClusWisard::new(INPUT_SIZE, ADDRESS_SIZE, 0.3, 20, 5, 0.1, true, false);
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

fn bench_clus_wisard_classify(c: &mut Criterion) {
    let mut group = c.benchmark_group("ClusWisard::classify");
    group.throughput(Throughput::Elements(1));

    let mut rng = rand::thread_rng();
    let mut clus = ClusWisard::new(INPUT_SIZE, ADDRESS_SIZE, 0.3, 20, 5, 0.1, true, false);
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

fn bench_regression_wisard_train(c: &mut Criterion) {
    let mut group = c.benchmark_group("RegressionWisard::train");
    group.throughput(Throughput::Elements(1));

    group.bench_function("train_single_sample", |b| {
        let mut rng = rand::thread_rng();
        let mut rew = RegressionWisard::new(INPUT_SIZE, ADDRESS_SIZE, 1);

        b.iter(|| {
            let input = random_input(&mut rng, INPUT_SIZE);
            let target = random_target(&mut rng);
            rew.train(&input, target);
        });
    });

    group.finish();
}

fn bench_regression_wisard_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("RegressionWisard::predict");
    group.throughput(Throughput::Elements(1));

    let mut rng = rand::thread_rng();
    let mut rew = RegressionWisard::new(INPUT_SIZE, ADDRESS_SIZE, 1);
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

fn bench_clus_regression_wisard_train(c: &mut Criterion) {
    let mut group = c.benchmark_group("ClusRegressionWisard::train");
    group.throughput(Throughput::Elements(1));

    group.bench_function("train_single_sample", |b| {
        let mut rng = rand::thread_rng();
        let mut model = ClusRegressionWisard::new(INPUT_SIZE, ADDRESS_SIZE, 0.3, 20, 5, 1);
        for g_id in 0..NUM_CLASSES {
            let input = random_input(&mut rng, INPUT_SIZE);
            let target = random_target(&mut rng);
            model.train(&input, &format!("group_{g_id}"), target);
        }

        b.iter(|| {
            let input = random_input(&mut rng, INPUT_SIZE);
            let target = random_target(&mut rng);
            let group_id = rng.gen_range(0..NUM_CLASSES);
            model.train(&input, &format!("group_{group_id}"), target);
        });
    });

    group.finish();
}

fn bench_clus_regression_wisard_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("ClusRegressionWisard::predict");
    group.throughput(Throughput::Elements(1));

    let mut rng = rand::thread_rng();
    let mut model = ClusRegressionWisard::new(INPUT_SIZE, ADDRESS_SIZE, 0.3, 20, 5, 1);
    for g_id in 0..NUM_CLASSES {
        for _ in 0..TRAIN_SAMPLES {
            let input = random_input(&mut rng, INPUT_SIZE);
            let target = random_target(&mut rng);
            model.train(&input, &format!("group_{g_id}"), target);
        }
    }

    group.bench_function("predict_single_sample", |b| {
        b.iter(|| {
            let input = random_input(&mut rng, INPUT_SIZE);
            model.predict(&input)
        });
    });

    group.finish();
}

/// Sweeps address_size to show how RAM count (and thus per-op cost)
/// scales, useful for tuning your own models against these benchmarks.
fn bench_wisard_address_size_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("Wisard::classify_by_address_size");
    let mut rng = rand::thread_rng();

    for &addr_size in &[4usize, 8, 16, 32] {
        let mut w = Wisard::new(INPUT_SIZE, addr_size, 0.1, true, false, true);
        for c_id in 0..NUM_CLASSES {
            for _ in 0..TRAIN_SAMPLES {
                let input = random_input(&mut rng, INPUT_SIZE);
                w.train(&input, &format!("class_{c_id}"));
            }
        }

        group.bench_with_input(BenchmarkId::from_parameter(addr_size), &addr_size, |b, _| {
            b.iter(|| {
                let input = random_input(&mut rng, INPUT_SIZE);
                w.classify(&input)
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_wisard_train,
    bench_wisard_classify,
    bench_clus_wisard_train,
    bench_clus_wisard_classify,
    bench_regression_wisard_train,
    bench_regression_wisard_predict,
    bench_clus_regression_wisard_train,
    bench_clus_regression_wisard_predict,
    bench_wisard_address_size_sweep,
);
criterion_main!(benches);