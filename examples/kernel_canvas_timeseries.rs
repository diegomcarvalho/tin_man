//! Classifies synthetic 2D motion strokes (diagonal vs. circular) using
//! `KernelCanvas` to turn variable-length point sequences into a
//! fixed-size binary pattern, then feeds that pattern into a `Wisard`.
//!
//! This mirrors KernelCanvas's original motivating use case:
//! time-series / spatio-temporal data where each sample may have a
//! different number of points, but WiSARD needs one fixed-size input.
//!
//! Run with:
//!     cargo run --example kernel_canvas_timeseries

use tin_man::KernelCanvas;
use tin_man::Wisard;

/// Generates a diagonal stroke from `(-1, -1)` to `(1, 1)` (or the
/// mirrored anti-diagonal), with `num_points` samples along the line
/// plus small random jitter to simulate natural hand-drawn variation.
fn generate_diagonal(num_points: usize, mirrored: bool, jitter: f64, seed: u64) -> Vec<Vec<f64>> {
    let mut rng_state = seed;
    let mut next_jitter = || {
        // Simple xorshift for reproducible, dependency-free jitter.
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let normalized = (rng_state % 2000) as f64 / 1000.0 - 1.0; // [-1, 1]
        normalized * jitter
    };

    (0..num_points)
        .map(|i| {
            let t = i as f64 / (num_points - 1).max(1) as f64;
            let x = -1.0 + 2.0 * t;
            let y = if mirrored { -x } else { x };
            vec![x + next_jitter(), y + next_jitter()]
        })
        .collect()
}

/// Generates a circular stroke of radius `radius` centered at the
/// origin, with `num_points` samples around the circle plus jitter.
fn generate_circle(num_points: usize, radius: f64, jitter: f64, seed: u64) -> Vec<Vec<f64>> {
    let mut rng_state = seed;
    let mut next_jitter = || {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let normalized = (rng_state % 2000) as f64 / 1000.0 - 1.0;
        normalized * jitter
    };

    (0..num_points)
        .map(|i| {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / (num_points as f64);
            let x = radius * angle.cos();
            let y = radius * angle.sin();
            vec![x + next_jitter(), y + next_jitter()]
        })
        .collect()
}

fn main() {
    // 24 kernels scattered across [-1, 1]^2, 4 thermometer bits per
    // kernel for graded proximity, activating the closest 25% of
    // kernels per point.
    let canvas = KernelCanvas::new(24, 2, 4, 0.25, 42);
    println!("Canvas output size: {} bits\n", canvas.output_size());

    let mut w = Wisard::new(canvas.output_size(), 6, 0.15, true, false, false);

    // --- Training set ---
    // Diagonal strokes: varying point counts and jitter, both
    // directions, to teach the model "diagonal" independent of stroke
    // length or exact orientation sign.
    let diagonal_train_seeds = [(15, 0.03, 1), (20, 0.05, 2), (12, 0.02, 3), (18, 0.04, 4)];
    for (i, &(n, jitter, seed)) in diagonal_train_seeds.iter().enumerate() {
        let mirrored = i % 2 == 0;
        let stroke = generate_diagonal(n, mirrored, jitter, seed);
        w.train(&canvas.encode_sequence(&stroke), "diagonal");
    }

    // Circular strokes: varying point counts, radii, and jitter.
    let circle_train_seeds = [(16, 0.6, 0.03, 10), (22, 0.7, 0.05, 11), (14, 0.65, 0.02, 12), (19, 0.55, 0.04, 13)];
    for &(n, radius, jitter, seed) in circle_train_seeds.iter() {
        let stroke = generate_circle(n, radius, jitter, seed);
        w.train(&canvas.encode_sequence(&stroke), "circle");
    }

    println!("Trained on {} diagonal strokes and {} circle strokes.\n",
        diagonal_train_seeds.len(), circle_train_seeds.len());

    let _ = w.save_to_file("wisard.json",tin_man::FileFormat::Json);


    // --- Test set ---
    // Fresh strokes with different point counts / jitter / seeds than
    // training, to confirm the model generalizes rather than
    // memorizing exact point sequences.
    let test_cases: Vec<(&str, Vec<Vec<f64>>)> = vec![
        ("diagonal (unseen jitter/length)", generate_diagonal(25, false, 0.06, 99)),
        ("diagonal, mirrored (unseen)", generate_diagonal(17, true, 0.03, 100)),
        ("circle (unseen radius/length)", generate_circle(30, 0.75, 0.06, 101)),
        ("circle, small radius (unseen)", generate_circle(13, 0.5, 0.02, 102)),
    ];

    println!("{:<34} {:<10} {:>10}", "Test stroke", "Predicted", "Confidence");
    println!("{}", "-".repeat(56));
    for (description, stroke) in &test_cases {
        let pattern = canvas.encode_sequence(stroke);
        match w.classify(&pattern) {
            Some((label, confidence)) => {
                println!("{description:<34} {label:<10} {confidence:>10.3}");
            }
            None => println!("{description:<34} <no prediction>"),
        }
    }
}
