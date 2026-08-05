use tin_man::{Wisard, KernelCanvas};

fn main() {
    // 32 kernels in 2D space, 4 thermometer bits per kernel,
    // top-20% closest kernels activated per point.
    let canvas = KernelCanvas::new(32, 2, 4, 0.2, 42);

    let stroke_down = vec![vec![-0.8, -0.8], vec![-0.4, -0.4], vec![0.0, 0.0]];
    let stroke_up = vec![vec![0.8, 0.8], vec![0.4, 0.4], vec![0.0, 0.0]];

    let mut w = Wisard::new(canvas.output_size(), 8, 0.1, true, false, false);
    w.train(&canvas.encode_sequence(&stroke_down), "diagonal_down");
    w.train(&canvas.encode_sequence(&stroke_up), "diagonal_up");

    let test = vec![vec![-0.6, -0.6], vec![-0.2, -0.2]];
    let (label, confidence) = w.classify(&canvas.encode_sequence(&test)).unwrap();
    println!("{label} ({confidence:.2})");
}
