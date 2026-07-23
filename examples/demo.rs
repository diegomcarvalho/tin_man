use tin_man::{ClusWisard, RegressionWisard, Wisard};

fn main() {
    // Standard WiSARD
    let mut w = Wisard::new(64, 8, 0.1, true, false, true);
    w.train(&vec![1, 0, 1, 0].repeat(16), "class_a");
    w.train(&vec![0, 1, 0, 1].repeat(16), "class_b");
    if let Some((label, conf)) = w.classify(&vec![1, 0, 1, 0].repeat(16)) {
        println!("WiSARD: {} ({:.2})", label, conf);
    }

    // ClusWiSARD
    let mut clus = ClusWisard::new(64, 8, 0.3, 20, 5, 0.1, true, false);
    clus.train(&vec![1, 0, 1, 1].repeat(16), "cold");
    clus.train(&vec![0, 1, 0, 0].repeat(16), "hot");
    if let Some((label, score)) = clus.classify(&vec![1, 0, 1, 1].repeat(16)) {
        println!("ClusWiSARD: {} ({:.2})", label, score);
    }

    // Regression WiSARD
    let mut rew = RegressionWisard::new(64, 8, 1);
    rew.train(&vec![1, 0, 1, 0].repeat(16), 10.5);
    rew.train(&vec![1, 0, 1, 1].repeat(16), 12.0);
    if let Some(pred) = rew.predict(&vec![1, 0, 1, 0].repeat(16)) {
        println!("RegressionWiSARD prediction: {:.2}", pred);
    }
}