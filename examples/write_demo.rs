use tin_man::{FileFormat, Wisard, ClusWisard, RegressionWisard};

fn main() -> std::io::Result<()> {
    let mut w = Wisard::new(64, 8, 0.1, true, false);
    w.train(&vec![1, 0, 1, 0].repeat(16), "class_a");
    w.train(&vec![0, 1, 0, 1].repeat(16), "class_b");

    w.save_to_file("wisard_model.json", FileFormat::Json)?;
    let w_loaded = Wisard::load_from_file("wisard_model.json", FileFormat::Json)?;
    println!("{:?}", w_loaded.classify(&vec![1, 0, 1, 0].repeat(16)));

    let mut clus = ClusWisard::new(64, 8, 0.3, 20, 5, 0.1, true, false);
    clus.train(&vec![1, 1, 0, 0].repeat(16), "cold");
    clus.save_to_file("clus_model.bin", FileFormat::Binary)?;
    let clus_loaded = ClusWisard::load_from_file("clus_model.bin", FileFormat::Binary)?;
    println!("{:?}", clus_loaded.classify(&vec![1, 1, 0, 0].repeat(16)));

    let mut rew = RegressionWisard::new(64, 8, 1);
    rew.train(&vec![1, 0, 1, 0].repeat(16), 10.5);
    rew.save_to_file("regression_model.json", FileFormat::Json)?;
    let rew_loaded = RegressionWisard::load_from_file("regression_model.json", FileFormat::Json)?;
    println!("{:?}", rew_loaded.predict(&vec![1, 0, 1, 0].repeat(16)));

    Ok(())
}