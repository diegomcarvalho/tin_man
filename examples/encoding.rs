use tin_man::{DistributiveThermometer, GaussianThermometer, LinearThermometer, Wisard};

fn main() {
    let data = vec![1.2, 3.4, 2.1, 5.6, 4.3, 2.9, 3.8, 1.9, 4.7, 3.1];

    // Linear: uniform bins across the observed range
    let linear = LinearThermometer::fit(&data, 8);
    println!("{:?}", linear.encode(3.0));

    // Gaussian: bins denser near the mean, sparser in the tails
    let gaussian = GaussianThermometer::fit(&data, 8);
    println!("{:?}", gaussian.encode(3.0));

    // Distributive: bins sized so each holds ~equal sample counts
    let distributive = DistributiveThermometer::fit(&data, 8);
    println!("{:?}", distributive.encode(3.0));

    // Encoding multiple features into one WiSARD-ready input vector
    let feature_1 = vec![1.2, 3.4, 2.1];
    let feature_2 = vec![10.0, 20.0, 15.0];
    let enc1 = LinearThermometer::fit(&feature_1, 4);
    let enc2 = LinearThermometer::fit(&feature_2, 4);

    let mut w = Wisard::new(8, 4, 0.1, true, false, true);
    let input: Vec<u8> = enc1
        .encode(feature_1[0])
        .into_iter()
        .chain(enc2.encode(feature_2[0]))
        .collect();
    w.train(&input, "sample_class");
}