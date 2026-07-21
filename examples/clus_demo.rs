use tin_man::ClusRegressionWisard;

fn main() {
    // min_score=0.3, threshold=20 cycles, up to 5 clusters per group, min_zero=1
    let mut model = ClusRegressionWisard::new(64, 8, 0.3, 20, 5, 1);

    let low_input = vec![1, 0, 1, 0].repeat(16);
    let high_input = vec![0, 1, 0, 1].repeat(16);

    model.train(&low_input, "sensor_a", 10.5);
    model.train(&high_input, "sensor_a", 95.0); // heterogeneous target -> spawns new cluster

    if let Some(pred) = model.predict(&low_input) {
        println!("Predicted: {:.2}", pred);
    }

    if let Some(pred) = model.predict_in_group(&high_input, "sensor_a") {
        println!("Predicted within group: {:.2}", pred);
    }
}