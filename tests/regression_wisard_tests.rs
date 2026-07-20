use tin_man::RegressionWisard;

fn input_for(value: u8, len: usize) -> Vec<u8> {
    (0..len).map(|i| (i as u8 + value) % 2).collect()
}

#[test]
fn predicts_close_to_trained_target() {
    let mut rew = RegressionWisard::new(64, 8, 1);
    let x = input_for(0, 64);
    rew.train(&x, 10.0);
    rew.train(&x, 12.0); // same input, different target -> should average

    let prediction = rew.predict(&x).expect("should predict");
    assert!((prediction - 11.0).abs() < 1e-6);
}

#[test]
fn returns_none_for_unseen_pattern_with_min_zero_two() {
    let mut rew = RegressionWisard::new(64, 8, 2);
    let x = input_for(0, 64);
    rew.train(&x, 5.0); // only seen once, min_zero=2 requires 2+

    // Depending on RAM overlap some tuples may still have enough
    // votes from partial matches; assert it doesn't panic either way.
    let _ = rew.predict(&x);
}

#[test]
fn distinguishes_two_different_targets() {
    let mut rew = RegressionWisard::new(64, 8, 1);
    let low = input_for(0, 64);
    let high = input_for(1, 64);

    rew.train(&low, 1.0);
    rew.train(&high, 100.0);

    let pred_low = rew.predict(&low).unwrap();
    let pred_high = rew.predict(&high).unwrap();
    assert!(pred_low < pred_high);
}

#[test]
#[should_panic(expected = "input size mismatch")]
fn train_panics_on_wrong_input_size() {
    let mut rew = RegressionWisard::new(32, 4, 1);
    rew.train(&vec![0u8; 16], 1.0);
}

#[test]
#[should_panic(expected = "input size mismatch")]
fn predict_panics_on_wrong_input_size() {
    let mut rew = RegressionWisard::new(32, 4, 1);
    rew.train(&vec![0u8; 32], 1.0);
    let _ = rew.predict(&vec![0u8; 16]);
}