use tin_man::Wisard;

fn pattern_a(len: usize) -> Vec<u8> {
    vec![1, 1, 0, 0].into_iter().cycle().take(len).collect()
}

fn pattern_b(len: usize) -> Vec<u8> {
    vec![0, 0, 1, 1].into_iter().cycle().take(len).collect()
}

#[test]
fn classifies_exact_training_sample_correctly() {
    let mut w = Wisard::new(64, 8, 0.1, false, false, true);
    w.train(&pattern_a(64), "a");
    w.train(&pattern_b(64), "b");

    let (label, confidence) = w.classify(&pattern_a(64)).expect("should classify");
    assert_eq!(label, "a");
    assert!(confidence > 0.0);
}

#[test]
fn distinguishes_two_well_separated_classes() {
    let mut w = Wisard::new(64, 8, 0.1, true, false, true);
    for _ in 0..5 {
        w.train(&pattern_a(64), "a");
    }
    for _ in 0..5 {
        w.train(&pattern_b(64), "b");
    }

    let (label_a, _) = w.classify(&pattern_a(64)).unwrap();
    let (label_b, _) = w.classify(&pattern_b(64)).unwrap();
    assert_eq!(label_a, "a");
    assert_eq!(label_b, "b");
}

#[test]
fn returns_none_when_untrained() {
    let w = Wisard::new(32, 4, 0.1, true, false, true);
    assert!(w.classify(&vec![0u8; 32]).is_none());
}

#[test]
fn bleaching_and_binary_modes_agree_on_clear_cases() {
    let mut w_binary = Wisard::new(64, 8, 0.1, false, false, true);
    let mut w_bleach = Wisard::new(64, 8, 0.1, true, false, true);

    for _ in 0..3 {
        w_binary.train(&pattern_a(64), "a");
        w_bleach.train(&pattern_a(64), "a");
    }
    w_binary.train(&pattern_b(64), "b");
    w_bleach.train(&pattern_b(64), "b");

    let (label_binary, _) = w_binary.classify(&pattern_a(64)).unwrap();
    let (label_bleach, _) = w_bleach.classify(&pattern_a(64)).unwrap();
    assert_eq!(label_binary, "a");
    assert_eq!(label_bleach, "a");
}

#[test]
fn ignore_zero_does_not_crash_and_still_classifies() {
    let mut w = Wisard::new(32, 4, 0.1, true, true, true);
    w.train(&vec![0u8; 32], "all_zero");
    w.train(&pattern_a(32), "mixed");

    let result = w.classify(&pattern_a(32));
    assert!(result.is_some());
}

#[test]
#[should_panic(expected = "input size mismatch")]
fn train_panics_on_wrong_input_size() {
    let mut w = Wisard::new(32, 4, 0.1, true, false, true);
    w.train(&vec![0u8; 16], "bad");
}

#[test]
#[should_panic(expected = "input size mismatch")]
fn classify_panics_on_wrong_input_size() {
    let mut w = Wisard::new(32, 4, 0.1, true, false,true);
    w.train(&vec![0u8; 32], "ok");
    let _ = w.classify(&vec![0u8; 16]);
}