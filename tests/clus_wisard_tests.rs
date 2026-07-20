use tin_man::ClusWisard;

fn pattern(seed: u8, len: usize) -> Vec<u8> {
    (0..len).map(|i| (i as u8 + seed) % 2).collect()
}

#[test]
fn classifies_simple_two_class_case() {
    let mut clus = ClusWisard::new(64, 8, 0.3, 20, 5, 0.1, true, false);
    clus.train(&pattern(0, 64), "even_start");
    clus.train(&pattern(1, 64), "odd_start");

    let (label, score) = clus.classify(&pattern(0, 64)).expect("should classify");
    assert_eq!(label, "even_start");
    assert!(score > 0.0);
}

#[test]
fn spawns_new_cluster_when_min_score_not_met() {
    // Very high min_score forces new clusters for any imperfect match.
    let mut clus = ClusWisard::new(64, 8, 0.99, 100, 10, 0.1, true, false);
    clus.train(&pattern(0, 64), "shape");
    clus.train(&pattern(3, 64), "shape"); // dissimilar pattern, same label

    // Should not panic and should still classify successfully.
    let result = clus.classify(&pattern(0, 64));
    assert!(result.is_some());
}

#[test]
fn respects_discriminator_limit() {
    // discriminator_limit = 1 forces all training into a single cluster
    // per class, regardless of min_score mismatches.
    let mut clus = ClusWisard::new(32, 4, 0.99, 1, 1, 0.1, true, false);
    for seed in 0..5u8 {
        clus.train(&pattern(seed, 32), "single_cluster_class");
    }
    let result = clus.classify(&pattern(0, 32));
    assert!(result.is_some());
}

#[test]
fn returns_none_when_untrained() {
    let clus = ClusWisard::new(32, 4, 0.3, 20, 5, 0.1, true, false);
    assert!(clus.classify(&vec![0u8; 32]).is_none());
}