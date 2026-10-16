mod common;

use p3_challenger::{CanObserve, FieldChallenger};
use p3_field::PrimeCharacteristicRing;
use spartan_whir::protocol::SpartanContextEngine;
use spartan_whir::{
    engine::F, DomainSeparator, KeccakQuarticEngine, MatrixClosingMode, Poseidon1QuarticEngine,
    PoseidonQuarticEngine, QuarticBinExtension, SecurityConfig, SparkWhirParams,
    WhirFoldingSchedule, WhirParams, FULL_ZK_PROTOCOL_ID, NO_ZK_PROTOCOL_ID,
    SPARK_MATRIX_CLOSING_VERSION,
};

fn expected_shared_direct_body(relation_digest: [u8; 32]) -> Vec<u8> {
    let mut expected = vec![
        0, // DirectSparse
        1, 0, 0, 0, 0, 0, 0, 0, // num_cons
        1, 0, 0, 0, 0, 0, 0, 0, // num_vars
        1, 0, 0, 0, 0, 0, 0, 0, // num_io
        100, 0, 0, 0, // security_level_bits
        100, 0, 0, 0, // merkle_security_bits
        2, // CapacityBound
        0, 0, 0, 0, // pow_bits
        4, 0, 0, 0, 0, 0, 0, 0, // folding_factor
        1, 0, 0, 0, 0, 0, 0, 0, // starting_log_inv_rate
        1, 0, 0, 0, 0, 0, 0, 0, // rs_domain_initial_reduction_factor
    ];
    expected.splice(25..25, relation_digest);
    expected
}

fn absorb_separator(separator: &DomainSeparator) -> QuarticBinExtension {
    let mut challenger = spartan_whir::poseidon_challenger();
    for byte in separator.to_bytes() {
        challenger.observe(F::from_u8(byte));
    }
    challenger.sample_algebra_element::<QuarticBinExtension>()
}

fn first_no_zk_challenge<E>(separator: &DomainSeparator) -> QuarticBinExtension
where
    E: SpartanContextEngine<EF = QuarticBinExtension>,
    E::Challenger: FieldChallenger<F>,
{
    let mut challenger = E::challenger();
    E::observe_spartan_context(&mut challenger, separator, &[F::from_u8(9)])
        .expect("context absorption succeeds");
    challenger.sample_algebra_element::<QuarticBinExtension>()
}

#[test]
fn domain_separator_encoding_is_deterministic() {
    let shape = common::koala_shape_single_constraint(1);
    let security = SecurityConfig::default();
    let whir = WhirParams::default();

    let a = DomainSeparator::new(&shape, &security, &whir).expect("valid relation");
    let b = DomainSeparator::new(&shape, &security, &whir).expect("valid relation");

    assert_eq!(a, b);
    assert_eq!(a.to_bytes(), b.to_bytes());
    let expected = [
        &(NO_ZK_PROTOCOL_ID.len() as u64).to_le_bytes(),
        b"spartan-whir-no-zk-v1".as_slice(),
        &expected_shared_direct_body(a.relation_digest),
    ]
    .concat();
    assert_eq!(a.to_bytes(), expected);
}

#[test]
fn no_zk_and_full_zk_separators_share_only_the_canonical_body() {
    let shape = common::koala_shape_single_constraint(1);
    let security = SecurityConfig::default();
    let whir = WhirParams::default();
    let no_zk = DomainSeparator::new(&shape, &security, &whir).expect("valid relation");
    let full_zk = DomainSeparator::new_full_zk(
        &shape,
        &security,
        &whir,
        MatrixClosingMode::DirectSparse,
        None,
    )
    .expect("valid relation");

    let expected_full_zk = [
        &(FULL_ZK_PROTOCOL_ID.len() as u64).to_le_bytes(),
        b"spartan-whir-full-zk-v1".as_slice(),
        &expected_shared_direct_body(full_zk.relation_digest),
    ]
    .concat();
    assert_eq!(full_zk.to_bytes(), expected_full_zk);
    assert_eq!(
        &no_zk.to_bytes()[8 + NO_ZK_PROTOCOL_ID.len()..],
        &full_zk.to_bytes()[8 + FULL_ZK_PROTOCOL_ID.len()..]
    );
    assert_ne!(absorb_separator(&no_zk), absorb_separator(&full_zk));
}

#[test]
fn protocol_identifier_length_frames_prefix_related_modes() {
    let shape = common::koala_shape_single_constraint(1);
    let separator =
        DomainSeparator::new(&shape, &SecurityConfig::default(), &WhirParams::default())
            .expect("valid relation");
    let mut suffixed = separator.clone();
    suffixed
        .protocol_id
        .extend_from_slice(b"-fresh-mask-batching-v1");

    for value in [&separator, &suffixed] {
        let bytes = value.to_bytes();
        let length = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        assert_eq!(length, value.protocol_id.len());
        assert_eq!(&bytes[8..8 + length], value.protocol_id);
        assert_eq!(
            &bytes[8 + length..],
            expected_shared_direct_body(value.relation_digest)
        );
    }
    assert!(suffixed.protocol_id.starts_with(&separator.protocol_id));
    assert_ne!(&separator.to_bytes()[..8], &suffixed.to_bytes()[..8]);
    assert_ne!(absorb_separator(&separator), absorb_separator(&suffixed));
    assert_ne!(
        first_no_zk_challenge::<KeccakQuarticEngine>(&separator),
        first_no_zk_challenge::<KeccakQuarticEngine>(&suffixed)
    );
}

#[test]
fn domain_separator_changes_before_challenges_when_relation_changes() {
    let shape = common::koala_shape_single_constraint(1);
    let security = SecurityConfig::default();
    let whir = WhirParams::default();

    let baseline = DomainSeparator::new(&shape, &security, &whir).expect("valid relation");
    let mut changed_relations = [shape.clone(), shape.clone(), shape];
    changed_relations[0].a.entries[0].val += F::ONE;
    changed_relations[1].b.entries[0].val += F::ONE;
    changed_relations[2].c.entries[0].val += F::ONE;

    for changed_relation in &changed_relations {
        let changed = DomainSeparator::new(changed_relation, &security, &whir)
            .expect("changed relation is valid");
        assert_ne!(baseline.relation_digest, changed.relation_digest);
        assert_ne!(baseline.to_bytes(), changed.to_bytes());
        assert_ne!(absorb_separator(&baseline), absorb_separator(&changed));
        assert_ne!(
            first_no_zk_challenge::<KeccakQuarticEngine>(&baseline),
            first_no_zk_challenge::<KeccakQuarticEngine>(&changed)
        );
        assert_ne!(
            first_no_zk_challenge::<PoseidonQuarticEngine>(&baseline),
            first_no_zk_challenge::<PoseidonQuarticEngine>(&changed)
        );
        assert_ne!(
            first_no_zk_challenge::<Poseidon1QuarticEngine>(&baseline),
            first_no_zk_challenge::<Poseidon1QuarticEngine>(&changed)
        );
    }
}

#[test]
fn domain_separator_changes_when_matrix_closing_changes() {
    let shape = common::koala_shape_single_constraint(1);
    let security = SecurityConfig::default();
    let whir = WhirParams::default();

    let direct = DomainSeparator::new_with_matrix_closing(
        &shape,
        &security,
        &whir,
        MatrixClosingMode::DirectSparse,
    )
    .expect("valid relation");
    let spark = DomainSeparator::new_with_matrix_closing(
        &shape,
        &security,
        &whir,
        MatrixClosingMode::Spark,
    )
    .expect("valid relation");

    assert_ne!(direct.to_bytes(), spark.to_bytes());
    assert_eq!(direct.to_bytes()[8 + NO_ZK_PROTOCOL_ID.len()], 0);
    assert_eq!(spark.to_bytes()[8 + NO_ZK_PROTOCOL_ID.len()], 1);
    assert_eq!(
        spark.to_bytes()[8 + NO_ZK_PROTOCOL_ID.len() + 1],
        SPARK_MATRIX_CLOSING_VERSION
    );
}

#[test]
fn domain_separator_canonicalizes_legacy_constant_schedule() {
    let shape = common::koala_shape_single_constraint(1);
    let security = SecurityConfig::default();
    let legacy = WhirParams::default();
    let explicit = WhirParams {
        folding_schedule: Some(WhirFoldingSchedule::Constant(legacy.folding_factor)),
        ..legacy.clone()
    };

    let legacy_bytes = DomainSeparator::new(&shape, &security, &legacy)
        .expect("valid relation")
        .to_bytes();
    let explicit_bytes = DomainSeparator::new(&shape, &security, &explicit)
        .expect("valid relation")
        .to_bytes();

    assert_eq!(legacy_bytes, explicit_bytes);
}

#[test]
fn domain_separator_binds_explicit_round_log_inv_rates() {
    let shape = common::koala_shape_single_constraint(1);
    let security = SecurityConfig::default();
    let rate_three = WhirParams {
        round_log_inv_rates: vec![3],
        ..WhirParams::default()
    };
    let rate_four = WhirParams {
        round_log_inv_rates: vec![4],
        ..rate_three.clone()
    };

    let rate_three = DomainSeparator::new(&shape, &security, &rate_three).expect("valid relation");
    let rate_four = DomainSeparator::new(&shape, &security, &rate_four).expect("valid relation");

    assert_ne!(rate_three.to_bytes(), rate_four.to_bytes());
    assert_ne!(absorb_separator(&rate_three), absorb_separator(&rate_four));
}

#[test]
fn domain_separator_ignores_spark_params_in_direct_mode() {
    let shape = common::koala_shape_single_constraint(1);
    let security = SecurityConfig::default();
    let whir = WhirParams::default();
    let spark_whir_params = SparkWhirParams {
        fixed_value: WhirParams {
            starting_log_inv_rate: 2,
            ..whir.clone()
        },
        fixed_audit: WhirParams {
            starting_log_inv_rate: 3,
            ..whir.clone()
        },
        read: WhirParams {
            starting_log_inv_rate: 4,
            ..whir.clone()
        },
    };

    let direct = DomainSeparator::new_with_matrix_closing(
        &shape,
        &security,
        &whir,
        MatrixClosingMode::DirectSparse,
    )
    .expect("valid relation");
    let direct_with_spark_params = DomainSeparator::new_with_matrix_closing_and_spark_whir_params(
        &shape,
        &security,
        &whir,
        MatrixClosingMode::DirectSparse,
        Some(spark_whir_params.clone()),
    )
    .expect("valid relation");
    let spark_with_spark_params = DomainSeparator::new_with_matrix_closing_and_spark_whir_params(
        &shape,
        &security,
        &whir,
        MatrixClosingMode::Spark,
        Some(spark_whir_params),
    )
    .expect("valid relation");

    assert_eq!(direct_with_spark_params.spark_whir_params, None);
    assert_eq!(direct.to_bytes(), direct_with_spark_params.to_bytes());
    assert_ne!(direct.to_bytes(), spark_with_spark_params.to_bytes());
}
