mod common;

use p3_challenger::FieldChallenger;
use p3_field::PrimeCharacteristicRing;
use rand::{rngs::StdRng, SeedableRng};
use spartan_whir::proof_compression::{CompressedZkProofFor, ProofCompressionOptions};
use spartan_whir::{
    generate_satisfiable_fixture, MatrixClosingMode, PoseidonZkSetupConfig, QuinticExtension,
    SparkWhirParams, SyntheticR1csConfig,
};

macro_rules! profile_test {
    ($name:ident, $setup:path, $challenger:path, $protocol:ident, $engine:ident, $batching:expr) => {
        #[test]
        fn $name() {
            let fixture = generate_satisfiable_fixture(&SyntheticR1csConfig {
                target_log2_witness_poly: 3,
                num_constraints: 4,
                num_io: 1,
                a_terms_per_constraint: 2,
                b_terms_per_constraint: 2,
                seed: 27183,
            })
            .unwrap();
            let params = common::phase3_whir_params();
            let config = PoseidonZkSetupConfig {
                matrix_closing: MatrixClosingMode::Spark,
                security: common::phase3_security(),
                whir_params: common::phase3_zk_whir_params(),
                spark_whir_params: Some(SparkWhirParams {
                    fixed_value: params.clone(),
                    fixed_audit: params.clone(),
                    read: params,
                }),
                ell_zk: 3,
                mask_log_inv_rate: 3,
            };
            let (pk, vk) = spartan_whir::poseidon::setup_poseidon_zk_with_fresh_mask_batching::<
                spartan_whir::$engine<QuinticExtension>,
            >(fixture.shape, config, $batching)
            .unwrap();
            for seed in [181, 293] {
                let mut rng = StdRng::seed_from_u64(seed);
                let mut prover = $challenger().with_trace();
                let (instance, ordinary) =
                    spartan_whir::$protocol::<QuinticExtension>::prove_with_rng(
                        &pk,
                        &fixture.public_inputs,
                        &fixture.witness,
                        &mut prover,
                        &mut rng,
                    )
                    .unwrap();
                let expected_prover_trace = prover.transcript_trace();
                let mut verifier = $challenger().with_trace();
                spartan_whir::$protocol::<QuinticExtension>::verify(
                    &vk,
                    &instance,
                    &ordinary,
                    &mut verifier,
                )
                .unwrap();
                let expected_trace = verifier.transcript_trace();
                let mut options = ProofCompressionOptions::default();
                for stage in 0..10 {
                    match stage {
                        1 => options.structured_rows = true,
                        2 => options.fresh_rows = true,
                        3 => options.packed_fields = true,
                        4 => options.compact_integers = true,
                        5 => options.factored_rounds = true,
                        6 => options.final_rows = true,
                        7 => options.derived_products = true,
                        8 => options.refined_metadata = true,
                        9 => options.duplicate_columns = true,
                        _ => {}
                    }
                    let mut rng = StdRng::seed_from_u64(seed);
                    let mut compact_prover = $challenger().with_trace();
                    let (compact_instance, compact) =
                        spartan_whir::$protocol::<QuinticExtension>::prove_compressed_with_rng(
                            &pk,
                            &fixture.public_inputs,
                            &fixture.witness,
                            &mut compact_prover,
                            &mut rng,
                            options,
                        )
                        .unwrap();
                    assert_eq!(
                        bincode::serialize(&compact_instance).unwrap(),
                        bincode::serialize(&instance).unwrap()
                    );
                    // Branches share a trace sink; parallel execution may interleave
                    // independent events. Serial runs compare the complete trace.
                    if p3_maybe_rayon::prelude::current_num_threads() == 1 {
                        assert!(
                            compact_prover.transcript_trace() == expected_prover_trace,
                            "prover stage {stage}"
                        );
                    }
                    let mut expected_prover = prover.clone();
                    for _ in 0..16 {
                        assert_eq!(
                            compact_prover.sample_algebra_element::<QuinticExtension>(),
                            expected_prover.sample_algebra_element::<QuinticExtension>()
                        );
                    }
                    let bytes = compact.to_bytes().unwrap();
                    type Compressed = CompressedZkProofFor<spartan_whir::$engine<QuinticExtension>>;
                    let decoded = Compressed::from_bytes(&bytes).unwrap();
                    assert_eq!(decoded.to_bytes().unwrap(), bytes);
                    let mut compact_verifier = $challenger().with_trace();
                    spartan_whir::$protocol::<QuinticExtension>::verify_compressed(
                        &vk,
                        &instance,
                        decoded,
                        &mut compact_verifier,
                    )
                    .unwrap();
                    if p3_maybe_rayon::prelude::current_num_threads() == 1 {
                        assert!(
                            compact_verifier.transcript_trace() == expected_trace,
                            "verifier stage {stage}"
                        );
                    }
                    let mut expected_verifier = verifier.clone();
                    for _ in 0..16 {
                        assert_eq!(
                            compact_verifier.sample_algebra_element::<QuinticExtension>(),
                            expected_verifier.sample_algebra_element::<QuinticExtension>()
                        );
                    }

                    let mut bad_instance = instance.clone();
                    bad_instance.public_inputs[0] += spartan_whir::engine::F::ONE;
                    assert!(
                        spartan_whir::$protocol::<QuinticExtension>::verify_compressed(
                            &vk,
                            &bad_instance,
                            Compressed::from_bytes(&bytes).unwrap(),
                            &mut $challenger(),
                        )
                        .is_err()
                    );
                    for truncated in [0, 4, 5, bytes.len() / 2, bytes.len() - 1] {
                        assert!(Compressed::from_bytes(&bytes[..truncated]).is_err());
                    }
                    let mut extra = bytes.clone();
                    extra.push(0);
                    assert!(Compressed::from_bytes(&extra).is_err());
                    let mut bad_flags = bytes.clone();
                    bad_flags[5] |= 128;
                    assert!(Compressed::from_bytes(&bad_flags).is_err());
                    if options.factored_rounds {
                        let mut wrong_mode = bytes.clone();
                        wrong_mode[4] ^= 64;
                        assert!(Compressed::from_bytes(&wrong_mode).is_err());
                    }
                }
            }
        }
    };
}

profile_test!(
    poseidon1_compression_preserves_full_transcript,
    spartan_whir::setup_poseidon1_zk,
    spartan_whir::poseidon1_challenger,
    Poseidon1ZkSpartanProtocol,
    Poseidon1Engine,
    spartan_whir::pcs_config::FreshMaskBatching::Separate
);
profile_test!(
    poseidon2_compression_preserves_full_transcript,
    spartan_whir::setup_poseidon_zk,
    spartan_whir::poseidon_zk_challenger,
    PoseidonZkSpartanProtocol,
    PoseidonEngine,
    spartan_whir::pcs_config::FreshMaskBatching::Separate
);

profile_test!(
    poseidon1_grouped_compression_preserves_full_transcript,
    spartan_whir::setup_poseidon1_zk,
    spartan_whir::poseidon1_challenger,
    Poseidon1ZkSpartanProtocol,
    Poseidon1Engine,
    spartan_whir::pcs_config::FreshMaskBatching::SameHeight
);
