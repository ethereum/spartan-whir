mod common;

use p3_challenger::FieldChallenger;
use p3_field::PrimeCharacteristicRing;
use rand::{rngs::StdRng, SeedableRng};
use spartan_whir::{
    generate_satisfiable_fixture,
    pcs_config::{FreshMaskBatching, MaskPacking},
    proof_compression::{CompressedZkProofFor, ProofCompressionOptions},
    MatrixClosingMode, PoseidonZkSetupConfig, QuinticExtension, SparkWhirParams,
    SyntheticR1csConfig,
};

fn config(closing: MatrixClosingMode) -> PoseidonZkSetupConfig {
    let params = common::phase3_whir_params();
    PoseidonZkSetupConfig {
        matrix_closing: closing,
        security: common::phase3_security(),
        whir_params: common::phase3_zk_whir_params(),
        spark_whir_params: (closing == MatrixClosingMode::Spark).then_some(SparkWhirParams {
            fixed_value: params.clone(),
            fixed_audit: params.clone(),
            read: params,
        }),
        ell_zk: 3,
        mask_log_inv_rate: 3,
    }
}

macro_rules! packing_test {
    ($name:ident, $engine:ident, $protocol:ident, $vk:ident, $challenger:path) => {
        #[test]
        fn $name() {
            type E = spartan_whir::$engine<QuinticExtension>;
            type Protocol = spartan_whir::$protocol<QuinticExtension>;
            type Vk = spartan_whir::$vk<QuinticExtension>;
            // Minimum supported outer arity, unpadded dimensions, and a larger
            // independently generated relation exercise different mask widths.
            for (constraints, log_witness, seed) in [(2, 2, 23), (3, 3, 37), (9, 4, 61)] {
                let fixture = generate_satisfiable_fixture(&SyntheticR1csConfig {
                    target_log2_witness_poly: log_witness,
                    num_constraints: constraints,
                    num_io: 1,
                    a_terms_per_constraint: 2,
                    b_terms_per_constraint: 2,
                    seed,
                })
                .unwrap();
                for closing in [MatrixClosingMode::DirectSparse, MatrixClosingMode::Spark] {
                    let (_, control) =
                        spartan_whir::poseidon::setup_poseidon_zk_with_mask_packing::<E>(
                            fixture.shape.clone(),
                            config(closing),
                            FreshMaskBatching::SameHeight,
                            MaskPacking::Off,
                        )
                        .unwrap();
                    for packing in [
                        MaskPacking::Off,
                        MaskPacking::Application,
                        MaskPacking::All,
                        MaskPacking::ApplicationFreeBasis,
                        MaskPacking::AllFreeBasis,
                    ] {
                        let (pk, vk) =
                            spartan_whir::poseidon::setup_poseidon_zk_with_mask_packing::<E>(
                                fixture.shape.clone(),
                                config(closing),
                                FreshMaskBatching::SameHeight,
                                packing,
                            )
                            .unwrap();
                        assert_eq!(
                            vk.domain_separator().relation_digest,
                            control.domain_separator().relation_digest
                        );
                        assert_eq!(
                            vk.domain_separator().whir_params,
                            control.domain_separator().whir_params
                        );
                        assert_eq!(
                            vk.domain_separator().spark_whir_params,
                            control.domain_separator().spark_whir_params
                        );
                        assert_eq!(
                            vk.domain_separator().protocol_id
                                == control.domain_separator().protocol_id,
                            packing == MaskPacking::Off
                        );
                        let mut prover = $challenger();
                        let (instance, proof) = Protocol::prove_with_rng(
                            &pk,
                            &fixture.public_inputs,
                            &fixture.witness,
                            &mut prover,
                            &mut StdRng::seed_from_u64(111),
                        )
                        .unwrap();
                        let mut verifier = $challenger();
                        Protocol::verify(&vk, &instance, &proof, &mut verifier).unwrap();
                        for _ in 0..8 {
                            assert_eq!(
                                prover.sample_algebra_element::<QuinticExtension>(),
                                verifier.sample_algebra_element::<QuinticExtension>()
                            );
                        }
                        if packing != MaskPacking::Off {
                            assert!(Protocol::verify(
                                &control,
                                &instance,
                                &proof,
                                &mut $challenger()
                            )
                            .is_err());
                            assert_eq!(
                                proof.pcs_proof.base_case.carried_mask_openings[0].rows[0].len(),
                                1
                            );
                        }
                        if packing.sumchecks() {
                            assert!(proof
                                .pcs_proof
                                .base_case
                                .carried_mask_openings
                                .iter()
                                .all(|opening| opening.rows.iter().all(|row| row.len() == 1)));
                        }
                        if constraints == 3 && closing == MatrixClosingMode::Spark {
                            let words = spartan_whir::encode_full_zk_spark_guest_words::<E>(
                                &instance, &proof,
                            )
                            .unwrap();
                            let (decoded_instance, decoded) =
                                spartan_whir::decode_full_zk_spark_guest_words::<E>(&words)
                                    .unwrap();
                            Protocol::verify(&vk, &decoded_instance, &decoded, &mut $challenger())
                                .unwrap();
                            for mutation in 0..4 {
                                let mut altered = proof.clone();
                                match mutation {
                                    0 => {
                                        altered.pcs_proof.base_case.carried_mask_openings[0].rows
                                            [0]
                                        .push(QuinticExtension::ZERO);
                                    }
                                    1 => {
                                        altered.pcs_proof.base_case.blinded_masks[0].message[0] +=
                                            QuinticExtension::ONE;
                                    }
                                    2 => {
                                        altered.pcs_proof.base_case.fresh_mask_commitments.pop();
                                    }
                                    3 => {
                                        altered.pcs_proof.base_case.carried_mask_openings.pop();
                                    }
                                    _ => unreachable!(),
                                }
                                assert!(Protocol::verify(
                                    &vk,
                                    &instance,
                                    &altered,
                                    &mut $challenger()
                                )
                                .is_err());
                            }
                            let mut restored: Vk =
                                bincode::deserialize(&bincode::serialize(&vk).unwrap()).unwrap();
                            assert!(Protocol::verify(
                                &restored,
                                &instance,
                                &proof,
                                &mut $challenger()
                            )
                            .is_err());
                            restored.authenticate().unwrap();
                            Protocol::verify(&restored, &instance, &proof, &mut $challenger())
                                .unwrap();
                            let mut key_json = serde_json::to_value(&vk).unwrap();
                            key_json["pcs_config"]["mask_packing"] =
                                serde_json::json!(if packing == MaskPacking::Off {
                                    "all"
                                } else {
                                    "off"
                                });
                            let mut altered_key: Vk = serde_json::from_value(key_json).unwrap();
                            assert!(altered_key.authenticate().is_err());
                            let (compressed_instance, compressed) =
                                Protocol::prove_compressed_with_rng(
                                    &pk,
                                    &fixture.public_inputs,
                                    &fixture.witness,
                                    &mut $challenger(),
                                    &mut StdRng::seed_from_u64(111),
                                    ProofCompressionOptions::recommended(),
                                )
                                .unwrap();
                            let bytes = compressed.to_bytes().unwrap();
                            let decoded = CompressedZkProofFor::<E>::from_bytes(&bytes).unwrap();
                            Protocol::verify_compressed(
                                &vk,
                                &compressed_instance,
                                decoded,
                                &mut $challenger(),
                            )
                            .unwrap();
                            let mut trailing = bytes.clone();
                            trailing.push(0);
                            assert!(CompressedZkProofFor::<E>::from_bytes(&trailing).is_err());
                            assert!(CompressedZkProofFor::<E>::from_bytes(
                                &bytes[..bytes.len() - 1]
                            )
                            .is_err());
                        }
                    }
                }
            }
        }
    };
}

packing_test!(
    poseidon1_mask_packing_roundtrips,
    Poseidon1Engine,
    Poseidon1ZkSpartanProtocol,
    Poseidon1ZkVerifyingKey,
    spartan_whir::poseidon1_challenger
);
packing_test!(
    poseidon2_mask_packing_roundtrips,
    PoseidonEngine,
    PoseidonZkSpartanProtocol,
    PoseidonZkVerifyingKey,
    spartan_whir::poseidon_zk_challenger
);
