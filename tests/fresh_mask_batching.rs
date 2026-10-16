mod common;

use p3_challenger::FieldChallenger;
use p3_field::PrimeCharacteristicRing;
use rand::{rngs::StdRng, SeedableRng};
use spartan_whir::{
    engine::F,
    fixed_oracle_cache::{CachedFixedOracleProofFor, FixedOracleCacheFor},
    generate_satisfiable_fixture,
    pcs_config::FreshMaskBatching,
    proof_compression::{CompressedZkProofFor, ProofCompressionOptions},
    MatrixClosingMode, PoseidonZkSetupConfig, QuinticExtension, SparkWhirParams,
    SyntheticR1csConfig,
};

fn config() -> PoseidonZkSetupConfig {
    let params = common::phase3_whir_params();
    PoseidonZkSetupConfig {
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
    }
}

macro_rules! batching_test {
    ($name:ident, $engine:ident, $protocol:ident, $vk:ident, $challenger:path) => {
        #[test]
        fn $name() {
            type E = spartan_whir::$engine<QuinticExtension>;
            type Protocol = spartan_whir::$protocol<QuinticExtension>;
            type Vk = spartan_whir::$vk<QuinticExtension>;
            type Compact = CompressedZkProofFor<E>;
            type Cached = CachedFixedOracleProofFor<E>;
            let fixture = generate_satisfiable_fixture(&SyntheticR1csConfig {
                target_log2_witness_poly: 3,
                num_constraints: 4,
                num_io: 1,
                a_terms_per_constraint: 2,
                b_terms_per_constraint: 2,
                seed: 27183,
            })
            .unwrap();
            let setup = |shape, mode| {
                spartan_whir::poseidon::setup_poseidon_zk_with_fresh_mask_batching::<E>(
                    shape,
                    config(),
                    mode,
                )
                .unwrap()
            };
            let (_, separate_vk) = setup(fixture.shape.clone(), FreshMaskBatching::Separate);
            let (pk, vk) = setup(fixture.shape.clone(), FreshMaskBatching::SameHeight);
            assert_ne!(
                vk.domain_separator().protocol_id,
                separate_vk.domain_separator().protocol_id
            );
            assert_eq!(
                vk.domain_separator().relation_digest,
                separate_vk.domain_separator().relation_digest
            );
            assert_eq!(
                vk.domain_separator().whir_params,
                separate_vk.domain_separator().whir_params
            );
            assert_eq!(
                vk.domain_separator().spark_whir_params,
                separate_vk.domain_separator().spark_whir_params
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
            assert!(
                proof.pcs_proof.base_case.fresh_mask_commitments.len()
                    < proof.pcs_proof.base_case.carried_mask_openings.len()
            );
            let mut verifier = $challenger();
            Protocol::verify(&vk, &instance, &proof, &mut verifier).unwrap();
            for _ in 0..16 {
                assert_eq!(
                    prover.sample_algebra_element::<QuinticExtension>(),
                    verifier.sample_algebra_element::<QuinticExtension>()
                );
            }
            assert!(Protocol::verify(&separate_vk, &instance, &proof, &mut $challenger()).is_err());
            let words =
                spartan_whir::encode_full_zk_spark_guest_words::<E>(&instance, &proof).unwrap();
            let (decoded_instance, decoded) =
                spartan_whir::decode_full_zk_spark_guest_words::<E>(&words).unwrap();
            Protocol::verify(&vk, &decoded_instance, &decoded, &mut $challenger()).unwrap();
            assert_eq!(
                bincode::serialize(&decoded).unwrap(),
                bincode::serialize(&proof).unwrap()
            );
            let ordinary = bincode::serialize(&proof).unwrap();
            for mutation in 0..6 {
                let mut altered: spartan_whir::ZkSpartanProofFor<E> =
                    bincode::deserialize(&ordinary).unwrap();
                let base = &mut altered.pcs_proof.base_case;
                match mutation {
                    0 => base.fresh_mask_commitments[0] = base.fresh_main_commitment.clone(),
                    1 => base.carried_mask_openings[0].rows[0][0] += QuinticExtension::ONE,
                    2 => base.fresh_mask_openings[0].rows[0][0] += QuinticExtension::ONE,
                    3 => {
                        base.fresh_mask_openings[0].rows[0].pop();
                    }
                    4 => {
                        base.carried_mask_openings[0].rows.pop();
                    }
                    5 => {
                        base.fresh_mask_commitments.pop();
                    }
                    _ => unreachable!(),
                }
                assert!(
                    Protocol::verify(&vk, &instance, &altered, &mut $challenger()).is_err(),
                    "mutation {mutation}"
                );
            }
            let mut restored: Vk = bincode::deserialize(&bincode::serialize(&vk).unwrap()).unwrap();
            assert!(Protocol::verify(&restored, &instance, &proof, &mut $challenger()).is_err());
            assert!(FixedOracleCacheFor::<E>::build(&restored).is_err());
            restored.authenticate().unwrap();
            Protocol::verify(&restored, &instance, &proof, &mut $challenger()).unwrap();
            let mut key_json = serde_json::to_value(&vk).unwrap();
            key_json["pcs_config"]["fresh_mask_batching"] = serde_json::json!("Separate");
            let mut altered_key: Vk = serde_json::from_value(key_json).unwrap();
            assert!(altered_key.authenticate().is_err());

            let mut changed_shape = fixture.shape.clone();
            changed_shape.a.entries[0].val += F::ONE;
            let (_, other_vk) = setup(changed_shape, FreshMaskBatching::SameHeight);
            assert_ne!(
                vk.domain_separator().relation_digest,
                other_vk.domain_separator().relation_digest
            );
            assert!(Protocol::verify(&other_vk, &instance, &proof, &mut $challenger()).is_err());
            let cache = FixedOracleCacheFor::<E>::build(&vk).unwrap();
            let other_cache = FixedOracleCacheFor::<E>::build(&other_vk).unwrap();
            let mut compact_prover = $challenger();
            let (compact_instance, compact) = Protocol::prove_compressed_with_rng(
                &pk,
                &fixture.public_inputs,
                &fixture.witness,
                &mut compact_prover,
                &mut StdRng::seed_from_u64(111),
                ProofCompressionOptions::recommended(),
            )
            .unwrap();
            let compact_bytes = compact.to_bytes().unwrap();
            let cached_bytes = Cached::from_compressed(&vk, compact)
                .unwrap()
                .to_bytes()
                .unwrap();
            assert!(cached_bytes.len() < compact_bytes.len());
            assert!(Compact::from_bytes(&cached_bytes).is_err());
            let mut cache_verifier = $challenger();
            cache
                .verify(
                    &vk,
                    &compact_instance,
                    Cached::from_bytes(&cached_bytes).unwrap(),
                    &mut cache_verifier,
                )
                .unwrap();
            for _ in 0..16 {
                assert_eq!(
                    compact_prover.sample_algebra_element::<QuinticExtension>(),
                    cache_verifier.sample_algebra_element::<QuinticExtension>()
                );
            }
            assert!(other_cache
                .verify(
                    &vk,
                    &compact_instance,
                    Cached::from_bytes(&cached_bytes).unwrap(),
                    &mut $challenger()
                )
                .is_err());
            assert!(cache
                .verify(
                    &other_vk,
                    &compact_instance,
                    Cached::from_bytes(&cached_bytes).unwrap(),
                    &mut $challenger()
                )
                .is_err());
            let mut bad_instance = compact_instance.clone();
            bad_instance.public_inputs[0] += F::ONE;
            assert!(cache
                .verify(
                    &vk,
                    &bad_instance,
                    Cached::from_bytes(&cached_bytes).unwrap(),
                    &mut $challenger()
                )
                .is_err());
            let mut wrong_identity = cached_bytes.clone();
            wrong_identity[4] ^= 1;
            assert!(cache
                .verify(
                    &vk,
                    &compact_instance,
                    Cached::from_bytes(&wrong_identity).unwrap(),
                    &mut $challenger()
                )
                .is_err());
            for end in [0, 4, 35, 36, cached_bytes.len() / 2, cached_bytes.len() - 1] {
                assert!(Cached::from_bytes(&cached_bytes[..end]).is_err());
            }
            let mut trailing = cached_bytes.clone();
            trailing.push(0);
            assert!(Cached::from_bytes(&trailing).is_err());
        }
    };
}

batching_test!(
    poseidon1_batched_proof_cache_and_relation_binding,
    Poseidon1Engine,
    Poseidon1ZkSpartanProtocol,
    Poseidon1ZkVerifyingKey,
    spartan_whir::poseidon1_challenger
);
batching_test!(
    poseidon2_batched_proof_cache_and_relation_binding,
    PoseidonEngine,
    PoseidonZkSpartanProtocol,
    PoseidonZkVerifyingKey,
    spartan_whir::poseidon_zk_challenger
);
