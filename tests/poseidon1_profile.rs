use p3_challenger::{CanObserve, CanSample};
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_koala_bear::{default_koalabear_poseidon1_16, KoalaBear};
use p3_symmetric::{CryptographicHasher, Permutation, PseudoCompressionFunction};
use spartan_whir::{
    generate_satisfiable_fixture, poseidon1_challenger, poseidon1_merkle_compress,
    poseidon1_merkle_hash, MatrixClosingMode, Plonky3WhirPcs, Poseidon1Challenger,
    Poseidon1QuarticEngine, SpartanProtocol, SpartanSnarkConfig, SyntheticR1csConfig,
    POSEIDON1_NO_ZK_PROTOCOL_ID,
};
use spartan_whir::{
    poseidon1_challenger as poseidon_zk_challenger, setup_poseidon1_zk as setup_poseidon_zk,
    Poseidon1ZkCommitment as PoseidonZkCommitment,
    Poseidon1ZkMatrixClosingProof as ZkMatrixClosingProof,
    Poseidon1ZkSpartanProof as ZkSpartanProof,
    Poseidon1ZkSpartanProtocol as PoseidonZkSpartanProtocol,
    Poseidon1ZkVerifyingKey as PoseidonZkVerifyingKey, PoseidonZkSetupConfig, QuarticBinExtension,
    R1csInstance,
};

fn verify_full_zk_rejects(
    vk: &PoseidonZkVerifyingKey<QuarticBinExtension>,
    instance: &R1csInstance<KoalaBear, PoseidonZkCommitment>,
    proof: &ZkSpartanProof<QuarticBinExtension>,
) {
    let mut verifier = poseidon_zk_challenger();
    assert!(PoseidonZkSpartanProtocol::<QuarticBinExtension>::verify(
        vk,
        instance,
        proof,
        &mut verifier,
    )
    .is_err());
}

#[test]
fn poseidon1_width_16_matches_the_pinned_permutation_vector() {
    let input = core::array::from_fn(|index| KoalaBear::from_usize(index));
    let output = default_koalabear_poseidon1_16().permute(input);

    let actual = output.map(|value| value.as_canonical_u32());
    assert_eq!(
        actual,
        [
            610_090_613,
            935_319_874,
            1_893_335_292,
            796_792_199,
            356_405_232,
            552_237_741,
            55_134_556,
            1_215_104_204,
            1_823_723_405,
            1_133_298_033,
            1_780_633_798,
            1_453_946_561,
            710_069_176,
            1_128_629_550,
            1_917_333_254,
            1_175_481_618,
        ]
    );
}

#[test]
fn poseidon1_challenger_preserves_plonky3_duplex_semantics() {
    let mut traced: Poseidon1Challenger = poseidon1_challenger();
    let mut direct = p3_challenger::DuplexChallenger::<KoalaBear, _, 16, 8>::new(
        default_koalabear_poseidon1_16(),
    );

    let observations = (0..23)
        .map(|index| KoalaBear::from_usize(index * index + 17))
        .collect::<Vec<_>>();
    traced.observe_slice(&observations);
    direct.observe_slice(&observations);

    let traced_samples = (0..12).map(|_| traced.sample()).collect::<Vec<KoalaBear>>();
    let direct_samples = (0..12).map(|_| direct.sample()).collect::<Vec<KoalaBear>>();
    assert_eq!(
        traced_samples
            .iter()
            .map(PrimeField32::as_canonical_u32)
            .collect::<Vec<_>>(),
        [
            454_253_751,
            1_906_113_756,
            377_596_994,
            792_299_923,
            2_053_520_120,
            1_687_138_493,
            1_335_122_479,
            1_913_328_900,
            336_914_164,
            269_792_076,
            241_800_815,
            850_435_158,
        ]
    );
    assert_eq!(traced_samples, direct_samples);
}

#[test]
fn poseidon1_leaf_and_node_hashes_use_width_16_rate_8() {
    let leaf = (0..24)
        .map(|index| KoalaBear::from_usize(index + 1))
        .collect::<Vec<_>>();
    let digest = poseidon1_merkle_hash().hash_iter(leaf);

    let left = core::array::from_fn(|index| KoalaBear::from_usize(index + 100));
    let right = core::array::from_fn(|index| KoalaBear::from_usize(index + 200));
    let compressed = poseidon1_merkle_compress().compress([left, right]);

    assert_eq!(
        digest.map(|value| value.as_canonical_u32()),
        [
            521_167_832,
            1_294_533_855,
            1_414_095_283,
            607_223_733,
            1_119_062_695,
            178_801_552,
            1_289_018_264,
            713_205_468,
        ]
    );
    assert_eq!(
        compressed.map(|value| value.as_canonical_u32()),
        [
            1_447_100_179,
            963_179_731,
            1_376_818_832,
            157_084_832,
            1_605_343_738,
            1_715_972_436,
            1_200_644_159,
            263_182_536,
        ]
    );
}

#[test]
fn poseidon1_no_zk_direct_sparse_roundtrip_uses_its_own_transcript_id() {
    let fixture = generate_satisfiable_fixture(&SyntheticR1csConfig {
        target_log2_witness_poly: 3,
        num_constraints: 4,
        num_io: 1,
        a_terms_per_constraint: 2,
        b_terms_per_constraint: 2,
        seed: 0x501D_01,
    })
    .expect("fixture generation succeeds");
    let config = SpartanSnarkConfig {
        matrix_closing: MatrixClosingMode::DirectSparse,
        security: common::phase3_security(),
        whir_params: common::phase3_whir_params(),
        spark_whir_params: None,
    };
    let (pk, vk) = SpartanProtocol::<Poseidon1QuarticEngine, Plonky3WhirPcs>::setup_with_config(
        &fixture.shape,
        &config,
    )
    .expect("Poseidon1 setup succeeds");
    assert_eq!(pk.domain_separator.protocol_id, POSEIDON1_NO_ZK_PROTOCOL_ID);

    let mut prover_challenger = poseidon1_challenger();
    let (instance, proof) =
        SpartanProtocol::<Poseidon1QuarticEngine, Plonky3WhirPcs>::prove_with_mode(
            &pk,
            &fixture.public_inputs,
            &fixture.witness,
            MatrixClosingMode::DirectSparse,
            &mut prover_challenger,
        )
        .expect("Poseidon1 proof succeeds");
    let mut verifier_challenger = poseidon1_challenger();
    SpartanProtocol::<Poseidon1QuarticEngine, Plonky3WhirPcs>::verify_with_mode(
        &vk,
        &instance,
        &proof,
        &mut verifier_challenger,
    )
    .expect("Poseidon1 proof verifies");
}

#[test]
fn poseidon1_full_zk_profile_uses_its_own_transcript_id() {
    let fixture = generate_satisfiable_fixture(&SyntheticR1csConfig {
        target_log2_witness_poly: 2,
        num_constraints: 2,
        num_io: 1,
        a_terms_per_constraint: 2,
        b_terms_per_constraint: 2,
        seed: 0x501D_F011,
    })
    .expect("fixture generation succeeds");
    let config = PoseidonZkSetupConfig {
        matrix_closing: MatrixClosingMode::DirectSparse,
        security: common::phase3_security(),
        whir_params: common::phase3_zk_whir_params(),
        spark_whir_params: None,
        ell_zk: spartan_whir::DEFAULT_ZK_ELL,
        mask_log_inv_rate: spartan_whir::DEFAULT_ZK_MASK_LOG_INV_RATE,
    };
    let (pk, vk) = setup_poseidon_zk::<QuarticBinExtension>(fixture.shape, config)
        .expect("Poseidon1 full-ZK setup succeeds");

    assert_eq!(
        pk.domain_separator().protocol_id,
        spartan_whir::POSEIDON1_FULL_ZK_PROTOCOL_ID
    );
    assert_eq!(
        vk.domain_separator().protocol_id,
        spartan_whir::POSEIDON1_FULL_ZK_PROTOCOL_ID
    );

    let mut prover = poseidon_zk_challenger();
    let (instance, proof) = PoseidonZkSpartanProtocol::<QuarticBinExtension>::prove(
        &pk,
        &fixture.public_inputs,
        &fixture.witness,
        &mut prover,
    )
    .expect("Poseidon1 full-ZK proof succeeds");
    let mut verifier = poseidon_zk_challenger();
    PoseidonZkSpartanProtocol::<QuarticBinExtension>::verify(&vk, &instance, &proof, &mut verifier)
        .expect("Poseidon1 full-ZK proof verifies");

    let mut changed_relation_json =
        serde_json::to_value(&vk).expect("Poseidon1 verifying key serializes");
    let matrix_value = &mut changed_relation_json["shape_canonical"]["a"]["entries"][0]["val"];
    let value = matrix_value
        .as_u64()
        .expect("matrix coefficient is an integer");
    *matrix_value = serde_json::json!(value + 1);
    let mut changed_relation_vk: PoseidonZkVerifyingKey<QuarticBinExtension> =
        serde_json::from_value(changed_relation_json).expect("changed relation decodes");
    assert!(matches!(
        changed_relation_vk.authenticate(),
        Err(spartan_whir::SpartanWhirError::InvalidConfig(_))
    ));

    let mut tampered_instance = instance.clone();
    tampered_instance.public_inputs[0] += KoalaBear::ONE;
    let mut verifier = poseidon_zk_challenger();
    assert!(PoseidonZkSpartanProtocol::<QuarticBinExtension>::verify(
        &vk,
        &tampered_instance,
        &proof,
        &mut verifier,
    )
    .is_err());

    let mut tampered_instance = instance.clone();
    let mut roots = tampered_instance.witness_commitment.roots().to_vec();
    roots[0][0] += KoalaBear::ONE;
    tampered_instance.witness_commitment = p3_symmetric::MerkleCap::new(roots);
    let mut verifier = poseidon_zk_challenger();
    assert!(PoseidonZkSpartanProtocol::<QuarticBinExtension>::verify(
        &vk,
        &tampered_instance,
        &proof,
        &mut verifier,
    )
    .is_err());

    let mut tampered = proof.clone();
    tampered.outer_sumcheck.mu_tilde += QuarticBinExtension::ONE;
    let mut verifier = poseidon_zk_challenger();
    assert!(PoseidonZkSpartanProtocol::<QuarticBinExtension>::verify(
        &vk,
        &instance,
        &tampered,
        &mut verifier,
    )
    .is_err());

    let mut tampered = proof.clone();
    tampered.outer_claims.0 += QuarticBinExtension::ONE;
    let mut verifier = poseidon_zk_challenger();
    assert!(PoseidonZkSpartanProtocol::<QuarticBinExtension>::verify(
        &vk,
        &instance,
        &tampered,
        &mut verifier,
    )
    .is_err());

    let mut tampered = proof.clone();
    let mut roots = tampered.application_mask_commitment.roots().to_vec();
    roots[0][0] += KoalaBear::ONE;
    tampered.application_mask_commitment = p3_symmetric::MerkleCap::new(roots);
    let mut verifier = poseidon_zk_challenger();
    assert!(PoseidonZkSpartanProtocol::<QuarticBinExtension>::verify(
        &vk,
        &instance,
        &tampered,
        &mut verifier,
    )
    .is_err());

    let mut tampered = proof.clone();
    let mut roots = tampered.inner_sumcheck_mask_commitment.roots().to_vec();
    roots[0][0] += KoalaBear::ONE;
    tampered.inner_sumcheck_mask_commitment = p3_symmetric::MerkleCap::new(roots);
    let mut verifier = poseidon_zk_challenger();
    assert!(PoseidonZkSpartanProtocol::<QuarticBinExtension>::verify(
        &vk,
        &instance,
        &tampered,
        &mut verifier,
    )
    .is_err());

    let mut tampered = proof.clone();
    tampered.inner_sumcheck.mu_tilde += QuarticBinExtension::ONE;
    let mut verifier = poseidon_zk_challenger();
    assert!(PoseidonZkSpartanProtocol::<QuarticBinExtension>::verify(
        &vk,
        &instance,
        &tampered,
        &mut verifier,
    )
    .is_err());

    let mut tampered = proof.clone();
    tampered.pcs_proof.sumchecks.pop();
    let mut verifier = poseidon_zk_challenger();
    assert!(PoseidonZkSpartanProtocol::<QuarticBinExtension>::verify(
        &vk,
        &instance,
        &tampered,
        &mut verifier,
    )
    .is_err());

    let mut wrong_profile_json =
        serde_json::to_value(&vk).expect("Poseidon1 verifying key serializes");
    wrong_profile_json["domain_separator"]["protocol_id"] =
        serde_json::to_value(spartan_whir::FULL_ZK_PROTOCOL_ID)
            .expect("default full-ZK protocol identifier serializes");
    let mut wrong_profile_vk: PoseidonZkVerifyingKey<QuarticBinExtension> =
        serde_json::from_value(wrong_profile_json)
            .expect("verifying key with a changed protocol identifier decodes");
    assert!(wrong_profile_vk.authenticate().is_err());

    let mut tampered = proof;
    tampered.pcs_proof.sumchecks[0].mu_tilde += QuarticBinExtension::ONE;
    let mut verifier = poseidon_zk_challenger();
    assert!(PoseidonZkSpartanProtocol::<QuarticBinExtension>::verify(
        &vk,
        &instance,
        &tampered,
        &mut verifier,
    )
    .is_err());
}

#[test]
fn poseidon1_full_zk_spark_binds_products_and_openings() {
    let fixture = generate_satisfiable_fixture(&SyntheticR1csConfig {
        target_log2_witness_poly: 4,
        num_constraints: 8,
        num_io: 1,
        a_terms_per_constraint: 3,
        b_terms_per_constraint: 3,
        seed: 0x501D_5A4A,
    })
    .expect("fixture generation succeeds");
    let config = PoseidonZkSetupConfig {
        matrix_closing: MatrixClosingMode::Spark,
        security: common::phase3_security(),
        whir_params: common::phase3_zk_whir_params(),
        spark_whir_params: None,
        ell_zk: spartan_whir::DEFAULT_ZK_ELL,
        mask_log_inv_rate: spartan_whir::DEFAULT_ZK_MASK_LOG_INV_RATE,
    };
    let (pk, vk) = setup_poseidon_zk::<QuarticBinExtension>(fixture.shape, config)
        .expect("Poseidon1 full-ZK SPARK setup succeeds");
    let mut prover = poseidon_zk_challenger();
    let (instance, proof) = PoseidonZkSpartanProtocol::<QuarticBinExtension>::prove(
        &pk,
        &fixture.public_inputs,
        &fixture.witness,
        &mut prover,
    )
    .expect("Poseidon1 full-ZK SPARK proof succeeds");
    let mut verifier = poseidon_zk_challenger();
    PoseidonZkSpartanProtocol::<QuarticBinExtension>::verify(&vk, &instance, &proof, &mut verifier)
        .expect("Poseidon1 full-ZK SPARK proof verifies");

    let mut tampered = proof.clone();
    let ZkMatrixClosingProof::Spark(closing) = &mut tampered.matrix_closing else {
        panic!("expected SPARK closing proof");
    };
    closing
        .spark_products
        .proof_ops
        .layers
        .iter_mut()
        .find_map(|layer| layer.rounds.first_mut())
        .expect("SPARK product proof has a sumcheck round")
        .0[0] += QuarticBinExtension::ONE;
    verify_full_zk_rejects(&vk, &instance, &tampered);

    let mut tampered = proof.clone();
    let ZkMatrixClosingProof::Spark(closing) = &mut tampered.matrix_closing else {
        panic!("expected SPARK closing proof");
    };
    closing.spark_products.matrix_evals[0] += QuarticBinExtension::ONE;
    verify_full_zk_rejects(&vk, &instance, &tampered);

    let mut tampered = proof.clone();
    let ZkMatrixClosingProof::Spark(closing) = &mut tampered.matrix_closing else {
        panic!("expected SPARK closing proof");
    };
    let mut roots = closing
        .spark_fixed_openings
        .value_commitment
        .roots()
        .to_vec();
    roots[0][0] += KoalaBear::ONE;
    closing.spark_fixed_openings.value_commitment = p3_symmetric::MerkleCap::new(roots);
    verify_full_zk_rejects(&vk, &instance, &tampered);

    let mut tampered = proof.clone();
    let ZkMatrixClosingProof::Spark(closing) = &mut tampered.matrix_closing else {
        panic!("expected SPARK closing proof");
    };
    let mut roots = closing.spark_read_openings.groups[0]
        .commitment
        .roots()
        .to_vec();
    roots[0][0] += KoalaBear::ONE;
    closing.spark_read_openings.groups[0].commitment = p3_symmetric::MerkleCap::new(roots);
    verify_full_zk_rejects(&vk, &instance, &tampered);

    let mut tampered = proof.clone();
    let ZkMatrixClosingProof::Spark(closing) = &mut tampered.matrix_closing else {
        panic!("expected SPARK closing proof");
    };
    closing.spark_fixed_openings.evals.val_a_low += QuarticBinExtension::ONE;
    verify_full_zk_rejects(&vk, &instance, &tampered);

    let mut tampered = proof.clone();
    let ZkMatrixClosingProof::Spark(closing) = &mut tampered.matrix_closing else {
        panic!("expected SPARK closing proof");
    };
    *closing.spark_read_openings.groups[0].evals[0]
        .first_mut()
        .expect("read opening has an evaluation") += QuarticBinExtension::ONE;
    verify_full_zk_rejects(&vk, &instance, &tampered);

    let mut tampered = proof.clone();
    let ZkMatrixClosingProof::Spark(closing) = &mut tampered.matrix_closing else {
        panic!("expected SPARK closing proof");
    };
    *closing
        .spark_fixed_openings
        .value_proof
        .initial_ood_answers
        .first_mut()
        .expect("fixed opening has an OOD answer") += QuarticBinExtension::ONE;
    verify_full_zk_rejects(&vk, &instance, &tampered);

    let mut tampered = proof;
    let ZkMatrixClosingProof::Spark(closing) = &mut tampered.matrix_closing else {
        panic!("expected SPARK closing proof");
    };
    *closing.spark_read_openings.groups[0]
        .proof
        .initial_ood_answers
        .first_mut()
        .expect("read opening has an OOD answer") += QuarticBinExtension::ONE;
    verify_full_zk_rejects(&vk, &instance, &tampered);
}
mod common;
