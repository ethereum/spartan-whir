mod common;

use spartan_whir::{
    generate_satisfiable_fixture, setup_poseidon1_zk, setup_poseidon_zk, MatrixClosingMode,
    PoseidonZkSetupConfig, QuarticBinExtension, SyntheticR1csConfig, FULL_ZK_PROTOCOL_ID,
    POSEIDON1_FULL_ZK_PROTOCOL_ID,
};

#[test]
fn poseidon1_and_poseidon2_full_zk_profiles_coexist() {
    let fixture = generate_satisfiable_fixture(&SyntheticR1csConfig {
        target_log2_witness_poly: 2,
        num_constraints: 2,
        num_io: 1,
        a_terms_per_constraint: 2,
        b_terms_per_constraint: 2,
        seed: 0xADD1_71E,
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

    let (_, poseidon2_vk) =
        setup_poseidon_zk::<QuarticBinExtension>(fixture.shape.clone(), config.clone())
            .expect("Poseidon2 full-ZK setup succeeds");
    let (_, poseidon1_vk) = setup_poseidon1_zk::<QuarticBinExtension>(fixture.shape, config)
        .expect("Poseidon1 full-ZK setup succeeds");

    assert_eq!(
        poseidon2_vk.domain_separator().protocol_id,
        FULL_ZK_PROTOCOL_ID
    );
    assert_eq!(
        poseidon1_vk.domain_separator().protocol_id,
        POSEIDON1_FULL_ZK_PROTOCOL_ID
    );
    assert_ne!(
        poseidon2_vk.domain_separator().to_bytes(),
        poseidon1_vk.domain_separator().to_bytes()
    );
}
