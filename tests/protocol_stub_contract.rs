mod common;

use p3_field::PrimeCharacteristicRing;

use spartan_whir::{
    engine::F, preprocess_spark_tables, MatrixClosingMode, Plonky3WhirPcs,
    PoseidonQuarticEngine as PoseidonEngine, QuarticBinExtension as EF, R1csShape, SparkLayoutKind,
    SpartanProtocol, SpartanSnarkConfig, SpartanWhirError,
};

fn regular_shape_two_constraints() -> R1csShape<F> {
    let one = F::ONE;
    R1csShape {
        num_cons: 2,
        num_vars: 2,
        num_io: 1,
        a: spartan_whir::SparseMatrix {
            num_rows: 2,
            num_cols: 4,
            entries: vec![
                spartan_whir::SparseMatEntry {
                    row: 0,
                    col: 0,
                    val: one,
                },
                spartan_whir::SparseMatEntry {
                    row: 1,
                    col: 0,
                    val: one,
                },
            ],
        },
        b: spartan_whir::SparseMatrix {
            num_rows: 2,
            num_cols: 4,
            entries: vec![
                spartan_whir::SparseMatEntry {
                    row: 0,
                    col: 2,
                    val: one,
                },
                spartan_whir::SparseMatEntry {
                    row: 1,
                    col: 2,
                    val: one,
                },
            ],
        },
        c: spartan_whir::SparseMatrix {
            num_rows: 2,
            num_cols: 4,
            entries: vec![
                spartan_whir::SparseMatEntry {
                    row: 0,
                    col: 3,
                    val: one,
                },
                spartan_whir::SparseMatEntry {
                    row: 1,
                    col: 3,
                    val: one,
                },
            ],
        },
    }
}

fn shared_union_shape_two_constraints() -> R1csShape<F> {
    let a_entries = vec![
        spartan_whir::SparseMatEntry {
            row: 0,
            col: 1,
            val: F::ONE,
        },
        spartan_whir::SparseMatEntry {
            row: 1,
            col: 1,
            val: F::ONE,
        },
    ];
    let b_entries = vec![
        spartan_whir::SparseMatEntry {
            row: 0,
            col: 1,
            val: F::from_u32(2),
        },
        spartan_whir::SparseMatEntry {
            row: 1,
            col: 1,
            val: F::from_u32(2),
        },
    ];
    let c_entries = vec![
        spartan_whir::SparseMatEntry {
            row: 0,
            col: 1,
            val: F::from_u32(3),
        },
        spartan_whir::SparseMatEntry {
            row: 1,
            col: 1,
            val: F::from_u32(3),
        },
    ];
    R1csShape {
        num_cons: 2,
        num_vars: 2,
        num_io: 1,
        a: spartan_whir::SparseMatrix {
            num_rows: 2,
            num_cols: 4,
            entries: a_entries,
        },
        b: spartan_whir::SparseMatrix {
            num_rows: 2,
            num_cols: 4,
            entries: b_entries,
        },
        c: spartan_whir::SparseMatrix {
            num_rows: 2,
            num_cols: 4,
            entries: c_entries,
        },
    }
}

fn mid_size_mixed_shape() -> R1csShape<F> {
    let num_cons = 16;
    let num_vars = 8;
    let num_cols = num_vars + 1 + 1;
    let mut a_entries = Vec::new();
    let mut b_entries = Vec::new();
    let mut c_entries = Vec::new();

    for row in 0..num_cons {
        a_entries.push(spartan_whir::SparseMatEntry {
            row,
            col: 0,
            val: F::ONE,
        });
        b_entries.push(spartan_whir::SparseMatEntry {
            row,
            col: num_vars,
            val: F::ONE,
        });
        c_entries.push(spartan_whir::SparseMatEntry {
            row,
            col: num_vars + 1,
            val: F::ONE,
        });
        if row % 2 == 0 {
            a_entries.push(spartan_whir::SparseMatEntry {
                row,
                col: 1,
                val: F::from_u32(3),
            });
        }
        if row % 3 == 0 {
            b_entries.push(spartan_whir::SparseMatEntry {
                row,
                col: 2,
                val: F::from_u32(5),
            });
        }
        if row % 4 == 0 {
            c_entries.push(spartan_whir::SparseMatEntry {
                row,
                col: 3,
                val: F::from_u32(7),
            });
        }
    }

    R1csShape {
        num_cons,
        num_vars,
        num_io: 1,
        a: spartan_whir::SparseMatrix {
            num_rows: num_cons,
            num_cols,
            entries: a_entries,
        },
        b: spartan_whir::SparseMatrix {
            num_rows: num_cons,
            num_cols,
            entries: b_entries,
        },
        c: spartan_whir::SparseMatrix {
            num_rows: num_cons,
            num_cols,
            entries: c_entries,
        },
    }
}

fn setup_keys(
    shape: &R1csShape<F>,
) -> (
    spartan_whir::ProvingKey<PoseidonEngine, Plonky3WhirPcs>,
    spartan_whir::VerifyingKey<PoseidonEngine, Plonky3WhirPcs>,
) {
    setup_keys_with_mode(shape, MatrixClosingMode::Spark)
}

fn setup_keys_with_mode(
    shape: &R1csShape<F>,
    matrix_closing: MatrixClosingMode,
) -> (
    spartan_whir::ProvingKey<PoseidonEngine, Plonky3WhirPcs>,
    spartan_whir::VerifyingKey<PoseidonEngine, Plonky3WhirPcs>,
) {
    SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::setup_with_config(
        shape,
        &SpartanSnarkConfig {
            matrix_closing,
            security: common::phase3_security(),
            whir_params: common::phase3_whir_params(),
            spark_whir_params: None,
        },
    )
    .expect("setup succeeds")
}

fn mid_size_witness() -> spartan_whir::R1csWitness<F> {
    let mut w = vec![F::ZERO; 8];
    w[0] = F::from_u32(7);
    spartan_whir::R1csWitness { w }
}

#[test]
fn protocol_roundtrip_regular_shape() {
    let shape = regular_shape_two_constraints();
    let (pk, vk) = setup_keys_with_mode(&shape, MatrixClosingMode::DirectSparse);
    let mut prover_challenger = spartan_whir::poseidon_challenger();
    let mut verifier_challenger = spartan_whir::poseidon_challenger();

    let witness = spartan_whir::R1csWitness {
        w: vec![F::from_u32(7), F::ZERO],
    };
    let public_inputs = common::koala_public_inputs(7);

    let (instance, proof) = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::prove(
        &pk,
        &public_inputs,
        &witness,
        &mut prover_challenger,
    )
    .expect("prove succeeds");

    let verified = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::verify(
        &vk,
        &instance,
        &proof,
        &mut verifier_challenger,
    );
    assert_eq!(verified, Ok(()));
}

#[test]
fn protocol_direct_and_spark_roundtrip_same_fixture_with_config_modes() {
    let shape = regular_shape_two_constraints();
    let witness = spartan_whir::R1csWitness {
        w: vec![F::from_u32(7), F::ZERO],
    };
    let public_inputs = common::koala_public_inputs(7);

    let direct_config = SpartanSnarkConfig {
        matrix_closing: MatrixClosingMode::DirectSparse,
        security: common::phase3_security(),
        whir_params: common::phase3_whir_params(),
        spark_whir_params: None,
    };
    let (direct_pk, direct_vk) =
        SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::setup_with_config(
            &shape,
            &direct_config,
        )
        .expect("direct setup succeeds");
    assert_eq!(direct_pk.spark_fixed_commitments, None);
    assert_eq!(direct_vk.spark_fixed_commitments(), None);

    let mut direct_prover_challenger = spartan_whir::poseidon_challenger();
    let mut direct_verifier_challenger = spartan_whir::poseidon_challenger();
    let (direct_instance, direct_proof) =
        SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::prove_with_mode(
            &direct_pk,
            &public_inputs,
            &witness,
            direct_config.matrix_closing,
            &mut direct_prover_challenger,
        )
        .expect("direct prove succeeds");
    let direct_verified = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::verify_with_mode(
        &direct_vk,
        &direct_instance,
        &direct_proof,
        &mut direct_verifier_challenger,
    );
    assert_eq!(direct_verified, Ok(()));

    let spark_config = SpartanSnarkConfig {
        matrix_closing: MatrixClosingMode::Spark,
        security: common::phase3_security(),
        whir_params: common::phase3_whir_params(),
        spark_whir_params: None,
    };
    let (spark_pk, spark_vk) =
        SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::setup_with_config(&shape, &spark_config)
            .expect("spark setup succeeds");
    assert!(spark_pk.spark_fixed_commitments.is_some());
    assert!(spark_vk.spark_fixed_commitments().is_some());

    let mut spark_prover_challenger = spartan_whir::poseidon_challenger();
    let mut spark_verifier_challenger = spartan_whir::poseidon_challenger();
    let (spark_instance, spark_proof) =
        SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::prove_with_mode(
            &spark_pk,
            &public_inputs,
            &witness,
            spark_config.matrix_closing,
            &mut spark_prover_challenger,
        )
        .expect("spark prove succeeds");
    let spark_verified = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::verify_with_mode(
        &spark_vk,
        &spark_instance,
        &spark_proof,
        &mut spark_verifier_challenger,
    );
    assert_eq!(spark_verified, Ok(()));

    assert_eq!(direct_proof.kind(), MatrixClosingMode::DirectSparse);
    assert_eq!(spark_proof.kind(), MatrixClosingMode::Spark);

    let mut spark_proof_direct_vk_challenger = spartan_whir::poseidon_challenger();
    assert_eq!(
        SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::verify_with_mode(
            &direct_vk,
            &spark_instance,
            &spark_proof,
            &mut spark_proof_direct_vk_challenger,
        ),
        Err(SpartanWhirError::ProofKindMismatch)
    );

    let mut direct_proof_spark_vk_challenger = spartan_whir::poseidon_challenger();
    assert_eq!(
        SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::verify_with_mode(
            &spark_vk,
            &direct_instance,
            &direct_proof,
            &mut direct_proof_spark_vk_challenger,
        ),
        Err(SpartanWhirError::ProofKindMismatch)
    );
}

#[test]
fn protocol_spark_roundtrip_regular_shape() {
    let shape = regular_shape_two_constraints();
    let (pk, vk) = setup_keys(&shape);
    let mut prover_challenger = spartan_whir::poseidon_challenger();
    let mut verifier_challenger = spartan_whir::poseidon_challenger();

    let witness = spartan_whir::R1csWitness {
        w: vec![F::from_u32(7), F::ZERO],
    };
    let public_inputs = common::koala_public_inputs(7);

    let (instance, proof) = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::prove_spark(
        &pk,
        &public_inputs,
        &witness,
        &mut prover_challenger,
    )
    .expect("spark prove succeeds");
    assert!(pk
        .spark_fixed_commitments
        .as_ref()
        .expect("SPARK fixed commitments exist")
        .audit
        .is_none());
    assert!(proof.spark_fixed_openings.audit_commitment.is_none());
    assert!(proof.spark_fixed_openings.audit_proof.is_none());

    let verified = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::verify_spark(
        &vk,
        &instance,
        &proof,
        &mut verifier_challenger,
    );
    assert_eq!(verified, Ok(()));
}

#[test]
fn protocol_spark_roundtrip_shared_union_shape() {
    let shape = shared_union_shape_two_constraints();
    let tables = preprocess_spark_tables(&shape.pad_regular().expect("shape pads"))
        .expect("spark preprocess succeeds");
    assert_eq!(tables.layout, SparkLayoutKind::SharedUnion);

    let (pk, vk) = setup_keys(&shape);
    let mut prover_challenger = spartan_whir::poseidon_challenger();
    let mut verifier_challenger = spartan_whir::poseidon_challenger();

    let witness = spartan_whir::R1csWitness {
        w: vec![F::from_u32(7), F::ZERO],
    };
    let public_inputs = common::koala_public_inputs(7);

    let (instance, proof) = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::prove_spark(
        &pk,
        &public_inputs,
        &witness,
        &mut prover_challenger,
    )
    .expect("spark prove succeeds");
    assert!(pk
        .spark_fixed_commitments
        .as_ref()
        .expect("SPARK fixed commitments exist")
        .audit
        .is_some());
    assert!(proof.spark_fixed_openings.audit_commitment.is_some());
    assert!(proof.spark_fixed_openings.audit_proof.is_some());

    let verified = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::verify_spark(
        &vk,
        &instance,
        &proof,
        &mut verifier_challenger,
    );
    assert_eq!(verified, Ok(()));
}

#[test]
fn protocol_roundtrip_irregular_shape_autopad() {
    let shape = common::koala_shape_single_constraint(2);
    let (pk, vk) = setup_keys_with_mode(&shape, MatrixClosingMode::DirectSparse);
    let mut prover_challenger = spartan_whir::poseidon_challenger();
    let mut verifier_challenger = spartan_whir::poseidon_challenger();

    let witness = common::koala_witness(9);
    let public_inputs = common::koala_public_inputs(9);
    let (instance, proof) = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::prove(
        &pk,
        &public_inputs,
        &witness,
        &mut prover_challenger,
    )
    .expect("prove succeeds");

    let verified = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::verify(
        &vk,
        &instance,
        &proof,
        &mut verifier_challenger,
    );
    assert_eq!(verified, Ok(()));
}

#[test]
fn protocol_spark_roundtrip_mid_size_mixed_shape() {
    let shape = mid_size_mixed_shape();
    let (pk, vk) = setup_keys(&shape);
    let mut prover_challenger = spartan_whir::poseidon_challenger();
    let mut verifier_challenger = spartan_whir::poseidon_challenger();

    let public_inputs = common::koala_public_inputs(7);
    let (instance, proof) = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::prove_spark(
        &pk,
        &public_inputs,
        &mid_size_witness(),
        &mut prover_challenger,
    )
    .expect("spark prove succeeds");

    let verified = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::verify_spark(
        &vk,
        &instance,
        &proof,
        &mut verifier_challenger,
    );
    assert_eq!(verified, Ok(()));
}

#[test]
fn protocol_spark_mid_size_tampered_memory_product_fails() {
    let shape = mid_size_mixed_shape();
    let (pk, vk) = setup_keys(&shape);
    let mut prover_challenger = spartan_whir::poseidon_challenger();
    let mut verifier_challenger = spartan_whir::poseidon_challenger();

    let public_inputs = common::koala_public_inputs(7);
    let (instance, mut proof) = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::prove_spark(
        &pk,
        &public_inputs,
        &mid_size_witness(),
        &mut prover_challenger,
    )
    .expect("spark prove succeeds");

    proof.spark_products.proof_ops.layers[0].product_left_evals[0] += EF::ONE;
    let verified = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::verify_spark(
        &vk,
        &instance,
        &proof,
        &mut verifier_challenger,
    );
    assert_eq!(verified, Err(SpartanWhirError::SumcheckFailed));
}

#[test]
fn protocol_spark_tampered_matrix_eval_fails() {
    let shape = regular_shape_two_constraints();
    let (pk, vk) = setup_keys(&shape);
    let mut prover_challenger = spartan_whir::poseidon_challenger();
    let mut verifier_challenger = spartan_whir::poseidon_challenger();

    let witness = spartan_whir::R1csWitness {
        w: vec![F::from_u32(7), F::ZERO],
    };
    let public_inputs = common::koala_public_inputs(7);

    let (instance, mut proof) = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::prove_spark(
        &pk,
        &public_inputs,
        &witness,
        &mut prover_challenger,
    )
    .expect("spark prove succeeds");

    proof.spark_products.matrix_evals[0] += EF::ONE;
    let verified = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::verify_spark(
        &vk,
        &instance,
        &proof,
        &mut verifier_challenger,
    );
    assert_eq!(verified, Err(SpartanWhirError::SumcheckFailed));
}

#[test]
fn protocol_spark_tampered_memory_product_fails() {
    let shape = regular_shape_two_constraints();
    let (pk, vk) = setup_keys(&shape);
    let mut prover_challenger = spartan_whir::poseidon_challenger();
    let mut verifier_challenger = spartan_whir::poseidon_challenger();

    let witness = spartan_whir::R1csWitness {
        w: vec![F::from_u32(7), F::ZERO],
    };
    let public_inputs = common::koala_public_inputs(7);

    let (instance, mut proof) = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::prove_spark(
        &pk,
        &public_inputs,
        &witness,
        &mut prover_challenger,
    )
    .expect("spark prove succeeds");

    proof.spark_products.proof_ops.layers[1].rounds[0].0[0] += EF::ONE;
    let verified = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::verify_spark(
        &vk,
        &instance,
        &proof,
        &mut verifier_challenger,
    );
    assert_eq!(verified, Err(SpartanWhirError::SumcheckFailed));
}

#[test]
fn protocol_spark_tampered_read_opening_commitment_fails() {
    let shape = regular_shape_two_constraints();
    let (pk, vk) = setup_keys(&shape);
    let mut prover_challenger = spartan_whir::poseidon_challenger();
    let mut verifier_challenger = spartan_whir::poseidon_challenger();

    let witness = spartan_whir::R1csWitness {
        w: vec![F::from_u32(7), F::ZERO],
    };
    let public_inputs = common::koala_public_inputs(7);

    let (instance, mut proof) = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::prove_spark(
        &pk,
        &public_inputs,
        &witness,
        &mut prover_challenger,
    )
    .expect("spark prove succeeds");

    let mut roots = proof.spark_read_openings.groups[0]
        .commitment
        .clone()
        .into_roots();
    roots[0][0] += F::ONE;
    proof.spark_read_openings.groups[0].commitment = roots.into();
    let verified = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::verify_spark(
        &vk,
        &instance,
        &proof,
        &mut verifier_challenger,
    );
    assert_eq!(verified, Err(SpartanWhirError::TranscriptMismatch));
}

#[test]
fn protocol_spark_tampered_read_opening_proof_fails() {
    let shape = regular_shape_two_constraints();
    let (pk, vk) = setup_keys(&shape);
    let mut prover_challenger = spartan_whir::poseidon_challenger();
    let mut verifier_challenger = spartan_whir::poseidon_challenger();

    let witness = spartan_whir::R1csWitness {
        w: vec![F::from_u32(7), F::ZERO],
    };
    let public_inputs = common::koala_public_inputs(7);

    let (instance, mut proof) = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::prove_spark(
        &pk,
        &public_inputs,
        &witness,
        &mut prover_challenger,
    )
    .expect("spark prove succeeds");

    *proof.spark_read_openings.groups[0]
        .proof
        .initial_ood_answers
        .first_mut()
        .expect("read-table WHIR proof has an initial OOD answer") += EF::ONE;
    let verified = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::verify_spark(
        &vk,
        &instance,
        &proof,
        &mut verifier_challenger,
    );
    assert_eq!(verified, Err(SpartanWhirError::WhirVerifyFailed));
}

#[test]
fn protocol_spark_tampered_read_opening_eval_fails() {
    let shape = regular_shape_two_constraints();
    let (pk, vk) = setup_keys(&shape);
    let mut prover_challenger = spartan_whir::poseidon_challenger();
    let mut verifier_challenger = spartan_whir::poseidon_challenger();

    let witness = spartan_whir::R1csWitness {
        w: vec![F::from_u32(7), F::ZERO],
    };
    let public_inputs = common::koala_public_inputs(7);

    let (instance, mut proof) = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::prove_spark(
        &pk,
        &public_inputs,
        &witness,
        &mut prover_challenger,
    )
    .expect("spark prove succeeds");

    proof.spark_read_openings.groups[0].evals[0][0] += EF::ONE;
    let verified = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::verify_spark(
        &vk,
        &instance,
        &proof,
        &mut verifier_challenger,
    );
    assert_eq!(verified, Err(SpartanWhirError::WhirVerifyFailed));
}

#[test]
fn protocol_spark_tampered_ecol_read_opening_eval_fails() {
    let shape = regular_shape_two_constraints();
    let (pk, vk) = setup_keys(&shape);
    let mut prover_challenger = spartan_whir::poseidon_challenger();
    let mut verifier_challenger = spartan_whir::poseidon_challenger();

    let witness = spartan_whir::R1csWitness {
        w: vec![F::from_u32(7), F::ZERO],
    };
    let public_inputs = common::koala_public_inputs(7);

    let (instance, mut proof) = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::prove_spark(
        &pk,
        &public_inputs,
        &witness,
        &mut prover_challenger,
    )
    .expect("spark prove succeeds");

    proof.spark_read_openings.groups[0].evals[3][0] += EF::ONE;
    let verified = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::verify_spark(
        &vk,
        &instance,
        &proof,
        &mut verifier_challenger,
    );
    assert_eq!(verified, Err(SpartanWhirError::WhirVerifyFailed));
}

#[test]
fn protocol_spark_tampered_fixed_opening_commitment_fails() {
    let shape = regular_shape_two_constraints();
    let (pk, vk) = setup_keys(&shape);
    let mut prover_challenger = spartan_whir::poseidon_challenger();
    let mut verifier_challenger = spartan_whir::poseidon_challenger();

    let witness = spartan_whir::R1csWitness {
        w: vec![F::from_u32(7), F::ZERO],
    };
    let public_inputs = common::koala_public_inputs(7);

    let (instance, mut proof) = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::prove_spark(
        &pk,
        &public_inputs,
        &witness,
        &mut prover_challenger,
    )
    .expect("spark prove succeeds");

    let mut roots = proof
        .spark_fixed_openings
        .value_commitment
        .clone()
        .into_roots();
    roots[0][0] += F::ONE;
    proof.spark_fixed_openings.value_commitment = roots.into();
    let verified = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::verify_spark(
        &vk,
        &instance,
        &proof,
        &mut verifier_challenger,
    );
    assert_eq!(verified, Err(SpartanWhirError::CommitmentMismatch));
}

#[test]
fn protocol_spark_tampered_fixed_opening_eval_fails() {
    let shape = regular_shape_two_constraints();
    let (pk, vk) = setup_keys(&shape);
    let mut prover_challenger = spartan_whir::poseidon_challenger();
    let mut verifier_challenger = spartan_whir::poseidon_challenger();

    let witness = spartan_whir::R1csWitness {
        w: vec![F::from_u32(7), F::ZERO],
    };
    let public_inputs = common::koala_public_inputs(7);

    let (instance, mut proof) = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::prove_spark(
        &pk,
        &public_inputs,
        &witness,
        &mut prover_challenger,
    )
    .expect("spark prove succeeds");

    proof.spark_fixed_openings.evals.val_a_low += EF::ONE;
    let verified = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::verify_spark(
        &vk,
        &instance,
        &proof,
        &mut verifier_challenger,
    );
    assert_eq!(verified, Err(SpartanWhirError::WhirVerifyFailed));
}

#[test]
fn protocol_spark_tampered_embedded_audit_opening_eval_fails() {
    let shape = regular_shape_two_constraints();
    let (pk, vk) = setup_keys(&shape);
    let mut prover_challenger = spartan_whir::poseidon_challenger();
    let mut verifier_challenger = spartan_whir::poseidon_challenger();
    let witness = spartan_whir::R1csWitness {
        w: vec![F::from_u32(7), F::ZERO],
    };
    let (instance, mut proof) = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::prove_spark(
        &pk,
        &common::koala_public_inputs(7),
        &witness,
        &mut prover_challenger,
    )
    .expect("spark prove succeeds");

    assert!(proof.spark_fixed_openings.audit_proof.is_none());
    proof.spark_fixed_openings.evals.row_audit_ts += EF::ONE;
    assert_eq!(
        SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::verify_spark(
            &vk,
            &instance,
            &proof,
            &mut verifier_challenger,
        ),
        Err(SpartanWhirError::WhirVerifyFailed)
    );
}

#[test]
fn protocol_spark_tampered_separate_audit_opening_eval_fails() {
    let shape = shared_union_shape_two_constraints();
    let (pk, vk) = setup_keys(&shape);
    let mut prover_challenger = spartan_whir::poseidon_challenger();
    let mut verifier_challenger = spartan_whir::poseidon_challenger();
    let witness = spartan_whir::R1csWitness {
        w: vec![F::from_u32(7), F::ZERO],
    };
    let (instance, mut proof) = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::prove_spark(
        &pk,
        &common::koala_public_inputs(7),
        &witness,
        &mut prover_challenger,
    )
    .expect("spark prove succeeds");

    assert!(proof.spark_fixed_openings.audit_proof.is_some());
    proof.spark_fixed_openings.evals.row_audit_ts += EF::ONE;
    assert_eq!(
        SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::verify_spark(
            &vk,
            &instance,
            &proof,
            &mut verifier_challenger,
        ),
        Err(SpartanWhirError::WhirVerifyFailed)
    );
}

#[test]
fn protocol_spark_tampered_shared_union_val_a_opening_fails() {
    let shape = shared_union_shape_two_constraints();
    let (pk, vk) = setup_keys(&shape);
    let mut prover_challenger = spartan_whir::poseidon_challenger();
    let mut verifier_challenger = spartan_whir::poseidon_challenger();

    let witness = spartan_whir::R1csWitness {
        w: vec![F::from_u32(7), F::ZERO],
    };
    let public_inputs = common::koala_public_inputs(7);

    let (instance, mut proof) = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::prove_spark(
        &pk,
        &public_inputs,
        &witness,
        &mut prover_challenger,
    )
    .expect("spark prove succeeds");

    proof.spark_fixed_openings.evals.val_a_low += EF::ONE;
    let verified = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::verify_spark(
        &vk,
        &instance,
        &proof,
        &mut verifier_challenger,
    );
    assert_eq!(verified, Err(SpartanWhirError::WhirVerifyFailed));
}

#[test]
fn protocol_spark_swapped_shared_union_val_ab_openings_fail() {
    let shape = shared_union_shape_two_constraints();
    let (pk, vk) = setup_keys(&shape);
    let mut prover_challenger = spartan_whir::poseidon_challenger();
    let mut verifier_challenger = spartan_whir::poseidon_challenger();

    let witness = spartan_whir::R1csWitness {
        w: vec![F::from_u32(7), F::ZERO],
    };
    let public_inputs = common::koala_public_inputs(7);

    let (instance, mut proof) = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::prove_spark(
        &pk,
        &public_inputs,
        &witness,
        &mut prover_challenger,
    )
    .expect("spark prove succeeds");

    let val_a = proof.spark_fixed_openings.evals.val_a_low;
    proof.spark_fixed_openings.evals.val_a_low = proof.spark_fixed_openings.evals.val_b_low;
    proof.spark_fixed_openings.evals.val_b_low = val_a;
    let verified = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::verify_spark(
        &vk,
        &instance,
        &proof,
        &mut verifier_challenger,
    );
    assert_eq!(verified, Err(SpartanWhirError::WhirVerifyFailed));
}

#[test]
fn protocol_spark_verifying_key_fixed_commitment_mismatch_fails() {
    let shape = regular_shape_two_constraints();
    let (_, vk) = setup_keys(&shape);

    let mut encoded = serde_json::to_value(&vk).expect("verifying key serializes");
    let first_root = &mut encoded["spark_fixed_commitments"]["value"]["cap"][0][0];
    let value = first_root.as_u64().expect("commitment limb is an integer");
    *first_root = serde_json::json!(value + 1);
    let mut forged: spartan_whir::VerifyingKey<PoseidonEngine, Plonky3WhirPcs> =
        serde_json::from_value(encoded).expect("forged key deserializes");
    assert_eq!(
        forged.authenticate_spark_fixed_commitments(),
        Err(SpartanWhirError::CommitmentMismatch)
    );
}

#[test]
fn protocol_restored_verifying_key_rejects_relation_change_with_stale_digest() {
    let shape = regular_shape_two_constraints();
    let (_, vk) = setup_keys_with_mode(&shape, MatrixClosingMode::DirectSparse);

    let mut encoded = serde_json::to_value(&vk).expect("verifying key serializes");
    let matrix_value = &mut encoded["shape_canonical"]["a"]["entries"][0]["val"];
    let value = matrix_value
        .as_u64()
        .expect("matrix coefficient is an integer");
    *matrix_value = serde_json::json!(value + 1);
    let mut forged: spartan_whir::VerifyingKey<PoseidonEngine, Plonky3WhirPcs> =
        serde_json::from_value(encoded).expect("changed relation decodes");
    assert!(matches!(
        forged.authenticate(),
        Err(SpartanWhirError::InvalidConfig(_))
    ));
}

#[test]
fn protocol_wrong_public_input_fails() {
    let shape = common::koala_shape_single_constraint(2);
    let (pk, vk) = setup_keys_with_mode(&shape, MatrixClosingMode::DirectSparse);
    let mut prover_challenger = spartan_whir::poseidon_challenger();
    let mut verifier_challenger = spartan_whir::poseidon_challenger();

    let (instance, proof) = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::prove(
        &pk,
        &common::koala_public_inputs(5),
        &common::koala_witness(5),
        &mut prover_challenger,
    )
    .expect("prove succeeds");

    let mut wrong_instance = instance;
    wrong_instance.public_inputs[0] = F::from_u32(6);
    let verified = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::verify(
        &vk,
        &wrong_instance,
        &proof,
        &mut verifier_challenger,
    );
    assert!(verified.is_err());
}

#[test]
fn protocol_tampered_commitment_fails() {
    let shape = common::koala_shape_single_constraint(2);
    let (pk, vk) = setup_keys_with_mode(&shape, MatrixClosingMode::DirectSparse);
    let mut prover_challenger = spartan_whir::poseidon_challenger();
    let mut verifier_challenger = spartan_whir::poseidon_challenger();

    let (instance, proof) = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::prove(
        &pk,
        &common::koala_public_inputs(5),
        &common::koala_witness(5),
        &mut prover_challenger,
    )
    .expect("prove succeeds");

    let mut bad_instance = instance;
    let mut roots = bad_instance.witness_commitment.clone().into_roots();
    roots[0][0] += F::ONE;
    bad_instance.witness_commitment = roots.into();
    let verified = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::verify(
        &vk,
        &bad_instance,
        &proof,
        &mut verifier_challenger,
    );
    assert_eq!(verified, Err(SpartanWhirError::SumcheckFailed));
}

#[test]
fn protocol_tampered_outer_claims_fail() {
    let shape = common::koala_shape_single_constraint(2);
    let (pk, vk) = setup_keys_with_mode(&shape, MatrixClosingMode::DirectSparse);
    let mut prover_challenger = spartan_whir::poseidon_challenger();
    let mut verifier_challenger = spartan_whir::poseidon_challenger();

    let (instance, mut proof) = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::prove(
        &pk,
        &common::koala_public_inputs(11),
        &common::koala_witness(11),
        &mut prover_challenger,
    )
    .expect("prove succeeds");

    proof.outer_claims.0 += EF::ONE;
    let verified = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::verify(
        &vk,
        &instance,
        &proof,
        &mut verifier_challenger,
    );
    assert_eq!(verified, Err(SpartanWhirError::SumcheckFailed));
}

#[test]
fn protocol_tampered_witness_eval_fails() {
    let shape = common::koala_shape_single_constraint(2);
    let (pk, vk) = setup_keys_with_mode(&shape, MatrixClosingMode::DirectSparse);
    let mut prover_challenger = spartan_whir::poseidon_challenger();
    let mut verifier_challenger = spartan_whir::poseidon_challenger();

    let (instance, mut proof) = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::prove(
        &pk,
        &common::koala_public_inputs(11),
        &common::koala_witness(11),
        &mut prover_challenger,
    )
    .expect("prove succeeds");

    proof.witness_eval += EF::ONE;
    let verified = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::verify(
        &vk,
        &instance,
        &proof,
        &mut verifier_challenger,
    );
    assert!(matches!(
        verified,
        Err(SpartanWhirError::SumcheckFailed) | Err(SpartanWhirError::WhirVerifyFailed)
    ));
}

#[test]
fn protocol_tampered_pcs_proof_fails() {
    let shape = common::koala_shape_single_constraint(2);
    let (pk, vk) = setup_keys_with_mode(&shape, MatrixClosingMode::DirectSparse);
    let mut prover_challenger = spartan_whir::poseidon_challenger();
    let mut verifier_challenger = spartan_whir::poseidon_challenger();

    let (instance, mut proof) = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::prove(
        &pk,
        &common::koala_public_inputs(15),
        &common::koala_witness(15),
        &mut prover_challenger,
    )
    .expect("prove succeeds");

    *proof
        .pcs_proof
        .initial_ood_answers
        .first_mut()
        .expect("plain WHIR proof has an initial OOD answer") += EF::ONE;

    let verified = SpartanProtocol::<PoseidonEngine, Plonky3WhirPcs>::verify(
        &vk,
        &instance,
        &proof,
        &mut verifier_challenger,
    );
    assert!(matches!(
        verified,
        Err(SpartanWhirError::WhirVerifyFailed)
            | Err(SpartanWhirError::CommitmentMismatch)
            | Err(SpartanWhirError::SumcheckFailed)
    ));
}
