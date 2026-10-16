#![allow(dead_code)]

use p3_field::PrimeCharacteristicRing;

use spartan_whir::{
    engine::F, DomainSeparator, Evaluations, LinearConstraintClaim, MlePcs, MultilinearPoint,
    PcsStatement, PcsStatementBuilder, PointEvalClaim, R1csInstance, R1csShape, R1csWitness,
    SecurityConfig, SoundnessAssumption, SparseMatEntry, SparseMatrix, SpartanWhirEngine,
    SpartanWhirError, WhirParams, WhirPcsConfig,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DummyChallenger;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DummyEngine;

impl SpartanWhirEngine for DummyEngine {
    type F = u64;
    type EF = u64;
    type Challenger = DummyChallenger;
    type Hash = ();
    type Compress = ();
    type W = u64;
    type PackedF = u64;
    type PackedW = u64;

    const DIGEST_ELEMS: usize = 4;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DummyPcs;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DummyPcsConfig;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DummyPcsProof;

impl MlePcs<DummyEngine> for DummyPcs {
    type Commitment = [u64; 4];
    type ProverData = ();
    type Proof = DummyPcsProof;
    type Config = DummyPcsConfig;

    fn commit(
        _config: &Self::Config,
        _poly: &Evaluations<<DummyEngine as SpartanWhirEngine>::F>,
        _challenger: &mut <DummyEngine as SpartanWhirEngine>::Challenger,
    ) -> Result<(Self::Commitment, Self::ProverData), SpartanWhirError> {
        Err(SpartanWhirError::Unimplemented("pcs::commit"))
    }

    fn open(
        _config: &Self::Config,
        _prover_data: Self::ProverData,
        _statement: &PcsStatement<DummyEngine>,
        _challenger: &mut <DummyEngine as SpartanWhirEngine>::Challenger,
    ) -> Result<Self::Proof, SpartanWhirError> {
        Err(SpartanWhirError::Unimplemented("pcs::open"))
    }

    fn verify(
        _config: &Self::Config,
        _commitment: &Self::Commitment,
        _statement: &PcsStatement<DummyEngine>,
        _proof: &Self::Proof,
        _challenger: &mut <DummyEngine as SpartanWhirEngine>::Challenger,
    ) -> Result<(), SpartanWhirError> {
        Err(SpartanWhirError::Unimplemented("pcs::verify"))
    }
}

pub fn sample_sparse_matrix() -> SparseMatrix<u64> {
    SparseMatrix {
        num_rows: 1,
        num_cols: 3,
        entries: vec![SparseMatEntry {
            row: 0,
            col: 0,
            val: 1,
        }],
    }
}

pub fn sample_shape() -> R1csShape<u64> {
    R1csShape {
        num_cons: 1,
        num_vars: 1,
        num_io: 1,
        a: sample_sparse_matrix(),
        b: sample_sparse_matrix(),
        c: sample_sparse_matrix(),
    }
}

pub fn sample_instance() -> R1csInstance<u64, [u64; 4]> {
    R1csInstance {
        public_inputs: vec![7],
        witness_commitment: [0; 4],
    }
}

pub fn sample_witness() -> R1csWitness<u64> {
    R1csWitness { w: vec![42] }
}

pub fn sample_statement() -> PcsStatement<DummyEngine> {
    PcsStatementBuilder::<DummyEngine>::new()
        .add_point_eval(PointEvalClaim {
            point: MultilinearPoint(vec![0, 1]),
            value: 9,
        })
        .finalize()
        .expect("non-empty statement finalizes")
}

pub fn sample_linear_statement() -> PcsStatement<DummyEngine> {
    PcsStatementBuilder::<DummyEngine>::new()
        .add_linear_constraint(LinearConstraintClaim {
            coefficients: vec![1, 2, 3],
            expected: 6,
        })
        .finalize()
        .expect("non-empty statement finalizes")
}

pub fn sample_domain_separator() -> DomainSeparator {
    DomainSeparator::new(
        &koala_shape_single_constraint(1),
        &SecurityConfig::default(),
        &WhirParams::default(),
    )
    .expect("sample relation has a domain separator")
}

pub fn koala_shape_single_constraint(num_cons: usize) -> R1csShape<F> {
    let a_entries = (0..num_cons)
        .map(|row| SparseMatEntry {
            row,
            col: 0,
            val: F::ONE,
        })
        .collect();
    let b_entries = (0..num_cons)
        .map(|row| SparseMatEntry {
            row,
            col: 1,
            val: F::ONE,
        })
        .collect();
    let c_entries = (0..num_cons)
        .map(|row| SparseMatEntry {
            row,
            col: 2,
            val: F::ONE,
        })
        .collect();

    R1csShape {
        num_cons,
        num_vars: 1,
        num_io: 1,
        a: SparseMatrix {
            num_rows: num_cons,
            num_cols: 3,
            entries: a_entries,
        },
        b: SparseMatrix {
            num_rows: num_cons,
            num_cols: 3,
            entries: b_entries,
        },
        c: SparseMatrix {
            num_rows: num_cons,
            num_cols: 3,
            entries: c_entries,
        },
    }
}

pub fn koala_witness(value: u32) -> R1csWitness<F> {
    R1csWitness {
        w: vec![F::from_u32(value)],
    }
}

pub fn koala_public_inputs(value: u32) -> Vec<F> {
    vec![F::from_u32(value)]
}

pub fn phase3_security() -> SecurityConfig {
    SecurityConfig {
        security_level_bits: 80,
        merkle_security_bits: 80,
        soundness_assumption: SoundnessAssumption::CapacityBound,
    }
}

pub fn phase3_whir_params() -> WhirParams {
    WhirParams {
        pow_bits: 0,
        folding_factor: 1,
        starting_log_inv_rate: 6,
        rs_domain_initial_reduction_factor: 1,
        ..WhirParams::default()
    }
}

pub fn phase3_zk_whir_params() -> WhirParams {
    let mut params = phase3_whir_params();
    // The hiding base case counts its query rows in the terminal source
    // code, so the small fixtures need additional rate for the strongest
    // security-boundary tests.
    params.starting_log_inv_rate = 8;
    params
}

pub fn phase3_pcs_config() -> WhirPcsConfig {
    WhirPcsConfig {
        num_variables: 0,
        security: phase3_security(),
        whir: phase3_whir_params(),
    }
}

pub fn k22_jb100_security() -> SecurityConfig {
    SecurityConfig {
        security_level_bits: 100,
        merkle_security_bits: 80,
        soundness_assumption: SoundnessAssumption::JohnsonBound,
    }
}

pub fn k22_jb100_whir_params() -> WhirParams {
    WhirParams {
        pow_bits: 30,
        folding_factor: 4,
        starting_log_inv_rate: 6,
        rs_domain_initial_reduction_factor: 1,
        ..WhirParams::default()
    }
}

pub fn k22_jb100_pcs_config() -> WhirPcsConfig {
    WhirPcsConfig {
        num_variables: 22,
        security: k22_jb100_security(),
        whir: k22_jb100_whir_params(),
    }
}
