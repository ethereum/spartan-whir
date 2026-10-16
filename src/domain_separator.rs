use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

use crate::{
    canonical_r1cs_relation_digest, engine::F, R1csShape, SecurityConfig, SoundnessAssumption,
    SparkWhirParams, SpartanWhirError, WhirParams,
};

pub const NO_ZK_PROTOCOL_ID: &[u8] = b"spartan-whir-no-zk-v1";
pub const FULL_ZK_PROTOCOL_ID: &[u8] = b"spartan-whir-full-zk-v1";
pub const POSEIDON1_NO_ZK_PROTOCOL_ID: &[u8] = b"spartan-whir-poseidon1-no-zk-v1";
pub const POSEIDON1_FULL_ZK_PROTOCOL_ID: &[u8] = b"spartan-whir-poseidon1-full-zk-v1";
/// Version of the verifier-visible SPARK matrix-closing transcript contract.
/// Version 3 uses tagged transcript branches for the fixed-table and read-table
/// openings.
pub const SPARK_MATRIX_CLOSING_VERSION: u8 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatrixClosingMode {
    DirectSparse,
    Spark,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DomainSeparator {
    pub protocol_id: Vec<u8>,
    pub matrix_closing: MatrixClosingMode,
    pub num_cons: usize,
    pub num_vars: usize,
    pub num_io: usize,
    /// Canonical digest of the complete padded A/B/C relation and its
    /// original dimensions. It is included in the transcript domain separator
    /// before the first Fiat-Shamir challenge, following the mitigation in
    /// Fenzi, ePrint 2026/1838: <https://eprint.iacr.org/2026/1838>.
    pub relation_digest: [u8; 32],
    pub security_level_bits: u32,
    pub merkle_security_bits: u32,
    pub soundness_assumption: SoundnessAssumption,
    pub whir_params: WhirParams,
    #[serde(default)]
    pub spark_whir_params: Option<SparkWhirParams>,
}

impl DomainSeparator {
    pub fn new(
        shape: &R1csShape<F>,
        security: &SecurityConfig,
        whir_params: &WhirParams,
    ) -> Result<Self, SpartanWhirError> {
        Self::new_with_matrix_closing(
            shape,
            security,
            whir_params,
            MatrixClosingMode::DirectSparse,
        )
    }

    pub fn new_with_matrix_closing(
        shape: &R1csShape<F>,
        security: &SecurityConfig,
        whir_params: &WhirParams,
        matrix_closing: MatrixClosingMode,
    ) -> Result<Self, SpartanWhirError> {
        Self::new_with_matrix_closing_and_spark_whir_params(
            shape,
            security,
            whir_params,
            matrix_closing,
            None,
        )
    }

    pub fn new_with_matrix_closing_and_spark_whir_params(
        shape: &R1csShape<F>,
        security: &SecurityConfig,
        whir_params: &WhirParams,
        matrix_closing: MatrixClosingMode,
        spark_whir_params: Option<SparkWhirParams>,
    ) -> Result<Self, SpartanWhirError> {
        Self::new_with_protocol_id(
            NO_ZK_PROTOCOL_ID,
            shape,
            shape.num_cons,
            shape.num_vars,
            shape.num_io,
            security,
            whir_params,
            matrix_closing,
            spark_whir_params,
        )
    }

    pub fn new_full_zk(
        shape: &R1csShape<F>,
        security: &SecurityConfig,
        whir_params: &WhirParams,
        matrix_closing: MatrixClosingMode,
        spark_whir_params: Option<SparkWhirParams>,
    ) -> Result<Self, SpartanWhirError> {
        Self::new_with_protocol_id(
            FULL_ZK_PROTOCOL_ID,
            shape,
            shape.num_cons,
            shape.num_vars,
            shape.num_io,
            security,
            whir_params,
            matrix_closing,
            spark_whir_params,
        )
    }

    pub(crate) fn new_with_protocol_id(
        protocol_id: &[u8],
        shape: &R1csShape<F>,
        num_cons_unpadded: usize,
        num_vars_unpadded: usize,
        num_io: usize,
        security: &SecurityConfig,
        whir_params: &WhirParams,
        matrix_closing: MatrixClosingMode,
        spark_whir_params: Option<SparkWhirParams>,
    ) -> Result<Self, SpartanWhirError> {
        let relation_digest =
            canonical_r1cs_relation_digest(shape, num_cons_unpadded, num_vars_unpadded, num_io)?;
        Ok(Self::new_with_protocol_id_and_relation_digest(
            protocol_id,
            shape,
            relation_digest,
            security,
            whir_params,
            matrix_closing,
            spark_whir_params,
        ))
    }

    pub(crate) fn new_with_protocol_id_and_relation_digest(
        protocol_id: &[u8],
        shape: &R1csShape<F>,
        relation_digest: [u8; 32],
        security: &SecurityConfig,
        whir_params: &WhirParams,
        matrix_closing: MatrixClosingMode,
        spark_whir_params: Option<SparkWhirParams>,
    ) -> Self {
        let spark_whir_params = match matrix_closing {
            MatrixClosingMode::DirectSparse => None,
            MatrixClosingMode::Spark => spark_whir_params,
        };
        Self {
            protocol_id: protocol_id.to_vec(),
            matrix_closing,
            num_cons: shape.num_cons,
            num_vars: shape.num_vars,
            num_io: shape.num_io,
            relation_digest,
            security_level_bits: security.security_level_bits,
            merkle_security_bits: security.merkle_security_bits,
            soundness_assumption: security.soundness_assumption,
            whir_params: whir_params.clone(),
            spark_whir_params,
        }
    }

    /// Encode the variable-length protocol identifier with its little-endian
    /// u64 byte length before the fixed and separately framed configuration.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(self.protocol_id.len() as u64).to_le_bytes());
        out.extend_from_slice(&self.protocol_id);
        out.push(matrix_closing_to_byte(self.matrix_closing));
        if self.matrix_closing == MatrixClosingMode::Spark {
            out.push(SPARK_MATRIX_CLOSING_VERSION);
        }
        out.extend_from_slice(&(self.num_cons as u64).to_le_bytes());
        out.extend_from_slice(&(self.num_vars as u64).to_le_bytes());
        out.extend_from_slice(&(self.num_io as u64).to_le_bytes());
        out.extend_from_slice(&self.relation_digest);
        out.extend_from_slice(&self.security_level_bits.to_le_bytes());
        out.extend_from_slice(&self.merkle_security_bits.to_le_bytes());
        out.push(soundness_to_byte(self.soundness_assumption));
        encode_whir_params(&self.whir_params, &mut out);
        if self.matrix_closing == MatrixClosingMode::Spark {
            if let Some(spark_whir_params) = &self.spark_whir_params {
                out.push(1);
                encode_whir_params(&spark_whir_params.fixed_value, &mut out);
                encode_whir_params(&spark_whir_params.fixed_audit, &mut out);
                encode_whir_params(&spark_whir_params.read, &mut out);
            }
        }
        out
    }
}

fn encode_whir_params(params: &WhirParams, out: &mut Vec<u8>) {
    out.extend_from_slice(&params.pow_bits.to_le_bytes());
    out.extend_from_slice(&(params.folding_factor as u64).to_le_bytes());
    out.extend_from_slice(&(params.starting_log_inv_rate as u64).to_le_bytes());
    out.extend_from_slice(&(params.rs_domain_initial_reduction_factor as u64).to_le_bytes());
    if should_encode_schedule_suffix(params) {
        encode_folding_schedule(params, out);
    }
}

fn should_encode_schedule_suffix(params: &WhirParams) -> bool {
    if !params.round_log_inv_rates.is_empty() {
        return true;
    }
    match &params.folding_schedule {
        None => false,
        Some(crate::WhirFoldingSchedule::Constant(factor)) => *factor != params.folding_factor,
        Some(_) => true,
    }
}

fn encode_folding_schedule(params: &WhirParams, out: &mut Vec<u8>) {
    match params.effective_folding_schedule() {
        crate::WhirFoldingSchedule::Constant(factor) => {
            out.push(0);
            out.extend_from_slice(&(factor as u64).to_le_bytes());
        }
        crate::WhirFoldingSchedule::ConstantFromSecondRound { first, rest } => {
            out.push(1);
            out.extend_from_slice(&(first as u64).to_le_bytes());
            out.extend_from_slice(&(rest as u64).to_le_bytes());
        }
        crate::WhirFoldingSchedule::PerRound(factors) => {
            out.push(2);
            out.extend_from_slice(&(factors.len() as u64).to_le_bytes());
            for factor in factors {
                out.extend_from_slice(&(factor as u64).to_le_bytes());
            }
        }
    }
    out.extend_from_slice(&(params.round_log_inv_rates.len() as u64).to_le_bytes());
    for rate in &params.round_log_inv_rates {
        out.extend_from_slice(&(*rate as u64).to_le_bytes());
    }
}

fn matrix_closing_to_byte(matrix_closing: MatrixClosingMode) -> u8 {
    match matrix_closing {
        MatrixClosingMode::DirectSparse => 0,
        MatrixClosingMode::Spark => 1,
    }
}

fn soundness_to_byte(soundness: SoundnessAssumption) -> u8 {
    match soundness {
        SoundnessAssumption::UniqueDecoding => 0,
        SoundnessAssumption::JohnsonBound => 1,
        SoundnessAssumption::CapacityBound => 2,
    }
}
