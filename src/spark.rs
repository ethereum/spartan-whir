use alloc::{vec, vec::Vec};
use core::cmp::Ordering;

use p3_challenger::FieldChallenger;
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing, PrimeField32};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use serde::{Deserialize, Serialize};

use crate::profiling::{profile_detail_scope, profile_scope};
use crate::sumcheck_replay::{observe_sumcheck_claim, replay_compact_rounds};
use crate::{
    engine::F, evaluate_mle_table, CubicRoundPoly, EqPolynomial, MultilinearPoint, R1csShape,
    SparseMatrix, SpartanWhirError,
};

// SPARK's memory checks and batched product/dotproduct shape follow Microsoft
// Spartan and the arkworks Spartan port. This crate adapts that verifier shape
// to WHIR commitments and uses a measured shared-union sparse-table layout when
// it is smaller than the compatibility fallback.

const SPARK_PARALLEL_ROUND_MIN_PAIRS: usize = 1 << 14;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SparkMatrixSlot {
    A = 0,
    B = 1,
    C = 2,
    Zero = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SparkLayoutKind {
    Joint,
    SharedUnion,
    PerMatrix,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SparkLayoutDecision {
    SharedUnion,
    PerMatrix,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SparkMemoryAxis {
    Row,
    Col,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparkLayoutEstimate {
    pub kind: SparkLayoutKind,
    pub value_domain_size: usize,
    pub value_rounds: usize,
    pub raw_nnz_a: usize,
    pub raw_nnz_b: usize,
    pub raw_nnz_c: usize,
    pub aggregated_nnz_a: usize,
    pub aggregated_nnz_b: usize,
    pub aggregated_nnz_c: usize,
    pub union_nnz: usize,
    pub union_ratio_ppm: u64,
    pub max_matrix_nnz_padded: usize,
    pub wasted_value_slots: usize,
    pub wasted_value_slot_ratio_ppm: u64,
    pub setup_commitments: usize,
    pub size_n_polynomials: usize,
    pub size_m_polynomials: usize,
    pub grand_product_checks: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparkShapeProfile {
    pub num_rows: usize,
    pub num_cols: usize,
    pub nnz_a: usize,
    pub nnz_b: usize,
    pub nnz_c: usize,
    pub union_nnz: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparkLayoutComparison {
    pub joint: SparkLayoutEstimate,
    pub per_matrix: SparkLayoutEstimate,
    pub decision: SparkLayoutDecision,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparkTables {
    pub layout: SparkLayoutKind,
    pub row_memory_size: usize,
    pub col_memory_size: usize,
    pub matrix_nnz_padded: usize,
    pub value_domain_size: usize,
    pub raw_nnz: [usize; 3],
    pub aggregated_nnz: [usize; 3],
    pub union_nnz: usize,
    pub union_ratio_ppm: u64,
    pub rows: Vec<F>,
    pub cols: Vec<F>,
    pub val_a: Vec<F>,
    pub val_b: Vec<F>,
    pub val_c: Vec<F>,
    // Compatibility alias for legacy joint-table helpers. In the shared-union
    // layout this is equal to `val_a` and is not the matrix RLC value.
    pub vals: Vec<F>,
    pub read_ts_row: Vec<F>,
    pub read_ts_col: Vec<F>,
    pub audit_ts_row: Vec<F>,
    pub audit_ts_col: Vec<F>,
    pub slot_mapping: [SparkMatrixSlot; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparkTableMetadata {
    pub layout: SparkLayoutKind,
    pub row_memory_size: usize,
    pub col_memory_size: usize,
    pub value_domain_size: usize,
    pub matrix_nnz_padded: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparkReadTables<EF> {
    pub erow: Vec<EF>,
    pub ecol: Vec<EF>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparkVerifierOperationReport {
    pub layout: SparkLayoutKind,
    pub union_nnz: usize,
    pub union_ratio_ppm: u64,
    pub padded_value_domain_size: usize,
    pub padded_memory_domain_size: usize,
    pub max_matrix_nnz_padded: usize,
    pub proof_ops_layers: usize,
    pub proof_mem_layers: usize,
    pub proof_ops_sumcheck_rounds: usize,
    pub proof_mem_sumcheck_rounds: usize,
    pub total_product_sumcheck_rounds: usize,
    pub setup_commitments: usize,
    pub per_proof_commitments: usize,
    pub fixed_value_columns: usize,
    pub fixed_audit_columns: usize,
    pub fixed_audit_embedded: bool,
    pub proof_time_read_columns: usize,
    pub read_commitment_count: usize,
    pub proof_ops_product_count: usize,
    pub proof_ops_dotproduct_count: usize,
    pub proof_mem_product_count: usize,
    pub proof_mem_dotproduct_count: usize,
    pub fixed_value_domain_slots: usize,
    pub proof_time_read_domain_slots: usize,
    pub extension_element_bytes: usize,
    pub product_round_poly_ext_elements: usize,
    pub product_layer_eval_ext_elements: usize,
    pub product_root_claim_ext_elements: usize,
    pub product_wrapper_ext_elements: usize,
    pub product_proof_ext_elements: usize,
    pub fixed_opening_eval_ext_elements: usize,
    pub read_opening_eval_ext_elements: usize,
    pub opening_eval_ext_elements: usize,
    pub duplicate_commitment_bytes: usize,
    pub estimated_product_proof_bytes: usize,
    pub estimated_opening_eval_bytes: usize,
    pub estimated_spark_payload_bytes_excluding_whir: usize,
    pub aggregation_soundness_error_numerator: usize,
}

/// Return whether both fixed audit timestamp tables fit in the unused eighth
/// column of the fixed value bundle.
///
/// The decision depends only on authenticated table dimensions. Shapes that do
/// not satisfy it retain the separate fixed audit commitment.
pub fn spark_fixed_audit_is_embedded(
    value_domain_size: usize,
    row_memory_size: usize,
    col_memory_size: usize,
) -> bool {
    let memory_domain_size = row_memory_size.max(col_memory_size);
    value_domain_size.is_power_of_two()
        && row_memory_size.is_power_of_two()
        && col_memory_size.is_power_of_two()
        && memory_domain_size
            .checked_mul(2)
            .is_some_and(|audit_slots| audit_slots <= value_domain_size)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparkSolidityGasModel {
    pub cubic_sumcheck_round_replay_gas: usize,
    pub whir_opening_execution_gas: usize,
    pub whir_opening_calldata_bytes: usize,
    pub calldata_gas_per_nonzero_byte: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparkSolidityGasEstimate {
    pub whir_opening_count: usize,
    pub product_sumcheck_replay_gas: usize,
    pub whir_opening_execution_gas: usize,
    pub spark_payload_calldata_gas_upper_bound: usize,
    pub whir_opening_calldata_gas_upper_bound: usize,
    pub total_gas_upper_bound: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparkValueRoundPoly<EF>(pub Vec<EF>);

impl<EF> AsRef<[EF]> for SparkValueRoundPoly<EF> {
    fn as_ref(&self) -> &[EF] {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparkValueSumcheckProof<EF> {
    pub rounds: Vec<SparkValueRoundPoly<EF>>,
    pub final_evals: SparkValueFinalEvals<EF>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparkValueFinalEvals<EF> {
    pub selector: EF,
    pub val: EF,
    pub val_a: EF,
    pub val_b: EF,
    pub val_c: EF,
    pub erow: EF,
    pub ecol: EF,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparkGrandProductTree<EF> {
    pub gamma: EF,
    pub layers: Vec<Vec<EF>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparkGrandProductLayerProof<EF> {
    pub rounds: Vec<CubicRoundPoly<EF>>,
    pub left_eval: EF,
    pub right_eval: EF,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparkGrandProductProof<EF> {
    pub root: EF,
    pub layers: Vec<SparkGrandProductLayerProof<EF>>,
    pub leaf_eval: EF,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Dotproduct claim batched into the leaf-adjacent layer of a SPARK batched
/// product proof.
///
/// For a product domain of size `N`, each vector here must have length `N / 2`.
/// The claim proved by this circuit is `sum_i left[i] * right[i] * weight[i]`.
pub struct SparkDotProductCircuit<EF> {
    pub left: Vec<EF>,
    pub right: Vec<EF>,
    pub weight: Vec<EF>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SparkProverDotProductWeight<EF> {
    Base(Vec<F>),
    Extension(Vec<EF>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SparkProverDotProductCircuit<EF> {
    left: Vec<EF>,
    right: Vec<EF>,
    weight: SparkProverDotProductWeight<EF>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SparkProverDotProductBatch<EF> {
    lefts: Vec<Vec<EF>>,
    rights: Vec<Vec<EF>>,
    weights: Vec<SparkProverDotProductWeight<EF>>,
    pair_indices: Vec<usize>,
}

impl<EF> SparkProverDotProductBatch<EF> {
    fn empty() -> Self {
        Self {
            lefts: Vec::new(),
            rights: Vec::new(),
            weights: Vec::new(),
            pair_indices: Vec::new(),
        }
    }

    fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }
}

impl<EF> From<SparkDotProductCircuit<EF>> for SparkProverDotProductCircuit<EF> {
    fn from(dotproduct: SparkDotProductCircuit<EF>) -> Self {
        Self {
            left: dotproduct.left,
            right: dotproduct.right,
            weight: SparkProverDotProductWeight::Extension(dotproduct.weight),
        }
    }
}

impl<EF> From<Vec<SparkProverDotProductCircuit<EF>>> for SparkProverDotProductBatch<EF> {
    fn from(dotproducts: Vec<SparkProverDotProductCircuit<EF>>) -> Self {
        let mut lefts = Vec::with_capacity(dotproducts.len());
        let mut rights = Vec::with_capacity(dotproducts.len());
        let mut weights = Vec::with_capacity(dotproducts.len());
        for dotproduct in dotproducts {
            lefts.push(dotproduct.left);
            rights.push(dotproduct.right);
            weights.push(dotproduct.weight);
        }
        let pair_indices = (0..weights.len()).collect();
        Self {
            lefts,
            rights,
            weights,
            pair_indices,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparkBatchedProductLayerProof<EF> {
    pub rounds: Vec<CubicRoundPoly<EF>>,
    pub product_left_evals: Vec<EF>,
    pub product_right_evals: Vec<EF>,
    pub dotproduct_left_evals: Vec<EF>,
    pub dotproduct_right_evals: Vec<EF>,
    pub dotproduct_weight_evals: Vec<EF>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparkBatchedProductProof<EF> {
    pub product_roots: Vec<EF>,
    pub dotproduct_claims: Vec<EF>,
    pub layers: Vec<SparkBatchedProductLayerProof<EF>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparkBatchedProductLeafClaims<EF> {
    /// Evaluation point for the product child-layer claims. Its length is
    /// `log2(N)`, where `N` is the product domain size.
    pub product_point: MultilinearPoint<EF>,
    /// Evaluation point for dotproduct leaf claims. Its length is `log2(N) - 1`
    /// because dotproduct vectors live on the split leaf domain `N / 2`.
    pub dotproduct_point: MultilinearPoint<EF>,
    pub product_evals: Vec<EF>,
    pub dotproduct_left_evals: Vec<EF>,
    pub dotproduct_right_evals: Vec<EF>,
    pub dotproduct_weight_evals: Vec<EF>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparkMemoryProductClaim<EF> {
    pub init_root: EF,
    pub read_root: EF,
    pub write_root: EF,
    pub audit_root: EF,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparkMemoryProductProof<EF> {
    pub beta: EF,
    pub gamma: EF,
    pub row: SparkMemoryProductClaim<EF>,
    pub col: SparkMemoryProductClaim<EF>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparkAxisGrandProductProof<EF> {
    pub init: SparkGrandProductProof<EF>,
    pub read: SparkGrandProductProof<EF>,
    pub write: SparkGrandProductProof<EF>,
    pub audit: SparkGrandProductProof<EF>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparkMemoryGrandProductProof<EF> {
    pub products: SparkMemoryProductProof<EF>,
    pub row: SparkAxisGrandProductProof<EF>,
    pub col: SparkAxisGrandProductProof<EF>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparkBatchedMemoryProductsProof<EF> {
    pub products: SparkMemoryProductProof<EF>,
    pub matrix_evals: [EF; 3],
    pub proof_ops: SparkBatchedProductProof<EF>,
    pub proof_mem: SparkBatchedProductProof<EF>,
}

/// Round messages for a product layer in the compact transport encoding.
///
/// The factored form stores the nonconstant coefficients of the quadratic
/// quotient by the known equality factor. Decoding restores the original
/// cubic observations before advancing the transcript.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompactSparkProductRounds<EF> {
    Factored(Vec<[EF; 2]>),
    Cubic(Vec<CubicRoundPoly<EF>>),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactSparkProductLayerProof<EF> {
    pub rounds: CompactSparkProductRounds<EF>,
    pub product_left_evals: Vec<EF>,
    pub product_right_evals: Vec<EF>,
    pub dotproduct_left_evals: Vec<EF>,
    pub dotproduct_right_evals: Vec<EF>,
    pub dotproduct_weight_evals: Vec<EF>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactSparkBatchedProductProof<EF> {
    pub product_roots: Vec<EF>,
    pub dotproduct_claims: Vec<EF>,
    pub layers: Vec<CompactSparkProductLayerProof<EF>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompactSparkProductHeader<EF> {
    Explicit {
        products: SparkMemoryProductProof<EF>,
        matrix_evals: [EF; 3],
    },
    Derived,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactSparkBatchedMemoryProductsProof<EF> {
    pub header: CompactSparkProductHeader<EF>,
    pub proof_ops: CompactSparkBatchedProductProof<EF>,
    pub proof_mem: CompactSparkBatchedProductProof<EF>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparkBatchedMemoryProductsLeafClaims<EF> {
    pub beta: EF,
    pub gamma: EF,
    pub matrix_evals: [EF; 3],
    /// Product claims for row read/write and column read/write, in that order.
    pub ops: SparkBatchedProductLeafClaims<EF>,
    /// Product claims for row init/audit and column init/audit, in that order.
    pub mem: SparkBatchedProductLeafClaims<EF>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparkGrandProductLeafClaim<EF> {
    pub point: MultilinearPoint<EF>,
    pub term_eval: EF,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparkAxisGrandProductLeafClaims<EF> {
    pub init: SparkGrandProductLeafClaim<EF>,
    pub read: SparkGrandProductLeafClaim<EF>,
    pub write: SparkGrandProductLeafClaim<EF>,
    pub audit: SparkGrandProductLeafClaim<EF>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparkMemoryGrandProductLeafClaims<EF> {
    pub beta: EF,
    pub gamma: EF,
    pub row: SparkAxisGrandProductLeafClaims<EF>,
    pub col: SparkAxisGrandProductLeafClaims<EF>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparkFixedTableOpeningEvals<EF> {
    pub val_a_low: EF,
    pub val_a_high: EF,
    pub val_b_low: EF,
    pub val_b_high: EF,
    pub val_c_low: EF,
    pub val_c_high: EF,
    pub row_addr: EF,
    pub col_addr: EF,
    pub row_read_ts: EF,
    pub col_read_ts: EF,
    pub row_audit_ts: EF,
    pub col_audit_ts: EF,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparkReadTableOpeningEvals<EF> {
    pub erow_low: EF,
    pub erow_high: EF,
    pub ecol_low: EF,
    pub ecol_high: EF,
    pub erow_ops: EF,
    pub ecol_ops: EF,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SparkMemoryProductValues<EF> {
    init: Vec<EF>,
    read: Vec<EF>,
    write: Vec<EF>,
    audit: Vec<EF>,
}

#[cfg(test)]
struct SparkAxisProductTerms<EF> {
    init: Vec<EF>,
    read: Vec<EF>,
    write: Vec<EF>,
    audit: Vec<EF>,
}

impl SparkTables {
    pub fn metadata(&self) -> SparkTableMetadata {
        SparkTableMetadata {
            layout: self.layout,
            row_memory_size: self.row_memory_size,
            col_memory_size: self.col_memory_size,
            value_domain_size: self.value_domain_size,
            matrix_nnz_padded: self.matrix_nnz_padded,
        }
    }

    pub fn verifier_operation_report(
        &self,
        extension_dimension: usize,
    ) -> Result<SparkVerifierOperationReport, SpartanWhirError> {
        spark_verifier_operation_report(
            self.layout,
            self.union_nnz,
            self.union_ratio_ppm,
            self.value_domain_size,
            self.matrix_nnz_padded,
            self.row_memory_size,
            self.col_memory_size,
            extension_dimension,
        )
    }
}

impl SparkTableMetadata {
    pub fn validate(&self) -> Result<(), SpartanWhirError> {
        if self.row_memory_size == 0
            || self.col_memory_size == 0
            || self.value_domain_size == 0
            || self.matrix_nnz_padded == 0
            || !self.row_memory_size.is_power_of_two()
            || !self.col_memory_size.is_power_of_two()
            || !self.value_domain_size.is_power_of_two()
            || !self.matrix_nnz_padded.is_power_of_two()
        {
            return Err(SpartanWhirError::InvalidR1csShape);
        }

        match self.layout {
            SparkLayoutKind::SharedUnion => {
                let max_value_domain = self
                    .matrix_nnz_padded
                    .checked_mul(4)
                    .ok_or(SpartanWhirError::InvalidR1csShape)?;
                if self.value_domain_size < self.matrix_nnz_padded
                    || self.value_domain_size > max_value_domain
                {
                    return Err(SpartanWhirError::InvalidR1csShape);
                }
            }
            SparkLayoutKind::Joint | SparkLayoutKind::PerMatrix => {
                if self.value_domain_size
                    != self
                        .matrix_nnz_padded
                        .checked_mul(4)
                        .ok_or(SpartanWhirError::InvalidR1csShape)?
                {
                    return Err(SpartanWhirError::InvalidR1csShape);
                }
            }
        }
        Ok(())
    }

    pub fn verifier_operation_report(
        &self,
        extension_dimension: usize,
    ) -> Result<SparkVerifierOperationReport, SpartanWhirError> {
        self.validate()?;
        spark_verifier_operation_report(
            self.layout,
            0,
            0,
            self.value_domain_size,
            self.matrix_nnz_padded,
            self.row_memory_size,
            self.col_memory_size,
            extension_dimension,
        )
    }
}

impl SparkLayoutEstimate {
    pub fn verifier_operation_report(
        &self,
        row_memory_size: usize,
        col_memory_size: usize,
        extension_dimension: usize,
    ) -> Result<SparkVerifierOperationReport, SpartanWhirError> {
        spark_verifier_operation_report(
            self.kind,
            self.union_nnz,
            self.union_ratio_ppm,
            self.value_domain_size,
            self.max_matrix_nnz_padded,
            row_memory_size,
            col_memory_size,
            extension_dimension,
        )
    }
}

impl SparkVerifierOperationReport {
    pub fn estimate_solidity_gas(
        &self,
        model: SparkSolidityGasModel,
    ) -> Result<SparkSolidityGasEstimate, SpartanWhirError> {
        let whir_opening_count = self
            .setup_commitments
            .checked_add(self.read_commitment_count)
            .ok_or(SpartanWhirError::invalid_config())?;
        let product_sumcheck_replay_gas = checked_mul(
            self.total_product_sumcheck_rounds,
            model.cubic_sumcheck_round_replay_gas,
        )?;
        let whir_opening_execution_gas =
            checked_mul(whir_opening_count, model.whir_opening_execution_gas)?;
        let spark_payload_calldata_gas_upper_bound = checked_mul(
            self.estimated_spark_payload_bytes_excluding_whir,
            model.calldata_gas_per_nonzero_byte,
        )?;
        let whir_opening_calldata_gas_upper_bound =
            checked_mul(whir_opening_count, model.whir_opening_calldata_bytes)
                .and_then(|bytes| checked_mul(bytes, model.calldata_gas_per_nonzero_byte))?;
        let total_gas_upper_bound = product_sumcheck_replay_gas
            .checked_add(whir_opening_execution_gas)
            .and_then(|n| n.checked_add(spark_payload_calldata_gas_upper_bound))
            .and_then(|n| n.checked_add(whir_opening_calldata_gas_upper_bound))
            .ok_or(SpartanWhirError::invalid_config())?;

        Ok(SparkSolidityGasEstimate {
            whir_opening_count,
            product_sumcheck_replay_gas,
            whir_opening_execution_gas,
            spark_payload_calldata_gas_upper_bound,
            whir_opening_calldata_gas_upper_bound,
            total_gas_upper_bound,
        })
    }
}

fn spark_verifier_operation_report(
    layout: SparkLayoutKind,
    union_nnz: usize,
    union_ratio_ppm: u64,
    value_domain_size: usize,
    max_matrix_nnz_padded: usize,
    row_memory_size: usize,
    col_memory_size: usize,
    extension_dimension: usize,
) -> Result<SparkVerifierOperationReport, SpartanWhirError> {
    if extension_dimension == 0 {
        return Err(SpartanWhirError::invalid_config());
    }
    let proof_time_read_columns = extension_dimension
        .checked_mul(2)
        .ok_or(SpartanWhirError::invalid_config())?;
    let read_commitment_count = proof_time_read_columns.count_ones() as usize;
    let memory_domain_size = row_memory_size
        .max(col_memory_size)
        .checked_next_power_of_two()
        .ok_or(SpartanWhirError::invalid_config())?;
    let fixed_audit_embedded =
        spark_fixed_audit_is_embedded(value_domain_size, row_memory_size, col_memory_size);
    let proof_ops_layers = log2_power_of_two(value_domain_size);
    let proof_mem_layers = log2_power_of_two(memory_domain_size);
    let proof_ops_sumcheck_rounds = triangular_round_count(proof_ops_layers)?;
    let proof_mem_sumcheck_rounds = triangular_round_count(proof_mem_layers)?;
    let total_product_sumcheck_rounds = proof_ops_sumcheck_rounds
        .checked_add(proof_mem_sumcheck_rounds)
        .ok_or(SpartanWhirError::invalid_config())?;

    let proof_ops_product_count: usize = 4;
    let proof_ops_dotproduct_count: usize = 6;
    let proof_mem_product_count: usize = 4;
    let proof_mem_dotproduct_count: usize = 0;

    let product_round_poly_ext_elements = checked_mul(3, total_product_sumcheck_rounds)?;
    let proof_ops_product_layer_evals = checked_mul(2 * proof_ops_product_count, proof_ops_layers)?;
    let proof_ops_dotproduct_leaf_evals = checked_mul(3, proof_ops_dotproduct_count)?;
    let proof_mem_product_layer_evals = checked_mul(2 * proof_mem_product_count, proof_mem_layers)?;
    let product_layer_eval_ext_elements = proof_ops_product_layer_evals
        .checked_add(proof_ops_dotproduct_leaf_evals)
        .and_then(|n| n.checked_add(proof_mem_product_layer_evals))
        .ok_or(SpartanWhirError::invalid_config())?;

    let product_root_claim_ext_elements = proof_ops_product_count
        .checked_add(proof_ops_dotproduct_count)
        .and_then(|n| n.checked_add(proof_mem_product_count))
        .ok_or(SpartanWhirError::invalid_config())?;
    // beta, gamma, row/column init/read/write/audit roots, and the three
    // matrix evaluation claims currently live in the wrapper proof object.
    let product_wrapper_ext_elements: usize = 2 + 8 + 3;
    let product_proof_ext_elements = product_round_poly_ext_elements
        .checked_add(product_layer_eval_ext_elements)
        .and_then(|n| n.checked_add(product_root_claim_ext_elements))
        .and_then(|n| n.checked_add(product_wrapper_ext_elements))
        .ok_or(SpartanWhirError::invalid_config())?;

    let fixed_opening_eval_ext_elements: usize = 12;
    let read_opening_eval_ext_elements = checked_mul(6, extension_dimension)?;
    let opening_eval_ext_elements = fixed_opening_eval_ext_elements
        .checked_add(read_opening_eval_ext_elements)
        .ok_or(SpartanWhirError::invalid_config())?;
    let extension_element_bytes = checked_mul(4, extension_dimension)?;
    let setup_commitments = if fixed_audit_embedded { 1usize } else { 2usize };
    let duplicate_commitment_bytes = setup_commitments
        .checked_add(read_commitment_count)
        .and_then(|count| count.checked_mul(32))
        .ok_or(SpartanWhirError::invalid_config())?;
    let estimated_product_proof_bytes =
        checked_mul(product_proof_ext_elements, extension_element_bytes)?;
    let estimated_opening_eval_bytes =
        checked_mul(opening_eval_ext_elements, extension_element_bytes)?;
    let estimated_spark_payload_bytes_excluding_whir = estimated_product_proof_bytes
        .checked_add(estimated_opening_eval_bytes)
        .and_then(|n| n.checked_add(duplicate_commitment_bytes))
        .ok_or(SpartanWhirError::invalid_config())?;

    Ok(SparkVerifierOperationReport {
        layout,
        union_nnz,
        union_ratio_ppm,
        padded_value_domain_size: value_domain_size,
        padded_memory_domain_size: memory_domain_size,
        max_matrix_nnz_padded,
        proof_ops_layers,
        proof_mem_layers,
        proof_ops_sumcheck_rounds,
        proof_mem_sumcheck_rounds,
        total_product_sumcheck_rounds,
        setup_commitments,
        per_proof_commitments: read_commitment_count,
        fixed_value_columns: 8,
        fixed_audit_columns: 2,
        fixed_audit_embedded,
        proof_time_read_columns,
        read_commitment_count,
        proof_ops_product_count,
        proof_ops_dotproduct_count,
        proof_mem_product_count,
        proof_mem_dotproduct_count,
        fixed_value_domain_slots: checked_mul(8, value_domain_size)?,
        proof_time_read_domain_slots: checked_mul(proof_time_read_columns, value_domain_size)?,
        extension_element_bytes,
        product_round_poly_ext_elements,
        product_layer_eval_ext_elements,
        product_root_claim_ext_elements,
        product_wrapper_ext_elements,
        product_proof_ext_elements,
        fixed_opening_eval_ext_elements,
        read_opening_eval_ext_elements,
        opening_eval_ext_elements,
        duplicate_commitment_bytes,
        estimated_product_proof_bytes,
        estimated_opening_eval_bytes,
        estimated_spark_payload_bytes_excluding_whir,
        aggregation_soundness_error_numerator: 2,
    })
}

fn triangular_round_count(depth: usize) -> Result<usize, SpartanWhirError> {
    depth
        .checked_mul(depth.saturating_sub(1))
        .and_then(|n| n.checked_div(2))
        .ok_or(SpartanWhirError::invalid_config())
}

fn checked_mul(lhs: usize, rhs: usize) -> Result<usize, SpartanWhirError> {
    lhs.checked_mul(rhs)
        .ok_or(SpartanWhirError::invalid_config())
}

impl<EF: Field> SparkValueRoundPoly<EF> {
    fn eval_at_zero(&self) -> EF {
        self.0[0]
    }

    fn eval_at_one_from_claim(&self, claim: EF) -> EF {
        claim - self.eval_at_zero()
    }

    fn evaluate_at(&self, r: EF, claim: EF) -> EF {
        let mut ys = Vec::with_capacity(self.0.len() + 1);
        ys.push(self.eval_at_zero());
        ys.push(self.eval_at_one_from_claim(claim));
        ys.extend_from_slice(&self.0[1..]);
        interpolate_at(&ys, r)
    }
}

impl<EF: Field> SparkGrandProductTree<EF> {
    pub fn build(values: &[EF], gamma: EF) -> Result<Self, SpartanWhirError>
    where
        EF: Send + Sync,
    {
        if values.is_empty() || !values.len().is_power_of_two() {
            return Err(SpartanWhirError::InvalidPolynomialLength);
        }

        let mut layers = Vec::new();
        layers.push(
            values
                .iter()
                .map(|&value| value - gamma)
                .collect::<Vec<_>>(),
        );
        while layers.last().is_some_and(|layer| layer.len() > 1) {
            let previous = layers
                .last()
                .ok_or(SpartanWhirError::InvalidPolynomialLength)?;
            layers.push(product_tree_next_layer(previous));
        }

        Ok(Self { gamma, layers })
    }

    pub fn root(&self) -> Result<EF, SpartanWhirError> {
        self.layers
            .last()
            .and_then(|layer| layer.first())
            .copied()
            .ok_or(SpartanWhirError::InvalidPolynomialLength)
    }

    pub fn verify_tree(&self) -> Result<(), SpartanWhirError> {
        if self.layers.is_empty() || self.layers[0].is_empty() {
            return Err(SpartanWhirError::InvalidPolynomialLength);
        }
        for adjacent in self.layers.windows(2) {
            let lower = &adjacent[0];
            let upper = &adjacent[1];
            if lower.len() != upper.len() * 2 {
                return Err(SpartanWhirError::InvalidPolynomialLength);
            }
            for (i, pair) in lower.chunks_exact(2).enumerate() {
                if pair[0] * pair[1] != upper[i] {
                    return Err(SpartanWhirError::SumcheckFailed);
                }
            }
        }
        if self.layers.last().map_or(0, Vec::len) != 1 {
            return Err(SpartanWhirError::InvalidPolynomialLength);
        }
        Ok(())
    }
}

pub fn compare_spark_layouts(
    shape: &R1csShape<F>,
) -> Result<SparkLayoutComparison, SpartanWhirError> {
    shape.validate()?;
    validate_packed_index_bounds(shape)?;
    let union_nnz = union_sparse_entries(&shape.a, &shape.b, &shape.c)?.len();

    compare_spark_layout_profile(SparkShapeProfile {
        num_rows: shape.num_cons,
        num_cols: spark_col_memory_size(shape)?,
        nnz_a: shape.a.nnz(),
        nnz_b: shape.b.nnz(),
        nnz_c: shape.c.nnz(),
        union_nnz,
    })
}

pub fn compare_spark_layout_profile(
    profile: SparkShapeProfile,
) -> Result<SparkLayoutComparison, SpartanWhirError> {
    validate_profile_packed_index_bounds(profile)?;

    let nnz_a = profile.nnz_a;
    let nnz_b = profile.nnz_b;
    let nnz_c = profile.nnz_c;
    let max_nnz_padded = padded_nonzero_domain(nnz_a.max(nnz_b).max(nnz_c))?;
    let union_nnz = if profile.union_nnz == 0 {
        nnz_a.max(nnz_b).max(nnz_c)
    } else {
        profile.union_nnz
    };
    let union_domain = padded_nonzero_domain(union_nnz)?;
    let union_ratio_ppm = ratio_ppm(union_nnz, nnz_a.max(nnz_b).max(nnz_c).max(1));
    let total_nnz = nnz_a
        .checked_add(nnz_b)
        .and_then(|n| n.checked_add(nnz_c))
        .ok_or(SpartanWhirError::InvalidR1csShape)?;

    let shared_union = SparkLayoutEstimate {
        kind: SparkLayoutKind::SharedUnion,
        value_domain_size: union_domain,
        value_rounds: log2_power_of_two(union_domain),
        raw_nnz_a: nnz_a,
        raw_nnz_b: nnz_b,
        raw_nnz_c: nnz_c,
        aggregated_nnz_a: nnz_a,
        aggregated_nnz_b: nnz_b,
        aggregated_nnz_c: nnz_c,
        union_nnz,
        union_ratio_ppm,
        max_matrix_nnz_padded: max_nnz_padded,
        wasted_value_slots: union_domain.saturating_sub(union_nnz),
        wasted_value_slot_ratio_ppm: ratio_ppm(
            union_domain.saturating_sub(union_nnz),
            union_domain,
        ),
        setup_commitments: if spark_fixed_audit_is_embedded(
            union_domain,
            profile.num_rows,
            profile.num_cols,
        ) {
            1
        } else {
            2
        },
        size_n_polynomials: 7,
        size_m_polynomials: 2,
        grand_product_checks: 2,
    };

    let joint_with_split_vals_domain = max_nnz_padded
        .checked_mul(4)
        .ok_or(SpartanWhirError::InvalidR1csShape)?;
    let per_matrix = SparkLayoutEstimate {
        kind: SparkLayoutKind::PerMatrix,
        value_domain_size: joint_with_split_vals_domain,
        value_rounds: log2_power_of_two(joint_with_split_vals_domain),
        raw_nnz_a: nnz_a,
        raw_nnz_b: nnz_b,
        raw_nnz_c: nnz_c,
        aggregated_nnz_a: nnz_a,
        aggregated_nnz_b: nnz_b,
        aggregated_nnz_c: nnz_c,
        union_nnz,
        union_ratio_ppm,
        max_matrix_nnz_padded: max_nnz_padded,
        wasted_value_slots: joint_with_split_vals_domain.saturating_sub(total_nnz),
        wasted_value_slot_ratio_ppm: ratio_ppm(
            joint_with_split_vals_domain.saturating_sub(total_nnz),
            joint_with_split_vals_domain,
        ),
        setup_commitments: if spark_fixed_audit_is_embedded(
            joint_with_split_vals_domain,
            profile.num_rows,
            profile.num_cols,
        ) {
            1
        } else {
            2
        },
        size_n_polynomials: 7,
        size_m_polynomials: 2,
        grand_product_checks: 8,
    };

    // 1.5x union-ratio cutoff from the gas-oriented plan. Until the real
    // Spartan-style per-matrix product layer lands, avoid routing to the
    // transitional joint-with-split-values fallback when the shared union has a
    // smaller value domain.
    let decision = if union_ratio_ppm <= 1_500_000 || union_domain < joint_with_split_vals_domain {
        SparkLayoutDecision::SharedUnion
    } else {
        SparkLayoutDecision::PerMatrix
    };

    Ok(SparkLayoutComparison {
        joint: shared_union,
        per_matrix,
        decision,
    })
}

pub fn preprocess_spark_tables(shape: &R1csShape<F>) -> Result<SparkTables, SpartanWhirError> {
    let comparison = compare_spark_layouts(shape)?;
    match comparison.decision {
        SparkLayoutDecision::SharedUnion => preprocess_shared_union_spark_tables(shape),
        SparkLayoutDecision::PerMatrix => preprocess_joint_with_split_vals_spark_tables(shape),
    }
}

pub fn preprocess_shared_union_spark_tables(
    shape: &R1csShape<F>,
) -> Result<SparkTables, SpartanWhirError> {
    shape.validate()?;
    validate_packed_index_bounds(shape)?;

    let a = aggregate_matrix_entries(&shape.a)?;
    let b = aggregate_matrix_entries(&shape.b)?;
    let c = aggregate_matrix_entries(&shape.c)?;
    let union = union_aggregated_entries(&a, &b, &c)?;
    let max_raw_nnz = shape.a.nnz().max(shape.b.nnz()).max(shape.c.nnz());
    let matrix_nnz_padded = padded_nonzero_domain(max_raw_nnz)?;
    let value_domain_size = padded_nonzero_domain(union.len())?;
    let union_ratio_ppm = ratio_ppm(union.len(), max_raw_nnz.max(1));

    validate_value_domain_field_bound(value_domain_size)?;

    let mut rows = vec![F::ZERO; value_domain_size];
    let mut cols = vec![F::ZERO; value_domain_size];
    let mut val_a = vec![F::ZERO; value_domain_size];
    let mut val_b = vec![F::ZERO; value_domain_size];
    let mut val_c = vec![F::ZERO; value_domain_size];

    for (i, entry) in union.iter().enumerate() {
        rows[i] = usize_to_field(entry.row)?;
        cols[i] = usize_to_field(entry.col)?;
        val_a[i] = entry.val_a;
        val_b[i] = entry.val_b;
        val_c[i] = entry.val_c;
    }

    let row_addrs: Vec<usize> = rows
        .iter()
        .map(|row| row.as_canonical_u32() as usize)
        .collect();
    let col_addrs: Vec<usize> = cols
        .iter()
        .map(|col| col.as_canonical_u32() as usize)
        .collect();
    let (read_ts_row, audit_ts_row) = memory_in_the_head(shape.num_cons, &row_addrs)?;
    let col_memory_size = spark_col_memory_size(shape)?;
    let (read_ts_col, audit_ts_col) = memory_in_the_head(col_memory_size, &col_addrs)?;

    Ok(SparkTables {
        layout: SparkLayoutKind::SharedUnion,
        row_memory_size: shape.num_cons,
        col_memory_size,
        matrix_nnz_padded,
        value_domain_size,
        raw_nnz: [shape.a.nnz(), shape.b.nnz(), shape.c.nnz()],
        aggregated_nnz: [a.len(), b.len(), c.len()],
        union_nnz: union.len(),
        union_ratio_ppm,
        rows,
        cols,
        val_a: val_a.clone(),
        val_b,
        val_c,
        vals: val_a,
        read_ts_row,
        read_ts_col,
        audit_ts_row,
        audit_ts_col,
        slot_mapping: [
            SparkMatrixSlot::A,
            SparkMatrixSlot::B,
            SparkMatrixSlot::C,
            SparkMatrixSlot::Zero,
        ],
    })
}

pub fn preprocess_joint_with_split_vals_spark_tables(
    shape: &R1csShape<F>,
) -> Result<SparkTables, SpartanWhirError> {
    // Compatibility fallback until the Spartan-style independent-stream product
    // layer lands. It keeps A/B/C in separate padded slots, avoids the shared
    // union skeleton, and uses the same val_A/val_B/val_C fixed-bundle columns
    // as the shared-union path.
    let mut tables = preprocess_joint_spark_tables(shape)?;
    tables.layout = SparkLayoutKind::PerMatrix;
    tables.val_a.fill(F::ZERO);
    tables.val_b.fill(F::ZERO);
    tables.val_c.fill(F::ZERO);

    for i in 0..tables.matrix_nnz_padded {
        tables.val_a[i] = tables.vals[i];
    }
    let b_offset = tables.matrix_nnz_padded;
    for i in 0..tables.matrix_nnz_padded {
        tables.val_b[b_offset + i] = tables.vals[b_offset + i];
    }
    let c_offset = 2 * tables.matrix_nnz_padded;
    for i in 0..tables.matrix_nnz_padded {
        tables.val_c[c_offset + i] = tables.vals[c_offset + i];
    }
    tables.union_nnz = shape.a.nnz() + shape.b.nnz() + shape.c.nnz();
    tables.union_ratio_ppm = ratio_ppm(
        tables.union_nnz,
        shape.a.nnz().max(shape.b.nnz()).max(shape.c.nnz()).max(1),
    );
    Ok(tables)
}

pub fn preprocess_per_matrix_spark_tables(
    shape: &R1csShape<F>,
) -> Result<SparkTables, SpartanWhirError> {
    preprocess_joint_with_split_vals_spark_tables(shape)
}

pub fn preprocess_joint_spark_tables(
    shape: &R1csShape<F>,
) -> Result<SparkTables, SpartanWhirError> {
    shape.validate()?;
    validate_packed_index_bounds(shape)?;

    let max_nnz = shape.a.nnz().max(shape.b.nnz()).max(shape.c.nnz());
    let matrix_nnz_padded = padded_nonzero_domain(max_nnz)?;
    let value_domain_size = matrix_nnz_padded
        .checked_mul(4)
        .ok_or(SpartanWhirError::InvalidR1csShape)?;

    let mut rows = vec![F::ZERO; value_domain_size];
    let mut cols = vec![F::ZERO; value_domain_size];
    let mut vals = vec![F::ZERO; value_domain_size];

    fill_joint_slot(
        SparkMatrixSlot::A,
        matrix_nnz_padded,
        &shape.a,
        &mut rows,
        &mut cols,
        &mut vals,
    )?;
    fill_joint_slot(
        SparkMatrixSlot::B,
        matrix_nnz_padded,
        &shape.b,
        &mut rows,
        &mut cols,
        &mut vals,
    )?;
    fill_joint_slot(
        SparkMatrixSlot::C,
        matrix_nnz_padded,
        &shape.c,
        &mut rows,
        &mut cols,
        &mut vals,
    )?;

    let row_addrs = collect_joint_addresses(matrix_nnz_padded, &shape.a, &shape.b, &shape.c, true);
    let col_addrs = collect_joint_addresses(matrix_nnz_padded, &shape.a, &shape.b, &shape.c, false);
    let (read_ts_row, audit_ts_row) = memory_in_the_head(shape.num_cons, &row_addrs)?;
    let col_memory_size = spark_col_memory_size(shape)?;
    let (read_ts_col, audit_ts_col) = memory_in_the_head(col_memory_size, &col_addrs)?;

    Ok(SparkTables {
        layout: SparkLayoutKind::Joint,
        row_memory_size: shape.num_cons,
        col_memory_size,
        matrix_nnz_padded,
        value_domain_size,
        raw_nnz: [shape.a.nnz(), shape.b.nnz(), shape.c.nnz()],
        aggregated_nnz: [shape.a.nnz(), shape.b.nnz(), shape.c.nnz()],
        union_nnz: value_domain_size,
        union_ratio_ppm: ratio_ppm(value_domain_size, max_nnz.max(1)),
        rows,
        cols,
        val_a: vals.clone(),
        val_b: vec![F::ZERO; value_domain_size],
        val_c: vec![F::ZERO; value_domain_size],
        vals,
        read_ts_row,
        read_ts_col,
        audit_ts_row,
        audit_ts_col,
        slot_mapping: [
            SparkMatrixSlot::A,
            SparkMatrixSlot::B,
            SparkMatrixSlot::C,
            SparkMatrixSlot::Zero,
        ],
    })
}

pub fn prove_spark_value_sumcheck<EF, C>(
    tables: &SparkTables,
    r_x: &MultilinearPoint<EF>,
    r_y: &MultilinearPoint<EF>,
    r: EF,
    challenger: &mut C,
) -> Result<(SparkValueSumcheckProof<EF>, MultilinearPoint<EF>, EF), SpartanWhirError>
where
    EF: ExtensionField<F> + Send + Sync,
    C: FieldChallenger<F>,
{
    let read_tables = compute_spark_read_tables(tables, r_x, r_y)?;
    prove_spark_value_sumcheck_with_reads(tables, &read_tables, r, challenger)
}

pub fn prove_spark_value_sumcheck_with_reads<EF, C>(
    tables: &SparkTables,
    read_tables: &SparkReadTables<EF>,
    r: EF,
    challenger: &mut C,
) -> Result<(SparkValueSumcheckProof<EF>, MultilinearPoint<EF>, EF), SpartanWhirError>
where
    EF: ExtensionField<F>,
    C: FieldChallenger<F>,
{
    validate_read_tables(tables, read_tables)?;

    let mut selector = if tables.layout == SparkLayoutKind::Joint {
        joint_selector_table::<EF>(tables, r)?
    } else {
        vec![EF::ONE; tables.value_domain_size]
    };
    let mut vals = if tables.layout == SparkLayoutKind::Joint {
        lift_base_table::<EF>(&tables.vals)
    } else {
        matrix_rlc_table::<EF>(tables, r)?
    };
    let mut erow = read_tables.erow.clone();
    let mut ecol = read_tables.ecol.clone();

    let mut claim = sum_value_product(&selector, &vals, &erow, &ecol)?;
    let initial_claim = claim;
    let num_rounds = log2_power_of_two(tables.value_domain_size);
    let mut rounds = Vec::with_capacity(num_rounds);
    let mut alpha = Vec::with_capacity(num_rounds);
    observe_sumcheck_claim::<F, EF, C>(challenger, claim);

    for _ in 0..num_rounds {
        let round = compute_value_round(
            &selector,
            &vals,
            &erow,
            &ecol,
            tables.layout == SparkLayoutKind::Joint,
        )?;
        challenger.observe_algebra_slice(&round.0);
        let challenge = challenger.sample_algebra_element::<EF>();
        claim = round.evaluate_at(challenge, claim);
        rounds.push(round);
        alpha.push(challenge);

        bind_value_table(&mut selector, challenge)?;
        bind_value_table(&mut vals, challenge)?;
        bind_value_table(&mut erow, challenge)?;
        bind_value_table(&mut ecol, challenge)?;
    }

    if selector.len() != 1 || vals.len() != 1 || erow.len() != 1 || ecol.len() != 1 {
        return Err(SpartanWhirError::SumcheckFailed);
    }

    let final_evals = SparkValueFinalEvals {
        selector: selector[0],
        val: vals[0],
        val_a: evaluate_base_table_as_extension(&tables.val_a, &alpha)?,
        val_b: evaluate_base_table_as_extension(&tables.val_b, &alpha)?,
        val_c: evaluate_base_table_as_extension(&tables.val_c, &alpha)?,
        erow: erow[0],
        ecol: ecol[0],
    };

    Ok((
        SparkValueSumcheckProof {
            rounds,
            final_evals,
        },
        MultilinearPoint(alpha),
        initial_claim,
    ))
}

pub fn verify_spark_value_sumcheck<EF, C>(
    proof: &SparkValueSumcheckProof<EF>,
    initial_claim: EF,
    expected_rounds: usize,
    challenger: &mut C,
) -> Result<(MultilinearPoint<EF>, EF), SpartanWhirError>
where
    EF: ExtensionField<F>,
    C: FieldChallenger<F>,
{
    if proof.rounds.len() != expected_rounds {
        return Err(SpartanWhirError::InvalidRoundCount);
    }
    let degree = proof.rounds.first().map_or(3, |round| round.0.len());
    if !matches!(degree, 3 | 4) {
        return Err(SpartanWhirError::InvalidRoundPolynomial);
    }
    let (alpha, claim) = replay_compact_rounds::<F, EF, C, _>(
        &proof.rounds,
        initial_claim,
        expected_rounds,
        degree,
        challenger,
    )?;

    let expected_final = proof.final_evals.selector
        * proof.final_evals.val
        * proof.final_evals.erow
        * proof.final_evals.ecol;
    if claim != expected_final {
        return Err(SpartanWhirError::SumcheckFailed);
    }

    Ok((alpha, claim))
}

pub fn verify_spark_value_sumcheck_with_tables<EF, C>(
    tables: &SparkTables,
    proof: &SparkValueSumcheckProof<EF>,
    initial_claim: EF,
    r: EF,
    challenger: &mut C,
) -> Result<(MultilinearPoint<EF>, EF), SpartanWhirError>
where
    EF: ExtensionField<F>,
    C: FieldChallenger<F>,
{
    let expected_rounds = log2_power_of_two(tables.value_domain_size);
    validate_value_round_widths(proof, value_round_width(tables.layout))?;
    let (alpha, claim) =
        verify_spark_value_sumcheck(proof, initial_claim, expected_rounds, challenger)?;
    let vals = if tables.layout == SparkLayoutKind::Joint {
        lift_base_table::<EF>(&tables.vals)
    } else {
        matrix_rlc_table::<EF>(tables, r)?
    };

    let expected_selector = if tables.layout == SparkLayoutKind::Joint {
        spark_selector_from_joint_point(tables.slot_mapping, &alpha, r)?
    } else {
        EF::ONE
    };
    let expected_val = evaluate_mle_table(&vals, &alpha.0)?;

    if proof.final_evals.selector != expected_selector || proof.final_evals.val != expected_val {
        return Err(SpartanWhirError::SumcheckFailed);
    }

    Ok((alpha, claim))
}

pub fn verify_spark_value_sumcheck_with_openings<EF, C>(
    layout: SparkLayoutKind,
    slot_mapping: [SparkMatrixSlot; 4],
    proof: &SparkValueSumcheckProof<EF>,
    initial_claim: EF,
    value_domain_size: usize,
    r: EF,
    openings: SparkValueFinalEvals<EF>,
    challenger: &mut C,
) -> Result<(MultilinearPoint<EF>, EF), SpartanWhirError>
where
    EF: ExtensionField<F>,
    C: FieldChallenger<F>,
{
    if value_domain_size == 0 || !value_domain_size.is_power_of_two() {
        return Err(SpartanWhirError::InvalidPolynomialLength);
    }
    let expected_rounds = log2_power_of_two(value_domain_size);
    validate_value_round_widths(proof, value_round_width(layout))?;
    let (alpha, claim) =
        verify_spark_value_sumcheck(proof, initial_claim, expected_rounds, challenger)?;
    let expected_selector = if layout == SparkLayoutKind::Joint {
        spark_selector_from_joint_point(slot_mapping, &alpha, r)?
    } else {
        EF::ONE
    };
    let expected_val = if layout == SparkLayoutKind::Joint {
        openings.val
    } else {
        openings.val_a + r * openings.val_b + r * r * openings.val_c
    };
    if proof.final_evals.selector != expected_selector
        || openings.selector != expected_selector
        || proof.final_evals.val != expected_val
        || openings.val != expected_val
        || proof.final_evals != openings
    {
        return Err(SpartanWhirError::SumcheckFailed);
    }
    Ok((alpha, claim))
}

pub fn verify_spark_value_sumcheck_with_read_tables<EF, C>(
    tables: &SparkTables,
    read_tables: &SparkReadTables<EF>,
    proof: &SparkValueSumcheckProof<EF>,
    initial_claim: EF,
    r: EF,
    challenger: &mut C,
) -> Result<(MultilinearPoint<EF>, EF), SpartanWhirError>
where
    EF: ExtensionField<F>,
    C: FieldChallenger<F>,
{
    validate_read_tables(tables, read_tables)?;
    let expected_rounds = log2_power_of_two(tables.value_domain_size);
    validate_value_round_widths(proof, value_round_width(tables.layout))?;
    let (alpha, claim) =
        verify_spark_value_sumcheck(proof, initial_claim, expected_rounds, challenger)?;
    let selector = if tables.layout == SparkLayoutKind::Joint {
        joint_selector_table::<EF>(tables, r)?
    } else {
        vec![EF::ONE; tables.value_domain_size]
    };
    let vals = if tables.layout == SparkLayoutKind::Joint {
        lift_base_table::<EF>(&tables.vals)
    } else {
        matrix_rlc_table::<EF>(tables, r)?
    };

    let expected_selector = evaluate_mle_table(&selector, &alpha.0)?;
    let expected_val = evaluate_mle_table(&vals, &alpha.0)?;
    let expected_erow = evaluate_mle_table(&read_tables.erow, &alpha.0)?;
    let expected_ecol = evaluate_mle_table(&read_tables.ecol, &alpha.0)?;

    let expected_val_a = evaluate_base_table_as_extension(&tables.val_a, &alpha.0)?;
    let expected_val_b = evaluate_base_table_as_extension(&tables.val_b, &alpha.0)?;
    let expected_val_c = evaluate_base_table_as_extension(&tables.val_c, &alpha.0)?;

    if proof.final_evals.selector != expected_selector
        || proof.final_evals.val != expected_val
        || proof.final_evals.val_a != expected_val_a
        || proof.final_evals.val_b != expected_val_b
        || proof.final_evals.val_c != expected_val_c
        || proof.final_evals.erow != expected_erow
        || proof.final_evals.ecol != expected_ecol
    {
        return Err(SpartanWhirError::SumcheckFailed);
    }

    Ok((alpha, claim))
}

fn value_round_width(layout: SparkLayoutKind) -> usize {
    if layout == SparkLayoutKind::Joint {
        4
    } else {
        3
    }
}

fn validate_value_round_widths<EF>(
    proof: &SparkValueSumcheckProof<EF>,
    expected_width: usize,
) -> Result<(), SpartanWhirError> {
    if proof
        .rounds
        .iter()
        .any(|round| round.0.len() != expected_width)
    {
        return Err(SpartanWhirError::InvalidRoundPolynomial);
    }
    Ok(())
}

pub fn compute_spark_read_tables<EF>(
    tables: &SparkTables,
    r_x: &MultilinearPoint<EF>,
    r_y: &MultilinearPoint<EF>,
) -> Result<SparkReadTables<EF>, SpartanWhirError>
where
    EF: ExtensionField<F> + Send + Sync,
{
    validate_value_inputs(tables, r_x, r_y)?;
    let erow = {
        let _profile = profile_scope("spark_compute_read_table_row");
        spark_read_eq_table(tables.row_memory_size, &tables.rows, &r_x.0)?
    };
    let ecol = {
        let _profile = profile_scope("spark_compute_read_table_col");
        spark_read_eq_table(tables.col_memory_size, &tables.cols, &r_y.0)?
    };
    Ok(SparkReadTables { erow, ecol })
}

pub fn prove_spark_memory_products<EF, C>(
    tables: &SparkTables,
    r_x: &MultilinearPoint<EF>,
    r_y: &MultilinearPoint<EF>,
    challenger: &mut C,
) -> Result<SparkMemoryProductProof<EF>, SpartanWhirError>
where
    EF: ExtensionField<F>,
    C: FieldChallenger<F>,
{
    validate_value_inputs(tables, r_x, r_y)?;
    let read_tables = compute_spark_read_tables(tables, r_x, r_y)?;
    observe_spark_memory_context::<EF, C>(tables, challenger)?;
    let beta = challenger.sample_algebra_element::<EF>();
    let gamma = challenger.sample_algebra_element::<EF>();

    let row = spark_memory_product_claim(
        SparkMemoryAxis::Row,
        tables.row_memory_size,
        &tables.rows,
        &tables.read_ts_row,
        &tables.audit_ts_row,
        &r_x.0,
        &read_tables.erow,
        beta,
        gamma,
    )?;
    let col = spark_memory_product_claim(
        SparkMemoryAxis::Col,
        tables.col_memory_size,
        &tables.cols,
        &tables.read_ts_col,
        &tables.audit_ts_col,
        &r_y.0,
        &read_tables.ecol,
        beta,
        gamma,
    )?;

    let proof = SparkMemoryProductProof {
        beta,
        gamma,
        row,
        col,
    };
    check_spark_memory_product_equations(&proof)?;
    Ok(proof)
}

pub fn verify_spark_memory_products_with_tables<EF, C>(
    tables: &SparkTables,
    proof: &SparkMemoryProductProof<EF>,
    r_x: &MultilinearPoint<EF>,
    r_y: &MultilinearPoint<EF>,
    challenger: &mut C,
) -> Result<(), SpartanWhirError>
where
    EF: ExtensionField<F>,
    C: FieldChallenger<F>,
{
    validate_value_inputs(tables, r_x, r_y)?;
    observe_spark_memory_context::<EF, C>(tables, challenger)?;
    let beta = challenger.sample_algebra_element::<EF>();
    let gamma = challenger.sample_algebra_element::<EF>();
    if proof.beta != beta || proof.gamma != gamma {
        return Err(SpartanWhirError::TranscriptMismatch);
    }

    let read_tables = compute_spark_read_tables(tables, r_x, r_y)?;
    let expected_row = spark_memory_product_claim(
        SparkMemoryAxis::Row,
        tables.row_memory_size,
        &tables.rows,
        &tables.read_ts_row,
        &tables.audit_ts_row,
        &r_x.0,
        &read_tables.erow,
        beta,
        gamma,
    )?;
    let expected_col = spark_memory_product_claim(
        SparkMemoryAxis::Col,
        tables.col_memory_size,
        &tables.cols,
        &tables.read_ts_col,
        &tables.audit_ts_col,
        &r_y.0,
        &read_tables.ecol,
        beta,
        gamma,
    )?;

    if proof.row != expected_row || proof.col != expected_col {
        return Err(SpartanWhirError::SumcheckFailed);
    }
    check_spark_memory_product_equations(proof)
}

pub fn prove_spark_memory_grand_products<EF, C>(
    tables: &SparkTables,
    r_x: &MultilinearPoint<EF>,
    r_y: &MultilinearPoint<EF>,
    challenger: &mut C,
) -> Result<SparkMemoryGrandProductProof<EF>, SpartanWhirError>
where
    EF: ExtensionField<F>,
    C: FieldChallenger<F>,
{
    prove_spark_memory_grand_products_with_leaf_claims(tables, r_x, r_y, challenger)
        .map(|(proof, _)| proof)
}

pub fn prove_spark_batched_memory_products<EF, C>(
    tables: &SparkTables,
    r_x: &MultilinearPoint<EF>,
    r_y: &MultilinearPoint<EF>,
    challenger: &mut C,
) -> Result<SparkBatchedMemoryProductsProof<EF>, SpartanWhirError>
where
    EF: ExtensionField<F> + Send + Sync,
    C: FieldChallenger<F>,
{
    prove_spark_batched_memory_products_with_leaf_claims(tables, r_x, r_y, challenger)
        .map(|(proof, _)| proof)
}

pub fn prove_spark_batched_memory_products_with_leaf_claims<EF, C>(
    tables: &SparkTables,
    r_x: &MultilinearPoint<EF>,
    r_y: &MultilinearPoint<EF>,
    challenger: &mut C,
) -> Result<
    (
        SparkBatchedMemoryProductsProof<EF>,
        SparkBatchedMemoryProductsLeafClaims<EF>,
    ),
    SpartanWhirError,
>
where
    EF: ExtensionField<F> + Send + Sync,
    C: FieldChallenger<F>,
{
    validate_value_inputs(tables, r_x, r_y)?;
    let read_tables = compute_spark_read_tables(tables, r_x, r_y)?;
    prove_spark_batched_memory_products_with_read_tables_and_leaf_claims(
        tables,
        r_x,
        r_y,
        &read_tables,
        challenger,
    )
}

pub fn prove_spark_batched_memory_products_with_read_tables_and_leaf_claims<EF, C>(
    tables: &SparkTables,
    r_x: &MultilinearPoint<EF>,
    r_y: &MultilinearPoint<EF>,
    read_tables: &SparkReadTables<EF>,
    challenger: &mut C,
) -> Result<
    (
        SparkBatchedMemoryProductsProof<EF>,
        SparkBatchedMemoryProductsLeafClaims<EF>,
    ),
    SpartanWhirError,
>
where
    EF: ExtensionField<F> + Send + Sync,
    C: FieldChallenger<F>,
{
    validate_value_inputs(tables, r_x, r_y)?;
    validate_read_tables(tables, read_tables)?;
    observe_spark_memory_context::<EF, C>(tables, challenger)?;
    let beta = challenger.sample_algebra_element::<EF>();
    let gamma = challenger.sample_algebra_element::<EF>();

    // Build and consume the operation leaves before allocating the memory
    // leaves so both four-vector batches are not resident at the same time.
    let ops_terms = {
        let _profile = profile_scope("spark_ops_product_terms");
        spark_ops_product_terms_for_axes(tables, read_tables, beta, gamma)?
    };
    let value_dotproducts = {
        let _profile = profile_scope("spark_value_dotproducts");
        spark_base_value_dotproducts(tables, read_tables)?
    };

    let (proof_ops, ops_claims) = {
        let _profile = profile_scope("spark_prove_ops_products");
        prove_spark_batched_product_with_dotproducts(ops_terms, value_dotproducts, challenger)?
    };
    let matrix_evals = matrix_evals_from_split_dotproduct_claims(&proof_ops.dotproduct_claims)?;
    let mem_domain_size = tables.row_memory_size.max(tables.col_memory_size);
    let mem_terms = {
        let _profile = profile_scope("spark_mem_product_terms");
        spark_mem_product_terms_for_axes(tables, r_x, r_y, beta, gamma, mem_domain_size)?
    };
    let (proof_mem, mem_claims) = {
        let _profile = profile_scope("spark_prove_mem_products");
        prove_spark_batched_product_with_dotproducts(
            mem_terms,
            SparkProverDotProductBatch::empty(),
            challenger,
        )?
    };
    let products = SparkMemoryProductProof {
        beta,
        gamma,
        row: SparkMemoryProductClaim {
            init_root: proof_mem.product_roots[0],
            read_root: proof_ops.product_roots[0],
            write_root: proof_ops.product_roots[1],
            audit_root: proof_mem.product_roots[1],
        },
        col: SparkMemoryProductClaim {
            init_root: proof_mem.product_roots[2],
            read_root: proof_ops.product_roots[2],
            write_root: proof_ops.product_roots[3],
            audit_root: proof_mem.product_roots[3],
        },
    };
    check_spark_memory_product_equations(&products)?;

    Ok((
        SparkBatchedMemoryProductsProof {
            products,
            matrix_evals,
            proof_ops,
            proof_mem,
        },
        SparkBatchedMemoryProductsLeafClaims {
            beta,
            gamma,
            matrix_evals,
            ops: ops_claims,
            mem: mem_claims,
        },
    ))
}

pub fn prove_spark_memory_grand_products_with_leaf_claims<EF, C>(
    tables: &SparkTables,
    r_x: &MultilinearPoint<EF>,
    r_y: &MultilinearPoint<EF>,
    challenger: &mut C,
) -> Result<
    (
        SparkMemoryGrandProductProof<EF>,
        SparkMemoryGrandProductLeafClaims<EF>,
    ),
    SpartanWhirError,
>
where
    EF: ExtensionField<F> + Send + Sync,
    C: FieldChallenger<F>,
{
    validate_value_inputs(tables, r_x, r_y)?;
    let read_tables = compute_spark_read_tables(tables, r_x, r_y)?;
    observe_spark_memory_context::<EF, C>(tables, challenger)?;
    let beta = challenger.sample_algebra_element::<EF>();
    let gamma = challenger.sample_algebra_element::<EF>();

    let row_values = spark_memory_product_values(
        tables.row_memory_size,
        &tables.rows,
        &tables.read_ts_row,
        &tables.audit_ts_row,
        &r_x.0,
        &read_tables.erow,
        beta,
    )?;
    let col_values = spark_memory_product_values(
        tables.col_memory_size,
        &tables.cols,
        &tables.read_ts_col,
        &tables.audit_ts_col,
        &r_y.0,
        &read_tables.ecol,
        beta,
    )?;

    let (row, row_leaf_claims) =
        prove_axis_grand_products_with_leaf_claims(&row_values, gamma, challenger)?;
    let (col, col_leaf_claims) =
        prove_axis_grand_products_with_leaf_claims(&col_values, gamma, challenger)?;
    let products = SparkMemoryProductProof {
        beta,
        gamma,
        row: axis_product_claim(&row),
        col: axis_product_claim(&col),
    };
    let leaf_claims = SparkMemoryGrandProductLeafClaims {
        beta,
        gamma,
        row: row_leaf_claims,
        col: col_leaf_claims,
    };

    Ok((
        SparkMemoryGrandProductProof { products, row, col },
        leaf_claims,
    ))
}

pub fn verify_spark_memory_grand_products_with_tables<EF, C>(
    tables: &SparkTables,
    proof: &SparkMemoryGrandProductProof<EF>,
    r_x: &MultilinearPoint<EF>,
    r_y: &MultilinearPoint<EF>,
    challenger: &mut C,
) -> Result<(), SpartanWhirError>
where
    EF: ExtensionField<F> + Send + Sync,
    C: FieldChallenger<F>,
{
    validate_value_inputs(tables, r_x, r_y)?;
    let read_tables = compute_spark_read_tables(tables, r_x, r_y)?;
    let leaf_claims = verify_spark_memory_grand_product_claims(tables, proof, challenger)?;
    verify_spark_memory_leaf_claims_with_tables(tables, &read_tables, &leaf_claims, r_x, r_y)
}

pub fn verify_spark_batched_memory_products_with_tables<EF, C>(
    tables: &SparkTables,
    proof: &SparkBatchedMemoryProductsProof<EF>,
    r_x: &MultilinearPoint<EF>,
    r_y: &MultilinearPoint<EF>,
    challenger: &mut C,
) -> Result<(), SpartanWhirError>
where
    EF: ExtensionField<F> + Send + Sync,
    C: FieldChallenger<F>,
{
    validate_value_inputs(tables, r_x, r_y)?;
    let read_tables = compute_spark_read_tables(tables, r_x, r_y)?;
    let claims = verify_spark_batched_memory_product_claims(tables, proof, challenger)?;
    let (row_values, col_values) =
        spark_memory_values_for_axes(tables, r_x, r_y, &read_tables, claims.beta)?;
    let ops_terms = spark_ops_product_terms(&row_values, &col_values, claims.gamma)?;
    let value_dotproducts = spark_value_dotproducts(tables, &read_tables)?;
    let mem_terms = spark_mem_product_terms(
        &row_values,
        &col_values,
        claims.gamma,
        tables.row_memory_size.max(tables.col_memory_size),
    )?;
    verify_batched_product_leaf_claims_with_terms_and_dotproducts(
        &ops_terms,
        &value_dotproducts,
        &claims.ops,
    )?;
    verify_batched_product_leaf_claims_with_terms_and_dotproducts(&mem_terms, &[], &claims.mem)
}

pub fn verify_spark_batched_memory_leaf_claims_with_openings<EF>(
    row_memory_size: usize,
    col_memory_size: usize,
    claims: &SparkBatchedMemoryProductsLeafClaims<EF>,
    fixed: &SparkFixedTableOpeningEvals<EF>,
    read: &SparkReadTableOpeningEvals<EF>,
    r_x: &MultilinearPoint<EF>,
    r_y: &MultilinearPoint<EF>,
) -> Result<(), SpartanWhirError>
where
    EF: ExtensionField<F> + Send + Sync,
{
    let mem_domain_size = row_memory_size.max(col_memory_size);
    if !mem_domain_size.is_power_of_two()
        || claims.ops.product_evals.len() != 4
        || claims.mem.product_evals.len() != 4
        || claims.ops.dotproduct_left_evals.len() != 6
        || claims.ops.dotproduct_right_evals.len() != 6
        || claims.ops.dotproduct_weight_evals.len() != 6
        || !claims.mem.dotproduct_point.0.is_empty()
        || !claims.mem.dotproduct_left_evals.is_empty()
        || !claims.mem.dotproduct_right_evals.is_empty()
        || !claims.mem.dotproduct_weight_evals.is_empty()
    {
        return Err(SpartanWhirError::SumcheckFailed);
    }

    let expected_ops = [
        spark_memory_tuple_hash_eval(
            fixed.row_addr,
            read.erow_ops,
            fixed.row_read_ts,
            claims.beta,
        ) - claims.gamma,
        spark_memory_tuple_hash_eval(
            fixed.row_addr,
            read.erow_ops,
            fixed.row_read_ts + EF::ONE,
            claims.beta,
        ) - claims.gamma,
        spark_memory_tuple_hash_eval(
            fixed.col_addr,
            read.ecol_ops,
            fixed.col_read_ts,
            claims.beta,
        ) - claims.gamma,
        spark_memory_tuple_hash_eval(
            fixed.col_addr,
            read.ecol_ops,
            fixed.col_read_ts + EF::ONE,
            claims.beta,
        ) - claims.gamma,
    ];
    if claims.ops.product_evals.as_slice() != expected_ops {
        return Err(SpartanWhirError::SumcheckFailed);
    }

    let expected_left = [
        read.erow_low,
        read.erow_high,
        read.erow_low,
        read.erow_high,
        read.erow_low,
        read.erow_high,
    ];
    let expected_right = [
        read.ecol_low,
        read.ecol_high,
        read.ecol_low,
        read.ecol_high,
        read.ecol_low,
        read.ecol_high,
    ];
    let expected_weight = [
        fixed.val_a_low,
        fixed.val_a_high,
        fixed.val_b_low,
        fixed.val_b_high,
        fixed.val_c_low,
        fixed.val_c_high,
    ];
    if claims.ops.dotproduct_left_evals.as_slice() != expected_left
        || claims.ops.dotproduct_right_evals.as_slice() != expected_right
        || claims.ops.dotproduct_weight_evals.as_slice() != expected_weight
    {
        return Err(SpartanWhirError::SumcheckFailed);
    }

    let expected_mem = [
        padded_virtual_memory_term_eval(
            row_memory_size,
            mem_domain_size,
            &claims.mem.product_point,
            r_x,
            EF::ZERO,
            claims.beta,
            claims.gamma,
        )?,
        padded_virtual_memory_term_eval(
            row_memory_size,
            mem_domain_size,
            &claims.mem.product_point,
            r_x,
            fixed.row_audit_ts,
            claims.beta,
            claims.gamma,
        )?,
        padded_virtual_memory_term_eval(
            col_memory_size,
            mem_domain_size,
            &claims.mem.product_point,
            r_y,
            EF::ZERO,
            claims.beta,
            claims.gamma,
        )?,
        padded_virtual_memory_term_eval(
            col_memory_size,
            mem_domain_size,
            &claims.mem.product_point,
            r_y,
            fixed.col_audit_ts,
            claims.beta,
            claims.gamma,
        )?,
    ];
    if claims.mem.product_evals.as_slice() != expected_mem {
        return Err(SpartanWhirError::SumcheckFailed);
    }

    Ok(())
}

pub fn check_spark_memory_product_equations<EF>(
    proof: &SparkMemoryProductProof<EF>,
) -> Result<(), SpartanWhirError>
where
    EF: Field,
{
    check_spark_memory_product_claim(&proof.row)?;
    check_spark_memory_product_claim(&proof.col)
}

pub fn verify_spark_memory_leaf_claims_with_tables<EF>(
    tables: &SparkTables,
    read_tables: &SparkReadTables<EF>,
    claims: &SparkMemoryGrandProductLeafClaims<EF>,
    r_x: &MultilinearPoint<EF>,
    r_y: &MultilinearPoint<EF>,
) -> Result<(), SpartanWhirError>
where
    EF: ExtensionField<F> + Send + Sync,
{
    validate_value_inputs(tables, r_x, r_y)?;
    validate_read_tables(tables, read_tables)?;

    verify_axis_leaf_claims_with_tables(
        tables.row_memory_size,
        &tables.rows,
        &tables.read_ts_row,
        &tables.audit_ts_row,
        &read_tables.erow,
        &claims.row,
        r_x,
        claims.beta,
        claims.gamma,
    )?;
    verify_axis_leaf_claims_with_tables(
        tables.col_memory_size,
        &tables.cols,
        &tables.read_ts_col,
        &tables.audit_ts_col,
        &read_tables.ecol,
        &claims.col,
        r_y,
        claims.beta,
        claims.gamma,
    )
}

pub fn verify_spark_memory_grand_product_claims<EF, C>(
    tables: &SparkTables,
    proof: &SparkMemoryGrandProductProof<EF>,
    challenger: &mut C,
) -> Result<SparkMemoryGrandProductLeafClaims<EF>, SpartanWhirError>
where
    EF: ExtensionField<F> + Send + Sync,
    C: FieldChallenger<F>,
{
    validate_table_metadata(tables)?;
    observe_spark_memory_context::<EF, C>(tables, challenger)?;
    let beta = challenger.sample_algebra_element::<EF>();
    let gamma = challenger.sample_algebra_element::<EF>();
    if proof.products.beta != beta || proof.products.gamma != gamma {
        return Err(SpartanWhirError::TranscriptMismatch);
    }

    let row_leaf_claims = verify_axis_grand_product_claims(
        &proof.products.row,
        &proof.row,
        tables.value_domain_size,
        tables.row_memory_size,
        challenger,
    )?;
    let col_leaf_claims = verify_axis_grand_product_claims(
        &proof.products.col,
        &proof.col,
        tables.value_domain_size,
        tables.col_memory_size,
        challenger,
    )?;

    let expected_products = SparkMemoryProductProof {
        beta,
        gamma,
        row: axis_product_claim(&proof.row),
        col: axis_product_claim(&proof.col),
    };
    if proof.products != expected_products {
        return Err(SpartanWhirError::SumcheckFailed);
    }
    check_spark_memory_product_equations(&proof.products)?;

    Ok(SparkMemoryGrandProductLeafClaims {
        beta,
        gamma,
        row: row_leaf_claims,
        col: col_leaf_claims,
    })
}

pub fn verify_spark_batched_memory_product_claims<EF, C>(
    tables: &SparkTables,
    proof: &SparkBatchedMemoryProductsProof<EF>,
    challenger: &mut C,
) -> Result<SparkBatchedMemoryProductsLeafClaims<EF>, SpartanWhirError>
where
    EF: ExtensionField<F>,
    C: FieldChallenger<F>,
{
    verify_spark_batched_memory_product_claims_with_metadata(&tables.metadata(), proof, challenger)
}

pub fn verify_spark_batched_memory_product_claims_with_metadata<EF, C>(
    metadata: &SparkTableMetadata,
    proof: &SparkBatchedMemoryProductsProof<EF>,
    challenger: &mut C,
) -> Result<SparkBatchedMemoryProductsLeafClaims<EF>, SpartanWhirError>
where
    EF: ExtensionField<F>,
    C: FieldChallenger<F>,
{
    metadata.validate()?;
    observe_spark_memory_metadata::<EF, C>(metadata, challenger)?;
    let beta = challenger.sample_algebra_element::<EF>();
    let gamma = challenger.sample_algebra_element::<EF>();
    if proof.products.beta != beta || proof.products.gamma != gamma {
        return Err(SpartanWhirError::TranscriptMismatch);
    }

    let ops_roots = [
        proof.products.row.read_root,
        proof.products.row.write_root,
        proof.products.col.read_root,
        proof.products.col.write_root,
    ];
    let mem_roots = [
        proof.products.row.init_root,
        proof.products.row.audit_root,
        proof.products.col.init_root,
        proof.products.col.audit_root,
    ];
    let matrix_evals =
        matrix_evals_from_split_dotproduct_claims(&proof.proof_ops.dotproduct_claims)?;
    if proof.matrix_evals != matrix_evals {
        return Err(SpartanWhirError::SumcheckFailed);
    }
    let ops_claims = verify_spark_batched_product(
        &proof.proof_ops,
        &ops_roots,
        &proof.proof_ops.dotproduct_claims,
        metadata.value_domain_size,
        challenger,
    )?;
    let mem_claims = verify_spark_batched_product(
        &proof.proof_mem,
        &mem_roots,
        &[],
        metadata.row_memory_size.max(metadata.col_memory_size),
        challenger,
    )?;

    check_spark_memory_product_equations(&proof.products)?;
    Ok(SparkBatchedMemoryProductsLeafClaims {
        beta,
        gamma,
        matrix_evals,
        ops: ops_claims,
        mem: mem_claims,
    })
}

/// Encode product rounds without changing their mathematical proof or transcript.
///
/// `challenger` must be a copy of the transcript immediately before the memory
/// metadata is observed by the ordinary SPARK product prover. This performs
/// scalar transcript replay; it does not rebuild product trees or table values.
pub fn compress_spark_memory_products<EF, C>(
    proof: &SparkBatchedMemoryProductsProof<EF>,
    metadata: &SparkTableMetadata,
    challenger: &mut C,
) -> Result<CompactSparkBatchedMemoryProductsProof<EF>, SpartanWhirError>
where
    EF: ExtensionField<F>,
    C: FieldChallenger<F>,
{
    replay_compact_spark_memory_context(metadata, Some(&proof.products), challenger)?;
    validate_compact_spark_roots(
        &proof.products,
        &proof.matrix_evals,
        &proof.proof_ops.product_roots,
        &proof.proof_mem.product_roots,
        &proof.proof_ops.dotproduct_claims,
    )?;
    let proof_ops =
        compress_spark_product(&proof.proof_ops, metadata.value_domain_size, 6, challenger)?;
    let proof_mem = compress_spark_product(
        &proof.proof_mem,
        metadata.row_memory_size.max(metadata.col_memory_size),
        0,
        challenger,
    )?;
    Ok(CompactSparkBatchedMemoryProductsProof {
        header: CompactSparkProductHeader::Explicit {
            products: proof.products.clone(),
            matrix_evals: proof.matrix_evals,
        },
        proof_ops,
        proof_mem,
    })
}

/// Omit redundant header values and repeated row/column read evaluations.
///
/// This refines an already compressed proof without replaying its transcript.
/// The decoder derives the header from the original transcript and tree roots,
/// and expands the two read evaluations back to the six original observations.
pub fn refine_spark_memory_products<EF: Field>(
    proof: &mut CompactSparkBatchedMemoryProductsProof<EF>,
) -> Result<(), SpartanWhirError> {
    let CompactSparkProductHeader::Explicit {
        products,
        matrix_evals,
    } = &proof.header
    else {
        validate_refined_spark_leaf(proof)?;
        return Ok(());
    };
    validate_compact_spark_roots(
        products,
        matrix_evals,
        &proof.proof_ops.product_roots,
        &proof.proof_mem.product_roots,
        &proof.proof_ops.dotproduct_claims,
    )?;
    let leaf = proof
        .proof_ops
        .layers
        .last_mut()
        .ok_or(SpartanWhirError::InvalidRoundCount)?;
    for values in [&leaf.dotproduct_left_evals, &leaf.dotproduct_right_evals] {
        if values.len() != 6 || values[2..4] != values[..2] || values[4..6] != values[..2] {
            return Err(SpartanWhirError::InvalidProofShape);
        }
    }
    if !matches!(leaf.rounds, CompactSparkProductRounds::Cubic(_))
        || leaf.dotproduct_weight_evals.len() != 6
    {
        return Err(SpartanWhirError::InvalidProofShape);
    }
    leaf.dotproduct_left_evals.truncate(2);
    leaf.dotproduct_right_evals.truncate(2);
    proof.header = CompactSparkProductHeader::Derived;
    Ok(())
}

fn validate_refined_spark_leaf<EF>(
    proof: &CompactSparkBatchedMemoryProductsProof<EF>,
) -> Result<(), SpartanWhirError> {
    let leaf = proof
        .proof_ops
        .layers
        .last()
        .ok_or(SpartanWhirError::InvalidRoundCount)?;
    if proof.proof_ops.product_roots.len() != 4
        || proof.proof_mem.product_roots.len() != 4
        || proof.proof_ops.dotproduct_claims.len() != 6
        || !matches!(leaf.rounds, CompactSparkProductRounds::Cubic(_))
        || leaf.dotproduct_left_evals.len() != 2
        || leaf.dotproduct_right_evals.len() != 2
        || leaf.dotproduct_weight_evals.len() != 6
    {
        return Err(SpartanWhirError::InvalidProofShape);
    }
    Ok(())
}

impl<EF> CompactSparkBatchedMemoryProductsProof<EF> {
    pub fn omits_derived_fields(&self) -> bool {
        matches!(self.header, CompactSparkProductHeader::Derived)
    }
}

/// Restore the ordinary proof and replay its original transcript observations.
///
/// The input transcript position matches [`compress_spark_memory_products`].
/// The returned proof still requires the ordinary SPARK verifier; decoding is
/// not a substitute for checking the product identities or committed openings.
pub fn decompress_spark_memory_products<EF, C>(
    proof: &CompactSparkBatchedMemoryProductsProof<EF>,
    metadata: &SparkTableMetadata,
    challenger: &mut C,
) -> Result<SparkBatchedMemoryProductsProof<EF>, SpartanWhirError>
where
    EF: ExtensionField<F>,
    C: FieldChallenger<F>,
{
    let explicit_products = match &proof.header {
        CompactSparkProductHeader::Explicit { products, .. } => Some(products),
        CompactSparkProductHeader::Derived => None,
    };
    let (beta, gamma) =
        replay_compact_spark_memory_context(metadata, explicit_products, challenger)?;
    let (products, matrix_evals) = match &proof.header {
        CompactSparkProductHeader::Explicit {
            products,
            matrix_evals,
        } => (products.clone(), *matrix_evals),
        CompactSparkProductHeader::Derived => {
            validate_refined_spark_leaf(proof)?;
            let ops = &proof.proof_ops.product_roots;
            let mem = &proof.proof_mem.product_roots;
            (
                SparkMemoryProductProof {
                    beta,
                    gamma,
                    row: SparkMemoryProductClaim {
                        init_root: mem[0],
                        read_root: ops[0],
                        write_root: ops[1],
                        audit_root: mem[1],
                    },
                    col: SparkMemoryProductClaim {
                        init_root: mem[2],
                        read_root: ops[2],
                        write_root: ops[3],
                        audit_root: mem[3],
                    },
                },
                matrix_evals_from_split_dotproduct_claims(&proof.proof_ops.dotproduct_claims)?,
            )
        }
    };
    validate_compact_spark_roots(
        &products,
        &matrix_evals,
        &proof.proof_ops.product_roots,
        &proof.proof_mem.product_roots,
        &proof.proof_ops.dotproduct_claims,
    )?;
    let proof_ops = decompress_spark_product(
        &proof.proof_ops,
        metadata.value_domain_size,
        6,
        proof.omits_derived_fields(),
        challenger,
    )?;
    let proof_mem = decompress_spark_product(
        &proof.proof_mem,
        metadata.row_memory_size.max(metadata.col_memory_size),
        0,
        false,
        challenger,
    )?;
    Ok(SparkBatchedMemoryProductsProof {
        products,
        matrix_evals,
        proof_ops,
        proof_mem,
    })
}

fn replay_compact_spark_memory_context<EF, C>(
    metadata: &SparkTableMetadata,
    products: Option<&SparkMemoryProductProof<EF>>,
    challenger: &mut C,
) -> Result<(EF, EF), SpartanWhirError>
where
    EF: ExtensionField<F>,
    C: FieldChallenger<F>,
{
    metadata.validate()?;
    observe_spark_memory_metadata::<EF, C>(metadata, challenger)?;
    let beta = challenger.sample_algebra_element::<EF>();
    let gamma = challenger.sample_algebra_element::<EF>();
    if products.is_some_and(|products| products.beta != beta || products.gamma != gamma) {
        return Err(SpartanWhirError::TranscriptMismatch);
    }
    Ok((beta, gamma))
}

fn validate_compact_spark_roots<EF: Field>(
    products: &SparkMemoryProductProof<EF>,
    matrix_evals: &[EF; 3],
    ops_roots: &[EF],
    mem_roots: &[EF],
    dotproduct_claims: &[EF],
) -> Result<(), SpartanWhirError> {
    if ops_roots
        != [
            products.row.read_root,
            products.row.write_root,
            products.col.read_root,
            products.col.write_root,
        ]
        || mem_roots
            != [
                products.row.init_root,
                products.row.audit_root,
                products.col.init_root,
                products.col.audit_root,
            ]
        || *matrix_evals != matrix_evals_from_split_dotproduct_claims(dotproduct_claims)?
    {
        return Err(SpartanWhirError::SumcheckFailed);
    }
    Ok(())
}

fn compact_spark_layer_claim<EF, C>(
    claims: &mut Vec<EF>,
    leaf_dotproducts: &[EF],
    challenger: &mut C,
) -> EF
where
    EF: ExtensionField<F>,
    C: FieldChallenger<F>,
{
    claims.extend_from_slice(leaf_dotproducts);
    let coeffs = sample_batched_product_coefficients(claims, challenger);
    let claim = claims
        .iter()
        .zip(coeffs)
        .fold(EF::ZERO, |sum, (&value, coeff)| sum + value * coeff);
    observe_sumcheck_claim::<F, EF, C>(challenger, claim);
    claim
}

fn compact_spark_next_layer<EF, C>(
    layer: &SparkBatchedProductLayerProof<EF>,
    alpha: Vec<EF>,
    challenger: &mut C,
) -> (Vec<EF>, Vec<EF>)
where
    EF: ExtensionField<F>,
    C: FieldChallenger<F>,
{
    let combine = observe_batched_product_layer_and_sample(layer, challenger);
    let claims = layer
        .product_left_evals
        .iter()
        .zip(&layer.product_right_evals)
        .map(|(&left, &right)| extrapolate(left, right, combine))
        .collect();
    let mut point = alpha;
    point.push(combine);
    (claims, point)
}

fn validate_compact_spark_layer<EF>(
    layer: &SparkBatchedProductLayerProof<EF>,
    round_count: usize,
    dotproduct_count: usize,
) -> Result<(), SpartanWhirError> {
    if layer.rounds.len() != round_count
        || layer.product_left_evals.len() != 4
        || layer.product_right_evals.len() != 4
        || layer.dotproduct_left_evals.len() != dotproduct_count
        || layer.dotproduct_right_evals.len() != dotproduct_count
        || layer.dotproduct_weight_evals.len() != dotproduct_count
    {
        return Err(SpartanWhirError::InvalidRoundCount);
    }
    Ok(())
}

fn compress_spark_product<EF, C>(
    proof: &SparkBatchedProductProof<EF>,
    domain_size: usize,
    dotproduct_count: usize,
    challenger: &mut C,
) -> Result<CompactSparkBatchedProductProof<EF>, SpartanWhirError>
where
    EF: ExtensionField<F>,
    C: FieldChallenger<F>,
{
    let layer_count = log2_power_of_two(domain_size);
    if proof.layers.len() != layer_count
        || proof.product_roots.len() != 4
        || proof.dotproduct_claims.len() != dotproduct_count
    {
        return Err(SpartanWhirError::InvalidRoundCount);
    }
    let mut claims = proof.product_roots.clone();
    let mut parent_point = Vec::new();
    let mut layers = Vec::with_capacity(layer_count);
    for (index, layer) in proof.layers.iter().enumerate() {
        let dotproducts = if index + 1 == layer_count {
            proof.dotproduct_claims.as_slice()
        } else {
            &[]
        };
        validate_compact_spark_layer(layer, index, dotproducts.len())?;
        let mut claim = compact_spark_layer_claim(&mut claims, dotproducts, challenger);
        let mut alpha = Vec::with_capacity(index);
        let mut factored = Vec::with_capacity(if dotproducts.is_empty() { index } else { 0 });
        let inverse_slopes = if dotproducts.is_empty() {
            let slopes = parent_point
                .iter()
                .map(|coordinate: &EF| {
                    let slope = coordinate.double() - EF::ONE;
                    if slope == EF::ZERO {
                        EF::ONE
                    } else {
                        slope
                    }
                })
                .collect::<Vec<_>>();
            let mut inverses = vec![EF::ZERO; slopes.len()];
            p3_field::batch_multiplicative_inverse_general(&slopes, &mut inverses, |x| x.inverse());
            inverses
        } else {
            Vec::new()
        };
        for (round_index, round) in layer.rounds.iter().enumerate() {
            if dotproducts.is_empty() {
                factored.push(compress_spark_product_round(
                    round,
                    claim,
                    parent_point[round_index],
                    inverse_slopes[round_index],
                )?);
            }
            challenger.observe_algebra_slice(&round.0);
            let challenge = challenger.sample_algebra_element::<EF>();
            claim = evaluate_spark_cubic_for_codec(round, claim, challenge);
            alpha.push(challenge);
        }
        let rounds = if dotproducts.is_empty() {
            CompactSparkProductRounds::Factored(factored)
        } else {
            CompactSparkProductRounds::Cubic(layer.rounds.clone())
        };
        layers.push(CompactSparkProductLayerProof {
            rounds,
            product_left_evals: layer.product_left_evals.clone(),
            product_right_evals: layer.product_right_evals.clone(),
            dotproduct_left_evals: layer.dotproduct_left_evals.clone(),
            dotproduct_right_evals: layer.dotproduct_right_evals.clone(),
            dotproduct_weight_evals: layer.dotproduct_weight_evals.clone(),
        });
        (claims, parent_point) = compact_spark_next_layer(layer, alpha, challenger);
    }
    Ok(CompactSparkBatchedProductProof {
        product_roots: proof.product_roots.clone(),
        dotproduct_claims: proof.dotproduct_claims.clone(),
        layers,
    })
}

fn decompress_spark_product<EF, C>(
    proof: &CompactSparkBatchedProductProof<EF>,
    domain_size: usize,
    dotproduct_count: usize,
    expand_read_pairs: bool,
    challenger: &mut C,
) -> Result<SparkBatchedProductProof<EF>, SpartanWhirError>
where
    EF: ExtensionField<F>,
    C: FieldChallenger<F>,
{
    let layer_count = log2_power_of_two(domain_size);
    if proof.layers.len() != layer_count
        || proof.product_roots.len() != 4
        || proof.dotproduct_claims.len() != dotproduct_count
    {
        return Err(SpartanWhirError::InvalidRoundCount);
    }
    let mut claims = proof.product_roots.clone();
    let mut parent_point = Vec::new();
    let mut layers = Vec::with_capacity(layer_count);
    for (index, compact) in proof.layers.iter().enumerate() {
        let dotproducts = if index + 1 == layer_count {
            proof.dotproduct_claims.as_slice()
        } else {
            &[]
        };
        let round_count = match &compact.rounds {
            CompactSparkProductRounds::Factored(rounds) if dotproducts.is_empty() => rounds.len(),
            CompactSparkProductRounds::Cubic(rounds) if !dotproducts.is_empty() => rounds.len(),
            _ => return Err(SpartanWhirError::InvalidRoundPolynomial),
        };
        if round_count != index {
            return Err(SpartanWhirError::InvalidRoundCount);
        }
        let mut claim = compact_spark_layer_claim(&mut claims, dotproducts, challenger);
        let mut alpha = Vec::with_capacity(index);
        let mut rounds = Vec::with_capacity(index);
        for round_index in 0..index {
            let round = match &compact.rounds {
                CompactSparkProductRounds::Factored(values) => decompress_spark_product_round(
                    values[round_index],
                    claim,
                    parent_point[round_index],
                ),
                CompactSparkProductRounds::Cubic(values) => values[round_index].clone(),
            };
            challenger.observe_algebra_slice(&round.0);
            let challenge = challenger.sample_algebra_element::<EF>();
            claim = evaluate_spark_cubic_for_codec(&round, claim, challenge);
            alpha.push(challenge);
            rounds.push(round);
        }
        let expand = |values: &[EF]| -> Result<Vec<EF>, SpartanWhirError> {
            if expand_read_pairs && !dotproducts.is_empty() {
                let [low, high] = values else {
                    return Err(SpartanWhirError::InvalidProofShape);
                };
                Ok(vec![*low, *high, *low, *high, *low, *high])
            } else {
                Ok(values.to_vec())
            }
        };
        let layer = SparkBatchedProductLayerProof {
            rounds,
            product_left_evals: compact.product_left_evals.clone(),
            product_right_evals: compact.product_right_evals.clone(),
            dotproduct_left_evals: expand(&compact.dotproduct_left_evals)?,
            dotproduct_right_evals: expand(&compact.dotproduct_right_evals)?,
            dotproduct_weight_evals: compact.dotproduct_weight_evals.clone(),
        };
        validate_compact_spark_layer(&layer, index, dotproducts.len())?;
        (claims, parent_point) = compact_spark_next_layer(&layer, alpha, challenger);
        layers.push(layer);
    }
    Ok(SparkBatchedProductProof {
        product_roots: proof.product_roots.clone(),
        dotproduct_claims: proof.dotproduct_claims.clone(),
        layers,
    })
}

fn spark_cubic_differences<EF: Field>(round: &CubicRoundPoly<EF>, claim: EF) -> [EF; 3] {
    let [h0, h2, h3] = round.0;
    let h1 = claim - h0;
    let first = h1 - h0;
    let second = h2 - h1.double() + h0;
    let third = h3 - h2 * EF::from_u32(3) + h1 * EF::from_u32(3) - h0;
    [first, second, third]
}

fn evaluate_spark_cubic_for_codec<EF: ExtensionField<F>>(
    round: &CubicRoundPoly<EF>,
    claim: EF,
    challenge: EF,
) -> EF {
    let [first, second, third] = spark_cubic_differences(round, claim);
    let half = F::TWO.inverse();
    let sixth = F::from_u32(6).inverse();
    round.0[0]
        + challenge
            * (first
                + (challenge - EF::ONE) * (second * half + (challenge - EF::TWO) * third * sixth))
}

fn compress_spark_product_round<EF: ExtensionField<F>>(
    round: &CubicRoundPoly<EF>,
    claim: EF,
    equality_coordinate: EF,
    inverse_slope: EF,
) -> Result<[EF; 2], SpartanWhirError> {
    let [first, second, third] = spark_cubic_differences(round, claim);
    let slope = equality_coordinate.double() - EF::ONE;
    let values = if slope == EF::ZERO {
        // E(t) = 1/2, so Q(t) = 2g(t). Reconstruction below also rejects
        // an input cubic with a nonzero third difference.
        [first.double() - second, second]
    } else {
        let q2 = third * F::from_u32(6).inverse() * inverse_slope;
        let quadratic = (second - third) * F::TWO.inverse();
        let q1 = (quadratic - (EF::ONE - equality_coordinate) * q2) * inverse_slope;
        [q1, q2]
    };
    if decompress_spark_product_round(values, claim, equality_coordinate) != *round {
        return Err(SpartanWhirError::InvalidRoundPolynomial);
    }
    Ok(values)
}

fn decompress_spark_product_round<EF: Field>(
    [q1, q2]: [EF; 2],
    claim: EF,
    equality_coordinate: EF,
) -> CubicRoundPoly<EF> {
    let q0 = claim - equality_coordinate * (q1 + q2);
    let constant = EF::ONE - equality_coordinate;
    let slope = equality_coordinate.double() - EF::ONE;
    let evaluate = |t| (constant + slope * t) * (q0 + t * (q1 + t * q2));
    CubicRoundPoly([constant * q0, evaluate(EF::TWO), evaluate(EF::from_u32(3))])
}

pub fn prove_spark_grand_product<EF, C>(
    values: &[EF],
    gamma: EF,
    challenger: &mut C,
) -> Result<SparkGrandProductProof<EF>, SpartanWhirError>
where
    EF: ExtensionField<F> + Send + Sync,
    C: FieldChallenger<F>,
{
    if values.is_empty() || !values.len().is_power_of_two() {
        return Err(SpartanWhirError::InvalidPolynomialLength);
    }
    let terms: Vec<EF> = values.iter().map(|&value| value - gamma).collect();
    prove_spark_grand_product_terms(&terms, challenger)
}

pub fn prove_spark_grand_product_terms<EF, C>(
    terms: &[EF],
    challenger: &mut C,
) -> Result<SparkGrandProductProof<EF>, SpartanWhirError>
where
    EF: ExtensionField<F> + Send + Sync,
    C: FieldChallenger<F>,
{
    prove_spark_grand_product_terms_with_leaf_claim(terms, challenger).map(|(proof, _)| proof)
}

fn prove_spark_grand_product_terms_with_leaf_claim<EF, C>(
    terms: &[EF],
    challenger: &mut C,
) -> Result<(SparkGrandProductProof<EF>, SparkGrandProductLeafClaim<EF>), SpartanWhirError>
where
    EF: ExtensionField<F> + Send + Sync,
    C: FieldChallenger<F>,
{
    if terms.is_empty() || !terms.len().is_power_of_two() {
        return Err(SpartanWhirError::InvalidPolynomialLength);
    }

    let tree = product_tree_from_terms(terms)?;
    let root = tree
        .last()
        .and_then(|layer| layer.first())
        .copied()
        .ok_or(SpartanWhirError::InvalidPolynomialLength)?;
    let mut claim = root;
    let mut parent_point = Vec::new();
    let mut proof_layers = Vec::with_capacity(log2_power_of_two(terms.len()));

    for child_layer_index in (0..tree.len() - 1).rev() {
        let (layer, alpha) = prove_spark_grand_product_layer(
            &tree[child_layer_index],
            &parent_point,
            claim,
            challenger,
        )?;
        let combine = observe_product_layer_and_sample(&layer, challenger);
        let mut child_point = alpha.0;
        child_point.push(combine);
        claim = extrapolate(layer.left_eval, layer.right_eval, combine);
        proof_layers.push(layer);
        parent_point = child_point;
    }

    Ok((
        SparkGrandProductProof {
            root,
            layers: proof_layers,
            leaf_eval: claim,
        },
        SparkGrandProductLeafClaim {
            point: MultilinearPoint(parent_point),
            term_eval: claim,
        },
    ))
}

pub fn verify_spark_grand_product<EF, C>(
    proof: &SparkGrandProductProof<EF>,
    expected_root: EF,
    expected_leaf_rounds: usize,
    challenger: &mut C,
) -> Result<(MultilinearPoint<EF>, EF), SpartanWhirError>
where
    EF: ExtensionField<F>,
    C: FieldChallenger<F>,
{
    if proof.root != expected_root || proof.layers.len() != expected_leaf_rounds {
        return Err(SpartanWhirError::SumcheckFailed);
    }

    let mut claim = expected_root;
    let mut parent_point = Vec::new();
    for layer in &proof.layers {
        if layer.rounds.len() != parent_point.len() {
            return Err(SpartanWhirError::InvalidRoundCount);
        }
        let (alpha, reduced_claim) =
            verify_spark_grand_product_layer(layer, &parent_point, claim, challenger)?;
        let eq_eval = eq_eval_at_point(&parent_point, &alpha.0)?;
        if reduced_claim != eq_eval * layer.left_eval * layer.right_eval {
            return Err(SpartanWhirError::SumcheckFailed);
        }

        let combine = observe_product_layer_and_sample(layer, challenger);
        parent_point = alpha.0;
        parent_point.push(combine);
        claim = extrapolate(layer.left_eval, layer.right_eval, combine);
    }

    if proof.leaf_eval != claim {
        return Err(SpartanWhirError::SumcheckFailed);
    }
    Ok((MultilinearPoint(parent_point), claim))
}

pub fn verify_spark_grand_product_with_values<EF, C>(
    values: &[EF],
    gamma: EF,
    proof: &SparkGrandProductProof<EF>,
    challenger: &mut C,
) -> Result<(MultilinearPoint<EF>, EF), SpartanWhirError>
where
    EF: ExtensionField<F>,
    C: FieldChallenger<F>,
{
    if values.is_empty() || !values.len().is_power_of_two() {
        return Err(SpartanWhirError::InvalidPolynomialLength);
    }
    let terms: Vec<EF> = values.iter().map(|&value| value - gamma).collect();
    let expected_root = terms.iter().fold(EF::ONE, |acc, &term| acc * term);
    let (point, eval) = verify_spark_grand_product(
        proof,
        expected_root,
        log2_power_of_two(terms.len()),
        challenger,
    )?;
    let expected_leaf = evaluate_mle_table(&terms, &point.0)?;
    if eval != expected_leaf {
        return Err(SpartanWhirError::SumcheckFailed);
    }
    Ok((point, eval))
}

/// Prove a batch of product-tree claims, optionally with leaf-domain
/// dotproduct claims.
///
/// `product_terms` must contain one or more power-of-two vectors of the same
/// length `N`. Each dotproduct vector has length `N / 2` and is combined only
/// in the leaf-adjacent layer. The returned leaf claims separate the product
/// point (`log2(N)` coordinates) from the dotproduct point (`log2(N) - 1`
/// coordinates). The Fiat-Shamir order is: observe current product claims,
/// append dotproduct claims only at the leaf-adjacent layer, sample batching
/// coefficients, then run the layer sumcheck.
pub fn prove_spark_batched_product<EF, C>(
    product_terms: &[Vec<EF>],
    dotproducts: &[SparkDotProductCircuit<EF>],
    challenger: &mut C,
) -> Result<
    (
        SparkBatchedProductProof<EF>,
        SparkBatchedProductLeafClaims<EF>,
    ),
    SpartanWhirError,
>
where
    EF: ExtensionField<F> + Send + Sync,
    C: FieldChallenger<F>,
{
    prove_spark_batched_product_owned(
        product_terms.to_vec(),
        dotproducts.iter().cloned().map(Into::into).collect(),
        challenger,
    )
}

fn prove_spark_batched_product_owned<EF, C>(
    product_terms: Vec<Vec<EF>>,
    dotproducts: Vec<SparkProverDotProductCircuit<EF>>,
    challenger: &mut C,
) -> Result<
    (
        SparkBatchedProductProof<EF>,
        SparkBatchedProductLeafClaims<EF>,
    ),
    SpartanWhirError,
>
where
    EF: ExtensionField<F> + Send + Sync,
    C: FieldChallenger<F>,
{
    prove_spark_batched_product_with_dotproducts(product_terms, dotproducts.into(), challenger)
}

fn prove_spark_batched_product_with_dotproducts<EF, C>(
    product_terms: Vec<Vec<EF>>,
    dotproducts: SparkProverDotProductBatch<EF>,
    challenger: &mut C,
) -> Result<
    (
        SparkBatchedProductProof<EF>,
        SparkBatchedProductLeafClaims<EF>,
    ),
    SpartanWhirError,
>
where
    EF: ExtensionField<F> + Send + Sync,
    C: FieldChallenger<F>,
{
    prove_spark_batched_product_impl(product_terms, dotproducts, true, challenger)
}

fn prove_spark_batched_product_impl<EF, C>(
    product_terms: Vec<Vec<EF>>,
    mut dotproducts: SparkProverDotProductBatch<EF>,
    bind_product_children_directly: bool,
    challenger: &mut C,
) -> Result<
    (
        SparkBatchedProductProof<EF>,
        SparkBatchedProductLeafClaims<EF>,
    ),
    SpartanWhirError,
>
where
    EF: ExtensionField<F> + Send + Sync,
    C: FieldChallenger<F>,
{
    if product_terms.is_empty() {
        return Err(SpartanWhirError::InvalidPolynomialLength);
    }
    let domain_size = product_terms[0].len();
    if domain_size == 0 || !domain_size.is_power_of_two() {
        return Err(SpartanWhirError::InvalidPolynomialLength);
    }
    if product_terms.iter().any(|terms| terms.len() != domain_size) {
        return Err(SpartanWhirError::InvalidPolynomialLength);
    }
    validate_dotproducts_for_batched_product(&dotproducts, domain_size)?;

    let trees = {
        let _profile = profile_scope("spark_batched_product_trees");
        product_terms
            .into_iter()
            .map(product_tree_from_terms_owned)
            .collect::<Result<Vec<_>, _>>()?
    };
    let product_roots = {
        let _profile = profile_scope("spark_batched_product_roots");
        trees
            .iter()
            .map(|tree| {
                tree.last()
                    .and_then(|layer| layer.first())
                    .copied()
                    .ok_or(SpartanWhirError::InvalidPolynomialLength)
            })
            .collect::<Result<Vec<_>, _>>()?
    };
    let dotproduct_claims = {
        let _profile = profile_scope("spark_batched_dotproduct_claims");
        evaluate_dotproducts(&dotproducts)?
    };

    let mut claims_to_verify = product_roots.clone();
    let mut parent_point = Vec::new();
    let mut proof_layers = Vec::with_capacity(log2_power_of_two(domain_size));
    let mut leaf_claims = SparkBatchedProductLeafClaims {
        product_point: MultilinearPoint(Vec::new()),
        dotproduct_point: MultilinearPoint(Vec::new()),
        product_evals: Vec::new(),
        dotproduct_left_evals: Vec::new(),
        dotproduct_right_evals: Vec::new(),
        dotproduct_weight_evals: Vec::new(),
    };
    let _profile = profile_scope("spark_batched_product_layers");
    for child_layer_index in (0..trees[0].len() - 1).rev() {
        let include_dotproducts = child_layer_index == 0 && !dotproducts.is_empty();
        if include_dotproducts {
            claims_to_verify.extend_from_slice(&dotproduct_claims);
        }
        let coeffs = sample_batched_product_coefficients(&claims_to_verify, challenger);
        let mut claim = claims_to_verify
            .iter()
            .zip(&coeffs)
            .fold(EF::ZERO, |acc, (&claim, &coeff)| acc + claim * coeff);

        let child_layers = trees
            .iter()
            .map(|tree| tree[child_layer_index].as_slice())
            .collect::<Vec<_>>();
        // The tree stores each left/right pair adjacently. Compute the first
        // round in that layout, then materialize only the already-bound tables.
        let defer_product_split = bind_product_children_directly && !parent_point.is_empty();
        let (mut product_lefts, mut product_rights) = if defer_product_split {
            (Vec::new(), Vec::new())
        } else {
            let _profile = profile_detail_scope("spark_batched_product_split_layer");
            split_product_child_layers(&child_layers)?
        };
        let (
            mut dotproduct_lefts,
            mut dotproduct_rights,
            mut dotproduct_weights,
            mut dotproduct_base_weights,
            dotproduct_pair_indices,
        ) = if include_dotproducts {
            let dotproducts =
                core::mem::replace(&mut dotproducts, SparkProverDotProductBatch::empty());
            let mut dotproduct_weights = Vec::with_capacity(dotproducts.weights.len());
            let mut dotproduct_base_weights = Vec::with_capacity(dotproducts.weights.len());
            for weight in dotproducts.weights {
                match weight {
                    SparkProverDotProductWeight::Base(weight) => {
                        dotproduct_base_weights.push(weight)
                    }
                    SparkProverDotProductWeight::Extension(weight) => {
                        dotproduct_weights.push(weight)
                    }
                }
            }
            (
                dotproducts.lefts,
                dotproducts.rights,
                dotproduct_weights,
                dotproduct_base_weights,
                dotproducts.pair_indices,
            )
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };
        let mut eq = {
            let _profile = profile_detail_scope("spark_batched_product_eq_table");
            EqPolynomial::evals_from_point_with_base::<F>(&parent_point)
        };
        let product_pair_count = child_layers[0].len() / 2;
        if product_pair_count != eq.len() {
            return Err(SpartanWhirError::InvalidRoundCount);
        }

        let mut rounds = Vec::with_capacity(parent_point.len());
        let mut alpha = Vec::with_capacity(parent_point.len());
        observe_sumcheck_claim::<F, EF, C>(challenger, claim);
        for round_index in 0..parent_point.len() {
            let round = {
                let _profile = profile_detail_scope("spark_batched_product_round_compute");
                let _round_profile = profile_detail_scope(if child_layer_index == 0 {
                    if round_index == 0 {
                        "spark_batched_product_leaf_first_round_compute"
                    } else {
                        "spark_batched_product_leaf_later_round_compute"
                    }
                } else if round_index == 0 {
                    "spark_batched_product_internal_first_round_compute"
                } else {
                    "spark_batched_product_internal_later_round_compute"
                });
                if defer_product_split && round_index == 0 {
                    compute_batched_product_round_from_child_layers(
                        &eq,
                        &child_layers,
                        &dotproduct_lefts,
                        &dotproduct_rights,
                        &dotproduct_weights,
                        &dotproduct_base_weights,
                        &dotproduct_pair_indices,
                        &coeffs,
                    )?
                } else {
                    compute_batched_product_round(
                        &eq,
                        &product_lefts,
                        &product_rights,
                        &dotproduct_lefts,
                        &dotproduct_rights,
                        &dotproduct_weights,
                        &dotproduct_base_weights,
                        &dotproduct_pair_indices,
                        &coeffs,
                    )?
                }
            };
            challenger.observe_algebra_slice(&round.0);
            let challenge = challenger.sample_algebra_element::<EF>();
            claim = round.evaluate_at(challenge, claim);
            rounds.push(round);
            alpha.push(challenge);

            {
                let _profile = profile_detail_scope("spark_batched_product_round_bind");
                let _round_profile = profile_detail_scope(if child_layer_index == 0 {
                    if round_index == 0 {
                        "spark_batched_product_leaf_first_round_bind"
                    } else {
                        "spark_batched_product_leaf_later_round_bind"
                    }
                } else if round_index == 0 {
                    "spark_batched_product_internal_first_round_bind"
                } else {
                    "spark_batched_product_internal_later_round_bind"
                });
                bind_value_table(&mut eq, challenge)?;
                if defer_product_split && round_index == 0 {
                    (product_lefts, product_rights) = {
                        let _profile =
                            profile_detail_scope("spark_batched_product_bind_child_layer");
                        bind_product_child_layers(&child_layers, challenge)?
                    };
                } else {
                    bind_all_value_tables(&mut product_lefts, challenge)?;
                    bind_all_value_tables(&mut product_rights, challenge)?;
                }
                bind_all_value_tables(&mut dotproduct_lefts, challenge)?;
                bind_all_value_tables(&mut dotproduct_rights, challenge)?;
                if dotproduct_base_weights.is_empty() {
                    bind_all_value_tables(&mut dotproduct_weights, challenge)?;
                } else {
                    dotproduct_weights =
                        bind_base_weight_tables(dotproduct_base_weights, challenge)?;
                    dotproduct_base_weights = Vec::new();
                }
            }
        }
        if !dotproduct_base_weights.is_empty() {
            dotproduct_weights = dotproduct_base_weights
                .into_iter()
                .map(|weight| lift_base_values_to_extension(&weight))
                .collect();
        }
        let layer = batched_product_layer_from_bound_tables(
            rounds,
            &product_lefts,
            &product_rights,
            &dotproduct_lefts,
            &dotproduct_rights,
            &dotproduct_weights,
            &dotproduct_pair_indices,
        )?;
        check_batched_product_layer_final_claim(&layer, &eq, &coeffs, claim)?;
        let combine = observe_batched_product_layer_and_sample(&layer, challenger);

        claims_to_verify = layer
            .product_left_evals
            .iter()
            .zip(&layer.product_right_evals)
            .map(|(&left, &right)| extrapolate(left, right, combine))
            .collect();
        parent_point = alpha;
        let dotproduct_point = parent_point.clone();
        parent_point.push(combine);

        if include_dotproducts {
            leaf_claims = SparkBatchedProductLeafClaims {
                product_point: MultilinearPoint(parent_point.clone()),
                dotproduct_point: MultilinearPoint(dotproduct_point),
                product_evals: claims_to_verify.clone(),
                dotproduct_left_evals: layer.dotproduct_left_evals.clone(),
                dotproduct_right_evals: layer.dotproduct_right_evals.clone(),
                dotproduct_weight_evals: layer.dotproduct_weight_evals.clone(),
            };
        }
        proof_layers.push(layer);
    }

    if leaf_claims.product_point.0.is_empty() {
        leaf_claims = SparkBatchedProductLeafClaims {
            product_point: MultilinearPoint(parent_point),
            dotproduct_point: MultilinearPoint(Vec::new()),
            product_evals: claims_to_verify,
            dotproduct_left_evals: Vec::new(),
            dotproduct_right_evals: Vec::new(),
            dotproduct_weight_evals: Vec::new(),
        };
    }

    Ok((
        SparkBatchedProductProof {
            product_roots,
            dotproduct_claims,
            layers: proof_layers,
        },
        leaf_claims,
    ))
}

/// Verify a SPARK batched product proof.
///
/// `expected_product_roots` and `expected_dotproduct_claims` are supplied by
/// the surrounding protocol and are observed in the same order as the prover.
/// The returned product evaluations are at `product_point`; dotproduct factor
/// evaluations are at `dotproduct_point`, which is one coordinate shorter.
pub fn verify_spark_batched_product<EF, C>(
    proof: &SparkBatchedProductProof<EF>,
    expected_product_roots: &[EF],
    expected_dotproduct_claims: &[EF],
    domain_size: usize,
    challenger: &mut C,
) -> Result<SparkBatchedProductLeafClaims<EF>, SpartanWhirError>
where
    EF: ExtensionField<F>,
    C: FieldChallenger<F>,
{
    if domain_size == 0 || !domain_size.is_power_of_two() {
        return Err(SpartanWhirError::InvalidPolynomialLength);
    }
    if proof.product_roots != expected_product_roots
        || proof.dotproduct_claims != expected_dotproduct_claims
        || proof.layers.len() != log2_power_of_two(domain_size)
    {
        return Err(SpartanWhirError::SumcheckFailed);
    }

    let mut claims_to_verify = expected_product_roots.to_vec();
    let mut parent_point = Vec::new();
    let mut leaf_claims = SparkBatchedProductLeafClaims {
        product_point: MultilinearPoint(Vec::new()),
        dotproduct_point: MultilinearPoint(Vec::new()),
        product_evals: Vec::new(),
        dotproduct_left_evals: Vec::new(),
        dotproduct_right_evals: Vec::new(),
        dotproduct_weight_evals: Vec::new(),
    };
    for (layer_index, layer) in proof.layers.iter().enumerate() {
        let include_dotproducts =
            layer_index + 1 == proof.layers.len() && !expected_dotproduct_claims.is_empty();
        if include_dotproducts {
            claims_to_verify.extend_from_slice(expected_dotproduct_claims);
        }
        let coeffs = sample_batched_product_coefficients(&claims_to_verify, challenger);
        let joint_claim = claims_to_verify
            .iter()
            .zip(&coeffs)
            .fold(EF::ZERO, |acc, (&claim, &coeff)| acc + claim * coeff);

        let (alpha, reduced_claim) =
            verify_batched_product_layer(layer, parent_point.len(), joint_claim, challenger)?;
        let eq_eval = eq_eval_at_point(&parent_point, &alpha.0)?;
        check_batched_product_layer_final_claim(layer, &[eq_eval], &coeffs, reduced_claim)?;
        let combine = observe_batched_product_layer_and_sample(layer, challenger);

        claims_to_verify = layer
            .product_left_evals
            .iter()
            .zip(&layer.product_right_evals)
            .map(|(&left, &right)| extrapolate(left, right, combine))
            .collect();
        parent_point = alpha.0;
        let dotproduct_point = parent_point.clone();
        parent_point.push(combine);

        if include_dotproducts {
            leaf_claims = SparkBatchedProductLeafClaims {
                product_point: MultilinearPoint(parent_point.clone()),
                dotproduct_point: MultilinearPoint(dotproduct_point),
                product_evals: claims_to_verify.clone(),
                dotproduct_left_evals: layer.dotproduct_left_evals.clone(),
                dotproduct_right_evals: layer.dotproduct_right_evals.clone(),
                dotproduct_weight_evals: layer.dotproduct_weight_evals.clone(),
            };
        }
    }

    if leaf_claims.product_point.0.is_empty() {
        leaf_claims = SparkBatchedProductLeafClaims {
            product_point: MultilinearPoint(parent_point),
            dotproduct_point: MultilinearPoint(Vec::new()),
            product_evals: claims_to_verify,
            dotproduct_left_evals: Vec::new(),
            dotproduct_right_evals: Vec::new(),
            dotproduct_weight_evals: Vec::new(),
        };
    }
    Ok(leaf_claims)
}

fn validate_dotproducts_for_batched_product<EF>(
    dotproducts: &SparkProverDotProductBatch<EF>,
    product_domain_size: usize,
) -> Result<(), SpartanWhirError> {
    if dotproducts.is_empty() {
        return if dotproducts.lefts.is_empty()
            && dotproducts.rights.is_empty()
            && dotproducts.pair_indices.is_empty()
        {
            Ok(())
        } else {
            Err(SpartanWhirError::InvalidPolynomialLength)
        };
    }
    let dotproduct_domain_size = product_domain_size
        .checked_div(2)
        .ok_or(SpartanWhirError::InvalidPolynomialLength)?;
    if dotproduct_domain_size == 0 {
        return Err(SpartanWhirError::InvalidPolynomialLength);
    }
    if dotproducts.lefts.len() != dotproducts.rights.len()
        || dotproducts.weights.len() != dotproducts.pair_indices.len()
        || dotproducts
            .pair_indices
            .iter()
            .any(|&pair| pair >= dotproducts.lefts.len())
    {
        return Err(SpartanWhirError::InvalidPolynomialLength);
    }
    for (left, right) in dotproducts.lefts.iter().zip(&dotproducts.rights) {
        if left.len() != dotproduct_domain_size || right.len() != dotproduct_domain_size {
            return Err(SpartanWhirError::InvalidPolynomialLength);
        }
    }
    for weight in &dotproducts.weights {
        let weight_len = match weight {
            SparkProverDotProductWeight::Base(weight) => weight.len(),
            SparkProverDotProductWeight::Extension(weight) => weight.len(),
        };
        if weight_len != dotproduct_domain_size {
            return Err(SpartanWhirError::InvalidPolynomialLength);
        }
    }
    let uses_base_weights = matches!(dotproducts.weights[0], SparkProverDotProductWeight::Base(_));
    if dotproducts
        .weights
        .iter()
        .any(|weight| matches!(weight, SparkProverDotProductWeight::Base(_)) != uses_base_weights)
    {
        return Err(SpartanWhirError::InvalidPolynomialLength);
    }
    Ok(())
}

fn evaluate_dotproducts<EF>(
    dotproducts: &SparkProverDotProductBatch<EF>,
) -> Result<Vec<EF>, SpartanWhirError>
where
    EF: ExtensionField<F> + Send + Sync,
{
    dotproducts
        .weights
        .iter()
        .zip(&dotproducts.pair_indices)
        .map(|(weight, &pair_index)| {
            evaluate_dotproduct(
                &dotproducts.lefts[pair_index],
                &dotproducts.rights[pair_index],
                weight,
            )
        })
        .collect()
}

fn evaluate_dotproduct<EF>(
    left: &[EF],
    right: &[EF],
    weight: &SparkProverDotProductWeight<EF>,
) -> Result<EF, SpartanWhirError>
where
    EF: ExtensionField<F> + Send + Sync,
{
    let weight_len = match weight {
        SparkProverDotProductWeight::Base(weight) => weight.len(),
        SparkProverDotProductWeight::Extension(weight) => weight.len(),
    };
    if left.len() != right.len() || right.len() != weight_len {
        return Err(SpartanWhirError::InvalidPolynomialLength);
    }
    match weight {
        SparkProverDotProductWeight::Base(weight) => {
            if should_parallelize_spark_round(left.len()) {
                Ok((0..left.len())
                    .into_par_iter()
                    .map(|i| (left[i] * right[i]) * weight[i])
                    .par_fold_reduce(|| EF::ZERO, |acc, value| acc + value, |a, b| a + b))
            } else {
                Ok(left
                    .iter()
                    .zip(right)
                    .zip(weight)
                    .fold(EF::ZERO, |acc, ((&left, &right), &weight)| {
                        acc + (left * right) * weight
                    }))
            }
        }
        SparkProverDotProductWeight::Extension(weight) => {
            if should_parallelize_spark_round(left.len()) {
                Ok((0..left.len())
                    .into_par_iter()
                    .map(|i| left[i] * right[i] * weight[i])
                    .par_fold_reduce(|| EF::ZERO, |acc, value| acc + value, |a, b| a + b))
            } else {
                Ok(left
                    .iter()
                    .zip(right)
                    .zip(weight)
                    .fold(EF::ZERO, |acc, ((&left, &right), &weight)| {
                        acc + left * right * weight
                    }))
            }
        }
    }
}

fn sample_batched_product_coefficients<EF, C>(claims: &[EF], challenger: &mut C) -> Vec<EF>
where
    EF: ExtensionField<F>,
    C: FieldChallenger<F>,
{
    challenger.observe_algebra_slice(claims);
    (0..claims.len())
        .map(|_| challenger.sample_algebra_element::<EF>())
        .collect()
}

fn split_product_child_layers<EF>(
    child_layers: &[&[EF]],
) -> Result<(Vec<Vec<EF>>, Vec<Vec<EF>>), SpartanWhirError>
where
    EF: Copy + Send + Sync,
{
    let mut lefts = Vec::with_capacity(child_layers.len());
    let mut rights = Vec::with_capacity(child_layers.len());
    for child_layer in child_layers {
        if child_layer.len() < 2 || !child_layer.len().is_power_of_two() {
            return Err(SpartanWhirError::InvalidPolynomialLength);
        }
        let pair_count = child_layer.len() / 2;
        let (left, right) = if should_parallelize_spark_round(pair_count) {
            (0..pair_count)
                .into_par_iter()
                .map(|i| (child_layer[2 * i], child_layer[2 * i + 1]))
                .unzip()
        } else {
            child_layer
                .chunks_exact(2)
                .map(|pair| (pair[0], pair[1]))
                .unzip()
        };
        lefts.push(left);
        rights.push(right);
    }
    Ok((lefts, rights))
}

fn bind_product_child_layers<EF>(
    child_layers: &[&[EF]],
    challenge: EF,
) -> Result<(Vec<Vec<EF>>, Vec<Vec<EF>>), SpartanWhirError>
where
    EF: Field + Send + Sync,
{
    let mut lefts = Vec::with_capacity(child_layers.len());
    let mut rights = Vec::with_capacity(child_layers.len());
    for child_layer in child_layers {
        if child_layer.len() < 4 || !child_layer.len().is_power_of_two() {
            return Err(SpartanWhirError::InvalidPolynomialLength);
        }
        let output_len = child_layer.len() / 4;
        let bind = |i: usize| {
            let high_pair = i + output_len;
            (
                extrapolate(child_layer[2 * i], child_layer[2 * high_pair], challenge),
                extrapolate(
                    child_layer[2 * i + 1],
                    child_layer[2 * high_pair + 1],
                    challenge,
                ),
            )
        };
        let (left, right) = if should_parallelize_spark_round(output_len) {
            (0..output_len).into_par_iter().map(bind).unzip()
        } else {
            (0..output_len).map(bind).unzip()
        };
        lefts.push(left);
        rights.push(right);
    }
    Ok((lefts, rights))
}

fn bind_all_value_tables<EF>(tables: &mut [Vec<EF>], challenge: EF) -> Result<(), SpartanWhirError>
where
    EF: Field + Send + Sync,
{
    if tables.is_empty() {
        return Ok(());
    }

    let mut total_pair_count = 0usize;
    let mut largest_pair_count = 0usize;
    for table in tables.iter() {
        if table.len() < 2 || !table.len().is_multiple_of(2) {
            return Err(SpartanWhirError::InvalidRoundPolynomial);
        }
        let pair_count = table.len() / 2;
        total_pair_count = total_pair_count
            .checked_add(pair_count)
            .ok_or(SpartanWhirError::InvalidRoundPolynomial)?;
        largest_pair_count = largest_pair_count.max(pair_count);
    }

    if cfg!(feature = "parallel")
        && tables.len() > 1
        && total_pair_count >= SPARK_PARALLEL_ROUND_MIN_PAIRS
        && largest_pair_count < SPARK_PARALLEL_ROUND_MIN_PAIRS
    {
        return tables
            .par_iter_mut()
            .try_for_each(|table| bind_value_table(table, challenge));
    }

    for table in tables {
        bind_value_table(table, challenge)?;
    }
    Ok(())
}

fn bind_base_weight_tables<EF>(
    tables: Vec<Vec<F>>,
    challenge: EF,
) -> Result<Vec<Vec<EF>>, SpartanWhirError>
where
    EF: ExtensionField<F> + Send + Sync,
{
    // Organization inspired by ProveKit's mixed_sumcheck_map_reduce/mixed_fold
    // (World Foundation, MIT; https://github.com/worldfnd/provekit): keep public
    // matrix coefficients in the base field through the first round.
    tables
        .into_iter()
        .map(|table| {
            if table.len() < 2 || !table.len().is_multiple_of(2) {
                return Err(SpartanWhirError::InvalidRoundPolynomial);
            }
            let half = table.len() / 2;
            let bind = |i: usize| {
                let low = table[i];
                EF::from(low) + challenge * (table[i + half] - low)
            };
            if should_parallelize_spark_round(half) {
                Ok((0..half).into_par_iter().map(bind).collect())
            } else {
                Ok((0..half).map(bind).collect())
            }
        })
        .collect()
}

fn compute_batched_product_round<EF>(
    eq: &[EF],
    product_lefts: &[Vec<EF>],
    product_rights: &[Vec<EF>],
    dotproduct_lefts: &[Vec<EF>],
    dotproduct_rights: &[Vec<EF>],
    dotproduct_weights: &[Vec<EF>],
    dotproduct_base_weights: &[Vec<F>],
    dotproduct_pair_indices: &[usize],
    coeffs: &[EF],
) -> Result<CubicRoundPoly<EF>, SpartanWhirError>
where
    EF: ExtensionField<F> + Send + Sync,
{
    let product_count = product_lefts.len();
    let dotproduct_count = dotproduct_weights.len() + dotproduct_base_weights.len();
    if product_count == 0
        || product_rights.len() != product_count
        || dotproduct_rights.len() != dotproduct_lefts.len()
        || (dotproduct_count != 0 && dotproduct_lefts.is_empty())
        || dotproduct_pair_indices.len() != dotproduct_count
        || dotproduct_pair_indices
            .iter()
            .any(|&pair| pair >= dotproduct_lefts.len())
        || (!dotproduct_weights.is_empty() && !dotproduct_base_weights.is_empty())
        || coeffs.len() != product_count + dotproduct_count
    {
        return Err(SpartanWhirError::InvalidRoundPolynomial);
    }
    if product_lefts
        .iter()
        .chain(product_rights)
        .any(|table| table.len() != eq.len())
        || dotproduct_lefts
            .iter()
            .chain(dotproduct_rights)
            .any(|table| table.len() != eq.len())
        || dotproduct_weights
            .iter()
            .any(|table| table.len() != eq.len())
        || dotproduct_base_weights
            .iter()
            .any(|table| table.len() != eq.len())
        || eq.len() < 2
        || !eq.len().is_multiple_of(2)
    {
        return Err(SpartanWhirError::InvalidRoundPolynomial);
    }

    let half = eq.len() / 2;
    let (h0, h2, h3) = if should_parallelize_spark_round(half) {
        (0..half)
            .into_par_iter()
            .map(|i| {
                compute_batched_product_round_partial(
                    i,
                    half,
                    eq,
                    product_lefts,
                    product_rights,
                    dotproduct_lefts,
                    dotproduct_rights,
                    dotproduct_weights,
                    dotproduct_base_weights,
                    dotproduct_pair_indices,
                    coeffs,
                )
            })
            .par_fold_reduce(
                || (EF::ZERO, EF::ZERO, EF::ZERO),
                add_cubic_accumulators,
                add_cubic_accumulators,
            )
    } else {
        (0..half)
            .map(|i| {
                compute_batched_product_round_partial(
                    i,
                    half,
                    eq,
                    product_lefts,
                    product_rights,
                    dotproduct_lefts,
                    dotproduct_rights,
                    dotproduct_weights,
                    dotproduct_base_weights,
                    dotproduct_pair_indices,
                    coeffs,
                )
            })
            .fold((EF::ZERO, EF::ZERO, EF::ZERO), add_cubic_accumulators)
    };
    Ok(CubicRoundPoly([h0, h2, h3]))
}

fn compute_batched_product_round_from_child_layers<EF>(
    eq: &[EF],
    product_child_layers: &[&[EF]],
    dotproduct_lefts: &[Vec<EF>],
    dotproduct_rights: &[Vec<EF>],
    dotproduct_weights: &[Vec<EF>],
    dotproduct_base_weights: &[Vec<F>],
    dotproduct_pair_indices: &[usize],
    coeffs: &[EF],
) -> Result<CubicRoundPoly<EF>, SpartanWhirError>
where
    EF: ExtensionField<F> + Send + Sync,
{
    let product_count = product_child_layers.len();
    let dotproduct_count = dotproduct_weights.len() + dotproduct_base_weights.len();
    if product_count == 0
        || dotproduct_rights.len() != dotproduct_lefts.len()
        || (dotproduct_count != 0 && dotproduct_lefts.is_empty())
        || dotproduct_pair_indices.len() != dotproduct_count
        || dotproduct_pair_indices
            .iter()
            .any(|&pair| pair >= dotproduct_lefts.len())
        || (!dotproduct_weights.is_empty() && !dotproduct_base_weights.is_empty())
        || coeffs.len() != product_count + dotproduct_count
        || product_child_layers
            .iter()
            .any(|layer| layer.len() != 2 * eq.len())
        || dotproduct_lefts
            .iter()
            .chain(dotproduct_rights)
            .any(|table| table.len() != eq.len())
        || dotproduct_weights
            .iter()
            .any(|table| table.len() != eq.len())
        || dotproduct_base_weights
            .iter()
            .any(|table| table.len() != eq.len())
        || eq.len() < 2
        || !eq.len().is_multiple_of(2)
    {
        return Err(SpartanWhirError::InvalidRoundPolynomial);
    }

    let half = eq.len() / 2;
    let accumulate = |i| {
        compute_batched_product_round_from_child_layers_partial(
            i,
            half,
            eq,
            product_child_layers,
            dotproduct_lefts,
            dotproduct_rights,
            dotproduct_weights,
            dotproduct_base_weights,
            dotproduct_pair_indices,
            coeffs,
        )
    };
    let (h0, h2, h3) = if should_parallelize_spark_round(half) {
        (0..half).into_par_iter().map(accumulate).par_fold_reduce(
            || (EF::ZERO, EF::ZERO, EF::ZERO),
            add_cubic_accumulators,
            add_cubic_accumulators,
        )
    } else {
        (0..half)
            .map(accumulate)
            .fold((EF::ZERO, EF::ZERO, EF::ZERO), add_cubic_accumulators)
    };
    Ok(CubicRoundPoly([h0, h2, h3]))
}

#[allow(clippy::too_many_arguments)]
fn compute_independent_dotproduct_round_partial<EF>(
    i: usize,
    half: usize,
    dotproduct_lefts: &[Vec<EF>],
    dotproduct_rights: &[Vec<EF>],
    dotproduct_weights: &[Vec<EF>],
    dotproduct_base_weights: &[Vec<F>],
    dotproduct_pair_indices: &[usize],
    coeffs: &[EF],
) -> (EF, EF, EF)
where
    EF: ExtensionField<F>,
{
    let mut h0 = EF::ZERO;
    let mut h2 = EF::ZERO;
    let mut h3 = EF::ZERO;

    // A, B, and C reuse the same row/column read pairs. Combine their weights
    // before multiplying the shared pair at each evaluation point.
    for pair_index in 0..dotproduct_lefts.len() {
        let mut combined_weight0 = EF::ZERO;
        let mut combined_weight2 = EF::ZERO;
        let mut combined_weight3 = EF::ZERO;
        let mut has_weight = false;
        for (dotproduct_index, &selected_pair) in dotproduct_pair_indices.iter().enumerate() {
            if selected_pair != pair_index {
                continue;
            }
            has_weight = true;
            let coeff = coeffs[dotproduct_index];
            if dotproduct_base_weights.is_empty() {
                let weight0 = dotproduct_weights[dotproduct_index][i];
                let weight1 = dotproduct_weights[dotproduct_index][i + half];
                let (weight2, weight3) = extrapolate_at_two_and_three(weight0, weight1);
                combined_weight0 += coeff * weight0;
                combined_weight2 += coeff * weight2;
                combined_weight3 += coeff * weight3;
            } else {
                let weight0 = dotproduct_base_weights[dotproduct_index][i];
                let weight1 = dotproduct_base_weights[dotproduct_index][i + half];
                let (weight2, weight3) = extrapolate_at_two_and_three(weight0, weight1);
                combined_weight0 += coeff * weight0;
                combined_weight2 += coeff * weight2;
                combined_weight3 += coeff * weight3;
            }
        }
        if !has_weight {
            continue;
        }

        let left0 = dotproduct_lefts[pair_index][i];
        let left1 = dotproduct_lefts[pair_index][i + half];
        let right0 = dotproduct_rights[pair_index][i];
        let right1 = dotproduct_rights[pair_index][i + half];
        let (left2, left3) = extrapolate_at_two_and_three(left0, left1);
        let (right2, right3) = extrapolate_at_two_and_three(right0, right1);
        h0 += left0 * right0 * combined_weight0;
        h2 += left2 * right2 * combined_weight2;
        h3 += left3 * right3 * combined_weight3;
    }
    (h0, h2, h3)
}

#[allow(clippy::too_many_arguments)]
fn compute_batched_product_round_from_child_layers_partial<EF>(
    i: usize,
    half: usize,
    eq: &[EF],
    product_child_layers: &[&[EF]],
    dotproduct_lefts: &[Vec<EF>],
    dotproduct_rights: &[Vec<EF>],
    dotproduct_weights: &[Vec<EF>],
    dotproduct_base_weights: &[Vec<F>],
    dotproduct_pair_indices: &[usize],
    coeffs: &[EF],
) -> (EF, EF, EF)
where
    EF: ExtensionField<F>,
{
    let product_count = product_child_layers.len();

    let eq0 = eq[i];
    let eq1 = eq[i + half];
    let (eq2, eq3) = extrapolate_at_two_and_three(eq0, eq1);
    // The equality value is shared by every product circuit. Combine the
    // circuits first so each evaluation needs one equality multiplication.
    let mut product_h0 = EF::ZERO;
    let mut product_h2 = EF::ZERO;
    let mut product_h3 = EF::ZERO;
    for product_index in 0..product_count {
        let coeff = coeffs[product_index];
        let child_layer = product_child_layers[product_index];
        let left0 = child_layer[2 * i];
        let left1 = child_layer[2 * (i + half)];
        let right0 = child_layer[2 * i + 1];
        let right1 = child_layer[2 * (i + half) + 1];
        let (left2, left3) = extrapolate_at_two_and_three(left0, left1);
        let (right2, right3) = extrapolate_at_two_and_three(right0, right1);
        product_h0 += coeff * left0 * right0;
        product_h2 += coeff * left2 * right2;
        product_h3 += coeff * left3 * right3;
    }
    let mut h0 = eq0 * product_h0;
    let mut h2 = eq2 * product_h2;
    let mut h3 = eq3 * product_h3;
    let dotproduct_accumulator = compute_independent_dotproduct_round_partial(
        i,
        half,
        dotproduct_lefts,
        dotproduct_rights,
        dotproduct_weights,
        dotproduct_base_weights,
        dotproduct_pair_indices,
        &coeffs[product_count..],
    );
    (h0, h2, h3) = add_cubic_accumulators((h0, h2, h3), dotproduct_accumulator);

    (h0, h2, h3)
}

fn compute_batched_product_round_partial<EF>(
    i: usize,
    half: usize,
    eq: &[EF],
    product_lefts: &[Vec<EF>],
    product_rights: &[Vec<EF>],
    dotproduct_lefts: &[Vec<EF>],
    dotproduct_rights: &[Vec<EF>],
    dotproduct_weights: &[Vec<EF>],
    dotproduct_base_weights: &[Vec<F>],
    dotproduct_pair_indices: &[usize],
    coeffs: &[EF],
) -> (EF, EF, EF)
where
    EF: ExtensionField<F>,
{
    let product_count = product_lefts.len();

    let eq0 = eq[i];
    let eq1 = eq[i + half];
    let (eq2, eq3) = extrapolate_at_two_and_three(eq0, eq1);
    // The equality value is shared by every product circuit. Combine the
    // circuits first so each evaluation needs one equality multiplication.
    let mut product_h0 = EF::ZERO;
    let mut product_h2 = EF::ZERO;
    let mut product_h3 = EF::ZERO;
    for product_index in 0..product_count {
        let coeff = coeffs[product_index];
        let left0 = product_lefts[product_index][i];
        let left1 = product_lefts[product_index][i + half];
        let right0 = product_rights[product_index][i];
        let right1 = product_rights[product_index][i + half];
        let (left2, left3) = extrapolate_at_two_and_three(left0, left1);
        let (right2, right3) = extrapolate_at_two_and_three(right0, right1);
        product_h0 += coeff * left0 * right0;
        product_h2 += coeff * left2 * right2;
        product_h3 += coeff * left3 * right3;
    }
    let mut h0 = eq0 * product_h0;
    let mut h2 = eq2 * product_h2;
    let mut h3 = eq3 * product_h3;
    let dotproduct_accumulator = compute_independent_dotproduct_round_partial(
        i,
        half,
        dotproduct_lefts,
        dotproduct_rights,
        dotproduct_weights,
        dotproduct_base_weights,
        dotproduct_pair_indices,
        &coeffs[product_count..],
    );
    (h0, h2, h3) = add_cubic_accumulators((h0, h2, h3), dotproduct_accumulator);

    (h0, h2, h3)
}

fn batched_product_layer_from_bound_tables<EF>(
    rounds: Vec<CubicRoundPoly<EF>>,
    product_lefts: &[Vec<EF>],
    product_rights: &[Vec<EF>],
    dotproduct_lefts: &[Vec<EF>],
    dotproduct_rights: &[Vec<EF>],
    dotproduct_weights: &[Vec<EF>],
    dotproduct_pair_indices: &[usize],
) -> Result<SparkBatchedProductLayerProof<EF>, SpartanWhirError>
where
    EF: Copy,
{
    let product_left_evals = singleton_evals(product_lefts)?;
    let product_right_evals = singleton_evals(product_rights)?;
    if dotproduct_pair_indices.len() != dotproduct_weights.len() {
        return Err(SpartanWhirError::InvalidRoundPolynomial);
    }
    let dotproduct_left_evals =
        selected_singleton_evals(dotproduct_lefts, dotproduct_pair_indices)?;
    let dotproduct_right_evals =
        selected_singleton_evals(dotproduct_rights, dotproduct_pair_indices)?;
    let dotproduct_weight_evals = singleton_evals(dotproduct_weights)?;
    Ok(SparkBatchedProductLayerProof {
        rounds,
        product_left_evals,
        product_right_evals,
        dotproduct_left_evals,
        dotproduct_right_evals,
        dotproduct_weight_evals,
    })
}

fn selected_singleton_evals<EF>(
    tables: &[Vec<EF>],
    indices: &[usize],
) -> Result<Vec<EF>, SpartanWhirError>
where
    EF: Copy,
{
    indices
        .iter()
        .map(|&index| {
            let table = tables
                .get(index)
                .ok_or(SpartanWhirError::InvalidRoundPolynomial)?;
            if table.len() != 1 {
                return Err(SpartanWhirError::SumcheckFailed);
            }
            Ok(table[0])
        })
        .collect()
}

fn singleton_evals<EF>(tables: &[Vec<EF>]) -> Result<Vec<EF>, SpartanWhirError>
where
    EF: Copy,
{
    tables
        .iter()
        .map(|table| {
            if table.len() != 1 {
                return Err(SpartanWhirError::SumcheckFailed);
            }
            Ok(table[0])
        })
        .collect()
}

fn check_batched_product_layer_final_claim<EF>(
    layer: &SparkBatchedProductLayerProof<EF>,
    eq: &[EF],
    coeffs: &[EF],
    claim: EF,
) -> Result<(), SpartanWhirError>
where
    EF: Field,
{
    if eq.len() != 1
        || layer.product_left_evals.len() != layer.product_right_evals.len()
        || layer.dotproduct_left_evals.len() != layer.dotproduct_right_evals.len()
        || layer.dotproduct_right_evals.len() != layer.dotproduct_weight_evals.len()
        || coeffs.len() != layer.product_left_evals.len() + layer.dotproduct_left_evals.len()
    {
        return Err(SpartanWhirError::InvalidRoundPolynomial);
    }
    let product_count = layer.product_left_evals.len();
    let expected_products = layer
        .product_left_evals
        .iter()
        .zip(&layer.product_right_evals)
        .zip(&coeffs[..product_count])
        .fold(EF::ZERO, |acc, ((&left, &right), &coeff)| {
            acc + coeff * eq[0] * left * right
        });
    let expected_dotproducts = layer
        .dotproduct_left_evals
        .iter()
        .zip(&layer.dotproduct_right_evals)
        .zip(&layer.dotproduct_weight_evals)
        .zip(&coeffs[product_count..])
        .fold(EF::ZERO, |acc, (((&left, &right), &weight), &coeff)| {
            acc + coeff * left * right * weight
        });
    if claim != expected_products + expected_dotproducts {
        return Err(SpartanWhirError::SumcheckFailed);
    }
    Ok(())
}

fn verify_batched_product_layer<EF, C>(
    layer: &SparkBatchedProductLayerProof<EF>,
    expected_rounds: usize,
    initial_claim: EF,
    challenger: &mut C,
) -> Result<(MultilinearPoint<EF>, EF), SpartanWhirError>
where
    EF: ExtensionField<F>,
    C: FieldChallenger<F>,
{
    replay_compact_rounds::<F, EF, C, _>(
        &layer.rounds,
        initial_claim,
        expected_rounds,
        3,
        challenger,
    )
}

fn observe_batched_product_layer_and_sample<EF, C>(
    layer: &SparkBatchedProductLayerProof<EF>,
    challenger: &mut C,
) -> EF
where
    EF: ExtensionField<F>,
    C: FieldChallenger<F>,
{
    challenger.observe_algebra_slice(&layer.product_left_evals);
    challenger.observe_algebra_slice(&layer.product_right_evals);
    challenger.observe_algebra_slice(&layer.dotproduct_left_evals);
    challenger.observe_algebra_slice(&layer.dotproduct_right_evals);
    challenger.observe_algebra_slice(&layer.dotproduct_weight_evals);
    challenger.sample_algebra_element::<EF>()
}

pub fn spark_selector_from_slot<EF>(slot: SparkMatrixSlot, r: EF) -> EF
where
    EF: PrimeCharacteristicRing + Copy,
{
    match slot {
        SparkMatrixSlot::A => EF::ONE,
        SparkMatrixSlot::B => r,
        SparkMatrixSlot::C => r * r,
        SparkMatrixSlot::Zero => EF::ZERO,
    }
}

pub fn spark_selector_from_high_bits<EF>(tag_hi: EF, tag_lo: EF, r: EF) -> EF
where
    EF: PrimeCharacteristicRing + Copy,
{
    let one = EF::ONE;
    let eq_00 = (one - tag_hi) * (one - tag_lo);
    let eq_01 = (one - tag_hi) * tag_lo;
    let eq_10 = tag_hi * (one - tag_lo);
    eq_00 + r * eq_01 + r * r * eq_10
}

pub fn spark_selector_from_joint_point<EF>(
    slot_mapping: [SparkMatrixSlot; 4],
    point: &MultilinearPoint<EF>,
    r: EF,
) -> Result<EF, SpartanWhirError>
where
    EF: PrimeCharacteristicRing + Copy,
{
    if point.0.len() < 2 {
        return Err(SpartanWhirError::InvalidRoundCount);
    }
    let one = EF::ONE;
    let tag_hi = point.0[0];
    let tag_lo = point.0[1];
    let tag_eqs = [
        (one - tag_hi) * (one - tag_lo),
        (one - tag_hi) * tag_lo,
        tag_hi * (one - tag_lo),
        tag_hi * tag_lo,
    ];

    Ok(tag_eqs
        .into_iter()
        .zip(slot_mapping)
        .fold(EF::ZERO, |acc, (eq, slot)| {
            acc + eq * spark_selector_from_slot(slot, r)
        }))
}

#[derive(Clone, Copy)]
struct AggregatedEntry {
    row: usize,
    col: usize,
    val: F,
}

#[derive(Clone, Copy)]
struct UnionEntry {
    row: usize,
    col: usize,
    val_a: F,
    val_b: F,
    val_c: F,
}

fn aggregate_matrix_entries(
    matrix: &SparseMatrix<F>,
) -> Result<Vec<AggregatedEntry>, SpartanWhirError> {
    let mut entries: Vec<AggregatedEntry> = matrix
        .entries
        .iter()
        .map(|entry| AggregatedEntry {
            row: entry.row,
            col: entry.col,
            val: entry.val,
        })
        .collect();
    entries.sort_by(compare_aggregated_entries);

    let mut out: Vec<AggregatedEntry> = Vec::with_capacity(entries.len());
    for entry in entries {
        if let Some(last) = out.last_mut() {
            if last.row == entry.row && last.col == entry.col {
                last.val += entry.val;
                continue;
            }
        }
        out.push(entry);
    }
    Ok(out)
}

fn compare_aggregated_entries(a: &AggregatedEntry, b: &AggregatedEntry) -> Ordering {
    match a.row.cmp(&b.row) {
        Ordering::Equal => a.col.cmp(&b.col),
        other => other,
    }
}

fn union_sparse_entries(
    a: &SparseMatrix<F>,
    b: &SparseMatrix<F>,
    c: &SparseMatrix<F>,
) -> Result<Vec<UnionEntry>, SpartanWhirError> {
    let a = aggregate_matrix_entries(a)?;
    let b = aggregate_matrix_entries(b)?;
    let c = aggregate_matrix_entries(c)?;
    union_aggregated_entries(&a, &b, &c)
}

fn union_aggregated_entries(
    a: &[AggregatedEntry],
    b: &[AggregatedEntry],
    c: &[AggregatedEntry],
) -> Result<Vec<UnionEntry>, SpartanWhirError> {
    let mut keys = Vec::with_capacity(a.len() + b.len() + c.len());
    keys.extend(a.iter().map(|entry| (entry.row, entry.col)));
    keys.extend(b.iter().map(|entry| (entry.row, entry.col)));
    keys.extend(c.iter().map(|entry| (entry.row, entry.col)));
    keys.sort();
    keys.dedup();

    let mut out = Vec::with_capacity(keys.len());
    for (row, col) in keys {
        out.push(UnionEntry {
            row,
            col,
            val_a: lookup_aggregated_value(a, row, col),
            val_b: lookup_aggregated_value(b, row, col),
            val_c: lookup_aggregated_value(c, row, col),
        });
    }
    Ok(out)
}

fn lookup_aggregated_value(entries: &[AggregatedEntry], row: usize, col: usize) -> F {
    entries
        .binary_search_by(|entry| match entry.row.cmp(&row) {
            Ordering::Equal => entry.col.cmp(&col),
            other => other,
        })
        .map(|index| entries[index].val)
        .unwrap_or(F::ZERO)
}

fn fill_joint_slot(
    slot: SparkMatrixSlot,
    matrix_nnz_padded: usize,
    matrix: &SparseMatrix<F>,
    rows: &mut [F],
    cols: &mut [F],
    vals: &mut [F],
) -> Result<(), SpartanWhirError> {
    let offset = (slot as usize)
        .checked_mul(matrix_nnz_padded)
        .ok_or(SpartanWhirError::InvalidR1csShape)?;
    for (i, entry) in matrix.entries.iter().enumerate() {
        let index = offset
            .checked_add(i)
            .ok_or(SpartanWhirError::InvalidR1csShape)?;
        rows[index] = usize_to_field(entry.row)?;
        cols[index] = usize_to_field(entry.col)?;
        vals[index] = entry.val;
    }
    Ok(())
}

fn collect_joint_addresses(
    matrix_nnz_padded: usize,
    a: &SparseMatrix<F>,
    b: &SparseMatrix<F>,
    c: &SparseMatrix<F>,
    rows: bool,
) -> Vec<usize> {
    let mut out = Vec::with_capacity(matrix_nnz_padded * 4);
    push_addresses(&mut out, matrix_nnz_padded, a, rows);
    push_addresses(&mut out, matrix_nnz_padded, b, rows);
    push_addresses(&mut out, matrix_nnz_padded, c, rows);
    out.resize(matrix_nnz_padded * 4, 0);
    out
}

fn push_addresses(
    out: &mut Vec<usize>,
    matrix_nnz_padded: usize,
    matrix: &SparseMatrix<F>,
    rows: bool,
) {
    for entry in &matrix.entries {
        out.push(if rows { entry.row } else { entry.col });
    }
    let padded_len = out.len().next_multiple_of(matrix_nnz_padded);
    out.resize(padded_len, 0);
}

fn memory_in_the_head(
    memory_size: usize,
    addrs: &[usize],
) -> Result<(Vec<F>, Vec<F>), SpartanWhirError> {
    if memory_size == 0 {
        return Err(SpartanWhirError::InvalidR1csShape);
    }
    let mut read_ts = Vec::with_capacity(addrs.len());
    let mut audit_ts = vec![0usize; memory_size];
    for &addr in addrs {
        if addr >= memory_size {
            return Err(SpartanWhirError::InvalidR1csShape);
        }
        let read = audit_ts[addr];
        let write = read
            .checked_add(1)
            .ok_or(SpartanWhirError::InvalidR1csShape)?;
        read_ts.push(usize_to_field(read)?);
        audit_ts[addr] = write;
    }

    let audit_ts = audit_ts
        .into_iter()
        .map(usize_to_field)
        .collect::<Result<Vec<_>, _>>()?;
    Ok((read_ts, audit_ts))
}

fn validate_packed_index_bounds(shape: &R1csShape<F>) -> Result<(), SpartanWhirError> {
    validate_profile_packed_index_bounds(SparkShapeProfile {
        num_rows: shape.num_cons,
        num_cols: spark_col_memory_size(shape)?,
        nnz_a: shape.a.nnz(),
        nnz_b: shape.b.nnz(),
        nnz_c: shape.c.nnz(),
        union_nnz: 0,
    })
}

fn validate_profile_packed_index_bounds(
    profile: SparkShapeProfile,
) -> Result<(), SpartanWhirError> {
    if profile.num_rows == 0 || profile.num_cols == 0 {
        return Err(SpartanWhirError::InvalidR1csShape);
    }
    if profile.num_rows >= F::ORDER_U32 as usize || profile.num_cols >= F::ORDER_U32 as usize {
        return Err(SpartanWhirError::InvalidR1csShape);
    }
    let max_nnz_padded =
        padded_nonzero_domain(profile.nnz_a.max(profile.nnz_b).max(profile.nnz_c))?;
    let max_possible_union = max_nnz_padded
        .checked_mul(3)
        .ok_or(SpartanWhirError::InvalidR1csShape)?;
    validate_value_domain_field_bound(max_possible_union)?;
    Ok(())
}

fn validate_value_domain_field_bound(value_domain_size: usize) -> Result<(), SpartanWhirError> {
    if value_domain_size >= F::ORDER_U32 as usize {
        return Err(SpartanWhirError::InvalidR1csShape);
    }
    Ok(())
}

pub fn spark_col_memory_size(shape: &R1csShape<F>) -> Result<usize, SpartanWhirError> {
    let width = shape
        .num_vars
        .checked_mul(2)
        .ok_or(SpartanWhirError::InvalidR1csShape)?;
    if width < shape.a.num_cols {
        return Err(SpartanWhirError::InvalidR1csShape);
    }
    width
        .checked_next_power_of_two()
        .ok_or(SpartanWhirError::InvalidR1csShape)
}

fn validate_value_inputs<EF>(
    tables: &SparkTables,
    r_x: &MultilinearPoint<EF>,
    r_y: &MultilinearPoint<EF>,
) -> Result<(), SpartanWhirError> {
    validate_table_metadata(tables)?;
    if r_x.0.len() != log2_power_of_two(tables.row_memory_size)
        || r_y.0.len() != log2_power_of_two(tables.col_memory_size)
    {
        return Err(SpartanWhirError::InvalidRoundCount);
    }
    Ok(())
}

fn validate_table_metadata(tables: &SparkTables) -> Result<(), SpartanWhirError> {
    if tables.value_domain_size == 0
        || !tables.value_domain_size.is_power_of_two()
        || tables.rows.len() != tables.value_domain_size
        || tables.cols.len() != tables.value_domain_size
        || tables.vals.len() != tables.value_domain_size
        || tables.val_a.len() != tables.value_domain_size
        || tables.val_b.len() != tables.value_domain_size
        || tables.val_c.len() != tables.value_domain_size
        || tables.read_ts_row.len() != tables.value_domain_size
        || tables.read_ts_col.len() != tables.value_domain_size
        || tables.audit_ts_row.len() != tables.row_memory_size
        || tables.audit_ts_col.len() != tables.col_memory_size
        || tables.row_memory_size == 0
        || tables.col_memory_size == 0
        || !tables.row_memory_size.is_power_of_two()
        || !tables.col_memory_size.is_power_of_two()
    {
        return Err(SpartanWhirError::InvalidR1csShape);
    }
    Ok(())
}

fn validate_read_tables<EF>(
    tables: &SparkTables,
    read_tables: &SparkReadTables<EF>,
) -> Result<(), SpartanWhirError> {
    if tables.value_domain_size == 0
        || !tables.value_domain_size.is_power_of_two()
        || read_tables.erow.len() != tables.value_domain_size
        || read_tables.ecol.len() != tables.value_domain_size
    {
        return Err(SpartanWhirError::InvalidR1csShape);
    }
    Ok(())
}

fn joint_selector_table<EF>(tables: &SparkTables, r: EF) -> Result<Vec<EF>, SpartanWhirError>
where
    EF: ExtensionField<F> + Send + Sync,
{
    if tables.matrix_nnz_padded == 0
        || !tables.matrix_nnz_padded.is_power_of_two()
        || tables.value_domain_size != tables.matrix_nnz_padded * 4
    {
        return Err(SpartanWhirError::InvalidR1csShape);
    }
    let mut out = Vec::with_capacity(tables.value_domain_size);
    for slot in tables.slot_mapping {
        out.resize(
            out.len() + tables.matrix_nnz_padded,
            spark_selector_from_slot(slot, r),
        );
    }
    Ok(out)
}

fn lift_base_table<EF>(table: &[F]) -> Vec<EF>
where
    EF: ExtensionField<F>,
{
    table.iter().map(|&value| EF::from(value)).collect()
}

fn matrix_rlc_table<EF>(tables: &SparkTables, r: EF) -> Result<Vec<EF>, SpartanWhirError>
where
    EF: ExtensionField<F>,
{
    if tables.val_a.len() != tables.value_domain_size
        || tables.val_b.len() != tables.value_domain_size
        || tables.val_c.len() != tables.value_domain_size
    {
        return Err(SpartanWhirError::InvalidR1csShape);
    }
    Ok(tables
        .val_a
        .iter()
        .zip(&tables.val_b)
        .zip(&tables.val_c)
        .map(|((&a, &b), &c)| EF::from(a) + r * EF::from(b) + r * r * EF::from(c))
        .collect())
}

fn evaluate_base_table_as_extension<EF>(table: &[F], point: &[EF]) -> Result<EF, SpartanWhirError>
where
    EF: ExtensionField<F>,
{
    evaluate_mle_table(&lift_base_table::<EF>(table), point)
}

fn virtual_index_eval<EF>(domain_size: usize, point: &[EF]) -> Result<EF, SpartanWhirError>
where
    EF: ExtensionField<F>,
{
    if domain_size == 0
        || !domain_size.is_power_of_two()
        || point.len() != log2_power_of_two(domain_size)
    {
        return Err(SpartanWhirError::InvalidRoundCount);
    }
    let mut value = EF::ZERO;
    for (i, &coordinate) in point.iter().enumerate() {
        let bit_weight = 1usize << (point.len() - i - 1);
        value += coordinate * EF::from(usize_to_field(bit_weight)?);
    }
    Ok(value)
}

fn spark_read_eq_table<EF>(
    memory_size: usize,
    packed_addrs: &[F],
    point: &[EF],
) -> Result<Vec<EF>, SpartanWhirError>
where
    EF: ExtensionField<F> + Send + Sync,
{
    if memory_size == 0 || !memory_size.is_power_of_two() {
        return Err(SpartanWhirError::InvalidR1csShape);
    }
    if point.len() != log2_power_of_two(memory_size) {
        return Err(SpartanWhirError::InvalidRoundCount);
    }

    let eq = EqPolynomial::evals_from_point_with_base::<F>(point);
    if should_parallelize_spark_round(packed_addrs.len()) {
        packed_addrs
            .par_iter()
            .map(|addr| {
                let index = addr.as_canonical_u32() as usize;
                eq.get(index)
                    .copied()
                    .ok_or(SpartanWhirError::InvalidR1csShape)
            })
            .collect()
    } else {
        packed_addrs
            .iter()
            .map(|addr| {
                let index = addr.as_canonical_u32() as usize;
                eq.get(index)
                    .copied()
                    .ok_or(SpartanWhirError::InvalidR1csShape)
            })
            .collect()
    }
}

fn observe_spark_memory_context<EF, C>(
    tables: &SparkTables,
    challenger: &mut C,
) -> Result<(), SpartanWhirError>
where
    EF: ExtensionField<F>,
    C: FieldChallenger<F>,
{
    observe_spark_memory_metadata::<EF, C>(&tables.metadata(), challenger)
}

fn observe_spark_memory_metadata<EF, C>(
    metadata: &SparkTableMetadata,
    challenger: &mut C,
) -> Result<(), SpartanWhirError>
where
    EF: ExtensionField<F>,
    C: FieldChallenger<F>,
{
    let layout_tag = match metadata.layout {
        SparkLayoutKind::Joint => 0,
        SparkLayoutKind::SharedUnion => 1,
        SparkLayoutKind::PerMatrix => 2,
    };
    let context = [
        EF::from(usize_to_field(metadata.row_memory_size)?),
        EF::from(usize_to_field(metadata.col_memory_size)?),
        EF::from(usize_to_field(metadata.value_domain_size)?),
        EF::from(usize_to_field(metadata.matrix_nnz_padded)?),
        EF::from(usize_to_field(layout_tag)?),
    ];
    challenger.observe_algebra_slice(&context);
    Ok(())
}

fn product_tree_from_terms<EF>(terms: &[EF]) -> Result<Vec<Vec<EF>>, SpartanWhirError>
where
    EF: Field + Send + Sync,
{
    product_tree_from_terms_owned(terms.to_vec())
}

fn product_tree_from_terms_owned<EF>(terms: Vec<EF>) -> Result<Vec<Vec<EF>>, SpartanWhirError>
where
    EF: Field + Send + Sync,
{
    if terms.is_empty() || !terms.len().is_power_of_two() {
        return Err(SpartanWhirError::InvalidPolynomialLength);
    }

    let mut layers = Vec::new();
    layers.push(terms);
    while layers.last().is_some_and(|layer| layer.len() > 1) {
        let previous = layers
            .last()
            .ok_or(SpartanWhirError::InvalidPolynomialLength)?;
        layers.push(product_tree_next_layer(previous));
    }
    Ok(layers)
}

fn product_tree_next_layer<EF>(previous: &[EF]) -> Vec<EF>
where
    EF: Field + Send + Sync,
{
    let pair_count = previous.len() / 2;
    if should_parallelize_spark_round(pair_count) {
        previous
            .par_chunks(2)
            .map(|pair| pair[0] * pair[1])
            .collect()
    } else {
        let mut next = Vec::with_capacity(pair_count);
        for pair in previous.chunks_exact(2) {
            next.push(pair[0] * pair[1]);
        }
        next
    }
}

fn prove_spark_grand_product_layer<EF, C>(
    child_layer: &[EF],
    parent_point: &[EF],
    initial_claim: EF,
    challenger: &mut C,
) -> Result<(SparkGrandProductLayerProof<EF>, MultilinearPoint<EF>), SpartanWhirError>
where
    EF: ExtensionField<F>,
    C: FieldChallenger<F>,
{
    if child_layer.len() < 2 || !child_layer.len().is_power_of_two() {
        return Err(SpartanWhirError::InvalidPolynomialLength);
    }
    let parent_len = child_layer.len() / 2;
    if parent_point.len() != log2_power_of_two(parent_len) {
        return Err(SpartanWhirError::InvalidRoundCount);
    }

    let mut eq = EqPolynomial::evals_from_point_with_base::<F>(parent_point);
    let mut left = Vec::with_capacity(parent_len);
    let mut right = Vec::with_capacity(parent_len);
    for pair in child_layer.chunks_exact(2) {
        left.push(pair[0]);
        right.push(pair[1]);
    }

    let mut claim = initial_claim;
    let mut rounds = Vec::with_capacity(parent_point.len());
    let mut alpha = Vec::with_capacity(parent_point.len());
    observe_sumcheck_claim::<F, EF, C>(challenger, claim);
    for _ in 0..parent_point.len() {
        let round = compute_grand_product_layer_round(&eq, &left, &right)?;
        challenger.observe_algebra_slice(&round.0);
        let challenge = challenger.sample_algebra_element::<EF>();
        claim = round.evaluate_at(challenge, claim);
        rounds.push(round);
        alpha.push(challenge);

        bind_value_table(&mut eq, challenge)?;
        bind_value_table(&mut left, challenge)?;
        bind_value_table(&mut right, challenge)?;
    }

    if eq.len() != 1 || left.len() != 1 || right.len() != 1 {
        return Err(SpartanWhirError::SumcheckFailed);
    }
    if claim != eq[0] * left[0] * right[0] {
        return Err(SpartanWhirError::SumcheckFailed);
    }

    Ok((
        SparkGrandProductLayerProof {
            rounds,
            left_eval: left[0],
            right_eval: right[0],
        },
        MultilinearPoint(alpha),
    ))
}

fn verify_spark_grand_product_layer<EF, C>(
    proof: &SparkGrandProductLayerProof<EF>,
    parent_point: &[EF],
    initial_claim: EF,
    challenger: &mut C,
) -> Result<(MultilinearPoint<EF>, EF), SpartanWhirError>
where
    EF: ExtensionField<F>,
    C: FieldChallenger<F>,
{
    replay_compact_rounds::<F, EF, C, _>(
        &proof.rounds,
        initial_claim,
        parent_point.len(),
        3,
        challenger,
    )
}

fn compute_grand_product_layer_round<EF>(
    eq: &[EF],
    left: &[EF],
    right: &[EF],
) -> Result<CubicRoundPoly<EF>, SpartanWhirError>
where
    EF: Field + Send + Sync,
{
    if eq.len() != left.len()
        || left.len() != right.len()
        || eq.len() < 2
        || !eq.len().is_multiple_of(2)
    {
        return Err(SpartanWhirError::InvalidRoundPolynomial);
    }

    let half = eq.len() / 2;
    let (h0, h2, h3) = if should_parallelize_spark_round(half) {
        (0..half)
            .into_par_iter()
            .map(|i| compute_grand_product_layer_round_partial(i, half, eq, left, right))
            .par_fold_reduce(
                || (EF::ZERO, EF::ZERO, EF::ZERO),
                add_cubic_accumulators,
                add_cubic_accumulators,
            )
    } else {
        (0..half)
            .map(|i| compute_grand_product_layer_round_partial(i, half, eq, left, right))
            .fold((EF::ZERO, EF::ZERO, EF::ZERO), add_cubic_accumulators)
    };

    Ok(CubicRoundPoly([h0, h2, h3]))
}

fn compute_grand_product_layer_round_partial<EF>(
    i: usize,
    half: usize,
    eq: &[EF],
    left: &[EF],
    right: &[EF],
) -> (EF, EF, EF)
where
    EF: Field,
{
    let eq0 = eq[i];
    let eq1 = eq[i + half];
    let left0 = left[i];
    let left1 = left[i + half];
    let right0 = right[i];
    let right1 = right[i + half];
    let (eq2, eq3) = extrapolate_at_two_and_three(eq0, eq1);
    let (left2, left3) = extrapolate_at_two_and_three(left0, left1);
    let (right2, right3) = extrapolate_at_two_and_three(right0, right1);

    (
        eq0 * left0 * right0,
        eq2 * left2 * right2,
        eq3 * left3 * right3,
    )
}

fn observe_product_layer_and_sample<EF, C>(
    layer: &SparkGrandProductLayerProof<EF>,
    challenger: &mut C,
) -> EF
where
    EF: ExtensionField<F>,
    C: FieldChallenger<F>,
{
    challenger.observe_algebra_slice(&[layer.left_eval, layer.right_eval]);
    challenger.sample_algebra_element::<EF>()
}

fn eq_eval_at_point<EF>(eq_point: &[EF], point: &[EF]) -> Result<EF, SpartanWhirError>
where
    EF: Field,
{
    if eq_point.len() != point.len() {
        return Err(SpartanWhirError::InvalidRoundCount);
    }
    Ok(Point::<EF>::eval_eq(eq_point, point))
}

fn prove_axis_grand_products_with_leaf_claims<EF, C>(
    values: &SparkMemoryProductValues<EF>,
    gamma: EF,
    challenger: &mut C,
) -> Result<
    (
        SparkAxisGrandProductProof<EF>,
        SparkAxisGrandProductLeafClaims<EF>,
    ),
    SpartanWhirError,
>
where
    EF: ExtensionField<F> + Send + Sync,
    C: FieldChallenger<F>,
{
    let init_terms: Vec<EF> = values.init.iter().map(|&value| value - gamma).collect();
    let read_terms: Vec<EF> = values.read.iter().map(|&value| value - gamma).collect();
    let write_terms: Vec<EF> = values.write.iter().map(|&value| value - gamma).collect();
    let audit_terms: Vec<EF> = values.audit.iter().map(|&value| value - gamma).collect();

    let (init, init_claim) =
        prove_spark_grand_product_terms_with_leaf_claim(&init_terms, challenger)?;
    let (read, read_claim) =
        prove_spark_grand_product_terms_with_leaf_claim(&read_terms, challenger)?;
    let (write, write_claim) =
        prove_spark_grand_product_terms_with_leaf_claim(&write_terms, challenger)?;
    let (audit, audit_claim) =
        prove_spark_grand_product_terms_with_leaf_claim(&audit_terms, challenger)?;

    Ok((
        SparkAxisGrandProductProof {
            init,
            read,
            write,
            audit,
        },
        SparkAxisGrandProductLeafClaims {
            init: init_claim,
            read: read_claim,
            write: write_claim,
            audit: audit_claim,
        },
    ))
}

fn verify_axis_grand_product_claims<EF, C>(
    roots: &SparkMemoryProductClaim<EF>,
    proof: &SparkAxisGrandProductProof<EF>,
    value_domain_size: usize,
    memory_size: usize,
    challenger: &mut C,
) -> Result<SparkAxisGrandProductLeafClaims<EF>, SpartanWhirError>
where
    EF: ExtensionField<F>,
    C: FieldChallenger<F>,
{
    let value_rounds = log2_power_of_two(value_domain_size);
    let memory_rounds = log2_power_of_two(memory_size);

    let (init_point, init_eval) =
        verify_spark_grand_product(&proof.init, roots.init_root, memory_rounds, challenger)?;
    let (read_point, read_eval) =
        verify_spark_grand_product(&proof.read, roots.read_root, value_rounds, challenger)?;
    let (write_point, write_eval) =
        verify_spark_grand_product(&proof.write, roots.write_root, value_rounds, challenger)?;
    let (audit_point, audit_eval) =
        verify_spark_grand_product(&proof.audit, roots.audit_root, memory_rounds, challenger)?;

    Ok(SparkAxisGrandProductLeafClaims {
        init: SparkGrandProductLeafClaim {
            point: init_point,
            term_eval: init_eval,
        },
        read: SparkGrandProductLeafClaim {
            point: read_point,
            term_eval: read_eval,
        },
        write: SparkGrandProductLeafClaim {
            point: write_point,
            term_eval: write_eval,
        },
        audit: SparkGrandProductLeafClaim {
            point: audit_point,
            term_eval: audit_eval,
        },
    })
}

fn axis_product_claim<EF>(proof: &SparkAxisGrandProductProof<EF>) -> SparkMemoryProductClaim<EF>
where
    EF: Copy,
{
    SparkMemoryProductClaim {
        init_root: proof.init.root,
        read_root: proof.read.root,
        write_root: proof.write.root,
        audit_root: proof.audit.root,
    }
}

fn spark_memory_values_for_axes<EF>(
    tables: &SparkTables,
    r_x: &MultilinearPoint<EF>,
    r_y: &MultilinearPoint<EF>,
    read_tables: &SparkReadTables<EF>,
    beta: EF,
) -> Result<(SparkMemoryProductValues<EF>, SparkMemoryProductValues<EF>), SpartanWhirError>
where
    EF: ExtensionField<F> + Send + Sync,
{
    let row_values = spark_memory_product_values(
        tables.row_memory_size,
        &tables.rows,
        &tables.read_ts_row,
        &tables.audit_ts_row,
        &r_x.0,
        &read_tables.erow,
        beta,
    )?;
    let col_values = spark_memory_product_values(
        tables.col_memory_size,
        &tables.cols,
        &tables.read_ts_col,
        &tables.audit_ts_col,
        &r_y.0,
        &read_tables.ecol,
        beta,
    )?;
    Ok((row_values, col_values))
}

fn spark_ops_product_terms_for_axes<EF>(
    tables: &SparkTables,
    read_tables: &SparkReadTables<EF>,
    beta: EF,
    gamma: EF,
) -> Result<Vec<Vec<EF>>, SpartanWhirError>
where
    EF: ExtensionField<F> + Send + Sync,
{
    validate_read_tables(tables, read_tables)?;
    let (row_read, row_write) = spark_axis_ops_product_terms(
        tables.row_memory_size,
        &tables.rows,
        &tables.read_ts_row,
        &read_tables.erow,
        beta,
        gamma,
    )?;
    let (col_read, col_write) = spark_axis_ops_product_terms(
        tables.col_memory_size,
        &tables.cols,
        &tables.read_ts_col,
        &read_tables.ecol,
        beta,
        gamma,
    )?;
    Ok(vec![row_read, row_write, col_read, col_write])
}

fn spark_mem_product_terms_for_axes<EF>(
    tables: &SparkTables,
    r_x: &MultilinearPoint<EF>,
    r_y: &MultilinearPoint<EF>,
    beta: EF,
    gamma: EF,
    padded_memory_size: usize,
) -> Result<Vec<Vec<EF>>, SpartanWhirError>
where
    EF: ExtensionField<F> + Send + Sync,
{
    let (row_init, row_audit) = spark_axis_mem_product_terms(
        tables.row_memory_size,
        padded_memory_size,
        &tables.audit_ts_row,
        &r_x.0,
        beta,
        gamma,
    )?;
    let (col_init, col_audit) = spark_axis_mem_product_terms(
        tables.col_memory_size,
        padded_memory_size,
        &tables.audit_ts_col,
        &r_y.0,
        beta,
        gamma,
    )?;
    Ok(vec![row_init, row_audit, col_init, col_audit])
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
fn spark_axis_product_terms<EF>(
    memory_size: usize,
    padded_memory_size: usize,
    addrs: &[F],
    read_ts: &[F],
    audit_ts: &[F],
    point: &[EF],
    read_values: &[EF],
    beta: EF,
    gamma: EF,
) -> Result<SparkAxisProductTerms<EF>, SpartanWhirError>
where
    EF: ExtensionField<F> + Send + Sync,
{
    if memory_size == 0
        || !memory_size.is_power_of_two()
        || padded_memory_size < memory_size
        || !padded_memory_size.is_power_of_two()
        || addrs.len() != read_ts.len()
        || read_values.len() != addrs.len()
        || audit_ts.len() != memory_size
        || point.len() != log2_power_of_two(memory_size)
    {
        return Err(SpartanWhirError::InvalidR1csShape);
    }

    let (init, audit) = spark_axis_mem_product_terms(
        memory_size,
        padded_memory_size,
        audit_ts,
        point,
        beta,
        gamma,
    )?;
    let (read, write) =
        spark_axis_ops_product_terms(memory_size, addrs, read_ts, read_values, beta, gamma)?;

    Ok(SparkAxisProductTerms {
        init,
        read,
        write,
        audit,
    })
}

fn spark_axis_mem_product_terms<EF>(
    memory_size: usize,
    padded_memory_size: usize,
    audit_ts: &[F],
    point: &[EF],
    beta: EF,
    gamma: EF,
) -> Result<(Vec<EF>, Vec<EF>), SpartanWhirError>
where
    EF: ExtensionField<F> + Send + Sync,
{
    if memory_size == 0
        || !memory_size.is_power_of_two()
        || padded_memory_size < memory_size
        || !padded_memory_size.is_power_of_two()
        || audit_ts.len() != memory_size
        || point.len() != log2_power_of_two(memory_size)
    {
        return Err(SpartanWhirError::InvalidR1csShape);
    }
    let beta_squared = beta.square();
    let memory_values = EqPolynomial::evals_from_point_with_base::<F>(point);
    let mut init = vec![EF::ONE; padded_memory_size];
    let mut audit = vec![EF::ONE; padded_memory_size];
    if should_parallelize_spark_round(memory_size) {
        init[..memory_size]
            .par_iter_mut()
            .zip(audit[..memory_size].par_iter_mut())
            .enumerate()
            .try_for_each(|(index, (init, audit))| {
                let addr = EF::from(usize_to_field(index)?);
                let value = memory_values[index];
                *init = addr + beta * value - gamma;
                *audit = *init + beta_squared * EF::from(audit_ts[index]);
                Ok::<(), SpartanWhirError>(())
            })?;
    } else {
        for index in 0..memory_size {
            let addr = EF::from(usize_to_field(index)?);
            let value = memory_values[index];
            init[index] = addr + beta * value - gamma;
            audit[index] = init[index] + beta_squared * EF::from(audit_ts[index]);
        }
    }
    Ok((init, audit))
}

fn spark_axis_ops_product_terms<EF>(
    memory_size: usize,
    addrs: &[F],
    read_ts: &[F],
    read_values: &[EF],
    beta: EF,
    gamma: EF,
) -> Result<(Vec<EF>, Vec<EF>), SpartanWhirError>
where
    EF: ExtensionField<F> + Send + Sync,
{
    if memory_size == 0
        || !memory_size.is_power_of_two()
        || addrs.len() != read_ts.len()
        || read_values.len() != addrs.len()
    {
        return Err(SpartanWhirError::InvalidR1csShape);
    }
    let beta_squared = beta.square();
    let mut read = vec![EF::ZERO; addrs.len()];
    let mut write = vec![EF::ZERO; addrs.len()];
    if should_parallelize_spark_round(addrs.len()) {
        read.par_iter_mut()
            .zip(write.par_iter_mut())
            .zip(addrs.par_iter())
            .zip(read_ts.par_iter())
            .zip(read_values.par_iter())
            .try_for_each(|((((read, write), &addr), &timestamp), &value)| {
                if addr.as_canonical_u32() as usize >= memory_size {
                    return Err(SpartanWhirError::InvalidR1csShape);
                }
                *read = EF::from(addr) + beta * value + beta_squared * EF::from(timestamp) - gamma;
                *write = *read + beta_squared;
                Ok::<(), SpartanWhirError>(())
            })?;
    } else {
        for index in 0..addrs.len() {
            let addr = addrs[index];
            if addr.as_canonical_u32() as usize >= memory_size {
                return Err(SpartanWhirError::InvalidR1csShape);
            }
            read[index] = EF::from(addr)
                + beta * read_values[index]
                + beta_squared * EF::from(read_ts[index])
                - gamma;
            write[index] = read[index] + beta_squared;
        }
    }
    Ok((read, write))
}

fn spark_ops_product_terms<EF>(
    row: &SparkMemoryProductValues<EF>,
    col: &SparkMemoryProductValues<EF>,
    gamma: EF,
) -> Result<Vec<Vec<EF>>, SpartanWhirError>
where
    EF: Field,
{
    let domain_size = row.read.len();
    if domain_size == 0
        || !domain_size.is_power_of_two()
        || row.write.len() != domain_size
        || col.read.len() != domain_size
        || col.write.len() != domain_size
    {
        return Err(SpartanWhirError::InvalidPolynomialLength);
    }
    Ok(vec![
        product_terms_minus_gamma(&row.read, gamma),
        product_terms_minus_gamma(&row.write, gamma),
        product_terms_minus_gamma(&col.read, gamma),
        product_terms_minus_gamma(&col.write, gamma),
    ])
}

fn spark_value_dotproducts<EF>(
    tables: &SparkTables,
    read_tables: &SparkReadTables<EF>,
) -> Result<Vec<SparkDotProductCircuit<EF>>, SpartanWhirError>
where
    EF: ExtensionField<F> + Send + Sync,
{
    validate_read_tables(tables, read_tables)?;
    let domain_size = tables.value_domain_size;
    if domain_size < 2 || !domain_size.is_power_of_two() {
        return Err(SpartanWhirError::InvalidPolynomialLength);
    }
    let mut dotproducts = Vec::with_capacity(6);
    push_split_value_dotproducts(
        &mut dotproducts,
        &tables.val_a,
        &read_tables.erow,
        &read_tables.ecol,
    )?;
    push_split_value_dotproducts(
        &mut dotproducts,
        &tables.val_b,
        &read_tables.erow,
        &read_tables.ecol,
    )?;
    push_split_value_dotproducts(
        &mut dotproducts,
        &tables.val_c,
        &read_tables.erow,
        &read_tables.ecol,
    )?;
    Ok(dotproducts)
}

fn spark_base_value_dotproducts<EF>(
    tables: &SparkTables,
    read_tables: &SparkReadTables<EF>,
) -> Result<SparkProverDotProductBatch<EF>, SpartanWhirError>
where
    EF: ExtensionField<F>,
{
    validate_read_tables(tables, read_tables)?;
    let domain_size = tables.value_domain_size;
    if domain_size < 2 || !domain_size.is_power_of_two() {
        return Err(SpartanWhirError::InvalidPolynomialLength);
    }
    let half = domain_size / 2;
    let weights = [&tables.val_a, &tables.val_b, &tables.val_c]
        .into_iter()
        .flat_map(|values| {
            [
                SparkProverDotProductWeight::Base(values[..half].to_vec()),
                SparkProverDotProductWeight::Base(values[half..].to_vec()),
            ]
        })
        .collect();
    Ok(SparkProverDotProductBatch {
        lefts: vec![
            read_tables.erow[..half].to_vec(),
            read_tables.erow[half..].to_vec(),
        ],
        rights: vec![
            read_tables.ecol[..half].to_vec(),
            read_tables.ecol[half..].to_vec(),
        ],
        weights,
        pair_indices: vec![0, 1, 0, 1, 0, 1],
    })
}

fn push_split_value_dotproducts<EF>(
    dotproducts: &mut Vec<SparkDotProductCircuit<EF>>,
    vals: &[F],
    erow: &[EF],
    ecol: &[EF],
) -> Result<(), SpartanWhirError>
where
    EF: ExtensionField<F> + Send + Sync,
{
    if vals.len() != erow.len() || vals.len() != ecol.len() || vals.len() < 2 {
        return Err(SpartanWhirError::InvalidPolynomialLength);
    }
    let half = vals.len() / 2;
    if half == 0 || vals.len() != half * 2 {
        return Err(SpartanWhirError::InvalidPolynomialLength);
    }
    dotproducts.push(SparkDotProductCircuit {
        left: erow[..half].to_vec(),
        right: ecol[..half].to_vec(),
        weight: lift_base_values_to_extension(&vals[..half]),
    });
    dotproducts.push(SparkDotProductCircuit {
        left: erow[half..].to_vec(),
        right: ecol[half..].to_vec(),
        weight: lift_base_values_to_extension(&vals[half..]),
    });
    Ok(())
}

#[cfg(test)]
fn push_split_base_value_dotproducts<EF>(
    dotproducts: &mut Vec<SparkProverDotProductCircuit<EF>>,
    vals: &[F],
    erow: &[EF],
    ecol: &[EF],
) -> Result<(), SpartanWhirError>
where
    EF: ExtensionField<F>,
{
    if vals.len() != erow.len() || vals.len() != ecol.len() || vals.len() < 2 {
        return Err(SpartanWhirError::InvalidPolynomialLength);
    }
    let half = vals.len() / 2;
    if half == 0 || vals.len() != half * 2 {
        return Err(SpartanWhirError::InvalidPolynomialLength);
    }
    dotproducts.push(SparkProverDotProductCircuit {
        left: erow[..half].to_vec(),
        right: ecol[..half].to_vec(),
        weight: SparkProverDotProductWeight::Base(vals[..half].to_vec()),
    });
    dotproducts.push(SparkProverDotProductCircuit {
        left: erow[half..].to_vec(),
        right: ecol[half..].to_vec(),
        weight: SparkProverDotProductWeight::Base(vals[half..].to_vec()),
    });
    Ok(())
}

fn lift_base_values_to_extension<EF>(values: &[F]) -> Vec<EF>
where
    EF: ExtensionField<F> + Send + Sync,
{
    if should_parallelize_spark_round(values.len()) {
        values.par_iter().map(|&value| EF::from(value)).collect()
    } else {
        values.iter().map(|&value| EF::from(value)).collect()
    }
}

fn matrix_evals_from_split_dotproduct_claims<EF>(claims: &[EF]) -> Result<[EF; 3], SpartanWhirError>
where
    EF: Field,
{
    if claims.len() != 6 {
        return Err(SpartanWhirError::InvalidPolynomialLength);
    }
    Ok([
        claims[0] + claims[1],
        claims[2] + claims[3],
        claims[4] + claims[5],
    ])
}

fn spark_mem_product_terms<EF>(
    row: &SparkMemoryProductValues<EF>,
    col: &SparkMemoryProductValues<EF>,
    gamma: EF,
    domain_size: usize,
) -> Result<Vec<Vec<EF>>, SpartanWhirError>
where
    EF: Field,
{
    if domain_size == 0 || !domain_size.is_power_of_two() {
        return Err(SpartanWhirError::InvalidPolynomialLength);
    }
    Ok(vec![
        padded_product_terms_minus_gamma(&row.init, gamma, domain_size)?,
        padded_product_terms_minus_gamma(&row.audit, gamma, domain_size)?,
        padded_product_terms_minus_gamma(&col.init, gamma, domain_size)?,
        padded_product_terms_minus_gamma(&col.audit, gamma, domain_size)?,
    ])
}

fn product_terms_minus_gamma<EF>(values: &[EF], gamma: EF) -> Vec<EF>
where
    EF: Field + Send + Sync,
{
    if should_parallelize_spark_round(values.len()) {
        values.par_iter().map(|&value| value - gamma).collect()
    } else {
        values.iter().map(|&value| value - gamma).collect()
    }
}

fn padded_product_terms_minus_gamma<EF>(
    values: &[EF],
    gamma: EF,
    domain_size: usize,
) -> Result<Vec<EF>, SpartanWhirError>
where
    EF: Field + Send + Sync,
{
    if values.is_empty() || values.len() > domain_size {
        return Err(SpartanWhirError::InvalidPolynomialLength);
    }
    let mut terms = product_terms_minus_gamma(values, gamma);
    terms.resize(domain_size, EF::ONE);
    Ok(terms)
}

fn verify_batched_product_leaf_claims_with_terms_and_dotproducts<EF>(
    terms: &[Vec<EF>],
    dotproducts: &[SparkDotProductCircuit<EF>],
    claims: &SparkBatchedProductLeafClaims<EF>,
) -> Result<(), SpartanWhirError>
where
    EF: Field,
{
    if terms.len() != claims.product_evals.len() {
        return Err(SpartanWhirError::SumcheckFailed);
    }
    for (terms, &claim) in terms.iter().zip(&claims.product_evals) {
        if evaluate_mle_table(terms, &claims.product_point.0)? != claim {
            return Err(SpartanWhirError::SumcheckFailed);
        }
    }
    if dotproducts.is_empty() {
        if !claims.dotproduct_point.0.is_empty()
            || !claims.dotproduct_left_evals.is_empty()
            || !claims.dotproduct_right_evals.is_empty()
            || !claims.dotproduct_weight_evals.is_empty()
        {
            return Err(SpartanWhirError::SumcheckFailed);
        }
    } else {
        if dotproducts.len() != claims.dotproduct_left_evals.len()
            || dotproducts.len() != claims.dotproduct_right_evals.len()
            || dotproducts.len() != claims.dotproduct_weight_evals.len()
        {
            return Err(SpartanWhirError::SumcheckFailed);
        }
        for (index, dotproduct) in dotproducts.iter().enumerate() {
            if evaluate_mle_table(&dotproduct.left, &claims.dotproduct_point.0)?
                != claims.dotproduct_left_evals[index]
                || evaluate_mle_table(&dotproduct.right, &claims.dotproduct_point.0)?
                    != claims.dotproduct_right_evals[index]
                || evaluate_mle_table(&dotproduct.weight, &claims.dotproduct_point.0)?
                    != claims.dotproduct_weight_evals[index]
            {
                return Err(SpartanWhirError::SumcheckFailed);
            }
        }
    }
    Ok(())
}

fn spark_memory_product_values<EF>(
    memory_size: usize,
    addrs: &[F],
    read_ts: &[F],
    audit_ts: &[F],
    point: &[EF],
    read_values: &[EF],
    beta: EF,
) -> Result<SparkMemoryProductValues<EF>, SpartanWhirError>
where
    EF: ExtensionField<F> + Send + Sync,
{
    if memory_size == 0
        || !memory_size.is_power_of_two()
        || addrs.len() != read_ts.len()
        || read_values.len() != addrs.len()
        || audit_ts.len() != memory_size
        || point.len() != log2_power_of_two(memory_size)
    {
        return Err(SpartanWhirError::InvalidR1csShape);
    }

    let memory_values = EqPolynomial::evals_from_point_with_base::<F>(point);
    let init_audit = if should_parallelize_spark_round(memory_size) {
        (0..memory_size)
            .into_par_iter()
            .map(|index| {
                let addr = usize_to_field(index)?;
                let value = memory_values[index];
                Ok((
                    spark_memory_tuple_hash(addr, value, F::ZERO, beta),
                    spark_memory_tuple_hash(addr, value, audit_ts[index], beta),
                ))
            })
            .collect::<Result<Vec<_>, SpartanWhirError>>()?
    } else {
        (0..memory_size)
            .map(|index| {
                let addr = usize_to_field(index)?;
                let value = memory_values[index];
                Ok((
                    spark_memory_tuple_hash(addr, value, F::ZERO, beta),
                    spark_memory_tuple_hash(addr, value, audit_ts[index], beta),
                ))
            })
            .collect::<Result<Vec<_>, SpartanWhirError>>()?
    };
    let (init, audit) = init_audit.into_iter().unzip();

    let read_write = if should_parallelize_spark_round(addrs.len()) {
        addrs
            .par_iter()
            .zip(read_ts.par_iter())
            .zip(read_values.par_iter())
            .map(|((&addr, &read_timestamp), &value)| {
                let index = addr.as_canonical_u32() as usize;
                if index >= memory_size {
                    return Err(SpartanWhirError::InvalidR1csShape);
                }
                Ok((
                    spark_memory_tuple_hash(addr, value, read_timestamp, beta),
                    spark_memory_tuple_hash(addr, value, read_timestamp + F::ONE, beta),
                ))
            })
            .collect::<Result<Vec<_>, SpartanWhirError>>()?
    } else {
        addrs
            .iter()
            .zip(read_ts.iter())
            .zip(read_values.iter())
            .map(|((&addr, &read_timestamp), &value)| {
                let index = addr.as_canonical_u32() as usize;
                if index >= memory_size {
                    return Err(SpartanWhirError::InvalidR1csShape);
                }
                Ok((
                    spark_memory_tuple_hash(addr, value, read_timestamp, beta),
                    spark_memory_tuple_hash(addr, value, read_timestamp + F::ONE, beta),
                ))
            })
            .collect::<Result<Vec<_>, SpartanWhirError>>()?
    };
    let (read, write) = read_write.into_iter().unzip();

    Ok(SparkMemoryProductValues {
        init,
        read,
        write,
        audit,
    })
}

fn verify_axis_leaf_claims_with_tables<EF>(
    memory_size: usize,
    addrs: &[F],
    read_ts: &[F],
    audit_ts: &[F],
    read_values: &[EF],
    claims: &SparkAxisGrandProductLeafClaims<EF>,
    memory_point: &MultilinearPoint<EF>,
    beta: EF,
    gamma: EF,
) -> Result<(), SpartanWhirError>
where
    EF: ExtensionField<F>,
{
    if memory_size == 0
        || !memory_size.is_power_of_two()
        || addrs.len() != read_ts.len()
        || read_values.len() != addrs.len()
        || audit_ts.len() != memory_size
        || memory_point.0.len() != log2_power_of_two(memory_size)
    {
        return Err(SpartanWhirError::InvalidR1csShape);
    }

    let init_term = virtual_memory_term_eval(
        memory_size,
        &claims.init.point,
        memory_point,
        EF::ZERO,
        beta,
        gamma,
    )?;
    let read_term = read_memory_term_eval(
        addrs,
        read_values,
        read_ts,
        &claims.read.point,
        beta,
        gamma,
        false,
    )?;
    let write_term = read_memory_term_eval(
        addrs,
        read_values,
        read_ts,
        &claims.write.point,
        beta,
        gamma,
        true,
    )?;
    let audit_ts_eval = evaluate_base_table_as_extension(audit_ts, &claims.audit.point.0)?;
    let audit_term = virtual_memory_term_eval(
        memory_size,
        &claims.audit.point,
        memory_point,
        audit_ts_eval,
        beta,
        gamma,
    )?;

    if claims.init.term_eval != init_term
        || claims.read.term_eval != read_term
        || claims.write.term_eval != write_term
        || claims.audit.term_eval != audit_term
    {
        return Err(SpartanWhirError::SumcheckFailed);
    }
    Ok(())
}

fn virtual_memory_term_eval<EF>(
    memory_size: usize,
    point: &MultilinearPoint<EF>,
    memory_point: &MultilinearPoint<EF>,
    timestamp_eval: EF,
    beta: EF,
    gamma: EF,
) -> Result<EF, SpartanWhirError>
where
    EF: ExtensionField<F>,
{
    if point.0.len() != log2_power_of_two(memory_size) {
        return Err(SpartanWhirError::InvalidRoundCount);
    }
    let addr_eval = virtual_index_eval(memory_size, &point.0)?;
    let value_eval = eq_eval_at_point(&memory_point.0, &point.0)?;
    Ok(spark_memory_tuple_hash_eval(addr_eval, value_eval, timestamp_eval, beta) - gamma)
}

fn padded_virtual_memory_term_eval<EF>(
    memory_size: usize,
    padded_domain_size: usize,
    padded_point: &MultilinearPoint<EF>,
    memory_point: &MultilinearPoint<EF>,
    timestamp_eval: EF,
    beta: EF,
    gamma: EF,
) -> Result<EF, SpartanWhirError>
where
    EF: ExtensionField<F>,
{
    if memory_size == 0
        || !memory_size.is_power_of_two()
        || padded_domain_size < memory_size
        || !padded_domain_size.is_power_of_two()
        || padded_point.0.len() != log2_power_of_two(padded_domain_size)
    {
        return Err(SpartanWhirError::InvalidRoundCount);
    }
    let memory_bits = log2_power_of_two(memory_size);
    let prefix_len = padded_point
        .0
        .len()
        .checked_sub(memory_bits)
        .ok_or(SpartanWhirError::InvalidRoundCount)?;
    let low_block_weight = padded_point.0[..prefix_len]
        .iter()
        .fold(EF::ONE, |acc, &coord| acc * (EF::ONE - coord));
    let real_point = MultilinearPoint(padded_point.0[prefix_len..].to_vec());
    let real_term = virtual_memory_term_eval(
        memory_size,
        &real_point,
        memory_point,
        timestamp_eval,
        beta,
        gamma,
    )?;
    Ok(low_block_weight * real_term + (EF::ONE - low_block_weight))
}

fn read_memory_term_eval<EF>(
    addrs: &[F],
    read_values: &[EF],
    read_ts: &[F],
    point: &MultilinearPoint<EF>,
    beta: EF,
    gamma: EF,
    write: bool,
) -> Result<EF, SpartanWhirError>
where
    EF: ExtensionField<F>,
{
    if addrs.len() != read_ts.len()
        || read_values.len() != addrs.len()
        || addrs.is_empty()
        || !addrs.len().is_power_of_two()
    {
        return Err(SpartanWhirError::InvalidR1csShape);
    }
    let addr_eval = evaluate_base_table_as_extension(addrs, &point.0)?;
    let value_eval = evaluate_mle_table(read_values, &point.0)?;
    let mut timestamp_eval = evaluate_base_table_as_extension(read_ts, &point.0)?;
    if write {
        timestamp_eval += EF::ONE;
    }
    Ok(spark_memory_tuple_hash_eval(addr_eval, value_eval, timestamp_eval, beta) - gamma)
}

fn spark_memory_product_claim<EF>(
    _axis: SparkMemoryAxis,
    memory_size: usize,
    addrs: &[F],
    read_ts: &[F],
    audit_ts: &[F],
    point: &[EF],
    read_values: &[EF],
    beta: EF,
    gamma: EF,
) -> Result<SparkMemoryProductClaim<EF>, SpartanWhirError>
where
    EF: ExtensionField<F>,
{
    if memory_size == 0
        || !memory_size.is_power_of_two()
        || addrs.len() != read_ts.len()
        || read_values.len() != addrs.len()
        || audit_ts.len() != memory_size
        || point.len() != log2_power_of_two(memory_size)
    {
        return Err(SpartanWhirError::InvalidR1csShape);
    }

    let memory_values = EqPolynomial::evals_from_point_with_base::<F>(point);
    let mut init_root = EF::ONE;
    let mut audit_root = EF::ONE;
    for index in 0..memory_size {
        let addr = usize_to_field(index)?;
        let value = memory_values[index];
        init_root *= spark_memory_tuple_hash(addr, value, F::ZERO, beta) - gamma;
        audit_root *= spark_memory_tuple_hash(addr, value, audit_ts[index], beta) - gamma;
    }

    let mut read_root = EF::ONE;
    let mut write_root = EF::ONE;
    for ((&addr, &read_timestamp), &value) in addrs.iter().zip(read_ts).zip(read_values) {
        let index = addr.as_canonical_u32() as usize;
        if index >= memory_size {
            return Err(SpartanWhirError::InvalidR1csShape);
        }
        let write_timestamp = read_timestamp + F::ONE;
        read_root *= spark_memory_tuple_hash(addr, value, read_timestamp, beta) - gamma;
        write_root *= spark_memory_tuple_hash(addr, value, write_timestamp, beta) - gamma;
    }

    Ok(SparkMemoryProductClaim {
        init_root,
        read_root,
        write_root,
        audit_root,
    })
}

fn spark_memory_tuple_hash<EF>(addr: F, value: EF, timestamp: F, beta: EF) -> EF
where
    EF: ExtensionField<F>,
{
    spark_memory_tuple_hash_eval(EF::from(addr), value, EF::from(timestamp), beta)
}

fn spark_memory_tuple_hash_eval<EF>(addr: EF, value: EF, timestamp: EF, beta: EF) -> EF
where
    EF: Field,
{
    // Same Schwartz-Zippel memory tuple compression as SPARK's
    // address/value/timestamp polynomial, with `beta` used as the per-proof
    // combiner and a coefficient order chosen to match the local transcript.
    addr + beta * value + beta * beta * timestamp
}

fn check_spark_memory_product_claim<EF>(
    claim: &SparkMemoryProductClaim<EF>,
) -> Result<(), SpartanWhirError>
where
    EF: Field,
{
    if claim.init_root * claim.write_root != claim.read_root * claim.audit_root {
        return Err(SpartanWhirError::SumcheckFailed);
    }
    Ok(())
}

fn sum_value_product<EF>(
    selector: &[EF],
    vals: &[EF],
    erow: &[EF],
    ecol: &[EF],
) -> Result<EF, SpartanWhirError>
where
    EF: Field,
{
    if selector.len() != vals.len()
        || vals.len() != erow.len()
        || erow.len() != ecol.len()
        || selector.is_empty()
        || !selector.len().is_power_of_two()
    {
        return Err(SpartanWhirError::InvalidRoundPolynomial);
    }
    Ok(selector
        .iter()
        .zip(vals)
        .zip(erow)
        .zip(ecol)
        .fold(EF::ZERO, |acc, (((&s, &v), &row), &col)| {
            acc + s * v * row * col
        }))
}

fn compute_value_round<EF>(
    selector: &[EF],
    vals: &[EF],
    erow: &[EF],
    ecol: &[EF],
    include_selector: bool,
) -> Result<SparkValueRoundPoly<EF>, SpartanWhirError>
where
    EF: Field,
{
    if selector.len() != vals.len()
        || vals.len() != erow.len()
        || erow.len() != ecol.len()
        || selector.len() < 2
        || !selector.len().is_multiple_of(2)
    {
        return Err(SpartanWhirError::InvalidRoundPolynomial);
    }

    let half = selector.len() / 2;
    let mut h0 = EF::ZERO;
    let mut h2 = EF::ZERO;
    let mut h3 = EF::ZERO;
    let mut h4 = EF::ZERO;
    for i in 0..half {
        let s0 = selector[i];
        let s1 = selector[i + half];
        let v0 = vals[i];
        let v1 = vals[i + half];
        let row0 = erow[i];
        let row1 = erow[i + half];
        let col0 = ecol[i];
        let col1 = ecol[i + half];

        h0 += s0 * v0 * row0 * col0;
        let h2_term = extrapolate(v0, v1, EF::TWO)
            * extrapolate(row0, row1, EF::TWO)
            * extrapolate(col0, col1, EF::TWO);
        let h3_term = extrapolate(v0, v1, EF::from_u32(3))
            * extrapolate(row0, row1, EF::from_u32(3))
            * extrapolate(col0, col1, EF::from_u32(3));
        h2 += if include_selector {
            extrapolate(s0, s1, EF::TWO) * h2_term
        } else {
            h2_term
        };
        h3 += if include_selector {
            extrapolate(s0, s1, EF::from_u32(3)) * h3_term
        } else {
            h3_term
        };
        if include_selector {
            h4 += extrapolate(s0, s1, EF::from_u32(4))
                * extrapolate(v0, v1, EF::from_u32(4))
                * extrapolate(row0, row1, EF::from_u32(4))
                * extrapolate(col0, col1, EF::from_u32(4));
        }
    }

    let mut evals = vec![h0, h2, h3];
    if include_selector {
        evals.push(h4);
    }
    Ok(SparkValueRoundPoly(evals))
}

fn bind_value_table<EF>(table: &mut Vec<EF>, r: EF) -> Result<(), SpartanWhirError>
where
    EF: Field + Send + Sync,
{
    if table.len() < 2 || !table.len().is_multiple_of(2) {
        return Err(SpartanWhirError::InvalidRoundPolynomial);
    }

    let half = table.len() / 2;
    {
        let (lo, hi) = table.split_at_mut(half);
        if should_parallelize_spark_round(half) {
            lo.par_iter_mut()
                .zip(hi.par_iter())
                .for_each(|(lo, &hi)| *lo = extrapolate(*lo, hi, r));
        } else {
            for (lo, &hi) in lo.iter_mut().zip(hi.iter()) {
                *lo = extrapolate(*lo, hi, r);
            }
        }
    }
    table.truncate(half);
    Ok(())
}

fn extrapolate<EF: Field>(lo: EF, hi: EF, r: EF) -> EF {
    lo + r * (hi - lo)
}

fn extrapolate_at_two_and_three<EF: Field>(lo: EF, hi: EF) -> (EF, EF) {
    let delta = hi - lo;
    let at_two = hi + delta;
    let at_three = at_two + delta;
    (at_two, at_three)
}

fn add_cubic_accumulators<EF: Field>(
    (a0, a2, a3): (EF, EF, EF),
    (b0, b2, b3): (EF, EF, EF),
) -> (EF, EF, EF) {
    (a0 + b0, a2 + b2, a3 + b3)
}

fn should_parallelize_spark_round(pair_count: usize) -> bool {
    cfg!(feature = "parallel") && pair_count >= SPARK_PARALLEL_ROUND_MIN_PAIRS
}

fn interpolate_at<EF: Field>(ys: &[EF], r: EF) -> EF {
    let mut out = EF::ZERO;
    for i in 0..ys.len() {
        let x_i = EF::from_u32(i as u32);
        let mut num = EF::ONE;
        let mut den = EF::ONE;
        for j in 0..ys.len() {
            if i == j {
                continue;
            }
            let x_j = EF::from_u32(j as u32);
            num *= r - x_j;
            den *= x_i - x_j;
        }
        out += ys[i] * num * den.inverse();
    }
    out
}

fn padded_nonzero_domain(nnz: usize) -> Result<usize, SpartanWhirError> {
    if nnz == 0 {
        Ok(1)
    } else {
        nnz.checked_next_power_of_two()
            .ok_or(SpartanWhirError::InvalidR1csShape)
    }
}

fn usize_to_field(value: usize) -> Result<F, SpartanWhirError> {
    if value >= F::ORDER_U32 as usize {
        return Err(SpartanWhirError::InvalidR1csShape);
    }
    Ok(F::from_u32(value as u32))
}

fn log2_power_of_two(value: usize) -> usize {
    debug_assert!(value.is_power_of_two());
    value.ilog2() as usize
}

fn ratio_ppm(num: usize, den: usize) -> u64 {
    if den == 0 {
        return 0;
    }
    ((num as u128 * 1_000_000u128) / den as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OcticBinExtension, SparseMatEntry};
    use p3_challenger::CanObserve;
    use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};

    fn entry(row: usize, col: usize, val: u32) -> SparseMatEntry<F> {
        SparseMatEntry {
            row,
            col,
            val: F::from_u32(val),
        }
    }

    #[test]
    fn equality_evaluation_matches_the_materialized_table() {
        type EF = OcticBinExtension;

        let value = |seed: usize| {
            EF::from_basis_coefficients_fn(|coordinate| {
                F::from_u32((19 * seed + 11 * coordinate + 1) as u32)
            })
        };

        for dimensions in 0..=8 {
            let eq_point = (0..dimensions).map(value).collect::<Vec<_>>();
            let point = (0..dimensions)
                .map(|index| value(32 + index))
                .collect::<Vec<_>>();
            let scalar_table = EqPolynomial::evals_from_point(&eq_point);

            assert_eq!(
                EqPolynomial::evals_from_point_with_base::<F>(&eq_point),
                scalar_table
            );
            assert_eq!(
                eq_eval_at_point(&eq_point, &point),
                evaluate_mle_table(&scalar_table, &point)
            );
        }

        assert_eq!(
            eq_eval_at_point::<EF>(&[EF::ONE], &[]),
            Err(SpartanWhirError::InvalidRoundCount)
        );
    }

    fn shape_with_entries(
        rows: usize,
        cols: usize,
        a: Vec<SparseMatEntry<F>>,
        b: Vec<SparseMatEntry<F>>,
        c: Vec<SparseMatEntry<F>>,
    ) -> R1csShape<F> {
        R1csShape {
            num_cons: rows,
            num_vars: cols - 1,
            num_io: 0,
            a: SparseMatrix {
                num_rows: rows,
                num_cols: cols,
                entries: a,
            },
            b: SparseMatrix {
                num_rows: rows,
                num_cols: cols,
                entries: b,
            },
            c: SparseMatrix {
                num_rows: rows,
                num_cols: cols,
                entries: c,
            },
        }
    }

    #[test]
    fn compact_product_rounds_cover_equality_factor_edge_cases() {
        type EF = crate::QuinticExtension;
        let q1 = EF::from_basis_coefficients_fn(|i| F::from_u32(7 + 11 * i as u32));
        let q2 = EF::from_basis_coefficients_fn(|i| F::from_u32(13 + 5 * i as u32));
        let claim = EF::from_basis_coefficients_fn(|i| F::from_u32(19 + 3 * i as u32));
        let generic = EF::from_basis_coefficients_fn(|i| F::from_u32(23 + 17 * i as u32));
        for coordinate in [EF::ZERO, EF::ONE, EF::TWO.inverse(), generic] {
            let slope = coordinate.double() - EF::ONE;
            let inverse_slope = if slope == EF::ZERO {
                EF::ONE
            } else {
                slope.inverse()
            };
            let round = decompress_spark_product_round([q1, q2], claim, coordinate);
            assert_eq!(
                compress_spark_product_round(&round, claim, coordinate, inverse_slope),
                Ok([q1, q2])
            );
            for point in [EF::ZERO, EF::ONE, EF::TWO, generic] {
                assert_eq!(
                    evaluate_spark_cubic_for_codec(&round, claim, point),
                    round.evaluate_at(point, claim)
                );
            }
            let mut malformed = round.clone();
            malformed.0[0] += EF::ONE;
            assert_eq!(
                compress_spark_product_round(&malformed, claim, coordinate, inverse_slope),
                Err(SpartanWhirError::InvalidRoundPolynomial)
            );
        }
    }

    #[test]
    fn compact_memory_products_restore_proof_and_transcript() {
        type EF = crate::QuinticExtension;
        let shape = shape_with_entries(
            16,
            16,
            (0..16).map(|i| entry(i, i, i as u32 + 1)).collect(),
            (0..16)
                .map(|i| entry(i, (i + 1) % 16, i as u32 + 3))
                .collect(),
            (0..16)
                .map(|i| entry(i, (i + 2) % 16, i as u32 + 7))
                .collect(),
        );
        let tables = preprocess_shared_union_spark_tables(&shape).expect("tables");
        let metadata = tables.metadata();
        let point = |size: usize, offset: u32| {
            MultilinearPoint(
                (0..size.ilog2())
                    .map(|i| {
                        EF::from_basis_coefficients_fn(|j| {
                            F::from_u32(offset + i * 11 + j as u32 * 7)
                        })
                    })
                    .collect(),
            )
        };
        let r_x = point(tables.row_memory_size, 17);
        let r_y = point(tables.col_memory_size, 31);
        let mut initial = crate::poseidon1_challenger();
        initial.observe(F::from_u32(123));
        let mut prover = initial.clone().with_trace();
        let proof =
            prove_spark_batched_memory_products(&tables, &r_x, &r_y, &mut prover).expect("proof");
        let mut encoder = initial.clone().with_trace();
        let compact =
            compress_spark_memory_products(&proof, &metadata, &mut encoder).expect("compress");
        let mut decoder = initial.clone().with_trace();
        let restored = decompress_spark_memory_products(&compact, &metadata, &mut decoder)
            .expect("decompress");
        assert_eq!(restored, proof);
        assert_eq!(encoder.transcript_trace(), prover.transcript_trace());
        assert_eq!(decoder.transcript_trace(), prover.transcript_trace());
        let next = prover.sample_algebra_element::<EF>();
        assert_eq!(encoder.sample_algebra_element::<EF>(), next);
        assert_eq!(decoder.sample_algebra_element::<EF>(), next);
        let mut verifier = initial.clone().with_trace();
        verify_spark_batched_memory_products_with_tables(
            &tables,
            &restored,
            &r_x,
            &r_y,
            &mut verifier,
        )
        .expect("restored proof verifies");
        assert_eq!(verifier.sample_algebra_element::<EF>(), next);
        assert_eq!(verifier.transcript_trace(), prover.transcript_trace());

        let pure_rounds = compact
            .proof_ops
            .layers
            .iter()
            .chain(&compact.proof_mem.layers)
            .map(|layer| match &layer.rounds {
                CompactSparkProductRounds::Factored(rounds) => rounds.len(),
                CompactSparkProductRounds::Cubic(_) => 0,
            })
            .sum::<usize>();
        let expected_saving = pure_rounds * <EF as BasedVectorSpace<F>>::DIMENSION * 4
            - 4 * (proof.proof_ops.layers.len() + proof.proof_mem.layers.len())
            - 4;
        assert_eq!(
            bincode::serialize(&proof)
                .expect("serialize ordinary")
                .len()
                - bincode::serialize(&compact)
                    .expect("serialize compact")
                    .len(),
            expected_saving
        );

        let mut refined = compact.clone();
        refine_spark_memory_products(&mut refined).expect("refine");
        assert!(refined.omits_derived_fields());
        let compact_bytes = bincode::serialize(&compact).expect("compact bytes");
        let refined_bytes = bincode::serialize(&refined).expect("refined bytes");
        assert_eq!(
            compact_bytes.len() - refined_bytes.len(),
            21 * <EF as BasedVectorSpace<F>>::DIMENSION * 4
        );
        let mut refined: CompactSparkBatchedMemoryProductsProof<EF> =
            bincode::deserialize(&refined_bytes).expect("deserialize refined");
        refine_spark_memory_products(&mut refined).expect("idempotent refinement");
        assert_eq!(
            bincode::serialize(&refined).expect("serialize"),
            refined_bytes
        );
        let mut refined_decoder = initial.clone().with_trace();
        let refined_restored =
            decompress_spark_memory_products(&refined, &metadata, &mut refined_decoder)
                .expect("restore refined");
        assert_eq!(refined_restored, proof);
        assert_eq!(refined_decoder.sample_algebra_element::<EF>(), next);
        assert_eq!(
            refined_decoder.transcript_trace(),
            prover.transcript_trace()
        );
        verify_spark_batched_memory_products_with_tables(
            &tables,
            &refined_restored,
            &r_x,
            &r_y,
            &mut initial.clone(),
        )
        .expect("refined proof verifies");

        let mut bad_pair = compact.clone();
        bad_pair
            .proof_ops
            .layers
            .last_mut()
            .expect("leaf")
            .dotproduct_left_evals[2] += EF::ONE;
        let unchanged = bad_pair.clone();
        assert!(refine_spark_memory_products(&mut bad_pair).is_err());
        assert_eq!(bad_pair, unchanged);
        let mut malformed_refined = refined.clone();
        malformed_refined
            .proof_ops
            .layers
            .last_mut()
            .expect("leaf")
            .dotproduct_right_evals
            .push(EF::ZERO);
        assert!(decompress_spark_memory_products(
            &malformed_refined,
            &metadata,
            &mut initial.clone(),
        )
        .is_err());
        let mut altered_refined = refined.clone();
        altered_refined
            .proof_ops
            .layers
            .last_mut()
            .expect("leaf")
            .dotproduct_left_evals[0] += EF::ONE;
        let altered_restored =
            decompress_spark_memory_products(&altered_refined, &metadata, &mut initial.clone())
                .expect("decode altered pair");
        assert!(verify_spark_batched_memory_products_with_tables(
            &tables,
            &altered_restored,
            &r_x,
            &r_y,
            &mut initial.clone(),
        )
        .is_err());

        let mut changed = compact.clone();
        let CompactSparkProductRounds::Factored(rounds) = &mut changed.proof_ops.layers[1].rounds
        else {
            panic!("internal product layer is factored");
        };
        rounds[0][0] += EF::ONE;
        let changed = decompress_spark_memory_products(&changed, &metadata, &mut initial.clone())
            .expect("coefficient mutation decodes to a different proof");
        assert_ne!(changed, proof);
        assert!(verify_spark_batched_memory_products_with_tables(
            &tables,
            &changed,
            &r_x,
            &r_y,
            &mut initial.clone(),
        )
        .is_err());

        let mut wrong_kind = compact.clone();
        wrong_kind.proof_ops.layers.last_mut().expect("leaf").rounds =
            CompactSparkProductRounds::Factored(vec![
                [EF::ZERO; 2];
                proof.proof_ops.layers.len() - 1
            ]);
        assert!(
            decompress_spark_memory_products(&wrong_kind, &metadata, &mut initial.clone(),)
                .is_err()
        );
        let mut wrong_count = compact.clone();
        wrong_count.proof_mem.layers.pop();
        assert!(
            decompress_spark_memory_products(&wrong_count, &metadata, &mut initial.clone(),)
                .is_err()
        );
        let mut wrong_context = metadata;
        wrong_context.layout = SparkLayoutKind::Joint;
        assert!(
            decompress_spark_memory_products(&compact, &wrong_context, &mut initial.clone(),)
                .is_err()
        );
        // The refined header has no challenge copies to compare. A different
        // transcript context must still fail ordinary proof verification.
        let wrong_context_proof =
            decompress_spark_memory_products(&refined, &wrong_context, &mut initial.clone())
                .expect("decode under another structurally valid context");
        assert!(verify_spark_batched_memory_product_claims_with_metadata(
            &wrong_context,
            &wrong_context_proof,
            &mut initial.clone(),
        )
        .is_err());
        let mut wrong_root = refined;
        wrong_root.proof_ops.product_roots[0] += EF::ONE;
        let wrong_root_proof =
            decompress_spark_memory_products(&wrong_root, &metadata, &mut initial.clone())
                .expect("decode altered tree root");
        assert!(verify_spark_batched_memory_product_claims_with_metadata(
            &metadata,
            &wrong_root_proof,
            &mut initial.clone(),
        )
        .is_err());
    }

    #[test]
    fn fixed_point_extrapolation_matches_generic_path() {
        for (lo, hi) in [
            (F::ZERO, F::ONE),
            (F::from_u32(2), F::from_u32(7)),
            (F::from_u32(17), F::from_u32(3)),
        ] {
            let (at_two, at_three) = extrapolate_at_two_and_three(lo, hi);
            assert_eq!(at_two, extrapolate(lo, hi, F::TWO));
            assert_eq!(at_three, extrapolate(lo, hi, F::from_u32(3)));
        }
    }

    #[test]
    fn factored_product_round_matches_unfactored_round() {
        type EF = OcticBinExtension;

        let value = |seed: usize| {
            EF::from_basis_coefficients_fn(|coordinate| {
                F::from_u32((23 * seed + 7 * coordinate + 1) as u32)
            })
        };
        let eq = (0..4).map(|index| value(10 + index)).collect::<Vec<_>>();
        let lefts = (0..4)
            .map(|circuit| {
                (0..4)
                    .map(|index| value(100 + 4 * circuit + index))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let rights = (0..4)
            .map(|circuit| {
                (0..4)
                    .map(|index| value(200 + 4 * circuit + index))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let coeffs = (0..4).map(|index| value(300 + index)).collect::<Vec<_>>();

        let optimized =
            compute_batched_product_round(&eq, &lefts, &rights, &[], &[], &[], &[], &[], &coeffs)
                .expect("optimized round computes");

        let half = eq.len() / 2;
        let reference = (0..half)
            .map(|i| {
                let (eq2, eq3) = extrapolate_at_two_and_three(eq[i], eq[i + half]);
                let mut h0 = EF::ZERO;
                let mut h2 = EF::ZERO;
                let mut h3 = EF::ZERO;
                for circuit in 0..lefts.len() {
                    let (left2, left3) =
                        extrapolate_at_two_and_three(lefts[circuit][i], lefts[circuit][i + half]);
                    let (right2, right3) =
                        extrapolate_at_two_and_three(rights[circuit][i], rights[circuit][i + half]);
                    h0 += coeffs[circuit] * eq[i] * lefts[circuit][i] * rights[circuit][i];
                    h2 += coeffs[circuit] * eq2 * left2 * right2;
                    h3 += coeffs[circuit] * eq3 * left3 * right3;
                }
                (h0, h2, h3)
            })
            .fold((EF::ZERO, EF::ZERO, EF::ZERO), add_cubic_accumulators);
        assert_eq!(
            optimized,
            CubicRoundPoly([reference.0, reference.1, reference.2])
        );

        let child_layers = lefts
            .iter()
            .zip(&rights)
            .map(|(left, right)| {
                left.iter()
                    .zip(right)
                    .flat_map(|(&left, &right)| [left, right])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let child_slices = child_layers.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let interleaved = compute_batched_product_round_from_child_layers(
            &eq,
            &child_slices,
            &[],
            &[],
            &[],
            &[],
            &[],
            &coeffs,
        )
        .expect("interleaved round computes");
        assert_eq!(interleaved, optimized);
    }

    #[test]
    fn direct_product_terms_match_reference_construction() {
        type EF = OcticBinExtension;

        let memory_size = 4;
        let padded_memory_size = 8;
        let addrs = [0, 1, 2, 3, 2, 0, 3, 1].map(F::from_u32);
        let read_ts = [0, 0, 0, 0, 1, 1, 1, 1].map(F::from_u32);
        let audit_ts = [2, 2, 2, 2].map(F::from_u32);
        let point = [EF::from(F::from_u32(5)), EF::from(F::from_u32(9))];
        let read_values = (0..addrs.len())
            .map(|index| EF::from(F::from_u32((3 * index + 7) as u32)))
            .collect::<Vec<_>>();
        let beta = EF::from(F::from_u32(13));
        let gamma = EF::from(F::from_u32(17));

        let reference = spark_memory_product_values(
            memory_size,
            &addrs,
            &read_ts,
            &audit_ts,
            &point,
            &read_values,
            beta,
        )
        .expect("reference terms build");
        let direct = spark_axis_product_terms(
            memory_size,
            padded_memory_size,
            &addrs,
            &read_ts,
            &audit_ts,
            &point,
            &read_values,
            beta,
            gamma,
        )
        .expect("direct terms build");

        assert_eq!(
            direct.read,
            product_terms_minus_gamma(&reference.read, gamma)
        );
        assert_eq!(
            direct.write,
            product_terms_minus_gamma(&reference.write, gamma)
        );
        assert_eq!(
            direct.init,
            padded_product_terms_minus_gamma(&reference.init, gamma, padded_memory_size)
                .expect("reference init pads")
        );
        assert_eq!(
            direct.audit,
            padded_product_terms_minus_gamma(&reference.audit, gamma, padded_memory_size)
                .expect("reference audit pads")
        );

        let reference_terms = vec![
            product_terms_minus_gamma(&reference.read, gamma),
            product_terms_minus_gamma(&reference.write, gamma),
            padded_product_terms_minus_gamma(&reference.init, gamma, padded_memory_size)
                .expect("reference init pads"),
            padded_product_terms_minus_gamma(&reference.audit, gamma, padded_memory_size)
                .expect("reference audit pads"),
        ];
        let direct_terms = vec![direct.read, direct.write, direct.init, direct.audit];
        let mut reference_challenger = crate::poseidon_challenger();
        let reference_proof = prove_spark_batched_product_owned(
            reference_terms,
            Vec::new(),
            &mut reference_challenger,
        )
        .expect("reference product proof succeeds");
        let mut direct_challenger = crate::poseidon_challenger();
        let direct_proof =
            prove_spark_batched_product_owned(direct_terms, Vec::new(), &mut direct_challenger)
                .expect("direct product proof succeeds");

        assert_eq!(direct_proof, reference_proof);
        assert_eq!(
            direct_challenger.sample_algebra_element::<EF>(),
            reference_challenger.sample_algebra_element::<EF>()
        );
    }

    #[test]
    fn base_weight_first_round_preserves_proof_and_transcript() {
        type EF = OcticBinExtension;

        let value = |seed: usize| {
            EF::from_basis_coefficients_fn(|coefficient| {
                F::from_u32((17 * seed + 5 * coefficient + 1) as u32)
            })
        };
        let product_terms = (0..4)
            .map(|product| {
                (0..32)
                    .map(|index| value(32 * product + index))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let base_dotproducts = (0..6)
            .map(|dotproduct| {
                let weight = (0..16)
                    .map(|index| F::from_u32((11 * dotproduct + 3 * index + 2) as u32))
                    .collect::<Vec<_>>();
                SparkProverDotProductCircuit {
                    left: (0..16)
                        .map(|index| value(200 + 32 * dotproduct + index))
                        .collect(),
                    right: (0..16)
                        .map(|index| value(216 + 32 * dotproduct + index))
                        .collect(),
                    weight: SparkProverDotProductWeight::Base(weight),
                }
            })
            .collect::<Vec<_>>();
        let extension_dotproducts = base_dotproducts
            .iter()
            .map(|dotproduct| SparkProverDotProductCircuit {
                left: dotproduct.left.clone(),
                right: dotproduct.right.clone(),
                weight: match &dotproduct.weight {
                    SparkProverDotProductWeight::Base(weight) => {
                        SparkProverDotProductWeight::Extension(
                            weight.iter().copied().map(EF::from).collect(),
                        )
                    }
                    SparkProverDotProductWeight::Extension(_) => unreachable!(),
                },
            })
            .collect();

        let mut base_challenger = crate::poseidon_challenger();
        let base = prove_spark_batched_product_owned(
            product_terms.clone(),
            base_dotproducts,
            &mut base_challenger,
        )
        .expect("base-weight product proof succeeds");
        let mut extension_challenger = crate::poseidon_challenger();
        let extension = prove_spark_batched_product_owned(
            product_terms,
            extension_dotproducts,
            &mut extension_challenger,
        )
        .expect("extension-weight product proof succeeds");

        assert_eq!(base, extension);
        assert_eq!(
            base_challenger.sample_algebra_element::<EF>(),
            extension_challenger.sample_algebra_element::<EF>()
        );
    }

    #[test]
    fn shared_read_dotproducts_preserve_proof_and_transcript() {
        type EF = OcticBinExtension;

        let extension_value = |seed: usize| {
            EF::from_basis_coefficients_fn(|coordinate| {
                F::from_u32((19 * seed + 7 * coordinate + 3) as u32)
            })
        };
        let product_terms = (0..4)
            .map(|product| {
                (0..32)
                    .map(|index| extension_value(32 * product + index))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let erow = (0..32)
            .map(|index| extension_value(200 + index))
            .collect::<Vec<_>>();
        let ecol = (0..32)
            .map(|index| extension_value(300 + index))
            .collect::<Vec<_>>();
        let matrix_weights = (0..3)
            .map(|matrix| {
                (0..32)
                    .map(|index| F::from_u32((13 * matrix + 5 * index + 1) as u32))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let mut independent = Vec::with_capacity(6);
        for weights in &matrix_weights {
            push_split_base_value_dotproducts(&mut independent, weights, &erow, &ecol)
                .expect("independent dotproducts build");
        }
        let half = erow.len() / 2;
        let shared = SparkProverDotProductBatch {
            lefts: vec![erow[..half].to_vec(), erow[half..].to_vec()],
            rights: vec![ecol[..half].to_vec(), ecol[half..].to_vec()],
            weights: matrix_weights
                .iter()
                .flat_map(|weights| {
                    [
                        SparkProverDotProductWeight::Base(weights[..half].to_vec()),
                        SparkProverDotProductWeight::Base(weights[half..].to_vec()),
                    ]
                })
                .collect(),
            pair_indices: vec![0, 1, 0, 1, 0, 1],
        };

        let mut independent_challenger = crate::poseidon_challenger();
        let independent_proof = prove_spark_batched_product_owned(
            product_terms.clone(),
            independent,
            &mut independent_challenger,
        )
        .expect("independent product proof succeeds");
        let mut split_challenger = crate::poseidon_challenger();
        let split_proof = prove_spark_batched_product_impl(
            product_terms.clone(),
            shared.clone(),
            false,
            &mut split_challenger,
        )
        .expect("split product proof succeeds");
        let mut shared_challenger = crate::poseidon_challenger();
        let shared_proof = prove_spark_batched_product_with_dotproducts(
            product_terms,
            shared,
            &mut shared_challenger,
        )
        .expect("shared product proof succeeds");

        assert_eq!(shared_proof, independent_proof);
        assert_eq!(shared_proof, split_proof);
        let shared_next = shared_challenger.sample_algebra_element::<EF>();
        assert_eq!(
            shared_next,
            independent_challenger.sample_algebra_element::<EF>()
        );
        assert_eq!(shared_next, split_challenger.sample_algebra_element::<EF>());
    }

    #[test]
    fn joint_preprocessing_pads_and_tags_slots() {
        let shape = shape_with_entries(
            4,
            4,
            vec![entry(0, 1, 7), entry(2, 3, 9)],
            vec![entry(1, 0, 11)],
            vec![],
        );

        let tables = preprocess_joint_spark_tables(&shape).expect("preprocess succeeds");

        assert_eq!(tables.matrix_nnz_padded, 2);
        assert_eq!(tables.value_domain_size, 8);
        assert_eq!(tables.rows[0], F::from_u32(0));
        assert_eq!(tables.cols[0], F::from_u32(1));
        assert_eq!(tables.vals[0], F::from_u32(7));
        assert_eq!(tables.rows[2], F::from_u32(1));
        assert_eq!(tables.cols[2], F::from_u32(0));
        assert_eq!(tables.vals[2], F::from_u32(11));
        assert_eq!(tables.vals[4], F::ZERO);
        assert_eq!(tables.slot_mapping[0], SparkMatrixSlot::A);
        assert_eq!(tables.slot_mapping[3], SparkMatrixSlot::Zero);
    }

    #[test]
    fn memory_metadata_counts_repeated_reads_per_address() {
        let (read, audit) = memory_in_the_head(3, &[2, 2, 0, 2]).expect("metadata succeeds");

        assert_eq!(read, vec![F::ZERO, F::ONE, F::ZERO, F::from_u32(2)]);
        assert_eq!(audit, vec![F::ONE, F::ZERO, F::from_u32(3)]);
    }

    #[test]
    fn layout_report_keeps_commitment_budget() {
        let shape = shape_with_entries(
            8,
            8,
            vec![entry(0, 0, 1), entry(1, 1, 1), entry(2, 2, 1)],
            vec![entry(3, 3, 1)],
            vec![entry(4, 4, 1), entry(5, 5, 1)],
        );

        let report = compare_spark_layouts(&shape).expect("layout comparison succeeds");

        assert_eq!(report.joint.setup_commitments, 2);
        assert_eq!(report.per_matrix.setup_commitments, 2);
        assert!(report.joint.wasted_value_slot_ratio_ppm > 0);
    }

    #[test]
    fn table_metadata_preserves_verifier_dimensions() {
        let shape = shape_with_entries(
            4,
            5,
            vec![entry(0, 0, 1), entry(1, 1, 2)],
            vec![entry(0, 0, 3)],
            vec![entry(2, 2, 4)],
        );
        let tables = preprocess_spark_tables(&shape).expect("preprocessing succeeds");
        let metadata = tables.metadata();

        assert_eq!(metadata.layout, tables.layout);
        assert_eq!(metadata.row_memory_size, tables.row_memory_size);
        assert_eq!(metadata.col_memory_size, tables.col_memory_size);
        assert_eq!(metadata.value_domain_size, tables.value_domain_size);
        assert_eq!(metadata.matrix_nnz_padded, tables.matrix_nnz_padded);
        metadata.validate().expect("metadata validates");

        let mut malformed = metadata;
        malformed.value_domain_size = 1;
        assert_eq!(
            malformed.validate(),
            Err(SpartanWhirError::InvalidR1csShape)
        );
    }

    #[test]
    fn profile_layout_report_does_not_need_matrix_materialization() {
        let report = compare_spark_layout_profile(SparkShapeProfile {
            num_rows: 345_399,
            num_cols: 612_724,
            nnz_a: 700_000,
            nnz_b: 700_000,
            nnz_c: 658_122,
            union_nnz: 760_000,
        })
        .expect("profile comparison succeeds");

        assert_eq!(report.joint.setup_commitments, 2);
        assert_eq!(report.per_matrix.setup_commitments, 2);
    }

    #[test]
    fn selector_matches_slot_values_on_boolean_tags() {
        let r = F::from_u32(5);
        assert_eq!(spark_selector_from_high_bits(F::ZERO, F::ZERO, r), F::ONE);
        assert_eq!(spark_selector_from_high_bits(F::ZERO, F::ONE, r), r);
        assert_eq!(spark_selector_from_high_bits(F::ONE, F::ZERO, r), r * r);
        assert_eq!(spark_selector_from_high_bits(F::ONE, F::ONE, r), F::ZERO);
    }
}
