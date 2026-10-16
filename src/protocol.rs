use alloc::{vec, vec::Vec};
use core::marker::PhantomData;

use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger};
use p3_commit::ExtensionMmcs;
use p3_field::{Field, PackedValue, PrimeCharacteristicRing};
use p3_keccak::Keccak256Hash;
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::{point::Point, poly::Poly};
use p3_sumcheck::{
    product_polynomial::ProductPolynomial,
    strategy::{SumcheckProver, VariableOrder},
    zk::{mask_residual_covectors_from_shape, ZkSumcheckData, ZkVerifier},
};
use p3_symmetric::{CryptographicHasher, Hash};
use p3_whir::pcs::zk::{
    CommittedMaskGroup, CommittedMaskGroupProverData, CommittedRelation, HidingWhirProver,
    HidingWhirVerifier, MaskCodeShape, MaskGroupProverData, MaskGroupShape,
};
use rand::{
    distr::{Distribution, StandardUniform},
    rngs::StdRng,
    CryptoRng, Rng, RngExt,
};
use serde::{Deserialize, Serialize};

use crate::engine::{ExtField, KeccakEngine, Poseidon1Engine, PoseidonEngine, F};
use crate::error::InvalidConfigReason;
use crate::plonky3_whir_pcs::{
    build_poseidon_full_zk_pcs, observe_poseidon_relation_domain_separator, FullZkPoseidonEngine,
    FullZkPoseidonPcs, PoseidonZkCommitmentFor, PoseidonZkHidingPcsFor, PoseidonZkMmcsFor,
    PoseidonZkRelationProofFor,
};
use crate::poseidon::{PoseidonZkProvingKeyFor, PoseidonZkVerifyingKeyFor};
use crate::profiling::profile_scope;
use crate::r1cs::{DirectBindLayout, DirectMultiplyLayout};
use crate::security::{
    derive_direct_component_security, derive_spark_component_security, SpartanSoundnessMode,
};
use crate::sumcheck::{
    prove_inner_base_first_unchecked, prove_outer_split_eq_base_first_owned_unchecked,
    prove_outer_zk_base_first_unchecked, verify_outer_zk,
};
use crate::{
    compute_spark_read_tables, preprocess_spark_tables,
    prove_spark_batched_memory_products_with_read_tables_and_leaf_claims,
    spark_fixed_audit_is_embedded, verify_inner, verify_outer,
    verify_spark_batched_memory_leaf_claims_with_openings,
    verify_spark_batched_memory_product_claims_with_metadata, CommittedPolynomialView,
    DomainSeparator, EqPolynomial, InnerSumcheckProof, MatrixClosingMode, MlePcs, MultilinearPoint,
    NoZkPcs, NoopObserver, OuterSumcheckProof, PcsStatementBuilder, Plonky3WhirPcs, PointEvalClaim,
    ProtocolObserver, ProtocolPcs, ProtocolStage, R1csInstance, R1csShape, R1csWitness,
    SecurityConfig, SparkBatchedMemoryProductsLeafClaims, SparkBatchedMemoryProductsProof,
    SparkFixedTableOpeningEvals, SparkLayoutKind, SparkReadPcs, SparkReadTableOpeningEvals,
    SparkReadTables, SparkTableMetadata, SparkTables, SpartanWhirEngine, SpartanWhirError,
    WhirParams, WhirPcsConfig, ZkOuterSumcheckProof, ZkWhirPcsConfig,
};

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::F: Serialize, Pcs::Commitment: Serialize, Pcs::ProverData: Serialize, Pcs::Config: Serialize",
    deserialize = "E::F: Deserialize<'de>, Pcs::Commitment: Deserialize<'de>, Pcs::ProverData: Deserialize<'de>, Pcs::Config: Deserialize<'de>"
))]
pub struct ProvingKey<E: SpartanWhirEngine, Pcs: MlePcs<E>> {
    pub matrix_closing: MatrixClosingMode,
    pub shape_canonical: R1csShape<E::F>,
    pub num_cons_unpadded: usize,
    pub num_vars_unpadded: usize,
    pub num_io: usize,
    pub security: SecurityConfig,
    pub whir_params: WhirParams,
    pub pcs_config: Pcs::Config,
    pub spark_fixed_commitments: Option<SparkFixedCommitments<Pcs::Commitment>>,
    pub spark_pcs_configs: Option<SparkPcsConfigs>,
    spark_fixed_prover_data: Option<SparkFixedProverData<E, Pcs>>,
    #[serde(default)]
    spark_tables: Option<crate::SparkTables>,
    #[serde(skip)]
    direct_bind_layout: Option<DirectBindLayout<E::F>>,
    #[serde(skip)]
    direct_multiply_layout: Option<DirectMultiplyLayout>,
    pub domain_separator: DomainSeparator,
    pub observer: Option<NoopObserver>,
    marker: PhantomData<(E, Pcs)>,
}

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::F: Serialize, Pcs::Commitment: Serialize, Pcs::Config: Serialize",
    deserialize = "E::F: Deserialize<'de>, Pcs::Commitment: Deserialize<'de>, Pcs::Config: Deserialize<'de>"
))]
pub struct VerifyingKey<E: SpartanWhirEngine, Pcs: MlePcs<E>> {
    matrix_closing: MatrixClosingMode,
    shape_canonical: R1csShape<E::F>,
    num_cons_unpadded: usize,
    num_vars_unpadded: usize,
    num_io: usize,
    security: SecurityConfig,
    whir_params: WhirParams,
    pcs_config: Pcs::Config,
    /// SPARK fixed-table Merkle roots produced at setup.
    spark_fixed_commitments: Option<SparkFixedCommitments<Pcs::Commitment>>,
    spark_pcs_configs: Option<SparkPcsConfigs>,
    #[serde(default)]
    spark_table_metadata: Option<SparkTableMetadata>,
    domain_separator: DomainSeparator,
    pub observer: Option<NoopObserver>,
    /// Setup authenticates the relation digest and, in SPARK mode, the fixed
    /// commitments. Serialization omits the marker, so every restored key must
    /// be authenticated once before verification.
    #[serde(skip)]
    verifying_key_authenticated: bool,
    marker: PhantomData<(E, Pcs)>,
}

impl<E, Pcs> VerifyingKey<E, Pcs>
where
    E: SpartanWhirEngine,
    Pcs: MlePcs<E>,
{
    pub fn matrix_closing(&self) -> MatrixClosingMode {
        self.matrix_closing
    }

    pub fn shape_canonical(&self) -> &R1csShape<E::F> {
        &self.shape_canonical
    }

    pub fn num_cons_unpadded(&self) -> usize {
        self.num_cons_unpadded
    }

    pub fn num_vars_unpadded(&self) -> usize {
        self.num_vars_unpadded
    }

    pub fn num_io(&self) -> usize {
        self.num_io
    }

    pub fn security(&self) -> SecurityConfig {
        self.security
    }

    pub fn whir_params(&self) -> &WhirParams {
        &self.whir_params
    }

    pub fn pcs_config(&self) -> &Pcs::Config {
        &self.pcs_config
    }

    pub fn spark_fixed_commitments(&self) -> Option<&SparkFixedCommitments<Pcs::Commitment>> {
        self.spark_fixed_commitments.as_ref()
    }

    pub fn spark_pcs_configs(&self) -> Option<&SparkPcsConfigs> {
        self.spark_pcs_configs.as_ref()
    }

    pub fn spark_table_metadata(&self) -> Option<SparkTableMetadata> {
        self.spark_table_metadata
    }

    pub fn domain_separator(&self) -> &DomainSeparator {
        &self.domain_separator
    }
}

impl<E, Pcs> ProvingKey<E, Pcs>
where
    E: SpartanWhirEngine,
    E::F: Copy + PartialEq,
    Pcs: MlePcs<E>,
{
    pub fn prepare_for_proving(&mut self) -> Result<(), SpartanWhirError> {
        self.direct_bind_layout = Some(self.shape_canonical.direct_bind_layout()?);
        self.direct_multiply_layout = Some(self.shape_canonical.direct_multiply_layout()?);
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::EF: Serialize, Pcs::Proof: Serialize",
    deserialize = "E::EF: Deserialize<'de>, Pcs::Proof: Deserialize<'de>"
))]
pub struct SpartanProof<E: SpartanWhirEngine, Pcs: MlePcs<E>> {
    pub outer_sumcheck: OuterSumcheckProof<E::EF>,
    pub outer_claims: (E::EF, E::EF, E::EF),
    pub inner_sumcheck: InnerSumcheckProof<E::EF>,
    pub witness_eval: E::EF,
    pub pcs_proof: Pcs::Proof,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "E::EF: ExtField, SparkFixedOpeningProof<E, Plonky3WhirPcs>: Serialize, SparkReadOpeningProof<E, Plonky3WhirPcs>: Serialize, PoseidonZkRelationProofFor<E>: Serialize",
    deserialize = "E::EF: ExtField, SparkFixedOpeningProof<E, Plonky3WhirPcs>: Deserialize<'de>, SparkReadOpeningProof<E, Plonky3WhirPcs>: Deserialize<'de>, PoseidonZkRelationProofFor<E>: Deserialize<'de>"
))]
pub struct ZkSpartanProofFor<E>
where
    E: FullZkPoseidonEngine,
    E::EF: ExtField,
    E::Challenger: FieldChallenger<F> + p3_challenger::GrindingChallenger<Witness = F>,
    StandardUniform: Distribution<E::EF>,
    Plonky3WhirPcs: FullZkPoseidonPcs<E>,
{
    pub application_mask_commitment: PoseidonZkCommitmentFor<E>,
    pub outer_sumcheck: ZkOuterSumcheckProof<E::EF>,
    pub outer_claims: (E::EF, E::EF, E::EF),
    pub outer_mask_evals: Vec<E::EF>,
    pub inner_sumcheck: ZkSumcheckData<F, E::EF>,
    pub inner_sumcheck_mask_commitment: PoseidonZkCommitmentFor<E>,
    pub matrix_closing: ZkMatrixClosingProofFor<E>,
    pub pcs_proof: PoseidonZkRelationProofFor<E>,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "E::EF: ExtField, SparkFixedOpeningProof<E, Plonky3WhirPcs>: Serialize, SparkReadOpeningProof<E, Plonky3WhirPcs>: Serialize",
    deserialize = "E::EF: ExtField, SparkFixedOpeningProof<E, Plonky3WhirPcs>: Deserialize<'de>, SparkReadOpeningProof<E, Plonky3WhirPcs>: Deserialize<'de>"
))]
pub struct ZkSparkClosingProofFor<E>
where
    E: FullZkPoseidonEngine,
    E::EF: ExtField,
    E::Challenger: FieldChallenger<F> + p3_challenger::GrindingChallenger<Witness = F>,
    StandardUniform: Distribution<E::EF>,
    Plonky3WhirPcs: FullZkPoseidonPcs<E>,
{
    pub spark_products: SparkBatchedMemoryProductsProof<E::EF>,
    pub spark_fixed_openings: SparkFixedOpeningProof<E, Plonky3WhirPcs>,
    pub spark_read_openings: SparkReadOpeningProof<E, Plonky3WhirPcs>,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "E::EF: ExtField, ZkSparkClosingProofFor<E>: Serialize",
    deserialize = "E::EF: ExtField, ZkSparkClosingProofFor<E>: Deserialize<'de>"
))]
pub enum ZkMatrixClosingProofFor<E>
where
    E: FullZkPoseidonEngine,
    E::EF: ExtField,
    E::Challenger: FieldChallenger<F> + p3_challenger::GrindingChallenger<Witness = F>,
    StandardUniform: Distribution<E::EF>,
    Plonky3WhirPcs: FullZkPoseidonPcs<E>,
{
    DirectSparse,
    Spark(ZkSparkClosingProofFor<E>),
}

impl<E> ZkMatrixClosingProofFor<E>
where
    E: FullZkPoseidonEngine,
    E::EF: ExtField,
    E::Challenger: FieldChallenger<F> + p3_challenger::GrindingChallenger<Witness = F>,
    StandardUniform: Distribution<E::EF>,
    Plonky3WhirPcs: FullZkPoseidonPcs<E>,
{
    pub fn mode(&self) -> MatrixClosingMode {
        match self {
            Self::DirectSparse => MatrixClosingMode::DirectSparse,
            Self::Spark(_) => MatrixClosingMode::Spark,
        }
    }
}

pub type ZkSpartanProof<Ext> = ZkSpartanProofFor<PoseidonEngine<Ext>>;
pub type Poseidon1ZkSpartanProof<Ext> = ZkSpartanProofFor<Poseidon1Engine<Ext>>;
pub type ZkSparkClosingProof<Ext> = ZkSparkClosingProofFor<PoseidonEngine<Ext>>;
pub type Poseidon1ZkSparkClosingProof<Ext> = ZkSparkClosingProofFor<Poseidon1Engine<Ext>>;
pub type ZkMatrixClosingProof<Ext> = ZkMatrixClosingProofFor<PoseidonEngine<Ext>>;
pub type Poseidon1ZkMatrixClosingProof<Ext> = ZkMatrixClosingProofFor<Poseidon1Engine<Ext>>;

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::EF: Serialize, Pcs::Commitment: Serialize, Pcs::Proof: Serialize",
    deserialize = "E::EF: Deserialize<'de>, Pcs::Commitment: Deserialize<'de>, Pcs::Proof: Deserialize<'de>"
))]
pub struct SparkSpartanProof<E: SpartanWhirEngine, Pcs: MlePcs<E>> {
    pub outer_sumcheck: OuterSumcheckProof<E::EF>,
    pub outer_claims: (E::EF, E::EF, E::EF),
    pub inner_sumcheck: InnerSumcheckProof<E::EF>,
    pub witness_eval: E::EF,
    pub spark_products: SparkBatchedMemoryProductsProof<E::EF>,
    pub spark_fixed_openings: SparkFixedOpeningProof<E, Pcs>,
    pub spark_read_openings: SparkReadOpeningProof<E, Pcs>,
    pub pcs_proof: Pcs::Proof,
}

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::EF: Serialize, Pcs::Commitment: Serialize, Pcs::Proof: Serialize",
    deserialize = "E::EF: Deserialize<'de>, Pcs::Commitment: Deserialize<'de>, Pcs::Proof: Deserialize<'de>"
))]
pub struct SparkFixedOpeningProof<E: SpartanWhirEngine, Pcs: MlePcs<E>> {
    pub value_num_variables: usize,
    pub value_column_bits: usize,
    pub audit_num_variables: usize,
    pub audit_column_bits: usize,
    pub value_commitment: Pcs::Commitment,
    pub audit_commitment: Option<Pcs::Commitment>,
    pub evals: SparkFixedTableOpeningEvals<E::EF>,
    pub value_proof: Pcs::Proof,
    pub audit_proof: Option<Pcs::Proof>,
    marker: PhantomData<E>,
}

impl<E, Pcs> SparkFixedOpeningProof<E, Pcs>
where
    E: SpartanWhirEngine,
    Pcs: MlePcs<E>,
{
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts(
        value_num_variables: usize,
        value_column_bits: usize,
        audit_num_variables: usize,
        audit_column_bits: usize,
        value_commitment: Pcs::Commitment,
        audit_commitment: Option<Pcs::Commitment>,
        evals: SparkFixedTableOpeningEvals<E::EF>,
        value_proof: Pcs::Proof,
        audit_proof: Option<Pcs::Proof>,
    ) -> Self {
        Self {
            value_num_variables,
            value_column_bits,
            audit_num_variables,
            audit_column_bits,
            value_commitment,
            audit_commitment,
            evals,
            value_proof,
            audit_proof,
            marker: PhantomData,
        }
    }
}

impl<E, Pcs> Clone for SparkFixedOpeningProof<E, Pcs>
where
    E: SpartanWhirEngine,
    Pcs: MlePcs<E>,
    E::EF: Clone,
    Pcs::Commitment: Clone,
    Pcs::Proof: Clone,
{
    fn clone(&self) -> Self {
        Self {
            value_num_variables: self.value_num_variables,
            value_column_bits: self.value_column_bits,
            audit_num_variables: self.audit_num_variables,
            audit_column_bits: self.audit_column_bits,
            value_commitment: self.value_commitment.clone(),
            audit_commitment: self.audit_commitment.clone(),
            evals: self.evals.clone(),
            value_proof: self.value_proof.clone(),
            audit_proof: self.audit_proof.clone(),
            marker: PhantomData,
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::EF: Serialize, Pcs::Commitment: Serialize, Pcs::Proof: Serialize",
    deserialize = "E::EF: Deserialize<'de>, Pcs::Commitment: Deserialize<'de>, Pcs::Proof: Deserialize<'de>"
))]
pub struct SparkReadOpeningProof<E: SpartanWhirEngine, Pcs: MlePcs<E>> {
    pub groups: Vec<SparkReadGroupOpeningProof<E, Pcs>>,
}

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::EF: Serialize, Pcs::Commitment: Serialize, Pcs::Proof: Serialize",
    deserialize = "E::EF: Deserialize<'de>, Pcs::Commitment: Deserialize<'de>, Pcs::Proof: Deserialize<'de>"
))]
pub struct SparkReadGroupOpeningProof<E: SpartanWhirEngine, Pcs: MlePcs<E>> {
    pub num_variables: usize,
    pub column_start: usize,
    pub column_count: usize,
    pub commitment: Pcs::Commitment,
    pub evals: Vec<Vec<E::EF>>,
    pub proof: Pcs::Proof,
    marker: PhantomData<E>,
}

impl<E, Pcs> SparkReadGroupOpeningProof<E, Pcs>
where
    E: SpartanWhirEngine,
    Pcs: MlePcs<E>,
{
    pub(crate) fn from_parts(
        num_variables: usize,
        column_start: usize,
        column_count: usize,
        commitment: Pcs::Commitment,
        evals: Vec<Vec<E::EF>>,
        proof: Pcs::Proof,
    ) -> Self {
        Self {
            num_variables,
            column_start,
            column_count,
            commitment,
            evals,
            proof,
            marker: PhantomData,
        }
    }
}

impl<E, Pcs> Clone for SparkReadOpeningProof<E, Pcs>
where
    E: SpartanWhirEngine,
    Pcs: MlePcs<E>,
    E::EF: Clone,
    Pcs::Commitment: Clone,
    Pcs::Proof: Clone,
{
    fn clone(&self) -> Self {
        Self {
            groups: self.groups.clone(),
        }
    }
}

impl<E, Pcs> Clone for SparkReadGroupOpeningProof<E, Pcs>
where
    E: SpartanWhirEngine,
    Pcs: MlePcs<E>,
    E::EF: Clone,
    Pcs::Commitment: Clone,
    Pcs::Proof: Clone,
{
    fn clone(&self) -> Self {
        Self {
            num_variables: self.num_variables,
            column_start: self.column_start,
            column_count: self.column_count,
            commitment: self.commitment.clone(),
            evals: self.evals.clone(),
            proof: self.proof.clone(),
            marker: PhantomData,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparkWhirParams {
    pub fixed_value: WhirParams,
    pub fixed_audit: WhirParams,
    pub read: WhirParams,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparkPcsConfigs {
    pub fixed_value: WhirPcsConfig,
    pub fixed_audit: WhirPcsConfig,
    pub read: Vec<WhirPcsConfig>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpartanSnarkConfig {
    pub matrix_closing: MatrixClosingMode,
    pub security: SecurityConfig,
    pub whir_params: WhirParams,
    #[serde(default)]
    pub spark_whir_params: Option<SparkWhirParams>,
}

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::EF: Serialize, Pcs::Commitment: Serialize, Pcs::Proof: Serialize",
    deserialize = "E::EF: Deserialize<'de>, Pcs::Commitment: Deserialize<'de>, Pcs::Proof: Deserialize<'de>"
))]
pub enum SpartanProofKind<E: SpartanWhirEngine, Pcs: MlePcs<E>> {
    Direct(SpartanProof<E, Pcs>),
    Spark(SparkSpartanProof<E, Pcs>),
}

impl<E: SpartanWhirEngine, Pcs: MlePcs<E>> SpartanProofKind<E, Pcs> {
    pub fn kind(&self) -> MatrixClosingMode {
        match self {
            Self::Direct(_) => MatrixClosingMode::DirectSparse,
            Self::Spark(_) => MatrixClosingMode::Spark,
        }
    }
}

struct SparkReadProverData<E: SpartanWhirEngine, Pcs: SparkReadPcs<E>> {
    groups: Vec<Pcs::ReadProverData>,
    marker: PhantomData<E>,
}

struct SparkReadCommitments<C> {
    groups: Vec<C>,
}

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Pcs::ProverData: Serialize",
    deserialize = "Pcs::ProverData: Deserialize<'de>"
))]
pub(crate) struct SparkFixedProverData<E: SpartanWhirEngine, Pcs: MlePcs<E>> {
    pub(crate) value: Pcs::ProverData,
    pub(crate) audit: Option<Pcs::ProverData>,
    marker: PhantomData<E>,
}

impl<E, Pcs> Clone for SparkFixedProverData<E, Pcs>
where
    E: SpartanWhirEngine,
    Pcs: MlePcs<E>,
    Pcs::ProverData: Clone,
{
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            audit: self.audit.clone(),
            marker: PhantomData,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparkFixedCommitments<C = [u64; 4]> {
    pub value: C,
    pub audit: Option<C>,
}

struct ParsedSparkReadOpenings<E, Pcs>
where
    E: SpartanWhirEngine,
    Pcs: SparkReadPcs<E>,
{
    groups: Vec<Pcs::ParsedReadCommitment>,
}

struct ParsedSparkFixedOpenings<E, Pcs>
where
    E: SpartanWhirEngine,
    Pcs: ProtocolPcs<E, Config = WhirPcsConfig>,
{
    value: Pcs::ParsedCommitment,
    audit: Option<Pcs::ParsedCommitment>,
}

pub trait SpartanContextEngine: SpartanWhirEngine<F = F>
where
    Self::EF: ExtField,
{
    const NO_ZK_PROTOCOL_ID: &'static [u8];

    fn challenger() -> Self::Challenger;

    fn observe_spartan_context(
        challenger: &mut Self::Challenger,
        domain_separator: &DomainSeparator,
        public_inputs: &[F],
    ) -> Result<(), SpartanWhirError>;
}

impl<Ext> SpartanContextEngine for KeccakEngine<Ext>
where
    Ext: ExtField,
{
    const NO_ZK_PROTOCOL_ID: &'static [u8] = crate::NO_ZK_PROTOCOL_ID;

    fn challenger() -> Self::Challenger {
        crate::engine::keccak_challenger()
    }

    fn observe_spartan_context(
        challenger: &mut Self::Challenger,
        domain_separator: &DomainSeparator,
        public_inputs: &[F],
    ) -> Result<(), SpartanWhirError> {
        let digest_bytes = Keccak256Hash {}.hash_iter(domain_separator.to_bytes());
        let digest_hash: Hash<F, u8, 32> = digest_bytes.into();
        challenger.observe(digest_hash);
        for &input in public_inputs {
            challenger.observe(input);
        }
        Ok(())
    }
}

impl<Ext> SpartanContextEngine for PoseidonEngine<Ext>
where
    Ext: ExtField,
{
    const NO_ZK_PROTOCOL_ID: &'static [u8] = crate::NO_ZK_PROTOCOL_ID;

    fn challenger() -> Self::Challenger {
        crate::engine::poseidon_challenger()
    }

    fn observe_spartan_context(
        challenger: &mut Self::Challenger,
        domain_separator: &DomainSeparator,
        public_inputs: &[F],
    ) -> Result<(), SpartanWhirError> {
        // TODO: decide whether the Poseidon transcript should first compress
        // the domain separator into field elements instead of absorbing bytes.
        for byte in domain_separator.to_bytes() {
            challenger.observe(F::from_u8(byte));
        }
        for &input in public_inputs {
            challenger.observe(input);
        }
        Ok(())
    }
}

impl<Ext> SpartanContextEngine for Poseidon1Engine<Ext>
where
    Ext: ExtField,
{
    const NO_ZK_PROTOCOL_ID: &'static [u8] = crate::POSEIDON1_NO_ZK_PROTOCOL_ID;

    fn challenger() -> Self::Challenger {
        crate::engine::poseidon1_challenger()
    }

    fn observe_spartan_context(
        challenger: &mut Self::Challenger,
        domain_separator: &DomainSeparator,
        public_inputs: &[F],
    ) -> Result<(), SpartanWhirError> {
        for byte in domain_separator.to_bytes() {
            challenger.observe(F::from_u8(byte));
        }
        for &input in public_inputs {
            challenger.observe(input);
        }
        Ok(())
    }
}

pub struct SpartanProtocol<E: SpartanWhirEngine, Pcs: MlePcs<E>> {
    marker: PhantomData<(E, Pcs)>,
}

pub struct PoseidonZkSpartanProtocolFor<E: FullZkPoseidonEngine>
where
    E::EF: ExtField,
    E::Challenger: FieldChallenger<F> + p3_challenger::GrindingChallenger<Witness = F>,
{
    marker: PhantomData<E>,
}

pub type PoseidonZkSpartanProtocol<Ext> = PoseidonZkSpartanProtocolFor<PoseidonEngine<Ext>>;
pub type Poseidon1ZkSpartanProtocol<Ext> = PoseidonZkSpartanProtocolFor<Poseidon1Engine<Ext>>;

pub(crate) fn validate_canonical_verifying_shape<F>(
    shape: &R1csShape<F>,
    num_cons_unpadded: usize,
    num_vars_unpadded: usize,
    num_io: usize,
) -> Result<(), SpartanWhirError> {
    if num_cons_unpadded == 0 || shape.num_io != num_io {
        return Err(SpartanWhirError::InvalidR1csShape);
    }

    let expected_num_cons = num_cons_unpadded
        .checked_next_power_of_two()
        .ok_or(SpartanWhirError::InvalidR1csShape)?;
    let min_num_vars = num_io
        .checked_add(1)
        .ok_or(SpartanWhirError::InvalidR1csShape)?;
    let expected_num_vars = core::cmp::max(num_vars_unpadded, min_num_vars)
        .checked_next_power_of_two()
        .ok_or(SpartanWhirError::InvalidR1csShape)?;
    let expected_num_cols = expected_num_vars
        .checked_add(num_io)
        .and_then(|value| value.checked_add(1))
        .ok_or(SpartanWhirError::InvalidR1csShape)?;

    if shape.num_cons != expected_num_cons
        || shape.num_vars != expected_num_vars
        || num_io >= expected_num_vars
        || shape.a.num_rows != expected_num_cons
        || shape.b.num_rows != expected_num_cons
        || shape.c.num_rows != expected_num_cons
        || shape.a.num_cols != expected_num_cols
        || shape.b.num_cols != expected_num_cols
        || shape.c.num_cols != expected_num_cols
    {
        return Err(SpartanWhirError::InvalidR1csShape);
    }

    Ok(())
}

pub(crate) fn validate_spark_table_metadata<EF>(
    shape: &R1csShape<F>,
    metadata: &SparkTableMetadata,
    configs: &SparkPcsConfigs,
) -> Result<(), SpartanWhirError>
where
    EF: ExtField,
{
    metadata.validate()?;
    if metadata.layout == SparkLayoutKind::Joint {
        return Err(SpartanWhirError::invalid_config());
    }

    let expected_col_memory_size = shape
        .num_vars
        .checked_mul(2)
        .ok_or_else(SpartanWhirError::invalid_config)?;
    let max_nnz = shape.a.nnz().max(shape.b.nnz()).max(shape.c.nnz());
    let expected_matrix_nnz_padded = if max_nnz == 0 {
        1
    } else {
        max_nnz
            .checked_next_power_of_two()
            .ok_or_else(SpartanWhirError::invalid_config)?
    };
    if metadata.row_memory_size != shape.num_cons
        || metadata.col_memory_size != expected_col_memory_size
        || metadata.matrix_nnz_padded != expected_matrix_nnz_padded
    {
        return Err(SpartanWhirError::invalid_config());
    }

    let value_bits = metadata.value_domain_size.ilog2() as usize;
    let audit_bits = metadata
        .row_memory_size
        .max(metadata.col_memory_size)
        .ilog2() as usize;
    if configs.fixed_value.num_variables
        != value_bits
            .checked_add(fixed_value_column_bits())
            .ok_or_else(SpartanWhirError::invalid_config)?
        || configs.fixed_audit.num_variables
            != audit_bits
                .checked_add(fixed_audit_column_bits())
                .ok_or_else(SpartanWhirError::invalid_config)?
        || configs.read.len() != read_coordinate_groups::<EF>()?.len()
        || configs
            .read
            .iter()
            .zip(read_coordinate_groups::<EF>()?)
            .any(|(config, group)| {
                Some(config.num_variables)
                    != value_bits.checked_add(group.column_count.ilog2() as usize)
            })
    {
        return Err(SpartanWhirError::invalid_config());
    }
    Ok(())
}

/// Validate structure and configuration using the stored relation digest.
/// This does not recompute the relation digest; use [`VerifyingKey::authenticate`]
/// before accepting restored key material.
fn validate_verifying_key<E, Pcs>(
    vk: &VerifyingKey<E, Pcs>,
) -> Result<Option<SparkTableMetadata>, SpartanWhirError>
where
    E: SpartanContextEngine,
    E::EF: ExtField,
    Pcs: ProtocolPcs<E, Config = WhirPcsConfig> + NoZkPcs,
{
    validate_canonical_verifying_shape(
        &vk.shape_canonical,
        vk.num_cons_unpadded,
        vk.num_vars_unpadded,
        vk.num_io,
    )?;
    vk.security.validate()?;
    vk.pcs_config.validate()?;

    if vk.pcs_config.num_variables != vk.shape_canonical.num_vars.ilog2() as usize
        || vk.pcs_config.whir != vk.whir_params
    {
        return Err(SpartanWhirError::invalid_config());
    }

    let expected_domain = DomainSeparator::new_with_protocol_id_and_relation_digest(
        E::NO_ZK_PROTOCOL_ID,
        &vk.shape_canonical,
        vk.domain_separator.relation_digest,
        &vk.security,
        &vk.whir_params,
        vk.matrix_closing,
        vk.domain_separator.spark_whir_params.clone(),
    );
    if vk.domain_separator != expected_domain {
        return Err(SpartanWhirError::invalid_config());
    }

    let spark_metadata = match vk.matrix_closing {
        MatrixClosingMode::DirectSparse => {
            let (expected_security, _) = derive_direct_component_security::<E::EF>(
                &vk.security,
                &vk.pcs_config,
                vk.shape_canonical.num_cons.ilog2() as usize,
                vk.shape_canonical.num_vars.ilog2() as usize + 1,
                SpartanSoundnessMode::NoZk,
            )?;
            if vk.pcs_config.security != expected_security
                || vk.spark_fixed_commitments.is_some()
                || vk.spark_pcs_configs.is_some()
                || vk.spark_table_metadata.is_some()
            {
                return Err(SpartanWhirError::invalid_config());
            }
            None
        }
        MatrixClosingMode::Spark => {
            if vk.spark_fixed_commitments.is_none() {
                return Err(SpartanWhirError::invalid_config());
            }
            let configs = vk
                .spark_pcs_configs
                .as_ref()
                .ok_or(SpartanWhirError::invalid_config())?;
            configs.fixed_value.validate()?;
            configs.fixed_audit.validate()?;
            configs.read.iter().try_for_each(WhirPcsConfig::validate)?;
            match &vk.domain_separator.spark_whir_params {
                Some(params)
                    if configs.fixed_value.whir == params.fixed_value
                        && configs.fixed_audit.whir == params.fixed_audit
                        && configs.read.iter().all(|config| config.whir == params.read) => {}
                None if configs.fixed_value.whir == vk.whir_params
                    && configs.fixed_audit.whir == vk.whir_params
                    && configs
                        .read
                        .iter()
                        .all(|config| config.whir == vk.whir_params) => {}
                _ => return Err(SpartanWhirError::invalid_config()),
            }
            let metadata = vk
                .spark_table_metadata
                .ok_or_else(SpartanWhirError::invalid_config)?;
            validate_spark_table_metadata::<E::EF>(&vk.shape_canonical, &metadata, configs)?;
            let (expected_security, _) = derive_spark_component_security::<E::EF>(
                &vk.security,
                &metadata,
                &vk.pcs_config,
                configs,
                vk.shape_canonical.num_cons.ilog2() as usize,
                vk.shape_canonical.num_vars.ilog2() as usize + 1,
                SpartanSoundnessMode::NoZk,
            )?;
            if vk.pcs_config.security != expected_security
                || configs.fixed_value.security != expected_security
                || configs.fixed_audit.security != expected_security
                || configs
                    .read
                    .iter()
                    .any(|config| config.security != expected_security)
            {
                return Err(SpartanWhirError::invalid_config());
            }
            Some(metadata)
        }
    };

    Ok(spark_metadata)
}

/// Recompute the SPARK fixed-table commitments from an R1CS shape and compare
/// them with the roots carried by a key.
///
/// This is the only check that binds the fixed-table Merkle roots to the
/// matrices they are supposed to commit to. It re-runs SPARK preprocessing
/// and committing, so it costs about as much as the SPARK part of `setup`.
pub(crate) fn authenticate_spark_fixed_commitments<E, EF, Pcs>(
    shape_canonical: &R1csShape<F>,
    configs: &SparkPcsConfigs,
    expected: &SparkFixedCommitments<<Pcs as MlePcs<E>>::Commitment>,
) -> Result<(), SpartanWhirError>
where
    EF: ExtField,
    E: SpartanContextEngine<EF = EF>,
    Pcs: SparkReadPcs<E>,
    <Pcs as MlePcs<E>>::ProverData: Clone + CommittedPolynomialView<EF>,
    <Pcs as MlePcs<E>>::Commitment: Clone + PartialEq,
{
    let tables = preprocess_spark_tables(shape_canonical)?;
    let (_, commitments) = setup_spark_fixed_commitments::<E, EF, Pcs>(configs, &tables)?
        .ok_or_else(SpartanWhirError::invalid_config)?;
    if commitments != *expected {
        return Err(SpartanWhirError::CommitmentMismatch);
    }
    Ok(())
}

impl<E, Pcs> VerifyingKey<E, Pcs>
where
    E: SpartanContextEngine,
    E::EF: ExtField,
    Pcs: ProtocolPcs<E, Config = WhirPcsConfig> + NoZkPcs,
    Pcs::ProverData: Clone + CommittedPolynomialView<E::EF>,
    Pcs::Commitment: Clone + PartialEq,
{
    /// Authenticate a setup-created or restored key against its embedded R1CS.
    ///
    /// This recomputes the canonical relation digest once. In SPARK mode it
    /// also rebuilds and checks the fixed-table commitments, which costs about
    /// as much as the SPARK part of `setup`. Verification then reuses the
    /// authenticated digest without scanning the matrices again.
    pub fn authenticate(&mut self) -> Result<(), SpartanWhirError>
    where
        Pcs: SparkReadPcs<E>,
    {
        self.verifying_key_authenticated = false;
        validate_verifying_key::<E, Pcs>(self)?;
        let expected_relation_digest = crate::canonical_r1cs_relation_digest(
            &self.shape_canonical,
            self.num_cons_unpadded,
            self.num_vars_unpadded,
            self.num_io,
        )?;
        if self.domain_separator.relation_digest != expected_relation_digest {
            return Err(SpartanWhirError::invalid_config());
        }
        match self.matrix_closing {
            MatrixClosingMode::DirectSparse => {}
            MatrixClosingMode::Spark => {
                let configs = self
                    .spark_pcs_configs
                    .as_ref()
                    .ok_or_else(SpartanWhirError::invalid_config)?;
                let expected = self
                    .spark_fixed_commitments
                    .as_ref()
                    .ok_or_else(SpartanWhirError::invalid_config)?;
                authenticate_spark_fixed_commitments::<E, E::EF, Pcs>(
                    &self.shape_canonical,
                    configs,
                    expected,
                )?;
            }
        }
        self.verifying_key_authenticated = true;
        Ok(())
    }

    /// Compatibility wrapper for callers that authenticated restored SPARK
    /// keys through the previous API.
    pub fn authenticate_spark_fixed_commitments(&mut self) -> Result<(), SpartanWhirError>
    where
        Pcs: SparkReadPcs<E>,
    {
        self.authenticate()
    }

    pub(crate) fn ensure_authenticated(&self) -> Result<(), SpartanWhirError> {
        if !self.verifying_key_authenticated {
            return Err(SpartanWhirError::invalid_config_reason(
                InvalidConfigReason::UnauthenticatedVerifyingKey,
            ));
        }
        Ok(())
    }
}

impl<E, Pcs> SpartanProtocol<E, Pcs>
where
    E: SpartanContextEngine,
    E::EF: ExtField,
    Pcs: ProtocolPcs<E, Config = WhirPcsConfig> + NoZkPcs,
    Pcs::ProverData: Clone + CommittedPolynomialView<E::EF>,
    Pcs::Commitment: Clone + PartialEq,
    E::Challenger: FieldChallenger<F>,
{
    pub fn setup(
        shape: &R1csShape<F>,
        security: &SecurityConfig,
        whir_params: &WhirParams,
    ) -> Result<(ProvingKey<E, Pcs>, VerifyingKey<E, Pcs>), SpartanWhirError>
    where
        Pcs: SparkReadPcs<E>,
    {
        // Defaults to SPARK setup; call `setup_with_config` with
        // `DirectSparse` to skip SPARK preprocessing.
        Self::setup_with_config(
            shape,
            &SpartanSnarkConfig {
                matrix_closing: MatrixClosingMode::Spark,
                security: *security,
                whir_params: whir_params.clone(),
                spark_whir_params: None,
            },
        )
    }

    pub fn setup_with_config(
        shape: &R1csShape<F>,
        config: &SpartanSnarkConfig,
    ) -> Result<(ProvingKey<E, Pcs>, VerifyingKey<E, Pcs>), SpartanWhirError>
    where
        Pcs: SparkReadPcs<E>,
    {
        Self::setup_for_mode(shape, config)
    }

    fn setup_for_mode(
        shape: &R1csShape<F>,
        config: &SpartanSnarkConfig,
    ) -> Result<(ProvingKey<E, Pcs>, VerifyingKey<E, Pcs>), SpartanWhirError>
    where
        Pcs: SparkReadPcs<E>,
    {
        let mut observer = NoopObserver;
        observer.on_stage(ProtocolStage::SetupStart);

        config.security.validate()?;
        shape.validate()?;

        let shape_canonical = shape.pad_regular()?;
        let num_variables = shape_canonical.num_vars.ilog2() as usize;

        let provisional_pcs_config = WhirPcsConfig {
            num_variables,
            security: config.security,
            whir: config.whir_params.clone(),
        };
        provisional_pcs_config.validate()?;
        let num_outer_rounds = shape_canonical.num_cons.ilog2() as usize;
        let num_inner_rounds = num_variables + 1;
        let spark_tables = match config.matrix_closing {
            MatrixClosingMode::DirectSparse => None,
            MatrixClosingMode::Spark => Some(preprocess_spark_tables(&shape_canonical)?),
        };
        let component_security = match spark_tables.as_ref() {
            None => {
                derive_direct_component_security::<E::EF>(
                    &config.security,
                    &provisional_pcs_config,
                    num_outer_rounds,
                    num_inner_rounds,
                    SpartanSoundnessMode::NoZk,
                )?
                .0
            }
            Some(tables) => {
                let provisional_spark_configs = spark_pcs_configs_for_tables::<E::EF>(
                    &provisional_pcs_config,
                    tables,
                    config.spark_whir_params.as_ref(),
                )?;
                derive_spark_component_security::<E::EF>(
                    &config.security,
                    &tables.metadata(),
                    &provisional_pcs_config,
                    &provisional_spark_configs,
                    num_outer_rounds,
                    num_inner_rounds,
                    SpartanSoundnessMode::NoZk,
                )?
                .0
            }
        };
        let canonical_pcs_config = WhirPcsConfig {
            security: component_security,
            ..provisional_pcs_config
        };
        Pcs::validate_spartan_config(
            &canonical_pcs_config,
            config.matrix_closing,
            num_outer_rounds,
            num_inner_rounds,
        )?;

        let transcript_spark_whir_params = match config.matrix_closing {
            MatrixClosingMode::DirectSparse => None,
            MatrixClosingMode::Spark => config.spark_whir_params.clone(),
        };
        let domain_separator = DomainSeparator::new_with_protocol_id(
            E::NO_ZK_PROTOCOL_ID,
            &shape_canonical,
            shape.num_cons,
            shape.num_vars,
            shape.num_io,
            &config.security,
            &config.whir_params,
            config.matrix_closing,
            transcript_spark_whir_params,
        )?;
        let (spark_pcs_configs, spark_fixed_setup) = match config.matrix_closing {
            MatrixClosingMode::DirectSparse => (None, None),
            MatrixClosingMode::Spark => {
                let _profile = profile_scope("spark_fixed_setup");
                let spark_tables = spark_tables
                    .as_ref()
                    .ok_or_else(SpartanWhirError::invalid_config)?;
                let spark_pcs_configs = spark_pcs_configs_for_tables::<E::EF>(
                    &canonical_pcs_config,
                    spark_tables,
                    config.spark_whir_params.as_ref(),
                )?;
                let setup = setup_spark_fixed_commitments::<E, E::EF, Pcs>(
                    &spark_pcs_configs,
                    spark_tables,
                )?;
                (Some(spark_pcs_configs), setup)
            }
        };
        let (spark_fixed_prover_data, spark_fixed_commitments) = match spark_fixed_setup {
            Some((prover_data, commitments)) => (Some(prover_data), Some(commitments)),
            None => (None, None),
        };
        let spark_table_metadata = spark_tables.as_ref().map(SparkTables::metadata);
        let mut pk = ProvingKey {
            matrix_closing: config.matrix_closing,
            shape_canonical: shape_canonical.clone(),
            num_cons_unpadded: shape.num_cons,
            num_vars_unpadded: shape.num_vars,
            num_io: shape.num_io,
            security: config.security,
            whir_params: config.whir_params.clone(),
            pcs_config: canonical_pcs_config.clone(),
            spark_fixed_commitments: spark_fixed_commitments.clone(),
            spark_pcs_configs: spark_pcs_configs.clone(),
            spark_fixed_prover_data,
            spark_tables,
            direct_bind_layout: None,
            direct_multiply_layout: None,
            domain_separator: domain_separator.clone(),
            observer: Some(NoopObserver),
            marker: PhantomData,
        };
        pk.prepare_for_proving()?;

        let vk = VerifyingKey {
            matrix_closing: config.matrix_closing,
            shape_canonical,
            num_cons_unpadded: shape.num_cons,
            num_vars_unpadded: shape.num_vars,
            num_io: shape.num_io,
            security: config.security,
            whir_params: config.whir_params.clone(),
            pcs_config: canonical_pcs_config,
            spark_fixed_commitments,
            spark_pcs_configs,
            spark_table_metadata,
            domain_separator,
            observer: Some(NoopObserver),
            verifying_key_authenticated: true,
            marker: PhantomData,
        };

        observer.on_stage(ProtocolStage::SetupEnd);
        Ok((pk, vk))
    }

    pub fn prove_with_mode(
        pk: &ProvingKey<E, Pcs>,
        public_inputs: &[F],
        witness: &R1csWitness<F>,
        mode: MatrixClosingMode,
        challenger: &mut E::Challenger,
    ) -> Result<
        (
            R1csInstance<F, <Pcs as MlePcs<E>>::Commitment>,
            SpartanProofKind<E, Pcs>,
        ),
        SpartanWhirError,
    >
    where
        E: Send,
        Pcs: SparkReadPcs<E>,
        Pcs::ProverData: Send,
        Pcs::Commitment: Send,
        Pcs::Proof: Send,
        <Pcs as SparkReadPcs<E>>::ReadProverData: Send,
        E::Challenger: Clone + Send,
    {
        match mode {
            MatrixClosingMode::DirectSparse => {
                let (instance, proof) = Self::prove(pk, public_inputs, witness, challenger)?;
                Ok((instance, SpartanProofKind::Direct(proof)))
            }
            MatrixClosingMode::Spark => {
                let (instance, proof) = Self::prove_spark(pk, public_inputs, witness, challenger)?;
                Ok((instance, SpartanProofKind::Spark(proof)))
            }
        }
    }

    pub fn verify_with_mode(
        vk: &VerifyingKey<E, Pcs>,
        instance: &R1csInstance<F, <Pcs as MlePcs<E>>::Commitment>,
        proof: &SpartanProofKind<E, Pcs>,
        challenger: &mut E::Challenger,
    ) -> Result<(), SpartanWhirError>
    where
        E: Sync,
        Pcs: SparkReadPcs<E>,
        Pcs::Commitment: Sync,
        Pcs::Proof: Sync,
        <Pcs as ProtocolPcs<E>>::ParsedCommitment: Send,
        <Pcs as SparkReadPcs<E>>::ParsedReadCommitment: Send,
        E::Challenger: Clone + Send,
    {
        match proof {
            SpartanProofKind::Direct(proof) => Self::verify(vk, instance, proof, challenger),
            SpartanProofKind::Spark(proof) => Self::verify_spark(vk, instance, proof, challenger),
        }
    }

    fn ensure_key_mode(
        actual: MatrixClosingMode,
        expected: MatrixClosingMode,
    ) -> Result<(), SpartanWhirError> {
        if actual != expected {
            return Err(SpartanWhirError::ProofKindMismatch);
        }
        Ok(())
    }

    pub fn prove(
        pk: &ProvingKey<E, Pcs>,
        public_inputs: &[F],
        witness: &R1csWitness<F>,
        challenger: &mut E::Challenger,
    ) -> Result<
        (
            R1csInstance<F, <Pcs as MlePcs<E>>::Commitment>,
            SpartanProof<E, Pcs>,
        ),
        SpartanWhirError,
    > {
        let mut observer = pk.observer.unwrap_or_default();
        observer.on_stage(ProtocolStage::ProveStart);
        Self::ensure_key_mode(pk.matrix_closing, MatrixClosingMode::DirectSparse)?;

        if public_inputs.len() != pk.num_io {
            return Err(SpartanWhirError::InvalidPublicInputLength);
        }
        if witness.w.len() != pk.num_vars_unpadded {
            return Err(SpartanWhirError::InvalidWitnessLength);
        }

        {
            let _profile = profile_scope("observe_spartan_context");
            observe_spartan_context::<E, E::EF>(challenger, &pk.domain_separator, public_inputs)?;
        }

        let witness_padded = {
            let _profile = profile_scope("pad_witness");
            let mut witness_padded = witness.w.clone();
            witness_padded.resize(pk.shape_canonical.num_vars, F::ZERO);
            witness_padded
        };
        {
            let _profile = profile_scope("witness_to_mle");
            debug_assert_eq!(witness_padded.len(), pk.shape_canonical.num_vars);
        }

        observer.on_stage(ProtocolStage::PcsCommit);
        let (witness_commitment, prover_data) = {
            let _profile = profile_scope("witness_pcs_commit");
            <Pcs as MlePcs<E>>::commit(&pk.pcs_config, &witness_padded, challenger)?
        };
        let instance = {
            let _profile = profile_scope("build_instance");
            R1csInstance {
                public_inputs: public_inputs.to_vec(),
                witness_commitment,
            }
        };

        let z_full = {
            let _profile = profile_scope("build_z_full");
            build_z_full(witness_padded, pk.shape_canonical.num_vars, public_inputs)
        };
        let z_short = {
            let _profile = profile_scope("build_matrix_z");
            matrix_z_slice(&z_full, pk.shape_canonical.num_vars, public_inputs.len())?
        };

        let (az_f, bz_f, cz_f) = {
            let _profile = profile_scope("r1cs_multiply_vec");
            let layout = pk.direct_multiply_layout.as_ref().ok_or_else(|| {
                SpartanWhirError::InvalidConfig(InvalidConfigReason::MissingDerivedProverData)
            })?;
            pk.shape_canonical
                .multiply_vec_parallel_with_layout_unchecked(layout, &z_short)?
        };

        let num_rounds_x = pk.shape_canonical.num_cons.ilog2() as usize;
        let tau_point = {
            let _profile = profile_scope("sample_outer_tau");
            MultilinearPoint(sample_algebra_vec::<E, E::EF>(challenger, num_rounds_x))
        };

        let (outer_sumcheck, r_x, outer_claims) = {
            let _profile = profile_scope("outer_sumcheck");
            prove_outer_split_eq_base_first_owned_unchecked::<F, E::EF, _>(
                &pk.shape_canonical,
                az_f,
                bz_f,
                cz_f,
                &tau_point,
                challenger,
            )?
        };

        let r = {
            let _profile = profile_scope("sample_inner_joint");
            challenger.observe_algebra_slice(&[outer_claims.0, outer_claims.1, outer_claims.2]);
            challenger.sample_algebra_element::<E::EF>()
        };
        let claim_inner_joint = outer_claims.0 + r * outer_claims.1 + r * r * outer_claims.2;

        let t_x = {
            let _profile = profile_scope("eq_evals_from_point");
            EqPolynomial::evals_from_point_with_base::<F>(&r_x.0)
        };
        let poly_abc: Vec<E::EF> = {
            let _profile = profile_scope("bind_row_vars_joint");
            let layout = pk.direct_bind_layout.as_ref().ok_or_else(|| {
                SpartanWhirError::invalid_config_reason(
                    InvalidConfigReason::MissingDerivedProverData,
                )
            })?;
            pk.shape_canonical
                .bind_row_vars_joint_with_layout_unchecked::<E::EF>(layout, &t_x, r)?
        };
        let (inner_sumcheck, r_y, eval_z) = {
            let _profile = profile_scope("inner_sumcheck");
            prove_inner_base_first_unchecked::<F, E::EF, _>(
                claim_inner_joint,
                poly_abc,
                &z_full,
                challenger,
            )?
        };

        let witness_eval = {
            let _profile = profile_scope("witness_eval");
            let eval_x =
                evaluate_public_half(pk.shape_canonical.num_vars, public_inputs, &r_y.0[1..])?;
            recover_witness_eval(r_y.0[0], eval_z, eval_x)?
        };

        let pcs_statement = {
            let _profile = profile_scope("build_pcs_statement");
            PcsStatementBuilder::<E>::new()
                .add_point_eval(PointEvalClaim {
                    point: MultilinearPoint(r_y.0[1..].to_vec()),
                    value: witness_eval,
                })
                .finalize()?
        };

        observer.on_stage(ProtocolStage::PcsOpen);
        let pcs_proof = {
            let _profile = profile_scope("witness_pcs_open");
            <Pcs as MlePcs<E>>::open(&pk.pcs_config, prover_data, &pcs_statement, challenger)?
        };
        observer.on_stage(ProtocolStage::ProveEnd);

        Ok((
            instance,
            SpartanProof {
                outer_sumcheck,
                outer_claims,
                inner_sumcheck,
                witness_eval,
                pcs_proof,
            },
        ))
    }

    pub fn verify(
        vk: &VerifyingKey<E, Pcs>,
        instance: &R1csInstance<F, <Pcs as MlePcs<E>>::Commitment>,
        proof: &SpartanProof<E, Pcs>,
        challenger: &mut E::Challenger,
    ) -> Result<(), SpartanWhirError> {
        vk.ensure_authenticated()?;
        let _ = validate_verifying_key::<E, Pcs>(vk)?;
        let mut observer = vk.observer.unwrap_or_default();
        observer.on_stage(ProtocolStage::VerifyStart);
        Self::ensure_key_mode(vk.matrix_closing, MatrixClosingMode::DirectSparse)?;

        if instance.public_inputs.len() != vk.num_io {
            return Err(SpartanWhirError::InvalidPublicInputLength);
        }

        observe_spartan_context::<E, E::EF>(
            challenger,
            &vk.domain_separator,
            &instance.public_inputs,
        )?;

        observer.on_stage(ProtocolStage::PcsVerify);
        let parsed_commitment = <Pcs as ProtocolPcs<E>>::verify_parse_commitment(
            &vk.pcs_config,
            &instance.witness_commitment,
            &proof.pcs_proof,
            challenger,
        )?;

        let num_rounds_x = vk.shape_canonical.num_cons.ilog2() as usize;
        let tau = sample_algebra_vec::<E, E::EF>(challenger, num_rounds_x);
        let (r_x, final_outer_claim) = {
            let _profile = profile_scope("verify_outer_sumcheck");
            verify_outer::<F, E::EF, _>(
                &proof.outer_sumcheck,
                E::EF::ZERO,
                num_rounds_x,
                challenger,
            )?
        };

        let expected_outer = eq_point_eval(&tau, &r_x.0)
            * (proof.outer_claims.0 * proof.outer_claims.1 - proof.outer_claims.2);
        if final_outer_claim != expected_outer {
            return Err(SpartanWhirError::SumcheckFailed);
        }

        challenger.observe_algebra_slice(&[
            proof.outer_claims.0,
            proof.outer_claims.1,
            proof.outer_claims.2,
        ]);
        let r = challenger.sample_algebra_element::<E::EF>();
        let claim_inner_joint =
            proof.outer_claims.0 + r * proof.outer_claims.1 + r * r * proof.outer_claims.2;

        let num_rounds_y = vk.shape_canonical.num_vars.ilog2() as usize + 1;
        let (r_y, inner_final_claim) = {
            let _profile = profile_scope("verify_inner_sumcheck");
            verify_inner::<F, E::EF, _>(
                &proof.inner_sumcheck,
                claim_inner_joint,
                num_rounds_y,
                challenger,
            )?
        };

        let t_x = {
            let _profile = profile_scope("verify_eq_table_x");
            EqPolynomial::evals_from_point_with_base::<F>(&r_x.0)
        };
        let t_y = {
            let _profile = profile_scope("verify_eq_table_y");
            EqPolynomial::evals_from_point_with_base::<F>(&r_y.0)
        };
        let (eval_a, eval_b, eval_c) = {
            let _profile = profile_scope("verify_matrix_evals");
            vk.shape_canonical
                .evaluate_with_tables::<E::EF>(&t_x, &t_y)?
        };

        let eval_x = {
            let _profile = profile_scope("verify_public_half");
            evaluate_public_half(
                vk.shape_canonical.num_vars,
                &instance.public_inputs,
                &r_y.0[1..],
            )?
        };
        let eval_z = (E::EF::ONE - r_y.0[0]) * proof.witness_eval + r_y.0[0] * eval_x;
        let expected_inner = (eval_a + r * eval_b + r * r * eval_c) * eval_z;
        if inner_final_claim != expected_inner {
            return Err(SpartanWhirError::SumcheckFailed);
        }

        let pcs_statement = PcsStatementBuilder::<E>::new()
            .add_point_eval(PointEvalClaim {
                point: MultilinearPoint(r_y.0[1..].to_vec()),
                value: proof.witness_eval,
            })
            .finalize()?;

        <Pcs as ProtocolPcs<E>>::verify_finalize(
            &vk.pcs_config,
            &parsed_commitment,
            &pcs_statement,
            &proof.pcs_proof,
            challenger,
        )?;

        observer.on_stage(ProtocolStage::VerifyEnd);
        Ok(())
    }

    pub fn prove_spark(
        pk: &ProvingKey<E, Pcs>,
        public_inputs: &[F],
        witness: &R1csWitness<F>,
        challenger: &mut E::Challenger,
    ) -> Result<
        (
            R1csInstance<F, <Pcs as MlePcs<E>>::Commitment>,
            SparkSpartanProof<E, Pcs>,
        ),
        SpartanWhirError,
    >
    where
        E: Send,
        Pcs: SparkReadPcs<E>,
        Pcs::ProverData: Send,
        Pcs::Commitment: Send,
        Pcs::Proof: Send,
        <Pcs as SparkReadPcs<E>>::ReadProverData: Send,
        E::Challenger: Clone + Send,
    {
        let mut observer = pk.observer.unwrap_or_default();
        observer.on_stage(ProtocolStage::ProveStart);
        Self::ensure_key_mode(pk.matrix_closing, MatrixClosingMode::Spark)?;

        if public_inputs.len() != pk.num_io {
            return Err(SpartanWhirError::InvalidPublicInputLength);
        }
        if witness.w.len() != pk.num_vars_unpadded {
            return Err(SpartanWhirError::InvalidWitnessLength);
        }

        {
            let _profile = profile_scope("spark_observe_context");
            observe_spartan_context::<E, E::EF>(challenger, &pk.domain_separator, public_inputs)?;
        }

        let witness_padded = {
            let _profile = profile_scope("spark_pad_witness");
            let mut witness_padded = witness.w.clone();
            witness_padded.resize(pk.shape_canonical.num_vars, F::ZERO);
            witness_padded
        };

        observer.on_stage(ProtocolStage::PcsCommit);
        let (witness_commitment, prover_data) = {
            let _profile = profile_scope("witness_pcs_commit");
            <Pcs as MlePcs<E>>::commit(&pk.pcs_config, &witness_padded, challenger)?
        };
        let instance = R1csInstance {
            public_inputs: public_inputs.to_vec(),
            witness_commitment,
        };

        let z_full = {
            let _profile = profile_scope("spark_build_z");
            build_z_full(witness_padded, pk.shape_canonical.num_vars, public_inputs)
        };
        let z_short = {
            let _profile = profile_scope("spark_build_matrix_z");
            matrix_z_slice(&z_full, pk.shape_canonical.num_vars, public_inputs.len())?
        };

        let (az_f, bz_f, cz_f) = {
            let _profile = profile_scope("r1cs_multiply_vec");
            let layout = pk.direct_multiply_layout.as_ref().ok_or_else(|| {
                SpartanWhirError::InvalidConfig(InvalidConfigReason::MissingDerivedProverData)
            })?;
            pk.shape_canonical
                .multiply_vec_parallel_with_layout_unchecked(layout, &z_short)?
        };

        let num_rounds_x = pk.shape_canonical.num_cons.ilog2() as usize;
        let tau_point = MultilinearPoint(sample_algebra_vec::<E, E::EF>(challenger, num_rounds_x));

        let (outer_sumcheck, r_x, outer_claims) = {
            let _profile = profile_scope("outer_sumcheck");
            prove_outer_split_eq_base_first_owned_unchecked::<F, E::EF, _>(
                &pk.shape_canonical,
                az_f,
                bz_f,
                cz_f,
                &tau_point,
                challenger,
            )?
        };

        challenger.observe_algebra_slice(&[outer_claims.0, outer_claims.1, outer_claims.2]);
        let r = challenger.sample_algebra_element::<E::EF>();
        let claim_inner_joint = outer_claims.0 + r * outer_claims.1 + r * r * outer_claims.2;

        let t_x = {
            let _profile = profile_scope("spark_eq_table_x");
            EqPolynomial::evals_from_point_with_base::<F>(&r_x.0)
        };
        let poly_abc = {
            let _profile = profile_scope("spark_bind_row_vars");
            let layout = pk.direct_bind_layout.as_ref().ok_or_else(|| {
                SpartanWhirError::invalid_config_reason(
                    InvalidConfigReason::MissingDerivedProverData,
                )
            })?;
            pk.shape_canonical
                .bind_row_vars_joint_with_layout_unchecked::<E::EF>(layout, &t_x, r)?
        };

        let (inner_sumcheck, r_y, eval_z) = {
            let _profile = profile_scope("inner_sumcheck");
            prove_inner_base_first_unchecked::<F, E::EF, _>(
                claim_inner_joint,
                poly_abc,
                &z_full,
                challenger,
            )?
        };

        let computed_spark_tables;
        let spark_tables = match pk.spark_tables.as_ref() {
            Some(tables) => tables,
            None => {
                let _profile = profile_scope("spark_preprocess_tables");
                computed_spark_tables = preprocess_spark_tables(&pk.shape_canonical)?;
                &computed_spark_tables
            }
        };
        let spark_pcs_configs = pk
            .spark_pcs_configs
            .clone()
            .ok_or(SpartanWhirError::invalid_config())?;
        let fixed_prover_data = pk
            .spark_fixed_prover_data
            .clone()
            .ok_or(SpartanWhirError::invalid_config())?;
        let expected_fixed_commitments = pk
            .spark_fixed_commitments
            .clone()
            .ok_or(SpartanWhirError::invalid_config())?;
        let fixed_prover_data = {
            let _profile = profile_scope("spark_prepare_fixed_openings");
            prepare_spark_fixed_openings::<E, E::EF, Pcs>(
                &spark_pcs_configs.fixed_value,
                &spark_pcs_configs.fixed_audit,
                fixed_prover_data,
                challenger,
            )?
        };
        let read_tables = {
            let _profile = profile_scope("spark_compute_read_tables");
            compute_spark_read_tables(spark_tables, &r_x, &r_y)?
        };
        let (read_prover_data, read_commitments) = {
            let _profile = profile_scope("spark_commit_read_tables");
            commit_spark_read_tables::<E, E::EF, Pcs>(
                &spark_pcs_configs.read,
                &read_tables,
                challenger,
            )?
        };
        let (spark_products, product_claims) = {
            let _profile = profile_scope("spark_memory_products");
            prove_spark_batched_memory_products_with_read_tables_and_leaf_claims(
                spark_tables,
                &r_x,
                &r_y,
                &read_tables,
                challenger,
            )?
        };
        let (spark_fixed_openings, spark_read_openings) = open_spark_table_openings::<E, E::EF, Pcs>(
            &spark_pcs_configs,
            fixed_prover_data,
            expected_fixed_commitments,
            read_prover_data,
            read_commitments,
            spark_tables,
            &product_claims,
            false,
            challenger,
        )?;

        let witness_eval = {
            let _profile = profile_scope("witness_eval");
            let eval_x =
                evaluate_public_half(pk.shape_canonical.num_vars, public_inputs, &r_y.0[1..])?;
            recover_witness_eval(r_y.0[0], eval_z, eval_x)?
        };

        let pcs_statement = {
            let _profile = profile_scope("spark_build_pcs_statement");
            PcsStatementBuilder::<E>::new()
                .add_point_eval(PointEvalClaim {
                    point: MultilinearPoint(r_y.0[1..].to_vec()),
                    value: witness_eval,
                })
                .finalize()?
        };

        observer.on_stage(ProtocolStage::PcsOpen);
        let pcs_proof = {
            let _profile = profile_scope("witness_pcs_open");
            <Pcs as MlePcs<E>>::open(&pk.pcs_config, prover_data, &pcs_statement, challenger)?
        };
        observer.on_stage(ProtocolStage::ProveEnd);

        Ok((
            instance,
            SparkSpartanProof {
                outer_sumcheck,
                outer_claims,
                inner_sumcheck,
                witness_eval,
                spark_products,
                spark_fixed_openings,
                spark_read_openings,
                pcs_proof,
            },
        ))
    }

    pub fn verify_spark(
        vk: &VerifyingKey<E, Pcs>,
        instance: &R1csInstance<F, <Pcs as MlePcs<E>>::Commitment>,
        proof: &SparkSpartanProof<E, Pcs>,
        challenger: &mut E::Challenger,
    ) -> Result<(), SpartanWhirError>
    where
        E: Sync,
        Pcs: SparkReadPcs<E>,
        Pcs::Commitment: Sync,
        Pcs::Proof: Sync,
        <Pcs as ProtocolPcs<E>>::ParsedCommitment: Send,
        <Pcs as SparkReadPcs<E>>::ParsedReadCommitment: Send,
        E::Challenger: Clone + Send,
    {
        vk.ensure_authenticated()?;
        let validated_spark_metadata = validate_verifying_key::<E, Pcs>(vk)?;
        let mut observer = vk.observer.unwrap_or_default();
        observer.on_stage(ProtocolStage::VerifyStart);
        Self::ensure_key_mode(vk.matrix_closing, MatrixClosingMode::Spark)?;
        let spark_metadata =
            validated_spark_metadata.ok_or_else(SpartanWhirError::invalid_config)?;

        if instance.public_inputs.len() != vk.num_io {
            return Err(SpartanWhirError::InvalidPublicInputLength);
        }

        observe_spartan_context::<E, E::EF>(
            challenger,
            &vk.domain_separator,
            &instance.public_inputs,
        )?;

        observer.on_stage(ProtocolStage::PcsVerify);
        let parsed_commitment = <Pcs as ProtocolPcs<E>>::verify_parse_commitment(
            &vk.pcs_config,
            &instance.witness_commitment,
            &proof.pcs_proof,
            challenger,
        )?;

        let num_rounds_x = vk.shape_canonical.num_cons.ilog2() as usize;
        let tau = sample_algebra_vec::<E, E::EF>(challenger, num_rounds_x);
        let (r_x, final_outer_claim) = verify_outer::<F, E::EF, _>(
            &proof.outer_sumcheck,
            E::EF::ZERO,
            num_rounds_x,
            challenger,
        )?;

        let expected_outer = eq_point_eval(&tau, &r_x.0)
            * (proof.outer_claims.0 * proof.outer_claims.1 - proof.outer_claims.2);
        if final_outer_claim != expected_outer {
            return Err(SpartanWhirError::SumcheckFailed);
        }

        challenger.observe_algebra_slice(&[
            proof.outer_claims.0,
            proof.outer_claims.1,
            proof.outer_claims.2,
        ]);
        let r = challenger.sample_algebra_element::<E::EF>();
        let claim_inner_joint =
            proof.outer_claims.0 + r * proof.outer_claims.1 + r * r * proof.outer_claims.2;

        let num_rounds_y = vk.shape_canonical.num_vars.ilog2() as usize + 1;
        let (r_y, inner_final_claim) = verify_inner::<F, E::EF, _>(
            &proof.inner_sumcheck,
            claim_inner_joint,
            num_rounds_y,
            challenger,
        )?;

        let spark_pcs_configs = vk
            .spark_pcs_configs
            .clone()
            .ok_or(SpartanWhirError::invalid_config())?;
        let expected_fixed_commitments = vk
            .spark_fixed_commitments
            .clone()
            .ok_or(SpartanWhirError::invalid_config())?;
        validate_spark_fixed_commitments::<E, E::EF, Pcs>(
            &proof.spark_fixed_openings,
            &expected_fixed_commitments,
        )?;
        let (parsed_fixed_openings, parsed_read_openings) =
            parse_spark_table_openings::<E, E::EF, Pcs>(
                &spark_pcs_configs,
                spark_fixed_audit_is_embedded(
                    spark_metadata.value_domain_size,
                    spark_metadata.row_memory_size,
                    spark_metadata.col_memory_size,
                ),
                &proof.spark_fixed_openings,
                &proof.spark_read_openings,
                challenger,
            )?;
        let product_claims = verify_spark_batched_memory_product_claims_with_metadata(
            &spark_metadata,
            &proof.spark_products,
            challenger,
        )?;
        let read_opening_evals = finalize_spark_table_openings::<E, E::EF, Pcs>(
            &spark_pcs_configs,
            spark_metadata.row_memory_size,
            spark_metadata.col_memory_size,
            &proof.spark_fixed_openings,
            &proof.spark_read_openings,
            parsed_fixed_openings,
            parsed_read_openings,
            &product_claims,
            false,
            None,
            challenger,
        )?;
        verify_spark_batched_memory_leaf_claims_with_openings(
            spark_metadata.row_memory_size,
            spark_metadata.col_memory_size,
            &product_claims,
            &proof.spark_fixed_openings.evals,
            &read_opening_evals,
            &r_x,
            &r_y,
        )?;

        let eval_x = evaluate_public_half(
            vk.shape_canonical.num_vars,
            &instance.public_inputs,
            &r_y.0[1..],
        )?;
        let eval_z = (E::EF::ONE - r_y.0[0]) * proof.witness_eval + r_y.0[0] * eval_x;
        let spark_matrix_eval = matrix_eval_rlc(product_claims.matrix_evals, r);
        if inner_final_claim != spark_matrix_eval * eval_z {
            return Err(SpartanWhirError::SumcheckFailed);
        }

        let pcs_statement = PcsStatementBuilder::<E>::new()
            .add_point_eval(PointEvalClaim {
                point: MultilinearPoint(r_y.0[1..].to_vec()),
                value: proof.witness_eval,
            })
            .finalize()?;

        <Pcs as ProtocolPcs<E>>::verify_finalize(
            &vk.pcs_config,
            &parsed_commitment,
            &pcs_statement,
            &proof.pcs_proof,
            challenger,
        )?;

        observer.on_stage(ProtocolStage::VerifyEnd);
        Ok(())
    }
}

impl<E> PoseidonZkSpartanProtocolFor<E>
where
    E: FullZkPoseidonEngine,
    E::EF: ExtField + Serialize + for<'de> Deserialize<'de>,
    E::Challenger: CanObserve<PoseidonZkCommitmentFor<E>>
        + CanSampleUniformBits<F>
        + FieldChallenger<F>
        + p3_challenger::GrindingChallenger<Witness = F>
        + Clone
        + Send,
    StandardUniform: Distribution<E::EF> + Distribution<F>,
    Plonky3WhirPcs: FullZkPoseidonPcs<E>,
    MaskGroupProverData<F, E::EF, PoseidonZkMmcsFor<E>>: Send,
{
    pub fn prove(
        pk: &PoseidonZkProvingKeyFor<E>,
        public_inputs: &[F],
        witness: &R1csWitness<F>,
        challenger: &mut E::Challenger,
    ) -> Result<
        (
            R1csInstance<F, PoseidonZkCommitmentFor<E>>,
            ZkSpartanProofFor<E>,
        ),
        SpartanWhirError,
    > {
        let mut rng = rand::make_rng::<StdRng>();
        Self::prove_with_rng(pk, public_inputs, witness, challenger, &mut rng)
    }

    pub fn prove_with_rng<R>(
        pk: &PoseidonZkProvingKeyFor<E>,
        public_inputs: &[F],
        witness: &R1csWitness<F>,
        challenger: &mut E::Challenger,
        rng: &mut R,
    ) -> Result<
        (
            R1csInstance<F, PoseidonZkCommitmentFor<E>>,
            ZkSpartanProofFor<E>,
        ),
        SpartanWhirError,
    >
    where
        R: Rng + CryptoRng,
    {
        Self::prove_with_rng_impl(pk, public_inputs, witness, challenger, rng, false, false)
            .map(|(instance, proof, _)| (instance, proof))
    }

    /// Produce a compact transport proof with the same protocol transcript.
    pub fn prove_compressed_with_rng<R>(
        pk: &PoseidonZkProvingKeyFor<E>,
        public_inputs: &[F],
        witness: &R1csWitness<F>,
        challenger: &mut E::Challenger,
        rng: &mut R,
        options: crate::proof_compression::ProofCompressionOptions,
    ) -> Result<
        (
            R1csInstance<F, PoseidonZkCommitmentFor<E>>,
            crate::proof_compression::CompressedZkProofFor<E>,
        ),
        SpartanWhirError,
    >
    where
        R: Rng + CryptoRng,
        E::PlainProof: crate::proof_compression::PlainProofRowAccess,
    {
        if pk.matrix_closing != MatrixClosingMode::Spark {
            return Err(SpartanWhirError::ProofKindMismatch);
        }
        let (instance, proof, products) = Self::prove_with_rng_impl(
            pk,
            public_inputs,
            witness,
            challenger,
            rng,
            options.factored_rounds,
            options.final_rows,
        )?;
        Ok((
            instance,
            crate::proof_compression::CompressedZkProofFor::from_proof(proof, options, products)?,
        ))
    }

    fn prove_with_rng_impl<R>(
        pk: &PoseidonZkProvingKeyFor<E>,
        public_inputs: &[F],
        witness: &R1csWitness<F>,
        challenger: &mut E::Challenger,
        rng: &mut R,
        factor_products: bool,
        final_rows: bool,
    ) -> Result<
        (
            R1csInstance<F, PoseidonZkCommitmentFor<E>>,
            ZkSpartanProofFor<E>,
            Option<crate::spark::CompactSparkBatchedMemoryProductsProof<E::EF>>,
        ),
        SpartanWhirError,
    >
    where
        R: Rng + CryptoRng,
    {
        let mut compressed_products = None;
        if public_inputs.len() != pk.num_io {
            return Err(SpartanWhirError::InvalidPublicInputLength);
        }
        if witness.w.len() != pk.num_vars_unpadded {
            return Err(SpartanWhirError::InvalidWitnessLength);
        }

        let pcs_config = &pk.pcs_config;
        let num_outer_rounds = pk.shape_canonical.num_cons.ilog2() as usize;
        let num_inner_rounds = pk.shape_canonical.num_vars.ilog2() as usize + 1;
        if num_outer_rounds == 0 {
            return Err(SpartanWhirError::invalid_config());
        }
        observe_poseidon_zk_context::<E>(
            challenger,
            &pk.domain_separator,
            &pk.pcs_config,
            num_outer_rounds,
            num_inner_rounds,
            public_inputs,
        );

        let (pcs, [inner_shape, outer_shape, inner_sumcheck_shape]) =
            build_poseidon_full_zk_pcs::<E>(
                &pk.pcs_config,
                num_outer_rounds,
                num_inner_rounds,
                pk.security.effective_security_bits(),
            )?;
        let application_shape =
            combined_application_mask_shape(inner_shape, outer_shape, pcs_config.mask_packing)?;
        let relation_shapes = [application_shape, inner_sumcheck_shape];
        let whir_prover = HidingWhirProver::new(&pcs.config, &pcs.dft, &pcs.mmcs);

        let inner_messages = {
            let _profile = profile_scope("zk_inner_mask_sample");
            sample_inner_masks::<E::EF, _>(num_outer_rounds, rng)
        };
        let outer_messages = {
            let _profile = profile_scope("zk_outer_mask_sample");
            sample_outer_masks::<E::EF, _>(num_outer_rounds, rng)
        };
        let application_messages = combine_application_mask_vectors(
            inner_messages.clone(),
            outer_messages.clone(),
            outer_shape.shape.message_len,
            pcs_config.mask_packing,
            false,
        )?;
        let mut application_group = {
            let _profile = profile_scope("zk_application_mask_commit");
            whir_prover
                .commit_mask_group(application_shape, application_messages, challenger, rng)
                .map_err(|_| SpartanWhirError::WhirCommitFailed)?
        };
        let application_mask_commitment = application_group.commitment.clone();

        observe_poseidon_relation_domain_separator::<E>(&pcs, &relation_shapes, challenger);
        let mut witness_padded = witness.w.clone();
        witness_padded.resize(pk.shape_canonical.num_vars, F::ZERO);
        let (witness_commitment, witness_prover_data) = {
            let _profile = profile_scope("zk_witness_commit");
            whir_prover.commit(Poly::new(witness_padded.clone()), challenger, rng)
        };
        let instance = R1csInstance {
            public_inputs: public_inputs.to_vec(),
            witness_commitment: witness_commitment.clone(),
        };

        let z_full = {
            let _profile = profile_scope("zk_build_z_full");
            build_z_full(witness_padded, pk.shape_canonical.num_vars, public_inputs)
        };
        let z_short = matrix_z_slice(&z_full, pk.shape_canonical.num_vars, public_inputs.len())?;
        let layout = pk.direct_multiply_layout.as_ref().ok_or({
            SpartanWhirError::InvalidConfig(InvalidConfigReason::MissingDerivedProverData)
        })?;
        let (az, bz, cz) = {
            let _profile = profile_scope("zk_r1cs_multiply_vec");
            pk.shape_canonical
                .multiply_vec_parallel_with_layout_unchecked(layout, z_short)?
        };
        let outer = {
            let _profile = profile_scope("zk_outer_sumcheck");
            prove_outer_zk_base_first_unchecked::<F, E::EF, _>(
                &pk.shape_canonical,
                az,
                bz,
                cz,
                &inner_messages,
                &outer_messages,
                challenger,
            )?
        };
        let (rho, batching) = {
            let _profile = profile_scope("zk_outer_claim_observe");
            challenger.observe_algebra_slice(&outer.outer_mask_evals);
            challenger.observe_algebra_slice(&[
                outer.masked_claims.0,
                outer.masked_claims.1,
                outer.masked_claims.2,
            ]);
            (
                challenger.sample_algebra_element::<E::EF>(),
                challenger.sample_algebra_element::<E::EF>(),
            )
        };
        let t_x = {
            let _profile = profile_scope("zk_outer_eq_table");
            EqPolynomial::evals_from_point_with_base::<F>(&outer.point.0)
        };
        let layout = pk.direct_bind_layout.as_ref().ok_or_else(|| {
            SpartanWhirError::invalid_config_reason(InvalidConfigReason::MissingDerivedProverData)
        })?;
        let product = if z_full.len() > <F as Field>::Packing::WIDTH {
            let packed_weights = {
                let _profile = profile_scope("zk_bind_row_vars_joint");
                pk.shape_canonical
                    .bind_row_vars_joint_packed_with_layout_unchecked::<E::EF>(layout, &t_x, rho)?
            };
            let _profile = profile_scope("zk_inner_product_build");
            ProductPolynomial::new_base_packed(Poly::new(z_full), Poly::new(packed_weights))
        } else {
            let weights = {
                let _profile = profile_scope("zk_bind_row_vars_joint");
                pk.shape_canonical
                    .bind_row_vars_joint_with_layout_unchecked::<E::EF>(layout, &t_x, rho)?
            };
            let _profile = profile_scope("zk_inner_product_build");
            unpacked_inner_product::<E::EF>(z_full, weights)?
        };

        let (inner_covectors, outer_covectors, joint_target) = {
            let _profile = profile_scope("zk_application_relation");
            application_relation(
                num_outer_rounds,
                &outer.point.0,
                outer.masked_claims,
                &outer.outer_mask_evals,
                rho,
                batching,
            )
        };
        application_group.covectors = combine_application_mask_vectors(
            inner_covectors,
            outer_covectors,
            outer_shape.shape.message_len,
            pcs_config.mask_packing,
            true,
        )?;
        let application_aux = application_group.claim();
        let source_claim = joint_target - application_aux;
        let sumcheck_prover = SumcheckProver::new(product, source_claim);
        let extension_mmcs = ExtensionMmcs::new(pcs.mmcs.clone());
        let encoding = pcs.config.sumcheck_mask.encoding::<E::EF>();
        let mut inner_sumcheck = ZkSumcheckData::default();
        let handoff = {
            let _profile = profile_scope("zk_inner_sumcheck");
            sumcheck_prover.into_zk_sumcheck_with_mask_packing(
                &mut inner_sumcheck,
                &encoding,
                &inner_sumcheck_shape.shape.encoding::<E::EF>(),
                pcs_config.mask_packing.sumchecks(),
                &extension_mmcs,
                num_inner_rounds,
                0,
                application_aux,
                challenger,
                rng,
            )
        };
        let (
            inner_sumcheck_mask_commitment,
            sumcheck_group,
            r_y,
            inner_epsilon,
            weighted_matrix_eval,
            joint_residual,
        ) = {
            let _profile = profile_scope("zk_inner_sumcheck_finalize");
            let inner_sumcheck_mask_commitment = handoff.mask_oracle.0.clone();
            let r_y = handoff.randomness.clone();
            let inner_epsilon = handoff.eps;
            let source_residual = handoff.residual_prover.claimed_sum();
            let residual_weights = handoff.residual_prover.weights();
            let [weighted_matrix_eval] = residual_weights.as_slice() else {
                return Err(SpartanWhirError::InvalidRoundCount);
            };
            let weighted_matrix_eval = *weighted_matrix_eval;

            let carry_scale = inner_epsilon * E::EF::TWO.exp_u64(num_inner_rounds as u64).inverse();
            scale_covectors(&mut application_group.covectors, carry_scale);
            let sumcheck_covectors = mask_residual_covectors_from_shape(
                num_inner_rounds,
                pcs.config.sumcheck_mask.message_len,
                r_y.as_slice(),
            );
            let sumcheck_group = CommittedMaskGroupProverData {
                shape: inner_sumcheck_shape,
                commitment: inner_sumcheck_mask_commitment.clone(),
                messages: pack_mask_vectors(
                    handoff.mask_messages,
                    pcs_config.mask_packing.sumchecks(),
                ),
                randomness: handoff.mask_randomness,
                covectors: pack_mask_vectors(
                    sumcheck_covectors,
                    pcs_config.mask_packing.sumchecks(),
                ),
                prover_data: handoff.mask_oracle.1,
            };

            let joint_residual =
                source_residual + application_group.claim() + sumcheck_group.claim();
            (
                inner_sumcheck_mask_commitment,
                sumcheck_group,
                r_y,
                inner_epsilon,
                weighted_matrix_eval,
                joint_residual,
            )
        };
        let matrix_closing = match pk.matrix_closing {
            MatrixClosingMode::DirectSparse => ZkMatrixClosingProofFor::DirectSparse,
            MatrixClosingMode::Spark => {
                let spark_tables = pk
                    .spark_tables
                    .as_ref()
                    .ok_or_else(SpartanWhirError::invalid_config)?;
                let spark_pcs_configs = pk
                    .spark_pcs_configs
                    .as_ref()
                    .ok_or_else(SpartanWhirError::invalid_config)?;
                let fixed_prover_data = pk
                    .spark_fixed_prover_data
                    .clone()
                    .ok_or_else(SpartanWhirError::invalid_config)?;
                let expected_fixed_commitments = pk
                    .spark_fixed_commitments
                    .clone()
                    .ok_or_else(SpartanWhirError::invalid_config)?;
                let fixed_prover_data = {
                    let _profile = profile_scope("zk_spark_prepare_fixed_openings");
                    prepare_spark_fixed_openings::<E, E::EF, Plonky3WhirPcs>(
                        &spark_pcs_configs.fixed_value,
                        &spark_pcs_configs.fixed_audit,
                        fixed_prover_data,
                        challenger,
                    )?
                };
                let r_y = MultilinearPoint(r_y.as_slice().to_vec());
                let read_tables = {
                    let _profile = profile_scope("zk_spark_compute_read_tables");
                    compute_spark_read_tables(spark_tables, &outer.point, &r_y)?
                };
                let (read_prover_data, read_commitments) = {
                    let _profile = profile_scope("zk_spark_commit_read_tables");
                    commit_spark_read_tables::<E, E::EF, Plonky3WhirPcs>(
                        &spark_pcs_configs.read,
                        &read_tables,
                        challenger,
                    )?
                };
                let product_replay = factor_products.then(|| E::challenger_for_replay(challenger));
                let (spark_products, product_claims) = {
                    let _profile = profile_scope("zk_spark_memory_products");
                    prove_spark_batched_memory_products_with_read_tables_and_leaf_claims(
                        spark_tables,
                        &outer.point,
                        &r_y,
                        &read_tables,
                        challenger,
                    )?
                };
                if let Some(mut replay) = product_replay {
                    compressed_products = Some(crate::spark::compress_spark_memory_products(
                        &spark_products,
                        &spark_tables.metadata(),
                        &mut replay,
                    )?);
                }
                let (spark_fixed_openings, spark_read_openings) =
                    open_spark_table_openings::<E, E::EF, Plonky3WhirPcs>(
                        spark_pcs_configs,
                        fixed_prover_data,
                        expected_fixed_commitments,
                        read_prover_data,
                        read_commitments,
                        spark_tables,
                        &product_claims,
                        final_rows,
                        challenger,
                    )?;
                let spark_matrix_eval = matrix_eval_rlc(product_claims.matrix_evals, rho);
                if weighted_matrix_eval != inner_epsilon * spark_matrix_eval {
                    return Err(SpartanWhirError::SparkMatrixEvaluationMismatch);
                }
                ZkMatrixClosingProofFor::Spark(ZkSparkClosingProofFor {
                    spark_products,
                    spark_fixed_openings,
                    spark_read_openings,
                })
            }
        };
        let selector = r_y.as_slice()[0];
        let eval_public = evaluate_public_half(
            pk.shape_canonical.num_vars,
            public_inputs,
            &r_y.as_slice()[1..],
        )?;
        let public_term = weighted_matrix_eval * selector * eval_public;
        let mut relation = CommittedRelation::empty();
        relation.source.push_eq(
            Point::new(r_y.as_slice()[1..].to_vec()),
            weighted_matrix_eval * (E::EF::ONE - selector),
        );
        relation.target = joint_residual - public_term;
        let pcs_proof = {
            let _profile = profile_scope("zk_pcs_relation");
            whir_prover
                .prove_relation(
                    witness_prover_data,
                    &relation,
                    vec![application_group, sumcheck_group],
                    challenger,
                    rng,
                )
                .map_err(|_| SpartanWhirError::WhirOpenFailed)?
        };

        Ok((
            instance,
            ZkSpartanProofFor {
                application_mask_commitment,
                outer_sumcheck: outer.proof,
                outer_claims: outer.masked_claims,
                outer_mask_evals: outer.outer_mask_evals,
                inner_sumcheck,
                inner_sumcheck_mask_commitment,
                matrix_closing,
                pcs_proof,
            },
            compressed_products,
        ))
    }

    pub fn verify(
        vk: &PoseidonZkVerifyingKeyFor<E>,
        instance: &R1csInstance<F, PoseidonZkCommitmentFor<E>>,
        proof: &ZkSpartanProofFor<E>,
        challenger: &mut E::Challenger,
    ) -> Result<(), SpartanWhirError> {
        Self::verify_with_compression(vk, instance, proof, challenger, None, false, false, None)
    }

    /// Decode compressed field values and run all original verification checks.
    pub fn verify_compressed(
        vk: &PoseidonZkVerifyingKeyFor<E>,
        instance: &R1csInstance<F, PoseidonZkCommitmentFor<E>>,
        mut proof: crate::proof_compression::CompressedZkProofFor<E>,
        challenger: &mut E::Challenger,
    ) -> Result<(), SpartanWhirError>
    where
        E::PlainProof: crate::proof_compression::PlainProofRowAccess,
    {
        vk.ensure_authenticated()?;
        proof.restore_static_rows()?;
        Self::verify_with_compression(
            vk,
            instance,
            &proof.proof,
            challenger,
            proof.products.as_ref(),
            proof.options.fresh_rows,
            proof.options.final_rows,
            None,
        )
    }

    pub(crate) fn verify_with_compression(
        vk: &PoseidonZkVerifyingKeyFor<E>,
        instance: &R1csInstance<F, PoseidonZkCommitmentFor<E>>,
        proof: &ZkSpartanProofFor<E>,
        challenger: &mut E::Challenger,
        compact_products: Option<&crate::spark::CompactSparkBatchedMemoryProductsProof<E::EF>>,
        restore_fresh: bool,
        final_rows: bool,
        restore_fixed: Option<
            &dyn crate::fixed_oracle_cache::FixedOpeningRestorer<E, Plonky3WhirPcs>,
        >,
    ) -> Result<(), SpartanWhirError> {
        vk.ensure_authenticated()?;
        let validated_spark_metadata = vk.validate()?;
        if proof.matrix_closing.mode() != vk.matrix_closing {
            return Err(SpartanWhirError::ProofKindMismatch);
        }
        if instance.public_inputs.len() != vk.num_io {
            return Err(SpartanWhirError::InvalidPublicInputLength);
        }
        let pcs_config = &vk.pcs_config;
        let num_outer_rounds = vk.shape_canonical.num_cons.ilog2() as usize;
        let num_inner_rounds = vk.shape_canonical.num_vars.ilog2() as usize + 1;
        let (pcs, [inner_shape, outer_shape, inner_sumcheck_shape]) =
            build_poseidon_full_zk_pcs::<E>(
                &vk.pcs_config,
                num_outer_rounds,
                num_inner_rounds,
                vk.security.effective_security_bits(),
            )?;
        validate_poseidon_zk_proof_commitments(
            vk,
            validated_spark_metadata.as_ref(),
            instance,
            proof,
            &pcs,
            final_rows,
            restore_fixed,
        )?;
        let application_shape =
            combined_application_mask_shape(inner_shape, outer_shape, pcs_config.mask_packing)?;
        observe_poseidon_zk_context::<E>(
            challenger,
            &vk.domain_separator,
            &vk.pcs_config,
            num_outer_rounds,
            num_inner_rounds,
            &instance.public_inputs,
        );
        let relation_shapes = [application_shape, inner_sumcheck_shape];
        challenger.observe(proof.application_mask_commitment.clone());
        observe_poseidon_relation_domain_separator::<E>(&pcs, &relation_shapes, challenger);
        challenger.observe(instance.witness_commitment.clone());

        let r_x = verify_outer_zk::<F, E::EF, _>(
            &proof.outer_sumcheck,
            proof.outer_claims,
            &proof.outer_mask_evals,
            num_outer_rounds,
            challenger,
        )?;
        challenger.observe_algebra_slice(&proof.outer_mask_evals);
        challenger.observe_algebra_slice(&[
            proof.outer_claims.0,
            proof.outer_claims.1,
            proof.outer_claims.2,
        ]);
        let rho = challenger.sample_algebra_element::<E::EF>();
        let batching = challenger.sample_algebra_element::<E::EF>();
        let (inner_covectors, outer_covectors, joint_target) = application_relation(
            num_outer_rounds,
            &r_x.0,
            proof.outer_claims,
            &proof.outer_mask_evals,
            rho,
            batching,
        );
        let handoff = ZkVerifier::<F, E::EF>::verify_claim::<
            ExtensionMmcs<F, E::EF, PoseidonZkMmcsFor<E>>,
            _,
        >(
            &proof.inner_sumcheck,
            &proof.inner_sumcheck_mask_commitment,
            pcs.config.sumcheck_mask.message_len,
            num_inner_rounds,
            0,
            joint_target,
            challenger,
        )
        .map_err(|_| SpartanWhirError::SumcheckFailed)?;
        let carry_scale = handoff.eps * E::EF::TWO.exp_u64(num_inner_rounds as u64).inverse();
        let mut application_covectors = combine_application_mask_vectors(
            inner_covectors,
            outer_covectors,
            outer_shape.shape.message_len,
            pcs_config.mask_packing,
            true,
        )?;
        scale_covectors(&mut application_covectors, carry_scale);
        let sumcheck_covectors = mask_residual_covectors_from_shape(
            num_inner_rounds,
            pcs.config.sumcheck_mask.message_len,
            handoff.randomness.as_slice(),
        );

        let matrix_eval = match &proof.matrix_closing {
            ZkMatrixClosingProofFor::DirectSparse => {
                let t_x = EqPolynomial::evals_from_point_with_base::<F>(&r_x.0);
                let t_y =
                    EqPolynomial::evals_from_point_with_base::<F>(handoff.randomness.as_slice());
                let (eval_a, eval_b, eval_c) = vk
                    .shape_canonical
                    .evaluate_with_tables::<E::EF>(&t_x, &t_y)?;
                eval_a + rho * eval_b + rho * rho * eval_c
            }
            ZkMatrixClosingProofFor::Spark(closing) => {
                let spark_metadata = validated_spark_metadata
                    .as_ref()
                    .ok_or_else(SpartanWhirError::invalid_config)?;
                let spark_pcs_configs = vk
                    .spark_pcs_configs
                    .as_ref()
                    .ok_or_else(SpartanWhirError::invalid_config)?;
                let expected_fixed_commitments = vk
                    .spark_fixed_commitments
                    .as_ref()
                    .ok_or_else(SpartanWhirError::invalid_config)?;
                validate_spark_fixed_commitments::<E, E::EF, Plonky3WhirPcs>(
                    &closing.spark_fixed_openings,
                    expected_fixed_commitments,
                )?;
                let (parsed_fixed_openings, parsed_read_openings) =
                    parse_spark_table_openings::<E, E::EF, Plonky3WhirPcs>(
                        spark_pcs_configs,
                        spark_fixed_audit_is_embedded(
                            spark_metadata.value_domain_size,
                            spark_metadata.row_memory_size,
                            spark_metadata.col_memory_size,
                        ),
                        &closing.spark_fixed_openings,
                        &closing.spark_read_openings,
                        challenger,
                    )?;
                let restored_products;
                let product_proof = if let Some(compact) = compact_products {
                    let mut replay = E::challenger_for_replay(challenger);
                    restored_products = crate::spark::decompress_spark_memory_products(
                        compact,
                        spark_metadata,
                        &mut replay,
                    )?;
                    &restored_products
                } else {
                    &closing.spark_products
                };
                let product_claims = verify_spark_batched_memory_product_claims_with_metadata(
                    spark_metadata,
                    product_proof,
                    challenger,
                )?;
                let read_opening_evals = finalize_spark_table_openings::<E, E::EF, Plonky3WhirPcs>(
                    spark_pcs_configs,
                    spark_metadata.row_memory_size,
                    spark_metadata.col_memory_size,
                    &closing.spark_fixed_openings,
                    &closing.spark_read_openings,
                    parsed_fixed_openings,
                    parsed_read_openings,
                    &product_claims,
                    final_rows,
                    restore_fixed,
                    challenger,
                )?;
                let r_y = MultilinearPoint(handoff.randomness.as_slice().to_vec());
                verify_spark_batched_memory_leaf_claims_with_openings(
                    spark_metadata.row_memory_size,
                    spark_metadata.col_memory_size,
                    &product_claims,
                    &closing.spark_fixed_openings.evals,
                    &read_opening_evals,
                    &r_x,
                    &r_y,
                )?;
                matrix_eval_rlc(product_claims.matrix_evals, rho)
            }
        };
        let selector = handoff.randomness.as_slice()[0];
        let eval_public = evaluate_public_half(
            vk.shape_canonical.num_vars,
            &instance.public_inputs,
            &handoff.randomness.as_slice()[1..],
        )?;
        let mut relation = CommittedRelation::empty();
        relation.source.push_eq(
            Point::new(handoff.randomness.as_slice()[1..].to_vec()),
            handoff.eps * matrix_eval * (E::EF::ONE - selector),
        );
        relation.target =
            handoff.claimed_residual - handoff.eps * matrix_eval * selector * eval_public;
        let groups = vec![
            CommittedMaskGroup {
                shape: application_shape,
                commitment: proof.application_mask_commitment.clone(),
                covectors: application_covectors,
            },
            CommittedMaskGroup {
                shape: inner_sumcheck_shape,
                commitment: proof.inner_sumcheck_mask_commitment.clone(),
                covectors: pack_mask_vectors(
                    sumcheck_covectors,
                    pcs_config.mask_packing.sumchecks(),
                ),
            },
        ];
        let restored_relation;
        let relation_proof = if restore_fresh {
            let mut restored = proof.pcs_proof.clone();
            crate::proof_compression_hiding::restore_fresh_rows::<E>(
                &pcs.config,
                &pcs.mmcs,
                &relation_shapes,
                relation.target,
                &mut restored,
                challenger,
            )?;
            restored_relation = restored;
            &restored_relation
        } else {
            &proof.pcs_proof
        };
        let _profile = profile_scope("zk_pcs_relation_verify");
        HidingWhirVerifier::new(&pcs.config, &pcs.mmcs)
            .verify_relation(
                relation_proof,
                &instance.witness_commitment,
                relation,
                groups,
                challenger,
            )
            .map_err(|_| SpartanWhirError::WhirVerifyFailed)
    }
}

fn validate_poseidon_zk_proof_commitments<E>(
    vk: &PoseidonZkVerifyingKeyFor<E>,
    spark_metadata: Option<&SparkTableMetadata>,
    instance: &R1csInstance<F, PoseidonZkCommitmentFor<E>>,
    proof: &ZkSpartanProofFor<E>,
    pcs: &PoseidonZkHidingPcsFor<E>,
    final_rows: bool,
    restore_fixed: Option<&dyn crate::fixed_oracle_cache::FixedOpeningRestorer<E, Plonky3WhirPcs>>,
) -> Result<(), SpartanWhirError>
where
    E: FullZkPoseidonEngine,
    E::EF: ExtField,
    E::Challenger: CanObserve<PoseidonZkCommitmentFor<E>>
        + CanSampleUniformBits<F>
        + FieldChallenger<F>
        + p3_challenger::GrindingChallenger<Witness = F>
        + Clone
        + Send,
    StandardUniform: Distribution<E::EF>,
    Plonky3WhirPcs: FullZkPoseidonPcs<E>,
{
    use crate::plonky3_whir_pcs::validate_poseidon_relation_proof_shape;

    E::validate_commitment(&instance.witness_commitment)?;
    E::validate_commitment(&proof.application_mask_commitment)?;
    E::validate_commitment(&proof.inner_sumcheck_mask_commitment)?;
    validate_poseidon_relation_proof_shape::<E>(&pcs.config, &proof.pcs_proof)?;

    if let ZkMatrixClosingProofFor::Spark(closing) = &proof.matrix_closing {
        let configs = vk
            .spark_pcs_configs
            .as_ref()
            .ok_or_else(SpartanWhirError::invalid_config)?;
        let metadata = spark_metadata.ok_or_else(SpartanWhirError::invalid_config)?;
        let audit_embedded = spark_fixed_audit_is_embedded(
            metadata.value_domain_size,
            metadata.row_memory_size,
            metadata.col_memory_size,
        );
        if closing.spark_read_openings.groups.len() != configs.read.len() {
            return Err(SpartanWhirError::InvalidProofShape);
        }
        validate_spark_fixed_opening_shape::<E, E::EF, Plonky3WhirPcs>(
            &configs.fixed_value,
            &configs.fixed_audit,
            audit_embedded,
            &closing.spark_fixed_openings,
        )?;
        validate_spark_read_opening_shape::<E, E::EF, Plonky3WhirPcs>(
            &configs.read,
            &closing.spark_read_openings,
        )?;

        E::validate_commitment(&closing.spark_fixed_openings.value_commitment)?;
        if let Some(restorer) = restore_fixed {
            restorer.validate_shape(
                &configs.fixed_value,
                &closing.spark_fixed_openings.value_proof,
                final_rows,
            )?;
        } else {
            E::validate_compressed_plain_proof_shape(
                &configs.fixed_value,
                &closing.spark_fixed_openings.value_proof,
                final_rows,
            )?;
        }
        match (
            &closing.spark_fixed_openings.audit_commitment,
            &closing.spark_fixed_openings.audit_proof,
        ) {
            (Some(commitment), Some(audit_proof)) => {
                E::validate_commitment(commitment)?;
                if let Some(restorer) = restore_fixed {
                    restorer.validate_shape(&configs.fixed_audit, audit_proof, final_rows)?;
                } else {
                    E::validate_compressed_plain_proof_shape(
                        &configs.fixed_audit,
                        audit_proof,
                        final_rows,
                    )?;
                }
            }
            (None, None) => {}
            _ => return Err(SpartanWhirError::InvalidProofShape),
        }
        for (config, group) in configs.read.iter().zip(&closing.spark_read_openings.groups) {
            E::validate_commitment(&group.commitment)?;
            E::validate_compressed_plain_proof_shape(config, &group.proof, final_rows)?;
        }
    }
    Ok(())
}

fn observe_poseidon_zk_context<E>(
    challenger: &mut E::Challenger,
    domain_separator: &DomainSeparator,
    pcs_config: &ZkWhirPcsConfig,
    num_outer_rounds: usize,
    num_inner_rounds: usize,
    public_inputs: &[F],
) where
    E: FullZkPoseidonEngine,
    E::EF: ExtField,
    E::Challenger: FieldChallenger<F> + p3_challenger::GrindingChallenger<Witness = F>,
{
    let _profile = profile_scope("zk_observe_context");
    for byte in domain_separator.to_bytes() {
        challenger.observe(F::from_u8(byte));
    }
    for value in [
        pcs_config.ell_zk,
        pcs_config.mask_log_inv_rate,
        num_outer_rounds,
        num_inner_rounds,
    ] {
        for byte in (value as u64).to_le_bytes() {
            challenger.observe(F::from_u8(byte));
        }
    }
    for &input in public_inputs {
        challenger.observe(input);
    }
}

fn sample_inner_masks<Ext, R>(num_rounds: usize, rng: &mut R) -> Vec<Vec<Ext>>
where
    Ext: Field,
    StandardUniform: Distribution<Ext>,
    R: Rng + ?Sized,
{
    (0..3 * num_rounds)
        .map(|_| {
            let linear = rng.random::<Ext>();
            let quadratic = rng.random::<Ext>();
            vec![Ext::ZERO, linear, quadratic, -linear - quadratic]
        })
        .collect()
}

fn sample_outer_masks<Ext, R>(num_rounds: usize, rng: &mut R) -> Vec<Vec<Ext>>
where
    Ext: Field,
    StandardUniform: Distribution<Ext>,
    R: Rng + ?Sized,
{
    (0..num_rounds)
        .map(|_| (0..8).map(|_| rng.random::<Ext>()).collect())
        .collect()
}

fn power_covector<Ext: Field>(point: Ext, len: usize) -> Vec<Ext> {
    let mut power = Ext::ONE;
    (0..len)
        .map(|_| {
            let current = power;
            power *= point;
            current
        })
        .collect()
}

/// Batches the masked matrix claim, the inner-mask endpoint constraints, and
/// the disclosed outer-mask evaluations into one committed linear relation.
fn application_relation<Ext: Field>(
    num_outer_rounds: usize,
    outer_point: &[Ext],
    masked_claims: (Ext, Ext, Ext),
    outer_mask_evals: &[Ext],
    rho: Ext,
    batching: Ext,
) -> (Vec<Vec<Ext>>, Vec<Vec<Ext>>, Ext) {
    debug_assert_eq!(outer_point.len(), num_outer_rounds);
    debug_assert_eq!(outer_mask_evals.len(), num_outer_rounds);

    let matrix_coefficients = [Ext::ONE, rho, rho.square()];
    let mut inner_covectors = Vec::with_capacity(3 * num_outer_rounds);
    for matrix_coefficient in matrix_coefficients {
        for &point in outer_point {
            let mut covector = power_covector(point, 4);
            for value in &mut covector {
                *value *= matrix_coefficient;
            }
            inner_covectors.push(covector);
        }
    }

    let mut target = masked_claims.0 + rho * masked_claims.1 + rho.square() * masked_claims.2;
    let at_zero = power_covector(Ext::ZERO, 4);
    let at_one = power_covector(Ext::ONE, 4);
    let mut coefficient = batching;
    for covector in &mut inner_covectors {
        for (value, endpoint) in covector.iter_mut().zip(&at_zero) {
            *value += coefficient * *endpoint;
        }
        coefficient *= batching;
        for (value, endpoint) in covector.iter_mut().zip(&at_one) {
            *value += coefficient * *endpoint;
        }
        coefficient *= batching;
    }

    let mut outer_covectors = Vec::with_capacity(num_outer_rounds);
    for (&point, &evaluation) in outer_point.iter().zip(outer_mask_evals) {
        let mut covector = power_covector(point, 8);
        for value in &mut covector {
            *value *= coefficient;
        }
        target += coefficient * evaluation;
        coefficient *= batching;
        outer_covectors.push(covector);
    }

    (inner_covectors, outer_covectors, target)
}

fn scale_covectors<Ext: Field>(covectors: &mut [Vec<Ext>], scale: Ext) {
    for covector in covectors {
        for value in covector {
            *value *= scale;
        }
    }
}

pub(crate) fn combined_application_mask_shape(
    inner: MaskGroupShape,
    outer: MaskGroupShape,
    packing: crate::pcs_config::MaskPacking,
) -> Result<MaskGroupShape, SpartanWhirError> {
    if inner.shape.message_len > outer.shape.message_len
        || inner.shape.randomness_len != outer.shape.randomness_len
    {
        return Err(SpartanWhirError::invalid_config());
    }
    if inner.shape.domain_size != outer.shape.domain_size {
        return Err(SpartanWhirError::invalid_config_reason(
            InvalidConfigReason::IncompatibleApplicationMaskDomains {
                inner_domain_size: inner.shape.domain_size,
                outer_domain_size: outer.shape.domain_size,
            },
        ));
    }
    if packing.application() {
        let inner_len = if packing.free_basis() {
            2
        } else {
            outer.shape.message_len
        };
        let message_len = inner
            .width
            .checked_mul(inner_len)
            .and_then(|n| {
                outer
                    .width
                    .checked_mul(outer.shape.message_len)
                    .and_then(|m| n.checked_add(m))
            })
            .ok_or_else(SpartanWhirError::invalid_config)?;
        let base_len = (outer.shape.message_len + outer.shape.randomness_len).next_power_of_two();
        let log_inv_rate = (outer.shape.domain_size / base_len).ilog2() as usize;
        let domain_size = message_len
            .checked_add(outer.shape.randomness_len)
            .and_then(usize::checked_next_power_of_two)
            .and_then(|n| n.checked_shl(log_inv_rate as u32))
            .ok_or_else(SpartanWhirError::invalid_config)?;
        return Ok(MaskGroupShape {
            shape: MaskCodeShape {
                message_len,
                randomness_len: outer.shape.randomness_len,
                domain_size,
            },
            width: 1,
        });
    }
    Ok(MaskGroupShape {
        shape: outer.shape,
        width: inner
            .width
            .checked_add(outer.width)
            .ok_or_else(SpartanWhirError::invalid_config)?,
    })
}

fn pack_mask_vectors<Ext>(vectors: Vec<Vec<Ext>>, packed: bool) -> Vec<Vec<Ext>> {
    if packed {
        vec![vectors.into_iter().flatten().collect()]
    } else {
        vectors
    }
}

fn combine_application_mask_vectors<Ext: Field>(
    inner: Vec<Vec<Ext>>,
    outer: Vec<Vec<Ext>>,
    message_len: usize,
    packing: crate::pcs_config::MaskPacking,
    covectors: bool,
) -> Result<Vec<Vec<Ext>>, SpartanWhirError> {
    if inner.iter().any(|values| values.len() > message_len)
        || outer.iter().any(|values| values.len() != message_len)
    {
        return Err(SpartanWhirError::InvalidRoundPolynomial);
    }
    if packing.free_basis() {
        if inner.iter().any(|v| v.len() != 4) {
            return Err(SpartanWhirError::InvalidRoundPolynomial);
        }
        let mut combined = Vec::new();
        for values in inner {
            if covectors {
                combined.extend([values[1] - values[3], values[2] - values[3]]);
            } else {
                if values[0] != Ext::ZERO || values[3] != -values[1] - values[2] {
                    return Err(SpartanWhirError::InvalidRoundPolynomial);
                }
                combined.extend([values[1], values[2]]);
            }
        }
        combined.extend(outer.into_iter().flatten());
        return Ok(vec![combined]);
    }
    let mut combined = Vec::with_capacity(inner.len() + outer.len());
    combined.extend(inner.into_iter().map(|mut values| {
        values.resize(message_len, Ext::ZERO);
        values
    }));
    combined.extend(outer);
    Ok(pack_mask_vectors(combined, packing.application()))
}

fn unpacked_inner_product<Ext>(
    evaluations: Vec<F>,
    weights: Vec<Ext>,
) -> Result<ProductPolynomial<F, Ext>, SpartanWhirError>
where
    Ext: ExtField,
{
    if evaluations.len() != weights.len()
        || evaluations.is_empty()
        || !evaluations.len().is_power_of_two()
    {
        return Err(SpartanWhirError::InvalidRoundPolynomial);
    }

    if evaluations.len() > <F as Field>::Packing::WIDTH {
        return Err(SpartanWhirError::InvalidRoundPolynomial);
    }

    Ok(ProductPolynomial::new_unpacked(
        VariableOrder::Prefix,
        Poly::new(evaluations.into_iter().map(Ext::from).collect()),
        Poly::new(weights),
    ))
}

fn observe_spartan_context<E, Ext>(
    challenger: &mut E::Challenger,
    domain_separator: &DomainSeparator,
    public_inputs: &[F],
) -> Result<(), SpartanWhirError>
where
    Ext: ExtField,
    E: SpartanContextEngine<EF = Ext>,
{
    E::observe_spartan_context(challenger, domain_separator, public_inputs)
}

fn sample_algebra_vec<E, Ext>(challenger: &mut E::Challenger, len: usize) -> Vec<Ext>
where
    Ext: ExtField,
    E: SpartanWhirEngine<F = F, EF = Ext>,
    E::Challenger: FieldChallenger<F>,
{
    (0..len)
        .map(|_| challenger.sample_algebra_element::<Ext>())
        .collect()
}

fn matrix_eval_rlc<EF>(matrix_evals: [EF; 3], r: EF) -> EF
where
    EF: Field,
{
    matrix_evals[0] + r * matrix_evals[1] + r * r * matrix_evals[2]
}

const SPARK_OPENING_BRANCH_DOMAIN: &[u8] = b"spartan-whir-spark-opening-branches-v3";
const SPARK_OPENING_BRANCH_END: &[u8] = b"spartan-whir-spark-opening-branch-end-v3";
const SPARK_OPENING_BRANCH_MERGE: &[u8] = b"spartan-whir-spark-opening-merge-v3";
const SPARK_OPENING_BRANCH_SEAL_ELEMENTS: usize = 8;
type SparkOpeningBranchSeal = [F; SPARK_OPENING_BRANCH_SEAL_ELEMENTS];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SparkOpeningBranch {
    Fixed = 0,
    Read = 1,
}

fn observe_spark_opening_tag<C>(challenger: &mut C, tag: &[u8])
where
    C: FieldChallenger<F>,
{
    challenger.observe(F::from_usize(tag.len()));
    for &byte in tag {
        challenger.observe(F::from_u8(byte));
    }
}

fn begin_spark_opening_branches<C>(challenger: &mut C)
where
    C: FieldChallenger<F>,
{
    observe_spark_opening_tag(challenger, SPARK_OPENING_BRANCH_DOMAIN);
    challenger.observe(F::from_usize(2));
}

fn spark_opening_branch<C>(challenger: &C, branch: SparkOpeningBranch) -> C
where
    C: Clone + FieldChallenger<F>,
{
    // Challenger clones must be independent deterministic copies of the
    // transcript state. The supplied Poseidon and Keccak challengers satisfy
    // this contract.
    let mut branch_challenger = challenger.clone();
    branch_challenger.observe(F::from_u8(branch as u8));
    branch_challenger
}

fn seal_spark_opening_branch<C>(challenger: &mut C) -> SparkOpeningBranchSeal
where
    C: FieldChallenger<F>,
{
    observe_spark_opening_tag(challenger, SPARK_OPENING_BRANCH_END);
    core::array::from_fn(|_| challenger.sample())
}

fn merge_spark_opening_branches<C>(
    challenger: &mut C,
    fixed: SparkOpeningBranchSeal,
    read: SparkOpeningBranchSeal,
) where
    C: FieldChallenger<F>,
{
    observe_spark_opening_tag(challenger, SPARK_OPENING_BRANCH_MERGE);
    for (branch, seal) in [
        (SparkOpeningBranch::Fixed, fixed),
        (SparkOpeningBranch::Read, read),
    ] {
        challenger.observe(F::from_u8(branch as u8));
        for value in seal {
            challenger.observe(value);
        }
    }
}

pub(crate) fn setup_spark_fixed_commitments<E, EF, Pcs>(
    configs: &SparkPcsConfigs,
    spark_tables: &crate::SparkTables,
) -> Result<
    Option<(
        SparkFixedProverData<E, Pcs>,
        SparkFixedCommitments<<Pcs as MlePcs<E>>::Commitment>,
    )>,
    SpartanWhirError,
>
where
    EF: ExtField,
    E: SpartanContextEngine<EF = EF>,
    Pcs: ProtocolPcs<E, Config = WhirPcsConfig>,
    <Pcs as MlePcs<E>>::ProverData: Clone + CommittedPolynomialView<EF>,
    <Pcs as MlePcs<E>>::Commitment: Clone + PartialEq,
{
    let mut spark_setup_challenger = E::challenger();
    let (prover_data, commitments) = commit_spark_fixed_tables::<E, EF, Pcs>(
        &configs.fixed_value,
        &configs.fixed_audit,
        spark_tables,
        &mut spark_setup_challenger,
    )?;
    Ok(Some((prover_data, commitments)))
}

pub(crate) fn spark_pcs_configs_for_tables<EF>(
    base: &WhirPcsConfig,
    spark_tables: &crate::SparkTables,
    spark_whir_params: Option<&SparkWhirParams>,
) -> Result<SparkPcsConfigs, SpartanWhirError>
where
    EF: ExtField,
{
    let fixed_value_whir = spark_whir_params
        .map(|params| &params.fixed_value)
        .unwrap_or(&base.whir);
    let fixed_audit_whir = spark_whir_params
        .map(|params| &params.fixed_audit)
        .unwrap_or(&base.whir);
    let read_whir = spark_whir_params
        .map(|params| &params.read)
        .unwrap_or(&base.whir);
    let value_shape_config = spark_table_shape_pcs_config(base, spark_tables.value_domain_size)?;

    Ok(SparkPcsConfigs {
        fixed_value: spark_fixed_value_pcs_config_with_whir(&value_shape_config, fixed_value_whir)?,
        fixed_audit: spark_fixed_audit_pcs_config_with_whir(
            base,
            spark_tables.row_memory_size,
            spark_tables.col_memory_size,
            fixed_audit_whir,
        )?,
        read: spark_read_pcs_configs_with_whir::<EF>(&value_shape_config, read_whir)?,
    })
}

fn spark_table_shape_pcs_config(
    base: &WhirPcsConfig,
    domain_size: usize,
) -> Result<WhirPcsConfig, SpartanWhirError> {
    if domain_size == 0 || !domain_size.is_power_of_two() {
        return Err(SpartanWhirError::InvalidPolynomialLength);
    }
    let mut config = base.clone();
    config.num_variables = domain_size.ilog2() as usize;
    Ok(config)
}

fn spark_table_pcs_config_with_whir(
    base: &WhirPcsConfig,
    domain_size: usize,
    whir_params: &WhirParams,
) -> Result<WhirPcsConfig, SpartanWhirError> {
    let mut config = spark_table_shape_pcs_config(base, domain_size)?;
    config.whir = whir_params.clone();
    config.validate()?;
    Ok(config)
}

fn spark_read_pcs_configs_with_whir<EF>(
    value_config: &WhirPcsConfig,
    whir_params: &WhirParams,
) -> Result<Vec<WhirPcsConfig>, SpartanWhirError>
where
    EF: ExtField,
{
    read_coordinate_groups::<EF>()?
        .into_iter()
        .map(|group| {
            let mut config = value_config.clone();
            config.num_variables = config
                .num_variables
                .checked_add(group.column_count.ilog2() as usize)
                .ok_or(SpartanWhirError::invalid_config())?;
            config.whir = whir_params.clone();
            config.validate()?;
            Ok(config)
        })
        .collect()
}

fn spark_fixed_value_pcs_config_with_whir(
    value_config: &WhirPcsConfig,
    whir_params: &WhirParams,
) -> Result<WhirPcsConfig, SpartanWhirError> {
    let mut config = value_config.clone();
    config.num_variables = config
        .num_variables
        .checked_add(fixed_value_column_bits())
        .ok_or(SpartanWhirError::invalid_config())?;
    config.whir = whir_params.clone();
    config.validate()?;
    Ok(config)
}

fn spark_fixed_audit_pcs_config_with_whir(
    base: &WhirPcsConfig,
    row_memory_size: usize,
    col_memory_size: usize,
    whir_params: &WhirParams,
) -> Result<WhirPcsConfig, SpartanWhirError> {
    let audit_memory_size = row_memory_size
        .max(col_memory_size)
        .checked_next_power_of_two()
        .ok_or(SpartanWhirError::invalid_config())?;
    let audit_domain_size = audit_memory_size
        .checked_mul(fixed_audit_column_count())
        .ok_or(SpartanWhirError::invalid_config())?;
    spark_table_pcs_config_with_whir(base, audit_domain_size, whir_params)
}

fn commit_spark_fixed_tables<E, EF, Pcs>(
    fixed_value_config: &WhirPcsConfig,
    audit_config: &WhirPcsConfig,
    tables: &crate::SparkTables,
    challenger: &mut E::Challenger,
) -> Result<
    (
        SparkFixedProverData<E, Pcs>,
        SparkFixedCommitments<<Pcs as MlePcs<E>>::Commitment>,
    ),
    SpartanWhirError,
>
where
    EF: ExtField,
    E: SpartanContextEngine<EF = EF>,
    Pcs: ProtocolPcs<E, Config = WhirPcsConfig>,
    <Pcs as MlePcs<E>>::ProverData: Clone + CommittedPolynomialView<EF>,
    <Pcs as MlePcs<E>>::Commitment: Clone + PartialEq,
{
    let value_bundle = fixed_value_bundle(tables, fixed_value_config)?;
    let (value_commitment, value) =
        <Pcs as MlePcs<E>>::commit(fixed_value_config, &value_bundle, challenger)?;
    let (audit_commitment, audit) = if spark_fixed_audit_is_embedded(
        tables.value_domain_size,
        tables.row_memory_size,
        tables.col_memory_size,
    ) {
        (None, None)
    } else {
        let audit_bundle = fixed_audit_bundle(tables, audit_config)?;
        let (commitment, prover_data) =
            <Pcs as MlePcs<E>>::commit(audit_config, &audit_bundle, challenger)?;
        (Some(commitment), Some(prover_data))
    };

    Ok((
        SparkFixedProverData {
            value,
            audit,
            marker: PhantomData,
        },
        SparkFixedCommitments {
            value: value_commitment,
            audit: audit_commitment,
        },
    ))
}

fn fixed_value_bundle(
    tables: &crate::SparkTables,
    config: &WhirPcsConfig,
) -> Result<Vec<F>, SpartanWhirError> {
    let domain_size = tables.value_domain_size;
    if domain_size == 0
        || !domain_size.is_power_of_two()
        || config.num_variables != domain_size.ilog2() as usize + fixed_value_column_bits()
    {
        return Err(SpartanWhirError::invalid_config());
    }
    let mut packed = vec![F::ZERO; domain_size * fixed_value_column_count()];
    copy_rectangular_base_column(&mut packed, domain_size, 0, &tables.rows)?;
    copy_rectangular_base_column(&mut packed, domain_size, 1, &tables.cols)?;
    copy_rectangular_base_column(&mut packed, domain_size, 2, &tables.val_a)?;
    copy_rectangular_base_column(&mut packed, domain_size, 3, &tables.val_b)?;
    copy_rectangular_base_column(&mut packed, domain_size, 4, &tables.val_c)?;
    copy_rectangular_base_column(&mut packed, domain_size, 5, &tables.read_ts_row)?;
    copy_rectangular_base_column(&mut packed, domain_size, 6, &tables.read_ts_col)?;
    if spark_fixed_audit_is_embedded(
        tables.value_domain_size,
        tables.row_memory_size,
        tables.col_memory_size,
    ) {
        copy_embedded_audit_tables(&mut packed, tables)?;
    }
    Ok(packed)
}

fn copy_embedded_audit_tables(
    packed: &mut [F],
    tables: &crate::SparkTables,
) -> Result<(), SpartanWhirError> {
    let domain_size = tables.value_domain_size;
    let memory_domain_size = tables.row_memory_size.max(tables.col_memory_size);
    if !spark_fixed_audit_is_embedded(domain_size, tables.row_memory_size, tables.col_memory_size) {
        return Err(SpartanWhirError::invalid_config());
    }
    let column_start = 7usize
        .checked_mul(domain_size)
        .ok_or_else(SpartanWhirError::invalid_config)?;
    let row_end = column_start
        .checked_add(tables.audit_ts_row.len())
        .ok_or_else(SpartanWhirError::invalid_config)?;
    let col_start = column_start
        .checked_add(memory_domain_size)
        .ok_or_else(SpartanWhirError::invalid_config)?;
    let col_end = col_start
        .checked_add(tables.audit_ts_col.len())
        .ok_or_else(SpartanWhirError::invalid_config)?;
    if col_end > packed.len() || row_end > col_start {
        return Err(SpartanWhirError::invalid_config());
    }
    packed[column_start..row_end].copy_from_slice(&tables.audit_ts_row);
    packed[col_start..col_end].copy_from_slice(&tables.audit_ts_col);
    Ok(())
}

fn fixed_audit_bundle(
    tables: &crate::SparkTables,
    config: &WhirPcsConfig,
) -> Result<Vec<F>, SpartanWhirError> {
    if config.num_variables < fixed_audit_column_bits() {
        return Err(SpartanWhirError::invalid_config());
    }
    let audit_memory_bits = config.num_variables - fixed_audit_column_bits();
    let audit_memory_size = 1usize
        .checked_shl(audit_memory_bits as u32)
        .ok_or(SpartanWhirError::invalid_config())?;
    if tables.row_memory_size > audit_memory_size || tables.col_memory_size > audit_memory_size {
        return Err(SpartanWhirError::invalid_config());
    }
    let mut packed = vec![F::ZERO; audit_memory_size * fixed_audit_column_count()];
    copy_rectangular_base_column(&mut packed, audit_memory_size, 0, &tables.audit_ts_row)?;
    copy_rectangular_base_column(&mut packed, audit_memory_size, 1, &tables.audit_ts_col)?;
    Ok(packed)
}

fn copy_rectangular_base_column(
    packed: &mut [F],
    domain_size: usize,
    column: usize,
    values: &[F],
) -> Result<(), SpartanWhirError> {
    let start = column
        .checked_mul(domain_size)
        .ok_or(SpartanWhirError::invalid_config())?;
    let end = start
        .checked_add(values.len())
        .ok_or(SpartanWhirError::invalid_config())?;
    if domain_size == 0 || values.len() > domain_size || end > packed.len() {
        return Err(SpartanWhirError::invalid_config());
    }
    packed[start..end].copy_from_slice(values);
    Ok(())
}

fn commit_spark_read_tables<E, EF, Pcs>(
    configs: &[WhirPcsConfig],
    read_tables: &SparkReadTables<EF>,
    challenger: &mut E::Challenger,
) -> Result<
    (
        SparkReadProverData<E, Pcs>,
        SparkReadCommitments<<Pcs as MlePcs<E>>::Commitment>,
    ),
    SpartanWhirError,
>
where
    EF: ExtField,
    E: SpartanContextEngine<EF = EF>,
    Pcs: SparkReadPcs<E>,
    <Pcs as MlePcs<E>>::Commitment: Clone + PartialEq,
{
    let coordinate_columns = {
        let _profile = profile_scope("spark_read_to_base_columns");
        extension_read_tables_to_base_columns(&read_tables.erow, &read_tables.ecol)?
    };
    let domain_size = read_tables.erow.len();
    let groups = read_coordinate_groups::<EF>()?;
    if domain_size == 0
        || !domain_size.is_power_of_two()
        || read_tables.ecol.len() != domain_size
        || configs.len() != groups.len()
    {
        return Err(SpartanWhirError::invalid_config());
    }
    let mut commitments = Vec::with_capacity(groups.len());
    let mut prover_data = Vec::with_capacity(groups.len());
    for (config, group) in configs.iter().zip(groups) {
        let start = group
            .column_start
            .checked_mul(domain_size)
            .ok_or_else(SpartanWhirError::invalid_config)?;
        let end = group
            .column_start
            .checked_add(group.column_count)
            .and_then(|column| column.checked_mul(domain_size))
            .ok_or_else(SpartanWhirError::invalid_config)?;
        let columns = coordinate_columns
            .get(start..end)
            .ok_or_else(SpartanWhirError::invalid_config)?
            .to_vec();
        let (commitment, data) =
            Pcs::commit_read_table(config, columns, domain_size, group.column_count, challenger)?;
        commitments.push(commitment);
        prover_data.push(data);
    }

    Ok((
        SparkReadProverData {
            groups: prover_data,
            marker: PhantomData,
        },
        SparkReadCommitments {
            groups: commitments,
        },
    ))
}

fn prepare_spark_fixed_openings<E, EF, Pcs>(
    fixed_value_config: &WhirPcsConfig,
    audit_config: &WhirPcsConfig,
    prover_data: SparkFixedProverData<E, Pcs>,
    challenger: &mut E::Challenger,
) -> Result<SparkFixedProverData<E, Pcs>, SpartanWhirError>
where
    EF: ExtField,
    E: SpartanContextEngine<EF = EF>,
    Pcs: ProtocolPcs<E, Config = WhirPcsConfig>,
    <Pcs as MlePcs<E>>::ProverData: Clone + CommittedPolynomialView<EF>,
{
    Ok(SparkFixedProverData {
        value: <Pcs as ProtocolPcs<E>>::prepare_committed_opening(
            fixed_value_config,
            prover_data.value,
            challenger,
        )?,
        audit: prover_data
            .audit
            .map(|audit| {
                <Pcs as ProtocolPcs<E>>::prepare_committed_opening(audit_config, audit, challenger)
            })
            .transpose()?,
        marker: PhantomData,
    })
}

fn open_spark_fixed_tables<E, EF, Pcs>(
    fixed_value_config: &WhirPcsConfig,
    audit_config: &WhirPcsConfig,
    prover_data: SparkFixedProverData<E, Pcs>,
    commitments: SparkFixedCommitments<<Pcs as MlePcs<E>>::Commitment>,
    tables: &crate::SparkTables,
    product_claims: &SparkBatchedMemoryProductsLeafClaims<EF>,
    final_rows: bool,
    challenger: &mut E::Challenger,
) -> Result<SparkFixedOpeningProof<E, Pcs>, SpartanWhirError>
where
    EF: ExtField,
    E: SpartanContextEngine<EF = EF>,
    Pcs: ProtocolPcs<E, Config = WhirPcsConfig>,
    <Pcs as MlePcs<E>>::ProverData: Clone + CommittedPolynomialView<EF>,
    <Pcs as MlePcs<E>>::Commitment: Clone + PartialEq,
{
    let (mut value_claims, value_evals) = fixed_value_opening_claims_from_prover_data(
        fixed_value_config,
        &prover_data.value,
        product_claims,
    )?;
    let expected_weights = [
        value_evals.val_a_low,
        value_evals.val_a_high,
        value_evals.val_b_low,
        value_evals.val_b_high,
        value_evals.val_c_low,
        value_evals.val_c_high,
    ];
    if product_claims.ops.dotproduct_weight_evals.as_slice() != expected_weights {
        return Err(SpartanWhirError::SumcheckFailed);
    }
    let audit_embedded = spark_fixed_audit_is_embedded(
        tables.value_domain_size,
        tables.row_memory_size,
        tables.col_memory_size,
    );
    let embedded_audit_evals = if audit_embedded {
        let (claims, evals) = embedded_fixed_audit_opening_claims_from_prover_data(
            fixed_value_config,
            &prover_data.value,
            tables.row_memory_size,
            tables.col_memory_size,
            product_claims,
        )?;
        value_claims.extend(claims);
        Some(evals)
    } else {
        None
    };
    let value_statement = point_eval_statement::<E, EF>(&value_claims)?;
    let value_proof = <Pcs as ProtocolPcs<E>>::open_compressed(
        fixed_value_config,
        prover_data.value,
        &value_statement,
        final_rows,
        challenger,
    )?;

    let (audit_evals, audit_proof) = if audit_embedded {
        (
            embedded_audit_evals.ok_or_else(SpartanWhirError::invalid_config)?,
            None,
        )
    } else {
        let audit_data = prover_data
            .audit
            .ok_or_else(SpartanWhirError::invalid_config)?;
        let (audit_claims, audit_evals) = fixed_audit_opening_claims_from_prover_data(
            audit_config,
            &audit_data,
            tables.row_memory_size,
            tables.col_memory_size,
            product_claims,
        )?;
        let audit_statement = point_eval_statement::<E, EF>(&audit_claims)?;
        let audit_proof = <Pcs as ProtocolPcs<E>>::open_compressed(
            audit_config,
            audit_data,
            &audit_statement,
            final_rows,
            challenger,
        )?;
        (audit_evals, Some(audit_proof))
    };

    Ok(SparkFixedOpeningProof {
        value_num_variables: fixed_value_config.num_variables,
        value_column_bits: fixed_value_column_bits(),
        audit_num_variables: audit_config.num_variables,
        audit_column_bits: fixed_audit_column_bits(),
        value_commitment: commitments.value,
        audit_commitment: commitments.audit,
        evals: SparkFixedTableOpeningEvals {
            val_a_low: value_evals.val_a_low,
            val_a_high: value_evals.val_a_high,
            val_b_low: value_evals.val_b_low,
            val_b_high: value_evals.val_b_high,
            val_c_low: value_evals.val_c_low,
            val_c_high: value_evals.val_c_high,
            row_addr: value_evals.row_addr,
            col_addr: value_evals.col_addr,
            row_read_ts: value_evals.row_read_ts,
            col_read_ts: value_evals.col_read_ts,
            row_audit_ts: audit_evals.row_audit_ts,
            col_audit_ts: audit_evals.col_audit_ts,
        },
        value_proof,
        audit_proof,
        marker: PhantomData,
    })
}

fn open_spark_read_tables<E, EF, Pcs>(
    configs: &[WhirPcsConfig],
    prover_data: SparkReadProverData<E, Pcs>,
    commitments: SparkReadCommitments<<Pcs as MlePcs<E>>::Commitment>,
    product_claims: &SparkBatchedMemoryProductsLeafClaims<EF>,
    final_rows: bool,
    challenger: &mut E::Challenger,
) -> Result<SparkReadOpeningProof<E, Pcs>, SpartanWhirError>
where
    EF: ExtField,
    E: SpartanContextEngine<EF = EF>,
    Pcs: SparkReadPcs<E>,
    <Pcs as MlePcs<E>>::Commitment: Clone + PartialEq,
{
    let value_num_variables = product_claims.ops.product_point.0.len();
    let groups = read_coordinate_groups::<EF>()?;
    if configs.len() != groups.len()
        || prover_data.groups.len() != groups.len()
        || commitments.groups.len() != groups.len()
        || configs.iter().zip(&groups).any(|(config, group)| {
            Some(config.num_variables)
                != value_num_variables.checked_add(group.column_count.ilog2() as usize)
        })
    {
        return Err(SpartanWhirError::invalid_config());
    }
    let points = read_opening_points(product_claims)?;
    let points = [
        points.erow_low,
        points.erow_high,
        points.erow_ops,
        points.ecol_low,
        points.ecol_high,
        points.ecol_ops,
    ];
    let mut opening_groups = Vec::with_capacity(groups.len());
    for (((config, group), data), commitment) in configs
        .iter()
        .zip(groups)
        .zip(prover_data.groups)
        .zip(commitments.groups)
    {
        let requests = spark_read_group_opening_requests::<EF>(group)?;
        let opening_columns = requests
            .iter()
            .map(|request| request.columns.clone())
            .collect::<Vec<_>>();
        let opening_points = requests
            .iter()
            .map(|request| points[request.logical_opening].clone())
            .collect::<Vec<_>>();
        let (proof, evals) = Pcs::open_read_table_compressed(
            config,
            data,
            group.column_count,
            &opening_columns,
            &opening_points,
            final_rows,
            challenger,
        )?;
        opening_groups.push(SparkReadGroupOpeningProof {
            num_variables: config.num_variables,
            column_start: group.column_start,
            column_count: group.column_count,
            commitment,
            evals,
            proof,
            marker: PhantomData,
        });
    }
    let proof = SparkReadOpeningProof {
        groups: opening_groups,
    };
    let read_evals = spark_read_table_opening_evals::<E, EF, Pcs>(&proof)?;
    validate_spark_read_opening_product_claims(product_claims, &read_evals)?;
    Ok(proof)
}

fn validate_spark_read_opening_product_claims<EF>(
    product_claims: &SparkBatchedMemoryProductsLeafClaims<EF>,
    read_evals: &SparkReadTableOpeningEvals<EF>,
) -> Result<(), SpartanWhirError>
where
    EF: ExtField,
{
    let expected_left = [
        read_evals.erow_low,
        read_evals.erow_high,
        read_evals.erow_low,
        read_evals.erow_high,
        read_evals.erow_low,
        read_evals.erow_high,
    ];
    let expected_right = [
        read_evals.ecol_low,
        read_evals.ecol_high,
        read_evals.ecol_low,
        read_evals.ecol_high,
        read_evals.ecol_low,
        read_evals.ecol_high,
    ];
    if product_claims.ops.dotproduct_left_evals.as_slice() != expected_left
        || product_claims.ops.dotproduct_right_evals.as_slice() != expected_right
        || product_claims.ops.product_evals.len() != 4
    {
        return Err(SpartanWhirError::SumcheckFailed);
    }
    Ok(())
}

fn spark_read_table_opening_evals<E, EF, Pcs>(
    proof: &SparkReadOpeningProof<E, Pcs>,
) -> Result<SparkReadTableOpeningEvals<EF>, SpartanWhirError>
where
    EF: ExtField,
    E: SpartanContextEngine<EF = EF>,
    Pcs: MlePcs<E, Config = WhirPcsConfig>,
{
    let [erow_low, erow_high, erow_ops, ecol_low, ecol_high, ecol_ops] =
        grouped_read_coordinate_evals::<E, EF, Pcs>(proof)?;
    split_read_coordinate_evals(
        &[erow_low, erow_high, erow_ops],
        &[ecol_low, ecol_high, ecol_ops],
    )
}

fn parse_spark_fixed_openings<E, EF, Pcs>(
    fixed_value_config: &WhirPcsConfig,
    audit_config: &WhirPcsConfig,
    audit_embedded: bool,
    proof: &SparkFixedOpeningProof<E, Pcs>,
    challenger: &mut E::Challenger,
) -> Result<ParsedSparkFixedOpenings<E, Pcs>, SpartanWhirError>
where
    EF: ExtField,
    E: SpartanContextEngine<EF = EF>,
    Pcs: ProtocolPcs<E, Config = WhirPcsConfig>,
    <Pcs as MlePcs<E>>::Commitment: Clone + PartialEq,
{
    validate_spark_fixed_opening_shape::<E, EF, Pcs>(
        fixed_value_config,
        audit_config,
        audit_embedded,
        proof,
    )?;
    let value = <Pcs as ProtocolPcs<E>>::verify_parse_commitment(
        fixed_value_config,
        &proof.value_commitment,
        &proof.value_proof,
        challenger,
    )?;
    let audit = if audit_embedded {
        None
    } else {
        Some(<Pcs as ProtocolPcs<E>>::verify_parse_commitment(
            audit_config,
            proof
                .audit_commitment
                .as_ref()
                .ok_or_else(SpartanWhirError::invalid_config)?,
            proof
                .audit_proof
                .as_ref()
                .ok_or_else(SpartanWhirError::invalid_config)?,
            challenger,
        )?)
    };
    Ok(ParsedSparkFixedOpenings { value, audit })
}

fn parse_spark_read_openings<E, EF, Pcs>(
    configs: &[WhirPcsConfig],
    proof: &SparkReadOpeningProof<E, Pcs>,
    challenger: &mut E::Challenger,
) -> Result<ParsedSparkReadOpenings<E, Pcs>, SpartanWhirError>
where
    EF: ExtField,
    E: SpartanContextEngine<EF = EF>,
    Pcs: SparkReadPcs<E>,
    <Pcs as MlePcs<E>>::Commitment: Clone + PartialEq,
{
    validate_spark_read_opening_shape::<E, EF, Pcs>(configs, proof)?;
    let groups = configs
        .iter()
        .zip(&proof.groups)
        .map(|(config, group)| {
            Pcs::verify_parse_read_commitment(config, &group.commitment, &group.proof, challenger)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ParsedSparkReadOpenings { groups })
}

fn finalize_spark_fixed_openings<E, EF, Pcs>(
    fixed_value_config: &WhirPcsConfig,
    audit_config: &WhirPcsConfig,
    row_memory_size: usize,
    col_memory_size: usize,
    proof: &SparkFixedOpeningProof<E, Pcs>,
    parsed: ParsedSparkFixedOpenings<E, Pcs>,
    product_claims: &SparkBatchedMemoryProductsLeafClaims<EF>,
    final_rows: bool,
    restore_fixed: Option<&dyn crate::fixed_oracle_cache::FixedOpeningRestorer<E, Pcs>>,
    challenger: &mut E::Challenger,
) -> Result<(), SpartanWhirError>
where
    EF: ExtField,
    E: SpartanContextEngine<EF = EF>,
    Pcs: ProtocolPcs<E, Config = WhirPcsConfig>,
    <Pcs as MlePcs<E>>::Commitment: Clone + PartialEq,
{
    let audit_embedded = spark_fixed_audit_is_embedded(
        1usize
            .checked_shl(
                fixed_value_config
                    .num_variables
                    .checked_sub(fixed_value_column_bits())
                    .ok_or_else(SpartanWhirError::invalid_config)? as u32,
            )
            .ok_or_else(SpartanWhirError::invalid_config)?,
        row_memory_size,
        col_memory_size,
    );
    validate_spark_fixed_opening_shape::<E, EF, Pcs>(
        fixed_value_config,
        audit_config,
        audit_embedded,
        proof,
    )?;
    let mut value_claims = fixed_value_opening_claims_from_evals(product_claims, &proof.evals)?;
    if audit_embedded {
        value_claims.extend(embedded_fixed_audit_opening_claims_from_evals(
            fixed_value_config,
            row_memory_size,
            col_memory_size,
            product_claims,
            &proof.evals,
        )?);
    }
    let value_statement = point_eval_statement::<E, EF>(&value_claims)?;
    let restored_value;
    let value_proof = if let Some(restorer) = restore_fixed {
        restored_value = restorer.restore(
            crate::fixed_oracle_cache::FixedOracleKind::Value,
            fixed_value_config,
            &proof.value_proof,
            &value_statement,
            challenger,
            final_rows,
        )?;
        &restored_value
    } else {
        &proof.value_proof
    };
    <Pcs as ProtocolPcs<E>>::verify_finalize_compressed(
        fixed_value_config,
        &parsed.value,
        &value_statement,
        value_proof,
        final_rows,
        challenger,
    )?;

    if audit_embedded {
        return Ok(());
    }

    let audit_claims = fixed_audit_opening_claims_from_evals(
        audit_config,
        row_memory_size,
        col_memory_size,
        product_claims,
        &proof.evals,
    )?;
    let audit_statement = point_eval_statement::<E, EF>(&audit_claims)?;
    let audit_proof = proof
        .audit_proof
        .as_ref()
        .ok_or_else(SpartanWhirError::invalid_config)?;
    let restored_audit;
    let audit_proof = if let Some(restorer) = restore_fixed {
        restored_audit = restorer.restore(
            crate::fixed_oracle_cache::FixedOracleKind::Audit,
            audit_config,
            audit_proof,
            &audit_statement,
            challenger,
            final_rows,
        )?;
        &restored_audit
    } else {
        audit_proof
    };
    <Pcs as ProtocolPcs<E>>::verify_finalize_compressed(
        audit_config,
        parsed
            .audit
            .as_ref()
            .ok_or_else(SpartanWhirError::invalid_config)?,
        &audit_statement,
        audit_proof,
        final_rows,
        challenger,
    )
}

fn finalize_spark_read_openings<E, EF, Pcs>(
    configs: &[WhirPcsConfig],
    proof: &SparkReadOpeningProof<E, Pcs>,
    parsed: ParsedSparkReadOpenings<E, Pcs>,
    product_claims: &SparkBatchedMemoryProductsLeafClaims<EF>,
    final_rows: bool,
    challenger: &mut E::Challenger,
) -> Result<SparkReadTableOpeningEvals<EF>, SpartanWhirError>
where
    EF: ExtField,
    E: SpartanContextEngine<EF = EF>,
    Pcs: SparkReadPcs<E>,
    <Pcs as MlePcs<E>>::Commitment: Clone + PartialEq,
{
    validate_spark_read_opening_shape::<E, EF, Pcs>(configs, proof)?;
    if parsed.groups.len() != proof.groups.len() {
        return Err(SpartanWhirError::invalid_config());
    }
    let value_num_variables = product_claims.ops.product_point.0.len();
    let groups = read_coordinate_groups::<EF>()?;
    if configs.iter().zip(&groups).any(|(config, group)| {
        Some(config.num_variables)
            != value_num_variables.checked_add(group.column_count.ilog2() as usize)
    }) {
        return Err(SpartanWhirError::InvalidNumVariables);
    }
    let points = read_opening_points(product_claims)?;
    let points = [
        points.erow_low,
        points.erow_high,
        points.erow_ops,
        points.ecol_low,
        points.ecol_high,
        points.ecol_ops,
    ];
    for (((config, group), opening), parsed_group) in configs
        .iter()
        .zip(groups)
        .zip(&proof.groups)
        .zip(&parsed.groups)
    {
        let requests = spark_read_group_opening_requests::<EF>(group)?;
        let opening_columns = requests
            .iter()
            .map(|request| request.columns.clone())
            .collect::<Vec<_>>();
        let opening_points = requests
            .iter()
            .map(|request| points[request.logical_opening].clone())
            .collect::<Vec<_>>();
        Pcs::verify_finalize_read_table_compressed(
            config,
            parsed_group,
            &opening.proof,
            group.column_count,
            &opening_columns,
            &opening_points,
            &opening.evals,
            final_rows,
            challenger,
        )?;
    }
    let [erow_low, erow_high, erow_ops, ecol_low, ecol_high, ecol_ops] =
        grouped_read_coordinate_evals::<E, EF, Pcs>(proof)?;
    split_read_coordinate_evals(
        &[erow_low, erow_high, erow_ops],
        &[ecol_low, ecol_high, ecol_ops],
    )
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn open_spark_table_openings<E, EF, Pcs>(
    configs: &SparkPcsConfigs,
    fixed_prover_data: SparkFixedProverData<E, Pcs>,
    fixed_commitments: SparkFixedCommitments<<Pcs as MlePcs<E>>::Commitment>,
    read_prover_data: SparkReadProverData<E, Pcs>,
    read_commitments: SparkReadCommitments<<Pcs as MlePcs<E>>::Commitment>,
    tables: &crate::SparkTables,
    product_claims: &SparkBatchedMemoryProductsLeafClaims<EF>,
    final_rows: bool,
    challenger: &mut E::Challenger,
) -> Result<
    (
        SparkFixedOpeningProof<E, Pcs>,
        SparkReadOpeningProof<E, Pcs>,
    ),
    SpartanWhirError,
>
where
    EF: ExtField,
    E: SpartanContextEngine<EF = EF> + Send,
    Pcs: SparkReadPcs<E>,
    <Pcs as MlePcs<E>>::ProverData: Clone + CommittedPolynomialView<EF> + Send,
    <Pcs as MlePcs<E>>::Commitment: Clone + PartialEq + Send,
    <Pcs as MlePcs<E>>::Proof: Send,
    <Pcs as SparkReadPcs<E>>::ReadProverData: Send,
    E::Challenger: Clone + Send + FieldChallenger<F>,
{
    begin_spark_opening_branches(challenger);
    let mut fixed_challenger = spark_opening_branch(challenger, SparkOpeningBranch::Fixed);
    let mut read_challenger = spark_opening_branch(challenger, SparkOpeningBranch::Read);
    let fixed_value_config = configs.fixed_value.clone();
    let fixed_audit_config = configs.fixed_audit.clone();
    let read_configs = configs.read.clone();
    let (fixed_result, read_result) = join(
        || {
            let _profile = profile_scope("spark_open_fixed_tables");
            let proof = open_spark_fixed_tables::<E, EF, Pcs>(
                &fixed_value_config,
                &fixed_audit_config,
                fixed_prover_data,
                fixed_commitments,
                tables,
                product_claims,
                final_rows,
                &mut fixed_challenger,
            )?;
            let seal = seal_spark_opening_branch(&mut fixed_challenger);
            Ok::<_, SpartanWhirError>((proof, seal))
        },
        || {
            let _profile = profile_scope("spark_open_read_tables");
            let proof = open_spark_read_tables::<E, EF, Pcs>(
                &read_configs,
                read_prover_data,
                read_commitments,
                product_claims,
                final_rows,
                &mut read_challenger,
            )?;
            let seal = seal_spark_opening_branch(&mut read_challenger);
            Ok::<_, SpartanWhirError>((proof, seal))
        },
    );
    let (fixed_proof, fixed_seal) = fixed_result?;
    let (read_proof, read_seal) = read_result?;
    merge_spark_opening_branches(challenger, fixed_seal, read_seal);
    Ok((fixed_proof, read_proof))
}
fn parse_spark_table_openings<E, EF, Pcs>(
    configs: &SparkPcsConfigs,
    audit_embedded: bool,
    fixed_proof: &SparkFixedOpeningProof<E, Pcs>,
    read_proof: &SparkReadOpeningProof<E, Pcs>,
    challenger: &mut E::Challenger,
) -> Result<
    (
        ParsedSparkFixedOpenings<E, Pcs>,
        ParsedSparkReadOpenings<E, Pcs>,
    ),
    SpartanWhirError,
>
where
    EF: ExtField,
    E: SpartanContextEngine<EF = EF>,
    Pcs: SparkReadPcs<E>,
    <Pcs as MlePcs<E>>::Commitment: Clone + PartialEq,
{
    Ok((
        parse_spark_fixed_openings::<E, EF, Pcs>(
            &configs.fixed_value,
            &configs.fixed_audit,
            audit_embedded,
            fixed_proof,
            challenger,
        )?,
        parse_spark_read_openings::<E, EF, Pcs>(&configs.read, read_proof, challenger)?,
    ))
}
#[allow(clippy::too_many_arguments)]
fn finalize_spark_table_openings<E, EF, Pcs>(
    configs: &SparkPcsConfigs,
    row_memory_size: usize,
    col_memory_size: usize,
    fixed_proof: &SparkFixedOpeningProof<E, Pcs>,
    read_proof: &SparkReadOpeningProof<E, Pcs>,
    parsed_fixed: ParsedSparkFixedOpenings<E, Pcs>,
    parsed_read: ParsedSparkReadOpenings<E, Pcs>,
    product_claims: &SparkBatchedMemoryProductsLeafClaims<EF>,
    final_rows: bool,
    restore_fixed: Option<&dyn crate::fixed_oracle_cache::FixedOpeningRestorer<E, Pcs>>,
    challenger: &mut E::Challenger,
) -> Result<SparkReadTableOpeningEvals<EF>, SpartanWhirError>
where
    EF: ExtField,
    E: SpartanContextEngine<EF = EF> + Sync,
    Pcs: SparkReadPcs<E>,
    <Pcs as MlePcs<E>>::Commitment: Clone + PartialEq + Sync,
    <Pcs as MlePcs<E>>::Proof: Sync,
    <Pcs as ProtocolPcs<E>>::ParsedCommitment: Send,
    <Pcs as SparkReadPcs<E>>::ParsedReadCommitment: Send,
    E::Challenger: Clone + Send + FieldChallenger<F>,
{
    begin_spark_opening_branches(challenger);
    let mut fixed_challenger = spark_opening_branch(challenger, SparkOpeningBranch::Fixed);
    let mut read_challenger = spark_opening_branch(challenger, SparkOpeningBranch::Read);
    let (fixed_result, read_result) = join(
        || {
            finalize_spark_fixed_openings::<E, EF, Pcs>(
                &configs.fixed_value,
                &configs.fixed_audit,
                row_memory_size,
                col_memory_size,
                fixed_proof,
                parsed_fixed,
                product_claims,
                final_rows,
                restore_fixed,
                &mut fixed_challenger,
            )?;
            let seal = seal_spark_opening_branch(&mut fixed_challenger);
            Ok::<_, SpartanWhirError>(seal)
        },
        || {
            let evals = finalize_spark_read_openings::<E, EF, Pcs>(
                &configs.read,
                read_proof,
                parsed_read,
                product_claims,
                final_rows,
                &mut read_challenger,
            )?;
            let seal = seal_spark_opening_branch(&mut read_challenger);
            Ok::<_, SpartanWhirError>((evals, seal))
        },
    );
    let fixed_seal = fixed_result?;
    let (read_evals, read_seal) = read_result?;
    merge_spark_opening_branches(challenger, fixed_seal, read_seal);
    Ok(read_evals)
}
fn fixed_value_opening_claims_from_prover_data<EF, D>(
    config: &WhirPcsConfig,
    prover_data: &D,
    product_claims: &SparkBatchedMemoryProductsLeafClaims<EF>,
) -> Result<
    (
        Vec<(MultilinearPoint<EF>, EF)>,
        SparkFixedValueOpeningEvals<EF>,
    ),
    SpartanWhirError,
>
where
    EF: ExtField,
    D: CommittedPolynomialView<EF>,
{
    let _profile = profile_scope("spark_fixed_opening_claims");
    if prover_data.num_variables() != config.num_variables {
        return Err(SpartanWhirError::invalid_config());
    }
    let value_num_variables = config
        .num_variables
        .checked_sub(fixed_value_column_bits())
        .ok_or(SpartanWhirError::invalid_config())?;
    if product_claims.ops.product_point.0.len() != value_num_variables {
        return Err(SpartanWhirError::InvalidNumVariables);
    }
    let polynomial = prover_data.polynomial();
    let (dot_low_point, dot_high_point) = dotproduct_full_domain_points(&product_claims.ops)?;
    let mut claims = Vec::with_capacity(10);
    let row_addr = push_rectangular_opening_claim(
        polynomial,
        fixed_value_column_bits(),
        0,
        &product_claims.ops.product_point,
        &mut claims,
    )?;
    let col_addr = push_rectangular_opening_claim(
        polynomial,
        fixed_value_column_bits(),
        1,
        &product_claims.ops.product_point,
        &mut claims,
    )?;
    let val_a_low = push_rectangular_opening_claim(
        polynomial,
        fixed_value_column_bits(),
        2,
        &dot_low_point,
        &mut claims,
    )?;
    let val_a_high = push_rectangular_opening_claim(
        polynomial,
        fixed_value_column_bits(),
        2,
        &dot_high_point,
        &mut claims,
    )?;
    let val_b_low = push_rectangular_opening_claim(
        polynomial,
        fixed_value_column_bits(),
        3,
        &dot_low_point,
        &mut claims,
    )?;
    let val_b_high = push_rectangular_opening_claim(
        polynomial,
        fixed_value_column_bits(),
        3,
        &dot_high_point,
        &mut claims,
    )?;
    let val_c_low = push_rectangular_opening_claim(
        polynomial,
        fixed_value_column_bits(),
        4,
        &dot_low_point,
        &mut claims,
    )?;
    let val_c_high = push_rectangular_opening_claim(
        polynomial,
        fixed_value_column_bits(),
        4,
        &dot_high_point,
        &mut claims,
    )?;
    let row_read_ts = push_rectangular_opening_claim(
        polynomial,
        fixed_value_column_bits(),
        5,
        &product_claims.ops.product_point,
        &mut claims,
    )?;
    let col_read_ts = push_rectangular_opening_claim(
        polynomial,
        fixed_value_column_bits(),
        6,
        &product_claims.ops.product_point,
        &mut claims,
    )?;
    Ok((
        claims,
        SparkFixedValueOpeningEvals {
            val_a_low,
            val_a_high,
            val_b_low,
            val_b_high,
            val_c_low,
            val_c_high,
            row_addr,
            col_addr,
            row_read_ts,
            col_read_ts,
        },
    ))
}

fn fixed_value_opening_claims_from_evals<EF>(
    product_claims: &SparkBatchedMemoryProductsLeafClaims<EF>,
    evals: &SparkFixedTableOpeningEvals<EF>,
) -> Result<Vec<(MultilinearPoint<EF>, EF)>, SpartanWhirError>
where
    EF: ExtField,
{
    let (dot_low_point, dot_high_point) = dotproduct_full_domain_points(&product_claims.ops)?;
    Ok(vec![
        (
            rectangular_point::<EF>(
                0,
                fixed_value_column_bits(),
                &product_claims.ops.product_point,
            )?,
            evals.row_addr,
        ),
        (
            rectangular_point::<EF>(
                1,
                fixed_value_column_bits(),
                &product_claims.ops.product_point,
            )?,
            evals.col_addr,
        ),
        (
            rectangular_point::<EF>(2, fixed_value_column_bits(), &dot_low_point)?,
            evals.val_a_low,
        ),
        (
            rectangular_point::<EF>(2, fixed_value_column_bits(), &dot_high_point)?,
            evals.val_a_high,
        ),
        (
            rectangular_point::<EF>(3, fixed_value_column_bits(), &dot_low_point)?,
            evals.val_b_low,
        ),
        (
            rectangular_point::<EF>(3, fixed_value_column_bits(), &dot_high_point)?,
            evals.val_b_high,
        ),
        (
            rectangular_point::<EF>(4, fixed_value_column_bits(), &dot_low_point)?,
            evals.val_c_low,
        ),
        (
            rectangular_point::<EF>(4, fixed_value_column_bits(), &dot_high_point)?,
            evals.val_c_high,
        ),
        (
            rectangular_point::<EF>(
                5,
                fixed_value_column_bits(),
                &product_claims.ops.product_point,
            )?,
            evals.row_read_ts,
        ),
        (
            rectangular_point::<EF>(
                6,
                fixed_value_column_bits(),
                &product_claims.ops.product_point,
            )?,
            evals.col_read_ts,
        ),
    ])
}

fn fixed_audit_opening_claims_from_prover_data<EF, D>(
    config: &WhirPcsConfig,
    prover_data: &D,
    row_memory_size: usize,
    col_memory_size: usize,
    product_claims: &SparkBatchedMemoryProductsLeafClaims<EF>,
) -> Result<
    (
        Vec<(MultilinearPoint<EF>, EF)>,
        SparkFixedAuditOpeningEvals<EF>,
    ),
    SpartanWhirError,
>
where
    EF: ExtField,
    D: CommittedPolynomialView<EF>,
{
    let _profile = profile_scope("spark_audit_opening_claims");
    if prover_data.num_variables() != config.num_variables {
        return Err(SpartanWhirError::invalid_config());
    }
    let audit_memory_bits = config
        .num_variables
        .checked_sub(fixed_audit_column_bits())
        .ok_or(SpartanWhirError::invalid_config())?;
    let polynomial = prover_data.polynomial();
    let row_point = low_block_memory_point(
        &product_claims.mem.product_point,
        row_memory_size,
        audit_memory_bits,
    )?;
    let col_point = low_block_memory_point(
        &product_claims.mem.product_point,
        col_memory_size,
        audit_memory_bits,
    )?;
    let mut claims = Vec::with_capacity(2);
    let row_audit_ts = push_rectangular_opening_claim(
        polynomial,
        fixed_audit_column_bits(),
        0,
        &row_point,
        &mut claims,
    )?;
    let col_audit_ts = push_rectangular_opening_claim(
        polynomial,
        fixed_audit_column_bits(),
        1,
        &col_point,
        &mut claims,
    )?;
    Ok((
        claims,
        SparkFixedAuditOpeningEvals {
            row_audit_ts,
            col_audit_ts,
        },
    ))
}

fn embedded_fixed_audit_opening_claims_from_prover_data<EF, D>(
    fixed_value_config: &WhirPcsConfig,
    prover_data: &D,
    row_memory_size: usize,
    col_memory_size: usize,
    product_claims: &SparkBatchedMemoryProductsLeafClaims<EF>,
) -> Result<
    (
        Vec<(MultilinearPoint<EF>, EF)>,
        SparkFixedAuditOpeningEvals<EF>,
    ),
    SpartanWhirError,
>
where
    EF: ExtField,
    D: CommittedPolynomialView<EF>,
{
    if prover_data.num_variables() != fixed_value_config.num_variables {
        return Err(SpartanWhirError::invalid_config());
    }
    let (row_point, col_point) = embedded_fixed_audit_opening_points(
        fixed_value_config,
        row_memory_size,
        col_memory_size,
        product_claims,
    )?;
    let polynomial = prover_data.polynomial();
    let row_audit_ts = evaluate_base_mle_table_as_extension(polynomial, &row_point.0)?;
    let col_audit_ts = evaluate_base_mle_table_as_extension(polynomial, &col_point.0)?;
    Ok((
        vec![(row_point, row_audit_ts), (col_point, col_audit_ts)],
        SparkFixedAuditOpeningEvals {
            row_audit_ts,
            col_audit_ts,
        },
    ))
}

fn embedded_fixed_audit_opening_claims_from_evals<EF>(
    fixed_value_config: &WhirPcsConfig,
    row_memory_size: usize,
    col_memory_size: usize,
    product_claims: &SparkBatchedMemoryProductsLeafClaims<EF>,
    evals: &SparkFixedTableOpeningEvals<EF>,
) -> Result<Vec<(MultilinearPoint<EF>, EF)>, SpartanWhirError>
where
    EF: ExtField,
{
    let (row_point, col_point) = embedded_fixed_audit_opening_points(
        fixed_value_config,
        row_memory_size,
        col_memory_size,
        product_claims,
    )?;
    Ok(vec![
        (row_point, evals.row_audit_ts),
        (col_point, evals.col_audit_ts),
    ])
}

fn embedded_fixed_audit_opening_points<EF>(
    fixed_value_config: &WhirPcsConfig,
    row_memory_size: usize,
    col_memory_size: usize,
    product_claims: &SparkBatchedMemoryProductsLeafClaims<EF>,
) -> Result<(MultilinearPoint<EF>, MultilinearPoint<EF>), SpartanWhirError>
where
    EF: ExtField,
{
    let value_bits = fixed_value_config
        .num_variables
        .checked_sub(fixed_value_column_bits())
        .ok_or_else(SpartanWhirError::invalid_config)?;
    let value_domain_size = 1usize
        .checked_shl(value_bits as u32)
        .ok_or_else(SpartanWhirError::invalid_config)?;
    if !spark_fixed_audit_is_embedded(value_domain_size, row_memory_size, col_memory_size) {
        return Err(SpartanWhirError::invalid_config());
    }
    let memory_bits = product_claims.mem.product_point.0.len();
    let prefix_len = value_bits
        .checked_sub(
            memory_bits
                .checked_add(1)
                .ok_or_else(SpartanWhirError::invalid_config)?,
        )
        .ok_or_else(SpartanWhirError::invalid_config)?;
    let row_memory_point = low_block_memory_point(
        &product_claims.mem.product_point,
        row_memory_size,
        memory_bits,
    )?;
    let col_memory_point = low_block_memory_point(
        &product_claims.mem.product_point,
        col_memory_size,
        memory_bits,
    )?;
    let make_point = |axis: EF, memory_point: &MultilinearPoint<EF>| {
        let mut base_point = vec![EF::ZERO; prefix_len];
        base_point.push(axis);
        base_point.extend_from_slice(&memory_point.0);
        rectangular_point::<EF>(7, fixed_value_column_bits(), &MultilinearPoint(base_point))
    };
    Ok((
        make_point(EF::ZERO, &row_memory_point)?,
        make_point(EF::ONE, &col_memory_point)?,
    ))
}

fn fixed_audit_opening_claims_from_evals<EF>(
    config: &WhirPcsConfig,
    row_memory_size: usize,
    col_memory_size: usize,
    product_claims: &SparkBatchedMemoryProductsLeafClaims<EF>,
    evals: &SparkFixedTableOpeningEvals<EF>,
) -> Result<Vec<(MultilinearPoint<EF>, EF)>, SpartanWhirError>
where
    EF: ExtField,
{
    let audit_memory_bits = config
        .num_variables
        .checked_sub(fixed_audit_column_bits())
        .ok_or(SpartanWhirError::invalid_config())?;
    let row_point = low_block_memory_point(
        &product_claims.mem.product_point,
        row_memory_size,
        audit_memory_bits,
    )?;
    let col_point = low_block_memory_point(
        &product_claims.mem.product_point,
        col_memory_size,
        audit_memory_bits,
    )?;
    Ok(vec![
        (
            rectangular_point::<EF>(0, fixed_audit_column_bits(), &row_point)?,
            evals.row_audit_ts,
        ),
        (
            rectangular_point::<EF>(1, fixed_audit_column_bits(), &col_point)?,
            evals.col_audit_ts,
        ),
    ])
}

#[derive(Debug, Clone, Copy)]
struct SparkFixedValueOpeningEvals<EF> {
    val_a_low: EF,
    val_a_high: EF,
    val_b_low: EF,
    val_b_high: EF,
    val_c_low: EF,
    val_c_high: EF,
    row_addr: EF,
    col_addr: EF,
    row_read_ts: EF,
    col_read_ts: EF,
}

#[derive(Debug, Clone, Copy)]
struct SparkFixedAuditOpeningEvals<EF> {
    row_audit_ts: EF,
    col_audit_ts: EF,
}

struct SparkReadOpeningPoints<EF> {
    erow_low: MultilinearPoint<EF>,
    erow_high: MultilinearPoint<EF>,
    erow_ops: MultilinearPoint<EF>,
    ecol_low: MultilinearPoint<EF>,
    ecol_high: MultilinearPoint<EF>,
    ecol_ops: MultilinearPoint<EF>,
}

fn read_opening_points<EF>(
    product_claims: &SparkBatchedMemoryProductsLeafClaims<EF>,
) -> Result<SparkReadOpeningPoints<EF>, SpartanWhirError>
where
    EF: ExtField,
{
    let (low, high) = dotproduct_full_domain_points(&product_claims.ops)?;
    Ok(SparkReadOpeningPoints {
        erow_low: low.clone(),
        erow_high: high.clone(),
        erow_ops: product_claims.ops.product_point.clone(),
        ecol_low: low,
        ecol_high: high,
        ecol_ops: product_claims.ops.product_point.clone(),
    })
}

fn dotproduct_full_domain_points<EF>(
    claims: &crate::SparkBatchedProductLeafClaims<EF>,
) -> Result<(MultilinearPoint<EF>, MultilinearPoint<EF>), SpartanWhirError>
where
    EF: ExtField,
{
    let mut low = Vec::with_capacity(claims.dotproduct_point.0.len() + 1);
    low.push(EF::ZERO);
    low.extend_from_slice(&claims.dotproduct_point.0);
    let mut high = Vec::with_capacity(claims.dotproduct_point.0.len() + 1);
    high.push(EF::ONE);
    high.extend_from_slice(&claims.dotproduct_point.0);
    if claims.product_point.0.len() != low.len() {
        return Err(SpartanWhirError::InvalidRoundCount);
    }
    Ok((MultilinearPoint(low), MultilinearPoint(high)))
}

fn low_block_memory_point<EF>(
    padded_point: &MultilinearPoint<EF>,
    memory_size: usize,
    target_len: usize,
) -> Result<MultilinearPoint<EF>, SpartanWhirError>
where
    EF: ExtField,
{
    if memory_size == 0 || !memory_size.is_power_of_two() {
        return Err(SpartanWhirError::invalid_config());
    }
    let memory_bits = memory_size.ilog2() as usize;
    if memory_bits > target_len || padded_point.0.len() < memory_bits {
        return Err(SpartanWhirError::invalid_config());
    }
    let suffix_start = padded_point.0.len() - memory_bits;
    let mut point = vec![EF::ZERO; target_len - memory_bits];
    point.extend_from_slice(&padded_point.0[suffix_start..]);
    Ok(MultilinearPoint(point))
}

fn push_rectangular_opening_claim<EF>(
    polynomial: &[F],
    column_bits: usize,
    column: usize,
    base_point: &MultilinearPoint<EF>,
    claims: &mut Vec<(MultilinearPoint<EF>, EF)>,
) -> Result<EF, SpartanWhirError>
where
    EF: ExtField,
{
    let point = rectangular_point(column, column_bits, base_point)?;
    let value =
        evaluate_base_rectangular_opening_claim(polynomial, column_bits, column, base_point)?;
    claims.push((point, value));
    Ok(value)
}

fn evaluate_base_rectangular_opening_claim<EF>(
    polynomial: &[F],
    column_bits: usize,
    column: usize,
    base_point: &MultilinearPoint<EF>,
) -> Result<EF, SpartanWhirError>
where
    EF: ExtField,
{
    let column_count = 1usize
        .checked_shl(column_bits as u32)
        .ok_or(SpartanWhirError::invalid_config())?;
    let domain_size = 1usize
        .checked_shl(base_point.0.len() as u32)
        .ok_or(SpartanWhirError::invalid_config())?;
    if column >= column_count
        || polynomial.len()
            != domain_size
                .checked_mul(column_count)
                .ok_or(SpartanWhirError::invalid_config())?
    {
        return Err(SpartanWhirError::invalid_config());
    }

    let start = column
        .checked_mul(domain_size)
        .ok_or(SpartanWhirError::invalid_config())?;
    let end = start
        .checked_add(domain_size)
        .ok_or(SpartanWhirError::invalid_config())?;
    evaluate_base_mle_table_as_extension(&polynomial[start..end], &base_point.0)
}

fn evaluate_base_mle_table_as_extension<EF>(
    table: &[F],
    point: &[EF],
) -> Result<EF, SpartanWhirError>
where
    EF: ExtField,
{
    if table.is_empty() || !table.len().is_power_of_two() {
        return Err(SpartanWhirError::InvalidRoundPolynomial);
    }
    let expected_len = 1usize
        .checked_shl(point.len() as u32)
        .ok_or(SpartanWhirError::InvalidRoundPolynomial)?;
    if table.len() != expected_len {
        return Err(SpartanWhirError::InvalidRoundPolynomial);
    }

    if point.is_empty() {
        return Ok(EF::from(table[0]));
    }

    let first_challenge = point[0];
    let first_half = table.len() / 2;
    let mut layer: Vec<EF> = if cfg!(feature = "parallel") && first_half >= (1 << 14) {
        (0..first_half)
            .into_par_iter()
            .map(|i| {
                let lo = table[i];
                let hi = table[i + first_half];
                EF::from(lo) + first_challenge * (hi - lo)
            })
            .collect()
    } else {
        (0..first_half)
            .map(|i| {
                let lo = table[i];
                let hi = table[i + first_half];
                EF::from(lo) + first_challenge * (hi - lo)
            })
            .collect()
    };

    let mut active = layer.len();
    for &r_i in &point[1..] {
        let half = active / 2;
        let (low, high) = layer[..active].split_at_mut(half);
        if cfg!(feature = "parallel") && half >= (1 << 14) {
            low.par_iter_mut()
                .zip(high.par_iter())
                .for_each(|(lo, &hi)| *lo += r_i * (hi - *lo));
        } else {
            for (lo, &hi) in low.iter_mut().zip(high.iter()) {
                *lo += r_i * (hi - *lo);
            }
        }
        active = half;
    }

    Ok(layer[0])
}

fn rectangular_point<EF>(
    column: usize,
    column_bits: usize,
    base_point: &MultilinearPoint<EF>,
) -> Result<MultilinearPoint<EF>, SpartanWhirError>
where
    EF: ExtField,
{
    let column_count = 1usize
        .checked_shl(column_bits as u32)
        .ok_or(SpartanWhirError::invalid_config())?;
    if column >= column_count {
        return Err(SpartanWhirError::invalid_config());
    }
    let mut point = Vec::with_capacity(column_bits + base_point.0.len());
    for i in 0..column_bits {
        let bit = (column >> (column_bits - i - 1)) & 1;
        point.push(if bit == 0 { EF::ZERO } else { EF::ONE });
    }
    point.extend_from_slice(&base_point.0);
    Ok(MultilinearPoint(point))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SparkReadCoordinateGroup {
    column_start: usize,
    column_count: usize,
}

fn read_coordinate_groups<EF>() -> Result<Vec<SparkReadCoordinateGroup>, SpartanWhirError>
where
    EF: ExtField,
{
    let total_columns = EF::DIMENSION
        .checked_mul(2)
        .ok_or_else(SpartanWhirError::invalid_config)?;
    if total_columns == 0 {
        return Err(SpartanWhirError::invalid_config());
    }

    let mut groups = Vec::new();
    let mut column_start = 0usize;
    let mut remaining = total_columns;
    while remaining != 0 {
        let column_count = 1usize << (usize::BITS - 1 - remaining.leading_zeros());
        groups.push(SparkReadCoordinateGroup {
            column_start,
            column_count,
        });
        column_start = column_start
            .checked_add(column_count)
            .ok_or_else(SpartanWhirError::invalid_config)?;
        remaining -= column_count;
    }
    Ok(groups)
}

/// Selector bits in the largest SPARK read-coordinate commitment.
///
/// Non-power-of-two widths are represented by several commitments whose
/// column counts are the descending powers in the binary decomposition of
/// `2 * EF::DIMENSION`.
pub fn read_table_column_bits<EF>() -> usize
where
    EF: ExtField,
{
    (EF::DIMENSION * 2).ilog2() as usize
}

/// Column counts for the SPARK read-coordinate commitments.
pub fn read_table_group_column_counts<EF>() -> Result<Vec<usize>, SpartanWhirError>
where
    EF: ExtField,
{
    Ok(read_coordinate_groups::<EF>()?
        .into_iter()
        .map(|group| group.column_count)
        .collect())
}

pub fn fixed_value_column_count() -> usize {
    8
}

pub fn fixed_value_column_bits() -> usize {
    3
}

pub fn fixed_audit_column_count() -> usize {
    2
}

pub fn fixed_audit_column_bits() -> usize {
    1
}

fn point_eval_statement<E, EF>(
    claims: &[(MultilinearPoint<EF>, EF)],
) -> Result<crate::PcsStatement<E>, SpartanWhirError>
where
    EF: ExtField,
    E: SpartanWhirEngine<F = F, EF = EF>,
{
    let mut builder = PcsStatementBuilder::<E>::new();
    for (point, value) in claims {
        builder = builder.add_point_eval(PointEvalClaim {
            point: point.clone(),
            value: *value,
        });
    }
    builder.finalize()
}

fn validate_spark_fixed_opening_shape<E, EF, Pcs>(
    fixed_value_config: &WhirPcsConfig,
    audit_config: &WhirPcsConfig,
    audit_embedded: bool,
    proof: &SparkFixedOpeningProof<E, Pcs>,
) -> Result<(), SpartanWhirError>
where
    EF: ExtField,
    E: SpartanContextEngine<EF = EF>,
    Pcs: MlePcs<E, Config = WhirPcsConfig>,
{
    if proof.value_num_variables != fixed_value_config.num_variables
        || proof.value_column_bits != fixed_value_column_bits()
        || proof.audit_num_variables != audit_config.num_variables
        || proof.audit_column_bits != fixed_audit_column_bits()
        || audit_embedded != proof.audit_commitment.is_none()
        || audit_embedded != proof.audit_proof.is_none()
    {
        return Err(SpartanWhirError::invalid_config());
    }
    Ok(())
}

fn validate_spark_fixed_commitments<E, EF, Pcs>(
    proof: &SparkFixedOpeningProof<E, Pcs>,
    expected: &SparkFixedCommitments<<Pcs as MlePcs<E>>::Commitment>,
) -> Result<(), SpartanWhirError>
where
    EF: ExtField,
    E: SpartanContextEngine<EF = EF>,
    Pcs: MlePcs<E, Config = WhirPcsConfig>,
    <Pcs as MlePcs<E>>::Commitment: PartialEq,
{
    if proof.value_commitment != expected.value || proof.audit_commitment != expected.audit {
        return Err(SpartanWhirError::CommitmentMismatch);
    }
    Ok(())
}

fn validate_spark_read_opening_shape<E, EF, Pcs>(
    configs: &[WhirPcsConfig],
    proof: &SparkReadOpeningProof<E, Pcs>,
) -> Result<(), SpartanWhirError>
where
    EF: ExtField,
    E: SpartanContextEngine<EF = EF>,
    Pcs: MlePcs<E, Config = WhirPcsConfig>,
{
    let groups = read_coordinate_groups::<EF>()?;
    if configs.len() != groups.len() || proof.groups.len() != groups.len() {
        return Err(SpartanWhirError::invalid_config());
    }
    for ((config, expected), group) in configs.iter().zip(groups).zip(&proof.groups) {
        let requests = spark_read_group_opening_requests::<EF>(expected)?;
        if group.num_variables != config.num_variables
            || group.column_start != expected.column_start
            || group.column_count != expected.column_count
            || group.evals.len() != requests.len()
            || group
                .evals
                .iter()
                .zip(requests)
                .any(|(evals, request)| evals.len() != request.columns.len())
        {
            return Err(SpartanWhirError::invalid_config());
        }
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SparkReadOpeningRequest {
    logical_opening: usize,
    columns: Vec<usize>,
}

fn spark_read_group_opening_requests<EF>(
    group: SparkReadCoordinateGroup,
) -> Result<Vec<SparkReadOpeningRequest>, SpartanWhirError>
where
    EF: ExtField,
{
    let group_end = group
        .column_start
        .checked_add(group.column_count)
        .ok_or_else(SpartanWhirError::invalid_config)?;
    let mut requests = Vec::with_capacity(6);
    for (table, table_start) in [0usize, EF::DIMENSION].into_iter().enumerate() {
        let table_end = table_start
            .checked_add(EF::DIMENSION)
            .ok_or_else(SpartanWhirError::invalid_config)?;
        let start = group.column_start.max(table_start);
        let end = group_end.min(table_end);
        if start >= end {
            continue;
        }
        let columns = (start - group.column_start..end - group.column_start).collect::<Vec<_>>();
        for opening in 0..3 {
            requests.push(SparkReadOpeningRequest {
                logical_opening: table * 3 + opening,
                columns: columns.clone(),
            });
        }
    }
    if requests.is_empty() {
        return Err(SpartanWhirError::invalid_config());
    }
    Ok(requests)
}

fn grouped_read_coordinate_evals<E, EF, Pcs>(
    proof: &SparkReadOpeningProof<E, Pcs>,
) -> Result<[Vec<EF>; 6], SpartanWhirError>
where
    EF: ExtField,
    E: SpartanContextEngine<EF = EF>,
    Pcs: MlePcs<E, Config = WhirPcsConfig>,
{
    let groups = read_coordinate_groups::<EF>()?;
    if proof.groups.len() != groups.len() {
        return Err(SpartanWhirError::invalid_config());
    }
    let mut coordinate_evals = core::array::from_fn(|_| vec![EF::ZERO; EF::DIMENSION]);
    let mut seen: [Vec<bool>; 6] = core::array::from_fn(|_| vec![false; EF::DIMENSION]);
    for (group, opening) in groups.into_iter().zip(&proof.groups) {
        let requests = spark_read_group_opening_requests::<EF>(group)?;
        if opening.evals.len() != requests.len() {
            return Err(SpartanWhirError::invalid_config());
        }
        for (request, evals) in requests.iter().zip(&opening.evals) {
            if request.columns.len() != evals.len() {
                return Err(SpartanWhirError::invalid_config());
            }
            for (&local_column, &eval) in request.columns.iter().zip(evals) {
                let global_column = group
                    .column_start
                    .checked_add(local_column)
                    .ok_or_else(SpartanWhirError::invalid_config)?;
                let coordinate = if request.logical_opening < 3 {
                    global_column
                } else {
                    global_column
                        .checked_sub(EF::DIMENSION)
                        .ok_or_else(SpartanWhirError::invalid_config)?
                };
                if coordinate >= EF::DIMENSION || seen[request.logical_opening][coordinate] {
                    return Err(SpartanWhirError::invalid_config());
                }
                coordinate_evals[request.logical_opening][coordinate] = eval;
                seen[request.logical_opening][coordinate] = true;
            }
        }
    }
    if seen.iter().flatten().any(|seen| !seen) {
        return Err(SpartanWhirError::invalid_config());
    }
    Ok(coordinate_evals)
}

fn split_read_coordinate_evals<EF>(
    erow_evals: &[Vec<EF>],
    ecol_evals: &[Vec<EF>],
) -> Result<SparkReadTableOpeningEvals<EF>, SpartanWhirError>
where
    EF: ExtField,
{
    if erow_evals.len() != 3 || ecol_evals.len() != 3 {
        return Err(SpartanWhirError::invalid_config());
    }
    Ok(SparkReadTableOpeningEvals {
        erow_low: recombine_coordinate_evals::<EF>(&erow_evals[0])?,
        erow_high: recombine_coordinate_evals::<EF>(&erow_evals[1])?,
        erow_ops: recombine_coordinate_evals::<EF>(&erow_evals[2])?,
        ecol_low: recombine_coordinate_evals::<EF>(&ecol_evals[0])?,
        ecol_high: recombine_coordinate_evals::<EF>(&ecol_evals[1])?,
        ecol_ops: recombine_coordinate_evals::<EF>(&ecol_evals[2])?,
    })
}

fn extension_read_tables_to_base_columns<EF>(
    erow: &[EF],
    ecol: &[EF],
) -> Result<Vec<F>, SpartanWhirError>
where
    EF: ExtField,
{
    if erow.is_empty() || !erow.len().is_power_of_two() || ecol.len() != erow.len() {
        return Err(SpartanWhirError::InvalidPolynomialLength);
    }

    let domain_size = erow.len();
    let coordinate_count = EF::DIMENSION;
    let len = domain_size
        .checked_mul(2)
        .and_then(|len| len.checked_mul(coordinate_count))
        .ok_or_else(SpartanWhirError::invalid_config)?;
    let mut columns = vec![F::ZERO; len];
    for (table_index, table) in [erow, ecol].into_iter().enumerate() {
        for (row, value) in table.iter().enumerate() {
            for (coordinate, &coefficient) in value.as_basis_coefficients_slice().iter().enumerate()
            {
                let column = table_index * coordinate_count + coordinate;
                columns[column * domain_size + row] = coefficient;
            }
        }
    }
    Ok(columns)
}

fn recombine_coordinate_evals<EF>(evals: &[EF]) -> Result<EF, SpartanWhirError>
where
    EF: ExtField,
{
    if evals.len() != EF::DIMENSION {
        return Err(SpartanWhirError::invalid_config());
    }
    let mut out = EF::ZERO;
    for (i, &eval) in evals.iter().enumerate() {
        let basis = EF::ith_basis_element(i).ok_or(SpartanWhirError::invalid_config())?;
        out += eval * basis;
    }
    Ok(out)
}

#[cfg(test)]
fn build_public_half(num_vars: usize, public_inputs: &[F]) -> Vec<F> {
    let mut out = vec![F::ZERO; num_vars];
    out[0] = F::ONE;
    for (i, &x) in public_inputs.iter().enumerate() {
        out[i + 1] = x;
    }
    out
}

fn build_z_full(mut witness_half: Vec<F>, num_vars: usize, public_inputs: &[F]) -> Vec<F> {
    debug_assert_eq!(witness_half.len(), num_vars);
    let public_offset = witness_half.len();
    witness_half.resize(public_offset + num_vars, F::ZERO);
    witness_half[public_offset] = F::ONE;
    for (i, &value) in public_inputs.iter().enumerate() {
        witness_half[public_offset + i + 1] = value;
    }
    witness_half
}

fn matrix_z_slice(
    z_full: &[F],
    num_vars: usize,
    public_input_len: usize,
) -> Result<&[F], SpartanWhirError> {
    let len = num_vars
        .checked_add(public_input_len)
        .and_then(|n| n.checked_add(1))
        .ok_or(SpartanWhirError::InvalidWitnessLength)?;
    z_full
        .get(..len)
        .ok_or(SpartanWhirError::InvalidWitnessLength)
}

fn evaluate_public_half<EF>(
    num_vars: usize,
    public_inputs: &[F],
    point: &[EF],
) -> Result<EF, SpartanWhirError>
where
    EF: ExtField,
{
    if num_vars == 0 || !num_vars.is_power_of_two() {
        return Err(SpartanWhirError::InvalidRoundPolynomial);
    }
    if point.len() != num_vars.ilog2() as usize {
        return Err(SpartanWhirError::InvalidRoundPolynomial);
    }
    if public_inputs.len() >= num_vars {
        return Err(SpartanWhirError::InvalidPublicInputLength);
    }

    let active_len = public_inputs
        .len()
        .checked_add(1)
        .and_then(usize::checked_next_power_of_two)
        .ok_or(SpartanWhirError::InvalidPublicInputLength)?;
    let active_vars = active_len.ilog2() as usize;
    let fixed_zero_vars = point.len() - active_vars;
    let fixed_zero_weight = point[..fixed_zero_vars]
        .iter()
        .fold(EF::ONE, |acc, &r| acc * (EF::ONE - r));

    let mut table = vec![EF::ZERO; active_len];
    table[0] = EF::ONE;
    for (i, &value) in public_inputs.iter().enumerate() {
        table[i + 1] = EF::from(value);
    }
    Ok(fixed_zero_weight * crate::evaluate_mle_table(&table, &point[fixed_zero_vars..])?)
}

#[cfg(test)]
fn build_matrix_z(witness: &[F], public_inputs: &[F]) -> Vec<F> {
    let mut z = Vec::with_capacity(witness.len() + 1 + public_inputs.len());
    z.extend_from_slice(witness);
    z.push(F::ONE);
    z.extend_from_slice(public_inputs);
    z
}

fn eq_point_eval<Ext>(a: &[Ext], b: &[Ext]) -> Ext
where
    Ext: Field,
{
    a.iter().zip(b.iter()).fold(Ext::ONE, |acc, (&x, &y)| {
        acc * ((Ext::ONE - x) * (Ext::ONE - y) + x * y)
    })
}

fn recover_witness_eval<EF: Field>(r0: EF, eval_z: EF, eval_x: EF) -> Result<EF, SpartanWhirError> {
    let denom = EF::ONE - r0;
    let denom_inv = denom
        .try_inverse()
        .ok_or(SpartanWhirError::NonInvertibleElement)?;
    Ok((eval_z - r0 * eval_x) * denom_inv)
}

#[cfg(test)]
mod tests {
    use super::{
        application_relation, begin_spark_opening_branches, build_matrix_z, build_public_half,
        build_z_full, combine_application_mask_vectors, evaluate_public_half,
        extension_read_tables_to_base_columns, matrix_z_slice, merge_spark_opening_branches,
        observe_poseidon_zk_context, observe_spark_opening_tag, power_covector,
        recover_witness_eval, sample_inner_masks, sample_outer_masks, seal_spark_opening_branch,
        spark_opening_branch, SparkOpeningBranch, SparkOpeningBranchSeal,
        SPARK_OPENING_BRANCH_DOMAIN, SPARK_OPENING_BRANCH_END, SPARK_OPENING_BRANCH_MERGE,
    };
    use crate::{
        engine::{Plonky3PoseidonEngine, Poseidon1QuarticEngine, PoseidonQuarticEngine, F},
        DomainSeparator, FullZkPoseidonEngine, MatrixClosingMode, QuarticBinExtension, R1csShape,
        SecurityConfig, SparseMatEntry, SparseMatrix, SpartanWhirEngine, WhirParams,
        ZkWhirPcsConfig, SPARK_MATRIX_CLOSING_VERSION,
    };
    use p3_challenger::{CanSample, FieldChallenger, GrindingChallenger};
    use p3_field::{BasedVectorSpace, HornerIter, PrimeCharacteristicRing};
    use rand::{rngs::StdRng, RngExt, SeedableRng};

    type EF = QuarticBinExtension;

    fn dot(lhs: &[EF], rhs: &[EF]) -> EF {
        lhs.iter().zip(rhs).map(|(&a, &b)| a * b).sum()
    }

    fn spark_opening_branch_transcript<C>(
        mut challenger: C,
        fixed_payload: F,
        read_payload: F,
        reverse_seals: bool,
    ) -> (SparkOpeningBranchSeal, SparkOpeningBranchSeal, EF)
    where
        C: Clone + FieldChallenger<F>,
    {
        challenger.observe(F::from_u32(101));
        begin_spark_opening_branches(&mut challenger);
        let mut fixed_challenger = spark_opening_branch(&challenger, SparkOpeningBranch::Fixed);
        let mut read_challenger = spark_opening_branch(&challenger, SparkOpeningBranch::Read);
        fixed_challenger.observe(fixed_payload);
        read_challenger.observe(read_payload);
        let fixed_seal = seal_spark_opening_branch(&mut fixed_challenger);
        let read_seal = seal_spark_opening_branch(&mut read_challenger);
        if reverse_seals {
            merge_spark_opening_branches(&mut challenger, read_seal, fixed_seal);
        } else {
            merge_spark_opening_branches(&mut challenger, fixed_seal, read_seal);
        }
        let continuation = challenger.sample_algebra_element::<EF>();
        (fixed_seal, read_seal, continuation)
    }

    fn assert_spark_opening_branch_binding<C>(new_challenger: impl Fn() -> C)
    where
        C: Clone + FieldChallenger<F>,
    {
        let fixed_payload = F::from_u32(17);
        let read_payload = F::from_u32(29);
        let baseline =
            spark_opening_branch_transcript(new_challenger(), fixed_payload, read_payload, false);
        assert_eq!(
            baseline,
            spark_opening_branch_transcript(new_challenger(), fixed_payload, read_payload, false,)
        );

        let changed_fixed = spark_opening_branch_transcript(
            new_challenger(),
            fixed_payload + F::ONE,
            read_payload,
            false,
        );
        assert_ne!(baseline.0, changed_fixed.0);
        assert_eq!(baseline.1, changed_fixed.1);
        assert_ne!(baseline.2, changed_fixed.2);

        let changed_read = spark_opening_branch_transcript(
            new_challenger(),
            fixed_payload,
            read_payload + F::ONE,
            false,
        );
        assert_eq!(baseline.0, changed_read.0);
        assert_ne!(baseline.1, changed_read.1);
        assert_ne!(baseline.2, changed_read.2);

        let reversed =
            spark_opening_branch_transcript(new_challenger(), fixed_payload, read_payload, true);
        assert_ne!(baseline.2, reversed.2);

        let same_payload =
            spark_opening_branch_transcript(new_challenger(), fixed_payload, fixed_payload, false);
        assert_ne!(same_payload.0, same_payload.1);
    }

    fn same_geometry_relations() -> (R1csShape<F>, [R1csShape<F>; 3]) {
        let matrix = |value| SparseMatrix {
            num_rows: 2,
            num_cols: 4,
            entries: vec![SparseMatEntry {
                row: 0,
                col: 0,
                val: value,
            }],
        };
        let first = R1csShape {
            num_cons: 2,
            num_vars: 2,
            num_io: 1,
            a: matrix(F::ONE),
            b: matrix(F::from_u8(2)),
            c: matrix(F::from_u8(3)),
        };
        let mut changed_a = first.clone();
        changed_a.a.entries[0].val += F::ONE;
        let mut changed_b = first.clone();
        changed_b.b.entries[0].val += F::ONE;
        let mut changed_c = first.clone();
        changed_c.c.entries[0].val += F::ONE;
        (first, [changed_a, changed_b, changed_c])
    }

    fn first_full_zk_context_challenge<E>(shape: &R1csShape<F>) -> EF
    where
        E: FullZkPoseidonEngine + SpartanWhirEngine<EF = EF>,
        E::Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        let domain_separator = DomainSeparator::new_with_protocol_id(
            E::FULL_ZK_PROTOCOL_ID,
            shape,
            shape.num_cons,
            shape.num_vars,
            shape.num_io,
            &SecurityConfig::default(),
            &WhirParams::default(),
            MatrixClosingMode::DirectSparse,
            None,
        )
        .expect("valid relation");
        let mut challenger = <E as Plonky3PoseidonEngine>::challenger();
        observe_poseidon_zk_context::<E>(
            &mut challenger,
            &domain_separator,
            &ZkWhirPcsConfig::default(),
            1,
            2,
            &[F::from_u8(7)],
        );
        challenger.sample_algebra_element::<EF>()
    }

    #[test]
    fn poseidon2_full_zk_context_binds_relation_before_first_challenge() {
        let (relation, changed_relations) = same_geometry_relations();
        let baseline = first_full_zk_context_challenge::<PoseidonQuarticEngine>(&relation);
        for changed in &changed_relations {
            assert_ne!(
                baseline,
                first_full_zk_context_challenge::<PoseidonQuarticEngine>(changed)
            );
        }
    }

    #[test]
    fn poseidon1_full_zk_context_binds_relation_before_first_challenge() {
        let (relation, changed_relations) = same_geometry_relations();
        let baseline = first_full_zk_context_challenge::<Poseidon1QuarticEngine>(&relation);
        for changed in &changed_relations {
            assert_ne!(
                baseline,
                first_full_zk_context_challenge::<Poseidon1QuarticEngine>(changed)
            );
        }
    }

    #[test]
    fn spark_opening_branch_tags_match_matrix_closing_version() {
        let suffix = format!("-v{SPARK_MATRIX_CLOSING_VERSION}");
        for tag in [
            SPARK_OPENING_BRANCH_DOMAIN,
            SPARK_OPENING_BRANCH_END,
            SPARK_OPENING_BRANCH_MERGE,
        ] {
            assert!(tag.ends_with(suffix.as_bytes()));
        }
    }

    #[test]
    fn spark_opening_branch_tag_changes_change_transcript() {
        for current in [
            SPARK_OPENING_BRANCH_DOMAIN,
            SPARK_OPENING_BRANCH_END,
            SPARK_OPENING_BRANCH_MERGE,
        ] {
            let mut changed = current.to_vec();
            *changed.last_mut().expect("opening tag is nonempty") ^= 1;
            for sample in [
                |tag: &[u8]| -> F {
                    let mut challenger = crate::engine::poseidon_challenger();
                    observe_spark_opening_tag(&mut challenger, tag);
                    challenger.sample()
                },
                |tag: &[u8]| -> F {
                    let mut challenger = crate::engine::keccak_challenger();
                    observe_spark_opening_tag(&mut challenger, tag);
                    challenger.sample()
                },
            ] {
                assert_ne!(sample(current), sample(&changed));
            }
        }
    }

    #[test]
    fn poseidon_spark_opening_branches_bind_tags_payloads_and_seals() {
        assert_spark_opening_branch_binding(crate::engine::poseidon_challenger);
    }

    #[test]
    fn keccak_spark_opening_branches_bind_tags_payloads_and_seals() {
        assert_spark_opening_branch_binding(crate::engine::keccak_challenger);
    }

    #[test]
    fn read_tables_are_packed_by_table_then_coordinate() {
        let value = |offset: u32| {
            EF::from_basis_coefficients_fn(|coordinate| F::from_u32(offset + coordinate as u32))
        };
        let erow = [value(10), value(20)];
        let ecol = [value(30), value(40)];

        let packed = extension_read_tables_to_base_columns(&erow, &ecol).unwrap();

        for (table_index, table) in [&erow[..], &ecol[..]].into_iter().enumerate() {
            for coordinate in 0..<EF as BasedVectorSpace<F>>::DIMENSION {
                let column = table_index * <EF as BasedVectorSpace<F>>::DIMENSION + coordinate;
                for (row, value) in table.iter().enumerate() {
                    assert_eq!(
                        packed[column * table.len() + row],
                        value.as_basis_coefficients_slice()[coordinate]
                    );
                }
            }
        }
    }

    #[test]
    fn zk_application_relation_matches_separate_equations() {
        let num_rounds = 2;
        let point = [EF::from_u32(2), EF::from_u32(3)];
        let rho = EF::from_u32(5);
        let batching = EF::from_u32(7);
        let inner_messages = [
            vec![EF::ZERO, EF::ONE, EF::from_u32(2), -EF::from_u32(3)],
            vec![EF::ZERO, EF::from_u32(4), EF::from_u32(5), -EF::from_u32(9)],
            vec![
                EF::ZERO,
                EF::from_u32(6),
                EF::from_u32(7),
                -EF::from_u32(13),
            ],
            vec![
                EF::ZERO,
                EF::from_u32(8),
                EF::from_u32(9),
                -EF::from_u32(17),
            ],
            vec![
                EF::ZERO,
                EF::from_u32(10),
                EF::from_u32(11),
                -EF::from_u32(21),
            ],
            vec![
                EF::ZERO,
                EF::from_u32(12),
                EF::from_u32(13),
                -EF::from_u32(25),
            ],
        ];
        let outer_messages = [
            (1..=8).map(EF::from_u32).collect::<Vec<_>>(),
            (11..=18).map(EF::from_u32).collect::<Vec<_>>(),
        ];
        let unmasked = (EF::from_u32(19), EF::from_u32(23), EF::from_u32(29));
        let mut masked = [unmasked.0, unmasked.1, unmasked.2];
        for matrix in 0..3 {
            for round in 0..num_rounds {
                masked[matrix] += inner_messages[matrix * num_rounds + round]
                    .iter()
                    .copied()
                    .horner::<EF, EF>(point[round]);
            }
        }
        let outer_evals = outer_messages
            .iter()
            .zip(point)
            .map(|(message, x)| message.iter().copied().horner::<EF, EF>(x))
            .collect::<Vec<_>>();
        let (inner_covectors, outer_covectors, target) = application_relation(
            num_rounds,
            &point,
            (masked[0], masked[1], masked[2]),
            &outer_evals,
            rho,
            batching,
        );
        let source = unmasked.0 + rho * unmasked.1 + rho.square() * unmasked.2;
        let inner_claim = inner_messages
            .iter()
            .zip(&inner_covectors)
            .map(|(message, covector)| dot(message, covector))
            .sum::<EF>();
        let outer_claim = outer_messages
            .iter()
            .zip(&outer_covectors)
            .map(|(message, covector)| dot(message, covector))
            .sum::<EF>();

        assert_eq!(source + inner_claim + outer_claim, target);
        assert_eq!(power_covector(EF::ONE, 4), vec![EF::ONE; 4]);
    }

    #[test]
    fn combined_application_masks_pad_inner_before_outer() {
        let inner = vec![
            vec![EF::ONE, EF::from_u32(2), EF::from_u32(3), EF::from_u32(4)],
            vec![EF::from_u32(5); 4],
        ];
        let outer = vec![vec![EF::from_u32(6); 8]];
        let combined = combine_application_mask_vectors(
            inner.clone(),
            outer.clone(),
            8,
            crate::pcs_config::MaskPacking::Off,
            false,
        )
        .unwrap();

        assert_eq!(combined.len(), 3);
        assert_eq!(&combined[0][..4], inner[0]);
        assert_eq!(&combined[1][..4], inner[1]);
        assert!(combined[0][4..].iter().all(|&value| value == EF::ZERO));
        assert!(combined[1][4..].iter().all(|&value| value == EF::ZERO));
        assert_eq!(combined[2], outer[0]);
    }

    #[test]
    fn packed_application_basis_preserves_arbitrary_linear_claims() {
        use crate::pcs_config::MaskPacking;
        for rounds in [1, 2, 5] {
            let mut rng = StdRng::seed_from_u64(700 + rounds as u64);
            let inner = sample_inner_masks::<EF, _>(rounds, &mut rng);
            let outer = sample_outer_masks::<EF, _>(rounds, &mut rng);
            let inner_covectors: Vec<Vec<EF>> = inner
                .iter()
                .map(|_| (0..4).map(|_| rng.random()).collect())
                .collect();
            let outer_covectors: Vec<Vec<EF>> = outer
                .iter()
                .map(|_| (0..8).map(|_| rng.random()).collect())
                .collect();
            let expected: EF = inner
                .iter()
                .zip(&inner_covectors)
                .chain(outer.iter().zip(&outer_covectors))
                .map(|(m, c)| m.iter().zip(c).map(|(a, b)| *a * *b).sum::<EF>())
                .sum();
            for packing in [MaskPacking::Application, MaskPacking::ApplicationFreeBasis] {
                let messages = combine_application_mask_vectors(
                    inner.clone(),
                    outer.clone(),
                    8,
                    packing,
                    false,
                )
                .unwrap();
                let covectors = combine_application_mask_vectors(
                    inner_covectors.clone(),
                    outer_covectors.clone(),
                    8,
                    packing,
                    true,
                )
                .unwrap();
                assert_eq!(messages.len(), 1);
                assert_eq!(
                    messages[0]
                        .iter()
                        .zip(&covectors[0])
                        .map(|(a, b)| *a * *b)
                        .sum::<EF>(),
                    expected
                );
                assert_eq!(
                    messages[0].len(),
                    rounds * if packing.free_basis() { 14 } else { 32 }
                );
            }
        }
    }

    #[test]
    fn zk_inner_masks_shift_every_disclosed_matrix_claim() {
        let num_rounds = 3;
        let point = [EF::from_u32(2), EF::from_u32(3), EF::from_u32(4)];
        let unmasked = [EF::from_u32(17), EF::from_u32(19), EF::from_u32(23)];
        for seed in 0..16 {
            let mut rng = StdRng::seed_from_u64(seed);
            let masks = sample_inner_masks::<EF, _>(num_rounds, &mut rng);
            for matrix in 0..3 {
                let shift = (0..num_rounds)
                    .map(|round| {
                        masks[matrix * num_rounds + round]
                            .iter()
                            .copied()
                            .horner::<EF, EF>(point[round])
                    })
                    .sum::<EF>();
                assert_ne!(unmasked[matrix] + shift, unmasked[matrix]);
            }
        }
    }

    #[test]
    fn recover_witness_eval_rejects_non_invertible_denominator() {
        let result = recover_witness_eval(
            QuarticBinExtension::ONE,
            QuarticBinExtension::ONE,
            QuarticBinExtension::ONE,
        );
        assert_eq!(result, Err(crate::SpartanWhirError::NonInvertibleElement));
    }

    #[test]
    fn recover_witness_eval_matches_formula() {
        let r0 = QuarticBinExtension::from(F::from_u32(5));
        let eval_z = QuarticBinExtension::from(F::from_u32(17));
        let eval_x = QuarticBinExtension::from(F::from_u32(3));
        let got = recover_witness_eval(r0, eval_z, eval_x).unwrap();

        let recomposed = (QuarticBinExtension::ONE - r0) * got + r0 * eval_x;
        assert_eq!(recomposed, eval_z);
    }

    #[test]
    fn evaluate_public_half_matches_full_table() {
        let public_inputs = vec![F::from_u32(3), F::ZERO, F::from_u32(5), F::from_u32(7)];
        let point = vec![
            QuarticBinExtension::from(F::from_u32(2)),
            QuarticBinExtension::from(F::from_u32(4)),
            QuarticBinExtension::from(F::from_u32(6)),
        ];
        let full_table = build_public_half(8, &public_inputs)
            .into_iter()
            .map(QuarticBinExtension::from)
            .collect::<Vec<_>>();
        let expected = crate::evaluate_mle_table(&full_table, &point).unwrap();

        let got = evaluate_public_half(8, &public_inputs, &point).unwrap();

        assert_eq!(got, expected);
    }

    #[test]
    fn build_z_full_appends_public_half() {
        let witness = vec![F::from_u32(11), F::from_u32(13), F::ZERO, F::from_u32(17)];
        let public_inputs = vec![F::from_u32(3), F::from_u32(5)];

        let got = build_z_full(witness.clone(), 4, &public_inputs);

        let mut expected = witness;
        expected.extend(build_public_half(4, &public_inputs));
        assert_eq!(got, expected);
    }

    #[test]
    fn matrix_z_slice_matches_explicit_matrix_z() {
        let witness = vec![F::from_u32(11), F::from_u32(13), F::ZERO, F::from_u32(17)];
        let public_inputs = vec![F::from_u32(3), F::from_u32(5)];
        let z_full = build_z_full(witness.clone(), 4, &public_inputs);

        let got = matrix_z_slice(&z_full, 4, public_inputs.len()).unwrap();
        let expected = build_matrix_z(&witness, &public_inputs);

        assert_eq!(got, expected.as_slice());
    }

    #[test]
    fn evaluate_public_half_sparse_matches_full_table() {
        let public_inputs = vec![F::from_u32(3), F::ZERO];
        let point = vec![
            QuarticBinExtension::from(F::from_u32(2)),
            QuarticBinExtension::from(F::from_u32(4)),
            QuarticBinExtension::from(F::from_u32(6)),
            QuarticBinExtension::from(F::from_u32(8)),
        ];
        let full_table = build_public_half(16, &public_inputs)
            .into_iter()
            .map(QuarticBinExtension::from)
            .collect::<Vec<_>>();
        let expected = crate::evaluate_mle_table(&full_table, &point).unwrap();

        let got = evaluate_public_half(16, &public_inputs, &point).unwrap();

        assert_eq!(got, expected);
    }

    #[test]
    fn evaluate_public_half_rejects_invalid_inputs() {
        let point = vec![QuarticBinExtension::from(F::from_u32(2))];
        assert_eq!(
            evaluate_public_half(3, &[], &point),
            Err(crate::SpartanWhirError::InvalidRoundPolynomial)
        );
        assert_eq!(
            evaluate_public_half(2, &[F::ONE, F::ONE], &point),
            Err(crate::SpartanWhirError::InvalidPublicInputLength)
        );
    }
}
