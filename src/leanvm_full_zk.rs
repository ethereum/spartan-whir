use alloc::{string::String, vec::Vec};
use core::fmt::{Display, Formatter};

use p3_challenger::{CanFinalizeDigest, CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{
    integers::QuotientMap, BasedVectorSpace, PrimeCharacteristicRing, PrimeField32, TwoAdicField,
};
use p3_merkle_tree::PrunedMerklePaths;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::zk::ZkSumcheckData;
use p3_symmetric::MerkleCap;
use p3_whir::pcs::{
    proof::{QueryOpenings, SharedProofOpening, WhirProof, WhirRoundProof},
    zk::{BaseCaseZkProof, BlindedMask, ZkRoundProof, ZkWhirRelationProof},
};
use rand::distr::{Distribution, StandardUniform};
use serde::Serialize;

use crate::{
    engine::{ExtField, Plonky3PoseidonEngine, F},
    plonky3_whir_pcs::{
        build_poseidon_full_zk_pcs, observe_poseidon_relation_domain_separator,
        FullZkPoseidonEngine, FullZkPoseidonPcs, Plonky3WhirPcs, PoseidonZkCommitmentFor,
        PoseidonZkMmcsFor, PoseidonZkRelationProofFor,
    },
    poseidon::PoseidonZkVerifyingKeyFor,
    protocol::{
        combined_application_mask_shape, SparkFixedOpeningProof, SparkReadGroupOpeningProof,
        SparkReadOpeningProof, ZkMatrixClosingProofFor, ZkSparkClosingProofFor, ZkSpartanProofFor,
    },
    CubicRoundPoly, MatrixClosingMode, PoseidonTranscriptTrace, R1csInstance,
    SparkBatchedMemoryProductsProof, SparkBatchedProductLayerProof, SparkBatchedProductProof,
    SparkFixedTableOpeningEvals, SparkMemoryProductClaim, SparkMemoryProductProof, SparkPcsConfigs,
    SparkTableMetadata, SpartanWhirEngine, ZkOuterSumcheckProof,
};

pub const FULL_ZK_GUEST_MAGIC: [u32; 4] = [0x4c, 0x56, 0x5a, 0x57];
pub const FULL_ZK_GUEST_VERSION: u32 = 1;
pub const FULL_ZK_GROUPED_GUEST_VERSION: u32 = 2;
pub const MAX_FULL_ZK_GUEST_WORDS: usize = 1_048_576;
pub const FULL_ZK_STATEMENT_SCHEMA_ID: &str = "sha256-digest-bits-msb-first-v1";
pub const FULL_ZK_STATEMENT_DOMAIN: &[u8] = b"leanvm-spartan-whir-statement-v1";
const DIGEST_ELEMENTS: usize = 8;
const DIRECT_SPARSE_TAG: u32 = 0;
const SPARK_TAG: u32 = 1;
const QUERY_BASE_TAG: u32 = 0;
const QUERY_EXTENSION_TAG: u32 = 1;
const HEADER_WORDS: usize = 10;

type GuestMultiProof = PrunedMerklePaths<F, DIGEST_ELEMENTS>;
type GuestQueryOpenings<Ext> = QueryOpenings<F, Ext, GuestMultiProof>;
type GuestSharedOpening<T> = SharedProofOpening<T, GuestMultiProof>;
type GuestCommitment = MerkleCap<F, [F; DIGEST_ELEMENTS]>;
type GuestPlainWhirProof<E> =
    WhirProof<F, <E as SpartanWhirEngine>::EF, <E as FullZkPoseidonEngine>::ZkMmcs>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FullZkGuestMaskCodeShape {
    pub message_len: usize,
    pub randomness_len: usize,
    pub domain_size: usize,
    pub domain_generator: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FullZkGuestMaskGroupShape {
    pub shape: FullZkGuestMaskCodeShape,
    pub width: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FullZkGuestRoundConfig {
    /// Requested count from the native WHIR security configuration.
    pub num_queries: usize,
    /// Number of distinct positions returned by the native sampler after
    /// capping the request at the folded evaluation domain.
    pub query_positions: usize,
    pub ood_samples: usize,
    pub num_variables: usize,
    pub folding_factor: usize,
    pub log_inv_rate: usize,
    pub domain_size: usize,
    pub folded_domain_generator: u32,
    pub pow_bits: usize,
    pub folding_pow_bits: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FullZkGuestSourceCode {
    pub message_len: usize,
    pub randomness_len: usize,
    pub domain_size: usize,
    pub domain_generator: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FullZkGuestMatrixEntry {
    pub row: usize,
    pub column: usize,
    pub value: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FullZkGuestMatrix {
    pub rows: usize,
    pub columns: usize,
    pub entries: Vec<FullZkGuestMatrixEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FullZkGuestCanonicalShape {
    pub constraints: usize,
    pub variables: usize,
    pub public_inputs: usize,
    pub a: FullZkGuestMatrix,
    pub b: FullZkGuestMatrix,
    pub c: FullZkGuestMatrix,
}

/// Fixed verifier values that the LeanVM compiler embeds in one full-ZK guest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FullZkGuestVerifierConfig {
    pub canonical_shape: FullZkGuestCanonicalShape,
    pub context_prefix: Vec<u32>,
    pub relation_domain_separator: Vec<u32>,
    pub num_outer_rounds: usize,
    pub num_inner_rounds: usize,
    pub ell_zk: usize,
    pub mask_log_inv_rate: usize,
    pub mask_queries: usize,
    pub starting_folding_pow_bits: usize,
    pub final_queries: usize,
    pub final_pow_bits: usize,
    pub folding_schedule: Vec<usize>,
    pub oracle_randomness: Vec<usize>,
    pub sumcheck_mask: FullZkGuestMaskCodeShape,
    pub switch_masks: Vec<FullZkGuestMaskCodeShape>,
    pub external_mask_groups: Vec<FullZkGuestMaskGroupShape>,
    pub base_case_mask_groups: Vec<FullZkGuestMaskGroupShape>,
    pub fresh_mask_batching: String,
    pub mask_packing: String,
    pub rounds: Vec<FullZkGuestRoundConfig>,
    pub final_round: FullZkGuestRoundConfig,
    pub source_code: FullZkGuestSourceCode,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FullZkSparkGuestVerifierConfig {
    pub common: FullZkGuestVerifierConfig,
    pub table_metadata: SparkTableMetadata,
    pub fixed_commitments: FullZkGuestFixedCommitments,
    pub pcs_configs: SparkPcsConfigs,
    pub fixed_value_whir: FullZkPlainWhirGuestVerifierConfig,
    pub fixed_audit_whir: FullZkPlainWhirGuestVerifierConfig,
    pub read_whir: Vec<FullZkPlainWhirGuestVerifierConfig>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FullZkGuestFixedCommitments {
    pub value: FullZkGuestCommitment,
    pub audit: Option<FullZkGuestCommitment>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FullZkGuestCommitment {
    pub cap: Vec<[u32; 8]>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FullZkPlainWhirGuestVerifierConfig {
    pub domain_separator: Vec<u32>,
    pub num_variables: usize,
    pub commitment_ood_samples: usize,
    pub starting_folding_pow_bits: usize,
    pub folding_schedule: Vec<usize>,
    pub rounds: Vec<FullZkGuestRoundConfig>,
    pub final_round: FullZkGuestRoundConfig,
    pub final_queries: usize,
    pub final_pow_bits: usize,
    pub final_sumcheck_rounds: usize,
    pub final_folding_pow_bits: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum FullZkGuestHashProfile {
    Poseidon2 = 1,
    Poseidon1 = 2,
}

impl FullZkGuestHashProfile {
    fn from_word(word: u32) -> Result<Self, FullZkGuestCodecError> {
        match word {
            1 => Ok(Self::Poseidon2),
            2 => Ok(Self::Poseidon1),
            _ => Err(FullZkGuestCodecError::InvalidTag {
                section: "hash profile",
                value: word,
            }),
        }
    }
}

fn hash_profile_from_tag(tag: u32) -> Result<FullZkGuestHashProfile, FullZkGuestCodecError> {
    FullZkGuestHashProfile::from_word(tag)
}

/// Canonical Poseidon transcript input for the fixed LeanVM application
/// statement. The byte domain and field-vector length make the two
/// variable-length parts unambiguous.
pub fn full_zk_statement_digest_preimage(public_inputs: &[F]) -> Vec<F> {
    let mut preimage = Vec::with_capacity(FULL_ZK_STATEMENT_DOMAIN.len() + public_inputs.len() + 2);
    preimage.push(F::from_usize(FULL_ZK_STATEMENT_DOMAIN.len()));
    preimage.extend(FULL_ZK_STATEMENT_DOMAIN.iter().copied().map(F::from_u8));
    preimage.push(F::from_usize(public_inputs.len()));
    preimage.extend_from_slice(public_inputs);
    preimage
}

pub const fn full_zk_statement_digest_id<E>() -> &'static str
where
    E: FullZkPoseidonEngine,
    E::EF: ExtField,
    E::Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    E::FULL_ZK_STATEMENT_DIGEST_ID
}

pub fn full_zk_statement_digest<E>(public_inputs: &[F]) -> [F; DIGEST_ELEMENTS]
where
    E: FullZkPoseidonEngine,
    E::EF: ExtField,
    E::Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanFinalizeDigest<Digest = [F; DIGEST_ELEMENTS]>,
{
    let mut challenger = <E as Plonky3PoseidonEngine>::challenger();
    challenger.observe_slice(&full_zk_statement_digest_preimage(public_inputs));
    challenger.finalize()
}

/// Derive every configuration value that a fixed full-ZK DirectSparse guest
/// must compile into its bytecode.
pub fn full_zk_direct_guest_verifier_config<E>(
    verifying_key: &PoseidonZkVerifyingKeyFor<E>,
) -> Result<FullZkGuestVerifierConfig, crate::SpartanWhirError>
where
    E: FullZkPoseidonEngine<Commitment = GuestCommitment>,
    E::EF: ExtField,
    E::Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + PoseidonTranscriptTrace
        + CanObserve<GuestCommitment>,
    StandardUniform: Distribution<E::EF>,
    Plonky3WhirPcs: FullZkPoseidonPcs<E>,
{
    verifying_key.ensure_authenticated()?;
    verifying_key.validate()?;
    if verifying_key.matrix_closing != MatrixClosingMode::DirectSparse {
        return Err(crate::SpartanWhirError::ProofKindMismatch);
    }
    full_zk_guest_verifier_config(verifying_key, true)
}

pub fn full_zk_spark_guest_verifier_config<E>(
    verifying_key: &PoseidonZkVerifyingKeyFor<E>,
) -> Result<FullZkSparkGuestVerifierConfig, crate::SpartanWhirError>
where
    E: FullZkPoseidonEngine<Commitment = GuestCommitment>,
    E::EF: ExtField,
    E::Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + PoseidonTranscriptTrace
        + CanObserve<GuestCommitment>,
    StandardUniform: Distribution<E::EF>,
    Plonky3WhirPcs: FullZkPoseidonPcs<E>,
{
    verifying_key.ensure_authenticated()?;
    verifying_key.validate()?;
    if verifying_key.matrix_closing != MatrixClosingMode::Spark {
        return Err(crate::SpartanWhirError::ProofKindMismatch);
    }
    let pcs_configs = verifying_key
        .spark_pcs_configs
        .clone()
        .ok_or_else(crate::SpartanWhirError::invalid_config)?;
    let fixed_value_whir = plain_whir_guest_verifier_config::<E>(&pcs_configs.fixed_value)?;
    let fixed_audit_whir = plain_whir_guest_verifier_config::<E>(&pcs_configs.fixed_audit)?;
    let read_whir = pcs_configs
        .read
        .iter()
        .map(plain_whir_guest_verifier_config::<E>)
        .collect::<Result<Vec<_>, _>>()?;
    let fixed_commitments = verifying_key
        .spark_fixed_commitments
        .as_ref()
        .ok_or_else(crate::SpartanWhirError::invalid_config)?;
    Ok(FullZkSparkGuestVerifierConfig {
        common: full_zk_guest_verifier_config(verifying_key, false)?,
        table_metadata: verifying_key
            .spark_table_metadata
            .ok_or_else(crate::SpartanWhirError::invalid_config)?,
        fixed_commitments: FullZkGuestFixedCommitments {
            value: canonical_guest_commitment(&fixed_commitments.value),
            audit: fixed_commitments
                .audit
                .as_ref()
                .map(canonical_guest_commitment),
        },
        pcs_configs,
        fixed_value_whir,
        fixed_audit_whir,
        read_whir,
    })
}

fn canonical_guest_commitment(commitment: &GuestCommitment) -> FullZkGuestCommitment {
    FullZkGuestCommitment {
        cap: commitment
            .roots()
            .iter()
            .map(|root| core::array::from_fn(|index| root[index].as_canonical_u32()))
            .collect(),
    }
}

fn plain_whir_guest_verifier_config<E>(
    pcs_config: &crate::WhirPcsConfig,
) -> Result<FullZkPlainWhirGuestVerifierConfig, crate::SpartanWhirError>
where
    E: FullZkPoseidonEngine<Commitment = GuestCommitment>,
    E::EF: ExtField,
    E::Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    StandardUniform: Distribution<E::EF>,
    Plonky3WhirPcs: FullZkPoseidonPcs<E>,
{
    let (config, domain_separator) = E::plain_whir_guest_config_parts(pcs_config)?;
    let final_round = config.final_round_config();
    Ok(FullZkPlainWhirGuestVerifierConfig {
        domain_separator,
        num_variables: config.num_variables,
        commitment_ood_samples: config.commitment_ood_samples,
        starting_folding_pow_bits: config.starting_folding_pow_bits,
        folding_schedule: config.folding_schedule.clone(),
        rounds: config.round_parameters.iter().map(round_config).collect(),
        final_round: round_config(&final_round),
        final_queries: config.final_queries,
        final_pow_bits: config.final_pow_bits,
        final_sumcheck_rounds: config.final_sumcheck_rounds,
        final_folding_pow_bits: config.final_folding_pow_bits,
    })
}

fn full_zk_guest_verifier_config<E>(
    verifying_key: &PoseidonZkVerifyingKeyFor<E>,
    include_matrices: bool,
) -> Result<FullZkGuestVerifierConfig, crate::SpartanWhirError>
where
    E: FullZkPoseidonEngine,
    E::EF: ExtField,
    E::Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F> + PoseidonTranscriptTrace,
    StandardUniform: Distribution<E::EF>,
    Plonky3WhirPcs: FullZkPoseidonPcs<E>,
{
    let num_outer_rounds = verifying_key.shape_canonical.num_cons.ilog2() as usize;
    let num_inner_rounds = verifying_key.shape_canonical.num_vars.ilog2() as usize + 1;
    let (pcs, [inner_shape, outer_shape, inner_sumcheck_shape]) = build_poseidon_full_zk_pcs::<E>(
        &verifying_key.pcs_config,
        num_outer_rounds,
        num_inner_rounds,
        verifying_key.security.effective_security_bits(),
    )?;
    let application_shape = combined_application_mask_shape(
        inner_shape,
        outer_shape,
        verifying_key.pcs_config.mask_packing,
    )?;
    let external_shapes = [application_shape, inner_sumcheck_shape];

    let mut context_prefix = verifying_key
        .domain_separator
        .to_bytes()
        .into_iter()
        .map(u32::from)
        .collect::<Vec<_>>();
    for value in [
        verifying_key.pcs_config.ell_zk,
        verifying_key.pcs_config.mask_log_inv_rate,
        num_outer_rounds,
        num_inner_rounds,
    ] {
        context_prefix.extend((value as u64).to_le_bytes().into_iter().map(u32::from));
    }

    let mut relation_challenger = <E as Plonky3PoseidonEngine>::challenger().with_trace();
    observe_poseidon_relation_domain_separator::<E>(
        &pcs,
        &external_shapes,
        &mut relation_challenger,
    );
    let relation_domain_separator = relation_challenger
        .transcript_trace()
        .into_iter()
        .find_map(|event| match event {
            crate::PoseidonTranscriptEvent::Observe { values } => Some(values),
            _ => None,
        })
        .ok_or_else(crate::SpartanWhirError::invalid_config)?;

    let final_round = pcs.config.final_round_config();
    let source_message_len = 1usize << final_round.num_variables;
    let source_randomness_len = pcs.config.oracle_randomness[pcs.config.n_rounds()];
    let source_domain_size = final_round.domain_size >> final_round.folding_factor;
    let source_domain_generator =
        F::two_adic_generator(source_domain_size.ilog2() as usize).as_canonical_u32();

    let mut base_case_mask_groups = external_shapes
        .iter()
        .copied()
        .map(mask_group_shape::<E::EF>)
        .collect::<Vec<_>>();
    base_case_mask_groups.extend(
        pcs.config
            .mask_groups()
            .into_iter()
            .map(mask_group_shape::<E::EF>),
    );

    Ok(FullZkGuestVerifierConfig {
        canonical_shape: FullZkGuestCanonicalShape {
            constraints: verifying_key.shape_canonical.num_cons,
            variables: verifying_key.shape_canonical.num_vars,
            public_inputs: verifying_key.shape_canonical.num_io,
            a: guest_matrix(&verifying_key.shape_canonical.a, include_matrices),
            b: guest_matrix(&verifying_key.shape_canonical.b, include_matrices),
            c: guest_matrix(&verifying_key.shape_canonical.c, include_matrices),
        },
        context_prefix,
        relation_domain_separator,
        num_outer_rounds,
        num_inner_rounds,
        ell_zk: pcs.config.zk.ell_zk,
        mask_log_inv_rate: pcs.config.zk.mask_log_inv_rate,
        mask_queries: pcs.config.mask_queries,
        starting_folding_pow_bits: pcs.config.starting_folding_pow_bits,
        final_queries: pcs.config.final_queries,
        final_pow_bits: pcs.config.final_pow_bits,
        folding_schedule: pcs.config.folding_schedule.clone(),
        oracle_randomness: pcs.config.oracle_randomness.clone(),
        sumcheck_mask: mask_code_shape::<E::EF>(pcs.config.sumcheck_mask),
        switch_masks: pcs
            .config
            .switch_masks
            .iter()
            .copied()
            .map(mask_code_shape::<E::EF>)
            .collect(),
        external_mask_groups: external_shapes
            .iter()
            .copied()
            .map(mask_group_shape::<E::EF>)
            .collect(),
        base_case_mask_groups,
        mask_packing: verifying_key.pcs_config.mask_packing.label().to_owned(),
        fresh_mask_batching: match pcs.config.fresh_mask_batching {
            crate::pcs_config::FreshMaskBatching::Separate => "separate",
            crate::pcs_config::FreshMaskBatching::SameHeight => "same_height",
        }
        .to_owned(),
        rounds: pcs
            .config
            .round_parameters
            .iter()
            .map(round_config)
            .collect(),
        final_round: round_config(&final_round),
        source_code: FullZkGuestSourceCode {
            message_len: source_message_len,
            randomness_len: source_randomness_len,
            domain_size: source_domain_size,
            domain_generator: source_domain_generator,
        },
    })
}

fn guest_matrix(matrix: &crate::SparseMatrix<F>, include_entries: bool) -> FullZkGuestMatrix {
    FullZkGuestMatrix {
        rows: matrix.num_rows,
        columns: matrix.num_cols,
        entries: if include_entries {
            matrix
                .entries
                .iter()
                .map(|entry| FullZkGuestMatrixEntry {
                    row: entry.row,
                    column: entry.col,
                    value: entry.val.as_canonical_u32(),
                })
                .collect()
        } else {
            Vec::new()
        },
    }
}

fn mask_code_shape<Ext: ExtField>(
    shape: p3_whir::pcs::zk::MaskCodeShape,
) -> FullZkGuestMaskCodeShape {
    let generator = Ext::two_adic_generator(shape.domain_size.ilog2() as usize);
    FullZkGuestMaskCodeShape {
        message_len: shape.message_len,
        randomness_len: shape.randomness_len,
        domain_size: shape.domain_size,
        domain_generator: generator
            .as_basis_coefficients_slice()
            .iter()
            .map(PrimeField32::as_canonical_u32)
            .collect(),
    }
}

fn mask_group_shape<Ext: ExtField>(
    shape: p3_whir::pcs::zk::MaskGroupShape,
) -> FullZkGuestMaskGroupShape {
    FullZkGuestMaskGroupShape {
        shape: mask_code_shape::<Ext>(shape.shape),
        width: shape.width,
    }
}

fn round_config(config: &p3_whir::parameters::RoundConfig<F>) -> FullZkGuestRoundConfig {
    let folded_domain_size = config.domain_size >> config.folding_factor;
    FullZkGuestRoundConfig {
        num_queries: config.num_queries,
        query_positions: config.num_queries.min(folded_domain_size),
        ood_samples: config.ood_samples,
        num_variables: config.num_variables,
        folding_factor: config.folding_factor,
        log_inv_rate: config.log_inv_rate,
        domain_size: config.domain_size,
        folded_domain_generator: config.folded_domain_gen.as_canonical_u32(),
        pow_bits: config.pow_bits,
        folding_pow_bits: config.folding_pow_bits,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FullZkGuestCodecError {
    InvalidHeader,
    InvalidTag {
        section: &'static str,
        value: u32,
    },
    InvalidLength {
        section: &'static str,
    },
    NonCanonicalField {
        index: usize,
        value: u32,
    },
    UnsupportedExtensionDegree {
        expected: usize,
        actual: u32,
    },
    UnsupportedMatrixClosing,
    InputTooLarge {
        actual: usize,
        maximum: usize,
    },
    UnexpectedEnd,
    TrailingWords {
        count: usize,
    },
    ProfileMismatch {
        expected: FullZkGuestHashProfile,
        actual: FullZkGuestHashProfile,
    },
    Message(String),
}

impl Display for FullZkGuestCodecError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidHeader => write!(formatter, "invalid full-ZK guest header"),
            Self::InvalidTag { section, value } => {
                write!(formatter, "invalid {section} tag {value}")
            }
            Self::InvalidLength { section } => write!(formatter, "invalid {section} length"),
            Self::NonCanonicalField { index, value } => {
                write!(
                    formatter,
                    "word {index} is not a canonical KoalaBear value: {value}"
                )
            }
            Self::UnsupportedExtensionDegree { expected, actual } => write!(
                formatter,
                "expected extension degree {expected}, got {actual}"
            ),
            Self::UnsupportedMatrixClosing => write!(formatter, "unexpected matrix-closing mode"),
            Self::InputTooLarge { actual, maximum } => {
                write!(
                    formatter,
                    "guest input has {actual} words, maximum is {maximum}"
                )
            }
            Self::UnexpectedEnd => write!(formatter, "guest input ended early"),
            Self::TrailingWords { count } => {
                write!(formatter, "guest input has {count} trailing words")
            }
            Self::ProfileMismatch { expected, actual } => {
                write!(formatter, "expected {expected:?} profile, got {actual:?}")
            }
            Self::Message(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for FullZkGuestCodecError {}

pub fn encode_full_zk_direct_guest_words<E>(
    instance: &R1csInstance<F, PoseidonZkCommitmentFor<E>>,
    proof: &ZkSpartanProofFor<E>,
) -> Result<Vec<u32>, FullZkGuestCodecError>
where
    E: FullZkPoseidonEngine<Commitment = GuestCommitment>,
    E::EF: ExtField,
    E::Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    StandardUniform: Distribution<E::EF>,
    Plonky3WhirPcs: FullZkPoseidonPcs<E>,
{
    let profile = hash_profile_from_tag(E::HASH_PROFILE_TAG)?;
    if proof.matrix_closing.mode() != MatrixClosingMode::DirectSparse {
        return Err(FullZkGuestCodecError::UnsupportedMatrixClosing);
    }
    let mut writer = WordWriter::new();
    writer.words.extend(FULL_ZK_GUEST_MAGIC);
    writer.word(
        if proof.pcs_proof.base_case.fresh_mask_openings.len()
            < proof.pcs_proof.base_case.carried_mask_openings.len()
        {
            FULL_ZK_GROUPED_GUEST_VERSION
        } else {
            FULL_ZK_GUEST_VERSION
        },
    );
    writer.word(profile as u32);
    writer.len(E::EF::DIMENSION, "extension degree")?;
    writer.word(DIRECT_SPARSE_TAG);
    let total_words_index = writer.words.len();
    writer.word(0);
    writer.len(instance.public_inputs.len(), "public inputs")?;
    for &value in &instance.public_inputs {
        writer.base(value);
    }
    writer.commitment(&instance.witness_commitment)?;
    writer.commitment(&proof.application_mask_commitment)?;
    writer.extension(proof.outer_sumcheck.mu_tilde);
    writer.len(proof.outer_sumcheck.rounds.len(), "outer rounds")?;
    for round in &proof.outer_sumcheck.rounds {
        writer.extensions(round, "outer round coefficients")?;
    }
    for value in [
        proof.outer_claims.0,
        proof.outer_claims.1,
        proof.outer_claims.2,
    ] {
        writer.extension(value);
    }
    writer.extensions(&proof.outer_mask_evals, "outer mask evaluations")?;
    writer.zk_sumcheck(&proof.inner_sumcheck)?;
    writer.commitment(&proof.inner_sumcheck_mask_commitment)?;
    writer.word(DIRECT_SPARSE_TAG);
    writer.relation_proof::<E>(&proof.pcs_proof)?;
    let total_words = writer.words.len();
    if total_words > MAX_FULL_ZK_GUEST_WORDS {
        return Err(FullZkGuestCodecError::InputTooLarge {
            actual: total_words,
            maximum: MAX_FULL_ZK_GUEST_WORDS,
        });
    }
    writer.words[total_words_index] =
        u32::try_from(total_words).map_err(|_| FullZkGuestCodecError::InvalidLength {
            section: "total words",
        })?;
    Ok(writer.words)
}

pub fn encode_full_zk_spark_guest_words<E>(
    instance: &R1csInstance<F, PoseidonZkCommitmentFor<E>>,
    proof: &ZkSpartanProofFor<E>,
) -> Result<Vec<u32>, FullZkGuestCodecError>
where
    E: FullZkPoseidonEngine<
        Commitment = GuestCommitment,
        PlainProof = WhirProof<
            F,
            <E as SpartanWhirEngine>::EF,
            <E as FullZkPoseidonEngine>::ZkMmcs,
        >,
    >,
    E::EF: ExtField,
    E::Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    StandardUniform: Distribution<E::EF>,
    Plonky3WhirPcs: FullZkPoseidonPcs<E>,
{
    let profile = hash_profile_from_tag(E::HASH_PROFILE_TAG)?;
    let ZkMatrixClosingProofFor::Spark(closing) = &proof.matrix_closing else {
        return Err(FullZkGuestCodecError::UnsupportedMatrixClosing);
    };
    let mut writer = WordWriter::new();
    writer.words.extend(FULL_ZK_GUEST_MAGIC);
    writer.word(
        if proof.pcs_proof.base_case.fresh_mask_openings.len()
            < proof.pcs_proof.base_case.carried_mask_openings.len()
        {
            FULL_ZK_GROUPED_GUEST_VERSION
        } else {
            FULL_ZK_GUEST_VERSION
        },
    );
    writer.word(profile as u32);
    writer.len(E::EF::DIMENSION, "extension degree")?;
    writer.word(SPARK_TAG);
    let total_words_index = writer.words.len();
    writer.word(0);
    writer.len(instance.public_inputs.len(), "public inputs")?;
    for &value in &instance.public_inputs {
        writer.base(value);
    }
    writer.commitment(&instance.witness_commitment)?;
    writer.commitment(&proof.application_mask_commitment)?;
    writer.extension(proof.outer_sumcheck.mu_tilde);
    writer.len(proof.outer_sumcheck.rounds.len(), "outer rounds")?;
    for round in &proof.outer_sumcheck.rounds {
        writer.extensions(round, "outer round coefficients")?;
    }
    for value in [
        proof.outer_claims.0,
        proof.outer_claims.1,
        proof.outer_claims.2,
    ] {
        writer.extension(value);
    }
    writer.extensions(&proof.outer_mask_evals, "outer mask evaluations")?;
    writer.zk_sumcheck(&proof.inner_sumcheck)?;
    writer.commitment(&proof.inner_sumcheck_mask_commitment)?;
    writer.word(SPARK_TAG);
    writer.spark_closing::<E>(closing)?;
    writer.relation_proof::<E>(&proof.pcs_proof)?;
    let total_words = writer.words.len();
    if total_words > MAX_FULL_ZK_GUEST_WORDS {
        return Err(FullZkGuestCodecError::InputTooLarge {
            actual: total_words,
            maximum: MAX_FULL_ZK_GUEST_WORDS,
        });
    }
    writer.words[total_words_index] =
        u32::try_from(total_words).map_err(|_| FullZkGuestCodecError::InvalidLength {
            section: "total words",
        })?;
    Ok(writer.words)
}

/// Pad one canonical full-ZK encoding to the fixed LeanVM witness allocation.
/// The header retains the canonical length, so the guest parses exactly one
/// proof and ignores only the allocation padding beyond that boundary.
pub fn pad_full_zk_guest_words(mut words: Vec<u32>) -> Result<Vec<u32>, FullZkGuestCodecError> {
    if words.len() < HEADER_WORDS {
        return Err(FullZkGuestCodecError::InvalidHeader);
    }
    if words.len() > MAX_FULL_ZK_GUEST_WORDS {
        return Err(FullZkGuestCodecError::InputTooLarge {
            actual: words.len(),
            maximum: MAX_FULL_ZK_GUEST_WORDS,
        });
    }
    if words[8] as usize != words.len() {
        return Err(FullZkGuestCodecError::InvalidLength {
            section: "total words",
        });
    }
    words.resize(MAX_FULL_ZK_GUEST_WORDS, 0);
    Ok(words)
}

pub fn decode_full_zk_direct_guest_words<E>(
    words: &[u32],
) -> Result<
    (
        R1csInstance<F, PoseidonZkCommitmentFor<E>>,
        ZkSpartanProofFor<E>,
    ),
    FullZkGuestCodecError,
>
where
    E: FullZkPoseidonEngine<Commitment = GuestCommitment>,
    E::EF: ExtField,
    E::Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    StandardUniform: Distribution<E::EF>,
    Plonky3WhirPcs: FullZkPoseidonPcs<E>,
{
    let expected_profile = hash_profile_from_tag(E::HASH_PROFILE_TAG)?;
    if words.len() > MAX_FULL_ZK_GUEST_WORDS {
        return Err(FullZkGuestCodecError::InputTooLarge {
            actual: words.len(),
            maximum: MAX_FULL_ZK_GUEST_WORDS,
        });
    }
    let mut reader = WordReader::new(words);
    for expected in FULL_ZK_GUEST_MAGIC {
        if reader.word()? != expected {
            return Err(FullZkGuestCodecError::InvalidHeader);
        }
    }
    let version = reader.word()?;
    if version != FULL_ZK_GUEST_VERSION && version != FULL_ZK_GROUPED_GUEST_VERSION {
        return Err(FullZkGuestCodecError::InvalidHeader);
    }
    let actual_profile = FullZkGuestHashProfile::from_word(reader.word()?)?;
    if actual_profile != expected_profile {
        return Err(FullZkGuestCodecError::ProfileMismatch {
            expected: expected_profile,
            actual: actual_profile,
        });
    }
    let extension_degree = reader.word()?;
    if extension_degree as usize != E::EF::DIMENSION {
        return Err(FullZkGuestCodecError::UnsupportedExtensionDegree {
            expected: E::EF::DIMENSION,
            actual: extension_degree,
        });
    }
    if reader.word()? != DIRECT_SPARSE_TAG {
        return Err(FullZkGuestCodecError::UnsupportedMatrixClosing);
    }
    let encoded_words = reader.length("total words")?;
    if encoded_words != words.len() || encoded_words < HEADER_WORDS {
        return Err(FullZkGuestCodecError::InvalidLength {
            section: "total words",
        });
    }
    let public_inputs = reader.base_vec("public inputs")?;
    let witness_commitment = reader.commitment()?;
    let application_mask_commitment = reader.commitment()?;
    let outer_mu_tilde = reader.extension::<E::EF>()?;
    let outer_round_count = reader.length("outer rounds")?;
    let mut outer_rounds = Vec::with_capacity(outer_round_count);
    for _ in 0..outer_round_count {
        outer_rounds.push(reader.extension_vec::<E::EF>("outer round coefficients")?);
    }
    let outer_claims = (
        reader.extension::<E::EF>()?,
        reader.extension::<E::EF>()?,
        reader.extension::<E::EF>()?,
    );
    let outer_mask_evals = reader.extension_vec::<E::EF>("outer mask evaluations")?;
    let inner_sumcheck = reader.zk_sumcheck::<E::EF>()?;
    let inner_sumcheck_mask_commitment = reader.commitment()?;
    if reader.word()? != DIRECT_SPARSE_TAG {
        return Err(FullZkGuestCodecError::UnsupportedMatrixClosing);
    }
    let pcs_proof = reader.relation_proof::<E>()?;
    reader.finish()?;
    Ok((
        R1csInstance {
            public_inputs,
            witness_commitment,
        },
        ZkSpartanProofFor {
            application_mask_commitment,
            outer_sumcheck: ZkOuterSumcheckProof {
                mu_tilde: outer_mu_tilde,
                rounds: outer_rounds,
            },
            outer_claims,
            outer_mask_evals,
            inner_sumcheck,
            inner_sumcheck_mask_commitment,
            matrix_closing: ZkMatrixClosingProofFor::DirectSparse,
            pcs_proof,
        },
    ))
}

pub fn decode_full_zk_spark_guest_words<E>(
    words: &[u32],
) -> Result<
    (
        R1csInstance<F, PoseidonZkCommitmentFor<E>>,
        ZkSpartanProofFor<E>,
    ),
    FullZkGuestCodecError,
>
where
    E: FullZkPoseidonEngine<
        Commitment = GuestCommitment,
        PlainProof = WhirProof<
            F,
            <E as SpartanWhirEngine>::EF,
            <E as FullZkPoseidonEngine>::ZkMmcs,
        >,
    >,
    E::EF: ExtField,
    E::Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    StandardUniform: Distribution<E::EF>,
    Plonky3WhirPcs: FullZkPoseidonPcs<E>,
{
    let expected_profile = hash_profile_from_tag(E::HASH_PROFILE_TAG)?;
    if words.len() > MAX_FULL_ZK_GUEST_WORDS {
        return Err(FullZkGuestCodecError::InputTooLarge {
            actual: words.len(),
            maximum: MAX_FULL_ZK_GUEST_WORDS,
        });
    }
    let mut reader = WordReader::new(words);
    for expected in FULL_ZK_GUEST_MAGIC {
        if reader.word()? != expected {
            return Err(FullZkGuestCodecError::InvalidHeader);
        }
    }
    let version = reader.word()?;
    if version != FULL_ZK_GUEST_VERSION && version != FULL_ZK_GROUPED_GUEST_VERSION {
        return Err(FullZkGuestCodecError::InvalidHeader);
    }
    let actual_profile = FullZkGuestHashProfile::from_word(reader.word()?)?;
    if actual_profile != expected_profile {
        return Err(FullZkGuestCodecError::ProfileMismatch {
            expected: expected_profile,
            actual: actual_profile,
        });
    }
    let extension_degree = reader.word()?;
    if extension_degree as usize != E::EF::DIMENSION {
        return Err(FullZkGuestCodecError::UnsupportedExtensionDegree {
            expected: E::EF::DIMENSION,
            actual: extension_degree,
        });
    }
    if reader.word()? != SPARK_TAG {
        return Err(FullZkGuestCodecError::UnsupportedMatrixClosing);
    }
    let encoded_words = reader.length("total words")?;
    if encoded_words != words.len() || encoded_words < HEADER_WORDS {
        return Err(FullZkGuestCodecError::InvalidLength {
            section: "total words",
        });
    }
    let public_inputs = reader.base_vec("public inputs")?;
    let witness_commitment = reader.commitment()?;
    let application_mask_commitment = reader.commitment()?;
    let outer_mu_tilde = reader.extension::<E::EF>()?;
    let outer_round_count = reader.length("outer rounds")?;
    let mut outer_rounds = Vec::with_capacity(outer_round_count);
    for _ in 0..outer_round_count {
        outer_rounds.push(reader.extension_vec::<E::EF>("outer round coefficients")?);
    }
    let outer_claims = (
        reader.extension::<E::EF>()?,
        reader.extension::<E::EF>()?,
        reader.extension::<E::EF>()?,
    );
    let outer_mask_evals = reader.extension_vec::<E::EF>("outer mask evaluations")?;
    let inner_sumcheck = reader.zk_sumcheck::<E::EF>()?;
    let inner_sumcheck_mask_commitment = reader.commitment()?;
    if reader.word()? != SPARK_TAG {
        return Err(FullZkGuestCodecError::UnsupportedMatrixClosing);
    }
    let spark_closing = reader.spark_closing::<E>()?;
    let pcs_proof = reader.relation_proof::<E>()?;
    reader.finish()?;
    Ok((
        R1csInstance {
            public_inputs,
            witness_commitment,
        },
        ZkSpartanProofFor {
            application_mask_commitment,
            outer_sumcheck: ZkOuterSumcheckProof {
                mu_tilde: outer_mu_tilde,
                rounds: outer_rounds,
            },
            outer_claims,
            outer_mask_evals,
            inner_sumcheck,
            inner_sumcheck_mask_commitment,
            matrix_closing: ZkMatrixClosingProofFor::Spark(spark_closing),
            pcs_proof,
        },
    ))
}

struct WordWriter {
    words: Vec<u32>,
}

impl WordWriter {
    fn new() -> Self {
        Self { words: Vec::new() }
    }

    fn word(&mut self, word: u32) {
        self.words.push(word);
    }

    fn len(&mut self, len: usize, section: &'static str) -> Result<(), FullZkGuestCodecError> {
        self.word(
            u32::try_from(len).map_err(|_| FullZkGuestCodecError::InvalidLength { section })?,
        );
        Ok(())
    }

    fn base(&mut self, value: F) {
        self.word(value.as_canonical_u32());
    }

    fn extension<Ext: ExtField>(&mut self, value: Ext) {
        for &coefficient in value.as_basis_coefficients_slice() {
            self.base(coefficient);
        }
    }

    fn bases(&mut self, values: &[F], section: &'static str) -> Result<(), FullZkGuestCodecError> {
        self.len(values.len(), section)?;
        for &value in values {
            self.base(value);
        }
        Ok(())
    }

    fn extensions<Ext: ExtField>(
        &mut self,
        values: &[Ext],
        section: &'static str,
    ) -> Result<(), FullZkGuestCodecError> {
        self.len(values.len(), section)?;
        for &value in values {
            self.extension(value);
        }
        Ok(())
    }

    fn commitment(&mut self, commitment: &GuestCommitment) -> Result<(), FullZkGuestCodecError> {
        self.len(commitment.num_roots(), "commitment roots")?;
        for root in commitment.roots() {
            for &value in root {
                self.base(value);
            }
        }
        Ok(())
    }

    fn multi_proof(&mut self, proof: &GuestMultiProof) -> Result<(), FullZkGuestCodecError> {
        self.len(proof.sibling_hashes.len(), "Merkle siblings")?;
        for digest in &proof.sibling_hashes {
            for &value in digest {
                self.base(value);
            }
        }
        Ok(())
    }

    fn shared_base_opening(
        &mut self,
        opening: &GuestSharedOpening<F>,
    ) -> Result<(), FullZkGuestCodecError> {
        self.len(opening.rows.len(), "base opening rows")?;
        for row in &opening.rows {
            self.bases(row, "base opening row")?;
        }
        self.multi_proof(&opening.proof)
    }

    fn shared_extension_opening<Ext: ExtField>(
        &mut self,
        opening: &GuestSharedOpening<Ext>,
    ) -> Result<(), FullZkGuestCodecError> {
        self.len(opening.rows.len(), "extension opening rows")?;
        for row in &opening.rows {
            self.extensions(row, "extension opening row")?;
        }
        self.multi_proof(&opening.proof)
    }

    fn query_opening<Ext: ExtField>(
        &mut self,
        opening: &GuestQueryOpenings<Ext>,
    ) -> Result<(), FullZkGuestCodecError> {
        match opening {
            QueryOpenings::Base(opening) => {
                self.word(QUERY_BASE_TAG);
                self.shared_base_opening(opening)
            }
            QueryOpenings::Extension(opening) => {
                self.word(QUERY_EXTENSION_TAG);
                self.shared_extension_opening(opening)
            }
        }
    }

    fn optional_commitment(
        &mut self,
        commitment: Option<&GuestCommitment>,
    ) -> Result<(), FullZkGuestCodecError> {
        match commitment {
            Some(commitment) => {
                self.word(1);
                self.commitment(commitment)
            }
            None => {
                self.word(0);
                Ok(())
            }
        }
    }

    fn plain_sumcheck<Ext: ExtField>(
        &mut self,
        data: &p3_sumcheck::SumcheckData<F, Ext>,
    ) -> Result<(), FullZkGuestCodecError> {
        self.len(data.polynomial_evaluations.len(), "plain sumcheck rounds")?;
        for evaluations in &data.polynomial_evaluations {
            self.extensions(evaluations, "plain sumcheck round evaluations")?;
        }
        self.bases(
            &data.pow_witnesses,
            "plain sumcheck proof-of-work witnesses",
        )
    }

    fn plain_whir<Ext, MT>(
        &mut self,
        proof: &WhirProof<F, Ext, MT>,
    ) -> Result<(), FullZkGuestCodecError>
    where
        Ext: ExtField,
        MT: Mmcs<F, Commitment = GuestCommitment, MultiProof = GuestMultiProof>,
    {
        self.extensions(&proof.initial_ood_answers, "plain WHIR initial OOD answers")?;
        self.plain_sumcheck(&proof.initial_sumcheck)?;
        self.len(proof.rounds.len(), "plain WHIR rounds")?;
        for round in &proof.rounds {
            self.optional_commitment(round.commitment.as_ref())?;
            self.extensions(&round.ood_answers, "plain WHIR round OOD answers")?;
            self.base(round.pow_witness);
            self.query_opening(&round.openings)?;
            self.plain_sumcheck(&round.sumcheck)?;
        }
        match &proof.final_poly {
            Some(poly) => {
                self.word(1);
                self.extensions(poly.as_slice(), "plain WHIR final polynomial")?;
            }
            None => self.word(0),
        }
        self.base(proof.final_pow_witness);
        self.query_opening(&proof.final_openings)?;
        match &proof.final_sumcheck {
            Some(sumcheck) => {
                self.word(1);
                self.plain_sumcheck(sumcheck)?;
            }
            None => self.word(0),
        }
        Ok(())
    }

    fn spark_batched_product<Ext: ExtField>(
        &mut self,
        proof: &SparkBatchedProductProof<Ext>,
    ) -> Result<(), FullZkGuestCodecError> {
        self.extensions(&proof.product_roots, "SPARK product roots")?;
        self.extensions(&proof.dotproduct_claims, "SPARK dot-product claims")?;
        self.len(proof.layers.len(), "SPARK product layers")?;
        for layer in &proof.layers {
            self.len(layer.rounds.len(), "SPARK product layer rounds")?;
            for round in &layer.rounds {
                self.extensions(&round.0, "SPARK product round coefficients")?;
            }
            self.extensions(&layer.product_left_evals, "SPARK product left evaluations")?;
            self.extensions(
                &layer.product_right_evals,
                "SPARK product right evaluations",
            )?;
            self.extensions(
                &layer.dotproduct_left_evals,
                "SPARK dot-product left evaluations",
            )?;
            self.extensions(
                &layer.dotproduct_right_evals,
                "SPARK dot-product right evaluations",
            )?;
            self.extensions(
                &layer.dotproduct_weight_evals,
                "SPARK dot-product weight evaluations",
            )?;
        }
        Ok(())
    }

    fn spark_memory_products<Ext: ExtField>(
        &mut self,
        proof: &SparkBatchedMemoryProductsProof<Ext>,
    ) -> Result<(), FullZkGuestCodecError> {
        self.extension(proof.products.beta);
        self.extension(proof.products.gamma);
        for claim in [proof.products.row, proof.products.col] {
            self.extension(claim.init_root);
            self.extension(claim.read_root);
            self.extension(claim.write_root);
            self.extension(claim.audit_root);
        }
        for &evaluation in &proof.matrix_evals {
            self.extension(evaluation);
        }
        self.spark_batched_product(&proof.proof_ops)?;
        self.spark_batched_product(&proof.proof_mem)
    }

    fn spark_fixed_evals<Ext: ExtField>(&mut self, evals: &SparkFixedTableOpeningEvals<Ext>) {
        for evaluation in [
            evals.val_a_low,
            evals.val_a_high,
            evals.val_b_low,
            evals.val_b_high,
            evals.val_c_low,
            evals.val_c_high,
            evals.row_addr,
            evals.col_addr,
            evals.row_read_ts,
            evals.col_read_ts,
            evals.row_audit_ts,
            evals.col_audit_ts,
        ] {
            self.extension(evaluation);
        }
    }

    fn spark_closing<E>(
        &mut self,
        proof: &ZkSparkClosingProofFor<E>,
    ) -> Result<(), FullZkGuestCodecError>
    where
        E: FullZkPoseidonEngine<
            Commitment = GuestCommitment,
            PlainProof = WhirProof<
                F,
                <E as SpartanWhirEngine>::EF,
                <E as FullZkPoseidonEngine>::ZkMmcs,
            >,
        >,
        E::EF: ExtField,
        E::Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
        StandardUniform: Distribution<E::EF>,
        Plonky3WhirPcs: FullZkPoseidonPcs<E>,
    {
        self.spark_memory_products(&proof.spark_products)?;

        let fixed = &proof.spark_fixed_openings;
        self.len(fixed.value_num_variables, "SPARK fixed value variables")?;
        self.len(fixed.value_column_bits, "SPARK fixed value column bits")?;
        self.len(fixed.audit_num_variables, "SPARK fixed audit variables")?;
        self.len(fixed.audit_column_bits, "SPARK fixed audit column bits")?;
        self.commitment(&fixed.value_commitment)?;
        self.optional_commitment(fixed.audit_commitment.as_ref())?;
        self.spark_fixed_evals(&fixed.evals);
        self.plain_whir(&fixed.value_proof)?;
        match &fixed.audit_proof {
            Some(proof) => {
                self.word(1);
                self.plain_whir(proof)?;
            }
            None => self.word(0),
        }

        self.len(proof.spark_read_openings.groups.len(), "SPARK read groups")?;
        for group in &proof.spark_read_openings.groups {
            self.len(group.num_variables, "SPARK read variables")?;
            self.len(group.column_start, "SPARK read column start")?;
            self.len(group.column_count, "SPARK read column count")?;
            self.commitment(&group.commitment)?;
            self.len(group.evals.len(), "SPARK read opening batches")?;
            for batch in &group.evals {
                self.extensions(batch, "SPARK read opening evaluations")?;
            }
            self.plain_whir(&group.proof)?;
        }
        Ok(())
    }

    fn zk_sumcheck<Ext: ExtField>(
        &mut self,
        data: &ZkSumcheckData<F, Ext>,
    ) -> Result<(), FullZkGuestCodecError> {
        self.len(data.ell_zk, "sumcheck ell")?;
        self.extension(data.mu_tilde);
        self.len(data.round_coefficients.len(), "sumcheck rounds")?;
        for round in &data.round_coefficients {
            self.extensions(round, "sumcheck round coefficients")?;
        }
        self.bases(&data.pow_witnesses, "sumcheck proof-of-work witnesses")
    }

    fn relation_proof<E>(
        &mut self,
        proof: &PoseidonZkRelationProofFor<E>,
    ) -> Result<(), FullZkGuestCodecError>
    where
        E: FullZkPoseidonEngine<Commitment = GuestCommitment>,
        E::EF: ExtField,
        E::Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        self.len(proof.sumchecks.len(), "relation sumchecks")?;
        for sumcheck in &proof.sumchecks {
            self.zk_sumcheck(sumcheck)?;
        }
        self.len(
            proof.sumcheck_mask_commitments.len(),
            "relation sumcheck commitments",
        )?;
        for commitment in &proof.sumcheck_mask_commitments {
            self.commitment(commitment)?;
        }
        self.len(proof.rounds.len(), "relation rounds")?;
        for round in &proof.rounds {
            self.commitment(&round.commitment)?;
            self.commitment(&round.mask_commitment)?;
            self.extensions(&round.ood_answers, "relation OOD answers")?;
            self.base(round.pow_witness);
            self.query_opening(&round.openings)?;
        }
        self.base_case::<E>(&proof.base_case)
    }

    fn base_case<E>(
        &mut self,
        proof: &BaseCaseZkProof<F, E::EF, PoseidonZkMmcsFor<E>>,
    ) -> Result<(), FullZkGuestCodecError>
    where
        E: FullZkPoseidonEngine<Commitment = GuestCommitment>,
        E::EF: ExtField,
        E::Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        self.commitment(&proof.fresh_main_commitment)?;
        self.len(proof.fresh_mask_commitments.len(), "fresh mask commitments")?;
        for commitment in &proof.fresh_mask_commitments {
            self.commitment(commitment)?;
        }
        self.extension(proof.masked_claim);
        self.extensions(&proof.blinded_message, "blinded message")?;
        self.extensions(&proof.blinded_randomness, "blinded randomness")?;
        self.len(proof.blinded_masks.len(), "blinded masks")?;
        for mask in &proof.blinded_masks {
            self.extensions(&mask.message, "blinded mask message")?;
            self.extensions(&mask.randomness, "blinded mask randomness")?;
        }
        self.base(proof.pow_witness);
        self.query_opening(&proof.source_openings)?;
        self.shared_extension_opening(&proof.fresh_main_openings)?;
        if proof.carried_mask_openings.len() == proof.fresh_mask_openings.len() {
            self.len(proof.carried_mask_openings.len(), "mask openings")?;
            for (carried, fresh) in proof
                .carried_mask_openings
                .iter()
                .zip(&proof.fresh_mask_openings)
            {
                self.shared_extension_opening(carried)?;
                self.shared_extension_opening(fresh)?;
            }
        } else {
            self.len(proof.carried_mask_openings.len(), "carried mask openings")?;
            for opening in &proof.carried_mask_openings {
                self.shared_extension_opening(opening)?;
            }
            self.len(proof.fresh_mask_openings.len(), "fresh mask openings")?;
            for opening in &proof.fresh_mask_openings {
                self.shared_extension_opening(opening)?;
            }
        }
        Ok(())
    }
}

struct WordReader<'a> {
    words: &'a [u32],
    cursor: usize,
}

impl<'a> WordReader<'a> {
    fn new(words: &'a [u32]) -> Self {
        Self { words, cursor: 0 }
    }

    fn word(&mut self) -> Result<u32, FullZkGuestCodecError> {
        let value = self
            .words
            .get(self.cursor)
            .copied()
            .ok_or(FullZkGuestCodecError::UnexpectedEnd)?;
        self.cursor += 1;
        Ok(value)
    }

    fn length(&mut self, section: &'static str) -> Result<usize, FullZkGuestCodecError> {
        let value = self.word()? as usize;
        if value > self.words.len() {
            return Err(FullZkGuestCodecError::InvalidLength { section });
        }
        Ok(value)
    }

    fn base(&mut self) -> Result<F, FullZkGuestCodecError> {
        let index = self.cursor;
        let value = self.word()?;
        F::from_canonical_checked(value)
            .ok_or(FullZkGuestCodecError::NonCanonicalField { index, value })
    }

    fn extension<Ext: ExtField>(&mut self) -> Result<Ext, FullZkGuestCodecError> {
        let mut coefficients = Vec::with_capacity(Ext::DIMENSION);
        for _ in 0..Ext::DIMENSION {
            coefficients.push(self.base()?);
        }
        Ext::from_basis_coefficients_slice(&coefficients).ok_or_else(|| {
            FullZkGuestCodecError::Message("invalid extension-field coefficients".into())
        })
    }

    fn base_vec(&mut self, section: &'static str) -> Result<Vec<F>, FullZkGuestCodecError> {
        let len = self.length(section)?;
        (0..len).map(|_| self.base()).collect()
    }

    fn extension_vec<Ext: ExtField>(
        &mut self,
        section: &'static str,
    ) -> Result<Vec<Ext>, FullZkGuestCodecError> {
        let len = self.length(section)?;
        (0..len).map(|_| self.extension()).collect()
    }

    fn commitment(&mut self) -> Result<GuestCommitment, FullZkGuestCodecError> {
        let root_count = self.length("commitment roots")?;
        if !root_count.is_power_of_two() {
            return Err(FullZkGuestCodecError::InvalidLength {
                section: "commitment roots",
            });
        }
        let mut roots = Vec::with_capacity(root_count);
        for _ in 0..root_count {
            let mut root = [F::ZERO; DIGEST_ELEMENTS];
            for value in &mut root {
                *value = self.base()?;
            }
            roots.push(root);
        }
        Ok(MerkleCap::new(roots))
    }

    fn optional_commitment(&mut self) -> Result<Option<GuestCommitment>, FullZkGuestCodecError> {
        match self.word()? {
            0 => Ok(None),
            1 => Ok(Some(self.commitment()?)),
            value => Err(FullZkGuestCodecError::InvalidTag {
                section: "optional commitment",
                value,
            }),
        }
    }

    fn multi_proof(&mut self) -> Result<GuestMultiProof, FullZkGuestCodecError> {
        let sibling_count = self.length("Merkle siblings")?;
        let mut sibling_hashes = Vec::with_capacity(sibling_count);
        for _ in 0..sibling_count {
            let mut digest = [F::ZERO; DIGEST_ELEMENTS];
            for value in &mut digest {
                *value = self.base()?;
            }
            sibling_hashes.push(digest);
        }
        Ok(PrunedMerklePaths { sibling_hashes })
    }

    fn shared_base_opening(&mut self) -> Result<GuestSharedOpening<F>, FullZkGuestCodecError> {
        let row_count = self.length("base opening rows")?;
        let mut rows = Vec::with_capacity(row_count);
        for _ in 0..row_count {
            rows.push(self.base_vec("base opening row")?);
        }
        Ok(SharedProofOpening {
            rows,
            proof: self.multi_proof()?,
        })
    }

    fn shared_extension_opening<Ext: ExtField>(
        &mut self,
    ) -> Result<GuestSharedOpening<Ext>, FullZkGuestCodecError> {
        let row_count = self.length("extension opening rows")?;
        let mut rows = Vec::with_capacity(row_count);
        for _ in 0..row_count {
            rows.push(self.extension_vec("extension opening row")?);
        }
        Ok(SharedProofOpening {
            rows,
            proof: self.multi_proof()?,
        })
    }

    fn query_opening<Ext: ExtField>(
        &mut self,
    ) -> Result<GuestQueryOpenings<Ext>, FullZkGuestCodecError> {
        match self.word()? {
            QUERY_BASE_TAG => Ok(QueryOpenings::Base(self.shared_base_opening()?)),
            QUERY_EXTENSION_TAG => Ok(QueryOpenings::Extension(self.shared_extension_opening()?)),
            value => Err(FullZkGuestCodecError::InvalidTag {
                section: "query opening",
                value,
            }),
        }
    }

    fn plain_sumcheck<Ext: ExtField>(
        &mut self,
    ) -> Result<p3_sumcheck::SumcheckData<F, Ext>, FullZkGuestCodecError> {
        let round_count = self.length("plain sumcheck rounds")?;
        let mut polynomial_evaluations = Vec::with_capacity(round_count);
        for _ in 0..round_count {
            let evaluations = self.extension_vec::<Ext>("plain sumcheck round evaluations")?;
            polynomial_evaluations.push(evaluations.try_into().map_err(|_| {
                FullZkGuestCodecError::InvalidLength {
                    section: "plain sumcheck round evaluations",
                }
            })?);
        }
        Ok(p3_sumcheck::SumcheckData {
            polynomial_evaluations,
            pow_witnesses: self.base_vec("plain sumcheck proof-of-work witnesses")?,
        })
    }

    fn plain_whir<E>(&mut self) -> Result<GuestPlainWhirProof<E>, FullZkGuestCodecError>
    where
        E: FullZkPoseidonEngine<Commitment = GuestCommitment>,
        E::EF: ExtField,
        E::Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        let initial_ood_answers = self.extension_vec::<E::EF>("plain WHIR initial OOD answers")?;
        let initial_sumcheck = self.plain_sumcheck::<E::EF>()?;
        let round_count = self.length("plain WHIR rounds")?;
        let mut rounds = Vec::with_capacity(round_count);
        for _ in 0..round_count {
            rounds.push(WhirRoundProof {
                commitment: self.optional_commitment()?,
                ood_answers: self.extension_vec::<E::EF>("plain WHIR round OOD answers")?,
                pow_witness: self.base()?,
                openings: self.query_opening::<E::EF>()?,
                sumcheck: self.plain_sumcheck::<E::EF>()?,
            });
        }
        let final_poly = match self.word()? {
            0 => None,
            1 => {
                let coefficients = self.extension_vec::<E::EF>("plain WHIR final polynomial")?;
                if !coefficients.len().is_power_of_two() {
                    return Err(FullZkGuestCodecError::InvalidLength {
                        section: "plain WHIR final polynomial",
                    });
                }
                Some(Poly::new(coefficients))
            }
            value => {
                return Err(FullZkGuestCodecError::InvalidTag {
                    section: "plain WHIR final polynomial",
                    value,
                });
            }
        };
        let final_pow_witness = self.base()?;
        let final_openings = self.query_opening::<E::EF>()?;
        let final_sumcheck = match self.word()? {
            0 => None,
            1 => Some(self.plain_sumcheck::<E::EF>()?),
            value => {
                return Err(FullZkGuestCodecError::InvalidTag {
                    section: "plain WHIR final sumcheck",
                    value,
                });
            }
        };
        Ok(WhirProof {
            initial_ood_answers,
            initial_sumcheck,
            rounds,
            final_poly,
            final_pow_witness,
            final_openings,
            final_sumcheck,
        })
    }

    fn spark_batched_product<Ext: ExtField>(
        &mut self,
    ) -> Result<SparkBatchedProductProof<Ext>, FullZkGuestCodecError> {
        let product_roots = self.extension_vec::<Ext>("SPARK product roots")?;
        let dotproduct_claims = self.extension_vec::<Ext>("SPARK dot-product claims")?;
        let layer_count = self.length("SPARK product layers")?;
        let mut layers = Vec::with_capacity(layer_count);
        for _ in 0..layer_count {
            let round_count = self.length("SPARK product layer rounds")?;
            let mut rounds = Vec::with_capacity(round_count);
            for _ in 0..round_count {
                let coefficients = self.extension_vec::<Ext>("SPARK product round coefficients")?;
                rounds.push(CubicRoundPoly(coefficients.try_into().map_err(|_| {
                    FullZkGuestCodecError::InvalidLength {
                        section: "SPARK product round coefficients",
                    }
                })?));
            }
            layers.push(SparkBatchedProductLayerProof {
                rounds,
                product_left_evals: self.extension_vec::<Ext>("SPARK product left evaluations")?,
                product_right_evals: self
                    .extension_vec::<Ext>("SPARK product right evaluations")?,
                dotproduct_left_evals: self
                    .extension_vec::<Ext>("SPARK dot-product left evaluations")?,
                dotproduct_right_evals: self
                    .extension_vec::<Ext>("SPARK dot-product right evaluations")?,
                dotproduct_weight_evals: self
                    .extension_vec::<Ext>("SPARK dot-product weight evaluations")?,
            });
        }
        Ok(SparkBatchedProductProof {
            product_roots,
            dotproduct_claims,
            layers,
        })
    }

    fn spark_memory_products<Ext: ExtField>(
        &mut self,
    ) -> Result<SparkBatchedMemoryProductsProof<Ext>, FullZkGuestCodecError> {
        let beta = self.extension::<Ext>()?;
        let gamma = self.extension::<Ext>()?;
        let mut read_claim = || {
            Ok(SparkMemoryProductClaim {
                init_root: self.extension::<Ext>()?,
                read_root: self.extension::<Ext>()?,
                write_root: self.extension::<Ext>()?,
                audit_root: self.extension::<Ext>()?,
            })
        };
        let row = read_claim()?;
        let col = read_claim()?;
        let matrix_evals = [
            self.extension::<Ext>()?,
            self.extension::<Ext>()?,
            self.extension::<Ext>()?,
        ];
        Ok(SparkBatchedMemoryProductsProof {
            products: SparkMemoryProductProof {
                beta,
                gamma,
                row,
                col,
            },
            matrix_evals,
            proof_ops: self.spark_batched_product::<Ext>()?,
            proof_mem: self.spark_batched_product::<Ext>()?,
        })
    }

    fn spark_fixed_evals<Ext: ExtField>(
        &mut self,
    ) -> Result<SparkFixedTableOpeningEvals<Ext>, FullZkGuestCodecError> {
        Ok(SparkFixedTableOpeningEvals {
            val_a_low: self.extension()?,
            val_a_high: self.extension()?,
            val_b_low: self.extension()?,
            val_b_high: self.extension()?,
            val_c_low: self.extension()?,
            val_c_high: self.extension()?,
            row_addr: self.extension()?,
            col_addr: self.extension()?,
            row_read_ts: self.extension()?,
            col_read_ts: self.extension()?,
            row_audit_ts: self.extension()?,
            col_audit_ts: self.extension()?,
        })
    }

    fn spark_closing<E>(&mut self) -> Result<ZkSparkClosingProofFor<E>, FullZkGuestCodecError>
    where
        E: FullZkPoseidonEngine<
            Commitment = GuestCommitment,
            PlainProof = WhirProof<
                F,
                <E as SpartanWhirEngine>::EF,
                <E as FullZkPoseidonEngine>::ZkMmcs,
            >,
        >,
        E::EF: ExtField,
        E::Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
        StandardUniform: Distribution<E::EF>,
        Plonky3WhirPcs: FullZkPoseidonPcs<E>,
    {
        let spark_products = self.spark_memory_products::<E::EF>()?;
        let value_num_variables = self.length("SPARK fixed value variables")?;
        let value_column_bits = self.length("SPARK fixed value column bits")?;
        let audit_num_variables = self.length("SPARK fixed audit variables")?;
        let audit_column_bits = self.length("SPARK fixed audit column bits")?;
        let value_commitment = self.commitment()?;
        let audit_commitment = self.optional_commitment()?;
        let evals = self.spark_fixed_evals::<E::EF>()?;
        let value_proof = self.plain_whir::<E>()?;
        let audit_proof = match self.word()? {
            0 => None,
            1 => Some(self.plain_whir::<E>()?),
            value => {
                return Err(FullZkGuestCodecError::InvalidTag {
                    section: "SPARK fixed audit proof",
                    value,
                });
            }
        };
        let spark_fixed_openings = SparkFixedOpeningProof::from_parts(
            value_num_variables,
            value_column_bits,
            audit_num_variables,
            audit_column_bits,
            value_commitment,
            audit_commitment,
            evals,
            value_proof,
            audit_proof,
        );

        let group_count = self.length("SPARK read groups")?;
        let mut groups = Vec::with_capacity(group_count);
        for _ in 0..group_count {
            let num_variables = self.length("SPARK read variables")?;
            let column_start = self.length("SPARK read column start")?;
            let column_count = self.length("SPARK read column count")?;
            let commitment = self.commitment()?;
            let batch_count = self.length("SPARK read opening batches")?;
            let mut evals = Vec::with_capacity(batch_count);
            for _ in 0..batch_count {
                evals.push(self.extension_vec::<E::EF>("SPARK read opening evaluations")?);
            }
            let proof = self.plain_whir::<E>()?;
            groups.push(SparkReadGroupOpeningProof::from_parts(
                num_variables,
                column_start,
                column_count,
                commitment,
                evals,
                proof,
            ));
        }
        Ok(ZkSparkClosingProofFor {
            spark_products,
            spark_fixed_openings,
            spark_read_openings: SparkReadOpeningProof { groups },
        })
    }

    fn zk_sumcheck<Ext: ExtField>(
        &mut self,
    ) -> Result<ZkSumcheckData<F, Ext>, FullZkGuestCodecError> {
        let ell_zk = self.length("sumcheck ell")?;
        let mu_tilde = self.extension()?;
        let round_count = self.length("sumcheck rounds")?;
        let mut round_coefficients = Vec::with_capacity(round_count);
        for _ in 0..round_count {
            round_coefficients.push(self.extension_vec("sumcheck round coefficients")?);
        }
        let pow_witnesses = self.base_vec("sumcheck proof-of-work witnesses")?;
        Ok(ZkSumcheckData {
            mu_tilde,
            ell_zk,
            round_coefficients,
            pow_witnesses,
        })
    }

    fn relation_proof<E>(&mut self) -> Result<PoseidonZkRelationProofFor<E>, FullZkGuestCodecError>
    where
        E: FullZkPoseidonEngine<Commitment = GuestCommitment>,
        E::EF: ExtField,
        E::Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        let sumcheck_count = self.length("relation sumchecks")?;
        let mut sumchecks = Vec::with_capacity(sumcheck_count);
        for _ in 0..sumcheck_count {
            sumchecks.push(self.zk_sumcheck::<E::EF>()?);
        }
        let commitment_count = self.length("relation sumcheck commitments")?;
        let mut sumcheck_mask_commitments = Vec::with_capacity(commitment_count);
        for _ in 0..commitment_count {
            sumcheck_mask_commitments.push(self.commitment()?);
        }
        let round_count = self.length("relation rounds")?;
        let mut rounds = Vec::with_capacity(round_count);
        for _ in 0..round_count {
            rounds.push(ZkRoundProof {
                commitment: self.commitment()?,
                mask_commitment: self.commitment()?,
                ood_answers: self.extension_vec::<E::EF>("relation OOD answers")?,
                pow_witness: self.base()?,
                openings: self.query_opening::<E::EF>()?,
            });
        }
        Ok(ZkWhirRelationProof {
            sumchecks,
            sumcheck_mask_commitments,
            rounds,
            base_case: self.base_case::<E>()?,
        })
    }

    fn base_case<E>(
        &mut self,
    ) -> Result<BaseCaseZkProof<F, E::EF, PoseidonZkMmcsFor<E>>, FullZkGuestCodecError>
    where
        E: FullZkPoseidonEngine<Commitment = GuestCommitment>,
        E::EF: ExtField,
        E::Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        let fresh_main_commitment = self.commitment()?;
        let commitment_count = self.length("fresh mask commitments")?;
        let mut fresh_mask_commitments = Vec::with_capacity(commitment_count);
        for _ in 0..commitment_count {
            fresh_mask_commitments.push(self.commitment()?);
        }
        let masked_claim = self.extension::<E::EF>()?;
        let blinded_message = self.extension_vec::<E::EF>("blinded message")?;
        let blinded_randomness = self.extension_vec::<E::EF>("blinded randomness")?;
        let mask_count = self.length("blinded masks")?;
        let mut blinded_masks = Vec::with_capacity(mask_count);
        for _ in 0..mask_count {
            blinded_masks.push(BlindedMask {
                message: self.extension_vec::<E::EF>("blinded mask message")?,
                randomness: self.extension_vec::<E::EF>("blinded mask randomness")?,
            });
        }
        let pow_witness = self.base()?;
        let source_openings = self.query_opening::<E::EF>()?;
        let fresh_main_openings = self.shared_extension_opening::<E::EF>()?;
        let carried_count = self.length("mask openings")?;
        let mut carried_mask_openings = Vec::with_capacity(carried_count);
        let mut fresh_mask_openings = Vec::with_capacity(commitment_count);
        if self.words[4] == FULL_ZK_GUEST_VERSION {
            if carried_count != commitment_count {
                return Err(FullZkGuestCodecError::InvalidHeader);
            }
            for _ in 0..carried_count {
                carried_mask_openings.push(self.shared_extension_opening::<E::EF>()?);
                fresh_mask_openings.push(self.shared_extension_opening::<E::EF>()?);
            }
        } else {
            if carried_count <= commitment_count {
                return Err(FullZkGuestCodecError::InvalidHeader);
            }
            for _ in 0..carried_count {
                carried_mask_openings.push(self.shared_extension_opening::<E::EF>()?);
            }
            let fresh_count = self.length("fresh mask openings")?;
            if fresh_count != commitment_count {
                return Err(FullZkGuestCodecError::InvalidHeader);
            }
            for _ in 0..fresh_count {
                fresh_mask_openings.push(self.shared_extension_opening::<E::EF>()?);
            }
        }
        Ok(BaseCaseZkProof {
            fresh_main_commitment,
            fresh_mask_commitments,
            masked_claim,
            blinded_message,
            blinded_randomness,
            blinded_masks,
            pow_witness,
            source_openings,
            fresh_main_openings,
            carried_mask_openings,
            fresh_mask_openings,
        })
    }

    fn finish(self) -> Result<(), FullZkGuestCodecError> {
        if self.cursor == self.words.len() {
            Ok(())
        } else {
            Err(FullZkGuestCodecError::TrailingWords {
                count: self.words.len() - self.cursor,
            })
        }
    }
}

#[cfg(test)]
mod query_export_tests {
    use super::*;

    #[test]
    fn exported_query_positions_are_capped_at_folded_domain() {
        for (domain_bits, requested, expected_positions) in [
            (1, 0, 0),
            (1, 1, 1),
            (1, 2, 2),
            (1, 3, 2),
            (3, 7, 7),
            (3, 8, 8),
            (3, 9, 8),
            (15, 7, 7),
            (24, 7, 7),
        ] {
            let folded_domain = 1usize << domain_bits;
            let config = p3_whir::parameters::RoundConfig {
                num_queries: requested,
                domain_size: folded_domain << 1,
                folding_factor: 1,
                folded_domain_gen: F::two_adic_generator(domain_bits),
                num_variables: 0,
                log_inv_rate: 1,
                ood_samples: 0,
                pow_bits: 0,
                folding_pow_bits: 0,
            };
            let exported = round_config(&config);
            assert_eq!(exported.num_queries, requested);
            assert_eq!(exported.query_positions, expected_positions);
            assert!(exported.query_positions <= folded_domain);
        }
    }
}
