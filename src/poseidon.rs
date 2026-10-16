use alloc::vec::Vec;
use core::marker::PhantomData;
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_whir::pcs::zk::MaskGroupProverData;
use rand::{
    distr::{Distribution, StandardUniform},
    CryptoRng, Rng,
};
use serde::{Deserialize, Serialize};

use crate::{
    engine::{
        poseidon_challenger, ExtField, Poseidon1Engine, PoseidonChallenger, PoseidonEngine, F,
    },
    plonky3_whir_pcs::{
        build_poseidon_full_zk_pcs, FullZkPoseidonEngine, FullZkPoseidonPcs,
        PoseidonZkCommitmentFor, PoseidonZkMmcsFor,
    },
    preprocess_spark_tables,
    protocol::{
        combined_application_mask_shape, setup_spark_fixed_commitments,
        spark_pcs_configs_for_tables, validate_canonical_verifying_shape,
        validate_spark_table_metadata, PoseidonZkSpartanProtocolFor, SparkFixedProverData,
        ZkSpartanProofFor,
    },
    r1cs::{DirectBindLayout, DirectMultiplyLayout},
    security::{
        derive_direct_component_security, derive_spark_component_security, SpartanSoundnessMode,
    },
    DomainSeparator, MatrixClosingMode, MlePcs, Plonky3WhirPcs, R1csInstance, R1csShape,
    R1csWitness, SecurityConfig, SparkFixedCommitments, SparkPcsConfigs, SparkTableMetadata,
    SparkTables, SparkWhirParams, SpartanProofKind, SpartanProtocol, SpartanSnarkConfig,
    SpartanWhirError, WhirParams, ZkWhirPcsConfig,
};

fn full_zk_protocol_id<E: FullZkPoseidonEngine>(
    batching: crate::pcs_config::FreshMaskBatching,
    packing: crate::pcs_config::MaskPacking,
) -> Vec<u8>
where
    E::EF: ExtField,
    E::Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let mut id = E::FULL_ZK_PROTOCOL_ID.to_vec();
    if batching == crate::pcs_config::FreshMaskBatching::SameHeight {
        id.extend_from_slice(b"-fresh-mask-batching-v1");
    }
    if packing != crate::pcs_config::MaskPacking::Off {
        id.extend_from_slice(b"-mask-packing-v1-");
        id.extend_from_slice(packing.label().as_bytes());
    }
    id
}

pub type PoseidonSetupConfig = SpartanSnarkConfig;
pub type PoseidonProofKind<Ext> = SpartanProofKind<PoseidonEngine<Ext>, Plonky3WhirPcs>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoseidonZkSetupConfig {
    pub matrix_closing: MatrixClosingMode,
    pub security: SecurityConfig,
    pub whir_params: WhirParams,
    #[serde(default)]
    pub spark_whir_params: Option<SparkWhirParams>,
    pub ell_zk: usize,
    pub mask_log_inv_rate: usize,
}

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "<Plonky3WhirPcs as MlePcs<E>>::ProverData: Serialize",
    deserialize = "<Plonky3WhirPcs as MlePcs<E>>::ProverData: Deserialize<'de>"
))]
pub struct PoseidonZkProvingKeyFor<E>
where
    E: FullZkPoseidonEngine,
    E::EF: ExtField,
    E::Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    Plonky3WhirPcs: FullZkPoseidonPcs<E>,
{
    pub matrix_closing: MatrixClosingMode,
    pub(crate) shape_canonical: R1csShape<F>,
    pub(crate) num_cons_unpadded: usize,
    pub(crate) num_vars_unpadded: usize,
    pub(crate) num_io: usize,
    pub(crate) security: SecurityConfig,
    pub(crate) whir_params: WhirParams,
    pub(crate) pcs_config: ZkWhirPcsConfig,
    pub(crate) spark_fixed_commitments: Option<SparkFixedCommitments<PoseidonZkCommitmentFor<E>>>,
    pub(crate) spark_pcs_configs: Option<SparkPcsConfigs>,
    pub(crate) spark_fixed_prover_data: Option<SparkFixedProverData<E, Plonky3WhirPcs>>,
    #[serde(default)]
    pub(crate) spark_tables: Option<SparkTables>,
    pub(crate) domain_separator: DomainSeparator,
    #[serde(skip)]
    pub(crate) direct_bind_layout: Option<DirectBindLayout<F>>,
    #[serde(skip)]
    pub(crate) direct_multiply_layout: Option<DirectMultiplyLayout>,
    marker: PhantomData<E>,
}

#[derive(Serialize, Deserialize)]
#[serde(bound(serialize = "", deserialize = ""))]
pub struct PoseidonZkVerifyingKeyFor<E>
where
    E: FullZkPoseidonEngine,
    E::EF: ExtField,
    E::Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    Plonky3WhirPcs: FullZkPoseidonPcs<E>,
{
    pub matrix_closing: MatrixClosingMode,
    pub(crate) shape_canonical: R1csShape<F>,
    pub(crate) num_cons_unpadded: usize,
    pub(crate) num_vars_unpadded: usize,
    pub(crate) num_io: usize,
    pub(crate) security: SecurityConfig,
    pub(crate) whir_params: WhirParams,
    pub(crate) pcs_config: ZkWhirPcsConfig,
    /// SPARK fixed-table Merkle roots produced at setup.
    pub(crate) spark_fixed_commitments: Option<SparkFixedCommitments<PoseidonZkCommitmentFor<E>>>,
    pub(crate) spark_pcs_configs: Option<SparkPcsConfigs>,
    #[serde(default)]
    pub(crate) spark_table_metadata: Option<SparkTableMetadata>,
    pub(crate) domain_separator: DomainSeparator,
    /// Setup authenticates the relation digest and, in SPARK mode, the fixed
    /// commitments. Serialization omits the marker, so every restored key must
    /// be authenticated once before verification.
    #[serde(skip)]
    pub(crate) verifying_key_authenticated: bool,
    marker: PhantomData<E>,
}

pub type PoseidonZkProvingKey<Ext> = PoseidonZkProvingKeyFor<PoseidonEngine<Ext>>;
pub type PoseidonZkVerifyingKey<Ext> = PoseidonZkVerifyingKeyFor<PoseidonEngine<Ext>>;
pub type Poseidon1ZkProvingKey<Ext> = PoseidonZkProvingKeyFor<Poseidon1Engine<Ext>>;
pub type Poseidon1ZkVerifyingKey<Ext> = PoseidonZkVerifyingKeyFor<Poseidon1Engine<Ext>>;

impl<E> PoseidonZkProvingKeyFor<E>
where
    E: FullZkPoseidonEngine,
    E::EF: ExtField,
    E::Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    Plonky3WhirPcs: FullZkPoseidonPcs<E>,
{
    pub fn domain_separator(&self) -> &DomainSeparator {
        &self.domain_separator
    }

    pub fn prepare_for_proving(&mut self) -> Result<(), SpartanWhirError> {
        self.direct_bind_layout = Some(self.shape_canonical.direct_bind_layout()?);
        self.direct_multiply_layout = Some(self.shape_canonical.direct_multiply_layout()?);
        match self.matrix_closing {
            MatrixClosingMode::DirectSparse => {
                if self.spark_fixed_commitments.is_some()
                    || self.spark_pcs_configs.is_some()
                    || self.spark_fixed_prover_data.is_some()
                    || self.spark_tables.is_some()
                {
                    return Err(SpartanWhirError::invalid_config());
                }
            }
            MatrixClosingMode::Spark => {
                if self.spark_fixed_commitments.is_none()
                    || self.spark_pcs_configs.is_none()
                    || self.spark_fixed_prover_data.is_none()
                    || self.spark_tables.is_none()
                {
                    return Err(SpartanWhirError::invalid_config());
                }
            }
        }
        Ok(())
    }
}

impl<E> PoseidonZkVerifyingKeyFor<E>
where
    E: FullZkPoseidonEngine,
    E::EF: ExtField,
    E::Challenger: CanObserve<PoseidonZkCommitmentFor<E>>
        + FieldChallenger<F>
        + GrindingChallenger<Witness = F>,
    StandardUniform: Distribution<E::EF>,
    Plonky3WhirPcs: FullZkPoseidonPcs<E>,
{
    pub fn domain_separator(&self) -> &DomainSeparator {
        &self.domain_separator
    }

    /// Authenticate a setup-created or restored key against its embedded R1CS.
    ///
    /// This recomputes the canonical relation digest once. In SPARK mode it
    /// also rebuilds and checks the fixed-table commitments, which costs about
    /// as much as the SPARK part of `setup`. Verification then reuses the
    /// authenticated digest without scanning the matrices again.
    pub fn authenticate(&mut self) -> Result<(), SpartanWhirError> {
        self.verifying_key_authenticated = false;
        self.validate()?;
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
                crate::protocol::authenticate_spark_fixed_commitments::<E, E::EF, Plonky3WhirPcs>(
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
    pub fn authenticate_spark_fixed_commitments(&mut self) -> Result<(), SpartanWhirError> {
        self.authenticate()
    }

    pub(crate) fn ensure_authenticated(&self) -> Result<(), SpartanWhirError> {
        if !self.verifying_key_authenticated {
            return Err(SpartanWhirError::invalid_config_reason(
                crate::InvalidConfigReason::UnauthenticatedVerifyingKey,
            ));
        }
        Ok(())
    }

    /// Validate structure and configuration using the stored relation digest.
    /// This does not recompute the relation digest; use [`Self::authenticate`]
    /// before accepting restored key material.
    pub(crate) fn validate(&self) -> Result<Option<SparkTableMetadata>, SpartanWhirError> {
        validate_canonical_verifying_shape(
            &self.shape_canonical,
            self.num_cons_unpadded,
            self.num_vars_unpadded,
            self.num_io,
        )?;
        self.security.validate()?;
        self.pcs_config.validate()?;

        let num_outer_rounds = self.shape_canonical.num_cons.ilog2() as usize;
        if num_outer_rounds == 0
            || self.pcs_config.base.num_variables != self.shape_canonical.num_vars.ilog2() as usize
            || self.pcs_config.base.whir != self.whir_params
        {
            return Err(SpartanWhirError::invalid_config());
        }
        let expected_domain = DomainSeparator::new_with_protocol_id_and_relation_digest(
            &full_zk_protocol_id::<E>(
                self.pcs_config.fresh_mask_batching,
                self.pcs_config.mask_packing,
            ),
            &self.shape_canonical,
            self.domain_separator.relation_digest,
            &self.security,
            &self.whir_params,
            self.matrix_closing,
            self.domain_separator.spark_whir_params.clone(),
        );
        if self.domain_separator != expected_domain {
            return Err(SpartanWhirError::invalid_config());
        }

        let spark_metadata = match self.matrix_closing {
            MatrixClosingMode::DirectSparse => {
                let (expected_security, _) = derive_direct_component_security::<E::EF>(
                    &self.security,
                    &self.pcs_config.base,
                    num_outer_rounds,
                    self.shape_canonical.num_vars.ilog2() as usize + 1,
                    SpartanSoundnessMode::FullZk {
                        inner_degree: self.pcs_config.ell_zk.saturating_sub(1).max(2),
                    },
                )?;
                if self.pcs_config.base.security != expected_security
                    || self.spark_fixed_commitments.is_some()
                    || self.spark_pcs_configs.is_some()
                    || self.spark_table_metadata.is_some()
                {
                    return Err(SpartanWhirError::invalid_config());
                }
                None
            }
            MatrixClosingMode::Spark => {
                let configs = self
                    .spark_pcs_configs
                    .as_ref()
                    .ok_or_else(SpartanWhirError::invalid_config)?;
                configs.fixed_value.validate()?;
                configs.fixed_audit.validate()?;
                configs
                    .read
                    .iter()
                    .try_for_each(|config| config.validate())?;
                if self.spark_fixed_commitments.is_none() {
                    return Err(SpartanWhirError::invalid_config());
                }
                match &self.domain_separator.spark_whir_params {
                    Some(params)
                        if configs.fixed_value.whir == params.fixed_value
                            && configs.fixed_audit.whir == params.fixed_audit
                            && configs.read.iter().all(|config| config.whir == params.read) => {}
                    None if configs.fixed_value.whir == self.whir_params
                        && configs.fixed_audit.whir == self.whir_params
                        && configs
                            .read
                            .iter()
                            .all(|config| config.whir == self.whir_params) => {}
                    _ => return Err(SpartanWhirError::invalid_config()),
                }
                let metadata = self
                    .spark_table_metadata
                    .ok_or_else(SpartanWhirError::invalid_config)?;
                validate_spark_table_metadata::<E::EF>(&self.shape_canonical, &metadata, configs)?;
                let (expected_security, _) = derive_spark_component_security::<E::EF>(
                    &self.security,
                    &metadata,
                    &self.pcs_config.base,
                    configs,
                    num_outer_rounds,
                    self.shape_canonical.num_vars.ilog2() as usize + 1,
                    SpartanSoundnessMode::FullZk {
                        inner_degree: self.pcs_config.ell_zk.saturating_sub(1).max(2),
                    },
                )?;
                if self.pcs_config.base.security != expected_security
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
}

/// Poseidon Spartan proof plus its public instance.
///
/// Serialize this value with any serde-compatible encoding chosen by the
/// deployment or benchmark layer. This is the no-ZK Spartan IOP with plain
/// WHIR. Use [`PoseidonZkProof`] when the application requires zero knowledge.
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Ext: ExtField, R1csInstance<F, <Plonky3WhirPcs as MlePcs<PoseidonEngine<Ext>>>::Commitment>: Serialize, PoseidonProofKind<Ext>: Serialize",
    deserialize = "Ext: ExtField, R1csInstance<F, <Plonky3WhirPcs as MlePcs<PoseidonEngine<Ext>>>::Commitment>: Deserialize<'de>, PoseidonProofKind<Ext>: Deserialize<'de>"
))]
pub struct PoseidonProof<Ext: ExtField>
where
    StandardUniform: Distribution<Ext>,
{
    pub instance: R1csInstance<F, <Plonky3WhirPcs as MlePcs<PoseidonEngine<Ext>>>::Commitment>,
    pub proof: PoseidonProofKind<Ext>,
}

/// Full witness-hiding Spartan-WHIR proof plus its public instance.
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "E::EF: ExtField, ZkSpartanProofFor<E>: Serialize",
    deserialize = "E::EF: ExtField, ZkSpartanProofFor<E>: Deserialize<'de>"
))]
pub struct PoseidonZkProofFor<E>
where
    E: FullZkPoseidonEngine,
    E::EF: ExtField,
    E::Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    StandardUniform: Distribution<E::EF>,
    Plonky3WhirPcs: FullZkPoseidonPcs<E>,
{
    pub instance: R1csInstance<F, PoseidonZkCommitmentFor<E>>,
    pub proof: ZkSpartanProofFor<E>,
}

pub type PoseidonZkProof<Ext> = PoseidonZkProofFor<PoseidonEngine<Ext>>;
pub type Poseidon1ZkProof<Ext> = PoseidonZkProofFor<Poseidon1Engine<Ext>>;

impl<E> PoseidonZkProofFor<E>
where
    E: FullZkPoseidonEngine,
    E::EF: ExtField,
    E::Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    StandardUniform: Distribution<E::EF>,
    Plonky3WhirPcs: FullZkPoseidonPcs<E>,
{
    pub fn closing_mode(&self) -> MatrixClosingMode {
        self.proof.matrix_closing.mode()
    }
}

impl<Ext: ExtField> PoseidonProof<Ext>
where
    StandardUniform: Distribution<Ext>,
{
    pub fn new(
        instance: R1csInstance<F, <Plonky3WhirPcs as MlePcs<PoseidonEngine<Ext>>>::Commitment>,
        proof: PoseidonProofKind<Ext>,
    ) -> Self {
        Self { instance, proof }
    }

    pub fn closing_mode(&self) -> MatrixClosingMode {
        self.proof.kind()
    }

    #[deprecated(since = "0.1.0", note = "use closing_mode")]
    pub fn kind(&self) -> MatrixClosingMode {
        self.closing_mode()
    }
}

pub fn setup_poseidon<Ext>(
    shape: R1csShape<F>,
    config: PoseidonSetupConfig,
) -> Result<
    (
        crate::PoseidonProvingKey<Ext>,
        crate::PoseidonVerifyingKey<Ext>,
    ),
    SpartanWhirError,
>
where
    Ext: ExtField,
    StandardUniform: Distribution<Ext>,
    PoseidonChallenger: CanObserve<<Plonky3WhirPcs as MlePcs<PoseidonEngine<Ext>>>::Commitment>
        + CanSampleUniformBits<F>
        + FieldChallenger<F>
        + GrindingChallenger<Witness = F>,
{
    SpartanProtocol::<PoseidonEngine<Ext>, Plonky3WhirPcs>::setup_with_config(&shape, &config)
}

pub fn setup_poseidon_zk_for<E>(
    shape: R1csShape<F>,
    config: PoseidonZkSetupConfig,
) -> Result<(PoseidonZkProvingKeyFor<E>, PoseidonZkVerifyingKeyFor<E>), SpartanWhirError>
where
    E: FullZkPoseidonEngine,
    E::EF: ExtField,
    E::Challenger: CanObserve<PoseidonZkCommitmentFor<E>>
        + FieldChallenger<F>
        + GrindingChallenger<Witness = F>,
    StandardUniform: Distribution<E::EF>,
    Plonky3WhirPcs: FullZkPoseidonPcs<E>,
{
    setup_poseidon_zk_with_fresh_mask_batching::<E>(
        shape,
        config,
        crate::pcs_config::FreshMaskBatching::Separate,
    )
}

pub fn setup_poseidon_zk_with_fresh_mask_batching<E>(
    shape: R1csShape<F>,
    config: PoseidonZkSetupConfig,
    fresh_mask_batching: crate::pcs_config::FreshMaskBatching,
) -> Result<(PoseidonZkProvingKeyFor<E>, PoseidonZkVerifyingKeyFor<E>), SpartanWhirError>
where
    E: FullZkPoseidonEngine,
    E::EF: ExtField,
    E::Challenger: CanObserve<PoseidonZkCommitmentFor<E>>
        + FieldChallenger<F>
        + GrindingChallenger<Witness = F>,
    StandardUniform: Distribution<E::EF>,
    Plonky3WhirPcs: FullZkPoseidonPcs<E>,
{
    setup_poseidon_zk_with_mask_packing::<E>(
        shape,
        config,
        fresh_mask_batching,
        crate::pcs_config::MaskPacking::Off,
    )
}

pub fn setup_poseidon_zk_with_mask_packing<E>(
    shape: R1csShape<F>,
    config: PoseidonZkSetupConfig,
    fresh_mask_batching: crate::pcs_config::FreshMaskBatching,
    mask_packing: crate::pcs_config::MaskPacking,
) -> Result<(PoseidonZkProvingKeyFor<E>, PoseidonZkVerifyingKeyFor<E>), SpartanWhirError>
where
    E: FullZkPoseidonEngine,
    E::EF: ExtField,
    E::Challenger: CanObserve<PoseidonZkCommitmentFor<E>>
        + FieldChallenger<F>
        + GrindingChallenger<Witness = F>,
    StandardUniform: Distribution<E::EF>,
    Plonky3WhirPcs: FullZkPoseidonPcs<E>,
{
    config.security.validate()?;
    shape.validate()?;

    let shape_canonical = shape.pad_regular()?;
    let num_variables = shape_canonical.num_vars.ilog2() as usize;
    let num_outer_rounds = shape_canonical.num_cons.ilog2() as usize;
    if num_outer_rounds == 0 {
        return Err(SpartanWhirError::invalid_config());
    }
    let provisional_base = crate::WhirPcsConfig {
        num_variables,
        security: config.security,
        whir: config.whir_params.clone(),
    };
    provisional_base.validate()?;
    let spark_tables = match config.matrix_closing {
        MatrixClosingMode::DirectSparse => None,
        MatrixClosingMode::Spark => Some(preprocess_spark_tables(&shape_canonical)?),
    };
    let component_security = match spark_tables.as_ref() {
        None => {
            derive_direct_component_security::<E::EF>(
                &config.security,
                &provisional_base,
                num_outer_rounds,
                num_variables + 1,
                SpartanSoundnessMode::FullZk {
                    inner_degree: config.ell_zk.saturating_sub(1).max(2),
                },
            )?
            .0
        }
        Some(tables) => {
            let provisional_spark_configs = spark_pcs_configs_for_tables::<E::EF>(
                &provisional_base,
                tables,
                config.spark_whir_params.as_ref(),
            )?;
            derive_spark_component_security::<E::EF>(
                &config.security,
                &tables.metadata(),
                &provisional_base,
                &provisional_spark_configs,
                num_outer_rounds,
                num_variables + 1,
                SpartanSoundnessMode::FullZk {
                    inner_degree: config.ell_zk.saturating_sub(1).max(2),
                },
            )?
            .0
        }
    };
    let pcs_config = ZkWhirPcsConfig {
        base: crate::WhirPcsConfig {
            security: component_security,
            ..provisional_base
        },
        ell_zk: config.ell_zk,
        mask_log_inv_rate: config.mask_log_inv_rate,
        fresh_mask_batching,
        mask_packing,
    };
    let (_, [inner_shape, outer_shape, _]) = build_poseidon_full_zk_pcs::<E>(
        &pcs_config,
        num_outer_rounds,
        num_variables + 1,
        config.security.effective_security_bits(),
    )?;
    combined_application_mask_shape(inner_shape, outer_shape, mask_packing)?;
    let transcript_spark_whir_params = match config.matrix_closing {
        MatrixClosingMode::DirectSparse => None,
        MatrixClosingMode::Spark => config.spark_whir_params.clone(),
    };
    let domain_separator = DomainSeparator::new_with_protocol_id(
        &full_zk_protocol_id::<E>(fresh_mask_batching, mask_packing),
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
            let spark_tables = spark_tables
                .as_ref()
                .ok_or_else(SpartanWhirError::invalid_config)?;
            let spark_pcs_configs = spark_pcs_configs_for_tables::<E::EF>(
                &pcs_config.base,
                spark_tables,
                config.spark_whir_params.as_ref(),
            )?;
            let setup = setup_spark_fixed_commitments::<E, E::EF, Plonky3WhirPcs>(
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
    let mut pk = PoseidonZkProvingKeyFor {
        matrix_closing: config.matrix_closing,
        shape_canonical: shape_canonical.clone(),
        num_cons_unpadded: shape.num_cons,
        num_vars_unpadded: shape.num_vars,
        num_io: shape.num_io,
        security: config.security,
        whir_params: config.whir_params.clone(),
        pcs_config: pcs_config.clone(),
        spark_fixed_commitments: spark_fixed_commitments.clone(),
        spark_pcs_configs: spark_pcs_configs.clone(),
        spark_fixed_prover_data,
        spark_tables,
        domain_separator: domain_separator.clone(),
        direct_bind_layout: None,
        direct_multiply_layout: None,
        marker: PhantomData,
    };
    pk.prepare_for_proving()?;
    let vk = PoseidonZkVerifyingKeyFor {
        matrix_closing: config.matrix_closing,
        shape_canonical,
        num_cons_unpadded: shape.num_cons,
        num_vars_unpadded: shape.num_vars,
        num_io: shape.num_io,
        security: config.security,
        whir_params: config.whir_params,
        pcs_config,
        spark_fixed_commitments,
        spark_pcs_configs,
        spark_table_metadata,
        domain_separator,
        verifying_key_authenticated: true,
        marker: PhantomData,
    };
    Ok((pk, vk))
}

pub fn setup_poseidon_zk<Ext>(
    shape: R1csShape<F>,
    config: PoseidonZkSetupConfig,
) -> Result<(PoseidonZkProvingKey<Ext>, PoseidonZkVerifyingKey<Ext>), SpartanWhirError>
where
    Ext: ExtField,
    StandardUniform: Distribution<Ext>,
{
    setup_poseidon_zk_for::<PoseidonEngine<Ext>>(shape, config)
}

pub fn setup_poseidon1_zk<Ext>(
    shape: R1csShape<F>,
    config: PoseidonZkSetupConfig,
) -> Result<(Poseidon1ZkProvingKey<Ext>, Poseidon1ZkVerifyingKey<Ext>), SpartanWhirError>
where
    Ext: ExtField,
    StandardUniform: Distribution<Ext>,
{
    setup_poseidon_zk_for::<Poseidon1Engine<Ext>>(shape, config)
}

impl<Ext> crate::PoseidonProvingKey<Ext>
where
    Ext: ExtField,
    StandardUniform: Distribution<Ext>,
    PoseidonChallenger: CanObserve<<Plonky3WhirPcs as MlePcs<PoseidonEngine<Ext>>>::Commitment>
        + CanSampleUniformBits<F>
        + FieldChallenger<F>
        + GrindingChallenger<Witness = F>,
{
    /// Set up a Poseidon Spartan proving/verifying key pair.
    ///
    /// Spark proving keys include fixed-table prover data and cached Spark
    /// preprocessing tables, so their serialized form is expected to be
    /// materially larger than direct-mode keys.
    pub fn setup(
        shape: R1csShape<F>,
        config: PoseidonSetupConfig,
    ) -> Result<
        (
            crate::PoseidonProvingKey<Ext>,
            crate::PoseidonVerifyingKey<Ext>,
        ),
        SpartanWhirError,
    > {
        setup_poseidon::<Ext>(shape, config)
    }

    pub fn prove(
        &self,
        witness: R1csWitness<F>,
        public_inputs: Vec<F>,
    ) -> Result<PoseidonProof<Ext>, SpartanWhirError> {
        let mut challenger = poseidon_challenger();
        let (instance, proof) =
            SpartanProtocol::<PoseidonEngine<Ext>, Plonky3WhirPcs>::prove_with_mode(
                self,
                &public_inputs,
                &witness,
                self.matrix_closing,
                &mut challenger,
            )?;
        Ok(PoseidonProof::new(instance, proof))
    }
}

impl<Ext> crate::PoseidonVerifyingKey<Ext>
where
    Ext: ExtField,
    StandardUniform: Distribution<Ext>,
    PoseidonChallenger: CanObserve<<Plonky3WhirPcs as MlePcs<PoseidonEngine<Ext>>>::Commitment>
        + CanSampleUniformBits<F>
        + FieldChallenger<F>
        + GrindingChallenger<Witness = F>,
{
    /// Verify a proof against public inputs selected by the verifier.
    ///
    /// The caller must obtain `expected_public_inputs` from the application or
    /// another trusted statement source, not from the party supplying `proof`.
    pub fn verify(
        &self,
        expected_public_inputs: &[F],
        proof: &PoseidonProof<Ext>,
    ) -> Result<(), SpartanWhirError> {
        self.ensure_authenticated()?;
        if expected_public_inputs.len() != self.num_io()
            || proof.instance.public_inputs.len() != self.num_io()
        {
            return Err(SpartanWhirError::InvalidPublicInputLength);
        }
        if proof.instance.public_inputs != expected_public_inputs {
            return Err(SpartanWhirError::PublicInputMismatch);
        }
        let mut challenger = poseidon_challenger();
        SpartanProtocol::<PoseidonEngine<Ext>, Plonky3WhirPcs>::verify_with_mode(
            self,
            &proof.instance,
            &proof.proof,
            &mut challenger,
        )
    }
}

impl<E> PoseidonZkProvingKeyFor<E>
where
    E: FullZkPoseidonEngine,
    E::EF: ExtField + Serialize + for<'de> Deserialize<'de>,
    E::Challenger: CanObserve<PoseidonZkCommitmentFor<E>>
        + CanSampleUniformBits<F>
        + FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + Clone
        + Send,
    StandardUniform: Distribution<E::EF> + Distribution<F>,
    Plonky3WhirPcs: FullZkPoseidonPcs<E>,
    MaskGroupProverData<F, E::EF, PoseidonZkMmcsFor<E>>: Send,
{
    pub fn setup(
        shape: R1csShape<F>,
        config: PoseidonZkSetupConfig,
    ) -> Result<(Self, PoseidonZkVerifyingKeyFor<E>), SpartanWhirError> {
        setup_poseidon_zk_for::<E>(shape, config)
    }

    /// Set up a SPARK key with an explicit fresh-mask commitment policy.
    pub fn setup_with_fresh_mask_batching(
        shape: R1csShape<F>,
        config: PoseidonZkSetupConfig,
        batching: crate::pcs_config::FreshMaskBatching,
    ) -> Result<(Self, PoseidonZkVerifyingKeyFor<E>), SpartanWhirError> {
        if config.matrix_closing != MatrixClosingMode::Spark {
            return Err(SpartanWhirError::invalid_config());
        }
        setup_poseidon_zk_with_fresh_mask_batching::<E>(shape, config, batching)
    }

    /// Set up a full-ZK key with authenticated physical mask packing.
    pub fn setup_with_mask_packing(
        shape: R1csShape<F>,
        config: PoseidonZkSetupConfig,
        batching: crate::pcs_config::FreshMaskBatching,
        packing: crate::pcs_config::MaskPacking,
    ) -> Result<(Self, PoseidonZkVerifyingKeyFor<E>), SpartanWhirError> {
        setup_poseidon_zk_with_mask_packing::<E>(shape, config, batching, packing)
    }

    pub fn prove(
        &self,
        witness: R1csWitness<F>,
        public_inputs: Vec<F>,
    ) -> Result<PoseidonZkProofFor<E>, SpartanWhirError> {
        let mut challenger = <E as crate::engine::Plonky3PoseidonEngine>::challenger();
        let (instance, proof) = PoseidonZkSpartanProtocolFor::<E>::prove(
            self,
            &public_inputs,
            &witness,
            &mut challenger,
        )?;
        Ok(PoseidonZkProofFor { instance, proof })
    }

    pub fn prove_with_rng<R>(
        &self,
        witness: R1csWitness<F>,
        public_inputs: Vec<F>,
        rng: &mut R,
    ) -> Result<PoseidonZkProofFor<E>, SpartanWhirError>
    where
        R: Rng + CryptoRng,
    {
        let mut challenger = <E as crate::engine::Plonky3PoseidonEngine>::challenger();
        let (instance, proof) = PoseidonZkSpartanProtocolFor::<E>::prove_with_rng(
            self,
            &public_inputs,
            &witness,
            &mut challenger,
            rng,
        )?;
        Ok(PoseidonZkProofFor { instance, proof })
    }
}

impl<E> PoseidonZkVerifyingKeyFor<E>
where
    E: FullZkPoseidonEngine,
    E::EF: ExtField + Serialize + for<'de> Deserialize<'de>,
    E::Challenger: CanObserve<PoseidonZkCommitmentFor<E>>
        + CanSampleUniformBits<F>
        + FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + Clone
        + Send,
    StandardUniform: Distribution<E::EF> + Distribution<F>,
    Plonky3WhirPcs: FullZkPoseidonPcs<E>,
    MaskGroupProverData<F, E::EF, PoseidonZkMmcsFor<E>>: Send,
{
    /// Verify a proof against public inputs selected by the verifier.
    ///
    /// The caller must obtain `expected_public_inputs` from the application or
    /// another trusted statement source, not from the party supplying `proof`.
    pub fn verify(
        &self,
        expected_public_inputs: &[F],
        proof: &PoseidonZkProofFor<E>,
    ) -> Result<(), SpartanWhirError> {
        self.ensure_authenticated()?;
        if expected_public_inputs.len() != self.num_io
            || proof.instance.public_inputs.len() != self.num_io
        {
            return Err(SpartanWhirError::InvalidPublicInputLength);
        }
        if proof.instance.public_inputs != expected_public_inputs {
            return Err(SpartanWhirError::PublicInputMismatch);
        }
        let mut challenger = <E as crate::engine::Plonky3PoseidonEngine>::challenger();
        PoseidonZkSpartanProtocolFor::<E>::verify(
            self,
            &proof.instance,
            &proof.proof,
            &mut challenger,
        )
    }
}

mod witness_generator {
    use alloc::{string::String, vec};
    use core::{
        ffi::{c_int, c_void},
        fmt,
        ptr::NonNull,
    };
    use p3_field::PrimeCharacteristicRing;

    use super::*;
    use crate::circom::{validate_satisfaction, CircomAdapterError, KOALABEAR_MODULUS};

    pub const LINKED_WITNESS_GENERATOR_OK: c_int = 0;

    /// Linked witness-generator circuit loader ABI.
    ///
    /// `circuit_ptr/circuit_len` is the linked Circom `.dat` payload. Return
    /// an opaque non-null circuit handle on success, or null on failure after
    /// optionally writing a UTF-8, nul-terminated error message to `error_msg`.
    pub type LinkedWitnessLoadCircuitFn = unsafe extern "C" fn(
        circuit_ptr: *const u8,
        circuit_len: usize,
        error_msg: *mut u8,
        error_msg_len: usize,
    ) -> *mut c_void;

    /// Linked witness-generator circuit release ABI.
    pub type LinkedWitnessFreeCircuitFn = unsafe extern "C" fn(circuit: *mut c_void);

    /// Linked witness-generator ABI.
    ///
    /// `circuit` is the opaque handle returned by `LinkedWitnessLoadCircuitFn`.
    /// `input_ptr/input_len` is an application-defined binary input buffer. The
    /// generator writes exactly `witness_len` private/internal witness values
    /// and `public_inputs_len` public values, all as canonical KoalaBear `u32`
    /// limbs. Public values must be ordered as Circom exposes them:
    /// `public_outputs || public_inputs`.
    ///
    /// Return `LINKED_WITNESS_GENERATOR_OK` on success. On failure, return a
    /// non-zero code and optionally write a UTF-8, nul-terminated error message
    /// into `error_msg`.
    pub type LinkedWitnessGeneratorFn = unsafe extern "C" fn(
        circuit: *mut c_void,
        input_ptr: *const u8,
        input_len: usize,
        witness_ptr: *mut u32,
        witness_len: usize,
        public_inputs_ptr: *mut u32,
        public_inputs_len: usize,
        error_msg: *mut u8,
        error_msg_len: usize,
    ) -> c_int;

    /// Linked native witness generator handle.
    pub struct PoseidonWitnessGenerator {
        name: &'static str,
        circuit: NonNull<c_void>,
        generate: LinkedWitnessGeneratorFn,
        free: LinkedWitnessFreeCircuitFn,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum PoseidonWitnessGeneratorError {
        CircuitLoadFailed {
            name: &'static str,
            message: String,
        },
        GeneratorFailed {
            name: &'static str,
            code: c_int,
            message: String,
        },
        InvalidFieldElement {
            value: u32,
        },
        Circom(CircomAdapterError),
        Protocol(SpartanWhirError),
    }

    impl fmt::Debug for PoseidonWitnessGenerator {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("PoseidonWitnessGenerator")
                .field("name", &self.name)
                .finish_non_exhaustive()
        }
    }

    impl fmt::Display for PoseidonWitnessGeneratorError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::CircuitLoadFailed { name, message } => {
                    write!(
                        f,
                        "linked witness generator {name} failed to load circuit: {message}"
                    )
                }
                Self::GeneratorFailed {
                    name,
                    code,
                    message,
                } => write!(
                    f,
                    "linked witness generator {name} failed with code {code}: {message}"
                ),
                Self::InvalidFieldElement { value } => {
                    write!(
                        f,
                        "linked witness generator returned non-canonical field element {value}"
                    )
                }
                Self::Circom(error) => write!(f, "Circom witness import failed: {error}"),
                Self::Protocol(error) => write!(f, "Poseidon proof failed: {error:?}"),
            }
        }
    }

    impl std::error::Error for PoseidonWitnessGeneratorError {}

    impl From<CircomAdapterError> for PoseidonWitnessGeneratorError {
        fn from(value: CircomAdapterError) -> Self {
            Self::Circom(value)
        }
    }

    impl From<SpartanWhirError> for PoseidonWitnessGeneratorError {
        fn from(value: SpartanWhirError) -> Self {
            Self::Protocol(value)
        }
    }

    impl PoseidonWitnessGenerator {
        /// Construct a witness generator from linked native callbacks.
        ///
        /// # Safety
        ///
        /// The callbacks must implement the documented ABI, must remain loaded
        /// until this value is dropped, and must use the returned circuit handle
        /// only through the supplied `generate` and `free` functions. `load` must
        /// not retain `circuit_data`, `generate` must respect every pointer/length
        /// pair, and `free` must accept the handle exactly once.
        pub unsafe fn linked(
            name: &'static str,
            circuit_data: &[u8],
            load: LinkedWitnessLoadCircuitFn,
            generate: LinkedWitnessGeneratorFn,
            free: LinkedWitnessFreeCircuitFn,
        ) -> Result<Self, PoseidonWitnessGeneratorError> {
            let mut error_msg = vec![0u8; 512];
            let circuit = unsafe {
                load(
                    circuit_data.as_ptr(),
                    circuit_data.len(),
                    error_msg.as_mut_ptr(),
                    error_msg.len(),
                )
            };
            let circuit = NonNull::new(circuit).ok_or_else(|| {
                PoseidonWitnessGeneratorError::CircuitLoadFailed {
                    name,
                    message: error_message(&error_msg),
                }
            })?;
            Ok(Self {
                name,
                circuit,
                generate,
                free,
            })
        }

        pub const fn name(&self) -> &'static str {
            self.name
        }

        pub fn generate_witness(
            &self,
            input: impl AsRef<[u8]>,
            num_vars: usize,
            num_io: usize,
        ) -> Result<(R1csWitness<F>, Vec<F>), PoseidonWitnessGeneratorError> {
            let input = input.as_ref();
            let mut raw_witness = vec![0u32; num_vars];
            let mut raw_public_inputs = vec![0u32; num_io];
            let mut error_msg = vec![0u8; 512];
            let code = unsafe {
                (self.generate)(
                    self.circuit.as_ptr(),
                    input.as_ptr(),
                    input.len(),
                    raw_witness.as_mut_ptr(),
                    raw_witness.len(),
                    raw_public_inputs.as_mut_ptr(),
                    raw_public_inputs.len(),
                    error_msg.as_mut_ptr(),
                    error_msg.len(),
                )
            };
            if code != LINKED_WITNESS_GENERATOR_OK {
                return Err(PoseidonWitnessGeneratorError::GeneratorFailed {
                    name: self.name,
                    code,
                    message: error_message(&error_msg),
                });
            }

            let public_inputs = raw_public_inputs
                .into_iter()
                .map(canonical_field)
                .collect::<Result<Vec<_>, _>>()?;
            let witness = R1csWitness {
                w: raw_witness
                    .into_iter()
                    .map(canonical_field)
                    .collect::<Result<Vec<_>, _>>()?,
            };
            Ok((witness, public_inputs))
        }
    }

    impl Drop for PoseidonWitnessGenerator {
        fn drop(&mut self) {
            unsafe {
                (self.free)(self.circuit.as_ptr());
            }
        }
    }

    fn canonical_field(value: u32) -> Result<F, PoseidonWitnessGeneratorError> {
        if value >= KOALABEAR_MODULUS {
            return Err(PoseidonWitnessGeneratorError::InvalidFieldElement { value });
        }
        Ok(F::from_u32(value))
    }

    fn error_message(buffer: &[u8]) -> String {
        let len = buffer
            .iter()
            .position(|&byte| byte == 0)
            .unwrap_or(buffer.len());
        String::from_utf8_lossy(&buffer[..len]).into_owned()
    }

    impl<Ext> crate::PoseidonProvingKey<Ext>
    where
        Ext: ExtField,
        StandardUniform: Distribution<Ext>,
        PoseidonChallenger: CanObserve<<Plonky3WhirPcs as MlePcs<PoseidonEngine<Ext>>>::Commitment>
            + CanSampleUniformBits<F>
            + FieldChallenger<F>
            + GrindingChallenger<Witness = F>,
    {
        fn witness_from_generator(
            &self,
            generator: &PoseidonWitnessGenerator,
            input: impl AsRef<[u8]>,
        ) -> Result<(R1csWitness<F>, Vec<F>), PoseidonWitnessGeneratorError> {
            generator.generate_witness(input, self.num_vars_unpadded, self.num_io)
        }

        /// Generate a witness and prove it without a full R1CS satisfaction check.
        ///
        /// The prover evaluates the R1CS matrices as part of the Spartan
        /// protocol. Use `prove_from_witness_generator_checked` when the caller
        /// wants an explicit R1CS satisfaction check.
        pub fn prove_from_witness_generator(
            &self,
            generator: &PoseidonWitnessGenerator,
            input: impl AsRef<[u8]>,
        ) -> Result<PoseidonProof<Ext>, PoseidonWitnessGeneratorError> {
            let (witness, public_inputs) = self.witness_from_generator(generator, input)?;
            self.prove(witness, public_inputs).map_err(Into::into)
        }

        /// Generate a witness, validate R1CS satisfaction, then prove it.
        ///
        /// This is intended for import tests and debugging native witness
        /// generators. It performs a full matrix-vector pass before proving and
        /// should not be used as the default benchmark or production prove path.
        pub fn prove_from_witness_generator_checked(
            &self,
            generator: &PoseidonWitnessGenerator,
            input: impl AsRef<[u8]>,
        ) -> Result<PoseidonProof<Ext>, PoseidonWitnessGeneratorError> {
            let (witness, public_inputs) = self.witness_from_generator(generator, input)?;
            validate_satisfaction(&self.shape_canonical, &witness, &public_inputs)?;
            self.prove(witness, public_inputs).map_err(Into::into)
        }
    }

    impl<Ext> PoseidonZkProvingKey<Ext>
    where
        Ext: ExtField + Serialize + for<'de> Deserialize<'de>,
        StandardUniform: Distribution<Ext> + Distribution<F>,
        PoseidonChallenger: CanObserve<crate::PoseidonZkCommitment>,
    {
        fn witness_from_generator(
            &self,
            generator: &PoseidonWitnessGenerator,
            input: impl AsRef<[u8]>,
        ) -> Result<(R1csWitness<F>, Vec<F>), PoseidonWitnessGeneratorError> {
            generator.generate_witness(input, self.num_vars_unpadded, self.num_io)
        }

        pub fn prove_from_witness_generator(
            &self,
            generator: &PoseidonWitnessGenerator,
            input: impl AsRef<[u8]>,
        ) -> Result<PoseidonZkProof<Ext>, PoseidonWitnessGeneratorError> {
            let (witness, public_inputs) = self.witness_from_generator(generator, input)?;
            self.prove(witness, public_inputs).map_err(Into::into)
        }

        pub fn prove_from_witness_generator_checked(
            &self,
            generator: &PoseidonWitnessGenerator,
            input: impl AsRef<[u8]>,
        ) -> Result<PoseidonZkProof<Ext>, PoseidonWitnessGeneratorError> {
            let (witness, public_inputs) = self.witness_from_generator(generator, input)?;
            validate_satisfaction(&self.shape_canonical, &witness, &public_inputs)?;
            self.prove(witness, public_inputs).map_err(Into::into)
        }
    }
}

pub use witness_generator::{
    LinkedWitnessFreeCircuitFn, LinkedWitnessGeneratorFn, LinkedWitnessLoadCircuitFn,
    PoseidonWitnessGenerator, PoseidonWitnessGeneratorError, LINKED_WITNESS_GENERATOR_OK,
};
