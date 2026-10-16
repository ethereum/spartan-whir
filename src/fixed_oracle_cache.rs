//! Authenticated public initial SPARK oracles for repeated native verification.
//!
//! The cache retains the public codeword and Merkle tree, constructed from an
//! authenticated verifying key. Cache-dependent transport omits only the initial
//! fixed openings. Reconstruction replays the existing transcript, and the
//! ordinary verifier still authenticates every reconstructed opening.

use p3_challenger::{
    CanObserve, CanSample, CanSampleUniformBits, FieldChallenger, GrindingChallenger,
};
use p3_commit::Mmcs;
use p3_field::{BasedVectorSpace, PrimeField32};
use p3_matrix::{dense::DenseMatrix, Matrix};
use p3_whir::MaskGroupProverData;
use p3_whir::{
    parameters::WhirConfig,
    pcs::proof::{QueryOpenings, SharedProofOpening, WhirProof},
};
use rand::distr::{Distribution, StandardUniform};
use sha2::{Digest, Sha256};

use crate::engine::{ExtField, F};
use crate::plonky3_whir_pcs::{FullZkPoseidonEngine, FullZkPoseidonPcs};
use crate::proof_compression::{CompressedZkProofFor, PlainProofRowAccess};
use crate::{
    MatrixClosingMode, PcsStatement, Plonky3WhirPcs, PoseidonZkVerifyingKeyFor, ProtocolPcs,
    R1csInstance, SpartanWhirEngine, SpartanWhirError, WhirPcsConfig, ZkMatrixClosingProofFor,
};

const MAGIC: &[u8; 4] = b"SPF1";
const HEADER_SIZE: usize = 36;
type Tree<E> = <<E as FullZkPoseidonEngine>::ZkMmcs as Mmcs<F>>::ProverData<DenseMatrix<F>>;
type Opening<E> =
    SharedProofOpening<F, <<E as FullZkPoseidonEngine>::ZkMmcs as Mmcs<F>>::MultiProof>;

/// Access to the retained initial Merkle tree of the supported plain PCS.
#[doc(hidden)]
pub trait FixedOracleProverData<M: Mmcs<F>> {
    fn into_initial_tree(self) -> Result<M::ProverData<DenseMatrix<F>>, SpartanWhirError>;
}

/// Access to the initial base-field opening of the supported plain proof.
#[doc(hidden)]
pub trait FixedOracleProofAccess<E: FullZkPoseidonEngine>
where
    E::EF: ExtField,
    E::Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    fn initial_opening(&self) -> Result<&Opening<E>, SpartanWhirError>;
    fn initial_opening_mut(&mut self) -> Result<&mut Opening<E>, SpartanWhirError>;
    fn initial_queries(
        &self,
        config: &WhirConfig<E::EF, F, E::Challenger>,
        challenger: &E::Challenger,
    ) -> Result<Vec<usize>, SpartanWhirError>
    where
        E::Challenger: CanSampleUniformBits<F> + CanObserve<E::Commitment>;
}

impl<E> FixedOracleProofAccess<E> for WhirProof<F, E::EF, E::ZkMmcs>
where
    E: FullZkPoseidonEngine,
    E::EF: ExtField,
    E::Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    fn initial_opening(&self) -> Result<&Opening<E>, SpartanWhirError> {
        match self
            .rounds
            .first()
            .map_or(&self.final_openings, |round| &round.openings)
        {
            QueryOpenings::Base(opening) => Ok(opening),
            _ => Err(SpartanWhirError::InvalidProofShape),
        }
    }
    fn initial_opening_mut(&mut self) -> Result<&mut Opening<E>, SpartanWhirError> {
        let opening = match self.rounds.first_mut() {
            Some(round) => &mut round.openings,
            None => &mut self.final_openings,
        };
        match opening {
            QueryOpenings::Base(opening) => Ok(opening),
            _ => Err(SpartanWhirError::InvalidProofShape),
        }
    }
    fn initial_queries(
        &self,
        config: &WhirConfig<E::EF, F, E::Challenger>,
        challenger: &E::Challenger,
    ) -> Result<Vec<usize>, SpartanWhirError>
    where
        E::Challenger: CanSampleUniformBits<F> + CanObserve<E::Commitment>,
    {
        if self.rounds.len() != config.n_rounds()
            || self.rounds.first().is_some_and(|round| {
                round.ood_answers.len() != config.round_parameters[0].ood_samples
            })
        {
            return Err(SpartanWhirError::InvalidProofShape);
        }
        let mut replay = E::challenger_for_replay(challenger);
        crate::proof_compression_plain::replay_sumcheck::<E>(
            &self.initial_sumcheck,
            config.round_folding_factor(0),
            config.starting_folding_pow_bits,
            &mut replay,
        )?;
        if let Some(round) = self.rounds.first() {
            let params = &config.round_parameters[0];
            replay.observe(
                round
                    .commitment
                    .as_ref()
                    .ok_or(SpartanWhirError::InvalidProofShape)?
                    .clone(),
            );
            for &answer in &round.ood_answers {
                let _: E::EF = replay.sample_algebra_element();
                replay.observe_algebra_element(answer);
            }
            if params.pow_bits > 0 && !replay.check_witness(params.pow_bits, round.pow_witness) {
                return Err(SpartanWhirError::WhirVerifyFailed);
            }
            let _: F = replay.sample();
            crate::proof_compression_hiding::sample_queries(
                params.domain_size >> params.folding_factor,
                params.num_queries,
                &mut replay,
            )
        } else {
            let final_poly = self
                .final_poly
                .as_ref()
                .ok_or(SpartanWhirError::InvalidProofShape)?;
            replay.observe_algebra_slice(final_poly.as_slice());
            if config.final_pow_bits > 0
                && !replay.check_witness(config.final_pow_bits, self.final_pow_witness)
            {
                return Err(SpartanWhirError::WhirVerifyFailed);
            }
            let params = config.final_round_config();
            crate::proof_compression_hiding::sample_queries(
                params.domain_size >> params.folding_factor,
                config.final_queries,
                &mut replay,
            )
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum FixedOracleKind {
    Value,
    Audit,
}

pub(crate) trait FixedOpeningRestorer<E: SpartanWhirEngine, Pcs: ProtocolPcs<E>>:
    Sync
{
    fn validate_shape(
        &self,
        config: &WhirPcsConfig,
        proof: &Pcs::Proof,
        final_rows: bool,
    ) -> Result<(), SpartanWhirError>;
    fn restore(
        &self,
        kind: FixedOracleKind,
        config: &WhirPcsConfig,
        proof: &Pcs::Proof,
        statement: &PcsStatement<E>,
        challenger: &E::Challenger,
        final_rows: bool,
    ) -> Result<Pcs::Proof, SpartanWhirError>;
}

/// Private immutable codeword and Merkle trees for one authenticated key.
///
/// There is deliberately no cache deserializer or mutable tree accessor.
pub struct FixedOracleCacheFor<E>
where
    E: FullZkPoseidonEngine,
    E::EF: ExtField,
    E::Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    fingerprint: [u8; 32],
    value: Tree<E>,
    audit: Option<Tree<E>>,
    matrix_bytes: usize,
}

/// Explicit cache-dependent transport; ordinary compact decoding rejects it.
pub struct CachedFixedOracleProofFor<E>
where
    E: FullZkPoseidonEngine,
    E::EF: ExtField,
    E::Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    StandardUniform: Distribution<E::EF>,
    Plonky3WhirPcs: FullZkPoseidonPcs<E>,
{
    fingerprint: [u8; 32],
    proof: CompressedZkProofFor<E>,
}

fn key_fingerprint<E>(vk: &PoseidonZkVerifyingKeyFor<E>) -> Result<[u8; 32], SpartanWhirError>
where
    E: FullZkPoseidonEngine,
    E::EF: ExtField,
    E::Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<E::Commitment>,
    StandardUniform: Distribution<E::EF>,
    Plonky3WhirPcs: FullZkPoseidonPcs<E>,
{
    vk.ensure_authenticated()?;
    vk.validate()?;
    if vk.matrix_closing != MatrixClosingMode::Spark {
        return Err(SpartanWhirError::ProofKindMismatch);
    }
    let identity = bincode::serialize(&(
        E::FULL_ZK_PROTOCOL_ID,
        E::HASH_PROFILE_TAG,
        F::ORDER_U32,
        E::EF::DIMENSION,
        &vk.domain_separator,
        vk.num_cons_unpadded,
        vk.num_vars_unpadded,
        vk.num_io,
        vk.security,
        &vk.whir_params,
        &vk.pcs_config,
        &vk.spark_pcs_configs,
        &vk.spark_table_metadata,
        &vk.spark_fixed_commitments,
    ))
    .map_err(|_| SpartanWhirError::InvalidProofShape)?;
    let mut digest = Sha256::new();
    digest.update(b"spartan-whir-authenticated-fixed-oracle-cache-v1");
    digest.update(identity);
    Ok(digest.finalize().into())
}

impl<E> CachedFixedOracleProofFor<E>
where
    E: FullZkPoseidonEngine,
    E::EF: ExtField,
    E::PlainProof: PlainProofRowAccess + FixedOracleProofAccess<E>,
    E::Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<E::Commitment>,
    StandardUniform: Distribution<E::EF>,
    Plonky3WhirPcs: FullZkPoseidonPcs<E>,
{
    /// Remove initial fixed openings after ordinary compact preparation.
    pub fn from_compressed(
        vk: &PoseidonZkVerifyingKeyFor<E>,
        mut proof: CompressedZkProofFor<E>,
    ) -> Result<Self, SpartanWhirError> {
        let fingerprint = key_fingerprint(vk)?;
        let ZkMatrixClosingProofFor::Spark(closing) = &mut proof.proof.matrix_closing else {
            return Err(SpartanWhirError::ProofKindMismatch);
        };
        let expected = vk
            .spark_fixed_commitments
            .as_ref()
            .ok_or(SpartanWhirError::CommitmentMismatch)?;
        if closing.spark_fixed_openings.value_commitment != expected.value
            || closing.spark_fixed_openings.audit_commitment != expected.audit
        {
            return Err(SpartanWhirError::CommitmentMismatch);
        }
        let opening = closing
            .spark_fixed_openings
            .value_proof
            .initial_opening_mut()?;
        opening.rows.clear();
        opening.proof.sibling_hashes.clear();
        if let Some(proof) = &mut closing.spark_fixed_openings.audit_proof {
            let opening = proof.initial_opening_mut()?;
            opening.rows.clear();
            opening.proof.sibling_hashes.clear();
        }
        Ok(Self { fingerprint, proof })
    }
    pub fn to_bytes(&self) -> Result<Vec<u8>, SpartanWhirError> {
        self.ensure_omitted()?;
        let payload = self.proof.to_bytes()?;
        let mut bytes = Vec::with_capacity(HEADER_SIZE + payload.len());
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&self.fingerprint);
        bytes.extend_from_slice(&payload);
        Ok(bytes)
    }
    /// Decode with the ordinary compact codec's size, sequence, and depth limits.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SpartanWhirError> {
        if bytes.get(..4) != Some(MAGIC.as_slice()) || bytes.len() < HEADER_SIZE {
            return Err(SpartanWhirError::InvalidProofShape);
        }
        let fingerprint = bytes[4..HEADER_SIZE]
            .try_into()
            .map_err(|_| SpartanWhirError::InvalidProofShape)?;
        let result = Self {
            fingerprint,
            proof: CompressedZkProofFor::from_bytes(&bytes[HEADER_SIZE..])?,
        };
        result.ensure_omitted()?;
        Ok(result)
    }
    fn ensure_omitted(&self) -> Result<(), SpartanWhirError> {
        let ZkMatrixClosingProofFor::Spark(closing) = &self.proof.proof.matrix_closing else {
            return Err(SpartanWhirError::ProofKindMismatch);
        };
        for proof in core::iter::once(&closing.spark_fixed_openings.value_proof)
            .chain(closing.spark_fixed_openings.audit_proof.iter())
        {
            let opening = proof.initial_opening()?;
            if !opening.rows.is_empty() || !opening.proof.sibling_hashes.is_empty() {
                return Err(SpartanWhirError::InvalidProofShape);
            }
        }
        Ok(())
    }
}

impl<E> FixedOracleCacheFor<E>
where
    E: FullZkPoseidonEngine,
    E::EF: ExtField,
    E::PlainProverData: FixedOracleProverData<E::ZkMmcs>,
    E::Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<E::Commitment>,
    StandardUniform: Distribution<E::EF>,
    Plonky3WhirPcs: FullZkPoseidonPcs<E>,
{
    /// Rebuild the public fixed codewords and require their roots to match the key.
    /// Restored keys must first pass `authenticate()`.
    pub fn build(vk: &PoseidonZkVerifyingKeyFor<E>) -> Result<Self, SpartanWhirError> {
        let fingerprint = key_fingerprint(vk)?;
        let configs = vk
            .spark_pcs_configs
            .as_ref()
            .ok_or(SpartanWhirError::InvalidProofShape)?;
        let tables = crate::preprocess_spark_tables(&vk.shape_canonical)?;
        let (data, commitments) =
            crate::protocol::setup_spark_fixed_commitments::<E, E::EF, Plonky3WhirPcs>(
                configs, &tables,
            )?
            .ok_or(SpartanWhirError::InvalidProofShape)?;
        if vk.spark_fixed_commitments.as_ref() != Some(&commitments) {
            return Err(SpartanWhirError::CommitmentMismatch);
        }
        let value = data.value.into_initial_tree()?;
        let audit = data
            .audit
            .map(FixedOracleProverData::into_initial_tree)
            .transpose()?;
        let mmcs = E::full_zk_mmcs();
        let mut matrix_bytes = 0usize;
        for (tree, config) in core::iter::once((&value, &configs.fixed_value))
            .chain(audit.iter().map(|tree| (tree, &configs.fixed_audit)))
        {
            let (whir, _) = E::plain_whir_guest_config_parts(config)?;
            let matrices = mmcs.get_matrices(tree);
            let width = 1usize
                .checked_shl(whir.round_folding_factor(0) as u32)
                .ok_or(SpartanWhirError::InvalidProofShape)?;
            let domain_size = whir.round_parameters.first().map_or_else(
                || whir.final_round_config().domain_size,
                |params| params.domain_size,
            );
            let height = domain_size / width;
            if matrices.len() != 1 || matrices[0].width() != width || matrices[0].height() != height
            {
                return Err(SpartanWhirError::InvalidProofShape);
            }
            matrix_bytes = matrix_bytes
                .checked_add(
                    width
                        .checked_mul(height)
                        .and_then(|count| count.checked_mul(core::mem::size_of::<F>()))
                        .ok_or(SpartanWhirError::InvalidProofShape)?,
                )
                .ok_or(SpartanWhirError::InvalidProofShape)?;
        }
        Ok(Self {
            fingerprint,
            value,
            audit,
            matrix_bytes,
        })
    }
    /// Dense codeword bytes; Merkle nodes and allocation metadata are additional.
    pub fn matrix_bytes(&self) -> usize {
        self.matrix_bytes
    }
}

impl<E> FixedOpeningRestorer<E, Plonky3WhirPcs> for FixedOracleCacheFor<E>
where
    E: FullZkPoseidonEngine,
    E::EF: ExtField,
    E::PlainProof: FixedOracleProofAccess<E>,
    Tree<E>: Sync,
    E::Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanObserve<E::Commitment>
        + CanSampleUniformBits<F>,
    StandardUniform: Distribution<E::EF>,
    Plonky3WhirPcs: FullZkPoseidonPcs<E>,
{
    fn validate_shape(
        &self,
        config: &WhirPcsConfig,
        proof: &E::PlainProof,
        final_rows: bool,
    ) -> Result<(), SpartanWhirError> {
        E::validate_cached_plain_proof_shape(config, proof, final_rows)
    }
    fn restore(
        &self,
        kind: FixedOracleKind,
        config: &WhirPcsConfig,
        proof: &E::PlainProof,
        statement: &PcsStatement<E>,
        challenger: &E::Challenger,
        final_rows: bool,
    ) -> Result<E::PlainProof, SpartanWhirError> {
        E::validate_cached_plain_proof_shape(config, proof, final_rows)?;
        let tree = match kind {
            FixedOracleKind::Value => &self.value,
            FixedOracleKind::Audit => self
                .audit
                .as_ref()
                .ok_or(SpartanWhirError::InvalidProofShape)?,
        };
        let claims =
            crate::plonky3_whir_pcs::statement_point_claims(statement, config.num_variables)?;
        let mut replay = E::challenger_for_replay(challenger);
        crate::plonky3_whir_pcs::observe_statement_point_claims(&claims, &mut replay);
        let _: E::EF = replay.sample_algebra_element();
        let (whir, _) = E::plain_whir_guest_config_parts(config)?;
        let positions = proof.initial_queries(&whir, &replay)?;
        let (values, multiproof) = E::full_zk_mmcs().open_multi_batch(&positions, tree);
        let rows = values
            .into_iter()
            .map(|mut values| {
                if values.len() == 1 {
                    Ok(values.swap_remove(0))
                } else {
                    Err(SpartanWhirError::InvalidProofShape)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut result = proof.clone();
        *result.initial_opening_mut()? = SharedProofOpening {
            rows,
            proof: multiproof,
        };
        E::validate_compressed_plain_proof_shape(config, &result, final_rows)?;
        Ok(result)
    }
}

impl<E> FixedOracleCacheFor<E>
where
    E: FullZkPoseidonEngine,
    E::EF: ExtField,
    E::PlainProof: PlainProofRowAccess + FixedOracleProofAccess<E>,
    Tree<E>: Sync,
    E::Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanObserve<E::Commitment>
        + CanSampleUniformBits<F>
        + Clone
        + Send,
    StandardUniform: Distribution<E::EF> + Distribution<F>,
    Plonky3WhirPcs: FullZkPoseidonPcs<E>,
    MaskGroupProverData<F, E::EF, E::ZkMmcs>: Send,
{
    /// Reconstruct omitted openings and run every ordinary verification check.
    pub fn verify(
        &self,
        vk: &PoseidonZkVerifyingKeyFor<E>,
        instance: &R1csInstance<F, E::Commitment>,
        mut proof: CachedFixedOracleProofFor<E>,
        challenger: &mut E::Challenger,
    ) -> Result<(), SpartanWhirError> {
        if self.fingerprint != key_fingerprint(vk)? || self.fingerprint != proof.fingerprint {
            return Err(SpartanWhirError::CommitmentMismatch);
        }
        proof.ensure_omitted()?;
        proof.proof.restore_static_rows()?;
        crate::protocol::PoseidonZkSpartanProtocolFor::<E>::verify_with_compression(
            vk,
            instance,
            &proof.proof.proof,
            challenger,
            proof.proof.products.as_ref(),
            proof.proof.options.fresh_rows,
            proof.proof.options.final_rows,
            Some(self),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        generate_satisfiable_fixture, poseidon1_challenger, setup_poseidon1_zk,
        Poseidon1QuinticEngine, Poseidon1ZkSpartanProtocol, PoseidonZkSetupConfig,
        QuinticExtension, SecurityConfig, SoundnessAssumption, SparkWhirParams,
        SyntheticR1csConfig, WhirParams,
    };
    use rand::{rngs::StdRng, SeedableRng};
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::proof_compression::ProofCompressionOptions;

    type E = Poseidon1QuinticEngine;
    type Plain = <E as FullZkPoseidonEngine>::PlainProof;
    type Cache = FixedOracleCacheFor<E>;
    type Cached = CachedFixedOracleProofFor<E>;
    type Compact = CompressedZkProofFor<E>;
    type Protocol = Poseidon1ZkSpartanProtocol<QuinticExtension>;

    fn setup_config() -> PoseidonZkSetupConfig {
        let whir = WhirParams {
            pow_bits: 0,
            folding_factor: 1,
            starting_log_inv_rate: 6,
            rs_domain_initial_reduction_factor: 1,
            ..WhirParams::default()
        };
        let mut witness = whir.clone();
        witness.starting_log_inv_rate = 8;
        PoseidonZkSetupConfig {
            matrix_closing: MatrixClosingMode::Spark,
            security: SecurityConfig {
                security_level_bits: 80,
                merkle_security_bits: 80,
                soundness_assumption: SoundnessAssumption::CapacityBound,
            },
            whir_params: witness,
            spark_whir_params: Some(SparkWhirParams {
                fixed_value: whir.clone(),
                fixed_audit: whir.clone(),
                read: whir,
            }),
            ell_zk: 3,
            mask_log_inv_rate: 3,
        }
    }

    fn fixture(seed: u64) -> crate::SyntheticR1csFixture {
        generate_satisfiable_fixture(&SyntheticR1csConfig {
            target_log2_witness_poly: 3,
            num_constraints: 4,
            num_io: 1,
            a_terms_per_constraint: 2,
            b_terms_per_constraint: 2,
            seed,
        })
        .unwrap()
    }

    struct CheckRestore<'a> {
        cache: &'a Cache,
        value: &'a Plain,
        audit: Option<&'a Plain>,
        checked: AtomicUsize,
    }
    impl FixedOpeningRestorer<E, Plonky3WhirPcs> for CheckRestore<'_> {
        fn validate_shape(
            &self,
            config: &WhirPcsConfig,
            proof: &Plain,
            final_rows: bool,
        ) -> Result<(), SpartanWhirError> {
            self.cache.validate_shape(config, proof, final_rows)
        }
        fn restore(
            &self,
            kind: FixedOracleKind,
            config: &WhirPcsConfig,
            proof: &Plain,
            statement: &PcsStatement<E>,
            challenger: &<E as SpartanWhirEngine>::Challenger,
            final_rows: bool,
        ) -> Result<Plain, SpartanWhirError> {
            let restored = self
                .cache
                .restore(kind, config, proof, statement, challenger, final_rows)?;
            let (expected, bit) = match kind {
                FixedOracleKind::Value => (self.value, 1),
                FixedOracleKind::Audit => (self.audit.unwrap(), 2),
            };
            assert_eq!(
                bincode::serialize(&restored).unwrap(),
                bincode::serialize(expected).unwrap(),
                "reconstruction returns every original row and sibling in its original order"
            );
            self.checked.fetch_or(bit, Ordering::Relaxed);
            Ok(restored)
        }
    }

    #[test]
    fn exact_reconstruction_and_corrupted_cache_data() {
        for target_log2_witness_poly in [3, 6] {
            check_exact_reconstruction_and_corruption(target_log2_witness_poly);
        }
    }

    fn check_exact_reconstruction_and_corruption(target_log2_witness_poly: usize) {
        let fixture = generate_satisfiable_fixture(&SyntheticR1csConfig {
            target_log2_witness_poly,
            num_constraints: 4,
            num_io: 1,
            a_terms_per_constraint: 2,
            b_terms_per_constraint: 2,
            seed: 817,
        })
        .unwrap();
        let (pk, vk) =
            setup_poseidon1_zk::<QuinticExtension>(fixture.shape, setup_config()).unwrap();
        let cache = Cache::build(&vk).unwrap();
        assert_eq!(cache.audit.is_some(), target_log2_witness_poly == 6);
        let (instance, compact) = Protocol::prove_compressed_with_rng(
            &pk,
            &fixture.public_inputs,
            &fixture.witness,
            &mut poseidon1_challenger(),
            &mut StdRng::seed_from_u64(611),
            ProofCompressionOptions::recommended(),
        )
        .unwrap();
        let compact_bytes = compact.to_bytes().unwrap();
        let bytes = Cached::from_compressed(&vk, compact)
            .unwrap()
            .to_bytes()
            .unwrap();
        assert!(bytes.len() < compact_bytes.len());
        let mut expected = Compact::from_bytes(&compact_bytes).unwrap();
        expected.restore_static_rows().unwrap();
        let ZkMatrixClosingProofFor::Spark(closing) = &expected.proof.matrix_closing else {
            unreachable!()
        };
        let restorer = CheckRestore {
            cache: &cache,
            value: &closing.spark_fixed_openings.value_proof,
            audit: closing.spark_fixed_openings.audit_proof.as_ref(),
            checked: AtomicUsize::new(0),
        };
        let mut cached = Cached::from_bytes(&bytes).unwrap();
        cached.proof.restore_static_rows().unwrap();
        let mut restored_challenger = poseidon1_challenger().with_trace();
        Protocol::verify_with_compression(
            &vk,
            &instance,
            &cached.proof.proof,
            &mut restored_challenger,
            cached.proof.products.as_ref(),
            cached.proof.options.fresh_rows,
            cached.proof.options.final_rows,
            Some(&restorer),
        )
        .unwrap();
        assert_eq!(
            restorer.checked.load(Ordering::Relaxed),
            if restorer.audit.is_some() { 3 } else { 1 }
        );
        let mut ordinary_challenger = poseidon1_challenger().with_trace();
        Protocol::verify_compressed(
            &vk,
            &instance,
            Compact::from_bytes(&compact_bytes).unwrap(),
            &mut ordinary_challenger,
        )
        .unwrap();
        for _ in 0..16 {
            assert_eq!(
                restored_challenger.sample_algebra_element::<QuinticExtension>(),
                ordinary_challenger.sample_algebra_element::<QuinticExtension>()
            );
        }
        if p3_maybe_rayon::prelude::current_num_threads() == 1 {
            assert_eq!(
                restored_challenger.transcript_trace(),
                ordinary_challenger.transcript_trace()
            );
        }

        // Only the test can replace private immutable data. Changing a cached
        // codeword while retaining its original digests fails ordinary MMCS verification.
        let mut corrupt = serde_json::to_value(&cache.value).unwrap();
        let values = corrupt["leaves"][0]["values"].as_array_mut().unwrap();
        for value in values {
            *value = serde_json::json!((value.as_u64().unwrap() + 1) % u64::from(F::ORDER_U32));
        }
        let corrupt = Cache {
            fingerprint: cache.fingerprint,
            value: serde_json::from_value(corrupt).unwrap(),
            audit: cache.audit.clone(),
            matrix_bytes: cache.matrix_bytes,
        };
        assert!(corrupt
            .verify(
                &vk,
                &instance,
                Cached::from_bytes(&bytes).unwrap(),
                &mut poseidon1_challenger()
            )
            .is_err());

        // Corrupt Merkle nodes independently of the cached leaf values.
        let mut corrupt = serde_json::to_value(&cache.value).unwrap();
        for layer in corrupt["digest_layers"].as_array_mut().unwrap() {
            for digest in layer.as_array_mut().unwrap() {
                for value in digest.as_array_mut().unwrap() {
                    *value =
                        serde_json::json!((value.as_u64().unwrap() + 1) % u64::from(F::ORDER_U32));
                }
            }
        }
        let corrupt = Cache {
            fingerprint: cache.fingerprint,
            value: serde_json::from_value(corrupt).unwrap(),
            audit: cache.audit.clone(),
            matrix_bytes: cache.matrix_bytes,
        };
        assert!(corrupt
            .verify(
                &vk,
                &instance,
                Cached::from_bytes(&bytes).unwrap(),
                &mut poseidon1_challenger()
            )
            .is_err());
    }

    #[test]
    fn construction_rejects_wrong_roots_geometry_and_unauthenticated_keys() {
        let original_fixture = fixture(912);
        let (_, mut vk) =
            setup_poseidon1_zk::<QuinticExtension>(original_fixture.shape, setup_config()).unwrap();
        let restored: PoseidonZkVerifyingKeyFor<E> =
            bincode::deserialize(&bincode::serialize(&vk).unwrap()).unwrap();
        assert!(Cache::build(&restored).is_err());
        let other = fixture(913);
        let (_, other_vk) =
            setup_poseidon1_zk::<QuinticExtension>(other.shape, setup_config()).unwrap();
        let original = vk.spark_fixed_commitments.take();
        vk.spark_fixed_commitments = other_vk.spark_fixed_commitments;
        assert!(Cache::build(&vk).is_err());
        vk.spark_fixed_commitments = original;
        vk.spark_pcs_configs
            .as_mut()
            .unwrap()
            .fixed_value
            .num_variables += 1;
        assert!(Cache::build(&vk).is_err());
    }
}
