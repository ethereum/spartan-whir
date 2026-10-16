//! Canonical host encoding for the fixed LeanVM Spartan-WHIR control guest.
//!
//! The control guest consumes KoalaBear words rather than a Rust serialization.
//! Every word is a canonical field representative. The schema has a fixed
//! profile identifier, explicit section tags and lengths, a fixed quintic
//! coefficient order, and a required end marker.

use alloc::vec::Vec;

use p3_challenger::{CanFinalizeDigest, CanObserve};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField32};
use p3_merkle_tree::{MerkleCap, PrunedMerklePaths};
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::SumcheckData;
use p3_whir::pcs::proof::{QueryOpenings, SharedProofOpening, WhirProof, WhirRoundProof};
use sha2::{Digest, Sha256};

use crate::plonky3_whir_pcs::{PoseidonCommitment, PoseidonMmcs};
use crate::{
    engine::{poseidon_challenger, F},
    CubicRoundPoly, InnerSumcheckProof, MatrixClosingMode, OuterSumcheckProof, Plonky3WhirPcs,
    PoseidonEngine, PoseidonProof, PoseidonTranscriptEvent, PoseidonVerifyingKey,
    QuadraticRoundPoly, QuinticExtension, R1csInstance, SecurityConfig, SpartanProof,
    SpartanProofKind, SpartanProtocol, SpartanWhirError,
};

pub const LEANVM_GUEST_INPUT_VERSION: u32 = 1;
pub const LEANVM_CONTROL_PROFILE_NUMBER: u32 = 1;
pub const LEANVM_CONTROL_PROFILE_ID: &str =
    "spartan-whir-poseidon2-quintic-no-zk-direct-control-v1";
pub const LEANVM_CONTROL_STATEMENT_SCHEMA_ID: &str = "control-synthetic-inputs-v1";
pub const LEANVM_CONTROL_STATEMENT_DIGEST_ID: &str =
    "poseidon2-width-16-length-prefixed-statement-v1";
pub const MAX_CONTROL_GUEST_WORDS: usize = 1 << 20;

const MAGIC: [u32; 4] = [b'L' as u32, b'V' as u32, b'S' as u32, b'W' as u32];
const TAG_STATEMENT: u32 = 1;
const TAG_SPARTAN_PROOF: u32 = 2;
const TAG_WHIR_PROOF: u32 = 3;
const TAG_END: u32 = 4;
const TAG_NONE: u32 = 0;
const TAG_SOME: u32 = 1;
const TAG_QUERY_BASE: u32 = 0;
const TAG_QUERY_EXTENSION: u32 = 1;
const DIGEST_WORDS: usize = 8;
const QUINTIC_DEGREE: usize = 5;
const STATEMENT_DIGEST_DOMAIN: &[u8] = b"leanvm-spartan-whir-statement-v1";
const VERIFYING_KEY_ID_DOMAIN: &[u8] = b"leanvm-spartan-whir-control-vk-v1";

type ControlEngine = PoseidonEngine<QuinticExtension>;
type ControlProof = PoseidonProof<QuinticExtension>;
type ControlVerifyingKey = PoseidonVerifyingKey<QuinticExtension>;
type ControlSpartanProof = SpartanProof<ControlEngine, Plonky3WhirPcs>;
type ControlWhirProof = WhirProof<F, QuinticExtension, PoseidonMmcs>;
type ControlQueryOpenings = QueryOpenings<F, QuinticExtension, PrunedMerklePaths<F, 8>>;

/// Encode the fixed no-ZK DirectSparse control proof into canonical KoalaBear words.
pub fn encode_control_guest_input(proof: &ControlProof) -> Result<Vec<u32>, SpartanWhirError> {
    let direct = match &proof.proof {
        SpartanProofKind::Direct(proof) => proof,
        SpartanProofKind::Spark(_) => return Err(SpartanWhirError::ProofKindMismatch),
    };

    let mut writer = WordWriter::default();
    writer.words.extend(MAGIC);
    writer.metadata(LEANVM_GUEST_INPUT_VERSION)?;
    writer.metadata(LEANVM_CONTROL_PROFILE_NUMBER)?;

    writer.metadata(TAG_STATEMENT)?;
    writer.base_vec(&proof.instance.public_inputs)?;
    writer.commitment(&proof.instance.witness_commitment)?;

    writer.metadata(TAG_SPARTAN_PROOF)?;
    writer.cubic_rounds(&direct.outer_sumcheck.rounds)?;
    writer.extension(&direct.outer_claims.0);
    writer.extension(&direct.outer_claims.1);
    writer.extension(&direct.outer_claims.2);
    writer.quadratic_rounds(&direct.inner_sumcheck.rounds)?;
    writer.extension(&direct.witness_eval);

    writer.metadata(TAG_WHIR_PROOF)?;
    writer.whir_proof(&direct.pcs_proof)?;
    writer.metadata(TAG_END)?;

    if writer.words.len() > MAX_CONTROL_GUEST_WORDS {
        return Err(SpartanWhirError::ProofEncodeFailed);
    }
    Ok(writer.words)
}

/// Decode the fixed control schema and reject non-canonical or trailing words.
pub fn decode_control_guest_input(words: &[u32]) -> Result<ControlProof, SpartanWhirError> {
    if words.len() > MAX_CONTROL_GUEST_WORDS {
        return Err(SpartanWhirError::ProofDecodeFailed);
    }
    let mut reader = WordReader::new(words)?;
    for expected in MAGIC {
        reader.expect(expected)?;
    }
    reader.expect(LEANVM_GUEST_INPUT_VERSION)?;
    reader.expect(LEANVM_CONTROL_PROFILE_NUMBER)?;

    reader.expect(TAG_STATEMENT)?;
    let public_inputs = reader.base_vec()?;
    let witness_commitment = reader.commitment()?;

    reader.expect(TAG_SPARTAN_PROOF)?;
    let outer_sumcheck = OuterSumcheckProof {
        rounds: reader.cubic_rounds()?,
    };
    let outer_claims = (
        reader.extension()?,
        reader.extension()?,
        reader.extension()?,
    );
    let inner_sumcheck = InnerSumcheckProof {
        rounds: reader.quadratic_rounds()?,
    };
    let witness_eval = reader.extension()?;

    reader.expect(TAG_WHIR_PROOF)?;
    let pcs_proof = reader.whir_proof()?;
    reader.expect(TAG_END)?;
    reader.finish()?;

    let instance = R1csInstance {
        public_inputs,
        witness_commitment,
    };
    let direct = ControlSpartanProof {
        outer_sumcheck,
        outer_claims,
        inner_sumcheck,
        witness_eval,
        pcs_proof,
    };
    Ok(PoseidonProof::new(
        instance,
        SpartanProofKind::Direct(direct),
    ))
}

/// Field elements absorbed to derive the eight-element control statement digest.
pub fn control_statement_digest_preimage(public_inputs: &[F]) -> Vec<F> {
    let mut preimage = Vec::with_capacity(STATEMENT_DIGEST_DOMAIN.len() + public_inputs.len() + 2);
    preimage.push(F::from_usize(STATEMENT_DIGEST_DOMAIN.len()));
    preimage.extend(STATEMENT_DIGEST_DOMAIN.iter().copied().map(F::from_u8));
    preimage.push(F::from_usize(public_inputs.len()));
    preimage.extend_from_slice(public_inputs);
    preimage
}

/// Domain-separated Poseidon2 digest exposed as the control guest's public input.
pub fn control_statement_digest(public_inputs: &[F]) -> [F; DIGEST_WORDS] {
    let mut challenger = poseidon_challenger();
    challenger.observe_slice(&control_statement_digest_preimage(public_inputs));
    challenger.finalize()
}

/// Verify with the native Rust verifier while recording every challenger operation.
pub fn verify_control_with_trace(
    verifying_key: &ControlVerifyingKey,
    expected_public_inputs: &[F],
    proof: &ControlProof,
) -> Result<Vec<PoseidonTranscriptEvent>, SpartanWhirError> {
    verifying_key.ensure_authenticated()?;
    if verifying_key.matrix_closing() != MatrixClosingMode::DirectSparse {
        return Err(SpartanWhirError::ProofKindMismatch);
    }
    if expected_public_inputs.len() != verifying_key.num_io()
        || proof.instance.public_inputs.len() != verifying_key.num_io()
    {
        return Err(SpartanWhirError::InvalidPublicInputLength);
    }
    if proof.instance.public_inputs != expected_public_inputs {
        return Err(SpartanWhirError::PublicInputMismatch);
    }

    let mut challenger = poseidon_challenger().with_trace();
    SpartanProtocol::<ControlEngine, Plonky3WhirPcs>::verify_with_mode(
        verifying_key,
        &proof.instance,
        &proof.proof,
        &mut challenger,
    )?;
    Ok(challenger.transcript_trace())
}

/// Canonical identifier for the fixed DirectSparse control verifying key.
///
/// The identifier binds the transcript parameters and the complete canonical
/// R1CS shape. DirectSparse has no setup commitment outside those values. A
/// restored key must be authenticated before deriving this identifier.
pub fn control_verifying_key_id(
    verifying_key: &ControlVerifyingKey,
) -> Result<[u8; 32], SpartanWhirError> {
    verifying_key.ensure_authenticated()?;
    if verifying_key.matrix_closing() != MatrixClosingMode::DirectSparse {
        return Err(SpartanWhirError::ProofKindMismatch);
    }
    let mut hasher = Sha256::new();
    hasher.update(VERIFYING_KEY_ID_DOMAIN);
    let domain = verifying_key.domain_separator().to_bytes();
    update_len_prefixed(&mut hasher, &domain);
    update_u64(&mut hasher, verifying_key.num_cons_unpadded() as u64);
    update_u64(&mut hasher, verifying_key.num_vars_unpadded() as u64);
    update_u64(&mut hasher, verifying_key.num_io() as u64);
    update_security(&mut hasher, verifying_key.security());
    let shape = verifying_key.shape_canonical();
    update_u64(&mut hasher, shape.num_cons as u64);
    update_u64(&mut hasher, shape.num_vars as u64);
    update_u64(&mut hasher, shape.num_io as u64);
    for matrix in [&shape.a, &shape.b, &shape.c] {
        update_u64(&mut hasher, matrix.num_rows as u64);
        update_u64(&mut hasher, matrix.num_cols as u64);
        update_u64(&mut hasher, matrix.entries.len() as u64);
        for entry in &matrix.entries {
            update_u64(&mut hasher, entry.row as u64);
            update_u64(&mut hasher, entry.col as u64);
            hasher.update(entry.val.as_canonical_u32().to_le_bytes());
        }
    }
    Ok(hasher.finalize().into())
}

fn update_security(hasher: &mut Sha256, security: SecurityConfig) {
    hasher.update(security.security_level_bits.to_le_bytes());
    hasher.update(security.merkle_security_bits.to_le_bytes());
    hasher.update([security.soundness_assumption as u8]);
}

fn update_u64(hasher: &mut Sha256, value: u64) {
    hasher.update(value.to_le_bytes());
}

fn update_len_prefixed(hasher: &mut Sha256, bytes: &[u8]) {
    update_u64(hasher, bytes.len() as u64);
    hasher.update(bytes);
}

#[derive(Default)]
struct WordWriter {
    words: Vec<u32>,
}

impl WordWriter {
    fn metadata(&mut self, value: u32) -> Result<(), SpartanWhirError> {
        if value >= F::ORDER_U32 {
            return Err(SpartanWhirError::ProofEncodeFailed);
        }
        self.words.push(value);
        Ok(())
    }

    fn len(&mut self, len: usize) -> Result<(), SpartanWhirError> {
        let value = u32::try_from(len).map_err(|_| SpartanWhirError::ProofEncodeFailed)?;
        self.metadata(value)
    }

    fn base(&mut self, value: F) {
        self.words.push(value.as_canonical_u32());
    }

    fn base_vec(&mut self, values: &[F]) -> Result<(), SpartanWhirError> {
        self.len(values.len())?;
        for &value in values {
            self.base(value);
        }
        Ok(())
    }

    fn extension(&mut self, value: &QuinticExtension) {
        let coefficients =
            <QuinticExtension as BasedVectorSpace<F>>::as_basis_coefficients_slice(value);
        debug_assert_eq!(coefficients.len(), QUINTIC_DEGREE);
        for &coefficient in coefficients {
            self.base(coefficient);
        }
    }

    fn extension_vec(&mut self, values: &[QuinticExtension]) -> Result<(), SpartanWhirError> {
        self.len(values.len())?;
        for value in values {
            self.extension(value);
        }
        Ok(())
    }

    fn commitment(&mut self, commitment: &PoseidonCommitment) -> Result<(), SpartanWhirError> {
        if commitment.num_roots() != 1 {
            return Err(SpartanWhirError::InvalidCommitmentShape);
        }
        self.len(commitment.num_roots())?;
        for root in commitment.roots() {
            for &value in root {
                self.base(value);
            }
        }
        Ok(())
    }

    fn cubic_rounds(
        &mut self,
        rounds: &[CubicRoundPoly<QuinticExtension>],
    ) -> Result<(), SpartanWhirError> {
        self.len(rounds.len())?;
        for round in rounds {
            for value in &round.0 {
                self.extension(value);
            }
        }
        Ok(())
    }

    fn quadratic_rounds(
        &mut self,
        rounds: &[QuadraticRoundPoly<QuinticExtension>],
    ) -> Result<(), SpartanWhirError> {
        self.len(rounds.len())?;
        for round in rounds {
            for value in &round.0 {
                self.extension(value);
            }
        }
        Ok(())
    }

    fn sumcheck(
        &mut self,
        sumcheck: &SumcheckData<F, QuinticExtension>,
    ) -> Result<(), SpartanWhirError> {
        self.len(sumcheck.polynomial_evaluations.len())?;
        for evaluations in &sumcheck.polynomial_evaluations {
            self.extension(&evaluations[0]);
            self.extension(&evaluations[1]);
        }
        self.base_vec(&sumcheck.pow_witnesses)
    }

    fn query_openings(&mut self, openings: &ControlQueryOpenings) -> Result<(), SpartanWhirError> {
        match openings {
            QueryOpenings::Base(opening) => {
                self.metadata(TAG_QUERY_BASE)?;
                self.len(opening.rows.len())?;
                for row in &opening.rows {
                    self.base_vec(row)?;
                }
                self.merkle_proof(&opening.proof)
            }
            QueryOpenings::Extension(opening) => {
                self.metadata(TAG_QUERY_EXTENSION)?;
                self.len(opening.rows.len())?;
                for row in &opening.rows {
                    self.extension_vec(row)?;
                }
                self.merkle_proof(&opening.proof)
            }
        }
    }

    fn merkle_proof(&mut self, proof: &PrunedMerklePaths<F, 8>) -> Result<(), SpartanWhirError> {
        self.len(proof.sibling_hashes.len())?;
        for digest in &proof.sibling_hashes {
            for &value in digest {
                self.base(value);
            }
        }
        Ok(())
    }

    fn whir_proof(&mut self, proof: &ControlWhirProof) -> Result<(), SpartanWhirError> {
        self.extension_vec(&proof.initial_ood_answers)?;
        self.sumcheck(&proof.initial_sumcheck)?;
        self.len(proof.rounds.len())?;
        for round in &proof.rounds {
            match &round.commitment {
                Some(commitment) => {
                    self.metadata(TAG_SOME)?;
                    self.commitment(commitment)?;
                }
                None => self.metadata(TAG_NONE)?,
            }
            self.extension_vec(&round.ood_answers)?;
            self.base(round.pow_witness);
            self.query_openings(&round.openings)?;
            self.sumcheck(&round.sumcheck)?;
        }
        match &proof.final_poly {
            Some(poly) => {
                self.metadata(TAG_SOME)?;
                self.extension_vec(poly.as_slice())?;
            }
            None => self.metadata(TAG_NONE)?,
        }
        self.base(proof.final_pow_witness);
        self.query_openings(&proof.final_openings)?;
        match &proof.final_sumcheck {
            Some(sumcheck) => {
                self.metadata(TAG_SOME)?;
                self.sumcheck(sumcheck)?;
            }
            None => self.metadata(TAG_NONE)?,
        }
        Ok(())
    }
}

struct WordReader<'a> {
    words: &'a [u32],
    offset: usize,
}

impl<'a> WordReader<'a> {
    fn new(words: &'a [u32]) -> Result<Self, SpartanWhirError> {
        if words.iter().any(|&word| word >= F::ORDER_U32) {
            return Err(SpartanWhirError::NonCanonicalEncoding);
        }
        Ok(Self { words, offset: 0 })
    }

    fn word(&mut self) -> Result<u32, SpartanWhirError> {
        let word = self
            .words
            .get(self.offset)
            .copied()
            .ok_or(SpartanWhirError::ProofDecodeFailed)?;
        self.offset += 1;
        Ok(word)
    }

    fn expect(&mut self, expected: u32) -> Result<(), SpartanWhirError> {
        if self.word()? != expected {
            return Err(SpartanWhirError::InvalidBlobHeader);
        }
        Ok(())
    }

    fn len(&mut self) -> Result<usize, SpartanWhirError> {
        let len = self.word()? as usize;
        if len > self.words.len().saturating_sub(self.offset) {
            return Err(SpartanWhirError::ProofDecodeFailed);
        }
        Ok(len)
    }

    fn base(&mut self) -> Result<F, SpartanWhirError> {
        Ok(F::from_u32(self.word()?))
    }

    fn base_vec(&mut self) -> Result<Vec<F>, SpartanWhirError> {
        let len = self.len()?;
        (0..len).map(|_| self.base()).collect()
    }

    fn extension(&mut self) -> Result<QuinticExtension, SpartanWhirError> {
        let mut coefficients = [F::ZERO; QUINTIC_DEGREE];
        for coefficient in &mut coefficients {
            *coefficient = self.base()?;
        }
        Ok(QuinticExtension::new(coefficients))
    }

    fn extension_vec(&mut self) -> Result<Vec<QuinticExtension>, SpartanWhirError> {
        let len = self.len()?;
        if len
            .checked_mul(QUINTIC_DEGREE)
            .is_none_or(|words| words > self.remaining())
        {
            return Err(SpartanWhirError::ProofDecodeFailed);
        }
        (0..len).map(|_| self.extension()).collect()
    }

    fn commitment(&mut self) -> Result<PoseidonCommitment, SpartanWhirError> {
        let roots = self.len()?;
        if roots != 1
            || roots
                .checked_mul(DIGEST_WORDS)
                .is_none_or(|words| words > self.remaining())
        {
            return Err(SpartanWhirError::InvalidCommitmentShape);
        }
        let mut cap = Vec::with_capacity(roots);
        for _ in 0..roots {
            let mut digest = [F::ZERO; DIGEST_WORDS];
            for value in &mut digest {
                *value = self.base()?;
            }
            cap.push(digest);
        }
        Ok(MerkleCap::new(cap))
    }

    fn cubic_rounds(&mut self) -> Result<Vec<CubicRoundPoly<QuinticExtension>>, SpartanWhirError> {
        let len = self.len()?;
        if len
            .checked_mul(3 * QUINTIC_DEGREE)
            .is_none_or(|words| words > self.remaining())
        {
            return Err(SpartanWhirError::ProofDecodeFailed);
        }
        (0..len)
            .map(|_| {
                Ok(CubicRoundPoly([
                    self.extension()?,
                    self.extension()?,
                    self.extension()?,
                ]))
            })
            .collect()
    }

    fn quadratic_rounds(
        &mut self,
    ) -> Result<Vec<QuadraticRoundPoly<QuinticExtension>>, SpartanWhirError> {
        let len = self.len()?;
        if len
            .checked_mul(2 * QUINTIC_DEGREE)
            .is_none_or(|words| words > self.remaining())
        {
            return Err(SpartanWhirError::ProofDecodeFailed);
        }
        (0..len)
            .map(|_| Ok(QuadraticRoundPoly([self.extension()?, self.extension()?])))
            .collect()
    }

    fn sumcheck(&mut self) -> Result<SumcheckData<F, QuinticExtension>, SpartanWhirError> {
        let rounds = self.len()?;
        if rounds
            .checked_mul(2 * QUINTIC_DEGREE)
            .is_none_or(|words| words > self.remaining())
        {
            return Err(SpartanWhirError::ProofDecodeFailed);
        }
        let polynomial_evaluations = (0..rounds)
            .map(|_| Ok([self.extension()?, self.extension()?]))
            .collect::<Result<Vec<_>, SpartanWhirError>>()?;
        let pow_witnesses = self.base_vec()?;
        Ok(SumcheckData {
            polynomial_evaluations,
            pow_witnesses,
        })
    }

    fn query_openings(&mut self) -> Result<ControlQueryOpenings, SpartanWhirError> {
        match self.word()? {
            TAG_QUERY_BASE => {
                let row_count = self.len()?;
                let rows = (0..row_count)
                    .map(|_| self.base_vec())
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(QueryOpenings::Base(SharedProofOpening {
                    rows,
                    proof: self.merkle_proof()?,
                }))
            }
            TAG_QUERY_EXTENSION => {
                let row_count = self.len()?;
                let rows = (0..row_count)
                    .map(|_| self.extension_vec())
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(QueryOpenings::Extension(SharedProofOpening {
                    rows,
                    proof: self.merkle_proof()?,
                }))
            }
            _ => Err(SpartanWhirError::InvalidBlobFlags),
        }
    }

    fn merkle_proof(&mut self) -> Result<PrunedMerklePaths<F, 8>, SpartanWhirError> {
        let len = self.len()?;
        if len
            .checked_mul(DIGEST_WORDS)
            .is_none_or(|words| words > self.remaining())
        {
            return Err(SpartanWhirError::ProofDecodeFailed);
        }
        let mut sibling_hashes = Vec::with_capacity(len);
        for _ in 0..len {
            let mut digest = [F::ZERO; DIGEST_WORDS];
            for value in &mut digest {
                *value = self.base()?;
            }
            sibling_hashes.push(digest);
        }
        Ok(PrunedMerklePaths { sibling_hashes })
    }

    fn option_tag(&mut self) -> Result<bool, SpartanWhirError> {
        match self.word()? {
            TAG_NONE => Ok(false),
            TAG_SOME => Ok(true),
            _ => Err(SpartanWhirError::InvalidBlobFlags),
        }
    }

    fn whir_proof(&mut self) -> Result<ControlWhirProof, SpartanWhirError> {
        let initial_ood_answers = self.extension_vec()?;
        let initial_sumcheck = self.sumcheck()?;
        let round_count = self.len()?;
        let mut rounds = Vec::with_capacity(round_count);
        for _ in 0..round_count {
            let commitment = self.option_tag()?.then(|| self.commitment()).transpose()?;
            let ood_answers = self.extension_vec()?;
            let pow_witness = self.base()?;
            let openings = self.query_openings()?;
            let sumcheck = self.sumcheck()?;
            rounds.push(WhirRoundProof {
                commitment,
                ood_answers,
                pow_witness,
                openings,
                sumcheck,
            });
        }
        let final_poly = if self.option_tag()? {
            let coefficients = self.extension_vec()?;
            if coefficients.is_empty() || !coefficients.len().is_power_of_two() {
                return Err(SpartanWhirError::InvalidPolynomialLength);
            }
            Some(Poly::new(coefficients))
        } else {
            None
        };
        let final_pow_witness = self.base()?;
        let final_openings = self.query_openings()?;
        let final_sumcheck = self.option_tag()?.then(|| self.sumcheck()).transpose()?;
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

    fn remaining(&self) -> usize {
        self.words.len().saturating_sub(self.offset)
    }

    fn finish(self) -> Result<(), SpartanWhirError> {
        if self.offset != self.words.len() {
            return Err(SpartanWhirError::TrailingBytes);
        }
        Ok(())
    }
}
