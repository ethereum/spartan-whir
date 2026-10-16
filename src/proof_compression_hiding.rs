//! Reconstruction of omitted fresh hiding-WHIR leaf values.
//!
//! This module replays transcript operations on a clone and restores the ordinary
//! proof. It does not authenticate openings or establish the committed relation;
//! the caller must pass the restored proof to the existing WHIR verifier.

use p3_challenger::{
    CanObserve, CanSample, CanSampleUniformBits, FieldChallenger, GrindingChallenger,
};
use p3_commit::{ExtensionMmcs, Mmcs};
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::dense::DenseMatrix;
use p3_multilinear_util::{point::Point, poly::Poly};
use p3_sumcheck::zk::{ZkSumcheckData, ZkVerifier};
use p3_whir::pcs::proof::QueryOpenings;
use p3_whir::pcs::zk::{MaskGroupShape, ZkWhirConfig};

use crate::engine::{ExtField, F};
use crate::plonky3_whir_pcs::{FullZkPoseidonEngine, PoseidonZkRelationProofFor};
use crate::SpartanWhirError;

/// Restores fresh base-case rows at the entry to `verify_relation`.
///
/// `config`, `mmcs`, and `external_mask_shapes` must come from the same validated
/// key and application relation supplied to the ordinary verifier. The
/// challenger must be at the exact entry to that verifier, after the application
/// has bound its relation and external mask commitments.
///
/// All fresh row vectors must have been omitted. The retained rows, reveals,
/// commitments, and authentication proofs are not changed. The input challenger
/// is borrowed immutably so the ordinary verifier replays the original transcript.
pub(crate) fn restore_fresh_rows<E>(
    config: &ZkWhirConfig<E::EF, F, E::Challenger>,
    mmcs: &E::ZkMmcs,
    external_mask_shapes: &[MaskGroupShape],
    initial_target: E::EF,
    proof: &mut PoseidonZkRelationProofFor<E>,
    challenger: &E::Challenger,
) -> Result<(), SpartanWhirError>
where
    E: FullZkPoseidonEngine,
    E::EF: ExtField,
    E::Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>
        + CanObserve<E::Commitment>,
{
    let reject = || SpartanWhirError::WhirVerifyFailed;
    let n_rounds = config.n_rounds();
    if proof.rounds.len() != n_rounds
        || proof.sumchecks.len() != n_rounds + 1
        || proof.sumcheck_mask_commitments.len() != n_rounds + 1
        || !proof.base_case.fresh_main_openings.rows.is_empty()
        || proof
            .base_case
            .fresh_mask_openings
            .iter()
            .any(|opening| !opening.rows.is_empty())
    {
        return Err(reject());
    }

    let mut replay = E::challenger_for_replay(challenger);
    let mut mask_groups = external_mask_shapes.to_vec();
    let mut target = initial_target;
    let mut randomness = replay_sumcheck::<E>(
        &proof.sumchecks[0],
        &proof.sumcheck_mask_commitments[0],
        config.zk.ell_zk,
        config.round_folding_factor(0),
        config.starting_folding_pow_bits,
        &mut target,
        &mut replay,
    )?;
    mask_groups.push(config.sumcheck_mask_group(config.round_folding_factor(0)));

    // Mirror only the observations and samples in the pinned hiding verifier.
    // Every masked sumcheck observes its inherited target. Recompute that target
    // from the retained rows; the ordinary verifier authenticates those rows.
    for (round_index, round) in proof.rounds.iter().enumerate() {
        let params = &config.round_parameters[round_index];
        if round.ood_answers.len() != params.ood_samples {
            return Err(reject());
        }
        replay.observe(round.commitment.clone());
        replay.observe(round.mask_commitment.clone());
        for &answer in &round.ood_answers {
            let _: E::EF = replay.sample_algebra_element();
            replay.observe_algebra_element(answer);
        }
        replay_pow::<E::Challenger>(&mut replay, params.pow_bits, round.pow_witness)?;
        let _: F = replay.sample();
        let positions = sample_queries(
            params.domain_size >> config.round_folding_factor(round_index),
            params.num_queries,
            &mut replay,
        )?;
        let width = 1usize << config.round_folding_factor(round_index);
        let folded_values = match &round.openings {
            QueryOpenings::Base(opening) if round_index == 0 => {
                check_rows(&opening.rows, positions.len(), width)?;
                opening
                    .rows
                    .iter()
                    .map(|row| Poly::new(row.clone()).eval_base(&randomness))
                    .collect::<Vec<_>>()
            }
            QueryOpenings::Extension(opening) if round_index > 0 => {
                check_rows(&opening.rows, positions.len(), width)?;
                opening
                    .rows
                    .iter()
                    .map(|row| Poly::new(row.clone()).eval_ext::<F>(&randomness))
                    .collect::<Vec<_>>()
            }
            _ => return Err(reject()),
        };
        let combination: E::EF = replay.sample_algebra_element();
        let mut power = combination;
        for &answer in round.ood_answers.iter().chain(&folded_values) {
            target += power * answer;
            power *= combination;
        }
        mask_groups.push(MaskGroupShape {
            shape: config.switch_masks[round_index],
            width: 1,
        });
        let folding = config.round_folding_factor(round_index + 1);
        randomness = replay_sumcheck::<E>(
            &proof.sumchecks[round_index + 1],
            &proof.sumcheck_mask_commitments[round_index + 1],
            config.zk.ell_zk,
            folding,
            params.folding_pow_bits,
            &mut target,
            &mut replay,
        )?;
        mask_groups.push(config.sumcheck_mask_group(folding));
    }

    let final_config = config.final_round_config();
    let source_message_len = 1usize << final_config.num_variables;
    let source_domain_size = final_config.domain_size >> final_config.folding_factor;
    let base = &mut proof.base_case;
    let batches = config.fresh_mask_batching.batches(&mask_groups);
    let num_masks: usize = mask_groups.iter().map(|group| group.width).sum();
    if base.blinded_message.len() != source_message_len
        || base.blinded_randomness.len() != config.oracle_randomness[n_rounds]
        || base.blinded_masks.len() != num_masks
        || base.carried_mask_openings.len() != mask_groups.len()
        || base.fresh_mask_commitments.len() != batches.len()
        || base.fresh_mask_openings.len() != batches.len()
    {
        return Err(reject());
    }
    let mut offset = 0;
    for group in &mask_groups {
        if group.width == 0
            || !group.shape.domain_size.is_power_of_two()
            || group.shape.domain_size.ilog2() > E::EF::TWO_ADICITY as u32
        {
            return Err(reject());
        }
        for mask in &base.blinded_masks[offset..offset + group.width] {
            if mask.message.len() != group.shape.message_len
                || mask.randomness.len() != group.shape.randomness_len
            {
                return Err(reject());
            }
        }
        offset += group.width;
    }

    replay.observe(base.fresh_main_commitment.clone());
    for commitment in &base.fresh_mask_commitments {
        replay.observe(commitment.clone());
    }
    replay.observe_algebra_element(base.masked_claim);
    let gamma: E::EF = replay.sample_algebra_element();

    // The base-case reveal digest is a Merkle commitment to the concatenated
    // extension elements, two per leaf, with one zero element when needed.
    // Keep this encoding identical to the pinned backend's reveal_digest.
    let mut reveals = Vec::with_capacity(
        base.blinded_message.len()
            + base.blinded_randomness.len()
            + base
                .blinded_masks
                .iter()
                .map(|mask| mask.message.len() + mask.randomness.len())
                .sum::<usize>()
            + 1,
    );
    reveals.extend_from_slice(&base.blinded_message);
    reveals.extend_from_slice(&base.blinded_randomness);
    for mask in &base.blinded_masks {
        reveals.extend_from_slice(&mask.message);
        reveals.extend_from_slice(&mask.randomness);
    }
    if reveals.len() % 2 != 0 {
        reveals.push(E::EF::ZERO);
    }
    let extension_mmcs = ExtensionMmcs::<F, E::EF, E::ZkMmcs>::new(mmcs.clone());
    replay.observe(extension_mmcs.commit_matrix(DenseMatrix::new(reveals, 2)).0);
    replay_pow::<E::Challenger>(&mut replay, config.final_pow_bits, base.pow_witness)?;

    let source_positions = sample_queries(source_domain_size, config.final_queries, &mut replay)?;
    let source_width = 1usize << final_config.folding_factor;
    let source_values = match &base.source_openings {
        QueryOpenings::Base(opening) if n_rounds == 0 => {
            check_rows(&opening.rows, source_positions.len(), source_width)?;
            opening
                .rows
                .iter()
                .map(|row| Poly::new(row.clone()).eval_base(&randomness))
                .collect::<Vec<_>>()
        }
        QueryOpenings::Extension(opening) if n_rounds > 0 => {
            check_rows(&opening.rows, source_positions.len(), source_width)?;
            opening
                .rows
                .iter()
                .map(|row| Poly::new(row.clone()).eval_ext::<F>(&randomness))
                .collect::<Vec<_>>()
        }
        _ => return Err(reject()),
    };
    let source_generator = F::two_adic_generator(source_domain_size.ilog2() as usize);
    base.fresh_main_openings.rows = source_positions
        .iter()
        .zip(source_values)
        .map(|(&position, source)| {
            let point = source_generator.exp_u64(position as u64);
            let revealed = eval_code(point, &base.blinded_message, &base.blinded_randomness);
            vec![revealed - gamma * source]
        })
        .collect();

    let mut mask_offsets = Vec::with_capacity(mask_groups.len());
    let mut offset = 0;
    for group in &mask_groups {
        mask_offsets.push(offset);
        offset += group.width;
    }
    for (batch, fresh) in batches.iter().zip(&mut base.fresh_mask_openings) {
        let positions = sample_queries(batch.domain_size, config.mask_queries, &mut replay)?;
        let generator = E::EF::two_adic_generator(batch.domain_size.ilog2() as usize);
        for &index in &batch.group_indices {
            check_rows(
                &base.carried_mask_openings[index].rows,
                positions.len(),
                mask_groups[index].width,
            )?;
        }
        fresh.rows = positions
            .iter()
            .enumerate()
            .map(|(row_index, &position)| {
                let point = generator.exp_u64(position as u64);
                let mut row = Vec::with_capacity(batch.width);
                for &index in &batch.group_indices {
                    let offset = mask_offsets[index];
                    let masks = &base.blinded_masks[offset..offset + mask_groups[index].width];
                    row.extend(
                        masks
                            .iter()
                            .zip(&base.carried_mask_openings[index].rows[row_index])
                            .map(|(mask, &value)| {
                                eval_code(point, &mask.message, &mask.randomness) - gamma * value
                            }),
                    );
                }
                row
            })
            .collect();
    }
    Ok(())
}

fn replay_sumcheck<E>(
    data: &ZkSumcheckData<F, E::EF>,
    commitment: &E::Commitment,
    ell_zk: usize,
    folding: usize,
    pow_bits: usize,
    target: &mut E::EF,
    challenger: &mut E::Challenger,
) -> Result<Point<E::EF>, SpartanWhirError>
where
    E: FullZkPoseidonEngine,
    E::EF: ExtField,
    E::Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<E::Commitment>,
{
    let handoff =
        ZkVerifier::<F, E::EF>::verify_claim::<ExtensionMmcs<F, E::EF, E::ZkMmcs>, E::Challenger>(
            data, commitment, ell_zk, folding, pow_bits, *target, challenger,
        )
        .map_err(|_| SpartanWhirError::WhirVerifyFailed)?;
    *target = handoff.claimed_residual;
    Ok(handoff.randomness)
}

fn replay_pow<C>(challenger: &mut C, bits: usize, witness: F) -> Result<(), SpartanWhirError>
where
    C: GrindingChallenger<Witness = F>,
{
    if bits > 0 && !challenger.check_witness(bits, witness) {
        return Err(SpartanWhirError::WhirVerifyFailed);
    }
    Ok(())
}

/// Same distinct, sorted query sampling used by the pinned WHIR backend.
pub(crate) fn sample_queries<C>(
    domain_size: usize,
    count: usize,
    challenger: &mut C,
) -> Result<Vec<usize>, SpartanWhirError>
where
    C: CanSampleUniformBits<F>,
{
    if !domain_size.is_power_of_two() {
        return Err(SpartanWhirError::WhirVerifyFailed);
    }
    let mut positions = Vec::with_capacity(count.min(domain_size));
    while positions.len() < count.min(domain_size) {
        let position = challenger
            .sample_uniform_bits::<true>(domain_size.ilog2() as usize)
            .map_err(|_| SpartanWhirError::WhirVerifyFailed)?;
        if !positions.contains(&position) {
            positions.push(position);
        }
    }
    positions.sort_unstable();
    Ok(positions)
}

fn check_rows<T>(rows: &[Vec<T>], count: usize, width: usize) -> Result<(), SpartanWhirError> {
    if rows.len() != count || rows.iter().any(|row| row.len() != width) {
        return Err(SpartanWhirError::WhirVerifyFailed);
    }
    Ok(())
}

/// Evaluates the coefficient vector `message || randomness` without allocating.
fn eval_code<B: Field, EF: ExtensionField<B>>(point: B, message: &[EF], randomness: &[EF]) -> EF {
    randomness
        .iter()
        .rev()
        .chain(message.iter().rev())
        .fold(EF::ZERO, |acc, &coefficient| acc * point + coefficient)
}

#[cfg(test)]
mod tests {
    use p3_dft::Radix2DFTSmallBatch;
    use p3_whir::parameters::{FoldingFactor, ProtocolParameters, SecurityAssumption};
    use p3_whir::pcs::zk::{
        CommittedMaskGroup, CommittedRelation, HidingWhirProver, HidingWhirVerifier, MaskCodeShape,
        ZkParameters,
    };
    use rand::{rngs::StdRng, RngExt, SeedableRng};

    use super::*;
    use crate::engine::{Plonky3PoseidonEngine, Poseidon1QuinticEngine, QuinticExtension};

    type E = Poseidon1QuinticEngine;
    type EF = QuinticExtension;
    type Ch = <E as crate::SpartanWhirEngine>::Challenger;
    type Commitment = <E as FullZkPoseidonEngine>::Commitment;

    struct Fixture {
        config: ZkWhirConfig<EF, F, Ch>,
        mmcs: <E as FullZkPoseidonEngine>::ZkMmcs,
        commitment: Commitment,
        proof: PoseidonZkRelationProofFor<E>,
        relation: CommittedRelation<EF>,
        masks: Vec<CommittedMaskGroup<EF, Commitment>>,
        before: Ch,
        after: Ch,
    }

    fn fixture(seed: u64, pow_bits: usize) -> Fixture {
        fixture_with_batching(
            seed,
            pow_bits,
            p3_whir::pcs::zk::FreshMaskBatching::Separate,
        )
    }

    fn fixture_with_batching(
        seed: u64,
        pow_bits: usize,
        batching: p3_whir::pcs::zk::FreshMaskBatching,
    ) -> Fixture {
        let config = ZkWhirConfig::new(
            12,
            ProtocolParameters {
                security_level: 32,
                pow_bits,
                round_log_inv_rates: vec![],
                folding_factor: FoldingFactor::Constant(4),
                soundness_type: SecurityAssumption::CapacityBound,
                starting_log_inv_rate: 2,
            },
            ZkParameters {
                ell_zk: 3,
                mask_log_inv_rate: 1,
            },
        )
        .unwrap()
        .with_fresh_mask_batching(batching);
        let dft = Radix2DFTSmallBatch::<F>::default();
        let mmcs = E::full_zk_mmcs();
        let prover = HidingWhirProver::new(&config, &dft, &mmcs);
        let mut rng = StdRng::seed_from_u64(seed);
        let mut challenger = E::challenger().with_trace();
        challenger.observe(F::from_u32(0x434f4445));
        let witness = Poly::<F>::rand(&mut rng, config.num_variables);
        let point = Point::<EF>::rand(&mut rng, config.num_variables);
        let mut relation = CommittedRelation::empty();
        relation.target = witness.eval_base(&point);
        relation.source.push_eq(point.clone(), EF::ONE);
        let (commitment, data) = prover.commit(witness, &mut challenger, &mut rng);

        let mut external_prover_masks = Vec::new();
        let mut masks = Vec::new();
        for (message_len, width) in [(8, 2), (3, 1)] {
            let shape = MaskGroupShape {
                shape: MaskCodeShape::new(message_len, config.mask_queries, 1),
                width,
            };
            let messages = (0..width)
                .map(|_| (0..message_len).map(|_| rng.random::<EF>()).collect())
                .collect();
            let mut group = prover
                .commit_mask_group(shape, messages, &mut challenger, &mut rng)
                .unwrap();
            group.covectors = (0..width)
                .map(|_| (0..message_len).map(|_| rng.random::<EF>()).collect())
                .collect();
            for (message, covector) in group.messages.iter().zip(&group.covectors) {
                relation.target += message
                    .iter()
                    .zip(covector)
                    .map(|(&a, &b)| a * b)
                    .sum::<EF>();
                challenger.observe_algebra_slice(covector);
            }
            masks.push(CommittedMaskGroup {
                shape,
                commitment: group.commitment.clone(),
                covectors: group.covectors.clone(),
            });
            external_prover_masks.push(group);
        }
        challenger.observe_algebra_slice(point.as_slice());
        challenger.observe_algebra_element(relation.target);
        let before = challenger.clone();
        let proof = prover
            .prove_relation(
                data,
                &relation,
                external_prover_masks,
                &mut challenger,
                &mut rng,
            )
            .unwrap();
        Fixture {
            config,
            mmcs,
            commitment,
            proof,
            relation,
            masks,
            before,
            after: challenger,
        }
    }

    fn omit_fresh_rows(proof: &mut PoseidonZkRelationProofFor<E>) {
        proof.base_case.fresh_main_openings.rows.clear();
        for opening in &mut proof.base_case.fresh_mask_openings {
            opening.rows.clear();
        }
    }

    fn restore(fixture: &Fixture, proof: &mut PoseidonZkRelationProofFor<E>) {
        let shapes = fixture
            .masks
            .iter()
            .map(|group| group.shape)
            .collect::<Vec<_>>();
        restore_fresh_rows::<E>(
            &fixture.config,
            &fixture.mmcs,
            &shapes,
            fixture.relation.target,
            proof,
            &fixture.before,
        )
        .unwrap();
    }

    #[test]
    fn fresh_rows_roundtrip_to_backend_proof_and_preserve_transcript() {
        for (seed, pow_bits, batching) in [
            (31, 0, p3_whir::pcs::zk::FreshMaskBatching::Separate),
            (42, 3, p3_whir::pcs::zk::FreshMaskBatching::Separate),
            (31, 0, p3_whir::pcs::zk::FreshMaskBatching::SameHeight),
            (42, 3, p3_whir::pcs::zk::FreshMaskBatching::SameHeight),
        ] {
            let fixture = fixture_with_batching(seed, pow_bits, batching);
            let mut restored = fixture.proof.clone();
            omit_fresh_rows(&mut restored);
            let trace_before = fixture.before.transcript_trace();
            restore(&fixture, &mut restored);
            assert_eq!(fixture.before.transcript_trace(), trace_before);
            let restored_bytes = bincode::serialize(&restored).unwrap();
            let original_bytes = bincode::serialize(&fixture.proof).unwrap();
            assert!(
                restored_bytes == original_bytes,
                "restored proof differs at byte {:?}",
                restored_bytes
                    .iter()
                    .zip(&original_bytes)
                    .position(|(a, b)| a != b)
            );
            let mut verifier_challenger = fixture.before.clone();
            HidingWhirVerifier::new(&fixture.config, &fixture.mmcs)
                .verify_relation(
                    &restored,
                    &fixture.commitment,
                    fixture.relation.clone(),
                    fixture.masks.clone(),
                    &mut verifier_challenger,
                )
                .unwrap();
            let mut prover_challenger = fixture.after;
            assert_eq!(
                verifier_challenger.sample_algebra_element::<EF>(),
                prover_challenger.sample_algebra_element::<EF>(),
            );
        }
    }

    #[test]
    fn reconstructed_rows_do_not_replace_merkle_authentication() {
        let fixture = fixture(51, 0);
        let mut altered = fixture.proof.clone();
        omit_fresh_rows(&mut altered);
        altered.base_case.carried_mask_openings[0].rows[0][0] += EF::ONE;
        restore(&fixture, &mut altered);
        assert!(HidingWhirVerifier::new(&fixture.config, &fixture.mmcs)
            .verify_relation(
                &altered,
                &fixture.commitment,
                fixture.relation.clone(),
                fixture.masks.clone(),
                &mut fixture.before.clone(),
            )
            .is_err());

        let mut changed_commitment = fixture.proof.clone();
        omit_fresh_rows(&mut changed_commitment);
        changed_commitment.base_case.fresh_main_commitment = fixture.commitment.clone();
        restore(&fixture, &mut changed_commitment);
        assert_ne!(
            changed_commitment.base_case.fresh_main_openings.rows,
            fixture.proof.base_case.fresh_main_openings.rows,
        );
        assert!(HidingWhirVerifier::new(&fixture.config, &fixture.mmcs)
            .verify_relation(
                &changed_commitment,
                &fixture.commitment,
                fixture.relation.clone(),
                fixture.masks.clone(),
                &mut fixture.before.clone(),
            )
            .is_err());
    }

    #[test]
    fn reconstruction_rejects_partial_omission_and_bad_reveal_lengths() {
        let fixture = fixture(62, 0);
        let shapes = fixture
            .masks
            .iter()
            .map(|group| group.shape)
            .collect::<Vec<_>>();
        let mut partial = fixture.proof.clone();
        partial.base_case.fresh_main_openings.rows.clear();
        assert!(restore_fresh_rows::<E>(
            &fixture.config,
            &fixture.mmcs,
            &shapes,
            fixture.relation.target,
            &mut partial,
            &fixture.before,
        )
        .is_err());
        let mut malformed = fixture.proof.clone();
        omit_fresh_rows(&mut malformed);
        malformed.base_case.blinded_masks[0].randomness.pop();
        assert!(restore_fresh_rows::<E>(
            &fixture.config,
            &fixture.mmcs,
            &shapes,
            fixture.relation.target,
            &mut malformed,
            &fixture.before,
        )
        .is_err());
    }

    #[test]
    fn fresh_equation_does_not_determine_the_retained_carried_value() {
        // The revealed evaluation fixes fresh + gamma * carried. After fresh
        // is omitted, carried remains free under this one equation.
        for gamma in [EF::ZERO, EF::ONE, EF::from_u32(29)] {
            let revealed = EF::from_u32(103);
            let carried = EF::from_u32(7);
            let fresh = revealed - gamma * carried;
            for delta in [EF::ONE, EF::from_u32(31)] {
                let alternate_carried = carried + delta;
                let alternate_fresh = fresh - gamma * delta;
                assert_ne!(carried, alternate_carried);
                assert_eq!(fresh + gamma * carried, revealed);
                assert_eq!(alternate_fresh + gamma * alternate_carried, revealed);
            }
        }
    }
}
