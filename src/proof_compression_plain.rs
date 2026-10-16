//! Reconstruction of one omitted coordinate in each final plain-WHIR row.
//!
//! The final polynomial fixes the row's multilinear fold. Both encoding and
//! decoding replay an isolated transcript to choose a nonzero folding weight.
//! Authentication and the remaining equations stay with the ordinary verifier.

use p3_challenger::{
    CanObserve, CanSample, CanSampleUniformBits, FieldChallenger, GrindingChallenger,
};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_multilinear_util::{point::Point, poly::Poly};
use p3_sumcheck::{
    layout::{LayoutStrategy, Verifier},
    strategy::VariableOrder,
    OpeningBatch, OpeningProtocol, SumcheckData,
};
use p3_whir::{
    parameters::WhirConfig,
    pcs::proof::{QueryOpenings, WhirProof},
};

use crate::engine::{ExtField, F};
use crate::plonky3_whir_pcs::FullZkPoseidonEngine;
use crate::proof_compression_hiding::sample_queries;
use crate::SpartanWhirError;

type PlainProofFor<E> =
    WhirProof<F, <E as crate::SpartanWhirEngine>::EF, <E as FullZkPoseidonEngine>::ZkMmcs>;

/// Omits a coordinate from every final extension-field row of a prefix WHIR proof.
///
/// `challenger` is the state at entry to `WhirVerifier::verify`, after initial
/// opening claims and their batching challenge. The key-derived config must
/// match that verifier. Initial base-field oracles are left unchanged.
pub(crate) fn compress_final_rows<E>(
    config: &WhirConfig<E::EF, F, E::Challenger>,
    proof: &mut PlainProofFor<E>,
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
    if matches!(proof.final_openings, QueryOpenings::Base(_)) {
        return Ok(());
    }
    let (weights, positions) = final_context::<E>(config, proof, challenger)?;
    let pivot = pivot(&weights)?;
    let QueryOpenings::Extension(opening) = &mut proof.final_openings else {
        unreachable!();
    };
    if opening.rows.len() != positions.len()
        || opening.rows.iter().any(|row| row.len() != weights.len())
    {
        return Err(SpartanWhirError::InvalidProofShape);
    }
    for row in &mut opening.rows {
        row.remove(pivot);
    }
    Ok(())
}

/// Restores final extension-field rows before ordinary WHIR verification.
///
/// This does not advance the caller's challenger or replace Merkle verification.
/// Every extension row must be exactly one coordinate shorter than its normal
/// width. Base-field rows are retained in full by the encoder.
pub(crate) fn restore_final_rows<E>(
    config: &WhirConfig<E::EF, F, E::Challenger>,
    proof: &mut PlainProofFor<E>,
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
    if matches!(proof.final_openings, QueryOpenings::Base(_)) {
        return Ok(());
    }
    let (weights, positions) = final_context::<E>(config, proof, challenger)?;
    let pivot = pivot(&weights)?;
    let inverse = weights[pivot].inverse();
    let final_poly = proof
        .final_poly
        .as_ref()
        .ok_or(SpartanWhirError::InvalidProofShape)?;
    let QueryOpenings::Extension(opening) = &mut proof.final_openings else {
        unreachable!();
    };
    if opening.rows.len() != positions.len()
        || opening
            .rows
            .iter()
            .any(|row| row.len() + 1 != weights.len())
    {
        return Err(SpartanWhirError::InvalidProofShape);
    }
    let generator = config.final_round_config().folded_domain_gen;
    for (row, position) in opening.rows.iter_mut().zip(positions) {
        let point = generator.exp_u64(position as u64);
        // STIR treats the revealed evaluation table as univariate coefficients.
        let folded_value = final_poly
            .as_slice()
            .iter()
            .rev()
            .fold(E::EF::ZERO, |acc, &coefficient| acc * point + coefficient);
        let retained = weights
            .iter()
            .enumerate()
            .filter_map(|(index, &weight)| (index != pivot).then_some(weight))
            .zip(row.iter())
            .fold(E::EF::ZERO, |acc, (weight, &value)| acc + weight * value);
        row.insert(pivot, (folded_value - retained) * inverse);
    }
    Ok(())
}

/// Replays the prescribed-point read adapter up to its raw WHIR verifier call.
///
/// The initial commitment was already observed by the surrounding protocol.
/// This uses the same layout verifier as the adapter for OOD and opening-claim
/// observations, and leaves the original challenger untouched.
pub(crate) fn read_whir_challenger<E>(
    config: &WhirConfig<E::EF, F, E::Challenger>,
    proof: &PlainProofFor<E>,
    protocol: &OpeningProtocol,
    points: &[Point<E::EF>],
    evals: &[OpeningBatch<E::EF>],
    challenger: &E::Challenger,
) -> Result<E::Challenger, SpartanWhirError>
where
    E: FullZkPoseidonEngine,
    E::EF: ExtField,
    E::Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<E::Commitment>,
{
    if proof.initial_ood_answers.len() != config.commitment_ood_samples
        || protocol.num_openings() != points.len()
        || protocol.num_openings() != evals.len()
    {
        return Err(SpartanWhirError::InvalidProofShape);
    }
    let mut replay = E::challenger_for_replay(challenger);
    let mut layout = Verifier::<F, E::EF>::new(
        &protocol.table_shapes(),
        LayoutStrategy::new(true, VariableOrder::Prefix),
    );
    for &eval in &proof.initial_ood_answers {
        layout.add_virtual_eval(eval, &mut replay);
    }
    for (((table, batch), values), point) in protocol.iter_openings().zip(evals).zip(points) {
        if !batch.has_same_shape(values) {
            return Err(SpartanWhirError::InvalidProofShape);
        }
        layout
            .add_claim_at(table, batch, point, values, &mut replay)
            .map_err(|_| SpartanWhirError::WhirVerifyFailed)?;
    }
    let _: E::EF = replay.sample_algebra_element();
    Ok(replay)
}

fn final_context<E>(
    config: &WhirConfig<E::EF, F, E::Challenger>,
    proof: &PlainProofFor<E>,
    challenger: &E::Challenger,
) -> Result<(Vec<E::EF>, Vec<usize>), SpartanWhirError>
where
    E: FullZkPoseidonEngine,
    E::EF: ExtField,
    E::Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>
        + CanObserve<E::Commitment>,
{
    if proof.rounds.len() != config.n_rounds() {
        return Err(SpartanWhirError::InvalidProofShape);
    }
    let mut replay = E::challenger_for_replay(challenger);
    let mut randomness = replay_sumcheck::<E>(
        &proof.initial_sumcheck,
        config.round_folding_factor(0),
        config.starting_folding_pow_bits,
        &mut replay,
    )?;
    for (index, round) in proof.rounds.iter().enumerate() {
        let params = &config.round_parameters[index];
        let root = round
            .commitment
            .as_ref()
            .ok_or(SpartanWhirError::InvalidProofShape)?;
        if round.ood_answers.len() != params.ood_samples {
            return Err(SpartanWhirError::InvalidProofShape);
        }
        replay.observe(root.clone());
        for &answer in &round.ood_answers {
            let _: E::EF = replay.sample_algebra_element();
            replay.observe_algebra_element(answer);
        }
        if params.pow_bits > 0 && !replay.check_witness(params.pow_bits, round.pow_witness) {
            return Err(SpartanWhirError::WhirVerifyFailed);
        }
        let _: F = replay.sample();
        sample_queries(
            params.domain_size >> params.folding_factor,
            params.num_queries,
            &mut replay,
        )?;
        let _: E::EF = replay.sample_algebra_element();
        randomness = replay_sumcheck::<E>(
            &round.sumcheck,
            config.round_folding_factor(index + 1),
            params.folding_pow_bits,
            &mut replay,
        )?;
    }
    let final_poly = proof
        .final_poly
        .as_ref()
        .ok_or(SpartanWhirError::InvalidProofShape)?;
    let final_config = config.final_round_config();
    if final_poly.num_evals() != 1usize << final_config.num_variables {
        return Err(SpartanWhirError::InvalidProofShape);
    }
    replay.observe_algebra_slice(final_poly.as_slice());
    if config.final_pow_bits > 0
        && !replay.check_witness(config.final_pow_bits, proof.final_pow_witness)
    {
        return Err(SpartanWhirError::WhirVerifyFailed);
    }
    let positions = sample_queries(
        final_config.domain_size >> final_config.folding_factor,
        config.final_queries,
        &mut replay,
    )?;
    let weights = Poly::new_from_point(randomness.as_slice(), E::EF::ONE)
        .as_slice()
        .to_vec();
    Ok((weights, positions))
}

fn pivot<EF: Field>(weights: &[EF]) -> Result<usize, SpartanWhirError> {
    weights
        .iter()
        .position(|weight| !weight.is_zero())
        .ok_or(SpartanWhirError::InvalidProofShape)
}

pub(crate) fn replay_sumcheck<E>(
    data: &SumcheckData<F, E::EF>,
    count: usize,
    pow_bits: usize,
    challenger: &mut E::Challenger,
) -> Result<Point<E::EF>, SpartanWhirError>
where
    E: FullZkPoseidonEngine,
    E::EF: ExtField,
    E::Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    if data.polynomial_evaluations.len() != count
        || (pow_bits > 0 && data.pow_witnesses.len() != count)
    {
        return Err(SpartanWhirError::InvalidProofShape);
    }
    let mut randomness = Vec::with_capacity(count);
    for (index, wire) in data.polynomial_evaluations.iter().enumerate() {
        challenger.observe_algebra_slice(wire);
        if pow_bits > 0 && !challenger.check_witness(pow_bits, data.pow_witnesses[index]) {
            return Err(SpartanWhirError::WhirVerifyFailed);
        }
        randomness.push(challenger.sample_algebra_element());
    }
    Ok(Point::new(randomness))
}

#[cfg(test)]
mod tests {
    use p3_commit::MultilinearPcs;
    use p3_dft::Radix2DFTSmallBatch;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_sumcheck::{
        layout::{Layout, PrefixProver, Table},
        PrescribedPointPcs, TableShape, TableSpec,
    };
    use p3_whir::{
        parameters::{FoldingFactor, ProtocolParameters, SecurityAssumption},
        pcs::prover::WhirProver,
    };
    use rand::{rngs::StdRng, RngExt, SeedableRng};

    use super::*;
    use crate::engine::{Plonky3PoseidonEngine, Poseidon1QuinticEngine, QuinticExtension};

    type E = Poseidon1QuinticEngine;
    type EF = QuinticExtension;
    type Ch = <E as crate::SpartanWhirEngine>::Challenger;
    type Pcs = WhirProver<
        EF,
        F,
        Radix2DFTSmallBatch<F>,
        <E as FullZkPoseidonEngine>::ZkMmcs,
        Ch,
        PrefixProver<F, EF>,
    >;

    #[test]
    fn final_rows_roundtrip_through_original_read_adapter() {
        for (seed, pow_bits) in [(75, 0), (89, 3)] {
            let config = WhirConfig::new(
                12,
                ProtocolParameters {
                    security_level: 32,
                    pow_bits,
                    round_log_inv_rates: vec![],
                    folding_factor: FoldingFactor::Constant(4),
                    soundness_type: SecurityAssumption::CapacityBound,
                    starting_log_inv_rate: 2,
                },
            )
            .unwrap();
            let pcs = Pcs::new(config, Radix2DFTSmallBatch::default(), E::full_zk_mmcs());
            let mut rng = StdRng::seed_from_u64(seed);
            let table = Table::new(RowMajorMatrix::new(
                (0..1 << pcs.config.num_variables)
                    .map(|_| rng.random::<F>())
                    .collect(),
                1 << pcs.config.num_variables,
            ));
            let witness = <PrefixProver<F, EF> as Layout<F, EF>>::new_witness(
                vec![table],
                pcs.config.round_folding_factor(0),
            );
            let mut prover_challenger = E::challenger().with_trace();
            prover_challenger.observe(F::from_u32(0x524f5753));
            let (commitment, data) = pcs.commit(witness, &mut prover_challenger);
            let before = prover_challenger.clone();
            let protocol = OpeningProtocol::new(vec![TableSpec::new(
                TableShape::new(pcs.config.num_variables, 1),
                vec![OpeningBatch::new(vec![0], Vec::new())],
            )]);
            let points = vec![Point::<EF>::rand(&mut rng, pcs.config.num_variables)];
            let mut proof = pcs.open_at(data, &protocol, &points, &mut prover_challenger);
            let original = bincode::serialize(&proof).unwrap();
            let original_trace = before.transcript_trace();
            let entry = read_whir_challenger::<E>(
                &pcs.config,
                &proof.whir,
                &protocol,
                &points,
                &proof.evals,
                &before,
            )
            .unwrap();
            compress_final_rows::<E>(&pcs.config, &mut proof.whir, &entry).unwrap();
            let compressed = bincode::serialize(&proof).unwrap();
            assert!(compressed.len() < original.len());
            restore_final_rows::<E>(&pcs.config, &mut proof.whir, &entry).unwrap();
            assert!(bincode::serialize(&proof).unwrap() == original);
            assert_eq!(before.transcript_trace(), original_trace);
            let mut verifier_challenger = before.clone();
            pcs.verify_at(
                &commitment,
                &proof,
                &protocol,
                &points,
                &mut verifier_challenger,
            )
            .unwrap();
            assert_eq!(
                verifier_challenger.sample_algebra_element::<EF>(),
                prover_challenger.sample_algebra_element::<EF>(),
            );

            let mut altered = proof.clone();
            compress_final_rows::<E>(&pcs.config, &mut altered.whir, &entry).unwrap();
            let QueryOpenings::Extension(opening) = &mut altered.whir.final_openings else {
                panic!("test requires an extension oracle");
            };
            opening.rows[0][0] += EF::ONE;
            restore_final_rows::<E>(&pcs.config, &mut altered.whir, &entry).unwrap();
            assert!(pcs
                .verify_at(
                    &commitment,
                    &altered,
                    &protocol,
                    &points,
                    &mut before.clone()
                )
                .is_err());
        }
    }

    #[test]
    fn pivot_handles_boolean_folding_challenges() {
        for point in [
            vec![EF::ZERO, EF::ONE, EF::ONE / EF::TWO],
            vec![EF::ONE, EF::ONE],
            vec![EF::ZERO, EF::ZERO],
        ] {
            let weights = Poly::new_from_point(&point, EF::ONE);
            assert_eq!(weights.as_slice().iter().copied().sum::<EF>(), EF::ONE);
            let chosen = pivot(weights.as_slice()).unwrap();
            assert!(!weights.as_slice()[chosen].is_zero());
            assert!(weights.as_slice()[..chosen].iter().all(Field::is_zero));
        }
    }

    #[test]
    fn final_fold_does_not_determine_a_second_omitted_coordinate() {
        // A fold is one nonzero linear equation. Every retained coordinate has
        // a direction in its kernel, so the same equation cannot recover it.
        for point in [
            vec![EF::from_u32(3), EF::from_u32(7), EF::from_u32(11)],
            vec![EF::ZERO, EF::ONE, EF::ONE / EF::TWO],
        ] {
            let weights = Poly::new_from_point(&point, EF::ONE);
            let weights = weights.as_slice();
            let chosen = pivot(weights).unwrap();
            let original = (0..weights.len())
                .map(|i| EF::from_usize(i + 19))
                .collect::<Vec<_>>();
            let dot = |row: &[EF]| row.iter().zip(weights).map(|(&a, &b)| a * b).sum::<EF>();
            for retained in (0..weights.len()).filter(|&i| i != chosen) {
                let mut alternate = original.clone();
                alternate[retained] += EF::ONE;
                alternate[chosen] -= weights[retained] / weights[chosen];
                assert_ne!(alternate[retained], original[retained]);
                assert_eq!(dot(&alternate), dot(&original));
            }
        }
    }

    macro_rules! local_adapter_test {
        ($name:ident, $engine:ty) => {
            #[test]
            fn $name() {
                use crate::{
                    MlePcs, MultilinearPoint, PcsStatementBuilder, Plonky3WhirPcs, PointEvalClaim,
                    ProtocolPcs, SecurityConfig, SparkReadPcs, WhirParams, WhirPcsConfig,
                };
                type Engine = $engine;
                let config = WhirPcsConfig {
                    num_variables: 12,
                    security: SecurityConfig {
                        security_level_bits: 80,
                        merkle_security_bits: 80,
                        ..SecurityConfig::default()
                    },
                    whir: WhirParams {
                        pow_bits: 0,
                        folding_factor: 4,
                        starting_log_inv_rate: 2,
                        rs_domain_initial_reduction_factor: 1,
                        folding_schedule: None,
                        round_log_inv_rates: vec![],
                    },
                };
                let mut rng = StdRng::seed_from_u64(203);
                let values = (0..1 << config.num_variables)
                    .map(|_| rng.random::<F>())
                    .collect::<Vec<_>>();
                let point = Point::<EF>::rand(&mut rng, config.num_variables);
                let value = Poly::new(values.clone()).eval_base(&point);
                let statement = PcsStatementBuilder::<Engine>::new()
                    .add_point_eval(PointEvalClaim {
                        point: MultilinearPoint(point.as_slice().to_vec()),
                        value,
                    })
                    .finalize()
                    .unwrap();
                let mut prover = Engine::challenger();
                let mut verifier = prover.clone();
                let (commitment, data) =
                    <Plonky3WhirPcs as MlePcs<Engine>>::commit(&config, &values, &mut prover)
                        .unwrap();
                let proof = <Plonky3WhirPcs as ProtocolPcs<Engine>>::open_compressed(
                    &config,
                    data,
                    &statement,
                    true,
                    &mut prover,
                )
                .unwrap();
                Engine::validate_compressed_plain_proof_shape(&config, &proof, true).unwrap();
                assert!(Engine::validate_plain_proof_shape(&config, &proof).is_err());
                let parsed = <Plonky3WhirPcs as ProtocolPcs<Engine>>::verify_parse_commitment(
                    &config,
                    &commitment,
                    &proof,
                    &mut verifier,
                )
                .unwrap();
                <Plonky3WhirPcs as ProtocolPcs<Engine>>::verify_finalize_compressed(
                    &config,
                    &parsed,
                    &statement,
                    &proof,
                    true,
                    &mut verifier,
                )
                .unwrap();
                assert_eq!(
                    prover.sample_algebra_element::<EF>(),
                    verifier.sample_algebra_element::<EF>()
                );

                let mut prover = Engine::challenger();
                let mut verifier = prover.clone();
                let (commitment, data) =
                    <Plonky3WhirPcs as SparkReadPcs<Engine>>::commit_read_table(
                        &config,
                        values,
                        1 << config.num_variables,
                        1,
                        &mut prover,
                    )
                    .unwrap();
                let points = vec![MultilinearPoint(point.as_slice().to_vec())];
                let columns = vec![vec![0]];
                let (proof, evals) =
                    <Plonky3WhirPcs as SparkReadPcs<Engine>>::open_read_table_compressed(
                        &config,
                        data,
                        1,
                        &columns,
                        &points,
                        true,
                        &mut prover,
                    )
                    .unwrap();
                Engine::validate_compressed_plain_proof_shape(&config, &proof, true).unwrap();
                let parsed =
                    <Plonky3WhirPcs as SparkReadPcs<Engine>>::verify_parse_read_commitment(
                        &config,
                        &commitment,
                        &proof,
                        &mut verifier,
                    )
                    .unwrap();
                <Plonky3WhirPcs as SparkReadPcs<Engine>>::verify_finalize_read_table_compressed(
                    &config,
                    &parsed,
                    &proof,
                    1,
                    &columns,
                    &points,
                    &evals,
                    true,
                    &mut verifier,
                )
                .unwrap();
                assert_eq!(
                    prover.sample_algebra_element::<EF>(),
                    verifier.sample_algebra_element::<EF>()
                );
            }
        };
    }

    local_adapter_test!(
        poseidon1_local_final_row_adapters,
        crate::Poseidon1QuinticEngine
    );
    local_adapter_test!(
        poseidon2_local_final_row_adapters,
        crate::PoseidonQuinticEngine
    );
}
