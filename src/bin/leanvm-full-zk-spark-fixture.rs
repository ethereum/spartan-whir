#[path = "../../benches/support/sha256.rs"]
mod sha256_fixture;

#[cfg(feature = "poseidon1")]
use spartan_whir::withdrawal::config::configured_withdrawal_spark_config;

use std::{
    collections::BTreeMap,
    env,
    error::Error,
    fs::{self, File},
    io::{self, Write},
    path::{Path, PathBuf},
    process,
    time::Instant,
};

use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_merkle_tree::MerkleCap;
use p3_whir::pcs::proof::QueryOpenings;
use rand::{rngs::StdRng, SeedableRng};
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use spartan_whir::{
    encode_full_zk_spark_guest_words, full_zk_spark_guest_verifier_config,
    full_zk_statement_digest, full_zk_statement_digest_id, full_zk_statement_digest_preimage,
    pad_full_zk_guest_words, preprocess_spark_tables, read_table_group_column_counts,
    recommended_quintic_spark_fixed_whir_params, recommended_quintic_spark_read_whir_params,
    recommended_quintic_spark_zk_whir_params, MatrixClosingMode, PoseidonZkSetupConfig,
    QuinticExtension, R1csShape, SecurityConfig, SoundnessAssumption, SparkWhirParams,
    DEFAULT_ZK_ELL, DEFAULT_ZK_MASK_LOG_INV_RATE, FULL_ZK_STATEMENT_SCHEMA_ID,
    MAX_FULL_ZK_GUEST_WORDS,
};

#[cfg(feature = "poseidon1")]
use spartan_whir::{
    poseidon1_challenger as poseidon_zk_challenger, Poseidon1QuinticEngine as GuestEngine,
    Poseidon1ZkCommitment as ZkCommitment, Poseidon1ZkMatrixClosingProof as ZkMatrixClosingProof,
    Poseidon1ZkSpartanProof as ZkSpartanProof,
    Poseidon1ZkSpartanProtocol as PoseidonZkSpartanProtocol,
    Poseidon1ZkVerifyingKey as PoseidonZkVerifyingKey,
};
#[cfg(not(feature = "poseidon1"))]
use spartan_whir::{
    poseidon_zk_challenger, PoseidonQuinticEngine as GuestEngine,
    PoseidonZkCommitment as ZkCommitment, PoseidonZkSpartanProtocol, PoseidonZkVerifyingKey,
    ZkMatrixClosingProof, ZkSpartanProof,
};

const MESSAGE_BYTES: usize = 2_048;
const PROOF_SEED: u64 = 0x5A17_5041_524B_0001;

#[derive(Serialize)]
struct Artifact {
    file: String,
    bytes: usize,
    sha256: String,
}

#[derive(Serialize)]
struct WordRange {
    offset: usize,
    words: usize,
}

#[derive(Serialize)]
struct GuestLayout {
    schema: &'static str,
    total_words: usize,
    entries: BTreeMap<String, WordRange>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    if !cfg!(feature = "poseidon1") {
        return Err(io::Error::other(
            "the production LeanVM SPARK fixture requires --features poseidon1",
        )
        .into());
    }

    let mut args = env::args_os().skip(1);
    let output = args.next().map(PathBuf::from).unwrap_or_else(|| usage());
    let mut circuit = String::from("sha256");
    let mut sample = 0_u64;
    let mut setup_config_path = None;
    while let Some(argument) = args.next() {
        match argument.to_str() {
            Some("--circuit") => {
                circuit = args
                    .next()
                    .and_then(|value| value.into_string().ok())
                    .unwrap_or_else(|| usage());
            }
            Some("--sample") => {
                sample = args
                    .next()
                    .and_then(|value| value.into_string().ok())
                    .and_then(|value| value.parse().ok())
                    .unwrap_or_else(|| usage());
            }
            Some("--setup-config") => {
                setup_config_path = Some(args.next().map(PathBuf::from).unwrap_or_else(|| usage()));
            }
            _ => usage(),
        }
    }
    fs::create_dir_all(&output)?;

    let load_started = Instant::now();
    let (shape, witness, public_inputs, application) = match circuit.as_str() {
        "sha256" => {
            let fixture = sha256_fixture::Sha256Fixture::load(MESSAGE_BYTES)?;
            let message = sha256_fixture::message(MESSAGE_BYTES, usize::try_from(sample)?);
            let input = sha256_fixture::input_binary(&message);
            fixture.validate_input(&message, &input)?;
            let (witness, public_inputs) = fixture.generator.generate_witness(
                &input,
                fixture.shape.num_vars,
                fixture.shape.num_io,
            )?;
            (
                fixture.shape,
                witness,
                public_inputs,
                json!({
                    "id": "sha256-2048-byte-optimized-v1",
                    "message_bytes": MESSAGE_BYTES,
                }),
            )
        }
        "privacy-pools" => {
            let fixture =
                spartan_whir::withdrawal::deterministic_withdrawal_fixture_for_sample(sample);
            let application = json!({
                "id": "privacy-pools-withdrawal-poseidon2-v1",
                "state_tree_depth": 32,
                "asp_tree_depth": 32,
                "hash": "poseidon2-koalabear-width24-rate16-capacity8-digest8",
                "amount_bits": 128,
                "constraints": fixture.stats.constraints,
                "witness_variables": fixture.stats.witness_variables,
                "public_inputs": fixture.stats.public_inputs,
                "poseidon_permutations": fixture.stats.poseidon_permutations,
                "matrix_nonzeros": fixture.stats.matrix_nonzeros,
            });
            (
                fixture.shape,
                fixture.witness,
                fixture.public_inputs,
                application,
            )
        }
        _ => return Err(io::Error::other("--circuit must be sha256 or privacy-pools").into()),
    };
    let preparation_elapsed = load_started.elapsed();
    spartan_whir::validate_satisfaction(&shape, &witness, &public_inputs)?;
    let relation_digest = spartan_whir::canonical_r1cs_relation_digest(
        &shape.pad_regular().map_err(protocol_error)?,
        shape.num_cons,
        shape.num_vars,
        shape.num_io,
    )
    .map_err(protocol_error)?;

    let config = match circuit.as_str() {
        #[cfg(feature = "poseidon1")]
        "privacy-pools" => configured_withdrawal_spark_config(setup_config_path.as_deref())?,
        _ => {
            if setup_config_path.is_some() {
                return Err(
                    io::Error::other("--setup-config requires --circuit privacy-pools").into(),
                );
            }
            production_spark_config(&shape)?
        }
    };
    let setup_started = Instant::now();
    let default_batching = if circuit == "privacy-pools" {
        "same_height"
    } else {
        "separate"
    };
    let batching = match env::var("SPARK_FRESH_MASK_BATCHING")
        .as_deref()
        .unwrap_or(default_batching)
    {
        "separate" => spartan_whir::pcs_config::FreshMaskBatching::Separate,
        "same_height" => spartan_whir::pcs_config::FreshMaskBatching::SameHeight,
        _ => {
            return Err(io::Error::other(
                "SPARK_FRESH_MASK_BATCHING must be separate or same_height",
            )
            .into())
        }
    };
    let default_packing = if circuit == "privacy-pools" {
        "all_free_basis"
    } else {
        "off"
    };
    let packing = match env::var("SPARK_MASK_PACKING")
        .as_deref()
        .unwrap_or(default_packing)
    {
        "off" => spartan_whir::pcs_config::MaskPacking::Off,
        "application" => spartan_whir::pcs_config::MaskPacking::Application,
        "all" => spartan_whir::pcs_config::MaskPacking::All,
        "application_free_basis" => spartan_whir::pcs_config::MaskPacking::ApplicationFreeBasis,
        "all_free_basis" => spartan_whir::pcs_config::MaskPacking::AllFreeBasis,
        _ => return Err(io::Error::other("unknown SPARK_MASK_PACKING").into()),
    };
    let (pk, vk) = spartan_whir::poseidon::setup_poseidon_zk_with_mask_packing::<GuestEngine>(
        shape.clone(),
        config.clone(),
        batching,
        packing,
    )
    .map_err(protocol_error)?;
    let setup_elapsed = setup_started.elapsed();
    println!(
        "setup completed in {:.3} ms; starting proof",
        setup_elapsed.as_secs_f64() * 1_000.0
    );

    let prove_started = Instant::now();
    let mut prover_challenger = poseidon_zk_challenger();
    let proof_seed = PROOF_SEED.wrapping_add(sample);
    let mut rng = StdRng::seed_from_u64(proof_seed);
    let (instance, proof) = PoseidonZkSpartanProtocol::<QuinticExtension>::prove_with_rng(
        &pk,
        &public_inputs,
        &witness,
        &mut prover_challenger,
        &mut rng,
    )
    .map_err(protocol_error)?;
    let prove_elapsed = prove_started.elapsed();
    println!(
        "proof completed in {:.3} ms; starting verification",
        prove_elapsed.as_secs_f64() * 1_000.0
    );

    let verify_started = Instant::now();
    let mut verifier_challenger = poseidon_zk_challenger().with_trace();
    PoseidonZkSpartanProtocol::<QuinticExtension>::verify(
        &vk,
        &instance,
        &proof,
        &mut verifier_challenger,
    )
    .map_err(protocol_error)?;
    let verify_elapsed = verify_started.elapsed();

    let canonical_words = encode_full_zk_spark_guest_words(&instance, &proof)?;
    let layout = parse_guest_layout(&canonical_words)?;
    let layout_artifact = write_json(&output, "full_zk_spark_guest_layout.json", &layout)?;
    let words = pad_full_zk_guest_words(canonical_words)?;
    let input_artifact = write_words(&output, "full_zk_spark_guest_input.words", &words)?;

    let verifier_config = full_zk_spark_guest_verifier_config(&vk).map_err(protocol_error)?;
    let ZkMatrixClosingProof::Spark(closing) = &proof.matrix_closing else {
        unreachable!("production fixture uses SPARK")
    };
    let proof_value_root =
        canonical(closing.spark_fixed_openings.value_commitment.roots()[0].as_slice());
    if verifier_config.fixed_commitments.value.cap[0].as_slice() != proof_value_root.as_slice()
        || verifier_config.fixed_commitments.audit.is_some()
            != closing.spark_fixed_openings.audit_commitment.is_some()
    {
        return Err(io::Error::other(
            "encoded SPARK fixed commitments do not match the verifying key",
        )
        .into());
    }
    let config_artifact = write_json(
        &output,
        "full_zk_spark_guest_constants.json",
        &verifier_config,
    )?;

    let trace = verifier_challenger.transcript_trace();
    let trace_artifact = write_json(&output, "full_zk_spark_transcript_trace.json", &trace)?;

    let statement_preimage = full_zk_statement_digest_preimage(&instance.public_inputs);
    let statement_digest = full_zk_statement_digest::<GuestEngine>(&instance.public_inputs);
    let statement_artifact = write_json(
        &output,
        "full_zk_spark_statement.json",
        &json!({
            "schema": FULL_ZK_STATEMENT_SCHEMA_ID,
            "digest_algorithm": full_zk_statement_digest_id::<GuestEngine>(),
            "inner_public_inputs": canonical(&instance.public_inputs),
            "digest_preimage": canonical(&statement_preimage),
            "digest": canonical(&statement_digest),
        }),
    )?;
    let mutations = write_mutations(&words, &instance, &proof, &vk)?;
    let mutations_artifact = write_json(
        &output,
        "full_zk_spark_mutations.json",
        &json!({
            "schema": "leanvm-full-zk-spark-word-patches-v1",
            "base_words": words.len(),
            "cases": mutations,
        }),
    )?;

    let manifest = json!({
        "schema": "leanvm-full-zk-spark-fixture-v1",
        "application": application["id"],
        "application_details": application,
        "statement_schema": FULL_ZK_STATEMENT_SCHEMA_ID,
        "hash_profile": "poseidon1-width16",
        "matrix_closing": "spark",
        "extension": "X^5 + X^2 - 1",
        "extension_degree": 5,
        "sample_index": sample,
        "proof_seed": proof_seed,
        "relation_digest": relation_digest,
        "raw_shape": {
            "constraints": shape.num_cons,
            "variables": shape.num_vars,
            "public_inputs": shape.num_io,
            "matrix_nonzeros": shape.a.nnz() + shape.b.nnz() + shape.c.nnz(),
        },
        "diagnostic_times_ms": {
            "circuit_and_input_preparation": preparation_elapsed.as_secs_f64() * 1_000.0,
            "setup": setup_elapsed.as_secs_f64() * 1_000.0,
            "prove": prove_elapsed.as_secs_f64() * 1_000.0,
            "verify": verify_elapsed.as_secs_f64() * 1_000.0,
        },
        "setup": config,
        "canonical_shape": {
            "constraints": verifier_config.common.canonical_shape.constraints,
            "variables": verifier_config.common.canonical_shape.variables,
            "public_inputs": verifier_config.common.canonical_shape.public_inputs,
        },
        "spark_table_metadata": verifier_config.table_metadata,
        "canonical_words": layout.total_words,
        "guest_words": words.len(),
        "guest_word_limit": MAX_FULL_ZK_GUEST_WORDS,
        "transcript_events": trace.len(),
        "proof_shape": proof_shape(&proof),
        "artifacts": {
            "guest_input": input_artifact,
            "guest_layout": layout_artifact,
            "guest_constants": config_artifact,
            "transcript_trace": trace_artifact,
            "statement": statement_artifact,
            "mutations": mutations_artifact,
        },
    });
    write_json(&output, "full_zk_spark_manifest.json", &manifest)?;

    println!(
        "wrote {} canonical words in a {}-word LeanVM witness",
        layout.total_words,
        words.len()
    );
    println!("recorded {} transcript events", trace.len());
    println!(
        "circuit and input preparation {:.3} ms, setup {:.3} ms, prove {:.3} ms, verify {:.3} ms",
        preparation_elapsed.as_secs_f64() * 1_000.0,
        setup_elapsed.as_secs_f64() * 1_000.0,
        prove_elapsed.as_secs_f64() * 1_000.0,
        verify_elapsed.as_secs_f64() * 1_000.0
    );
    Ok(())
}

fn production_spark_config(
    shape: &R1csShape<spartan_whir::engine::F>,
) -> Result<PoseidonZkSetupConfig, Box<dyn Error>> {
    let tables = preprocess_spark_tables(shape).map_err(protocol_error)?;
    let value_variables = tables.value_domain_size.ilog2() as usize;
    let fixed_value_variables = value_variables + spartan_whir::protocol::fixed_value_column_bits();
    let audit_variables = tables
        .row_memory_size
        .max(tables.col_memory_size)
        .next_power_of_two()
        .ilog2() as usize
        + spartan_whir::protocol::fixed_audit_column_bits();
    let read_group_columns =
        read_table_group_column_counts::<QuinticExtension>().map_err(protocol_error)?;
    let read_variables = read_group_columns
        .iter()
        .map(|columns| value_variables + columns.next_power_of_two().ilog2() as usize)
        .max()
        .ok_or_else(|| io::Error::other("SPARK read groups are empty"))?;
    let mut read = recommended_quintic_spark_read_whir_params(read_variables);
    read.round_log_inv_rates.clear();

    Ok(PoseidonZkSetupConfig {
        matrix_closing: MatrixClosingMode::Spark,
        security: SecurityConfig {
            security_level_bits: 116,
            merkle_security_bits: 116,
            soundness_assumption: SoundnessAssumption::JohnsonBound,
        },
        whir_params: recommended_quintic_spark_zk_whir_params(
            shape.num_vars.next_power_of_two().ilog2() as usize,
        ),
        spark_whir_params: Some(SparkWhirParams {
            fixed_value: recommended_quintic_spark_fixed_whir_params(fixed_value_variables),
            fixed_audit: recommended_quintic_spark_fixed_whir_params(audit_variables),
            read,
        }),
        ell_zk: DEFAULT_ZK_ELL,
        mask_log_inv_rate: DEFAULT_ZK_MASK_LOG_INV_RATE,
    })
}

fn proof_shape(proof: &ZkSpartanProof<QuinticExtension>) -> serde_json::Value {
    let ZkMatrixClosingProof::Spark(closing) = &proof.matrix_closing else {
        unreachable!("production fixture uses SPARK")
    };
    let product_shape = |proof: &spartan_whir::SparkBatchedProductProof<QuinticExtension>| {
        json!({
            "product_roots": proof.product_roots.len(),
            "dotproduct_claims": proof.dotproduct_claims.len(),
            "layers": proof.layers.len(),
            "rounds_per_layer": proof.layers.iter().map(|layer| layer.rounds.len()).collect::<Vec<_>>(),
            "round_widths": proof.layers.iter().map(|layer| layer.rounds.iter().map(|round| round.0.len()).collect::<Vec<_>>()).collect::<Vec<_>>(),
        })
    };
    json!({
        "outer_rounds": proof.outer_sumcheck.rounds.len(),
        "outer_round_widths": proof.outer_sumcheck.rounds.iter().map(Vec::len).collect::<Vec<_>>(),
        "outer_mask_evals": proof.outer_mask_evals.len(),
        "inner_rounds": proof.inner_sumcheck.round_coefficients.len(),
        "inner_round_widths": proof.inner_sumcheck.round_coefficients.iter().map(Vec::len).collect::<Vec<_>>(),
        "spark": {
            "matrix_evals": closing.spark_products.matrix_evals.len(),
            "operations_product": product_shape(&closing.spark_products.proof_ops),
            "memory_product": product_shape(&closing.spark_products.proof_mem),
            "fixed_value_variables": closing.spark_fixed_openings.value_num_variables,
            "fixed_audit_variables": closing.spark_fixed_openings.audit_num_variables,
            "fixed_audit_present": closing.spark_fixed_openings.audit_proof.is_some(),
            "read_groups": closing.spark_read_openings.groups.iter().map(|group| json!({
                "variables": group.num_variables,
                "column_start": group.column_start,
                "column_count": group.column_count,
                "batches": group.evals.len(),
                "batch_widths": group.evals.iter().map(Vec::len).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
        },
        "relation_sumchecks": proof.pcs_proof.sumchecks.len(),
        "relation_sumcheck_rounds": proof.pcs_proof.sumchecks.iter().map(|sumcheck| sumcheck.round_coefficients.len()).collect::<Vec<_>>(),
        "relation_rounds": proof.pcs_proof.rounds.len(),
        "relation_ood_answers": proof.pcs_proof.rounds.iter().map(|round| round.ood_answers.len()).collect::<Vec<_>>(),
        "base_case_blinded_message": proof.pcs_proof.base_case.blinded_message.len(),
        "base_case_blinded_randomness": proof.pcs_proof.base_case.blinded_randomness.len(),
        "base_case_blinded_masks": proof.pcs_proof.base_case.blinded_masks.len(),
        "base_case_mask_openings": proof.pcs_proof.base_case.carried_mask_openings.len(),
    })
}

fn write_mutations(
    honest_words: &[u32],
    instance: &spartan_whir::R1csInstance<spartan_whir::engine::F, ZkCommitment>,
    proof: &ZkSpartanProof<QuinticExtension>,
    vk: &PoseidonZkVerifyingKey<QuinticExtension>,
) -> Result<Vec<serde_json::Value>, Box<dyn Error>> {
    let mut mutations = Vec::new();
    mutations.push(write_verified_mutation(
        honest_words,
        instance,
        proof,
        vk,
        "changed_public_input",
        "public statement",
        |instance, _| instance.public_inputs[0] += spartan_whir::engine::F::ONE,
    )?);
    mutations.push(write_verified_mutation(
        honest_words,
        instance,
        proof,
        vk,
        "changed_application_mask_commitment",
        "application mask commitment",
        |_, proof| {
            let mut roots = proof.application_mask_commitment.roots().to_vec();
            roots[0][0] += spartan_whir::engine::F::ONE;
            proof.application_mask_commitment = MerkleCap::new(roots);
        },
    )?);
    mutations.push(write_verified_mutation(
        honest_words,
        instance,
        proof,
        vk,
        "changed_spark_product_sumcheck",
        "SPARK product sumcheck",
        |_, proof| {
            let ZkMatrixClosingProof::Spark(closing) = &mut proof.matrix_closing else {
                unreachable!("production fixture uses SPARK")
            };
            closing.spark_products.proof_ops.layers[1].rounds[0].0[0] += QuinticExtension::ONE;
        },
    )?);
    mutations.push(write_verified_mutation(
        honest_words,
        instance,
        proof,
        vk,
        "changed_spark_fixed_opening",
        "SPARK fixed-table opening",
        |_, proof| {
            let ZkMatrixClosingProof::Spark(closing) = &mut proof.matrix_closing else {
                unreachable!("production fixture uses SPARK")
            };
            closing.spark_fixed_openings.evals.val_a_low += QuinticExtension::ONE;
        },
    )?);
    mutations.push(write_verified_mutation(
        honest_words,
        instance,
        proof,
        vk,
        "changed_spark_read_opening",
        "SPARK read-table opening",
        |_, proof| {
            let ZkMatrixClosingProof::Spark(closing) = &mut proof.matrix_closing else {
                unreachable!("production fixture uses SPARK")
            };
            closing.spark_read_openings.groups[0].evals[0][0] += QuinticExtension::ONE;
        },
    )?);
    mutations.push(write_verified_mutation(
        honest_words,
        instance,
        proof,
        vk,
        "changed_hiding_whir_relation",
        "hiding-WHIR relation",
        |_, proof| proof.pcs_proof.base_case.masked_claim += QuinticExtension::ONE,
    )?);
    mutations.push(write_verified_mutation(
        honest_words,
        instance,
        proof,
        vk,
        "changed_relation_sumcheck",
        "hiding-WHIR relation sumcheck",
        |_, proof| {
            proof.pcs_proof.sumchecks[0].round_coefficients[0][0] += QuinticExtension::ONE;
        },
    )?);
    mutations.push(write_verified_mutation(
        honest_words,
        instance,
        proof,
        vk,
        "changed_code_switch_commitment",
        "hiding-WHIR code-switch commitment",
        |_, proof| {
            let mut roots = proof.pcs_proof.rounds[0].commitment.roots().to_vec();
            roots[0][0] += spartan_whir::engine::F::ONE;
            proof.pcs_proof.rounds[0].commitment = MerkleCap::new(roots);
        },
    )?);
    mutations.push(write_verified_mutation(
        honest_words,
        instance,
        proof,
        vk,
        "changed_code_switch_opening",
        "hiding-WHIR code-switch opening",
        |_, proof| match &mut proof.pcs_proof.rounds[0].openings {
            QueryOpenings::Base(opening) => {
                opening.rows[0][0] += spartan_whir::engine::F::ONE;
            }
            QueryOpenings::Extension(opening) => {
                opening.rows[0][0] += QuinticExtension::ONE;
            }
        },
    )?);
    mutations.push(write_verified_mutation(
        honest_words,
        instance,
        proof,
        vk,
        "changed_mask_reveal",
        "hiding-WHIR mask reveal",
        |_, proof| {
            proof.pcs_proof.base_case.blinded_masks[0].message[0] += QuinticExtension::ONE;
        },
    )?);
    mutations.push(write_verified_mutation(
        honest_words,
        instance,
        proof,
        vk,
        "changed_mask_opening",
        "hiding-WHIR mask opening",
        |_, proof| {
            proof.pcs_proof.base_case.carried_mask_openings[0].rows[0][0] += QuinticExtension::ONE;
        },
    )?);
    mutations.push(write_verified_mutation(
        honest_words,
        instance,
        proof,
        vk,
        "changed_source_opening",
        "hiding-WHIR source opening",
        |_, proof| match &mut proof.pcs_proof.base_case.source_openings {
            QueryOpenings::Base(opening) => {
                opening.rows[0][0] += spartan_whir::engine::F::ONE;
            }
            QueryOpenings::Extension(opening) => {
                opening.rows[0][0] += QuinticExtension::ONE;
            }
        },
    )?);

    let canonical = encode_full_zk_spark_guest_words(instance, proof)?;
    for (name, field, mutate) in [
        ("wrong_version", "encoding version", (4usize, 3u32)),
        ("wrong_hash_profile", "hash profile", (5usize, 1u32)),
        ("wrong_matrix_closing", "matrix-closing tag", (7usize, 0u32)),
    ] {
        let mut changed = canonical.clone();
        changed[mutate.0] = mutate.1;
        let padded = pad_malformed_words(changed)?;
        mutations.push(mutation_record(
            honest_words,
            name,
            field,
            "guest header check",
            &padded,
        )?);
    }
    let mut trailing = canonical.clone();
    trailing.push(0);
    trailing[8] = trailing.len() as u32;
    let padded = pad_malformed_words(trailing)?;
    mutations.push(mutation_record(
        honest_words,
        "trailing_word",
        "exact canonical proof consumption",
        "guest exact-consumption check",
        &padded,
    )?);
    let mut wrong_length = canonical;
    wrong_length[8] -= 1;
    let padded = pad_malformed_words(wrong_length)?;
    mutations.push(mutation_record(
        honest_words,
        "wrong_total_words",
        "canonical proof length",
        "guest exact-consumption check",
        &padded,
    )?);
    Ok(mutations)
}

fn write_verified_mutation<FN>(
    honest_words: &[u32],
    instance: &spartan_whir::R1csInstance<spartan_whir::engine::F, ZkCommitment>,
    proof: &ZkSpartanProof<QuinticExtension>,
    vk: &PoseidonZkVerifyingKey<QuinticExtension>,
    name: &str,
    field: &str,
    mutate: FN,
) -> Result<serde_json::Value, Box<dyn Error>>
where
    FN: FnOnce(
        &mut spartan_whir::R1csInstance<spartan_whir::engine::F, ZkCommitment>,
        &mut ZkSpartanProof<QuinticExtension>,
    ),
{
    let mut changed_instance = instance.clone();
    let mut changed_proof = proof.clone();
    mutate(&mut changed_instance, &mut changed_proof);
    let mut challenger = poseidon_zk_challenger();
    if PoseidonZkSpartanProtocol::<QuinticExtension>::verify(
        vk,
        &changed_instance,
        &changed_proof,
        &mut challenger,
    )
    .is_ok()
    {
        return Err(io::Error::other(format!("mutation {name} verified natively")).into());
    }
    let canonical = encode_full_zk_spark_guest_words(&changed_instance, &changed_proof)?;
    let words = pad_full_zk_guest_words(canonical)?;
    mutation_record(
        honest_words,
        name,
        field,
        "native verifier rejected the mutated proof",
        &words,
    )
}

fn mutation_record(
    honest_words: &[u32],
    name: &str,
    field: &str,
    preflight: &str,
    words: &[u32],
) -> Result<serde_json::Value, Box<dyn Error>> {
    if words.len() != honest_words.len() {
        return Err(io::Error::other(format!(
            "mutation {name} has {} words, expected {}",
            words.len(),
            honest_words.len()
        ))
        .into());
    }
    let patches = honest_words
        .iter()
        .zip(words)
        .enumerate()
        .filter_map(|(offset, (&before, &after))| {
            (before != after).then_some(json!({
                "offset": offset,
                "before": before,
                "after": after,
            }))
        })
        .collect::<Vec<_>>();
    if patches.is_empty() {
        return Err(io::Error::other(format!("mutation {name} changes no words")).into());
    }
    Ok(json!({
        "name": name,
        "field": field,
        "expected": "reject",
        "preflight": preflight,
        "patches": patches,
    }))
}

fn pad_malformed_words(mut words: Vec<u32>) -> Result<Vec<u32>, Box<dyn Error>> {
    if words.len() > MAX_FULL_ZK_GUEST_WORDS {
        return Err(io::Error::other("malformed fixture exceeds guest word limit").into());
    }
    words.resize(MAX_FULL_ZK_GUEST_WORDS, 0);
    Ok(words)
}

fn parse_guest_layout(words: &[u32]) -> Result<GuestLayout, Box<dyn Error>> {
    let mut reader = LayoutReader::new(words);
    reader.fixed("header.magic", 4)?;
    reader.word("header.version")?;
    reader.word("header.hash_profile")?;
    let extension_degree = reader.word("header.extension_degree")? as usize;
    if extension_degree != 5 {
        return Err(io::Error::other("production fixture must use extension degree 5").into());
    }
    reader.word("header.matrix_closing")?;
    let total_words = reader.word("header.total_words")? as usize;
    reader.base_vec("instance.public_inputs")?;
    reader.commitment("instance.witness_commitment")?;
    reader.commitment("proof.application_mask_commitment")?;
    reader.extension("proof.outer_sumcheck.mu_tilde")?;
    let outer_rounds = reader.length("proof.outer_sumcheck.rounds.count")?;
    for round in 0..outer_rounds {
        reader.extension_vec(&format!("proof.outer_sumcheck.rounds[{round}]"))?;
    }
    for claim in 0..3 {
        reader.extension(&format!("proof.outer_claims[{claim}]"))?;
    }
    reader.extension_vec("proof.outer_mask_evals")?;
    reader.zk_sumcheck("proof.inner_sumcheck")?;
    reader.commitment("proof.inner_sumcheck_mask_commitment")?;
    reader.word("proof.matrix_closing")?;
    reader.spark_closing("proof.spark")?;
    reader.relation_proof("proof.pcs")?;
    if reader.cursor != words.len() || total_words != words.len() {
        return Err(io::Error::other(format!(
            "layout consumed {} of {} words; header reports {total_words}",
            reader.cursor,
            words.len()
        ))
        .into());
    }
    Ok(GuestLayout {
        schema: "leanvm-full-zk-spark-word-layout-v1",
        total_words,
        entries: reader.entries,
    })
}

struct LayoutReader<'a> {
    words: &'a [u32],
    cursor: usize,
    entries: BTreeMap<String, WordRange>,
}

impl<'a> LayoutReader<'a> {
    fn new(words: &'a [u32]) -> Self {
        Self {
            words,
            cursor: 0,
            entries: BTreeMap::new(),
        }
    }

    fn record(&mut self, path: &str, offset: usize, words: usize) -> Result<(), Box<dyn Error>> {
        if self
            .entries
            .insert(path.to_owned(), WordRange { offset, words })
            .is_some()
        {
            return Err(io::Error::other(format!("duplicate layout path {path}")).into());
        }
        Ok(())
    }

    fn fixed(&mut self, path: &str, words: usize) -> Result<(), Box<dyn Error>> {
        let start = self.cursor;
        self.cursor = self
            .cursor
            .checked_add(words)
            .ok_or_else(|| io::Error::other("layout cursor overflow"))?;
        if self.cursor > self.words.len() {
            return Err(io::Error::other(format!("layout path {path} exceeds input")).into());
        }
        self.record(path, start, words)
    }

    fn word(&mut self, path: &str) -> Result<u32, Box<dyn Error>> {
        let value = *self
            .words
            .get(self.cursor)
            .ok_or_else(|| io::Error::other(format!("missing layout word {path}")))?;
        self.fixed(path, 1)?;
        Ok(value)
    }

    fn length(&mut self, path: &str) -> Result<usize, Box<dyn Error>> {
        Ok(self.word(path)? as usize)
    }

    fn base(&mut self, path: &str) -> Result<(), Box<dyn Error>> {
        self.fixed(path, 1)
    }

    fn extension(&mut self, path: &str) -> Result<(), Box<dyn Error>> {
        self.fixed(path, 5)
    }

    fn base_vec(&mut self, path: &str) -> Result<(), Box<dyn Error>> {
        let count = self.length(&format!("{path}.count"))?;
        self.fixed(&format!("{path}.values"), count)
    }

    fn extension_vec(&mut self, path: &str) -> Result<(), Box<dyn Error>> {
        let count = self.length(&format!("{path}.count"))?;
        self.fixed(&format!("{path}.values"), count * 5)
    }

    fn commitment(&mut self, path: &str) -> Result<(), Box<dyn Error>> {
        let roots = self.length(&format!("{path}.roots_count"))?;
        self.fixed(&format!("{path}.roots"), roots * 8)
    }

    fn optional_commitment(&mut self, path: &str) -> Result<(), Box<dyn Error>> {
        match self.word(&format!("{path}.present"))? {
            0 => Ok(()),
            1 => self.commitment(path),
            tag => Err(io::Error::other(format!(
                "invalid optional commitment tag {tag} at {path}"
            ))
            .into()),
        }
    }

    fn multi_proof(&mut self, path: &str) -> Result<(), Box<dyn Error>> {
        let siblings = self.length(&format!("{path}.siblings_count"))?;
        self.fixed(&format!("{path}.siblings"), siblings * 8)
    }

    fn query_opening(&mut self, path: &str) -> Result<(), Box<dyn Error>> {
        let tag = self.word(&format!("{path}.tag"))?;
        match tag {
            0 => self.shared_base_opening(path),
            1 => self.shared_extension_opening(path),
            _ => Err(io::Error::other(format!("invalid query opening tag {tag} at {path}")).into()),
        }
    }

    fn shared_base_opening(&mut self, path: &str) -> Result<(), Box<dyn Error>> {
        let rows = self.length(&format!("{path}.rows_count"))?;
        for row in 0..rows {
            self.base_vec(&format!("{path}.rows[{row}]"))?;
        }
        self.multi_proof(path)
    }

    fn shared_extension_opening(&mut self, path: &str) -> Result<(), Box<dyn Error>> {
        let rows = self.length(&format!("{path}.rows_count"))?;
        for row in 0..rows {
            self.extension_vec(&format!("{path}.rows[{row}]"))?;
        }
        self.multi_proof(path)
    }

    fn plain_sumcheck(&mut self, path: &str) -> Result<(), Box<dyn Error>> {
        let rounds = self.length(&format!("{path}.rounds_count"))?;
        for round in 0..rounds {
            self.extension_vec(&format!("{path}.rounds[{round}]"))?;
        }
        self.base_vec(&format!("{path}.pow_witnesses"))
    }

    fn plain_whir(&mut self, path: &str) -> Result<(), Box<dyn Error>> {
        self.extension_vec(&format!("{path}.initial_ood_answers"))?;
        self.plain_sumcheck(&format!("{path}.initial_sumcheck"))?;
        let rounds = self.length(&format!("{path}.rounds_count"))?;
        for round in 0..rounds {
            let round_path = format!("{path}.rounds[{round}]");
            self.optional_commitment(&format!("{round_path}.commitment"))?;
            self.extension_vec(&format!("{round_path}.ood_answers"))?;
            self.base(&format!("{round_path}.pow_witness"))?;
            self.query_opening(&format!("{round_path}.openings"))?;
            self.plain_sumcheck(&format!("{round_path}.sumcheck"))?;
        }
        match self.word(&format!("{path}.final_poly.present"))? {
            0 => {}
            1 => self.extension_vec(&format!("{path}.final_poly"))?,
            tag => {
                return Err(io::Error::other(format!(
                    "invalid final polynomial tag {tag} at {path}"
                ))
                .into())
            }
        }
        self.base(&format!("{path}.final_pow_witness"))?;
        self.query_opening(&format!("{path}.final_openings"))?;
        match self.word(&format!("{path}.final_sumcheck.present"))? {
            0 => Ok(()),
            1 => self.plain_sumcheck(&format!("{path}.final_sumcheck")),
            tag => {
                Err(io::Error::other(format!("invalid final sumcheck tag {tag} at {path}")).into())
            }
        }
    }

    fn spark_batched_product(&mut self, path: &str) -> Result<(), Box<dyn Error>> {
        self.extension_vec(&format!("{path}.product_roots"))?;
        self.extension_vec(&format!("{path}.dotproduct_claims"))?;
        let layers = self.length(&format!("{path}.layers_count"))?;
        for layer in 0..layers {
            let layer_path = format!("{path}.layers[{layer}]");
            let rounds = self.length(&format!("{layer_path}.rounds_count"))?;
            for round in 0..rounds {
                self.extension_vec(&format!("{layer_path}.rounds[{round}]"))?;
            }
            self.extension_vec(&format!("{layer_path}.product_left_evals"))?;
            self.extension_vec(&format!("{layer_path}.product_right_evals"))?;
            self.extension_vec(&format!("{layer_path}.dotproduct_left_evals"))?;
            self.extension_vec(&format!("{layer_path}.dotproduct_right_evals"))?;
            self.extension_vec(&format!("{layer_path}.dotproduct_weight_evals"))?;
        }
        Ok(())
    }

    fn spark_closing(&mut self, path: &str) -> Result<(), Box<dyn Error>> {
        self.extension(&format!("{path}.products.beta"))?;
        self.extension(&format!("{path}.products.gamma"))?;
        for axis in ["row", "col"] {
            for root in ["init_root", "read_root", "write_root", "audit_root"] {
                self.extension(&format!("{path}.products.{axis}.{root}"))?;
            }
        }
        for matrix in 0..3 {
            self.extension(&format!("{path}.matrix_evals[{matrix}]"))?;
        }
        self.spark_batched_product(&format!("{path}.proof_ops"))?;
        self.spark_batched_product(&format!("{path}.proof_mem"))?;

        for field in [
            "value_num_variables",
            "value_column_bits",
            "audit_num_variables",
            "audit_column_bits",
        ] {
            self.word(&format!("{path}.fixed.{field}"))?;
        }
        self.commitment(&format!("{path}.fixed.value_commitment"))?;
        self.optional_commitment(&format!("{path}.fixed.audit_commitment"))?;
        for field in [
            "val_a_low",
            "val_a_high",
            "val_b_low",
            "val_b_high",
            "val_c_low",
            "val_c_high",
            "row_addr",
            "col_addr",
            "row_read_ts",
            "col_read_ts",
            "row_audit_ts",
            "col_audit_ts",
        ] {
            self.extension(&format!("{path}.fixed.evals.{field}"))?;
        }
        self.plain_whir(&format!("{path}.fixed.value_proof"))?;
        match self.word(&format!("{path}.fixed.audit_proof.present"))? {
            0 => {}
            1 => self.plain_whir(&format!("{path}.fixed.audit_proof"))?,
            tag => {
                return Err(io::Error::other(format!("invalid fixed audit proof tag {tag}")).into())
            }
        }

        let groups = self.length(&format!("{path}.read.groups_count"))?;
        for group in 0..groups {
            let group_path = format!("{path}.read.groups[{group}]");
            for field in ["num_variables", "column_start", "column_count"] {
                self.word(&format!("{group_path}.{field}"))?;
            }
            self.commitment(&format!("{group_path}.commitment"))?;
            let batches = self.length(&format!("{group_path}.evals.batches_count"))?;
            for batch in 0..batches {
                self.extension_vec(&format!("{group_path}.evals[{batch}]"))?;
            }
            self.plain_whir(&format!("{group_path}.proof"))?;
        }
        Ok(())
    }

    fn zk_sumcheck(&mut self, path: &str) -> Result<(), Box<dyn Error>> {
        self.word(&format!("{path}.ell_zk"))?;
        self.extension(&format!("{path}.mu_tilde"))?;
        let rounds = self.length(&format!("{path}.rounds_count"))?;
        for round in 0..rounds {
            self.extension_vec(&format!("{path}.rounds[{round}]"))?;
        }
        self.base_vec(&format!("{path}.pow_witnesses"))
    }

    fn relation_proof(&mut self, path: &str) -> Result<(), Box<dyn Error>> {
        let sumchecks = self.length(&format!("{path}.sumchecks_count"))?;
        for sumcheck in 0..sumchecks {
            self.zk_sumcheck(&format!("{path}.sumchecks[{sumcheck}]"))?;
        }
        let commitments = self.length(&format!("{path}.sumcheck_commitments_count"))?;
        for commitment in 0..commitments {
            self.commitment(&format!("{path}.sumcheck_commitments[{commitment}]"))?;
        }
        let rounds = self.length(&format!("{path}.rounds_count"))?;
        for round in 0..rounds {
            let round_path = format!("{path}.rounds[{round}]");
            self.commitment(&format!("{round_path}.commitment"))?;
            self.commitment(&format!("{round_path}.mask_commitment"))?;
            self.extension_vec(&format!("{round_path}.ood_answers"))?;
            self.base(&format!("{round_path}.pow_witness"))?;
            self.query_opening(&format!("{round_path}.openings"))?;
        }
        self.base_case(&format!("{path}.base_case"))
    }

    fn base_case(&mut self, path: &str) -> Result<(), Box<dyn Error>> {
        self.commitment(&format!("{path}.fresh_main_commitment"))?;
        let commitments = self.length(&format!("{path}.fresh_mask_commitments_count"))?;
        for commitment in 0..commitments {
            self.commitment(&format!("{path}.fresh_mask_commitments[{commitment}]"))?;
        }
        self.extension(&format!("{path}.masked_claim"))?;
        self.extension_vec(&format!("{path}.blinded_message"))?;
        self.extension_vec(&format!("{path}.blinded_randomness"))?;
        let masks = self.length(&format!("{path}.blinded_masks_count"))?;
        for mask in 0..masks {
            self.extension_vec(&format!("{path}.blinded_masks[{mask}].message"))?;
            self.extension_vec(&format!("{path}.blinded_masks[{mask}].randomness"))?;
        }
        self.base(&format!("{path}.pow_witness"))?;
        self.query_opening(&format!("{path}.source_openings"))?;
        self.shared_extension_opening(&format!("{path}.fresh_main_openings"))?;
        if self.words[4] == 1 {
            let openings = self.length(&format!("{path}.mask_openings_count"))?;
            for opening in 0..openings {
                self.shared_extension_opening(&format!("{path}.mask_openings[{opening}].carried"))?;
                self.shared_extension_opening(&format!("{path}.mask_openings[{opening}].fresh"))?;
            }
        } else {
            let carried = self.length(&format!("{path}.carried_mask_openings_count"))?;
            for opening in 0..carried {
                self.shared_extension_opening(&format!("{path}.carried_mask_openings[{opening}]"))?;
            }
            let fresh = self.length(&format!("{path}.fresh_mask_openings_count"))?;
            for opening in 0..fresh {
                self.shared_extension_opening(&format!("{path}.fresh_mask_openings[{opening}]"))?;
            }
        }
        Ok(())
    }
}

fn canonical(values: &[spartan_whir::engine::F]) -> Vec<u32> {
    values.iter().map(PrimeField32::as_canonical_u32).collect()
}

fn write_words(output: &Path, name: &str, words: &[u32]) -> Result<Artifact, Box<dyn Error>> {
    let path = output.join(name);
    let mut file = File::create(&path)?;
    for word in words {
        file.write_all(&word.to_le_bytes())?;
    }
    artifact(output, &path)
}

fn write_json<T: Serialize>(
    output: &Path,
    name: &str,
    value: &T,
) -> Result<Artifact, Box<dyn Error>> {
    let path = output.join(name);
    let mut file = File::create(&path)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    artifact(output, &path)
}

fn artifact(output: &Path, path: &Path) -> Result<Artifact, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    Ok(Artifact {
        file: path
            .strip_prefix(output)?
            .to_string_lossy()
            .replace('\\', "/"),
        bytes: bytes.len(),
        sha256: format!("{:x}", Sha256::digest(&bytes)),
    })
}

fn usage() -> ! {
    eprintln!("usage: leanvm-full-zk-spark-fixture <output-directory> [--circuit sha256|privacy-pools] [--sample N] [--setup-config JSON]");
    process::exit(2)
}

fn protocol_error(error: spartan_whir::SpartanWhirError) -> io::Error {
    io::Error::other(error.to_string())
}
