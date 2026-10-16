use std::{
    env,
    error::Error,
    fs::{self, File},
    io::{self, Write},
    path::{Path, PathBuf},
    process,
};

use p3_challenger::{CanFinalizeDigest, CanObserve};
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_merkle_tree::MerkleCap;
use rand::{rngs::StdRng, SeedableRng};
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use spartan_whir::{
    encode_full_zk_direct_guest_words, full_zk_direct_guest_verifier_config,
    generate_satisfiable_fixture, MatrixClosingMode, PoseidonZkSetupConfig, QuinticExtension,
    SecurityConfig, SoundnessAssumption, SyntheticR1csConfig, WhirParams, DEFAULT_ZK_ELL,
    DEFAULT_ZK_MASK_LOG_INV_RATE,
};

#[cfg(feature = "poseidon1")]
use spartan_whir::{
    poseidon1_challenger as poseidon_zk_challenger, setup_poseidon1_zk as setup_poseidon_zk,
    Poseidon1ZkCommitment as ZkCommitment, Poseidon1ZkSpartanProof as ZkSpartanProof,
    Poseidon1ZkSpartanProtocol as PoseidonZkSpartanProtocol,
    Poseidon1ZkVerifyingKey as PoseidonZkVerifyingKey,
};
#[cfg(not(feature = "poseidon1"))]
use spartan_whir::{
    poseidon_zk_challenger, setup_poseidon_zk, PoseidonZkCommitment as ZkCommitment,
    PoseidonZkSpartanProtocol, PoseidonZkVerifyingKey, ZkSpartanProof,
};

const FIXTURE_SEED: u64 = 0x1EA0_F011_2A;
const PROOF_SEED: u64 = 0xF011_2A11_CE;

#[derive(Serialize)]
struct Artifact {
    file: String,
    bytes: usize,
    sha256: String,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = env::args_os().skip(1);
    let output = args.next().map(PathBuf::from).unwrap_or_else(|| usage());
    if args.next().is_some() {
        usage();
    }
    fs::create_dir_all(&output)?;
    fs::create_dir_all(output.join("mutations"))?;

    let fixture_config = SyntheticR1csConfig {
        target_log2_witness_poly: 3,
        num_constraints: 4,
        num_io: 1,
        a_terms_per_constraint: 2,
        b_terms_per_constraint: 2,
        seed: FIXTURE_SEED,
    };
    let fixture = generate_satisfiable_fixture(&fixture_config).map_err(protocol_error)?;
    let config = PoseidonZkSetupConfig {
        matrix_closing: MatrixClosingMode::DirectSparse,
        security: SecurityConfig {
            security_level_bits: 80,
            merkle_security_bits: 80,
            soundness_assumption: SoundnessAssumption::CapacityBound,
        },
        whir_params: WhirParams {
            pow_bits: 0,
            folding_factor: 1,
            starting_log_inv_rate: 8,
            rs_domain_initial_reduction_factor: 1,
            ..WhirParams::default()
        },
        spark_whir_params: None,
        ell_zk: DEFAULT_ZK_ELL,
        mask_log_inv_rate: DEFAULT_ZK_MASK_LOG_INV_RATE,
    };
    let (pk, vk) = setup_poseidon_zk::<QuinticExtension>(fixture.shape, config.clone())
        .map_err(protocol_error)?;
    let mut prover_challenger = poseidon_zk_challenger();
    let mut rng = StdRng::seed_from_u64(PROOF_SEED);
    let (instance, proof) = PoseidonZkSpartanProtocol::<QuinticExtension>::prove_with_rng(
        &pk,
        &fixture.public_inputs,
        &fixture.witness,
        &mut prover_challenger,
        &mut rng,
    )
    .map_err(protocol_error)?;
    let mut verifier_challenger = poseidon_zk_challenger().with_trace();
    PoseidonZkSpartanProtocol::<QuinticExtension>::verify(
        &vk,
        &instance,
        &proof,
        &mut verifier_challenger,
    )
    .map_err(protocol_error)?;

    let words = encode_full_zk_direct_guest_words(&instance, &proof)?;
    let input_artifact = write_words(&output, "full_zk_direct_guest_input.words", &words)?;
    let vk_artifact = write_json(&output, "full_zk_direct_verifying_key.json", &vk)?;
    let verifier_config = full_zk_direct_guest_verifier_config(&vk).map_err(protocol_error)?;
    let verifier_config_artifact = write_json(
        &output,
        "full_zk_direct_guest_constants.json",
        &verifier_config,
    )?;
    let trace = verifier_challenger.transcript_trace();
    let trace_artifact = write_json(&output, "full_zk_direct_transcript_trace.json", &trace)?;

    let mut digest_challenger = poseidon_zk_challenger();
    digest_challenger.observe_slice(&[
        spartan_whir::engine::F::from_u32(1),
        instance.public_inputs[0],
    ]);
    let statement_digest = digest_challenger.finalize();
    let statement_artifact = write_json(
        &output,
        "full_zk_direct_statement.json",
        &json!({
            "schema": "full-zk-synthetic-inputs-v1",
            "inner_public_inputs": instance.public_inputs.iter().map(PrimeField32::as_canonical_u32).collect::<Vec<_>>(),
            "digest_preimage": [1, instance.public_inputs[0].as_canonical_u32()],
            "digest": statement_digest.iter().map(PrimeField32::as_canonical_u32).collect::<Vec<_>>(),
        }),
    )?;

    let mutation_artifacts = write_mutations(&output, &instance, &proof, &vk)?;
    let manifest = json!({
        "schema": "leanvm-full-zk-direct-fixture-v1",
        "hash_profile": if cfg!(feature = "poseidon1") { "poseidon1-width16" } else { "poseidon2-width16-width24" },
        "matrix_closing": "direct_sparse",
        "extension_degree": 5,
        "fixture_seed": FIXTURE_SEED,
        "proof_seed": PROOF_SEED,
        "fixture": {
            "target_log2_witness_poly": fixture_config.target_log2_witness_poly,
            "num_constraints": fixture_config.num_constraints,
            "num_io": fixture_config.num_io,
            "a_terms_per_constraint": fixture_config.a_terms_per_constraint,
            "b_terms_per_constraint": fixture_config.b_terms_per_constraint,
            "seed": fixture_config.seed,
        },
        "setup": config,
        "guest_words": words.len(),
        "transcript_events": trace.len(),
        "proof_shape": proof_shape(&proof),
        "artifacts": {
            "guest_input": input_artifact,
            "verifying_key": vk_artifact,
            "guest_constants": verifier_config_artifact,
            "transcript_trace": trace_artifact,
            "statement": statement_artifact,
            "mutations": mutation_artifacts,
        },
    });
    write_json(&output, "full_zk_direct_manifest.json", &manifest)?;
    println!("wrote {} canonical guest words", words.len());
    println!("recorded {} transcript events", trace.len());
    Ok(())
}

fn write_mutations(
    output: &Path,
    instance: &spartan_whir::R1csInstance<spartan_whir::engine::F, ZkCommitment>,
    proof: &ZkSpartanProof<QuinticExtension>,
    vk: &PoseidonZkVerifyingKey<QuinticExtension>,
) -> Result<Vec<Artifact>, Box<dyn Error>> {
    let mut cases = Vec::new();

    let mut changed_instance = instance.clone();
    changed_instance.public_inputs[0] += spartan_whir::engine::F::ONE;
    cases.push(("changed_public_input", changed_instance, proof.clone()));

    let mut changed_instance = instance.clone();
    let mut roots = changed_instance.witness_commitment.roots().to_vec();
    roots[0][0] += spartan_whir::engine::F::ONE;
    changed_instance.witness_commitment = MerkleCap::new(roots);
    cases.push((
        "changed_witness_commitment",
        changed_instance,
        proof.clone(),
    ));

    let mut changed_proof = proof.clone();
    let mut roots = changed_proof.application_mask_commitment.roots().to_vec();
    roots[0][0] += spartan_whir::engine::F::ONE;
    changed_proof.application_mask_commitment = MerkleCap::new(roots);
    cases.push((
        "changed_application_mask_commitment",
        instance.clone(),
        changed_proof,
    ));

    let mut changed_proof = proof.clone();
    changed_proof.outer_sumcheck.rounds[0][0] += QuinticExtension::ONE;
    cases.push(("changed_outer_sumcheck", instance.clone(), changed_proof));

    let mut changed_proof = proof.clone();
    changed_proof.inner_sumcheck.round_coefficients[0][0] += QuinticExtension::ONE;
    cases.push(("changed_inner_sumcheck", instance.clone(), changed_proof));

    let mut changed_proof = proof.clone();
    changed_proof.pcs_proof.base_case.masked_claim += QuinticExtension::ONE;
    cases.push((
        "changed_hiding_whir_base_claim",
        instance.clone(),
        changed_proof,
    ));

    let mut artifacts = Vec::new();
    for (name, changed_instance, changed_proof) in cases {
        let mut challenger = poseidon_zk_challenger();
        if PoseidonZkSpartanProtocol::<QuinticExtension>::verify(
            vk,
            &changed_instance,
            &changed_proof,
            &mut challenger,
        )
        .is_ok()
        {
            return Err(io::Error::other(format!("mutation {name} verified")).into());
        }
        let words = encode_full_zk_direct_guest_words(&changed_instance, &changed_proof)?;
        artifacts.push(write_words(
            output,
            &format!("mutations/{name}.words"),
            &words,
        )?);
    }
    Ok(artifacts)
}

fn proof_shape(proof: &ZkSpartanProof<QuinticExtension>) -> serde_json::Value {
    json!({
        "outer_rounds": proof.outer_sumcheck.rounds.len(),
        "outer_round_widths": proof.outer_sumcheck.rounds.iter().map(Vec::len).collect::<Vec<_>>(),
        "outer_mask_evals": proof.outer_mask_evals.len(),
        "inner_rounds": proof.inner_sumcheck.round_coefficients.len(),
        "inner_round_widths": proof.inner_sumcheck.round_coefficients.iter().map(Vec::len).collect::<Vec<_>>(),
        "relation_sumchecks": proof.pcs_proof.sumchecks.len(),
        "relation_rounds": proof.pcs_proof.rounds.len(),
        "relation_ood_answers": proof.pcs_proof.rounds.iter().map(|round| round.ood_answers.len()).collect::<Vec<_>>(),
        "base_case_blinded_message": proof.pcs_proof.base_case.blinded_message.len(),
        "base_case_blinded_randomness": proof.pcs_proof.base_case.blinded_randomness.len(),
        "base_case_blinded_masks": proof.pcs_proof.base_case.blinded_masks.len(),
        "base_case_mask_openings": proof.pcs_proof.base_case.carried_mask_openings.len(),
    })
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
    eprintln!("usage: leanvm-full-zk-fixture <output-directory>");
    process::exit(2)
}

fn protocol_error(error: spartan_whir::SpartanWhirError) -> io::Error {
    io::Error::other(error.to_string())
}
