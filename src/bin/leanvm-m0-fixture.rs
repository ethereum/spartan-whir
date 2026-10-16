use std::{
    env,
    error::Error,
    fs, io,
    path::{Path, PathBuf},
    process,
};

use p3_field::{Field, PrimeCharacteristicRing, PrimeField32, TwoAdicField};
use p3_koala_bear::{
    KOALABEAR_POSEIDON2_RC_16_EXTERNAL_FINAL, KOALABEAR_POSEIDON2_RC_16_EXTERNAL_INITIAL,
    KOALABEAR_POSEIDON2_RC_16_INTERNAL, KOALABEAR_POSEIDON2_RC_24_EXTERNAL_FINAL,
    KOALABEAR_POSEIDON2_RC_24_EXTERNAL_INITIAL, KOALABEAR_POSEIDON2_RC_24_INTERNAL,
};
use p3_merkle_tree::MerkleCap;
use p3_whir::pcs::proof::QueryOpenings;
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use spartan_whir::{
    control_statement_digest, control_statement_digest_preimage, decode_control_guest_input,
    encode_control_guest_input, generate_satisfiable_fixture, verify_control_with_trace,
    MatrixClosingMode, PoseidonProof, PoseidonSpartanProof, PoseidonSpartanProtocol,
    PoseidonTranscriptEvent, QuinticExtension, SecurityConfig, SoundnessAssumption,
    SpartanProofKind, SpartanSnarkConfig, SyntheticR1csConfig, WhirParams,
    LEANVM_CONTROL_PROFILE_ID, LEANVM_CONTROL_PROFILE_NUMBER, LEANVM_CONTROL_STATEMENT_DIGEST_ID,
    LEANVM_CONTROL_STATEMENT_SCHEMA_ID, LEANVM_GUEST_INPUT_VERSION, MAX_CONTROL_GUEST_WORDS,
};

type ControlProof = PoseidonProof<QuinticExtension>;
type ControlDirectProof = PoseidonSpartanProof<QuinticExtension>;

#[derive(Debug, Serialize)]
struct ArtifactRecord {
    file: String,
    sha256: String,
    bytes: usize,
}

#[derive(Debug, Serialize)]
struct MutationRecord {
    target: &'static str,
    artifact: ArtifactRecord,
    expected_result: String,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = env::args_os().skip(1);
    let output_dir = args.next().map(PathBuf::from).unwrap_or_else(|| usage());
    if args.next().is_some() {
        usage();
    }
    fs::create_dir_all(&output_dir)?;
    fs::create_dir_all(output_dir.join("mutations"))?;

    let fixture_config = SyntheticR1csConfig {
        target_log2_witness_poly: 2,
        num_constraints: 2,
        num_io: 1,
        a_terms_per_constraint: 2,
        b_terms_per_constraint: 2,
        seed: 0xC11E_1715_7A7E,
    };
    let fixture = generate_satisfiable_fixture(&fixture_config).map_err(protocol_error)?;
    let control_config = control_config();
    let (pk, vk) = PoseidonSpartanProtocol::<QuinticExtension>::setup_with_config(
        &fixture.shape,
        &control_config,
    )
    .map_err(protocol_error)?;
    let proof = pk
        .prove(fixture.witness, fixture.public_inputs.clone())
        .map_err(protocol_error)?;
    vk.verify(&fixture.public_inputs, &proof)
        .map_err(protocol_error)?;

    let guest_words = encode_control_guest_input(&proof).map_err(protocol_error)?;
    let decoded = decode_control_guest_input(&guest_words).map_err(protocol_error)?;
    vk.verify(&fixture.public_inputs, &decoded)
        .map_err(protocol_error)?;
    let reproduced = encode_control_guest_input(&decoded).map_err(protocol_error)?;
    if reproduced != guest_words {
        return Err(io::Error::other("control proof did not re-encode exactly").into());
    }

    let guest_input = write_words(&output_dir, "control_guest_input.words", &guest_words)?;
    let trace =
        verify_control_with_trace(&vk, &fixture.public_inputs, &proof).map_err(protocol_error)?;
    let trace_artifact = write_json(&output_dir, "control_transcript_trace.json", &trace)?;

    let guest_constants = control_guest_constants(&vk, &trace, &fixture.public_inputs)?;
    let guest_constants_artifact = write_json(
        &output_dir,
        "control_guest_constants.json",
        &guest_constants,
    )?;

    let digest_preimage = control_statement_digest_preimage(&fixture.public_inputs);
    let digest = control_statement_digest(&fixture.public_inputs);
    let statement = json!({
        "schema": LEANVM_CONTROL_STATEMENT_SCHEMA_ID,
        "inner_public_inputs": canonical(&fixture.public_inputs),
        "digest_algorithm": LEANVM_CONTROL_STATEMENT_DIGEST_ID,
        "digest_preimage": canonical(&digest_preimage),
        "digest": canonical(&digest),
    });
    let statement_artifact = write_json(&output_dir, "control_statement.json", &statement)?;

    let mut mutations = Vec::new();
    mutations.push(write_proof_mutation(
        &output_dir,
        &guest_words,
        &fixture.public_inputs,
        &vk,
        "changed_public_input",
        "application public input",
        |proof| proof.instance.public_inputs[0] += spartan_whir::engine::F::ONE,
    )?);
    mutations.push(write_proof_mutation(
        &output_dir,
        &guest_words,
        &fixture.public_inputs,
        &vk,
        "changed_witness_commitment",
        "witness commitment",
        |proof| {
            let mut roots = proof.instance.witness_commitment.roots().to_vec();
            roots[0][0] += spartan_whir::engine::F::ONE;
            proof.instance.witness_commitment = MerkleCap::new(roots);
        },
    )?);
    mutations.push(write_direct_mutation(
        &output_dir,
        &guest_words,
        &fixture.public_inputs,
        &vk,
        "changed_outer_sumcheck",
        "outer sumcheck",
        |proof| proof.outer_sumcheck.rounds[0].0[0] += QuinticExtension::ONE,
    )?);
    mutations.push(write_direct_mutation(
        &output_dir,
        &guest_words,
        &fixture.public_inputs,
        &vk,
        "changed_outer_claim",
        "outer claims",
        |proof| proof.outer_claims.0 += QuinticExtension::ONE,
    )?);
    mutations.push(write_direct_mutation(
        &output_dir,
        &guest_words,
        &fixture.public_inputs,
        &vk,
        "changed_inner_sumcheck",
        "inner sumcheck",
        |proof| proof.inner_sumcheck.rounds[0].0[0] += QuinticExtension::ONE,
    )?);
    mutations.push(write_direct_mutation(
        &output_dir,
        &guest_words,
        &fixture.public_inputs,
        &vk,
        "changed_witness_evaluation",
        "witness evaluation",
        |proof| proof.witness_eval += QuinticExtension::ONE,
    )?);
    mutations.push(write_direct_mutation(
        &output_dir,
        &guest_words,
        &fixture.public_inputs,
        &vk,
        "changed_whir_opening",
        "WHIR opening",
        |proof| match &mut proof.pcs_proof.final_openings {
            QueryOpenings::Base(opening) => {
                opening.rows[0][0] += spartan_whir::engine::F::ONE;
            }
            QueryOpenings::Extension(opening) => {
                opening.rows[0][0] += QuinticExtension::ONE;
            }
        },
    )?);

    let mut wrong_version = guest_words.clone();
    wrong_version[4] = LEANVM_GUEST_INPUT_VERSION + 1;
    mutations.push(write_decode_mutation(
        &output_dir,
        "wrong_version",
        "encoding version",
        &wrong_version,
    )?);
    let mut trailing = guest_words.clone();
    trailing.push(0);
    mutations.push(write_decode_mutation(
        &output_dir,
        "trailing_word",
        "exact proof consumption",
        &trailing,
    )?);

    let fixture_manifest = json!({
        "manifest_version": 1,
        "profile_id": LEANVM_CONTROL_PROFILE_ID,
        "profile_number": LEANVM_CONTROL_PROFILE_NUMBER,
        "guest_input_version": LEANVM_GUEST_INPUT_VERSION,
        "field": {
            "base": "KoalaBear",
            "modulus": spartan_whir::engine::F::ORDER_U32,
            "extension": "X^5 + X^2 - 1",
            "extension_coordinate_order": ["1", "X", "X^2", "X^3", "X^4"],
        },
        "privacy_mode": "no_zk",
        "matrix_closing": "direct_sparse",
        "security": control_config.security,
        "whir_params": control_config.whir_params,
        "synthetic_r1cs": {
            "target_log2_witness_poly": fixture_config.target_log2_witness_poly,
            "num_constraints": fixture_config.num_constraints,
            "num_io": fixture_config.num_io,
            "a_terms_per_constraint": fixture_config.a_terms_per_constraint,
            "b_terms_per_constraint": fixture_config.b_terms_per_constraint,
            "seed": fixture_config.seed,
        },
        "canonical_shape": {
            "constraints": vk.shape_canonical().num_cons,
            "variables": vk.shape_canonical().num_vars,
            "public_inputs": vk.shape_canonical().num_io,
        },
        "guest_input": guest_input,
        "guest_input_words": guest_words.len(),
        "guest_input_word_limit": MAX_CONTROL_GUEST_WORDS,
        "statement": statement_artifact,
        "guest_constants": guest_constants_artifact,
        "transcript_trace": trace_artifact,
        "transcript_events": trace.len(),
        "mutations": mutations,
    });
    write_json(
        &output_dir,
        "control_fixture_manifest.json",
        &fixture_manifest,
    )?;

    println!("wrote LeanVM M0 fixtures to {}", output_dir.display());
    Ok(())
}

fn control_guest_constants(
    verifying_key: &spartan_whir::PoseidonVerifyingKey<QuinticExtension>,
    trace: &[PoseidonTranscriptEvent],
    public_inputs: &[spartan_whir::engine::F],
) -> Result<serde_json::Value, Box<dyn Error>> {
    let spartan_domain_separator_len = verifying_key.domain_separator().to_bytes().len();
    let spartan_domain_separator = trace
        .get(..spartan_domain_separator_len)
        .ok_or_else(|| io::Error::other("control trace is missing the Spartan domain separator"))?
        .iter()
        .map(|event| match event {
            PoseidonTranscriptEvent::Observe { values } if values.len() == 1 => Ok(values[0]),
            _ => Err(io::Error::other(
                "control trace Spartan domain separator has an unexpected event",
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    match trace.get(spartan_domain_separator_len) {
        Some(PoseidonTranscriptEvent::Observe { values })
            if values == &canonical(public_inputs) => {}
        _ => {
            return Err(io::Error::other(
                "control trace public input checkpoint has an unexpected value",
            )
            .into());
        }
    }
    let whir_domain_separator = match trace.get(spartan_domain_separator_len + 1) {
        Some(PoseidonTranscriptEvent::Observe { values }) => values.clone(),
        _ => {
            return Err(io::Error::other(
                "control trace WHIR domain separator has an unexpected event",
            )
            .into());
        }
    };

    let inverse_power_of_two =
        |exponent: u64| spartan_whir::engine::F::TWO.exp_u64(exponent).inverse();
    let inv2 = inverse_power_of_two(1);
    let inv4 = inverse_power_of_two(2);
    let inv8 = inverse_power_of_two(3);
    let inv16 = inverse_power_of_two(4);
    let inv32 = inverse_power_of_two(5);
    let inv64 = inverse_power_of_two(6);
    let inv128 = inverse_power_of_two(7);
    let inv256 = inverse_power_of_two(8);
    let inv512 = inverse_power_of_two(9);
    let inv2_24 = inverse_power_of_two(24);
    let f = |value: i32| {
        if value >= 0 {
            spartan_whir::engine::F::from_u32(value as u32)
        } else {
            -spartan_whir::engine::F::from_u32((-value) as u32)
        }
    };
    let internal_diagonal_16 = [
        f(-2),
        f(1),
        f(2),
        inv2,
        f(3),
        f(4),
        -inv2,
        f(-3),
        f(-4),
        inv256,
        inv8,
        inv2_24,
        -inv256,
        -inv8,
        -inv16,
        -inv2_24,
    ];
    let internal_diagonal_24 = [
        f(-2),
        f(1),
        f(2),
        inv2,
        f(3),
        f(4),
        -inv2,
        f(-3),
        f(-4),
        inv256,
        inv4,
        inv8,
        inv16,
        inv32,
        inv64,
        inv2_24,
        -inv256,
        -inv8,
        -inv16,
        -inv32,
        -inv64,
        -inv128,
        -inv512,
        -inv2_24,
    ];
    let shape = verifying_key.shape_canonical();

    Ok(json!({
        "schema": "leanvm-control-guest-constants-v1",
        "spartan_domain_separator": spartan_domain_separator,
        "whir_domain_separator": whir_domain_separator,
        "canonical_shape": {
            "constraints": shape.num_cons,
            "variables": shape.num_vars,
            "public_inputs": shape.num_io,
            "a": canonical_matrix(&shape.a),
            "b": canonical_matrix(&shape.b),
            "c": canonical_matrix(&shape.c),
        },
        "poseidon2": {
            "width_16": {
                "external_initial": canonical_2d(&KOALABEAR_POSEIDON2_RC_16_EXTERNAL_INITIAL),
                "internal": canonical(&KOALABEAR_POSEIDON2_RC_16_INTERNAL),
                "external_final": canonical_2d(&KOALABEAR_POSEIDON2_RC_16_EXTERNAL_FINAL),
                "internal_diagonal": canonical(&internal_diagonal_16),
            },
            "width_24": {
                "external_initial": canonical_2d(&KOALABEAR_POSEIDON2_RC_24_EXTERNAL_INITIAL),
                "internal": canonical(&KOALABEAR_POSEIDON2_RC_24_INTERNAL),
                "external_final": canonical_2d(&KOALABEAR_POSEIDON2_RC_24_EXTERNAL_FINAL),
                "internal_diagonal": canonical(&internal_diagonal_24),
            },
        },
        "whir": {
            "num_variables": 2,
            "folding_factor": 1,
            "starting_domain_size": 256,
            "folded_domain_size": 128,
            "folded_domain_generator": spartan_whir::engine::F::two_adic_generator(7).as_canonical_u32(),
            "final_queries": 14,
            "uniform_query_bits": 7,
        },
    }))
}

fn canonical_matrix(
    matrix: &spartan_whir::SparseMatrix<spartan_whir::engine::F>,
) -> serde_json::Value {
    json!({
        "rows": matrix.num_rows,
        "columns": matrix.num_cols,
        "entries": matrix.entries.iter().map(|entry| json!({
            "row": entry.row,
            "column": entry.col,
            "value": entry.val.as_canonical_u32(),
        })).collect::<Vec<_>>(),
    })
}

fn canonical_2d<const ROWS: usize, const COLUMNS: usize>(
    values: &[[spartan_whir::engine::F; COLUMNS]; ROWS],
) -> Vec<Vec<u32>> {
    values.iter().map(|row| canonical(row)).collect()
}

fn control_config() -> SpartanSnarkConfig {
    SpartanSnarkConfig {
        matrix_closing: MatrixClosingMode::DirectSparse,
        security: SecurityConfig {
            security_level_bits: 80,
            merkle_security_bits: 80,
            soundness_assumption: SoundnessAssumption::CapacityBound,
        },
        whir_params: WhirParams {
            pow_bits: 0,
            folding_factor: 1,
            starting_log_inv_rate: 6,
            rs_domain_initial_reduction_factor: 1,
            ..WhirParams::default()
        },
        spark_whir_params: None,
    }
}

fn write_proof_mutation(
    output_dir: &Path,
    original: &[u32],
    public_inputs: &[spartan_whir::engine::F],
    verifying_key: &spartan_whir::PoseidonVerifyingKey<QuinticExtension>,
    file_stem: &str,
    target: &'static str,
    mutate: impl FnOnce(&mut ControlProof),
) -> Result<MutationRecord, Box<dyn Error>> {
    let mut proof = decode_control_guest_input(original).map_err(protocol_error)?;
    mutate(&mut proof);
    let words = encode_control_guest_input(&proof).map_err(protocol_error)?;
    let expected_result = verifying_key
        .verify(public_inputs, &proof)
        .err()
        .ok_or_else(|| io::Error::other(format!("{target} mutation was accepted")))?
        .to_string();
    let artifact = write_words(output_dir, &format!("mutations/{file_stem}.words"), &words)?;
    Ok(MutationRecord {
        target,
        artifact,
        expected_result,
    })
}

fn write_direct_mutation(
    output_dir: &Path,
    original: &[u32],
    public_inputs: &[spartan_whir::engine::F],
    verifying_key: &spartan_whir::PoseidonVerifyingKey<QuinticExtension>,
    file_stem: &str,
    target: &'static str,
    mutate: impl FnOnce(&mut ControlDirectProof),
) -> Result<MutationRecord, Box<dyn Error>> {
    write_proof_mutation(
        output_dir,
        original,
        public_inputs,
        verifying_key,
        file_stem,
        target,
        |proof| mutate(direct_mut(proof)),
    )
}

fn direct_mut(proof: &mut ControlProof) -> &mut ControlDirectProof {
    match &mut proof.proof {
        SpartanProofKind::Direct(proof) => proof,
        SpartanProofKind::Spark(_) => unreachable!("control decoder fixes DirectSparse"),
    }
}

fn write_decode_mutation(
    output_dir: &Path,
    file_stem: &str,
    target: &'static str,
    words: &[u32],
) -> Result<MutationRecord, Box<dyn Error>> {
    let expected_result = decode_control_guest_input(words)
        .err()
        .ok_or_else(|| io::Error::other(format!("{target} mutation decoded")))?
        .to_string();
    let artifact = write_words(output_dir, &format!("mutations/{file_stem}.words"), words)?;
    Ok(MutationRecord {
        target,
        artifact,
        expected_result,
    })
}

fn write_words(
    output_dir: &Path,
    relative: &str,
    words: &[u32],
) -> Result<ArtifactRecord, Box<dyn Error>> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(words));
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    write_bytes(output_dir, relative, &bytes)
}

fn write_json(
    output_dir: &Path,
    relative: &str,
    value: &impl Serialize,
) -> Result<ArtifactRecord, Box<dyn Error>> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    write_bytes(output_dir, relative, &bytes)
}

fn write_bytes(
    output_dir: &Path,
    relative: &str,
    bytes: &[u8],
) -> Result<ArtifactRecord, Box<dyn Error>> {
    let path = output_dir.join(relative);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&path, bytes)?;
    Ok(ArtifactRecord {
        file: relative.to_owned(),
        sha256: hex(&Sha256::digest(bytes)),
        bytes: bytes.len(),
    })
}

fn canonical(values: &[spartan_whir::engine::F]) -> Vec<u32> {
    values.iter().map(PrimeField32::as_canonical_u32).collect()
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(DIGITS[(byte >> 4) as usize] as char);
        out.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    out
}

fn protocol_error(error: spartan_whir::SpartanWhirError) -> io::Error {
    io::Error::other(error.to_string())
}

fn usage() -> ! {
    eprintln!("usage: leanvm-m0-fixture <output-dir>");
    process::exit(2);
}
