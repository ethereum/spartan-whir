use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_merkle_tree::MerkleCap;
use sha2::{Digest, Sha256};

use spartan_whir::{
    decode_control_guest_input, encode_control_guest_input, generate_satisfiable_fixture,
    leanvm::{
        control_statement_digest, control_verifying_key_id, verify_control_with_trace,
        LEANVM_CONTROL_STATEMENT_DIGEST_ID, LEANVM_CONTROL_STATEMENT_SCHEMA_ID,
        LEANVM_GUEST_INPUT_VERSION,
    },
    InvalidConfigReason, MatrixClosingMode, PoseidonSpartanProtocol, PoseidonTranscriptEvent,
    QuinticExtension, SecurityConfig, SoundnessAssumption, SpartanSnarkConfig, SpartanWhirError,
    SyntheticR1csConfig, WhirParams,
};

fn control_fixture() -> spartan_whir::SyntheticR1csFixture {
    generate_satisfiable_fixture(&SyntheticR1csConfig {
        target_log2_witness_poly: 2,
        num_constraints: 2,
        num_io: 1,
        a_terms_per_constraint: 2,
        b_terms_per_constraint: 2,
        seed: 0xC11E_1715_7A7E,
    })
    .expect("control fixture generation succeeds")
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

#[test]
fn control_codec_roundtrips_and_is_deterministic() {
    type Protocol = PoseidonSpartanProtocol<QuinticExtension>;

    let fixture = control_fixture();
    let (pk, vk) = Protocol::setup_with_config(&fixture.shape, &control_config())
        .expect("control setup succeeds");
    let proof_a = pk
        .prove(fixture.witness.clone(), fixture.public_inputs.clone())
        .expect("control proof succeeds");
    let proof_b = pk
        .prove(fixture.witness, fixture.public_inputs.clone())
        .expect("repeated control proof succeeds");

    vk.verify(&fixture.public_inputs, &proof_a)
        .expect("native proof verifies");
    let encoded_a = encode_control_guest_input(&proof_a).expect("proof encodes");
    let encoded_b = encode_control_guest_input(&proof_b).expect("repeated proof encodes");
    assert_eq!(
        encoded_a, encoded_b,
        "control proof words must reproduce exactly"
    );
    assert_eq!(
        encoded_a,
        checked_words(include_bytes!(
            "../testdata/leanvm-m0/control_guest_input.words"
        )),
        "checked control fixture must reproduce exactly"
    );

    let decoded = decode_control_guest_input(&encoded_a).expect("guest words decode");
    vk.verify(&fixture.public_inputs, &decoded)
        .expect("decoded proof verifies");
    assert_eq!(
        encode_control_guest_input(&decoded).expect("decoded proof re-encodes"),
        encoded_a
    );

    let mut wrong_commitment_shape = proof_b;
    let mut roots = wrong_commitment_shape
        .instance
        .witness_commitment
        .roots()
        .to_vec();
    roots.push(roots[0]);
    wrong_commitment_shape.instance.witness_commitment = MerkleCap::new(roots);
    assert_eq!(
        encode_control_guest_input(&wrong_commitment_shape).unwrap_err(),
        SpartanWhirError::InvalidCommitmentShape
    );

    assert_eq!(
        control_verifying_key_id(&vk).expect("control key ID derives"),
        control_verifying_key_id(&vk).expect("control key ID reproduces")
    );
    assert_ne!(
        control_statement_digest(&fixture.public_inputs),
        control_statement_digest(&[fixture.public_inputs[0] + spartan_whir::engine::F::ONE])
    );
}

#[test]
fn restored_control_verifying_key_requires_authentication_for_export() {
    type Protocol = PoseidonSpartanProtocol<QuinticExtension>;

    let fixture = control_fixture();
    let (_, vk) = Protocol::setup_with_config(&fixture.shape, &control_config())
        .expect("control setup succeeds");
    let mut restored_vk =
        bincode::deserialize(&bincode::serialize(&vk).expect("control verifying key serializes"))
            .expect("control verifying key deserializes");
    assert_eq!(
        control_verifying_key_id(&restored_vk),
        Err(SpartanWhirError::InvalidConfig(
            InvalidConfigReason::UnauthenticatedVerifyingKey
        ))
    );
    restored_vk
        .authenticate()
        .expect("restored control key authenticates");
    assert_eq!(
        control_verifying_key_id(&restored_vk).expect("restored control key ID derives"),
        control_verifying_key_id(&vk).expect("setup control key ID derives")
    );
}

#[test]
fn control_trace_reproduces_and_covers_transcript_operation_kinds() {
    type Protocol = PoseidonSpartanProtocol<QuinticExtension>;

    let fixture = control_fixture();
    let (pk, vk) = Protocol::setup_with_config(&fixture.shape, &control_config())
        .expect("control setup succeeds");
    let proof = pk
        .prove(fixture.witness, fixture.public_inputs.clone())
        .expect("control proof succeeds");

    let trace_a = verify_control_with_trace(&vk, &fixture.public_inputs, &proof)
        .expect("traced verification succeeds");
    let trace_b = verify_control_with_trace(&vk, &fixture.public_inputs, &proof)
        .expect("repeated traced verification succeeds");
    assert_eq!(trace_a, trace_b);
    let checked_trace: Vec<PoseidonTranscriptEvent> = serde_json::from_slice(include_bytes!(
        "../testdata/leanvm-m0/control_transcript_trace.json"
    ))
    .expect("checked trace decodes");
    assert_eq!(
        trace_a, checked_trace,
        "checked trace must reproduce exactly"
    );
    assert!(trace_a
        .iter()
        .any(|event| matches!(event, PoseidonTranscriptEvent::Observe { .. })));
    assert!(trace_a
        .iter()
        .any(|event| matches!(event, PoseidonTranscriptEvent::ObserveCommitment { .. })));
    assert!(trace_a
        .iter()
        .any(|event| matches!(event, PoseidonTranscriptEvent::Sample { .. })));
    assert!(trace_a
        .iter()
        .any(|event| matches!(event, PoseidonTranscriptEvent::SampleUniformBits { .. })));
    serde_json::to_vec(&trace_a).expect("trace serializes");
}

#[test]
fn control_codec_rejects_header_canonicality_length_and_proof_mutations() {
    type Protocol = PoseidonSpartanProtocol<QuinticExtension>;

    let fixture = control_fixture();
    let (pk, vk) = Protocol::setup_with_config(&fixture.shape, &control_config())
        .expect("control setup succeeds");
    let proof = pk
        .prove(fixture.witness, fixture.public_inputs.clone())
        .expect("control proof succeeds");
    let encoded = encode_control_guest_input(&proof).expect("proof encodes");

    let mut wrong_version = encoded.clone();
    wrong_version[4] = LEANVM_GUEST_INPUT_VERSION + 1;
    assert_eq!(
        decode_control_guest_input(&wrong_version)
            .err()
            .expect("wrong version rejects"),
        SpartanWhirError::InvalidBlobHeader
    );

    let mut trailing = encoded.clone();
    trailing.push(0);
    assert_eq!(
        decode_control_guest_input(&trailing)
            .err()
            .expect("trailing word rejects"),
        SpartanWhirError::TrailingBytes
    );

    let mut noncanonical = encoded.clone();
    noncanonical[8] = spartan_whir::engine::F::ORDER_U32;
    assert_eq!(
        decode_control_guest_input(&noncanonical)
            .err()
            .expect("non-canonical word rejects"),
        SpartanWhirError::NonCanonicalEncoding
    );

    assert!(decode_control_guest_input(&encoded[..encoded.len() - 1]).is_err());

    let mut changed_statement = encoded.clone();
    changed_statement[8] = (changed_statement[8] + 1) % spartan_whir::engine::F::ORDER_U32;
    let changed_statement =
        decode_control_guest_input(&changed_statement).expect("changed statement decodes");
    assert_eq!(
        vk.verify(&fixture.public_inputs, &changed_statement)
            .unwrap_err(),
        SpartanWhirError::PublicInputMismatch
    );

    let mut changed_outer_round = encoded;
    changed_outer_round[20] = (changed_outer_round[20] + 1) % spartan_whir::engine::F::ORDER_U32;
    let changed_outer_round =
        decode_control_guest_input(&changed_outer_round).expect("changed proof decodes");
    assert!(vk
        .verify(&fixture.public_inputs, &changed_outer_round)
        .is_err());
}

#[test]
fn checked_manifest_artifacts_and_mutations_are_consistent() {
    type Protocol = PoseidonSpartanProtocol<QuinticExtension>;

    let fixture = control_fixture();
    let (_, vk) = Protocol::setup_with_config(&fixture.shape, &control_config())
        .expect("control setup succeeds");
    let manifest: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../testdata/leanvm-m0/control_fixture_manifest.json"
    ))
    .expect("control fixture manifest decodes");
    let fixture_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/leanvm-m0");

    check_artifact(&fixture_dir, &manifest["guest_input"]);
    let statement_bytes = check_artifact(&fixture_dir, &manifest["statement"]);
    check_artifact(&fixture_dir, &manifest["transcript_trace"]);
    let statement: serde_json::Value =
        serde_json::from_slice(&statement_bytes).expect("control statement decodes");
    assert_eq!(
        statement["schema"].as_str().unwrap(),
        LEANVM_CONTROL_STATEMENT_SCHEMA_ID
    );
    assert_eq!(
        statement["digest_algorithm"].as_str().unwrap(),
        LEANVM_CONTROL_STATEMENT_DIGEST_ID
    );
    let mutations = manifest["mutations"]
        .as_array()
        .expect("mutations are an array");
    assert_eq!(mutations.len(), 9);
    for mutation in mutations {
        let bytes = check_artifact(&fixture_dir, &mutation["artifact"]);
        let words = checked_words(&bytes);
        let actual = match decode_control_guest_input(&words) {
            Ok(proof) => vk
                .verify(&fixture.public_inputs, &proof)
                .expect_err("proof mutation rejects")
                .to_string(),
            Err(error) => error.to_string(),
        };
        assert_eq!(
            actual,
            mutation["expected_result"].as_str().unwrap(),
            "wrong rejection for {}",
            mutation["target"].as_str().unwrap()
        );
    }
}

fn check_artifact(fixture_dir: &std::path::Path, artifact: &serde_json::Value) -> Vec<u8> {
    let file = artifact["file"]
        .as_str()
        .expect("artifact file is a string");
    let bytes = std::fs::read(fixture_dir.join(file)).expect("artifact reads");
    assert_eq!(bytes.len(), artifact["bytes"].as_u64().unwrap() as usize);
    assert_eq!(sha256_hex(&bytes), artifact["sha256"].as_str().unwrap());
    bytes
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn checked_words(bytes: &[u8]) -> Vec<u32> {
    assert_eq!(bytes.len() % size_of::<u32>(), 0);
    bytes
        .chunks_exact(size_of::<u32>())
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four-byte word")))
        .collect()
}
