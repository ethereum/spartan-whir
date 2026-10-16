use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use p3_field::{PrimeCharacteristicRing, PrimeField32};
use sha2::{Digest, Sha256};
use spartan_whir::{
    decode_full_zk_direct_guest_words, decode_full_zk_spark_guest_words,
    encode_full_zk_direct_guest_words, encode_full_zk_spark_guest_words,
    full_zk_direct_guest_verifier_config, full_zk_spark_guest_verifier_config,
    full_zk_statement_digest, full_zk_statement_digest_id, full_zk_statement_digest_preimage,
    generate_satisfiable_fixture, pad_full_zk_guest_words, setup_poseidon_zk_for,
    FullZkGuestCodecError, FullZkGuestHashProfile, InvalidConfigReason, MatrixClosingMode,
    Plonky3PoseidonEngine, Poseidon1QuinticEngine, PoseidonQuinticEngine, PoseidonZkSetupConfig,
    PoseidonZkSpartanProtocolFor, PoseidonZkVerifyingKeyFor, SparkWhirParams, SpartanWhirError,
    SyntheticR1csConfig, FULL_ZK_STATEMENT_DOMAIN, MAX_FULL_ZK_GUEST_WORDS,
};

fn fixture() -> spartan_whir::SyntheticR1csFixture {
    generate_satisfiable_fixture(&SyntheticR1csConfig {
        target_log2_witness_poly: 3,
        num_constraints: 4,
        num_io: 1,
        a_terms_per_constraint: 2,
        b_terms_per_constraint: 2,
        seed: 0x1EA0_F011_2A,
    })
    .expect("fixture generation succeeds")
}

fn config(matrix_closing: MatrixClosingMode) -> PoseidonZkSetupConfig {
    let params = common::phase3_whir_params();
    PoseidonZkSetupConfig {
        matrix_closing,
        security: common::phase3_security(),
        whir_params: common::phase3_zk_whir_params(),
        spark_whir_params: (matrix_closing == MatrixClosingMode::Spark).then_some(
            SparkWhirParams {
                fixed_value: params.clone(),
                fixed_audit: params.clone(),
                read: params,
            },
        ),
        ell_zk: spartan_whir::DEFAULT_ZK_ELL,
        mask_log_inv_rate: spartan_whir::DEFAULT_ZK_MASK_LOG_INV_RATE,
    }
}

macro_rules! check_direct_profile {
    ($engine:ty, $profile:expr, $other_engine:ty) => {{
        let fixture = fixture();
        let (pk, vk) = setup_poseidon_zk_for::<$engine>(
            fixture.shape,
            config(MatrixClosingMode::DirectSparse),
        )
        .expect("full-ZK setup succeeds");
        let mut prover = <$engine as Plonky3PoseidonEngine>::challenger();
        let (instance, proof) = PoseidonZkSpartanProtocolFor::<$engine>::prove(
            &pk,
            &fixture.public_inputs,
            &fixture.witness,
            &mut prover,
        )
        .expect("full-ZK proof succeeds");
        let words = encode_full_zk_direct_guest_words::<$engine>(&instance, &proof)
            .expect("guest encoding succeeds");
        assert_eq!(words[5], $profile as u32);
        assert_eq!(words[8] as usize, words.len());
        let canonical_words = words.len();
        let padded = pad_full_zk_guest_words(words.clone()).expect("guest padding succeeds");
        assert_eq!(padded.len(), MAX_FULL_ZK_GUEST_WORDS);
        assert_eq!(padded[8] as usize, canonical_words);
        assert!(padded[canonical_words..].iter().all(|&word| word == 0));

        let (decoded_instance, decoded_proof) =
            decode_full_zk_direct_guest_words::<$engine>(&words).expect("guest decoding succeeds");
        let mut verifier = <$engine as Plonky3PoseidonEngine>::challenger();
        PoseidonZkSpartanProtocolFor::<$engine>::verify(
            &vk,
            &decoded_instance,
            &decoded_proof,
            &mut verifier,
        )
        .expect("decoded proof verifies");
        assert_eq!(
            encode_full_zk_direct_guest_words::<$engine>(&decoded_instance, &decoded_proof)
                .expect("decoded proof re-encodes"),
            words
        );

        let mut changed_public_input = words.clone();
        changed_public_input[10] += 1;
        let (changed_instance, changed_proof) =
            decode_full_zk_direct_guest_words::<$engine>(&changed_public_input)
                .expect("canonical changed input decodes");
        let mut verifier = <$engine as Plonky3PoseidonEngine>::challenger();
        assert!(PoseidonZkSpartanProtocolFor::<$engine>::verify(
            &vk,
            &changed_instance,
            &changed_proof,
            &mut verifier,
        )
        .is_err());

        let mut wrong_version = words.clone();
        wrong_version[4] += 1;
        assert!(decode_full_zk_direct_guest_words::<$engine>(&wrong_version).is_err());

        let mut wrong_profile = words.clone();
        wrong_profile[5] = match $profile {
            FullZkGuestHashProfile::Poseidon2 => FullZkGuestHashProfile::Poseidon1 as u32,
            FullZkGuestHashProfile::Poseidon1 => FullZkGuestHashProfile::Poseidon2 as u32,
        };
        assert!(matches!(
            decode_full_zk_direct_guest_words::<$engine>(&wrong_profile),
            Err(FullZkGuestCodecError::ProfileMismatch { .. })
        ));
        assert!(matches!(
            decode_full_zk_direct_guest_words::<$other_engine>(&words),
            Err(FullZkGuestCodecError::ProfileMismatch { .. })
        ));

        let mut wrong_extension = words.clone();
        wrong_extension[6] = 4;
        assert!(decode_full_zk_direct_guest_words::<$engine>(&wrong_extension).is_err());

        let mut wrong_mode = words.clone();
        wrong_mode[7] = 1;
        assert!(decode_full_zk_direct_guest_words::<$engine>(&wrong_mode).is_err());

        let mut non_canonical = words.clone();
        non_canonical[10] = spartan_whir::engine::F::ORDER_U32;
        assert!(decode_full_zk_direct_guest_words::<$engine>(&non_canonical).is_err());

        let mut invalid_cap = words.clone();
        invalid_cap[11] = 3;
        assert!(matches!(
            decode_full_zk_direct_guest_words::<$engine>(&invalid_cap),
            Err(FullZkGuestCodecError::InvalidLength {
                section: "commitment roots"
            })
        ));

        let mut trailing = words;
        trailing.push(0);
        trailing[8] = trailing.len() as u32;
        assert!(matches!(
            decode_full_zk_direct_guest_words::<$engine>(&trailing),
            Err(FullZkGuestCodecError::TrailingWords { count: 1 })
        ));
    }};
}

#[test]
fn full_zk_direct_codec_supports_both_profiles_in_one_build() {
    check_direct_profile!(
        PoseidonQuinticEngine,
        FullZkGuestHashProfile::Poseidon2,
        Poseidon1QuinticEngine
    );
    check_direct_profile!(
        Poseidon1QuinticEngine,
        FullZkGuestHashProfile::Poseidon1,
        PoseidonQuinticEngine
    );
}

macro_rules! check_spark_profile {
    ($engine:ty, $profile:expr) => {{
        let fixture = fixture();
        let (pk, vk) =
            setup_poseidon_zk_for::<$engine>(fixture.shape, config(MatrixClosingMode::Spark))
                .expect("full-ZK SPARK setup succeeds");
        let mut prover = <$engine as Plonky3PoseidonEngine>::challenger();
        let (instance, proof) = PoseidonZkSpartanProtocolFor::<$engine>::prove(
            &pk,
            &fixture.public_inputs,
            &fixture.witness,
            &mut prover,
        )
        .expect("full-ZK SPARK proof succeeds");
        let words = encode_full_zk_spark_guest_words::<$engine>(&instance, &proof)
            .expect("SPARK guest encoding succeeds");
        assert_eq!(words[5], $profile as u32);
        let (decoded_instance, decoded_proof) = decode_full_zk_spark_guest_words::<$engine>(&words)
            .expect("SPARK guest decoding succeeds");
        assert_eq!(
            encode_full_zk_spark_guest_words::<$engine>(&decoded_instance, &decoded_proof)
                .expect("decoded SPARK proof re-encodes"),
            words
        );
        let mut verifier = <$engine as Plonky3PoseidonEngine>::challenger();
        PoseidonZkSpartanProtocolFor::<$engine>::verify(
            &vk,
            &decoded_instance,
            &decoded_proof,
            &mut verifier,
        )
        .expect("decoded SPARK proof verifies");

        let mut wrong_profile = words.clone();
        wrong_profile[5] = match $profile {
            FullZkGuestHashProfile::Poseidon2 => FullZkGuestHashProfile::Poseidon1 as u32,
            FullZkGuestHashProfile::Poseidon1 => FullZkGuestHashProfile::Poseidon2 as u32,
        };
        assert!(matches!(
            decode_full_zk_spark_guest_words::<$engine>(&wrong_profile),
            Err(FullZkGuestCodecError::ProfileMismatch { .. })
        ));

        let mut wrong_mode = words.clone();
        wrong_mode[7] = 0;
        assert!(decode_full_zk_spark_guest_words::<$engine>(&wrong_mode).is_err());

        let mut non_canonical = words.clone();
        non_canonical[10] = spartan_whir::engine::F::ORDER_U32;
        assert!(decode_full_zk_spark_guest_words::<$engine>(&non_canonical).is_err());

        let mut invalid_cap = words.clone();
        invalid_cap[11] = 3;
        assert!(matches!(
            decode_full_zk_spark_guest_words::<$engine>(&invalid_cap),
            Err(FullZkGuestCodecError::InvalidLength {
                section: "commitment roots"
            })
        ));

        let mut oversized_length = words.clone();
        oversized_length[8] = u32::MAX;
        assert!(decode_full_zk_spark_guest_words::<$engine>(&oversized_length).is_err());
        assert!(decode_full_zk_spark_guest_words::<$engine>(&words[..words.len() - 1]).is_err());

        let mut trailing = words;
        trailing.push(0);
        trailing[8] = trailing.len() as u32;
        assert!(matches!(
            decode_full_zk_spark_guest_words::<$engine>(&trailing),
            Err(FullZkGuestCodecError::TrailingWords { count: 1 })
        ));
    }};
}

#[test]
fn full_zk_spark_codec_roundtrips_both_profiles() {
    check_spark_profile!(PoseidonQuinticEngine, FullZkGuestHashProfile::Poseidon2);
    check_spark_profile!(Poseidon1QuinticEngine, FullZkGuestHashProfile::Poseidon1);
}

#[test]
fn restored_full_zk_guest_configs_require_authenticated_verifying_keys() {
    let fixture = fixture();
    let (_, direct_vk) = setup_poseidon_zk_for::<PoseidonQuinticEngine>(
        fixture.shape.clone(),
        config(MatrixClosingMode::DirectSparse),
    )
    .expect("full-ZK DirectSparse setup succeeds");
    let mut restored_direct: PoseidonZkVerifyingKeyFor<PoseidonQuinticEngine> =
        bincode::deserialize(&bincode::serialize(&direct_vk).expect("DirectSparse key serializes"))
            .expect("DirectSparse key deserializes");
    assert_eq!(
        full_zk_direct_guest_verifier_config(&restored_direct).unwrap_err(),
        SpartanWhirError::InvalidConfig(InvalidConfigReason::UnauthenticatedVerifyingKey)
    );
    restored_direct
        .authenticate()
        .expect("restored DirectSparse key authenticates");
    full_zk_direct_guest_verifier_config(&restored_direct)
        .expect("authenticated DirectSparse key exports guest config");

    let (_, spark_vk) = setup_poseidon_zk_for::<PoseidonQuinticEngine>(
        fixture.shape,
        config(MatrixClosingMode::Spark),
    )
    .expect("full-ZK SPARK setup succeeds");
    let mut restored_spark: PoseidonZkVerifyingKeyFor<PoseidonQuinticEngine> =
        bincode::deserialize(&bincode::serialize(&spark_vk).expect("SPARK key serializes"))
            .expect("SPARK key deserializes");
    assert_eq!(
        full_zk_spark_guest_verifier_config(&restored_spark).unwrap_err(),
        SpartanWhirError::InvalidConfig(InvalidConfigReason::UnauthenticatedVerifyingKey)
    );
    restored_spark
        .authenticate()
        .expect("restored SPARK key authenticates");
    full_zk_spark_guest_verifier_config(&restored_spark)
        .expect("authenticated SPARK key exports guest config");
}

#[test]
fn full_zk_statement_digest_binds_domain_length_profile_and_public_inputs() {
    let first = [spartan_whir::engine::F::ZERO, spartan_whir::engine::F::ONE];
    let second = [spartan_whir::engine::F::ONE, spartan_whir::engine::F::ZERO];
    let preimage = full_zk_statement_digest_preimage(&first);
    assert_eq!(
        preimage.len(),
        FULL_ZK_STATEMENT_DOMAIN.len() + first.len() + 2
    );
    assert_eq!(
        preimage[0],
        spartan_whir::engine::F::from_usize(FULL_ZK_STATEMENT_DOMAIN.len())
    );
    assert_eq!(
        preimage[FULL_ZK_STATEMENT_DOMAIN.len() + 1],
        spartan_whir::engine::F::from_usize(first.len())
    );
    assert_ne!(
        full_zk_statement_digest::<PoseidonQuinticEngine>(&first),
        full_zk_statement_digest::<PoseidonQuinticEngine>(&second)
    );
    assert_ne!(
        full_zk_statement_digest::<PoseidonQuinticEngine>(&first),
        full_zk_statement_digest::<Poseidon1QuinticEngine>(&first)
    );
    assert_ne!(
        full_zk_statement_digest_id::<PoseidonQuinticEngine>(),
        full_zk_statement_digest_id::<Poseidon1QuinticEngine>()
    );
}

#[test]
fn checked_full_zk_direct_fixtures_reproduce_and_match_their_manifests() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    for directory in [
        "testdata/leanvm-full-zk-direct-poseidon1",
        "testdata/leanvm-full-zk-direct-poseidon2",
    ] {
        check_direct_manifest(&manifest_dir.join(directory));
    }

    let checked = manifest_dir.join(if cfg!(feature = "poseidon1") {
        "testdata/leanvm-full-zk-direct-poseidon1"
    } else {
        "testdata/leanvm-full-zk-direct-poseidon2"
    });
    let generated = temporary_fixture_directory();
    fs::create_dir_all(&generated).expect("temporary fixture directory creates");
    let status = Command::new(env!("CARGO_BIN_EXE_leanvm-full-zk-fixture"))
        .arg(&generated)
        .status()
        .expect("fixture generator starts");
    assert!(status.success(), "fixture generator succeeds");
    assert_eq!(relative_files(&generated), relative_files(&checked));
    for relative in relative_files(&checked) {
        assert_eq!(
            fs::read(generated.join(&relative)).expect("generated fixture reads"),
            fs::read(checked.join(&relative)).expect("checked fixture reads"),
            "fixture differs at {}",
            relative.display()
        );
    }
    fs::remove_dir_all(&generated).expect("temporary fixture directory removes");
}

fn check_direct_manifest(fixture_dir: &Path) {
    let manifest: serde_json::Value = serde_json::from_slice(
        &fs::read(fixture_dir.join("full_zk_direct_manifest.json")).expect("manifest reads"),
    )
    .expect("manifest decodes");
    for name in [
        "guest_input",
        "verifying_key",
        "guest_constants",
        "transcript_trace",
        "statement",
    ] {
        check_artifact(fixture_dir, &manifest["artifacts"][name]);
    }
    for artifact in manifest["artifacts"]["mutations"]
        .as_array()
        .expect("mutations are an array")
    {
        check_artifact(fixture_dir, artifact);
    }
}

fn check_artifact(fixture_dir: &Path, artifact: &serde_json::Value) {
    let relative = artifact["file"]
        .as_str()
        .expect("artifact file is a string");
    let bytes = fs::read(fixture_dir.join(relative)).expect("artifact reads");
    assert_eq!(bytes.len(), artifact["bytes"].as_u64().unwrap() as usize);
    assert_eq!(sha256_hex(&bytes), artifact["sha256"].as_str().unwrap());
}

fn relative_files(root: &Path) -> BTreeSet<PathBuf> {
    fn visit(root: &Path, directory: &Path, files: &mut BTreeSet<PathBuf>) {
        for entry in fs::read_dir(directory).expect("fixture directory reads") {
            let path = entry.expect("fixture entry reads").path();
            if path.is_dir() {
                visit(root, &path, files);
            } else {
                files.insert(path.strip_prefix(root).unwrap().to_owned());
            }
        }
    }
    let mut files = BTreeSet::new();
    visit(root, root, &mut files);
    files
}

fn temporary_fixture_directory() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock follows Unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "spartan-whir-full-zk-fixture-{}-{nonce}",
        std::process::id()
    ))
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

mod common;
