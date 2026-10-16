#![recursion_limit = "256"]

#[path = "support/sha256.rs"]
mod sha256;

use std::{env, fs, hint::black_box, path::PathBuf, time::Duration};

use criterion::{
    criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use p3_challenger::FieldChallenger;
use p3_field::PrimeField32;
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use serde_json::json;
use sha2::{Digest, Sha256};
use sha256::{input_binary, message, Sha256Fixture};
use spartan_whir::{
    engine::F,
    fixed_oracle_cache::{CachedFixedOracleProofFor, FixedOracleCacheFor},
    pcs_config::{FreshMaskBatching, MaskPacking},
    poseidon1_challenger, preprocess_spark_tables,
    proof_compression::{CompressedZkProofFor, ProofCompressionOptions},
    recommended_quintic_spark_fixed_whir_params, recommended_quintic_spark_read_whir_params,
    recommended_quintic_spark_zk_whir_params, MatrixClosingMode, Poseidon1Challenger,
    Poseidon1Engine, Poseidon1ZkCommitment, Poseidon1ZkProvingKey, Poseidon1ZkSpartanProof,
    Poseidon1ZkSpartanProtocol, Poseidon1ZkVerifyingKey, PoseidonZkSetupConfig, QuinticExtension,
    R1csInstance, SecurityConfig, SoundnessAssumption, SparkWhirParams,
};

const MESSAGE_BYTES: usize = 2048;
const SECURITY_BITS: u32 = 116;
const RNG_SEED: u64 = 0x5A25_6B32_4655_4C4C;
type Protocol = Poseidon1ZkSpartanProtocol<QuinticExtension>;
type ProvingKey = Poseidon1ZkProvingKey<QuinticExtension>;
type VerifyingKey = Poseidon1ZkVerifyingKey<QuinticExtension>;
type Proof = Poseidon1ZkSpartanProof<QuinticExtension>;
type CompressedProof = CompressedZkProofFor<Poseidon1Engine<QuinticExtension>>;
type CachedProof = CachedFixedOracleProofFor<Poseidon1Engine<QuinticExtension>>;
type FixedCache = FixedOracleCacheFor<Poseidon1Engine<QuinticExtension>>;
type Instance = R1csInstance<F, Poseidon1ZkCommitment>;

fn mask_packing(name: &str) -> MaskPacking {
    match name {
        "off" => MaskPacking::Off,
        "application" => MaskPacking::Application,
        "all" => MaskPacking::All,
        "application_free_basis" => MaskPacking::ApplicationFreeBasis,
        "all_free_basis" => MaskPacking::AllFreeBasis,
        _ => panic!("unknown mask packing {name}; expected off, application, all, application_free_basis or all_free_basis"),
    }
}

#[derive(Clone, Copy)]
struct Variant {
    name: &'static str,
    options: Option<ProofCompressionOptions>,
}

impl Variant {
    fn all() -> Vec<Self> {
        let rows = ProofCompressionOptions {
            structured_rows: true,
            ..ProofCompressionOptions::default()
        };
        let fresh = ProofCompressionOptions {
            fresh_rows: true,
            ..rows
        };
        let packed = ProofCompressionOptions {
            packed_fields: true,
            ..fresh
        };
        let metadata = ProofCompressionOptions {
            compact_integers: true,
            ..packed
        };
        let factored = ProofCompressionOptions {
            factored_rounds: true,
            ..metadata
        };
        let final_rows = ProofCompressionOptions {
            final_rows: true,
            ..factored
        };
        let derived_products = ProofCompressionOptions {
            derived_products: true,
            ..final_rows
        };
        let refined_metadata = ProofCompressionOptions {
            refined_metadata: true,
            ..derived_products
        };
        let duplicate_columns = ProofCompressionOptions {
            duplicate_columns: true,
            ..refined_metadata
        };
        vec![
            Self {
                name: "baseline",
                options: None,
            },
            Self {
                name: "rows",
                options: Some(rows),
            },
            Self {
                name: "rows_fresh",
                options: Some(fresh),
            },
            Self {
                name: "rows_fresh_packed",
                options: Some(packed),
            },
            Self {
                name: "rows_fresh_packed_metadata",
                options: Some(metadata),
            },
            Self {
                name: "rows_fresh_packed_metadata_factored",
                options: Some(factored),
            },
            Self {
                name: "final_rows",
                options: Some(final_rows),
            },
            Self {
                name: "derived_products",
                options: Some(derived_products),
            },
            Self {
                name: "refined_metadata",
                options: Some(refined_metadata),
            },
            Self {
                name: "duplicate_columns",
                options: Some(duplicate_columns),
            },
            Self {
                name: "fixed_cache",
                options: Some(ProofCompressionOptions::recommended()),
            },
        ]
    }
}

enum PreparedProof {
    Baseline(Proof),
    Compressed(CompressedProof),
    Cached(CachedProof),
}

impl PreparedProof {
    fn encode(&self) -> Vec<u8> {
        match self {
            Self::Baseline(proof) => bincode::serialize(proof).expect("baseline encodes"),
            Self::Compressed(proof) => proof.to_bytes().expect("compressed proof encodes"),
            Self::Cached(proof) => proof.to_bytes().expect("cache-dependent proof encodes"),
        }
    }
}

struct Sample {
    instance: Instance,
    proof: PreparedProof,
    bytes: Vec<u8>,
    compact_bytes: Option<Vec<u8>>,
}

struct Corpus {
    variant: Variant,
    samples: Vec<Sample>,
}

fn selected_names(name: &str, allowed: &[&str]) -> Vec<String> {
    let value = env::var(name).unwrap_or_else(|_| String::from("all"));
    let selected = if value == "all" {
        allowed
            .iter()
            .map(|name| String::from(*name))
            .collect::<Vec<_>>()
    } else {
        value
            .split(',')
            .map(|value| value.trim().to_owned())
            .collect::<Vec<_>>()
    };
    assert!(!selected.is_empty(), "{name} must not be empty");
    for item in &selected {
        assert!(
            allowed.contains(&item.as_str()),
            "unknown {name} value {item}; expected {allowed:?}"
        );
    }
    selected
}

fn env_usize(name: &str, default: usize) -> usize {
    match env::var(name) {
        Ok(value) => value
            .parse()
            .unwrap_or_else(|_| panic!("{name} must be an integer")),
        Err(env::VarError::NotPresent) => default,
        Err(error) => panic!("invalid {name}: {error}"),
    }
}

fn config(fixture: &Sha256Fixture) -> PoseidonZkSetupConfig {
    let witness_variables = fixture.shape.num_vars.next_power_of_two().ilog2() as usize;
    let tables = preprocess_spark_tables(&fixture.shape).expect("SPARK tables preprocess");
    let value_variables = tables.value_domain_size.ilog2() as usize;
    let fixed_variables = value_variables + spartan_whir::protocol::fixed_value_column_bits();
    let audit_variables = tables.row_memory_size.max(tables.col_memory_size).ilog2() as usize
        + spartan_whir::protocol::fixed_audit_column_bits();
    let read_variables =
        value_variables + spartan_whir::protocol::read_table_column_bits::<QuinticExtension>();
    let mut read = recommended_quintic_spark_read_whir_params(read_variables);
    // The same read schedule serves both power-of-two coordinate groups.
    read.round_log_inv_rates.clear();
    PoseidonZkSetupConfig {
        matrix_closing: MatrixClosingMode::Spark,
        security: SecurityConfig {
            security_level_bits: SECURITY_BITS,
            merkle_security_bits: SECURITY_BITS,
            soundness_assumption: SoundnessAssumption::JohnsonBound,
        },
        whir_params: recommended_quintic_spark_zk_whir_params(witness_variables),
        spark_whir_params: Some(SparkWhirParams {
            fixed_value: recommended_quintic_spark_fixed_whir_params(fixed_variables),
            fixed_audit: recommended_quintic_spark_fixed_whir_params(audit_variables),
            read,
        }),
        ell_zk: spartan_whir::DEFAULT_ZK_ELL,
        mask_log_inv_rate: spartan_whir::DEFAULT_ZK_MASK_LOG_INV_RATE,
    }
}

fn sample_rng(sample: usize) -> StdRng {
    StdRng::seed_from_u64(RNG_SEED.wrapping_add(sample as u64))
}

fn challenge_sample(mut challenger: Poseidon1Challenger) -> Vec<QuinticExtension> {
    (0..16)
        .map(|_| challenger.sample_algebra_element())
        .collect()
}

fn decode_and_verify(
    vk: &VerifyingKey,
    cache: Option<&FixedCache>,
    sample: &Sample,
    variant: Variant,
) {
    let mut challenger = poseidon1_challenger();
    if variant.name == "fixed_cache" {
        let proof = CachedProof::from_bytes(black_box(&sample.bytes))
            .expect("cache-dependent proof decodes");
        cache
            .expect("fixed cache exists")
            .verify(vk, black_box(&sample.instance), proof, &mut challenger)
            .expect("cache-dependent proof verifies");
    } else if variant.options.is_some() {
        let proof = CompressedProof::from_bytes(black_box(&sample.bytes)).expect("proof decodes");
        Protocol::verify_compressed(vk, black_box(&sample.instance), proof, &mut challenger)
            .expect("compressed proof verifies");
    } else {
        let proof: Proof = bincode::deserialize(black_box(&sample.bytes)).expect("proof decodes");
        Protocol::verify(vk, black_box(&sample.instance), &proof, &mut challenger)
            .expect("baseline proof verifies");
    }
}

fn build_corpora(
    witnesses: &[(spartan_whir::R1csWitness<F>, Vec<F>)],
    pk: &ProvingKey,
    vk: &VerifyingKey,
    variants: &[Variant],
    cache: Option<&FixedCache>,
) -> Vec<Corpus> {
    // Baseline proofs are always present for matched size and challenge checks,
    // even when their timing is excluded by the variant filter.
    let mut corpora = vec![Corpus {
        variant: Variant::all()[0],
        samples: Vec::new(),
    }];
    corpora.extend(
        variants
            .iter()
            .filter(|variant| variant.options.is_some())
            .map(|&variant| Corpus {
                variant,
                samples: Vec::new(),
            }),
    );
    for (sample_index, (witness, public_inputs)) in witnesses.iter().enumerate() {
        let mut prover = poseidon1_challenger();
        let (instance, proof) = Protocol::prove_with_rng(
            pk,
            public_inputs,
            witness,
            &mut prover,
            &mut sample_rng(sample_index),
        )
        .expect("baseline corpus proving succeeds");
        let expected_challenges = challenge_sample(prover);
        let expected_instance = bincode::serialize(&instance).expect("instance serializes");
        let mut verifier = poseidon1_challenger();
        Protocol::verify(vk, &instance, &proof, &mut verifier).expect("baseline corpus verifies");
        assert_eq!(
            challenge_sample(verifier),
            expected_challenges,
            "baseline prover/verifier challenge samples match"
        );
        let proof = PreparedProof::Baseline(proof);
        let bytes = proof.encode();
        corpora[0].samples.push(Sample {
            instance,
            proof,
            bytes,
            compact_bytes: None,
        });

        for corpus in &mut corpora[1..] {
            let mut prover = poseidon1_challenger();
            let (instance, proof) = Protocol::prove_compressed_with_rng(
                pk,
                public_inputs,
                witness,
                &mut prover,
                &mut sample_rng(sample_index),
                corpus.variant.options.expect("compressed variant"),
            )
            .expect("compressed corpus proving succeeds");
            assert_eq!(
                bincode::serialize(&instance).unwrap(),
                expected_instance,
                "matched proof instances agree"
            );
            assert_eq!(
                challenge_sample(prover),
                expected_challenges,
                "compressed and baseline prover challenge samples match"
            );
            let bytes = proof.to_bytes().expect("compressed proof encodes");
            let decoded = CompressedProof::from_bytes(&bytes).expect("compressed proof decodes");
            assert_eq!(
                decoded.to_bytes().unwrap(),
                bytes,
                "compressed encoding is canonical"
            );
            let mut verifier = poseidon1_challenger();
            Protocol::verify_compressed(vk, &instance, decoded, &mut verifier)
                .expect("compressed corpus verifies");
            assert_eq!(
                challenge_sample(verifier),
                expected_challenges,
                "compressed verifier challenge samples match"
            );
            let (proof, bytes, compact_bytes) = if corpus.variant.name == "fixed_cache" {
                let cached = CachedProof::from_compressed(vk, proof)
                    .expect("cache-dependent preparation succeeds");
                let cached_bytes = cached.to_bytes().expect("cache-dependent proof encodes");
                let decoded =
                    CachedProof::from_bytes(&cached_bytes).expect("cache-dependent proof decodes");
                assert_eq!(
                    decoded.to_bytes().unwrap(),
                    cached_bytes,
                    "cache-dependent encoding is canonical"
                );
                let mut verifier = poseidon1_challenger();
                cache
                    .expect("fixed cache exists")
                    .verify(vk, &instance, decoded, &mut verifier)
                    .expect("cache-dependent corpus verifies");
                assert_eq!(
                    challenge_sample(verifier),
                    expected_challenges,
                    "cache-dependent verifier challenge samples match"
                );
                (PreparedProof::Cached(cached), cached_bytes, Some(bytes))
            } else {
                (PreparedProof::Compressed(proof), bytes, None)
            };
            corpus.samples.push(Sample {
                instance,
                proof,
                bytes,
                compact_bytes,
            });
        }
        println!("corpus_sample_verified: {}", sample_index + 1);
    }
    for corpus in &corpora {
        for sample in &corpus.samples {
            decode_and_verify(vk, cache, sample, corpus.variant);
        }
    }
    corpora
}

fn workload_artifact_fingerprints() -> serde_json::Value {
    let root = env::var_os("SHA256_BENCH_WORKDIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/sha256-cache"));
    let circuit = format!("sha256_{MESSAGE_BYTES}b");
    let directory = root.join(&circuit);
    let files = [
        format!("{circuit}.r1cs"),
        format!(
            "{}{circuit}_witness.{}",
            env::consts::DLL_PREFIX,
            env::consts::DLL_EXTENSION
        ),
        format!("{circuit}_cpp/{circuit}.dat"),
    ];
    serde_json::to_value(
        files
            .into_iter()
            .map(|name| {
                let bytes = fs::read(directory.join(&name)).expect("workload artifact reads");
                (name, format!("{:x}", Sha256::digest(bytes)))
            })
            .collect::<std::collections::BTreeMap<_, _>>(),
    )
    .unwrap()
}

fn size_report(
    fixture: &Sha256Fixture,
    setup: &PoseidonZkSetupConfig,
    corpora: &[Corpus],
    selected: &[Variant],
    phases: &[String],
    cache: Option<&FixedCache>,
    inputs: &[Vec<u8>],
    vk: &VerifyingKey,
    artifact_fingerprints: &serde_json::Value,
    batching: &str,
    packing: &str,
) -> serde_json::Value {
    let baseline = &corpora[0].samples;
    let variants = corpora.iter().map(|corpus| {
        let bytes = corpus.samples.iter().map(|sample| sample.bytes.len()).collect::<Vec<_>>();
        let saved = bytes.iter().zip(baseline).map(|(size, baseline)| {
            baseline.bytes.len() as i64 - *size as i64
        }).collect::<Vec<_>>();
        let mut sorted = bytes.clone();
        sorted.sort_unstable();
        let options = corpus.variant.options.map(|options| json!({
            "structured_rows": options.structured_rows,
            "fresh_rows": options.fresh_rows,
            "packed_fields": options.packed_fields,
            "compact_integers": options.compact_integers,
            "factored_rounds": options.factored_rounds,
            "final_rows": options.final_rows,
            "derived_products": options.derived_products,
            "refined_metadata": options.refined_metadata,
            "duplicate_columns": options.duplicate_columns,
        }));
        let statistics = corpus.samples.iter().map(|sample| match &sample.proof {
            PreparedProof::Baseline(_) | PreparedProof::Cached(_) => None,
            PreparedProof::Compressed(proof) => {
                let mut statistics = proof.encoding_statistics().expect("encoding statistics succeed");
                let field_count = statistics["field_elements"].as_u64().unwrap() as f64;
                statistics["ideal_field_range_saving_bytes"] = json!(field_count * (31.0 - f64::from(F::ORDER_U32).log2()) / 8.0);
                // Removing every non-field bit is a conservative upper bound
                // for any optimization that only omits vector lengths.
                statistics["all_non_field_saving_bound_bytes"] = json!(statistics["refined_other_bits"].as_u64().unwrap() as f64 / 8.0);
                Some(statistics)
            },
        }).collect::<Vec<_>>();
        json!({
            "variant": corpus.variant.name,
            "options": options,
            "fixed_oracle_cache": corpus.variant.name == "fixed_cache",
            "encoding_statistics": statistics,
            "proof_bytes": bytes,
            "bytes_saved_against_matched_baseline": saved,
            "min_bytes": sorted[0],
            "median_bytes": (sorted[(sorted.len() - 1) / 2] as f64 + sorted[sorted.len() / 2] as f64) / 2.0,
            "max_bytes": sorted[sorted.len() - 1],
            "mean_bytes": sorted.iter().sum::<usize>() as f64 / sorted.len() as f64,
        })
    }).collect::<Vec<_>>();
    let revision = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned());
    let PreparedProof::Baseline(first_proof) = &baseline[0].proof else {
        unreachable!()
    };
    let report = json!({
        "schema": 1,
        "workload": "sha256_2048b_full_zk_poseidon1_quintic_spark",
        "features": "parallel,poseidon1",
        "security_bits": SECURITY_BITS,
        "setup": setup,
        "fresh_mask_batching": batching,
        "mask_packing": packing,
        "shape": { "num_cons": fixture.shape.num_cons, "num_vars": fixture.shape.num_vars, "num_io": fixture.shape.num_io },
        "git_revision": revision,
        "benchmark_binary_sha256": format!("{:x}", Sha256::digest(fs::read(env::current_exe().expect("benchmark executable path")).expect("benchmark executable reads"))),
        "build_rustflags": option_env!("CARGO_ENCODED_RUSTFLAGS").or(option_env!("RUSTFLAGS")),
        "build_target_arch": env::consts::ARCH,
        "build_target_os": env::consts::OS,
        "workload_artifact_sha256": artifact_fingerprints,
        "relation_digest": vk.domain_separator().relation_digest.iter().map(|byte| format!("{byte:02x}")).collect::<String>(),
        "input_binary_sha256": inputs.iter().map(|input| format!("{:x}", Sha256::digest(input))).collect::<Vec<_>>(),
        "hiding_whir_rounds": first_proof.pcs_proof.rounds.len(),
        "carried_mask_group_count": first_proof.pcs_proof.base_case.carried_mask_openings.len(),
        "carried_mask_opening_widths": first_proof.pcs_proof.base_case.carried_mask_openings.iter().map(|opening| opening.rows.first().map_or(0, Vec::len)).collect::<Vec<_>>(),
        "revealed_mask_message_lengths": first_proof.pcs_proof.base_case.blinded_masks.iter().map(|mask| mask.message.len()).collect::<Vec<_>>(),
        "revealed_mask_randomness_lengths": first_proof.pcs_proof.base_case.blinded_masks.iter().map(|mask| mask.randomness.len()).collect::<Vec<_>>(),
        "fresh_mask_root_count": first_proof.pcs_proof.base_case.fresh_mask_commitments.len(),
        "fresh_mask_opening_widths": first_proof.pcs_proof.base_case.fresh_mask_openings.iter().map(|opening| opening.rows.first().map_or(0, Vec::len)).collect::<Vec<_>>(),
        "rustflags": env::var("RUSTFLAGS").ok(),
        "threads": p3_maybe_rayon::prelude::current_num_threads(),
        "corpus_threads": 1,
        "cache_matrix_bytes": cache.map(FixedCache::matrix_bytes),
        "cache_memory_scope": "dense codewords only; Merkle nodes and allocation metadata are additional",
        "fixture": "optimized SHA-256 2048-byte linked-witness artifacts",
        "corpus_size": baseline.len(),
        "rng_seed": RNG_SEED,
        "verified_every_sample": true,
        "matched_instance_bytes": true,
        "matched_postproof_extension_challenges": 16,
        "timed_variants": selected.iter().map(|variant| variant.name).collect::<Vec<_>>(),
        "phases": phases,
        "sample_size": env_usize("SPARK_COMPRESSION_SAMPLES", 10),
        "warmup_seconds": env_usize("SPARK_COMPRESSION_WARMUP_SECONDS", 3),
        "measurement_seconds": env_usize("SPARK_COMPRESSION_MEASUREMENT_SECONDS", 15),
        "variants": variants,
    });
    report
}

fn write_json(path: &std::path::Path, report: &serde_json::Value) {
    if let Some(parent) = path.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent).expect("report directory exists");
    }
    fs::write(path, serde_json::to_vec_pretty(report).unwrap()).expect("report writes");
}

fn write_size_report(
    fixture: &Sha256Fixture,
    setup: &PoseidonZkSetupConfig,
    corpora: &[Corpus],
    selected: &[Variant],
    phases: &[String],
    cache: Option<&FixedCache>,
    inputs: &[Vec<u8>],
    vk: &VerifyingKey,
    artifact_fingerprints: &serde_json::Value,
) {
    let batching = env::var("SPARK_FRESH_MASK_BATCHING").unwrap_or_else(|_| "separate".into());
    let packing = env::var("SPARK_MASK_PACKING").unwrap_or_else(|_| "off".into());
    let report = size_report(
        fixture,
        setup,
        corpora,
        selected,
        phases,
        cache,
        inputs,
        vk,
        artifact_fingerprints,
        &batching,
        &packing,
    );
    let path = env::var_os("SPARK_COMPRESSION_REPORT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/spark-proof-compression-sizes.json"));
    write_json(&path, &report);
    println!("proof_size_report: {}", path.display());
    println!(
        "proof_size_json: {}",
        serde_json::to_string(&report).unwrap()
    );
}

fn dump_corpora(directory: &std::path::Path, inputs: &[Vec<u8>], corpora: &[Corpus]) {
    fs::create_dir_all(directory).expect("proof dump directory exists");
    for (sample, input) in inputs.iter().enumerate() {
        fs::write(directory.join(format!("input-{sample}.bin")), input)
            .expect("sample input writes");
        fs::write(
            directory.join(format!("instance-{sample}.bin")),
            bincode::serialize(&corpora[0].samples[sample].instance).expect("instance serializes"),
        )
        .expect("sample instance writes");
    }
    for corpus in corpora {
        for (index, sample) in corpus.samples.iter().enumerate() {
            fs::write(
                directory.join(format!("{}-{index}.bin", corpus.variant.name)),
                &sample.bytes,
            )
            .expect("sample proof writes");
        }
        fs::write(
            directory.join(format!("{}.bin", corpus.variant.name)),
            &corpus.samples[0].bytes,
        )
        .expect("first sample alias writes");
    }
}

fn paired_witness_blocks(
    c: &mut Criterion,
    directory: &std::path::Path,
    fixture: &Sha256Fixture,
    setup: &PoseidonZkSetupConfig,
    inputs: &[Vec<u8>],
    artifact_fingerprints: &serde_json::Value,
) {
    const BLOCKS: usize = 10;
    const SAMPLES: usize = 10;
    const ORDER_SEED: u64 = RNG_SEED ^ 0x424C_4F43_4B53;
    const BLOCK_SEED_START: usize = 1_000_000;
    const BLOCK_SEED_STRIDE: usize = 10_000;
    for flag in ["SHA256_BENCH_PROFILE", "SHA256_BENCH_PROFILE_DETAIL"] {
        let enabled = env::var(flag).is_ok_and(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        });
        assert!(!enabled, "paired acceptance requires {flag} disabled");
    }
    assert_eq!(inputs.len(), 4, "paired comparison fixes four valid inputs");
    assert!(
        !directory.join("blocks.json").exists(),
        "paired comparison requires a new output directory"
    );
    let variant = Variant::all()
        .into_iter()
        .find(|variant| variant.name == "refined_metadata")
        .unwrap();
    let options = variant.options.unwrap();
    let corpus_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let witnesses = inputs
        .iter()
        .map(|input| {
            fixture
                .generator
                .generate_witness(input, fixture.shape.num_vars, fixture.shape.num_io)
                .expect("paired corpus witness generation succeeds")
        })
        .collect::<Vec<_>>();
    let candidate_packing = env::var("SPARK_COMPRESSION_PAIRED_PACKING").ok();
    let control_packing = env::var("SPARK_COMPRESSION_PAIRED_CONTROL_PACKING").ok();
    assert!(
        control_packing.is_none() || candidate_packing.is_some(),
        "an explicit paired control packing requires a candidate packing"
    );
    let protocol_modes = if let Some(candidate) = candidate_packing.as_deref() {
        assert_ne!(candidate, "off", "paired packing requires a candidate");
        let control = control_packing.as_deref().unwrap_or("off");
        assert_ne!(control, candidate, "paired packing requires distinct modes");
        [
            (
                control,
                "same_height",
                FreshMaskBatching::SameHeight,
                control,
                mask_packing(control),
            ),
            (
                candidate,
                "same_height",
                FreshMaskBatching::SameHeight,
                candidate,
                mask_packing(candidate),
            ),
        ]
    } else {
        [
            (
                "separate",
                "separate",
                FreshMaskBatching::Separate,
                "off",
                MaskPacking::Off,
            ),
            (
                "same_height",
                "same_height",
                FreshMaskBatching::SameHeight,
                "off",
                MaskPacking::Off,
            ),
        ]
    };
    let arm_names = [protocol_modes[0].0, protocol_modes[1].0];
    let mut modes = Vec::new();
    for (name, batching_name, batching, packing_name, packing) in protocol_modes {
        let (pk, vk) = ProvingKey::setup_with_mask_packing(
            fixture.shape.clone(),
            setup.clone(),
            batching,
            packing,
        )
        .expect("paired setup succeeds");
        let corpora = corpus_pool.install(|| build_corpora(&witnesses, &pk, &vk, &[variant], None));
        dump_corpora(&directory.join(name).join("corpus"), inputs, &corpora);
        let mut report = size_report(
            fixture,
            setup,
            &corpora,
            &[variant],
            &["witness_prove".into()],
            None,
            inputs,
            &vk,
            artifact_fingerprints,
            batching_name,
            packing_name,
        );
        report["sample_size"] = json!(SAMPLES);
        report["warmup_seconds"] = json!(0.5);
        report["measurement_seconds"] = json!(1);
        report["process_scope"] = json!("both modes' proving/verifying keys and validated corpora remain resident throughout every timed block");
        report["profiling_disabled"] = json!(true);
        write_json(&directory.join(name).join("sizes.json"), &report);
        modes.push((name, pk, vk, corpora));
    }
    let mut candidate_first = vec![false; BLOCKS];
    candidate_first[BLOCKS / 2..].fill(true);
    candidate_first.shuffle(&mut StdRng::seed_from_u64(ORDER_SEED));
    let blocks = candidate_first
        .iter()
        .enumerate()
        .map(|(block, &reverse)| {
            json!({
                "block": block,
                "order": if reverse { [arm_names[1], arm_names[0]] } else { arm_names },
                "phase": format!("paired_witness_prove/block_{block:02}"),
                "seed_index_base": BLOCK_SEED_START + block * BLOCK_SEED_STRIDE,
                "arms": {},
            })
        })
        .collect::<Vec<_>>();
    let mut manifest = json!({
        "schema": 1,
        "phase": "witness_prove",
        "variant": "refined_metadata",
        "block_count": BLOCKS,
        "samples_per_arm": SAMPLES,
        "order_seed": ORDER_SEED,
        "rng_seed": RNG_SEED,
        "block_seed_stride": BLOCK_SEED_STRIDE,
        "sampling_mode": "Flat",
        "warmup_ms_per_arm": 500,
        "measurement_target_seconds_per_arm": 1,
        "process_scope": "both modes' proving/verifying keys and validated corpora remain resident throughout every timed block",
        "profiling_disabled": true,
        "control_name": arm_names[0],
        "candidate_name": arm_names[1],
        "control_size_report": format!("{}/sizes.json", arm_names[0]),
        "candidate_size_report": format!("{}/sizes.json", arm_names[1]),
        "protocol_modes": protocol_modes.iter().map(|(name, batching_name, _, packing_name, _)| (*name, json!({"fresh_mask_batching": batching_name, "mask_packing": packing_name}))).collect::<std::collections::BTreeMap<_, _>>(),
        "timing": "Criterion iter_batched, BatchSize::PerIteration; witness generation, compressed proving and compact preparation; output cleanup outside timing",
        "seed_schedule": "sample_rng(seed_index_base + invocation_index); input index is the same seed index modulo four. Invocation indices include warmup; comparison must validate retained sample schedules match.",
        "completion_rule": "exactly ten preselected blocks, no optional stopping or sample removal",
        "complete": false,
        "blocks": blocks,
    });
    let manifest_path = directory.join("blocks.json");
    write_json(&manifest_path, &manifest);
    for (block, &reverse) in candidate_first.iter().enumerate() {
        let phase = format!("paired_witness_prove/block_{block:02}");
        let mut group = c.benchmark_group(format!("spark_proof_compression/{phase}"));
        group
            .sample_size(SAMPLES)
            .sampling_mode(SamplingMode::Flat)
            .warm_up_time(Duration::from_millis(500))
            .measurement_time(Duration::from_secs(1))
            .throughput(Throughput::Elements(1));
        for mode_index in if reverse { [1, 0] } else { [0, 1] } {
            let (name, pk, _, _) = &modes[mode_index];
            let seed_base = BLOCK_SEED_START + block * BLOCK_SEED_STRIDE;
            let mut next_sample = 0usize;
            let mut callbacks = Vec::new();
            group.bench_function(BenchmarkId::from_parameter(*name), |bencher| {
                let first_sample = next_sample;
                bencher.iter_batched(
                    || {
                        let sample = seed_base + next_sample;
                        next_sample += 1;
                        (sample, sample_rng(sample))
                    },
                    |(sample, mut rng)| {
                        let (witness, public_inputs) = fixture
                            .generator
                            .generate_witness(
                                black_box(&inputs[sample % inputs.len()]),
                                fixture.shape.num_vars,
                                fixture.shape.num_io,
                            )
                            .expect("paired timed witness generation succeeds");
                        let (instance, proof) = Protocol::prove_compressed_with_rng(
                            black_box(pk),
                            &public_inputs,
                            &witness,
                            &mut poseidon1_challenger(),
                            &mut rng,
                            options,
                        )
                        .expect("paired timed compressed proving succeeds");
                        black_box((instance, PreparedProof::Compressed(proof), Vec::<u8>::new()))
                    },
                    BatchSize::PerIteration,
                );
                callbacks.push(json!({
                    "seed_index_start": seed_base + first_sample,
                    "iterations": next_sample - first_sample,
                }));
            });
            assert!(
                next_sample < BLOCK_SEED_STRIDE,
                "block RNG schedules must not overlap"
            );
            manifest["blocks"][block]["arms"][*name] = json!({
                "sample_file": format!("criterion/spark_proof_compression_paired_witness_prove_block_{block:02}/{name}/new/sample.json"),
                "callbacks_including_warmup": callbacks,
            });
            write_json(&manifest_path, &manifest);
        }
        group.finish();
    }
    manifest["complete"] = json!(true);
    write_json(&manifest_path, &manifest);
    // The bindings keep both complete configurations and corpora live until
    // every arm has finished; no arm gains a smaller resident setup.
    black_box(&modes);
    println!("paired_block_manifest: {}", manifest_path.display());
}

fn benchmark(c: &mut Criterion) {
    if spartan_whir::profiling::profile_enabled() {
        tracing_subscriber::fmt()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE)
            .with_ansi(false)
            .with_max_level(tracing::Level::INFO)
            .try_init()
            .expect("benchmark profiling subscriber initializes");
    }
    let all = Variant::all();
    let names = selected_names(
        "SPARK_COMPRESSION_VARIANTS",
        &all.iter().map(|variant| variant.name).collect::<Vec<_>>(),
    );
    let mut variants = all
        .into_iter()
        .filter(|variant| names.iter().any(|name| name == variant.name))
        .collect::<Vec<_>>();
    if env::var("SPARK_COMPRESSION_REVERSE").is_ok_and(|value| value == "1") {
        variants.reverse();
    }
    let phases = selected_names(
        "SPARK_COMPRESSION_PHASES",
        &[
            "witness_prove",
            "end_to_end",
            "encode",
            "decode_verify",
            "native_verify",
            "cache_prepare",
            "cache_build",
        ],
    );
    let corpus_size = env_usize("SPARK_COMPRESSION_CORPUS_SIZE", 4);
    assert!(
        corpus_size >= 4,
        "SPARK_COMPRESSION_CORPUS_SIZE must be at least four"
    );
    let artifact_fingerprints = workload_artifact_fingerprints();
    let fixture = Sha256Fixture::load(MESSAGE_BYTES).expect("cached SHA256 fixture loads");
    let inputs = (0..corpus_size)
        .map(|sample| {
            let message = message(MESSAGE_BYTES, sample);
            let input = input_binary(&message);
            fixture
                .validate_input(&message, &input)
                .expect("linked witness matches SHA256");
            input
        })
        .collect::<Vec<_>>();
    let setup = config(&fixture);
    if let Some(directory) = env::var_os("SPARK_COMPRESSION_PAIRED_DIR") {
        paired_witness_blocks(
            c,
            &PathBuf::from(directory),
            &fixture,
            &setup,
            &inputs,
            &artifact_fingerprints,
        );
        return;
    }
    let batching = match env::var("SPARK_FRESH_MASK_BATCHING")
        .as_deref()
        .unwrap_or("separate")
    {
        "separate" => spartan_whir::pcs_config::FreshMaskBatching::Separate,
        "same_height" => spartan_whir::pcs_config::FreshMaskBatching::SameHeight,
        _ => panic!("SPARK_FRESH_MASK_BATCHING must be separate or same_height"),
    };
    let packing = mask_packing(&env::var("SPARK_MASK_PACKING").unwrap_or_else(|_| "off".into()));
    let (pk, vk) = ProvingKey::setup_with_mask_packing(
        fixture.shape.clone(),
        setup.clone(),
        batching,
        packing,
    )
    .expect("selected setup succeeds");
    // Parallel grinding may select a different valid witness on repeated
    // proofs. Build the matched corpus deterministically; timed proving still
    // uses the caller's configured Rayon pool.
    let corpus_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let witnesses = inputs
        .iter()
        .map(|input| {
            fixture
                .generator
                .generate_witness(input, fixture.shape.num_vars, fixture.shape.num_io)
                .expect("corpus witness generation succeeds")
        })
        .collect::<Vec<_>>();
    let cache = variants
        .iter()
        .any(|variant| variant.name == "fixed_cache")
        .then(|| FixedCache::build(&vk).expect("authenticated fixed cache builds"));
    let corpora =
        corpus_pool.install(|| build_corpora(&witnesses, &pk, &vk, &variants, cache.as_ref()));
    if let Some(directory) = env::var_os("SPARK_COMPRESSION_DUMP_DIR") {
        dump_corpora(&PathBuf::from(directory), &inputs, &corpora);
    }
    write_size_report(
        &fixture,
        &setup,
        &corpora,
        &variants,
        &phases,
        cache.as_ref(),
        &inputs,
        &vk,
        &artifact_fingerprints,
    );
    if env::var("SPARK_COMPRESSION_SIZES_ONLY").is_ok_and(|value| value == "1") {
        return;
    }

    for phase in &phases {
        let mut group = c.benchmark_group(format!("spark_proof_compression/{phase}"));
        group.throughput(Throughput::Elements(1));
        if matches!(
            phase.as_str(),
            "end_to_end" | "witness_prove" | "decode_verify" | "native_verify" | "cache_build"
        ) {
            group.sampling_mode(SamplingMode::Flat);
        }
        for &variant in &variants {
            if (phase == "cache_prepare" || phase == "cache_build") && variant.name != "fixed_cache"
            {
                continue;
            }
            let corpus = corpora
                .iter()
                .find(|corpus| corpus.variant.name == variant.name)
                .unwrap();
            let mut next_sample = 0usize;
            group.bench_function(BenchmarkId::from_parameter(variant.name), |bencher| {
                match phase.as_str() {
                    "cache_build" => bencher.iter_batched(
                        || (),
                        |()| {
                            black_box(
                                FixedCache::build(black_box(&vk))
                                    .expect("timed fixed cache builds"),
                            )
                        },
                        BatchSize::PerIteration,
                    ),
                    "cache_prepare" => bencher.iter_batched(
                        || {
                            let sample = next_sample % corpus.samples.len();
                            next_sample += 1;
                            CompressedProof::from_bytes(
                                corpus.samples[sample]
                                    .compact_bytes
                                    .as_ref()
                                    .expect("compact source exists"),
                            )
                            .expect("compact source decodes")
                        },
                        |proof| {
                            black_box(
                                CachedProof::from_compressed(black_box(&vk), proof)
                                    .expect("timed cache-dependent preparation succeeds"),
                            )
                        },
                        BatchSize::PerIteration,
                    ),
                    "encode" => bencher.iter_batched(
                        || {
                            let sample = next_sample % corpus.samples.len();
                            next_sample += 1;
                            sample
                        },
                        |sample| black_box(corpus.samples[sample].proof.encode()),
                        BatchSize::PerIteration,
                    ),
                    "decode_verify" => bencher.iter_batched(
                        || {
                            let sample = next_sample % corpus.samples.len();
                            next_sample += 1;
                            sample
                        },
                        |sample| {
                            decode_and_verify(
                                black_box(&vk),
                                cache.as_ref(),
                                &corpus.samples[sample],
                                variant,
                            )
                        },
                        BatchSize::PerIteration,
                    ),
                    "native_verify" => bencher.iter_batched(
                        || {
                            let sample = next_sample % corpora[0].samples.len();
                            next_sample += 1;
                            sample
                        },
                        |sample| {
                            let sample = &corpora[0].samples[sample];
                            let PreparedProof::Baseline(proof) = &sample.proof else {
                                unreachable!();
                            };
                            Protocol::verify(
                                black_box(&vk),
                                black_box(&sample.instance),
                                black_box(proof),
                                &mut poseidon1_challenger(),
                            )
                            .expect("ordinary proof verifies");
                        },
                        BatchSize::PerIteration,
                    ),
                    "end_to_end" | "witness_prove" => bencher.iter_batched(
                        || {
                            let sample = next_sample;
                            next_sample += 1;
                            (sample, sample_rng(sample))
                        },
                        |(sample, mut rng)| {
                            let (witness, public_inputs) = fixture
                                .generator
                                .generate_witness(
                                    black_box(&inputs[sample % inputs.len()]),
                                    fixture.shape.num_vars,
                                    fixture.shape.num_io,
                                )
                                .expect("timed witness generation succeeds");
                            let mut challenger = poseidon1_challenger();
                            let (instance, proof) = match variant.options {
                                None => {
                                    let (instance, proof) = Protocol::prove_with_rng(
                                        &pk,
                                        &public_inputs,
                                        &witness,
                                        &mut challenger,
                                        &mut rng,
                                    )
                                    .expect("timed baseline proving succeeds");
                                    (instance, PreparedProof::Baseline(proof))
                                }
                                Some(options) => {
                                    let (instance, proof) = Protocol::prove_compressed_with_rng(
                                        &pk,
                                        &public_inputs,
                                        &witness,
                                        &mut challenger,
                                        &mut rng,
                                        options,
                                    )
                                    .expect("timed compressed proving succeeds");
                                    (instance, PreparedProof::Compressed(proof))
                                }
                            };
                            let proof = if phase == "end_to_end" && variant.name == "fixed_cache" {
                                let PreparedProof::Compressed(proof) = proof else {
                                    unreachable!();
                                };
                                PreparedProof::Cached(
                                    CachedProof::from_compressed(&vk, proof)
                                        .expect("timed cache-dependent preparation succeeds"),
                                )
                            } else {
                                proof
                            };
                            let bytes = if phase == "end_to_end" {
                                proof.encode()
                            } else {
                                Vec::new()
                            };
                            // Criterion drops large proof objects and output bytes
                            // after timing, consistently across all variants.
                            black_box((instance, proof, bytes))
                        },
                        BatchSize::PerIteration,
                    ),
                    _ => unreachable!("validated phase"),
                }
            });
        }
        group.finish();
    }
}

fn criterion_config() -> Criterion {
    let samples = env_usize("SPARK_COMPRESSION_SAMPLES", 10);
    let warmup = env_usize("SPARK_COMPRESSION_WARMUP_SECONDS", 3);
    let measurement = env_usize("SPARK_COMPRESSION_MEASUREMENT_SECONDS", 15);
    assert!(samples >= 10, "Criterion requires at least ten samples");
    assert!(warmup >= 1, "warmup must be at least one second");
    assert!(
        measurement >= 10,
        "measurement must be at least ten seconds"
    );
    let criterion = Criterion::default()
        .sample_size(samples)
        .warm_up_time(Duration::from_secs(warmup as u64))
        .measurement_time(Duration::from_secs(measurement as u64));
    if let Some(directory) = env::var_os("SPARK_COMPRESSION_PAIRED_DIR") {
        criterion.output_directory(&PathBuf::from(directory).join("criterion"))
    } else {
        criterion
    }
}

criterion_group! { name = benches; config = criterion_config(); targets = benchmark }
criterion_main!(benches);
