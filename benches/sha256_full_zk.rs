#[path = "support/sha256.rs"]
mod sha256;

use std::{env, hint::black_box, time::Duration};

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use rand::{
    distr::{Distribution, StandardUniform},
    rngs::StdRng,
    SeedableRng,
};
use serde::{Deserialize, Serialize};
use sha256::{input_binary, message, Sha256Fixture};
use spartan_whir::{
    engine::{ExtField, F},
    parse_whir_params_label, preprocess_spark_tables, recommended_octic_spark_fixed_whir_params,
    recommended_octic_spark_read_whir_params, recommended_octic_whir_params,
    recommended_octic_zk_whir_params, recommended_quintic_spark_fixed_whir_params,
    recommended_quintic_spark_read_whir_params, recommended_quintic_spark_whir_params,
    recommended_quintic_spark_zk_whir_params, recommended_quintic_whir_params,
    recommended_quintic_zk_whir_params, MatrixClosingMode, MlePcs, OcticBinExtension,
    Plonky3WhirPcs, PoseidonChallenger, PoseidonEngine, PoseidonSpartanProtocol,
    PoseidonZkSetupConfig, ProvingKey, QuinticExtension, R1csInstance, SecurityConfig,
    SoundnessAssumption, SparkWhirParams, SpartanProofKind, SpartanSnarkConfig, VerifyingKey,
    WhirParams, MAX_SECURITY_BITS, MIN_SECURITY_BITS,
};

#[cfg(feature = "poseidon1")]
use spartan_whir::{
    poseidon1_challenger as poseidon_zk_challenger, Poseidon1Challenger as PoseidonZkChallenger,
    Poseidon1ZkCommitment as PoseidonZkCommitment,
    Poseidon1ZkMatrixClosingProof as ZkMatrixClosingProof,
    Poseidon1ZkProvingKey as PoseidonZkProvingKey, Poseidon1ZkSpartanProof as ZkSpartanProof,
    Poseidon1ZkSpartanProtocol as PoseidonZkSpartanProtocol,
    Poseidon1ZkVerifyingKey as PoseidonZkVerifyingKey,
};
#[cfg(not(feature = "poseidon1"))]
use spartan_whir::{
    poseidon_zk_challenger, PoseidonZkChallenger, PoseidonZkCommitment, PoseidonZkProvingKey,
    PoseidonZkSpartanProtocol, PoseidonZkVerifyingKey, ZkMatrixClosingProof, ZkSpartanProof,
};

const DEFAULT_SHA256_SIZE: usize = 2048;
const DEFAULT_CORPUS_SIZE: usize = 16;
const FULL_ZK_SEED: u64 = 0x5A25_6B32_4655_4C4C;
#[cfg(feature = "poseidon1")]
const HASH_PROFILE: &str = "poseidon1";
#[cfg(not(feature = "poseidon1"))]
const HASH_PROFILE: &str = "poseidon2";

type PlainProtocol<Ext> = PoseidonSpartanProtocol<Ext>;
type FullZkProtocol<Ext> = PoseidonZkSpartanProtocol<Ext>;
type Engine<Ext> = PoseidonEngine<Ext>;
type PlainProvingKey<Ext> = ProvingKey<Engine<Ext>, Plonky3WhirPcs>;
type PlainVerifyingKey<Ext> = VerifyingKey<Engine<Ext>, Plonky3WhirPcs>;
type FullZkProvingKey<Ext> = PoseidonZkProvingKey<Ext>;
type FullZkVerifyingKey<Ext> = PoseidonZkVerifyingKey<Ext>;
type Commitment<Ext> = <Plonky3WhirPcs as MlePcs<Engine<Ext>>>::Commitment;
type Instance<Ext> = R1csInstance<F, Commitment<Ext>>;
type FullZkInstance = R1csInstance<F, PoseidonZkCommitment>;
type PlainProof<Ext> = SpartanProofKind<Engine<Ext>, Plonky3WhirPcs>;
type FullZkProof<Ext> = ZkSpartanProof<Ext>;

#[derive(Clone)]
struct BenchmarkConfigs {
    no_zk_direct: SpartanSnarkConfig,
    no_zk_spark: SpartanSnarkConfig,
    full_zk_direct: PoseidonZkSetupConfig,
    full_zk_spark: PoseidonZkSetupConfig,
}

struct BenchmarkKeys<Ext: ExtField> {
    no_zk_direct: Option<(PlainProvingKey<Ext>, PlainVerifyingKey<Ext>)>,
    no_zk_spark: Option<(PlainProvingKey<Ext>, PlainVerifyingKey<Ext>)>,
    full_zk_direct: Option<(FullZkProvingKey<Ext>, FullZkVerifyingKey<Ext>)>,
    full_zk_spark: Option<(FullZkProvingKey<Ext>, FullZkVerifyingKey<Ext>)>,
}

struct ProofCorpus<Ext: ExtField>
where
    StandardUniform: Distribution<Ext>,
{
    no_zk_direct: Vec<(Instance<Ext>, PlainProof<Ext>)>,
    no_zk_spark: Vec<(Instance<Ext>, PlainProof<Ext>)>,
    full_zk_direct: Vec<(FullZkInstance, FullZkProof<Ext>)>,
    full_zk_spark: Vec<(FullZkInstance, FullZkProof<Ext>)>,
}

#[derive(Clone, Copy)]
enum SingleProvingVariant {
    NoZkDirect,
    FullZkDirect,
    NoZkSpark,
    FullZkSpark,
}

impl SingleProvingVariant {
    const fn benchmark_variants(self) -> BenchmarkVariants {
        match self {
            Self::NoZkDirect | Self::FullZkDirect => BenchmarkVariants::DIRECT,
            Self::NoZkSpark | Self::FullZkSpark => BenchmarkVariants::SPARK,
        }
    }
}

#[derive(Clone, Copy)]
struct BenchmarkVariants {
    direct: bool,
    spark: bool,
}

impl BenchmarkVariants {
    const ALL: Self = Self {
        direct: true,
        spark: true,
    };
    const DIRECT: Self = Self {
        direct: true,
        spark: false,
    };
    const SPARK: Self = Self {
        direct: false,
        spark: true,
    };
}

fn benchmark_sha256_full_zk(c: &mut Criterion) {
    init_profile_tracing();
    println!("hash_profile: {HASH_PROFILE}");
    let sha256_size = env_usize("SHA256_ZK_BENCH_SIZE", DEFAULT_SHA256_SIZE);
    let single_variant = single_proving_variant();
    match env_string("SHA256_ZK_BENCH_EXTENSION", "selected").as_str() {
        "selected" if single_variant.is_some() => {
            benchmark_extension::<QuinticExtension>(
                c,
                sha256_size,
                "quintic",
                single_variant
                    .expect("single variant is present")
                    .benchmark_variants(),
            );
        }
        "selected" => benchmark_extension::<QuinticExtension>(
            c,
            sha256_size,
            "quintic",
            BenchmarkVariants::ALL,
        ),
        "octic" => benchmark_extension::<OcticBinExtension>(
            c,
            sha256_size,
            "octic",
            BenchmarkVariants::ALL,
        ),
        "quintic" => benchmark_extension::<QuinticExtension>(
            c,
            sha256_size,
            "quintic",
            BenchmarkVariants::ALL,
        ),
        "spark" => {
            benchmark_extension::<QuinticExtension>(
                c,
                sha256_size,
                "quintic",
                BenchmarkVariants::SPARK,
            );
            benchmark_extension::<OcticBinExtension>(
                c,
                sha256_size,
                "octic",
                BenchmarkVariants::SPARK,
            );
        }
        extension => {
            panic!(
                "SHA256_ZK_BENCH_EXTENSION must be selected, octic, quintic, or spark, got {extension}"
            )
        }
    }
}

fn init_profile_tracing() {
    if !spartan_whir::profiling::profile_enabled() {
        return;
    }
    let _ = tracing_subscriber::fmt()
        .with_target(false)
        .without_time()
        .with_level(false)
        .try_init();
}

fn benchmark_extension<Ext>(
    c: &mut Criterion,
    sha256_size: usize,
    extension: &str,
    variants: BenchmarkVariants,
) where
    Ext: ExtField + Serialize + for<'de> Deserialize<'de>,
    StandardUniform: Distribution<Ext> + Distribution<F>,
    PoseidonChallenger: CanObserve<Commitment<Ext>>
        + CanSampleUniformBits<F>
        + FieldChallenger<F>
        + GrindingChallenger<Witness = F>,
    PoseidonZkChallenger: CanObserve<PoseidonZkCommitment>,
{
    let fixture = Sha256Fixture::load(sha256_size)
        .unwrap_or_else(|error| panic!("failed to load SHA-256 benchmark fixture: {error}"));
    let corpus_size = env_usize("SHA256_ZK_BENCH_CORPUS_SIZE", DEFAULT_CORPUS_SIZE);
    assert!(
        corpus_size > 1,
        "SHA256_ZK_BENCH_CORPUS_SIZE must exceed one"
    );
    let messages = (0..corpus_size)
        .map(|sample| message(sha256_size, sample))
        .collect::<Vec<_>>();
    let inputs = messages
        .iter()
        .map(|message| input_binary(message))
        .collect::<Vec<_>>();
    for (message, input) in messages.iter().zip(&inputs) {
        fixture
            .validate_input(message, input)
            .expect("cached linked witness generator matches SHA-256");
    }

    if let Some(variant) = single_proving_variant() {
        assert!(
            env_flag("SHA256_ZK_BENCH_PROVING_ONLY"),
            "SHA256_ZK_BENCH_SINGLE_PROVING_VARIANT requires SHA256_ZK_BENCH_PROVING_ONLY=1"
        );
        let security_bits = env_usize("SHA256_ZK_BENCH_SECURITY_BITS", 116) as u32;
        benchmark_single_proving::<Ext>(
            c,
            sha256_size,
            extension,
            &fixture,
            &inputs,
            security_bits,
            variant,
        );
        return;
    }

    let (configs, keys, security_bits) =
        select_configs_and_keys::<Ext>(&fixture, extension, variants);
    println!("benchmark_security_bits: {security_bits}");

    if env_flag("SHA256_ZK_BENCH_PROVING_ONLY") {
        benchmark_proving::<Ext>(
            c,
            sha256_size,
            extension,
            &fixture,
            &inputs,
            &configs,
            &keys,
        );
        return;
    }

    benchmark_proving::<Ext>(
        c,
        sha256_size,
        extension,
        &fixture,
        &inputs,
        &configs,
        &keys,
    );

    let proofs = build_proof_corpus::<Ext>(&fixture, &messages, &inputs, &configs, &keys);
    report_proof_sizes::<Ext>(&proofs);
    benchmark_verification::<Ext>(c, sha256_size, extension, &keys, &proofs);
}

fn benchmark_single_proving<Ext>(
    c: &mut Criterion,
    sha256_size: usize,
    extension: &str,
    fixture: &Sha256Fixture,
    inputs: &[Vec<u8>],
    security_bits: u32,
    variant: SingleProvingVariant,
) where
    Ext: ExtField + Serialize + for<'de> Deserialize<'de>,
    StandardUniform: Distribution<Ext> + Distribution<F>,
    PoseidonChallenger: CanObserve<Commitment<Ext>>
        + CanSampleUniformBits<F>
        + FieldChallenger<F>
        + GrindingChallenger<Witness = F>,
    PoseidonZkChallenger: CanObserve<PoseidonZkCommitment>,
{
    assert!(
        (MIN_SECURITY_BITS..=MAX_SECURITY_BITS).contains(&security_bits),
        "SHA256_ZK_BENCH_SECURITY_BITS must be between {MIN_SECURITY_BITS} and {MAX_SECURITY_BITS}"
    );
    let num_variables = fixture.shape.num_vars.next_power_of_two().ilog2() as usize;
    let security = benchmark_security(security_bits);
    println!("benchmark_security_bits: {security_bits}");
    let mut group = c.benchmark_group(format!(
        "sha256_{sha256_size}b_{extension}/witness_and_prove"
    ));

    match variant {
        SingleProvingVariant::NoZkDirect => {
            let config = SpartanSnarkConfig {
                matrix_closing: MatrixClosingMode::DirectSparse,
                security,
                whir_params: benchmark_whir_params(
                    num_variables,
                    extension,
                    "SHA256_ZK_BENCH_NO_ZK_DIRECT_SCHEDULE",
                ),
                spark_whir_params: None,
            };
            let (pk, _) = PlainProtocol::<Ext>::setup_with_config(&fixture.shape, &config)
                .expect("no-ZK DirectSparse setup succeeds at selected security");
            let mut sample_index = 0usize;
            group.bench_function(BenchmarkId::from_parameter("no_zk_direct"), |bencher| {
                bencher.iter_batched(
                    || {
                        let input = inputs[sample_index % inputs.len()].clone();
                        sample_index += 1;
                        input
                    },
                    |input| {
                        let (witness, public_inputs) = fixture
                            .generator
                            .generate_witness(&input, fixture.shape.num_vars, fixture.shape.num_io)
                            .expect("no-ZK DirectSparse witness generation succeeds");
                        let mut challenger = spartan_whir::poseidon_challenger();
                        black_box(
                            PlainProtocol::<Ext>::prove_with_mode(
                                &pk,
                                &public_inputs,
                                &witness,
                                MatrixClosingMode::DirectSparse,
                                &mut challenger,
                            )
                            .expect("no-ZK DirectSparse proving succeeds"),
                        )
                    },
                    BatchSize::PerIteration,
                );
            });
        }
        SingleProvingVariant::FullZkDirect => {
            let config = PoseidonZkSetupConfig {
                matrix_closing: MatrixClosingMode::DirectSparse,
                security,
                whir_params: benchmark_whir_params(
                    num_variables,
                    extension,
                    "SHA256_ZK_BENCH_FULL_ZK_DIRECT_SCHEDULE",
                ),
                spark_whir_params: None,
                ell_zk: env_usize("SHA256_BENCH_ZK_ELL", spartan_whir::DEFAULT_ZK_ELL),
                mask_log_inv_rate: env_usize(
                    "SHA256_BENCH_ZK_MASK_LOG_INV_RATE",
                    spartan_whir::DEFAULT_ZK_MASK_LOG_INV_RATE,
                ),
            };
            let (pk, vk) = FullZkProvingKey::<Ext>::setup(fixture.shape.clone(), config)
                .expect("full-ZK DirectSparse setup succeeds at selected security");
            if env_flag("SHA256_ZK_BENCH_DIAGNOSTICS") {
                let mut proof_sizes = Vec::with_capacity(inputs.len());
                let mut verification_time = Duration::ZERO;
                for (sample, input) in inputs.iter().enumerate() {
                    let (witness, public_inputs) = fixture
                        .generator
                        .generate_witness(input, fixture.shape.num_vars, fixture.shape.num_io)
                        .expect("full-ZK DirectSparse diagnostic witness generation succeeds");
                    let mut prover = poseidon_zk_challenger();
                    let mut rng = sample_rng(sample);
                    let (instance, proof) = FullZkProtocol::<Ext>::prove_with_rng(
                        &pk,
                        &public_inputs,
                        &witness,
                        &mut prover,
                        &mut rng,
                    )
                    .expect("full-ZK DirectSparse diagnostic proving succeeds");
                    proof_sizes.push(
                        bincode::serialize(&proof)
                            .expect("full-ZK DirectSparse diagnostic proof serializes")
                            .len(),
                    );
                    let verify_start = std::time::Instant::now();
                    let mut verifier = poseidon_zk_challenger();
                    FullZkProtocol::<Ext>::verify(&vk, &instance, &proof, &mut verifier)
                        .expect("full-ZK DirectSparse diagnostic verification succeeds");
                    verification_time += verify_start.elapsed();
                }
                proof_sizes.sort_unstable();
                println!(
                    "diagnostic: profile={HASH_PROFILE} variant=full_zk_direct samples={} proof_size_bytes_min={} proof_size_bytes_median={} proof_size_bytes_max={} verify_mean_ms={:.3}",
                    proof_sizes.len(),
                    proof_sizes[0],
                    proof_sizes[proof_sizes.len() / 2],
                    proof_sizes[proof_sizes.len() - 1],
                    verification_time.as_secs_f64() * 1_000.0 / proof_sizes.len() as f64,
                );
            }
            let mut sample_index = 0usize;
            group.bench_function(BenchmarkId::from_parameter("full_zk_direct"), |bencher| {
                bencher.iter_batched(
                    || {
                        let sample = sample_index;
                        sample_index += 1;
                        (inputs[sample % inputs.len()].clone(), sample_rng(sample))
                    },
                    |(input, mut rng)| {
                        let (witness, public_inputs) = fixture
                            .generator
                            .generate_witness(&input, fixture.shape.num_vars, fixture.shape.num_io)
                            .expect("full-ZK DirectSparse witness generation succeeds");
                        let mut challenger = poseidon_zk_challenger();
                        black_box(
                            FullZkProtocol::<Ext>::prove_with_rng(
                                &pk,
                                &public_inputs,
                                &witness,
                                &mut challenger,
                                &mut rng,
                            )
                            .expect("full-ZK DirectSparse proving succeeds"),
                        )
                    },
                    BatchSize::PerIteration,
                );
            });
        }
        SingleProvingVariant::NoZkSpark => {
            let config = SpartanSnarkConfig {
                matrix_closing: MatrixClosingMode::Spark,
                security,
                whir_params: benchmark_no_zk_spark_whir_params(num_variables, extension),
                spark_whir_params: Some(benchmark_spark_table_whir_params_for_shape::<Ext>(
                    &fixture.shape,
                    extension,
                )),
            };
            let (pk, _) = PlainProtocol::<Ext>::setup_with_config(&fixture.shape, &config)
                .expect("no-ZK SPARK setup succeeds at selected security");
            let mut sample_index = 0usize;
            group.bench_function(BenchmarkId::from_parameter("no_zk_spark"), |bencher| {
                bencher.iter_batched(
                    || {
                        let input = inputs[sample_index % inputs.len()].clone();
                        sample_index += 1;
                        input
                    },
                    |input| {
                        let _profile_context = spartan_whir::profiling::set_profile_context(
                            "criterion",
                            "no_zk_spark",
                        );
                        let _profile =
                            spartan_whir::profiling::profile_scope("criterion_witness_and_prove");
                        let (witness, public_inputs) = fixture
                            .generator
                            .generate_witness(&input, fixture.shape.num_vars, fixture.shape.num_io)
                            .expect("no-ZK SPARK witness generation succeeds");
                        let mut challenger = spartan_whir::poseidon_challenger();
                        black_box(
                            PlainProtocol::<Ext>::prove_with_mode(
                                &pk,
                                &public_inputs,
                                &witness,
                                MatrixClosingMode::Spark,
                                &mut challenger,
                            )
                            .expect("no-ZK SPARK proving succeeds"),
                        )
                    },
                    BatchSize::PerIteration,
                );
            });
        }
        SingleProvingVariant::FullZkSpark => {
            let config = PoseidonZkSetupConfig {
                matrix_closing: MatrixClosingMode::Spark,
                security,
                whir_params: benchmark_full_zk_spark_whir_params(num_variables, extension),
                spark_whir_params: Some(benchmark_spark_table_whir_params_for_shape::<Ext>(
                    &fixture.shape,
                    extension,
                )),
                ell_zk: env_usize("SHA256_BENCH_ZK_ELL", spartan_whir::DEFAULT_ZK_ELL),
                mask_log_inv_rate: env_usize(
                    "SHA256_BENCH_ZK_MASK_LOG_INV_RATE",
                    spartan_whir::DEFAULT_ZK_MASK_LOG_INV_RATE,
                ),
            };
            let (pk, vk) = FullZkProvingKey::<Ext>::setup(fixture.shape.clone(), config)
                .expect("full-ZK SPARK setup succeeds at selected security");
            if env_flag("SHA256_ZK_BENCH_DIAGNOSTICS") {
                let mut proof_sizes = Vec::with_capacity(inputs.len());
                let mut verification_time = Duration::ZERO;
                for (sample, input) in inputs.iter().enumerate() {
                    let (witness, public_inputs) = fixture
                        .generator
                        .generate_witness(input, fixture.shape.num_vars, fixture.shape.num_io)
                        .expect("full-ZK SPARK diagnostic witness generation succeeds");
                    let mut prover = poseidon_zk_challenger();
                    let mut rng = sample_rng(sample);
                    let (instance, proof) = FullZkProtocol::<Ext>::prove_with_rng(
                        &pk,
                        &public_inputs,
                        &witness,
                        &mut prover,
                        &mut rng,
                    )
                    .expect("full-ZK SPARK diagnostic proving succeeds");
                    proof_sizes.push(
                        bincode::serialize(&proof)
                            .expect("full-ZK SPARK diagnostic proof serializes")
                            .len(),
                    );
                    let verify_start = std::time::Instant::now();
                    let mut verifier = poseidon_zk_challenger();
                    FullZkProtocol::<Ext>::verify(&vk, &instance, &proof, &mut verifier)
                        .expect("full-ZK SPARK diagnostic verification succeeds");
                    verification_time += verify_start.elapsed();
                }
                proof_sizes.sort_unstable();
                println!(
                    "diagnostic: profile={HASH_PROFILE} variant=full_zk_spark samples={} proof_size_bytes_min={} proof_size_bytes_median={} proof_size_bytes_max={} verify_mean_ms={:.3}",
                    proof_sizes.len(),
                    proof_sizes[0],
                    proof_sizes[proof_sizes.len() / 2],
                    proof_sizes[proof_sizes.len() - 1],
                    verification_time.as_secs_f64() * 1_000.0 / proof_sizes.len() as f64,
                );
            }
            let mut sample_index = 0usize;
            group.bench_function(BenchmarkId::from_parameter("full_zk_spark"), |bencher| {
                bencher.iter_batched(
                    || {
                        let sample = sample_index;
                        sample_index += 1;
                        (inputs[sample % inputs.len()].clone(), sample_rng(sample))
                    },
                    |(input, mut rng)| {
                        let _profile_context = spartan_whir::profiling::set_profile_context(
                            "criterion",
                            "full_zk_spark",
                        );
                        let _profile =
                            spartan_whir::profiling::profile_scope("criterion_witness_and_prove");
                        let (witness, public_inputs) = fixture
                            .generator
                            .generate_witness(&input, fixture.shape.num_vars, fixture.shape.num_io)
                            .expect("full-ZK SPARK witness generation succeeds");
                        let mut challenger = poseidon_zk_challenger();
                        black_box(
                            FullZkProtocol::<Ext>::prove_with_rng(
                                &pk,
                                &public_inputs,
                                &witness,
                                &mut challenger,
                                &mut rng,
                            )
                            .expect("full-ZK SPARK proving succeeds"),
                        )
                    },
                    BatchSize::PerIteration,
                );
            });
        }
    }
    group.finish();
}

fn select_configs_and_keys<Ext>(
    fixture: &Sha256Fixture,
    extension: &str,
    variants: BenchmarkVariants,
) -> (BenchmarkConfigs, BenchmarkKeys<Ext>, u32)
where
    Ext: ExtField + Serialize + for<'de> Deserialize<'de>,
    StandardUniform: Distribution<Ext> + Distribution<F>,
    PoseidonChallenger: CanObserve<Commitment<Ext>>
        + CanSampleUniformBits<F>
        + FieldChallenger<F>
        + GrindingChallenger<Witness = F>,
    PoseidonZkChallenger: CanObserve<PoseidonZkCommitment>,
{
    let security_bits = env_usize("SHA256_ZK_BENCH_SECURITY_BITS", 116) as u32;
    assert!(
        (MIN_SECURITY_BITS..=MAX_SECURITY_BITS).contains(&security_bits),
        "SHA256_ZK_BENCH_SECURITY_BITS must be between {MIN_SECURITY_BITS} and {MAX_SECURITY_BITS}"
    );
    let configs = benchmark_configs::<Ext>(&fixture.shape, extension, security_bits);
    let full_zk_spark = variants.spark.then(|| {
        FullZkProvingKey::<Ext>::setup(fixture.shape.clone(), configs.full_zk_spark.clone())
            .expect("full-ZK SPARK setup succeeds at selected security")
    });
    let full_zk_direct = variants.direct.then(|| {
        FullZkProvingKey::<Ext>::setup(fixture.shape.clone(), configs.full_zk_direct.clone())
            .expect("full-ZK DirectSparse setup succeeds at selected security")
    });
    let no_zk_direct = variants.direct.then(|| {
        PlainProtocol::<Ext>::setup_with_config(&fixture.shape, &configs.no_zk_direct)
            .expect("no-ZK DirectSparse setup succeeds at selected security")
    });
    let no_zk_spark = variants.spark.then(|| {
        PlainProtocol::<Ext>::setup_with_config(&fixture.shape, &configs.no_zk_spark)
            .expect("no-ZK SPARK setup succeeds at selected security")
    });
    (
        configs,
        BenchmarkKeys {
            no_zk_direct,
            no_zk_spark,
            full_zk_direct,
            full_zk_spark,
        },
        security_bits,
    )
}

fn benchmark_proving<Ext>(
    c: &mut Criterion,
    sha256_size: usize,
    extension: &str,
    fixture: &Sha256Fixture,
    inputs: &[Vec<u8>],
    configs: &BenchmarkConfigs,
    keys: &BenchmarkKeys<Ext>,
) where
    Ext: ExtField + Serialize + for<'de> Deserialize<'de>,
    StandardUniform: Distribution<Ext> + Distribution<F>,
    PoseidonChallenger: CanObserve<Commitment<Ext>>
        + CanSampleUniformBits<F>
        + FieldChallenger<F>
        + GrindingChallenger<Witness = F>,
    PoseidonZkChallenger: CanObserve<PoseidonZkCommitment>,
{
    let mut group = c.benchmark_group(format!(
        "sha256_{sha256_size}b_{extension}/witness_and_prove"
    ));
    if let Some((pk, _)) = &keys.no_zk_direct {
        let mut no_zk_direct_sample = 0usize;
        group.bench_function(BenchmarkId::from_parameter("no_zk_direct"), |bencher| {
            bencher.iter_batched(
                || {
                    let input = inputs[no_zk_direct_sample % inputs.len()].clone();
                    no_zk_direct_sample += 1;
                    input
                },
                |input| {
                    let (witness, public_inputs) = fixture
                        .generator
                        .generate_witness(&input, fixture.shape.num_vars, fixture.shape.num_io)
                        .expect("plain-WHIR witness generation succeeds");
                    let mut challenger = spartan_whir::poseidon_challenger();
                    black_box(
                        PlainProtocol::<Ext>::prove_with_mode(
                            pk,
                            &public_inputs,
                            &witness,
                            configs.no_zk_direct.matrix_closing,
                            &mut challenger,
                        )
                        .expect("no-ZK DirectSparse proving succeeds"),
                    )
                },
                BatchSize::PerIteration,
            );
        });
    }

    if let Some((pk, _)) = &keys.no_zk_spark {
        let mut no_zk_spark_sample = 0usize;
        group.bench_function(BenchmarkId::from_parameter("no_zk_spark"), |bencher| {
            bencher.iter_batched(
                || {
                    let input = inputs[no_zk_spark_sample % inputs.len()].clone();
                    no_zk_spark_sample += 1;
                    input
                },
                |input| {
                    let _profile_context =
                        spartan_whir::profiling::set_profile_context("criterion", "no_zk_spark");
                    let _profile =
                        spartan_whir::profiling::profile_scope("criterion_witness_and_prove");
                    let (witness, public_inputs) = fixture
                        .generator
                        .generate_witness(&input, fixture.shape.num_vars, fixture.shape.num_io)
                        .expect("no-ZK SPARK witness generation succeeds");
                    let mut challenger = spartan_whir::poseidon_challenger();
                    black_box(
                        PlainProtocol::<Ext>::prove_with_mode(
                            pk,
                            &public_inputs,
                            &witness,
                            configs.no_zk_spark.matrix_closing,
                            &mut challenger,
                        )
                        .expect("no-ZK SPARK proving succeeds"),
                    )
                },
                BatchSize::PerIteration,
            );
        });
    }

    if let Some((pk, _)) = &keys.full_zk_direct {
        let mut full_zk_direct_sample = 0usize;
        group.bench_function(BenchmarkId::from_parameter("full_zk_direct"), |bencher| {
            bencher.iter_batched(
                || {
                    let sample = full_zk_direct_sample;
                    full_zk_direct_sample += 1;
                    let input = inputs[sample % inputs.len()].clone();
                    (input, sample_rng(sample))
                },
                |(input, mut rng)| {
                    let (witness, public_inputs) = fixture
                        .generator
                        .generate_witness(&input, fixture.shape.num_vars, fixture.shape.num_io)
                        .expect("full-ZK witness generation succeeds");
                    let mut challenger = poseidon_zk_challenger();
                    black_box(
                        FullZkProtocol::<Ext>::prove_with_rng(
                            pk,
                            &public_inputs,
                            &witness,
                            &mut challenger,
                            &mut rng,
                        )
                        .expect("full-ZK DirectSparse proving succeeds"),
                    )
                },
                BatchSize::PerIteration,
            );
        });
    }

    if let Some((pk, _)) = &keys.full_zk_spark {
        let mut full_zk_spark_sample = 0usize;
        group.bench_function(BenchmarkId::from_parameter("full_zk_spark"), |bencher| {
            bencher.iter_batched(
                || {
                    let sample = full_zk_spark_sample;
                    full_zk_spark_sample += 1;
                    let input = inputs[sample % inputs.len()].clone();
                    (input, sample_rng(sample))
                },
                |(input, mut rng)| {
                    let _profile_context =
                        spartan_whir::profiling::set_profile_context("criterion", "full_zk_spark");
                    let _profile =
                        spartan_whir::profiling::profile_scope("criterion_witness_and_prove");
                    let (witness, public_inputs) = fixture
                        .generator
                        .generate_witness(&input, fixture.shape.num_vars, fixture.shape.num_io)
                        .expect("full-ZK SPARK witness generation succeeds");
                    let mut challenger = poseidon_zk_challenger();
                    black_box(
                        FullZkProtocol::<Ext>::prove_with_rng(
                            pk,
                            &public_inputs,
                            &witness,
                            &mut challenger,
                            &mut rng,
                        )
                        .expect("full-ZK SPARK proving succeeds"),
                    )
                },
                BatchSize::PerIteration,
            );
        });
    }
    group.finish();
}

fn build_proof_corpus<Ext>(
    fixture: &Sha256Fixture,
    messages: &[Vec<u8>],
    inputs: &[Vec<u8>],
    configs: &BenchmarkConfigs,
    keys: &BenchmarkKeys<Ext>,
) -> ProofCorpus<Ext>
where
    Ext: ExtField + Serialize + for<'de> Deserialize<'de>,
    StandardUniform: Distribution<Ext> + Distribution<F>,
    PoseidonChallenger: CanObserve<Commitment<Ext>>
        + CanSampleUniformBits<F>
        + FieldChallenger<F>
        + GrindingChallenger<Witness = F>,
    PoseidonZkChallenger: CanObserve<PoseidonZkCommitment>,
{
    let mut corpus = ProofCorpus {
        no_zk_direct: Vec::with_capacity(inputs.len()),
        no_zk_spark: Vec::with_capacity(inputs.len()),
        full_zk_direct: Vec::with_capacity(inputs.len()),
        full_zk_spark: Vec::with_capacity(inputs.len()),
    };
    for (sample, input) in inputs.iter().enumerate() {
        let (witness, public_inputs) = fixture
            .generator
            .generate_witness(input, fixture.shape.num_vars, fixture.shape.num_io)
            .expect("verification-corpus witness generation succeeds");
        fixture
            .validate_input(&messages[sample], input)
            .expect("verification-corpus input matches SHA-256");

        if let Some((pk, vk)) = &keys.no_zk_direct {
            let mut prover = spartan_whir::poseidon_challenger();
            let proof = PlainProtocol::<Ext>::prove_with_mode(
                pk,
                &public_inputs,
                &witness,
                configs.no_zk_direct.matrix_closing,
                &mut prover,
            )
            .expect("verification-corpus no-ZK DirectSparse proof succeeds");
            let mut verifier = spartan_whir::poseidon_challenger();
            PlainProtocol::<Ext>::verify_with_mode(vk, &proof.0, &proof.1, &mut verifier)
                .expect("verification-corpus no-ZK DirectSparse proof verifies");
            corpus.no_zk_direct.push(proof);
        }

        if let Some((pk, vk)) = &keys.no_zk_spark {
            let mut prover = spartan_whir::poseidon_challenger();
            let proof = PlainProtocol::<Ext>::prove_with_mode(
                pk,
                &public_inputs,
                &witness,
                configs.no_zk_spark.matrix_closing,
                &mut prover,
            )
            .expect("verification-corpus no-ZK SPARK proof succeeds");
            let mut verifier = spartan_whir::poseidon_challenger();
            PlainProtocol::<Ext>::verify_with_mode(vk, &proof.0, &proof.1, &mut verifier)
                .expect("verification-corpus no-ZK SPARK proof verifies");
            corpus.no_zk_spark.push(proof);
        }

        if let Some((pk, vk)) = &keys.full_zk_direct {
            let mut rng = sample_rng(sample);
            let mut prover = poseidon_zk_challenger();
            let proof = FullZkProtocol::<Ext>::prove_with_rng(
                pk,
                &public_inputs,
                &witness,
                &mut prover,
                &mut rng,
            )
            .expect("verification-corpus full-ZK DirectSparse proof succeeds");
            let mut verifier = poseidon_zk_challenger();
            FullZkProtocol::<Ext>::verify(vk, &proof.0, &proof.1, &mut verifier)
                .expect("verification-corpus full-ZK DirectSparse proof verifies");
            corpus.full_zk_direct.push(proof);
        }

        if let Some((pk, vk)) = &keys.full_zk_spark {
            let mut rng = sample_rng(sample);
            let mut prover = poseidon_zk_challenger();
            let proof = FullZkProtocol::<Ext>::prove_with_rng(
                pk,
                &public_inputs,
                &witness,
                &mut prover,
                &mut rng,
            )
            .expect("verification-corpus full-ZK SPARK proof succeeds");
            let mut verifier = poseidon_zk_challenger();
            FullZkProtocol::<Ext>::verify(vk, &proof.0, &proof.1, &mut verifier)
                .expect("verification-corpus full-ZK SPARK proof verifies");
            corpus.full_zk_spark.push(proof);
        }
    }
    corpus
}

fn benchmark_verification<Ext>(
    c: &mut Criterion,
    sha256_size: usize,
    extension: &str,
    keys: &BenchmarkKeys<Ext>,
    proofs: &ProofCorpus<Ext>,
) where
    Ext: ExtField + Serialize + for<'de> Deserialize<'de>,
    StandardUniform: Distribution<Ext> + Distribution<F>,
    PoseidonChallenger: CanObserve<Commitment<Ext>>
        + CanSampleUniformBits<F>
        + FieldChallenger<F>
        + GrindingChallenger<Witness = F>,
    PoseidonZkChallenger: CanObserve<PoseidonZkCommitment>,
{
    let mut group = c.benchmark_group(format!("sha256_{sha256_size}b_{extension}/verify"));
    if let Some((_, vk)) = &keys.no_zk_direct {
        let mut no_zk_direct_sample = 0usize;
        group.bench_function(BenchmarkId::from_parameter("no_zk_direct"), |bencher| {
            bencher.iter_batched(
                || {
                    let (instance, proof) =
                        &proofs.no_zk_direct[no_zk_direct_sample % proofs.no_zk_direct.len()];
                    no_zk_direct_sample += 1;
                    (instance, proof)
                },
                |(instance, proof)| {
                    let mut challenger = spartan_whir::poseidon_challenger();
                    black_box(
                        PlainProtocol::<Ext>::verify_with_mode(
                            vk,
                            instance,
                            proof,
                            &mut challenger,
                        )
                        .expect("no-ZK DirectSparse verification succeeds"),
                    )
                },
                BatchSize::PerIteration,
            );
        });
    }

    if let Some((_, vk)) = &keys.no_zk_spark {
        let mut no_zk_spark_sample = 0usize;
        group.bench_function(BenchmarkId::from_parameter("no_zk_spark"), |bencher| {
            bencher.iter_batched(
                || {
                    let (instance, proof) =
                        &proofs.no_zk_spark[no_zk_spark_sample % proofs.no_zk_spark.len()];
                    no_zk_spark_sample += 1;
                    (instance, proof)
                },
                |(instance, proof)| {
                    let mut challenger = spartan_whir::poseidon_challenger();
                    black_box(
                        PlainProtocol::<Ext>::verify_with_mode(
                            vk,
                            instance,
                            proof,
                            &mut challenger,
                        )
                        .expect("no-ZK SPARK verification succeeds"),
                    )
                },
                BatchSize::PerIteration,
            );
        });
    }

    if let Some((_, vk)) = &keys.full_zk_direct {
        let mut full_zk_direct_sample = 0usize;
        group.bench_function(BenchmarkId::from_parameter("full_zk_direct"), |bencher| {
            bencher.iter_batched(
                || {
                    let (instance, proof) =
                        &proofs.full_zk_direct[full_zk_direct_sample % proofs.full_zk_direct.len()];
                    full_zk_direct_sample += 1;
                    (instance, proof)
                },
                |(instance, proof)| {
                    let mut challenger = poseidon_zk_challenger();
                    black_box(
                        FullZkProtocol::<Ext>::verify(vk, instance, proof, &mut challenger)
                            .expect("full-ZK DirectSparse verification succeeds"),
                    )
                },
                BatchSize::PerIteration,
            );
        });
    }

    if let Some((_, vk)) = &keys.full_zk_spark {
        let mut full_zk_spark_sample = 0usize;
        group.bench_function(BenchmarkId::from_parameter("full_zk_spark"), |bencher| {
            bencher.iter_batched(
                || {
                    let (instance, proof) =
                        &proofs.full_zk_spark[full_zk_spark_sample % proofs.full_zk_spark.len()];
                    full_zk_spark_sample += 1;
                    (instance, proof)
                },
                |(instance, proof)| {
                    let mut challenger = poseidon_zk_challenger();
                    black_box(
                        FullZkProtocol::<Ext>::verify(vk, instance, proof, &mut challenger)
                            .expect("full-ZK SPARK verification succeeds"),
                    )
                },
                BatchSize::PerIteration,
            );
        });
    }
    group.finish();
}

fn report_proof_sizes<Ext>(proofs: &ProofCorpus<Ext>)
where
    Ext: ExtField + Serialize,
    StandardUniform: Distribution<Ext>,
{
    if !proofs.no_zk_direct.is_empty() {
        report_variant_size("no_zk_direct", &proofs.no_zk_direct);
    }
    if !proofs.no_zk_spark.is_empty() {
        report_variant_size("no_zk_spark", &proofs.no_zk_spark);
    }
    if !proofs.full_zk_direct.is_empty() {
        report_variant_size("full_zk_direct", &proofs.full_zk_direct);
        report_full_zk_sections("full_zk_direct", &proofs.full_zk_direct);
    }
    if proofs.full_zk_spark.is_empty() {
        return;
    }
    report_variant_size("full_zk_spark", &proofs.full_zk_spark);
    report_full_zk_sections("full_zk_spark", &proofs.full_zk_spark);

    let spark_products = size_stats(proofs.full_zk_spark.iter().map(|(_, proof)| {
        let ZkMatrixClosingProof::Spark(closing) = &proof.matrix_closing else {
            panic!("full-ZK SPARK corpus contains direct proof");
        };
        bincode::serialize(&closing.spark_products)
            .expect("SPARK product proof serializes")
            .len()
    }));
    let spark_fixed = size_stats(proofs.full_zk_spark.iter().map(|(_, proof)| {
        let ZkMatrixClosingProof::Spark(closing) = &proof.matrix_closing else {
            panic!("full-ZK SPARK corpus contains direct proof");
        };
        bincode::serialize(&closing.spark_fixed_openings)
            .expect("SPARK fixed openings serialize")
            .len()
    }));
    let spark_read = size_stats(proofs.full_zk_spark.iter().map(|(_, proof)| {
        let ZkMatrixClosingProof::Spark(closing) = &proof.matrix_closing else {
            panic!("full-ZK SPARK corpus contains direct proof");
        };
        bincode::serialize(&closing.spark_read_openings)
            .expect("SPARK read openings serialize")
            .len()
    }));
    println!(
        "full_zk_spark_closing_sections_median_bytes: products={} fixed_openings={} read_openings={}",
        spark_products.median, spark_fixed.median, spark_read.median
    );
}

fn report_variant_size<T, P: Serialize>(variant: &str, proofs: &[(T, P)]) {
    let stats = size_stats(
        proofs
            .iter()
            .map(|(_, proof)| bincode::serialize(proof).expect("proof serializes").len()),
    );
    println!(
        "proof_size_bytes: profile={HASH_PROFILE} variant={variant} min={} median={} max={}",
        stats.min, stats.median, stats.max
    );
}

fn report_full_zk_sections<Ext>(variant: &str, proofs: &[(FullZkInstance, FullZkProof<Ext>)])
where
    Ext: ExtField + Serialize,
    StandardUniform: Distribution<Ext>,
{
    let application_masks = size_stats(proofs.iter().map(|(_, proof)| {
        bincode::serialize(&(proof.application_mask_commitment.clone(),))
            .expect("application-mask commitments serialize")
            .len()
    }));
    let outer_iop = size_stats(proofs.iter().map(|(_, proof)| {
        bincode::serialize(&(
            &proof.outer_sumcheck,
            proof.outer_claims,
            &proof.outer_mask_evals,
        ))
        .expect("outer IOP proof serializes")
        .len()
    }));
    let inner_iop = size_stats(proofs.iter().map(|(_, proof)| {
        bincode::serialize(&(
            &proof.inner_sumcheck,
            proof.inner_sumcheck_mask_commitment.clone(),
        ))
        .expect("inner IOP proof serializes")
        .len()
    }));
    let matrix_closing = size_stats(proofs.iter().map(|(_, proof)| {
        bincode::serialize(&proof.matrix_closing)
            .expect("matrix closing proof serializes")
            .len()
    }));
    let pcs_relation = size_stats(proofs.iter().map(|(_, proof)| {
        bincode::serialize(&proof.pcs_proof)
            .expect("PCS relation proof serializes")
            .len()
    }));
    println!(
        "full_zk_proof_sections_median_bytes: profile={HASH_PROFILE} variant={variant} application_mask_commitments={} outer_iop={} inner_iop={} matrix_closing={} pcs_relation={}",
        application_masks.median,
        outer_iop.median,
        inner_iop.median,
        matrix_closing.median,
        pcs_relation.median
    );
}

struct SizeStats {
    min: usize,
    median: usize,
    max: usize,
}

fn size_stats(values: impl IntoIterator<Item = usize>) -> SizeStats {
    let mut values = values.into_iter().collect::<Vec<_>>();
    assert!(!values.is_empty(), "proof-size corpus must not be empty");
    values.sort_unstable();
    let median = if values.len().is_multiple_of(2) {
        (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2
    } else {
        values[values.len() / 2]
    };
    SizeStats {
        min: values[0],
        median,
        max: values[values.len() - 1],
    }
}

fn sample_rng(sample: usize) -> StdRng {
    StdRng::seed_from_u64(FULL_ZK_SEED.wrapping_add(sample as u64))
}

fn env_usize(name: &str, default: usize) -> usize {
    match env::var(name) {
        Ok(raw) => raw
            .parse()
            .unwrap_or_else(|error| panic!("{name} must be a usize: {error}")),
        Err(env::VarError::NotPresent) => default,
        Err(env::VarError::NotUnicode(_)) => panic!("{name} must be valid UTF-8"),
    }
}

fn env_string(name: &str, default: &str) -> String {
    match env::var(name) {
        Ok(value) => value,
        Err(env::VarError::NotPresent) => default.to_owned(),
        Err(env::VarError::NotUnicode(_)) => panic!("{name} must be valid UTF-8"),
    }
}

fn single_proving_variant() -> Option<SingleProvingVariant> {
    let raw = env::var_os("SHA256_ZK_BENCH_SINGLE_PROVING_VARIANT")?;
    match raw
        .into_string()
        .unwrap_or_else(|_| panic!("SHA256_ZK_BENCH_SINGLE_PROVING_VARIANT must be valid UTF-8"))
        .as_str()
    {
        "no_zk_direct" => Some(SingleProvingVariant::NoZkDirect),
        "full_zk_direct" => Some(SingleProvingVariant::FullZkDirect),
        "no_zk_spark" => Some(SingleProvingVariant::NoZkSpark),
        "full_zk_spark" => Some(SingleProvingVariant::FullZkSpark),
        value => panic!(
            "SHA256_ZK_BENCH_SINGLE_PROVING_VARIANT must be no_zk_direct, full_zk_direct, no_zk_spark, or full_zk_spark, got {value}"
        ),
    }
}

fn env_flag(name: &str) -> bool {
    match env::var(name) {
        Ok(value) => matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"),
        Err(env::VarError::NotPresent) => false,
        Err(env::VarError::NotUnicode(_)) => panic!("{name} must be valid UTF-8"),
    }
}

fn benchmark_configs<Ext: ExtField>(
    shape: &spartan_whir::R1csShape<F>,
    extension: &str,
    security_bits: u32,
) -> BenchmarkConfigs {
    let num_variables = shape.num_vars.next_power_of_two().ilog2() as usize;
    let security = benchmark_security(security_bits);
    let no_zk_direct_whir = benchmark_whir_params(
        num_variables,
        extension,
        "SHA256_ZK_BENCH_NO_ZK_DIRECT_SCHEDULE",
    );
    let no_zk_spark_whir = benchmark_no_zk_spark_whir_params(num_variables, extension);
    let full_zk_direct_whir = benchmark_whir_params(
        num_variables,
        extension,
        "SHA256_ZK_BENCH_FULL_ZK_DIRECT_SCHEDULE",
    );
    let full_zk_spark_whir = benchmark_full_zk_spark_whir_params(num_variables, extension);
    let spark_whir_params = benchmark_spark_table_whir_params_for_shape::<Ext>(shape, extension);
    let no_zk_direct = SpartanSnarkConfig {
        matrix_closing: MatrixClosingMode::DirectSparse,
        security,
        whir_params: no_zk_direct_whir,
        spark_whir_params: None,
    };
    let no_zk_spark = SpartanSnarkConfig {
        matrix_closing: MatrixClosingMode::Spark,
        security,
        whir_params: no_zk_spark_whir,
        spark_whir_params: Some(spark_whir_params.clone()),
    };
    let full_zk_direct = PoseidonZkSetupConfig {
        matrix_closing: MatrixClosingMode::DirectSparse,
        security,
        whir_params: full_zk_direct_whir,
        spark_whir_params: None,
        ell_zk: env_usize("SHA256_BENCH_ZK_ELL", spartan_whir::DEFAULT_ZK_ELL),
        mask_log_inv_rate: env_usize(
            "SHA256_BENCH_ZK_MASK_LOG_INV_RATE",
            spartan_whir::DEFAULT_ZK_MASK_LOG_INV_RATE,
        ),
    };
    let full_zk_spark = PoseidonZkSetupConfig {
        matrix_closing: MatrixClosingMode::Spark,
        security,
        whir_params: full_zk_spark_whir,
        spark_whir_params: Some(spark_whir_params),
        ell_zk: full_zk_direct.ell_zk,
        mask_log_inv_rate: full_zk_direct.mask_log_inv_rate,
    };
    BenchmarkConfigs {
        no_zk_direct,
        no_zk_spark,
        full_zk_direct,
        full_zk_spark,
    }
}

fn benchmark_no_zk_spark_whir_params(num_variables: usize, extension: &str) -> WhirParams {
    match env::var_os("SHA256_ZK_BENCH_NO_ZK_SPARK_SCHEDULE")
        .or_else(|| env::var_os("SHA256_ZK_BENCH_SCHEDULE"))
    {
        None if extension == "quintic" => recommended_quintic_spark_whir_params(num_variables),
        _ => benchmark_whir_params(
            num_variables,
            extension,
            "SHA256_ZK_BENCH_NO_ZK_SPARK_SCHEDULE",
        ),
    }
}

fn benchmark_full_zk_spark_whir_params(num_variables: usize, extension: &str) -> WhirParams {
    match env::var_os("SHA256_ZK_BENCH_FULL_ZK_SPARK_SCHEDULE")
        .or_else(|| env::var_os("SHA256_ZK_BENCH_SCHEDULE"))
    {
        None if extension == "quintic" => recommended_quintic_spark_zk_whir_params(num_variables),
        _ => benchmark_whir_params(
            num_variables,
            extension,
            "SHA256_ZK_BENCH_FULL_ZK_SPARK_SCHEDULE",
        ),
    }
}

fn benchmark_spark_table_whir_params_for_shape<Ext: ExtField>(
    shape: &spartan_whir::R1csShape<F>,
    extension: &str,
) -> SparkWhirParams {
    let tables = preprocess_spark_tables(shape).expect("SPARK tables preprocess");
    let value_variables = tables.value_domain_size.ilog2() as usize;
    let fixed_value_variables = value_variables + spartan_whir::protocol::fixed_value_column_bits();
    let audit_variables = tables
        .row_memory_size
        .max(tables.col_memory_size)
        .next_power_of_two()
        .ilog2() as usize
        + spartan_whir::protocol::fixed_audit_column_bits();
    let read_variables = value_variables + spartan_whir::protocol::read_table_column_bits::<Ext>();
    let fixed_value = benchmark_spark_table_whir_params(
        fixed_value_variables,
        extension,
        "SHA256_ZK_BENCH_SPARK_FIXED_VALUE_SCHEDULE",
        match extension {
            "quintic" => recommended_quintic_spark_fixed_whir_params(fixed_value_variables),
            "octic" => recommended_octic_spark_fixed_whir_params(fixed_value_variables),
            _ => unreachable!("validated extension"),
        },
    );
    let mut read = benchmark_spark_table_whir_params(
        read_variables,
        extension,
        "SHA256_ZK_BENCH_SPARK_READ_SCHEDULE",
        match extension {
            "quintic" => recommended_quintic_spark_read_whir_params(read_variables),
            "octic" => recommended_octic_spark_read_whir_params(read_variables),
            _ => unreachable!("validated extension"),
        },
    );
    assert!(
        env::var_os("SHA256_ZK_BENCH_SPARK_READ_SCHEDULE").is_none()
            || read.round_log_inv_rates.is_empty(),
        "SHA256_ZK_BENCH_SPARK_READ_SCHEDULE must use derived round_log_inv_rates for SPARK read openings"
    );
    read.round_log_inv_rates.clear();
    SparkWhirParams {
        fixed_value,
        fixed_audit: benchmark_spark_table_whir_params(
            audit_variables,
            extension,
            "SHA256_ZK_BENCH_SPARK_FIXED_AUDIT_SCHEDULE",
            match extension {
                "quintic" => recommended_quintic_spark_fixed_whir_params(audit_variables),
                "octic" => recommended_octic_spark_fixed_whir_params(audit_variables),
                _ => unreachable!("validated extension"),
            },
        ),
        read,
    }
}

fn benchmark_security(security_bits: u32) -> SecurityConfig {
    SecurityConfig {
        security_level_bits: security_bits,
        merkle_security_bits: security_bits,
        soundness_assumption: SoundnessAssumption::JohnsonBound,
    }
}

fn benchmark_whir_params(num_variables: usize, extension: &str, variant_env: &str) -> WhirParams {
    let Some(raw) = env::var_os(variant_env).or_else(|| env::var_os("SHA256_ZK_BENCH_SCHEDULE"))
    else {
        return match (extension, variant_env) {
            ("octic", "SHA256_ZK_BENCH_NO_ZK_DIRECT_SCHEDULE")
            | ("octic", "SHA256_ZK_BENCH_NO_ZK_SPARK_SCHEDULE") => {
                recommended_octic_whir_params(num_variables)
            }
            ("octic", "SHA256_ZK_BENCH_FULL_ZK_DIRECT_SCHEDULE")
            | ("octic", "SHA256_ZK_BENCH_FULL_ZK_SPARK_SCHEDULE") => {
                recommended_octic_zk_whir_params(num_variables)
            }
            ("quintic", "SHA256_ZK_BENCH_NO_ZK_DIRECT_SCHEDULE")
            | ("quintic", "SHA256_ZK_BENCH_NO_ZK_SPARK_SCHEDULE") => {
                recommended_quintic_whir_params(num_variables)
            }
            ("quintic", "SHA256_ZK_BENCH_FULL_ZK_DIRECT_SCHEDULE")
            | ("quintic", "SHA256_ZK_BENCH_FULL_ZK_SPARK_SCHEDULE") => {
                recommended_quintic_zk_whir_params(num_variables)
            }
            _ => panic!("unsupported benchmark extension or variant: {extension} {variant_env}"),
        };
    };
    let label = raw
        .into_string()
        .unwrap_or_else(|_| panic!("{variant_env} must be valid UTF-8"));
    parse_benchmark_schedule(num_variables, extension, &label)
}

fn benchmark_spark_table_whir_params(
    num_variables: usize,
    extension: &str,
    schedule_env: &str,
    default: WhirParams,
) -> WhirParams {
    let Some(raw) = env::var_os(schedule_env) else {
        return default;
    };
    let label = raw
        .into_string()
        .unwrap_or_else(|_| panic!("{schedule_env} must be valid UTF-8"));
    parse_benchmark_schedule(num_variables, extension, &label)
}

fn parse_benchmark_schedule(num_variables: usize, extension: &str, label: &str) -> WhirParams {
    parse_whir_params_label(num_variables, extension, label)
        .unwrap_or_else(|error| panic!("unsupported SHA256_ZK_BENCH_SCHEDULE {label}: {error}"))
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_secs(4))
        .measurement_time(Duration::from_secs(12));
    targets = benchmark_sha256_full_zk
}
criterion_main!(benches);
