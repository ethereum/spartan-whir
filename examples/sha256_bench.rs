use std::{
    env,
    error::Error,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant},
};

use libloading::Library;
use p3_field::PrimeField32;
use sha2::{Digest, Sha256};
use spartan_whir::{
    compare_spark_layouts, engine::F, format_whir_params_label, import_r1cs_path,
    parse_whir_params_label, recommended_octic_spark_fixed_whir_params,
    recommended_octic_spark_read_whir_params, recommended_octic_whir_params,
    recommended_octic_zk_whir_params, MatrixClosingMode, OcticBinExtension,
    PoseidonSpartanProtocol, PoseidonWitnessGenerator, PoseidonZkProvingKey, PoseidonZkSetupConfig,
    PoseidonZkSpartanProtocol, R1csShape, SecurityConfig, SoundnessAssumption, SparkLayoutDecision,
    SparkWhirParams, SpartanSnarkConfig, WhirParams,
};
use spartan_whir::{
    protocol::{fixed_audit_column_count, fixed_value_column_bits, read_table_column_bits},
    spark::spark_col_memory_size,
};
const DEFAULT_SIZES: &[usize] = &[128, 256, 512, 1024, 2048];
const DEFAULT_POSEIDON_SECURITY_BITS: u32 = 116;

#[derive(Debug)]
struct ArtifactPaths {
    r1cs: PathBuf,
    linked_library: PathBuf,
    circuit_data: Vec<u8>,
    run_name: String,
    reused: bool,
}

struct LoadedWitnessGenerator {
    generator: PoseidonWitnessGenerator,
    _library: Library,
}

fn main() -> Result<(), Box<dyn Error>> {
    init_profile_tracing();
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workdir = env::var_os("SHA256_BENCH_WORKDIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.join("target/sha256-cache"));
    let sizes = parse_sizes()?;
    let proof_modes = parse_proof_modes()?;
    let modes = parse_modes()?;
    let repeats = parse_repeats()?;

    println!("security: {}-bit JohnsonBound", benchmark_security_bits());
    println!("sizes: {:?}", sizes);
    println!("proof_modes: {:?}", proof_modes);
    println!("modes: {:?}", modes);
    println!("repeats: {repeats}");
    if proof_modes.len() > 1 && repeats > 1 {
        println!("proof_mode_order: alternating_per_sample");
    }
    if reuse_artifacts() {
        println!("reuse_artifacts: enabled");
    }
    if spartan_whir::profiling::profile_enabled() {
        println!("profile: enabled");
    }

    for size in sizes {
        run_size(&manifest_dir, &workdir, size, &proof_modes, &modes, repeats)?;
    }

    Ok(())
}

fn init_profile_tracing() {
    if !spartan_whir::profiling::profile_enabled() {
        return;
    }
    if spartan_whir::profiling::profile_detail_enabled() {
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE)
            .try_init();
    } else {
        let _ = tracing_subscriber::fmt()
            .with_target(false)
            .without_time()
            .with_level(false)
            .try_init();
    }
}

fn run_size(
    manifest_dir: &Path,
    workdir: &Path,
    size: usize,
    proof_modes: &[ProofMode],
    modes: &[BenchMode],
    repeats: usize,
) -> Result<(), Box<dyn Error>> {
    let profile_mode = format!("sha256_{size}b");
    let _profile_context = spartan_whir::profiling::set_profile_context("bench", &profile_mode);
    let circuit = manifest_dir.join(format!("tests/circuits/sha256_{size}b.circom"));
    let size_workdir = workdir.join(format!("sha256_{size}b"));
    let message = reference_message(size);
    let expected_digest = Sha256::digest(&message);

    println!("size_bytes: {size}");

    let compile_start = Instant::now();
    let artifacts = generate_artifacts(manifest_dir, &circuit, &size_workdir, size)?;
    let compile_elapsed = compile_start.elapsed();
    if artifacts.reused {
        println!("reuse_artifacts: true");
        println!("compile_and_build_ms: 0");
        spartan_whir::profiling::record_profile_phase(
            "frontend_compile_build",
            Duration::from_secs(0),
        );
        spartan_whir::profiling::record_profile_phase("artifact_load", compile_elapsed);
    } else {
        println!("compile_and_build_ms: {}", compile_elapsed.as_millis());
        spartan_whir::profiling::record_profile_phase("frontend_compile_build", compile_elapsed);
    }

    let import_start = Instant::now();
    let imported = import_r1cs_path(&artifacts.r1cs)?;
    let shape = imported.shape;
    let input_binary = input_binary(&message);
    let loaded_generator = load_linked_witness_generator(&artifacts)?;
    let import_elapsed = import_start.elapsed();
    println!("shape_import_ms: {}", import_elapsed.as_millis());
    spartan_whir::profiling::record_profile_phase("shape_import", import_elapsed);

    let (_validation_witness, validation_public_inputs) = loaded_generator
        .generator
        .generate_witness(&input_binary, shape.num_vars, shape.num_io)?;
    let actual_digest_bits = public_digest_bits(&validation_public_inputs)?;
    let expected_digest_bits = expected_digest_bits(&expected_digest);
    if actual_digest_bits != expected_digest_bits {
        return Err(format!("digest mismatch for {size}B circuit").into());
    }

    let blocks = (size + 8) / 64 + 1;
    println!(
        "shape: blocks={} constraints={} constraints_per_block={} vars={} public={}",
        blocks,
        shape.num_cons,
        shape.num_cons / blocks,
        shape.num_vars,
        validation_public_inputs.len()
    );

    let padded_shape = shape
        .pad_regular()
        .map_err(|err| format!("padding failed for {size}B: {err}"))?;
    let spark_layout = compare_spark_layouts(&padded_shape)
        .map_err(|err| format!("Spark layout failed for {size}B: {err}"))?;
    let selected_layout = match spark_layout.decision {
        SparkLayoutDecision::SharedUnion => &spark_layout.joint,
        SparkLayoutDecision::PerMatrix => &spark_layout.per_matrix,
    };
    println!(
        "spark_layout: decision={:?} value_domain={} union_nnz={} max_matrix_nnz_padded={}",
        spark_layout.decision,
        selected_layout.value_domain_size,
        selected_layout.union_nnz,
        selected_layout.max_matrix_nnz_padded
    );
    let no_zk_direct_configs = poseidon_direct_configs(shape.num_vars, ProofMode::NoZk)?;
    let full_zk_direct_configs = poseidon_direct_configs(shape.num_vars, ProofMode::FullZk)?;
    let spark_config =
        spark_protocol_config(&padded_shape, selected_layout.value_domain_size, false)?;
    let spark_full_zk_config =
        spark_protocol_config(&padded_shape, selected_layout.value_domain_size, true)?;

    for sample in 0..repeats {
        if repeats > 1 {
            println!("sample: {}", sample + 1);
        }
        for proof_mode_position in 0..proof_modes.len() {
            let proof_mode_index = if sample.is_multiple_of(2) {
                proof_mode_position
            } else {
                proof_modes.len() - 1 - proof_mode_position
            };
            let proof_mode = proof_modes[proof_mode_index];
            for &mode in modes {
                match mode {
                    BenchMode::Direct => {
                        let direct_configs = match proof_mode {
                            ProofMode::NoZk => &no_zk_direct_configs,
                            ProofMode::FullZk => &full_zk_direct_configs,
                        };
                        for direct in direct_configs {
                            prove_and_verify(
                                proof_mode,
                                &format!("direct_sparse_{}", direct.label),
                                &direct.config,
                                &shape,
                                &loaded_generator.generator,
                                &input_binary,
                            )?;
                        }
                    }
                    BenchMode::Spark => {
                        let config = match proof_mode {
                            ProofMode::NoZk => &spark_config,
                            ProofMode::FullZk => &spark_full_zk_config,
                        };
                        prove_and_verify(
                            proof_mode,
                            "spark",
                            config,
                            &shape,
                            &loaded_generator.generator,
                            &input_binary,
                        )?;
                    }
                }
            }
        }
    }

    Ok(())
}

fn generate_artifacts(
    manifest_dir: &Path,
    circuit: &Path,
    workdir: &Path,
    size: usize,
) -> Result<ArtifactPaths, Box<dyn Error>> {
    let circom_bin = env::var_os("CIRCOM_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.join("../circom/target/debug/circom"));
    fs::create_dir_all(workdir)?;

    if reuse_artifacts() {
        if let Some(artifacts) = load_existing_artifacts(workdir, size)? {
            return Ok(artifacts);
        }
    }

    clear_previous_outputs(workdir, size)?;

    run(Command::new(&circom_bin)
        .arg(circuit)
        .arg("--prime")
        .arg("koalabear")
        .arg("--r1cs")
        .arg("--c")
        .arg("-o")
        .arg(workdir))?;

    let cpp_dir = workdir.join(format!("sha256_{size}b_cpp"));
    let linked_library = workdir.join(dynamic_library_name(size));
    build_linked_witness_library(&cpp_dir, size, &linked_library)?;
    let circuit_data = fs::read(cpp_dir.join(format!("sha256_{size}b.dat")))?;

    Ok(ArtifactPaths {
        r1cs: workdir.join(format!("sha256_{size}b.r1cs")),
        linked_library,
        circuit_data,
        run_name: format!("sha256_{size}b"),
        reused: false,
    })
}

fn load_existing_artifacts(
    workdir: &Path,
    size: usize,
) -> Result<Option<ArtifactPaths>, Box<dyn Error>> {
    let r1cs = workdir.join(format!("sha256_{size}b.r1cs"));
    let linked_library = workdir.join(dynamic_library_name(size));
    let circuit_data_path = workdir
        .join(format!("sha256_{size}b_cpp"))
        .join(format!("sha256_{size}b.dat"));

    if !(r1cs.exists() && linked_library.exists() && circuit_data_path.exists()) {
        return Ok(None);
    }

    Ok(Some(ArtifactPaths {
        r1cs,
        linked_library,
        circuit_data: fs::read(circuit_data_path)?,
        run_name: format!("sha256_{size}b"),
        reused: true,
    }))
}

fn reuse_artifacts() -> bool {
    env_flag("SHA256_BENCH_REUSE_ARTIFACTS")
}

fn env_flag(name: &str) -> bool {
    env::var_os(name)
        .and_then(|value| value.into_string().ok())
        .is_some_and(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
}

fn clear_previous_outputs(workdir: &Path, size: usize) -> Result<(), Box<dyn Error>> {
    remove_file_if_exists(&workdir.join(format!("sha256_{size}b.r1cs")))?;
    remove_file_if_exists(&workdir.join(dynamic_library_name(size)))?;

    let cpp_dir = workdir.join(format!("sha256_{size}b_cpp"));
    match fs::remove_dir_all(&cpp_dir) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(format!("failed to remove {}: {err}", cpp_dir.display()).into()),
    }
}

fn build_linked_witness_library(
    cpp_dir: &Path,
    size: usize,
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    let cxx = env::var_os("CXX").unwrap_or_else(|| OsString::from("c++"));
    let mut command = Command::new(cxx);
    command
        .arg("-std=c++11")
        .arg("-O3")
        .arg("-fPIC")
        .arg("-fvisibility=hidden")
        .arg("-UNDEBUG")
        .arg("-DCIRCOM_LINKED_WITNESS_ONLY")
        .arg("-I")
        .arg(cpp_dir)
        .arg(cpp_dir.join("calcwit.cpp"))
        .arg(cpp_dir.join("fr.cpp"))
        .arg(cpp_dir.join("main.cpp"))
        .arg(cpp_dir.join(format!("sha256_{size}b.cpp")));
    add_include_path_if_exists(&mut command, Path::new("/opt/homebrew/include"), "gmp.h");
    add_include_path_if_exists(&mut command, Path::new("/usr/local/include"), "gmp.h");
    match env::consts::OS {
        "macos" | "ios" => {
            command.arg("-dynamiclib").arg(format!(
                "-Wl,-install_name,@rpath/{}",
                output.file_name().unwrap().to_str().unwrap()
            ));
        }
        _ => {
            command.arg("-shared");
        }
    }
    run(command.arg("-o").arg(output))
}

fn add_include_path_if_exists(command: &mut Command, dir: &Path, marker: &str) {
    if dir.join(marker).exists() {
        command.arg("-I").arg(dir);
    }
}

fn load_linked_witness_generator(
    artifacts: &ArtifactPaths,
) -> Result<LoadedWitnessGenerator, Box<dyn Error>> {
    let library = unsafe { Library::new(&artifacts.linked_library)? };
    let mut load_symbol = format!("{}_load_circuit", artifacts.run_name).into_bytes();
    load_symbol.push(0);
    let mut generate_symbol = format!("{}_linked_witness", artifacts.run_name).into_bytes();
    generate_symbol.push(0);
    let mut free_symbol = format!("{}_free_circuit", artifacts.run_name).into_bytes();
    free_symbol.push(0);
    let load = unsafe { *library.get::<spartan_whir::LinkedWitnessLoadCircuitFn>(&load_symbol)? };
    let generate =
        unsafe { *library.get::<spartan_whir::LinkedWitnessGeneratorFn>(&generate_symbol)? };
    let free = unsafe { *library.get::<spartan_whir::LinkedWitnessFreeCircuitFn>(&free_symbol)? };
    // SAFETY: the symbols come from `library`, which is retained in the return
    // value and dropped after the generator.
    let generator = unsafe {
        PoseidonWitnessGenerator::linked(
            "sha256_bench",
            &artifacts.circuit_data,
            load,
            generate,
            free,
        )
    }?;
    Ok(LoadedWitnessGenerator {
        generator,
        _library: library,
    })
}

fn dynamic_library_name(size: usize) -> String {
    format!(
        "{}sha256_{size}b_witness.{}",
        env::consts::DLL_PREFIX,
        env::consts::DLL_EXTENSION
    )
}

fn prove_and_verify(
    proof_mode: ProofMode,
    label: &str,
    config: &SpartanSnarkConfig,
    shape: &R1csShape<F>,
    generator: &PoseidonWitnessGenerator,
    input_binary: &[u8],
) -> Result<(), Box<dyn Error>> {
    match proof_mode {
        ProofMode::NoZk => {
            prove_and_verify_poseidon_no_zk(label, config, shape, generator, input_binary)
        }
        ProofMode::FullZk => {
            prove_and_verify_poseidon_plonky3_full_zk(label, config, shape, generator, input_binary)
        }
    }
}

fn prove_and_verify_poseidon_plonky3_full_zk(
    label: &str,
    config: &SpartanSnarkConfig,
    shape: &R1csShape<F>,
    generator: &PoseidonWitnessGenerator,
    input_binary: &[u8],
) -> Result<(), Box<dyn Error>> {
    type Protocol = PoseidonZkSpartanProtocol<OcticBinExtension>;

    let _profile_context =
        spartan_whir::profiling::set_profile_context("poseidon-plonky3-full-zk", label);
    let setup_start = Instant::now();
    let _setup_profile = spartan_whir::profiling::profile_scope("setup");
    let (pk, vk) = PoseidonZkProvingKey::<OcticBinExtension>::setup(
        shape.clone(),
        full_zk_setup_config(config),
    )
    .map_err(|err| format!("{label} full-ZK Poseidon setup failed: {err}"))?;
    drop(_setup_profile);
    let setup_ms = setup_start.elapsed().as_millis();

    let witness_and_prove_start = Instant::now();
    let _prove_profile = spartan_whir::profiling::profile_scope("witness_and_prove");
    let (witness, public_inputs) = {
        let _profile = spartan_whir::profiling::profile_scope("linked_witness_generation");
        generator.generate_witness(input_binary, shape.num_vars, shape.num_io)?
    };
    let mut prover_challenger = spartan_whir::poseidon_zk_challenger();
    let (instance, proof) = Protocol::prove(&pk, &public_inputs, &witness, &mut prover_challenger)
        .map_err(|err| format!("{label} full-ZK Poseidon prove failed: {err}"))?;
    drop(_prove_profile);
    let witness_and_prove_ms = witness_and_prove_start.elapsed().as_millis();

    let verify_start = Instant::now();
    let _verify_profile = spartan_whir::profiling::profile_scope("verify");
    let mut verifier_challenger = spartan_whir::poseidon_zk_challenger();
    Protocol::verify(&vk, &instance, &proof, &mut verifier_challenger)
        .map_err(|err| format!("{label} full-ZK Poseidon verify failed: {err}"))?;
    drop(_verify_profile);
    let verify_ms = verify_start.elapsed().as_millis();
    let proof_size_bytes = bincode::serialize(&proof)?.len();
    let application_mask_commitments_bytes =
        bincode::serialize(&(proof.application_mask_commitment.clone(),))?.len();
    let outer_iop_bytes = bincode::serialize(&(
        &proof.outer_sumcheck,
        proof.outer_claims,
        &proof.outer_mask_evals,
    ))?
    .len();
    let inner_iop_bytes = bincode::serialize(&(
        &proof.inner_sumcheck,
        proof.inner_sumcheck_mask_commitment.clone(),
    ))?
    .len();
    let pcs_relation_bytes = bincode::serialize(&proof.pcs_proof)?.len();

    println!(
        "proof_mode: full-zk matrix_closing: {label} setup_ms={setup_ms} witness_and_prove_ms={witness_and_prove_ms} verify_ms={verify_ms} proof_size_bytes={proof_size_bytes}"
    );
    println!(
        "full_zk_proof_sections: application_mask_commitments_bytes={application_mask_commitments_bytes} outer_iop_bytes={outer_iop_bytes} inner_iop_bytes={inner_iop_bytes} pcs_relation_bytes={pcs_relation_bytes}"
    );
    Ok(())
}

fn prove_and_verify_poseidon_no_zk(
    label: &str,
    config: &SpartanSnarkConfig,
    shape: &R1csShape<F>,
    generator: &PoseidonWitnessGenerator,
    input_binary: &[u8],
) -> Result<(), Box<dyn Error>> {
    type Protocol = PoseidonSpartanProtocol<OcticBinExtension>;

    let _profile_context =
        spartan_whir::profiling::set_profile_context("poseidon-plonky3-no-zk", label);
    let setup_start = Instant::now();
    let _setup_profile = spartan_whir::profiling::profile_scope("setup");
    let (pk, vk) = Protocol::setup_with_config(shape, config)
        .map_err(|err| format!("{label} no-ZK Poseidon setup failed: {err}"))?;
    drop(_setup_profile);
    let setup_ms = setup_start.elapsed().as_millis();

    let witness_and_prove_start = Instant::now();
    let _prove_profile = spartan_whir::profiling::profile_scope("witness_and_prove");
    let (witness, public_inputs) = {
        let _profile = spartan_whir::profiling::profile_scope("linked_witness_generation");
        generator.generate_witness(input_binary, shape.num_vars, shape.num_io)?
    };
    let mut prover_challenger = spartan_whir::poseidon_challenger();
    let (instance, proof) = Protocol::prove_with_mode(
        &pk,
        &public_inputs,
        &witness,
        config.matrix_closing,
        &mut prover_challenger,
    )
    .map_err(|err| format!("{label} no-ZK Poseidon prove failed: {err}"))?;
    drop(_prove_profile);
    let witness_and_prove_ms = witness_and_prove_start.elapsed().as_millis();

    let verify_start = Instant::now();
    let _verify_profile = spartan_whir::profiling::profile_scope("verify");
    let mut verifier_challenger = spartan_whir::poseidon_challenger();
    Protocol::verify_with_mode(&vk, &instance, &proof, &mut verifier_challenger)
        .map_err(|err| format!("{label} no-ZK Poseidon verify failed: {err}"))?;
    drop(_verify_profile);
    let verify_ms = verify_start.elapsed().as_millis();
    let proof_size_bytes = bincode::serialize(&proof)?.len();

    println!(
        "proof_mode: no-zk matrix_closing: {label} setup_ms={setup_ms} witness_and_prove_ms={witness_and_prove_ms} verify_ms={verify_ms} proof_size_bytes={proof_size_bytes}"
    );
    Ok(())
}
fn run(command: &mut Command) -> Result<(), Box<dyn Error>> {
    let status = command.status()?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("command failed with status {status}: {command:?}").into())
    }
}

fn parse_sizes() -> Result<Vec<usize>, Box<dyn Error>> {
    let Some(raw) = env::var_os("SHA256_BENCH_SIZES") else {
        return Ok(DEFAULT_SIZES.to_vec());
    };
    let raw = raw
        .into_string()
        .map_err(|_| "SHA256_BENCH_SIZES must be valid UTF-8")?;
    let mut sizes = Vec::new();
    for part in raw.split(',') {
        let size = part.trim().parse::<usize>()?;
        if !DEFAULT_SIZES.contains(&size) {
            return Err(format!("unsupported SHA benchmark size: {size}").into());
        }
        sizes.push(size);
    }
    if sizes.is_empty() {
        return Err("SHA256_BENCH_SIZES must not be empty".into());
    }
    Ok(sizes)
}

#[derive(Debug, Clone, Copy)]
enum ProofMode {
    NoZk,
    FullZk,
}

fn parse_proof_modes() -> Result<Vec<ProofMode>, Box<dyn Error>> {
    let Some(raw) = env::var_os("SHA256_BENCH_PROOF_MODES") else {
        return Ok(vec![ProofMode::NoZk]);
    };
    let raw = raw
        .into_string()
        .map_err(|_| "SHA256_BENCH_PROOF_MODES must be valid UTF-8")?;
    let mut proof_modes = Vec::new();
    for part in raw.split(',') {
        match part.trim() {
            "no-zk" => proof_modes.push(ProofMode::NoZk),
            "full-zk" => proof_modes.push(ProofMode::FullZk),
            other => return Err(format!("unsupported SHA proof mode: {other}").into()),
        }
    }
    if proof_modes.is_empty() {
        return Err("SHA256_BENCH_PROOF_MODES must not be empty".into());
    }
    Ok(proof_modes)
}

#[derive(Debug, Clone, Copy)]
enum BenchMode {
    Direct,
    Spark,
}

fn parse_modes() -> Result<Vec<BenchMode>, Box<dyn Error>> {
    let Some(raw) = env::var_os("SHA256_BENCH_MODES") else {
        return Ok(vec![BenchMode::Direct, BenchMode::Spark]);
    };
    let raw = raw
        .into_string()
        .map_err(|_| "SHA256_BENCH_MODES must be valid UTF-8")?;
    let mut modes = Vec::new();
    for part in raw.split(',') {
        match part.trim() {
            "direct" | "direct-sparse" | "direct_sparse_no_spark" => modes.push(BenchMode::Direct),
            "spark" => modes.push(BenchMode::Spark),
            "both" | "all" => {
                modes.push(BenchMode::Direct);
                modes.push(BenchMode::Spark);
            }
            other => return Err(format!("unsupported SHA benchmark mode: {other}").into()),
        }
    }
    if modes.is_empty() {
        return Err("SHA256_BENCH_MODES must not be empty".into());
    }
    Ok(modes)
}

fn parse_repeats() -> Result<usize, Box<dyn Error>> {
    let Some(raw) = env::var_os("SHA256_BENCH_REPEATS") else {
        return Ok(1);
    };
    let repeats = raw
        .into_string()
        .map_err(|_| "SHA256_BENCH_REPEATS must be valid UTF-8")?
        .parse::<usize>()?;
    if repeats == 0 {
        return Err("SHA256_BENCH_REPEATS must be greater than zero".into());
    }
    Ok(repeats)
}

fn reference_message(size: usize) -> Vec<u8> {
    (0..size)
        .map(|i| (i as u8).wrapping_mul(17).wrapping_add(3))
        .collect()
}

fn input_binary(message: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(message.len() * 8 * core::mem::size_of::<u32>());
    for bit in message
        .iter()
        .flat_map(|byte| (0..8).rev().map(move |bit| u32::from((byte >> bit) & 1)))
    {
        out.extend_from_slice(&bit.to_le_bytes());
    }
    out
}

fn expected_digest_bits(digest: &[u8]) -> Vec<u8> {
    digest
        .iter()
        .flat_map(|byte| (0..8).rev().map(move |bit| (byte >> bit) & 1))
        .collect()
}

fn public_digest_bits(public_inputs: &[F]) -> Result<Vec<u8>, Box<dyn Error>> {
    public_inputs
        .iter()
        .enumerate()
        .map(|(i, x)| {
            let bit = x.as_canonical_u32();
            if bit < 2 {
                Ok(bit as u8)
            } else {
                Err(format!("public digest bit {i} is not boolean: {bit}").into())
            }
        })
        .collect()
}

fn remove_file_if_exists(path: &Path) -> Result<(), Box<dyn Error>> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(format!("failed to remove {}: {err}", path.display()).into()),
    }
}

fn protocol_config(
    matrix_closing: MatrixClosingMode,
    whir_params: WhirParams,
    spark_whir_params: Option<SparkWhirParams>,
) -> SpartanSnarkConfig {
    let security_bits = benchmark_security_bits();
    let security = SecurityConfig {
        security_level_bits: security_bits,
        merkle_security_bits: security_bits,
        soundness_assumption: SoundnessAssumption::JohnsonBound,
    };
    SpartanSnarkConfig {
        matrix_closing,
        security,
        whir_params,
        spark_whir_params,
    }
}

fn benchmark_security_bits() -> u32 {
    match env::var("SHA256_BENCH_SECURITY_BITS") {
        Ok(raw) => raw
            .parse()
            .unwrap_or_else(|err| panic!("SHA256_BENCH_SECURITY_BITS must be a u32: {err}")),
        Err(env::VarError::NotPresent) => DEFAULT_POSEIDON_SECURITY_BITS,
        Err(env::VarError::NotUnicode(_)) => {
            panic!("SHA256_BENCH_SECURITY_BITS must be valid UTF-8")
        }
    }
}

fn full_zk_setup_config(config: &SpartanSnarkConfig) -> PoseidonZkSetupConfig {
    PoseidonZkSetupConfig {
        matrix_closing: config.matrix_closing,
        security: config.security,
        whir_params: config.whir_params.clone(),
        spark_whir_params: config.spark_whir_params.clone(),
        ell_zk: env_usize("SHA256_BENCH_ZK_ELL", spartan_whir::DEFAULT_ZK_ELL),
        mask_log_inv_rate: env_usize(
            "SHA256_BENCH_ZK_MASK_LOG_INV_RATE",
            spartan_whir::DEFAULT_ZK_MASK_LOG_INV_RATE,
        ),
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    match env::var(name) {
        Ok(raw) => raw
            .parse()
            .unwrap_or_else(|err| panic!("{name} must be a usize: {err}")),
        Err(env::VarError::NotPresent) => default,
        Err(env::VarError::NotUnicode(_)) => panic!("{name} must be valid UTF-8"),
    }
}

struct DirectBenchConfig {
    label: String,
    config: SpartanSnarkConfig,
}

fn poseidon_direct_configs(
    num_vars: usize,
    proof_mode: ProofMode,
) -> Result<Vec<DirectBenchConfig>, Box<dyn Error>> {
    let num_variables = num_vars.next_power_of_two().ilog2() as usize;
    let labels = match env::var_os("SHA256_BENCH_DIRECT_SCHEDULES") {
        Some(raw) => raw
            .into_string()
            .map_err(|_| "SHA256_BENCH_DIRECT_SCHEDULES must be valid UTF-8")?
            .split(',')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>(),
        None => {
            let whir_params = match proof_mode {
                ProofMode::NoZk => recommended_octic_whir_params(num_variables),
                ProofMode::FullZk => recommended_octic_zk_whir_params(num_variables),
            };
            return Ok(vec![DirectBenchConfig {
                label: format_whir_params_label("octic", &whir_params),
                config: protocol_config(MatrixClosingMode::DirectSparse, whir_params, None),
            }]);
        }
    };
    if labels.is_empty() {
        return Err("SHA256_BENCH_DIRECT_SCHEDULES must not be empty".into());
    }

    labels
        .into_iter()
        .map(|label| {
            let whir_params = poseidon_direct_whir_params_from_label(&label, num_variables)?;
            Ok(DirectBenchConfig {
                label,
                config: protocol_config(MatrixClosingMode::DirectSparse, whir_params, None),
            })
        })
        .collect()
}

fn poseidon_direct_whir_params_from_label(
    label: &str,
    num_variables: usize,
) -> Result<WhirParams, Box<dyn Error>> {
    Ok(parse_whir_params_label(num_variables, "octic", label)?)
}

fn spark_protocol_config(
    padded_shape: &R1csShape<F>,
    value_domain_size: usize,
    full_zk: bool,
) -> Result<SpartanSnarkConfig, Box<dyn Error>> {
    let witness_vars = log2_power_of_two(padded_shape.num_vars)?;
    let value_vars = log2_power_of_two(value_domain_size)?;
    let col_memory_size =
        spark_col_memory_size(padded_shape).map_err(|err| format!("Spark layout failed: {err}"))?;
    let audit_memory_size = padded_shape
        .num_cons
        .max(col_memory_size)
        .checked_next_power_of_two()
        .ok_or("audit memory size overflow")?;
    let fixed_audit_domain_size = audit_memory_size
        .checked_mul(fixed_audit_column_count())
        .ok_or("fixed audit domain size overflow")?;
    let fixed_value_vars = value_vars + fixed_value_column_bits();
    let fixed_audit_vars = log2_power_of_two(fixed_audit_domain_size)?;
    let read_vars = value_vars + read_table_column_bits::<OcticBinExtension>();

    println!(
        "spark_vars: witness={witness_vars} fixed_value={fixed_value_vars} fixed_audit={fixed_audit_vars} read={read_vars}"
    );

    let witness = if full_zk {
        recommended_octic_zk_whir_params(witness_vars)
    } else {
        recommended_octic_whir_params(witness_vars)
    };
    let fixed_value = recommended_octic_spark_fixed_whir_params(fixed_value_vars);
    let fixed_audit = recommended_octic_spark_fixed_whir_params(fixed_audit_vars);
    let read = recommended_octic_spark_read_whir_params(read_vars);

    println!(
        "spark_schedules: witness={} fixed_value={} fixed_audit={} read={}",
        format_whir_params_label("octic", &witness),
        format_whir_params_label("octic", &fixed_value),
        format_whir_params_label("octic", &fixed_audit),
        format_whir_params_label("octic", &read)
    );

    Ok(protocol_config(
        MatrixClosingMode::Spark,
        witness,
        Some(SparkWhirParams {
            fixed_value,
            fixed_audit,
            read,
        }),
    ))
}

fn log2_power_of_two(value: usize) -> Result<usize, Box<dyn Error>> {
    if value == 0 || !value.is_power_of_two() {
        return Err(format!("expected power-of-two value, got {value}").into());
    }
    Ok(value.ilog2() as usize)
}
