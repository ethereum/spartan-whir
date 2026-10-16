use std::{
    env, fs,
    io::{self, Read},
    path::{Path, PathBuf},
    process,
    time::Instant,
};

use libloading::Library;
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_field::TwoAdicField;
use p3_maybe_rayon::prelude::current_num_threads;
use rand::distr::{Distribution, StandardUniform};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use spartan_whir::{
    engine::{ExtField, F},
    import_paths, import_r1cs_path, validate_satisfaction, LinkedWitnessFreeCircuitFn,
    LinkedWitnessGeneratorFn, LinkedWitnessLoadCircuitFn, MlePcs, Plonky3WhirPcs,
    PoseidonSetupConfig, PoseidonWitnessGenerator, PoseidonZkSetupConfig, QuarticBinExtension,
    R1csShape, R1csWitness,
};

#[cfg(feature = "poseidon1")]
use spartan_whir::{
    poseidon1_challenger as poseidon_challenger, setup_poseidon1_zk as setup_poseidon_zk,
    Poseidon1Challenger as PoseidonChallenger, Poseidon1Engine as PoseidonEngine,
    Poseidon1ProvingKey as PoseidonProvingKey, Poseidon1SpartanProtocol as PoseidonSpartanProtocol,
    Poseidon1VerifyingKey as PoseidonVerifyingKey, Poseidon1ZkProvingKey as PoseidonZkProvingKey,
    Poseidon1ZkVerifyingKey as PoseidonZkVerifyingKey,
};
#[cfg(not(feature = "poseidon1"))]
use spartan_whir::{
    poseidon_challenger, setup_poseidon_zk, PoseidonChallenger, PoseidonEngine, PoseidonProvingKey,
    PoseidonSpartanProtocol, PoseidonVerifyingKey, PoseidonZkProvingKey, PoseidonZkVerifyingKey,
};

mod poseidon_schedule_support;
use poseidon_schedule_support::{
    collect as collect_provenance, enabled_features, BenchmarkProvenance,
};

type OcticBinExtension = spartan_whir::OcticBinExtension;
type QuinticExtension = spartan_whir::QuinticExtension;

const DEFAULT_REPEATS: usize = 3;
const DEFAULT_WARMUPS: usize = 1;
const DEFAULT_MAX_ROWS: usize = 5;

#[derive(Debug)]
struct Args {
    r1cs: PathBuf,
    wtns: PathBuf,
    linked_witness_library: PathBuf,
    linked_circuit_data: PathBuf,
    linked_input: PathBuf,
    linked_run_name: Option<String>,
    report: PathBuf,
    out: PathBuf,
    labels: Option<Vec<String>>,
    case_label: Option<String>,
    extension: Option<String>,
    repeats: usize,
    warmups: usize,
    max_rows: usize,
    include_strata: bool,
    row_source: RowSource,
    proof_mode: ProofMode,
    randomize_linked_input_bits: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RowSource {
    Auto,
    Scores,
    Candidates,
    MeasurementShortlist,
    MeasurementCandidates,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProofMode {
    NoZk,
    FullZk,
}

#[derive(Debug, Serialize)]
struct HeldoutDump {
    schema_version: u32,
    provenance: BenchmarkProvenance,
    workload_identity: WorkloadIdentity,
    measurement_kind: &'static str,
    units: &'static str,
    source_report: String,
    r1cs: String,
    wtns: Option<String>,
    linked_witness_library: Option<String>,
    linked_circuit_data: Option<String>,
    linked_input: Option<String>,
    linked_run_name: Option<String>,
    repeats: usize,
    warmups: usize,
    #[serde(flatten)]
    selection: RowSelectionMetadata,
    proof_mode: &'static str,
    hash_profile: &'static str,
    rayon_threads: usize,
    verifier_measurement: &'static str,
    interleaved_repeats: bool,
    randomize_linked_input_bits: bool,
    rows: Vec<Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkloadIdentity {
    label: String,
    r1cs_sha256: String,
    constraint_work: usize,
}

#[derive(Debug, Serialize)]
struct RowSelectionMetadata {
    row_source_requested: &'static str,
    row_source_resolved: &'static str,
    max_rows: usize,
    include_strata: bool,
    extension: Option<String>,
    explicit_labels: Option<Vec<String>>,
    reference_labels: Vec<String>,
    required_labels: Vec<String>,
    selected_row_count: usize,
}

#[derive(Debug)]
struct SelectedRows {
    rows: Vec<Value>,
    metadata: RowSelectionMetadata,
}

struct MeasuredRows {
    rows: Vec<Value>,
    interleaved_repeats: bool,
}

fn main() {
    if let Err(error) = run_main() {
        eprintln!("{error}");
        process::exit(1);
    }
}

fn run_main() -> Result<(), String> {
    let args = parse_args().unwrap_or_else(|error| {
        eprintln!("{error}");
        usage();
        process::exit(2);
    });

    let report = read_json(&args.report)?;
    let provenance = collect_provenance(enabled_features())?;
    require_matching_code_provenance(&report, &provenance)?;
    let workload_identity = report_workload_identity(&report)?;
    let SelectedRows {
        rows,
        metadata: selection,
    } = select_rows(&report, &args)?;
    if rows.is_empty() {
        return Err("no heldout rows selected".to_owned());
    }
    require_selected_row_workload_identities(&rows, &workload_identity)?;

    let (shape, input_source) = load_input_source(&args)?;
    require_workload_matches_inputs(&workload_identity, &args, &shape)?;
    let measured = measure_rows(&shape, &input_source, rows, &args)?;

    let dump = HeldoutDump {
        schema_version: 5,
        provenance,
        workload_identity,
        measurement_kind: "poseidon_schedule_full_proof_heldout",
        units: "seconds",
        source_report: args.report.display().to_string(),
        r1cs: args.r1cs.display().to_string(),
        wtns: nonempty_path(&args.wtns).map(path_to_string),
        linked_witness_library: nonempty_path(&args.linked_witness_library).map(path_to_string),
        linked_circuit_data: nonempty_path(&args.linked_circuit_data).map(path_to_string),
        linked_input: nonempty_path(&args.linked_input).map(path_to_string),
        linked_run_name: args.linked_run_name.clone(),
        repeats: args.repeats,
        warmups: args.warmups,
        selection,
        proof_mode: args.proof_mode.name(),
        hash_profile: if cfg!(feature = "poseidon1") {
            "poseidon1"
        } else {
            "poseidon2"
        },
        rayon_threads: current_num_threads(),
        verifier_measurement:
            "native wall time; use RAYON_NUM_THREADS=1 as the sequential recursive-verifier proxy",
        interleaved_repeats: measured.interleaved_repeats,
        randomize_linked_input_bits: args.randomize_linked_input_bits,
        rows: measured.rows,
    };
    if let Some(parent) = args
        .out
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
    }
    let file = fs::File::create(&args.out)
        .map_err(|err| format!("failed to create {}: {err}", args.out.display()))?;
    serde_json::to_writer_pretty(file, &dump)
        .map_err(|err| format!("failed to write {}: {err}", args.out.display()))?;
    Ok(())
}

impl RowSource {
    fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Scores => "scores",
            Self::Candidates => "candidates",
            Self::MeasurementShortlist => "measurement_shortlist",
            Self::MeasurementCandidates => "measurement_candidates",
        }
    }
}

impl ProofMode {
    fn name(self) -> &'static str {
        match self {
            Self::NoZk => "no-zk",
            Self::FullZk => "full-zk",
        }
    }
}

enum InputSource {
    Static {
        witness: R1csWitness<F>,
        public_inputs: Vec<F>,
    },
    Linked(LinkedInputSource),
}

struct LinkedInputSource {
    generator: PoseidonWitnessGenerator,
    _library: Library,
    base_input: Vec<u8>,
    randomize_bits: bool,
}

impl InputSource {
    fn sample(
        &self,
        shape: &R1csShape<F>,
        sample_index: usize,
    ) -> Result<(R1csWitness<F>, Vec<F>), String> {
        match self {
            Self::Static {
                witness,
                public_inputs,
            } => Ok((witness.clone(), public_inputs.clone())),
            Self::Linked(source) => {
                let input = source.input_for_sample(sample_index);
                source
                    .generator
                    .generate_witness(&input, shape.num_vars, shape.num_io)
                    .map_err(|err| format!("linked witness generation failed: {err}"))
            }
        }
    }
}

impl LinkedInputSource {
    fn input_for_sample(&self, sample_index: usize) -> Vec<u8> {
        if !self.randomize_bits {
            return self.base_input.clone();
        }
        randomize_binary_field_input(&self.base_input, sample_index as u64)
    }
}

fn load_input_source(args: &Args) -> Result<(R1csShape<F>, InputSource), String> {
    if has_linked_witness_args(args) {
        load_linked_input_source(args)
    } else {
        let (shape, witness, public_inputs) = import_paths(&args.r1cs, &args.wtns)
            .map_err(|err| format!("failed to import frontend artifacts: {err}"))?;
        Ok((
            shape,
            InputSource::Static {
                witness,
                public_inputs,
            },
        ))
    }
}

fn load_linked_input_source(args: &Args) -> Result<(R1csShape<F>, InputSource), String> {
    let imported = import_r1cs_path(&args.r1cs)
        .map_err(|err| format!("failed to import {}: {err}", args.r1cs.display()))?;
    let circuit_data = fs::read(&args.linked_circuit_data).map_err(|err| {
        format!(
            "failed to read {}: {err}",
            args.linked_circuit_data.display()
        )
    })?;
    let base_input = fs::read(&args.linked_input)
        .map_err(|err| format!("failed to read {}: {err}", args.linked_input.display()))?;
    let run_name = args
        .linked_run_name
        .as_deref()
        .ok_or_else(|| "--linked-run-name is required for linked witness input".to_owned())?;

    let library = unsafe { Library::new(&args.linked_witness_library) }.map_err(|err| {
        format!(
            "failed to load {}: {err}",
            args.linked_witness_library.display()
        )
    })?;
    let load_symbol = linked_symbol(run_name, "load_circuit");
    let generate_symbol = linked_symbol(run_name, "linked_witness");
    let free_symbol = linked_symbol(run_name, "free_circuit");
    let load = unsafe { library.get::<LinkedWitnessLoadCircuitFn>(&load_symbol) }
        .map_err(|err| {
            format!(
                "failed to load symbol {}: {err}",
                display_symbol(&load_symbol)
            )
        })
        .map(|symbol| *symbol)?;
    let generate = unsafe { library.get::<LinkedWitnessGeneratorFn>(&generate_symbol) }
        .map_err(|err| {
            format!(
                "failed to load symbol {}: {err}",
                display_symbol(&generate_symbol)
            )
        })
        .map(|symbol| *symbol)?;
    let free = unsafe { library.get::<LinkedWitnessFreeCircuitFn>(&free_symbol) }
        .map_err(|err| {
            format!(
                "failed to load symbol {}: {err}",
                display_symbol(&free_symbol)
            )
        })
        .map(|symbol| *symbol)?;
    // SAFETY: the callbacks come from `library`, which remains loaded until
    // after the generator is dropped.
    let generator = unsafe {
        PoseidonWitnessGenerator::linked(
            "poseidon_schedule_heldout",
            &circuit_data,
            load,
            generate,
            free,
        )
    }
    .map_err(|err| format!("failed to initialize linked witness generator: {err}"))?;
    let (witness, public_inputs) = generator
        .generate_witness(&base_input, imported.shape.num_vars, imported.shape.num_io)
        .map_err(|err| format!("linked witness generation failed: {err}"))?;
    validate_satisfaction(&imported.shape, &witness, &public_inputs)
        .map_err(|err| format!("linked witness does not satisfy R1CS: {err}"))?;
    if args.randomize_linked_input_bits {
        let randomized_input = randomize_binary_field_input(&base_input, 0);
        let (randomized_witness, randomized_public_inputs) = generator
            .generate_witness(
                &randomized_input,
                imported.shape.num_vars,
                imported.shape.num_io,
            )
            .map_err(|err| format!("randomized linked witness generation failed: {err}"))?;
        validate_satisfaction(
            &imported.shape,
            &randomized_witness,
            &randomized_public_inputs,
        )
        .map_err(|err| format!("randomized linked witness does not satisfy R1CS: {err}"))?;
    }
    Ok((
        imported.shape,
        InputSource::Linked(LinkedInputSource {
            generator,
            _library: library,
            base_input,
            randomize_bits: args.randomize_linked_input_bits,
        }),
    ))
}

fn linked_symbol(run_name: &str, suffix: &str) -> Vec<u8> {
    let mut symbol = format!("{run_name}_{suffix}").into_bytes();
    symbol.push(0);
    symbol
}

fn display_symbol(symbol: &[u8]) -> String {
    String::from_utf8_lossy(symbol.strip_suffix(&[0]).unwrap_or(symbol)).into_owned()
}

fn nonempty_path(path: &PathBuf) -> Option<&Path> {
    (!path.as_os_str().is_empty()).then_some(path.as_path())
}

fn path_to_string(path: &Path) -> String {
    path.display().to_string()
}

struct Measurement {
    median_seconds: f64,
    mean_seconds: f64,
    median_ci_seconds: (f64, f64),
    samples_seconds: Vec<f64>,
    verifier_median_seconds: f64,
    verifier_mean_seconds: f64,
    verifier_median_ci_seconds: (f64, f64),
    verifier_samples_seconds: Vec<f64>,
    proof_size_min_bytes: usize,
    proof_size_median_bytes: usize,
    proof_size_max_bytes: usize,
}

struct RowToMeasure {
    original_index: usize,
    row: Value,
    label: String,
    setup_config: RowSetupConfig,
}

enum RowSetupConfig {
    NoZk(PoseidonSetupConfig),
    FullZk(PoseidonZkSetupConfig),
}

struct PreparedFullZkRow<Ext>
where
    Ext: ExtField,
    StandardUniform: Distribution<Ext>,
{
    original_index: usize,
    row: Value,
    label: String,
    pk: PoseidonZkProvingKey<Ext>,
    vk: PoseidonZkVerifyingKey<Ext>,
}

enum PreparedFullZkRowAny {
    Quartic(PreparedFullZkRow<QuarticBinExtension>),
    Quintic(PreparedFullZkRow<QuinticExtension>),
    Octic(PreparedFullZkRow<OcticBinExtension>),
}

impl PreparedFullZkRowAny {
    fn original_index(&self) -> usize {
        match self {
            Self::Quartic(row) => row.original_index,
            Self::Quintic(row) => row.original_index,
            Self::Octic(row) => row.original_index,
        }
    }

    fn label(&self) -> &str {
        match self {
            Self::Quartic(row) => &row.label,
            Self::Quintic(row) => &row.label,
            Self::Octic(row) => &row.label,
        }
    }

    fn into_row(self) -> Value {
        match self {
            Self::Quartic(row) => row.row,
            Self::Quintic(row) => row.row,
            Self::Octic(row) => row.row,
        }
    }

    fn prove_once(
        &self,
        witness: &R1csWitness<F>,
        public_inputs: &[F],
        phase: &'static str,
    ) -> Result<(), String> {
        match self {
            Self::Quartic(row) => prove_once(row, witness, public_inputs, phase),
            Self::Quintic(row) => prove_once(row, witness, public_inputs, phase),
            Self::Octic(row) => prove_once(row, witness, public_inputs, phase),
        }
    }

    fn prove_timed(
        &self,
        witness: &R1csWitness<F>,
        public_inputs: &[F],
    ) -> Result<(f64, f64, usize), String> {
        match self {
            Self::Quartic(row) => prove_timed(row, witness, public_inputs),
            Self::Quintic(row) => prove_timed(row, witness, public_inputs),
            Self::Octic(row) => prove_timed(row, witness, public_inputs),
        }
    }
}

struct PreparedNoZkRow<Ext>
where
    Ext: ExtField + TwoAdicField,
{
    original_index: usize,
    row: Value,
    label: String,
    pk: PoseidonProvingKey<Ext>,
    vk: PoseidonVerifyingKey<Ext>,
}

enum PreparedNoZkRowAny {
    Quartic(PreparedNoZkRow<QuarticBinExtension>),
    Quintic(PreparedNoZkRow<QuinticExtension>),
    Octic(PreparedNoZkRow<OcticBinExtension>),
}

impl PreparedNoZkRowAny {
    fn original_index(&self) -> usize {
        match self {
            Self::Quartic(row) => row.original_index,
            Self::Quintic(row) => row.original_index,
            Self::Octic(row) => row.original_index,
        }
    }

    fn label(&self) -> &str {
        match self {
            Self::Quartic(row) => &row.label,
            Self::Quintic(row) => &row.label,
            Self::Octic(row) => &row.label,
        }
    }

    fn into_row(self) -> Value {
        match self {
            Self::Quartic(row) => row.row,
            Self::Quintic(row) => row.row,
            Self::Octic(row) => row.row,
        }
    }

    fn prove_once(
        &self,
        witness: &R1csWitness<F>,
        public_inputs: &[F],
        phase: &'static str,
    ) -> Result<(), String> {
        match self {
            Self::Quartic(row) => prove_no_zk_once(row, witness, public_inputs, phase),
            Self::Quintic(row) => prove_no_zk_once(row, witness, public_inputs, phase),
            Self::Octic(row) => prove_no_zk_once(row, witness, public_inputs, phase),
        }
    }

    fn prove_timed(
        &self,
        witness: &R1csWitness<F>,
        public_inputs: &[F],
    ) -> Result<(f64, f64, usize), String> {
        match self {
            Self::Quartic(row) => prove_no_zk_timed(row, witness, public_inputs),
            Self::Quintic(row) => prove_no_zk_timed(row, witness, public_inputs),
            Self::Octic(row) => prove_no_zk_timed(row, witness, public_inputs),
        }
    }
}

fn measure_rows(
    shape: &R1csShape<F>,
    input_source: &InputSource,
    rows: Vec<Value>,
    args: &Args,
) -> Result<MeasuredRows, String> {
    let mut measured = vec![Value::Null; rows.len()];
    let mut quartic = Vec::new();
    let mut quintic = Vec::new();
    let mut octic = Vec::new();
    for (index, row) in rows.into_iter().enumerate() {
        let label = row
            .get("label")
            .and_then(Value::as_str)
            .unwrap_or("<missing-label>")
            .to_owned();
        let extension = row
            .get("extension")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("{label}: missing extension"))?
            .to_owned();
        let setup_config = row
            .get("setup_config")
            .cloned()
            .ok_or_else(|| format!("{label}: missing setup_config"))?;
        let setup_config = match args.proof_mode {
            ProofMode::NoZk => RowSetupConfig::NoZk(
                serde_json::from_value(setup_config)
                    .map_err(|err| format!("{label}: invalid no-ZK setup_config: {err}"))?,
            ),
            ProofMode::FullZk => RowSetupConfig::FullZk(
                serde_json::from_value(setup_config)
                    .map_err(|err| format!("{label}: invalid full-ZK setup_config: {err}"))?,
            ),
        };
        let row_to_measure = RowToMeasure {
            original_index: index,
            row,
            label,
            setup_config,
        };
        match extension.as_str() {
            "quartic" => quartic.push(row_to_measure),
            "quintic" => quintic.push(row_to_measure),
            "octic" => octic.push(row_to_measure),
            other => return Err(format!("unsupported extension {other}")),
        }
    }
    let interleaved_repeats = match args.proof_mode {
        ProofMode::FullZk => measure_full_zk_rows_interleaved(
            shape,
            input_source,
            quartic,
            quintic,
            octic,
            args,
            &mut measured,
        )?,
        ProofMode::NoZk => measure_no_zk_rows_interleaved(
            shape,
            input_source,
            quartic,
            quintic,
            octic,
            args,
            &mut measured,
        )?,
    };
    let mut rows = measured
        .into_iter()
        .enumerate()
        .map(|(index, row)| {
            if row.is_null() {
                Err(format!("internal error: missing measured row {index}"))
            } else {
                Ok(row)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    if interleaved_repeats {
        annotate_paired_comparisons(&mut rows)?;
    }
    Ok(MeasuredRows {
        rows,
        interleaved_repeats,
    })
}

fn measure_no_zk_rows_interleaved(
    shape: &R1csShape<F>,
    input_source: &InputSource,
    quartic: Vec<RowToMeasure>,
    quintic: Vec<RowToMeasure>,
    octic: Vec<RowToMeasure>,
    args: &Args,
    measured: &mut [Value],
) -> Result<bool, String> {
    if quartic.is_empty() && quintic.is_empty() && octic.is_empty() {
        return Ok(false);
    }
    let mut prepared = Vec::with_capacity(quartic.len() + quintic.len() + octic.len());
    prepared.extend(
        prepare_no_zk_extension_rows::<QuarticBinExtension>(shape, quartic)?
            .into_iter()
            .map(PreparedNoZkRowAny::Quartic),
    );
    prepared.extend(
        prepare_no_zk_extension_rows::<QuinticExtension>(shape, quintic)?
            .into_iter()
            .map(PreparedNoZkRowAny::Quintic),
    );
    prepared.extend(
        prepare_no_zk_extension_rows::<OcticBinExtension>(shape, octic)?
            .into_iter()
            .map(PreparedNoZkRowAny::Octic),
    );

    for (warmup, order) in round_robin_orders(prepared.len(), args.warmups, 0xa51c_0000)
        .into_iter()
        .enumerate()
    {
        let (witness, public_inputs) = input_source.sample(shape, usize::MAX - warmup)?;
        for index in order {
            prepared[index].prove_once(&witness, &public_inputs, "warmup")?;
        }
    }

    let repeat_orders = round_robin_orders(prepared.len(), args.repeats, 0);
    let interleaved_repeats =
        orders_cover_each_row_once(&repeat_orders, prepared.len(), args.repeats);
    let mut samples = vec![Vec::with_capacity(args.repeats); prepared.len()];
    let mut verifier_samples = vec![Vec::with_capacity(args.repeats); prepared.len()];
    let mut proof_sizes = vec![Vec::with_capacity(args.repeats); prepared.len()];
    for (repeat, order) in repeat_orders.into_iter().enumerate() {
        let (witness, public_inputs) = input_source.sample(shape, repeat)?;
        for index in order {
            let (elapsed, verifier_elapsed, proof_size) =
                prepared[index].prove_timed(&witness, &public_inputs)?;
            samples[index].push(elapsed);
            verifier_samples[index].push(verifier_elapsed);
            proof_sizes[index].push(proof_size);
        }
    }

    for (prepared_index, prepared_row) in prepared.into_iter().enumerate() {
        let original_index = prepared_row.original_index();
        let label = prepared_row.label().to_owned();
        let row = prepared_row.into_row();
        let measurement = summarize_samples(
            &samples[prepared_index],
            &verifier_samples[prepared_index],
            &proof_sizes[prepared_index],
        );
        measured[original_index] =
            measured_row(shape, row, &label, measurement, args, interleaved_repeats)?;
    }
    Ok(interleaved_repeats)
}

fn prepare_no_zk_extension_rows<Ext>(
    shape: &R1csShape<F>,
    rows: Vec<RowToMeasure>,
) -> Result<Vec<PreparedNoZkRow<Ext>>, String>
where
    Ext: ExtField + TwoAdicField,
    PoseidonChallenger: CanObserve<<Plonky3WhirPcs as MlePcs<PoseidonEngine<Ext>>>::Commitment>
        + CanSampleUniformBits<F>
        + FieldChallenger<F>
        + GrindingChallenger<Witness = F>,
{
    let mut prepared = Vec::with_capacity(rows.len());
    for row in rows {
        let RowSetupConfig::NoZk(setup_config) = row.setup_config else {
            return Err(format!("{}: expected no-ZK setup config", row.label));
        };
        let (pk, vk) = PoseidonSpartanProtocol::<Ext>::setup_with_config(shape, &setup_config)
            .map_err(|err| format!("{}: no-ZK setup failed: {err:?}", row.label))?;
        prepared.push(PreparedNoZkRow {
            original_index: row.original_index,
            row: row.row,
            label: row.label,
            pk,
            vk,
        });
    }
    Ok(prepared)
}

fn prove_no_zk_once<Ext>(
    row: &PreparedNoZkRow<Ext>,
    witness: &R1csWitness<F>,
    public_inputs: &[F],
    phase: &'static str,
) -> Result<(), String>
where
    Ext: ExtField + TwoAdicField,
    PoseidonChallenger: CanObserve<<Plonky3WhirPcs as MlePcs<PoseidonEngine<Ext>>>::Commitment>
        + CanSampleUniformBits<F>
        + FieldChallenger<F>
        + GrindingChallenger<Witness = F>,
{
    let mut prover_challenger = poseidon_challenger();
    let (instance, proof) = PoseidonSpartanProtocol::<Ext>::prove_with_mode(
        &row.pk,
        public_inputs,
        witness,
        row.pk.matrix_closing,
        &mut prover_challenger,
    )
    .map_err(|err| format!("{}: {phase} no-ZK failed: {err:?}", row.label))?;
    let mut verifier_challenger = poseidon_challenger();
    PoseidonSpartanProtocol::<Ext>::verify_with_mode(
        &row.vk,
        &instance,
        &proof,
        &mut verifier_challenger,
    )
    .map_err(|err| format!("{}: {phase} no-ZK verify failed: {err:?}", row.label))
}

fn prove_no_zk_timed<Ext>(
    row: &PreparedNoZkRow<Ext>,
    witness: &R1csWitness<F>,
    public_inputs: &[F],
) -> Result<(f64, f64, usize), String>
where
    Ext: ExtField + TwoAdicField,
    PoseidonChallenger: CanObserve<<Plonky3WhirPcs as MlePcs<PoseidonEngine<Ext>>>::Commitment>
        + CanSampleUniformBits<F>
        + FieldChallenger<F>
        + GrindingChallenger<Witness = F>,
{
    let mut prover_challenger = poseidon_challenger();
    let start = Instant::now();
    let (instance, proof) = PoseidonSpartanProtocol::<Ext>::prove_with_mode(
        &row.pk,
        public_inputs,
        witness,
        row.pk.matrix_closing,
        &mut prover_challenger,
    )
    .map_err(|err| format!("{}: no-ZK prove failed: {err:?}", row.label))?;
    let elapsed = start.elapsed().as_secs_f64();
    let proof_size = bincode::serialize(&proof)
        .map_err(|err| format!("{}: no-ZK proof serialization failed: {err}", row.label))?
        .len();
    let mut verifier_challenger = poseidon_challenger();
    let verify_start = Instant::now();
    PoseidonSpartanProtocol::<Ext>::verify_with_mode(
        &row.vk,
        &instance,
        &proof,
        &mut verifier_challenger,
    )
    .map_err(|err| format!("{}: no-ZK verify failed: {err:?}", row.label))?;
    let verifier_elapsed = verify_start.elapsed().as_secs_f64();
    Ok((elapsed, verifier_elapsed, proof_size))
}

fn measure_full_zk_rows_interleaved(
    shape: &R1csShape<F>,
    input_source: &InputSource,
    quartic: Vec<RowToMeasure>,
    quintic: Vec<RowToMeasure>,
    octic: Vec<RowToMeasure>,
    args: &Args,
    measured: &mut [Value],
) -> Result<bool, String> {
    if quartic.is_empty() && quintic.is_empty() && octic.is_empty() {
        return Ok(false);
    }
    let mut prepared = Vec::with_capacity(quartic.len() + quintic.len() + octic.len());
    prepared.extend(
        prepare_full_zk_extension_rows::<QuarticBinExtension>(shape, quartic)?
            .into_iter()
            .map(PreparedFullZkRowAny::Quartic),
    );
    prepared.extend(
        prepare_full_zk_extension_rows::<QuinticExtension>(shape, quintic)?
            .into_iter()
            .map(PreparedFullZkRowAny::Quintic),
    );
    prepared.extend(
        prepare_full_zk_extension_rows::<OcticBinExtension>(shape, octic)?
            .into_iter()
            .map(PreparedFullZkRowAny::Octic),
    );

    for (warmup, order) in round_robin_orders(prepared.len(), args.warmups, 0xa51c_0000)
        .into_iter()
        .enumerate()
    {
        let (witness, public_inputs) = input_source.sample(shape, usize::MAX - warmup)?;
        for index in order {
            prepared[index].prove_once(&witness, &public_inputs, "warmup")?;
        }
    }

    let repeat_orders = round_robin_orders(prepared.len(), args.repeats, 0);
    let interleaved_repeats =
        orders_cover_each_row_once(&repeat_orders, prepared.len(), args.repeats);
    let mut samples = vec![Vec::with_capacity(args.repeats); prepared.len()];
    let mut verifier_samples = vec![Vec::with_capacity(args.repeats); prepared.len()];
    let mut proof_sizes = vec![Vec::with_capacity(args.repeats); prepared.len()];
    for (repeat, order) in repeat_orders.into_iter().enumerate() {
        let (witness, public_inputs) = input_source.sample(shape, repeat)?;
        for index in order {
            let (elapsed, verifier_elapsed, proof_size) =
                prepared[index].prove_timed(&witness, &public_inputs)?;
            samples[index].push(elapsed);
            verifier_samples[index].push(verifier_elapsed);
            proof_sizes[index].push(proof_size);
        }
    }

    for (prepared_index, prepared_row) in prepared.into_iter().enumerate() {
        let original_index = prepared_row.original_index();
        let label = prepared_row.label().to_owned();
        let row = prepared_row.into_row();
        let measurement = summarize_samples(
            &samples[prepared_index],
            &verifier_samples[prepared_index],
            &proof_sizes[prepared_index],
        );
        measured[original_index] =
            measured_row(shape, row, &label, measurement, args, interleaved_repeats)?;
    }
    Ok(interleaved_repeats)
}

fn prepare_full_zk_extension_rows<Ext>(
    shape: &R1csShape<F>,
    rows: Vec<RowToMeasure>,
) -> Result<Vec<PreparedFullZkRow<Ext>>, String>
where
    Ext: ExtField,
    StandardUniform: Distribution<Ext>,
    PoseidonChallenger: CanObserve<<Plonky3WhirPcs as MlePcs<PoseidonEngine<Ext>>>::Commitment>,
{
    let mut prepared = Vec::with_capacity(rows.len());
    for row in rows {
        let RowSetupConfig::FullZk(zk_config) = row.setup_config else {
            return Err(format!("{}: expected full-ZK setup config", row.label));
        };
        let (pk, vk) = setup_poseidon_zk::<Ext>(shape.clone(), zk_config)
            .map_err(|err| format!("{}: full-ZK setup failed: {err:?}", row.label))?;
        prepared.push(PreparedFullZkRow {
            original_index: row.original_index,
            row: row.row,
            label: row.label,
            pk,
            vk,
        });
    }
    Ok(prepared)
}

fn prove_once<Ext>(
    row: &PreparedFullZkRow<Ext>,
    witness: &R1csWitness<F>,
    public_inputs: &[F],
    phase: &'static str,
) -> Result<(), String>
where
    Ext: ExtField,
    StandardUniform: Distribution<Ext>,
    PoseidonChallenger: CanObserve<<Plonky3WhirPcs as MlePcs<PoseidonEngine<Ext>>>::Commitment>,
{
    let proof = row
        .pk
        .prove(witness.clone(), public_inputs.to_vec())
        .map_err(|err| format!("{}: {phase} failed: {err:?}", row.label))?;
    row.vk
        .verify(&proof.instance.public_inputs, &proof)
        .map_err(|err| format!("{}: {phase} verify failed: {err:?}", row.label))
}

fn prove_timed<Ext>(
    row: &PreparedFullZkRow<Ext>,
    witness: &R1csWitness<F>,
    public_inputs: &[F],
) -> Result<(f64, f64, usize), String>
where
    Ext: ExtField,
    StandardUniform: Distribution<Ext>,
    PoseidonChallenger: CanObserve<<Plonky3WhirPcs as MlePcs<PoseidonEngine<Ext>>>::Commitment>,
{
    let start = Instant::now();
    let proof = row
        .pk
        .prove(witness.clone(), public_inputs.to_vec())
        .map_err(|err| format!("{}: prove failed: {err:?}", row.label))?;
    let elapsed = start.elapsed().as_secs_f64();
    let proof_size = bincode::serialize(&proof.proof)
        .map_err(|err| format!("{}: proof serialization failed: {err}", row.label))?
        .len();
    let verify_start = Instant::now();
    row.vk
        .verify(&proof.instance.public_inputs, &proof)
        .map_err(|err| format!("{}: verify failed: {err:?}", row.label))?;
    let verifier_elapsed = verify_start.elapsed().as_secs_f64();
    Ok((elapsed, verifier_elapsed, proof_size))
}

fn measured_row(
    shape: &R1csShape<F>,
    mut row: Value,
    label: &str,
    measurement: Measurement,
    args: &Args,
    interleaved_repeats: bool,
) -> Result<Value, String> {
    let object = row
        .as_object_mut()
        .ok_or_else(|| format!("{label}: row is not an object"))?;
    object.insert(
        "measured_seconds".to_owned(),
        Value::from(measurement.median_seconds),
    );
    object.insert(
        "heldout_prove_seconds".to_owned(),
        Value::from(measurement.median_seconds),
    );
    object.insert(
        "heldout_mean_seconds".to_owned(),
        Value::from(measurement.mean_seconds),
    );
    object.insert(
        "heldout_median_ci_seconds".to_owned(),
        Value::Array(vec![
            Value::from(measurement.median_ci_seconds.0),
            Value::from(measurement.median_ci_seconds.1),
        ]),
    );
    object.insert("heldout_median_ci_confidence".to_owned(), Value::from(0.95));
    object.insert(
        "constraint_work".to_owned(),
        Value::from(shape.num_cons as u64),
    );
    object.insert(
        "witness_work".to_owned(),
        Value::from(shape.num_vars as u64),
    );
    if let Some(case_label) = &args.case_label {
        object.insert(
            "heldout_case_label".to_owned(),
            Value::from(case_label.clone()),
        );
    }
    object.insert(
        "heldout_repeats".to_owned(),
        Value::from(args.repeats as u64),
    );
    object.insert(
        "heldout_warmups".to_owned(),
        Value::from(args.warmups as u64),
    );
    object.insert(
        "heldout_interleaved".to_owned(),
        Value::from(interleaved_repeats),
    );
    object.insert(
        "heldout_proof_mode".to_owned(),
        Value::from(args.proof_mode.name()),
    );
    object.insert(
        "heldout_randomized_linked_input_bits".to_owned(),
        Value::from(args.randomize_linked_input_bits),
    );
    object.insert(
        "heldout_samples_seconds".to_owned(),
        Value::Array(
            measurement
                .samples_seconds
                .iter()
                .copied()
                .map(Value::from)
                .collect(),
        ),
    );
    object.insert(
        "heldout_verify_seconds".to_owned(),
        Value::from(measurement.verifier_median_seconds),
    );
    object.insert(
        "heldout_verify_mean_seconds".to_owned(),
        Value::from(measurement.verifier_mean_seconds),
    );
    object.insert(
        "heldout_verify_median_ci_seconds".to_owned(),
        Value::Array(vec![
            Value::from(measurement.verifier_median_ci_seconds.0),
            Value::from(measurement.verifier_median_ci_seconds.1),
        ]),
    );
    object.insert(
        "heldout_verify_samples_seconds".to_owned(),
        Value::Array(
            measurement
                .verifier_samples_seconds
                .iter()
                .copied()
                .map(Value::from)
                .collect(),
        ),
    );
    object.insert(
        "heldout_proof_size_min_bytes".to_owned(),
        Value::from(measurement.proof_size_min_bytes as u64),
    );
    object.insert(
        "heldout_proof_size_median_bytes".to_owned(),
        Value::from(measurement.proof_size_median_bytes as u64),
    );
    object.insert(
        "heldout_proof_size_max_bytes".to_owned(),
        Value::from(measurement.proof_size_max_bytes as u64),
    );
    Ok(row)
}

fn summarize_samples(
    samples: &[f64],
    verifier_samples: &[f64],
    proof_sizes: &[usize],
) -> Measurement {
    assert_eq!(samples.len(), proof_sizes.len());
    assert_eq!(samples.len(), verifier_samples.len());
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mut sorted_sizes = proof_sizes.to_vec();
    sorted_sizes.sort_unstable();
    let median_seconds = sorted[sorted.len() / 2];
    let mean_seconds = samples.iter().sum::<f64>() / samples.len() as f64;
    let mut sorted_verifier = verifier_samples.to_vec();
    sorted_verifier.sort_by(f64::total_cmp);
    let verifier_median_seconds = sorted_verifier[sorted_verifier.len() / 2];
    let verifier_mean_seconds =
        verifier_samples.iter().sum::<f64>() / verifier_samples.len() as f64;
    Measurement {
        median_seconds,
        mean_seconds,
        median_ci_seconds: bootstrap_median_ci(samples, 2_000, 0x5eed_5eed),
        samples_seconds: samples.to_vec(),
        verifier_median_seconds,
        verifier_mean_seconds,
        verifier_median_ci_seconds: bootstrap_median_ci(verifier_samples, 2_000, 0x5eed_71f1),
        verifier_samples_seconds: verifier_samples.to_vec(),
        proof_size_min_bytes: sorted_sizes[0],
        proof_size_median_bytes: sorted_sizes[sorted_sizes.len() / 2],
        proof_size_max_bytes: sorted_sizes[sorted_sizes.len() - 1],
    }
}

fn bootstrap_median_ci(samples: &[f64], bootstrap_samples: usize, seed: u64) -> (f64, f64) {
    if samples.is_empty() {
        return (0.0, 0.0);
    }
    if samples.len() == 1 || bootstrap_samples == 0 {
        return (samples[0], samples[0]);
    }
    let mut rng = SplitMix64::new(seed ^ samples.len() as u64);
    let mut medians = Vec::with_capacity(bootstrap_samples);
    let mut draw = vec![0.0; samples.len()];
    for _ in 0..bootstrap_samples {
        for item in &mut draw {
            let index = (rng.next_u64() as usize) % samples.len();
            *item = samples[index];
        }
        draw.sort_by(f64::total_cmp);
        medians.push(draw[draw.len() / 2]);
    }
    medians.sort_by(f64::total_cmp);
    let low = percentile_index(bootstrap_samples, 0.025);
    let high = percentile_index(bootstrap_samples, 0.975);
    (medians[low], medians[high])
}

fn annotate_paired_comparisons(rows: &mut [Value]) -> Result<(), String> {
    if rows.is_empty() {
        return Ok(());
    }
    let fastest_index = rows
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| {
            measured_seconds(left)
                .unwrap_or(f64::INFINITY)
                .total_cmp(&measured_seconds(right).unwrap_or(f64::INFINITY))
        })
        .map(|(index, _)| index)
        .ok_or_else(|| "heldout rows are empty".to_owned())?;
    let fastest_median = measured_seconds(&rows[fastest_index])?;
    let fastest_samples = heldout_samples(&rows[fastest_index])?;

    for (index, row) in rows.iter_mut().enumerate() {
        let median = measured_seconds(row)?;
        let samples = heldout_samples(row)?;
        if samples.len() != fastest_samples.len() {
            return Err(format!(
                "heldout row {index} has {} samples, fastest row has {}",
                samples.len(),
                fastest_samples.len()
            ));
        }
        let relative = median / fastest_median - 1.0;
        let ci = bootstrap_paired_relative_median_ci(
            &samples,
            &fastest_samples,
            2_000,
            0x51ec_7100 ^ index as u64,
        );
        let object = row
            .as_object_mut()
            .ok_or_else(|| format!("heldout row {index} is not an object"))?;
        object.insert(
            "heldout_fastest_row_index".to_owned(),
            Value::from(fastest_index as u64),
        );
        object.insert(
            "heldout_relative_median_difference".to_owned(),
            Value::from(relative),
        );
        object.insert(
            "heldout_paired_relative_median_ci".to_owned(),
            Value::Array(vec![Value::from(ci.0), Value::from(ci.1)]),
        );
        object.insert("heldout_paired_ci_confidence".to_owned(), Value::from(0.95));
    }
    Ok(())
}

fn measured_seconds(row: &Value) -> Result<f64, String> {
    row.get("heldout_prove_seconds")
        .and_then(Value::as_f64)
        .ok_or_else(|| "heldout row is missing heldout_prove_seconds".to_owned())
}

fn heldout_samples(row: &Value) -> Result<Vec<f64>, String> {
    row.get("heldout_samples_seconds")
        .and_then(Value::as_array)
        .ok_or_else(|| "heldout row is missing heldout_samples_seconds".to_owned())?
        .iter()
        .map(|value| {
            value
                .as_f64()
                .ok_or_else(|| "heldout sample is not a number".to_owned())
        })
        .collect()
}

fn bootstrap_paired_relative_median_ci(
    candidate: &[f64],
    baseline: &[f64],
    bootstrap_samples: usize,
    seed: u64,
) -> (f64, f64) {
    assert_eq!(candidate.len(), baseline.len());
    if candidate.is_empty() {
        return (0.0, 0.0);
    }
    if candidate.len() == 1 || bootstrap_samples == 0 {
        let relative = candidate[0] / baseline[0] - 1.0;
        return (relative, relative);
    }
    let mut rng = SplitMix64::new(seed ^ candidate.len() as u64);
    let mut candidate_draw = vec![0.0; candidate.len()];
    let mut baseline_draw = vec![0.0; baseline.len()];
    let mut differences = Vec::with_capacity(bootstrap_samples);
    for _ in 0..bootstrap_samples {
        for index in 0..candidate.len() {
            let sampled = (rng.next_u64() as usize) % candidate.len();
            candidate_draw[index] = candidate[sampled];
            baseline_draw[index] = baseline[sampled];
        }
        candidate_draw.sort_by(f64::total_cmp);
        baseline_draw.sort_by(f64::total_cmp);
        differences.push(
            candidate_draw[candidate_draw.len() / 2] / baseline_draw[baseline_draw.len() / 2] - 1.0,
        );
    }
    differences.sort_by(f64::total_cmp);
    (
        differences[percentile_index(bootstrap_samples, 0.025)],
        differences[percentile_index(bootstrap_samples, 0.975)],
    )
}

fn percentile_index(len: usize, percentile: f64) -> usize {
    let last = len.saturating_sub(1);
    ((last as f64) * percentile).round() as usize
}

fn shuffled_order(len: usize, seed: u64) -> Vec<usize> {
    let mut order = (0..len).collect::<Vec<_>>();
    let mut rng = SplitMix64::new(seed ^ ((len as u64) << 32) ^ 0x9e37_79b9_7f4a_7c15);
    for i in (1..order.len()).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        order.swap(i, j);
    }
    order
}

fn round_robin_orders(row_count: usize, round_count: usize, seed_mask: u64) -> Vec<Vec<usize>> {
    (0..round_count)
        .map(|round| shuffled_order(row_count, round as u64 ^ seed_mask))
        .collect()
}

fn orders_cover_each_row_once(orders: &[Vec<usize>], row_count: usize, round_count: usize) -> bool {
    orders.len() == round_count
        && orders.iter().all(|order| {
            if order.len() != row_count {
                return false;
            }
            let mut seen = vec![false; row_count];
            order
                .iter()
                .all(|&index| index < row_count && !std::mem::replace(&mut seen[index], true))
        })
}

fn randomize_binary_field_input(input: &[u8], seed: u64) -> Vec<u8> {
    let mut out = input.to_vec();
    let mut rng = SplitMix64::new(seed ^ 0x0b1f_1a5c_0ded_1ced);
    for chunk in out.chunks_exact_mut(4) {
        let value = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if value <= 1 {
            let bit = (rng.next_u64() & 1) as u32;
            chunk.copy_from_slice(&bit.to_le_bytes());
        }
    }
    out
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

fn select_rows(report: &Value, args: &Args) -> Result<SelectedRows, String> {
    require_report_proof_mode(report, args.proof_mode)?;
    let (source, row_source_resolved) = select_source_rows(report, args.row_source)?;
    let (mut reference_labels, mut required_labels) =
        if row_source_resolved == RowSource::MeasurementShortlist {
            measurement_shortlist_labels(report)?
        } else {
            (Vec::new(), Vec::new())
        };

    let mut rows = Vec::new();
    for row in source {
        let label = row.get("label").and_then(Value::as_str).unwrap_or_default();
        if let Some(labels) = &args.labels {
            if !labels.iter().any(|wanted| wanted == label) {
                continue;
            }
        }
        if let Some(extension) = &args.extension {
            if row.get("extension").and_then(Value::as_str) != Some(extension.as_str()) {
                continue;
            }
        }
        if row.get("accepted_for_ranking").and_then(Value::as_bool) == Some(false) {
            continue;
        }
        if row.get("valid").and_then(Value::as_bool) == Some(false) {
            continue;
        }
        if row.get("setup_config").is_none() {
            continue;
        }
        rows.push(row.clone());
    }
    if let Some(explicit_labels) = &args.labels {
        let selected_labels = rows
            .iter()
            .filter_map(|row| row.get("label").and_then(Value::as_str))
            .collect::<Vec<_>>();
        for label in explicit_labels {
            if !selected_labels.iter().any(|selected| *selected == label) {
                return Err(format!("explicit label was not selected: {label}"));
            }
        }
        if rows.len() > args.max_rows {
            return Err(format!(
                "{} explicitly selected rows exceed --max-rows {}",
                rows.len(),
                args.max_rows
            ));
        }
        reference_labels.retain(|label| {
            selected_labels
                .iter()
                .any(|selected| *selected == label.as_str())
        });
        required_labels.retain(|label| {
            selected_labels
                .iter()
                .any(|selected| *selected == label.as_str())
        });
    } else {
        rows = if row_source_resolved == RowSource::MeasurementShortlist {
            limited_rows_preserving_labels(
                &rows,
                args.max_rows,
                &required_labels,
                args.include_strata,
            )?
        } else if rows.len() <= args.max_rows {
            rows
        } else if args.include_strata {
            stratified_rows(&rows, args.max_rows)
        } else {
            rows.into_iter().take(args.max_rows).collect()
        };
    }
    require_selected_row_proof_modes(&rows, args.proof_mode)?;
    let metadata = RowSelectionMetadata {
        row_source_requested: args.row_source.name(),
        row_source_resolved: row_source_resolved.name(),
        max_rows: args.max_rows,
        include_strata: args.include_strata,
        extension: args.extension.clone(),
        explicit_labels: args.labels.clone(),
        reference_labels,
        required_labels,
        selected_row_count: rows.len(),
    };
    Ok(SelectedRows { rows, metadata })
}

fn select_source_rows(
    report: &Value,
    row_source: RowSource,
) -> Result<(&Vec<Value>, RowSource), String> {
    match row_source {
        RowSource::Auto => {
            for resolved in [
                RowSource::MeasurementShortlist,
                RowSource::MeasurementCandidates,
                RowSource::Scores,
                RowSource::Candidates,
            ] {
                if let Some(rows) = report.get(resolved.name()).and_then(Value::as_array) {
                    return Ok((rows, resolved));
                }
            }
            Err(
                "report must contain measurement_shortlist, measurement_candidates, scores, or candidates array"
                    .to_owned(),
            )
        }
        resolved => report
            .get(resolved.name())
            .and_then(Value::as_array)
            .map(|rows| (rows, resolved))
            .ok_or_else(|| format!("report must contain {} array", resolved.name())),
    }
}

fn require_report_proof_mode(report: &Value, proof_mode: ProofMode) -> Result<(), String> {
    let actual = report
        .get("proof_mode")
        .and_then(Value::as_str)
        .ok_or_else(|| "source report must declare proof_mode as no-zk or full-zk".to_owned())?;
    if actual != proof_mode.name() {
        return Err(format!(
            "source report uses proof_mode={actual}, expected {}",
            proof_mode.name()
        ));
    }
    Ok(())
}

fn require_selected_row_proof_modes(rows: &[Value], proof_mode: ProofMode) -> Result<(), String> {
    for (index, row) in rows.iter().enumerate() {
        if row.get("proof_mode").and_then(Value::as_str) != Some(proof_mode.name()) {
            let label = row
                .get("label")
                .and_then(Value::as_str)
                .unwrap_or("<missing-label>");
            return Err(format!(
                "selected row {index} ({label}) must use proof_mode={}",
                proof_mode.name()
            ));
        }
    }
    Ok(())
}

fn measurement_shortlist_labels(report: &Value) -> Result<(Vec<String>, Vec<String>), String> {
    let Some(meta) = report.get("measurement_shortlist_meta") else {
        return Ok((Vec::new(), Vec::new()));
    };
    let reference_labels = string_array_field(meta, "reference_labels")?;
    let declared_required_labels = string_array_field(meta, "required_labels")?;
    let mut required_labels = reference_labels.clone();
    for label in declared_required_labels {
        if !required_labels.contains(&label) {
            required_labels.push(label);
        }
    }
    Ok((reference_labels, required_labels))
}

fn string_array_field(value: &Value, field: &str) -> Result<Vec<String>, String> {
    let Some(labels) = value.get(field) else {
        return Ok(Vec::new());
    };
    let labels = labels
        .as_array()
        .ok_or_else(|| format!("measurement shortlist {field} must be an array"))?;
    labels
        .iter()
        .map(|label| {
            label
                .as_str()
                .map(ToOwned::to_owned)
                .ok_or_else(|| format!("measurement shortlist {field} must contain strings"))
        })
        .collect()
}

fn stratified_rows(rows: &[Value], max_rows: usize) -> Vec<Value> {
    stratified_indices(rows.len(), max_rows)
        .into_iter()
        .map(|index| rows[index].clone())
        .collect()
}

fn limited_rows_preserving_labels(
    rows: &[Value],
    max_rows: usize,
    required_labels: &[String],
    include_strata: bool,
) -> Result<Vec<Value>, String> {
    if required_labels.is_empty() {
        return Ok(if include_strata {
            stratified_rows(rows, max_rows)
        } else {
            rows.iter().take(max_rows).cloned().collect()
        });
    }

    let mut required_indices = Vec::new();
    for required in required_labels {
        let mut found = false;
        for (index, row) in rows.iter().enumerate() {
            if row.get("label").and_then(Value::as_str) == Some(required.as_str()) {
                found = true;
                if !required_indices.contains(&index) {
                    required_indices.push(index);
                }
            }
        }
        if !found {
            return Err(format!(
                "measurement shortlist is missing required reference label {required}"
            ));
        }
    }
    if required_indices.len() > max_rows {
        return Err(format!(
            "{} required reference rows exceed --max-rows {max_rows}",
            required_indices.len()
        ));
    }

    let available = (0..rows.len())
        .filter(|index| !required_indices.contains(index))
        .collect::<Vec<_>>();
    let remaining_slots = max_rows - required_indices.len();
    let remaining_indices = if include_strata {
        stratified_indices(available.len(), remaining_slots)
    } else {
        (0..available.len().min(remaining_slots)).collect()
    };
    let mut selected_indices = remaining_indices
        .into_iter()
        .map(|index| available[index])
        .collect::<Vec<_>>();
    selected_indices.extend(required_indices);
    selected_indices.sort_unstable();
    Ok(selected_indices
        .into_iter()
        .map(|index| rows[index].clone())
        .collect())
}

fn stratified_indices(len: usize, max_rows: usize) -> Vec<usize> {
    if max_rows == 0 || len == 0 {
        return Vec::new();
    }
    if len <= max_rows {
        return (0..len).collect();
    }
    if max_rows == 1 {
        return vec![0];
    }
    let last = len - 1;
    let mut selected = Vec::with_capacity(max_rows);
    let mut last_index = None;
    for slot in 0..max_rows {
        let index = (slot * last + (max_rows - 1) / 2) / (max_rows - 1);
        if Some(index) != last_index {
            selected.push(index);
            last_index = Some(index);
        }
    }
    selected
}

fn read_json(path: &PathBuf) -> Result<Value, String> {
    let file =
        fs::File::open(path).map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    serde_json::from_reader(file)
        .map_err(|err| format!("failed to parse {}: {err}", path.display()))
}

fn report_workload_identity(report: &Value) -> Result<WorkloadIdentity, String> {
    let value = report
        .get("workload_identity")
        .ok_or_else(|| "source report is missing workload_identity".to_owned())?;
    let identity: WorkloadIdentity = serde_json::from_value(value.clone())
        .map_err(|err| format!("source report has invalid workload_identity: {err}"))?;
    validate_workload_identity(&identity)?;
    Ok(identity)
}

fn validate_workload_identity(identity: &WorkloadIdentity) -> Result<(), String> {
    if identity.label.is_empty() {
        return Err("source report workload_identity label must not be empty".to_owned());
    }
    if identity.constraint_work == 0 {
        return Err("source report workload_identity constraint_work must be positive".to_owned());
    }
    if identity.r1cs_sha256.len() != 64
        || !identity
            .r1cs_sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(
            "source report workload_identity r1cs_sha256 must be 64 lowercase hex characters"
                .to_owned(),
        );
    }
    Ok(())
}

fn require_selected_row_workload_identities(
    rows: &[Value],
    expected: &WorkloadIdentity,
) -> Result<(), String> {
    for (index, row) in rows.iter().enumerate() {
        let label = row
            .get("label")
            .and_then(Value::as_str)
            .unwrap_or("<missing-label>");
        let value = row.get("workload_identity").ok_or_else(|| {
            format!("selected row {index} ({label}) is missing workload_identity")
        })?;
        let identity: WorkloadIdentity = serde_json::from_value(value.clone()).map_err(|err| {
            format!("selected row {index} ({label}) has invalid workload_identity: {err}")
        })?;
        validate_workload_identity(&identity)?;
        if &identity != expected {
            return Err(format!(
                "selected row {index} ({label}) workload_identity differs from the source report"
            ));
        }
    }
    Ok(())
}

fn require_workload_matches_inputs(
    identity: &WorkloadIdentity,
    args: &Args,
    shape: &R1csShape<F>,
) -> Result<(), String> {
    let r1cs_sha256 = sha256_file(&args.r1cs)?;
    require_workload_metadata_matches(
        identity,
        &r1cs_sha256,
        shape.num_cons,
        args.case_label.as_deref(),
        args.linked_run_name.as_deref(),
    )
}

fn require_workload_metadata_matches(
    identity: &WorkloadIdentity,
    r1cs_sha256: &str,
    constraint_work: usize,
    case_label: Option<&str>,
    linked_run_name: Option<&str>,
) -> Result<(), String> {
    if identity.r1cs_sha256 != r1cs_sha256 {
        return Err("source report workload_identity does not match --r1cs".to_owned());
    }
    if identity.constraint_work != constraint_work {
        return Err(format!(
            "source report workload_identity constraint_work={} does not match R1CS constraint count {constraint_work}",
            identity.constraint_work
        ));
    }
    let case_label = case_label
        .ok_or_else(|| "--case-label is required to validate workload_identity".to_owned())?;
    if identity.label != case_label {
        return Err(format!(
            "source report workload_identity label={} does not match --case-label {case_label}",
            identity.label
        ));
    }
    if let Some(linked_run_name) = linked_run_name {
        if identity.label != linked_run_name {
            return Err(format!(
                "source report workload_identity label={} does not match --linked-run-name {linked_run_name}",
                identity.label
            ));
        }
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let file = fs::File::open(path)
        .map_err(|err| format!("failed to open {} for SHA-256: {err}", path.display()))?;
    sha256_reader(file)
        .map_err(|err| format!("failed to read {} for SHA-256: {err}", path.display()))
}

fn sha256_reader(mut reader: impl Read) -> io::Result<String> {
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn require_matching_code_provenance(
    report: &Value,
    current: &BenchmarkProvenance,
) -> Result<(), String> {
    let source = report
        .get("provenance")
        .ok_or_else(|| "source report is missing provenance".to_owned())?;
    let current = serde_json::to_value(current)
        .map_err(|error| format!("failed to serialize current provenance: {error}"))?;
    if source != &current {
        return Err("source report provenance does not match the current run".to_owned());
    }
    Ok(())
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        r1cs: PathBuf::new(),
        wtns: PathBuf::new(),
        linked_witness_library: PathBuf::new(),
        linked_circuit_data: PathBuf::new(),
        linked_input: PathBuf::new(),
        linked_run_name: None,
        report: PathBuf::new(),
        out: PathBuf::new(),
        labels: None,
        case_label: None,
        extension: None,
        repeats: DEFAULT_REPEATS,
        warmups: DEFAULT_WARMUPS,
        max_rows: DEFAULT_MAX_ROWS,
        include_strata: false,
        row_source: RowSource::Auto,
        proof_mode: ProofMode::FullZk,
        randomize_linked_input_bits: false,
    };
    let mut iter = env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--r1cs" => args.r1cs = PathBuf::from(parse_next_string(&mut iter, &arg)?),
            "--wtns" => args.wtns = PathBuf::from(parse_next_string(&mut iter, &arg)?),
            "--linked-witness-library" => {
                args.linked_witness_library = PathBuf::from(parse_next_string(&mut iter, &arg)?)
            }
            "--linked-circuit-data" => {
                args.linked_circuit_data = PathBuf::from(parse_next_string(&mut iter, &arg)?)
            }
            "--linked-input" => {
                args.linked_input = PathBuf::from(parse_next_string(&mut iter, &arg)?)
            }
            "--linked-run-name" => args.linked_run_name = Some(parse_next_string(&mut iter, &arg)?),
            "--report" => args.report = PathBuf::from(parse_next_string(&mut iter, &arg)?),
            "--out" => args.out = PathBuf::from(parse_next_string(&mut iter, &arg)?),
            "--labels" => {
                args.labels = Some(
                    parse_next_string(&mut iter, &arg)?
                        .split(',')
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(ToOwned::to_owned)
                        .collect(),
                )
            }
            "--case-label" => args.case_label = Some(parse_next_string(&mut iter, &arg)?),
            "--extension" => args.extension = Some(parse_next_string(&mut iter, &arg)?),
            "--repeats" => args.repeats = parse_next(&mut iter, &arg)?,
            "--warmups" => args.warmups = parse_next(&mut iter, &arg)?,
            "--max-rows" => args.max_rows = parse_next(&mut iter, &arg)?,
            "--include-strata" => args.include_strata = true,
            "--row-source" => {
                args.row_source = parse_row_source(&parse_next_string(&mut iter, &arg)?)?
            }
            "--proof-mode" => {
                args.proof_mode = parse_proof_mode(&parse_next_string(&mut iter, &arg)?)?
            }
            "--randomize-linked-input-bits" => args.randomize_linked_input_bits = true,
            "--help" | "-h" => {
                usage();
                process::exit(0);
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if args.r1cs.as_os_str().is_empty()
        || args.report.as_os_str().is_empty()
        || args.out.as_os_str().is_empty()
    {
        return Err("--r1cs, --report, and --out are required".to_owned());
    }
    let has_wtns = !args.wtns.as_os_str().is_empty();
    let has_linked = has_linked_witness_args(&args);
    if has_wtns == has_linked {
        return Err(
            "pass exactly one witness source: --wtns or linked witness arguments".to_owned(),
        );
    }
    if has_linked
        && (args.linked_witness_library.as_os_str().is_empty()
            || args.linked_circuit_data.as_os_str().is_empty()
            || args.linked_input.as_os_str().is_empty()
            || args
                .linked_run_name
                .as_ref()
                .map_or(true, |run_name| run_name.is_empty()))
    {
        return Err(
            "--linked-witness-library, --linked-circuit-data, --linked-input, and --linked-run-name are required for linked witness input"
                .to_owned(),
        );
    }
    if args.repeats == 0 {
        return Err("--repeats must be positive".to_owned());
    }
    if args.labels.as_ref().is_some_and(Vec::is_empty) {
        return Err("--labels must not be empty".to_owned());
    }
    if args.max_rows == 0 {
        return Err("--max-rows must be positive".to_owned());
    }
    if args.randomize_linked_input_bits && !has_linked {
        return Err("--randomize-linked-input-bits requires linked witness input".to_owned());
    }
    Ok(args)
}

fn parse_row_source(raw: &str) -> Result<RowSource, String> {
    match raw {
        "auto" => Ok(RowSource::Auto),
        "scores" => Ok(RowSource::Scores),
        "candidates" => Ok(RowSource::Candidates),
        "measurement-shortlist" | "measurement_shortlist" => Ok(RowSource::MeasurementShortlist),
        "measurement-candidates" | "measurement_candidates" => {
            Ok(RowSource::MeasurementCandidates)
        }
        other => Err(format!(
            "--row-source must be auto, scores, candidates, measurement-shortlist, or measurement-candidates; got {other}"
        )),
    }
}

fn parse_proof_mode(raw: &str) -> Result<ProofMode, String> {
    match raw {
        "no-zk" => Ok(ProofMode::NoZk),
        "full-zk" => Ok(ProofMode::FullZk),
        other => Err(format!(
            "--proof-mode must be no-zk or full-zk; got {other}"
        )),
    }
}

fn parse_next(iter: &mut impl Iterator<Item = String>, name: &str) -> Result<usize, String> {
    parse_next_string(iter, name)?
        .parse()
        .map_err(|_| format!("{name} must be a non-negative integer"))
}

fn parse_next_string(
    iter: &mut impl Iterator<Item = String>,
    name: &str,
) -> Result<String, String> {
    iter.next()
        .ok_or_else(|| format!("{name} requires a value"))
}

fn usage() {
    eprintln!(
        "usage: poseidon-schedule-heldout --r1cs circuit.r1cs (--wtns witness.wtns | --linked-witness-library lib.so --linked-circuit-data circuit.dat --linked-input input.bin --linked-run-name symbol_prefix) --report report.json --out heldout.json [--labels label-a,label-b] [--case-label sha256_512b] [--extension octic] [--row-source auto|scores|candidates|measurement-shortlist|measurement-candidates] [--proof-mode no-zk|full-zk] [--max-rows 5] [--include-strata] [--randomize-linked-input-bits] [--repeats 3] [--warmups 1]"
    );
}

fn has_linked_witness_args(args: &Args) -> bool {
    !args.linked_witness_library.as_os_str().is_empty()
        || !args.linked_circuit_data.as_os_str().is_empty()
        || !args.linked_input.as_os_str().is_empty()
        || args.linked_run_name.is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn args(include_strata: bool, max_rows: usize) -> Args {
        Args {
            r1cs: PathBuf::new(),
            wtns: PathBuf::new(),
            linked_witness_library: PathBuf::new(),
            linked_circuit_data: PathBuf::new(),
            linked_input: PathBuf::new(),
            linked_run_name: None,
            report: PathBuf::new(),
            out: PathBuf::new(),
            labels: None,
            case_label: None,
            extension: None,
            repeats: 1,
            warmups: 0,
            max_rows,
            include_strata,
            row_source: RowSource::Scores,
            proof_mode: ProofMode::FullZk,
            randomize_linked_input_bits: false,
        }
    }

    fn report() -> Value {
        json!({
            "proof_mode": "full-zk",
            "scores": (0..7)
                .map(|i| json!({
                    "label": format!("row{i}"),
                    "extension": "quintic",
                    "proof_mode": "full-zk",
                    "valid": true,
                    "accepted_for_ranking": true,
                    "setup_config": {"matrix_closing": "DirectSparse"}
                }))
                .collect::<Vec<_>>()
        })
    }

    fn report_with_shortlist() -> Value {
        json!({
            "proof_mode": "full-zk",
            "measurement_shortlist": [
                {
                    "label": "short0",
                    "extension": "quintic",
                    "proof_mode": "full-zk",
                    "valid": true,
                    "accepted_for_ranking": true,
                    "setup_config": {"matrix_closing": "DirectSparse"}
                },
            ],
            "scores": [
                {
                    "label": "score0",
                    "extension": "quintic",
                    "proof_mode": "full-zk",
                    "valid": true,
                    "accepted_for_ranking": true,
                    "setup_config": {"matrix_closing": "DirectSparse"}
                },
            ]
        })
    }

    fn report_with_measurement_candidates() -> Value {
        json!({
            "proof_mode": "full-zk",
            "measurement_candidates": [
                {
                    "label": "candidate0",
                    "extension": "quintic",
                    "proof_mode": "full-zk",
                    "valid": true,
                    "accepted_for_ranking": true,
                    "setup_config": {"matrix_closing": "DirectSparse"}
                },
            ],
            "scores": [
                {
                    "label": "score0",
                    "extension": "quintic",
                    "proof_mode": "full-zk",
                    "valid": true,
                    "accepted_for_ranking": true,
                    "setup_config": {"matrix_closing": "DirectSparse"}
                },
            ]
        })
    }

    fn workload_identity() -> WorkloadIdentity {
        WorkloadIdentity {
            label: "case".to_owned(),
            r1cs_sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
                .to_owned(),
            constraint_work: 123,
        }
    }

    #[test]
    fn source_report_requires_a_well_formed_workload_identity() {
        let error = report_workload_identity(&report()).unwrap_err();
        assert!(error.contains("source report is missing workload_identity"));

        let mut report = report();
        report["workload_identity"] = json!({
            "label": "case",
            "r1cs_sha256": "ABC",
            "constraint_work": 123,
        });
        let error = report_workload_identity(&report).unwrap_err();
        assert!(error.contains("64 lowercase hex characters"));

        report["workload_identity"] = serde_json::to_value(workload_identity()).unwrap();
        assert_eq!(
            report_workload_identity(&report).unwrap(),
            workload_identity()
        );
    }

    #[test]
    fn every_selected_row_must_repeat_the_report_workload_identity() {
        let expected = workload_identity();
        let exact = json!({
            "label": "exact",
            "workload_identity": expected,
        });
        require_selected_row_workload_identities(&[exact], &expected).unwrap();

        let missing = json!({"label": "missing"});
        let error = require_selected_row_workload_identities(&[missing], &expected).unwrap_err();
        assert!(error.contains("selected row 0 (missing) is missing workload_identity"));

        let foreign = json!({
            "label": "foreign",
            "workload_identity": {
                "label": "other",
                "r1cs_sha256": expected.r1cs_sha256,
                "constraint_work": expected.constraint_work,
            },
        });
        let error = require_selected_row_workload_identities(&[foreign], &expected).unwrap_err();
        assert!(error.contains("workload_identity differs from the source report"));
    }

    #[test]
    fn workload_identity_must_match_r1cs_shape_and_labels() {
        let identity = workload_identity();
        require_workload_metadata_matches(
            &identity,
            &identity.r1cs_sha256,
            identity.constraint_work,
            Some("case"),
            None,
        )
        .unwrap();
        require_workload_metadata_matches(
            &identity,
            &identity.r1cs_sha256,
            identity.constraint_work,
            Some("case"),
            Some("case"),
        )
        .unwrap();

        assert!(require_workload_metadata_matches(
            &identity,
            &"0".repeat(64),
            identity.constraint_work,
            Some("case"),
            None,
        )
        .unwrap_err()
        .contains("does not match --r1cs"));
        assert!(require_workload_metadata_matches(
            &identity,
            &identity.r1cs_sha256,
            identity.constraint_work + 1,
            Some("case"),
            None,
        )
        .unwrap_err()
        .contains("does not match R1CS constraint count"));
        assert!(require_workload_metadata_matches(
            &identity,
            &identity.r1cs_sha256,
            identity.constraint_work,
            None,
            None,
        )
        .unwrap_err()
        .contains("--case-label is required"));
        assert!(require_workload_metadata_matches(
            &identity,
            &identity.r1cs_sha256,
            identity.constraint_work,
            Some("other"),
            None,
        )
        .unwrap_err()
        .contains("does not match --case-label"));
        assert!(require_workload_metadata_matches(
            &identity,
            &identity.r1cs_sha256,
            identity.constraint_work,
            Some("case"),
            Some("other"),
        )
        .unwrap_err()
        .contains("does not match --linked-run-name"));
    }

    #[test]
    fn sha256_reader_hashes_the_exact_r1cs_bytes() {
        assert_eq!(
            sha256_reader(&b"abc"[..]).unwrap(),
            workload_identity().r1cs_sha256
        );
    }

    #[test]
    fn select_rows_defaults_to_top_rows() {
        let selected = select_rows(&report(), &args(false, 3)).unwrap();
        let labels = selected
            .rows
            .iter()
            .map(|row| row["label"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(labels, vec!["row0", "row1", "row2"]);
    }

    #[test]
    fn select_rows_can_sample_strata() {
        let selected = select_rows(&report(), &args(true, 3)).unwrap();
        let labels = selected
            .rows
            .iter()
            .map(|row| row["label"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(labels, vec!["row0", "row3", "row6"]);
    }

    #[test]
    fn stratified_shortlist_preserves_required_reference_labels() {
        let mut report = report();
        report["measurement_shortlist"] = report["scores"].clone();
        report["measurement_shortlist_meta"] = json!({
            "reference_labels": ["row4"]
        });
        let mut args = args(true, 3);
        args.row_source = RowSource::MeasurementShortlist;

        let selected = select_rows(&report, &args).unwrap();
        let labels = selected
            .rows
            .iter()
            .map(|row| row["label"].as_str().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(labels, vec!["row0", "row4", "row6"]);
    }

    #[test]
    fn top_shortlist_preserves_required_reference_labels() {
        let mut report = report();
        report["measurement_shortlist"] = report["scores"].clone();
        report["measurement_shortlist_meta"] = json!({
            "reference_labels": ["row4"]
        });
        let mut args = args(false, 3);
        args.row_source = RowSource::MeasurementShortlist;

        let selected = select_rows(&report, &args).unwrap();
        let labels = selected
            .rows
            .iter()
            .map(|row| row["label"].as_str().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(labels, vec!["row0", "row1", "row4"]);
    }

    #[test]
    fn stratified_shortlist_rejects_more_reference_rows_than_the_cap() {
        let mut report = report();
        report["measurement_shortlist"] = report["scores"].clone();
        report["measurement_shortlist_meta"] = json!({
            "reference_labels": ["row1", "row4"]
        });
        let mut args = args(true, 1);
        args.row_source = RowSource::MeasurementShortlist;

        let error = select_rows(&report, &args).unwrap_err();

        assert!(error.contains("required reference rows exceed --max-rows"));
    }

    #[test]
    fn shortlist_validates_reference_labels_without_truncation() {
        let rows = vec![json!({"label": "row0"}), json!({"label": "row1"})];
        let required = vec!["missing".to_owned()];

        let error = limited_rows_preserving_labels(&rows, 5, &required, false).unwrap_err();

        assert!(error.contains("missing required reference label missing"));
    }

    #[test]
    fn shortlist_preserves_duplicate_rows_for_a_reference_label() {
        let rows = vec![
            json!({"id": "top", "label": "other"}),
            json!({"id": "reference-a", "label": "reference"}),
            json!({"id": "reference-b", "label": "reference"}),
            json!({"id": "tail", "label": "other"}),
        ];
        let required = vec!["reference".to_owned()];

        let selected = limited_rows_preserving_labels(&rows, 3, &required, false).unwrap();
        let ids = selected
            .iter()
            .map(|row| row["id"].as_str().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(ids, vec!["top", "reference-a", "reference-b"]);
    }

    #[test]
    fn shortlist_rejects_a_cap_that_cannot_fit_duplicate_reference_rows() {
        let rows = vec![
            json!({"label": "reference"}),
            json!({"label": "reference"}),
            json!({"label": "other"}),
        ];
        let required = vec!["reference".to_owned()];

        let error = limited_rows_preserving_labels(&rows, 1, &required, false).unwrap_err();

        assert!(error.contains("2 required reference rows exceed --max-rows 1"));
    }

    #[test]
    fn select_rows_auto_prefers_measurement_shortlist() {
        let mut args = args(false, 10);
        args.row_source = RowSource::Auto;
        let selected = select_rows(&report_with_shortlist(), &args).unwrap();
        let labels = selected
            .rows
            .iter()
            .map(|row| row["label"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(labels, vec!["short0"]);
    }

    #[test]
    fn select_rows_can_force_scores_source() {
        let mut args = args(false, 10);
        args.row_source = RowSource::Scores;
        let selected = select_rows(&report_with_shortlist(), &args).unwrap();
        let labels = selected
            .rows
            .iter()
            .map(|row| row["label"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(labels, vec!["score0"]);
    }

    #[test]
    fn select_rows_auto_uses_measurement_candidates() {
        let mut args = args(false, 10);
        args.row_source = RowSource::Auto;
        let selected = select_rows(&report_with_measurement_candidates(), &args).unwrap();
        let labels = selected
            .rows
            .iter()
            .map(|row| row["label"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(labels, vec!["candidate0"]);
    }

    #[test]
    fn select_rows_can_force_measurement_candidates() {
        let mut args = args(false, 10);
        args.row_source = RowSource::MeasurementCandidates;
        let selected = select_rows(&report_with_measurement_candidates(), &args).unwrap();
        let labels = selected
            .rows
            .iter()
            .map(|row| row["label"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(labels, vec!["candidate0"]);
    }

    #[test]
    fn selection_metadata_records_requested_and_resolved_sources_and_knobs() {
        let mut report = report_with_shortlist();
        report["measurement_shortlist_meta"] = json!({
            "reference_labels": ["short0"]
        });
        let mut args = args(true, 7);
        args.row_source = RowSource::Auto;
        args.extension = Some("quintic".to_owned());
        args.labels = Some(vec!["short0".to_owned()]);

        let selected = select_rows(&report, &args).unwrap();
        let metadata = serde_json::to_value(&selected.metadata).unwrap();

        assert_eq!(
            metadata,
            json!({
                "row_source_requested": "auto",
                "row_source_resolved": "measurement_shortlist",
                "max_rows": 7,
                "include_strata": true,
                "extension": "quintic",
                "explicit_labels": ["short0"],
                "reference_labels": ["short0"],
                "required_labels": ["short0"],
                "selected_row_count": 1
            })
        );
    }

    #[test]
    fn shortlist_preserves_coverage_labels_separately_from_references() {
        let mut report = report();
        report["measurement_shortlist"] = report["scores"].clone();
        report["measurement_shortlist_meta"] = json!({
            "reference_labels": ["row1"],
            "required_labels": ["row1", "row5"]
        });
        let mut args = args(false, 3);
        args.row_source = RowSource::MeasurementShortlist;

        let selected = select_rows(&report, &args).unwrap();
        let labels = selected
            .rows
            .iter()
            .map(|row| row["label"].as_str().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(labels, vec!["row0", "row1", "row5"]);
        assert_eq!(selected.metadata.reference_labels, vec!["row1"]);
        assert_eq!(selected.metadata.required_labels, vec!["row1", "row5"]);
    }

    #[test]
    fn shortlist_unions_references_into_declared_required_labels() {
        let mut report = report();
        report["measurement_shortlist"] = report["scores"].clone();
        report["measurement_shortlist_meta"] = json!({
            "reference_labels": ["row5"],
            "required_labels": ["row1"]
        });
        let mut args = args(false, 3);
        args.row_source = RowSource::MeasurementShortlist;

        let selected = select_rows(&report, &args).unwrap();
        let labels = selected
            .rows
            .iter()
            .map(|row| row["label"].as_str().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(labels, vec!["row0", "row1", "row5"]);
        assert_eq!(selected.metadata.required_labels, vec!["row5", "row1"]);
    }

    #[test]
    fn explicit_labels_do_not_claim_unselected_required_rows() {
        let mut report = report();
        report["measurement_shortlist"] = report["scores"].clone();
        report["measurement_shortlist_meta"] = json!({
            "reference_labels": ["row4"],
            "required_labels": ["row4", "row5"]
        });
        let mut args = args(false, 1);
        args.row_source = RowSource::MeasurementShortlist;
        args.labels = Some(vec!["row0".to_owned()]);

        let selected = select_rows(&report, &args).unwrap();

        assert_eq!(selected.rows.len(), 1);
        assert!(selected.metadata.reference_labels.is_empty());
        assert!(selected.metadata.required_labels.is_empty());
    }

    #[test]
    fn explicit_labels_cannot_exceed_the_row_cap() {
        let mut args = args(false, 1);
        args.labels = Some(vec!["row0".to_owned(), "row1".to_owned()]);

        let error = select_rows(&report(), &args).unwrap_err();

        assert!(error.contains("2 explicitly selected rows exceed --max-rows 1"));
    }

    #[test]
    fn every_explicit_label_must_be_selected() {
        let mut args = args(false, 2);
        args.labels = Some(vec!["row0".to_owned(), "missing".to_owned()]);

        let error = select_rows(&report(), &args).unwrap_err();

        assert!(error.contains("explicit label was not selected: missing"));
    }

    #[test]
    fn report_proof_mode_must_match_the_cli_mode() {
        let mut report = report();
        report["proof_mode"] = Value::from("no-zk");

        let error = select_rows(&report, &args(false, 1)).unwrap_err();

        assert!(error.contains("source report uses proof_mode=no-zk, expected full-zk"));
    }

    #[test]
    fn every_selected_row_must_match_the_report_proof_mode() {
        let mut report = report();
        report["scores"][0]["proof_mode"] = Value::from("no-zk");

        let error = select_rows(&report, &args(false, 1)).unwrap_err();

        assert!(error.contains("selected row 0 (row0) must use proof_mode=full-zk"));
    }

    #[test]
    fn parse_proof_mode_accepts_only_current_labels() {
        assert_eq!(parse_proof_mode("no-zk").unwrap(), ProofMode::NoZk);
        assert_eq!(parse_proof_mode("full-zk").unwrap(), ProofMode::FullZk);
        for legacy in ["hiding", "zk", "plain", "non-zk", "legacy"] {
            assert!(parse_proof_mode(legacy).is_err());
        }
    }

    #[test]
    fn round_robin_scheduler_visits_every_row_once_per_repeat() {
        let orders = round_robin_orders(4, 5, 0);

        assert!(orders_cover_each_row_once(&orders, 4, 5));
        for order in orders {
            let mut sorted = order;
            sorted.sort_unstable();
            assert_eq!(sorted, vec![0, 1, 2, 3]);
        }
    }

    #[test]
    fn incomplete_order_does_not_claim_interleaving() {
        let orders = vec![vec![0, 1, 2], vec![0, 2]];

        assert!(!orders_cover_each_row_once(&orders, 3, 2));
    }

    #[test]
    fn randomize_binary_field_input_only_changes_boolean_limbs() {
        let mut input = Vec::new();
        input.extend_from_slice(&0u32.to_le_bytes());
        input.extend_from_slice(&1u32.to_le_bytes());
        input.extend_from_slice(&7u32.to_le_bytes());
        let randomized = randomize_binary_field_input(&input, 12);
        assert!(matches!(
            u32::from_le_bytes(randomized[0..4].try_into().unwrap()),
            0 | 1
        ));
        assert!(matches!(
            u32::from_le_bytes(randomized[4..8].try_into().unwrap()),
            0 | 1
        ));
        assert_eq!(u32::from_le_bytes(randomized[8..12].try_into().unwrap()), 7);
    }

    #[test]
    fn paired_bootstrap_preserves_matched_speed_difference() {
        let baseline = [1.00, 1.10, 0.90, 1.05, 0.95];
        let candidate = baseline.map(|sample| sample * 1.02);
        let ci = bootstrap_paired_relative_median_ci(&candidate, &baseline, 2_000, 7);

        assert!(ci.0 > 0.019);
        assert!(ci.1 < 0.021);
    }

    #[test]
    fn sample_summary_records_exact_proof_size_statistics() {
        let measurement = summarize_samples(&[3.0, 1.0, 2.0], &[0.3, 0.1, 0.2], &[120, 100, 110]);

        assert_eq!(measurement.median_seconds, 2.0);
        assert_eq!(measurement.verifier_median_seconds, 0.2);
        assert_eq!(measurement.proof_size_min_bytes, 100);
        assert_eq!(measurement.proof_size_median_bytes, 110);
        assert_eq!(measurement.proof_size_max_bytes, 120);
    }

    #[test]
    fn paired_annotations_identify_the_fastest_row() {
        let mut rows = vec![
            json!({
                "heldout_prove_seconds": 1.0,
                "heldout_samples_seconds": [1.0, 1.1, 0.9]
            }),
            json!({
                "heldout_prove_seconds": 1.02,
                "heldout_samples_seconds": [1.02, 1.122, 0.918]
            }),
        ];

        annotate_paired_comparisons(&mut rows).unwrap();

        assert_eq!(rows[1]["heldout_fastest_row_index"], 0);
        assert!(
            rows[1]["heldout_relative_median_difference"]
                .as_f64()
                .unwrap()
                > 0.019
        );
        assert!(
            rows[1]["heldout_paired_relative_median_ci"][0]
                .as_f64()
                .unwrap()
                > 0.0
        );
    }
}
