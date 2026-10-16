use std::{
    env,
    fs::File,
    hint::black_box,
    path::PathBuf,
    process,
    time::{Duration, Instant},
};

use p3_challenger::{CanObserve, GrindingChallenger};
use p3_commit::Mmcs;
use p3_dft::{Radix2DFTSmallBatch, TwoAdicSubgroupDft};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::current_num_threads;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use serde::Serialize;
use spartan_whir::{engine::F, OcticBinExtension, QuarticBinExtension, QuinticExtension};

#[cfg(feature = "poseidon1")]
use spartan_whir::{
    poseidon1_challenger as poseidon_challenger,
    poseidon1_merkle_compress as poseidon_merkle_compress,
    poseidon1_merkle_hash as poseidon_merkle_hash, Poseidon1FieldHash as PoseidonFieldHash,
    Poseidon1NodeCompress as PoseidonNodeCompress,
};
#[cfg(not(feature = "poseidon1"))]
use spartan_whir::{
    poseidon_challenger, poseidon_merkle_compress, poseidon_merkle_hash, PoseidonFieldHash,
    PoseidonNodeCompress,
};

mod poseidon_schedule_support;
use poseidon_schedule_support::{
    collect as collect_provenance, enabled_features, BenchmarkProvenance,
};

const DEFAULT_MIN_LOG_SIZE: usize = 14;
const DEFAULT_MAX_LOG_SIZE: usize = 18;
const DEFAULT_STEP: usize = 2;
const DEFAULT_REPEATS: usize = 5;
const DEFAULT_POW_ATTEMPTS_LOG2: usize = 18;
const DEFAULT_ROW_OPENINGS: usize = 256;
const DEFAULT_ROW_WIDTH: usize = 16;
const DEFAULT_VALIDATION_TOLERANCE: f64 = 0.20;

type PoseidonMmcs = MerkleTreeMmcs<
    <F as Field>::Packing,
    <F as Field>::Packing,
    PoseidonFieldHash,
    PoseidonNodeCompress,
    2,
    8,
>;

#[derive(Debug, Clone)]
struct Args {
    min_log_size: usize,
    max_log_size: usize,
    step: usize,
    repeats: usize,
    pow_attempts_log2: usize,
    row_openings: usize,
    row_width: usize,
    out: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct Calibration {
    schema_version: u32,
    provenance: BenchmarkProvenance,
    measurement_kind: &'static str,
    units: &'static str,
    build_profile: String,
    target_cpu_native: bool,
    rayon_threads: usize,
    features: String,
    hash_profile: &'static str,
    args: CalibrationArgs,
    coefficients: Coefficients,
    verifier_coefficients: VerifierCoefficients,
    measurements: Measurements,
    validation: Validation,
}

#[derive(Debug, Serialize)]
struct CalibrationArgs {
    min_log_size: usize,
    max_log_size: usize,
    step: usize,
    repeats: usize,
    pow_attempts_log2: usize,
    row_openings: usize,
    row_width: usize,
}

#[derive(Debug, Serialize)]
struct Coefficients {
    fixed_overhead: f64,
    spartan: f64,
    dft: f64,
    merkle: f64,
    merkle_path: f64,
    row_opening: f64,
    sumcheck: SumcheckCoefficients,
    pow: f64,
}

#[derive(Debug, Serialize)]
struct Measurements {
    dft: Vec<PointMeasurement>,
    merkle: Vec<PointMeasurement>,
    merkle_path: Vec<PointMeasurement>,
    row_opening: Vec<PointMeasurement>,
    sumcheck: SumcheckMeasurements,
    pow: PointMeasurement,
    verifier_merkle_hash: PointMeasurement,
    verifier_leaf_hash: Vec<PointMeasurement>,
}

#[derive(Debug, Serialize)]
struct VerifierCoefficients {
    fixed_overhead: f64,
    merkle_hash: f64,
    leaf_field_element: f64,
    row_field_element: f64,
    extension_operation: SumcheckCoefficients,
    pow_check: f64,
}

#[derive(Debug, Serialize)]
struct SumcheckCoefficients {
    quartic: f64,
    quintic: f64,
    octic: f64,
}

#[derive(Debug, Serialize)]
struct SumcheckMeasurements {
    quartic: Vec<PointMeasurement>,
    quintic: Vec<PointMeasurement>,
    octic: Vec<PointMeasurement>,
}

#[derive(Debug, Serialize)]
struct PointMeasurement {
    label: String,
    work_units: u128,
    median_seconds: f64,
    seconds_per_unit: f64,
}

#[derive(Debug, Serialize)]
struct Validation {
    max_relative_error: f64,
    heldout: Vec<serde_json::Value>,
    note: &'static str,
}

fn main() {
    let args = parse_args().unwrap_or_else(|error| {
        eprintln!("{error}");
        usage();
        process::exit(2);
    });

    let dft = measure_size_points(&args, "dft", measure_dft);
    let merkle = measure_size_points(&args, "merkle", |log_size, repeats| {
        measure_merkle(log_size, args.row_width, repeats)
    });
    let merkle_path = measure_size_points(&args, "merkle_path", |log_size, repeats| {
        measure_merkle_path(log_size, args.row_openings, repeats)
    });
    let row_opening = measure_size_points(&args, "row_opening", |log_size, repeats| {
        measure_row_payload(log_size, args.row_width, args.row_openings, repeats)
    });
    let sumcheck = SumcheckMeasurements {
        quartic: measure_size_points(
            &args,
            "sumcheck_quartic",
            measure_sumcheck_kernel::<QuarticBinExtension>,
        ),
        quintic: measure_size_points(
            &args,
            "sumcheck_quintic",
            measure_sumcheck_kernel::<QuinticExtension>,
        ),
        octic: measure_size_points(
            &args,
            "sumcheck_octic",
            measure_sumcheck_kernel::<OcticBinExtension>,
        ),
    };
    let pow = measure_pow(args.pow_attempts_log2, args.repeats);
    let verifier_merkle_hash = measure_verifier_merkle_hash(
        args.row_openings.saturating_mul(args.max_log_size),
        args.repeats,
    );
    let verifier_leaf_hash = [8usize, 16, 32, 64, 128]
        .into_iter()
        .map(|width| measure_verifier_leaf_hash(width, args.row_openings, args.repeats))
        .collect::<Vec<_>>();

    let coefficients = Coefficients {
        fixed_overhead: 0.0,
        spartan: 0.0,
        dft: median_seconds_per_unit(&dft),
        merkle: median_seconds_per_unit(&merkle),
        merkle_path: median_seconds_per_unit(&merkle_path),
        row_opening: median_seconds_per_unit(&row_opening),
        sumcheck: SumcheckCoefficients {
            quartic: median_seconds_per_unit(&sumcheck.quartic),
            quintic: median_seconds_per_unit(&sumcheck.quintic),
            octic: median_seconds_per_unit(&sumcheck.octic),
        },
        pow: pow.seconds_per_unit,
    };
    let verifier_coefficients = VerifierCoefficients {
        fixed_overhead: 0.0,
        merkle_hash: verifier_merkle_hash.seconds_per_unit,
        leaf_field_element: median_seconds_per_unit(&verifier_leaf_hash),
        row_field_element: median_seconds_per_unit(&row_opening),
        extension_operation: SumcheckCoefficients {
            quartic: median_seconds_per_unit(&sumcheck.quartic),
            quintic: median_seconds_per_unit(&sumcheck.quintic),
            octic: median_seconds_per_unit(&sumcheck.octic),
        },
        pow_check: pow.seconds_per_unit,
    };

    let calibration = Calibration {
        schema_version: 3,
        provenance: collect_provenance(enabled_features()).expect("collect benchmark provenance"),
        measurement_kind: "poseidon_schedule_component_calibration",
        units: "seconds",
        build_profile: if cfg!(debug_assertions) {
            "debug".to_owned()
        } else {
            "release".to_owned()
        },
        target_cpu_native: env::var("RUSTFLAGS")
            .map(|flags| flags.contains("target-cpu=native"))
            .unwrap_or(false),
        rayon_threads: current_num_threads(),
        features: enabled_features().to_owned(),
        hash_profile: if cfg!(feature = "poseidon1") {
            "poseidon1"
        } else {
            "poseidon2"
        },
        args: CalibrationArgs {
            min_log_size: args.min_log_size,
            max_log_size: args.max_log_size,
            step: args.step,
            repeats: args.repeats,
            pow_attempts_log2: args.pow_attempts_log2,
            row_openings: args.row_openings,
            row_width: args.row_width,
        },
        coefficients,
        verifier_coefficients,
        measurements: Measurements {
            dft,
            merkle,
            merkle_path,
            row_opening,
            sumcheck,
            pow,
            verifier_merkle_hash,
            verifier_leaf_hash,
        },
        validation: Validation {
            max_relative_error: DEFAULT_VALIDATION_TOLERANCE,
            heldout: Vec::new(),
            note: "Full-proof heldout rows are intentionally supplied separately; scorer recommendations remain untrusted until they are present.",
        },
    };

    match &args.out {
        Some(path) => {
            let file = File::create(path).expect("create calibration output");
            serde_json::to_writer_pretty(file, &calibration).expect("write calibration JSON");
        }
        None => {
            serde_json::to_writer_pretty(std::io::stdout(), &calibration)
                .expect("write calibration JSON");
            println!();
        }
    }
}

fn measure_size_points(
    args: &Args,
    label_prefix: &str,
    mut measure: impl FnMut(usize, usize) -> PointMeasurement,
) -> Vec<PointMeasurement> {
    let mut out = Vec::new();
    let mut log_size = args.min_log_size;
    while log_size <= args.max_log_size {
        let mut point = measure(log_size, args.repeats);
        point.label = format!("{label_prefix}_log{log_size}");
        out.push(point);
        log_size = log_size.saturating_add(args.step);
        if args.step == 0 {
            break;
        }
    }
    out
}

fn measure_dft(log_size: usize, repeats: usize) -> PointMeasurement {
    let len = 1usize << log_size;
    let dft = Radix2DFTSmallBatch::default();
    let mut timings = Vec::with_capacity(repeats);
    for repeat in 0..repeats {
        let input = field_vec(len, repeat as u32 + 1);
        let start = Instant::now();
        let output = dft.dft_batch(RowMajorMatrix::new_col(input));
        black_box(output);
        timings.push(start.elapsed());
    }
    point("", len as u128, median(timings))
}

fn measure_merkle(log_size: usize, width: usize, repeats: usize) -> PointMeasurement {
    let height = 1usize << log_size;
    let mmcs = PoseidonMmcs::new(poseidon_merkle_hash(), poseidon_merkle_compress(), 0);
    let mut timings = Vec::with_capacity(repeats);
    for repeat in 0..repeats {
        let input = RowMajorMatrix::new(field_vec(height * width, repeat as u32 + 11), width);
        let start = Instant::now();
        let committed = mmcs.commit_matrix(input);
        black_box(committed);
        timings.push(start.elapsed());
    }
    point("", (height * width) as u128, median(timings))
}

fn measure_merkle_path(log_size: usize, openings: usize, repeats: usize) -> PointMeasurement {
    let height = 1usize << log_size;
    let mmcs = PoseidonMmcs::new(poseidon_merkle_hash(), poseidon_merkle_compress(), 0);
    let input = RowMajorMatrix::new(field_vec(height, 29), 1);
    let (_commitment, prover_data) = mmcs.commit_matrix(input);
    let mut timings = Vec::with_capacity(repeats);
    for repeat in 0..repeats {
        let start = Instant::now();
        for i in 0..openings {
            let index = (i * 7919 + repeat * 104729) & (height - 1);
            let opening = mmcs.open_batch(index, &prover_data);
            black_box(opening);
        }
        timings.push(start.elapsed());
    }
    point("", (openings * log_size) as u128, median(timings))
}

fn measure_row_payload(
    log_size: usize,
    width: usize,
    openings: usize,
    repeats: usize,
) -> PointMeasurement {
    let height = 1usize << log_size;
    let input = field_vec(height * width, 29);
    let mut timings = Vec::with_capacity(repeats);
    for repeat in 0..repeats {
        let mut acc = F::ZERO;
        let start = Instant::now();
        for i in 0..openings {
            let index = (i * 7919 + repeat * 104729) & (height - 1);
            let row = &input[index * width..(index + 1) * width];
            for value in row {
                acc += *value;
            }
        }
        let _ = black_box(acc);
        timings.push(start.elapsed());
    }
    point("", (openings * width) as u128, median(timings))
}

fn measure_sumcheck_kernel<Ext>(log_size: usize, repeats: usize) -> PointMeasurement
where
    Ext: PrimeCharacteristicRing + Copy,
{
    let len = 1usize << log_size;
    let rounds = log_size.max(1);
    let work_units = (rounds as u128) * (len as u128);
    let mut timings = Vec::with_capacity(repeats);
    for repeat in 0..repeats {
        let mut acc = Ext::from_u32(repeat as u32 + 1);
        let a = Ext::from_u32(3);
        let b = Ext::from_u32(5);
        let start = Instant::now();
        for i in 0..(rounds * len) {
            let tweak = Ext::from_u32((i as u32).wrapping_mul(17).wrapping_add(7));
            acc += (a + tweak) * (b - tweak);
        }
        let _ = black_box(acc);
        timings.push(start.elapsed());
    }
    point("", work_units, median(timings))
}

fn measure_pow(pow_attempts_log2: usize, repeats: usize) -> PointMeasurement {
    let attempts = 1usize << pow_attempts_log2;
    let mut timings = Vec::with_capacity(repeats);
    for repeat in 0..repeats {
        let mut base = poseidon_challenger();
        base.observe(F::from_u32(0x5eed_u32.wrapping_add(repeat as u32)));
        let start = Instant::now();
        let mut hits = 0usize;
        for i in 0..attempts {
            let mut challenger = base.clone();
            if challenger.check_witness(8, F::from_u32(i as u32)) {
                hits += 1;
            }
        }
        black_box(hits);
        timings.push(start.elapsed());
    }
    point("pow_check_witness", attempts as u128, median(timings))
}

fn measure_verifier_merkle_hash(hashes: usize, repeats: usize) -> PointMeasurement {
    let compressor = poseidon_merkle_compress();
    let mut timings = Vec::with_capacity(repeats);
    for repeat in 0..repeats {
        let mut left = [F::ZERO; 8];
        let mut right = [F::ZERO; 8];
        left[0] = F::from_u32(repeat as u32 + 1);
        right[0] = F::from_u32(repeat as u32 + 17);
        let start = Instant::now();
        for i in 0..hashes {
            let digest = compressor.compress([left, right]);
            left = right;
            right = digest;
            right[0] += F::from_u32(i as u32 + 1);
        }
        let _ = black_box((left, right));
        timings.push(start.elapsed());
    }
    point("verifier_merkle_hash", hashes as u128, median(timings))
}

fn measure_verifier_leaf_hash(width: usize, hashes: usize, repeats: usize) -> PointMeasurement {
    let hasher = poseidon_merkle_hash();
    let mut timings = Vec::with_capacity(repeats);
    for repeat in 0..repeats {
        let mut row = field_vec(width, repeat as u32 + 41);
        let start = Instant::now();
        for i in 0..hashes {
            let digest = hasher.hash_slice(&row);
            row[i % width] += digest[i % digest.len()];
            let _ = black_box(digest);
        }
        let _ = black_box(row);
        timings.push(start.elapsed());
    }
    let mut measurement = point("", (width * hashes) as u128, median(timings));
    measurement.label = format!("verifier_leaf_hash_width_{width}");
    measurement
}

fn field_vec(len: usize, seed: u32) -> Vec<F> {
    (0..len)
        .map(|i| F::from_u32((i as u32).wrapping_mul(31).wrapping_add(seed)))
        .collect()
}

fn point(label: impl Into<String>, work_units: u128, duration: Duration) -> PointMeasurement {
    let median_seconds = duration.as_secs_f64();
    PointMeasurement {
        label: label.into(),
        work_units,
        median_seconds,
        seconds_per_unit: median_seconds / work_units as f64,
    }
}

fn median_seconds_per_unit(points: &[PointMeasurement]) -> f64 {
    let mut values = points
        .iter()
        .map(|point| point.seconds_per_unit)
        .collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn median(mut timings: Vec<Duration>) -> Duration {
    timings.sort();
    timings[timings.len() / 2]
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        min_log_size: DEFAULT_MIN_LOG_SIZE,
        max_log_size: DEFAULT_MAX_LOG_SIZE,
        step: DEFAULT_STEP,
        repeats: DEFAULT_REPEATS,
        pow_attempts_log2: DEFAULT_POW_ATTEMPTS_LOG2,
        row_openings: DEFAULT_ROW_OPENINGS,
        row_width: DEFAULT_ROW_WIDTH,
        out: None,
    };
    let mut iter = env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--min-log-size" => args.min_log_size = parse_next(&mut iter, &arg)?,
            "--max-log-size" => args.max_log_size = parse_next(&mut iter, &arg)?,
            "--step" => args.step = parse_next(&mut iter, &arg)?,
            "--repeats" => args.repeats = parse_next(&mut iter, &arg)?,
            "--pow-attempts-log2" => args.pow_attempts_log2 = parse_next(&mut iter, &arg)?,
            "--row-openings" => args.row_openings = parse_next(&mut iter, &arg)?,
            "--row-width" => args.row_width = parse_next(&mut iter, &arg)?,
            "--out" => args.out = Some(PathBuf::from(parse_next_string(&mut iter, &arg)?)),
            "--help" | "-h" => {
                usage();
                process::exit(0);
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if args.step == 0 {
        return Err("--step must be positive".to_owned());
    }
    if args.repeats == 0 {
        return Err("--repeats must be positive".to_owned());
    }
    if args.max_log_size < args.min_log_size {
        return Err("--max-log-size must be >= --min-log-size".to_owned());
    }
    if args.pow_attempts_log2 >= usize::BITS as usize {
        return Err("--pow-attempts-log2 is too large".to_owned());
    }
    if args.row_openings == 0 || args.row_width == 0 {
        return Err("--row-openings and --row-width must be positive".to_owned());
    }
    Ok(args)
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
        "usage: poseidon-schedule-calibration [--min-log-size 14] [--max-log-size 18] [--step 2] [--repeats 5] [--pow-attempts-log2 18] [--out calibration.json]"
    );
}
