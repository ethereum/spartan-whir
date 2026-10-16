use std::{env, panic, process};

use num_bigint::BigUint;
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{DuplexChallenger, FieldChallenger, GrindingChallenger};
use p3_field::{extension::BinomialExtensionField, ExtensionField, Field, TwoAdicField};
use p3_whir::parameters::{
    FoldingFactor, ProtocolParameters, SecurityAssumption as P3SecurityAssumption,
    WhirConfig as P3WhirConfig,
};
use serde::Serialize;
use spartan_whir::plonky3_whir_pcs::hiding_terminal_budget;
use spartan_whir::{
    engine::F, format_whir_params_label, MatrixClosingMode, OcticBinExtension,
    PoseidonZkSetupConfig, QuarticBinExtension, SecurityConfig, SoundnessAssumption,
    SpartanSnarkConfig, WhirFoldingSchedule, WhirParams, FINAL_SUMCHECK_MAX_VARIABLES,
    MAX_SECURITY_BITS,
};

#[cfg(feature = "poseidon1")]
use spartan_whir::Poseidon1Challenger as PoseidonChallenger;
#[cfg(not(feature = "poseidon1"))]
use spartan_whir::PoseidonChallenger;

mod poseidon_schedule_support;
use poseidon_schedule_support::{
    collect as collect_provenance, enabled_features, BenchmarkProvenance,
};

const DEFAULT_SECURITY_BITS: usize = 116;
const DEFAULT_K_MAX: usize = 8;
const DEFAULT_LIR_MAX: usize = 8;
const DEFAULT_MAX_POW_BITS: usize = 22;
const DEFAULT_BEAM_WIDTH: usize = 64;
const FULL_ZK_RELATION_SECURITY_SLACK_BITS: usize = 2;
const FIELD_BYTES: u128 = 4;
const POSEIDON_DIGEST_BYTES: u128 = 32;

type KoalaBearQuinticExtension = spartan_whir::QuinticExtension;
type BabyBearQuarticExtension = BinomialExtensionField<BabyBear, 4>;
type BabyBearQuinticExtension = BinomialExtensionField<BabyBear, 5>;
type BabyBearOcticExtension = BinomialExtensionField<BabyBear, 8>;
type BabyBearPoseidonChallenger = DuplexChallenger<BabyBear, Poseidon2BabyBear<16>, 16, 8>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FieldProfile {
    KoalaBear,
    BabyBear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExtensionFilter {
    All,
    Quartic,
    Quintic,
    Octic,
}

impl ExtensionFilter {
    const fn label(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Quartic => "quartic",
            Self::Quintic => "quintic",
            Self::Octic => "octic",
        }
    }

    fn includes(self, extension: &str) -> bool {
        matches!(self, Self::All)
            || matches!(
                (self, extension),
                (Self::Quartic, "quartic") | (Self::Quintic, "quintic") | (Self::Octic, "octic")
            )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProofMode {
    NoZk,
    FullZk,
}

impl ProofMode {
    const fn label(self) -> &'static str {
        match self {
            Self::NoZk => "no-zk",
            Self::FullZk => "full-zk",
        }
    }

    const fn is_full_zk(self) -> bool {
        matches!(self, Self::FullZk)
    }
}

impl FieldProfile {
    const fn label(self) -> &'static str {
        match self {
            Self::KoalaBear => "koalabear",
            Self::BabyBear => "babybear",
        }
    }
}

#[derive(Debug, Clone)]
struct Args {
    field: FieldProfile,
    extension_filter: ExtensionFilter,
    num_variables: usize,
    num_outer_rounds: usize,
    security_bits: usize,
    merkle_security_bits: usize,
    component_security_bits: Option<usize>,
    component_merkle_security_bits: Option<usize>,
    k_max: usize,
    starting_log_inv_rate_max: usize,
    round_log_inv_rate_offset_max: usize,
    max_pow_bits: usize,
    final_sumcheck_max_variables: usize,
    beam_width: usize,
    include_invalid: bool,
    proof_mode: ProofMode,
    zk_ell: usize,
    zk_mask_log_inv_rate: usize,
}

#[derive(Debug, Serialize)]
struct CandidateDump {
    schema_version: u32,
    provenance: BenchmarkProvenance,
    matrix_closing: MatrixClosingMode,
    base_field: &'static str,
    extension_filter: &'static str,
    num_variables: usize,
    num_outer_rounds: usize,
    target_security_bits: usize,
    target_merkle_security_bits: usize,
    component_security_override_bits: Option<usize>,
    component_merkle_security_override_bits: Option<usize>,
    soundness: SoundnessAssumption,
    max_pow_bits: usize,
    round_log_inv_rate_offset_max: usize,
    proof_mode: &'static str,
    hash_profile: &'static str,
    zk_ell: Option<usize>,
    zk_mask_log_inv_rate: Option<usize>,
    candidates: Vec<CandidateRow>,
}

#[derive(Debug, Serialize)]
struct CandidateRow {
    label: String,
    proof_mode: &'static str,
    base_field: &'static str,
    base_two_adicity: usize,
    extension: &'static str,
    extension_degree: usize,
    extension_two_adicity: usize,
    field_bits: usize,
    round_log_inv_rate_offset: usize,
    valid: bool,
    rejection_reason: Option<String>,
    security_bits_achieved: Option<f64>,
    whir_component_security_bits: Option<usize>,
    merkle_component_security_bits: Option<usize>,
    max_derived_pow_bits: Option<usize>,
    pow_work_units: u128,
    dft_work: u128,
    merkle_work: u128,
    merkle_path_work: u128,
    row_work: u128,
    sumcheck_work: u128,
    verifier_merkle_hashes: u128,
    verifier_leaf_field_elements: u128,
    verifier_row_field_elements: u128,
    verifier_extension_operations: u128,
    verifier_pow_checks: u128,
    proof_size_bytes_estimate: u128,
    zk_dft_work: Option<u128>,
    zk_merkle_work: Option<u128>,
    zk_merkle_path_work: Option<u128>,
    zk_row_work: Option<u128>,
    zk_sumcheck_work: Option<u128>,
    zk_verifier_merkle_hashes: Option<u128>,
    zk_verifier_leaf_field_elements: Option<u128>,
    zk_verifier_row_field_elements: Option<u128>,
    zk_verifier_extension_operations: Option<u128>,
    zk_verifier_pow_checks: Option<u128>,
    zk_proof_size_bytes_estimate: Option<u128>,
    zk_mask_queries: Option<usize>,
    zk_application_mask_domain: Option<usize>,
    zk_application_mask_width: Option<usize>,
    zk_ell: Option<usize>,
    zk_mask_log_inv_rate: Option<usize>,
    commitment_ood_samples: Option<usize>,
    starting_folding_pow_bits: Option<usize>,
    final_queries: Option<usize>,
    final_pow_bits: Option<usize>,
    final_sumcheck_rounds: Option<usize>,
    final_folding_pow_bits: Option<usize>,
    rounds: Vec<RoundRow>,
    whir_params: WhirParams,
    setup_config: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct RoundRow {
    round_index: usize,
    num_variables: usize,
    folding_factor: usize,
    log_inv_rate: usize,
    domain_size: usize,
    num_queries: usize,
    ood_samples: usize,
    pow_bits: usize,
    folding_pow_bits: usize,
}

fn main() {
    let args = parse_args().unwrap_or_else(|error| {
        eprintln!("{error}");
        usage();
        process::exit(2);
    });

    let provenance = collect_provenance(enabled_features()).unwrap_or_else(|error| {
        eprintln!("failed to collect benchmark provenance: {error}");
        process::exit(1);
    });
    let mut candidates = Vec::new();
    for schedule in schedules(&args) {
        for pow_bits in 0..=args.max_pow_bits {
            for starting_log_inv_rate in 1..=args.starting_log_inv_rate_max {
                let first = schedule.first_round();
                for rsv in 1..=first {
                    let derived_round_log_inv_rates = derived_round_log_inv_rates(
                        args.num_variables,
                        &schedule,
                        starting_log_inv_rate,
                        rsv,
                    );
                    let backend_derived = WhirParams {
                        pow_bits: pow_bits as u32,
                        folding_factor: first,
                        starting_log_inv_rate,
                        rs_domain_initial_reduction_factor: rsv,
                        folding_schedule: Some(schedule.clone()),
                        round_log_inv_rates: Vec::new(),
                    };
                    push_candidates(&args, &mut candidates, backend_derived, 0);

                    let Ok(derived_round_log_inv_rates) = derived_round_log_inv_rates else {
                        continue;
                    };
                    for round_log_inv_rate_offset in 1..=args.round_log_inv_rate_offset_max {
                        let Some(round_log_inv_rates) = offset_round_log_inv_rates(
                            &derived_round_log_inv_rates,
                            round_log_inv_rate_offset,
                        ) else {
                            continue;
                        };
                        let whir_params = WhirParams {
                            pow_bits: pow_bits as u32,
                            folding_factor: first,
                            starting_log_inv_rate,
                            rs_domain_initial_reduction_factor: rsv,
                            folding_schedule: Some(schedule.clone()),
                            round_log_inv_rates,
                        };
                        push_candidates(
                            &args,
                            &mut candidates,
                            whir_params,
                            round_log_inv_rate_offset,
                        );
                    }
                }
            }
        }
    }

    let dump = CandidateDump {
        schema_version: 5,
        provenance,
        matrix_closing: MatrixClosingMode::DirectSparse,
        base_field: args.field.label(),
        extension_filter: args.extension_filter.label(),
        num_variables: args.num_variables,
        num_outer_rounds: args.num_outer_rounds,
        target_security_bits: args.security_bits,
        target_merkle_security_bits: args.merkle_security_bits,
        component_security_override_bits: args.component_security_bits,
        component_merkle_security_override_bits: args.component_merkle_security_bits,
        soundness: SoundnessAssumption::JohnsonBound,
        max_pow_bits: args.max_pow_bits,
        round_log_inv_rate_offset_max: args.round_log_inv_rate_offset_max,
        proof_mode: args.proof_mode.label(),
        hash_profile: if cfg!(feature = "poseidon1") {
            "poseidon1"
        } else {
            "poseidon2"
        },
        zk_ell: args.proof_mode.is_full_zk().then_some(args.zk_ell),
        zk_mask_log_inv_rate: args
            .proof_mode
            .is_full_zk()
            .then_some(args.zk_mask_log_inv_rate),
        candidates,
    };
    serde_json::to_writer_pretty(std::io::stdout(), &dump).expect("write candidate JSON");
    println!();
}

fn push_candidates(
    args: &Args,
    out: &mut Vec<CandidateRow>,
    whir_params: WhirParams,
    round_log_inv_rate_offset: usize,
) {
    match args.field {
        FieldProfile::KoalaBear => {
            if args.extension_filter.includes("quartic") {
                derive_for_extension::<F, QuarticBinExtension, PoseidonChallenger>(
                    args,
                    out,
                    whir_params.clone(),
                    "quartic",
                    4,
                    round_log_inv_rate_offset,
                );
            }
            if args.extension_filter.includes("quintic") {
                derive_for_extension::<F, KoalaBearQuinticExtension, PoseidonChallenger>(
                    args,
                    out,
                    whir_params.clone(),
                    "quintic",
                    5,
                    round_log_inv_rate_offset,
                );
            }
            if args.extension_filter.includes("octic") {
                derive_for_extension::<F, OcticBinExtension, PoseidonChallenger>(
                    args,
                    out,
                    whir_params,
                    "octic",
                    8,
                    round_log_inv_rate_offset,
                );
            }
        }
        FieldProfile::BabyBear => {
            if args.extension_filter.includes("quartic") {
                derive_for_extension::<
                    BabyBear,
                    BabyBearQuarticExtension,
                    BabyBearPoseidonChallenger,
                >(
                    args,
                    out,
                    whir_params.clone(),
                    "quartic",
                    4,
                    round_log_inv_rate_offset,
                );
            }
            if args.extension_filter.includes("quintic") {
                derive_for_extension::<
                    BabyBear,
                    BabyBearQuinticExtension,
                    BabyBearPoseidonChallenger,
                >(
                    args,
                    out,
                    whir_params.clone(),
                    "quintic",
                    5,
                    round_log_inv_rate_offset,
                );
            }
            if args.extension_filter.includes("octic") {
                derive_for_extension::<BabyBear, BabyBearOcticExtension, BabyBearPoseidonChallenger>(
                    args,
                    out,
                    whir_params,
                    "octic",
                    8,
                    round_log_inv_rate_offset,
                );
            }
        }
    }
}

fn derive_for_extension<Base, Ext, Challenger>(
    args: &Args,
    out: &mut Vec<CandidateRow>,
    whir_params: WhirParams,
    extension: &'static str,
    extension_degree: usize,
    round_log_inv_rate_offset: usize,
) where
    Base: TwoAdicField,
    Ext: ExtensionField<Base> + Field + TwoAdicField,
    Challenger: FieldChallenger<Base> + GrindingChallenger<Witness = Base>,
{
    let label = format_whir_params_label(extension, &whir_params);
    let component_security_bits = match whir_component_security_bits(args) {
        Ok(bits) => bits,
        Err(reason) => {
            if args.include_invalid {
                out.push(invalid_row(
                    args,
                    label,
                    args.field.label(),
                    Base::TWO_ADICITY,
                    extension,
                    extension_degree,
                    Ext::TWO_ADICITY,
                    Ext::bits(),
                    round_log_inv_rate_offset,
                    whir_params,
                    reason,
                ));
            }
            return;
        }
    };
    let protocol_security_bits = if args.proof_mode.is_full_zk() {
        component_security_bits
            .checked_add(FULL_ZK_RELATION_SECURITY_SLACK_BITS)
            .unwrap_or(usize::MAX)
    } else {
        component_security_bits
    };
    let result = catch_unwind_silent(|| -> Result<_, String> {
        let protocol_params =
            protocol_parameters(args.num_variables, &whir_params, protocol_security_bits)?;
        P3WhirConfig::<Ext, Base, Challenger>::new(args.num_variables, protocol_params)
            .map_err(|reason| reason.to_string())
    });

    let config = match result {
        Ok(Ok(config)) => config,
        Ok(Err(reason)) => {
            if args.include_invalid {
                out.push(invalid_row(
                    args,
                    label,
                    args.field.label(),
                    Base::TWO_ADICITY,
                    extension,
                    extension_degree,
                    Ext::TWO_ADICITY,
                    Ext::bits(),
                    round_log_inv_rate_offset,
                    whir_params,
                    format!("backend rejected candidate: {reason}"),
                ));
            }
            return;
        }
        Err(reason) => {
            if args.include_invalid {
                out.push(invalid_row(
                    args,
                    label,
                    args.field.label(),
                    Base::TWO_ADICITY,
                    extension,
                    extension_degree,
                    Ext::TWO_ADICITY,
                    Ext::bits(),
                    round_log_inv_rate_offset,
                    whir_params,
                    format!("backend panicked while deriving candidate: {reason}"),
                ));
            }
            return;
        }
    };

    let achieved = achieved_security_bits::<Base, Ext, Challenger>(&config);
    let (terminal_queries, terminal_pow_bits) = if args.proof_mode.is_full_zk() {
        hiding_terminal_budget::<Base, Ext, Challenger>(&config)
    } else {
        (config.final_queries, config.final_pow_bits)
    };
    let max_pow = max_derived_pow_bits::<Base, Ext, Challenger>(&config).max(terminal_pow_bits);
    let merkle_component_security = direct_merkle_component_security_bits(args, config.n_rounds());
    let merkle_rejection = match &merkle_component_security {
        Ok(bits) if *bits <= MAX_SECURITY_BITS as usize => None,
        Ok(bits) => Some(format!(
            "derived Merkle component target {bits} exceeds maximum {MAX_SECURITY_BITS}"
        )),
        Err(reason) => Some(reason.clone()),
    };
    let zk_rejection = if args.proof_mode.is_full_zk() {
        full_zk_compatibility_error::<Base, Ext, Challenger>(args, &config)
    } else {
        None
    };
    let direct_setup_rejection = if args.uses_component_security_overrides() {
        None
    } else {
        direct_composed_security_error::<Ext>(args, config.n_rounds())
    };
    let valid = achieved >= protocol_security_bits as f64
        && max_pow <= args.max_pow_bits
        && merkle_rejection.is_none()
        && zk_rejection.is_none()
        && direct_setup_rejection.is_none();
    if !valid && !args.include_invalid {
        return;
    }

    let setup_config = (valid && !args.uses_component_security_overrides())
        .then(|| setup_config(args, whir_params.clone()));
    let rounds = config
        .round_parameters
        .iter()
        .enumerate()
        .map(|(round_index, round)| RoundRow {
            round_index,
            num_variables: round.num_variables,
            folding_factor: round.folding_factor,
            log_inv_rate: round.log_inv_rate,
            domain_size: round.domain_size,
            num_queries: round.num_queries,
            ood_samples: round.ood_samples,
            pow_bits: round.pow_bits,
            folding_pow_bits: round.folding_pow_bits,
        })
        .collect::<Vec<_>>();
    let pow_work_units = pow_work_units::<Base, Ext, Challenger>(&config, terminal_pow_bits);
    let dft_work = dft_work::<Base, Ext, Challenger>(&config);
    let merkle_work = merkle_work::<Base, Ext, Challenger>(&config);
    let merkle_path_work = merkle_path_work::<Base, Ext, Challenger>(&config);
    let row_work = row_work::<Base, Ext, Challenger>(&config);
    let sumcheck_work = sumcheck_work::<Base, Ext, Challenger>(&config);
    let verifier = verifier_work::<Base, Ext, Challenger>(&config, terminal_pow_bits);
    let plain_proof_size_bytes_estimate =
        proof_size_bytes_estimate::<Base, Ext, Challenger>(&config);
    let zk_estimates = if args.proof_mode.is_full_zk() && zk_rejection.is_none() {
        Some(zk_estimates::<Base, Ext, Challenger>(
            args,
            &config,
            ZkBaseEstimates {
                dft_work,
                merkle_work,
                merkle_path_work,
                row_work,
                sumcheck_work,
                verifier,
            },
        ))
    } else {
        None
    };
    let proof_size_bytes_estimate = zk_estimates
        .as_ref()
        .map(|estimate| estimate.proof_size_bytes_estimate)
        .unwrap_or(plain_proof_size_bytes_estimate);

    out.push(CandidateRow {
        label,
        proof_mode: args.proof_mode.label(),
        base_field: args.field.label(),
        base_two_adicity: Base::TWO_ADICITY,
        extension,
        extension_degree,
        extension_two_adicity: Ext::TWO_ADICITY,
        field_bits: Ext::bits(),
        round_log_inv_rate_offset,
        valid,
        rejection_reason: (!valid).then(|| {
            if achieved < protocol_security_bits as f64 {
                format!(
                    "achieved security {:.3} below target {}",
                    achieved, protocol_security_bits
                )
            } else if max_pow > args.max_pow_bits {
                format!("derived PoW {max_pow} exceeds max {}", args.max_pow_bits)
            } else if let Some(reason) = merkle_rejection {
                reason
            } else if let Some(reason) = direct_setup_rejection {
                reason
            } else {
                zk_rejection.unwrap_or_else(|| "candidate rejected".to_owned())
            }
        }),
        security_bits_achieved: Some(achieved),
        whir_component_security_bits: Some(component_security_bits),
        merkle_component_security_bits: merkle_component_security.ok(),
        max_derived_pow_bits: Some(max_pow),
        pow_work_units,
        dft_work,
        merkle_work,
        merkle_path_work,
        row_work,
        sumcheck_work,
        verifier_merkle_hashes: verifier.merkle_hashes,
        verifier_leaf_field_elements: verifier.leaf_field_elements,
        verifier_row_field_elements: verifier.row_field_elements,
        verifier_extension_operations: verifier.extension_operations,
        verifier_pow_checks: verifier.pow_checks,
        proof_size_bytes_estimate,
        zk_dft_work: zk_estimates.as_ref().map(|estimate| estimate.dft_work),
        zk_merkle_work: zk_estimates.as_ref().map(|estimate| estimate.merkle_work),
        zk_merkle_path_work: zk_estimates
            .as_ref()
            .map(|estimate| estimate.merkle_path_work),
        zk_row_work: zk_estimates.as_ref().map(|estimate| estimate.row_work),
        zk_sumcheck_work: zk_estimates.as_ref().map(|estimate| estimate.sumcheck_work),
        zk_verifier_merkle_hashes: zk_estimates
            .as_ref()
            .map(|estimate| estimate.verifier.merkle_hashes),
        zk_verifier_leaf_field_elements: zk_estimates
            .as_ref()
            .map(|estimate| estimate.verifier.leaf_field_elements),
        zk_verifier_row_field_elements: zk_estimates
            .as_ref()
            .map(|estimate| estimate.verifier.row_field_elements),
        zk_verifier_extension_operations: zk_estimates
            .as_ref()
            .map(|estimate| estimate.verifier.extension_operations),
        zk_verifier_pow_checks: zk_estimates
            .as_ref()
            .map(|estimate| estimate.verifier.pow_checks),
        zk_proof_size_bytes_estimate: zk_estimates
            .as_ref()
            .map(|estimate| estimate.proof_size_bytes_estimate),
        zk_mask_queries: zk_estimates.as_ref().map(|estimate| estimate.mask_queries),
        zk_application_mask_domain: zk_estimates
            .as_ref()
            .map(|estimate| estimate.application_mask.domain_size),
        zk_application_mask_width: zk_estimates
            .as_ref()
            .map(|estimate| estimate.application_mask.width),
        zk_ell: args.proof_mode.is_full_zk().then_some(args.zk_ell),
        zk_mask_log_inv_rate: args
            .proof_mode
            .is_full_zk()
            .then_some(args.zk_mask_log_inv_rate),
        commitment_ood_samples: Some(config.commitment_ood_samples),
        starting_folding_pow_bits: Some(config.starting_folding_pow_bits),
        final_queries: Some(terminal_queries),
        final_pow_bits: Some(terminal_pow_bits),
        final_sumcheck_rounds: Some(config.final_sumcheck_rounds),
        final_folding_pow_bits: Some(config.final_folding_pow_bits),
        rounds,
        whir_params,
        setup_config,
    });
}

fn protocol_parameters(
    num_variables: usize,
    whir_params: &WhirParams,
    security_bits: usize,
) -> Result<ProtocolParameters, String> {
    let folding_schedule = whir_params.effective_folding_schedule();
    let round_log_inv_rates = if whir_params.round_log_inv_rates.is_empty() {
        derived_round_log_inv_rates(
            num_variables,
            &folding_schedule,
            whir_params.starting_log_inv_rate,
            whir_params.rs_domain_initial_reduction_factor,
        )?
    } else {
        whir_params.round_log_inv_rates.clone()
    };
    Ok(ProtocolParameters {
        starting_log_inv_rate: whir_params.starting_log_inv_rate,
        round_log_inv_rates,
        folding_factor: map_schedule(&folding_schedule),
        soundness_type: P3SecurityAssumption::JohnsonBound,
        security_level: security_bits,
        pow_bits: whir_params.pow_bits as usize,
    })
}

struct ZkBaseEstimates {
    dft_work: u128,
    merkle_work: u128,
    merkle_path_work: u128,
    row_work: u128,
    sumcheck_work: u128,
    verifier: VerifierWorkEstimate,
}

struct ZkEstimates {
    dft_work: u128,
    merkle_work: u128,
    merkle_path_work: u128,
    row_work: u128,
    sumcheck_work: u128,
    verifier: VerifierWorkEstimate,
    proof_size_bytes_estimate: u128,
    mask_queries: usize,
    application_mask: ApplicationMaskEstimate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ApplicationMaskEstimate {
    domain_size: usize,
    log_domain_size: usize,
    width: usize,
}

fn zk_estimates<Base, Ext, Challenger>(
    args: &Args,
    config: &P3WhirConfig<Ext, Base, Challenger>,
    base: ZkBaseEstimates,
) -> ZkEstimates
where
    Base: TwoAdicField,
    Ext: ExtensionField<Base> + Field + TwoAdicField,
    Challenger: FieldChallenger<Base> + GrindingChallenger<Witness = Base>,
{
    let n_rounds = config.n_rounds();
    let oracle_randomness = zk_oracle_randomness(config);
    let mask_queries = zk_mask_queries(args, config);
    let application_mask = application_mask_estimate::<Ext>(args, mask_queries)
        .expect("application mask geometry was validated before estimation");
    let sumcheck_rounds = zk_sumcheck_rounds(config);
    let sumcheck_mask_domain = mask_domain(args.zk_ell, mask_queries, args.zk_mask_log_inv_rate);
    let switch_mask_domains = (0..n_rounds)
        .map(|round| {
            mask_domain(
                oracle_randomness[round] + config.round_parameters[round].ood_samples,
                mask_queries,
                args.zk_mask_log_inv_rate,
            )
        })
        .collect::<Vec<_>>();

    let ext_dim = Ext::DIMENSION as u128;
    let sumcheck_mask_width: u128 = sumcheck_rounds.iter().sum::<usize>() as u128;
    let switch_mask_width = n_rounds as u128;
    let base_mask_width = sumcheck_mask_width.saturating_add(switch_mask_width);

    let sumcheck_mask_rows = (sumcheck_mask_domain as u128).saturating_mul(sumcheck_mask_width);
    let switch_mask_rows = switch_mask_domains
        .iter()
        .map(|domain| *domain as u128)
        .fold(0, u128::saturating_add);
    let mask_rows = sumcheck_mask_rows.saturating_add(switch_mask_rows);

    let sumcheck_work_extra = sumcheck_rounds
        .iter()
        .map(|rounds| (*rounds as u128).saturating_mul(args.zk_ell.max(3) as u128))
        .fold(0, u128::saturating_add);
    let verifier_extension_operations_extra = sumcheck_rounds
        .iter()
        .map(|rounds| {
            (*rounds as u128)
                .saturating_mul(args.zk_ell.max(3) as u128)
                .saturating_mul(2)
        })
        .fold(0, u128::saturating_add);
    let application_mask_rows = (application_mask.domain_size as u128)
        .saturating_mul(application_mask.width as u128)
        .saturating_mul(2);
    let dft_work_extra = mask_rows
        .saturating_add(application_mask_rows)
        .saturating_mul(ext_dim);
    let merkle_work_extra = mask_rows
        .saturating_add(application_mask_rows)
        .saturating_mul(ext_dim);

    let switch_mask_path_work = switch_mask_domains
        .iter()
        .map(|domain| (mask_queries as u128).saturating_mul(log2_usize(*domain)))
        .fold(0, u128::saturating_add);
    let sumcheck_mask_path_work = (mask_queries as u128)
        .saturating_mul(log2_usize(sumcheck_mask_domain))
        .saturating_mul((n_rounds + 1) as u128);
    let base_mask_path_work = (mask_queries as u128)
        .saturating_mul(base_mask_width)
        .saturating_mul(log2_usize(sumcheck_mask_domain));
    let merkle_path_work_extra = switch_mask_path_work
        .saturating_add(sumcheck_mask_path_work)
        .saturating_add(base_mask_path_work)
        .saturating_add(
            (mask_queries as u128)
                .saturating_mul(application_mask.log_domain_size as u128)
                .saturating_mul(2),
        );

    let row_work_extra = (mask_queries as u128)
        .saturating_mul(
            switch_mask_width
                .saturating_add(sumcheck_mask_width)
                .saturating_add(base_mask_width.saturating_mul(2)),
        )
        .saturating_mul(ext_dim)
        .saturating_add(
            (mask_queries as u128)
                .saturating_mul(application_mask.width as u128)
                .saturating_mul(ext_dim)
                .saturating_mul(2),
        );

    let proof_size_bytes_estimate = zk_proof_size_bytes_estimate::<Base, Ext, Challenger>(
        args,
        config,
        &oracle_randomness,
        &sumcheck_rounds,
        mask_queries,
        sumcheck_mask_domain,
        &switch_mask_domains,
        application_mask,
    );

    ZkEstimates {
        dft_work: base.dft_work.saturating_add(dft_work_extra),
        merkle_work: base.merkle_work.saturating_add(merkle_work_extra),
        merkle_path_work: base.merkle_path_work.saturating_add(merkle_path_work_extra),
        row_work: base.row_work.saturating_add(row_work_extra),
        sumcheck_work: base.sumcheck_work.saturating_add(sumcheck_work_extra),
        verifier: VerifierWorkEstimate {
            merkle_hashes: base
                .verifier
                .merkle_hashes
                .saturating_add(zk_merkle_multiproof_hashes(
                    config,
                    mask_queries,
                    sumcheck_mask_domain,
                    &switch_mask_domains,
                    application_mask,
                    base_mask_width,
                )),
            leaf_field_elements: base
                .verifier
                .leaf_field_elements
                .saturating_add(row_work_extra),
            row_field_elements: base
                .verifier
                .row_field_elements
                .saturating_add(row_work_extra),
            extension_operations: base
                .verifier
                .extension_operations
                .saturating_add(verifier_extension_operations_extra),
            pow_checks: base.verifier.pow_checks,
        },
        proof_size_bytes_estimate,
        mask_queries,
        application_mask,
    }
}

fn zk_proof_size_bytes_estimate<Base, Ext, Challenger>(
    args: &Args,
    config: &P3WhirConfig<Ext, Base, Challenger>,
    oracle_randomness: &[usize],
    sumcheck_rounds: &[usize],
    mask_queries: usize,
    sumcheck_mask_domain: usize,
    switch_mask_domains: &[usize],
    application_mask: ApplicationMaskEstimate,
) -> u128
where
    Base: TwoAdicField,
    Ext: ExtensionField<Base> + Field + TwoAdicField,
    Challenger: FieldChallenger<Base> + GrindingChallenger<Witness = Base>,
{
    let n_rounds = config.n_rounds();
    let ext_bytes = extension_bytes::<Ext>();
    let field_bytes = FIELD_BYTES;
    let sumcheck_coeffs_per_round = args.zk_ell.saturating_sub(1).max(2) as u128;
    let sumcheck_bytes = sumcheck_rounds
        .iter()
        .enumerate()
        .map(|(index, rounds)| {
            let pow_bytes = if zk_sumcheck_pow_bits(config, index) > 0 {
                (*rounds as u128).saturating_mul(field_bytes)
            } else {
                0
            };
            ext_bytes
                .saturating_add(
                    (*rounds as u128)
                        .saturating_mul(sumcheck_coeffs_per_round)
                        .saturating_mul(ext_bytes),
                )
                .saturating_add(pow_bytes)
        })
        .fold(0, u128::saturating_add);
    let sumcheck_mask_commitments = ((n_rounds + 1) as u128).saturating_mul(POSEIDON_DIGEST_BYTES);
    let round_bytes = config
        .round_parameters
        .iter()
        .enumerate()
        .map(|(round, params)| {
            let query_bytes = actual_query_count(
                params.num_queries,
                params.domain_size,
                params.folding_factor,
            )
            .saturating_mul(
                row_width(params.folding_factor)
                    .saturating_mul(query_payload_degree::<Base, Ext>(round))
                    .saturating_mul(FIELD_BYTES)
                    .saturating_add(
                        path_depth(params.domain_size, params.folding_factor)
                            .saturating_mul(POSEIDON_DIGEST_BYTES),
                    ),
            );
            POSEIDON_DIGEST_BYTES
                .saturating_mul(2)
                .saturating_add((params.ood_samples as u128).saturating_mul(ext_bytes))
                .saturating_add(FIELD_BYTES)
                .saturating_add(query_bytes)
        })
        .fold(0, u128::saturating_add);

    let final_round = final_round_estimate(config);
    let final_queries = actual_query_count(
        oracle_randomness[n_rounds],
        final_round.domain_size,
        final_round.folding_factor,
    );
    let source_query_bytes = final_queries.saturating_mul(
        row_width(final_round.folding_factor)
            .saturating_mul(final_payload_degree::<Base, Ext, Challenger>(config))
            .saturating_mul(FIELD_BYTES)
            .saturating_add(
                path_depth(final_round.domain_size, final_round.folding_factor)
                    .saturating_mul(POSEIDON_DIGEST_BYTES),
            ),
    );
    let fresh_main_query_bytes = final_queries.saturating_mul(
        ext_bytes.saturating_add(
            path_depth(final_round.domain_size, final_round.folding_factor)
                .saturating_mul(POSEIDON_DIGEST_BYTES),
        ),
    );
    let sumcheck_mask_width: u128 = sumcheck_rounds.iter().sum::<usize>() as u128;
    let switch_mask_width = n_rounds as u128;
    let base_mask_width = sumcheck_mask_width.saturating_add(switch_mask_width);
    let mask_query_bytes = (mask_queries as u128)
        .saturating_mul(base_mask_width)
        .saturating_mul(2)
        .saturating_mul(ext_bytes.saturating_add(
            log2_usize(sumcheck_mask_domain).saturating_mul(POSEIDON_DIGEST_BYTES),
        ));
    let switch_mask_query_bytes = switch_mask_domains
        .iter()
        .map(|domain| {
            (mask_queries as u128).saturating_mul(
                ext_bytes.saturating_add(log2_usize(*domain).saturating_mul(POSEIDON_DIGEST_BYTES)),
            )
        })
        .fold(0, u128::saturating_add);
    let base_blinded_bytes = final_message_len(config)
        .saturating_mul(ext_bytes)
        .saturating_add((oracle_randomness[n_rounds] as u128).saturating_mul(ext_bytes))
        .saturating_add(
            carried_mask_message_len(args, config, oracle_randomness).saturating_mul(ext_bytes),
        );
    let base_commitment_count = 1u128.saturating_add(base_mask_width);
    let base_case_bytes = base_commitment_count
        .saturating_mul(POSEIDON_DIGEST_BYTES)
        .saturating_add(ext_bytes)
        .saturating_add(base_blinded_bytes)
        .saturating_add(FIELD_BYTES)
        .saturating_add(source_query_bytes)
        .saturating_add(fresh_main_query_bytes)
        .saturating_add(mask_query_bytes);

    let application_mask_bytes = POSEIDON_DIGEST_BYTES
        .saturating_mul(2)
        .saturating_add(
            (application_mask.width as u128)
                .saturating_mul((8usize.saturating_add(mask_queries)) as u128)
                .saturating_mul(ext_bytes),
        )
        .saturating_add(
            (mask_queries as u128)
                .saturating_mul(application_mask.width as u128)
                .saturating_mul(ext_bytes)
                .saturating_mul(2),
        )
        .saturating_add(
            (mask_queries as u128)
                .saturating_mul(application_mask.log_domain_size as u128)
                .saturating_mul(POSEIDON_DIGEST_BYTES)
                .saturating_mul(2),
        );

    ext_bytes
        .saturating_add(sumcheck_bytes)
        .saturating_add(sumcheck_mask_commitments)
        .saturating_add(round_bytes)
        .saturating_add(switch_mask_query_bytes)
        .saturating_add(base_case_bytes)
        .saturating_add(application_mask_bytes)
}

fn zk_oracle_randomness<Base, Ext, Challenger>(
    config: &P3WhirConfig<Ext, Base, Challenger>,
) -> Vec<usize>
where
    Base: TwoAdicField,
    Ext: ExtensionField<Base> + Field + TwoAdicField,
    Challenger: FieldChallenger<Base> + GrindingChallenger<Witness = Base>,
{
    let (final_queries, _) = hiding_terminal_budget(config);
    (0..=config.n_rounds())
        .map(|round| {
            if round < config.n_rounds() {
                config.round_parameters[round].num_queries
            } else {
                final_queries
            }
        })
        .collect()
}

fn zk_mask_queries<Base, Ext, Challenger>(
    args: &Args,
    config: &P3WhirConfig<Ext, Base, Challenger>,
) -> usize
where
    Base: TwoAdicField,
    Ext: ExtensionField<Base> + Field + TwoAdicField,
    Challenger: FieldChallenger<Base> + GrindingChallenger<Witness = Base>,
{
    let union = log2_ceil_usize(2 * config.n_rounds() + 2);
    query_count_for_assumption(
        config.params.soundness_type,
        config.params.security_level + union,
        args.zk_mask_log_inv_rate,
    )
}

fn zk_sumcheck_rounds<Base, Ext, Challenger>(
    config: &P3WhirConfig<Ext, Base, Challenger>,
) -> Vec<usize>
where
    Base: TwoAdicField,
    Ext: ExtensionField<Base> + Field + TwoAdicField,
    Challenger: FieldChallenger<Base> + GrindingChallenger<Witness = Base>,
{
    (0..=config.n_rounds())
        .map(|round| config.round_folding_factor(round))
        .collect()
}

fn zk_sumcheck_pow_bits<Base, Ext, Challenger>(
    config: &P3WhirConfig<Ext, Base, Challenger>,
    batch: usize,
) -> usize
where
    Base: Field,
    Ext: ExtensionField<Base> + Field,
{
    if batch == 0 {
        config.starting_folding_pow_bits
    } else if batch - 1 < config.round_parameters.len() {
        config.round_parameters[batch - 1].folding_pow_bits
    } else {
        config.final_folding_pow_bits
    }
}

fn mask_domain(message_len: usize, randomness_len: usize, log_inv_rate: usize) -> usize {
    (message_len + randomness_len).next_power_of_two() << log_inv_rate
}

fn extension_bytes<Ext>() -> u128
where
    Ext: Field,
{
    (Ext::bits() as u128).div_ceil(8)
}

fn final_message_len<Base, Ext, Challenger>(config: &P3WhirConfig<Ext, Base, Challenger>) -> u128
where
    Base: TwoAdicField,
    Ext: ExtensionField<Base> + Field + TwoAdicField,
    Challenger: FieldChallenger<Base> + GrindingChallenger<Witness = Base>,
{
    let final_round = final_round_estimate(config);
    folded_row_count(final_round.domain_size, final_round.folding_factor) as u128
}

fn carried_mask_message_len<Base, Ext, Challenger>(
    args: &Args,
    config: &P3WhirConfig<Ext, Base, Challenger>,
    oracle_randomness: &[usize],
) -> u128
where
    Base: TwoAdicField,
    Ext: ExtensionField<Base> + Field + TwoAdicField,
    Challenger: FieldChallenger<Base> + GrindingChallenger<Witness = Base>,
{
    let sumcheck_messages = zk_sumcheck_rounds(config).len().saturating_mul(args.zk_ell) as u128;
    let switch_messages = (0..config.n_rounds())
        .map(|round| oracle_randomness[round] + config.round_parameters[round].ood_samples)
        .sum::<usize>() as u128;
    sumcheck_messages.saturating_add(switch_messages)
}

fn full_zk_compatibility_error<Base, Ext, Challenger>(
    args: &Args,
    config: &P3WhirConfig<Ext, Base, Challenger>,
) -> Option<String>
where
    Base: TwoAdicField,
    Ext: ExtensionField<Base> + Field + TwoAdicField,
    Challenger: FieldChallenger<Base> + GrindingChallenger<Witness = Base>,
{
    let soundness_error_terms = match direct_algebraic_error_terms(args) {
        Ok(terms) => terms,
        Err(reason) => return Some(reason),
    };
    let requested_bits = args.security_bits.min(args.merkle_security_bits);
    let Some(required_bits) = requested_bits.checked_add(FULL_ZK_RELATION_SECURITY_SLACK_BITS)
    else {
        return Some("full-ZK relation security level overflows".to_owned());
    };
    let required_order = BigUint::from(soundness_error_terms) << required_bits;
    if Ext::order() < required_order {
        return Some(format!(
            "full-ZK security target {} exceeds extension field capacity after {} local error terms",
            requested_bits, soundness_error_terms
        ));
    }
    if args.zk_ell < 3 {
        return Some(format!(
            "ZK mask length {} is below the minimum of 3",
            args.zk_ell
        ));
    }
    if args.zk_mask_log_inv_rate == 0 {
        return Some("ZK mask log inverse rate must be at least 1".to_owned());
    }

    let n_rounds = config.n_rounds();
    let oracle_randomness = zk_oracle_randomness(config);

    for (round, &randomness) in oracle_randomness.iter().enumerate() {
        let Some((message_rows, height)) = oracle_shape(config, round) else {
            return Some(format!("ZK oracle shape overflows at round {round}"));
        };
        let slack = height.saturating_sub(message_rows);
        if randomness > slack {
            return Some(format!(
                "ZK round {round} randomness rows {randomness} exceed slack {slack}"
            ));
        }
    }

    let union = log2_ceil_usize(2 * n_rounds + 2);
    let mask_queries = query_count_for_assumption(
        config.params.soundness_type,
        config.params.security_level + union,
        args.zk_mask_log_inv_rate,
    );
    let mask_message_lens =
        core::iter::once(args.zk_ell)
            .chain((0..n_rounds).map(|round| {
                oracle_randomness[round] + config.round_parameters[round].ood_samples
            }));
    for message_len in mask_message_lens {
        let Some(unencoded_len) = message_len.checked_add(mask_queries) else {
            return Some("ZK mask message and randomness length overflow".to_owned());
        };
        let Some(log_domain_size) =
            log2_ceil_usize(unencoded_len).checked_add(args.zk_mask_log_inv_rate)
        else {
            return Some("ZK mask domain log size overflows".to_owned());
        };
        if log_domain_size > Ext::TWO_ADICITY {
            return Some(format!(
                "ZK mask domain 2^{log_domain_size} exceeds extension-field two-adicity 2^{}",
                Ext::TWO_ADICITY
            ));
        }
    }
    if let Err(reason) = application_mask_estimate::<Ext>(args, mask_queries) {
        return Some(reason);
    }
    None
}

fn application_mask_estimate<Ext>(
    args: &Args,
    mask_queries: usize,
) -> Result<ApplicationMaskEstimate, String>
where
    Ext: TwoAdicField,
{
    let inner_log = checked_mask_log_domain::<Ext>(
        4,
        mask_queries,
        args.zk_mask_log_inv_rate,
        "inner application",
    )?;
    let outer_log = checked_mask_log_domain::<Ext>(
        8,
        mask_queries,
        args.zk_mask_log_inv_rate,
        "outer application",
    )?;
    if inner_log != outer_log {
        return Err(format!(
            "IncompatibleApplicationMaskDomains: inner domain 2^{inner_log}, outer domain 2^{outer_log}"
        ));
    }
    let width = args
        .num_outer_rounds
        .checked_mul(4)
        .ok_or_else(|| "combined application mask width overflows".to_owned())?;
    let domain_size = 1usize
        .checked_shl(inner_log as u32)
        .ok_or_else(|| "combined application mask domain size overflows".to_owned())?;
    Ok(ApplicationMaskEstimate {
        domain_size,
        log_domain_size: inner_log,
        width,
    })
}

fn checked_mask_log_domain<Ext>(
    message_len: usize,
    randomness_len: usize,
    log_inv_rate: usize,
    label: &str,
) -> Result<usize, String>
where
    Ext: TwoAdicField,
{
    let unencoded_len = message_len
        .checked_add(randomness_len)
        .ok_or_else(|| format!("{label} mask message and randomness length overflow"))?;
    let log_domain_size = log2_ceil_usize(unencoded_len)
        .checked_add(log_inv_rate)
        .ok_or_else(|| format!("{label} mask domain log size overflows"))?;
    if log_domain_size > Ext::TWO_ADICITY || log_domain_size >= usize::BITS as usize {
        return Err(format!(
            "{label} mask domain 2^{log_domain_size} exceeds extension-field two-adicity 2^{}",
            Ext::TWO_ADICITY
        ));
    }
    Ok(log_domain_size)
}

fn oracle_shape<Base, Ext, Challenger>(
    config: &P3WhirConfig<Ext, Base, Challenger>,
    round: usize,
) -> Option<(usize, usize)>
where
    Base: TwoAdicField,
    Ext: ExtensionField<Base> + Field + TwoAdicField,
    Challenger: FieldChallenger<Base> + GrindingChallenger<Witness = Base>,
{
    if round == 0 {
        let rows = 1usize.checked_shl(
            config
                .num_variables
                .checked_sub(config.round_folding_factor(0))? as u32,
        )?;
        Some((rows, rows.checked_shl(config.starting_log_inv_rate as u32)?))
    } else {
        let prev = &config.round_parameters[round - 1];
        let rows = 1usize.checked_shl(
            prev.num_variables
                .checked_sub(config.round_folding_factor(round))? as u32,
        )?;
        Some((rows, rows.checked_mul(config.inv_rate(round - 1))?))
    }
}

fn query_count_for_assumption(
    soundness: P3SecurityAssumption,
    protocol_security_level: usize,
    log_inv_rate: usize,
) -> usize {
    let num_queries_f = -(protocol_security_level as f64) / log_1_delta(soundness, log_inv_rate);
    num_queries_f.ceil() as usize
}

fn log_1_delta(soundness: P3SecurityAssumption, log_inv_rate: usize) -> f64 {
    let rate = 1.0 / ((1usize << log_inv_rate) as f64);
    let delta = match soundness {
        P3SecurityAssumption::UniqueDecoding => 0.5 * (1.0 - rate),
        P3SecurityAssumption::JohnsonBound => {
            1.0 - rate.sqrt() - 2f64.powf(-(0.5 * log_inv_rate as f64 + 10f64.log2() + 1.0))
        }
        P3SecurityAssumption::CapacityBound => {
            1.0 - rate - 2f64.powf(-(log_inv_rate as f64 + 10f64.log2() + 1.0))
        }
    };
    (1.0 - delta).log2()
}

fn log2_ceil_usize(value: usize) -> usize {
    if value <= 1 {
        0
    } else {
        usize::BITS as usize - (value - 1).leading_zeros() as usize
    }
}

fn catch_unwind_silent<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    let previous = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let result = panic::catch_unwind(panic::AssertUnwindSafe(f));
    panic::set_hook(previous);
    result.map_err(|payload| {
        if let Some(message) = payload.downcast_ref::<&str>() {
            (*message).to_owned()
        } else if let Some(message) = payload.downcast_ref::<String>() {
            message.clone()
        } else {
            "non-string panic payload".to_owned()
        }
    })
}

fn invalid_row(
    args: &Args,
    label: String,
    base_field: &'static str,
    base_two_adicity: usize,
    extension: &'static str,
    extension_degree: usize,
    extension_two_adicity: usize,
    field_bits: usize,
    round_log_inv_rate_offset: usize,
    whir_params: WhirParams,
    reason: impl Into<String>,
) -> CandidateRow {
    CandidateRow {
        label,
        proof_mode: args.proof_mode.label(),
        base_field,
        base_two_adicity,
        extension,
        extension_degree,
        extension_two_adicity,
        field_bits,
        round_log_inv_rate_offset,
        valid: false,
        rejection_reason: Some(reason.into()),
        security_bits_achieved: None,
        whir_component_security_bits: None,
        merkle_component_security_bits: None,
        max_derived_pow_bits: None,
        pow_work_units: 0,
        dft_work: 0,
        merkle_work: 0,
        merkle_path_work: 0,
        row_work: 0,
        sumcheck_work: 0,
        verifier_merkle_hashes: 0,
        verifier_leaf_field_elements: 0,
        verifier_row_field_elements: 0,
        verifier_extension_operations: 0,
        verifier_pow_checks: 0,
        proof_size_bytes_estimate: 0,
        zk_dft_work: None,
        zk_merkle_work: None,
        zk_merkle_path_work: None,
        zk_row_work: None,
        zk_sumcheck_work: None,
        zk_verifier_merkle_hashes: None,
        zk_verifier_leaf_field_elements: None,
        zk_verifier_row_field_elements: None,
        zk_verifier_extension_operations: None,
        zk_verifier_pow_checks: None,
        zk_proof_size_bytes_estimate: None,
        zk_mask_queries: None,
        zk_application_mask_domain: None,
        zk_application_mask_width: None,
        zk_ell: args.proof_mode.is_full_zk().then_some(args.zk_ell),
        zk_mask_log_inv_rate: args
            .proof_mode
            .is_full_zk()
            .then_some(args.zk_mask_log_inv_rate),
        commitment_ood_samples: None,
        starting_folding_pow_bits: None,
        final_queries: None,
        final_pow_bits: None,
        final_sumcheck_rounds: None,
        final_folding_pow_bits: None,
        rounds: Vec::new(),
        whir_params,
        setup_config: None,
    }
}

fn setup_config(args: &Args, whir_params: WhirParams) -> serde_json::Value {
    let security = requested_security_config(args)
        .expect("candidate arguments carry a valid requested security config");
    match args.proof_mode {
        ProofMode::NoZk => serde_json::to_value(SpartanSnarkConfig {
            matrix_closing: MatrixClosingMode::DirectSparse,
            security,
            whir_params,
            spark_whir_params: None,
        }),
        ProofMode::FullZk => serde_json::to_value(PoseidonZkSetupConfig {
            matrix_closing: MatrixClosingMode::DirectSparse,
            security,
            whir_params,
            spark_whir_params: None,
            ell_zk: args.zk_ell,
            mask_log_inv_rate: args.zk_mask_log_inv_rate,
        }),
    }
    .expect("setup config serializes")
}

fn requested_security_config(args: &Args) -> Result<SecurityConfig, String> {
    let security = SecurityConfig {
        security_level_bits: u32::try_from(args.security_bits)
            .map_err(|_| "requested security level does not fit in u32".to_owned())?,
        merkle_security_bits: u32::try_from(args.merkle_security_bits)
            .map_err(|_| "requested Merkle security level does not fit in u32".to_owned())?,
        soundness_assumption: SoundnessAssumption::JohnsonBound,
    };
    security
        .validate()
        .map_err(|error| format!("invalid requested security config: {error}"))?;
    Ok(security)
}

fn validate_requested_security_config(args: &Args) -> Result<(), String> {
    requested_security_config(args).map(|_| ())
}

impl Args {
    fn uses_component_security_overrides(&self) -> bool {
        self.component_security_bits.is_some()
    }
}

fn whir_component_security_bits(args: &Args) -> Result<usize, String> {
    if let Some(bits) = args.component_security_bits {
        return Ok(bits);
    }
    args.security_bits
        .min(args.merkle_security_bits)
        .checked_add(three_way_budget_slack(1)?)
        .ok_or_else(|| "WHIR component security target overflows".to_owned())
}

fn direct_merkle_component_security_bits(
    args: &Args,
    witness_rounds: usize,
) -> Result<usize, String> {
    if let Some(bits) = args.component_merkle_security_bits {
        return Ok(bits);
    }
    let events = direct_commitment_binding_events(args, witness_rounds)?;
    args.security_bits
        .min(args.merkle_security_bits)
        .checked_add(three_way_budget_slack(events)?)
        .ok_or_else(|| "Merkle component security target overflows".to_owned())
}

fn direct_commitment_binding_events(args: &Args, witness_rounds: usize) -> Result<usize, String> {
    match args.proof_mode {
        ProofMode::NoZk => witness_rounds.checked_add(1),
        // Keep the separate-tree upper bound used by setup, including when
        // the SPARK consumer batches fresh masks at an unchanged schedule.
        ProofMode::FullZk => witness_rounds
            .checked_mul(2)
            .and_then(|n| n.checked_add(3))
            .and_then(|fresh| {
                spartan_whir::security::full_zk_witness_commitment_events(witness_rounds, fresh)
                    .ok()
            }),
    }
    .ok_or_else(|| "DirectSparse commitment event count overflows".to_owned())
}

fn direct_algebraic_error_terms(args: &Args) -> Result<usize, String> {
    let num_inner_rounds = args
        .num_variables
        .checked_add(1)
        .ok_or_else(|| "DirectSparse inner round count overflows".to_owned())?;
    match args.proof_mode {
        ProofMode::NoZk => args
            .num_outer_rounds
            .checked_mul(4)
            .and_then(|outer| {
                num_inner_rounds
                    .checked_mul(2)
                    .and_then(|inner| outer.checked_add(inner))
            })
            .and_then(|terms| terms.checked_add(2)),
        ProofMode::FullZk => {
            let inner_degree = args.zk_ell.saturating_sub(1).max(2);
            args.num_outer_rounds
                .checked_mul(15)
                .and_then(|outer| {
                    inner_degree
                        .checked_mul(num_inner_rounds)
                        .and_then(|inner| outer.checked_add(inner))
                })
                .and_then(|terms| terms.checked_add(4))
        }
    }
    .ok_or_else(|| "DirectSparse algebraic error term count overflows".to_owned())
}

fn direct_composed_security_error<Ext>(args: &Args, witness_rounds: usize) -> Option<String>
where
    Ext: Field,
{
    let algebraic_error_terms = match direct_algebraic_error_terms(args) {
        Ok(terms) => terms,
        Err(reason) => return Some(reason),
    };
    let commitment_binding_events = match direct_commitment_binding_events(args, witness_rounds) {
        Ok(events) => events,
        Err(reason) => return Some(reason),
    };
    let whir_slack_bits = match three_way_budget_slack(1) {
        Ok(bits) => bits,
        Err(reason) => return Some(reason),
    };
    let merkle_slack_bits = match three_way_budget_slack(commitment_binding_events) {
        Ok(bits) => bits,
        Err(reason) => return Some(reason),
    };
    let field_denominator = BigUint::from(algebraic_error_terms) * BigUint::from(3u8);
    let field_attainable = (Ext::order() / field_denominator)
        .bits()
        .saturating_sub(1)
        .min(u32::MAX as u64) as usize;
    let whir_attainable = (MAX_SECURITY_BITS as usize).saturating_sub(whir_slack_bits);
    let merkle_attainable = (MAX_SECURITY_BITS as usize).saturating_sub(merkle_slack_bits);
    let attainable = field_attainable.min(whir_attainable).min(merkle_attainable);
    let requested = args.security_bits.min(args.merkle_security_bits);
    (requested > attainable).then(|| {
        format!(
            "DirectSparse requested security {requested} exceeds composed-security limit {attainable}"
        )
    })
}

fn three_way_budget_slack(events: usize) -> Result<usize, String> {
    let weighted = events
        .checked_mul(3)
        .ok_or_else(|| "composed security event count overflows".to_owned())?;
    Ok(usize::BITS as usize - weighted.saturating_sub(1).leading_zeros() as usize)
}

fn achieved_security_bits<Base, Ext, Challenger>(
    config: &P3WhirConfig<Ext, Base, Challenger>,
) -> f64
where
    Base: Field,
    Ext: ExtensionField<Base> + Field,
{
    config.params.security_level as f64
}

#[cfg(any())]
fn folding_security(
    soundness: P3SecurityAssumption,
    field_bits: usize,
    num_variables: usize,
    log_inv_rate: usize,
    pow_bits: usize,
) -> f64 {
    soundness
        .prox_gaps_error(num_variables, log_inv_rate, field_bits, 2)
        .min(soundness.fold_sumcheck_error(field_bits, num_variables, log_inv_rate))
        + pow_bits as f64
}

fn max_derived_pow_bits<Base, Ext, Challenger>(
    config: &P3WhirConfig<Ext, Base, Challenger>,
) -> usize
where
    Base: Field,
    Ext: ExtensionField<Base> + Field,
{
    config.round_parameters.iter().fold(
        config
            .starting_folding_pow_bits
            .max(config.final_pow_bits)
            .max(config.final_folding_pow_bits),
        |acc, round| acc.max(round.pow_bits).max(round.folding_pow_bits),
    )
}

fn pow_work_units<Base, Ext, Challenger>(
    config: &P3WhirConfig<Ext, Base, Challenger>,
    terminal_pow_bits: usize,
) -> u128
where
    Base: Field,
    Ext: ExtensionField<Base> + Field,
{
    let mut bits = vec![
        config.starting_folding_pow_bits,
        terminal_pow_bits,
        config.final_folding_pow_bits,
    ];
    for round in &config.round_parameters {
        bits.push(round.pow_bits);
        bits.push(round.folding_pow_bits);
    }
    bits.into_iter()
        .filter(|bits| *bits > 0)
        .map(|bits| 1u128.checked_shl(bits as u32).unwrap_or(u128::MAX))
        .fold(0u128, u128::saturating_add)
}

fn dft_work<Base, Ext, Challenger>(config: &P3WhirConfig<Ext, Base, Challenger>) -> u128
where
    Base: Field,
    Ext: ExtensionField<Base> + Field,
{
    let initial = 1u128 << (config.num_variables + config.params.starting_log_inv_rate);
    config.round_parameters.iter().fold(initial, |acc, round| {
        acc.saturating_add(round.domain_size as u128)
    })
}

fn merkle_work<Base, Ext, Challenger>(config: &P3WhirConfig<Ext, Base, Challenger>) -> u128
where
    Base: TwoAdicField,
    Ext: ExtensionField<Base> + Field + TwoAdicField,
    Challenger: FieldChallenger<Base> + GrindingChallenger<Witness = Base>,
{
    let initial = config.starting_domain_size() as u128;
    (0..config.n_rounds())
        .map(|round| {
            let committed_domain =
                config.round_parameters[round].domain_size >> config.rs_reduction_factor(round);
            (committed_domain as u128).saturating_mul(Ext::DIMENSION as u128)
        })
        .fold(initial, u128::saturating_add)
}

fn merkle_path_work<Base, Ext, Challenger>(config: &P3WhirConfig<Ext, Base, Challenger>) -> u128
where
    Base: TwoAdicField,
    Ext: ExtensionField<Base> + Field + TwoAdicField,
    Challenger: FieldChallenger<Base> + GrindingChallenger<Witness = Base>,
{
    config
        .round_parameters
        .iter()
        .map(|round| {
            actual_query_count(round.num_queries, round.domain_size, round.folding_factor)
                .saturating_mul(path_depth(round.domain_size, round.folding_factor))
        })
        .fold(0, u128::saturating_add)
        .saturating_add(final_query_count(config).saturating_mul(final_path_depth(config)))
}

fn row_work<Base, Ext, Challenger>(config: &P3WhirConfig<Ext, Base, Challenger>) -> u128
where
    Base: TwoAdicField,
    Ext: ExtensionField<Base> + Field + TwoAdicField,
    Challenger: FieldChallenger<Base> + GrindingChallenger<Witness = Base>,
{
    let final_round = final_round_estimate(config);
    config
        .round_parameters
        .iter()
        .enumerate()
        .map(|(round_index, round)| {
            actual_query_count(round.num_queries, round.domain_size, round.folding_factor)
                .saturating_mul(row_width(round.folding_factor))
                .saturating_mul(query_payload_degree::<Base, Ext>(round_index))
        })
        .fold(0, u128::saturating_add)
        .saturating_add(
            final_query_count(config)
                .saturating_mul(row_width(final_round.folding_factor))
                .saturating_mul(final_payload_degree::<Base, Ext, Challenger>(config)),
        )
}

#[derive(Debug, Clone, Copy)]
struct VerifierWorkEstimate {
    merkle_hashes: u128,
    leaf_field_elements: u128,
    row_field_elements: u128,
    extension_operations: u128,
    pow_checks: u128,
}

fn verifier_work<Base, Ext, Challenger>(
    config: &P3WhirConfig<Ext, Base, Challenger>,
    terminal_pow_bits: usize,
) -> VerifierWorkEstimate
where
    Base: TwoAdicField,
    Ext: ExtensionField<Base> + Field + TwoAdicField,
    Challenger: FieldChallenger<Base> + GrindingChallenger<Witness = Base>,
{
    let final_round = final_round_estimate(config);
    let final_poly_field_elements =
        row_width(config.final_sumcheck_rounds).saturating_mul(Ext::DIMENSION as u128);
    let queried_row_field_elements = row_work::<Base, Ext, Challenger>(config);

    let sumcheck_rounds = (0..=config.n_rounds())
        .map(|round| config.round_folding_factor(round) as u128)
        .fold(config.final_sumcheck_rounds as u128, u128::saturating_add);
    let constraint_operations = config
        .round_parameters
        .iter()
        .map(|round| {
            let queries =
                actual_query_count(round.num_queries, round.domain_size, round.folding_factor);
            (queries.saturating_add(round.ood_samples as u128))
                .saturating_mul(round.num_variables as u128)
        })
        .fold(0, u128::saturating_add)
        .saturating_add(
            final_query_count(config).saturating_mul(final_round.folding_factor as u128),
        );

    VerifierWorkEstimate {
        merkle_hashes: verifier_merkle_hashes(config),
        leaf_field_elements: queried_row_field_elements,
        row_field_elements: queried_row_field_elements.saturating_add(final_poly_field_elements),
        // Each sumcheck round reconstructs and evaluates a low-degree univariate;
        // constraint materialization is linear in the number of claims and variables.
        extension_operations: sumcheck_rounds
            .saturating_mul(6)
            .saturating_add(constraint_operations.saturating_mul(2)),
        pow_checks: verifier_pow_checks(config, terminal_pow_bits),
    }
}

fn verifier_merkle_hashes<Base, Ext, Challenger>(
    config: &P3WhirConfig<Ext, Base, Challenger>,
) -> u128
where
    Base: TwoAdicField,
    Ext: ExtensionField<Base> + Field + TwoAdicField,
    Challenger: FieldChallenger<Base> + GrindingChallenger<Witness = Base>,
{
    config
        .round_parameters
        .iter()
        .map(|round| {
            expected_merkle_multiproof_hashes(
                folded_row_count(round.domain_size, round.folding_factor),
                actual_query_count(round.num_queries, round.domain_size, round.folding_factor)
                    as usize,
            )
        })
        .fold(0, u128::saturating_add)
        .saturating_add(expected_merkle_multiproof_hashes(
            folded_row_count(
                final_round_estimate(config).domain_size,
                final_round_estimate(config).folding_factor,
            ),
            final_query_count(config) as usize,
        ))
}

fn zk_merkle_multiproof_hashes<Base, Ext, Challenger>(
    config: &P3WhirConfig<Ext, Base, Challenger>,
    mask_queries: usize,
    sumcheck_mask_domain: usize,
    switch_mask_domains: &[usize],
    application_mask: ApplicationMaskEstimate,
    base_mask_width: u128,
) -> u128
where
    Base: TwoAdicField,
    Ext: ExtensionField<Base> + Field + TwoAdicField,
    Challenger: FieldChallenger<Base> + GrindingChallenger<Witness = Base>,
{
    let switch_masks = switch_mask_domains
        .iter()
        .map(|domain| expected_merkle_multiproof_hashes(*domain, mask_queries))
        .fold(0, u128::saturating_add);
    let sumcheck_masks = expected_merkle_multiproof_hashes(sumcheck_mask_domain, mask_queries)
        .saturating_mul((config.n_rounds() + 1) as u128);
    let carried_masks = expected_merkle_multiproof_hashes(sumcheck_mask_domain, mask_queries)
        .saturating_mul(base_mask_width);
    let application_masks =
        expected_merkle_multiproof_hashes(application_mask.domain_size, mask_queries)
            .saturating_mul(2);
    switch_masks
        .saturating_add(sumcheck_masks)
        .saturating_add(carried_masks)
        .saturating_add(application_masks)
}

fn expected_merkle_multiproof_hashes(leaves: usize, queries: usize) -> u128 {
    if leaves <= 1 || queries == 0 {
        return 0;
    }
    let queries = queries.min(leaves);
    let mut block_size = 2usize;
    let mut hashes = 0.0;
    while block_size <= leaves {
        let blocks = leaves / block_size;
        let mut none_probability = 1.0;
        for selected in 0..queries {
            let remaining = leaves - selected;
            let outside = leaves.saturating_sub(block_size).saturating_sub(selected);
            if outside == 0 {
                none_probability = 0.0;
                break;
            }
            none_probability *= outside as f64 / remaining as f64;
        }
        hashes += blocks as f64 * (1.0 - none_probability);
        block_size = block_size.saturating_mul(2);
        if block_size == 0 {
            break;
        }
    }
    hashes.round() as u128
}

fn verifier_pow_checks<Base, Ext, Challenger>(
    config: &P3WhirConfig<Ext, Base, Challenger>,
    terminal_pow_bits: usize,
) -> u128
where
    Base: TwoAdicField,
    Ext: ExtensionField<Base> + Field + TwoAdicField,
    Challenger: FieldChallenger<Base> + GrindingChallenger<Witness = Base>,
{
    let mut checks = 0u128;
    if config.starting_folding_pow_bits > 0 {
        checks = checks.saturating_add(config.round_folding_factor(0) as u128);
    }
    for (round_index, round) in config.round_parameters.iter().enumerate() {
        if round.pow_bits > 0 {
            checks = checks.saturating_add(1);
        }
        if round.folding_pow_bits > 0 {
            checks = checks.saturating_add(config.round_folding_factor(round_index + 1) as u128);
        }
    }
    if terminal_pow_bits > 0 {
        checks = checks.saturating_add(1);
    }
    if config.final_folding_pow_bits > 0 {
        checks = checks.saturating_add(config.final_sumcheck_rounds as u128);
    }
    checks
}

fn proof_size_bytes_estimate<Base, Ext, Challenger>(
    config: &P3WhirConfig<Ext, Base, Challenger>,
) -> u128
where
    Base: TwoAdicField,
    Ext: ExtensionField<Base> + Field + TwoAdicField,
    Challenger: FieldChallenger<Base> + GrindingChallenger<Witness = Base>,
{
    let query_bytes = config
        .round_parameters
        .iter()
        .enumerate()
        .map(|(round_index, round)| {
            let leaf_bytes = row_width(round.folding_factor)
                .saturating_mul(query_payload_degree::<Base, Ext>(round_index))
                .saturating_mul(FIELD_BYTES);
            let path_bytes = path_depth(round.domain_size, round.folding_factor)
                .saturating_mul(POSEIDON_DIGEST_BYTES);
            actual_query_count(round.num_queries, round.domain_size, round.folding_factor)
                .saturating_mul(leaf_bytes.saturating_add(path_bytes))
        })
        .fold(0, u128::saturating_add);
    let final_round = final_round_estimate(config);
    let final_poly_bytes = row_width(config.final_sumcheck_rounds)
        .saturating_mul(Ext::DIMENSION as u128)
        .saturating_mul(FIELD_BYTES);
    let final_leaf_bytes = row_width(final_round.folding_factor)
        .saturating_mul(final_payload_degree::<Base, Ext, Challenger>(config))
        .saturating_mul(FIELD_BYTES);
    let final_path_bytes = path_depth(final_round.domain_size, final_round.folding_factor)
        .saturating_mul(POSEIDON_DIGEST_BYTES);
    let commitment_bytes =
        (config.round_parameters.len() as u128 + 1).saturating_mul(POSEIDON_DIGEST_BYTES);
    query_bytes
        .saturating_add(
            final_query_count(config)
                .saturating_mul(final_leaf_bytes.saturating_add(final_path_bytes)),
        )
        .saturating_add(final_poly_bytes)
        .saturating_add(commitment_bytes)
}

fn sumcheck_work<Base, Ext, Challenger>(config: &P3WhirConfig<Ext, Base, Challenger>) -> u128
where
    Base: Field,
    Ext: ExtensionField<Base> + Field,
{
    let initial = (config.num_variables as u128) * (1u128 << config.num_variables);
    let whir_rounds = config
        .round_parameters
        .iter()
        .map(|round| (round.folding_factor as u128) * (1u128 << round.num_variables))
        .fold(0, u128::saturating_add);
    initial.saturating_add(whir_rounds).saturating_add(
        config.final_sumcheck_rounds as u128 * (1u128 << config.final_sumcheck_rounds),
    )
}

fn row_width(folding_factor: usize) -> u128 {
    1u128
        .checked_shl(folding_factor as u32)
        .unwrap_or(u128::MAX)
}

fn query_payload_degree<Base, Ext>(round_index: usize) -> u128
where
    Base: Field,
    Ext: ExtensionField<Base>,
{
    if round_index == 0 {
        1
    } else {
        Ext::DIMENSION as u128
    }
}

fn final_payload_degree<Base, Ext, Challenger>(config: &P3WhirConfig<Ext, Base, Challenger>) -> u128
where
    Base: TwoAdicField,
    Ext: ExtensionField<Base> + Field + TwoAdicField,
    Challenger: FieldChallenger<Base> + GrindingChallenger<Witness = Base>,
{
    if config.n_rounds() == 0 {
        1
    } else {
        Ext::DIMENSION as u128
    }
}

fn final_query_count<Base, Ext, Challenger>(config: &P3WhirConfig<Ext, Base, Challenger>) -> u128
where
    Base: TwoAdicField,
    Ext: ExtensionField<Base> + Field + TwoAdicField,
    Challenger: FieldChallenger<Base> + GrindingChallenger<Witness = Base>,
{
    let final_round = final_round_estimate(config);
    actual_query_count(
        config.final_queries,
        final_round.domain_size,
        final_round.folding_factor,
    )
}

fn actual_query_count(num_queries: usize, domain_size: usize, folding_factor: usize) -> u128 {
    num_queries.min(folded_row_count(domain_size, folding_factor)) as u128
}

fn final_path_depth<Base, Ext, Challenger>(config: &P3WhirConfig<Ext, Base, Challenger>) -> u128
where
    Base: TwoAdicField,
    Ext: ExtensionField<Base> + Field + TwoAdicField,
    Challenger: FieldChallenger<Base> + GrindingChallenger<Witness = Base>,
{
    let final_round = final_round_estimate(config);
    path_depth(final_round.domain_size, final_round.folding_factor)
}

struct FinalRoundEstimate {
    domain_size: usize,
    folding_factor: usize,
}

fn final_round_estimate<Base, Ext, Challenger>(
    config: &P3WhirConfig<Ext, Base, Challenger>,
) -> FinalRoundEstimate
where
    Base: TwoAdicField,
    Ext: ExtensionField<Base> + Field + TwoAdicField,
    Challenger: FieldChallenger<Base> + GrindingChallenger<Witness = Base>,
{
    if config.round_parameters.is_empty() {
        FinalRoundEstimate {
            domain_size: config.starting_domain_size(),
            folding_factor: config.round_folding_factor(0),
        }
    } else {
        let last_round = config.n_rounds() - 1;
        let last = &config.round_parameters[last_round];
        FinalRoundEstimate {
            domain_size: last.domain_size >> config.rs_reduction_factor(last_round),
            folding_factor: config.round_folding_factor(config.n_rounds()),
        }
    }
}

fn path_depth(domain_size: usize, folding_factor: usize) -> u128 {
    log2_usize(folded_row_count(domain_size, folding_factor))
}

fn folded_row_count(domain_size: usize, folding_factor: usize) -> usize {
    domain_size >> folding_factor
}

fn log2_usize(value: usize) -> u128 {
    if value <= 1 {
        0
    } else {
        value.ilog2() as u128
    }
}

fn schedules(args: &Args) -> Vec<WhirFoldingSchedule> {
    let mut out = Vec::new();
    for factor in 1..=args.k_max.min(args.num_variables) {
        out.push(WhirFoldingSchedule::Constant(factor));
    }
    for first in 1..=args.k_max.min(args.num_variables) {
        for rest in 1..=args.k_max.min(args.num_variables) {
            if first != rest {
                out.push(WhirFoldingSchedule::ConstantFromSecondRound { first, rest });
            }
        }
    }
    out.extend(per_round_schedules(args));
    out
}

fn per_round_schedules(args: &Args) -> Vec<WhirFoldingSchedule> {
    #[derive(Clone)]
    struct Partial {
        factors: Vec<usize>,
        remaining: usize,
        cost: u128,
    }

    let mut partials = vec![Partial {
        factors: Vec::new(),
        remaining: args.num_variables,
        cost: 0,
    }];
    let mut done = Vec::new();
    while !partials.is_empty() {
        let mut next = Vec::new();
        for partial in partials {
            let max_factor = args.k_max.min(partial.remaining);
            for factor in 1..=max_factor {
                let remaining = partial.remaining.saturating_sub(factor);
                let mut factors = partial.factors.clone();
                factors.push(factor);
                let cost = partial.cost.saturating_add(1u128 << factor);
                if remaining <= args.final_sumcheck_max_variables {
                    done.push(Partial {
                        factors,
                        remaining,
                        cost,
                    });
                } else {
                    next.push(Partial {
                        factors,
                        remaining,
                        cost,
                    });
                }
            }
        }
        next.sort_by_key(|partial| (partial.remaining, partial.cost));
        next.truncate(args.beam_width);
        partials = next;
    }
    done.sort_by_key(|partial| (partial.cost, partial.remaining, partial.factors.len()));
    done.truncate(args.beam_width);
    done.into_iter()
        .map(|partial| WhirFoldingSchedule::PerRound(partial.factors))
        .collect()
}

fn derived_round_log_inv_rates(
    num_variables: usize,
    schedule: &WhirFoldingSchedule,
    starting_log_inv_rate: usize,
    rs_domain_initial_reduction_factor: usize,
) -> Result<Vec<usize>, String> {
    let folding = map_schedule(schedule);
    let (num_rounds, _) = folding
        .compute_number_of_rounds(num_variables)
        .map_err(|reason| format!("round count derivation failed: {reason}"))?;
    let mut rates = Vec::with_capacity(num_rounds);
    let mut rate = starting_log_inv_rate;
    for round in 0..num_rounds {
        let reduction = if round == 0 {
            rs_domain_initial_reduction_factor
        } else {
            1
        };
        rate = rate
            .checked_add(folding.at_round(round))
            .ok_or_else(|| format!("rate overflow at round {round}"))?
            .checked_sub(reduction)
            .ok_or_else(|| format!("rate underflow at round {round}"))?;
        rates.push(rate);
    }
    Ok(rates)
}

fn offset_round_log_inv_rates(rates: &[usize], offset: usize) -> Option<Vec<usize>> {
    if offset == 0 {
        return Some(Vec::new());
    }
    rates.iter().map(|rate| rate.checked_add(offset)).collect()
}

fn map_schedule(schedule: &WhirFoldingSchedule) -> FoldingFactor {
    match schedule {
        WhirFoldingSchedule::Constant(factor) => FoldingFactor::Constant(*factor),
        WhirFoldingSchedule::ConstantFromSecondRound { first, rest } => {
            FoldingFactor::ConstantFromSecondRound(*first, *rest)
        }
        WhirFoldingSchedule::PerRound(factors) => FoldingFactor::PerRound(factors.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_field::BasedVectorSpace;
    use spartan_whir::{
        recommended_quintic_zk_whir_params, setup_poseidon, setup_poseidon_zk, InvalidConfigReason,
        R1csShape, SecurityBoundComponent, SparseMatrix, SpartanWhirError, MIN_SECURITY_BITS,
    };

    fn test_config() -> P3WhirConfig<OcticBinExtension, F, PoseidonChallenger> {
        let params = WhirParams {
            pow_bits: 0,
            folding_factor: 4,
            starting_log_inv_rate: 1,
            rs_domain_initial_reduction_factor: 1,
            folding_schedule: Some(WhirFoldingSchedule::Constant(4)),
            round_log_inv_rates: vec![4, 7],
        };
        let protocol_params = ProtocolParameters {
            starting_log_inv_rate: params.starting_log_inv_rate,
            round_log_inv_rates: params.round_log_inv_rates.clone(),
            folding_factor: map_schedule(&params.effective_folding_schedule()),
            soundness_type: P3SecurityAssumption::JohnsonBound,
            security_level: DEFAULT_SECURITY_BITS,
            pow_bits: params.pow_bits as usize,
        };
        P3WhirConfig::<OcticBinExtension, F, PoseidonChallenger>::new(18, protocol_params)
            .expect("test WHIR config is valid")
    }

    fn direct_security_test_args(proof_mode: ProofMode, security_bits: usize) -> Args {
        Args {
            field: FieldProfile::KoalaBear,
            extension_filter: ExtensionFilter::Quartic,
            num_variables: 25,
            num_outer_rounds: 19,
            security_bits,
            merkle_security_bits: security_bits,
            component_security_bits: None,
            component_merkle_security_bits: None,
            k_max: DEFAULT_K_MAX,
            starting_log_inv_rate_max: DEFAULT_LIR_MAX,
            round_log_inv_rate_offset_max: 0,
            max_pow_bits: 64,
            final_sumcheck_max_variables: FINAL_SUMCHECK_MAX_VARIABLES,
            beam_width: DEFAULT_BEAM_WIDTH,
            include_invalid: true,
            proof_mode,
            zk_ell: 3,
            zk_mask_log_inv_rate: 3,
        }
    }

    fn direct_security_test_whir_params() -> WhirParams {
        WhirParams {
            pow_bits: 64,
            folding_factor: 8,
            starting_log_inv_rate: 1,
            rs_domain_initial_reduction_factor: 8,
            folding_schedule: Some(WhirFoldingSchedule::ConstantFromSecondRound {
                first: 8,
                rest: 4,
            }),
            round_log_inv_rates: Vec::new(),
        }
    }

    fn empty_shape(num_variables: usize, num_outer_rounds: usize) -> R1csShape<F> {
        let num_vars = 1usize << num_variables;
        let num_cons = 1usize << num_outer_rounds;
        let matrix = SparseMatrix {
            num_rows: num_cons,
            num_cols: num_vars + 1,
            entries: Vec::new(),
        };
        R1csShape {
            num_cons,
            num_vars,
            num_io: 0,
            a: matrix.clone(),
            b: matrix.clone(),
            c: matrix,
        }
    }

    fn assert_candidate_composed_rejection(
        args: &Args,
        whir_params: WhirParams,
        attainable_bits: usize,
    ) {
        let mut candidates = Vec::new();
        push_candidates(args, &mut candidates, whir_params, 0);

        assert_eq!(candidates.len(), 1);
        let candidate = &candidates[0];
        assert!(!candidate.valid);
        let expected_reason = format!(
            "DirectSparse requested security {} exceeds composed-security limit {attainable_bits}",
            args.security_bits.min(args.merkle_security_bits)
        );
        assert_eq!(
            candidate.rejection_reason.as_deref(),
            Some(expected_reason.as_str())
        );
        assert!(candidate.setup_config.is_none());
    }

    #[test]
    fn no_zk_candidate_matches_direct_sparse_setup_security_rejection() {
        let args = direct_security_test_args(ProofMode::NoZk, 116);
        let whir_params = direct_security_test_whir_params();
        assert_candidate_composed_rejection(&args, whir_params.clone(), 115);

        let setup_error = match setup_poseidon::<QuarticBinExtension>(
            empty_shape(args.num_variables, args.num_outer_rounds),
            SpartanSnarkConfig {
                matrix_closing: MatrixClosingMode::DirectSparse,
                security: requested_security_config(&args).expect("requested security is valid"),
                whir_params,
                spark_whir_params: None,
            },
        ) {
            Ok(_) => panic!("DirectSparse setup unexpectedly accepted the quartic config"),
            Err(error) => error,
        };
        assert_eq!(
            setup_error,
            SpartanWhirError::InvalidConfig(InvalidConfigReason::ComposedSecurityUnavailable {
                requested_bits: 116,
                attainable_bits: 115,
                dominant_component: SecurityBoundComponent::ExtensionField,
            })
        );
    }

    #[test]
    fn full_zk_candidate_matches_direct_sparse_setup_security_rejection() {
        let args = direct_security_test_args(ProofMode::FullZk, 115);
        let whir_params = direct_security_test_whir_params();
        assert_candidate_composed_rejection(&args, whir_params.clone(), 113);

        let setup_error = match setup_poseidon_zk::<QuarticBinExtension>(
            empty_shape(args.num_variables, args.num_outer_rounds),
            PoseidonZkSetupConfig {
                matrix_closing: MatrixClosingMode::DirectSparse,
                security: requested_security_config(&args).expect("requested security is valid"),
                whir_params,
                spark_whir_params: None,
                ell_zk: args.zk_ell,
                mask_log_inv_rate: args.zk_mask_log_inv_rate,
            },
        ) {
            Ok(_) => panic!("full-ZK DirectSparse setup unexpectedly accepted the quartic config"),
            Err(error) => error,
        };
        assert_eq!(
            setup_error,
            SpartanWhirError::InvalidConfig(InvalidConfigReason::ComposedSecurityUnavailable {
                requested_bits: 115,
                attainable_bits: 113,
                dominant_component: SecurityBoundComponent::ExtensionField,
            })
        );
    }

    #[test]
    fn full_zk_candidate_counts_every_commitment_binding_event() {
        let args = direct_security_test_args(ProofMode::FullZk, 116);
        assert_eq!(
            direct_commitment_binding_events(&args, 5).expect("commitment count fits"),
            34
        );
        assert_eq!(
            three_way_budget_slack(34).expect("slack calculation fits"),
            7
        );
    }

    #[test]
    fn requested_security_config_enforces_supported_ranges() {
        let mut args = direct_security_test_args(ProofMode::NoZk, MIN_SECURITY_BITS as usize);
        args.merkle_security_bits = MAX_SECURITY_BITS as usize;
        assert!(requested_security_config(&args).is_ok());

        args.security_bits = MIN_SECURITY_BITS as usize - 1;
        assert!(requested_security_config(&args).is_err());
        args.security_bits = MAX_SECURITY_BITS as usize + 1;
        assert!(requested_security_config(&args).is_err());

        args.security_bits = MIN_SECURITY_BITS as usize;
        args.merkle_security_bits = MIN_SECURITY_BITS as usize - 1;
        assert!(requested_security_config(&args).is_err());
        args.merkle_security_bits = MAX_SECURITY_BITS as usize + 1;
        assert!(requested_security_config(&args).is_err());

        if let Some(too_wide) = (u32::MAX as usize).checked_add(1) {
            args.security_bits = too_wide;
            let error = requested_security_config(&args).unwrap_err();
            assert!(error.contains("does not fit in u32"));
        }
    }

    #[test]
    fn full_zk_estimates_use_the_shared_terminal_budget() {
        let whir = recommended_quintic_zk_whir_params(20);
        let config = P3WhirConfig::<KoalaBearQuinticExtension, F, PoseidonChallenger>::new(
            20,
            ProtocolParameters {
                starting_log_inv_rate: whir.starting_log_inv_rate,
                round_log_inv_rates: whir.round_log_inv_rates.clone(),
                folding_factor: map_schedule(&whir.effective_folding_schedule()),
                soundness_type: P3SecurityAssumption::JohnsonBound,
                security_level: 120,
                pow_bits: whir.pow_bits as usize,
            },
        )
        .expect("selected full-ZK WHIR config is valid");

        assert_eq!(hiding_terminal_budget(&config), (125, 4));
        assert_eq!(zk_oracle_randomness(&config).last(), Some(&125));
    }

    #[test]
    fn round_rate_offset_zero_is_derived_and_positive_offsets_are_explicit() {
        let schedule = WhirFoldingSchedule::Constant(4);
        let derived =
            derived_round_log_inv_rates(14, &schedule, 1, 2).expect("round rates are derived");

        assert_eq!(derived, vec![3]);
        assert_eq!(offset_round_log_inv_rates(&derived, 0), Some(Vec::new()));
        assert_eq!(offset_round_log_inv_rates(&derived, 1), Some(vec![4]));

        let mut params = WhirParams {
            pow_bits: 0,
            folding_factor: 4,
            starting_log_inv_rate: 1,
            rs_domain_initial_reduction_factor: 2,
            folding_schedule: Some(schedule),
            round_log_inv_rates: Vec::new(),
        };
        let derived_label = format_whir_params_label("quintic", &params);
        let derived_protocol = protocol_parameters(14, &params, DEFAULT_SECURITY_BITS)
            .expect("derived protocol parameters are valid");
        assert_eq!(derived_protocol.round_log_inv_rates, vec![3]);
        assert!(params.round_log_inv_rates.is_empty());

        params.round_log_inv_rates = vec![4];
        let explicit_label = format_whir_params_label("quintic", &params);
        let explicit_protocol = protocol_parameters(14, &params, DEFAULT_SECURITY_BITS)
            .expect("explicit protocol parameters are valid");
        assert_eq!(explicit_protocol.round_log_inv_rates, vec![4]);
        assert!(derived_label.ends_with("_round_log_inv_rates_derived"));
        assert!(explicit_label.ends_with("_round_log_inv_rates_4"));
        assert_ne!(derived_label, explicit_label);
    }

    #[test]
    fn derived_rsv_rates_match_explicit_config_rounds_and_metrics() {
        let schedule = WhirFoldingSchedule::ConstantFromSecondRound { first: 8, rest: 4 };
        let derived_params = WhirParams {
            pow_bits: 9,
            folding_factor: 8,
            starting_log_inv_rate: 1,
            rs_domain_initial_reduction_factor: 8,
            folding_schedule: Some(schedule.clone()),
            round_log_inv_rates: Vec::new(),
        };
        let expected_rates =
            derived_round_log_inv_rates(25, &schedule, 1, 8).expect("round rates are derived");
        assert_eq!(expected_rates, vec![1, 4, 7]);

        let mut explicit_params = derived_params.clone();
        explicit_params.round_log_inv_rates = expected_rates.clone();
        let derived_config = P3WhirConfig::<KoalaBearQuinticExtension, F, PoseidonChallenger>::new(
            25,
            protocol_parameters(25, &derived_params, DEFAULT_SECURITY_BITS)
                .expect("derived protocol parameters are valid"),
        )
        .expect("derived WHIR config is valid");
        let explicit_config =
            P3WhirConfig::<KoalaBearQuinticExtension, F, PoseidonChallenger>::new(
                25,
                protocol_parameters(25, &explicit_params, DEFAULT_SECURITY_BITS)
                    .expect("explicit protocol parameters are valid"),
            )
            .expect("explicit WHIR config is valid");

        assert!(derived_params.round_log_inv_rates.is_empty());
        assert_eq!(derived_config.params.round_log_inv_rates, expected_rates);
        assert_eq!(
            derived_config.params.round_log_inv_rates,
            explicit_config.params.round_log_inv_rates
        );
        assert_eq!(
            derived_config.folding_schedule,
            explicit_config.folding_schedule
        );
        assert_eq!(
            (
                derived_config.commitment_ood_samples,
                derived_config.starting_folding_pow_bits,
                derived_config.final_queries,
                derived_config.final_pow_bits,
                derived_config.final_sumcheck_rounds,
                derived_config.final_folding_pow_bits,
            ),
            (
                explicit_config.commitment_ood_samples,
                explicit_config.starting_folding_pow_bits,
                explicit_config.final_queries,
                explicit_config.final_pow_bits,
                explicit_config.final_sumcheck_rounds,
                explicit_config.final_folding_pow_bits,
            )
        );
        assert_eq!(
            derived_config.round_parameters.len(),
            explicit_config.round_parameters.len()
        );
        for (derived, explicit) in derived_config
            .round_parameters
            .iter()
            .zip(&explicit_config.round_parameters)
        {
            assert_eq!(
                (
                    derived.pow_bits,
                    derived.folding_pow_bits,
                    derived.num_queries,
                    derived.ood_samples,
                    derived.num_variables,
                    derived.folding_factor,
                    derived.log_inv_rate,
                    derived.domain_size,
                    derived.folded_domain_gen,
                ),
                (
                    explicit.pow_bits,
                    explicit.folding_pow_bits,
                    explicit.num_queries,
                    explicit.ood_samples,
                    explicit.num_variables,
                    explicit.folding_factor,
                    explicit.log_inv_rate,
                    explicit.domain_size,
                    explicit.folded_domain_gen,
                )
            );
        }
        assert_eq!(
            (
                pow_work_units(&derived_config, derived_config.final_pow_bits),
                dft_work(&derived_config),
                merkle_work(&derived_config),
                merkle_path_work(&derived_config),
                row_work(&derived_config),
                sumcheck_work(&derived_config),
                proof_size_bytes_estimate(&derived_config),
            ),
            (
                pow_work_units(&explicit_config, explicit_config.final_pow_bits),
                dft_work(&explicit_config),
                merkle_work(&explicit_config),
                merkle_path_work(&explicit_config),
                row_work(&explicit_config),
                sumcheck_work(&explicit_config),
                proof_size_bytes_estimate(&explicit_config),
            )
        );

        let args = Args {
            field: FieldProfile::KoalaBear,
            extension_filter: ExtensionFilter::Quintic,
            num_variables: 25,
            num_outer_rounds: 25,
            security_bits: DEFAULT_SECURITY_BITS,
            merkle_security_bits: DEFAULT_SECURITY_BITS,
            component_security_bits: None,
            component_merkle_security_bits: None,
            k_max: DEFAULT_K_MAX,
            starting_log_inv_rate_max: DEFAULT_LIR_MAX,
            round_log_inv_rate_offset_max: 0,
            max_pow_bits: DEFAULT_MAX_POW_BITS,
            final_sumcheck_max_variables: FINAL_SUMCHECK_MAX_VARIABLES,
            beam_width: DEFAULT_BEAM_WIDTH,
            include_invalid: false,
            proof_mode: ProofMode::NoZk,
            zk_ell: 3,
            zk_mask_log_inv_rate: 3,
        };
        let mut candidates = Vec::new();
        push_candidates(&args, &mut candidates, derived_params, 0);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].round_log_inv_rate_offset, 0);
        assert!(candidates[0]
            .label
            .ends_with("_round_log_inv_rates_derived"));
        assert!(candidates[0].whir_params.round_log_inv_rates.is_empty());
        let setup: SpartanSnarkConfig = serde_json::from_value(
            candidates[0]
                .setup_config
                .clone()
                .expect("candidate has a setup config"),
        )
        .expect("candidate setup config deserializes");
        assert!(setup.whir_params.round_log_inv_rates.is_empty());
    }

    #[test]
    fn extension_filter_limits_backend_candidate_derivation() {
        let args = Args {
            field: FieldProfile::KoalaBear,
            extension_filter: ExtensionFilter::Quintic,
            num_variables: 18,
            num_outer_rounds: 17,
            security_bits: DEFAULT_SECURITY_BITS,
            merkle_security_bits: DEFAULT_SECURITY_BITS,
            component_security_bits: None,
            component_merkle_security_bits: None,
            k_max: DEFAULT_K_MAX,
            starting_log_inv_rate_max: DEFAULT_LIR_MAX,
            round_log_inv_rate_offset_max: 0,
            max_pow_bits: DEFAULT_MAX_POW_BITS,
            final_sumcheck_max_variables: FINAL_SUMCHECK_MAX_VARIABLES,
            beam_width: DEFAULT_BEAM_WIDTH,
            include_invalid: true,
            proof_mode: ProofMode::NoZk,
            zk_ell: 3,
            zk_mask_log_inv_rate: 3,
        };
        let params = WhirParams {
            pow_bits: 0,
            folding_factor: 4,
            starting_log_inv_rate: 1,
            rs_domain_initial_reduction_factor: 1,
            folding_schedule: Some(WhirFoldingSchedule::Constant(4)),
            round_log_inv_rates: vec![4, 7],
        };
        let mut candidates = Vec::new();

        push_candidates(&args, &mut candidates, params, 1);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].extension, "quintic");
        assert_eq!(candidates[0].round_log_inv_rate_offset, 1);
    }

    #[test]
    fn setup_config_is_mode_specific() {
        let mut args = Args {
            field: FieldProfile::KoalaBear,
            extension_filter: ExtensionFilter::All,
            num_variables: 20,
            num_outer_rounds: 20,
            security_bits: DEFAULT_SECURITY_BITS,
            merkle_security_bits: DEFAULT_SECURITY_BITS,
            component_security_bits: None,
            component_merkle_security_bits: None,
            k_max: DEFAULT_K_MAX,
            starting_log_inv_rate_max: DEFAULT_LIR_MAX,
            round_log_inv_rate_offset_max: 0,
            max_pow_bits: DEFAULT_MAX_POW_BITS,
            final_sumcheck_max_variables: FINAL_SUMCHECK_MAX_VARIABLES,
            beam_width: DEFAULT_BEAM_WIDTH,
            include_invalid: false,
            proof_mode: ProofMode::FullZk,
            zk_ell: 3,
            zk_mask_log_inv_rate: 1,
        };

        let config = setup_config(&args, WhirParams::default());

        let config: PoseidonZkSetupConfig =
            serde_json::from_value(config).expect("full-ZK setup config deserializes");
        assert_eq!(config.matrix_closing, MatrixClosingMode::DirectSparse);
        assert_eq!(config.whir_params, WhirParams::default());
        assert_eq!(config.ell_zk, 3);
        assert_eq!(config.mask_log_inv_rate, 1);

        args.proof_mode = ProofMode::NoZk;
        let config: SpartanSnarkConfig =
            serde_json::from_value(setup_config(&args, WhirParams::default()))
                .expect("no-ZK setup config deserializes");
        assert_eq!(config.matrix_closing, MatrixClosingMode::DirectSparse);
        assert_eq!(config.whir_params, WhirParams::default());
    }

    #[test]
    fn direct_component_targets_are_derived_from_end_to_end_security() {
        let mut args = Args {
            field: FieldProfile::KoalaBear,
            extension_filter: ExtensionFilter::All,
            num_variables: 20,
            num_outer_rounds: 20,
            security_bits: 116,
            merkle_security_bits: 116,
            component_security_bits: None,
            component_merkle_security_bits: None,
            k_max: DEFAULT_K_MAX,
            starting_log_inv_rate_max: DEFAULT_LIR_MAX,
            round_log_inv_rate_offset_max: 0,
            max_pow_bits: DEFAULT_MAX_POW_BITS,
            final_sumcheck_max_variables: FINAL_SUMCHECK_MAX_VARIABLES,
            beam_width: DEFAULT_BEAM_WIDTH,
            include_invalid: false,
            proof_mode: ProofMode::NoZk,
            zk_ell: 3,
            zk_mask_log_inv_rate: 3,
        };

        assert_eq!(whir_component_security_bits(&args), Ok(118));
        assert_eq!(direct_merkle_component_security_bits(&args, 2), Ok(120));

        args.proof_mode = ProofMode::FullZk;
        assert_eq!(whir_component_security_bits(&args), Ok(118));
        assert_eq!(direct_merkle_component_security_bits(&args, 2), Ok(122));

        args.component_security_bits = Some(120);
        args.component_merkle_security_bits = Some(123);
        assert_eq!(whir_component_security_bits(&args), Ok(120));
        assert_eq!(direct_merkle_component_security_bits(&args, 2), Ok(123));
    }

    #[test]
    fn cost_model_units_match_constant_schedule_layout() {
        let config = test_config();
        assert_eq!(config.n_rounds(), 2);
        assert_eq!(config.final_sumcheck_rounds, 6);
        assert_eq!(config.round_parameters[0].domain_size, 1 << 19);
        assert_eq!(config.round_parameters[1].domain_size, 1 << 18);
        assert_eq!(config.rs_reduction_factor(0), 1);
        assert_eq!(config.rs_reduction_factor(1), 1);

        let first_queries = actual_query_count(
            config.round_parameters[0].num_queries,
            config.round_parameters[0].domain_size,
            config.round_parameters[0].folding_factor,
        );
        let second_queries = actual_query_count(
            config.round_parameters[1].num_queries,
            config.round_parameters[1].domain_size,
            config.round_parameters[1].folding_factor,
        );
        let final_round = config.final_round_config();
        let final_queries = final_query_count(&config);

        assert_eq!(
            merkle_work(&config),
            (1u128 << 19) + 8 * (1u128 << 18) + 8 * (1u128 << 17)
        );
        assert_eq!(
            row_work(&config),
            first_queries * 16 + second_queries * 16 * 8 + final_queries * 16 * 8
        );
        assert_eq!(
            merkle_path_work(&config),
            first_queries * 15
                + second_queries * 14
                + final_queries * path_depth(final_round.domain_size, final_round.folding_factor)
        );
        assert_eq!(
            proof_size_bytes_estimate(&config),
            first_queries * (16 * FIELD_BYTES + 15 * POSEIDON_DIGEST_BYTES)
                + second_queries * (16 * 8 * FIELD_BYTES + 14 * POSEIDON_DIGEST_BYTES)
                + final_queries
                    * (16 * 8 * FIELD_BYTES
                        + path_depth(final_round.domain_size, final_round.folding_factor)
                            * POSEIDON_DIGEST_BYTES)
                + (1u128 << 6) * 8 * FIELD_BYTES
                + 3 * POSEIDON_DIGEST_BYTES
        );
    }

    #[test]
    fn final_round_estimate_matches_backend_final_round_config() {
        let config = test_config();
        let estimate = final_round_estimate(&config);
        let backend = config.final_round_config();

        assert_eq!(estimate.domain_size, backend.domain_size);
        assert_eq!(estimate.folding_factor, backend.folding_factor);
    }

    #[test]
    fn actual_query_count_caps_at_folded_row_count() {
        assert_eq!(actual_query_count(100, 32, 3), 4);
        assert_eq!(path_depth(32, 3), 2);
    }

    #[test]
    fn verifier_merkle_work_counts_reconstructed_multiproof_nodes() {
        assert_eq!(expected_merkle_multiproof_hashes(8, 0), 0);
        assert_eq!(expected_merkle_multiproof_hashes(8, 1), 3);
        assert_eq!(expected_merkle_multiproof_hashes(8, 8), 7);
    }

    #[test]
    fn application_mask_rejects_mismatched_domains() {
        let args = Args {
            field: FieldProfile::KoalaBear,
            extension_filter: ExtensionFilter::All,
            num_variables: 20,
            num_outer_rounds: 19,
            security_bits: DEFAULT_SECURITY_BITS,
            merkle_security_bits: DEFAULT_SECURITY_BITS,
            component_security_bits: None,
            component_merkle_security_bits: None,
            k_max: DEFAULT_K_MAX,
            starting_log_inv_rate_max: DEFAULT_LIR_MAX,
            round_log_inv_rate_offset_max: 0,
            max_pow_bits: DEFAULT_MAX_POW_BITS,
            final_sumcheck_max_variables: FINAL_SUMCHECK_MAX_VARIABLES,
            beam_width: DEFAULT_BEAM_WIDTH,
            include_invalid: false,
            proof_mode: ProofMode::FullZk,
            zk_ell: 3,
            zk_mask_log_inv_rate: 2,
        };

        let error = application_mask_estimate::<OcticBinExtension>(&args, 25).unwrap_err();
        assert!(error.contains("IncompatibleApplicationMaskDomains"));
    }

    #[test]
    fn application_mask_rejects_domains_beyond_two_adicity() {
        let mut args = Args {
            field: FieldProfile::KoalaBear,
            extension_filter: ExtensionFilter::All,
            num_variables: 20,
            num_outer_rounds: 19,
            security_bits: DEFAULT_SECURITY_BITS,
            merkle_security_bits: DEFAULT_SECURITY_BITS,
            component_security_bits: None,
            component_merkle_security_bits: None,
            k_max: DEFAULT_K_MAX,
            starting_log_inv_rate_max: DEFAULT_LIR_MAX,
            round_log_inv_rate_offset_max: 0,
            max_pow_bits: DEFAULT_MAX_POW_BITS,
            final_sumcheck_max_variables: FINAL_SUMCHECK_MAX_VARIABLES,
            beam_width: DEFAULT_BEAM_WIDTH,
            include_invalid: false,
            proof_mode: ProofMode::FullZk,
            zk_ell: 3,
            zk_mask_log_inv_rate: OcticBinExtension::TWO_ADICITY,
        };

        let error = application_mask_estimate::<OcticBinExtension>(&args, 1).unwrap_err();
        assert!(error.contains("exceeds extension-field two-adicity"));

        args.num_outer_rounds = usize::MAX;
        args.zk_mask_log_inv_rate = 1;
        let error = application_mask_estimate::<OcticBinExtension>(&args, 8).unwrap_err();
        assert!(error.contains("width overflows"));
    }

    #[test]
    fn application_mask_cost_and_size_terms_are_accounted_for() {
        let args = Args {
            field: FieldProfile::KoalaBear,
            extension_filter: ExtensionFilter::All,
            num_variables: 18,
            num_outer_rounds: 17,
            security_bits: DEFAULT_SECURITY_BITS,
            merkle_security_bits: DEFAULT_SECURITY_BITS,
            component_security_bits: None,
            component_merkle_security_bits: None,
            k_max: DEFAULT_K_MAX,
            starting_log_inv_rate_max: DEFAULT_LIR_MAX,
            round_log_inv_rate_offset_max: 0,
            max_pow_bits: DEFAULT_MAX_POW_BITS,
            final_sumcheck_max_variables: FINAL_SUMCHECK_MAX_VARIABLES,
            beam_width: DEFAULT_BEAM_WIDTH,
            include_invalid: false,
            proof_mode: ProofMode::FullZk,
            zk_ell: 3,
            zk_mask_log_inv_rate: 3,
        };
        let config = test_config();
        let mask_queries = zk_mask_queries(&args, &config);
        let application = application_mask_estimate::<OcticBinExtension>(&args, mask_queries)
            .expect("application mask geometry is valid");
        let base = ZkBaseEstimates {
            dft_work: 0,
            merkle_work: 0,
            merkle_path_work: 0,
            row_work: 0,
            sumcheck_work: 0,
            verifier: VerifierWorkEstimate {
                merkle_hashes: 0,
                leaf_field_elements: 0,
                row_field_elements: 0,
                extension_operations: 0,
                pow_checks: 0,
            },
        };
        let estimate = zk_estimates(&args, &config, base);
        let application_rows = (application.domain_size as u128)
            * (application.width as u128)
            * 2
            * (<OcticBinExtension as BasedVectorSpace<F>>::DIMENSION as u128);

        assert!(estimate.dft_work >= application_rows);
        assert!(estimate.merkle_work >= application_rows);
        assert!(estimate.proof_size_bytes_estimate > proof_size_bytes_estimate(&config));
        assert_eq!(estimate.application_mask, application);
    }
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        field: FieldProfile::KoalaBear,
        extension_filter: ExtensionFilter::All,
        num_variables: 0,
        num_outer_rounds: 0,
        security_bits: DEFAULT_SECURITY_BITS,
        merkle_security_bits: DEFAULT_SECURITY_BITS,
        component_security_bits: None,
        component_merkle_security_bits: None,
        k_max: DEFAULT_K_MAX,
        starting_log_inv_rate_max: DEFAULT_LIR_MAX,
        round_log_inv_rate_offset_max: 0,
        max_pow_bits: DEFAULT_MAX_POW_BITS,
        final_sumcheck_max_variables: FINAL_SUMCHECK_MAX_VARIABLES,
        beam_width: DEFAULT_BEAM_WIDTH,
        include_invalid: false,
        proof_mode: ProofMode::NoZk,
        zk_ell: spartan_whir::DEFAULT_ZK_ELL,
        zk_mask_log_inv_rate: spartan_whir::DEFAULT_ZK_MASK_LOG_INV_RATE,
    };
    let mut iter = env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--field" => args.field = parse_field_profile(&mut iter, &arg)?,
            "--extension" => args.extension_filter = parse_extension_filter(&mut iter, &arg)?,
            "--num-variables" => args.num_variables = parse_next(&mut iter, &arg)?,
            "--num-outer-rounds" => args.num_outer_rounds = parse_next(&mut iter, &arg)?,
            "--security-bits" => args.security_bits = parse_next(&mut iter, &arg)?,
            "--merkle-security-bits" => args.merkle_security_bits = parse_next(&mut iter, &arg)?,
            "--component-security-bits" => {
                args.component_security_bits = Some(parse_next(&mut iter, &arg)?)
            }
            "--component-merkle-security-bits" => {
                args.component_merkle_security_bits = Some(parse_next(&mut iter, &arg)?)
            }
            "--k-max" => args.k_max = parse_next(&mut iter, &arg)?,
            "--starting-log-inv-rate-max" => {
                args.starting_log_inv_rate_max = parse_next(&mut iter, &arg)?
            }
            "--round-log-inv-rate-offset-max" => {
                args.round_log_inv_rate_offset_max = parse_next(&mut iter, &arg)?
            }
            "--max-pow-bits" => args.max_pow_bits = parse_next(&mut iter, &arg)?,
            "--final-sumcheck-max-variables" => {
                args.final_sumcheck_max_variables = parse_next(&mut iter, &arg)?
            }
            "--beam-width" => args.beam_width = parse_next(&mut iter, &arg)?,
            "--include-invalid" => args.include_invalid = true,
            "--proof-mode" => args.proof_mode = parse_proof_mode(&mut iter, &arg)?,
            "--zk-ell" => args.zk_ell = parse_next(&mut iter, &arg)?,
            "--zk-mask-log-inv-rate" => args.zk_mask_log_inv_rate = parse_next(&mut iter, &arg)?,
            "--help" | "-h" => {
                usage();
                process::exit(0);
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if args.num_variables == 0 {
        return Err("--num-variables is required".to_owned());
    }
    if args.num_outer_rounds == 0 {
        args.num_outer_rounds = args.num_variables;
    }
    if args.component_security_bits.is_some() != args.component_merkle_security_bits.is_some() {
        return Err(
            "--component-security-bits and --component-merkle-security-bits must be supplied together"
                .to_owned(),
        );
    }
    if let (Some(component), Some(component_merkle)) = (
        args.component_security_bits,
        args.component_merkle_security_bits,
    ) {
        let requested = args.security_bits.min(args.merkle_security_bits);
        if component < requested || component_merkle < requested {
            return Err(
                "component security targets must not be below the requested end-to-end target"
                    .to_owned(),
            );
        }
    }
    validate_requested_security_config(&args)?;
    Ok(args)
}

fn parse_next(iter: &mut impl Iterator<Item = String>, name: &str) -> Result<usize, String> {
    iter.next()
        .ok_or_else(|| format!("{name} requires a value"))?
        .parse()
        .map_err(|_| format!("{name} must be a non-negative integer"))
}

fn parse_field_profile(
    iter: &mut impl Iterator<Item = String>,
    name: &str,
) -> Result<FieldProfile, String> {
    match iter
        .next()
        .ok_or_else(|| format!("{name} requires a value"))?
        .as_str()
    {
        "koalabear" | "koala-bear" => Ok(FieldProfile::KoalaBear),
        "babybear" | "baby-bear" => Ok(FieldProfile::BabyBear),
        other => Err(format!(
            "{name} must be one of koalabear, koala-bear, babybear, baby-bear; got {other}"
        )),
    }
}

fn parse_extension_filter(
    iter: &mut impl Iterator<Item = String>,
    name: &str,
) -> Result<ExtensionFilter, String> {
    match iter
        .next()
        .ok_or_else(|| format!("{name} requires a value"))?
        .as_str()
    {
        "all" => Ok(ExtensionFilter::All),
        "quartic" => Ok(ExtensionFilter::Quartic),
        "quintic" => Ok(ExtensionFilter::Quintic),
        "octic" => Ok(ExtensionFilter::Octic),
        other => Err(format!(
            "{name} must be one of all, quartic, quintic, or octic; got {other}"
        )),
    }
}

fn parse_proof_mode(
    iter: &mut impl Iterator<Item = String>,
    name: &str,
) -> Result<ProofMode, String> {
    match iter
        .next()
        .ok_or_else(|| format!("{name} requires a value"))?
        .as_str()
    {
        "no-zk" => Ok(ProofMode::NoZk),
        "full-zk" => Ok(ProofMode::FullZk),
        other => Err(format!(
            "{name} must be one of no-zk or full-zk; got {other}"
        )),
    }
}

fn usage() {
    eprintln!(
        "usage: poseidon-schedule-candidates --num-variables N [--num-outer-rounds N] [--field koalabear|babybear] [--extension all|quartic|quintic|octic] [--security-bits 116] [--merkle-security-bits 116] [--component-security-bits N --component-merkle-security-bits N] [--max-pow-bits 22] [--round-log-inv-rate-offset-max N] [--proof-mode no-zk|full-zk] [--include-invalid]"
    );
}
