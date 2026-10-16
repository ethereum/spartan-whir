use std::{env, fs, io, path::Path};

use crate::{
    engine::F,
    pcs_config::{FreshMaskBatching, MaskPacking},
    MatrixClosingMode, Poseidon1ZkProvingKey, Poseidon1ZkVerifyingKey, PoseidonZkSetupConfig,
    QuinticExtension, R1csShape, SecurityConfig, SoundnessAssumption, SparkWhirParams,
    SpartanWhirError, WhirFoldingSchedule, WhirParams,
};

/// SPARK configuration for withdrawal proving and recursive fixture generation.
/// The application proof uses a 116-bit configured security target.
pub fn withdrawal_spark_config() -> PoseidonZkSetupConfig {
    let schedule = |pow_bits, first, rest, reduction| WhirParams {
        pow_bits,
        folding_factor: first,
        starting_log_inv_rate: 1,
        rs_domain_initial_reduction_factor: reduction,
        folding_schedule: Some(WhirFoldingSchedule::ConstantFromSecondRound { first, rest }),
        round_log_inv_rates: Vec::new(),
    };
    PoseidonZkSetupConfig {
        matrix_closing: MatrixClosingMode::Spark,
        security: SecurityConfig {
            security_level_bits: 116,
            merkle_security_bits: 116,
            soundness_assumption: SoundnessAssumption::JohnsonBound,
        },
        whir_params: schedule(7, 4, 3, 3),
        spark_whir_params: Some(SparkWhirParams {
            fixed_value: schedule(9, 5, 4, 5),
            fixed_audit: WhirParams {
                pow_bits: 6,
                folding_factor: 8,
                starting_log_inv_rate: 1,
                rs_domain_initial_reduction_factor: 8,
                folding_schedule: None,
                round_log_inv_rates: Vec::new(),
            },
            read: schedule(9, 5, 4, 5),
        }),
        ell_zk: 3,
        mask_log_inv_rate: 3,
    }
}

/// Load the same explicit setup parameters in fixture export and Criterion.
/// Setup validates the complete schedule and derives its transcript binding.
pub fn configured_withdrawal_spark_config(
    path: Option<&Path>,
) -> io::Result<PoseidonZkSetupConfig> {
    let environment_path = env::var_os("WITHDRAWAL_SPARK_CONFIG");
    match path.or_else(|| environment_path.as_deref().map(Path::new)) {
        Some(path) => parse_withdrawal_spark_config(&fs::read(path)?),
        None => Ok(withdrawal_spark_config()),
    }
}

fn parse_withdrawal_spark_config(bytes: &[u8]) -> io::Result<PoseidonZkSetupConfig> {
    let config: PoseidonZkSetupConfig = serde_json::from_slice(bytes)?;
    if config.matrix_closing != MatrixClosingMode::Spark
        || config.security.security_level_bits != 116
        || config.security.merkle_security_bits != 116
        || config.security.soundness_assumption != SoundnessAssumption::JohnsonBound
        || config.spark_whir_params.is_none()
    {
        return Err(io::Error::other(
            "withdrawal experiments require SPARK, explicit table schedules, and the 116-bit Johnson-bound security configuration",
        ));
    }
    Ok(config)
}

pub fn setup_withdrawal(
    shape: R1csShape<F>,
    config: PoseidonZkSetupConfig,
) -> Result<
    (
        Poseidon1ZkProvingKey<QuinticExtension>,
        Poseidon1ZkVerifyingKey<QuinticExtension>,
    ),
    SpartanWhirError,
> {
    Poseidon1ZkProvingKey::setup_with_mask_packing(
        shape,
        config,
        FreshMaskBatching::SameHeight,
        MaskPacking::AllFreeBasis,
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn configuration_preserves_explicit_schedule_representation() {
        use super::*;

        let mut config = withdrawal_spark_config();
        config.whir_params.starting_log_inv_rate = 3;
        config.whir_params.round_log_inv_rates = vec![4, 5];
        let parsed = parse_withdrawal_spark_config(&serde_json::to_vec(&config).unwrap())
            .expect("explicit parameters round trip");
        assert_eq!(parsed, config);
    }

    #[test]
    fn configuration_rejects_security_and_mode_changes() {
        use super::*;

        let baseline = withdrawal_spark_config();
        let mut weaker = baseline.clone();
        weaker.security.security_level_bits = 100;
        assert!(parse_withdrawal_spark_config(&serde_json::to_vec(&weaker).unwrap()).is_err());
        let mut direct = baseline;
        direct.matrix_closing = MatrixClosingMode::DirectSparse;
        assert!(parse_withdrawal_spark_config(&serde_json::to_vec(&direct).unwrap()).is_err());
    }
}
