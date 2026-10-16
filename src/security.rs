use num_bigint::BigUint;

use crate::SpartanWhirError;
use crate::{
    engine::ExtField,
    error::{InvalidConfigReason, SecurityBoundComponent},
    protocol::SparkPcsConfigs,
    spark::{spark_fixed_audit_is_embedded, SparkTableMetadata},
    whir_params::whir_folding_round_count,
    WhirPcsConfig,
};
use serde::{Deserialize, Serialize};

pub const MIN_SECURITY_BITS: u32 = 80;
pub const DEFAULT_SECURITY_BITS: u32 = 100;
/// Maximum target supported by the current eight-element KoalaBear Poseidon digest.
///
/// TODO: Raise this to 128 after widening the Poseidon Merkle digest and its
/// compression construction to provide at least 128 bits of collision security.
pub const MAX_SECURITY_BITS: u32 = 123;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SoundnessAssumption {
    UniqueDecoding,
    JohnsonBound,
    CapacityBound,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub security_level_bits: u32,
    pub merkle_security_bits: u32,
    pub soundness_assumption: SoundnessAssumption,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            security_level_bits: DEFAULT_SECURITY_BITS,
            merkle_security_bits: DEFAULT_SECURITY_BITS,
            soundness_assumption: SoundnessAssumption::CapacityBound,
        }
    }
}

impl SecurityConfig {
    pub fn validate(&self) -> Result<(), SpartanWhirError> {
        if self.security_level_bits < MIN_SECURITY_BITS {
            return Err(SpartanWhirError::SecurityBelowMinimum);
        }
        if self.security_level_bits > MAX_SECURITY_BITS {
            return Err(SpartanWhirError::SecurityAboveMaximum);
        }
        if self.merkle_security_bits < MIN_SECURITY_BITS {
            return Err(SpartanWhirError::MerkleSecurityBelowMinimum);
        }
        if self.merkle_security_bits > MAX_SECURITY_BITS {
            return Err(SpartanWhirError::MerkleSecurityAboveMaximum);
        }
        Ok(())
    }

    pub fn effective_security_bits(&self) -> u32 {
        core::cmp::min(self.security_level_bits, self.merkle_security_bits)
    }

    pub fn merkle_override_weaker_than_security(&self) -> bool {
        self.merkle_security_bits < self.security_level_bits
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SpartanSoundnessMode {
    NoZk,
    FullZk { inner_degree: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ComposedSecurityBudget {
    pub requested_bits: u32,
    pub attainable_bits: u32,
    pub algebraic_error_terms: usize,
    pub whir_argument_count: usize,
    pub commitment_binding_events: usize,
    pub whir_slack_bits: u32,
    pub merkle_slack_bits: u32,
    pub dominant_component: SecurityBoundComponent,
}

/// Derive the witness PCS target for a DirectSparse Spartan proof.
pub(crate) fn derive_direct_component_security<Ext>(
    requested: &SecurityConfig,
    witness_config: &WhirPcsConfig,
    num_outer_rounds: usize,
    num_inner_rounds: usize,
    mode: SpartanSoundnessMode,
) -> Result<(SecurityConfig, ComposedSecurityBudget), SpartanWhirError>
where
    Ext: ExtField,
{
    requested.validate()?;
    let algebraic_error_terms =
        spartan_algebraic_error_terms(num_outer_rounds, num_inner_rounds, mode)?;
    let witness_rounds =
        whir_folding_round_count(witness_config.num_variables, &witness_config.whir)?;
    let commitment_binding_events = witness_commitment_events(witness_rounds, mode)?;
    compose_security_budget::<Ext>(
        requested,
        algebraic_error_terms,
        1,
        commitment_binding_events,
    )
}

/// Derive per-argument targets for the WHIR arguments in a SPARK proof.
///
/// The composed error budget is divided equally between algebraic checks,
/// WHIR soundness, and Merkle binding. All arithmetic that counts protocol
/// events is checked integer arithmetic.
pub(crate) fn derive_spark_component_security<Ext>(
    requested: &SecurityConfig,
    metadata: &SparkTableMetadata,
    witness_config: &WhirPcsConfig,
    spark_configs: &SparkPcsConfigs,
    num_outer_rounds: usize,
    num_inner_rounds: usize,
    mode: SpartanSoundnessMode,
) -> Result<(SecurityConfig, ComposedSecurityBudget), SpartanWhirError>
where
    Ext: ExtField,
{
    requested.validate()?;
    let report = metadata.verifier_operation_report(Ext::DIMENSION)?;
    // Tuple compression and the two row/column grand-product identities have
    // degree proportional to the corresponding public table domains.
    let tuple_terms = metadata
        .value_domain_size
        .checked_mul(2)
        .and_then(|value| value.checked_add(metadata.row_memory_size))
        .and_then(|value| value.checked_add(metadata.col_memory_size))
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(composed_budget_overflow)?;
    let product_sumcheck_terms = report
        .total_product_sumcheck_rounds
        .checked_mul(3)
        .ok_or_else(composed_budget_overflow)?;
    let layer_terms = report
        .proof_ops_layers
        .checked_add(report.proof_mem_layers)
        .and_then(|layers| layers.checked_mul(2))
        .ok_or_else(composed_budget_overflow)?;
    let opening_batch_terms = Ext::DIMENSION
        .checked_mul(6)
        .and_then(|terms| terms.checked_add(8))
        .ok_or_else(composed_budget_overflow)?;
    let spartan_terms = spartan_algebraic_error_terms(num_outer_rounds, num_inner_rounds, mode)?;
    let algebraic_error_terms = spartan_terms
        .checked_add(tuple_terms)
        .and_then(|terms| terms.checked_add(product_sumcheck_terms))
        .and_then(|terms| terms.checked_add(layer_terms))
        .and_then(|terms| terms.checked_add(opening_batch_terms))
        .ok_or_else(composed_budget_overflow)?;

    let witness_rounds =
        whir_folding_round_count(witness_config.num_variables, &witness_config.whir)?;
    let fixed_value_rounds = whir_folding_round_count(
        spark_configs.fixed_value.num_variables,
        &spark_configs.fixed_value.whir,
    )?;
    let fixed_audit_embedded = spark_fixed_audit_is_embedded(
        metadata.value_domain_size,
        metadata.row_memory_size,
        metadata.col_memory_size,
    );
    let fixed_audit_rounds = if fixed_audit_embedded {
        None
    } else {
        Some(whir_folding_round_count(
            spark_configs.fixed_audit.num_variables,
            &spark_configs.fixed_audit.whir,
        )?)
    };
    let read_commitment_events = spark_configs.read.iter().try_fold(
        0usize,
        |events, config| -> Result<usize, SpartanWhirError> {
            let rounds = whir_folding_round_count(config.num_variables, &config.whir)?;
            events
                .checked_add(rounds.checked_add(1).ok_or_else(composed_budget_overflow)?)
                .ok_or_else(composed_budget_overflow)
        },
    )?;
    let mut table_commitment_events = fixed_value_rounds
        .checked_add(1)
        .and_then(|events| events.checked_add(read_commitment_events))
        .ok_or_else(composed_budget_overflow)?;
    if let Some(rounds) = fixed_audit_rounds {
        table_commitment_events = table_commitment_events
            .checked_add(rounds.checked_add(1).ok_or_else(composed_budget_overflow)?)
            .ok_or_else(composed_budget_overflow)?;
    }
    let witness_commitment_events = witness_commitment_events(witness_rounds, mode)?;
    let commitment_binding_events = table_commitment_events
        .checked_add(witness_commitment_events)
        .ok_or_else(composed_budget_overflow)?;

    // Embedded audit tables share the fixed value argument. Read coordinates
    // use one argument per power-of-two group.
    let whir_argument_count = spark_configs
        .read
        .len()
        .checked_add(if fixed_audit_embedded { 2 } else { 3 })
        .ok_or_else(composed_budget_overflow)?;
    compose_security_budget::<Ext>(
        requested,
        algebraic_error_terms,
        whir_argument_count,
        commitment_binding_events,
    )
}

fn spartan_algebraic_error_terms(
    num_outer_rounds: usize,
    num_inner_rounds: usize,
    mode: SpartanSoundnessMode,
) -> Result<usize, SpartanWhirError> {
    // The constant terms include the degree-two A/B/C batching challenge and
    // the fixed relation-batching checks for the selected Spartan protocol.
    match mode {
        SpartanSoundnessMode::NoZk => num_outer_rounds
            .checked_mul(4)
            .and_then(|terms| {
                num_inner_rounds
                    .checked_mul(2)
                    .and_then(|inner| terms.checked_add(inner))
            })
            .and_then(|terms| terms.checked_add(2)),
        SpartanSoundnessMode::FullZk { inner_degree } => num_outer_rounds
            .checked_mul(15)
            .and_then(|terms| {
                inner_degree
                    .checked_mul(num_inner_rounds)
                    .and_then(|inner| terms.checked_add(inner))
            })
            .and_then(|terms| terms.checked_add(4)),
    }
    .ok_or_else(composed_budget_overflow)
}

fn witness_commitment_events(
    witness_rounds: usize,
    mode: SpartanSoundnessMode,
) -> Result<usize, SpartanWhirError> {
    match mode {
        SpartanSoundnessMode::NoZk => witness_rounds
            .checked_add(1)
            .ok_or_else(composed_budget_overflow),
        // Setup retains the separate-tree budget so batching never weakens the
        // selected security target or changes the derived query schedules.
        SpartanSoundnessMode::FullZk { .. } => full_zk_witness_commitment_events(
            witness_rounds,
            witness_rounds
                .checked_mul(2)
                .and_then(|n| n.checked_add(3))
                .ok_or_else(composed_budget_overflow)?,
        ),
    }
}

/// Count witness commitment binding events for the actual fresh-mask batch count.
///
/// The setup budget uses the separate-tree upper bound. Reporting a smaller
/// same-height count does not reduce the configured Merkle security target.
pub fn full_zk_witness_commitment_events(
    witness_rounds: usize,
    fresh_mask_batches: usize,
) -> Result<usize, SpartanWhirError> {
    let max_groups = witness_rounds
        .checked_mul(2)
        .and_then(|n| n.checked_add(3))
        .ok_or_else(composed_budget_overflow)?;
    if fresh_mask_batches == 0 || fresh_mask_batches > max_groups {
        return Err(composed_budget_overflow());
    }
    // Three application roots, n+1 relation sumcheck roots, 2n switch
    // roots, the fresh main root, the reveal digest, and one fresh root per batch.
    witness_rounds
        .checked_mul(3)
        .and_then(|n| n.checked_add(6))
        .and_then(|n| n.checked_add(fresh_mask_batches))
        .ok_or_else(composed_budget_overflow)
}

fn compose_security_budget<Ext>(
    requested: &SecurityConfig,
    algebraic_error_terms: usize,
    whir_argument_count: usize,
    commitment_binding_events: usize,
) -> Result<(SecurityConfig, ComposedSecurityBudget), SpartanWhirError>
where
    Ext: ExtField,
{
    if algebraic_error_terms == 0 || whir_argument_count == 0 || commitment_binding_events == 0 {
        return Err(composed_budget_overflow());
    }
    let requested_bits = requested.effective_security_bits();
    let whir_slack_bits = three_way_budget_slack(whir_argument_count)?;
    let merkle_slack_bits = three_way_budget_slack(commitment_binding_events)?;

    let field_denominator = BigUint::from(algebraic_error_terms) * BigUint::from(3u8);
    let field_quotient = Ext::order() / field_denominator;
    let field_attainable = field_quotient.bits().saturating_sub(1).min(u32::MAX as u64) as u32;
    let whir_attainable = MAX_SECURITY_BITS.saturating_sub(whir_slack_bits);
    let merkle_attainable = MAX_SECURITY_BITS.saturating_sub(merkle_slack_bits);
    let (attainable_bits, dominant_component) = [
        (field_attainable, SecurityBoundComponent::ExtensionField),
        (whir_attainable, SecurityBoundComponent::WhirArguments),
        (
            merkle_attainable,
            SecurityBoundComponent::PoseidonCommitments,
        ),
    ]
    .into_iter()
    .min_by_key(|(bits, _)| *bits)
    .expect("three security components");

    let budget = ComposedSecurityBudget {
        requested_bits,
        attainable_bits,
        algebraic_error_terms,
        whir_argument_count,
        commitment_binding_events,
        whir_slack_bits,
        merkle_slack_bits,
        dominant_component,
    };
    if requested_bits > attainable_bits {
        return Err(SpartanWhirError::invalid_config_reason(
            InvalidConfigReason::ComposedSecurityUnavailable {
                requested_bits,
                attainable_bits,
                dominant_component,
            },
        ));
    }

    let component_security = SecurityConfig {
        security_level_bits: requested_bits
            .checked_add(whir_slack_bits)
            .ok_or_else(composed_budget_overflow)?,
        merkle_security_bits: requested_bits
            .checked_add(merkle_slack_bits)
            .ok_or_else(composed_budget_overflow)?,
        soundness_assumption: requested.soundness_assumption,
    };
    component_security.validate()?;
    Ok((component_security, budget))
}

fn three_way_budget_slack(events: usize) -> Result<u32, SpartanWhirError> {
    let weighted = events.checked_mul(3).ok_or_else(composed_budget_overflow)?;
    Ok(usize::BITS - weighted.saturating_sub(1).leading_zeros())
}

fn composed_budget_overflow() -> SpartanWhirError {
    SpartanWhirError::invalid_config_reason(InvalidConfigReason::ComposedSecurityBudgetOverflow)
}

#[cfg(test)]
mod composed_tests {
    use super::*;
    use crate::OcticBinExtension;

    fn config(bits: u32) -> SecurityConfig {
        SecurityConfig {
            security_level_bits: bits,
            merkle_security_bits: bits,
            soundness_assumption: SoundnessAssumption::CapacityBound,
        }
    }

    #[test]
    fn component_targets_cover_all_additive_events() {
        let (internal, budget) =
            compose_security_budget::<OcticBinExtension>(&config(100), 1_000, 5, 25)
                .expect("budget is attainable");
        assert_eq!(internal.security_level_bits, 104);
        assert_eq!(internal.merkle_security_bits, 107);
        assert_eq!(budget.whir_slack_bits, 4);
        assert_eq!(budget.merkle_slack_bits, 7);
    }

    #[test]
    fn larger_protocols_cannot_increase_attainable_security() {
        let (_, small) = compose_security_budget::<OcticBinExtension>(&config(100), 100, 5, 10)
            .expect("small budget is attainable");
        let (_, large) = compose_security_budget::<OcticBinExtension>(&config(100), 10_000, 5, 100)
            .expect("large budget is attainable");
        assert!(large.attainable_bits <= small.attainable_bits);
    }

    #[test]
    fn direct_mode_reserves_component_slack() {
        let requested = config(100);
        let (internal, budget) =
            compose_security_budget::<OcticBinExtension>(&requested, 42, 1, 10)
                .expect("DirectSparse budget is attainable");
        assert_eq!(internal.security_level_bits, 102);
        assert_eq!(internal.merkle_security_bits, 105);
        assert_eq!(budget.whir_argument_count, 1);
        assert_eq!(budget.whir_slack_bits, 2);
        assert_eq!(budget.merkle_slack_bits, 5);
    }

    #[test]
    fn full_zk_counts_every_poseidon_commitment() {
        let events = witness_commitment_events(5, SpartanSoundnessMode::FullZk { inner_degree: 3 })
            .expect("commitment count fits");
        assert_eq!(events, 34);

        let (internal, budget) =
            compose_security_budget::<OcticBinExtension>(&config(116), 100, 1, events)
                .expect("116-bit target remains attainable");
        assert_eq!(internal.security_level_bits, 118);
        assert_eq!(internal.merkle_security_bits, 123);
        assert_eq!(budget.merkle_slack_bits, 7);
    }

    #[test]
    fn direct_and_spark_share_the_spartan_terms() {
        assert_eq!(
            spartan_algebraic_error_terms(5, 10, SpartanSoundnessMode::NoZk),
            Ok(42)
        );
        assert_eq!(
            spartan_algebraic_error_terms(5, 10, SpartanSoundnessMode::FullZk { inner_degree: 3 }),
            Ok(109)
        );
    }

    #[test]
    fn budget_arithmetic_rejects_overflow() {
        let error = compose_security_budget::<OcticBinExtension>(&config(100), 1, usize::MAX, 1)
            .expect_err("overflow is rejected");
        assert_eq!(
            error,
            SpartanWhirError::invalid_config_reason(
                InvalidConfigReason::ComposedSecurityBudgetOverflow
            )
        );
    }

    #[test]
    fn unattainable_target_reports_the_limiting_component() {
        let error = compose_security_budget::<OcticBinExtension>(&config(123), 1_000, 5, 25)
            .expect_err("component slack makes 123 bits unattainable");
        assert!(matches!(
            error,
            SpartanWhirError::InvalidConfig(InvalidConfigReason::ComposedSecurityUnavailable {
                requested_bits: 123,
                dominant_component: SecurityBoundComponent::PoseidonCommitments
                    | SecurityBoundComponent::WhirArguments,
                ..
            })
        ));
    }
}
