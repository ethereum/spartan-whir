#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecurityBoundComponent {
    ExtensionField,
    WhirArguments,
    PoseidonCommitments,
}

impl core::fmt::Display for SecurityBoundComponent {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ExtensionField => write!(f, "extension field"),
            Self::WhirArguments => write!(f, "WHIR arguments"),
            Self::PoseidonCommitments => write!(f, "Poseidon commitments"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvalidConfigReason {
    Generic,
    ZeroFoldingFactor,
    ZeroRsDomainInitialReductionFactor,
    RsDomainInitialReductionFactorExceedsFirstFoldingFactor {
        rs_domain_initial_reduction_factor: usize,
        first_folding_factor: usize,
    },
    InvalidFoldingSchedule {
        num_variables: usize,
    },
    FoldedDomainSizeOverflow {
        num_variables: usize,
        starting_log_inv_rate: usize,
    },
    FirstFoldingFactorExceedsDomain {
        first_folding_factor: usize,
        log_domain_size: usize,
    },
    FoldedDomainExceedsBaseTwoAdicity {
        log_folded_domain_size: usize,
        base_two_adicity: usize,
        min_first_folding_factor: usize,
    },
    ZkWhirMaskLengthTooSmall {
        ell_zk: usize,
    },
    ZkWhirMaskRateTooHigh,
    ZkWhirRandomnessExceedsSlack {
        round: usize,
        randomness: usize,
        slack: usize,
    },
    ZkWhirMaskDomainExceedsTwoAdicity {
        log_domain_size: usize,
        two_adicity: usize,
    },
    IncompatibleApplicationMaskDomains {
        inner_domain_size: usize,
        outer_domain_size: usize,
    },
    FullZkSecurityExceedsExtensionField {
        requested_bits: u32,
        extension_field_bits: usize,
        soundness_error_terms: usize,
    },
    ComposedSecurityUnavailable {
        requested_bits: u32,
        attainable_bits: u32,
        dominant_component: SecurityBoundComponent,
    },
    ComposedSecurityBudgetOverflow,
    MissingDerivedProverData,
    UnauthenticatedVerifyingKey,
}

impl core::fmt::Display for InvalidConfigReason {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Generic => write!(f, "unspecified reason"),
            Self::ZeroFoldingFactor => write!(f, "first folding factor is zero"),
            Self::ZeroRsDomainInitialReductionFactor => {
                write!(f, "RS domain initial reduction factor is zero")
            }
            Self::RsDomainInitialReductionFactorExceedsFirstFoldingFactor {
                rs_domain_initial_reduction_factor,
                first_folding_factor,
            } => write!(
                f,
                "RS domain initial reduction factor {rs_domain_initial_reduction_factor} exceeds first folding factor {first_folding_factor}"
            ),
            Self::InvalidFoldingSchedule { num_variables } => write!(
                f,
                "folding schedule is invalid for {num_variables} variables"
            ),
            Self::FoldedDomainSizeOverflow {
                num_variables,
                starting_log_inv_rate,
            } => write!(
                f,
                "domain log size overflows for {num_variables} variables and starting log inverse rate {starting_log_inv_rate}"
            ),
            Self::FirstFoldingFactorExceedsDomain {
                first_folding_factor,
                log_domain_size,
            } => write!(
                f,
                "first folding factor {first_folding_factor} exceeds domain log size {log_domain_size}"
            ),
            Self::FoldedDomainExceedsBaseTwoAdicity {
                log_folded_domain_size,
                base_two_adicity,
                min_first_folding_factor,
            } => write!(
                f,
                "folded domain log size {log_folded_domain_size} exceeds base two-adicity {base_two_adicity}; first folding factor must be at least {min_first_folding_factor}"
            ),
            Self::ZkWhirMaskLengthTooSmall { ell_zk } => write!(
                f,
                "ZK WHIR mask length {ell_zk} is below the minimum of 3"
            ),
            Self::ZkWhirMaskRateTooHigh => {
                write!(f, "ZK WHIR mask log inverse rate must be at least 1")
            }
            Self::ZkWhirRandomnessExceedsSlack {
                round,
                randomness,
                slack,
            } => write!(
                f,
                "ZK WHIR round {round} randomness rows {randomness} exceed slack {slack}"
            ),
            Self::ZkWhirMaskDomainExceedsTwoAdicity {
                log_domain_size,
                two_adicity,
            } => write!(
                f,
                "ZK WHIR mask domain 2^{log_domain_size} exceeds extension-field two-adicity 2^{two_adicity}"
            ),
            Self::IncompatibleApplicationMaskDomains {
                inner_domain_size,
                outer_domain_size,
            } => write!(
                f,
                "inner and outer application masks require different domains ({inner_domain_size} and {outer_domain_size})"
            ),
            Self::FullZkSecurityExceedsExtensionField {
                requested_bits,
                extension_field_bits,
                soundness_error_terms,
            } => write!(
                f,
                "full-ZK target of {requested_bits} bits is not supported by the {extension_field_bits}-bit extension field with {soundness_error_terms} algebraic error terms"
            ),
            Self::ComposedSecurityUnavailable {
                requested_bits,
                attainable_bits,
                dominant_component,
            } => write!(
                f,
                "composed target of {requested_bits} bits exceeds the attainable {attainable_bits} bits limited by {dominant_component}"
            ),
            Self::ComposedSecurityBudgetOverflow => {
                write!(f, "composed security budget arithmetic overflowed")
            }
            Self::MissingDerivedProverData => write!(f, "derived prover data is missing"),
            Self::UnauthenticatedVerifyingKey => {
                write!(f, "verifying key is not authenticated")
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpartanWhirError {
    Unimplemented(&'static str),
    InvalidR1csShape,
    InvalidWitnessLength,
    InvalidPublicInputLength,
    PublicInputMismatch,
    PaddingError,
    SecurityBelowMinimum,
    SecurityAboveMaximum,
    MerkleSecurityBelowMinimum,
    MerkleSecurityAboveMaximum,
    TranscriptMismatch,
    SumcheckFailed,
    SparkMatrixEvaluationMismatch,
    PcsVerificationFailed,
    ProofDecodeFailed,
    ProofKindMismatch,
    UnsupportedStatementType,
    InvalidConfig(InvalidConfigReason),
    WhirCommitFailed,
    WhirOpenFailed,
    WhirVerifyFailed,
    CommitmentMismatch,
    InvalidCommitmentShape,
    InvalidProofShape,
    InvalidPolynomialLength,
    InvalidNumVariables,
    InvalidRoundCount,
    InvalidRoundPolynomial,
    NonInvertibleElement,
    ProofEncodeFailed,
    UnsupportedBlobVersion,
    InvalidBlobHeader,
    InvalidBlobLayout,
    InvalidBlobFlags,
    TrailingBytes,
    DigestBytesMismatch,
    NonCanonicalEncoding,
}

impl SpartanWhirError {
    pub const fn invalid_config() -> Self {
        Self::InvalidConfig(InvalidConfigReason::Generic)
    }

    pub const fn invalid_config_reason(reason: InvalidConfigReason) -> Self {
        Self::InvalidConfig(reason)
    }
}

impl core::fmt::Display for SpartanWhirError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Unimplemented(where_) => write!(f, "unimplemented: {where_}"),
            Self::InvalidR1csShape => write!(f, "invalid R1CS shape"),
            Self::InvalidWitnessLength => write!(f, "invalid witness length"),
            Self::InvalidPublicInputLength => write!(f, "invalid public input length"),
            Self::PublicInputMismatch => write!(f, "public inputs do not match expected statement"),
            Self::PaddingError => write!(f, "padding error"),
            Self::SecurityBelowMinimum => write!(f, "security level below minimum"),
            Self::SecurityAboveMaximum => write!(f, "security level above supported maximum"),
            Self::MerkleSecurityBelowMinimum => write!(f, "merkle security below minimum"),
            Self::MerkleSecurityAboveMaximum => {
                write!(f, "merkle security above supported maximum")
            }
            Self::TranscriptMismatch => write!(f, "transcript mismatch"),
            Self::SumcheckFailed => write!(f, "sumcheck verification failed"),
            Self::SparkMatrixEvaluationMismatch => {
                write!(
                    f,
                    "SPARK matrix evaluation does not match the inner sumcheck"
                )
            }
            Self::PcsVerificationFailed => write!(f, "PCS verification failed"),
            Self::ProofDecodeFailed => write!(f, "proof decode failed"),
            Self::ProofKindMismatch => write!(f, "proof kind mismatch"),
            Self::UnsupportedStatementType => write!(f, "unsupported statement type"),
            Self::InvalidConfig(reason) => write!(f, "invalid configuration: {reason}"),
            Self::WhirCommitFailed => write!(f, "WHIR commitment failed"),
            Self::WhirOpenFailed => write!(f, "WHIR opening failed"),
            Self::WhirVerifyFailed => write!(f, "WHIR verification failed"),
            Self::CommitmentMismatch => write!(f, "commitment mismatch"),
            Self::InvalidCommitmentShape => write!(f, "invalid commitment shape"),
            Self::InvalidProofShape => write!(f, "invalid proof shape"),
            Self::InvalidPolynomialLength => write!(f, "invalid polynomial length"),
            Self::InvalidNumVariables => write!(f, "invalid number of variables"),
            Self::InvalidRoundCount => write!(f, "invalid sumcheck round count"),
            Self::InvalidRoundPolynomial => write!(f, "invalid sumcheck round polynomial"),
            Self::NonInvertibleElement => write!(f, "non-invertible field element"),
            Self::ProofEncodeFailed => write!(f, "proof encode failed"),
            Self::UnsupportedBlobVersion => write!(f, "unsupported blob version"),
            Self::InvalidBlobHeader => write!(f, "invalid blob header"),
            Self::InvalidBlobLayout => write!(f, "invalid blob layout"),
            Self::InvalidBlobFlags => write!(f, "invalid blob flags"),
            Self::TrailingBytes => write!(f, "trailing bytes"),
            Self::DigestBytesMismatch => write!(f, "digest bytes mismatch"),
            Self::NonCanonicalEncoding => write!(f, "non-canonical encoding"),
        }
    }
}
