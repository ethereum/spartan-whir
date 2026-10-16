use crate::{
    Evaluations, MatrixClosingMode, MultilinearPoint, PcsStatement, SpartanWhirEngine,
    SpartanWhirError, WhirPcsConfig,
};

mod sealed {
    pub trait SealedNoZkPcs {}
}

/// PCS families that may be composed with the ordinary, non-ZK Spartan IOP.
///
/// The capability is attached to the PCS family rather than an engine pairing,
/// so future engines can reuse a supported plain PCS implementation.
pub trait NoZkPcs: sealed::SealedNoZkPcs {}

pub(crate) use sealed::SealedNoZkPcs;

pub trait MlePcs<E: SpartanWhirEngine> {
    type Commitment;
    type ProverData;
    type Proof;
    type Config;

    fn commit(
        config: &Self::Config,
        poly: &Evaluations<E::F>,
        challenger: &mut E::Challenger,
    ) -> Result<(Self::Commitment, Self::ProverData), SpartanWhirError>;

    fn open(
        config: &Self::Config,
        prover_data: Self::ProverData,
        statement: &PcsStatement<E>,
        challenger: &mut E::Challenger,
    ) -> Result<Self::Proof, SpartanWhirError>;

    fn verify(
        config: &Self::Config,
        commitment: &Self::Commitment,
        statement: &PcsStatement<E>,
        proof: &Self::Proof,
        challenger: &mut E::Challenger,
    ) -> Result<(), SpartanWhirError>;
}

pub trait ProtocolPcs<E: SpartanWhirEngine>: MlePcs<E> {
    type ParsedCommitment;

    fn open_compressed(
        config: &Self::Config,
        prover_data: Self::ProverData,
        statement: &PcsStatement<E>,
        final_rows: bool,
        challenger: &mut E::Challenger,
    ) -> Result<Self::Proof, SpartanWhirError> {
        if final_rows {
            return Err(SpartanWhirError::InvalidProofShape);
        }
        Self::open(config, prover_data, statement, challenger)
    }

    fn verify_finalize_compressed(
        config: &Self::Config,
        parsed: &Self::ParsedCommitment,
        statement: &PcsStatement<E>,
        proof: &Self::Proof,
        final_rows: bool,
        challenger: &mut E::Challenger,
    ) -> Result<(), SpartanWhirError> {
        if final_rows {
            return Err(SpartanWhirError::InvalidProofShape);
        }
        Self::verify_finalize(config, parsed, statement, proof, challenger)
    }

    fn validate_spartan_config(
        config: &WhirPcsConfig,
        _matrix_closing: MatrixClosingMode,
        _num_outer_rounds: usize,
        _num_inner_rounds: usize,
    ) -> Result<(), SpartanWhirError>
    where
        Self: MlePcs<E, Config = WhirPcsConfig>,
    {
        config.validate()
    }

    fn prepare_committed_opening(
        config: &Self::Config,
        prover_data: Self::ProverData,
        challenger: &mut E::Challenger,
    ) -> Result<Self::ProverData, SpartanWhirError>;

    fn verify_parse_commitment(
        config: &Self::Config,
        commitment: &Self::Commitment,
        proof: &Self::Proof,
        challenger: &mut E::Challenger,
    ) -> Result<Self::ParsedCommitment, SpartanWhirError>;

    fn verify_finalize(
        config: &Self::Config,
        parsed: &Self::ParsedCommitment,
        statement: &PcsStatement<E>,
        proof: &Self::Proof,
        challenger: &mut E::Challenger,
    ) -> Result<(), SpartanWhirError>;
}

/// Plain PCS support for one power-of-two group of SPARK read coordinates.
///
/// The surrounding protocol decomposes all row and column coordinates into
/// power-of-two groups. Each group has one commitment and one opening argument.
/// Opening column lists and points are fixed by the surrounding SPARK
/// transcript before this interface is called.
pub trait SparkReadPcs<E: SpartanWhirEngine>: ProtocolPcs<E, Config = WhirPcsConfig> {
    type ReadProverData;
    type ParsedReadCommitment;

    fn open_read_table_compressed(
        config: &WhirPcsConfig,
        prover_data: Self::ReadProverData,
        column_count: usize,
        opening_columns: &[Vec<usize>],
        points: &[MultilinearPoint<E::EF>],
        final_rows: bool,
        challenger: &mut E::Challenger,
    ) -> Result<(Self::Proof, Vec<Vec<E::EF>>), SpartanWhirError> {
        if final_rows {
            return Err(SpartanWhirError::InvalidProofShape);
        }
        Self::open_read_table(
            config,
            prover_data,
            column_count,
            opening_columns,
            points,
            challenger,
        )
    }

    fn verify_finalize_read_table_compressed(
        config: &WhirPcsConfig,
        parsed: &Self::ParsedReadCommitment,
        proof: &Self::Proof,
        column_count: usize,
        opening_columns: &[Vec<usize>],
        points: &[MultilinearPoint<E::EF>],
        evals: &[Vec<E::EF>],
        final_rows: bool,
        challenger: &mut E::Challenger,
    ) -> Result<(), SpartanWhirError> {
        if final_rows {
            return Err(SpartanWhirError::InvalidProofShape);
        }
        Self::verify_finalize_read_table(
            config,
            parsed,
            proof,
            column_count,
            opening_columns,
            points,
            evals,
            challenger,
        )
    }

    fn commit_read_table(
        config: &WhirPcsConfig,
        coordinate_columns: Evaluations<E::F>,
        domain_size: usize,
        column_count: usize,
        challenger: &mut E::Challenger,
    ) -> Result<(Self::Commitment, Self::ReadProverData), SpartanWhirError>;

    fn open_read_table(
        config: &WhirPcsConfig,
        prover_data: Self::ReadProverData,
        column_count: usize,
        opening_columns: &[Vec<usize>],
        points: &[MultilinearPoint<E::EF>],
        challenger: &mut E::Challenger,
    ) -> Result<(Self::Proof, Vec<Vec<E::EF>>), SpartanWhirError>;

    fn verify_parse_read_commitment(
        config: &WhirPcsConfig,
        commitment: &Self::Commitment,
        proof: &Self::Proof,
        challenger: &mut E::Challenger,
    ) -> Result<Self::ParsedReadCommitment, SpartanWhirError>;

    fn verify_finalize_read_table(
        config: &WhirPcsConfig,
        parsed: &Self::ParsedReadCommitment,
        proof: &Self::Proof,
        column_count: usize,
        opening_columns: &[Vec<usize>],
        points: &[MultilinearPoint<E::EF>],
        evals: &[Vec<E::EF>],
        challenger: &mut E::Challenger,
    ) -> Result<(), SpartanWhirError>;
}

pub trait CommittedPolynomialView<EF> {
    fn num_variables(&self) -> usize;
    fn polynomial(&self) -> &[crate::engine::F];
}
