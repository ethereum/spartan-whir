//! Compact transport encoding for the existing full-ZK SPARK proof.
//!
//! Compressed proofs are verified through the ordinary protocol after restoring
//! its field values. Encoding flags never enter the Fiat–Shamir transcript.

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_whir::pcs::proof::{QueryOpenings, WhirProof};
use rand::distr::{Distribution, StandardUniform};
use serde::{Deserialize, Serialize};

use crate::engine::{ExtField, F};
use crate::plonky3_whir_pcs::{FullZkPoseidonEngine, FullZkPoseidonPcs};
use crate::spark::CompactSparkBatchedMemoryProductsProof;
use crate::{Plonky3WhirPcs, SpartanWhirError, ZkMatrixClosingProofFor, ZkSpartanProofFor};

const MAGIC: &[u8; 4] = b"SPC2";
const MAX_EXPANDED_FIELDS: usize = 1 << 24;
const MAX_ROW_WIDTH: usize = 1 << 16;

/// Independent encoding choices for matched size and time comparisons.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProofCompressionOptions {
    pub structured_rows: bool,
    pub fresh_rows: bool,
    pub packed_fields: bool,
    pub compact_integers: bool,
    pub factored_rounds: bool,
    pub final_rows: bool,
    pub derived_products: bool,
    pub refined_metadata: bool,
    pub duplicate_columns: bool,
}

impl ProofCompressionOptions {
    /// Encoding choices retained on the selected 2048-byte SHA-256 workload.
    ///
    /// Nonadjacent column detection remains available as an experiment; it
    /// provided no additional size reduction on that workload.
    pub const fn recommended() -> Self {
        Self {
            structured_rows: true,
            fresh_rows: true,
            packed_fields: true,
            compact_integers: true,
            factored_rounds: true,
            final_rows: true,
            derived_products: true,
            refined_metadata: true,
            duplicate_columns: false,
        }
    }

    fn flags(self) -> u16 {
        u16::from(self.structured_rows)
            | (u16::from(self.fresh_rows) << 1)
            | (u16::from(self.packed_fields) << 2)
            | (u16::from(self.compact_integers) << 3)
            | (u16::from(self.factored_rounds) << 4)
            | (u16::from(self.final_rows) << 5)
            | (u16::from(self.derived_products) << 6)
            | (u16::from(self.refined_metadata) << 7)
            | (u16::from(self.duplicate_columns) << 8)
    }

    fn from_flags(flags: u16) -> Result<Self, SpartanWhirError> {
        if flags & !511 != 0 {
            return Err(SpartanWhirError::InvalidProofShape);
        }
        Ok(Self {
            structured_rows: flags & 1 != 0,
            fresh_rows: flags & 2 != 0,
            packed_fields: flags & 4 != 0,
            compact_integers: flags & 8 != 0,
            factored_rounds: flags & 16 != 0,
            final_rows: flags & 32 != 0,
            derived_products: flags & 64 != 0,
            refined_metadata: flags & 128 != 0,
            duplicate_columns: flags & 256 != 0,
        })
    }
}

/// Access to the initial base-field rows of a plain WHIR proof.
pub trait PlainProofRowAccess {
    fn initial_base_rows(&mut self) -> Result<&mut Vec<Vec<F>>, SpartanWhirError>;
}

impl<Ext, M> PlainProofRowAccess for WhirProof<F, Ext, M>
where
    M: Mmcs<F>,
{
    fn initial_base_rows(&mut self) -> Result<&mut Vec<Vec<F>>, SpartanWhirError> {
        let opening = match self.rounds.first_mut() {
            Some(round) => &mut round.openings,
            None => &mut self.final_openings,
        };
        match opening {
            QueryOpenings::Base(opening) => Ok(&mut opening.rows),
            QueryOpenings::Extension(_) => Err(SpartanWhirError::InvalidProofShape),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RowEncoding {
    width: u64,
    zero_columns: Vec<u16>,
    tail_start: u64,
    duplicates: Vec<(u16, u16)>,
}

/// An opaque transport proof; use the compressed verification entry point.
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "ZkSpartanProofFor<E>: Serialize, CompactSparkBatchedMemoryProductsProof<E::EF>: Serialize",
    deserialize = "ZkSpartanProofFor<E>: Deserialize<'de>, CompactSparkBatchedMemoryProductsProof<E::EF>: Deserialize<'de>"
))]
pub struct CompressedZkProofFor<E>
where
    E: FullZkPoseidonEngine,
    E::EF: ExtField,
    E::Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    StandardUniform: Distribution<E::EF>,
    Plonky3WhirPcs: FullZkPoseidonPcs<E>,
{
    pub(crate) proof: ZkSpartanProofFor<E>,
    row_encodings: Vec<RowEncoding>,
    pub(crate) products: Option<CompactSparkBatchedMemoryProductsProof<E::EF>>,
    #[serde(skip)]
    pub(crate) options: ProofCompressionOptions,
}

impl<E> CompressedZkProofFor<E>
where
    E: FullZkPoseidonEngine,
    E::EF: ExtField,
    E::PlainProof: PlainProofRowAccess,
    E::Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    StandardUniform: Distribution<E::EF>,
    Plonky3WhirPcs: FullZkPoseidonPcs<E>,
{
    pub(crate) fn from_proof(
        mut proof: ZkSpartanProofFor<E>,
        options: ProofCompressionOptions,
        mut products: Option<CompactSparkBatchedMemoryProductsProof<E::EF>>,
    ) -> Result<Self, SpartanWhirError> {
        if !matches!(proof.matrix_closing, ZkMatrixClosingProofFor::Spark(_))
            || options.factored_rounds != products.is_some()
            || (options.derived_products && !options.factored_rounds)
            || (options.duplicate_columns && !options.structured_rows)
        {
            return Err(SpartanWhirError::InvalidProofShape);
        }
        let mut row_encodings = Vec::new();
        if options.structured_rows {
            for rows in initial_rows_mut::<E>(&mut proof)? {
                let mut encoding = compress_rows(rows)?;
                if options.duplicate_columns {
                    encoding.duplicates = compress_duplicate_columns(rows);
                }
                row_encodings.push(encoding);
            }
        }
        if options.fresh_rows {
            proof.pcs_proof.base_case.fresh_main_openings.rows.clear();
            for opening in &mut proof.pcs_proof.base_case.fresh_mask_openings {
                opening.rows.clear();
            }
        }
        if options.factored_rounds {
            if options.derived_products {
                crate::spark::refine_spark_memory_products(products.as_mut().unwrap())?;
            }
            let ZkMatrixClosingProofFor::Spark(closing) = &mut proof.matrix_closing else {
                unreachable!();
            };
            // The compact product object contains the endpoints and roots too.
            // Keep a unique empty placeholder in the ordinary proof field.
            closing.spark_products = empty_products();
        }
        Ok(Self {
            proof,
            row_encodings,
            products,
            options,
        })
    }

    pub fn options(&self) -> ProofCompressionOptions {
        self.options
    }

    /// Return encoding size statistics using untimed serializations.
    pub fn encoding_statistics(&self) -> Result<serde_json::Value, SpartanWhirError> {
        let statistics = crate::proof_compression_encoding::encoding_statistics(self)
            .map_err(|_| SpartanWhirError::InvalidProofShape)?;
        serde_json::to_value(statistics).map_err(|_| SpartanWhirError::InvalidProofShape)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, SpartanWhirError> {
        let payload = crate::proof_compression_encoding::encode_refined(
            self,
            self.options.packed_fields,
            self.options.compact_integers,
            self.options.refined_metadata,
        )
        .map_err(|_| SpartanWhirError::InvalidProofShape)?;
        let mut bytes = Vec::with_capacity(6 + payload.len());
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&self.options.flags().to_le_bytes());
        bytes.extend_from_slice(&payload);
        Ok(bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SpartanWhirError> {
        if bytes.get(..4) != Some(MAGIC.as_slice()) || bytes.len() < 6 {
            return Err(SpartanWhirError::InvalidProofShape);
        }
        let options =
            ProofCompressionOptions::from_flags(u16::from_le_bytes([bytes[4], bytes[5]]))?;
        let mut value: Self = crate::proof_compression_encoding::decode_refined(
            &bytes[6..],
            options.packed_fields,
            options.compact_integers,
            options.refined_metadata,
        )
        .map_err(|_| SpartanWhirError::InvalidProofShape)?;
        value.options = options;
        if !matches!(
            value.proof.matrix_closing,
            ZkMatrixClosingProofFor::Spark(_)
        ) || options.factored_rounds != value.products.is_some()
            || value
                .products
                .as_ref()
                .is_some_and(|products| products.omits_derived_fields() != options.derived_products)
            || (!options.structured_rows && !value.row_encodings.is_empty())
            || (options.derived_products && !options.factored_rounds)
            || (options.duplicate_columns && !options.structured_rows)
            || (!options.duplicate_columns
                && value
                    .row_encodings
                    .iter()
                    .any(|encoding| !encoding.duplicates.is_empty()))
        {
            return Err(SpartanWhirError::InvalidProofShape);
        }
        if options.factored_rounds {
            let ZkMatrixClosingProofFor::Spark(closing) = &value.proof.matrix_closing else {
                return Err(SpartanWhirError::InvalidProofShape);
            };
            if closing.spark_products != empty_products() {
                return Err(SpartanWhirError::InvalidProofShape);
            }
        }
        Ok(value)
    }

    pub(crate) fn restore_static_rows(&mut self) -> Result<(), SpartanWhirError> {
        if self.options.structured_rows {
            let rows = initial_rows_mut::<E>(&mut self.proof)?;
            if rows.len() != self.row_encodings.len() {
                return Err(SpartanWhirError::InvalidProofShape);
            }
            let mut remaining = MAX_EXPANDED_FIELDS;
            for (rows, encoding) in rows.into_iter().zip(&self.row_encodings) {
                restore_rows(rows, encoding, &mut remaining)?;
            }
        }
        Ok(())
    }
}

fn empty_products<Ext: ExtField>() -> crate::SparkBatchedMemoryProductsProof<Ext> {
    let axis = crate::SparkMemoryProductClaim {
        init_root: Ext::ZERO,
        read_root: Ext::ZERO,
        write_root: Ext::ZERO,
        audit_root: Ext::ZERO,
    };
    let tree = crate::SparkBatchedProductProof {
        product_roots: Vec::new(),
        dotproduct_claims: Vec::new(),
        layers: Vec::new(),
    };
    crate::SparkBatchedMemoryProductsProof {
        products: crate::SparkMemoryProductProof {
            beta: Ext::ZERO,
            gamma: Ext::ZERO,
            row: axis,
            col: axis,
        },
        matrix_evals: [Ext::ZERO; 3],
        proof_ops: tree.clone(),
        proof_mem: tree,
    }
}

fn initial_rows_mut<E>(
    proof: &mut ZkSpartanProofFor<E>,
) -> Result<Vec<&mut Vec<Vec<F>>>, SpartanWhirError>
where
    E: FullZkPoseidonEngine,
    E::EF: ExtField,
    E::PlainProof: PlainProofRowAccess,
    E::Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    StandardUniform: Distribution<E::EF>,
    Plonky3WhirPcs: FullZkPoseidonPcs<E>,
{
    let ZkMatrixClosingProofFor::Spark(closing) = &mut proof.matrix_closing else {
        return Err(SpartanWhirError::InvalidProofShape);
    };
    let mut result = vec![closing
        .spark_fixed_openings
        .value_proof
        .initial_base_rows()?];
    if let Some(proof) = &mut closing.spark_fixed_openings.audit_proof {
        result.push(proof.initial_base_rows()?);
    }
    for group in &mut closing.spark_read_openings.groups {
        result.push(group.proof.initial_base_rows()?);
    }
    Ok(result)
}

fn compress_rows(rows: &mut Vec<Vec<F>>) -> Result<RowEncoding, SpartanWhirError> {
    let width = rows
        .first()
        .ok_or(SpartanWhirError::InvalidProofShape)?
        .len();
    if width == 0 || width > MAX_ROW_WIDTH || rows.iter().any(|row| row.len() != width) {
        return Err(SpartanWhirError::InvalidProofShape);
    }
    let zero_columns: Vec<u16> = (0..width)
        .filter(|&i| rows.iter().all(|row| row[i] == F::ZERO))
        .map(|i| i as u16)
        .collect();
    let kept: Vec<usize> = (0..width)
        .filter(|i| zero_columns.binary_search(&(*i as u16)).is_err())
        .collect();
    let mut tail_start = kept.len().saturating_sub(1);
    while tail_start > 0
        && rows
            .iter()
            .all(|row| row[kept[tail_start - 1]] == row[*kept.last().unwrap()])
    {
        tail_start -= 1;
    }
    let stored = if kept.is_empty() { 0 } else { tail_start + 1 };
    for row in rows {
        for (destination, &source) in kept.iter().take(stored).enumerate() {
            row[destination] = row[source];
        }
        row.truncate(stored);
    }
    Ok(RowEncoding {
        width: width as u64,
        zero_columns,
        tail_start: tail_start as u64,
        duplicates: Vec::new(),
    })
}

fn compress_duplicate_columns(rows: &mut Vec<Vec<F>>) -> Vec<(u16, u16)> {
    let width = rows.first().map_or(0, Vec::len);
    let mut candidates = std::collections::HashMap::<u32, Vec<usize>>::new();
    let mut kept = Vec::new();
    let mut duplicates = Vec::new();
    for column in 0..width {
        let candidates = candidates
            .entry(rows[0][column].as_canonical_u32())
            .or_default();
        if let Some(&source) = candidates
            .iter()
            .find(|&&source| rows.iter().all(|row| row[source] == row[column]))
        {
            duplicates.push((column as u16, source as u16));
        } else {
            kept.push(column);
            candidates.push(column);
        }
    }
    for row in rows {
        for (destination, &source) in kept.iter().enumerate() {
            row[destination] = row[source];
        }
        row.truncate(kept.len());
    }
    duplicates
}

fn restore_rows(
    rows: &mut Vec<Vec<F>>,
    encoding: &RowEncoding,
    remaining: &mut usize,
) -> Result<(), SpartanWhirError> {
    let reject = || SpartanWhirError::InvalidProofShape;
    let width = usize::try_from(encoding.width).map_err(|_| reject())?;
    let tail_start = usize::try_from(encoding.tail_start).map_err(|_| reject())?;
    if width == 0
        || width > MAX_ROW_WIDTH
        || encoding.zero_columns.len() > width
        || encoding
            .zero_columns
            .iter()
            .any(|&i| usize::from(i) >= width)
        || encoding
            .zero_columns
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
    {
        return Err(reject());
    }
    let kept = width - encoding.zero_columns.len();
    if (kept == 0 && tail_start != 0) || (kept != 0 && tail_start >= kept) {
        return Err(reject());
    }
    let count = width.checked_mul(rows.len()).ok_or_else(reject)?;
    *remaining = remaining.checked_sub(count).ok_or_else(reject)?;
    let stored = if kept == 0 { 0 } else { tail_start + 1 };
    let duplicates = &encoding.duplicates;
    if duplicates.len() > stored
        || duplicates.iter().any(|&(destination, source)| {
            usize::from(destination) >= stored || source >= destination
        })
        || duplicates.windows(2).any(|pair| pair[0].0 >= pair[1].0)
        || rows
            .iter()
            .any(|row| row.len() != stored - duplicates.len())
    {
        return Err(reject());
    }
    for row in rows {
        if !duplicates.is_empty() {
            let mut expanded = Vec::with_capacity(stored);
            let mut duplicate = 0;
            let mut input = row.iter();
            for column in 0..stored {
                if let Some(&(_, source)) = duplicates
                    .get(duplicate)
                    .filter(|&&(destination, _)| usize::from(destination) == column)
                {
                    expanded.push(expanded[usize::from(source)]);
                    duplicate += 1;
                } else {
                    expanded.push(*input.next().ok_or_else(reject)?);
                }
            }
            *row = expanded;
        }
        let mut restored = Vec::with_capacity(width);
        let mut zero = 0;
        let mut index = 0;
        for column in 0..width {
            if encoding.zero_columns.get(zero).copied() == Some(column as u16) {
                restored.push(F::ZERO);
                zero += 1;
            } else {
                restored.push(row[index.min(tail_start)]);
                index += 1;
            }
        }
        *row = restored;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn structured_rows_roundtrip_and_reject_invalid_layout() {
        let original = vec![
            vec![F::ZERO, F::ONE, F::TWO, F::TWO, F::ZERO],
            vec![F::ZERO, F::TWO, F::ONE, F::ONE, F::ZERO],
        ];
        let mut rows = original.clone();
        let encoding = compress_rows(&mut rows).unwrap();
        assert_eq!(rows[0].len(), 2);
        let compressed = rows.clone();
        restore_rows(&mut rows, &encoding, &mut 100).unwrap();
        assert_eq!(rows, original);
        let mut bad = encoding.clone();
        bad.zero_columns = vec![0, 0];
        assert!(restore_rows(&mut compressed.clone(), &bad, &mut 100).is_err());
        assert!(restore_rows(&mut compressed.clone(), &encoding, &mut 2).is_err());
        let mut zeros = vec![vec![F::ZERO; 4]; 2];
        let encoding = compress_rows(&mut zeros).unwrap();
        assert!(zeros[0].is_empty());
        restore_rows(&mut zeros, &encoding, &mut 100).unwrap();
        assert_eq!(zeros, vec![vec![F::ZERO; 4]; 2]);
    }

    #[test]
    fn duplicate_columns_restore_nonadjacent_values_and_reject_bad_sources() {
        let original = vec![
            vec![F::ONE, F::TWO, F::ONE, F::ZERO, F::TWO],
            vec![F::TWO, F::ONE, F::TWO, F::ZERO, F::ONE],
        ];
        let mut rows = original.clone();
        let mut encoding = compress_rows(&mut rows).unwrap();
        encoding.duplicates = compress_duplicate_columns(&mut rows);
        assert_eq!(encoding.duplicates, vec![(2, 0), (3, 1)]);
        assert_eq!(rows[0].len(), 2);
        let compressed = rows.clone();
        restore_rows(&mut rows, &encoding, &mut 100).unwrap();
        assert_eq!(rows, original);
        let mut invalid = encoding.clone();
        invalid.duplicates[0].1 = 2;
        assert!(restore_rows(&mut compressed.clone(), &invalid, &mut 100).is_err());
        invalid.duplicates = vec![(2, 0), (2, 1)];
        assert!(restore_rows(&mut compressed.clone(), &invalid, &mut 100).is_err());
    }
}
