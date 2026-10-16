use alloc::{vec, vec::Vec};
use core::cmp::max;

use p3_field::{
    ExtensionField, Field, PackedFieldExtension, PackedValue, PrimeCharacteristicRing, PrimeField32,
};
use p3_maybe_rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{engine::F, Evaluations, SpartanWhirError};

/// Domain for the canonical digest of a KoalaBear R1CS relation.
pub const R1CS_RELATION_DIGEST_DOMAIN: &[u8] = b"spartan-whir-r1cs-relation-digest";
/// Encoding version for [`canonical_r1cs_relation_digest`].
pub const R1CS_RELATION_DIGEST_VERSION: u32 = 1;

const R1CS_RELATION_FIELD_ID: &[u8] = b"koalabear-prime-field";
const R1CS_RELATION_COLUMN_LAYOUT_ID: &[u8] = b"witness-constant-public";

const R1CS_PARALLEL_MATRIX_MIN_NNZ: usize = 1 << 14;
const R1CS_PARALLEL_BIND_MIN_NNZ: usize = 1 << 15;
const R1CS_PARALLEL_BIND_CHUNKS_PER_THREAD: usize = 4;
const R1CS_PARALLEL_BIND_FALLBACK_MAX_SHARDS: usize = 4;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparseMatEntry<F> {
    pub row: usize,
    pub col: usize,
    pub val: F,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparseMatrix<F> {
    pub num_rows: usize,
    pub num_cols: usize,
    pub entries: Vec<SparseMatEntry<F>>,
}

impl<F> SparseMatrix<F> {
    pub fn nnz(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct R1csShape<F> {
    pub num_cons: usize,
    pub num_vars: usize,
    pub num_io: usize,
    pub a: SparseMatrix<F>,
    pub b: SparseMatrix<F>,
    pub c: SparseMatrix<F>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct R1csWitness<F> {
    pub w: Vec<F>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DirectBindLayout<F> {
    width: usize,
    a: ColumnBindMatrixLayout<F>,
    b: ColumnBindMatrixLayout<F>,
    c: ColumnBindMatrixLayout<F>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DirectMultiplyLayout {
    a: RowMatrixLayout,
    b: RowMatrixLayout,
    c: RowMatrixLayout,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ColumnBindMatrixLayout<F> {
    entries_by_col: Vec<ColumnBindEntry<F>>,
    col_starts: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ColumnBindEntry<F> {
    col: usize,
    row: usize,
    val: F,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RowMatrixLayout {
    entry_indices_by_row: Vec<usize>,
    row_starts: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct R1csInstance<F, C> {
    pub public_inputs: Vec<F>,
    pub witness_commitment: C,
}

/// Compute a canonical SHA-256 digest of a padded KoalaBear R1CS relation.
///
/// The digest includes both the original dimensions and the padded shape. A
/// sparse matrix is encoded by sorted `(row, column)` coordinates after
/// duplicate entries have been added in the field and zero sums removed.
/// This is the generated-statement digest used by the Fiat-Shamir mitigation
/// described in Fenzi, ePrint 2026/1838:
/// <https://eprint.iacr.org/2026/1838>.
pub fn canonical_r1cs_relation_digest(
    shape_canonical: &R1csShape<F>,
    num_cons_unpadded: usize,
    num_vars_unpadded: usize,
    num_io: usize,
) -> Result<[u8; 32], SpartanWhirError> {
    shape_canonical.validate()?;

    let mut hasher = Sha256::new();
    relation_digest_bytes(&mut hasher, R1CS_RELATION_DIGEST_DOMAIN)?;
    hasher.update(R1CS_RELATION_DIGEST_VERSION.to_le_bytes());
    relation_digest_bytes(&mut hasher, R1CS_RELATION_FIELD_ID)?;
    hasher.update((F::ORDER_U32 as u64).to_le_bytes());
    relation_digest_bytes(&mut hasher, R1CS_RELATION_COLUMN_LAYOUT_ID)?;

    for dimension in [
        num_cons_unpadded,
        num_vars_unpadded,
        num_io,
        shape_canonical.num_cons,
        shape_canonical.num_vars,
        shape_canonical.num_io,
    ] {
        relation_digest_usize(&mut hasher, dimension)?;
    }

    for (tag, matrix) in [
        (b"A".as_slice(), &shape_canonical.a),
        (b"B".as_slice(), &shape_canonical.b),
        (b"C".as_slice(), &shape_canonical.c),
    ] {
        relation_digest_matrix(&mut hasher, tag, matrix)?;
    }

    Ok(hasher.finalize().into())
}

fn relation_digest_matrix(
    hasher: &mut Sha256,
    tag: &[u8],
    matrix: &SparseMatrix<F>,
) -> Result<(), SpartanWhirError> {
    relation_digest_bytes(hasher, tag)?;
    relation_digest_usize(hasher, matrix.num_rows)?;
    relation_digest_usize(hasher, matrix.num_cols)?;

    if matrix.entries.iter().all(|entry| entry.val != F::ZERO)
        && matrix
            .entries
            .windows(2)
            .all(|entries| (entries[0].row, entries[0].col) < (entries[1].row, entries[1].col))
    {
        relation_digest_usize(hasher, matrix.entries.len())?;
        for entry in &matrix.entries {
            relation_digest_entry(hasher, entry.row, entry.col, entry.val)?;
        }
        return Ok(());
    }

    let mut entries = matrix.entries.iter().collect::<Vec<_>>();
    entries.sort_unstable_by_key(|entry| (entry.row, entry.col));
    let canonical_len = canonical_matrix_entries(&entries)
        .filter(|(_, _, value)| *value != F::ZERO)
        .count();
    relation_digest_usize(hasher, canonical_len)?;
    for (row, col, value) in canonical_matrix_entries(&entries) {
        if value != F::ZERO {
            relation_digest_entry(hasher, row, col, value)?;
        }
    }
    Ok(())
}

fn canonical_matrix_entries<'a>(
    entries: &'a [&'a SparseMatEntry<F>],
) -> impl Iterator<Item = (usize, usize, F)> + 'a {
    let mut index = 0;
    core::iter::from_fn(move || {
        let first = entries.get(index)?;
        let (row, col) = (first.row, first.col);
        let mut value = F::ZERO;
        while let Some(entry) = entries.get(index) {
            if (entry.row, entry.col) != (row, col) {
                break;
            }
            value += entry.val;
            index += 1;
        }
        Some((row, col, value))
    })
}

fn relation_digest_entry(
    hasher: &mut Sha256,
    row: usize,
    col: usize,
    value: F,
) -> Result<(), SpartanWhirError> {
    relation_digest_usize(hasher, row)?;
    relation_digest_usize(hasher, col)?;
    hasher.update(value.as_canonical_u32().to_le_bytes());
    Ok(())
}

fn relation_digest_bytes(hasher: &mut Sha256, bytes: &[u8]) -> Result<(), SpartanWhirError> {
    relation_digest_usize(hasher, bytes.len())?;
    hasher.update(bytes);
    Ok(())
}

fn relation_digest_usize(hasher: &mut Sha256, value: usize) -> Result<(), SpartanWhirError> {
    let value = u64::try_from(value).map_err(|_| SpartanWhirError::InvalidR1csShape)?;
    hasher.update(value.to_le_bytes());
    Ok(())
}

impl<F> R1csShape<F> {
    pub fn validate(&self) -> Result<(), SpartanWhirError> {
        let _profile = crate::profiling::profile_scope("r1cs_shape_validate");

        if self.num_cons == 0 {
            return Err(SpartanWhirError::InvalidR1csShape);
        }

        let expected_cols = self
            .num_vars
            .checked_add(self.num_io)
            .and_then(|n| n.checked_add(1))
            .ok_or(SpartanWhirError::InvalidR1csShape)?;

        validate_matrix_dimensions(&self.a, self.num_cons, expected_cols)?;
        validate_matrix_dimensions(&self.b, self.num_cons, expected_cols)?;
        validate_matrix_dimensions(&self.c, self.num_cons, expected_cols)?;

        validate_matrix_entries(&self.a)?;
        validate_matrix_entries(&self.b)?;
        validate_matrix_entries(&self.c)?;

        Ok(())
    }

    pub fn pad_regular(&self) -> Result<Self, SpartanWhirError>
    where
        F: Clone,
    {
        self.validate()?;

        let num_vars_target = max(
            self.num_vars,
            self.num_io
                .checked_add(1)
                .ok_or(SpartanWhirError::PaddingError)?,
        );
        let num_vars_padded = num_vars_target
            .checked_next_power_of_two()
            .ok_or(SpartanWhirError::PaddingError)?;
        let num_cons_padded = self
            .num_cons
            .checked_next_power_of_two()
            .ok_or(SpartanWhirError::PaddingError)?;

        if self.num_io >= num_vars_padded {
            return Err(SpartanWhirError::PaddingError);
        }

        let vars_delta = num_vars_padded.saturating_sub(self.num_vars);
        let num_cols_padded = num_vars_padded
            .checked_add(self.num_io)
            .and_then(|n| n.checked_add(1))
            .ok_or(SpartanWhirError::PaddingError)?;

        let pad_matrix = |m: &SparseMatrix<F>| -> SparseMatrix<F> {
            let mut entries = m.entries.clone();
            if vars_delta > 0 {
                for entry in &mut entries {
                    // Preserve layout [W | 1 | X] while adding witness columns.
                    if entry.col >= self.num_vars {
                        entry.col += vars_delta;
                    }
                }
            }
            SparseMatrix {
                num_rows: num_cons_padded,
                num_cols: num_cols_padded,
                entries,
            }
        };

        Ok(Self {
            num_cons: num_cons_padded,
            num_vars: num_vars_padded,
            num_io: self.num_io,
            a: pad_matrix(&self.a),
            b: pad_matrix(&self.b),
            c: pad_matrix(&self.c),
        })
    }

    pub(crate) fn direct_bind_layout(&self) -> Result<DirectBindLayout<F>, SpartanWhirError>
    where
        F: Copy + PartialEq,
    {
        let _profile = crate::profiling::profile_scope("direct_bind_layout");
        self.validate()?;
        DirectBindLayout::new(self)
    }

    pub(crate) fn direct_multiply_layout(&self) -> Result<DirectMultiplyLayout, SpartanWhirError> {
        let _profile = crate::profiling::profile_scope("direct_multiply_layout");
        self.validate()?;
        DirectMultiplyLayout::new(self)
    }
}

impl<F: Field> R1csShape<F> {
    pub fn multiply_vec(
        &self,
        z: &[F],
    ) -> Result<(Evaluations<F>, Evaluations<F>, Evaluations<F>), SpartanWhirError> {
        self.validate()?;
        self.multiply_vec_unchecked(z)
    }

    pub(crate) fn multiply_vec_unchecked(
        &self,
        z: &[F],
    ) -> Result<(Evaluations<F>, Evaluations<F>, Evaluations<F>), SpartanWhirError> {
        self.validate_matrix_vector_input_len(z)?;
        let az = multiply_sparse_matrix_vector(&self.a, z)?;
        let bz = multiply_sparse_matrix_vector(&self.b, z)?;
        let cz = multiply_sparse_matrix_vector(&self.c, z)?;
        Ok((az, bz, cz))
    }

    pub fn multiply_vec_parallel(
        &self,
        z: &[F],
    ) -> Result<(Evaluations<F>, Evaluations<F>, Evaluations<F>), SpartanWhirError>
    where
        F: Send + Sync,
    {
        self.validate()?;
        self.multiply_vec_parallel_unchecked(z)
    }

    pub(crate) fn multiply_vec_parallel_unchecked(
        &self,
        z: &[F],
    ) -> Result<(Evaluations<F>, Evaluations<F>, Evaluations<F>), SpartanWhirError>
    where
        F: Send + Sync,
    {
        self.validate_matrix_vector_input_len(z)?;

        let total_nnz = self
            .a
            .nnz()
            .checked_add(self.b.nnz())
            .and_then(|n| n.checked_add(self.c.nnz()))
            .ok_or(SpartanWhirError::InvalidR1csShape)?;

        if !(cfg!(feature = "parallel") && total_nnz >= R1CS_PARALLEL_MATRIX_MIN_NNZ) {
            return self.multiply_vec_unchecked(z);
        }

        let (az, (bz, cz)) = join(
            || multiply_sparse_matrix_vector(&self.a, z),
            || {
                join(
                    || multiply_sparse_matrix_vector(&self.b, z),
                    || multiply_sparse_matrix_vector(&self.c, z),
                )
            },
        );
        Ok((az?, bz?, cz?))
    }

    pub(crate) fn multiply_vec_parallel_with_layout_unchecked(
        &self,
        layout: &DirectMultiplyLayout,
        z: &[F],
    ) -> Result<(Evaluations<F>, Evaluations<F>, Evaluations<F>), SpartanWhirError>
    where
        F: Send + Sync,
    {
        self.validate_matrix_vector_input_len(z)?;

        let total_nnz = self
            .a
            .nnz()
            .checked_add(self.b.nnz())
            .and_then(|n| n.checked_add(self.c.nnz()))
            .ok_or(SpartanWhirError::InvalidR1csShape)?;

        if !(cfg!(feature = "parallel") && total_nnz >= R1CS_PARALLEL_MATRIX_MIN_NNZ) {
            return self.multiply_vec_unchecked(z);
        }

        let (az, (bz, cz)) = join(
            || multiply_sparse_matrix_vector_by_row(&self.a, &layout.a, z),
            || {
                join(
                    || multiply_sparse_matrix_vector_by_row(&self.b, &layout.b, z),
                    || multiply_sparse_matrix_vector_by_row(&self.c, &layout.c, z),
                )
            },
        );
        Ok((az?, bz?, cz?))
    }

    fn validate_matrix_vector_input_len(&self, z: &[F]) -> Result<(), SpartanWhirError> {
        let expected_len = self
            .num_vars
            .checked_add(self.num_io)
            .and_then(|n| n.checked_add(1))
            .ok_or(SpartanWhirError::InvalidWitnessLength)?;
        if z.len() != expected_len {
            return Err(SpartanWhirError::InvalidWitnessLength);
        }

        Ok(())
    }

    pub fn witness_to_mle(&self, witness: &[F]) -> Result<Evaluations<F>, SpartanWhirError> {
        self.validate()?;
        self.witness_to_mle_unchecked(witness)
    }

    pub(crate) fn witness_to_mle_unchecked(
        &self,
        witness: &[F],
    ) -> Result<Evaluations<F>, SpartanWhirError> {
        if witness.len() > self.num_vars {
            return Err(SpartanWhirError::InvalidWitnessLength);
        }

        let mut out = witness.to_vec();
        if out.len() < self.num_vars {
            out.resize(self.num_vars, F::ZERO);
        }
        Ok(out)
    }

    pub fn bind_row_vars<EF>(
        &self,
        eq_rx: &[EF],
    ) -> Result<(Vec<EF>, Vec<EF>, Vec<EF>), SpartanWhirError>
    where
        EF: ExtensionField<F>,
    {
        self.validate()?;

        if eq_rx.len() != self.num_cons {
            return Err(SpartanWhirError::InvalidWitnessLength);
        }

        let width = self
            .num_vars
            .checked_mul(2)
            .ok_or(SpartanWhirError::InvalidR1csShape)?;

        let mut a_evals = vec![EF::ZERO; width];
        let mut b_evals = vec![EF::ZERO; width];
        let mut c_evals = vec![EF::ZERO; width];

        accumulate_bound_rows(&self.a, eq_rx, &mut a_evals)?;
        accumulate_bound_rows(&self.b, eq_rx, &mut b_evals)?;
        accumulate_bound_rows(&self.c, eq_rx, &mut c_evals)?;

        Ok((a_evals, b_evals, c_evals))
    }

    pub fn bind_row_vars_joint<EF>(&self, eq_rx: &[EF], r: EF) -> Result<Vec<EF>, SpartanWhirError>
    where
        F: Send + Sync,
        EF: ExtensionField<F> + Send + Sync,
    {
        self.validate()?;

        if eq_rx.len() != self.num_cons {
            return Err(SpartanWhirError::InvalidWitnessLength);
        }

        let width = self
            .num_vars
            .checked_mul(2)
            .ok_or(SpartanWhirError::InvalidR1csShape)?;

        let mut out = vec![EF::ZERO; width];
        let r_squared = r * r;
        let (eq_b, eq_c) = join(
            || scale_eq_table(eq_rx, r),
            || scale_eq_table(eq_rx, r_squared),
        );

        accumulate_bound_rows_for_joint(&self.a, eq_rx, &mut out)?;
        accumulate_bound_rows_for_joint(&self.b, &eq_b, &mut out)?;
        accumulate_bound_rows_for_joint(&self.c, &eq_c, &mut out)?;

        Ok(out)
    }

    pub(crate) fn bind_row_vars_joint_with_layout_unchecked<EF>(
        &self,
        layout: &DirectBindLayout<F>,
        eq_rx: &[EF],
        r: EF,
    ) -> Result<Vec<EF>, SpartanWhirError>
    where
        F: Send + Sync,
        EF: ExtensionField<F> + Send + Sync,
    {
        if eq_rx.len() != self.num_cons {
            return Err(SpartanWhirError::InvalidWitnessLength);
        }

        let width = self
            .num_vars
            .checked_mul(2)
            .ok_or(SpartanWhirError::InvalidR1csShape)?;
        if layout.width != width {
            return Err(SpartanWhirError::InvalidR1csShape);
        }

        let mut out = vec![EF::ZERO; width];
        let r_squared = r * r;
        let (eq_b, eq_c) = {
            let _profile = crate::profiling::profile_scope("bind_row_scale_eq");
            join(
                || scale_eq_table(eq_rx, r),
                || scale_eq_table(eq_rx, r_squared),
            )
        };
        {
            let _profile = crate::profiling::profile_scope("bind_row_accumulate");
            accumulate_bound_rows_for_joint_column_layout(
                self, layout, eq_rx, &eq_b, &eq_c, &mut out,
            )?;
        }

        Ok(out)
    }

    pub(crate) fn bind_row_vars_joint_packed_with_layout_unchecked<EF>(
        &self,
        layout: &DirectBindLayout<F>,
        eq_rx: &[EF],
        r: EF,
    ) -> Result<Vec<EF::ExtensionPacking>, SpartanWhirError>
    where
        F: Send + Sync,
        EF: ExtensionField<F> + Send + Sync,
    {
        if eq_rx.len() != self.num_cons {
            return Err(SpartanWhirError::InvalidWitnessLength);
        }

        let width = self
            .num_vars
            .checked_mul(2)
            .ok_or(SpartanWhirError::InvalidR1csShape)?;
        if layout.width != width || !width.is_multiple_of(F::Packing::WIDTH) {
            return Err(SpartanWhirError::InvalidR1csShape);
        }

        let r_squared = r * r;
        let (eq_b, eq_c) = {
            let _profile = crate::profiling::profile_scope("bind_row_scale_eq");
            join(
                || scale_eq_table(eq_rx, r),
                || scale_eq_table(eq_rx, r_squared),
            )
        };
        let packed = {
            let _profile = crate::profiling::profile_scope("bind_row_accumulate");
            accumulate_bound_rows_for_joint_column_layout_packed(
                self, layout, eq_rx, &eq_b, &eq_c, width,
            )?
        };

        Ok(packed)
    }

    pub fn evaluate_with_tables<EF>(
        &self,
        t_x: &[EF],
        t_y: &[EF],
    ) -> Result<(EF, EF, EF), SpartanWhirError>
    where
        EF: ExtensionField<F>,
    {
        self.validate()?;

        if t_x.len() != self.num_cons || t_y.len() < self.a.num_cols {
            return Err(SpartanWhirError::InvalidWitnessLength);
        }

        Ok((
            evaluate_sparse_matrix_with_tables(&self.a, t_x, t_y)?,
            evaluate_sparse_matrix_with_tables(&self.b, t_x, t_y)?,
            evaluate_sparse_matrix_with_tables(&self.c, t_x, t_y)?,
        ))
    }
}

impl<F> DirectBindLayout<F>
where
    F: Copy + PartialEq,
{
    fn new(shape: &R1csShape<F>) -> Result<Self, SpartanWhirError> {
        let width = shape
            .num_vars
            .checked_mul(2)
            .ok_or(SpartanWhirError::InvalidR1csShape)?;

        let layout = Self {
            width,
            a: ColumnBindMatrixLayout::new(&shape.a),
            b: ColumnBindMatrixLayout::new(&shape.b),
            c: ColumnBindMatrixLayout::new(&shape.c),
        };
        layout.validate_for(shape, width)?;
        Ok(layout)
    }

    fn validate_for(&self, shape: &R1csShape<F>, width: usize) -> Result<(), SpartanWhirError> {
        if self.width != width {
            return Err(SpartanWhirError::InvalidR1csShape);
        }
        self.a.validate_for(&shape.a)?;
        self.b.validate_for(&shape.b)?;
        self.c.validate_for(&shape.c)?;
        Ok(())
    }
}

impl DirectMultiplyLayout {
    fn new<F>(shape: &R1csShape<F>) -> Result<Self, SpartanWhirError> {
        let layout = Self {
            a: RowMatrixLayout::new(&shape.a),
            b: RowMatrixLayout::new(&shape.b),
            c: RowMatrixLayout::new(&shape.c),
        };
        layout.validate_for(shape)?;
        Ok(layout)
    }

    fn validate_for<F>(&self, shape: &R1csShape<F>) -> Result<(), SpartanWhirError> {
        self.a.validate_for(&shape.a)?;
        self.b.validate_for(&shape.b)?;
        self.c.validate_for(&shape.c)?;
        Ok(())
    }
}

impl<F> ColumnBindMatrixLayout<F>
where
    F: Copy + PartialEq,
{
    fn new(mat: &SparseMatrix<F>) -> Self {
        let mut col_starts = vec![0usize; mat.num_cols + 1];
        for entry in &mat.entries {
            col_starts[entry.col + 1] += 1;
        }
        for col in 1..=mat.num_cols {
            col_starts[col] += col_starts[col - 1];
        }

        let mut next = col_starts.clone();
        let mut entry_indices_by_col = vec![0usize; mat.entries.len()];
        for (idx, entry) in mat.entries.iter().enumerate() {
            let pos = next[entry.col];
            entry_indices_by_col[pos] = idx;
            next[entry.col] += 1;
        }

        let entries_by_col = entry_indices_by_col
            .into_iter()
            .map(|idx| {
                let entry = &mat.entries[idx];
                ColumnBindEntry {
                    col: entry.col,
                    row: entry.row,
                    val: entry.val,
                }
            })
            .collect();

        Self {
            entries_by_col,
            col_starts,
        }
    }

    fn validate_for(&self, mat: &SparseMatrix<F>) -> Result<(), SpartanWhirError> {
        if self.entries_by_col.len() != mat.entries.len()
            || self.col_starts.len() != mat.num_cols + 1
        {
            return Err(SpartanWhirError::InvalidR1csShape);
        }

        if self.col_starts.first().copied() != Some(0)
            || self.col_starts.last().copied() != Some(mat.entries.len())
        {
            return Err(SpartanWhirError::InvalidR1csShape);
        }
        for window in self.col_starts.windows(2) {
            if window[0] > window[1] || window[1] > mat.entries.len() {
                return Err(SpartanWhirError::InvalidR1csShape);
            }
        }

        let mut next = self.col_starts[..mat.num_cols].to_vec();
        for entry in &mat.entries {
            let pos = next[entry.col];
            if pos >= self.col_starts[entry.col + 1] {
                return Err(SpartanWhirError::InvalidR1csShape);
            }
            let layout_entry = self.entries_by_col[pos];
            if layout_entry.col != entry.col
                || layout_entry.row != entry.row
                || layout_entry.val != entry.val
            {
                return Err(SpartanWhirError::InvalidR1csShape);
            }
            next[entry.col] += 1;
        }
        Ok(())
    }

    fn range_for_cols(&self, col_start: usize, col_end: usize) -> &[ColumnBindEntry<F>] {
        debug_assert!(col_start <= col_end);
        let col_count = self.col_starts.len().saturating_sub(1);
        let col_start = col_start.min(col_count);
        let col_end = col_end.min(col_count);
        &self.entries_by_col[self.col_starts[col_start]..self.col_starts[col_end]]
    }
}

impl RowMatrixLayout {
    fn new<F>(mat: &SparseMatrix<F>) -> Self {
        let mut entry_indices_by_row: Vec<usize> = (0..mat.entries.len()).collect();
        entry_indices_by_row.sort_unstable_by_key(|&idx| mat.entries[idx].row);

        let mut row_starts = vec![0usize; mat.num_rows + 1];
        let mut pos = 0usize;
        for (row, start) in row_starts.iter_mut().take(mat.num_rows).enumerate() {
            *start = pos;
            while pos < entry_indices_by_row.len()
                && mat.entries[entry_indices_by_row[pos]].row == row
            {
                pos += 1;
            }
        }
        row_starts[mat.num_rows] = entry_indices_by_row.len();

        Self {
            entry_indices_by_row,
            row_starts,
        }
    }

    fn validate_for<F>(&self, mat: &SparseMatrix<F>) -> Result<(), SpartanWhirError> {
        if self.entry_indices_by_row.len() != mat.entries.len()
            || self.row_starts.len() != mat.num_rows + 1
        {
            return Err(SpartanWhirError::InvalidR1csShape);
        }

        let mut seen = vec![false; mat.entries.len()];
        let mut prev_row = None;
        for &idx in &self.entry_indices_by_row {
            if idx >= mat.entries.len() || seen[idx] {
                return Err(SpartanWhirError::InvalidR1csShape);
            }
            seen[idx] = true;

            let row = mat.entries[idx].row;
            if prev_row.is_some_and(|prev| row < prev) {
                return Err(SpartanWhirError::InvalidR1csShape);
            }
            prev_row = Some(row);
        }

        if self.row_starts.first().copied() != Some(0)
            || self.row_starts.last().copied() != Some(mat.entries.len())
        {
            return Err(SpartanWhirError::InvalidR1csShape);
        }
        for window in self.row_starts.windows(2) {
            if window[0] > window[1] || window[1] > mat.entries.len() {
                return Err(SpartanWhirError::InvalidR1csShape);
            }
        }
        for row in 0..mat.num_rows {
            for &idx in self.range_for_row(row) {
                if mat.entries[idx].row != row {
                    return Err(SpartanWhirError::InvalidR1csShape);
                }
            }
        }
        Ok(())
    }

    fn range_for_row(&self, row: usize) -> &[usize] {
        &self.entry_indices_by_row[self.row_starts[row]..self.row_starts[row + 1]]
    }
}

fn validate_matrix_dimensions<F>(
    mat: &SparseMatrix<F>,
    expected_rows: usize,
    expected_cols: usize,
) -> Result<(), SpartanWhirError> {
    if mat.num_rows != expected_rows || mat.num_cols != expected_cols {
        return Err(SpartanWhirError::InvalidR1csShape);
    }
    Ok(())
}

fn validate_matrix_entries<F>(mat: &SparseMatrix<F>) -> Result<(), SpartanWhirError> {
    for entry in &mat.entries {
        if entry.row >= mat.num_rows || entry.col >= mat.num_cols {
            return Err(SpartanWhirError::InvalidR1csShape);
        }
    }
    Ok(())
}

fn multiply_sparse_matrix_vector<F: Field>(
    mat: &SparseMatrix<F>,
    z: &[F],
) -> Result<Vec<F>, SpartanWhirError> {
    if z.len() != mat.num_cols {
        return Err(SpartanWhirError::InvalidWitnessLength);
    }

    let mut out = vec![F::ZERO; mat.num_rows];
    for entry in &mat.entries {
        add_base_scaled(&mut out[entry.row], z[entry.col], entry.val);
    }
    Ok(out)
}

fn multiply_sparse_matrix_vector_by_row<F>(
    mat: &SparseMatrix<F>,
    layout: &RowMatrixLayout,
    z: &[F],
) -> Result<Vec<F>, SpartanWhirError>
where
    F: Field + Send + Sync,
{
    if z.len() != mat.num_cols {
        return Err(SpartanWhirError::InvalidWitnessLength);
    }

    let mut out = vec![F::ZERO; mat.num_rows];
    let rows_per_chunk = 1024usize;
    out.par_chunks_mut(rows_per_chunk)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let row_start = chunk_idx * rows_per_chunk;
            for (offset, out) in out_chunk.iter_mut().enumerate() {
                let row = row_start + offset;
                let mut acc = F::ZERO;
                for &entry_idx in layout.range_for_row(row) {
                    let entry = &mat.entries[entry_idx];
                    add_base_scaled(&mut acc, z[entry.col], entry.val);
                }
                *out = acc;
            }
        });

    Ok(out)
}

fn accumulate_bound_rows<F, EF>(
    mat: &SparseMatrix<F>,
    eq_rx: &[EF],
    out: &mut [EF],
) -> Result<(), SpartanWhirError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    if out.len() < mat.num_cols {
        return Err(SpartanWhirError::InvalidR1csShape);
    }

    for entry in &mat.entries {
        add_extension_scaled(&mut out[entry.col], eq_rx[entry.row], entry.val);
    }
    Ok(())
}

fn scale_eq_table<EF>(eq_rx: &[EF], scale: EF) -> Vec<EF>
where
    EF: Field + Send + Sync,
{
    if cfg!(feature = "parallel") && eq_rx.len() >= R1CS_PARALLEL_BIND_MIN_NNZ {
        eq_rx.par_iter().map(|&v| scale * v).collect()
    } else {
        eq_rx.iter().map(|&v| scale * v).collect()
    }
}

fn accumulate_bound_rows_for_joint<F, EF>(
    mat: &SparseMatrix<F>,
    eq_rx: &[EF],
    out: &mut [EF],
) -> Result<(), SpartanWhirError>
where
    F: Field + Send + Sync,
    EF: ExtensionField<F> + Send + Sync,
{
    if out.len() < mat.num_cols {
        return Err(SpartanWhirError::InvalidR1csShape);
    }

    if !(cfg!(feature = "parallel") && mat.entries.len() >= R1CS_PARALLEL_BIND_MIN_NNZ) {
        return accumulate_bound_rows(mat, eq_rx, out);
    }

    let shard_count = current_num_threads()
        .min(R1CS_PARALLEL_BIND_FALLBACK_MAX_SHARDS)
        .min(mat.entries.len())
        .max(1);
    let chunk_len = mat.entries.len().div_ceil(shard_count);
    let out_len = out.len();
    let partials: Vec<Vec<EF>> = mat
        .entries
        .par_chunks(chunk_len)
        .map(|entries| {
            let mut local = vec![EF::ZERO; out_len];
            for entry in entries {
                add_extension_scaled(&mut local[entry.col], eq_rx[entry.row], entry.val);
            }
            local
        })
        .collect();

    for partial in partials {
        for (dst, value) in out.iter_mut().zip(partial) {
            *dst += value;
        }
    }

    Ok(())
}

fn accumulate_bound_rows_for_joint_column_layout<F, EF>(
    shape: &R1csShape<F>,
    layout: &DirectBindLayout<F>,
    eq_a: &[EF],
    eq_b: &[EF],
    eq_c: &[EF],
    out: &mut [EF],
) -> Result<(), SpartanWhirError>
where
    F: Field + Send + Sync,
    EF: ExtensionField<F> + Send + Sync,
{
    let total_nnz = shape
        .a
        .nnz()
        .checked_add(shape.b.nnz())
        .and_then(|n| n.checked_add(shape.c.nnz()))
        .ok_or(SpartanWhirError::InvalidR1csShape)?;

    if !(cfg!(feature = "parallel") && total_nnz >= R1CS_PARALLEL_BIND_MIN_NNZ) {
        accumulate_bound_rows_for_column_range(&layout.a, eq_a, 0, out.len(), out);
        accumulate_bound_rows_for_column_range(&layout.b, eq_b, 0, out.len(), out);
        accumulate_bound_rows_for_column_range(&layout.c, eq_c, 0, out.len(), out);
        return Ok(());
    }

    let shard_count = current_num_threads()
        .saturating_mul(R1CS_PARALLEL_BIND_CHUNKS_PER_THREAD)
        .min(out.len())
        .max(1);
    let chunk_len = out.len().div_ceil(shard_count).max(1);

    out.par_chunks_mut(chunk_len)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let col_start = chunk_idx * chunk_len;
            let col_end = col_start + out_chunk.len();

            accumulate_bound_rows_for_column_range(&layout.a, eq_a, col_start, col_end, out_chunk);
            accumulate_bound_rows_for_column_range(&layout.b, eq_b, col_start, col_end, out_chunk);
            accumulate_bound_rows_for_column_range(&layout.c, eq_c, col_start, col_end, out_chunk);
        });

    Ok(())
}

fn accumulate_bound_rows_for_joint_column_layout_packed<F, EF>(
    shape: &R1csShape<F>,
    layout: &DirectBindLayout<F>,
    eq_a: &[EF],
    eq_b: &[EF],
    eq_c: &[EF],
    out_len: usize,
) -> Result<Vec<EF::ExtensionPacking>, SpartanWhirError>
where
    F: Field + Send + Sync,
    EF: ExtensionField<F> + Send + Sync,
{
    let packing_width = F::Packing::WIDTH;
    if !out_len.is_multiple_of(packing_width) {
        return Err(SpartanWhirError::InvalidR1csShape);
    }

    let total_nnz = shape
        .a
        .nnz()
        .checked_add(shape.b.nnz())
        .and_then(|n| n.checked_add(shape.c.nnz()))
        .ok_or(SpartanWhirError::InvalidR1csShape)?;
    let packed_len = out_len / packing_width;
    let mut out = EF::ExtensionPacking::zero_vec(packed_len);
    let fill_chunk = |packed_start: usize, out_chunk: &mut [EF::ExtensionPacking]| {
        let mut scratch = vec![EF::ZERO; packing_width];
        for (offset, dst) in out_chunk.iter_mut().enumerate() {
            scratch.fill(EF::ZERO);
            let col_start = (packed_start + offset) * packing_width;
            let col_end = col_start + packing_width;
            accumulate_bound_rows_for_column_range(
                &layout.a,
                eq_a,
                col_start,
                col_end,
                &mut scratch,
            );
            accumulate_bound_rows_for_column_range(
                &layout.b,
                eq_b,
                col_start,
                col_end,
                &mut scratch,
            );
            accumulate_bound_rows_for_column_range(
                &layout.c,
                eq_c,
                col_start,
                col_end,
                &mut scratch,
            );
            *dst = EF::ExtensionPacking::from_ext_slice(&scratch);
        }
    };

    if cfg!(feature = "parallel") && total_nnz >= R1CS_PARALLEL_BIND_MIN_NNZ {
        let shard_count = current_num_threads()
            .saturating_mul(R1CS_PARALLEL_BIND_CHUNKS_PER_THREAD)
            .min(packed_len)
            .max(1);
        let chunk_len = packed_len.div_ceil(shard_count).max(1);
        out.par_chunks_mut(chunk_len)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| fill_chunk(chunk_idx * chunk_len, out_chunk));
    } else {
        fill_chunk(0, &mut out);
    }

    Ok(out)
}

fn accumulate_bound_rows_for_column_range<F, EF>(
    layout: &ColumnBindMatrixLayout<F>,
    eq_rx: &[EF],
    col_start: usize,
    col_end: usize,
    out_chunk: &mut [EF],
) where
    F: Field,
    EF: ExtensionField<F>,
{
    for entry in layout.range_for_cols(col_start, col_end) {
        add_extension_scaled(
            &mut out_chunk[entry.col - col_start],
            eq_rx[entry.row],
            entry.val,
        );
    }
}

fn evaluate_sparse_matrix_with_tables<F, EF>(
    mat: &SparseMatrix<F>,
    t_x: &[EF],
    t_y: &[EF],
) -> Result<EF, SpartanWhirError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    if t_x.len() != mat.num_rows || t_y.len() < mat.num_cols {
        return Err(SpartanWhirError::InvalidWitnessLength);
    }

    let mut acc = EF::ZERO;
    for entry in &mat.entries {
        add_extension_scaled(&mut acc, t_x[entry.row] * t_y[entry.col], entry.val);
    }
    Ok(acc)
}

fn add_base_scaled<F: Field>(dst: &mut F, value: F, scale: F) {
    if scale.is_one() {
        *dst += value;
    } else if scale == -F::ONE {
        *dst -= value;
    } else {
        *dst += scale * value;
    }
}

fn add_extension_scaled<F, EF>(dst: &mut EF, value: EF, scale: F)
where
    F: Field,
    EF: ExtensionField<F>,
{
    if scale.is_one() {
        *dst += value;
    } else if scale == -F::ONE {
        *dst -= value;
    } else {
        *dst += value * scale;
    }
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;

    use super::*;
    use crate::{EqPolynomial, QuarticBinExtension as EF};

    fn digest_test_shape() -> R1csShape<F> {
        R1csShape {
            num_cons: 2,
            num_vars: 2,
            num_io: 0,
            a: SparseMatrix {
                num_rows: 2,
                num_cols: 3,
                entries: vec![SparseMatEntry {
                    row: 0,
                    col: 0,
                    val: F::from_u32(3),
                }],
            },
            b: SparseMatrix {
                num_rows: 2,
                num_cols: 3,
                entries: vec![SparseMatEntry {
                    row: 0,
                    col: 1,
                    val: F::from_u32(5),
                }],
            },
            c: SparseMatrix {
                num_rows: 2,
                num_cols: 3,
                entries: vec![SparseMatEntry {
                    row: 0,
                    col: 2,
                    val: F::from_u32(15),
                }],
            },
        }
    }

    #[test]
    fn relation_digest_has_a_stable_encoding() {
        let digest = canonical_r1cs_relation_digest(&digest_test_shape(), 2, 2, 0)
            .expect("valid shape hashes");

        assert_eq!(
            digest,
            [
                201, 22, 54, 128, 54, 224, 52, 41, 209, 185, 15, 138, 87, 234, 221, 180, 10, 97,
                89, 226, 32, 49, 56, 235, 131, 55, 43, 233, 49, 174, 104, 235,
            ]
        );
    }

    #[test]
    fn relation_digest_canonicalizes_sparse_matrix_entries() {
        let shape = digest_test_shape();
        let mut equivalent = shape.clone();
        equivalent.a.entries = vec![
            SparseMatEntry {
                row: 1,
                col: 1,
                val: F::from_u32(7),
            },
            SparseMatEntry {
                row: 0,
                col: 0,
                val: F::ONE,
            },
            SparseMatEntry {
                row: 1,
                col: 1,
                val: -F::from_u32(7),
            },
            SparseMatEntry {
                row: 0,
                col: 2,
                val: F::ZERO,
            },
            SparseMatEntry {
                row: 0,
                col: 0,
                val: F::from_u32(2),
            },
        ];

        assert_eq!(
            canonical_r1cs_relation_digest(&shape, 2, 2, 0).expect("valid shape hashes"),
            canonical_r1cs_relation_digest(&equivalent, 2, 2, 0).expect("equivalent shape hashes")
        );
    }

    #[test]
    fn relation_digest_changes_with_relation_and_matrix_tag() {
        let shape = digest_test_shape();
        let expected = canonical_r1cs_relation_digest(&shape, 2, 2, 0).expect("valid shape hashes");

        let mut changed_value = shape.clone();
        changed_value.a.entries[0].val += F::ONE;
        assert_ne!(
            expected,
            canonical_r1cs_relation_digest(&changed_value, 2, 2, 0).expect("changed shape hashes")
        );

        let mut changed_matrix = shape.clone();
        changed_matrix
            .b
            .entries
            .push(changed_matrix.a.entries.remove(0));
        assert_ne!(
            expected,
            canonical_r1cs_relation_digest(&changed_matrix, 2, 2, 0)
                .expect("retagged shape hashes")
        );
    }

    #[test]
    fn relation_digest_binds_original_and_padded_dimensions() {
        let shape = digest_test_shape();
        let expected = canonical_r1cs_relation_digest(&shape, 2, 2, 0).expect("valid shape hashes");
        assert_ne!(
            expected,
            canonical_r1cs_relation_digest(&shape, 1, 2, 0)
                .expect("different original dimensions hash")
        );

        let mut padded = shape.clone();
        padded.num_cons = 4;
        padded.a.num_rows = 4;
        padded.b.num_rows = 4;
        padded.c.num_rows = 4;
        assert_ne!(
            expected,
            canonical_r1cs_relation_digest(&padded, 2, 2, 0)
                .expect("different padded dimensions hash")
        );
    }

    #[test]
    fn relation_digest_rejects_invalid_shape() {
        let mut shape = digest_test_shape();
        shape.a.entries[0].row = shape.num_cons;

        assert_eq!(
            canonical_r1cs_relation_digest(&shape, 2, 2, 0),
            Err(SpartanWhirError::InvalidR1csShape)
        );
    }

    fn repeated_entries(seed: usize, num_rows: usize, num_cols: usize) -> Vec<SparseMatEntry<F>> {
        let entry_count = (R1CS_PARALLEL_BIND_MIN_NNZ / 3) + 257;
        (0..entry_count)
            .map(|idx| {
                let val = if idx % 5 == 0 {
                    F::ONE
                } else {
                    F::from_u32(((idx + seed) % 17) as u32 + 2)
                };
                SparseMatEntry {
                    row: (idx * 7 + seed) % num_rows,
                    col: (idx * 11 + seed * 3) % num_cols,
                    val,
                }
            })
            .collect()
    }

    #[test]
    fn direct_bind_layout_matches_joint_bind() {
        let num_cons = 64;
        let num_vars = 64;
        let num_io = 0;
        let num_cols = num_vars + num_io + 1;
        let shape = R1csShape {
            num_cons,
            num_vars,
            num_io,
            a: SparseMatrix {
                num_rows: num_cons,
                num_cols,
                entries: repeated_entries(1, num_cons, num_cols),
            },
            b: SparseMatrix {
                num_rows: num_cons,
                num_cols,
                entries: repeated_entries(2, num_cons, num_cols),
            },
            c: SparseMatrix {
                num_rows: num_cons,
                num_cols,
                entries: repeated_entries(3, num_cons, num_cols),
            },
        };
        let point: Vec<EF> = [2, 3, 5, 7, 11, 13]
            .into_iter()
            .map(|v| EF::from(F::from_u32(v)))
            .collect();
        let eq_rx = EqPolynomial::evals_from_point_with_base::<F>(&point);
        let r = EF::from(F::from_u32(19));

        let layout = shape.direct_bind_layout().expect("layout builds");
        let expected = shape
            .bind_row_vars_joint(&eq_rx, r)
            .expect("joint bind succeeds");
        let actual = shape
            .bind_row_vars_joint_with_layout_unchecked(&layout, &eq_rx, r)
            .expect("layout bind succeeds");
        let packed = shape
            .bind_row_vars_joint_packed_with_layout_unchecked(&layout, &eq_rx, r)
            .expect("packed layout bind succeeds");
        let unpacked = p3_multilinear_util::poly::Poly::new(packed)
            .unpack::<F, EF>()
            .into_evals();

        assert_eq!(actual, expected);
        assert_eq!(unpacked, expected);
    }

    #[test]
    fn direct_multiply_layout_matches_matrix_multiply() {
        let num_cons = 64;
        let num_vars = 64;
        let num_io = 0;
        let num_cols = num_vars + num_io + 1;
        let shape = R1csShape {
            num_cons,
            num_vars,
            num_io,
            a: SparseMatrix {
                num_rows: num_cons,
                num_cols,
                entries: repeated_entries(1, num_cons, num_cols),
            },
            b: SparseMatrix {
                num_rows: num_cons,
                num_cols,
                entries: repeated_entries(2, num_cons, num_cols),
            },
            c: SparseMatrix {
                num_rows: num_cons,
                num_cols,
                entries: repeated_entries(3, num_cons, num_cols),
            },
        };
        let z: Vec<F> = (0..num_cols)
            .map(|idx| F::from_u32((idx % 23) as u32 + 1))
            .collect();

        let layout = shape.direct_multiply_layout().expect("layout builds");
        let expected = shape
            .multiply_vec_parallel_unchecked(&z)
            .expect("multiply succeeds");
        let actual = shape
            .multiply_vec_parallel_with_layout_unchecked(&layout, &z)
            .expect("layout multiply succeeds");

        assert_eq!(actual, expected);
    }

    #[test]
    fn direct_bind_layout_validation_rejects_bad_entries() {
        let shape = R1csShape {
            num_cons: 2,
            num_vars: 2,
            num_io: 0,
            a: SparseMatrix {
                num_rows: 2,
                num_cols: 3,
                entries: vec![
                    SparseMatEntry {
                        row: 0,
                        col: 1,
                        val: F::ONE,
                    },
                    SparseMatEntry {
                        row: 1,
                        col: 0,
                        val: F::ONE,
                    },
                ],
            },
            b: SparseMatrix {
                num_rows: 2,
                num_cols: 3,
                entries: vec![],
            },
            c: SparseMatrix {
                num_rows: 2,
                num_cols: 3,
                entries: vec![],
            },
        };
        let mut layout = shape.direct_bind_layout().expect("layout builds");
        layout.a.entries_by_col[0].row = 0;

        assert_eq!(
            layout.validate_for(&shape, 4),
            Err(SpartanWhirError::InvalidR1csShape)
        );

        let mut layout = shape.direct_bind_layout().expect("layout builds");
        layout.a.col_starts = vec![0, 2, 2, 2];
        assert_eq!(
            layout.validate_for(&shape, 4),
            Err(SpartanWhirError::InvalidR1csShape)
        );
    }

    #[test]
    fn direct_multiply_layout_validation_rejects_bad_row_ranges() {
        let shape = R1csShape {
            num_cons: 2,
            num_vars: 2,
            num_io: 0,
            a: SparseMatrix {
                num_rows: 2,
                num_cols: 3,
                entries: vec![
                    SparseMatEntry {
                        row: 0,
                        col: 1,
                        val: F::ONE,
                    },
                    SparseMatEntry {
                        row: 1,
                        col: 0,
                        val: F::ONE,
                    },
                ],
            },
            b: SparseMatrix {
                num_rows: 2,
                num_cols: 3,
                entries: vec![],
            },
            c: SparseMatrix {
                num_rows: 2,
                num_cols: 3,
                entries: vec![],
            },
        };
        let mut layout = shape.direct_multiply_layout().expect("layout builds");
        layout.a.row_starts = vec![0, 0, 2];

        assert_eq!(
            layout.validate_for(&shape),
            Err(SpartanWhirError::InvalidR1csShape)
        );
    }
}
