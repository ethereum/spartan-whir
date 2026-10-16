//! A field-native Privacy Pools-style partial-withdrawal benchmark.
//!
//! The relation uses two full depth-32 binary Merkle paths, a persistent private
//! association label, fresh change-note secrets, and 128-bit integer amounts.
//! It is a benchmark relation, not the deployed Privacy Pools byte encoding or
//! LeanIMT implementation. The application must authenticate both public roots,
//! pin its pool and scope identifiers, reject spent nullifiers, and execute the
//! public recipient/amount/fee fields.

pub mod config;

use std::collections::BTreeMap;

use p3_field::{Field, PrimeCharacteristicRing, PrimeField32};
use p3_koala_bear::{
    default_koalabear_poseidon2_24, KOALABEAR_POSEIDON2_RC_24_EXTERNAL_FINAL,
    KOALABEAR_POSEIDON2_RC_24_EXTERNAL_INITIAL, KOALABEAR_POSEIDON2_RC_24_INTERNAL,
};
use p3_symmetric::Permutation;

use crate::{engine::F, R1csShape, R1csWitness, SparseMatEntry, SparseMatrix};

pub const WITHDRAWAL_TREE_DEPTH: usize = 32;
pub const WITHDRAWAL_DIGEST_ELEMS: usize = 8;
pub const WITHDRAWAL_HASH_PROFILE: &str = "koalabear-poseidon2-width24-rate16-capacity8-digest8-v1";
pub type WithdrawalDigest = [F; WITHDRAWAL_DIGEST_ELEMS];

const PRECOMMIT: u32 = 1;
const NOTE: u32 = 2;
const NULLIFIER: u32 = 3;
const POOL_NODE: u32 = 4;
const ASSOCIATION_NODE: u32 = 5;
const TRANSACTION: u32 = 6;

#[derive(Clone, Debug)]
pub struct WithdrawalWitness {
    pub balance: u128,
    pub change: u128,
    pub label: WithdrawalDigest,
    pub nullifier: WithdrawalDigest,
    pub secret: WithdrawalDigest,
    pub new_nullifier: WithdrawalDigest,
    pub new_secret: WithdrawalDigest,
    pub pool_siblings: [WithdrawalDigest; WITHDRAWAL_TREE_DEPTH],
    pub pool_index: u32,
    pub association_siblings: [WithdrawalDigest; WITHDRAWAL_TREE_DEPTH],
    pub association_index: u32,
}

#[derive(Clone, Debug)]
pub struct WithdrawalStatement {
    pub pool: WithdrawalDigest,
    pub scope: WithdrawalDigest,
    pub context: WithdrawalDigest,
    /// Little-endian 16-bit limbs of a 160-bit recipient identifier.
    pub recipient: [u16; 10],
    /// Amount sent to the recipient, excluding the separately charged fee.
    pub withdrawn: u128,
    pub fee: u128,
    pub pool_root: WithdrawalDigest,
    pub association_root: WithdrawalDigest,
    pub nullifier_hash: WithdrawalDigest,
    pub change_commitment: WithdrawalDigest,
    pub transaction_digest: WithdrawalDigest,
}

impl WithdrawalStatement {
    /// The public field order consumed by the ordinary Spartan statement API.
    pub fn public_inputs(&self) -> Vec<F> {
        let mut out = self.transaction_fields();
        out.extend(self.transaction_digest);
        out
    }

    fn transaction_fields(&self) -> Vec<F> {
        let mut out = Vec::new();
        for digest in [self.pool, self.scope, self.context] {
            out.extend(digest);
        }
        out.extend(self.recipient.map(|x| F::from_u16(x)));
        out.extend(amount_limbs(self.withdrawn));
        out.extend(amount_limbs(self.fee));
        for digest in [
            self.pool_root,
            self.association_root,
            self.nullifier_hash,
            self.change_commitment,
        ] {
            out.extend(digest);
        }
        out
    }

    /// Recompute derived public values without checking the balance equation.
    /// Keeping that check inside R1CS permits adversarial witness tests.
    pub fn derive(&mut self, witness: &WithdrawalWitness) {
        let old_note = note_commitment(
            self.pool,
            witness.label,
            witness.balance,
            witness.nullifier,
            witness.secret,
        );
        self.pool_root = merkle_root(
            POOL_NODE,
            old_note,
            &witness.pool_siblings,
            witness.pool_index,
        );
        self.association_root = merkle_root(
            ASSOCIATION_NODE,
            witness.label,
            &witness.association_siblings,
            witness.association_index,
        );
        // Scope does not change a note's nullifier. The same note cannot acquire
        // another nullifier merely by selecting different withdrawal metadata.
        self.nullifier_hash = hash_fields(NULLIFIER, &[self.pool, witness.nullifier].concat());
        self.change_commitment = note_commitment(
            self.pool,
            witness.label,
            witness.change,
            witness.new_nullifier,
            witness.new_secret,
        );
        self.transaction_digest = hash_fields(TRANSACTION, &self.transaction_fields());
    }
}

#[derive(Clone, Debug)]
pub struct WithdrawalCircuitStats {
    pub constraints: usize,
    pub witness_variables: usize,
    pub public_inputs: usize,
    pub poseidon_permutations: usize,
    pub matrix_nonzeros: usize,
}

#[derive(Clone, Debug)]
pub struct WithdrawalCircuit {
    pub shape: R1csShape<F>,
    pub witness: R1csWitness<F>,
    pub public_inputs: Vec<F>,
    pub stats: WithdrawalCircuitStats,
}

/// Poseidon2 sponge with a distinct domain and exact field-element length in
/// the initial capacity. These two values make the fixed input grammar
/// unambiguous, including trailing zero fields. State positions 16..24 provide
/// eight capacity elements, and the digest contains eight field elements.
pub fn hash_fields(domain: u32, input: &[F]) -> WithdrawalDigest {
    assert!(domain < F::ORDER_U32);
    assert!(input.len() < F::ORDER_U32 as usize);
    let mut state = [F::ZERO; 24];
    state[16] = F::from_u32(domain);
    state[17] = F::from_usize(input.len());
    state[18] = F::ONE;
    let permutation = default_koalabear_poseidon2_24();
    for block in input.chunks(16) {
        for (target, value) in state.iter_mut().zip(block) {
            *target += *value;
        }
        permutation.permute_mut(&mut state);
    }
    if input.is_empty() {
        permutation.permute_mut(&mut state);
    }
    state[..8].try_into().unwrap()
}

pub fn note_commitment(
    pool: WithdrawalDigest,
    label: WithdrawalDigest,
    value: u128,
    nullifier: WithdrawalDigest,
    secret: WithdrawalDigest,
) -> WithdrawalDigest {
    let precommitment = hash_fields(PRECOMMIT, &[nullifier, secret].concat());
    hash_fields(
        NOTE,
        &[pool, label, amount_limbs(value), precommitment].concat(),
    )
}

fn merkle_root(
    domain: u32,
    mut leaf: WithdrawalDigest,
    siblings: &[WithdrawalDigest; WITHDRAWAL_TREE_DEPTH],
    index: u32,
) -> WithdrawalDigest {
    for (depth, sibling) in siblings.iter().enumerate() {
        let input = if (index >> depth) & 1 == 0 {
            [leaf, *sibling].concat()
        } else {
            [*sibling, leaf].concat()
        };
        leaf = hash_fields(domain, &input);
    }
    leaf
}

fn amount_limbs(value: u128) -> [F; 8] {
    core::array::from_fn(|i| F::from_u16((value >> (16 * i)) as u16))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Wire {
    One,
    Public(usize),
    Witness(usize),
}

#[derive(Clone, Debug)]
struct Linear(Vec<(Wire, F)>);

impl Linear {
    fn constant(value: F) -> Self {
        Self(if value == F::ZERO {
            vec![]
        } else {
            vec![(Wire::One, value)]
        })
    }

    fn wire(wire: Wire) -> Self {
        Self(vec![(wire, F::ONE)])
    }

    fn add(&self, rhs: &Self) -> Self {
        let mut entries = BTreeMap::new();
        for &(wire, value) in self.0.iter().chain(&rhs.0) {
            *entries.entry(wire).or_insert(F::ZERO) += value;
        }
        Self(
            entries
                .into_iter()
                .filter(|(_, value)| *value != F::ZERO)
                .collect(),
        )
    }

    fn scale(&self, scalar: F) -> Self {
        Self(
            self.0
                .iter()
                .map(|&(wire, value)| (wire, value * scalar))
                .filter(|(_, value)| *value != F::ZERO)
                .collect(),
        )
    }

    fn sub(&self, rhs: &Self) -> Self {
        self.add(&rhs.scale(-F::ONE))
    }
}

struct Builder {
    public: Vec<F>,
    witness: Vec<F>,
    constraints: Vec<[Linear; 3]>,
    permutations: usize,
    record_constraints: bool,
}

impl Builder {
    fn new(public: Vec<F>, record_constraints: bool) -> Self {
        Self {
            public,
            witness: Vec::new(),
            constraints: Vec::new(),
            permutations: 0,
            record_constraints,
        }
    }

    fn value(&self, linear: &Linear) -> F {
        linear
            .0
            .iter()
            .map(|&(wire, coefficient)| {
                coefficient
                    * match wire {
                        Wire::One => F::ONE,
                        Wire::Public(i) => self.public[i],
                        Wire::Witness(i) => self.witness[i],
                    }
            })
            .sum()
    }

    fn private(&mut self, value: F) -> Linear {
        let result = Linear::wire(Wire::Witness(self.witness.len()));
        self.witness.push(value);
        result
    }

    fn private_digest(&mut self, values: WithdrawalDigest) -> Vec<Linear> {
        values
            .into_iter()
            .map(|value| self.private(value))
            .collect()
    }

    fn enforce(&mut self, a: Linear, b: Linear, c: Linear) {
        if self.record_constraints {
            self.constraints.push([a, b, c]);
        }
    }

    fn equal(&mut self, left: Linear, right: Linear) {
        self.enforce(left, Linear::constant(F::ONE), right);
    }

    fn mul(&mut self, left: Linear, right: Linear) -> Linear {
        let out = self.private(self.value(&left) * self.value(&right));
        self.enforce(left, right, out.clone());
        out
    }

    fn compact(&mut self, expression: Linear) -> Linear {
        if expression.0.len() <= 4 {
            expression
        } else {
            let out = self.private(self.value(&expression));
            self.equal(expression, out.clone());
            out
        }
    }

    fn boolean(&mut self, value: bool) -> Linear {
        let bit = self.private(F::from_bool(value));
        self.enforce(
            bit.clone(),
            bit.sub(&Linear::constant(F::ONE)),
            Linear::constant(F::ZERO),
        );
        bit
    }

    fn range(&mut self, input: Linear, bits: usize) {
        let value = self.value(&input).as_canonical_u32();
        let mut reconstructed = Linear::constant(F::ZERO);
        for bit in 0..bits {
            let wire = self.boolean((value >> bit) & 1 != 0);
            reconstructed = reconstructed.add(&wire.scale(F::from_u32(1 << bit)));
        }
        self.equal(input, reconstructed);
    }

    fn cube(&mut self, value: Linear) -> Linear {
        let square = self.mul(value.clone(), value.clone());
        self.mul(square, value)
    }

    fn external_layer(&mut self, state: &mut [Linear; 24]) {
        let matrix = [[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]];
        for chunk in state.chunks_exact_mut(4) {
            let previous = chunk.to_vec();
            for row in 0..4 {
                let expression = (0..4).fold(Linear::constant(F::ZERO), |acc, col| {
                    acc.add(&previous[col].scale(F::from_u32(matrix[row][col])))
                });
                chunk[row] = self.compact(expression);
            }
        }
        let sums: [Linear; 4] = core::array::from_fn(|col| {
            let sum = (0..24)
                .step_by(4)
                .fold(Linear::constant(F::ZERO), |acc, row| {
                    acc.add(&state[row + col])
                });
            self.compact(sum)
        });
        for i in 0..24 {
            state[i] = self.compact(state[i].add(&sums[i % 4]));
        }
    }

    fn permutation(&mut self, state: &mut [Linear; 24]) {
        self.permutations += 1;
        self.external_layer(state);
        for constants in KOALABEAR_POSEIDON2_RC_24_EXTERNAL_INITIAL {
            for i in 0..24 {
                state[i] = self.cube(state[i].add(&Linear::constant(constants[i])));
            }
            self.external_layer(state);
        }
        let half = F::TWO.inverse();
        let diagonal = [
            -F::TWO,
            F::ONE,
            F::TWO,
            half,
            F::from_u32(3),
            F::from_u32(4),
            -half,
            -F::from_u32(3),
            -F::from_u32(4),
            half.exp_u64(8),
            half.exp_u64(2),
            half.exp_u64(3),
            half.exp_u64(4),
            half.exp_u64(5),
            half.exp_u64(6),
            half.exp_u64(24),
            -half.exp_u64(8),
            -half.exp_u64(3),
            -half.exp_u64(4),
            -half.exp_u64(5),
            -half.exp_u64(6),
            -half.exp_u64(7),
            -half.exp_u64(9),
            -half.exp_u64(24),
        ];
        for constant in KOALABEAR_POSEIDON2_RC_24_INTERNAL {
            state[0] = self.cube(state[0].add(&Linear::constant(constant)));
            let sum = self.compact(
                state
                    .iter()
                    .fold(Linear::constant(F::ZERO), |acc, value| acc.add(value)),
            );
            for i in 0..24 {
                state[i] = self.compact(sum.add(&state[i].scale(diagonal[i])));
            }
        }
        for constants in KOALABEAR_POSEIDON2_RC_24_EXTERNAL_FINAL {
            for i in 0..24 {
                state[i] = self.cube(state[i].add(&Linear::constant(constants[i])));
            }
            self.external_layer(state);
        }
    }

    fn hash(&mut self, domain: u32, input: &[Linear]) -> Vec<Linear> {
        let mut state: [Linear; 24] = core::array::from_fn(|_| Linear::constant(F::ZERO));
        state[16] = Linear::constant(F::from_u32(domain));
        state[17] = Linear::constant(F::from_usize(input.len()));
        state[18] = Linear::constant(F::ONE);
        for block in input.chunks(16) {
            for (target, value) in state.iter_mut().zip(block) {
                *target = target.add(value);
            }
            self.permutation(&mut state);
        }
        if input.is_empty() {
            self.permutation(&mut state);
        }
        state[..8].to_vec()
    }

    fn merkle(
        &mut self,
        domain: u32,
        mut node: Vec<Linear>,
        siblings: &[WithdrawalDigest; 32],
        index: u32,
    ) -> Vec<Linear> {
        for (depth, sibling) in siblings.iter().enumerate() {
            let sibling = self.private_digest(*sibling);
            let bit = self.boolean((index >> depth) & 1 != 0);
            let mut left = Vec::new();
            let mut right = Vec::new();
            for i in 0..8 {
                let difference = self.mul(bit.clone(), sibling[i].sub(&node[i]));
                left.push(node[i].add(&difference));
                right.push(sibling[i].sub(&difference));
            }
            left.extend(right);
            node = self.hash(domain, &left);
        }
        node
    }

    fn finish(self) -> WithdrawalCircuit {
        let num_vars = self.witness.len();
        let num_io = self.public.len();
        let num_cons = self.constraints.len();
        let mut matrices: [SparseMatrix<F>; 3] = core::array::from_fn(|_| SparseMatrix {
            num_rows: num_cons,
            num_cols: num_vars + 1 + num_io,
            entries: Vec::new(),
        });
        for (row, constraint) in self.constraints.into_iter().enumerate() {
            for (matrix, expression) in matrices.iter_mut().zip(constraint) {
                matrix
                    .entries
                    .extend(expression.0.into_iter().map(|(wire, val)| SparseMatEntry {
                        row,
                        col: match wire {
                            Wire::One => num_vars,
                            Wire::Public(i) => num_vars + 1 + i,
                            Wire::Witness(i) => i,
                        },
                        val,
                    }));
            }
        }
        let stats = WithdrawalCircuitStats {
            constraints: num_cons,
            witness_variables: num_vars,
            public_inputs: num_io,
            poseidon_permutations: self.permutations,
            matrix_nonzeros: matrices.iter().map(SparseMatrix::nnz).sum(),
        };
        let [a, b, c] = matrices;
        WithdrawalCircuit {
            shape: R1csShape {
                num_cons,
                num_vars,
                num_io,
                a,
                b,
                c,
            },
            witness: R1csWitness { w: self.witness },
            public_inputs: self.public,
            stats,
        }
    }
}

/// Build a fixed-shape R1CS, including constraints for malformed witnesses.
/// Call the ordinary satisfaction checker before producing a proof.
pub fn build_withdrawal_circuit(
    statement: &WithdrawalStatement,
    witness: &WithdrawalWitness,
) -> WithdrawalCircuit {
    synthesize_withdrawal(statement, witness, true).finish()
}

/// Generate an assignment for a previously synthesized withdrawal relation.
/// The witness uses the same operations and wire order, while omitting sparse
/// matrix construction. This is the native witness step for client benchmarks.
pub fn generate_withdrawal_witness(
    statement: &WithdrawalStatement,
    witness: &WithdrawalWitness,
) -> R1csWitness<F> {
    R1csWitness {
        w: synthesize_withdrawal(statement, witness, false).witness,
    }
}

fn synthesize_withdrawal(
    statement: &WithdrawalStatement,
    witness: &WithdrawalWitness,
    record_constraints: bool,
) -> Builder {
    let public = statement.public_inputs();
    let mut b = Builder::new(public, record_constraints);
    let inputs: Vec<_> = (0..b.public.len())
        .map(|i| Linear::wire(Wire::Public(i)))
        .collect();
    let pool = &inputs[0..8];
    let recipient = &inputs[24..34];
    let withdrawn = &inputs[34..42];
    let fee = &inputs[42..50];
    let pool_root = &inputs[50..58];
    let association_root = &inputs[58..66];
    let nullifier_hash = &inputs[66..74];
    let change_commitment = &inputs[74..82];
    let transaction_digest = &inputs[82..90];

    let balance: Vec<_> = amount_limbs(witness.balance)
        .into_iter()
        .map(|value| b.private(value))
        .collect();
    let change: Vec<_> = amount_limbs(witness.change)
        .into_iter()
        .map(|value| b.private(value))
        .collect();
    for value in balance
        .iter()
        .chain(&change)
        .chain(withdrawn)
        .chain(fee)
        .chain(recipient)
    {
        b.range(value.clone(), 16);
    }
    // Every limb equation is bounded far below the field modulus. The final
    // carry is zero, preventing modular wraparound of the 128-bit balance.
    let mut carry_value = 0u32;
    let mut carry = Linear::constant(F::ZERO);
    for i in 0..8 {
        let sum = change[i].add(&withdrawn[i]).add(&fee[i]).add(&carry);
        carry_value = (b.value(&change[i]).as_canonical_u32()
            + b.value(&withdrawn[i]).as_canonical_u32()
            + b.value(&fee[i]).as_canonical_u32()
            + carry_value)
            >> 16;
        let next = b.private(F::from_u32(carry_value));
        b.range(next.clone(), 2);
        b.equal(sum, balance[i].add(&next.scale(F::from_u32(1 << 16))));
        carry = next;
    }
    b.equal(carry, Linear::constant(F::ZERO));

    let label = b.private_digest(witness.label);
    let nullifier = b.private_digest(witness.nullifier);
    let secret = b.private_digest(witness.secret);
    let new_nullifier = b.private_digest(witness.new_nullifier);
    let new_secret = b.private_digest(witness.new_secret);
    let mut difference_sum = Linear::constant(F::ZERO);
    let mut selected = false;
    for i in 0..8 {
        let difference = new_nullifier[i].sub(&nullifier[i]);
        let value = b.value(&difference);
        let inverse = if !selected && value != F::ZERO {
            selected = true;
            value.inverse()
        } else {
            F::ZERO
        };
        let inverse = b.private(inverse);
        difference_sum = difference_sum.add(&b.mul(difference, inverse));
    }
    b.equal(difference_sum, Linear::constant(F::ONE));

    let old_precommitment = b.hash(PRECOMMIT, &[nullifier.clone(), secret].concat());
    let old_note = b.hash(
        NOTE,
        &[pool.to_vec(), label.clone(), balance, old_precommitment].concat(),
    );
    let new_precommitment = b.hash(PRECOMMIT, &[new_nullifier, new_secret].concat());
    let new_note = b.hash(
        NOTE,
        &[pool.to_vec(), label.clone(), change, new_precommitment].concat(),
    );
    let computed_nullifier = b.hash(NULLIFIER, &[pool.to_vec(), nullifier].concat());
    let computed_pool_root = b.merkle(
        POOL_NODE,
        old_note,
        &witness.pool_siblings,
        witness.pool_index,
    );
    let computed_association_root = b.merkle(
        ASSOCIATION_NODE,
        label,
        &witness.association_siblings,
        witness.association_index,
    );
    let computed_transaction = b.hash(TRANSACTION, &inputs[..82]);
    for (computed, expected) in [
        (new_note, change_commitment),
        (computed_nullifier, nullifier_hash),
        (computed_pool_root, pool_root),
        (computed_association_root, association_root),
        (computed_transaction, transaction_digest),
    ] {
        for (actual, expected) in computed.into_iter().zip(expected) {
            b.equal(actual, expected.clone());
        }
    }
    b
}

/// Deterministic, non-secret inputs for reproducible benchmark samples.
pub fn deterministic_withdrawal_inputs(sample: u64) -> (WithdrawalStatement, WithdrawalWitness) {
    let mut rng = sample.wrapping_add(0x9e3779b97f4a7c15);
    if rng == 0 {
        rng = 1;
    }
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        F::from_u64(rng)
    };
    let mut digest = || core::array::from_fn(|_| next());
    let witness = WithdrawalWitness {
        balance: (1u128 << 100) + 1_000_000,
        change: (1u128 << 100) + 749_000,
        label: digest(),
        nullifier: digest(),
        secret: digest(),
        new_nullifier: digest(),
        new_secret: digest(),
        pool_siblings: core::array::from_fn(|_| digest()),
        pool_index: 0xa59136c7,
        association_siblings: core::array::from_fn(|_| digest()),
        association_index: 0x317f94a2,
    };
    let mut statement = WithdrawalStatement {
        pool: [F::from_u32(123); 8],
        scope: [F::from_u32(456); 8],
        context: digest(),
        recipient: core::array::from_fn(|i| (sample as u16).wrapping_add(i as u16 + 1)),
        withdrawn: 250_000,
        fee: 1_000,
        pool_root: [F::ZERO; 8],
        association_root: [F::ZERO; 8],
        nullifier_hash: [F::ZERO; 8],
        change_commitment: [F::ZERO; 8],
        transaction_digest: [F::ZERO; 8],
    };
    statement.derive(&witness);
    (statement, witness)
}

pub fn deterministic_withdrawal_fixture_for_sample(sample: u64) -> WithdrawalCircuit {
    let (statement, witness) = deterministic_withdrawal_inputs(sample);
    build_withdrawal_circuit(&statement, &witness)
}

pub fn deterministic_withdrawal_fixture() -> WithdrawalCircuit {
    deterministic_withdrawal_fixture_for_sample(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn satisfied(circuit: &WithdrawalCircuit) -> bool {
        let mut z = circuit.witness.w.clone();
        z.push(F::ONE);
        z.extend_from_slice(&circuit.public_inputs);
        let (a, b, c) = circuit.shape.multiply_vec(&z).unwrap();
        a.iter().zip(b).zip(c).all(|((&a, b), c)| a * b == c)
    }

    #[test]
    fn poseidon2_gadget_matches_pinned_native_permutation() {
        for domain in [
            PRECOMMIT,
            NOTE,
            NULLIFIER,
            POOL_NODE,
            ASSOCIATION_NODE,
            TRANSACTION,
        ] {
            for length in [0, 8, 16, 17, 32, 82] {
                let values: Vec<_> = (0..length)
                    .map(|i| match i % 3 {
                        0 => F::ZERO,
                        1 => -F::ONE,
                        _ => F::from_usize(i * 31 + 7),
                    })
                    .collect();
                let mut b = Builder::new(Vec::new(), true);
                let inputs: Vec<_> = values.iter().map(|&value| b.private(value)).collect();
                let output = b.hash(domain, &inputs);
                let actual: Vec<_> = output.iter().map(|value| b.value(value)).collect();
                assert_eq!(actual, hash_fields(domain, &values));
                assert!(satisfied(&b.finish()));
            }
        }
        assert_ne!(
            hash_fields(NOTE, &[F::ONE]),
            hash_fields(PRECOMMIT, &[F::ONE])
        );
        assert_ne!(
            hash_fields(NOTE, &[F::ONE]),
            hash_fields(NOTE, &[F::ONE, F::ZERO])
        );
    }

    #[test]
    fn withdrawal_satisfies_and_has_stable_smaller_shape() {
        let first = deterministic_withdrawal_fixture_for_sample(0);
        let second = deterministic_withdrawal_fixture_for_sample(1);
        first.shape.validate().unwrap();
        assert!(satisfied(&first));
        assert!(satisfied(&second));
        assert_eq!(first.shape, second.shape);
        assert_ne!(first.public_inputs, second.public_inputs);
        assert!(first.stats.constraints < 605_424);
        let (statement, witness) = deterministic_withdrawal_inputs(1);
        assert_eq!(
            generate_withdrawal_witness(&statement, &witness),
            second.witness
        );
    }

    #[test]
    fn every_public_field_is_constrained() {
        let mut circuit = deterministic_withdrawal_fixture();
        for i in 0..circuit.public_inputs.len() {
            circuit.public_inputs[i] += F::ONE;
            assert!(!satisfied(&circuit), "unconstrained public field {i}");
            circuit.public_inputs[i] -= F::ONE;
        }
    }

    #[test]
    fn rejects_invalid_membership_balance_and_reused_nullifier() {
        let (statement, witness) = deterministic_withdrawal_inputs(7);
        let mut changed = witness.clone();
        changed.pool_siblings[0][0] += F::ONE;
        assert!(!satisfied(&build_withdrawal_circuit(&statement, &changed)));
        changed = witness.clone();
        changed.association_index ^= 1;
        assert!(!satisfied(&build_withdrawal_circuit(&statement, &changed)));
        changed = witness.clone();
        changed.change += 1;
        let mut rederived = statement.clone();
        rederived.derive(&changed);
        assert!(!satisfied(&build_withdrawal_circuit(&rederived, &changed)));
        changed = witness.clone();
        changed.new_nullifier = changed.nullifier;
        rederived.derive(&changed);
        assert!(!satisfied(&build_withdrawal_circuit(&rederived, &changed)));
        changed = witness;
        changed.balance = u128::MAX;
        changed.change = u128::MAX;
        rederived.withdrawn = 1;
        rederived.fee = 0;
        rederived.derive(&changed);
        assert!(!satisfied(&build_withdrawal_circuit(&rederived, &changed)));
    }

    #[test]
    fn rejects_non_boolean_bits_and_out_of_range_limbs() {
        let mut b = Builder::new(Vec::new(), true);
        let value = b.private(F::from_u32(1 << 16));
        b.range(value, 16);
        assert!(!satisfied(&b.finish()));
        let mut b = Builder::new(Vec::new(), true);
        b.boolean(false);
        let mut circuit = b.finish();
        circuit.witness.w[0] = F::TWO;
        assert!(!satisfied(&circuit));
    }

    #[test]
    fn accepts_integer_boundaries_and_carries() {
        let (mut statement, mut witness) = deterministic_withdrawal_inputs(9);
        for (balance, change, withdrawn, fee) in [
            (0, 0, 0, 0),
            (u128::MAX, u128::MAX, 0, 0),
            (u128::MAX, 0, u128::MAX, 0),
            (u128::MAX, 0, 0, u128::MAX),
            (196_605, 65_535, 65_535, 65_535),
            ((1u128 << 64) + 1, (1u128 << 64) - 1, 1, 1),
        ] {
            witness.balance = balance;
            witness.change = change;
            statement.withdrawn = withdrawn;
            statement.fee = fee;
            statement.derive(&witness);
            assert!(satisfied(&build_withdrawal_circuit(&statement, &witness)));
        }
    }

    #[test]
    fn scope_cannot_change_nullifier_and_pool_is_in_note() {
        let (statement, witness) = deterministic_withdrawal_inputs(11);
        let mut changed = statement.clone();
        changed.scope[0] += F::ONE;
        changed.derive(&witness);
        assert_eq!(statement.nullifier_hash, changed.nullifier_hash);
        assert_eq!(statement.pool_root, changed.pool_root);
        assert_ne!(statement.transaction_digest, changed.transaction_digest);
        assert!(satisfied(&build_withdrawal_circuit(&changed, &witness)));
        changed = statement;
        changed.pool[0] += F::ONE;
        // Recompute the metadata hash, retaining the authenticated pool root
        // and note nullifier. A transaction-digest mismatch cannot mask this test.
        changed.transaction_digest = hash_fields(TRANSACTION, &changed.transaction_fields());
        assert!(!satisfied(&build_withdrawal_circuit(&changed, &witness)));
    }

    #[test]
    fn rejects_mutated_private_bits_carries_and_hash_intermediates() {
        let mut circuit = deterministic_withdrawal_fixture();
        // First two amounts, a range bit, first carry and its bit, and two
        // intermediate variables in the Poseidon2 arithmetic.
        for i in [0, 8, 16, 16 + 42 * 16, 17 + 42 * 16, 900, 30_000] {
            circuit.witness.w[i] += F::ONE;
            assert!(!satisfied(&circuit), "unconstrained private field {i}");
            circuit.witness.w[i] -= F::ONE;
        }
    }
}
