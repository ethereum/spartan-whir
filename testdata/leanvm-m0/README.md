# Spartan-WHIR recursive verification fixtures for LeanVM

#### Scope

This directory defines the input and statement boundary for recursively verifying Spartan-WHIR proofs in LeanVM. The verifier guest decodes a Spartan-WHIR proof from private witness words, verifies it inside the VM, recomputes the application statement digest, and exposes that digest as the public input of the LeanVM execution proof. The execution proof attests that the fixed guest accepted the child Spartan-WHIR proof for the bound application statement.

The control fixture is the small no-ZK DirectSparse correctness profile. Both full-ZK encoders are implemented; the production SPARK guest and its complete fixture live in the sibling LeanVM checkout.

The checked fixture uses:

- KoalaBear base field;
- the quintic extension `X^5 + X^2 - 1` with coordinates ordered as `1, X, X^2, X^3, X^4`;
- the current Spartan-WHIR Poseidon2 challenger and Merkle hashing;
- no ZK and DirectSparse matrix closing;
- the deterministic synthetic R1CS parameters recorded in `control_fixture_manifest.json`.

#### Guest word encoding

`control_guest_input.words` is a sequence of little-endian `u32` words. Every word must be less than the KoalaBear modulus `2,130,706,433`; metadata, tags, lengths, base-field values, and extension coordinates all use canonical KoalaBear representatives.

The fixed control schema is:

```text
header:
  ['L', 'V', 'S', 'W']
  format version = 1
  profile number = 1

statement section, tag = 1:
  public input count
  public inputs
  witness commitment root count
  eight base-field words per root

Spartan section, tag = 2:
  outer round count
  three quintic elements per cubic round
  three outer claims
  inner round count
  two quintic elements per quadratic round
  witness evaluation

WHIR section, tag = 3:
  initial OOD answer vector
  initial sumcheck
  round vector
  optional final polynomial
  final proof-of-work witness
  final query openings
  optional final sumcheck

end tag = 4
end of input
```

Every vector starts with its element count. An optional value uses tag `0` for absent and `1` for present. Query openings use tag `0` for base-field rows and `1` for extension-field rows. A WHIR sumcheck contains the round count, two quintic values per round in `[h(0), h(inf)]` order, then the proof-of-work witness vector. A Merkle multiproof contains its sibling digest count followed by eight base-field words per digest.

The decoder rejects an unknown version or profile, a non-canonical word, a wrong tag, a commitment with anything other than one root, a malformed length, an invalid final-polynomial length, and any word after the end tag.

#### Statement digest

The control statement schema is `control-synthetic-inputs-v1`; its field count is fixed by the synthetic control R1CS rather than the production SHA-256 application. The control guest exposes eight public KoalaBear elements by absorbing this length-prefixed field-element preimage into the Poseidon2 width-16, rate-8 duplex challenger and finalizing the challenger:

```text
[domain byte length]
[bytes of "leanvm-spartan-whir-statement-v1"]
[inner public input count]
[inner public inputs]
```

`control_statement.json` records the exact preimage and result under the digest identifier `poseidon2-width-16-length-prefixed-statement-v1`. The production application uses the separate `sha256-digest-bits-msb-first-v1` statement schema. The production hash profile receives its own statement-digest identifier if the Poseidon gate changes this construction.

#### Transcript trace

`control_transcript_trace.json` records challenger calls in verifier execution order. Observations contain canonical base-field words, extension samples contain five coefficients, commitment observations contain the root count and flattened roots, and bit-sampling events record the requested width and result. The trace schema also represents grinding and proof-of-work verification for profiles with nonzero grinding.

The trace records the actual operations made by the Rust verifier. Semantic order is fixed by `spartan-whir/src/protocol.rs`, `spartan-whir/src/sumcheck.rs`, and the pinned Plonky3 WHIR verifier. The control profile has no transcript branch because its matrix-closing mode is fixed to DirectSparse.

#### Fiat-Shamir binding

The verifier assigns authority to its inputs as follows:

| Value | Authority | Binding rule |
| --- | --- | --- |
| Application public inputs | Trusted expected statement supplied separately to native verification | The proof-carried copy must equal the expected slice, and the values are absorbed with the fixed domain separator before any proof-dependent challenge. |
| Fixed R1CS shape, security configuration, WHIR parameters, and matrix-closing mode | Fixed profile and verifying key | The guest does not decode these values from the proof. The v1 context binds the canonical SHA-256 digest of the complete padded A/B/C relation and the fixed profile parameters before the first challenge. The protocol identifier has an eight-byte little-endian length prefix. |
| Witness commitment and every Spartan or WHIR proof value | Untrusted proof data | Each value is checked by the native verifier and is absorbed before the first challenge that depends on it. |
| Eight-element LeanVM public input | Derived from the application statement | The guest recomputes the domain-separated statement digest; the terminal adapter supplies the same digest. |
| Guest bytecode hash | LeanVM proof statement | M1 fixes this value after the guest has been compiled. |

The challenge audit for the control profile is:

| Challenge or verifier randomness | Values fixed first | First use |
| --- | --- | --- |
| Outer point `tau` | Spartan domain separator, application public inputs, plain-WHIR domain separator, witness commitment, and initial WHIR OOD material parsed from the proof | Defines the equality polynomial in the outer R1CS sumcheck. |
| Each outer sumcheck round challenge | Current outer claim and that round's three cubic coefficients | Binds the outer multilinear tables for the next round. |
| Inner batching challenge | All three outer claims | Combines the three matrix-product claims. |
| Each inner sumcheck round challenge | Current inner claim and that round's two quadratic coefficients | Binds the inner multilinear tables for the next round. |
| WHIR opening batching challenge | Witness evaluation point and claimed value, plus parsed initial OOD claims | Combines the equality constraints that define the opening statement. |
| WHIR folding, grinding, and query challenges | The commitments, OOD answers, sumcheck coefficients, and proof-of-work witnesses required by the pinned WHIR verifier at each step | Selects the next fold, verifies grinding, and selects queried codeword positions. |

`control_transcript_trace.json` is the machine-readable verifier-order record for this audit. The mutation tests change the public statement and one value in every top-level proof section and require rejection.

#### Production input limit

The production guest caps its ordinary input at 1,048,576 field words, or 4,194,304 bytes. A larger encoding is rejected instead of allocating an unbounded input. The [guest documentation](../../../leanVM/crates/spartan_whir_guest/README.md) records the current ordinary and compact input sizes.

#### SPARK candidate dimensions

The production SHA-256 R1CS has 3,251,928 SPARK union entries, which round to a `2^22` value domain. Eight fixed-value columns give `22 + 3 = 25` variables. The `2^21` column memory dominates the `2^20` row memory, and two fixed-audit columns give a `2^22` audit domain. The ten quintic read coordinates split into eight- and two-column groups, giving 25 and 23 variables. The selected schedule uses 25 variables for both read groups and is validated for both dimensions.

#### LeanVM outer-proof controls

The M1 control proof uses LeanVM `log_inv_rate = 2`, an inverse rate of 1/4. LeanVM's deterministic padding rule sets each table to `max(ceil(log2(non_padded_rows + 1)), profile_min_log_rows, 8)`, and memory to the next power of two covering the initialized memory, execution cycles, and the 256-cell minimum. The [guest documentation](../../../leanVM/crates/spartan_whir_guest/README.md) records current execution measurements and recursion results.

#### Rejection fixtures

The `mutations` directory contains one-value changes to the application public input, witness commitment, outer sumcheck, outer claims, inner sumcheck, witness evaluation, and WHIR opening. It also contains a wrong encoding version and a proof with one trailing word. `control_fixture_manifest.json` records the expected native rejection and SHA-256 digest for every file.

#### Regeneration and verification

Run the generator from the `spartan-whir` checkout. Regenerate the complete fixture and copy its consumer files together:


```bash
cargo run --release --features poseidon1 -j4 --bin leanvm-m0-fixture -- \
  testdata/leanvm-m0
```

Verify codec, trace, fixture reproduction, and mutations with:

```bash
cargo test --release --features poseidon1 -j4 --test leanvm_m0
```
