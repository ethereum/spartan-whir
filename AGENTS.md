# spartan-whir — Agent Instructions

## Project Intent

`spartan-whir` is part of the client-side proving work on a proof system that is:

- post-quantum and transparent,
- competitive for client-side proving,
- suitable for comparison benchmarks against other WHIR-based systems.

The two matrix-closing variants have different optimization objectives:

- DirectSparse prioritizes client-side proving time. Proof size and recursive verification cost are not selection criteria for DirectSparse.
- Spark balances client-side proving performance with recursion friendliness. Evaluate proof size and the work imposed on the recursive verifier together with prover time.

On-chain verification targets the final recursive root proof produced by a separate wrapper SNARK. That wrapper and direct EVM verification are outside this repository.

## Default Path Invariants

- Keep DirectSparse aligned with fast client-side proving and feasible memory use.
- Keep Spark aligned with both client-side proving and recursive verification.
- Keep the no-ZK and full-ZK APIs separate.
- Unqualified `Plonky3WhirPcs` means plain WHIR and is the no-ZK PCS. Full ZK uses dedicated Poseidon ZK keys and the lower-level hiding-WHIR committed-relation APIs.
- Treat privacy and matrix closing as independent axes: no ZK and full ZK both support DirectSparse and Spark.
- Preserve verifier-facing protocol details deliberately, including the proof structure consumed by a recursive verifier.
- Do not switch transcript, hash, or commitment choices as a refactor convenience. Treat those as protocol choices that need an explicit benchmark or verifier-design reason.
- Treat transcript ordering, proof encoding, digest layout, and other verifier-facing details as protocol surface, not incidental implementation details.
- The project is pre-production / PoC. Protocol-surface changes are allowed when they improve the design, reduce meaningful complexity, or improve the applicable DirectSparse or Spark objective. Do not block them solely to preserve existing unpublished proofs or fixtures; instead, call out what changes and update affected Rust proof structs, codecs, tests, and benchmark fixtures.

## What Is Flexible

- Field choices, extension choices, proof-system tuning, and internal structure may evolve when supported by client-side prover benchmarks and, for Spark, recursion analysis.
- Additive experimental paths are welcome when they are clearly separated from the default path.
- Add alternative primitives or challengers as explicit experiments, features, or separate APIs instead of silently changing the default.

## Refactoring Rules

- Do not broaden abstractions around cryptographic backends unless there is a concrete need in this crate.
- Do not make "cleanups" that change verifier-relevant behavior without calling that out explicitly.
- If a change can affect client-side proving, recursive verifier work, proof size, transcript compatibility, or benchmark comparability, state that impact in your summary. Recursive verifier work and proof size are optimization criteria for Spark, not DirectSparse.
- When changing extension choices or adding a new extension, document extension-specific algebraic limits and support level explicitly. This includes `TWO_ADICITY`-dependent behavior, skip-path feasibility, and which paths are currently exercised by tests or benchmarks.
- Prefer changes that keep `spartan-whir` representative for future client-side proving comparison benchmarks.

## Docs and Benchmarks

- When you add, rename, or materially change a public benchmark target or its environment knobs, update `README.md` in the same change.
- Keep schedule calibration, candidate search, heldout validation, schedule tradeoff analysis, and model-interpretation workflows in this file rather than `README.md`.
- Keep benchmark commands, benchmark target names, environment knobs, and output examples aligned with the code.
- For SHA256 comparisons across no-ZK/full-ZK and DirectSparse/Spark, use the Criterion target with native CPU tuning:
  `RUSTFLAGS='-C target-cpu=native -C debuginfo=0' SHA256_BENCH_WORKDIR=target/sha256-optimized-cache cargo bench --features parallel --bench sha256_full_zk`.
- Build the cached optimized 2048-byte SHA-256 artifact bundle before running
  that benchmark:
  `tests/circuits/build_sha256_optimized_fixture.sh ../circom/target/release/circom`.
- The Criterion target only loads existing circuit artifacts from `SHA256_BENCH_WORKDIR`; it must not compile the circuit as part of a benchmark run. Use `sha256_bench` for schedule screening and detailed tracing, and treat its `Instant` timings as diagnostic rather than comparison results.
- The default `SHA256_ZK_BENCH_EXTENSION=selected` run uses quintic DirectSparse and quintic Spark with the selected per-mode schedules. Set `SHA256_ZK_BENCH_EXTENSION=octic|quintic` to run all four variants over one extension, or `SHA256_ZK_BENCH_EXTENSION=spark` to compare quintic and octic Spark in one invocation. In a forced single-extension run, `SHA256_ZK_BENCH_SCHEDULE` applies one schedule to every witness variant; the variant-specific `SHA256_ZK_BENCH_{NO_ZK_DIRECT,NO_ZK_SPARK,FULL_ZK_DIRECT,FULL_ZK_SPARK}_SCHEDULE` values override it. `SHA256_ZK_BENCH_SPARK_{FIXED_VALUE,FIXED_AUDIT,READ}_SCHEDULE` overrides the corresponding SPARK table schedule.
- For an independent schedule measurement, set `SHA256_ZK_BENCH_PROVING_ONLY=1`, `SHA256_ZK_BENCH_SINGLE_PROVING_VARIANT=no_zk_direct|full_zk_direct|no_zk_spark|full_zk_spark`, and `SHA256_ZK_BENCH_SECURITY_BITS=<end-to-end target>`. This constructs only the selected configuration and key, so the run does not depend on another mode accepting the same extension or schedule.
- Profile the 2048-byte full-ZK Spark path without rebuilding the circuit with:
  `SHA256_BENCH_WORKDIR=target/sha256-optimized-cache SHA256_BENCH_REUSE_ARTIFACTS=1 SHA256_BENCH_SIZES=2048 SHA256_BENCH_PROOF_MODES=full-zk SHA256_BENCH_MODES=spark SHA256_BENCH_SECURITY_BITS=116 SHA256_BENCH_PROFILE=1 SHA256_BENCH_PROFILE_DETAIL=1 RUSTFLAGS='-C target-cpu=native -C debuginfo=0' cargo run --release --features parallel --example sha256_bench`.
- Benchmark/profiling output intended for direct human inspection should be stable and human-readable. Prefer labeled `key: value` fields and clear tree/group structure over raw debug dumps.
- Write benchmark output under `benchmark-results/` for local analysis. The directory is ignored and must not be committed. Document the latest relevant measurements directly in `README.md`, including the workload, commit or working-tree description, machine, toolchain, flags, sample count, confidence intervals, proof sizes, and exact commands needed to reproduce them. Generate raw samples, control results, logs, and analysis artifacts during development.
- If a benchmark fixture is only shape-similar to a real circuit, document that approximation explicitly instead of describing it as the real circuit.
- Keep `README.md` focused on the current codebase state rather than changelog-style history; describe the format and behavior that exist now.
- Any protocol change that affects Spark fixed-table commitments must regenerate serialized `spark_fixed_commitments` and fixtures containing them.

## Poseidon Schedule Optimization

The Poseidon Plonky3-WHIR prover uses an offline schedule-scoring workflow with
Johnson-bound soundness. The component scorer ranks one WHIR commitment at a
time. Full-ZK Spark uses `poseidon_spark_schedule_scorer.py` to compose the
witness, fixed-value, fixed-audit, and read reports. The scorers are
not part of setup. Generate and measure schedules for the target circuit, then
pass the selected setup configuration into setup.

Add the `poseidon1` feature to every calibration, candidate, and heldout command
when tuning the Poseidon1 profile. The report provenance must say
`parallel,poseidon1`; a report that says only `parallel` does not identify the
hash profile even if another field names it.

For DirectSparse, optimize prover time while keeping memory feasible. Do not use
proof size or recursive verifier cost to select a DirectSparse schedule. For
Spark, retain candidates that balance prover time, proof size, and recursive
verifier work. Use the linked native witness generator for end-to-end client
benchmarks; reserve `.wtns` inputs for schedule-model calibration.
Rerun component calibration after changes to Plonky3 kernels, SPARK batching,
or the number or shape of commitments and openings represented by the model.
Do not rank schedules with calibration data from a different implementation.

### Workflow

1. Measure local component costs:

```bash
RUSTFLAGS="-C target-cpu=native -C debuginfo=0" \
cargo run --release --features parallel \
  --bin poseidon-schedule-calibration -- \
  --out /tmp/poseidon-calibration.json
```

2. Enumerate backend-validated candidate schedules:

```bash
RUSTFLAGS="-C target-cpu=native -C debuginfo=0" \
cargo run --release --features parallel -q --bin poseidon-schedule-candidates -- \
  --num-variables 19 \
  --field koalabear \
  --extension quintic \
  --security-bits 116 \
  --merkle-security-bits 116 \
  --max-pow-bits 22 \
  --proof-mode no-zk \
  > /tmp/poseidon-candidates.json
```

Use `--proof-mode full-zk` for schedules that must satisfy hiding-WHIR mask
slack and extension-field two-adicity limits. The fallback parameter helpers
are `recommended_quintic_whir_params` and
`recommended_quintic_zk_whir_params` for DirectSparse,
`recommended_quintic_spark_whir_params` and
`recommended_quintic_spark_zk_whir_params` for the Spark witness,
`recommended_quintic_spark_fixed_whir_params` for fixed tables, and
`recommended_quintic_spark_read_whir_params` for read tables. The corresponding
octic helpers are explicit octic controls and fallbacks. Use scheduler-selected
`WhirParams` for benchmarked deployments.

Use `--extension quartic|quintic|octic` to bound a candidate file or component
report to one extension; the default is `all`. The scorer passes this option to
the Rust candidate generator. `--round-log-inv-rate-offset-max N` keeps offset
zero as the backend-derived schedule with an empty `round_log_inv_rates` list
and also searches explicit schedules formed by adding each offset from one
through `N` to every backend-derived round rate.

DirectSparse searches interpret `--security-bits` and
`--merkle-security-bits` as end-to-end targets and derive the stronger WHIR
and Merkle component targets for each candidate. The selected row includes a
standalone DirectSparse setup configuration at the requested end-to-end target.

3. Score a candidate file:

```bash
python3 scripts/poseidon_schedule_scorer.py \
  --candidates /tmp/poseidon-candidates.json \
  --calibration /tmp/poseidon-calibration.json \
  --constraint-work 483648 \
  --case-label circuit \
  --workload-r1cs circuit.r1cs \
  --out-report /tmp/poseidon-report.json \
  --out-config /tmp/poseidon-config.json
```

For full-ZK tuning, let the scorer generate the joint WHIR and ZK parameter
grid:

```bash
python3 scripts/poseidon_schedule_scorer.py \
  --num-variables 20 \
  --extension quintic \
  --calibration /tmp/poseidon-calibration.json \
  --constraint-work 605424 \
  --case-label sha256_2048b \
  --workload-r1cs target/sha256-optimized-cache/sha256_2048b/sha256_2048b.r1cs \
  --security-bits 116 \
  --merkle-security-bits 116 \
  --proof-mode full-zk \
  --round-log-inv-rate-offset-max 3 \
  --reference-label quintic_cfsr_pow4_ff8_rest6_lir1_rsv6_round_log_inv_rates_4 \
  --measurement-shortlist-margin-ratio 0.01 \
  --out-report /tmp/poseidon-zk-report.json \
  --out-config /tmp/poseidon-zk-config.json
```

Pass `--reference-label <accepted-label>` once for each accepted baseline or
finalist that must remain in `measurement_shortlist` even when it lies outside
the model margin. The option is repeatable and rejects labels that are absent
from the accepted candidate rows.

The default ZK sweep is `ell_zk = 3,4,8,16` and
`mask_log_inv_rate = 1,2,3,4,5`. Include low mask rates in the sweep and let
the backend slack checks reject invalid rows.

#### Spark component reports

For the 2048-byte Spark workload at 116-bit composed security, score the
witness, fixed-value, fixed-audit, and each read group independently. Their
variable counts are 20, 25, 22, 25, and 23. The composed budget requires
120-bit WHIR components and 123-bit Merkle binding. Generate separate reports
for both read groups even though the protocol stores one shared read schedule.
The table reports are shared by both privacy modes; generate a witness report
for each privacy mode. Component searches omit `--out-config`:

```bash
RUSTFLAGS="-C target-cpu=native -C debuginfo=0" \
python3 scripts/poseidon_schedule_scorer.py \
  --num-variables 20 \
  --field koalabear \
  --extension quintic \
  --calibration /tmp/poseidon-calibration.json \
  --out-report /tmp/spark-witness-no-zk.json \
  --max-pow-bits 22 \
  --security-bits 116 \
  --merkle-security-bits 116 \
  --component-security-bits 120 \
  --component-merkle-security-bits 123 \
  --constraint-work 605424 \
  --case-label sha256_2048b \
  --workload-r1cs target/sha256-optimized-cache/sha256_2048b/sha256_2048b.r1cs \
  --proof-mode no-zk \
  --reference-label quintic_constant_pow6_ff8_lir1_rsv8_round_log_inv_rates_derived

RUSTFLAGS="-C target-cpu=native -C debuginfo=0" \
python3 scripts/poseidon_schedule_scorer.py \
  --num-variables 20 \
  --field koalabear \
  --extension quintic \
  --calibration /tmp/poseidon-calibration.json \
  --out-report /tmp/spark-witness-full-zk.json \
  --max-pow-bits 22 \
  --security-bits 116 \
  --merkle-security-bits 116 \
  --component-security-bits 120 \
  --component-merkle-security-bits 123 \
  --constraint-work 605424 \
  --case-label sha256_2048b \
  --workload-r1cs target/sha256-optimized-cache/sha256_2048b/sha256_2048b.r1cs \
  --proof-mode full-zk \
  --zk-ell-values 3 \
  --zk-mask-log-inv-rate-values 3 \
  --round-log-inv-rate-offset-max 3 \
  --reference-label quintic_cfsr_pow7_ff8_rest3_lir1_rsv7_round_log_inv_rates_derived

RUSTFLAGS="-C target-cpu=native -C debuginfo=0" \
python3 scripts/poseidon_schedule_scorer.py \
  --num-variables 25 \
  --field koalabear \
  --extension quintic \
  --calibration /tmp/poseidon-calibration.json \
  --out-report /tmp/spark-fixed-value.json \
  --max-pow-bits 22 \
  --security-bits 116 \
  --merkle-security-bits 116 \
  --component-security-bits 120 \
  --component-merkle-security-bits 123 \
  --proof-mode no-zk \
  --reference-label quintic_cfsr_pow9_ff8_rest4_lir1_rsv8_round_log_inv_rates_derived

RUSTFLAGS="-C target-cpu=native -C debuginfo=0" \
python3 scripts/poseidon_schedule_scorer.py \
  --num-variables 22 \
  --field koalabear \
  --extension quintic \
  --calibration /tmp/poseidon-calibration.json \
  --out-report /tmp/spark-fixed-audit.json \
  --max-pow-bits 22 \
  --security-bits 116 \
  --merkle-security-bits 116 \
  --component-security-bits 120 \
  --component-merkle-security-bits 123 \
  --proof-mode no-zk \
  --reference-label quintic_constant_pow6_ff8_lir1_rsv8_round_log_inv_rates_derived

RUSTFLAGS="-C target-cpu=native -C debuginfo=0" \
python3 scripts/poseidon_schedule_scorer.py \
  --num-variables 25 \
  --field koalabear \
  --extension quintic \
  --calibration /tmp/poseidon-calibration.json \
  --out-report /tmp/spark-read-25.json \
  --max-pow-bits 22 \
  --security-bits 116 \
  --merkle-security-bits 116 \
  --component-security-bits 120 \
  --component-merkle-security-bits 123 \
  --proof-mode no-zk \
  --reference-label quintic_cfsr_pow9_ff8_rest4_lir1_rsv8_round_log_inv_rates_derived

RUSTFLAGS="-C target-cpu=native -C debuginfo=0" \
python3 scripts/poseidon_schedule_scorer.py \
  --num-variables 23 \
  --field koalabear \
  --extension quintic \
  --calibration /tmp/poseidon-calibration.json \
  --out-report /tmp/spark-read-23.json \
  --max-pow-bits 22 \
  --security-bits 116 \
  --merkle-security-bits 116 \
  --component-security-bits 120 \
  --component-merkle-security-bits 123 \
  --proof-mode no-zk \
  --reference-label quintic_cfsr_pow9_ff8_rest4_lir1_rsv8_round_log_inv_rates_derived
```

For the grouped quintic read argument, intersect schedules accepted by both
read reports and sum their projected work and proof-size estimates. Compare the
complete parameter tuple, not only the schedule label, and keep
`round_log_inv_rates` derived because the two groups can have different round
counts. Do not use the 25-variable report alone as the read cost. Confirm the
shortlist with the full Criterion target.

#### Spark composition

The Spark composer requires an explicit `--extension` and accepts one
`--read-report` for each read group. With multiple read reports, it intersects
the complete `WhirParams` tuple except `round_log_inv_rates`, verifies that
each source rate list is empty or backend-derived, writes an empty shared rate
list into the setup configuration, and sums the projected time and proof size
for every read group. Quintic composition requires read dimensions equal to
the fixed-value dimension and that dimension minus two. Octic composition
requires one read report at the fixed-value dimension plus one. The setup
configuration uses separate Spark openings. The composer derives whether the
fixed audit tables fit in the fixed value bundle from their dimensions and
requires `--fixed-audit-embedded` to match. Embedded audit configuration and
security checks remain present while its per-proof time and proof size are
zero. The embedded audit schedule is pinned to its explicit reference, or to
the component report's selected row when no reference is supplied.

When verifier coefficients are available, the component scorer preserves the
complete prover-time, verifier-time, and proof-size Pareto set before applying
`--max-report-rows`. It also preserves the selected row, measurement shortlist,
and explicit references, and fails if the cap cannot hold that required set.
The composer retains the three-axis Pareto set after bounding each component's
excess prover projection by 1% of the best composed prover projection.
`--top-per-component` limits only the alternatives used to build the
calibration shortlist. `--max-report-rows` must fit the composed Pareto rows,
shortlist rows, and measured rows; composition fails instead of silently
dropping any of them. Proof size and one-thread native verifier time remain
separate proxies; neither substitutes for LeanVM cycles or trace rows.

Compose both 2048-byte privacy modes before heldout measurement:

```bash
python3 scripts/poseidon_spark_schedule_scorer.py \
  --witness-report /tmp/spark-witness-no-zk.json \
  --fixed-value-report /tmp/spark-fixed-value.json \
  --fixed-audit-report /tmp/spark-fixed-audit.json \
  --read-report /tmp/spark-read-25.json \
  --read-report /tmp/spark-read-23.json \
  --out-report /tmp/spark-no-zk-combined.json \
  --extension quintic \
  --proof-mode no-zk \
  --security-bits 116 \
  --merkle-security-bits 116 \
  --top-per-component 16 \
  --max-report-rows 5000 \
  --measurement-rows 10 \
  --max-fixed-value-log-domain 26 \
  --max-fixed-audit-log-domain 23 \
  --fixed-audit-embedded \
  --workload-r1cs target/sha256-optimized-cache/sha256_2048b/sha256_2048b.r1cs \
  --workload-label sha256_2048b \
  --reference-witness-label quintic_constant_pow6_ff8_lir1_rsv8_round_log_inv_rates_derived \
  --reference-fixed-value-label quintic_cfsr_pow9_ff8_rest4_lir1_rsv8_round_log_inv_rates_derived \
  --reference-fixed-audit-label quintic_constant_pow6_ff8_lir1_rsv8_round_log_inv_rates_derived \
  --reference-read-label quintic_cfsr_pow9_ff8_rest4_lir1_rsv8_round_log_inv_rates_derived \
  --reference-read-label quintic_cfsr_pow9_ff8_rest4_lir1_rsv8_round_log_inv_rates_derived

python3 scripts/poseidon_spark_schedule_scorer.py \
  --witness-report /tmp/spark-witness-full-zk.json \
  --fixed-value-report /tmp/spark-fixed-value.json \
  --fixed-audit-report /tmp/spark-fixed-audit.json \
  --read-report /tmp/spark-read-25.json \
  --read-report /tmp/spark-read-23.json \
  --out-report /tmp/spark-full-zk-combined.json \
  --extension quintic \
  --proof-mode full-zk \
  --security-bits 116 \
  --merkle-security-bits 116 \
  --ell-zk 3 \
  --mask-log-inv-rate 3 \
  --top-per-component 16 \
  --max-report-rows 5000 \
  --measurement-rows 10 \
  --max-fixed-value-log-domain 26 \
  --max-fixed-audit-log-domain 23 \
  --fixed-audit-embedded \
  --workload-r1cs target/sha256-optimized-cache/sha256_2048b/sha256_2048b.r1cs \
  --workload-label sha256_2048b \
  --reference-witness-label quintic_cfsr_pow7_ff8_rest3_lir1_rsv7_round_log_inv_rates_derived \
  --reference-fixed-value-label quintic_cfsr_pow9_ff8_rest4_lir1_rsv8_round_log_inv_rates_derived \
  --reference-fixed-audit-label quintic_constant_pow6_ff8_lir1_rsv8_round_log_inv_rates_derived \
  --reference-read-label quintic_cfsr_pow9_ff8_rest4_lir1_rsv8_round_log_inv_rates_derived \
  --reference-read-label quintic_cfsr_pow9_ff8_rest4_lir1_rsv8_round_log_inv_rates_derived
```

Every component report must carry the requested 116-bit
end-to-end target and the same explicit 120-bit WHIR and 123-bit Merkle
component targets. Its `setup_config` retains the 116-bit end-to-end target.

Use the matching combined report for heldout measurement. After measurement,
rerun the composer with `--measurements /tmp/spark-heldout.json` and a separate
`RAYON_NUM_THREADS=1` heldout report through
`--verifier-measurements /tmp/spark-heldout-verifier-1thread.json`. Require
`model.calibration.validation_within_ten_percent` and
`model.verifier_calibration.validation_within_twenty_percent` before selecting
a schedule.
To include a configured baseline when it is outside the model's top component
rows, pass the witness, fixed-value, and fixed-audit reference labels once and
pass `--reference-read-label` once per read report in the same order as the
reports.

4. Measure shortlisted rows. For a `.wtns` calibration fixture:

```bash
RUSTFLAGS="-C target-cpu=native -C debuginfo=0" \
cargo run --release --features parallel \
  --bin poseidon-schedule-heldout -- \
  --r1cs circuit.r1cs \
  --wtns witness.wtns \
  --case-label circuit \
  --report /tmp/poseidon-report.json \
  --out /tmp/poseidon-heldout.json \
  --extension octic \
  --proof-mode no-zk \
  --row-source measurement-shortlist \
  --max-rows 5 \
  --repeats 21 \
  --warmups 3
```

For cached SHA-256 linked-witness artifacts:

```bash
RUSTFLAGS="-C target-cpu=native -C debuginfo=0" \
cargo run --release --features parallel \
  --bin poseidon-schedule-heldout -- \
  --r1cs target/sha256-optimized-cache/sha256_2048b/sha256_2048b.r1cs \
  --linked-witness-library target/sha256-optimized-cache/sha256_2048b/libsha256_2048b_witness.dylib \
  --linked-circuit-data target/sha256-optimized-cache/sha256_2048b/sha256_2048b_cpp/sha256_2048b.dat \
  --linked-input target/poseidon-schedule/sha256_2048b_input.bin \
  --linked-run-name sha256_2048b \
  --case-label sha256_2048b \
  --report /tmp/poseidon-report.json \
  --out /tmp/poseidon-heldout.json \
  --extension octic \
  --proof-mode full-zk \
  --row-source measurement-shortlist \
  --max-rows 10 \
  --randomize-linked-input-bits \
  --repeats 21 \
  --warmups 3
```

5. Merge heldout measurements and refit component scales:

```bash
python3 scripts/poseidon_schedule_add_heldout.py \
  --calibration /tmp/poseidon-calibration.json \
  --heldout /tmp/poseidon-heldout.json \
  --out /tmp/poseidon-calibration-heldout.json \
  --replace \
  --recalibrate
```

Use `--proof-mode full-zk` throughout candidate generation, scoring, heldout
measurement, and rescoring for a full-ZK search. Full-ZK heldout rows must come
from the full-ZK prover; never relabel a full-ZK report as no ZK. Pass
`--measurements /tmp/poseidon-heldout.json` to the scorer to add
`selected_measured` without replacing the model's `selected` row.

### Selection Rules

- Treat the scorer as a pruning model, not a sub-percent ordering oracle.
- A bounded component report must retain its complete three-axis Pareto set,
  shortlist, selected row, and explicit references. The scorer and Spark
  composer fail when a report cap cannot contain their required rows.
- For Spark, admit schedules whose measured client prover median is no more
  than 1% above the fastest measured median. Prover confidence intervals are
  diagnostic and do not expand this band.
- For DirectSparse, prefer lower PoW and then stable label order among time-tied
  rows. Proof size is not a DirectSparse selection criterion.
- Within the Spark prover band, prefer one-thread native verifier time. Treat verifier medians within 1% or with overlapping bootstrap median intervals as tied, then prefer the smaller measured serialized proof and stable label order.
- When recursive-verifier cycles or trace rows have not been measured, keep the
  native-time/proof-size finalists and state that the recursive choice remains
  unresolved. Do not treat proof size as recursive-verifier work.
- A higher-PoW Spark row can win a prover-time tie through its proof size or
  recursive verifier cost, but it carries a tail-latency cost. Revisit the
  PoW-free tied row if p99 proving latency becomes an objective.
- Compare absolute timings only within one heldout run. Cite the heldout
  artifact when quoting a value because medians drift across batches.
- Use full-proof Criterion benchmarks as the final decision point.

`poseidon-schedule-heldout` runs repeats in shuffled round-robin passes. With
`--randomize-linked-input-bits`, it varies SHA-style inputs per repeat to
average deterministic PoW grind luck. Its output includes samples, the median,
the mean, and a bootstrap median confidence interval for both proving and
verification. The `parallel` verifier is partly multithreaded; use
`RAYON_NUM_THREADS=1` for the sequential verifier proxy used by the scorer.
The source report and every measured row carry one workload identity that binds
the case label, R1CS SHA-256 digest, and constraint count. Heldout measurement
checks it against `--r1cs`, `--case-label`, and the linked run name before setup
or timing.

### Artifacts and Model Limits

- `poseidon-schedule-calibration` writes component coefficients and raw
  microbenchmark measurements by extension. Its verifier coefficients cover
  Merkle compression, leaf field elements, opened row field elements,
  extension operations, and proof-of-work checks.
- `poseidon-schedule-candidates` writes backend-derived schedules, component
  security targets, PoW, round data, work units, and `proof_mode`. End-to-end
  DirectSparse rows include a setup configuration. Explicit component-target
  rows are inputs to the Spark composer and do not include one. Full-ZK rows
  also contain `ell_zk`, `mask_log_inv_rate`, `zk_*` work, and proof-size
  estimates.
- `poseidon_schedule_scorer.py` writes projected time, `cost_breakdown`,
  verifier projected time and breakdown, validation status, `selected`, and
  `measurement_shortlist`. With measured
  input it also writes `selected_measured`, and `--out-config` writes that
  measured selection rather than the model-only selection. Pass `--workload-r1cs`,
  `--constraint-work`, and `--case-label` together for a report that will be
  measured by `poseidon-schedule-heldout`.
- `poseidon-schedule-heldout` consumes `measurement_shortlist`,
  `measurement_candidates`, `scores`, or `candidates`, in that order. Use
  `--include-strata` to sample across the accepted ranking.
- `poseidon_schedule_add_heldout.py` merges heldout rows and refits component
  scales. Heldout rows must contain every modeled component metric. Keep each
  refit scoped to one proof mode and extension; mixed workloads can be too
  collinear to identify the component coefficients.
Candidate WHIR validity and achieved security come from constructing Plonky3
WHIR configs. DirectSparse rows also reproduce the composed algebraic, WHIR,
and Merkle security gate used by setup. Rows are rejected when backend
derivation fails, achieved security is below target, PoW exceeds the cap, field
two-adicity is insufficient, or the requested end-to-end security is
unattainable.

The linear model is:

```text
projected_time = fixed_overhead + dft + merkle + merkle_path + row_opening + sumcheck + pow + spartan
```

The verifier model is calibrated independently from proof size:

```text
projected_verifier_time = fixed_overhead + merkle_hashes + leaf_field_elements + row_field_elements + extension_operations + pow_checks
```

Do not multiply verifier time by proof bytes. That product is measured in
byte-seconds and double-counts proof-shape effects already present in verifier
time. For an explicit input-delivery model, use
`verifier_time + proof_bytes / effective_input_bandwidth` and report the two
terms separately.

The SPARK composer adds the witness relation, post-setup fixed-table work, and
every read group's commitment and opening. It excludes each fixed table's
initial setup commitment. The fixed-value and fixed-audit log-domain caps keep
the search within the configured setup footprint.

Recommendations are trusted only after heldout error for the target circuit and
extension falls within tolerance. The model omits cache behavior and shared
backend interactions, so clustered schedules require heldout confirmation.
`proof_size_bytes_estimate` counts opened field elements, Merkle path digests,
and round commitments; it is a ranking proxy rather than a serialized byte
count.

The `--field babybear` mode tests schedule validity and ranking under BabyBear's
two-adicity; it is not a BabyBear prover benchmark. The relevant validity bound
is `num_variables + starting_log_inv_rate - first_folding_factor <=
F::TWO_ADICITY`. Heldout recalibration updates only extensions represented in
the measured rows.

## Decision Heuristic

- For DirectSparse, reject changes that slow client-side proving unless they are required for correctness, security, or feasible client memory use. Do not trade DirectSparse proving speed for smaller proofs or cheaper recursion.
- For Spark, evaluate client-side prover performance and recursion friendliness together. Require benchmark evidence and recursion analysis for changes that improve one by making the other worse.
- Treat direct EVM verification as outside the scope of `spartan-whir`. The on-chain verifier checks the final recursive root proof produced by a separate wrapper SNARK.
