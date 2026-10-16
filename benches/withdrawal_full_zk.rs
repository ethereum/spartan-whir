use std::{
    env,
    hint::black_box,
    time::{Duration, Instant},
};

use criterion::{criterion_group, criterion_main, BatchSize, Criterion, SamplingMode};
use rand::{rngs::StdRng, SeedableRng};
use spartan_whir::withdrawal::{
    config::{configured_withdrawal_spark_config, setup_withdrawal},
    deterministic_withdrawal_fixture, deterministic_withdrawal_inputs, generate_withdrawal_witness,
    WITHDRAWAL_HASH_PROFILE,
};

fn benchmark_withdrawal(c: &mut Criterion) {
    let config = configured_withdrawal_spark_config(None)
        .expect("withdrawal setup configuration is valid JSON with the required security target");
    let fixture = deterministic_withdrawal_fixture();
    let corpus_size = env::var("WITHDRAWAL_BENCH_CORPUS_SIZE")
        .ok()
        .map(|value| value.parse::<usize>().expect("corpus size is an integer"))
        .unwrap_or(4);
    assert!(
        corpus_size >= 2,
        "WITHDRAWAL_BENCH_CORPUS_SIZE must be at least two"
    );
    let corpus: Vec<_> = (0..corpus_size)
        .map(|sample| deterministic_withdrawal_inputs(sample as u64))
        .collect();
    let public_corpus: Vec<_> = corpus
        .iter()
        .map(|(statement, _)| statement.public_inputs())
        .collect();
    println!("application_hash: {WITHDRAWAL_HASH_PROFILE}");
    println!("proof_hash: poseidon1");
    println!("application_security_bits: 116");
    println!("circuit: {:?}", fixture.stats);
    println!("setup_config: {}", serde_json::to_string(&config).unwrap());
    let setup_start = Instant::now();
    let (pk, vk) = setup_withdrawal(fixture.shape.clone(), config)
        .expect("withdrawal full-ZK SPARK setup succeeds at 116 bits");
    println!(
        "diagnostic_setup_ms: {:.3}",
        setup_start.elapsed().as_secs_f64() * 1000.0
    );

    let mut proofs = Vec::with_capacity(corpus_size);
    for (sample, (statement, input)) in corpus.iter().enumerate() {
        let witness = generate_withdrawal_witness(statement, input);
        let public_inputs = public_corpus[sample].clone();
        spartan_whir::validate_satisfaction(&fixture.shape, &witness, &public_inputs)
            .expect("sample satisfies the reusable withdrawal relation");
        let start = Instant::now();
        let proof = pk
            .prove_with_rng(witness, public_inputs.clone(), &mut sample_rng(sample))
            .expect("withdrawal diagnostic proof succeeds");
        println!(
            "diagnostic_prove_ms_sample_{sample}: {:.3}",
            start.elapsed().as_secs_f64() * 1000.0
        );
        vk.verify(&public_inputs, &proof)
            .expect("withdrawal diagnostic proof verifies");
        println!(
            "proof_bytes_sample_{sample}: {}",
            bincode::serialize(&proof.proof).unwrap().len()
        );
        proofs.push(proof);
    }
    // Every public vector is supplied by the application, including during
    // recursive fixture export. Another valid withdrawal statement cannot
    // replace the statement against which this proof is verified.
    assert!(vk.verify(&public_corpus[1], &proofs[0]).is_err());

    let mut group = c.benchmark_group("privacy_pools_withdrawal_poseidon1_quintic");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(30));
    group.sampling_mode(SamplingMode::Flat);
    let mut sample = 0usize;
    let maximum_client_seconds = env::var("WITHDRAWAL_BENCH_MAX_SECONDS").ok().map(|value| {
        let seconds: f64 = value.parse().expect("client time limit is a number");
        assert!(seconds.is_finite() && seconds > 0.0);
        seconds
    });
    let mut client_operation_times_ms = Vec::new();
    group.bench_function("witness", |bencher| {
        bencher.iter_batched(
            || {
                let item = corpus[sample % corpus.len()].clone();
                sample += 1;
                item
            },
            |(statement, input)| {
                black_box(generate_withdrawal_witness(
                    black_box(&statement),
                    black_box(&input),
                ))
            },
            BatchSize::PerIteration,
        );
    });
    group.bench_function("witness_and_prove", |bencher| {
        bencher.iter_batched(
            || {
                let item = corpus[sample % corpus.len()].clone();
                let rng = sample_rng(sample);
                sample += 1;
                (item, rng)
            },
            |((statement, input), mut rng)| {
                let started = Instant::now();
                let witness = generate_withdrawal_witness(black_box(&statement), black_box(&input));
                let proof = black_box(
                    pk.prove_with_rng(witness, statement.public_inputs(), &mut rng)
                        .expect("withdrawal full-ZK SPARK proving succeeds"),
                );
                let seconds = started.elapsed().as_secs_f64();
                client_operation_times_ms.push(seconds * 1000.0);
                if let Some(limit) = maximum_client_seconds {
                    assert!(
                        seconds < limit,
                        "client witness and proving took {seconds:.3}s, exceeding {limit:.3}s"
                    );
                }
                proof
            },
            BatchSize::PerIteration,
        );
    });
    group.bench_function("verify", |bencher| {
        bencher.iter_batched(
            || {
                let index = sample % corpus.len();
                sample += 1;
                index
            },
            |index| {
                vk.verify(black_box(&public_corpus[index]), black_box(&proofs[index]))
                    .expect("withdrawal full-ZK SPARK verification succeeds")
            },
            BatchSize::PerIteration,
        );
    });
    group.finish();
    println!(
        "individual_client_operation_ms_including_warmup: {}",
        serde_json::to_string(&client_operation_times_ms).unwrap()
    );
}

fn sample_rng(sample: usize) -> StdRng {
    // Reproducible benchmark randomness; every iteration has a distinct seed.
    StdRng::seed_from_u64(0x5050_5749_5448_4452u64.wrapping_add(sample as u64))
}

criterion_group!(benches, benchmark_withdrawal);
criterion_main!(benches);
