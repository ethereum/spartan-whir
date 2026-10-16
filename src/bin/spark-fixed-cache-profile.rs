//! Profile native verifier startup from a complete serialized SPARK key.
//!
//! Loading, key authentication, and one cache construction are timed separately.
//! Authentication rebuilds the fixed trees, so the subsequent construction uses
//! warmed allocator and DFT state. An external process peak-RSS measurement covers
//! all three phases. This mode never constructs a proving key, witness, or proof.
//! `prepare-key` is a separate untimed invocation that creates the input artifact.

#[cfg(not(feature = "poseidon1"))]
fn main() {
    eprintln!("error: spark-fixed-cache-profile requires --features poseidon1");
    std::process::exit(1);
}

#[cfg(feature = "poseidon1")]
fn main() {
    if let Err(error) = profile::run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

#[cfg(feature = "poseidon1")]
mod profile {
    use std::{
        env,
        error::Error,
        fs::File,
        hint::black_box,
        io::{self, BufReader, BufWriter, Read, Write},
        path::Path,
        time::Instant,
    };

    use p3_field::PrimeField32;
    use serde::Deserialize;
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use spartan_whir::{
        engine::F, fixed_oracle_cache::FixedOracleCacheFor, import_r1cs_path,
        pcs_config::FreshMaskBatching, MatrixClosingMode, Poseidon1QuinticEngine,
        Poseidon1ZkProvingKey, Poseidon1ZkVerifyingKey, PoseidonZkSetupConfig, QuinticExtension,
        SpartanWhirError,
    };

    type VerifyingKey = Poseidon1ZkVerifyingKey<QuinticExtension>;
    type Cache = FixedOracleCacheFor<Poseidon1QuinticEngine>;

    const USAGE: &str = "usage: spark-fixed-cache-profile <verifying-key.json>\n       spark-fixed-cache-profile prepare-key <circuit.r1cs> <size-report.json> <verifying-key.json>";

    struct HashingReader<R> {
        reader: R,
        digest: Sha256,
        bytes: u64,
    }

    impl<R: Read> Read for HashingReader<R> {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            let count = self.reader.read(buffer)?;
            self.digest.update(&buffer[..count]);
            self.bytes += count as u64;
            Ok(count)
        }
    }

    pub fn run() -> Result<(), Box<dyn Error>> {
        let args = env::args_os().skip(1).collect::<Vec<_>>();
        match args.as_slice() {
            [key] => startup(Path::new(key)),
            [command, r1cs, report, output] if command == "prepare-key" => {
                prepare_key(Path::new(r1cs), Path::new(report), Path::new(output))
            }
            _ => Err(io::Error::other(USAGE).into()),
        }
    }

    fn startup(path: &Path) -> Result<(), Box<dyn Error>> {
        let startup_start = Instant::now();
        let load_start = Instant::now();
        // Stream the complete key to avoid retaining a second copy of its JSON.
        // The input format is the ordinary serde representation of the full VK,
        // including canonical A/B/C matrices; a guest-config projection is insufficient.
        let mut input = BufReader::new(HashingReader {
            reader: File::open(path)?,
            digest: Sha256::new(),
            bytes: 0,
        });
        let mut vk: VerifyingKey = serde_json::from_reader(&mut input)?;
        let input = input.into_inner();
        let key_bytes = input.bytes;
        let key_sha256 = format!("{:x}", input.digest.finalize());
        drop(input.reader);
        let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
        if vk.matrix_closing != MatrixClosingMode::Spark {
            return Err(
                io::Error::other("the verification key must use SPARK matrix closing").into(),
            );
        }

        let authentication_start = Instant::now();
        vk.authenticate().map_err(protocol_error)?;
        let authentication_ms = authentication_start.elapsed().as_secs_f64() * 1000.0;

        let construction_start = Instant::now();
        let cache = Cache::build(&vk).map_err(protocol_error)?;
        let build_ms = construction_start.elapsed().as_secs_f64() * 1000.0;
        let startup_ms = startup_start.elapsed().as_secs_f64() * 1000.0;

        let domain = vk.domain_separator();
        let report = json!({
            "schema": "spark-fixed-cache-startup-v1",
            "scope": "key loading, authentication, and one native fixed-oracle cache",
            "peak_rss_scope": "complete process; authentication and cache construction share allocator and DFT state",
            "verifying_key_json_sha256": key_sha256,
            "verifying_key_json_bytes": key_bytes,
            "load_ms": load_ms,
            "authentication_ms": authentication_ms,
            "build_ms": build_ms,
            "startup_ms": startup_ms,
            "matrix_bytes": cache.matrix_bytes(),
            "matrix_bytes_scope": "dense codewords; Merkle nodes, key, temporary allocations, and allocator metadata are additional",
            "threads": p3_maybe_rayon::prelude::current_num_threads(),
            "features": if cfg!(feature = "parallel") { "parallel,poseidon1" } else { "poseidon1" },
            "config_identity": {
                "hash_profile": "poseidon1-width16",
                "field_modulus": F::ORDER_U32,
                "extension_degree": 5,
                "domain_separator_sha256": format!("{:x}", Sha256::digest(domain.to_bytes())),
                "domain_separator": domain,
            },
        });
        write_report(&report)?;
        // Keep the authenticated key and retained cache alive through reporting.
        black_box((&vk, &cache));
        Ok(())
    }

    #[derive(Deserialize)]
    struct SelectedSetup {
        setup: PoseidonZkSetupConfig,
        fresh_mask_batching: String,
        relation_digest: String,
        shape: Shape,
    }

    #[derive(Deserialize)]
    struct Shape {
        num_cons: usize,
        num_vars: usize,
        num_io: usize,
    }

    fn prepare_key(r1cs: &Path, report: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
        let selected: SelectedSetup = serde_json::from_reader(BufReader::new(File::open(report)?))?;
        let batching = match selected.fresh_mask_batching.as_str() {
            "separate" => FreshMaskBatching::Separate,
            "same_height" => FreshMaskBatching::SameHeight,
            _ => return Err(io::Error::other("unknown fresh_mask_batching in size report").into()),
        };
        let circuit = import_r1cs_path(r1cs)?;
        if (
            circuit.shape.num_cons,
            circuit.shape.num_vars,
            circuit.shape.num_io,
        ) != (
            selected.shape.num_cons,
            selected.shape.num_vars,
            selected.shape.num_io,
        ) {
            return Err(
                io::Error::other("R1CS dimensions do not match the selected size report").into(),
            );
        }
        let (pk, vk) = Poseidon1ZkProvingKey::<QuinticExtension>::setup_with_fresh_mask_batching(
            circuit.shape,
            selected.setup,
            batching,
        )
        .map_err(protocol_error)?;
        drop(pk);
        let relation_digest = vk
            .domain_separator()
            .relation_digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        if relation_digest != selected.relation_digest {
            return Err(io::Error::other(
                "canonical relation digest does not match the selected size report",
            )
            .into());
        }
        let mut output = BufWriter::new(File::options().write(true).create_new(true).open(output)?);
        serde_json::to_writer(&mut output, &vk)?;
        output.write_all(b"\n")?;
        output.flush()?;
        write_report(&json!({
            "schema": "spark-fixed-cache-key-v1",
            "scope": "separate key preparation; excluded from verifier startup measurements",
            "relation_digest": relation_digest,
            "fresh_mask_batching": selected.fresh_mask_batching,
        }))
    }

    fn protocol_error(error: SpartanWhirError) -> io::Error {
        io::Error::other(error.to_string())
    }

    fn write_report(report: &serde_json::Value) -> Result<(), Box<dyn Error>> {
        let mut output = io::stdout().lock();
        serde_json::to_writer_pretty(&mut output, report)?;
        output.write_all(b"\n")?;
        Ok(())
    }
}
