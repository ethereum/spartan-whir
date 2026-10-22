use std::{env, error::Error, fs, path::PathBuf};

use p3_field::PrimeField32;
use sha2::{Digest, Sha256};
use spartan_whir::{engine::F, import_r1cs_path, PoseidonWitnessGenerator, R1csShape};

pub struct Sha256Fixture {
    pub shape: R1csShape<F>,
    pub generator: PoseidonWitnessGenerator,
}

impl Sha256Fixture {
    pub fn load(size: usize) -> Result<Self, Box<dyn Error>> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let cache_root = env::var_os("SHA256_BENCH_WORKDIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| manifest_dir.join("target/sha256-cache"));
        let workdir = cache_root.join(format!("sha256_{size}b"));
        let r1cs = workdir.join(format!("sha256_{size}b.r1cs"));
        let linked_library = workdir.join(dynamic_library_name(size));
        let circuit_data_path = workdir
            .join(format!("sha256_{size}b_cpp"))
            .join(format!("sha256_{size}b.dat"));

        for path in [&r1cs, &linked_library, &circuit_data_path] {
            if !path.is_file() {
                return Err(format!(
                    "missing cached SHA-256 artifact {}; generate it with the sha256_bench example before running Criterion",
                    path.display()
                )
                .into());
            }
        }

        let shape = import_r1cs_path(&r1cs)?.shape;
        let run_name = format!("sha256_{size}b");
        let circuit_data = fs::read(circuit_data_path)?;
        // SAFETY: the cached circuit library implements the linked witness ABI.
        let generator = unsafe {
            PoseidonWitnessGenerator::load_library(
                "sha256_full_zk_criterion",
                &linked_library,
                &run_name,
                &circuit_data,
            )
        }?;

        Ok(Self { shape, generator })
    }

    pub fn validate_input(&self, message: &[u8], input: &[u8]) -> Result<(), Box<dyn Error>> {
        let (_, public_inputs) =
            self.generator
                .generate_witness(input, self.shape.num_vars, self.shape.num_io)?;
        let actual = public_inputs
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let bit = value.as_canonical_u32();
                if bit < 2 {
                    Ok(bit as u8)
                } else {
                    Err(format!("public digest bit {index} is not boolean: {bit}"))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let expected = Sha256::digest(message)
            .iter()
            .flat_map(|byte| (0..8).rev().map(move |bit| (byte >> bit) & 1))
            .collect::<Vec<_>>();
        if actual != expected {
            return Err("linked witness generator returned the wrong SHA-256 digest".into());
        }
        Ok(())
    }
}

pub fn message(size: usize, sample: usize) -> Vec<u8> {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64 ^ sample as u64;
    (0..size)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut mixed = state;
            mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            (mixed ^ (mixed >> 31)) as u8
        })
        .collect()
}

pub fn input_binary(message: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(message.len() * 8 * size_of::<u32>());
    for bit in message
        .iter()
        .flat_map(|byte| (0..8).rev().map(move |bit| u32::from((byte >> bit) & 1)))
    {
        out.extend_from_slice(&bit.to_le_bytes());
    }
    out
}

fn dynamic_library_name(size: usize) -> String {
    format!(
        "{}sha256_{size}b_witness.{}",
        env::consts::DLL_PREFIX,
        env::consts::DLL_EXTENSION
    )
}
