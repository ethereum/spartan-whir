#![cfg(unix)]

use std::{
    env, fs,
    os::unix::ffi::OsStrExt,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use p3_field::PrimeCharacteristicRing;
use spartan_whir::{
    circom::KOALABEAR_MODULUS, engine::F, PoseidonWitnessGenerator, PoseidonWitnessGeneratorError,
    PoseidonWitnessLibraryError,
};

// The fixture exercises the published native ABI and dynamic-library lifetime.
// It performs no circuit computation and links only the C runtime.
const NATIVE_SOURCE: &str = r#"
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EXPORT __attribute__((visibility("default")))
static const char *event_file = __EVENT_FILE__;

static void event(const char *name) {
    FILE *file = fopen(event_file, "ab");
    if (file) {
        fprintf(file, "%s\n", name);
        fclose(file);
    }
}

__attribute__((constructor)) static void opened(void) { event("open"); }
__attribute__((destructor)) static void unloaded(void) { event("unload"); }

static void error_message(uint8_t *destination, size_t size, const char *message) {
    if (size) {
        size_t length = strlen(message);
        if (length >= size) length = size - 1;
        memcpy(destination, message, length);
        destination[length] = 0;
    }
}

typedef struct { uint32_t magic; } Circuit;

#ifndef OMIT_LOAD
EXPORT void *fixture_load_circuit(const uint8_t *data, size_t length,
                                 uint8_t *error, size_t error_length) {
    event("load");
    if (length != 3 || memcmp(data, "dat", 3)) {
        error_message(error, error_length, "fixture rejected circuit data");
        return NULL;
    }
    Circuit *circuit = malloc(sizeof(Circuit));
    if (!circuit) {
        error_message(error, error_length, "fixture allocation failed");
        return NULL;
    }
    circuit->magic = 17;
    return circuit;
}
#endif

#ifndef OMIT_GENERATE
EXPORT int fixture_linked_witness(void *handle,
                                 const uint8_t *input, size_t input_length,
                                 uint32_t *witness, size_t witness_length,
                                 uint32_t *public_inputs, size_t public_length,
                                 uint8_t *error, size_t error_length) {
    event("generate");
    Circuit *circuit = handle;
    if (!circuit || circuit->magic != 17 || input_length != 1 ||
        witness_length != 2 || public_length != 2) {
        error_message(error, error_length, "fixture received invalid ABI arguments");
        return 9;
    }
    if (input[0] == 1) {
        error_message(error, error_length, "fixture generation failed");
        return 7;
    }
    witness[0] = input[0] == 2 ? 2130706433u : 7u;
    witness[1] = 11;
    public_inputs[0] = 13;
    public_inputs[1] = input[0] == 3 ? 2130706433u : circuit->magic;
    return 0;
}
#endif

#ifndef OMIT_FREE
EXPORT void fixture_free_circuit(void *handle) {
    Circuit *circuit = handle;
    if (circuit && circuit->magic == 17) {
        circuit->magic = 0;
        free(circuit);
        event("free");
    } else {
        event("invalid free");
    }
}
#endif
"#;

struct Fixture {
    directory: PathBuf,
    library: PathBuf,
    events: PathBuf,
}

impl Fixture {
    fn build(omit: Option<&str>) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = env::temp_dir().join(format!(
            "spartan-witness-library-{}-{nonce}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&directory).unwrap();
        let library = directory.join(format!(
            "{}fixture.{}",
            env::consts::DLL_PREFIX,
            env::consts::DLL_EXTENSION
        ));
        let events = directory.join("events.txt");
        let fixture = Self {
            directory,
            library,
            events,
        };
        let source = fixture.directory.join("fixture.c");
        fs::write(
            &source,
            NATIVE_SOURCE.replace("__EVENT_FILE__", &c_string(&fixture.events)),
        )
        .unwrap();
        let mut compiler = Command::new(env::var_os("CC").unwrap_or_else(|| "cc".into()));
        compiler.args(["-std=c11", "-O0", "-fPIC", "-fvisibility=hidden"]);
        if cfg!(target_os = "macos") {
            compiler.arg("-dynamiclib");
        } else {
            compiler.arg("-shared");
        }
        if let Some(omit) = omit {
            compiler.arg(format!("-D{omit}"));
        }
        let output = compiler
            .arg(&source)
            .arg("-o")
            .arg(&fixture.library)
            .output()
            .expect("run C compiler for native witness fixture");
        assert!(
            output.status.success(),
            "native fixture compilation failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        fixture
    }

    fn events(&self) -> Vec<String> {
        match fs::read_to_string(&self.events) {
            Ok(contents) => contents.lines().map(str::to_owned).collect(),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(error) => panic!("read native fixture events: {error}"),
        }
    }

    fn load(&self, data: &[u8]) -> Result<PoseidonWitnessGenerator, PoseidonWitnessLibraryError> {
        // SAFETY: this fixture is compiled above from ABI-compatible source.
        // Its initializers and callbacks obey the loader's documented contract.
        unsafe { PoseidonWitnessGenerator::load_library("fixture", &self.library, "fixture", data) }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}

fn c_string(path: &Path) -> String {
    let mut text = String::from("\"");
    for byte in path.as_os_str().as_bytes() {
        use std::fmt::Write;
        write!(&mut text, "\\x{byte:02x}").unwrap();
    }
    text.push('"');
    text
}

#[test]
fn owned_library_resolves_callbacks_and_frees_before_unloading() {
    let fixture = Fixture::build(None);
    let data = Vec::from(&b"dat"[..]);
    let generator = fixture.load(&data).expect("load native fixture");
    drop(data);
    assert_eq!(generator.name(), "fixture");
    assert_eq!(fixture.events(), ["open", "load"]);
    let (witness, public) = generator.generate_witness([0], 2, 2).unwrap();
    assert_eq!(witness.w, [F::from_u32(7), F::from_u32(11)]);
    assert_eq!(public, [F::from_u32(13), F::from_u32(17)]);
    assert_eq!(fixture.events(), ["open", "load", "generate"]);
    drop(generator);
    assert_eq!(
        fixture.events(),
        ["open", "load", "generate", "free", "unload"]
    );
}

#[test]
fn missing_library_and_each_missing_callback_report_loader_errors() {
    let fixture = Fixture::build(None);
    // SAFETY: the nonexistent path cannot initialize or call native code.
    let error = unsafe {
        PoseidonWitnessGenerator::load_library(
            "missing",
            fixture.directory.join("absent"),
            "fixture",
            b"dat",
        )
    }
    .expect_err("missing library must fail");
    assert!(
        matches!(error, PoseidonWitnessLibraryError::LibraryLoadFailed { name: "missing", ref message } if !message.is_empty())
    );
    assert!(fixture.events().is_empty());

    for (omit, symbol) in [
        ("OMIT_LOAD", "fixture_load_circuit"),
        ("OMIT_GENERATE", "fixture_linked_witness"),
        ("OMIT_FREE", "fixture_free_circuit"),
    ] {
        let fixture = Fixture::build(Some(omit));
        let error = fixture
            .load(b"dat")
            .expect_err("missing callback must fail");
        match error {
            PoseidonWitnessLibraryError::SymbolLoadFailed {
                name,
                symbol: actual,
                message,
            } => {
                assert_eq!(name, "fixture");
                assert_eq!(actual, symbol);
                assert!(!message.is_empty());
            }
            other => panic!("unexpected error for {symbol}: {other:?}"),
        }
        assert_eq!(
            fixture.events(),
            ["open", "unload"],
            "failed resolution must unload without loading or freeing a circuit"
        );
    }
}

#[test]
fn rejected_native_load_unloads_the_library_without_freeing_a_null_handle() {
    let fixture = Fixture::build(None);
    assert_eq!(
        fixture.load(b"reject").unwrap_err(),
        PoseidonWitnessLibraryError::Generator(PoseidonWitnessGeneratorError::CircuitLoadFailed {
            name: "fixture",
            message: "fixture rejected circuit data".to_owned()
        })
    );
    assert_eq!(fixture.events(), ["open", "load", "unload"]);
}

#[test]
fn dynamic_callbacks_keep_generation_errors_and_canonical_output_checks() {
    let fixture = Fixture::build(None);
    let generator = fixture.load(b"dat").unwrap();
    assert_eq!(
        generator.generate_witness([1], 2, 2).unwrap_err(),
        PoseidonWitnessGeneratorError::GeneratorFailed {
            name: "fixture",
            code: 7,
            message: "fixture generation failed".to_owned()
        }
    );
    for mode in [2, 3] {
        assert_eq!(
            generator.generate_witness([mode], 2, 2).unwrap_err(),
            PoseidonWitnessGeneratorError::InvalidFieldElement {
                value: KOALABEAR_MODULUS
            }
        );
    }
    let (witness, public) = generator.generate_witness([0], 2, 2).unwrap();
    assert_eq!(witness.w, [F::from_u32(7), F::from_u32(11)]);
    assert_eq!(public, [F::from_u32(13), F::from_u32(17)]);
    assert_eq!(
        fixture.events(),
        ["open", "load", "generate", "generate", "generate", "generate"]
    );
    drop(generator);
    assert_eq!(
        fixture.events(),
        ["open", "load", "generate", "generate", "generate", "generate", "free", "unload"]
    );
}
