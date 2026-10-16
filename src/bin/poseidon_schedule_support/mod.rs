use std::{
    env,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkProvenance {
    pub schema_version: u32,
    pub code: CodeProvenance,
    pub build: BuildProvenance,
}

#[derive(Debug, Clone, Serialize)]
pub struct CodeProvenance {
    pub spartan_whir: RepositoryProvenance,
    pub plonky3: RepositoryProvenance,
}

#[derive(Debug, Clone, Serialize)]
pub struct RepositoryProvenance {
    pub head: String,
    pub branch: Option<String>,
    pub dirty: bool,
    pub status_porcelain: String,
    pub tracked_diff_hash: String,
    pub untracked_files: Vec<UntrackedFile>,
}

#[derive(Debug, Clone, Serialize)]
pub struct UntrackedFile {
    pub path: String,
    pub blob_hash: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct BuildProvenance {
    pub rustc_version: String,
    pub profile: &'static str,
    pub features: String,
    pub rustflags: String,
    pub target_cpu_native: bool,
}

pub fn collect(features: impl Into<String>) -> Result<BenchmarkProvenance, String> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let plonky3 = manifest_dir
        .parent()
        .ok_or_else(|| "spartan-whir manifest has no parent directory".to_owned())?
        .join("Plonky3");
    let rustflags = env::var("RUSTFLAGS").unwrap_or_default();
    Ok(BenchmarkProvenance {
        schema_version: 1,
        code: CodeProvenance {
            spartan_whir: repository(&manifest_dir)?,
            plonky3: repository(&plonky3)?,
        },
        build: BuildProvenance {
            rustc_version: command_text(Path::new("."), "rustc", &["--version", "--verbose"])?
                .trim()
                .to_owned(),
            profile: if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            },
            features: features.into(),
            target_cpu_native: rustflags.contains("target-cpu=native"),
            rustflags,
        },
    })
}

fn repository(path: &Path) -> Result<RepositoryProvenance, String> {
    let head = git_text(path, &["rev-parse", "HEAD"])?;
    let branch = git_text(path, &["branch", "--show-current"])?;
    let status_porcelain = git_text(path, &["status", "--porcelain=v1", "--untracked-files=all"])?;
    let diff = git_bytes(path, &["diff", "--binary", "HEAD"])?;
    let tracked_diff_hash = hash_stdin(path, &diff)?;
    let untracked_raw = git_bytes(path, &["ls-files", "--others", "--exclude-standard", "-z"])?;
    let mut untracked_files = Vec::new();
    for raw in untracked_raw
        .split(|byte| *byte == 0)
        .filter(|raw| !raw.is_empty())
    {
        let relative = String::from_utf8(raw.to_vec())
            .map_err(|_| format!("{} contains a non-UTF-8 untracked path", path.display()))?;
        let blob_hash = git_text(path, &["hash-object", "--", &relative])?;
        untracked_files.push(UntrackedFile {
            path: relative,
            blob_hash: blob_hash.trim().to_owned(),
        });
    }
    Ok(RepositoryProvenance {
        head: head.trim().to_owned(),
        branch: (!branch.trim().is_empty()).then(|| branch.trim().to_owned()),
        dirty: !status_porcelain.trim().is_empty(),
        status_porcelain: status_porcelain.trim_end().to_owned(),
        tracked_diff_hash: tracked_diff_hash.trim().to_owned(),
        untracked_files,
    })
}

fn hash_stdin(path: &Path, bytes: &[u8]) -> Result<String, String> {
    let mut child = Command::new("git")
        .args(["hash-object", "--stdin"])
        .current_dir(path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("failed to start git hash-object: {error}"))?;
    child
        .stdin
        .take()
        .ok_or_else(|| "git hash-object stdin unavailable".to_owned())?
        .write_all(bytes)
        .map_err(|error| format!("failed to write git hash-object input: {error}"))?;
    let output = child
        .wait_with_output()
        .map_err(|error| format!("failed to wait for git hash-object: {error}"))?;
    checked_output("git hash-object --stdin", output)
}

fn git_text(path: &Path, args: &[&str]) -> Result<String, String> {
    command_text(path, "git", args)
}

fn git_bytes(path: &Path, args: &[&str]) -> Result<Vec<u8>, String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(path)
        .output()
        .map_err(|error| format!("failed to run git {}: {error}", args.join(" ")))?;
    if output.status.success() {
        Ok(output.stdout)
    } else {
        Err(format!(
            "git {} failed in {}: {}",
            args.join(" "),
            path.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    }
}

fn command_text(path: &Path, program: &str, args: &[&str]) -> Result<String, String> {
    let output = Command::new(program)
        .args(args)
        .current_dir(path)
        .output()
        .map_err(|error| format!("failed to run {program} {}: {error}", args.join(" ")))?;
    checked_output(&format!("{program} {}", args.join(" ")), output)
}

fn checked_output(command: &str, output: std::process::Output) -> Result<String, String> {
    if !output.status.success() {
        return Err(format!(
            "{command} failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    String::from_utf8(output.stdout).map_err(|_| format!("{command} emitted non-UTF-8 output"))
}

pub const fn enabled_features() -> &'static str {
    match (cfg!(feature = "parallel"), cfg!(feature = "poseidon1")) {
        (true, true) => "parallel,poseidon1",
        (true, false) => "parallel",
        (false, true) => "poseidon1",
        (false, false) => "",
    }
}
