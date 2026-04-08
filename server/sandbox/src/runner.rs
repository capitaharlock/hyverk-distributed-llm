// Runs Rust code in a temporary Cargo project, captures output.
// MVP: no network isolation or memory limits — just a temp dir + timeout.
// Production: wrap in Docker/container for full isolation.

use crate::{QualitySignals, VerificationResult, VerificationStage};
use std::path::Path;
use std::time::{Duration, Instant};
use tracing::debug;

/// Cargo.toml template for the sandbox project
const CARGO_TOML: &str = r#"[package]
name = "sandbox"
version = "0.1.0"
edition = "2021"

[dependencies]
"#;

/// Wrapper to make any code compile as a library (adds test harness support)
fn wrap_code(code: &str) -> String {
    // If code already has fn main or mod tests, use as-is
    // Otherwise wrap in a way that allows compilation
    if code.contains("fn main()") || code.contains("fn main ") {
        code.to_string()
    } else {
        // Pure library code — ensure it has #![allow(...)] to reduce noise
        format!("#![allow(dead_code, unused_variables, unused_imports)]\n\n{}", code)
    }
}

pub async fn run_in_sandbox(
    code: &str,
    timeout: Duration,
    mut signals: QualitySignals,
) -> VerificationResult {
    let start = Instant::now();

    // Create temp directory
    let tmp = match tempfile::Builder::new().prefix("hyverk-sandbox-").tempdir() {
        Ok(d) => d,
        Err(e) => {
            return VerificationResult {
                passed: false,
                stage: VerificationStage::CompileFailed,
                stdout: String::new(),
                stderr: format!("Failed to create temp dir: {e}"),
                duration_ms: 0,
                signals,
            };
        }
    };

    let src_dir = tmp.path().join("src");
    if let Err(e) = std::fs::create_dir_all(&src_dir) {
        return failed(format!("mkdir: {e}"), signals);
    }

    // Write Cargo.toml
    if let Err(e) = std::fs::write(tmp.path().join("Cargo.toml"), CARGO_TOML) {
        return failed(format!("write Cargo.toml: {e}"), signals);
    }

    // Write src/lib.rs with the code
    let wrapped = wrap_code(code);
    let lib_path = src_dir.join("lib.rs");
    if let Err(e) = std::fs::write(&lib_path, &wrapped) {
        return failed(format!("write lib.rs: {e}"), signals);
    }

    let has_tests = signals.has_tests;

    // First: try to compile
    match run_cargo(tmp.path(), &["build", "--quiet"], timeout).await {
        Err(stage) => {
            let duration_ms = start.elapsed().as_millis() as u64;
            return VerificationResult {
                passed: false,
                stage,
                stdout: String::new(),
                stderr: "Build timed out".to_string(),
                duration_ms,
                signals,
            };
        }
        Ok(output) => {
            let warning_count = output.stderr
                .lines()
                .filter(|l| l.contains("warning[") || l.trim_start().starts_with("warning: "))
                .count();
            signals.warning_count = warning_count;

            if !output.success {
                return VerificationResult {
                    passed: false,
                    stage: VerificationStage::CompileFailed,
                    stdout: output.stdout,
                    stderr: output.stderr,
                    duration_ms: start.elapsed().as_millis() as u64,
                    signals,
                };
            }

            // If tests exist, run them
            if has_tests {
                let remaining = timeout.saturating_sub(start.elapsed());
                match run_cargo(tmp.path(), &["test", "--quiet"], remaining).await {
                    Err(stage) => VerificationResult {
                        passed: false,
                        stage,
                        stdout: String::new(),
                        stderr: "Test timed out".to_string(),
                        duration_ms: start.elapsed().as_millis() as u64,
                        signals,
                    },
                    Ok(test_output) => {
                        let passed = test_output.success;
                        VerificationResult {
                            passed,
                            stage: if passed {
                                VerificationStage::TestsPassed
                            } else {
                                VerificationStage::TestsFailed
                            },
                            stdout: test_output.stdout,
                            stderr: test_output.stderr,
                            duration_ms: start.elapsed().as_millis() as u64,
                            signals,
                        }
                    }
                }
            } else {
                VerificationResult {
                    passed: true,  // compiled successfully, no tests to fail
                    stage: VerificationStage::CompiledNoTests,
                    stdout: output.stdout,
                    stderr: output.stderr,
                    duration_ms: start.elapsed().as_millis() as u64,
                    signals,
                }
            }
        }
    }
}

struct CargoOutput {
    success: bool,
    stdout: String,
    stderr: String,
}

async fn run_cargo(
    cwd: &Path,
    args: &[&str],
    timeout: Duration,
) -> Result<CargoOutput, VerificationStage> {
    debug!(?args, "Running cargo");

    let result = tokio::time::timeout(timeout, async {
        tokio::process::Command::new("cargo")
            .args(args)
            .current_dir(cwd)
            .env("CARGO_TERM_COLOR", "never")
            // Use a shared target dir to avoid recompiling std every time
            .env(
                "CARGO_TARGET_DIR",
                std::env::temp_dir().join("hyverk-sandbox-target"),
            )
            .output()
            .await
    })
    .await;

    match result {
        Err(_) => Err(VerificationStage::Timeout),
        Ok(Err(_e)) => Err(VerificationStage::CompileFailed),
        Ok(Ok(output)) => Ok(CargoOutput {
            success: output.status.success(),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        }),
    }
}

fn failed(msg: String, signals: QualitySignals) -> VerificationResult {
    VerificationResult {
        passed: false,
        stage: VerificationStage::CompileFailed,
        stdout: String::new(),
        stderr: msg,
        duration_ms: 0,
        signals,
    }
}
