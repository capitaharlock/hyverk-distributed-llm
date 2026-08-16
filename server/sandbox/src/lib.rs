// @llm-context: .meshkore/modules/general/README.md
// Phase 3: Execution Verification
// Extracts Rust code from LLM responses, compiles it in a temp Cargo project,
// runs tests if present, returns pass/fail + compiler output.
// Goal: only execution-verified examples enter the training dataset.

pub mod extractor;
pub mod runner;

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Result of executing code in the sandbox
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub passed: bool,
    pub stage: VerificationStage,
    pub stdout: String,
    pub stderr: String,
    pub duration_ms: u64,
    /// Quality signals derived from compilation
    pub signals: QualitySignals,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum VerificationStage {
    /// No Rust code block found in response
    NoCode,
    /// Code extracted but failed to compile
    CompileFailed,
    /// Compiled, no tests found
    CompiledNoTests,
    /// Compiled, tests ran and passed
    TestsPassed,
    /// Compiled, tests ran but some failed
    TestsFailed,
    /// Process timed out
    Timeout,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QualitySignals {
    pub has_doc_comments: bool,
    pub has_tests: bool,
    pub has_error_handling: bool,
    pub function_count: usize,
    pub line_count: usize,
    pub warning_count: usize,
}

/// Verify Rust code extracted from a response string.
/// Creates a temp Cargo project, compiles + optionally tests.
/// Timeout defaults to 60 seconds (llama.cpp takes time to compile heavy deps).
pub async fn verify_response(
    response: &str,
    timeout: Option<Duration>,
) -> VerificationResult {
    let timeout = timeout.unwrap_or(Duration::from_secs(60));

    // Extract rust code blocks
    let code_blocks = extractor::extract_rust_blocks(response);
    if code_blocks.is_empty() {
        return VerificationResult {
            passed: false,
            stage: VerificationStage::NoCode,
            stdout: String::new(),
            stderr: String::new(),
            duration_ms: 0,
            signals: QualitySignals {
                line_count: response.lines().count(),
                ..Default::default()
            },
        };
    }

    // Use the largest code block (most likely the main implementation)
    let code = code_blocks.into_iter().max_by_key(|s| s.len()).unwrap();
    let signals = extractor::analyze_code(&code);

    runner::run_in_sandbox(&code, timeout, signals).await
}

/// Quick check: does the response contain any Rust code?
pub fn has_rust_code(response: &str) -> bool {
    extractor::extract_rust_blocks(response).into_iter().any(|b| b.len() > 20)
}
