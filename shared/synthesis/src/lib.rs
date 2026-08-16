// @llm-context: .meshkore/modules/general/README.md
// Phase 2: Distributed Synthesis Engine
// Each node calls free LLM APIs to generate (instruction, code) training pairs.
// With 1M nodes × 14.400 Groq calls/day = 14.4B examples/day.

pub mod prompts;
pub mod provider;
pub mod worker;

use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

/// Configuration for a single LLM provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub name: String,       // "groq", "gemini", "deepseek", "mistral", "openrouter"
    pub api_key: String,
    pub model: String,
    /// Requests per minute (respect free tier)
    pub rpm_limit: Option<u32>,
    /// Requests per day (respect free tier)
    pub rpd_limit: Option<u32>,
}

/// Synthesis configuration in config.toml [synthesis]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SynthesisConfig {
    #[serde(default)]
    pub enabled: bool,
    /// Target examples to generate per hour
    #[serde(default = "default_target_per_hour")]
    pub target_per_hour: u32,
    /// Enable multi-LLM refinement (higher quality, uses 3x API calls)
    #[serde(default)]
    pub enable_refinement: bool,
    /// Coordinator URL for submitting examples
    #[serde(default = "default_coordinator_url")]
    pub coordinator_url: String,
    #[serde(default)]
    pub providers: Vec<ProviderConfig>,
}

fn default_target_per_hour() -> u32 { 50 }
fn default_coordinator_url() -> String { "http://127.0.0.1:17000".to_string() }

/// A generated training example ready to submit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedExample {
    pub id: String,
    pub instruction: String,
    pub response: String,
    pub category: String,
    pub provider: String,
    pub model: String,
    /// Was this refined through multi-LLM pipeline?
    #[serde(default)]
    pub refined: bool,
    /// Execution verified (code compiled/tested)?
    #[serde(default)]
    pub execution_verified: bool,
}

/// Start the synthesis engine. Returns when cancelled or on fatal error.
pub async fn run_synthesis(
    config: &SynthesisConfig,
    node_id: &str,
    shutdown: CancellationToken,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if config.providers.is_empty() {
        tracing::warn!("Synthesis enabled but no providers configured. Add [[synthesis.providers]] to config.toml");
        return Ok(());
    }

    tracing::info!(
        providers = config.providers.len(),
        target_per_hour = config.target_per_hour,
        "Starting synthesis engine"
    );

    let worker = worker::SynthesisWorker::new(config.clone(), node_id.to_string());
    worker.run(shutdown).await
}
