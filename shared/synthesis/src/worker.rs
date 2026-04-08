// SynthesisWorker: main loop that generates training examples and sends them to coordinator.
// Rate-limited per provider to stay within free tier. Runs indefinitely until cancelled.

use crate::{
    prompts::{self, HYVERK_SYSTEM, CRITIC_SYSTEM, REFINER_SYSTEM},
    provider::{self, ProviderError},
    GeneratedExample, SynthesisConfig,
};
use hyverk_sandbox::VerificationStage;
use serde_json::json;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};
use uuid::Uuid;

pub struct SynthesisWorker {
    config: SynthesisConfig,
    node_id: String,
    /// Monotonically increasing counter used to pick prompts (cycles through bank)
    counter: Arc<AtomicU64>,
    client: reqwest::Client,
}

impl SynthesisWorker {
    pub fn new(config: SynthesisConfig, node_id: String) -> Self {
        Self {
            config,
            node_id,
            counter: Arc::new(AtomicU64::new(rand_seed())),
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .unwrap(),
        }
    }

    pub async fn run(
        self,
        shutdown: CancellationToken,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Build provider list
        let providers: Vec<_> = self
            .config
            .providers
            .iter()
            .map(|c| provider::build_provider(c))
            .collect();

        if providers.is_empty() {
            warn!("No synthesis providers configured");
            return Ok(());
        }

        // Calculate inter-request delay to hit target_per_hour
        // Divide target across all providers
        let total_rpm_budget: u32 = providers.iter()
            .map(|p| p.default_rpm().min(5)) // cap at 5 RPM per provider to be conservative
            .sum();
        let delay_secs = if total_rpm_budget > 0 {
            60.0 / total_rpm_budget as f64
        } else {
            12.0 // default: 5/min
        };

        let target_interval = Duration::from_secs_f64(delay_secs.max(2.0));
        info!(
            providers = providers.len(),
            delay_ms = target_interval.as_millis(),
            "Synthesis worker starting"
        );

        // Daily counters (reset each day)
        let mut daily_counts: Vec<u32> = vec![0; providers.len()];
        let mut day_start = Instant::now();

        let mut provider_idx: usize = 0;

        loop {
            tokio::select! {
                _ = shutdown.cancelled() => {
                    info!("Synthesis worker shutting down");
                    return Ok(());
                }
                _ = tokio::time::sleep(target_interval) => {}
            }

            // Reset daily counters after 24h
            if day_start.elapsed() > Duration::from_secs(86_400) {
                daily_counts.iter_mut().for_each(|c| *c = 0);
                day_start = Instant::now();
            }

            // Pick provider (round-robin, skip if daily limit hit)
            let start_idx = provider_idx;
            let provider = loop {
                let p = &providers[provider_idx];
                let daily_limit = self.config.providers[provider_idx]
                    .rpd_limit
                    .unwrap_or(p.default_rpd());

                if daily_counts[provider_idx] < daily_limit {
                    break p;
                }

                provider_idx = (provider_idx + 1) % providers.len();
                if provider_idx == start_idx {
                    // All providers exhausted for the day
                    warn!("All synthesis providers hit daily limit, sleeping 1h");
                    tokio::select! {
                        _ = shutdown.cancelled() => return Ok(()),
                        _ = tokio::time::sleep(Duration::from_secs(3600)) => {}
                    }
                    daily_counts.iter_mut().for_each(|c| *c = 0);
                    break &providers[0];
                }
            };

            // Pick prompt
            let seed = self.counter.fetch_add(1, Ordering::Relaxed);
            let prompt = prompts::random_prompt(seed);

            // Generate
            info!(
                category = %prompt.category,
                provider = provider.name(),
                "Generating example"
            );

            match self.generate_example(provider.as_ref(), &prompt, seed).await {
                Ok(mut example) => {
                    daily_counts[provider_idx] += 1;

                    // Verify Rust code compiles (Phase 3 execution verification)
                    if hyverk_sandbox::has_rust_code(&example.response) {
                        let result = hyverk_sandbox::verify_response(
                            &example.response,
                            Some(std::time::Duration::from_secs(60)),
                        ).await;

                        example.execution_verified = result.passed;

                        match result.stage {
                            VerificationStage::TestsPassed => {
                                info!(category = %example.category, "Code verified: tests passed");
                            }
                            VerificationStage::CompiledNoTests => {
                                info!(category = %example.category, "Code verified: compiled OK");
                            }
                            VerificationStage::CompileFailed => {
                                info!(category = %example.category, "Code compile failed — submitting unverified");
                            }
                            VerificationStage::TestsFailed => {
                                info!(category = %example.category, "Tests failed — submitting unverified");
                            }
                            _ => {}
                        }
                    }

                    if let Err(e) = self.submit_example(&example).await {
                        error!("Failed to submit example: {e}");
                    } else {
                        info!(
                            category = %example.category,
                            provider = %example.provider,
                            verified = example.execution_verified,
                            "Example submitted"
                        );
                    }
                }
                Err(ProviderError::RateLimit) => {
                    warn!("Rate limit hit for {}, skipping", provider.name());
                    daily_counts[provider_idx] = daily_counts[provider_idx].saturating_add(100);
                }
                Err(e) => {
                    error!("Generation error ({}): {e}", provider.name());
                }
            }

            provider_idx = (provider_idx + 1) % providers.len();
        }
    }

    async fn generate_example(
        &self,
        provider: &dyn crate::provider::SynthesisProvider,
        prompt: &prompts::Prompt,
        seed: u64,
    ) -> Result<GeneratedExample, ProviderError> {
        let response = provider.generate(HYVERK_SYSTEM, &prompt.instruction).await?;

        // Basic quality filter
        if response.len() < 50 {
            return Err(ProviderError::EmptyResponse);
        }

        let mut example = GeneratedExample {
            id: Uuid::new_v4().to_string(),
            instruction: prompt.instruction.clone(),
            response: response.clone(),
            category: prompt.category.clone(),
            provider: provider.name().to_string(),
            model: String::new(), // provider knows its model
            refined: false,
            execution_verified: false,
        };

        // Optional multi-LLM refinement (only if enabled and we have 2+ providers)
        if self.config.enable_refinement && self.config.providers.len() >= 2 {
            let refiner_idx = (seed as usize + 1) % self.config.providers.len();
            let refiner = crate::provider::build_provider(&self.config.providers[refiner_idx]);

            if let Ok(refined) = self.refine_with_critique(
                provider,
                refiner.as_ref(),
                &prompt.instruction,
                &response,
            ).await {
                example.response = refined;
                example.refined = true;
            }
        }

        Ok(example)
    }

    /// Multi-LLM refinement: generate → critique → improve
    async fn refine_with_critique(
        &self,
        _generator: &dyn crate::provider::SynthesisProvider,
        critiquer: &dyn crate::provider::SynthesisProvider,
        instruction: &str,
        draft: &str,
    ) -> Result<String, ProviderError> {
        // Step 1: get critique
        let critique_prompt = format!(
            "Review this code response to the instruction:\n\n## Instruction\n{}\n\n## Response\n{}\n\nProvide a concise critique (max 200 words): bugs, missing error handling, style issues.",
            instruction, draft
        );
        let critique = critiquer.generate(CRITIC_SYSTEM, &critique_prompt).await?;

        // Step 2: refine based on critique (use same critiquer to save API calls)
        let refine_prompt = format!(
            "Improve this code based on the critique:\n\n## Original instruction\n{}\n\n## Draft code\n{}\n\n## Critique\n{}\n\nProvide only the improved code with a brief explanation.",
            instruction, draft, critique
        );
        let refined = critiquer.generate(REFINER_SYSTEM, &refine_prompt).await?;

        Ok(refined)
    }

    async fn submit_example(
        &self,
        example: &GeneratedExample,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let url = format!("{}/api/v1/dataset/examples", self.config.coordinator_url);
        let body = json!({
            "example": example,
            "node_id": self.node_id
        });

        let resp = self.client.post(&url).json(&body).send().await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("Submit failed: {status} — {text}").into());
        }

        Ok(())
    }
}

/// Simple entropy from process ID + time for initial counter seed
fn rand_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    let pid = std::process::id() as u64;
    t ^ (pid << 32) ^ (pid * 6364136223846793005)
}
