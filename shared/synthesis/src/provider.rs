// LLM provider implementations for synthesis.
// Each provider stays within its documented free tier limits.
// Only official APIs are used — no browser simulation, no ToS violations.

use crate::ProviderConfig;
use async_trait::async_trait;
use serde_json::{json, Value};
use std::time::Duration;

#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("API error {status}: {message}")]
    Api { status: u16, message: String },
    #[error("Empty response from provider")]
    EmptyResponse,
    #[error("Rate limit exceeded")]
    RateLimit,
}

/// Core trait for any LLM provider
#[async_trait]
pub trait SynthesisProvider: Send + Sync {
    fn name(&self) -> &str;
    /// Default RPM for this provider's free tier
    fn default_rpm(&self) -> u32;
    /// Default RPD for this provider's free tier
    fn default_rpd(&self) -> u32;
    async fn generate(&self, system: &str, user: &str) -> Result<String, ProviderError>;
}

/// Build a provider from config
pub fn build_provider(config: &ProviderConfig) -> Box<dyn SynthesisProvider> {
    match config.name.as_str() {
        "groq" => Box::new(GroqProvider::new(config.api_key.clone(), config.model.clone())),
        "gemini" => Box::new(GeminiProvider::new(config.api_key.clone(), config.model.clone())),
        "deepseek" => Box::new(OpenAICompatProvider::new(
            config.api_key.clone(),
            config.model.clone(),
            "https://api.deepseek.com/v1/chat/completions".to_string(),
            "deepseek",
            14,    // ~$0.14/M input tokens, use conservatively
            1000,
        )),
        "mistral" => Box::new(OpenAICompatProvider::new(
            config.api_key.clone(),
            config.model.clone(),
            "https://api.mistral.ai/v1/chat/completions".to_string(),
            "mistral",
            1,    // Free tier: 1 RPM for most models
            200,
        )),
        "openrouter" => Box::new(OpenAICompatProvider::new(
            config.api_key.clone(),
            config.model.clone(),
            "https://openrouter.ai/api/v1/chat/completions".to_string(),
            "openrouter",
            20,
            200,
        )),
        "together" => Box::new(OpenAICompatProvider::new(
            config.api_key.clone(),
            config.model.clone(),
            "https://api.together.xyz/v1/chat/completions".to_string(),
            "together",
            60,
            1000,
        )),
        other => {
            tracing::warn!("Unknown provider '{}', defaulting to OpenAI-compatible", other);
            Box::new(OpenAICompatProvider::new(
                config.api_key.clone(),
                config.model.clone(),
                format!("https://api.{}.com/v1/chat/completions", other),
                other,
                10,
                100,
            ))
        }
    }
}

// ─── Groq ───────────────────────────────────────────────────────────────────

pub struct GroqProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
}

impl GroqProvider {
    pub fn new(api_key: String, model: String) -> Self {
        let model = if model.is_empty() { "llama-3.3-70b-versatile".to_string() } else { model };
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(60))
                .build()
                .unwrap(),
            api_key,
            model,
        }
    }
}

#[async_trait]
impl SynthesisProvider for GroqProvider {
    fn name(&self) -> &str { "groq" }
    fn default_rpm(&self) -> u32 { 30 }
    fn default_rpd(&self) -> u32 { 14_400 }

    async fn generate(&self, system: &str, user: &str) -> Result<String, ProviderError> {
        let body = json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "max_tokens": 2048,
            "temperature": 0.7
        });

        let resp = self.client
            .post("https://api.groq.com/openai/v1/chat/completions")
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        parse_openai_response(resp).await
    }
}

// ─── Google Gemini ───────────────────────────────────────────────────────────

pub struct GeminiProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
}

impl GeminiProvider {
    pub fn new(api_key: String, model: String) -> Self {
        let model = if model.is_empty() { "gemini-1.5-flash".to_string() } else { model };
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(60))
                .build()
                .unwrap(),
            api_key,
            model,
        }
    }
}

#[async_trait]
impl SynthesisProvider for GeminiProvider {
    fn name(&self) -> &str { "gemini" }
    fn default_rpm(&self) -> u32 { 15 }
    fn default_rpd(&self) -> u32 { 1_500 }

    async fn generate(&self, system: &str, user: &str) -> Result<String, ProviderError> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            self.model, self.api_key
        );

        let body = json!({
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": user}]}],
            "generationConfig": {
                "maxOutputTokens": 2048,
                "temperature": 0.7
            }
        });

        let resp = self.client.post(&url).json(&body).send().await?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            if status == 429 { return Err(ProviderError::RateLimit); }
            let text = resp.text().await.unwrap_or_default();
            return Err(ProviderError::Api { status, message: text });
        }

        let data: Value = resp.json().await?;
        let text = data["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .ok_or(ProviderError::EmptyResponse)?
            .to_string();

        Ok(text)
    }
}

// ─── OpenAI-compatible (DeepSeek, Mistral, OpenRouter, Together) ────────────

pub struct OpenAICompatProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
    endpoint: String,
    name: String,
    rpm: u32,
    rpd: u32,
}

impl OpenAICompatProvider {
    pub fn new(
        api_key: String,
        model: String,
        endpoint: String,
        name: &str,
        rpm: u32,
        rpd: u32,
    ) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(90))
                .build()
                .unwrap(),
            api_key,
            model,
            endpoint,
            name: name.to_string(),
            rpm,
            rpd,
        }
    }
}

#[async_trait]
impl SynthesisProvider for OpenAICompatProvider {
    fn name(&self) -> &str { &self.name }
    fn default_rpm(&self) -> u32 { self.rpm }
    fn default_rpd(&self) -> u32 { self.rpd }

    async fn generate(&self, system: &str, user: &str) -> Result<String, ProviderError> {
        let body = json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "max_tokens": 2048,
            "temperature": 0.7
        });

        let resp = self.client
            .post(&self.endpoint)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        parse_openai_response(resp).await
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

async fn parse_openai_response(resp: reqwest::Response) -> Result<String, ProviderError> {
    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        if status == 429 { return Err(ProviderError::RateLimit); }
        let text = resp.text().await.unwrap_or_default();
        return Err(ProviderError::Api { status, message: text });
    }

    let data: Value = resp.json().await?;
    let text = data["choices"][0]["message"]["content"]
        .as_str()
        .ok_or(ProviderError::EmptyResponse)?
        .to_string();

    Ok(text)
}
