// @llm-context: .meshkore/docs/architecture/overview.md

use hyverk_coordinator::run_coordinator;
use hyverk_core::config::{load_config, HyverkConfig};
use tokio_util::sync::CancellationToken;
use tracing::info;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    let config_path = std::env::var("HYVERK_CONFIG").unwrap_or_else(|_| "config.toml".to_string());
    let config = load_config(&config_path).unwrap_or_else(|e| {
        info!("Using default config: {e}");
        HyverkConfig::default()
    });

    let shutdown = CancellationToken::new();
    let shutdown_sig = shutdown.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        shutdown_sig.cancel();
    });

    run_coordinator(&config.coordinator, shutdown)
        .await
        .map_err(|e| -> Box<dyn std::error::Error> { e })?;
    Ok(())
}
