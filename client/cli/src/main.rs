// @llm-context: .meshkore/docs/architecture/overview.md
// @llm-critical: This is the unified entry point. Runs coordinator, node, or both based on config/CLI.

use clap::{Parser, Subcommand};
use hyverk_core::config::{load_config, HyverkConfig, Mode};
use tokio_util::sync::CancellationToken;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "hyverk", about = "Distributed LLM inference network")]
struct Cli {
    /// Config file path
    #[arg(short, long, default_value = "config.toml", env = "HYVERK_CONFIG")]
    config: String,

    /// Override mode from config
    #[arg(short, long)]
    mode: Option<String>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the network (default behavior)
    Run,
    /// Generate training data using free LLM APIs (synthesis mode)
    Synthesize {
        /// Target examples per hour (overrides config)
        #[arg(long)]
        target_per_hour: Option<u32>,
        /// Run for N hours then exit (0 = run indefinitely)
        #[arg(long, default_value = "0")]
        hours: u64,
    },
    /// Index a source into the RAG knowledge base
    Index {
        /// Source to index: crate:NAME, dir:PATH, or url:URL
        source: String,
        /// RAG database path (default: ~/.hyverk/rag.db)
        #[arg(long, default_value = "~/.hyverk/rag.db")]
        db: String,
    },
    /// Search the RAG knowledge base
    Search {
        /// Search query
        query: String,
        /// Number of results
        #[arg(short, long, default_value = "5")]
        k: usize,
        /// RAG database path
        #[arg(long, default_value = "~/.hyverk/rag.db")]
        db: String,
    },
    /// Run LoRA fine-tuning on a dataset shard (training mode)
    Train {
        /// Path to model directory (safetensors + config.json)
        #[arg(long)]
        model_dir: String,
        /// Path to tokenizer.json
        #[arg(long)]
        tokenizer: String,
        /// Path to dataset JSONL file (or - for stdin)
        #[arg(long)]
        dataset: String,
        /// Output path for LoRA adapter weights (.safetensors)
        #[arg(long, default_value = "adapter.safetensors")]
        output: String,
        /// LoRA rank
        #[arg(long, default_value = "16")]
        lora_rank: usize,
        /// Training epochs
        #[arg(long, default_value = "3")]
        epochs: usize,
        /// Learning rate
        #[arg(long, default_value = "0.0002")]
        lr: f64,
        /// Mini-batch size (lower = less memory). Default 1 for safety.
        #[arg(long, default_value = "1")]
        batch_size: usize,
        /// Max sequence length in tokens (lower = less memory). Default 256.
        #[arg(long, default_value = "256")]
        max_seq_len: usize,
        /// Limit to N examples for fast dev iteration (0 = no limit)
        #[arg(long, default_value = "0")]
        max_examples: usize,
        /// Hard memory limit in GB — aborts training if exceeded (default: 16)
        #[arg(long, default_value = "16")]
        max_memory_gb: u64,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    let cli = Cli::parse();

    let config = load_config(&cli.config).unwrap_or_else(|e| {
        info!("Using default config: {e}");
        HyverkConfig::default()
    });

    // CLI mode override
    let mode = match cli.mode.as_deref() {
        Some("node") => Mode::Node,
        Some("coordinator") => Mode::Coordinator,
        Some("both") => Mode::Both,
        Some(other) => {
            error!("Invalid mode: {other}. Use: node, coordinator, both");
            std::process::exit(1);
        }
        None => config.mode.clone(),
    };

    match cli.command {
        Some(Commands::Synthesize { target_per_hour, hours }) => {
            let syn_config = hyverk_synthesis::SynthesisConfig {
                enabled: true,
                target_per_hour: target_per_hour.unwrap_or(config.synthesis.target_per_hour),
                enable_refinement: config.synthesis.enable_refinement,
                coordinator_url: if config.synthesis.coordinator_url.is_empty() {
                    format!(
                        "http://{}:{}",
                        config.coordinator.bind_addr.replace("0.0.0.0", "127.0.0.1"),
                        config.coordinator.http_port
                    )
                } else {
                    config.synthesis.coordinator_url.clone()
                },
                providers: config.synthesis.providers.iter().map(|p| hyverk_synthesis::ProviderConfig {
                    name: p.name.clone(),
                    api_key: p.api_key.clone(),
                    model: p.model.clone(),
                    rpm_limit: p.rpm_limit,
                    rpd_limit: p.rpd_limit,
                }).collect(),
            };

            if syn_config.providers.is_empty() {
                error!("No synthesis providers configured. Add [[synthesis.providers]] to config.toml");
                error!("Example:\n[[synthesis.providers]]\nname = \"groq\"\napi_key = \"gsk_...\"\nmodel = \"llama-3.3-70b-versatile\"");
                std::process::exit(1);
            }

            let shutdown = CancellationToken::new();
            let shutdown_sig = shutdown.clone();
            tokio::spawn(async move {
                tokio::signal::ctrl_c().await.ok();
                info!("Shutdown signal received");
                shutdown_sig.cancel();
            });

            // Optional time limit
            if hours > 0 {
                let shutdown_timer = shutdown.clone();
                tokio::spawn(async move {
                    tokio::time::sleep(std::time::Duration::from_secs(hours * 3600)).await;
                    info!("Time limit reached ({hours}h), stopping synthesis");
                    shutdown_timer.cancel();
                });
            }

            let node_id = config.node.name.clone();
            hyverk_synthesis::run_synthesis(&syn_config, &node_id, shutdown).await?;
            return Ok(());
        }
        Some(Commands::Index { source, db }) => {
            use hyverk_rag::{RagConfig, SourceType, store::RagStore};
            let config = RagConfig { db_path: db, ..RagConfig::default() };
            let store = RagStore::open(&config.db_path)?;
            let (source_type, source_ref) = match source.split_once(':') {
                Some(("crate", name)) => (SourceType::CrateDocs, name.to_string()),
                Some(("dir", path)) => (SourceType::LocalDir, path.to_string()),
                Some(("url", url)) => (SourceType::Url, url.to_string()),
                _ => {
                    error!("Source must be crate:NAME, dir:PATH, or url:URL");
                    std::process::exit(1);
                }
            };
            let chunks = hyverk_rag::sources::index_source(&store, &config, source_type, &source_ref).await?;
            info!(chunks, source = source_ref, "Indexed successfully");
            return Ok(());
        }
        Some(Commands::Search { query, k, db }) => {
            use hyverk_rag::{RagConfig, store::RagStore};
            let config = RagConfig { db_path: db, top_k: k, ..RagConfig::default() };
            let store = RagStore::open(&config.db_path)?;
            match store.search(&query, k) {
                Ok(results) if results.is_empty() => {
                    println!("No results found for: {query}");
                    println!("Hint: index sources first with: hyverk index crate:tokio");
                }
                Ok(results) => {
                    println!("Top {} results for '{}' ({} total chunks indexed):\n", results.len(), query, store.chunk_count());
                    for (i, r) in results.iter().enumerate() {
                        println!("─── [{}/{}] {} (score: {:.2})", i + 1, results.len(), r.chunk.title, r.score);
                        println!("    Source: {}", r.chunk.source_ref);
                        let preview: String = r.chunk.content.lines().take(8).collect::<Vec<_>>().join("\n");
                        println!("{preview}\n");
                    }
                }
                Err(e) => error!("Search failed: {e}"),
            }
            return Ok(());
        }
        Some(Commands::Train { model_dir, tokenizer, dataset, output, lora_rank, epochs, lr, batch_size, max_seq_len, max_examples, max_memory_gb }) => {
            use hyverk_training::TrainingConfig;
            use tokenizers::Tokenizer;

            let shard_content = if dataset == "-" {
                use std::io::Read;
                let mut s = String::new();
                std::io::stdin().read_to_string(&mut s)?;
                s
            } else {
                std::fs::read_to_string(&dataset)?
            };

            let tokenizer = Tokenizer::from_file(&tokenizer)
                .map_err(|e| format!("Failed to load tokenizer: {e}"))?;

            let train_config = TrainingConfig {
                lora_rank,
                num_epochs: epochs,
                learning_rate: lr,
                batch_size,
                max_seq_len,
                max_examples,
                max_memory_gb,
                ..TrainingConfig::default()
            };

            info!("Starting LoRA training...");
            let (adapter_bytes, result) = hyverk_training::trainer::train_on_shard(
                std::path::Path::new(&model_dir),
                &tokenizer,
                &shard_content,
                &train_config,
            ).await?;

            std::fs::write(&output, &adapter_bytes)?;
            info!(
                steps = result.steps,
                final_loss = result.final_loss,
                secs = result.duration_secs,
                output = %output,
                "Training complete. Adapter saved."
            );
            return Ok(());
        }
        Some(Commands::Run) | None => {}
    }

    // Run mode
    info!(mode = ?mode, "Starting hyverk");

    let shutdown = CancellationToken::new();
    let shutdown_sig = shutdown.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        info!("Shutdown signal received");
        shutdown_sig.cancel();
    });

    match mode {
        Mode::Coordinator => {
            hyverk_coordinator::run_coordinator(&config.coordinator, shutdown).await?;
        }
        Mode::Node => {
            hyverk_node::run_node(&config.node, shutdown).await?;
        }
        Mode::Both => {
            let coord_shutdown = shutdown.clone();
            let coord_config = config.coordinator.clone();
            let coord_handle = tokio::spawn(async move {
                if let Err(e) =
                    hyverk_coordinator::run_coordinator(&coord_config, coord_shutdown).await
                {
                    error!("Coordinator error: {e}");
                }
            });

            // Brief delay to let coordinator bind ports
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;

            let node_shutdown = shutdown.clone();
            let node_config = config.node.clone();
            let node_handle = tokio::spawn(async move {
                if let Err(e) = hyverk_node::run_node(&node_config, node_shutdown).await {
                    error!("Node error: {e}");
                }
            });

            tokio::select! {
                _ = coord_handle => error!("Coordinator exited unexpectedly"),
                _ = node_handle => error!("Node exited unexpectedly"),
            }
        }
    }

    info!("Hyverk stopped.");
    Ok(())
}

