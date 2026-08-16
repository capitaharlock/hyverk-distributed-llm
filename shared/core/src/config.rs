// @llm-context: .meshkore/docs/architecture/overview.md
// @llm-depends: hyverk-coordinator/src/main.rs, hyverk-node/src/main.rs

use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize)]
pub struct HyverkConfig {
    pub mode: Mode,
    pub node: NodeConfig,
    pub coordinator: CoordinatorConfig,
    #[serde(default)]
    pub synthesis: SynthesisConfig,
}

/// Synthesis configuration — opt-in, node contributes to dataset generation
#[derive(Debug, Clone, Deserialize, Default)]
pub struct SynthesisConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_target_per_hour")]
    pub target_per_hour: u32,
    #[serde(default)]
    pub enable_refinement: bool,
    #[serde(default)]
    pub coordinator_url: String,
    #[serde(default)]
    pub providers: Vec<SynthesisProviderConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SynthesisProviderConfig {
    pub name: String,
    pub api_key: String,
    #[serde(default)]
    pub model: String,
    pub rpm_limit: Option<u32>,
    pub rpd_limit: Option<u32>,
}

fn default_target_per_hour() -> u32 { 50 }

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Node,
    Coordinator,
    Both,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NodeConfig {
    pub name: String,
    pub coordinator_url: String,
    pub models_dir: PathBuf,
    pub max_concurrent_tasks: u32,
    pub poll_interval_ms: u64,
    pub hardware_info: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CoordinatorConfig {
    pub grpc_port: u16,
    pub http_port: u16,
    pub bind_addr: String,
    pub heartbeat_timeout_secs: u64,
}

impl Default for HyverkConfig {
    fn default() -> Self {
        Self {
            mode: Mode::Node,
            node: NodeConfig {
                name: "hyverk-node".to_string(),
                coordinator_url: "http://127.0.0.1:17001".to_string(),
                models_dir: default_models_dir(),
                max_concurrent_tasks: 1,
                poll_interval_ms: 1000,
                hardware_info: String::new(),
            },
            coordinator: CoordinatorConfig {
                grpc_port: 17001,
                http_port: 17000,
                bind_addr: "0.0.0.0".to_string(),
                heartbeat_timeout_secs: 30,
            },
            synthesis: SynthesisConfig::default(),
        }
    }
}

fn default_models_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".hyverk")
        .join("models")
}

/// Return a stable node identity for this machine.
///
/// Reads `~/.hyverk/node_id` if it exists; otherwise derives one from the
/// OS hostname, sanitises it, writes it to that file, and returns it.
/// The config.toml `node.name` field is ignored for identity purposes —
/// this prevents duplicate registrations when operators run multiple sessions
/// or change their config name.
pub fn stable_node_id() -> String {
    let id_path = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".hyverk")
        .join("node_id");

    if let Ok(contents) = std::fs::read_to_string(&id_path) {
        let id = contents.trim().to_string();
        if !id.is_empty() {
            return id;
        }
    }

    // Derive from hostname: lowercase, non-alphanumeric → '-', collapse runs, strip edges.
    let raw = hostname::get()
        .ok()
        .and_then(|h| h.into_string().ok())
        .unwrap_or_else(|| "node".to_string());
    let mut id: String = raw
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '-' })
        .collect();
    // Collapse consecutive dashes and trim them
    while id.contains("--") {
        id = id.replace("--", "-");
    }
    let id = id.trim_matches('-').to_string();
    let id = if id.is_empty() { "node".to_string() } else { id };

    if let Some(parent) = id_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(&id_path, &id);
    id
}

/// Expand ~ to home directory in paths.
fn expand_tilde(path: &PathBuf) -> PathBuf {
    let s = path.to_string_lossy();
    if s.starts_with("~/") || s == "~" {
        if let Some(home) = dirs::home_dir() {
            return home.join(s.strip_prefix("~/").unwrap_or(""));
        }
    }
    path.clone()
}

pub fn load_config(path: &str) -> Result<HyverkConfig, crate::error::HyverkError> {
    let contents = std::fs::read_to_string(path)
        .map_err(|e| crate::error::HyverkError::Config(format!("Failed to read {path}: {e}")))?;
    let mut config: HyverkConfig = toml::from_str(&contents)
        .map_err(|e| crate::error::HyverkError::Config(format!("Failed to parse config: {e}")))?;

    // Expand tilde in paths
    config.node.models_dir = expand_tilde(&config.node.models_dir);

    // Always override node name with the stable machine ID (~/.hyverk/node_id).
    // Prevents duplicate registrations when config.toml name changes across sessions.
    config.node.name = stable_node_id();

    Ok(config)
}
