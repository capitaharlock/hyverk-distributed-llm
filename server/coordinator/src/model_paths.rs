//! On-disk model location for the coordinator.
//!
//! Override with `HYVERK_MODEL_DIR` (default `/data/model` for backwards compat).
//! Local runs should point this at e.g. `~/.hyverk/model`.

use std::path::{Path, PathBuf};

pub fn model_dir() -> PathBuf {
    std::env::var("HYVERK_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/data/model"))
}

pub fn model_file(name: &str) -> PathBuf {
    model_dir().join(name)
}

/// Tokenizer search order: env override, model dir, legacy Fly paths.
pub fn tokenizer_candidates() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Ok(p) = std::env::var("HYVERK_TOKENIZER") {
        paths.push(PathBuf::from(p));
    }
    paths.push(model_file("tokenizer.json"));
    paths.push(PathBuf::from("/data/tokenizer.json"));
    paths.push(PathBuf::from("/app/tokenizer.json"));
    paths
}

pub fn first_existing_tokenizer() -> Option<PathBuf> {
    tokenizer_candidates()
        .into_iter()
        .find(|p| Path::new(p).is_file())
}
