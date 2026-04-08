// Index a local directory: reads all .rs, .ts, .md, .toml files.
// Returns (relative_path, file_content) pairs for chunking.

use anyhow::{Context, Result};
use std::path::Path;

const MAX_FILE_SIZE: u64 = 512 * 1024; // 512 KB per file

/// Walk a local directory and return (title, content) for all indexable files.
pub fn fetch(dir: &str) -> Result<Vec<(String, String)>> {
    let expanded = shellexpand::tilde(dir).to_string();
    let base = Path::new(&expanded);
    if !base.exists() {
        anyhow::bail!("Directory not found: {dir}");
    }
    let mut sections = Vec::new();
    walk_dir(base, base, &mut sections)?;
    tracing::info!(dir = dir, files = sections.len(), "Indexed local dir");
    Ok(sections)
}

fn walk_dir(root: &Path, dir: &Path, out: &mut Vec<(String, String)>) -> Result<()> {
    let entries = std::fs::read_dir(dir).context("read dir")?;
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

        // Skip hidden dirs and common non-source dirs
        if name.starts_with('.') || matches!(name, "target" | "node_modules" | ".git") {
            continue;
        }

        if path.is_dir() {
            walk_dir(root, &path, out)?;
        } else if is_indexable(&path) {
            let meta = std::fs::metadata(&path)?;
            if meta.len() > MAX_FILE_SIZE {
                tracing::debug!(file = ?path, "Skipping large file");
                continue;
            }
            let content = std::fs::read_to_string(&path).unwrap_or_default();
            if content.trim().is_empty() {
                continue;
            }
            let title = path
                .strip_prefix(root)
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| path.to_string_lossy().to_string());
            out.push((title, content));
        }
    }
    Ok(())
}

fn is_indexable(path: &Path) -> bool {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    matches!(ext, "rs" | "ts" | "tsx" | "js" | "md" | "toml" | "yaml" | "yml" | "txt")
}
