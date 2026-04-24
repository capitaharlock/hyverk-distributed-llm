use std::process::Command;

fn main() {
    // Embed git short hash so the dashboard can verify every node runs the same commit.
    // Falls back to "unknown" in environments without git (CI containers, etc.).
    let hash = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=GIT_HASH={hash}");
    // Re-run whenever HEAD changes (new commit / checkout)
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/refs");
}
