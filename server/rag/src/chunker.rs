// Text/code chunker: splits documents into overlapping windows.
// Code-aware: tries to break at function/struct/impl boundaries first.
// Falls back to paragraph breaks, then hard char limits.

use regex::Regex;
use std::sync::OnceLock;

static CODE_BOUNDARY: OnceLock<Regex> = OnceLock::new();
static PARA_BREAK: OnceLock<Regex> = OnceLock::new();

fn code_boundary() -> &'static Regex {
    CODE_BOUNDARY.get_or_init(|| {
        Regex::new(r"(?m)^(?:pub\s+)?(?:fn|struct|enum|impl|trait|mod|type|const|static)\s").unwrap()
    })
}

fn para_break() -> &'static Regex {
    PARA_BREAK.get_or_init(|| Regex::new(r"\n\s*\n").unwrap())
}

/// Split text into overlapping chunks.
/// Prefers splitting at code boundaries, then paragraph breaks, then hard limit.
pub fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    if text.len() <= chunk_size {
        return vec![text.to_string()];
    }

    // Try code-aware splitting first
    let chunks = split_at_pattern(text, code_boundary(), chunk_size, overlap);
    if !chunks.is_empty() {
        return chunks;
    }

    // Fall back to paragraph splitting
    let chunks = split_at_pattern(text, para_break(), chunk_size, overlap);
    if !chunks.is_empty() {
        return chunks;
    }

    // Hard split
    hard_split(text, chunk_size, overlap)
}

/// Split at natural boundaries (regex matches) while respecting chunk_size.
fn split_at_pattern(text: &str, re: &Regex, chunk_size: usize, overlap: usize) -> Vec<String> {
    let boundaries: Vec<usize> = re.find_iter(text).map(|m| m.start()).collect();
    if boundaries.is_empty() {
        return vec![];
    }

    let mut chunks = Vec::new();
    let mut start = 0usize;

    for &boundary in &boundaries {
        if boundary <= start {
            continue;
        }
        // If adding up to this boundary would exceed chunk_size, emit current chunk
        if boundary - start >= chunk_size {
            let end = (start + chunk_size).min(text.len());
            chunks.push(text[start..end].to_string());
            // Next chunk starts with overlap
            start = end.saturating_sub(overlap);
        }
    }
    // Emit remaining
    if start < text.len() {
        chunks.push(text[start..].to_string());
    }

    if chunks.is_empty() {
        return vec![];
    }
    chunks
}

fn hard_split(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let bytes = text.as_bytes();
    let mut start = 0;

    while start < bytes.len() {
        let end = (start + chunk_size).min(bytes.len());
        // Snap to char boundary
        let end = snap_to_char_boundary(text, end);
        chunks.push(text[start..end].to_string());
        if end >= bytes.len() {
            break;
        }
        start = end.saturating_sub(overlap);
        start = snap_to_char_boundary(text, start);
    }
    chunks
}

fn snap_to_char_boundary(s: &str, mut idx: usize) -> usize {
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_short_text() {
        let chunks = chunk_text("hello world", 1500, 200);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "hello world");
    }

    #[test]
    fn test_hard_split() {
        let text = "a".repeat(4000);
        let chunks = chunk_text(&text, 1500, 200);
        assert!(chunks.len() >= 2);
        assert!(chunks[0].len() <= 1500);
    }

    #[test]
    fn test_code_split() {
        let text = "use std::io;\n\npub fn foo() {\n    let x = 1;\n}\n\npub fn bar() {\n    let y = 2;\n}\n\npub struct Baz {\n    val: u32,\n}";
        let chunks = chunk_text(text, 50, 10);
        assert!(chunks.len() >= 2);
    }
}
