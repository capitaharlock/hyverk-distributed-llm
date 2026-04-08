// Extracts Rust code blocks from LLM markdown responses.
// Also computes quality signals from the extracted code.

use crate::QualitySignals;

/// Extract all ```rust ... ``` code blocks from a markdown string
pub fn extract_rust_blocks(text: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut in_block = false;
    let mut current = String::new();
    let mut lang_marker = false;

    for line in text.lines() {
        let trimmed = line.trim();

        if !in_block {
            if trimmed.starts_with("```rust") || trimmed == "```rust" {
                in_block = true;
                lang_marker = true;
                current.clear();
                continue;
            }
            // Also catch unmarked code blocks that look like Rust
            if trimmed == "```" && !lang_marker {
                // Skip — only capture explicitly-tagged rust blocks
            }
        } else {
            if trimmed == "```" {
                let block = current.trim().to_string();
                if !block.is_empty() {
                    blocks.push(block);
                }
                in_block = false;
                lang_marker = false;
                current.clear();
            } else {
                current.push_str(line);
                current.push('\n');
            }
        }
    }

    // If there's an unclosed block, try it anyway
    if in_block && !current.trim().is_empty() {
        blocks.push(current.trim().to_string());
    }

    blocks
}

/// Analyze code quality signals without executing it
pub fn analyze_code(code: &str) -> QualitySignals {
    let line_count = code.lines().count();
    let has_doc_comments = code.contains("///") || code.contains("//!");
    let has_tests = code.contains("#[test]") || code.contains("#[cfg(test)]");
    let has_error_handling = code.contains("Result<")
        || code.contains("Option<")
        || code.contains(".unwrap_or")
        || code.contains("map_err")
        || code.contains("?;");

    // Count fn declarations (rough function count)
    let function_count = code
        .lines()
        .filter(|l| {
            let t = l.trim();
            (t.starts_with("fn ") || t.starts_with("pub fn ") || t.starts_with("async fn ")
                || t.starts_with("pub async fn "))
                && !t.starts_with("//")
        })
        .count();

    QualitySignals {
        has_doc_comments,
        has_tests,
        has_error_handling,
        function_count,
        line_count,
        warning_count: 0, // filled in by runner after compilation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_basic() {
        let text = "Here is code:\n```rust\nfn main() {}\n```\nDone.";
        let blocks = extract_rust_blocks(text);
        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].contains("fn main()"));
    }

    #[test]
    fn test_extract_multiple() {
        let text = "```rust\nfn foo() {}\n```\nand\n```rust\nfn bar() {}\n```";
        let blocks = extract_rust_blocks(text);
        assert_eq!(blocks.len(), 2);
    }

    #[test]
    fn test_no_blocks() {
        let blocks = extract_rust_blocks("No code here, just text.");
        assert!(blocks.is_empty());
    }

    #[test]
    fn test_analyze_code() {
        let code = "/// Does something\npub fn foo() -> Result<i32, String> {\n    Ok(42)\n}\n\n#[cfg(test)]\nmod tests {\n    #[test]\n    fn test_foo() {}\n}";
        let signals = analyze_code(code);
        assert!(signals.has_doc_comments);
        assert!(signals.has_tests);
        assert!(signals.has_error_handling);
        assert_eq!(signals.function_count, 2);
    }
}
