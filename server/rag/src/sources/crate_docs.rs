// Fetch crate documentation from docs.rs.
// Fetches the main lib.rs documentation page and parses it into sections.
// Each public item (fn, struct, trait, enum) becomes one section.

use anyhow::{Context, Result};

/// Fetch docs for a crate from docs.rs.
/// Returns Vec<(title, content)> where title is the item name.
pub async fn fetch(crate_name: &str) -> Result<Vec<(String, String)>> {
    // Parse "crate_name" or "crate_name@version"
    let (name, version) = if let Some((n, v)) = crate_name.split_once('@') {
        (n, v.to_string())
    } else {
        (crate_name, "latest".to_string())
    };

    let url = if version == "latest" {
        format!("https://docs.rs/{name}/latest/{name}/")
    } else {
        format!("https://docs.rs/{name}/{version}/{name}/")
    };

    tracing::info!(crate = name, version = version, url = %url, "Fetching crate docs");

    let html = reqwest::get(&url)
        .await
        .context("fetch docs.rs")?
        .text()
        .await
        .context("read docs.rs body")?;

    parse_docs_rs_html(name, &html)
}

/// Parse docs.rs HTML into (title, content) sections.
/// Strips HTML tags and extracts module-level docs + item docs.
fn parse_docs_rs_html(crate_name: &str, html: &str) -> Result<Vec<(String, String)>> {
    let mut sections = Vec::new();

    // Very simple HTML stripper — good enough for docs.rs
    let text = strip_html(html);

    // Split at common docs.rs section headers
    let lines: Vec<&str> = text.lines().collect();
    let mut current_title = format!("{crate_name} — overview");
    let mut current_content = String::new();
    let mut in_content = false;

    for line in &lines {
        let trimmed = line.trim();
        // docs.rs item headings are typically short lines ending with a type keyword
        if is_section_heading(trimmed) && !current_content.trim().is_empty() {
            if in_content {
                sections.push((current_title.clone(), current_content.trim().to_string()));
                current_content.clear();
            }
            current_title = trimmed.to_string();
            in_content = true;
        } else {
            if !trimmed.is_empty() {
                in_content = true;
            }
            current_content.push_str(line);
            current_content.push('\n');
        }
    }
    if !current_content.trim().is_empty() {
        sections.push((current_title, current_content.trim().to_string()));
    }

    // Fallback: if no sections found, return the whole text as one chunk
    if sections.is_empty() && !text.trim().is_empty() {
        sections.push((
            format!("{crate_name} docs"),
            text.chars().take(8000).collect(),
        ));
    }

    Ok(sections)
}

fn is_section_heading(line: &str) -> bool {
    // Looks like a Rust item declaration
    let keywords = ["fn ", "pub fn ", "struct ", "pub struct ", "enum ", "pub enum ",
                    "trait ", "pub trait ", "impl ", "type ", "pub type ", "mod ", "pub mod "];
    keywords.iter().any(|k| line.starts_with(k)) && line.len() < 120
}

fn strip_html(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut in_script = false;
    let mut tag_buf = String::new();

    let chars: Vec<char> = html.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        if in_tag {
            tag_buf.push(ch);
            if ch == '>' {
                let tag_lower = tag_buf.to_lowercase();
                if tag_lower.starts_with("<script") {
                    in_script = true;
                } else if tag_lower.starts_with("</script") {
                    in_script = false;
                }
                in_tag = false;
                tag_buf.clear();
                // Replace block tags with newline
                out.push('\n');
            }
        } else if ch == '<' {
            in_tag = true;
            tag_buf.push(ch);
        } else if !in_script {
            // Decode common HTML entities
            if ch == '&' {
                // Collect entity
                let mut entity = String::from('&');
                let mut j = i + 1;
                while j < chars.len() && chars[j] != ';' && chars[j] != ' ' && j - i < 10 {
                    entity.push(chars[j]);
                    j += 1;
                }
                if j < chars.len() && chars[j] == ';' {
                    entity.push(';');
                    match entity.as_str() {
                        "&amp;" => out.push('&'),
                        "&lt;" => out.push('<'),
                        "&gt;" => out.push('>'),
                        "&quot;" => out.push('"'),
                        "&#39;" | "&apos;" => out.push('\''),
                        "&nbsp;" => out.push(' '),
                        _ => out.push_str(&entity),
                    }
                    i = j;
                } else {
                    out.push(ch);
                }
            } else {
                out.push(ch);
            }
        }
        i += 1;
    }
    out
}
