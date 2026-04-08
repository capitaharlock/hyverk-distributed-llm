// Fetch and index a URL (markdown docs, GitHub READMEs, etc.).
// Strips HTML, returns (title, content).

use anyhow::{Context, Result};

pub async fn fetch(url: &str) -> Result<Vec<(String, String)>> {
    tracing::info!(url = url, "Fetching URL");
    let response = reqwest::Client::new()
        .get(url)
        .header("User-Agent", "hyverk-rag/0.1 (documentation indexer)")
        .send()
        .await
        .context("fetch URL")?;

    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    let body = response.text().await.context("read body")?;

    let (title, text) = if content_type.contains("html") {
        let text = strip_html_basic(&body);
        let title = extract_title(&body).unwrap_or_else(|| url.to_string());
        (title, text)
    } else {
        // Plain text / markdown
        (url.to_string(), body)
    };

    if text.trim().is_empty() {
        return Ok(vec![]);
    }

    Ok(vec![(title, text)])
}

fn extract_title(html: &str) -> Option<String> {
    let lower = html.to_lowercase();
    let start = lower.find("<title>")?  + "<title>".len();
    let end = lower.find("</title>")?;
    if start < end {
        Some(html[start..end].trim().to_string())
    } else {
        None
    }
}

fn strip_html_basic(html: &str) -> String {
    let mut out = String::with_capacity(html.len() / 2);
    let mut in_tag = false;
    let mut in_script = false;
    let mut tag_start = false;
    let mut tag_content = String::new();

    for ch in html.chars() {
        match ch {
            '<' => {
                in_tag = true;
                tag_start = true;
                tag_content.clear();
            }
            '>' if in_tag => {
                in_tag = false;
                let tl = tag_content.to_lowercase();
                if tl.starts_with("script") || tl.starts_with("/script") {
                    in_script = tl.starts_with("script");
                }
                out.push(' ');
            }
            _ if in_tag => {
                if tag_start {
                    tag_content.push(ch);
                }
            }
            _ if !in_script => {
                out.push(ch);
                tag_start = false;
            }
            _ => {}
        }
    }

    // Collapse whitespace
    let mut result = String::new();
    let mut last_was_ws = false;
    for ch in out.chars() {
        if ch.is_whitespace() {
            if !last_was_ws {
                result.push('\n');
                last_was_ws = true;
            }
        } else {
            result.push(ch);
            last_was_ws = false;
        }
    }
    result
}
