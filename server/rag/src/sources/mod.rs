// Source adapters: fetch content from various origins, chunk it, store in RAG.
// Each source returns Vec<(title, content)> for chunking.

pub mod crate_docs;
pub mod local_dir;
pub mod url;

use crate::{Chunk, IndexedSource, RagConfig, SourceType, store::RagStore};
use anyhow::Result;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Index a source into the store. Returns number of chunks indexed.
pub async fn index_source(
    store: &RagStore,
    config: &RagConfig,
    source_type: SourceType,
    source_ref: &str,
) -> Result<usize> {
    let source_id = Uuid::new_v4().to_string();
    let now = now_secs();

    // Fetch raw (title, content) pairs from the source
    let sections: Vec<(String, String)> = match &source_type {
        SourceType::CrateDocs => crate_docs::fetch(source_ref).await?,
        SourceType::LocalDir => local_dir::fetch(source_ref)?,
        SourceType::Url => url::fetch(source_ref).await?,
        SourceType::GitRepo => url::fetch(source_ref).await?, // fallback
    };

    let mut chunk_count = 0;
    for (title, content) in &sections {
        let chunks = crate::chunker::chunk_text(content, config.chunk_size, config.chunk_overlap);
        for (i, chunk_text) in chunks.iter().enumerate() {
            let chunk = Chunk {
                id: Uuid::new_v4().to_string(),
                source_id: source_id.clone(),
                source_type: source_type.clone(),
                source_ref: source_ref.to_string(),
                title: if chunks.len() > 1 {
                    format!("{} (part {})", title, i + 1)
                } else {
                    title.clone()
                },
                content: chunk_text.clone(),
                indexed_at: now,
            };
            store.upsert_chunk(&chunk)?;
            chunk_count += 1;
        }
    }

    store.upsert_source(&IndexedSource {
        id: source_id,
        source_type,
        source_ref: source_ref.to_string(),
        chunk_count,
        indexed_at: now,
        last_updated: now,
    })?;

    tracing::info!(source = source_ref, chunks = chunk_count, "Indexed source");
    Ok(chunk_count)
}
