// hyverk-rag: Real-time knowledge layer for the Hyverk coding assistant.
//
// Architecture:
//   Source (URL / dir / crate) → Chunker → BM25 index + SQLite store
//   Query → BM25 retrieval → top-K chunks → injected into system prompt
//
// Why BM25 instead of dense embeddings?
//   - Zero ML infrastructure: no embedding model to download or run
//   - Excellent for code (exact symbol names, method signatures, crate names)
//   - Instant indexing, sub-millisecond search
//   - Can be upgraded to hybrid BM25+dense later without API changes

pub mod chunker;
pub mod index;
pub mod store;
pub mod sources;

use serde::{Deserialize, Serialize};

/// A single chunk of indexed content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    pub source_id: String,
    pub source_type: SourceType,
    /// URL / file path / crate name
    pub source_ref: String,
    pub title: String,
    pub content: String,
    pub indexed_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SourceType {
    CrateDocs,
    LocalDir,
    Url,
    GitRepo,
}

/// A search result with BM25 relevance score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub chunk: Chunk,
    /// BM25 score (higher = more relevant)
    pub score: f32,
}

/// Indexed source metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexedSource {
    pub id: String,
    pub source_type: SourceType,
    pub source_ref: String,
    pub chunk_count: usize,
    pub indexed_at: u64,
    pub last_updated: u64,
}

pub struct RagConfig {
    /// Path to SQLite database file
    pub db_path: String,
    /// Chunk size in characters
    pub chunk_size: usize,
    /// Overlap between consecutive chunks in characters
    pub chunk_overlap: usize,
    /// Maximum chunks to return per query
    pub top_k: usize,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            db_path: "~/.hyverk/rag.db".to_string(),
            chunk_size: 1500,
            chunk_overlap: 200,
            top_k: 5,
        }
    }
}
