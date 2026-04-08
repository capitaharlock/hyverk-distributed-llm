// SQLite-backed persistent store for RAG chunks + metadata.
// Keeps chunks on disk so the index survives restarts.
// The BM25 index is rebuilt from SQLite on startup (fast: ~1ms per 1k docs).

use crate::{Chunk, IndexedSource, SearchResult, SourceType, index::BM25Index};
use anyhow::{Context, Result};
use rusqlite::{Connection, params};
use std::sync::{Arc, Mutex};


pub struct RagStore {
    conn: Mutex<Connection>,
    index: Arc<Mutex<BM25Index>>,
}

impl RagStore {
    /// Open or create the SQLite store at `db_path`.
    pub fn open(db_path: &str) -> Result<Self> {
        let expanded = shellexpand::tilde(db_path).to_string();
        // Ensure parent directory exists
        if let Some(parent) = std::path::Path::new(&expanded).parent() {
            std::fs::create_dir_all(parent).context("create db directory")?;
        }
        let conn = Connection::open(&expanded).context("open sqlite")?;
        conn.execute_batch(SCHEMA).context("create schema")?;

        let store = Self {
            conn: Mutex::new(conn),
            index: Arc::new(Mutex::new(BM25Index::new())),
        };
        store.rebuild_index()?;
        Ok(store)
    }

    /// Index a new chunk (or update if id already exists).
    pub fn upsert_chunk(&self, chunk: &Chunk) -> Result<()> {
        {
            let conn = self.conn.lock().unwrap();
            conn.execute(
                "INSERT OR REPLACE INTO chunks (id, source_id, source_type, source_ref, title, content, indexed_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    chunk.id,
                    chunk.source_id,
                    serde_json::to_string(&chunk.source_type)?,
                    chunk.source_ref,
                    chunk.title,
                    chunk.content,
                    chunk.indexed_at,
                ],
            )?;
        }
        // Update in-memory BM25 index
        let text = format!("{} {}", chunk.title, chunk.content);
        self.index.lock().unwrap().add_document(&chunk.id, &text);
        Ok(())
    }

    /// Remove all chunks for a source.
    pub fn remove_source(&self, source_id: &str) -> Result<()> {
        let chunk_ids: Vec<String> = {
            let conn = self.conn.lock().unwrap();
            let mut stmt = conn.prepare("SELECT id FROM chunks WHERE source_id = ?1")?;
            let ids: Vec<String> = stmt.query_map(params![source_id], |r| r.get(0))?.flatten().collect();
            ids
        };
        {
            let mut idx = self.index.lock().unwrap();
            for id in &chunk_ids {
                idx.remove_document(id);
            }
        }
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM chunks WHERE source_id = ?1", params![source_id])?;
        conn.execute("DELETE FROM sources WHERE id = ?1", params![source_id])?;
        Ok(())
    }

    /// Record source metadata.
    pub fn upsert_source(&self, src: &IndexedSource) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO sources (id, source_type, source_ref, chunk_count, indexed_at, last_updated)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                src.id,
                serde_json::to_string(&src.source_type)?,
                src.source_ref,
                src.chunk_count as i64,
                src.indexed_at as i64,
                src.last_updated as i64,
            ],
        )?;
        Ok(())
    }

    /// BM25 search returning top-K results.
    pub fn search(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        let hits = {
            let idx = self.index.lock().unwrap();
            idx.search(query, top_k)
        };
        if hits.is_empty() {
            return Ok(vec![]);
        }
        let conn = self.conn.lock().unwrap();
        let mut results = Vec::new();
        for (chunk_id, score) in hits {
            let chunk = self.load_chunk_by_id(&conn, &chunk_id)?;
            if let Some(chunk) = chunk {
                results.push(SearchResult { chunk, score });
            }
        }
        Ok(results)
    }

    /// Format top-K results as a system prompt injection string.
    pub fn build_context(&self, query: &str, top_k: usize) -> Result<String> {
        let results = self.search(query, top_k)?;
        if results.is_empty() {
            return Ok(String::new());
        }
        let mut ctx = String::from("# Relevant Documentation\n\n");
        for (i, r) in results.iter().enumerate() {
            ctx.push_str(&format!(
                "## [{}/{}] {} (from: {})\n\n{}\n\n---\n\n",
                i + 1,
                results.len(),
                r.chunk.title,
                r.chunk.source_ref,
                r.chunk.content
            ));
        }
        Ok(ctx)
    }

    pub fn list_sources(&self) -> Result<Vec<IndexedSource>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, source_type, source_ref, chunk_count, indexed_at, last_updated FROM sources ORDER BY last_updated DESC"
        )?;
        let sources = stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, i64>(3)?,
                r.get::<_, i64>(4)?,
                r.get::<_, i64>(5)?,
            ))
        })?.flatten().map(|(id, st, sr, cc, ia, lu)| {
            let source_type = serde_json::from_str(&st).unwrap_or(SourceType::Url);
            IndexedSource {
                id,
                source_type,
                source_ref: sr,
                chunk_count: cc as usize,
                indexed_at: ia as u64,
                last_updated: lu as u64,
            }
        }).collect();
        Ok(sources)
    }

    pub fn chunk_count(&self) -> usize {
        self.index.lock().unwrap().doc_count()
    }

    // ── private helpers ────────────────────────────────────────────────────────

    fn rebuild_index(&self) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT id, title, content FROM chunks")?;
        let rows: Vec<(String, String, String)> = stmt
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
            .flatten()
            .collect();
        let mut idx = self.index.lock().unwrap();
        for (id, title, content) in rows {
            let text = format!("{title} {content}");
            idx.add_document(&id, &text);
        }
        tracing::info!(chunks = idx.doc_count(), "BM25 index rebuilt from SQLite");
        Ok(())
    }

    fn load_chunk_by_id(&self, conn: &Connection, id: &str) -> Result<Option<Chunk>> {
        let mut stmt = conn.prepare(
            "SELECT id, source_id, source_type, source_ref, title, content, indexed_at FROM chunks WHERE id = ?1"
        )?;
        let chunk = stmt
            .query_map(params![id], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, String>(4)?,
                    r.get::<_, String>(5)?,
                    r.get::<_, i64>(6)?,
                ))
            })?
            .flatten()
            .next()
            .map(|(id, sid, st, sr, title, content, ia)| {
                let source_type = serde_json::from_str(&st).unwrap_or(SourceType::Url);
                Chunk {
                    id,
                    source_id: sid,
                    source_type,
                    source_ref: sr,
                    title,
                    content,
                    indexed_at: ia as u64,
                }
            });
        Ok(chunk)
    }
}

const SCHEMA: &str = "
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    source_id   TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_ref  TEXT NOT NULL,
    title       TEXT NOT NULL,
    content     TEXT NOT NULL,
    indexed_at  INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_id);

CREATE TABLE IF NOT EXISTS sources (
    id           TEXT PRIMARY KEY,
    source_type  TEXT NOT NULL,
    source_ref   TEXT NOT NULL,
    chunk_count  INTEGER NOT NULL,
    indexed_at   INTEGER NOT NULL,
    last_updated INTEGER NOT NULL
);
";
