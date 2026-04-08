// Dataset Store — SQLite-backed for unlimited capacity.
// Dedup hashes kept in memory for fast duplicate rejection.
// Examples stored in SQLite — no OOM regardless of dataset size.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use rusqlite::{Connection, params};
use std::sync::Mutex as StdMutex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetExample {
    pub id: String,
    pub instruction: String,
    pub response: String,
    pub category: String,
    pub provider: String,
    pub model: String,
    pub node_id: String,
    pub refined: bool,
    pub execution_verified: bool,
    pub quality_score: Option<f32>,
    pub submitted_at_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStats {
    pub total_examples: usize,
    pub by_category: HashMap<String, usize>,
    pub by_provider: HashMap<String, usize>,
    pub by_node: HashMap<String, usize>,
    pub execution_verified: usize,
    pub refined: usize,
    pub deduplicated: usize,
}

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS examples (
    id TEXT PRIMARY KEY,
    instruction TEXT NOT NULL,
    response TEXT NOT NULL,
    category TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT,
    node_id TEXT,
    refined INTEGER DEFAULT 0,
    execution_verified INTEGER DEFAULT 0,
    quality_score REAL,
    submitted_at INTEGER
);
CREATE INDEX IF NOT EXISTS idx_examples_category ON examples(category);
CREATE INDEX IF NOT EXISTS idx_examples_provider ON examples(provider);
";

#[derive(Clone)]
pub struct DatasetStore {
    db: Arc<StdMutex<Connection>>,
    dedup: Arc<RwLock<DedupState>>,
}

struct DedupState {
    seen_hashes: HashSet<u64>,
    deduplicated: usize,
}

impl DatasetStore {
    pub fn new() -> Self {
        Self::open(":memory:")
    }

    pub fn open(db_path: &str) -> Self {
        let conn = Connection::open(db_path).expect("open dataset DB");
        conn.execute_batch(SCHEMA).expect("create dataset schema");
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;").ok();

        // Load dedup hashes from existing data
        let mut hashes = HashSet::new();
        {
            let mut stmt = conn.prepare("SELECT instruction FROM examples").unwrap();
            let rows: Vec<String> = stmt.query_map([], |r| r.get(0)).unwrap().flatten().collect();
            for inst in &rows {
                hashes.insert(instruction_hash(inst));
            }
        }
        let count = hashes.len();
        tracing::info!(examples = count, "Dataset store loaded from SQLite");

        Self {
            db: Arc::new(StdMutex::new(conn)),
            dedup: Arc::new(RwLock::new(DedupState { seen_hashes: hashes, deduplicated: 0 })),
        }
    }

    // Keep old constructor for compatibility
    pub fn with_persist_path(path: std::path::PathBuf) -> Self {
        Self::open(&path.to_string_lossy())
    }

    pub async fn add_example(&self, example: DatasetExample) -> bool {
        if !passes_quality_filter(&example) { return false; }
        let hash = instruction_hash(&example.instruction);

        {
            let mut dedup = self.dedup.write().await;
            if dedup.seen_hashes.contains(&hash) {
                dedup.deduplicated += 1;
                return false;
            }
            dedup.seen_hashes.insert(hash);
        }

        self.insert_example_sync(&example);
        true
    }

    pub async fn add_bulk(&self, examples: Vec<DatasetExample>) -> (usize, usize) {
        let mut accepted = 0;
        let mut rejected = 0;

        // First pass: dedup check (async-safe)
        let mut to_insert = Vec::new();
        {
            let mut dedup = self.dedup.write().await;
            for ex in examples {
                if !passes_quality_filter(&ex) { rejected += 1; continue; }
                let hash = instruction_hash(&ex.instruction);
                if dedup.seen_hashes.contains(&hash) { dedup.deduplicated += 1; rejected += 1; continue; }
                dedup.seen_hashes.insert(hash);
                to_insert.push(ex);
            }
        }

        // Second pass: DB insert (sync, no await)
        accepted = to_insert.len();
        self.insert_examples_batch_sync(&to_insert);
        (accepted, rejected)
    }

    fn insert_example_sync(&self, ex: &DatasetExample) {
        let db = self.db.lock().unwrap();
        db.execute(
            "INSERT OR IGNORE INTO examples (id,instruction,response,category,provider,model,node_id,refined,execution_verified,quality_score,submitted_at) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",
            params![ex.id, ex.instruction, ex.response, ex.category, ex.provider, ex.model, ex.node_id, ex.refined as i32, ex.execution_verified as i32, ex.quality_score, ex.submitted_at_secs as i64],
        ).ok();
    }

    fn insert_examples_batch_sync(&self, examples: &[DatasetExample]) {
        let db = self.db.lock().unwrap();
        for ex in examples {
            db.execute(
                "INSERT OR IGNORE INTO examples (id,instruction,response,category,provider,model,node_id,refined,execution_verified,quality_score,submitted_at) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",
                params![ex.id, ex.instruction, ex.response, ex.category, ex.provider, ex.model, ex.node_id, ex.refined as i32, ex.execution_verified as i32, ex.quality_score, ex.submitted_at_secs as i64],
            ).ok();
        }
    }

    pub async fn stats(&self) -> DatasetStats {
        let (total, verified, refined, by_category, by_provider, by_node) = self.stats_sync();
        let dedup = self.dedup.read().await;
        DatasetStats {
            total_examples: total,
            by_category, by_provider, by_node,
            execution_verified: verified,
            refined,
            deduplicated: dedup.deduplicated,
        }
    }

    fn stats_sync(&self) -> (usize, usize, usize, HashMap<String, usize>, HashMap<String, usize>, HashMap<String, usize>) {
        let db = self.db.lock().unwrap();
        let total: usize = db.query_row("SELECT COUNT(*) FROM examples", [], |r| r.get(0)).unwrap_or(0);
        let verified: usize = db.query_row("SELECT COUNT(*) FROM examples WHERE execution_verified=1", [], |r| r.get(0)).unwrap_or(0);
        let refined: usize = db.query_row("SELECT COUNT(*) FROM examples WHERE refined=1", [], |r| r.get(0)).unwrap_or(0);

        let mut by_category = HashMap::new();
        { let mut s = db.prepare("SELECT category,COUNT(*) FROM examples GROUP BY category").unwrap(); for r in s.query_map([],|r|Ok((r.get::<_,String>(0)?,r.get::<_,usize>(1)?))).unwrap().flatten() { by_category.insert(r.0,r.1); } }
        let mut by_provider = HashMap::new();
        { let mut s = db.prepare("SELECT provider,COUNT(*) FROM examples GROUP BY provider").unwrap(); for r in s.query_map([],|r|Ok((r.get::<_,String>(0)?,r.get::<_,usize>(1)?))).unwrap().flatten() { by_provider.insert(r.0,r.1); } }
        let mut by_node = HashMap::new();
        { let mut s = db.prepare("SELECT node_id,COUNT(*) FROM examples GROUP BY node_id").unwrap(); for r in s.query_map([],|r|Ok((r.get::<_,String>(0)?,r.get::<_,usize>(1)?))).unwrap().flatten() { by_node.insert(r.0,r.1); } }
        (total, verified, refined, by_category, by_provider, by_node)
    }

    pub async fn export_jsonl(&self) -> String {
        self.export_jsonl_sync()
    }

    fn export_jsonl_sync(&self) -> String {
        let db = self.db.lock().unwrap();
        let mut stmt = db.prepare("SELECT instruction,response,category,provider FROM examples LIMIT 100000").unwrap();
        let rows: Vec<String> = stmt.query_map([], |r| {
            Ok(serde_json::json!({"instruction":r.get::<_,String>(0)?,"response":r.get::<_,String>(1)?,"category":r.get::<_,String>(2)?,"provider":r.get::<_,String>(3)?}).to_string())
        }).unwrap().flatten().collect();
        rows.join("\n")
    }

    pub async fn list(&self, offset: usize, limit: usize) -> Vec<DatasetExample> {
        self.list_sync(offset, limit)
    }

    fn list_sync(&self, offset: usize, limit: usize) -> Vec<DatasetExample> {
        let db = self.db.lock().unwrap();
        let mut stmt = db.prepare("SELECT id,instruction,response,category,provider,model,node_id,refined,execution_verified,quality_score,submitted_at FROM examples LIMIT ?1 OFFSET ?2").unwrap();
        stmt.query_map(params![limit as i64, offset as i64], |r| {
            Ok(DatasetExample {
                id: r.get(0)?, instruction: r.get(1)?, response: r.get(2)?,
                category: r.get(3)?, provider: r.get(4)?, model: r.get::<_,String>(5).unwrap_or_default(),
                node_id: r.get::<_,String>(6).unwrap_or_default(),
                refined: r.get::<_,i32>(7).unwrap_or(0) != 0,
                execution_verified: r.get::<_,i32>(8).unwrap_or(0) != 0,
                quality_score: r.get(9).ok(),
                submitted_at_secs: r.get::<_,i64>(10).unwrap_or(0) as u64,
            })
        }).unwrap().flatten().collect()
    }
}

fn passes_quality_filter(ex: &DatasetExample) -> bool {
    if ex.instruction.len() < 20 { return false; }
    if ex.response.len() < 30 { return false; }
    let has_code = ex.response.contains("```");
    let is_substantial = ex.response.len() > 200;
    if !has_code && !is_substantial { return false; }
    let rl = ex.response.to_lowercase();
    let refusals = ["i cannot", "i'm unable", "i am unable", "as an ai"];
    if refusals.iter().any(|p| rl.starts_with(p)) { return false; }
    true
}

fn instruction_hash(instruction: &str) -> u64 {
    let normalized = instruction.trim().to_lowercase();
    let text: String = normalized.chars().take(200).collect();
    let mut hash: u64 = 14695981039346656037;
    for byte in text.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    hash
}
