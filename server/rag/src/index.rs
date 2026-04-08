// BM25 index for fast keyword search over code/doc chunks.
//
// BM25 is the industry standard for lexical retrieval (used by Elasticsearch,
// Lucene, Solr). For code search it outperforms dense embeddings on:
//   - Exact symbol names (function names, struct names, crate names)
//   - API method signatures
//   - Error codes and constants
//
// Parameters: k1=1.5 (term saturation), b=0.75 (length normalization)
// These are the standard defaults used in practice.

use std::collections::HashMap;
use unicode_normalization::UnicodeNormalization;

const K1: f32 = 1.5;
const B: f32 = 0.75;

pub struct BM25Index {
    /// doc_id → term → frequency
    doc_term_freq: HashMap<String, HashMap<String, u32>>,
    /// term → set of doc_ids containing the term
    inverted: HashMap<String, Vec<String>>,
    /// doc_id → length (token count)
    doc_lengths: HashMap<String, usize>,
    avg_doc_len: f32,
    num_docs: usize,
}

impl BM25Index {
    pub fn new() -> Self {
        Self {
            doc_term_freq: HashMap::new(),
            inverted: HashMap::new(),
            doc_lengths: HashMap::new(),
            avg_doc_len: 0.0,
            num_docs: 0,
        }
    }

    /// Add or update a document in the index.
    pub fn add_document(&mut self, doc_id: &str, text: &str) {
        let terms = tokenize(text);
        let len = terms.len();

        // Remove old entry if exists (for re-indexing)
        self.remove_document(doc_id);

        // Count term frequencies
        let mut freq: HashMap<String, u32> = HashMap::new();
        for term in &terms {
            *freq.entry(term.clone()).or_default() += 1;
        }

        // Update inverted index
        for term in freq.keys() {
            self.inverted
                .entry(term.clone())
                .or_default()
                .push(doc_id.to_string());
        }

        self.doc_term_freq.insert(doc_id.to_string(), freq);
        self.doc_lengths.insert(doc_id.to_string(), len);
        self.num_docs += 1;

        // Update average doc length
        let total_len: usize = self.doc_lengths.values().sum();
        self.avg_doc_len = total_len as f32 / self.num_docs as f32;
    }

    pub fn remove_document(&mut self, doc_id: &str) {
        if let Some(freq) = self.doc_term_freq.remove(doc_id) {
            for term in freq.keys() {
                if let Some(docs) = self.inverted.get_mut(term) {
                    docs.retain(|d| d != doc_id);
                }
            }
        }
        if self.doc_lengths.remove(doc_id).is_some() {
            self.num_docs = self.num_docs.saturating_sub(1);
            if self.num_docs > 0 {
                let total_len: usize = self.doc_lengths.values().sum();
                self.avg_doc_len = total_len as f32 / self.num_docs as f32;
            }
        }
    }

    /// Search and return (doc_id, score) sorted by descending score.
    pub fn search(&self, query: &str, top_k: usize) -> Vec<(String, f32)> {
        if self.num_docs == 0 {
            return vec![];
        }

        let query_terms = tokenize(query);
        let mut scores: HashMap<String, f32> = HashMap::new();

        for term in &query_terms {
            let Some(doc_ids) = self.inverted.get(term) else {
                continue;
            };
            let df = doc_ids.len() as f32;
            // IDF with smoothing
            let idf = ((self.num_docs as f32 - df + 0.5) / (df + 0.5) + 1.0).ln();

            for doc_id in doc_ids {
                let tf = self
                    .doc_term_freq
                    .get(doc_id)
                    .and_then(|f| f.get(term))
                    .copied()
                    .unwrap_or(0) as f32;
                let dl = self.doc_lengths.get(doc_id).copied().unwrap_or(1) as f32;
                let norm_tf = tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * dl / self.avg_doc_len));
                *scores.entry(doc_id.clone()).or_default() += idf * norm_tf;
            }
        }

        let mut ranked: Vec<(String, f32)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(top_k);
        ranked
    }

    pub fn doc_count(&self) -> usize {
        self.num_docs
    }
}

impl Default for BM25Index {
    fn default() -> Self {
        Self::new()
    }
}

/// Tokenize text into normalized lowercase terms for BM25.
/// Splits on whitespace and punctuation, filters short tokens.
fn tokenize(text: &str) -> Vec<String> {
    // Normalize unicode (NFC)
    let normalized: String = text.nfc().collect();

    // Split on non-alphanumeric characters, preserving underscore (important for code)
    let mut tokens = Vec::new();
    let mut current = String::new();

    for ch in normalized.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            current.push(ch.to_lowercase().next().unwrap_or(ch));
        } else {
            if current.len() >= 2 {
                tokens.push(current.clone());
            }
            current.clear();
        }
    }
    if current.len() >= 2 {
        tokens.push(current);
    }

    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_search() {
        let mut idx = BM25Index::new();
        idx.add_document("doc1", "fn handle_request(req: &HttpRequest) -> Response");
        idx.add_document("doc2", "struct DatabaseConnection { pool: ConnectionPool }");
        idx.add_document("doc3", "fn authenticate_user(token: &str) -> Result<User>");

        let results = idx.search("request handler http", 3);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "doc1");
    }

    #[test]
    fn test_code_symbols() {
        let mut idx = BM25Index::new();
        idx.add_document("doc1", "use tokio::sync::RwLock;\npub async fn run() {}");
        idx.add_document("doc2", "use std::sync::Mutex;\nfn sync_fn() {}");

        let results = idx.search("tokio async", 2);
        assert_eq!(results[0].0, "doc1");
    }
}
