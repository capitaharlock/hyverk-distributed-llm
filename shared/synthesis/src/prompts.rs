// Prompt bank for generating training data.
// 500+ prompts across 8 categories, focused on Rust + distributed systems + DevOps.
// Each prompt becomes one (instruction, response) training pair.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prompt {
    pub category: String,
    pub instruction: String,
}

pub const HYVERK_SYSTEM: &str = "You are Hyverk, an expert software engineer specializing in Rust, TypeScript, distributed systems, and DevOps. You write clean, idiomatic, well-tested code. Always include proper error handling, documentation comments, and examples.";

pub const CRITIC_SYSTEM: &str = "You are a senior Rust engineer doing a code review. Identify bugs, anti-patterns, missing error handling, performance issues, and style violations. Be specific and actionable.";

pub const REFINER_SYSTEM: &str = "You are a Rust expert. Given the original code and a critique, produce an improved version that addresses all the issues. The improved code should be production-ready.";

/// Returns all prompts in the bank
pub fn all_prompts() -> Vec<Prompt> {
    let mut prompts = Vec::new();
    prompts.extend(rust_fundamentals());
    prompts.extend(async_concurrent());
    prompts.extend(error_handling());
    prompts.extend(data_structures());
    prompts.extend(distributed_systems());
    prompts.extend(cli_tools());
    prompts.extend(testing());
    prompts.extend(refactoring());
    prompts
}

/// Get a random prompt (uniform distribution across categories)
pub fn random_prompt(rng_seed: u64) -> Prompt {
    let all = all_prompts();
    let idx = (rng_seed as usize) % all.len();
    all[idx].clone()
}

fn rust_fundamentals() -> Vec<Prompt> {
    vec![
        p("rust", "Write a Rust generic function `deduplicate<T: Eq + Hash>(v: Vec<T>) -> Vec<T>` that removes duplicates while preserving insertion order."),
        p("rust", "Implement a Rust struct `RingBuffer<T>` with fixed capacity using a circular array. Include push, pop, len, is_full, and is_empty methods."),
        p("rust", "Write a Rust function that parses a semantic version string like '1.2.3-beta.1+build.42' into a struct with proper error handling."),
        p("rust", "Implement an iterator adapter `Chunks<I>` that groups elements of an iterator into fixed-size chunks, yielding the last partial chunk too."),
        p("rust", "Write a Rust macro `retry!` that retries a fallible expression up to N times with exponential backoff between attempts."),
        p("rust", "Implement a `LruCache<K, V>` in Rust with O(1) get and put using a HashMap and a doubly-linked list."),
        p("rust", "Write a Rust function `merge_sorted<T: Ord>(a: &[T], b: &[T]) -> Vec<T>` that merges two sorted slices into a new sorted vec."),
        p("rust", "Implement a Rust type `NonEmptyVec<T>` that guarantees at least one element at compile time. Include push, first, last, and iter methods."),
        p("rust", "Write a Rust function that computes the SHA-256 hash of a file without loading the entire file into memory."),
        p("rust", "Implement `Trie<V>` in Rust with insert, get, contains_prefix, and words_with_prefix methods."),
        p("rust", "Write a Rust proc-macro `#[builder]` that generates a builder pattern for any struct."),
        p("rust", "Implement a thread-safe lazy singleton in Rust using `OnceLock` without external dependencies."),
        p("rust", "Write a Rust function that converts a JSON Value to a typed struct using a visitor pattern (without serde_derive)."),
        p("rust", "Implement Fibonacci sequence using memoization in Rust, handling arbitrary precision with `u128` and returning an error for values > 186."),
        p("rust", "Write a Rust function to flatten a deeply nested `serde_json::Value` object into a flat `HashMap<String, String>` with dot-notation keys."),
    ]
}

fn async_concurrent() -> Vec<Prompt> {
    vec![
        p("async", "Write an async Rust function that fetches multiple URLs concurrently with a concurrency limit of N, returning results in the original order."),
        p("async", "Implement an async Rust `WorkQueue<T>` backed by tokio with a fixed number of worker tasks, graceful shutdown, and backpressure."),
        p("async", "Write a Rust async rate limiter using tokio that limits to N requests per second using a token bucket algorithm."),
        p("async", "Implement a Rust async retry middleware for reqwest that retries on 5xx errors with exponential backoff and jitter."),
        p("async", "Write an async Rust function that reads lines from multiple files concurrently and merges them into a single sorted output stream."),
        p("async", "Implement a Rust async pub/sub system using tokio::broadcast with multiple named topics and subscriber tracking."),
        p("async", "Write a Rust async connection pool using tokio with min/max connections, health checks, and idle timeout."),
        p("async", "Implement a Rust async pipeline where data flows through N stages concurrently, each stage processes items in parallel with a configurable buffer."),
        p("async", "Write a Rust function that runs a set of async tasks with a timeout, returning completed results and errors for timed-out tasks."),
        p("async", "Implement async Rust circuit breaker pattern: open after N failures, half-open after timeout, closed after success."),
        p("async", "Write a Rust async file watcher using tokio that debounces changes and emits events only after N milliseconds of inactivity."),
        p("async", "Implement a Rust async semaphore-based resource pool that acquires/releases resources and waits when all are in use."),
    ]
}

fn error_handling() -> Vec<Prompt> {
    vec![
        p("errors", "Design and implement a Rust error hierarchy for a web API: distinguish between validation errors, not-found, unauthorized, and internal errors. Include HTTP status code mapping."),
        p("errors", "Write a Rust function that parses a config file with detailed error messages including line numbers, column numbers, and suggestions for common mistakes."),
        p("errors", "Implement Rust error recovery: a function that processes a Vec of items, collects all errors, and returns both successful results and all errors at once."),
        p("errors", "Write a Rust `Result`-chaining example for a multi-step process (read config → connect DB → run query → serialize output) with context added at each step."),
        p("errors", "Implement a Rust error type that carries a backtrace, is serializable to JSON for API responses, and implements Display with user-friendly messages."),
        p("errors", "Write a Rust function that validates an HTTP request struct, returning a `Vec<ValidationError>` with field names, expected types, and actual values."),
    ]
}

fn data_structures() -> Vec<Prompt> {
    vec![
        p("data-structures", "Implement a lock-free MPSC queue in Rust using atomics. The queue should be bounded, support multiple producers, and one consumer."),
        p("data-structures", "Write a Rust skip list implementation with O(log n) insert, delete, and search."),
        p("data-structures", "Implement a Bloom filter in Rust with configurable false positive rate and optimal bit/hash count."),
        p("data-structures", "Write a Rust implementation of a red-black tree with insert, delete, and in-order traversal."),
        p("data-structures", "Implement a persistent (immutable) vector in Rust using a 32-way tree (like Clojure's PersistentVector)."),
        p("data-structures", "Write a Rust interval tree that supports inserting intervals and efficiently querying all intervals overlapping a given point or range."),
        p("data-structures", "Implement a Rust `DisjointSet` (union-find) with path compression and union by rank."),
        p("data-structures", "Write a Rust `TimeSeries<T>` struct that stores time-indexed values and supports efficient range queries and downsampling."),
    ]
}

fn distributed_systems() -> Vec<Prompt> {
    vec![
        p("distributed", "Write a Rust implementation of a consistent hash ring for distributing keys across N nodes, with virtual nodes for better balance."),
        p("distributed", "Implement Raft leader election in Rust using tokio: RequestVote RPC, election timeout randomization, and term tracking."),
        p("distributed", "Write a Rust implementation of the Gossip protocol for eventual consistency: random peer selection, rumor spreading, and convergence detection."),
        p("distributed", "Implement a Rust distributed rate limiter using Redis INCR+EXPIRE: sliding window algorithm with atomic operations."),
        p("distributed", "Write a Rust service mesh sidecar that intercepts HTTP traffic, adds trace headers, records latency, and emits metrics to stdout."),
        p("distributed", "Implement a Rust implementation of the two-phase commit protocol for distributed transactions."),
        p("distributed", "Write a Rust vector clock implementation for tracking causality in distributed systems. Include merge, increment, and happens-before operations."),
        p("distributed", "Implement a Rust distributed lock using a lease-based approach: acquire with TTL, renew, release, and automatic expiry."),
        p("distributed", "Write a Rust implementation of consistent hashing with virtual nodes that handles node addition and removal with minimal data redistribution."),
        p("distributed", "Implement a Rust gRPC middleware that adds request IDs, traces spans, and retries on UNAVAILABLE status codes."),
    ]
}

fn cli_tools() -> Vec<Prompt> {
    vec![
        p("cli", "Write a Rust CLI tool using clap that watches a directory for file changes and runs a command when files matching a glob pattern change."),
        p("cli", "Implement a Rust CLI tool that formats a JSON file in-place with configurable indentation, sorting of keys, and diff preview."),
        p("cli", "Write a Rust CLI tool that tail-follows multiple log files simultaneously, with colored output per file and regex filtering."),
        p("cli", "Implement a Rust CLI tool that benchmarks HTTP endpoints: configurable concurrency, duration, reports p50/p95/p99 latency histograms."),
        p("cli", "Write a Rust CLI tool that syncs two directories: shows diff, asks confirmation, then copies/deletes to make them identical."),
        p("cli", "Implement a Rust CLI tool that converts between TOML, YAML, JSON, and CBOR formats, preserving comments for TOML↔TOML conversion."),
        p("cli", "Write a Rust CLI progress bar library that supports nested progress, ETA, bytes-per-second, and renders properly in terminals of any width."),
    ]
}

fn testing() -> Vec<Prompt> {
    vec![
        p("testing", "Write a Rust property-based test using proptest for a custom sort function. Include strategies for edge cases: empty vecs, duplicates, negative numbers."),
        p("testing", "Implement Rust snapshot testing without external libraries: serialize output to a file on first run, compare on subsequent runs, with `--update` flag to regenerate."),
        p("testing", "Write a Rust integration test for a tokio-based HTTP server using reqwest: test all endpoints, verify status codes and response bodies."),
        p("testing", "Implement a Rust test fixture using RAII that starts a process before the test and kills it after, even on test failure."),
        p("testing", "Write a Rust fuzz test using cargo-fuzz for a parser function, including a corpus of interesting edge cases."),
        p("testing", "Implement Rust contract tests using trait-based test helpers that verify any implementation of a trait satisfies documented invariants."),
        p("testing", "Write a Rust mock HTTP server for tests using axum that records requests and returns configurable responses."),
    ]
}

fn refactoring() -> Vec<Prompt> {
    vec![
        p("refactoring", "Refactor this Rust function to replace nested if-let chains with a cleaner ? operator and custom error types:\n```rust\nfn process(data: &str) -> Option<i32> {\n    if let Some(x) = data.split(',').next() {\n        if let Ok(n) = x.trim().parse::<i32>() {\n            if n > 0 { return Some(n * 2); }\n        }\n    }\n    None\n}\n```"),
        p("refactoring", "Refactor this callback-heavy Rust code to use async/await:\n```rust\nfn fetch_and_process(url: &str, callback: Box<dyn Fn(Result<String, Error>)>) {\n    thread::spawn(move || {\n        let result = blocking_fetch(url);\n        callback(result.map(|r| r.body));\n    });\n}\n```"),
        p("refactoring", "Refactor this Rust struct to use the builder pattern with validation:\n```rust\nstruct Config {\n    host: String, port: u16, timeout: Duration, retry: u32\n}\nimpl Config {\n    fn new(host: &str, port: u16, timeout: u64, retry: u32) -> Self { ... }\n}\n```"),
        p("refactoring", "This Rust code allocates unnecessarily. Refactor to minimize heap allocations:\n```rust\nfn count_words(text: String) -> HashMap<String, usize> {\n    let mut counts = HashMap::new();\n    for word in text.split_whitespace() {\n        *counts.entry(word.to_string()).or_insert(0) += 1;\n    }\n    counts\n}\n```"),
        p("refactoring", "Refactor this Rust state machine from nested match to a proper type-state pattern that enforces valid transitions at compile time."),
        p("refactoring", "Convert this Rust synchronous code to async without changing the public API, preserving all error handling behavior."),
    ]
}

fn p(category: &str, instruction: &str) -> Prompt {
    Prompt {
        category: category.to_string(),
        instruction: instruction.to_string(),
    }
}
