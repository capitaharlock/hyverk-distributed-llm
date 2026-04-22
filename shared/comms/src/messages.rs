// Message types for coordinator ↔ client WebSocket communication.

use serde::{Deserialize, Serialize};

/// Messages FROM coordinator TO client
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CoordinatorMessage {
    /// Assign training shard
    TrainShard {
        assignment_id: String,
        layer_start: usize,
        layer_end: usize,
        data_url: String,
    },
    /// Start inference chain (first node: embed tokens, forward through layers)
    InferenceStart {
        request_id: String,
        token_ids: Vec<u32>,
        layer_start: usize,
        layer_end: usize,
        max_tokens: usize,
        #[serde(default)]
        temperature: Option<f32>,
        #[serde(default)]
        top_p: Option<f32>,
        #[serde(default)]
        top_k: Option<u32>,
    },
    /// Continue inference chain (middle/last: receive hidden states, forward)
    InferenceForward {
        request_id: String,
        /// Raw binary hidden states — NOT base64, sent as separate binary WS frame
        hidden_states_ref: String,
        layer_start: usize,
        layer_end: usize,
        is_last: bool,
        #[serde(default)]
        temperature: Option<f32>,
        #[serde(default)]
        top_p: Option<f32>,
        #[serde(default)]
        top_k: Option<u32>,
    },
    /// Decode one new token on the first node after prefill (embed single `new_token_id` with KV cache).
    InferenceContinue {
        request_id: String,
        new_token_id: u32,
        layer_start: usize,
        layer_end: usize,
        max_tokens: usize,
        temperature: f32,
        #[serde(default)]
        top_p: Option<f32>,
        #[serde(default)]
        top_k: Option<u32>,
    },
    /// Drop KV / state for `request_id` on this node (broadcast when a request finishes or aborts).
    InferenceEnd {
        request_id: String,
    },
    /// Assign layers to this node (sent after GPU node registers)
    LayerAssignment {
        layer_start: usize,
        layer_end: usize,
    },
    /// Switch role
    SwitchRole {
        role: String,    // "trainer" or "executor"
        urgency: String, // "graceful" or "immediate"
    },
    /// Keepalive
    Ping,
}

/// Messages FROM client TO coordinator
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ClientMessage {
    /// Initial registration
    Register {
        node_name: String,
        hardware_info: String,
        models: Vec<String>,
        ram_mb: u64,
        has_gpu: bool,
        layer_range: Option<(u32, u32)>,
        #[serde(default)]
        client_version: String,
        #[serde(default)]
        os: String,
    },
    /// Status heartbeat
    Heartbeat {
        active_tasks: u32,
        current_role: String,
        gpu_memory_used_mb: u64,
    },
    /// Training shard completed
    ShardComplete {
        assignment_id: String,
        loss: f32,
        steps: usize,
        adapter_data: Vec<u8>,
    },
    /// Hidden states after forward pass (for inference chain)
    ForwardResult {
        request_id: String,
        /// Hidden states as raw bytes (f16)
        hidden_states: Vec<u8>,
        shape: Vec<usize>,
    },
    /// Token generated (last node in chain)
    TokenGenerated {
        request_id: String,
        token_id: u32,
        is_eos: bool,
    },
    /// Node state update (downloading, ready, error, processing)
    StateUpdate { state: String, detail: String },
    /// Keepalive response
    Pong,
}
