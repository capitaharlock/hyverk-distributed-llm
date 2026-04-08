use thiserror::Error;

#[derive(Error, Debug)]
pub enum HyverkError {
    #[error("Config error: {0}")]
    Config(String),

    #[error("Inference error: {0}")]
    Inference(String),

    #[error("Coordinator error: {0}")]
    Coordinator(String),

    #[error("gRPC error: {0}")]
    Grpc(#[from] tonic::Status),

    #[error("Transport error: {0}")]
    Transport(String),

    #[error("Node not found: {0}")]
    NodeNotFound(String),

    #[error("Task not found: {0}")]
    TaskNotFound(String),
}
