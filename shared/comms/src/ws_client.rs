// WebSocket client — connects to coordinator from behind NAT/firewall.
// Maintains persistent connection. Receives work, sends results.

use crate::messages::{ClientMessage, CoordinatorMessage};
use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{info, warn, error};

pub struct WsClient {
    coordinator_url: String,
    node_name: String,
}

impl WsClient {
    pub fn new(coordinator_ws_url: &str, node_name: &str) -> Self {
        Self {
            coordinator_url: coordinator_ws_url.to_string(),
            node_name: node_name.to_string(),
        }
    }

    /// Connect and run the message loop. Reconnects on failure.
    pub async fn run<F>(&self, handler: F)
    where
        F: Fn(CoordinatorMessage) -> Option<ClientMessage> + Send + Sync + 'static,
    {
        loop {
            match self.connect_and_handle(&handler).await {
                Ok(()) => info!("WebSocket closed cleanly"),
                Err(e) => warn!("WebSocket error: {e}"),
            }
            info!("Reconnecting in 5s...");
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        }
    }

    async fn connect_and_handle<F>(&self, handler: &F) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
    where
        F: Fn(CoordinatorMessage) -> Option<ClientMessage>,
    {
        let url = format!("{}/ws", self.coordinator_url);
        info!(url = %url, "Connecting WebSocket...");

        let (ws, _) = connect_async(&url).await?;
        let (mut sink, mut stream) = ws.split();
        info!("WebSocket connected");

        // Register
        let reg = ClientMessage::Register {
            node_name: self.node_name.clone(),
            hardware_info: String::new(),
            models: vec![],
            ram_mb: 0,
            has_gpu: false,
        };
        let msg = serde_json::to_string(&reg)?;
        sink.send(Message::Text(msg.into())).await?;

        // Keepalive + message loop
        let mut ping_interval = tokio::time::interval(std::time::Duration::from_secs(30));

        loop {
            tokio::select! {
                _ = ping_interval.tick() => {
                    let pong = serde_json::to_string(&ClientMessage::Pong)?;
                    sink.send(Message::Text(pong.into())).await?;
                }
                msg = stream.next() => {
                    match msg {
                        Some(Ok(Message::Text(text))) => {
                            match serde_json::from_str::<CoordinatorMessage>(&text) {
                                Ok(coord_msg) => {
                                    if let Some(response) = handler(coord_msg) {
                                        let resp_json = serde_json::to_string(&response)?;
                                        sink.send(Message::Text(resp_json.into())).await?;
                                    }
                                }
                                Err(e) => warn!("Failed to parse coordinator message: {e}"),
                            }
                        }
                        Some(Ok(Message::Binary(data))) => {
                            // Binary frame = hidden states for inference
                            info!(size = data.len(), "Received binary hidden states");
                            // TODO: process and forward
                        }
                        Some(Ok(Message::Ping(d))) => {
                            sink.send(Message::Pong(d)).await?;
                        }
                        Some(Ok(Message::Close(_))) | None => {
                            info!("WebSocket closed by coordinator");
                            break;
                        }
                        Some(Err(e)) => {
                            error!("WebSocket error: {e}");
                            break;
                        }
                        _ => {}
                    }
                }
            }
        }

        Ok(())
    }
}
