// hyverk-comms: WebSocket communication layer for Hyverk.
//
// Clients connect to coordinator via WebSocket (persistent connection).
// Works through NAT/firewalls because clients initiate the connection.
// Coordinator sends work and receives results through these connections.
//
// This is a native Rust module — NOT browser-based WebSocket.
// Clients are desktop applications running in terminals/background,
// not web browsers. The WebSocket protocol is used because:
// - Bidirectional communication on a single TCP connection
// - Works through corporate firewalls (port 443)
// - Efficient binary frame support for hidden states
// - Keepalive/reconnection built into the protocol

pub mod messages;
pub mod ws_client;
pub mod ws_server;
