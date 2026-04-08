// @llm-context: proto/hyverk.proto
// @llm-depends: registry.rs, task_store.rs, router.rs

use crate::metrics::LiveCounters;
use crate::registry::NodeRegistry;
use crate::router::TaskRouter;
use crate::task_store::TaskStore;
use hyverk_proto::node_coordinator_server::NodeCoordinator;
use hyverk_proto::*;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use tonic::{Request, Response, Status};

pub struct GrpcService {
    pub registry: NodeRegistry,
    pub task_store: TaskStore,
    pub router: TaskRouter,
    pub heartbeat_interval_secs: u64,
    pub counters: Arc<LiveCounters>,
}

#[tonic::async_trait]
impl NodeCoordinator for GrpcService {
    async fn register(
        &self,
        req: Request<RegisterRequest>,
    ) -> Result<Response<RegisterResponse>, Status> {
        self.counters.grpc_requests.fetch_add(1, Ordering::Relaxed);
        let r = req.into_inner();
        let caps = r.capabilities.unwrap_or_default();
        let node_id = self.registry.register(r.node_name, caps).await;
        Ok(Response::new(RegisterResponse {
            node_id,
            heartbeat_interval_secs: self.heartbeat_interval_secs,
        }))
    }

    async fn heartbeat(
        &self,
        req: Request<HeartbeatRequest>,
    ) -> Result<Response<HeartbeatResponse>, Status> {
        self.counters.grpc_requests.fetch_add(1, Ordering::Relaxed);
        self.counters.heartbeats.fetch_add(1, Ordering::Relaxed);
        self.counters.bytes_in.fetch_add(64, Ordering::Relaxed); // ~64 bytes per heartbeat
        let r = req.into_inner();
        if !self.registry.heartbeat(&r.node_id, r.active_tasks).await {
            return Err(Status::not_found("Node not registered"));
        }
        Ok(Response::new(HeartbeatResponse { acknowledged: true }))
    }

    async fn deregister(
        &self,
        req: Request<DeregisterRequest>,
    ) -> Result<Response<DeregisterResponse>, Status> {
        let r = req.into_inner();
        self.registry.deregister(&r.node_id).await;
        Ok(Response::new(DeregisterResponse { acknowledged: true }))
    }

    async fn poll_task(
        &self,
        req: Request<PollTaskRequest>,
    ) -> Result<Response<PollTaskResponse>, Status> {
        let r = req.into_inner();
        match self.router.assign_task_for_node(&r.node_id).await {
            Some(task) => Ok(Response::new(PollTaskResponse {
                has_task: true,
                task: Some(task),
            })),
            None => Ok(Response::new(PollTaskResponse {
                has_task: false,
                task: None,
            })),
        }
    }

    async fn submit_result(
        &self,
        req: Request<SubmitResultRequest>,
    ) -> Result<Response<SubmitResultResponse>, Status> {
        let r = req.into_inner();
        let found = if r.success {
            self.task_store
                .complete_task(&r.task_id, r.response_text, r.duration_ms)
                .await
        } else {
            self.task_store
                .fail_task(&r.task_id, r.error_message)
                .await
        };
        if !found {
            return Err(Status::not_found("Task not found"));
        }
        Ok(Response::new(SubmitResultResponse { acknowledged: true }))
    }
}
