//! Server-hosted channel control plane for the agent runtime.
//!
//! This crate introduces sessions, approvals, normalized channel envelopes, an
//! HTTP API, and adapter scaffolding for Slack and WhatsApp. It keeps the
//! existing `AgentRuntime` as a single-turn execution engine and layers
//! multi-turn orchestration above it.
//!
//! Public modules are organized so the boundaries remain explicit:
//! - `types` contains session/channel records shared across APIs and adapters
//! - `service` owns orchestration and persistence abstractions
//! - `runner` bridges the control plane to the single-turn runtime
//! - `api` exposes HTTP and SSE routes
//! - `adapters` normalize inbound and outbound channel payloads

pub mod adapters;
pub mod api;
pub mod observability;
pub mod runner;
pub mod service;
pub mod types;
pub mod whatsapp;

pub use adapters::{ChannelAdapter, SlackChannelAdapter, WhatsAppChannelAdapter};
pub use api::{
    ApiError, ApiState, ReqwestSlackDeliveryClient, SlackConnector, SlackDeliveryClient,
    SlackDeliveryError, SlackFileUpload, SlackMessagePayload, router, router_with_channels,
    router_with_debug_history, router_with_slack,
};
pub use observability::{
    ConsoleRuntimeHarnessListener, DebugHistoryAvailableFilters, DebugHistoryEventRow,
    DebugHistoryFilterOption, DebugHistoryRawArtifactRow, DebugHistorySessionSummary,
    DebugHistoryStore, DebugHistoryTurnDetail, DebugHistoryTurnFilters, DebugHistoryTurnSummary,
    PostgresRuntimeDebugListener, RuntimeHarnessEventEnvelope, RuntimeHarnessFanoutSink,
    RuntimeHarnessListener, RuntimeHarnessListenerError, RuntimeHarnessObservation,
    RuntimeHarnessRawArtifactEnvelope, SseRuntimeHarnessListener, TurnDebugSnapshot,
};
pub use runner::{
    RuntimeExecutionConfig, RuntimeTurnRunner, TurnRunner, TurnRunnerInput, TurnRunnerOutput,
};
pub use service::{
    ControlPlaneError, ConversationStore, ConversationStoreError, InMemoryConversationStore,
    JsonlConversationStore, SessionService,
};
pub use types::*;
pub use whatsapp::{
    LoggingWhatsAppDeliveryClient, ReqwestWhatsAppWebBridgeClient, WhatsAppConnector,
    WhatsAppControlClient, WhatsAppControlStatus, WhatsAppDeliveryClient, WhatsAppDeliveryError,
    WhatsAppGatewayError, WhatsAppGatewayHandle, WhatsAppMessagePayload,
};
