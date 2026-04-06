//! MCP-first agent runtime.
//!
//! This crate owns the canonical run/turn/step/message state model, prompt
//! assembly, eventing, and the execution loop. MCP is the only active
//! capability in this slice. Local tools, memory, skills, and sub-agents are
//! represented as strongly typed placeholders so they can be added later
//! without changing the core loop structure.
//!
//! Public modules are split by concern:
//! - `state` holds canonical records used across prompts, events, and tests
//! - `runtime` holds the execution loop and MCP session preparation
//! - `model` defines the provider-neutral adapter contract
//! - `prompt` renders prompt snapshots from the canonical state
//! - `events` exposes the lifecycle stream emitted during execution

pub mod events;
mod ids;
pub mod model;
pub mod policy;
pub mod prompt;
pub mod runtime;
pub mod state;
pub mod subagent;
pub mod tools;

pub use events::{
    EventSink, InMemoryEventSink, RuntimeDebugSink, RuntimeEvent, RuntimeExecutor,
    RuntimeExecutorKind, RuntimeRawArtifact, RuntimeRawArtifactKind,
};
pub use ids::{MessageId, StepId, TurnId};
pub use runtime::{AgentRuntime, McpSession, RuntimeError};
pub use state::{
    CapabilityPlaceholders, ConversationMessage, ConversationRole, LocalToolName, McpCapability,
    MessageRecord, ModelConfig, PromptSection, PromptSnapshot, ResponseClient, ResponseFormat,
    ResponseTarget, RunRequest, RuntimeLimits, ServerName, StepOutcomeKind, StepRecord,
    TerminationReason, TurnOutcome, TurnRecord, UsageSummary,
};
pub use tools::{ToolCallResultEnvelope, ToolDescriptor, ToolFamily, builtin_local_tool_catalog};
