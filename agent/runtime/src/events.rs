//! Typed runtime events and sink abstractions.
//!
//! The runtime emits events as it progresses, but in the first slice those
//! events are only retained in memory. The sink interface exists so callers can
//! later add persistence or tracing without changing the execution loop.

use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    ids::{StepId, TurnId},
    policy::{ToolMaskDecision, ToolMaskEnforcementMode},
    state::{DelegationTarget, TerminationReason, UsageSummary},
};

/// Event sink abstraction used by the runtime loop.
pub trait EventSink: Send {
    /// Records one runtime event.
    fn record(&mut self, event: RuntimeEvent);
}

/// Sink abstraction that can also receive raw debug artifacts.
pub trait RuntimeDebugSink: EventSink {
    /// Records one raw request/response artifact.
    fn record_raw_artifact(&mut self, artifact: RuntimeRawArtifact);
}

/// Default sink that retains all events in memory and returns them to callers.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InMemoryEventSink {
    events: Vec<RuntimeEvent>,
}

impl InMemoryEventSink {
    /// Returns the collected events in their recorded order.
    pub fn into_events(self) -> Vec<RuntimeEvent> {
        self.events
    }
}

impl EventSink for InMemoryEventSink {
    fn record(&mut self, event: RuntimeEvent) {
        self.events.push(event);
    }
}

impl RuntimeDebugSink for InMemoryEventSink {
    fn record_raw_artifact(&mut self, _artifact: RuntimeRawArtifact) {}
}

/// Actor responsible for one runtime event or debug artifact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeExecutorKind {
    MainAgent,
    Subagent,
}

/// Human-readable executor details surfaced in debug history.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeExecutor {
    pub kind: RuntimeExecutorKind,
    pub display_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subagent_type: Option<String>,
}

impl RuntimeExecutor {
    pub fn main_agent() -> Self {
        Self {
            kind: RuntimeExecutorKind::MainAgent,
            display_name: "Main Agent".to_owned(),
            subagent_type: None,
        }
    }

    pub fn subagent(display_name: impl Into<String>, subagent_type: impl Into<String>) -> Self {
        Self {
            kind: RuntimeExecutorKind::Subagent,
            display_name: display_name.into(),
            subagent_type: Some(subagent_type.into()),
        }
    }
}

/// Debug-only raw artifact kinds captured alongside runtime events.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeRawArtifactKind {
    ModelRequest,
    ModelResponse,
    ModelError,
    McpRequest,
    McpResponse,
    McpError,
    LocalToolRequest,
    LocalToolResponse,
    LocalToolError,
    PolicyDecision,
}

/// Raw transport payload retained for internal debugging.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimeRawArtifact {
    /// Turn identifier.
    pub turn_id: TurnId,
    /// Optional step identifier when the artifact belongs to a step.
    pub step_id: Option<StepId>,
    /// Artifact timestamp.
    pub occurred_at: SystemTime,
    /// Artifact kind used for rendering and persistence.
    pub kind: RuntimeRawArtifactKind,
    /// Source label such as `openai_responses` or `mcp_client`.
    pub source: String,
    /// Executor responsible for producing the artifact.
    pub executor: RuntimeExecutor,
    /// Short summary shown in the debug UI.
    pub summary: Option<String>,
    /// Redacted raw payload.
    pub payload: Value,
}

/// Runtime event stream emitted during one turn.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum RuntimeEvent {
    TurnStarted {
        /// Turn identifier.
        turn_id: TurnId,
        /// Executor responsible for the event.
        executor: RuntimeExecutor,
        /// Event timestamp.
        at: SystemTime,
    },
    StepStarted {
        /// Turn identifier.
        turn_id: TurnId,
        /// Step identifier.
        step_id: StepId,
        /// One-based step number.
        step_number: u32,
        /// Executor responsible for the event.
        executor: RuntimeExecutor,
        /// Event timestamp.
        at: SystemTime,
    },
    PromptBuilt {
        /// Turn identifier.
        turn_id: TurnId,
        /// Step identifier.
        step_id: StepId,
        /// Executor responsible for the event.
        executor: RuntimeExecutor,
        /// Event timestamp.
        at: SystemTime,
    },
    HandoffToSubagent {
        /// Turn identifier.
        turn_id: TurnId,
        /// Step identifier.
        step_id: StepId,
        /// Executor responsible for the event.
        executor: RuntimeExecutor,
        /// Event timestamp.
        at: SystemTime,
        /// Configured sub-agent type receiving control.
        subagent_type: String,
        /// Delegated goal passed to the sub-agent.
        goal: String,
        /// Delegated execution target selected by the main agent.
        target: DelegationTarget,
    },
    ModelCalled {
        /// Turn identifier.
        turn_id: TurnId,
        /// Step identifier.
        step_id: StepId,
        /// Executor responsible for the event.
        executor: RuntimeExecutor,
        /// Event timestamp.
        at: SystemTime,
    },
    ModelResponded {
        /// Turn identifier.
        turn_id: TurnId,
        /// Step identifier.
        step_id: StepId,
        /// Event timestamp.
        at: SystemTime,
        /// Elapsed model-call latency.
        latency: Duration,
        /// Executor responsible for the event.
        executor: RuntimeExecutor,
        /// Provider-reported token usage for the model call.
        usage: UsageSummary,
    },
    AnswerRenderStarted {
        /// Turn identifier.
        turn_id: TurnId,
        /// Step identifier that triggered final rendering.
        step_id: StepId,
        /// Executor responsible for the render event.
        executor: RuntimeExecutor,
        /// Event timestamp.
        at: SystemTime,
    },
    AnswerTextDelta {
        /// Turn identifier.
        turn_id: TurnId,
        /// Step identifier that triggered final rendering.
        step_id: StepId,
        /// Executor responsible for the render event.
        executor: RuntimeExecutor,
        /// Event timestamp.
        at: SystemTime,
        /// Incremental answer text emitted by the renderer.
        delta: String,
    },
    AnswerRenderCompleted {
        /// Turn identifier.
        turn_id: TurnId,
        /// Step identifier that triggered final rendering.
        step_id: StepId,
        /// Executor responsible for the render event.
        executor: RuntimeExecutor,
        /// Event timestamp.
        at: SystemTime,
    },
    AnswerRenderFailed {
        /// Turn identifier.
        turn_id: TurnId,
        /// Step identifier that triggered final rendering.
        step_id: StepId,
        /// Executor responsible for the render event.
        executor: RuntimeExecutor,
        /// Event timestamp.
        at: SystemTime,
        /// Human-readable failure message.
        error: String,
    },
    HandoffToMainAgent {
        /// Turn identifier.
        turn_id: TurnId,
        /// Step identifier.
        step_id: StepId,
        /// Executor responsible for the event.
        executor: RuntimeExecutor,
        /// Event timestamp.
        at: SystemTime,
        /// Configured sub-agent type returning control.
        subagent_type: String,
        /// Final delegated execution status.
        status: String,
    },
    ToolMaskEvaluated {
        turn_id: TurnId,
        step_id: StepId,
        executor: RuntimeExecutor,
        at: SystemTime,
        enforcement_mode: ToolMaskEnforcementMode,
        allowed_tool_ids: Vec<String>,
        denied_tool_ids: Vec<String>,
        decisions: Vec<ToolMaskDecision>,
    },
    McpCalled {
        /// Turn identifier.
        turn_id: TurnId,
        /// Step identifier.
        step_id: StepId,
        /// Target server name.
        server_name: String,
        /// Target tool name.
        tool_name: String,
        /// Executor responsible for the event.
        executor: RuntimeExecutor,
        /// Event timestamp.
        at: SystemTime,
    },
    McpResponded {
        /// Turn identifier.
        turn_id: TurnId,
        /// Step identifier.
        step_id: StepId,
        /// Server that handled the tool call.
        server_name: String,
        /// Tool that was invoked.
        tool_name: String,
        /// Executor responsible for the event.
        executor: RuntimeExecutor,
        /// Event timestamp.
        at: SystemTime,
        /// Tool-call latency.
        latency: Duration,
        /// Whether the tool result represented an error outcome.
        was_error: bool,
        /// Human-readable summary of the tool result.
        result_summary: String,
        /// Error text when the call failed or the tool returned an MCP error.
        error: Option<String>,
        /// Full response payload retained for debugging.
        response_payload: Value,
    },
    LocalToolCalled {
        turn_id: TurnId,
        step_id: StepId,
        tool_name: String,
        executor: RuntimeExecutor,
        at: SystemTime,
    },
    LocalToolResponded {
        turn_id: TurnId,
        step_id: StepId,
        tool_name: String,
        executor: RuntimeExecutor,
        at: SystemTime,
        latency: Duration,
        was_error: bool,
        result_summary: String,
        error: Option<String>,
        response_payload: Value,
    },
    StepEnded {
        /// Turn identifier.
        turn_id: TurnId,
        /// Step identifier.
        step_id: StepId,
        /// Executor responsible for the event.
        executor: RuntimeExecutor,
        /// Event timestamp.
        at: SystemTime,
    },
    TurnEnded {
        /// Turn identifier.
        turn_id: TurnId,
        /// Executor responsible for the event.
        executor: RuntimeExecutor,
        /// Event timestamp.
        at: SystemTime,
        /// Why the turn ended.
        termination: TerminationReason,
        /// Aggregate turn usage.
        usage: UsageSummary,
    },
}
