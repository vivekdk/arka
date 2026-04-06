//! Provider-neutral model adapter contracts.
//!
//! The runtime depends only on these types. Concrete provider crates translate
//! their native request and response shapes into this strongly typed boundary.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::{
    policy::ToolMaskPlan,
    state::{
        DelegationTarget, LocalToolName, ModelConfig, PromptSnapshot, ResponseTarget,
        ServerName, UsageSummary,
    },
    tools::ToolDescriptor,
};

/// Request sent from the runtime to a model adapter for one step.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelStepRequest {
    /// One-based step number within the current turn.
    pub step_number: u32,
    /// Fully assembled prompt snapshot for this step.
    pub prompt: PromptSnapshot,
    /// Provider and model selection for the call.
    pub model_config: ModelConfig,
    /// Static registered tool catalog relevant to the current request.
    pub registered_tools: Vec<ToolDescriptor>,
    /// Harness-evaluated tool constraints for the current step.
    pub tool_mask_plan: ToolMaskPlan,
}

/// Result returned by a model adapter after evaluating one step prompt.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelAdapterResponse {
    /// Normalized structured decision returned by the provider.
    pub decision: ModelStepDecision,
    /// Provider-reported token usage for the step.
    pub usage: UsageSummary,
}

/// Request sent to a model adapter to render the final user-facing answer.
#[derive(Clone, Debug, PartialEq)]
pub struct FinalAnswerRenderRequest {
    /// Provider and model selection for the render pass.
    pub model_config: ModelConfig,
    /// Fully rendered prompt for the render pass.
    pub prompt: String,
    /// Planner-produced answer brief used as the default non-streaming fallback.
    pub answer_brief: String,
    /// Target client and formatting profile for the final reply.
    pub response_target: ResponseTarget,
}

/// Result returned after rendering the final user-facing answer.
#[derive(Clone, Debug, PartialEq)]
pub struct FinalAnswerRenderResponse {
    /// Canonical assistant text preserved in the transcript.
    pub canonical_text: String,
    /// Client-formatted assistant text delivered to the caller.
    pub display_text: String,
    /// Provider-reported token usage for the render pass.
    pub usage: UsageSummary,
}

/// Sink used by model adapters to stream final answer text incrementally.
pub trait FinalAnswerStreamSink: Send {
    fn record_text_delta(&mut self, delta: &str);
}

/// Artifact kinds emitted by model adapters for debug persistence.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelAdapterArtifactKind {
    Request,
    Response,
    Error,
}

/// Raw transport artifact emitted by a model adapter.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelAdapterArtifact {
    /// Artifact direction or outcome.
    pub kind: ModelAdapterArtifactKind,
    /// Adapter-specific source label such as `openai_responses`.
    pub source: String,
    /// Short human-readable description shown in debug history.
    pub summary: Option<String>,
    /// Raw JSON payload after any adapter-side redaction.
    pub payload: Value,
}

/// Sink used by model adapters to publish raw request/response artifacts.
pub trait ModelAdapterDebugSink: Send {
    fn record_model_artifact(&mut self, artifact: ModelAdapterArtifact);
}

/// Structured decision emitted by the model for one loop iteration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ModelStepDecision {
    /// End the turn and return the supplied assistant message.
    Final { content: String },
    /// Delegate work to one configured sub-agent.
    DelegateSubagent {
        delegation: SubagentDelegationRequest,
    },
}

/// One configured sub-agent delegation requested by the main model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SubagentDelegationRequest {
    /// Configured sub-agent type to invoke.
    #[serde(rename = "type")]
    pub subagent_type: String,
    /// Structured target selected by the main agent.
    pub target: DelegationTarget,
    /// Short statement of what the delegated action should achieve.
    #[serde(default)]
    pub goal: String,
}

/// Request sent to a sub-agent renderer/model for a delegated step.
#[derive(Clone, Debug, PartialEq)]
pub struct SubagentStepRequest {
    pub subagent_type: String,
    pub prompt: String,
    pub model_config: ModelConfig,
    pub registered_tools: Vec<ToolDescriptor>,
    pub tool_mask_plan: ToolMaskPlan,
}

/// Result returned by a sub-agent model call.
#[derive(Clone, Debug, PartialEq)]
pub struct SubagentAdapterResponse {
    pub decision: SubagentDecision,
    pub usage: UsageSummary,
}

/// Execution decision emitted by a typed sub-agent.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SubagentDecision {
    McpToolCall {
        server_name: ServerName,
        tool_name: String,
        #[serde(default)]
        arguments: Value,
    },
    McpResourceRead {
        server_name: ServerName,
        resource_uri: String,
    },
    LocalToolCall {
        tool_name: LocalToolName,
        #[serde(default)]
        arguments: Value,
    },
    Done {
        summary: String,
    },
    Partial {
        summary: String,
        reason: String,
    },
    CannotExecute {
        reason: String,
    },
}

/// Abstract interface implemented by concrete model providers.
#[async_trait]
pub trait ModelAdapter: Send + Sync {
    /// Produces the next runtime decision from the assembled prompt snapshot.
    async fn generate_step(
        &self,
        request: ModelStepRequest,
        debug_sink: Option<&mut dyn ModelAdapterDebugSink>,
    ) -> Result<ModelAdapterResponse, ModelAdapterError>;

    /// Produces one delegated sub-agent execution decision.
    async fn generate_subagent_step(
        &self,
        request: SubagentStepRequest,
        debug_sink: Option<&mut dyn ModelAdapterDebugSink>,
    ) -> Result<SubagentAdapterResponse, ModelAdapterError>;

    /// Renders the final user-facing assistant answer.
    ///
    /// The default implementation preserves current behavior by returning the
    /// planner-produced brief verbatim while still notifying any streaming sink.
    async fn render_final_answer(
        &self,
        request: FinalAnswerRenderRequest,
        mut stream_sink: Option<&mut dyn FinalAnswerStreamSink>,
        _debug_sink: Option<&mut dyn ModelAdapterDebugSink>,
    ) -> Result<FinalAnswerRenderResponse, ModelAdapterError> {
        if let Some(sink) = stream_sink.as_mut() {
            sink.record_text_delta(&request.answer_brief);
        }
        Ok(FinalAnswerRenderResponse {
            canonical_text: request.answer_brief.clone(),
            display_text: request.answer_brief,
            usage: UsageSummary::default(),
        })
    }
}

/// Provider-normalized model failure modes.
#[derive(Debug, Error, PartialEq)]
pub enum ModelAdapterError {
    /// Structured model output could not be parsed or validated.
    #[error("model returned invalid structured output: {0}")]
    InvalidDecision(String),
    /// Transport to the model provider failed.
    #[error("model transport failed: {0}")]
    Transport(String),
    /// Provider returned an application-level error.
    #[error("model provider failed: {0}")]
    Provider(String),
}
