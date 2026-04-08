//! Canonical runtime state records and supporting types.
//!
//! The runtime is intentionally record-first: turns, steps, messages, events,
//! and usage are stored in structured types and prompts are derived from them.
//! This makes retries, observability, and future capability families much safer
//! than treating prompt text as the system of record.

use std::{
    fmt,
    path::PathBuf,
    time::{Duration, SystemTime},
};

use mcp_metadata::CapabilityKind;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use time::{OffsetDateTime, UtcOffset, format_description::well_known::Rfc3339};

use crate::{
    ids::{MessageId, StepId, TurnId},
    model::ModelStepDecision,
    policy::ToolMaskPlan,
};

/// Strongly typed MCP server identifier.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ServerName(String);

impl ServerName {
    /// Creates a server name after rejecting blank values.
    pub fn new(value: impl Into<String>) -> Result<Self, StateValidationError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(StateValidationError::BlankServerName);
        }
        Ok(Self(value))
    }

    /// Borrows the inner server name.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ServerName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl TryFrom<&str> for ServerName {
    type Error = StateValidationError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::new(value.to_owned())
    }
}

impl std::str::FromStr for ServerName {
    type Err = StateValidationError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::new(value.to_owned())
    }
}

/// Strongly typed local runtime tool identifier.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LocalToolName(String);

impl LocalToolName {
    /// Creates a local tool name after rejecting blank values.
    pub fn new(value: impl Into<String>) -> Result<Self, StateValidationError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(StateValidationError::BlankLocalToolName);
        }
        Ok(Self(value))
    }

    /// Borrows the inner local tool name.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for LocalToolName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Limits that guard one runtime turn.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimeLimits {
    /// Upper bound on reasoning iterations in a single turn.
    pub max_steps_per_turn: u32,
    /// Upper bound on MCP tool calls the model may request in one step.
    pub max_mcp_calls_per_step: u32,
    /// Upper bound on sub-agent reasoning iterations inside one delegation.
    pub max_subagent_steps_per_invocation: u32,
    /// Upper bound on MCP actions a sub-agent may execute inside one delegation.
    pub max_subagent_mcp_calls_per_invocation: u32,
    /// Timeout applied to each individual MCP tool call.
    pub mcp_call_timeout: Duration,
    /// Total wall-clock budget for the full turn.
    pub turn_timeout: Duration,
}

impl Default for RuntimeLimits {
    fn default() -> Self {
        Self {
            max_steps_per_turn: 8,
            max_mcp_calls_per_step: 4,
            max_subagent_steps_per_invocation: 8,
            max_subagent_mcp_calls_per_invocation: 4,
            mcp_call_timeout: Duration::from_secs(10),
            turn_timeout: Duration::from_secs(60),
        }
    }
}

/// Provider selection and provider-specific options for one turn.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Provider-specific model identifier.
    pub model_name: String,
    /// Opaque provider options forwarded to the model adapter.
    pub provider_options: Value,
}

impl ModelConfig {
    /// Creates a model config with an empty provider options object.
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            provider_options: Value::Object(Default::default()),
        }
    }
}

/// External client family requesting the turn output.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseClient {
    Api,
    Cli,
    Slack,
    WhatsApp,
}

/// Output formatting profile requested for the final assistant reply.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormat {
    PlainText,
    Markdown,
    SlackMrkdwn,
    WhatsAppText,
}

/// Resolved client and formatting target for one turn.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseTarget {
    pub client: ResponseClient,
    pub format: ResponseFormat,
}

/// One prior conversation message carried into a new turn.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConversationMessage {
    /// When the historical message was originally created.
    pub timestamp: SystemTime,
    /// Historical speaker role.
    pub role: ConversationRole,
    /// Historical message text.
    pub content: String,
}

/// Role used in historical conversation messages.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConversationRole {
    /// User-authored prior message.
    User,
    /// Assistant-authored prior message.
    Assistant,
}

impl fmt::Display for ConversationRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::User => write!(f, "user"),
            Self::Assistant => write!(f, "assistant"),
        }
    }
}

/// Aggregate token usage information.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct UsageSummary {
    /// Provider-reported prompt/input token count.
    pub input_tokens: u32,
    /// Provider-reported cached prompt/input token count.
    #[serde(default)]
    pub cached_tokens: u32,
    /// Provider-reported completion/output token count.
    pub output_tokens: u32,
    /// Provider-reported total token count.
    pub total_tokens: u32,
}

impl UsageSummary {
    /// Adds another usage record into this summary.
    pub fn add_assign(&mut self, other: UsageSummary) {
        self.input_tokens += other.input_tokens;
        self.cached_tokens += other.cached_tokens;
        self.output_tokens += other.output_tokens;
        self.total_tokens += other.total_tokens;
    }
}

/// One MCP capability surfaced to the model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct McpCapability {
    /// MCP server that owns this capability.
    pub server_name: ServerName,
    /// Optional server-level description surfaced from minimal metadata.
    pub server_description: Option<String>,
    /// Capability family exposed by the server.
    pub kind: CapabilityKind,
    /// Stable capability identifier.
    pub capability_id: String,
    /// Optional human-friendly title surfaced to the model.
    pub title: Option<String>,
    /// Optional description surfaced to the model.
    pub description: Option<String>,
}

/// Inactive capability families rendered into the prompt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CapabilityPlaceholders {
    /// Local tool names reserved for future activation.
    pub local_tools: Vec<String>,
    /// Placeholder text rendered for memory capability slots.
    pub memory_description: String,
    /// Placeholder text rendered for skill capability slots.
    pub skills_description: String,
    /// Placeholder text rendered for memory capability slots.
    pub sub_agents_description: String,
}

impl Default for CapabilityPlaceholders {
    fn default() -> Self {
        Self {
            local_tools: vec![
                "glob".to_owned(),
                "read_file".to_owned(),
                "edit_file".to_owned(),
                "write_file".to_owned(),
                "bash".to_owned(),
            ],
            memory_description: "Memory capability exists structurally but is inactive.".to_owned(),
            skills_description: "Skill capability exists structurally but is inactive.".to_owned(),
            sub_agents_description: "No sub-agents are configured.".to_owned(),
        }
    }
}

/// Main-agent-visible summary of one configured sub-agent.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentCard {
    pub subagent_type: String,
    pub display_name: String,
    pub purpose: String,
    pub when_to_use: String,
    pub target_requirements: String,
    pub result_summary: String,
}

/// One prompt section captured for observability and tests.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PromptSection {
    /// Human-readable section title.
    pub title: String,
    /// Fully rendered content for the section.
    pub content: String,
}

/// Full assembled prompt snapshot for one step.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PromptSnapshot {
    /// Final prompt text handed to the model adapter.
    pub rendered: String,
    /// Structured prompt sections retained for observability and tests.
    pub sections: Vec<PromptSection>,
}

/// The different reasons a turn can terminate.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TerminationReason {
    /// The model produced a final answer.
    Final,
    /// The turn exhausted its configured step budget.
    MaxStepsReached,
    /// The turn exceeded its configured time budget.
    Timeout,
    /// The runtime rejected invalid caller or model state.
    ValidationError,
    /// An internal runtime or transport failure aborted the turn.
    RuntimeError,
}

/// One per-step status classification.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StepOutcomeKind {
    /// The step completed and the turn should continue.
    Continue,
    /// The step produced the final turn answer.
    Final,
    /// The step ended in failure.
    Failed,
}

/// Top-level request accepted by the runtime.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunRequest {
    /// Path to the system prompt template file for this turn.
    pub system_prompt_path: PathBuf,
    /// Root directory used for prompt rendering and local tool execution.
    pub working_directory: PathBuf,
    /// Prior conversation history to inject before the new user message.
    pub conversation_history: Vec<ConversationMessage>,
    /// Recent session-scoped runtime messages carried over from prior turns.
    pub recent_session_messages: Vec<MessageRecord>,
    /// Fresh user message that starts this turn.
    pub user_message: String,
    /// Resolved client and formatting target for the final reply.
    pub response_target: ResponseTarget,
    /// Path to the MCP registry used to prepare server connections.
    pub registry_path: PathBuf,
    /// Path to the configured sub-agent registry.
    pub subagent_registry_path: PathBuf,
    /// Optional JSON overlay used for per-environment tool policy rules.
    pub tool_policy_path: Option<PathBuf>,
    /// Optional allowlist of server names enabled for the turn.
    pub enabled_servers: Option<Vec<ServerName>>,
    /// Runtime safety and timeout limits.
    pub limits: RuntimeLimits,
    /// Provider and model selection for this turn.
    pub model_config: ModelConfig,
}

/// Final runtime result returned to callers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TurnOutcome {
    /// Final assistant text returned to the caller.
    pub final_text: String,
    /// Final assistant text rendered for the active client target.
    pub display_text: String,
    /// Canonical turn record built during execution.
    pub turn: TurnRecord,
    /// Lifecycle events emitted while the turn ran.
    pub events: Vec<crate::events::RuntimeEvent>,
    /// Aggregate token usage across all model calls in the turn.
    pub usage: UsageSummary,
    /// Why the turn ended.
    pub termination: TerminationReason,
}

/// Canonical record for one turn.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TurnRecord {
    /// Stable identifier for this turn execution.
    pub turn_id: TurnId,
    /// Turn start timestamp.
    pub started_at: SystemTime,
    /// Turn end timestamp.
    pub ended_at: SystemTime,
    /// Ordered step records executed during the turn.
    pub steps: Vec<StepRecord>,
    /// Flattened message timeline for the entire turn.
    pub messages: Vec<MessageRecord>,
    /// Final assistant text when the turn reached a final answer.
    pub final_text: Option<String>,
    /// Why the turn ended.
    pub termination: TerminationReason,
    /// Aggregate token usage across the turn.
    pub usage: UsageSummary,
}

/// Canonical record for one step.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StepRecord {
    /// Stable identifier for the step.
    pub step_id: StepId,
    /// One-based ordinal within the turn.
    pub step_number: u32,
    /// Step start timestamp.
    pub started_at: SystemTime,
    /// Step end timestamp.
    pub ended_at: SystemTime,
    /// Prompt snapshot shown to the model.
    pub prompt: PromptSnapshot,
    /// Model decision returned for the step, when one was produced.
    pub decision: Option<ModelStepDecision>,
    /// Messages generated during this step.
    pub messages: Vec<MessageRecord>,
    /// Final step status.
    pub outcome: StepOutcomeKind,
    /// Token usage attributed to this step.
    pub usage: UsageSummary,
}

/// Canonical runtime message record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum MessageRecord {
    /// User-authored message for the current turn.
    User(UserMessageRecord),
    /// Assistant-authored message for the current turn.
    Llm(LlmMessageRecord),
    /// MCP execution request prepared by runtime or sub-agent.
    McpCall(McpCallMessageRecord),
    /// Result returned from the executed MCP action.
    McpResult(McpResultMessageRecord),
    /// Placeholder local-tool invocation.
    LocalToolCall(LocalToolCallMessageRecord),
    /// Placeholder local-tool result.
    LocalToolResult(LocalToolResultMessageRecord),
    /// Placeholder memory read event.
    MemoryRead(PlaceholderMessageRecord),
    /// Placeholder memory write event.
    MemoryWrite(PlaceholderMessageRecord),
    /// Placeholder skill invocation.
    SkillCall(PlaceholderMessageRecord),
    /// Typed sub-agent invocation.
    SubAgentCall(SubAgentCallMessageRecord),
    /// Typed sub-agent result.
    SubAgentResult(SubAgentResultMessageRecord),
}

impl MessageRecord {
    /// Produces a concise human-readable line used in prompt assembly.
    pub fn summary_line(&self) -> String {
        match self {
            Self::User(record) => format!(
                "- user_message [{}]: {}",
                format_system_time(record.timestamp),
                record.content
            ),
            Self::Llm(record) => format!(
                "- llm_message [{}]: {}",
                format_system_time(record.timestamp),
                record.content
            ),
            Self::McpCall(record) => format!(
                "- mcp_call: server={} kind={:?} target={} args={}",
                record.target.server_name,
                record.target.capability_kind,
                record.target.capability_id,
                record.arguments
            ),
            Self::McpResult(record) => format!(
                "- mcp_result: server={} kind={:?} target={} error={} payload={}",
                record.target.server_name,
                record.target.capability_kind,
                record.target.capability_id,
                record.error.is_some(),
                record.result_summary
            ),
            Self::LocalToolCall(record) => {
                format!(
                    "- tool_call: local_tool={} args={}",
                    record.tool_name, record.arguments
                )
            }
            Self::LocalToolResult(record) => format!(
                "- tool_result: local_tool={} status={} error={} payload={}",
                record.tool_name,
                record.status,
                record.error.is_some(),
                record.result_summary
            ),
            Self::MemoryRead(record) => format!("- memory_read: {}", record.label),
            Self::MemoryWrite(record) => format!("- memory_write: {}", record.label),
            Self::SkillCall(record) => format!("- skill_call: {}", record.label),
            Self::SubAgentCall(record) => format!(
                "- sub_agent_call: type={} goal={} target={}",
                record.subagent_type,
                record.goal,
                record.target.summary()
            ),
            Self::SubAgentResult(record) => format!(
                "- sub_agent_result: type={} status={} actions={} detail={}",
                record.subagent_type, record.status, record.executed_action_count, record.detail
            ),
        }
    }
}

pub(crate) fn format_system_time(timestamp: SystemTime) -> String {
    let offset = UtcOffset::current_local_offset().unwrap_or(UtcOffset::UTC);
    let timestamp = OffsetDateTime::from(timestamp).to_offset(offset);

    timestamp
        .format(&Rfc3339)
        .unwrap_or_else(|_| timestamp.unix_timestamp().to_string())
}

/// User-authored message captured as part of the current turn.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UserMessageRecord {
    /// Stable identifier for the message record.
    pub message_id: MessageId,
    /// Message timestamp.
    pub timestamp: SystemTime,
    /// User-authored text.
    pub content: String,
}

/// Assistant-authored message captured as part of the current turn.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LlmMessageRecord {
    /// Stable identifier for the message record.
    pub message_id: MessageId,
    /// Message timestamp.
    pub timestamp: SystemTime,
    /// Assistant-authored text.
    pub content: String,
}

/// One selected MCP capability target.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpCapabilityTarget {
    pub server_name: ServerName,
    pub capability_kind: CapabilityKind,
    pub capability_id: String,
}

/// One local-tools delegation scope selected by the main agent.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalToolsScopeTarget {
    pub scope: String,
}

impl LocalToolsScopeTarget {
    pub fn working_directory() -> Self {
        Self {
            scope: "working_directory".to_owned(),
        }
    }
}

/// Typed delegation target for one sub-agent invocation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum DelegationTarget {
    McpCapability(McpCapabilityTarget),
    LocalToolsScope(LocalToolsScopeTarget),
}

impl DelegationTarget {
    pub fn summary(&self) -> String {
        match self {
            Self::McpCapability(target) => format!(
                "{}::{:?}::{}",
                target.server_name, target.capability_kind, target.capability_id
            ),
            Self::LocalToolsScope(target) => format!("local_tools::{}", target.scope),
        }
    }
}

/// One MCP invocation requested by runtime.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct McpCallMessageRecord {
    /// Stable identifier for the message record.
    pub message_id: MessageId,
    /// Message timestamp.
    pub timestamp: SystemTime,
    /// Target capability selected for execution.
    pub target: McpCapabilityTarget,
    /// Arguments forwarded to the tool, or `null` for resource reads.
    pub arguments: Value,
}

/// Result from one executed MCP invocation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct McpResultMessageRecord {
    /// Stable identifier for the message record.
    pub message_id: MessageId,
    /// Message timestamp.
    pub timestamp: SystemTime,
    /// Target capability that produced the result.
    pub target: McpCapabilityTarget,
    /// Rendered result payload text reused in prompt history.
    pub result_summary: String,
    /// Error text when the call failed before a result could be decoded.
    pub error: Option<String>,
}

/// Placeholder message shape for local tool calls.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LocalToolCallMessageRecord {
    /// Stable identifier for the message record.
    pub message_id: MessageId,
    /// Message timestamp.
    pub timestamp: SystemTime,
    /// Local tool name.
    pub tool_name: LocalToolName,
    /// JSON arguments forwarded to the local tool.
    pub arguments: Value,
}

/// Placeholder message shape for local tool results.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LocalToolResultMessageRecord {
    /// Stable identifier for the message record.
    pub message_id: MessageId,
    /// Message timestamp.
    pub timestamp: SystemTime,
    /// Local tool name.
    pub tool_name: LocalToolName,
    /// Tool execution status.
    pub status: String,
    /// Human-readable payload summary reused in prompt history.
    pub result_summary: String,
    /// Error text when the tool failed.
    pub error: Option<String>,
}

/// Typed sub-agent invocation record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SubAgentCallMessageRecord {
    pub message_id: MessageId,
    pub timestamp: SystemTime,
    pub subagent_type: String,
    pub goal: String,
    pub target: DelegationTarget,
}

/// Typed sub-agent result record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SubAgentResultMessageRecord {
    pub message_id: MessageId,
    pub timestamp: SystemTime,
    pub subagent_type: String,
    pub status: String,
    pub executed_action_count: u32,
    pub detail: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_mask: Option<ToolMaskPlan>,
}

/// Placeholder message used by inactive capability families.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlaceholderMessageRecord {
    /// Stable identifier for the message record.
    pub message_id: MessageId,
    /// Message timestamp.
    pub timestamp: SystemTime,
    /// Human-readable placeholder label.
    pub label: String,
}

/// Validation failures for strongly typed state inputs.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum StateValidationError {
    #[error("server name cannot be blank")]
    BlankServerName,
    #[error("local tool name cannot be blank")]
    BlankLocalToolName,
}

#[cfg(test)]
mod tests {
    //! Unit tests for runtime state validation and prompt-facing summaries.

    use std::time::SystemTime;

    use serde_json::json;

    use crate::ids::MessageId;

    use super::{
        LlmMessageRecord, LocalToolCallMessageRecord, LocalToolName, McpCallMessageRecord,
        McpCapabilityTarget, MessageRecord, ServerName, StateValidationError, UsageSummary,
        UserMessageRecord, format_system_time,
    };

    #[test]
    fn rejects_blank_strongly_typed_names() {
        assert_eq!(
            ServerName::new("  ").expect_err("blank server names must fail"),
            StateValidationError::BlankServerName
        );
        assert_eq!(
            LocalToolName::new("  ").expect_err("blank local tool names must fail"),
            StateValidationError::BlankLocalToolName
        );
    }

    #[test]
    fn message_summaries_keep_mcp_and_local_tools_distinct() {
        let mcp_message = MessageRecord::McpCall(McpCallMessageRecord {
            message_id: MessageId::new(),
            timestamp: SystemTime::now(),
            target: McpCapabilityTarget {
                server_name: ServerName::new("sqlite").expect("valid server"),
                capability_kind: mcp_metadata::CapabilityKind::Tool,
                capability_id: "run-sql".to_owned(),
            },
            arguments: json!({"query": "select 1"}),
        });
        let local_tool_message = MessageRecord::LocalToolCall(LocalToolCallMessageRecord {
            message_id: MessageId::new(),
            timestamp: SystemTime::now(),
            tool_name: LocalToolName::new("read_file").expect("valid tool"),
            arguments: json!({"path": "README.md"}),
        });

        assert!(mcp_message.summary_line().contains("mcp_call"));
        assert!(local_tool_message.summary_line().contains("tool_call"));
        assert!(!local_tool_message.summary_line().contains("mcp_call"));
    }

    #[test]
    fn user_and_llm_message_summaries_include_timestamps() {
        let timestamp = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000);
        let expected = format_system_time(timestamp);
        let user_message = MessageRecord::User(UserMessageRecord {
            message_id: MessageId::new(),
            timestamp,
            content: "hello".to_owned(),
        });
        let llm_message = MessageRecord::Llm(LlmMessageRecord {
            message_id: MessageId::new(),
            timestamp,
            content: "world".to_owned(),
        });

        assert!(user_message.summary_line().contains(&expected));
        assert!(user_message.summary_line().contains("hello"));
        assert!(llm_message.summary_line().contains(&expected));
        assert!(llm_message.summary_line().contains("world"));
    }

    #[test]
    fn usage_summary_deserializes_missing_cached_tokens_as_zero() {
        let usage: UsageSummary = serde_json::from_value(json!({
            "input_tokens": 12,
            "output_tokens": 3,
            "total_tokens": 15
        }))
        .expect("legacy usage should parse");

        assert_eq!(
            usage,
            UsageSummary {
                input_tokens: 12,
                cached_tokens: 0,
                output_tokens: 3,
                total_tokens: 15,
            }
        );
    }

    #[test]
    fn usage_summary_add_assign_accumulates_cached_tokens() {
        let mut total = UsageSummary {
            input_tokens: 10,
            cached_tokens: 4,
            output_tokens: 2,
            total_tokens: 12,
        };

        total.add_assign(UsageSummary {
            input_tokens: 5,
            cached_tokens: 3,
            output_tokens: 1,
            total_tokens: 6,
        });

        assert_eq!(
            total,
            UsageSummary {
                input_tokens: 15,
                cached_tokens: 7,
                output_tokens: 3,
                total_tokens: 18,
            }
        );
    }
}
