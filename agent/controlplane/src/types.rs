//! Channel-facing control-plane types.
//!
//! These records sit above the single-turn runtime and own session identity,
//! message persistence, approvals, and channel bindings.

use std::time::SystemTime;

use agent_runtime::{TerminationReason, UsageSummary};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Unique identifier for one channel session.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SessionId(Uuid);

impl SessionId {
    /// Creates a new session identifier.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Returns the inner UUID for this session identifier.
    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl Default for SessionId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// Unique identifier for one persisted session message.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SessionMessageId(Uuid);

impl SessionMessageId {
    /// Creates a new session message identifier.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Returns the inner UUID for this session message identifier.
    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl Default for SessionMessageId {
    fn default() -> Self {
        Self::new()
    }
}

/// Unique identifier for one approval request.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ApprovalId(Uuid);

impl ApprovalId {
    /// Creates a new approval identifier.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Returns the inner UUID for this approval identifier.
    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl Default for ApprovalId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ApprovalId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// External channel types supported by the control plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChannelKind {
    Api,
    Cli,
    Slack,
    WhatsApp,
}

/// Stable binding between an external channel thread and one internal session.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChannelBinding {
    /// External channel family.
    pub channel: ChannelKind,
    /// External workspace/account identifier when the channel scopes ids by account.
    pub external_workspace_id: Option<String>,
    /// External conversation/thread identifier.
    pub external_conversation_id: String,
    /// External channel/container identifier when distinct from the conversation id.
    pub external_channel_id: Option<String>,
    /// External thread identifier when the channel supports threaded replies.
    pub external_thread_id: Option<String>,
    /// External end-user identifier.
    pub external_user_id: String,
}

/// Session lifecycle states managed above the runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionStatus {
    /// No turn is currently running.
    Idle,
    /// A turn is actively executing.
    Running,
    /// Execution is paused pending an approval decision.
    WaitingForApproval,
    /// Session execution was interrupted by an external command.
    Interrupted,
    /// The last turn failed.
    Failed,
    /// The last turn completed successfully.
    Completed,
}

/// Message roles persisted in session history.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionMessageRole {
    /// User-authored session message.
    User,
    /// Assistant-authored session message.
    Assistant,
    /// System-authored session note.
    System,
}

/// One persisted message in a session transcript.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SessionMessage {
    /// Stable identifier for the message.
    pub message_id: SessionMessageId,
    /// Session that owns the message.
    pub session_id: SessionId,
    /// Speaker role for the message.
    pub role: SessionMessageRole,
    /// Message text persisted in the transcript.
    pub content: String,
    /// When the message was persisted.
    pub created_at: SystemTime,
    /// Originating channel when applicable.
    pub channel: Option<ChannelKind>,
    /// Originating external message id when applicable.
    pub external_message_id: Option<String>,
}

/// One approval request waiting for a human decision.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ApprovalRequestRecord {
    /// Stable identifier for the approval request.
    pub approval_id: ApprovalId,
    /// Session waiting on the approval.
    pub session_id: SessionId,
    /// Text shown to the approver.
    pub prompt: String,
    /// Current approval state.
    pub state: ApprovalState,
    /// Creation timestamp.
    pub created_at: SystemTime,
    /// Resolution timestamp when the approval leaves the pending state.
    pub resolved_at: Option<SystemTime>,
}

/// Approval state machine values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalState {
    /// Waiting for a human decision.
    Pending,
    /// Approved by a human.
    Approved,
    /// Rejected by a human.
    Rejected,
    /// Expired without a decision.
    Expired,
    /// Cancelled because the session moved on.
    Cancelled,
}

/// Decision submitted against a pending approval request.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalDecision {
    /// Continue execution.
    Approve,
    /// Deny the requested action.
    Reject,
}

/// Minimal stored summary for one runtime turn.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TurnRecordSummary {
    /// One-based turn number inside the session.
    pub turn_number: u32,
    /// Model name used for the turn.
    pub model_name: String,
    /// Turn duration in milliseconds.
    pub elapsed_ms: u64,
    /// Final assistant text returned by the turn.
    pub final_text: String,
    /// Why the turn ended.
    pub termination: TerminationReason,
    /// Aggregate token usage reported by the runtime.
    pub usage: UsageSummary,
    /// Completion timestamp.
    pub completed_at: SystemTime,
}

/// Top-level session record stored by the control plane.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SessionRecord {
    /// Stable session identifier.
    pub session_id: SessionId,
    /// Session creation timestamp.
    pub created_at: SystemTime,
    /// Last mutation timestamp.
    pub updated_at: SystemTime,
    /// Current session lifecycle state.
    pub status: SessionStatus,
    /// Known external bindings attached to the session.
    pub bindings: Vec<ChannelBinding>,
    /// Summary of the last completed turn, if any.
    pub last_turn: Option<TurnRecordSummary>,
}

/// Normalized inbound channel envelope.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChannelEnvelope {
    /// Originating channel family.
    pub channel: ChannelKind,
    /// External workspace/account identifier when the channel scopes ids by account.
    pub external_workspace_id: Option<String>,
    /// External conversation/thread identifier.
    pub external_conversation_id: String,
    /// External channel/container identifier when distinct from the conversation id.
    pub external_channel_id: Option<String>,
    /// External thread identifier when the channel supports threaded replies.
    pub external_thread_id: Option<String>,
    /// External end-user identifier.
    pub external_user_id: String,
    /// External message identifier when the channel supplies one.
    pub external_message_id: Option<String>,
    /// Idempotency key used to deduplicate delivery.
    pub idempotency_key: String,
    /// Time the external event occurred or was ingested.
    pub occurred_at: SystemTime,
    /// Normalized channel intent.
    pub intent: ChannelIntent,
}

/// Channel-neutral intent extracted from inbound traffic.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChannelIntent {
    /// User-authored free text.
    UserText { text: String },
    /// Reset the active session bound to the current channel route.
    ResetSession,
    /// Human response to a prior approval prompt.
    ApprovalResponse {
        approval_id: ApprovalId,
        decision: ApprovalDecision,
    },
    /// Interrupt the current turn.
    Interrupt,
    /// Resume a previously interrupted turn.
    Resume,
    /// Request a status update without adding user text.
    StatusRequest,
}

/// Outbound message shape before channel-specific rendering.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChannelResponseEnvelope {
    /// Session that produced the outbound message.
    pub session_id: SessionId,
    /// High-level outbound message category.
    pub kind: OutboundMessageKind,
    /// Channel-neutral text payload.
    pub text: String,
    /// Optional delivery target used by channel-specific connectors.
    pub delivery_target: Option<ChannelDeliveryTarget>,
}

/// Delivery address used by connectors to send one outbound message.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelDeliveryTarget {
    /// Channel family that should receive the message.
    pub channel: ChannelKind,
    /// External workspace/account identifier when present.
    pub external_workspace_id: Option<String>,
    /// External conversation/thread identifier.
    pub external_conversation_id: String,
    /// External channel/container identifier when distinct from the conversation id.
    pub external_channel_id: Option<String>,
    /// External thread identifier when present.
    pub external_thread_id: Option<String>,
    /// External end-user identifier for rendering or direct addressing.
    pub external_user_id: String,
}

/// Kind of outbound message emitted to a channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutboundMessageKind {
    /// Normal assistant reply.
    Reply,
    /// Session status update.
    Status,
    /// Approval request prompt.
    ApprovalPrompt,
    /// Error message intended for the channel.
    Error,
}

/// Result returned after one inbound envelope is processed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChannelDispatchResult {
    /// Updated session record after processing the inbound envelope.
    pub session: SessionRecord,
    /// Outbound messages the caller should attempt to deliver.
    pub outbound: Vec<ChannelResponseEnvelope>,
    /// Approval request created by this dispatch, if any.
    pub approval: Option<ApprovalRequestRecord>,
    /// Whether the inbound envelope was ignored as a duplicate.
    pub was_duplicate: bool,
}

/// Delivery outcome for one adapter send attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryStatus {
    /// Delivery has been queued or attempted but not confirmed.
    Pending,
    /// Delivery succeeded.
    Sent,
    /// Delivery failed.
    Failed,
}

/// Session-scoped event stream exposed to APIs and adapters.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionEvent {
    SessionCreated {
        /// Newly created session record.
        session: SessionRecord,
    },
    SessionMessageReceived {
        /// Session that received the message.
        session_id: SessionId,
        /// Newly persisted session message.
        message: SessionMessage,
    },
    TurnQueued {
        /// Session whose turn was queued.
        session_id: SessionId,
    },
    TurnStarted {
        /// Session whose turn began.
        session_id: SessionId,
    },
    TurnCompleted {
        /// Session whose turn completed.
        session_id: SessionId,
        /// Summary of the completed turn.
        summary: TurnRecordSummary,
    },
    ApprovalRequested {
        /// Session that is now waiting for approval.
        session_id: SessionId,
        /// Newly created approval request.
        approval: ApprovalRequestRecord,
    },
    ApprovalResolved {
        /// Session that was waiting for approval.
        session_id: SessionId,
        /// Resolved approval record.
        approval: ApprovalRequestRecord,
    },
    RuntimeEvent {
        /// Session that emitted the runtime event.
        session_id: SessionId,
        /// Raw runtime event forwarded from the runner.
        event: agent_runtime::RuntimeEvent,
    },
    ChannelDeliveryAttempted {
        /// Session whose outbound message was delivered.
        session_id: SessionId,
        /// Channel targeted by the delivery attempt.
        channel: ChannelKind,
        /// Delivery result.
        status: DeliveryStatus,
    },
}

/// Slack webhook payload normalized by the Slack adapter.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SlackWebhookPayload {
    /// Stable Slack event identifier used for idempotency.
    pub event_id: String,
    /// Slack channel id.
    pub channel_id: String,
    /// Slack thread timestamp when the event belongs to a thread.
    pub thread_ts: Option<String>,
    /// Slack user id.
    pub user_id: String,
    /// Slack message text.
    pub text: String,
}

/// WhatsApp webhook payload normalized by the WhatsApp adapter.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WhatsAppWebhookPayload {
    /// Stable inbound message identifier.
    pub message_id: String,
    /// External conversation identifier.
    pub conversation_id: String,
    /// External user identifier.
    pub from_user_id: String,
    /// Incoming message text.
    pub text: String,
}

/// Delivery access policy for inbound WhatsApp direct messages.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WhatsAppDmPolicy {
    /// Only explicitly allowed senders may open sessions.
    Allowlist,
    /// Any sender may open a session.
    Open,
}

/// Connection state for the WhatsApp gateway worker.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WhatsAppGatewayConnectionState {
    /// Connector is configured but not paired.
    NeedsLogin,
    /// Connector is paired and ready to send or receive.
    Ready,
}

/// Snapshot of the WhatsApp gateway status exposed over HTTP and CLI.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WhatsAppGatewayStatus {
    /// Logical account identifier used for routing.
    pub account_id: String,
    /// Current connection state.
    pub connection_state: WhatsAppGatewayConnectionState,
    /// Access policy for new direct messages.
    pub dm_policy: WhatsAppDmPolicy,
    /// Explicitly allowed senders when the policy requires it.
    pub allow_from: Vec<String>,
    /// Number of queued outbound deliveries waiting on the worker.
    pub pending_outbound: usize,
    /// Active login session identifier when a QR is pending.
    pub active_login_session_id: Option<String>,
    /// Human-readable error from the last send attempt, if any.
    pub last_error: Option<String>,
}

/// Response returned when a WhatsApp login session starts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StartWhatsAppLoginResponse {
    /// Logical account identifier used for routing.
    pub account_id: String,
    /// Stable login session identifier expected by login completion.
    pub login_session_id: String,
    /// Opaque QR payload that a real Web transport would surface to the user.
    pub qr_code: String,
}

/// Request body used to complete a pending WhatsApp login session.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompleteWhatsAppLoginRequest {
    /// Login session created by the start-login endpoint.
    pub login_session_id: String,
}

/// Normalized inbound WhatsApp message sent by the gateway worker.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReceiveWhatsAppMessageRequest {
    /// Stable inbound message identifier.
    pub message_id: String,
    /// Logical account identifier used for routing.
    pub account_id: Option<String>,
    /// Stable conversation identifier for the peer chat.
    pub conversation_id: String,
    /// Sender identifier, usually a phone number or WhatsApp jid.
    pub from_user_id: String,
    /// Incoming message text.
    pub text: String,
    /// Optional quoted message identifier when the inbound message is a reply.
    pub quoted_message_id: Option<String>,
    /// Optional quoted message text appended into the normalized prompt.
    pub quoted_text: Option<String>,
}

/// Response returned after the gateway accepts one inbound WhatsApp message.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReceiveWhatsAppMessageResponse {
    /// Updated session after dispatch.
    pub session: SessionRecord,
    /// Number of outbound messages queued for background delivery.
    pub queued_outbound: usize,
    /// Whether the inbound message was ignored as a duplicate.
    pub was_duplicate: bool,
}

/// Aggregate response returned after a provider-native WhatsApp webhook is processed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WhatsAppWebhookDispatchResponse {
    /// Number of inbound WhatsApp user messages processed from the webhook.
    pub processed_messages: usize,
    /// Total number of outbound replies queued for delivery.
    pub queued_outbound: usize,
    /// Number of inbound messages ignored as duplicates.
    pub duplicate_messages: usize,
}

/// API request body for creating a session.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CreateSessionRequest {
    /// Optional initial channel binding kind.
    pub channel: Option<ChannelKind>,
    /// Optional initial external conversation id.
    pub external_conversation_id: Option<String>,
    /// Optional initial external user id.
    pub external_user_id: Option<String>,
}

/// API request body for sending a session message.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SendSessionMessageRequest {
    /// User text to append and execute.
    pub text: String,
    /// Optional caller-supplied idempotency key.
    pub idempotency_key: Option<String>,
    /// Optional response format override for API callers.
    pub response_format: Option<ApiResponseFormat>,
}

/// API request body for submitting an approval decision.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SubmitApprovalRequest {
    /// Human decision applied to the approval request.
    pub decision: ApprovalDecision,
}

/// API response after processing a session message.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SendSessionMessageResponse {
    /// Combined session/outbound result returned after processing the message.
    pub result: ChannelDispatchResult,
}

/// API-selectable response formatting profiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApiResponseFormat {
    PlainText,
    Markdown,
}
