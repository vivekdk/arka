//! OpenAI-backed implementation of the runtime model adapter.
//!
//! This crate intentionally isolates all provider-specific HTTP details from
//! the core runtime. The runtime depends only on the `ModelAdapter` trait,
//! which keeps unit tests deterministic and avoids leaking OpenAI wire types
//! into the rest of the repo.
//!
//! # Request/response flow
//!
//! `OpenAiModelAdapter` translates one runtime `ModelStepRequest` into a
//! single call to the OpenAI Responses API. The adapter:
//!
//! 1. Builds a request body whose `input` is the fully rendered runtime prompt.
//! 2. Attaches a strict JSON schema so the model must emit a
//!    `ModelStepDecision`-compatible payload.
//! 3. Sends the request with bearer authentication.
//! 4. Normalizes the provider response back into the provider-agnostic
//!    `ModelAdapterResponse` used by the runtime loop.
//!
//! The runtime therefore never needs to know about OpenAI-specific response
//! envelopes, error formats, or output-content nesting rules.

use agent_runtime::{
    model::{
        FinalAnswerRenderRequest, FinalAnswerRenderResponse, FinalAnswerStreamSink, ModelAdapter,
        ModelAdapterArtifact, ModelAdapterArtifactKind, ModelAdapterDebugSink, ModelAdapterError,
        ModelAdapterResponse, ModelStepDecision, ModelStepRequest, SubagentAdapterResponse,
        SubagentDecision, SubagentDelegationRequest, SubagentStepRequest,
    },
    policy::ToolMaskPlan,
    state::{DelegationTarget, ResponseClient, ResponseFormat, UsageSummary},
    tools::{ToolDescriptor, ToolFamily},
};
use async_trait::async_trait;
use reqwest::StatusCode;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::Deserialize;
use serde_json::{Value, json};
use thiserror::Error;

const RESPONSES_ENDPOINT: &str = "https://api.openai.com/v1/responses";

#[derive(Debug)]
/// OpenAI-backed implementation of the runtime model adapter.
///
/// The adapter owns a reusable `reqwest::Client`, the API key used to
/// authenticate requests, and the target endpoint. In production the endpoint
/// is the public OpenAI Responses API, while tests can substitute a custom
/// base URL to exercise request construction and error handling without
/// reaching the network.
pub struct OpenAiModelAdapter {
    /// Reusable HTTP client for Responses API calls.
    http_client: reqwest::Client,
    /// Bearer token used for authorization.
    api_key: String,
    /// Target Responses API endpoint, overridable in tests.
    base_url: String,
}

impl OpenAiModelAdapter {
    /// Creates an adapter that targets the public OpenAI Responses API.
    ///
    /// This is the normal constructor used by application code. It validates
    /// that the API key is not blank and then delegates to
    /// [`OpenAiModelAdapter::with_base_url`].
    pub fn new(api_key: impl Into<String>) -> Result<Self, OpenAiAdapterError> {
        Self::with_base_url(api_key, RESPONSES_ENDPOINT)
    }

    /// Creates an adapter with a custom base URL, primarily for tests.
    ///
    /// A custom base URL is useful when wiring the adapter against a mock
    /// server or a local fixture. The adapter still behaves identically with
    /// respect to header generation and payload shape.
    pub fn with_base_url(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Result<Self, OpenAiAdapterError> {
        let api_key = api_key.into();
        if api_key.trim().is_empty() {
            return Err(OpenAiAdapterError::MissingApiKey);
        }

        Ok(Self {
            http_client: reqwest::Client::new(),
            api_key,
            base_url: base_url.into(),
        })
    }

    /// Builds the HTTP headers required by the Responses API.
    ///
    /// The adapter always sends bearer authentication and JSON content type.
    /// Header construction can fail only if the supplied API key would produce
    /// an invalid header value.
    fn headers(&self) -> Result<HeaderMap, OpenAiAdapterError> {
        let mut headers = HeaderMap::new();
        let bearer = format!("Bearer {}", self.api_key);
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&bearer).map_err(|error| {
                OpenAiAdapterError::InvalidHeader(format!("authorization header: {error}"))
            })?,
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        Ok(headers)
    }

    /// Converts a provider-neutral runtime step request into an OpenAI
    /// Responses API payload.
    ///
    /// The prompt text is passed through verbatim as `input`. The important
    /// provider-specific behavior here is the `json_schema` response format:
    /// it forces the model to emit a decision payload that can be losslessly
    /// converted into `ModelStepDecision`.
    fn build_request_body(&self, request: &ModelStepRequest) -> Value {
        json!({
            "model": request.model_config.model_name,
            "input": request.prompt.rendered,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "model_step_decision",
                    "schema": decision_schema(),
                    "strict": true
                }
            },
            "metadata": {
                "step_number": request.step_number.to_string()
            }
        })
    }

    fn build_subagent_request_body(&self, request: &SubagentStepRequest) -> Value {
        json!({
            "model": request.model_config.model_name,
            "input": request.prompt,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "subagent_step_decision",
                    "schema": subagent_decision_schema(
                        &request.tool_mask_plan,
                        &request.registered_tools,
                    ),
                    "strict": true
                }
            },
            "metadata": {
                "subagent_type": request.subagent_type
            }
        })
    }

    fn build_final_answer_render_body(&self, request: &FinalAnswerRenderRequest) -> Value {
        json!({
            "model": request.model_config.model_name,
            "input": request.prompt,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "final_answer_render",
                    "schema": final_answer_render_schema(),
                    "strict": true
                }
            },
            "metadata": {
                "response_client": response_client_name(request.response_target.client),
                "response_format": response_format_name(request.response_target.format)
            }
        })
    }

    async fn execute_responses_call(
        &self,
        body: Value,
        request_summary: &'static str,
        response_summary: &'static str,
        debug_sink: Option<&mut dyn ModelAdapterDebugSink>,
    ) -> Result<ResponsesCreateResponse, ModelAdapterError> {
        let mut debug = OpenAiDebugEmitter { sink: debug_sink };
        debug.emit(
            ModelAdapterArtifactKind::Request,
            request_summary,
            redact_json_value(&body),
        );

        let headers = self
            .headers()
            .map_err(|error| ModelAdapterError::Provider(error.to_string()))?;
        let response = self
            .http_client
            .post(&self.base_url)
            .headers(headers)
            .json(&body)
            .send()
            .await
            .map_err(|error| {
                debug.emit(
                    ModelAdapterArtifactKind::Error,
                    "transport_error",
                    json!({
                        "message": error.to_string(),
                    }),
                );
                ModelAdapterError::Transport(error.to_string())
            })?;

        let status = response.status();
        let response_text = response.text().await.map_err(|error| {
            debug.emit(
                ModelAdapterArtifactKind::Error,
                "response_body_read_error",
                json!({
                    "message": error.to_string(),
                }),
            );
            ModelAdapterError::Transport(error.to_string())
        })?;

        if !status.is_success() {
            debug.emit(
                ModelAdapterArtifactKind::Error,
                &format!("provider error {status}"),
                json!({
                    "status": status.as_u16(),
                    "body": parse_json_or_text(&response_text),
                }),
            );
            return Err(ModelAdapterError::Provider(format_provider_error(
                status,
                &response_text,
            )));
        }

        let raw_payload = serde_json::from_str::<Value>(&response_text).map_err(|error| {
            debug.emit(
                ModelAdapterArtifactKind::Error,
                "response_json_parse_error",
                json!({
                    "message": error.to_string(),
                    "body": parse_json_or_text(&response_text),
                }),
            );
            ModelAdapterError::Transport(error.to_string())
        })?;
        debug.emit(
            ModelAdapterArtifactKind::Response,
            response_summary,
            redact_json_value(&raw_payload),
        );

        serde_json::from_value(raw_payload)
            .map_err(|error| ModelAdapterError::Transport(error.to_string()))
    }
}

#[async_trait]
impl ModelAdapter for OpenAiModelAdapter {
    async fn generate_step(
        &self,
        request: ModelStepRequest,
        debug_sink: Option<&mut dyn ModelAdapterDebugSink>,
    ) -> Result<ModelAdapterResponse, ModelAdapterError> {
        // Convert the runtime request into an OpenAI-specific HTTP payload.
        let body = self.build_request_body(&request);
        let payload = self
            .execute_responses_call(
                body,
                "model step request",
                "model step response",
                debug_sink,
            )
            .await?;

        // The Responses API can surface generated text in multiple places.
        // Normalize those variants into one decision string before parsing it.
        let decision_text = payload.extract_output_text()?;
        let decision = parse_decision_text(&decision_text)?;

        Ok(ModelAdapterResponse {
            decision,
            usage: payload.usage.map(UsageSummary::from).unwrap_or_default(),
        })
    }

    async fn generate_subagent_step(
        &self,
        request: SubagentStepRequest,
        debug_sink: Option<&mut dyn ModelAdapterDebugSink>,
    ) -> Result<SubagentAdapterResponse, ModelAdapterError> {
        let body = self.build_subagent_request_body(&request);
        let payload = self
            .execute_responses_call(body, "subagent request", "subagent response", debug_sink)
            .await?;
        let decision_text = payload.extract_output_text()?;
        let decision = parse_subagent_decision_text(&decision_text)?;
        Ok(SubagentAdapterResponse {
            decision,
            usage: payload.usage.map(UsageSummary::from).unwrap_or_default(),
        })
    }

    async fn render_final_answer(
        &self,
        request: FinalAnswerRenderRequest,
        mut stream_sink: Option<&mut dyn FinalAnswerStreamSink>,
        debug_sink: Option<&mut dyn ModelAdapterDebugSink>,
    ) -> Result<FinalAnswerRenderResponse, ModelAdapterError> {
        let body = self.build_final_answer_render_body(&request);
        let payload = self
            .execute_responses_call(
                body,
                "final answer render request",
                "final answer render response",
                debug_sink,
            )
            .await?;
        let response_text = payload.extract_output_text()?;
        let parsed = parse_final_answer_render_text(&response_text)?;
        let canonical_text = if parsed.canonical_text.trim().is_empty() {
            request.answer_brief.clone()
        } else {
            parsed.canonical_text
        };
        let display_text = if parsed.display_text.trim().is_empty() {
            canonical_text.clone()
        } else {
            parsed.display_text
        };

        if let Some(sink) = stream_sink.as_deref_mut() {
            sink.record_text_delta(&display_text);
        }

        Ok(FinalAnswerRenderResponse {
            canonical_text,
            display_text,
            usage: payload.usage.map(UsageSummary::from).unwrap_or_default(),
        })
    }
}

/// Formats a provider error into a stable, human-readable runtime message.
///
/// When the body matches OpenAI's standard error envelope, the formatter
/// extracts the structured fields that are most helpful during debugging. If
/// the body is not JSON, it falls back to the full raw body so transport
/// callers still get actionable context.
fn format_provider_error(status: StatusCode, body: &str) -> String {
    let mut message = format!("OpenAI Responses API returned status {status}");
    let body = body.trim();

    if body.is_empty() {
        return message;
    }

    if let Ok(payload) = serde_json::from_str::<ResponsesErrorEnvelope>(body) {
        if let Some(error) = payload.error {
            message.push_str(": ");
            message.push_str(&error.message);
            if let Some(error_type) = error.error_type {
                message.push_str(" [type=");
                message.push_str(&error_type);
                message.push(']');
            }
            if let Some(param) = error.param {
                message.push_str(" [param=");
                message.push_str(&param);
                message.push(']');
            }
            if let Some(code) = error.code {
                message.push_str(" [code=");
                message.push_str(&code);
                message.push(']');
            }
            return message;
        }
    }

    message.push_str(": ");
    message.push_str(body);
    message
}

struct OpenAiDebugEmitter<'a> {
    sink: Option<&'a mut dyn ModelAdapterDebugSink>,
}

impl OpenAiDebugEmitter<'_> {
    fn emit(&mut self, kind: ModelAdapterArtifactKind, summary: &str, payload: Value) {
        if let Some(sink) = self.sink.as_deref_mut() {
            sink.record_model_artifact(ModelAdapterArtifact {
                kind,
                source: "openai_responses".to_owned(),
                summary: Some(summary.to_owned()),
                payload,
            });
        }
    }
}

fn parse_json_or_text(body: &str) -> Value {
    serde_json::from_str(body).unwrap_or_else(|_| Value::String(body.to_owned()))
}

fn redact_json_value(value: &Value) -> Value {
    match value {
        Value::Array(items) => Value::Array(items.iter().map(redact_json_value).collect()),
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(key, value)| {
                    if is_sensitive_key(key) {
                        (key.clone(), Value::String("[REDACTED]".to_owned()))
                    } else {
                        (key.clone(), redact_json_value(value))
                    }
                })
                .collect(),
        ),
        _ => value.clone(),
    }
}

fn is_sensitive_key(key: &str) -> bool {
    let lower = key.to_ascii_lowercase();
    lower.contains("authorization")
        || lower.contains("api_key")
        || lower == "apikey"
        || lower.contains("token")
        || lower.contains("password")
        || lower.contains("secret")
}

/// Parses the model-emitted decision JSON into the runtime's strongly typed
/// decision enum.
///
/// The wire shape is slightly different from the runtime shape:
/// sub-agent tool arguments are accepted as structured JSON, with temporary
/// fallback support for the older `arguments_json` string field. This function
/// normalizes those variants before constructing `SubagentDecision`.
fn parse_decision_text(decision_text: &str) -> Result<ModelStepDecision, ModelAdapterError> {
    let wire: OpenAiDecisionWire = serde_json::from_str(decision_text)
        .map_err(|error| ModelAdapterError::InvalidDecision(error.to_string()))?;

    match wire.decision_type.as_str() {
        "final" => Ok(ModelStepDecision::Final {
            content: wire.content,
        }),
        "delegate_subagent" => Ok(ModelStepDecision::DelegateSubagent {
            delegation: SubagentDelegationRequest {
                subagent_type: wire.subagent_type,
                target: wire.target.ok_or_else(|| {
                    ModelAdapterError::InvalidDecision(
                        "delegate_subagent requires target".to_owned(),
                    )
                })?,
                goal: wire.goal,
            },
        }),
        other => Err(ModelAdapterError::InvalidDecision(format!(
            "unsupported decision type `{other}`"
        ))),
    }
}

fn parse_subagent_decision_text(
    decision_text: &str,
) -> Result<SubagentDecision, ModelAdapterError> {
    let wire: OpenAiSubagentDecisionWire = serde_json::from_str(decision_text)
        .map_err(|error| ModelAdapterError::InvalidDecision(error.to_string()))?;
    match wire.decision_type.as_str() {
        "mcp_tool_call" => {
            let arguments = parse_subagent_arguments(&wire, "mcp_tool_call")?;
            Ok(SubagentDecision::McpToolCall {
                server_name: wire.server_name.ok_or_else(|| {
                    ModelAdapterError::InvalidDecision(
                        "mcp_tool_call requires server_name".to_owned(),
                    )
                })?,
                tool_name: wire.tool_name.unwrap_or_default(),
                arguments,
            })
        }
        "mcp_resource_read" => Ok(SubagentDecision::McpResourceRead {
            server_name: wire.server_name.ok_or_else(|| {
                ModelAdapterError::InvalidDecision(
                    "mcp_resource_read requires server_name".to_owned(),
                )
            })?,
            resource_uri: wire.resource_uri.unwrap_or_default(),
        }),
        "local_tool_call" => {
            let arguments = parse_subagent_arguments(&wire, "local_tool_call")?;
            Ok(SubagentDecision::LocalToolCall {
                tool_name: agent_runtime::state::LocalToolName::new(
                    wire.tool_name.unwrap_or_default(),
                )
                .map_err(|error| ModelAdapterError::InvalidDecision(error.to_string()))?,
                arguments,
            })
        }
        "done" => Ok(SubagentDecision::Done {
            summary: wire.summary.unwrap_or_default(),
        }),
        "partial" => Ok(SubagentDecision::Partial {
            summary: wire.summary.unwrap_or_default(),
            reason: wire.reason.unwrap_or_default(),
        }),
        "cannot_execute" => Ok(SubagentDecision::CannotExecute {
            reason: wire.reason.unwrap_or_default(),
        }),
        other => Err(ModelAdapterError::InvalidDecision(format!(
            "unsupported subagent decision type `{other}`"
        ))),
    }
}

fn parse_subagent_arguments(
    wire: &OpenAiSubagentDecisionWire,
    decision_type: &str,
) -> Result<Value, ModelAdapterError> {
    if let Some(arguments) = &wire.arguments {
        return Ok(prune_null_json(arguments.clone()));
    }

    if let Some(arguments_json) = &wire.arguments_json {
        return serde_json::from_str(arguments_json)
            .map(prune_null_json)
            .map_err(|error| {
                ModelAdapterError::InvalidDecision(format!(
                    "invalid {decision_type} arguments_json: {error}"
                ))
            });
    }

    Ok(json!({}))
}

fn prune_null_json(value: Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.into_iter()
                .filter_map(|(key, value)| {
                    let normalized = prune_null_json(value);
                    if normalized.is_null() {
                        None
                    } else {
                        Some((key, normalized))
                    }
                })
                .collect(),
        ),
        Value::Array(items) => Value::Array(
            items
                .into_iter()
                .map(prune_null_json)
                .filter(|item| !item.is_null())
                .collect(),
        ),
        other => other,
    }
}

fn parse_final_answer_render_text(
    response_text: &str,
) -> Result<OpenAiFinalAnswerRenderWire, ModelAdapterError> {
    serde_json::from_str(response_text)
        .map_err(|error| ModelAdapterError::InvalidDecision(error.to_string()))
}

/// OpenAI adapter construction errors.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum OpenAiAdapterError {
    #[error("OpenAI API key cannot be blank")]
    MissingApiKey,
    #[error("failed to construct request headers: {0}")]
    InvalidHeader(String),
}

/// Minimal subset of the Responses API success payload used by the runtime.
///
/// The runtime only needs:
/// - the generated text that encodes a structured decision, and
/// - optional usage information for observability and accounting.
///
/// OpenAI can place generated text either in the top-level `output_text` field
/// or inside nested `output` content items, so both are modeled here.
#[derive(Clone, Debug, Deserialize)]
struct ResponsesCreateResponse {
    #[serde(default)]
    output_text: Option<String>,
    #[serde(default)]
    output: Vec<Value>,
    #[serde(default)]
    usage: Option<OpenAiUsage>,
}

impl ResponsesCreateResponse {
    /// Returns the single distinct non-empty assistant text found in the response.
    ///
    /// The Responses API can repeat the same structured decision in multiple
    /// nested message phases. Those duplicates are tolerated, but multiple
    /// distinct decision texts are rejected so the runtime never silently picks
    /// an arbitrary tool action.
    ///
    /// When OpenAI emits both an actionable decision and a terminal summary in
    /// the same response, prefer the actionable one. Within the same class,
    /// prefer `phase="final_answer"` candidates over commentary candidates.
    ///
    /// If the model emits multiple actionable commentary candidates in one
    /// response, execute the first one in response order and let the runtime
    /// continue the delegated loop on the next turn.
    fn extract_output_text(&self) -> Result<String, ModelAdapterError> {
        let mut candidates = Vec::new();
        if let Some(text) = self.output_text.as_deref().and_then(non_empty_text) {
            push_unique_output_text_candidate(&mut candidates, text, None);
        }
        for item in &self.output {
            let phase = output_value_phase(item);
            collect_output_text_candidates_from_output_value(item, phase, &mut candidates);
        }

        if candidates.is_empty() {
            return Err(ModelAdapterError::InvalidDecision(
                "missing output_text and no assistant text found in output".to_owned(),
            ));
        }

        if let Some(text) = select_output_text_candidate(
            candidates
                .iter()
                .filter(|candidate| candidate.kind.is_action())
                .filter(|candidate| candidate.phase.as_deref() == Some("final_answer")),
            OutputTextSelectionMode::RequireUnique,
        )? {
            return Ok(text);
        }

        if let Some(text) = select_output_text_candidate(
            candidates
                .iter()
                .filter(|candidate| candidate.kind.is_action()),
            OutputTextSelectionMode::FirstInResponse,
        )? {
            return Ok(text);
        }

        if let Some(text) = select_output_text_candidate(
            candidates
                .iter()
                .filter(|candidate| candidate.phase.as_deref() == Some("final_answer")),
            OutputTextSelectionMode::RequireUnique,
        )? {
            return Ok(text);
        }

        select_output_text_candidate(candidates.iter(), OutputTextSelectionMode::RequireUnique)?
            .ok_or_else(|| {
                ModelAdapterError::InvalidDecision(
                    "missing output_text and no assistant text found in output".to_owned(),
                )
            })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OutputTextCandidateKind {
    Action,
    Terminal,
    Unknown,
}

impl OutputTextCandidateKind {
    fn is_action(self) -> bool {
        matches!(self, Self::Action)
    }
}

#[derive(Clone, Debug)]
struct OutputTextCandidate {
    text: String,
    phase: Option<String>,
    kind: OutputTextCandidateKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OutputTextSelectionMode {
    RequireUnique,
    FirstInResponse,
}

fn output_value_phase(value: &Value) -> Option<&str> {
    value.as_object()?.get("phase")?.as_str()
}

fn collect_output_text_candidates_from_output_value(
    value: &Value,
    phase: Option<&str>,
    candidates: &mut Vec<OutputTextCandidate>,
) {
    match value {
        Value::String(text) => {
            if let Some(text) = non_empty_text(text) {
                push_unique_output_text_candidate(candidates, text, phase);
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_output_text_candidates_from_output_value(item, phase, candidates);
            }
        }
        Value::Object(map) => {
            if let Some(content) = map.get("content") {
                collect_output_text_candidates_from_output_value(content, phase, candidates);
            }
            if let Some(text) = map.get("text") {
                collect_output_text_candidates_from_output_value(text, phase, candidates);
            }
            if let Some(value) = map.get("value") {
                collect_output_text_candidates_from_output_value(value, phase, candidates);
            }
        }
        _ => {}
    }
}

fn push_unique_output_text_candidate(
    candidates: &mut Vec<OutputTextCandidate>,
    text: &str,
    phase: Option<&str>,
) {
    if candidates.iter().any(|candidate| candidate.text == text) {
        return;
    }

    candidates.push(OutputTextCandidate {
        text: text.to_owned(),
        phase: phase.map(str::to_owned),
        kind: classify_output_text_candidate(text),
    });
}

fn classify_output_text_candidate(text: &str) -> OutputTextCandidateKind {
    let Ok(value) = serde_json::from_str::<Value>(text) else {
        return OutputTextCandidateKind::Unknown;
    };

    match value.get("type").and_then(Value::as_str) {
        Some("delegate_subagent" | "mcp_tool_call" | "mcp_resource_read" | "local_tool_call") => {
            OutputTextCandidateKind::Action
        }
        Some("final" | "done" | "partial" | "cannot_execute") => OutputTextCandidateKind::Terminal,
        _ => OutputTextCandidateKind::Unknown,
    }
}

fn select_output_text_candidate<'a>(
    candidates: impl Iterator<Item = &'a OutputTextCandidate>,
    mode: OutputTextSelectionMode,
) -> Result<Option<String>, ModelAdapterError> {
    let selected: Vec<&OutputTextCandidate> = candidates.collect();
    match selected.len() {
        0 => Ok(None),
        1 => Ok(Some(selected[0].text.clone())),
        count => match mode {
            OutputTextSelectionMode::RequireUnique => Err(ModelAdapterError::InvalidDecision(
                format!("multiple distinct output_text candidates found in response: {count}"),
            )),
            OutputTextSelectionMode::FirstInResponse => Ok(Some(selected[0].text.clone())),
        },
    }
}

#[derive(Clone, Debug, Deserialize)]
struct OpenAiUsage {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    input_tokens_details: OpenAiInputTokensDetails,
    #[serde(default)]
    output_tokens: u32,
    #[serde(default)]
    total_tokens: u32,
}

impl From<OpenAiUsage> for UsageSummary {
    fn from(value: OpenAiUsage) -> Self {
        Self {
            input_tokens: value.input_tokens,
            cached_tokens: value.input_tokens_details.cached_tokens,
            output_tokens: value.output_tokens,
            total_tokens: value.total_tokens,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
struct OpenAiInputTokensDetails {
    #[serde(default)]
    cached_tokens: u32,
}

/// Wire representation of the strict JSON decision requested from OpenAI.
///
/// This mirrors the schema produced by [`decision_schema`]. The `calls` field
/// is present for both variants so the schema can stay uniform and strict even
/// when the decision is a final answer.
#[derive(Clone, Debug, Deserialize)]
struct OpenAiDecisionWire {
    #[serde(rename = "type")]
    decision_type: String,
    content: String,
    #[serde(default)]
    subagent_type: String,
    #[serde(default)]
    goal: String,
    #[serde(default)]
    target: Option<DelegationTarget>,
}

#[derive(Clone, Debug, Deserialize)]
struct OpenAiSubagentDecisionWire {
    #[serde(rename = "type")]
    decision_type: String,
    #[serde(default)]
    server_name: Option<agent_runtime::state::ServerName>,
    #[serde(default)]
    tool_name: Option<String>,
    #[serde(default)]
    arguments: Option<Value>,
    #[serde(default)]
    arguments_json: Option<String>,
    #[serde(default)]
    resource_uri: Option<String>,
    #[serde(default)]
    summary: Option<String>,
    #[serde(default)]
    reason: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct OpenAiFinalAnswerRenderWire {
    canonical_text: String,
    display_text: String,
}

/// OpenAI error envelope returned on non-success responses.
#[derive(Clone, Debug, Deserialize)]
struct ResponsesErrorEnvelope {
    #[serde(default)]
    error: Option<ResponsesErrorBody>,
}

/// Structured contents of the `error` member returned by OpenAI.
#[derive(Clone, Debug, Deserialize)]
struct ResponsesErrorBody {
    message: String,
    #[serde(rename = "type", default)]
    error_type: Option<String>,
    #[serde(default)]
    param: Option<String>,
    #[serde(default)]
    code: Option<String>,
}

/// Returns the strict JSON schema enforced for every model step response.
///
/// The schema is intentionally narrow:
/// - only two decision types are allowed,
/// - all fields are required, and
/// - additional properties are forbidden.
///
/// This pushes output-shape enforcement onto the provider so the runtime can
/// reject malformed generations early and deterministically.
fn decision_schema() -> Value {
    json!({
        "type": "object",
        "required": ["type", "content", "subagent_type", "goal", "target"],
        "properties": {
            "type": {
                "type": "string",
                "enum": ["final", "delegate_subagent"]
            },
            "content": { "type": "string" },
            "subagent_type": { "type": "string" },
            "goal": { "type": "string" },
            "target": {
                "anyOf": [
                    { "type": "null" },
                    {
                        "type": "object",
                        "required": ["kind", "value"],
                        "properties": {
                            "kind": { "type": "string", "enum": ["mcp_capability"] },
                            "value": {
                                "type": "object",
                                "required": ["server_name", "capability_kind", "capability_id"],
                                "properties": {
                                    "server_name": { "type": "string", "minLength": 1 },
                                    "capability_kind": { "type": "string", "enum": ["tool", "resource"] },
                                    "capability_id": { "type": "string", "minLength": 1 }
                                },
                                "additionalProperties": false
                            }
                        },
                        "additionalProperties": false
                    },
                    {
                        "type": "object",
                        "required": ["kind", "value"],
                        "properties": {
                            "kind": { "type": "string", "enum": ["mcp_server_scope"] },
                            "value": {
                                "type": "object",
                                "required": ["server_name"],
                                "properties": {
                                    "server_name": { "type": "string", "minLength": 1 }
                                },
                                "additionalProperties": false
                            }
                        },
                        "additionalProperties": false
                    },
                    {
                        "type": "object",
                        "required": ["kind", "value"],
                        "properties": {
                            "kind": { "type": "string", "enum": ["local_tools_scope"] },
                            "value": {
                                "type": "object",
                                "required": ["scope"],
                                "properties": {
                                    "scope": { "type": "string", "minLength": 1 }
                                },
                                "additionalProperties": false
                            }
                        },
                        "additionalProperties": false
                    }
                ]
            }
        },
        "additionalProperties": false
    })
}

fn final_answer_render_schema() -> Value {
    json!({
        "type": "object",
        "required": ["canonical_text", "display_text"],
        "properties": {
            "canonical_text": { "type": "string" },
            "display_text": { "type": "string" }
        },
        "additionalProperties": false
    })
}

fn subagent_decision_schema(
    tool_mask_plan: &ToolMaskPlan,
    registered_tools: &[ToolDescriptor],
) -> Value {
    let action_types = supported_subagent_action_types(tool_mask_plan);
    let server_names = supported_server_names(tool_mask_plan);
    let tool_names = supported_tool_names(tool_mask_plan);
    let resource_uris = supported_resource_uris(tool_mask_plan);
    let arguments_schema = merged_arguments_schema(tool_mask_plan, registered_tools);

    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["type", "server_name", "tool_name", "arguments", "resource_uri", "summary", "reason"],
        "properties": {
            "type": { "type": "string", "enum": action_types },
            "server_name": enum_or_null_schema(&server_names),
            "tool_name": enum_or_null_schema(&tool_names),
            "arguments": arguments_schema,
            "resource_uri": enum_or_null_schema(&resource_uris),
            "summary": { "type": ["string", "null"] },
            "reason": { "type": ["string", "null"] }
        }
    })
}

fn supported_subagent_action_types(tool_mask_plan: &ToolMaskPlan) -> Vec<String> {
    let mut action_types = vec![
        "done".to_owned(),
        "partial".to_owned(),
        "cannot_execute".to_owned(),
    ];
    if !tool_mask_plan.allowed_mcp_tools.is_empty() {
        action_types.push("mcp_tool_call".to_owned());
    }
    if !tool_mask_plan.allowed_mcp_resources.is_empty() {
        action_types.push("mcp_resource_read".to_owned());
    }
    if !tool_mask_plan.allowed_local_tools.is_empty() {
        action_types.push("local_tool_call".to_owned());
    }
    action_types
}

fn supported_server_names(tool_mask_plan: &ToolMaskPlan) -> Vec<String> {
    let mut names = tool_mask_plan
        .allowed_mcp_tools
        .iter()
        .map(|tool| tool.server_name.clone())
        .chain(
            tool_mask_plan
                .allowed_mcp_resources
                .iter()
                .map(|resource| resource.server_name.clone()),
        )
        .collect::<Vec<_>>();
    names.sort();
    names.dedup();
    names
}

fn supported_tool_names(tool_mask_plan: &ToolMaskPlan) -> Vec<String> {
    let mut names = tool_mask_plan
        .allowed_mcp_tools
        .iter()
        .map(|tool| tool.tool_name.clone())
        .chain(tool_mask_plan.allowed_local_tools.iter().cloned())
        .collect::<Vec<_>>();
    names.sort();
    names.dedup();
    names
}

fn supported_resource_uris(tool_mask_plan: &ToolMaskPlan) -> Vec<String> {
    let mut uris = tool_mask_plan
        .allowed_mcp_resources
        .iter()
        .map(|resource| resource.resource_uri.clone())
        .collect::<Vec<_>>();
    uris.sort();
    uris.dedup();
    uris
}

fn enum_or_null_schema(values: &[String]) -> Value {
    if values.is_empty() {
        json!({ "type": "null" })
    } else {
        json!({
            "anyOf": [
                { "type": "null" },
                { "type": "string", "enum": values }
            ]
        })
    }
}

fn merged_arguments_schema(
    tool_mask_plan: &ToolMaskPlan,
    registered_tools: &[ToolDescriptor],
) -> Value {
    let mut properties = serde_json::Map::new();

    for allowed in &tool_mask_plan.allowed_local_tools {
        if let Some(tool) = registered_tools
            .iter()
            .find(|tool| tool.family == ToolFamily::Local && tool.name == *allowed)
        {
            merge_argument_properties(&mut properties, &strict_json_schema(&tool.input_schema));
        }
    }

    for allowed in &tool_mask_plan.allowed_mcp_tools {
        if let Some(tool) = registered_tools.iter().find(|tool| {
            tool.family == ToolFamily::McpTool
                && tool.server_name.as_ref().map(|name| name.as_str())
                    == Some(allowed.server_name.as_str())
                && tool.name == allowed.tool_name
        }) {
            merge_argument_properties(&mut properties, &strict_json_schema(&tool.input_schema));
        }
    }

    let mut required = properties.keys().cloned().collect::<Vec<_>>();
    required.sort();

    json!({
        "anyOf": [
            { "type": "null" },
            {
                "type": "object",
                "additionalProperties": false,
                "properties": properties,
                "required": required
            }
        ]
    })
}

fn merge_argument_properties(
    merged_properties: &mut serde_json::Map<String, Value>,
    schema: &Value,
) {
    let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
        return;
    };

    for (name, property_schema) in properties {
        let nullable_schema = ensure_nullable_schema(property_schema.clone());
        match merged_properties.get_mut(name) {
            Some(existing) => {
                if *existing != nullable_schema {
                    *existing = merge_schema_variants(existing.clone(), nullable_schema);
                }
            }
            None => {
                merged_properties.insert(name.clone(), nullable_schema);
            }
        }
    }
}

fn ensure_nullable_schema(schema: Value) -> Value {
    if schema_allows_null(&schema) {
        schema
    } else {
        json!({
            "anyOf": [
                schema,
                { "type": "null" }
            ]
        })
    }
}

fn schema_allows_null(schema: &Value) -> bool {
    match schema {
        Value::Object(map) => {
            if map.get("type").and_then(Value::as_str) == Some("null") {
                return true;
            }
            map.get("anyOf")
                .and_then(Value::as_array)
                .map(|variants| variants.iter().any(schema_allows_null))
                .unwrap_or(false)
        }
        _ => false,
    }
}

fn merge_schema_variants(left: Value, right: Value) -> Value {
    let mut variants = Vec::new();
    append_schema_variant(&mut variants, left);
    append_schema_variant(&mut variants, right);
    json!({ "anyOf": variants })
}

fn append_schema_variant(variants: &mut Vec<Value>, schema: Value) {
    if let Some(items) = schema.get("anyOf").and_then(Value::as_array) {
        for item in items {
            append_schema_variant(variants, item.clone());
        }
        return;
    }

    if !variants.iter().any(|existing| existing == &schema) {
        variants.push(schema);
    }
}

fn strict_json_schema(schema: &Value) -> Value {
    match schema {
        Value::Object(map) => {
            let mut normalized = serde_json::Map::new();
            let mut property_keys = map
                .get("properties")
                .and_then(Value::as_object)
                .map(|properties| properties.keys().cloned().collect::<Vec<_>>());
            if let Some(keys) = property_keys.as_mut() {
                keys.sort();
            }
            for (key, value) in map {
                let normalized_value = match key.as_str() {
                    "properties" => Value::Object(
                        value
                            .as_object()
                            .map(|properties| {
                                properties
                                    .iter()
                                    .map(|(name, schema)| {
                                        (name.clone(), strict_json_schema(schema))
                                    })
                                    .collect()
                            })
                            .unwrap_or_default(),
                    ),
                    "items" => strict_json_schema(value),
                    "prefixItems" | "anyOf" | "oneOf" | "allOf" => Value::Array(
                        value
                            .as_array()
                            .map(|items| items.iter().map(strict_json_schema).collect())
                            .unwrap_or_default(),
                    ),
                    _ => strict_json_schema(value),
                };
                normalized.insert(key.clone(), normalized_value);
            }

            if map.get("type").and_then(Value::as_str) == Some("object")
                || map.contains_key("properties")
            {
                normalized.insert("additionalProperties".to_owned(), Value::Bool(false));
                normalized.insert(
                    "required".to_owned(),
                    Value::Array(
                        property_keys
                            .unwrap_or_default()
                            .into_iter()
                            .map(Value::String)
                            .collect(),
                    ),
                );
            }

            Value::Object(normalized)
        }
        Value::Array(items) => Value::Array(items.iter().map(strict_json_schema).collect()),
        _ => schema.clone(),
    }
}

fn response_client_name(client: ResponseClient) -> &'static str {
    match client {
        ResponseClient::Api => "api",
        ResponseClient::Cli => "cli",
        ResponseClient::Slack => "slack",
        ResponseClient::WhatsApp => "whatsapp",
    }
}

fn response_format_name(format: ResponseFormat) -> &'static str {
    match format {
        ResponseFormat::PlainText => "plain_text",
        ResponseFormat::Markdown => "markdown",
        ResponseFormat::SlackMrkdwn => "slack_mrkdwn",
        ResponseFormat::WhatsAppText => "whatsapp_text",
    }
}

/// Returns `Some(text)` only when the string contains non-whitespace content.
fn non_empty_text(text: &str) -> Option<&str> {
    if text.trim().is_empty() {
        None
    } else {
        Some(text)
    }
}

#[cfg(test)]
mod tests {
    use agent_runtime::{
        model::ModelStepRequest,
        policy::ToolMaskPlan,
        state::{ModelConfig, PromptSection, PromptSnapshot, UsageSummary},
        tools::{ToolDescriptor, builtin_local_tool_catalog},
    };
    use reqwest::StatusCode;
    use serde_json::json;

    use super::{
        OpenAiAdapterError, OpenAiModelAdapter, OpenAiUsage, ResponsesCreateResponse,
        decision_schema, format_provider_error, parse_decision_text, parse_subagent_decision_text,
        prune_null_json, subagent_decision_schema,
    };

    #[test]
    fn adapter_rejects_blank_api_key() {
        let error = OpenAiModelAdapter::new("   ").expect_err("blank keys must fail");
        assert_eq!(error, OpenAiAdapterError::MissingApiKey);
    }

    #[test]
    fn request_body_uses_structured_schema() {
        let adapter = OpenAiModelAdapter::with_base_url("test-key", "https://example.invalid")
            .expect("adapter should construct");
        let request = ModelStepRequest {
            step_number: 2,
            prompt: PromptSnapshot {
                rendered: "hello".to_owned(),
                sections: vec![PromptSection {
                    title: "System Prompt".to_owned(),
                    content: "hello".to_owned(),
                }],
            },
            model_config: ModelConfig::new("gpt-5.4"),
            registered_tools: Vec::new(),
            tool_mask_plan: ToolMaskPlan::terminal_only("main agent has no direct tools"),
        };

        let body = adapter.build_request_body(&request);
        assert_eq!(body["model"], "gpt-5.4");
        assert_eq!(body["input"], "hello");
        assert_eq!(body["text"]["format"]["type"], "json_schema");
        assert_eq!(body["text"]["format"]["strict"], true);
        assert_eq!(body["text"]["format"]["schema"], decision_schema());
        assert_eq!(body["metadata"]["step_number"], "2");
    }

    #[test]
    fn parses_delegate_subagent_decision() {
        let decision = parse_decision_text(
            r#"{"type":"delegate_subagent","content":"","subagent_type":"mcp-executor","goal":"run a query","target":{"kind":"mcp_capability","value":{"server_name":"postgres","capability_kind":"tool","capability_id":"run-sql"}}}"#,
        )
        .expect("decision should parse");

        assert_eq!(
            decision,
            agent_runtime::model::ModelStepDecision::DelegateSubagent {
                delegation: agent_runtime::model::SubagentDelegationRequest {
                    subagent_type: "mcp-executor".to_owned(),
                    goal: "run a query".to_owned(),
                    target: agent_runtime::state::DelegationTarget::McpCapability(
                        agent_runtime::state::McpCapabilityTarget {
                            server_name: agent_runtime::state::ServerName::new("postgres")
                                .expect("valid server name"),
                            capability_kind: mcp_metadata::CapabilityKind::Tool,
                            capability_id: "run-sql".to_owned(),
                        }
                    ),
                }
            }
        );
    }

    #[test]
    fn parses_server_scoped_delegate_subagent_decision() {
        let decision = parse_decision_text(
            r#"{"type":"delegate_subagent","content":"","subagent_type":"mcp-executor","goal":"inspect one server","target":{"kind":"mcp_server_scope","value":{"server_name":"postgres"}}}"#,
        )
        .expect("decision should parse");

        assert_eq!(
            decision,
            agent_runtime::model::ModelStepDecision::DelegateSubagent {
                delegation: agent_runtime::model::SubagentDelegationRequest {
                    subagent_type: "mcp-executor".to_owned(),
                    goal: "inspect one server".to_owned(),
                    target: agent_runtime::state::DelegationTarget::McpServerScope(
                        agent_runtime::state::McpServerScopeTarget {
                            server_name: agent_runtime::state::ServerName::new("postgres")
                                .expect("valid server name"),
                        }
                    ),
                }
            }
        );
    }

    #[test]
    fn parses_final_decision_with_required_calls_field() {
        let decision = parse_decision_text(
            r#"{"type":"final","content":"done","subagent_type":"","goal":"","target":null}"#,
        )
        .expect("decision should parse");

        assert_eq!(
            decision,
            agent_runtime::model::ModelStepDecision::Final {
                content: "done".to_owned(),
            }
        );
    }

    #[test]
    fn parses_subagent_done_decision() {
        let decision = parse_subagent_decision_text(
            r#"{"type":"done","server_name":null,"tool_name":null,"arguments":null,"resource_uri":null,"summary":"completed task","reason":null}"#,
        )
        .expect("subagent done should parse");

        assert_eq!(
            decision,
            agent_runtime::model::SubagentDecision::Done {
                summary: "completed task".to_owned(),
            }
        );
    }

    #[test]
    fn parses_subagent_local_tool_write_file_decision() {
        let decision = parse_subagent_decision_text(
            r#"{"type":"local_tool_call","server_name":null,"tool_name":"write_file","arguments":{"path":"scripts/chart.py","content":"print('ok')\n"},"resource_uri":null,"summary":"write plotting script","reason":"need a reproducible script"}"#,
        )
        .expect("local tool call should parse");

        assert_eq!(
            decision,
            agent_runtime::model::SubagentDecision::LocalToolCall {
                tool_name: agent_runtime::state::LocalToolName::new("write_file")
                    .expect("valid local tool"),
                arguments: json!({
                    "path": "scripts/chart.py",
                    "content": "print('ok')\n",
                }),
            }
        );
    }

    #[test]
    fn parses_subagent_arguments_and_prunes_null_placeholders() {
        let decision = parse_subagent_decision_text(
            r#"{"type":"mcp_tool_call","server_name":"ex-vol","tool_name":"list_tables","arguments":{"database":"volonte","include_detailed_columns":true,"like":null},"resource_uri":null,"summary":"list tables","reason":"inspect schema"}"#,
        )
        .expect("mcp tool call should parse");

        assert_eq!(
            decision,
            agent_runtime::model::SubagentDecision::McpToolCall {
                server_name: agent_runtime::state::ServerName::new("ex-vol")
                    .expect("valid server name"),
                tool_name: "list_tables".to_owned(),
                arguments: json!({
                    "database": "volonte",
                    "include_detailed_columns": true,
                }),
            }
        );
    }

    #[test]
    fn parses_legacy_subagent_local_tool_arguments_json() {
        let decision = parse_subagent_decision_text(
            r#"{"type":"local_tool_call","server_name":null,"tool_name":"write_file","arguments":null,"arguments_json":"{\"path\":\"scripts/chart.py\",\"content\":\"print('ok')\\n\"}","resource_uri":null,"summary":"write plotting script","reason":"need a reproducible script"}"#,
        )
        .expect("legacy arguments_json should still parse");

        assert_eq!(
            decision,
            agent_runtime::model::SubagentDecision::LocalToolCall {
                tool_name: agent_runtime::state::LocalToolName::new("write_file")
                    .expect("valid local tool"),
                arguments: json!({
                    "path": "scripts/chart.py",
                    "content": "print('ok')\n",
                }),
            }
        );
    }

    #[test]
    fn rejects_subagent_local_tool_call_with_unescaped_multiline_arguments_json() {
        let error = parse_subagent_decision_text(
            "{\"type\":\"local_tool_call\",\"server_name\":null,\"tool_name\":\"bash\",\"arguments_json\":\"{\\\"command\\\":\\\"python3 - <<'PY'\nprint('hi')\nPY\\\",\\\"timeout_ms\\\":10000}\",\"resource_uri\":null,\"summary\":\"run inline python\",\"reason\":\"test malformed nested json\"}",
        )
        .expect_err("malformed nested json should fail");

        assert!(matches!(
            error,
            agent_runtime::model::ModelAdapterError::InvalidDecision(message)
                if message.contains("control character")
        ));
    }

    #[test]
    fn subagent_schema_supports_terminal_loop_variants() {
        let schema = subagent_decision_schema(&ToolMaskPlan::terminal_only("none"), &[]);
        assert_eq!(schema["type"], json!("object"));
        assert_eq!(schema["additionalProperties"], json!(false));
        assert!(schema.get("anyOf").is_none());
        assert!(schema.get("allOf").is_none());
        assert!(schema.get("enum").is_none());
        assert!(schema.get("not").is_none());
        assert_eq!(
            schema["properties"]["type"]["enum"],
            json!(["done", "partial", "cannot_execute"])
        );
        assert_eq!(
            schema["properties"]["arguments"]["anyOf"][0]["type"],
            json!("null")
        );
        assert_eq!(
            schema["properties"]["arguments"]["anyOf"][1]["required"],
            json!([])
        );
        assert_eq!(
            schema["properties"]["arguments"]["anyOf"][1]["additionalProperties"],
            json!(false)
        );
    }

    #[test]
    fn subagent_schema_lists_allowed_identifiers_without_conditionals() {
        let server = agent_runtime::state::ServerName::new("ex-vol").expect("valid server");
        let mut tools = builtin_local_tool_catalog();
        tools.push(ToolDescriptor::mcp_tool(
            &server,
            "list_tables",
            None,
            None,
            json!({
                "type": "object",
                "required": ["database"],
                "properties": {
                    "database": { "type": "string" },
                    "include_detailed_columns": {
                        "type": "boolean",
                        "default": true
                    }
                }
            }),
        ));
        let schema = subagent_decision_schema(
            &ToolMaskPlan {
                enforcement_mode: agent_runtime::policy::ToolMaskEnforcementMode::DecodeTimeMask,
                allowed_tool_ids: vec![
                    "local.bash".to_owned(),
                    "mcp.ex-vol.tool.list_tables".to_owned(),
                    "mcp.ex-vol.resource.schema://users".to_owned(),
                ],
                denied_tool_ids: Vec::new(),
                decisions: Vec::new(),
                allowed_local_tools: vec!["bash".to_owned()],
                allowed_mcp_tools: vec![agent_runtime::policy::AllowedMcpTool {
                    server_name: "ex-vol".to_owned(),
                    tool_name: "list_tables".to_owned(),
                }],
                allowed_mcp_resources: vec![agent_runtime::policy::AllowedMcpResource {
                    server_name: "ex-vol".to_owned(),
                    resource_uri: "schema://users".to_owned(),
                }],
            },
            &tools,
        );

        assert_eq!(schema["type"], json!("object"));
        assert!(schema.get("anyOf").is_none());
        assert!(schema.get("oneOf").is_none());
        assert!(schema.get("allOf").is_none());
        assert_eq!(
            schema["properties"]["type"]["enum"],
            json!([
                "done",
                "partial",
                "cannot_execute",
                "mcp_tool_call",
                "mcp_resource_read",
                "local_tool_call"
            ])
        );
        assert_eq!(
            schema["properties"]["server_name"]["anyOf"][1]["enum"],
            json!(["ex-vol"])
        );
        assert_eq!(
            schema["properties"]["tool_name"]["anyOf"][1]["enum"],
            json!(["bash", "list_tables"])
        );
        assert_eq!(
            schema["properties"]["resource_uri"]["anyOf"][1]["enum"],
            json!(["schema://users"])
        );
        assert_eq!(
            schema["properties"]["arguments"]["anyOf"][1]["required"],
            json!([
                "command",
                "database",
                "include_detailed_columns",
                "timeout_ms"
            ])
        );
        assert_eq!(
            schema["properties"]["arguments"]["anyOf"][1]["properties"]["database"]["anyOf"],
            json!([
                { "type": "string" },
                { "type": "null" }
            ])
        );
    }

    #[test]
    fn prune_null_json_removes_nested_null_placeholders() {
        assert_eq!(
            prune_null_json(json!({
                "database": "volonte",
                "like": null,
                "options": {
                    "include_detailed_columns": true,
                    "page_token": null
                },
                "items": [1, null, 2]
            })),
            json!({
                "database": "volonte",
                "options": {
                    "include_detailed_columns": true
                },
                "items": [1, 2]
            })
        );
    }

    #[test]
    fn extracts_decision_text_from_output_message_content() {
        let payload: ResponsesCreateResponse = serde_json::from_value(json!({
            "output": [{
                "type": "message",
                "content": [{
                    "type": "output_text",
                    "text": "{\"type\":\"final\",\"content\":\"done\",\"subagent_type\":\"\",\"goal\":\"\",\"target\":null}"
                }]
            }]
        }))
        .expect("payload should parse");

        assert_eq!(
            payload
                .extract_output_text()
                .expect("payload should contain one decision"),
            "{\"type\":\"final\",\"content\":\"done\",\"subagent_type\":\"\",\"goal\":\"\",\"target\":null}"
        );
    }

    #[test]
    fn extracts_decision_text_from_nested_text_value() {
        let payload: ResponsesCreateResponse = serde_json::from_value(json!({
            "output": [{
                "type": "message",
                "content": [{
                    "type": "output_text",
                    "text": {
                        "value": "{\"type\":\"final\",\"content\":\"done\",\"calls\":[]}"
                    }
                }]
            }]
        }))
        .expect("payload should parse");

        assert_eq!(
            payload
                .extract_output_text()
                .expect("payload should contain one decision"),
            "{\"type\":\"final\",\"content\":\"done\",\"calls\":[]}"
        );
    }

    #[test]
    fn extracts_decision_text_when_duplicate_candidates_match() {
        let decision = "{\"type\":\"done\",\"server_name\":null,\"tool_name\":null,\"arguments\":null,\"resource_uri\":null,\"summary\":\"completed task\",\"reason\":null}";
        let payload: ResponsesCreateResponse = serde_json::from_value(json!({
            "output_text": decision,
            "output": [{
                "type": "message",
                "content": [{
                    "type": "output_text",
                    "text": decision
                }]
            }]
        }))
        .expect("payload should parse");

        assert_eq!(
            payload
                .extract_output_text()
                .expect("duplicate candidates should collapse"),
            decision
        );
    }

    #[test]
    fn rejects_multiple_distinct_decision_text_candidates() {
        let payload: ResponsesCreateResponse = serde_json::from_value(json!({
            "output": [
                {
                    "type": "message",
                    "phase": "final_answer",
                    "content": [{
                        "type": "output_text",
                        "text": "{\"type\":\"local_tool_call\",\"server_name\":null,\"tool_name\":\"write_file\",\"arguments\":{\"path\":\"outputs/data.csv\",\"content\":\"a,b\\n1,2\\n\"},\"resource_uri\":null,\"summary\":\"write csv\",\"reason\":\"stage data\"}"
                    }]
                },
                {
                    "type": "message",
                    "phase": "final_answer",
                    "content": [{
                        "type": "output_text",
                        "text": "{\"type\":\"local_tool_call\",\"server_name\":null,\"tool_name\":\"bash\",\"arguments\":{\"command\":\"python3 scripts/plot.py\",\"timeout_ms\":120000},\"resource_uri\":null,\"summary\":\"run plot\",\"reason\":\"generate chart\"}"
                    }]
                }
            ]
        }))
        .expect("payload should parse");

        let error = payload
            .extract_output_text()
            .expect_err("multiple distinct candidates should fail");
        assert!(matches!(
            error,
            agent_runtime::model::ModelAdapterError::InvalidDecision(message)
                if message.contains("multiple distinct output_text candidates")
        ));
    }

    #[test]
    fn extract_output_text_prefers_final_answer_phase_over_commentary_candidates() {
        let payload: ResponsesCreateResponse = serde_json::from_value(json!({
            "output": [
                {
                    "type": "message",
                    "phase": "commentary",
                    "content": [{
                        "type": "output_text",
                        "text": "{\"type\":\"mcp_tool_call\",\"server_name\":\"ex-vol\",\"tool_name\":\"run_select_query\",\"arguments\":{\"database\":\"volonte\",\"include_detailed_columns\":false,\"like\":\"%checkin%\",\"not_like\":\"%response%\",\"page_size\":50,\"page_token\":null,\"query\":null},\"resource_uri\":null,\"summary\":\"Schema discovery suggests weekly active users may be inferred from check-in responses. Next step is to run a SELECT query for weekly distinct users and compare recent weeks.\",\"reason\":\"Need a query to compute week-on-week user counts from the most relevant engagement table(s).\"}"
                    }]
                },
                {
                    "type": "message",
                    "phase": "final_answer",
                    "content": [{
                        "type": "output_text",
                        "text": "{\"type\":\"mcp_tool_call\",\"server_name\":\"ex-vol\",\"tool_name\":\"run_select_query\",\"arguments\":{\"database\":null,\"include_detailed_columns\":null,\"like\":null,\"not_like\":null,\"page_size\":null,\"page_token\":null,\"query\":\"select 1\"},\"resource_uri\":null,\"summary\":\"Running query to calculate recent weekly user counts and week-on-week change for Volonte.\",\"reason\":\"Execute the weekly distinct user count query on volonte.checkin_responses to complete the delegated goal.\"}"
                    }]
                }
            ]
        }))
        .expect("payload should parse");

        assert_eq!(
            payload
                .extract_output_text()
                .expect("final answer candidate should win"),
            "{\"type\":\"mcp_tool_call\",\"server_name\":\"ex-vol\",\"tool_name\":\"run_select_query\",\"arguments\":{\"database\":null,\"include_detailed_columns\":null,\"like\":null,\"not_like\":null,\"page_size\":null,\"page_token\":null,\"query\":\"select 1\"},\"resource_uri\":null,\"summary\":\"Running query to calculate recent weekly user counts and week-on-week change for Volonte.\",\"reason\":\"Execute the weekly distinct user count query on volonte.checkin_responses to complete the delegated goal.\"}"
        );
    }

    #[test]
    fn extract_output_text_prefers_actionable_commentary_over_terminal_final_answer() {
        let payload: ResponsesCreateResponse = serde_json::from_value(json!({
            "output": [
                {
                    "type": "message",
                    "phase": "commentary",
                    "content": [{
                        "type": "output_text",
                        "text": "{\"type\":\"delegate_subagent\",\"content\":\"Generate the requested bar/line combo visualization from the already-computed weekly series using a reproducible Python script and save output files in the workspace.\",\"goal\":\"Create chart artifact for week-on-week user count visualization\",\"subagent_type\":\"tool-executor\",\"target\":{\"kind\":\"local_tools_scope\",\"value\":{\"scope\":\"workspace\"}}}"
                    }]
                },
                {
                    "type": "message",
                    "phase": "final_answer",
                    "content": [{
                        "type": "output_text",
                        "text": "{\"type\":\"final\",\"content\":\"I’ve generated the chart and saved it in the workspace.\",\"goal\":\"Respond to user with completed visualization result and file paths\",\"subagent_type\":\"tool-executor\",\"target\":null}"
                    }]
                }
            ]
        }))
        .expect("payload should parse");

        assert_eq!(
            payload
                .extract_output_text()
                .expect("actionable commentary candidate should win"),
            "{\"type\":\"delegate_subagent\",\"content\":\"Generate the requested bar/line combo visualization from the already-computed weekly series using a reproducible Python script and save output files in the workspace.\",\"goal\":\"Create chart artifact for week-on-week user count visualization\",\"subagent_type\":\"tool-executor\",\"target\":{\"kind\":\"local_tools_scope\",\"value\":{\"scope\":\"workspace\"}}}"
        );
    }

    #[test]
    fn extract_output_text_prefers_actionable_tool_call_over_terminal_done() {
        let payload: ResponsesCreateResponse = serde_json::from_value(json!({
            "output": [
                {
                    "type": "message",
                    "phase": "commentary",
                    "content": [{
                        "type": "output_text",
                        "text": "{\"type\":\"local_tool_call\",\"server_name\":null,\"tool_name\":\"bash\",\"arguments\":{\"command\":\"mkdir -p outputs scripts && python3 scripts/generate_missing_file_viz.py\",\"timeout_ms\":120000},\"resource_uri\":null,\"summary\":\"Generate the visualization artifact and print its path.\",\"reason\":\"Need to create the missing visualization before returning a path.\"}"
                    }]
                },
                {
                    "type": "message",
                    "phase": "final_answer",
                    "content": [{
                        "type": "output_text",
                        "text": "{\"type\":\"done\",\"server_name\":null,\"tool_name\":null,\"arguments\":null,\"resource_uri\":null,\"summary\":\"Created a visualization artifact acknowledging the missing file/data context. Exact path: outputs/missing-file-visualization.png\",\"reason\":null}"
                    }]
                }
            ]
        }))
        .expect("payload should parse");

        assert_eq!(
            payload
                .extract_output_text()
                .expect("actionable tool call should win over terminal done"),
            "{\"type\":\"local_tool_call\",\"server_name\":null,\"tool_name\":\"bash\",\"arguments\":{\"command\":\"mkdir -p outputs scripts && python3 scripts/generate_missing_file_viz.py\",\"timeout_ms\":120000},\"resource_uri\":null,\"summary\":\"Generate the visualization artifact and print its path.\",\"reason\":\"Need to create the missing visualization before returning a path.\"}"
        );
    }

    #[test]
    fn extract_output_text_uses_first_commentary_action_when_multiple_actions_are_emitted() {
        let payload: ResponsesCreateResponse = serde_json::from_value(json!({
            "output": [
                {
                    "type": "message",
                    "phase": "commentary",
                    "content": [{
                        "type": "output_text",
                        "text": "{\"type\":\"local_tool_call\",\"server_name\":null,\"tool_name\":\"bash\",\"arguments\":{\"command\":\"mkdir -p outputs scripts && printf 'week_start,user_count\\n2026-04-05,25\\n' > outputs/week_on_week_user_count.csv\",\"timeout_ms\":120000},\"resource_uri\":null,\"summary\":\"Materialize the computed rows into CSV.\",\"reason\":\"Need a local data file before plotting.\"}"
                    }]
                },
                {
                    "type": "message",
                    "phase": "commentary",
                    "content": [{
                        "type": "output_text",
                        "text": "{\"type\":\"local_tool_call\",\"server_name\":null,\"tool_name\":\"bash\",\"arguments\":{\"command\":\"python3 scripts/plot_week_on_week_user_count.py\",\"timeout_ms\":120000},\"resource_uri\":null,\"summary\":\"Generate the line chart from the CSV.\",\"reason\":\"Produce the requested visualization artifact.\"}"
                    }]
                },
                {
                    "type": "message",
                    "phase": "final_answer",
                    "content": [{
                        "type": "output_text",
                        "text": "{\"type\":\"done\",\"server_name\":null,\"tool_name\":null,\"arguments\":null,\"resource_uri\":null,\"summary\":\"Generated the requested line chart and saved it as outputs/week_on_week_user_count_line_chart.png.\",\"reason\":null}"
                    }]
                }
            ]
        }))
        .expect("payload should parse");

        assert_eq!(
            payload
                .extract_output_text()
                .expect("first commentary action should be selected"),
            "{\"type\":\"local_tool_call\",\"server_name\":null,\"tool_name\":\"bash\",\"arguments\":{\"command\":\"mkdir -p outputs scripts && printf 'week_start,user_count\\n2026-04-05,25\\n' > outputs/week_on_week_user_count.csv\",\"timeout_ms\":120000},\"resource_uri\":null,\"summary\":\"Materialize the computed rows into CSV.\",\"reason\":\"Need a local data file before plotting.\"}"
        );
    }

    #[test]
    fn parses_cached_tokens_from_openai_usage_details() {
        let usage: OpenAiUsage = serde_json::from_value(json!({
            "input_tokens": 2006,
            "input_tokens_details": {
                "cached_tokens": 1920
            },
            "output_tokens": 300,
            "total_tokens": 2306
        }))
        .expect("usage should parse");

        assert_eq!(
            UsageSummary::from(usage),
            UsageSummary {
                input_tokens: 2006,
                cached_tokens: 1920,
                output_tokens: 300,
                total_tokens: 2306,
            }
        );
    }

    #[test]
    fn provider_error_uses_openai_error_message_when_present() {
        let error = format_provider_error(
            StatusCode::BAD_REQUEST,
            r#"{"error":{"message":"Unknown parameter: text.format","type":"invalid_request_error","param":"text.format","code":"unknown_parameter"}}"#,
        );

        assert_eq!(
            error,
            "OpenAI Responses API returned status 400 Bad Request: Unknown parameter: text.format [type=invalid_request_error] [param=text.format] [code=unknown_parameter]"
        );
    }

    #[test]
    fn provider_error_falls_back_to_raw_body_when_not_json() {
        let error = format_provider_error(StatusCode::BAD_REQUEST, "plain failure");

        assert_eq!(
            error,
            "OpenAI Responses API returned status 400 Bad Request: plain failure"
        );
    }

    #[test]
    fn provider_error_keeps_full_non_json_body() {
        let body = "x".repeat(500);
        let error = format_provider_error(StatusCode::BAD_REQUEST, &body);

        assert_eq!(
            error,
            format!("OpenAI Responses API returned status 400 Bad Request: {body}")
        );
        assert!(!error.ends_with("..."));
    }
}
