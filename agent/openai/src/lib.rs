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
    state::{ResponseClient, ResponseFormat, UsageSummary},
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
                    "schema": subagent_decision_schema(),
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
        let decision_text = payload.extract_output_text().ok_or_else(|| {
            ModelAdapterError::InvalidDecision(
                "missing output_text and no assistant text found in output".to_owned(),
            )
        })?;
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
        let decision_text = payload.extract_output_text().ok_or_else(|| {
            ModelAdapterError::InvalidDecision(
                "missing output_text and no assistant text found in output".to_owned(),
            )
        })?;
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
        let response_text = payload.extract_output_text().ok_or_else(|| {
            ModelAdapterError::InvalidDecision(
                "missing output_text and no assistant text found in output".to_owned(),
            )
        })?;
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
/// `arguments_json` is emitted as a string so the schema can stay simple and
/// strict. This function performs the second-stage parse for every MCP action
/// before constructing `ModelStepDecision`.
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
        "tool_call" => Ok(SubagentDecision::ToolCall {
            server_name: wire.server_name.ok_or_else(|| {
                ModelAdapterError::InvalidDecision("tool_call requires server_name".to_owned())
            })?,
            tool_name: wire.tool_name.unwrap_or_default(),
            arguments: serde_json::from_str(
                &wire.arguments_json.unwrap_or_else(|| "{}".to_owned()),
            )
            .map_err(|error| {
                ModelAdapterError::InvalidDecision(format!(
                    "invalid tool_call arguments_json: {error}"
                ))
            })?,
        }),
        "resource_read" => Ok(SubagentDecision::ResourceRead {
            server_name: wire.server_name.ok_or_else(|| {
                ModelAdapterError::InvalidDecision("resource_read requires server_name".to_owned())
            })?,
            resource_uri: wire.resource_uri.unwrap_or_default(),
        }),
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
    /// Returns the first non-empty assistant text found in the response.
    ///
    /// The top-level `output_text` shortcut is preferred when present. If it is
    /// absent or blank, the adapter walks the nested `output` structure to find
    /// the first text-bearing content item.
    fn extract_output_text(&self) -> Option<String> {
        self.output_text
            .as_deref()
            .and_then(non_empty_text)
            .map(ToOwned::to_owned)
            .or_else(|| {
                self.output
                    .iter()
                    .find_map(extract_text_from_output_value)
                    .map(ToOwned::to_owned)
            })
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
    target: Option<agent_runtime::state::McpCapabilityTarget>,
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
                "type": ["object", "null"],
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

fn subagent_decision_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["type", "server_name", "tool_name", "arguments_json", "resource_uri", "summary", "reason"],
        "properties": {
            "type": {
                "type": "string",
                "enum": ["tool_call", "resource_read", "done", "partial", "cannot_execute"]
            },
            "server_name": { "type": ["string", "null"] },
            "tool_name": { "type": ["string", "null"] },
            "arguments_json": { "type": ["string", "null"] },
            "resource_uri": { "type": ["string", "null"] },
            "summary": { "type": ["string", "null"] },
            "reason": { "type": ["string", "null"] }
        }
    })
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

/// Recursively searches a Responses API `output` item for the first text value.
///
/// The Responses API can nest text inside arrays of output items or inside a
/// `content` list on message objects. This helper walks those structures
/// without caring about unrelated keys.
fn extract_text_from_output_value(value: &Value) -> Option<&str> {
    match value {
        Value::Array(items) => items.iter().find_map(extract_text_from_output_value),
        Value::Object(map) => {
            if let Some(text) = map.get("text").and_then(extract_text_field) {
                return Some(text);
            }

            map.get("content")
                .and_then(Value::as_array)
                .and_then(|items| items.iter().find_map(extract_text_from_output_value))
        }
        _ => None,
    }
}

/// Extracts text from one `text` field in either string or object form.
///
/// Some Responses API payloads expose text directly as a string, while others
/// wrap it in an object containing a `value` field. This helper normalizes both
/// representations and rejects blank strings.
fn extract_text_field(value: &Value) -> Option<&str> {
    match value {
        Value::String(text) => non_empty_text(text),
        Value::Object(map) => map
            .get("value")
            .and_then(Value::as_str)
            .and_then(non_empty_text),
        _ => None,
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
        state::{ModelConfig, PromptSection, PromptSnapshot, UsageSummary},
    };
    use reqwest::StatusCode;
    use serde_json::json;

    use super::{
        OpenAiAdapterError, OpenAiModelAdapter, OpenAiUsage, ResponsesCreateResponse,
        decision_schema, format_provider_error, parse_decision_text, parse_subagent_decision_text,
        subagent_decision_schema,
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
            r#"{"type":"delegate_subagent","content":"","subagent_type":"tool-executor","goal":"run a query","target":{"server_name":"postgres","capability_kind":"tool","capability_id":"run-sql"}}"#,
        )
        .expect("decision should parse");

        assert_eq!(
            decision,
            agent_runtime::model::ModelStepDecision::DelegateSubagent {
                delegation: agent_runtime::model::SubagentDelegationRequest {
                    subagent_type: "tool-executor".to_owned(),
                    goal: "run a query".to_owned(),
                    target: agent_runtime::state::McpCapabilityTarget {
                        server_name: agent_runtime::state::ServerName::new("postgres")
                            .expect("valid server name"),
                        capability_kind: mcp_metadata::CapabilityKind::Tool,
                        capability_id: "run-sql".to_owned(),
                    },
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
            r#"{"type":"done","server_name":null,"tool_name":null,"arguments_json":null,"resource_uri":null,"summary":"completed task","reason":null}"#,
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
    fn subagent_schema_supports_terminal_loop_variants() {
        let schema = subagent_decision_schema();
        assert_eq!(
            schema["properties"]["type"]["enum"],
            json!([
                "tool_call",
                "resource_read",
                "done",
                "partial",
                "cannot_execute"
            ])
        );
        assert_eq!(
            schema["required"],
            json!([
                "type",
                "server_name",
                "tool_name",
                "arguments_json",
                "resource_uri",
                "summary",
                "reason"
            ])
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
            payload.extract_output_text().as_deref(),
            Some(
                "{\"type\":\"final\",\"content\":\"done\",\"subagent_type\":\"\",\"goal\":\"\",\"target\":null}"
            )
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
            payload.extract_output_text().as_deref(),
            Some("{\"type\":\"final\",\"content\":\"done\",\"calls\":[]}")
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
