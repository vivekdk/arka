//! HTTP API and webhook routes for the control plane.
//!
//! The API surface is intentionally thin. Route handlers mostly validate basic
//! request state, delegate to `SessionService`, and translate errors into HTTP
//! responses or SSE frames.

use std::{convert::Infallible, sync::Arc};

use agent_runtime::{
    ResponseClient, ResponseFormat, ResponseTarget, RuntimeEvent, TerminationReason,
};
use async_trait::async_trait;
use axum::{
    Json, Router,
    body::Bytes,
    extract::{Path, Query, State},
    http::HeaderMap,
    response::sse::{Event, KeepAlive, Sse},
    response::{Html, IntoResponse},
    routing::{get, post},
};
use futures_util::{Stream, StreamExt};
use tokio::sync::mpsc;
use tokio_stream::wrappers::BroadcastStream;
use tower_http::trace::TraceLayer;
use tracing::{info_span, warn};
use uuid::Uuid;

use crate::{
    adapters::{
        ChannelAdapter, ChannelAdapterError, SlackChannelAdapter, SlackDispatchRequest,
        SlackIngressResult, WhatsAppChannelAdapter,
    },
    observability::{
        DebugHistoryStore, RuntimeHarnessListener, RuntimeHarnessListenerError,
        RuntimeHarnessObservation,
    },
    runner::TurnRunner,
    service::{ControlPlaneError, ConversationStore, SessionService},
    types::{
        ApiResponseFormat, ChannelDeliveryTarget, ChannelKind, CompleteWhatsAppLoginRequest,
        CreateSessionRequest, DeliveryStatus, ReceiveWhatsAppMessageRequest,
        ReceiveWhatsAppMessageResponse, SendSessionMessageRequest, SendSessionMessageResponse,
        SessionEvent, SessionId, StartWhatsAppLoginResponse, SubmitApprovalRequest,
        WhatsAppGatewayStatus, WhatsAppWebhookPayload,
    },
    whatsapp::{WhatsAppConnector, WhatsAppGatewayError, WhatsAppGatewayHandle},
};

const DEFAULT_SLACK_API_BASE_URL: &str = "https://slack.com/api";

/// Slack connector settings used by the HTTP API.
pub struct SlackConnector {
    pub signing_secret: String,
    pub delivery_client: Arc<dyn SlackDeliveryClient>,
    pub event_queue_capacity: usize,
}

/// Minimal outbound Slack transport.
#[async_trait]
pub trait SlackDeliveryClient: Send + Sync {
    async fn post_message(
        &self,
        target: &ChannelDeliveryTarget,
        text: &str,
    ) -> Result<(), SlackDeliveryError>;

    async fn start_stream(
        &self,
        target: &ChannelDeliveryTarget,
        markdown_text: &str,
    ) -> Result<SlackStreamHandle, SlackDeliveryError> {
        let _ = target;
        let _ = markdown_text;
        Err(SlackDeliveryError::Api(
            "slack streaming is not supported by this client".to_owned(),
        ))
    }

    async fn append_stream(
        &self,
        handle: &SlackStreamHandle,
        markdown_text: &str,
    ) -> Result<(), SlackDeliveryError> {
        let _ = handle;
        let _ = markdown_text;
        Err(SlackDeliveryError::Api(
            "slack streaming is not supported by this client".to_owned(),
        ))
    }

    async fn stop_stream(
        &self,
        handle: &SlackStreamHandle,
        markdown_text: Option<&str>,
    ) -> Result<(), SlackDeliveryError> {
        let _ = handle;
        let _ = markdown_text;
        Err(SlackDeliveryError::Api(
            "slack streaming is not supported by this client".to_owned(),
        ))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlackStreamHandle {
    pub channel: String,
    pub ts: String,
}

/// Reqwest-backed Slack Web API client.
#[derive(Clone)]
pub struct ReqwestSlackDeliveryClient {
    http: reqwest::Client,
    bot_token: String,
    api_base_url: String,
}

impl ReqwestSlackDeliveryClient {
    pub fn new(bot_token: String, api_base_url: Option<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            bot_token,
            api_base_url: api_base_url.unwrap_or_else(|| DEFAULT_SLACK_API_BASE_URL.to_owned()),
        }
    }
}

#[async_trait]
impl SlackDeliveryClient for ReqwestSlackDeliveryClient {
    async fn post_message(
        &self,
        target: &ChannelDeliveryTarget,
        text: &str,
    ) -> Result<(), SlackDeliveryError> {
        let channel = target.external_channel_id.as_ref().ok_or_else(|| {
            SlackDeliveryError::InvalidTarget(
                "slack delivery target is missing channel id".to_owned(),
            )
        })?;
        let thread_ts = target.external_thread_id.as_ref().ok_or_else(|| {
            SlackDeliveryError::InvalidTarget(
                "slack delivery target is missing thread id".to_owned(),
            )
        })?;
        let response = self
            .http
            .post(format!(
                "{}/chat.postMessage",
                self.api_base_url.trim_end_matches('/')
            ))
            .bearer_auth(&self.bot_token)
            .json(&serde_json::json!({
                "channel": channel,
                "thread_ts": thread_ts,
                "text": text,
            }))
            .send()
            .await?
            .error_for_status()?;
        let payload: SlackPostMessageResponse = response.json().await?;
        if payload.ok {
            Ok(())
        } else {
            Err(SlackDeliveryError::Api(payload.error.unwrap_or_else(
                || "slack api returned ok=false".to_owned(),
            )))
        }
    }

    async fn start_stream(
        &self,
        target: &ChannelDeliveryTarget,
        markdown_text: &str,
    ) -> Result<SlackStreamHandle, SlackDeliveryError> {
        let channel = target.external_channel_id.as_ref().ok_or_else(|| {
            SlackDeliveryError::InvalidTarget(
                "slack delivery target is missing channel id".to_owned(),
            )
        })?;
        let thread_ts = target.external_thread_id.as_ref().ok_or_else(|| {
            SlackDeliveryError::InvalidTarget(
                "slack delivery target is missing thread id".to_owned(),
            )
        })?;
        let team_id = target.external_workspace_id.as_ref().ok_or_else(|| {
            SlackDeliveryError::InvalidTarget(
                "slack delivery target is missing workspace id".to_owned(),
            )
        })?;
        let response = self
            .http
            .post(format!(
                "{}/chat.startStream",
                self.api_base_url.trim_end_matches('/')
            ))
            .bearer_auth(&self.bot_token)
            .json(&serde_json::json!({
                "channel": channel,
                "thread_ts": thread_ts,
                "recipient_user_id": target.external_user_id,
                "recipient_team_id": team_id,
                "markdown_text": markdown_text,
            }))
            .send()
            .await?
            .error_for_status()?;
        let payload: SlackStreamResponse = response.json().await?;
        if payload.ok {
            Ok(SlackStreamHandle {
                channel: payload.channel.unwrap_or_else(|| channel.clone()),
                ts: payload.ts.ok_or_else(|| {
                    SlackDeliveryError::Api("slack startStream response is missing ts".to_owned())
                })?,
            })
        } else {
            Err(SlackDeliveryError::Api(payload.error.unwrap_or_else(
                || "slack api returned ok=false".to_owned(),
            )))
        }
    }

    async fn append_stream(
        &self,
        handle: &SlackStreamHandle,
        markdown_text: &str,
    ) -> Result<(), SlackDeliveryError> {
        let response = self
            .http
            .post(format!(
                "{}/chat.appendStream",
                self.api_base_url.trim_end_matches('/')
            ))
            .bearer_auth(&self.bot_token)
            .json(&serde_json::json!({
                "channel": handle.channel,
                "ts": handle.ts,
                "markdown_text": markdown_text,
            }))
            .send()
            .await?
            .error_for_status()?;
        let payload: SlackStreamResponse = response.json().await?;
        if payload.ok {
            Ok(())
        } else {
            Err(SlackDeliveryError::Api(payload.error.unwrap_or_else(
                || "slack api returned ok=false".to_owned(),
            )))
        }
    }

    async fn stop_stream(
        &self,
        handle: &SlackStreamHandle,
        markdown_text: Option<&str>,
    ) -> Result<(), SlackDeliveryError> {
        let body = slack_stop_stream_body(handle, markdown_text);
        let response = self
            .http
            .post(format!(
                "{}/chat.stopStream",
                self.api_base_url.trim_end_matches('/')
            ))
            .bearer_auth(&self.bot_token)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;
        let payload: SlackStreamResponse = response.json().await?;
        if payload.ok {
            Ok(())
        } else {
            Err(SlackDeliveryError::Api(payload.error.unwrap_or_else(
                || "slack api returned ok=false".to_owned(),
            )))
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SlackDeliveryError {
    #[error("invalid slack delivery target: {0}")]
    InvalidTarget(String),
    #[error("slack delivery request failed: {0}")]
    Request(#[from] reqwest::Error),
    #[error("slack api returned an error: {0}")]
    Api(String),
}

#[derive(Debug, serde::Deserialize)]
struct SlackPostMessageResponse {
    ok: bool,
    error: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct SlackStreamResponse {
    ok: bool,
    error: Option<String>,
    channel: Option<String>,
    ts: Option<String>,
}

fn slack_stop_stream_body(
    handle: &SlackStreamHandle,
    markdown_text: Option<&str>,
) -> serde_json::Value {
    let mut body = serde_json::Map::from_iter([
        (
            "channel".to_owned(),
            serde_json::Value::String(handle.channel.clone()),
        ),
        (
            "ts".to_owned(),
            serde_json::Value::String(handle.ts.clone()),
        ),
    ]);
    if let Some(markdown_text) = markdown_text {
        body.insert(
            "markdown_text".to_owned(),
            serde_json::Value::String(markdown_text.to_owned()),
        );
    }
    serde_json::Value::Object(body)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SlackStreamTaskResult {
    streamed_text: bool,
}

/// Shared HTTP application state.
pub struct ApiState<R, S> {
    /// Shared session orchestration service used by all routes.
    pub service: Arc<SessionService<R, S>>,
    /// Optional Postgres-backed debug history store.
    pub debug_history: Option<Arc<DebugHistoryStore>>,
    /// Slack payload normalizer and renderer.
    pub slack_adapter: SlackChannelAdapter,
    /// Queue used to process Slack events asynchronously.
    pub slack_queue: Option<mpsc::Sender<SlackDispatchRequest>>,
    /// WhatsApp payload normalizer and renderer.
    pub whatsapp_adapter: WhatsAppChannelAdapter,
    /// Optional stateful WhatsApp gateway handle.
    pub whatsapp_gateway: Option<Arc<WhatsAppGatewayHandle>>,
}

impl<R, S> Clone for ApiState<R, S> {
    fn clone(&self) -> Self {
        Self {
            service: Arc::clone(&self.service),
            debug_history: self.debug_history.clone(),
            slack_adapter: self.slack_adapter.clone(),
            slack_queue: self.slack_queue.clone(),
            whatsapp_adapter: self.whatsapp_adapter.clone(),
            whatsapp_gateway: self.whatsapp_gateway.clone(),
        }
    }
}

/// Builds the HTTP router for the control plane.
pub fn router<R, S>(service: SessionService<R, S>) -> Router
where
    R: TurnRunner + Send + Sync + 'static,
    S: ConversationStore + Send + Sync + 'static,
{
    router_with_channels(service, None, None, None)
}

/// Builds the HTTP router for the control plane with optional debug history store.
pub fn router_with_debug_history<R, S>(
    service: SessionService<R, S>,
    debug_history: Option<DebugHistoryStore>,
) -> Router
where
    R: TurnRunner + Send + Sync + 'static,
    S: ConversationStore + Send + Sync + 'static,
{
    router_with_channels(service, debug_history, None, None)
}

/// Builds the HTTP router for the control plane with optional Slack connector support.
pub fn router_with_slack<R, S>(
    service: SessionService<R, S>,
    debug_history: Option<DebugHistoryStore>,
    slack_connector: Option<SlackConnector>,
) -> Router
where
    R: TurnRunner + Send + Sync + 'static,
    S: ConversationStore + Send + Sync + 'static,
{
    router_with_channels(service, debug_history, slack_connector, None)
}

/// Builds the HTTP router for the control plane with optional Slack and WhatsApp connectors.
pub fn router_with_channels<R, S>(
    service: SessionService<R, S>,
    debug_history: Option<DebugHistoryStore>,
    slack_connector: Option<SlackConnector>,
    whatsapp_connector: Option<WhatsAppConnector>,
) -> Router
where
    R: TurnRunner + Send + Sync + 'static,
    S: ConversationStore + Send + Sync + 'static,
{
    let service = Arc::new(service);
    let (slack_adapter, slack_queue) = if let Some(connector) = slack_connector {
        let adapter = SlackChannelAdapter::new(Some(connector.signing_secret));
        let (tx, rx) = mpsc::channel(connector.event_queue_capacity.max(1));
        tokio::spawn(run_slack_worker(
            Arc::clone(&service),
            adapter.clone(),
            connector.delivery_client,
            rx,
        ));
        (adapter, Some(tx))
    } else {
        (SlackChannelAdapter::new(None), None)
    };
    let whatsapp_gateway = whatsapp_connector.map(|connector| {
        Arc::new(
            WhatsAppGatewayHandle::spawn(Arc::clone(&service), connector)
                .expect("whatsapp connector should initialize"),
        )
    });
    let state = ApiState {
        service,
        debug_history: debug_history.map(Arc::new),
        slack_adapter,
        slack_queue,
        whatsapp_adapter: WhatsAppChannelAdapter,
        whatsapp_gateway,
    };

    Router::new()
        .route("/sessions", post(create_session::<R, S>))
        .route("/sessions/{session_id}", get(get_session::<R, S>))
        .route(
            "/sessions/{session_id}/messages",
            get(get_messages::<R, S>).post(send_session_message::<R, S>),
        )
        .route(
            "/sessions/{session_id}/approvals/{approval_id}",
            post(submit_approval::<R, S>),
        )
        .route("/sessions/{session_id}/events", get(stream_events::<R, S>))
        .route("/debug/history", get(debug_history_page::<R, S>))
        .route(
            "/debug/history/sessions",
            get(list_debug_history_sessions::<R, S>),
        )
        .route(
            "/debug/history/sessions/{session_id}",
            get(list_debug_history_session_turns::<R, S>),
        )
        .route(
            "/debug/history/turns/{turn_id}",
            get(get_debug_history_turn::<R, S>),
        )
        .route("/channels/slack/events", post(handle_slack_event::<R, S>))
        .route(
            "/channels/whatsapp/status",
            get(get_whatsapp_status::<R, S>),
        )
        .route(
            "/channels/whatsapp/login/start",
            post(start_whatsapp_login::<R, S>),
        )
        .route(
            "/channels/whatsapp/login/complete",
            post(complete_whatsapp_login::<R, S>),
        )
        .route("/channels/whatsapp/logout", post(logout_whatsapp::<R, S>))
        .route(
            "/channels/whatsapp/inbound",
            post(handle_whatsapp_inbound::<R, S>),
        )
        .route(
            "/channels/whatsapp/webhook",
            post(handle_whatsapp_event::<R, S>),
        )
        .layer(
            TraceLayer::new_for_http().make_span_with(|request: &axum::http::Request<_>| {
                info_span!(
                    "http_request",
                    method = %request.method(),
                    uri = %request.uri()
                )
            }),
        )
        .with_state(state)
}

async fn create_session<R, S>(
    State(state): State<ApiState<R, S>>,
    Json(request): Json<CreateSessionRequest>,
) -> Result<Json<crate::types::SessionRecord>, ApiError>
where
    R: TurnRunner + Send + Sync + 'static,
    S: ConversationStore + Send + Sync + 'static,
{
    Ok(Json(state.service.create_session(request).await?))
}

async fn get_session<R, S>(
    State(state): State<ApiState<R, S>>,
    Path(session_id): Path<SessionId>,
) -> Result<Json<crate::types::SessionRecord>, ApiError>
where
    R: TurnRunner + Send + Sync + 'static,
    S: ConversationStore + Send + Sync + 'static,
{
    state
        .service
        .get_session(&session_id)
        .await
        .map(Json)
        .ok_or(ApiError::NotFound("session not found".to_owned()))
}

async fn get_messages<R, S>(
    State(state): State<ApiState<R, S>>,
    Path(session_id): Path<SessionId>,
) -> Result<Json<Vec<crate::types::SessionMessage>>, ApiError>
where
    R: TurnRunner + Send + Sync + 'static,
    S: ConversationStore + Send + Sync + 'static,
{
    if state.service.get_session(&session_id).await.is_none() {
        return Err(ApiError::NotFound("session not found".to_owned()));
    }
    Ok(Json(state.service.get_messages(&session_id).await))
}

async fn send_session_message<R, S>(
    State(state): State<ApiState<R, S>>,
    Path(session_id): Path<SessionId>,
    Json(request): Json<SendSessionMessageRequest>,
) -> Result<Json<SendSessionMessageResponse>, ApiError>
where
    R: TurnRunner + Send + Sync + 'static,
    S: ConversationStore + Send + Sync + 'static,
{
    let session = state
        .service
        .get_session(&session_id)
        .await
        .ok_or(ApiError::NotFound("session not found".to_owned()))?;
    let result = state
        .service
        .send_session_text(
            &session.session_id,
            request.text,
            request
                .idempotency_key
                .unwrap_or_else(|| format!("api-{}-{}", session_id, uuid::Uuid::new_v4())),
            ChannelKind::Api,
            api_response_target(request.response_format),
        )
        .await?;
    state
        .service
        .emit_channel_delivery(session_id.clone(), ChannelKind::Api, DeliveryStatus::Sent);
    Ok(Json(SendSessionMessageResponse { result }))
}

fn api_response_target(response_format: Option<ApiResponseFormat>) -> ResponseTarget {
    ResponseTarget {
        client: ResponseClient::Api,
        format: match response_format.unwrap_or(ApiResponseFormat::PlainText) {
            ApiResponseFormat::PlainText => ResponseFormat::PlainText,
            ApiResponseFormat::Markdown => ResponseFormat::Markdown,
        },
    }
}

async fn submit_approval<R, S>(
    State(state): State<ApiState<R, S>>,
    Path((session_id, approval_id)): Path<(SessionId, crate::types::ApprovalId)>,
    Json(request): Json<SubmitApprovalRequest>,
) -> Result<Json<crate::types::ApprovalRequestRecord>, ApiError>
where
    R: TurnRunner + Send + Sync + 'static,
    S: ConversationStore + Send + Sync + 'static,
{
    Ok(Json(
        state
            .service
            .submit_approval(&session_id, &approval_id, request)
            .await?,
    ))
}

async fn stream_events<R, S>(
    State(state): State<ApiState<R, S>>,
    Path(session_id): Path<SessionId>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, ApiError>
where
    R: TurnRunner + Send + Sync + 'static,
    S: ConversationStore + Send + Sync + 'static,
{
    if state.service.get_session(&session_id).await.is_none() {
        return Err(ApiError::NotFound("session not found".to_owned()));
    }

    let stream = BroadcastStream::new(state.service.subscribe()).filter_map(move |result| {
        let session_id = session_id.clone();
        async move {
            match result {
                Ok(event) if event_matches_session(&event, &session_id) => {
                    // Session events already have a typed schema, so SSE just
                    // carries their JSON representation verbatim.
                    let json = serde_json::to_string(&event).expect("session events serialize");
                    Some(Ok(Event::default().data(json)))
                }
                _ => None,
            }
        }
    });

    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

async fn handle_slack_event<R, S>(
    State(state): State<ApiState<R, S>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<impl IntoResponse, ApiError>
where
    R: TurnRunner + Send + Sync + 'static,
    S: ConversationStore + Send + Sync + 'static,
{
    let parsed = state.slack_adapter.ingest_http_request(&headers, &body)?;
    match parsed {
        SlackIngressResult::UrlVerification { challenge } => {
            Ok(Json(serde_json::json!({ "challenge": challenge })).into_response())
        }
        SlackIngressResult::AckOnly => Ok(Json(serde_json::json!({ "ok": true })).into_response()),
        SlackIngressResult::Dispatch(request) => {
            let queue = state
                .slack_queue
                .clone()
                .ok_or(ApiError::ServiceUnavailable(
                    "slack connector is not configured".to_owned(),
                ))?;
            queue.try_send(request).map_err(|error| match error {
                mpsc::error::TrySendError::Full(_) => ApiError::ServiceUnavailable(
                    "slack event queue is full; retry later".to_owned(),
                ),
                mpsc::error::TrySendError::Closed(_) => {
                    ApiError::ServiceUnavailable("slack event queue is unavailable".to_owned())
                }
            })?;
            Ok(Json(serde_json::json!({ "ok": true })).into_response())
        }
    }
}

async fn handle_whatsapp_event<R, S>(
    State(state): State<ApiState<R, S>>,
    Json(payload): Json<WhatsAppWebhookPayload>,
) -> Result<Json<ReceiveWhatsAppMessageResponse>, ApiError>
where
    R: TurnRunner + Send + Sync + 'static,
    S: ConversationStore + Send + Sync + 'static,
{
    handle_whatsapp_message(
        state,
        ReceiveWhatsAppMessageRequest {
            message_id: payload.message_id,
            account_id: None,
            conversation_id: payload.conversation_id,
            from_user_id: payload.from_user_id,
            text: payload.text,
            quoted_message_id: None,
            quoted_text: None,
        },
    )
    .await
}

async fn handle_whatsapp_inbound<R, S>(
    State(state): State<ApiState<R, S>>,
    Json(payload): Json<ReceiveWhatsAppMessageRequest>,
) -> Result<Json<ReceiveWhatsAppMessageResponse>, ApiError>
where
    R: TurnRunner + Send + Sync + 'static,
    S: ConversationStore + Send + Sync + 'static,
{
    handle_whatsapp_message(state, payload).await
}

async fn get_whatsapp_status<R, S>(
    State(state): State<ApiState<R, S>>,
) -> Result<Json<WhatsAppGatewayStatus>, ApiError>
where
    R: TurnRunner + Send + Sync + 'static,
    S: ConversationStore + Send + Sync + 'static,
{
    let gateway = state
        .whatsapp_gateway
        .as_ref()
        .ok_or(ApiError::ServiceUnavailable(
            "whatsapp gateway is not configured".to_owned(),
        ))?;
    Ok(Json(gateway.status().await))
}

async fn start_whatsapp_login<R, S>(
    State(state): State<ApiState<R, S>>,
) -> Result<Json<StartWhatsAppLoginResponse>, ApiError>
where
    R: TurnRunner + Send + Sync + 'static,
    S: ConversationStore + Send + Sync + 'static,
{
    let gateway = state
        .whatsapp_gateway
        .as_ref()
        .ok_or(ApiError::ServiceUnavailable(
            "whatsapp gateway is not configured".to_owned(),
        ))?;
    Ok(Json(gateway.start_login().await?))
}

async fn complete_whatsapp_login<R, S>(
    State(state): State<ApiState<R, S>>,
    Json(request): Json<CompleteWhatsAppLoginRequest>,
) -> Result<Json<WhatsAppGatewayStatus>, ApiError>
where
    R: TurnRunner + Send + Sync + 'static,
    S: ConversationStore + Send + Sync + 'static,
{
    let gateway = state
        .whatsapp_gateway
        .as_ref()
        .ok_or(ApiError::ServiceUnavailable(
            "whatsapp gateway is not configured".to_owned(),
        ))?;
    gateway.complete_login(&request.login_session_id).await?;
    Ok(Json(gateway.status().await))
}

async fn logout_whatsapp<R, S>(
    State(state): State<ApiState<R, S>>,
) -> Result<Json<WhatsAppGatewayStatus>, ApiError>
where
    R: TurnRunner + Send + Sync + 'static,
    S: ConversationStore + Send + Sync + 'static,
{
    let gateway = state
        .whatsapp_gateway
        .as_ref()
        .ok_or(ApiError::ServiceUnavailable(
            "whatsapp gateway is not configured".to_owned(),
        ))?;
    Ok(Json(gateway.logout().await?))
}

async fn handle_whatsapp_message<R, S>(
    state: ApiState<R, S>,
    payload: ReceiveWhatsAppMessageRequest,
) -> Result<Json<ReceiveWhatsAppMessageResponse>, ApiError>
where
    R: TurnRunner + Send + Sync + 'static,
    S: ConversationStore + Send + Sync + 'static,
{
    if let Some(gateway) = state.whatsapp_gateway.as_ref() {
        return Ok(Json(
            gateway
                .dispatch_inbound(state.service.as_ref(), payload)
                .await?,
        ));
    }

    let envelope = state.whatsapp_adapter.ingest(WhatsAppWebhookPayload {
        message_id: payload.message_id,
        conversation_id: payload.conversation_id,
        from_user_id: payload.from_user_id,
        text: payload.text,
    })?;
    let dispatch = state.service.dispatch_envelope(envelope).await?;
    Ok(Json(ReceiveWhatsAppMessageResponse {
        session: dispatch.session,
        queued_outbound: 0,
        was_duplicate: dispatch.was_duplicate,
    }))
}

#[derive(Clone, Debug, Default, serde::Deserialize)]
struct DebugHistorySessionsQuery {
    limit: Option<usize>,
}

#[derive(Clone, Debug, Default, serde::Deserialize)]
struct DebugHistoryTurnQuery {
    event_types: Option<String>,
    artifact_kinds: Option<String>,
    executors: Option<String>,
    steps: Option<String>,
}

async fn debug_history_page<R, S>(State(_state): State<ApiState<R, S>>) -> Html<String>
where
    R: TurnRunner + Send + Sync + 'static,
    S: ConversationStore + Send + Sync + 'static,
{
    Html(
        DEBUG_HISTORY_PAGE
            .replace("{{ARKA_LOGO}}", ARKA_LOGO_SVG)
            .replace("{{ARKA_FAVICON}}", &svg_to_data_uri(ARKA_LOGO_SVG)),
    )
}

fn svg_to_data_uri(svg: &str) -> String {
    let mut encoded = String::with_capacity(svg.len() * 2);
    for ch in svg.trim().chars() {
        match ch {
            '%' => encoded.push_str("%25"),
            '#' => encoded.push_str("%23"),
            '"' => encoded.push_str("%22"),
            '<' => encoded.push_str("%3C"),
            '>' => encoded.push_str("%3E"),
            '?' => encoded.push_str("%3F"),
            '&' => encoded.push_str("%26"),
            '\n' => encoded.push_str("%0A"),
            '\r' => {}
            _ => encoded.push(ch),
        }
    }
    format!("data:image/svg+xml,{encoded}")
}

async fn list_debug_history_sessions<R, S>(
    State(state): State<ApiState<R, S>>,
    Query(query): Query<DebugHistorySessionsQuery>,
) -> Result<Json<Vec<crate::observability::DebugHistorySessionSummary>>, ApiError>
where
    R: TurnRunner + Send + Sync + 'static,
    S: ConversationStore + Send + Sync + 'static,
{
    let store = state
        .debug_history
        .as_ref()
        .ok_or(ApiError::NotFound("debug history unavailable".to_owned()))?;
    Ok(Json(
        store
            .list_recent_sessions(query.limit.unwrap_or(50))
            .await?,
    ))
}

async fn list_debug_history_session_turns<R, S>(
    State(state): State<ApiState<R, S>>,
    Path(session_id): Path<String>,
) -> Result<Json<Vec<crate::observability::DebugHistoryTurnSummary>>, ApiError>
where
    R: TurnRunner + Send + Sync + 'static,
    S: ConversationStore + Send + Sync + 'static,
{
    let store = state
        .debug_history
        .as_ref()
        .ok_or(ApiError::NotFound("debug history unavailable".to_owned()))?;
    let session_id = Uuid::parse_str(&session_id)
        .map_err(|_| ApiError::BadRequest("invalid session id".to_owned()))?;
    Ok(Json(store.list_session_turns(session_id).await?))
}

async fn get_debug_history_turn<R, S>(
    State(state): State<ApiState<R, S>>,
    Path(turn_id): Path<String>,
    Query(query): Query<DebugHistoryTurnQuery>,
) -> Result<Json<crate::observability::DebugHistoryTurnDetail>, ApiError>
where
    R: TurnRunner + Send + Sync + 'static,
    S: ConversationStore + Send + Sync + 'static,
{
    let store = state
        .debug_history
        .as_ref()
        .ok_or(ApiError::NotFound("debug history unavailable".to_owned()))?;
    let turn_id = Uuid::parse_str(&turn_id)
        .map_err(|_| ApiError::BadRequest("invalid turn id".to_owned()))?;
    let filters = parse_debug_history_turn_filters(query)?;
    let detail = store
        .get_turn_detail(turn_id, &filters)
        .await?
        .ok_or(ApiError::NotFound("debug turn not found".to_owned()))?;
    Ok(Json(detail))
}

fn parse_debug_history_turn_filters(
    query: DebugHistoryTurnQuery,
) -> Result<crate::observability::DebugHistoryTurnFilters, ApiError> {
    let event_types = parse_csv_filter_values(query.event_types);
    let artifact_kinds = parse_csv_filter_values(query.artifact_kinds);
    let executors = parse_csv_filter_values(query.executors);
    let steps = parse_csv_filter_values(query.steps);
    for step in &steps {
        if step == "none" {
            continue;
        }
        let parsed = step
            .parse::<u32>()
            .map_err(|_| ApiError::BadRequest(format!("invalid step filter `{step}`")))?;
        if parsed == 0 {
            return Err(ApiError::BadRequest(
                "step filter must be a positive integer or `none`".to_owned(),
            ));
        }
    }
    Ok(crate::observability::DebugHistoryTurnFilters {
        event_types,
        artifact_kinds,
        executors,
        steps,
    })
}

fn parse_csv_filter_values(raw: Option<String>) -> std::collections::BTreeSet<String> {
    raw.unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn event_matches_session(event: &SessionEvent, session_id: &SessionId) -> bool {
    match event {
        SessionEvent::SessionCreated { session } => &session.session_id == session_id,
        SessionEvent::SessionMessageReceived { session_id: id, .. } => id == session_id,
        SessionEvent::TurnQueued { session_id: id } => id == session_id,
        SessionEvent::TurnStarted { session_id: id } => id == session_id,
        SessionEvent::TurnCompleted { session_id: id, .. } => id == session_id,
        SessionEvent::ApprovalRequested { session_id: id, .. } => id == session_id,
        SessionEvent::ApprovalResolved { session_id: id, .. } => id == session_id,
        SessionEvent::RuntimeEvent { session_id: id, .. } => id == session_id,
        SessionEvent::ChannelDeliveryAttempted { session_id: id, .. } => id == session_id,
    }
}

/// API-level error mapping.
#[derive(Debug)]
pub enum ApiError {
    NotFound(String),
    BadRequest(String),
    ServiceUnavailable(String),
    Internal(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let (status, message) = match self {
            Self::NotFound(message) => (axum::http::StatusCode::NOT_FOUND, message),
            Self::BadRequest(message) => (axum::http::StatusCode::BAD_REQUEST, message),
            Self::ServiceUnavailable(message) => {
                (axum::http::StatusCode::SERVICE_UNAVAILABLE, message)
            }
            Self::Internal(message) => (axum::http::StatusCode::INTERNAL_SERVER_ERROR, message),
        };
        (status, Json(serde_json::json!({ "error": message }))).into_response()
    }
}

impl From<ControlPlaneError> for ApiError {
    fn from(value: ControlPlaneError) -> Self {
        match value {
            ControlPlaneError::SessionNotFound | ControlPlaneError::ApprovalNotFound => {
                Self::NotFound(value.to_string())
            }
            ControlPlaneError::ApprovalNotPending => Self::BadRequest(value.to_string()),
            ControlPlaneError::Store(_) => Self::Internal(value.to_string()),
            ControlPlaneError::Runner(_) => Self::Internal(value.to_string()),
        }
    }
}

impl From<crate::adapters::ChannelAdapterError> for ApiError {
    fn from(value: crate::adapters::ChannelAdapterError) -> Self {
        match value {
            ChannelAdapterError::InvalidPayload(_) => Self::BadRequest(value.to_string()),
            ChannelAdapterError::NotConfigured(_) => Self::ServiceUnavailable(value.to_string()),
        }
    }
}

impl From<RuntimeHarnessListenerError> for ApiError {
    fn from(value: RuntimeHarnessListenerError) -> Self {
        Self::Internal(value.to_string())
    }
}

impl From<WhatsAppGatewayError> for ApiError {
    fn from(value: WhatsAppGatewayError) -> Self {
        match value {
            WhatsAppGatewayError::NotReady => Self::ServiceUnavailable(value.to_string()),
            WhatsAppGatewayError::SenderBlocked(_)
            | WhatsAppGatewayError::LoginSessionNotFound
            | WhatsAppGatewayError::InvalidPayload(_)
            | WhatsAppGatewayError::UnsupportedOperation(_) => Self::BadRequest(value.to_string()),
            WhatsAppGatewayError::Delivery(_) => Self::ServiceUnavailable(value.to_string()),
            WhatsAppGatewayError::Io { .. }
            | WhatsAppGatewayError::Serialize { .. }
            | WhatsAppGatewayError::Deserialize { .. } => Self::Internal(value.to_string()),
        }
    }
}

async fn run_slack_worker<R, S>(
    service: Arc<SessionService<R, S>>,
    adapter: SlackChannelAdapter,
    delivery_client: Arc<dyn SlackDeliveryClient>,
    mut rx: mpsc::Receiver<SlackDispatchRequest>,
) where
    R: TurnRunner + Send + Sync + 'static,
    S: ConversationStore + Send + Sync + 'static,
{
    while let Some(request) = rx.recv().await {
        if request.requires_existing_session
            && service
                .find_session_by_binding(&request.binding)
                .await
                .is_none()
        {
            continue;
        }

        if let Some(text) = request.immediate_text.clone() {
            if let Err(error) = deliver_slack_text(
                &service,
                &adapter,
                delivery_client.as_ref(),
                &request.delivery_target,
                None,
                text,
            )
            .await
            {
                warn!(error = %error, "slack immediate reply delivery failed");
            }
            continue;
        }

        let mut stream_listener = None;
        let mut stream_task = None;
        if matches!(
            &request.envelope.intent,
            crate::types::ChannelIntent::UserText { .. }
        ) {
            let initial_markdown = format!("<@{}> ", request.delivery_target.external_user_id);
            match delivery_client
                .start_stream(&request.delivery_target, &initial_markdown)
                .await
            {
                Ok(handle) => {
                    let (listener, task) =
                        spawn_slack_stream_listener(delivery_client.clone(), handle);
                    stream_listener = Some(listener);
                    stream_task = Some(task);
                }
                Err(error) => {
                    warn!(error = %error, "slack stream start failed; falling back to final reply");
                }
            }
        }

        let extra_runtime_harness_listeners = stream_listener
            .as_ref()
            .map(|listener| vec![Arc::clone(listener)])
            .unwrap_or_default();
        let dispatch = match service
            .dispatch_envelope_with_runtime_listeners(
                request.envelope.clone(),
                extra_runtime_harness_listeners,
            )
            .await
        {
            Ok(dispatch) => dispatch,
            Err(error) => {
                drop(stream_listener.take());
                if let Some(task) = stream_task.take() {
                    let _ = task.await;
                }
                warn!(error = %error, "slack dispatch failed");
                continue;
            }
        };
        drop(stream_listener.take());
        let stream_result = match stream_task.take() {
            Some(task) => match task.await {
                Ok(Ok(result)) => Some(result),
                Ok(Err(error)) => {
                    warn!(error = %error, "slack stream delivery failed");
                    service.emit_channel_delivery(
                        dispatch.session.session_id.clone(),
                        ChannelKind::Slack,
                        DeliveryStatus::Failed,
                    );
                    None
                }
                Err(error) => {
                    warn!(error = %error, "slack stream worker task failed");
                    service.emit_channel_delivery(
                        dispatch.session.session_id.clone(),
                        ChannelKind::Slack,
                        DeliveryStatus::Failed,
                    );
                    None
                }
            },
            None => None,
        };

        let stream_replaced_reply = stream_result.is_some();

        if stream_replaced_reply {
            service.emit_channel_delivery(
                dispatch.session.session_id.clone(),
                ChannelKind::Slack,
                DeliveryStatus::Sent,
            );
        }

        for outbound in dispatch.outbound {
            if matches!(
                outbound.kind,
                crate::types::OutboundMessageKind::Reply | crate::types::OutboundMessageKind::Error
            ) && stream_replaced_reply
            {
                continue;
            }
            let target = outbound
                .delivery_target
                .clone()
                .unwrap_or_else(|| request.delivery_target.clone());
            if let Err(error) = deliver_slack_text(
                &service,
                &adapter,
                delivery_client.as_ref(),
                &target,
                Some(dispatch.session.session_id.clone()),
                outbound.text.clone(),
            )
            .await
            {
                warn!(error = %error, "slack outbound delivery failed");
                service.emit_channel_delivery(
                    dispatch.session.session_id.clone(),
                    ChannelKind::Slack,
                    DeliveryStatus::Failed,
                );
            }
        }
    }
}

#[derive(Debug)]
struct SlackRuntimeStreamListener {
    tx: mpsc::Sender<RuntimeHarnessObservation>,
}

impl RuntimeHarnessListener for SlackRuntimeStreamListener {
    fn try_observe(
        &self,
        observation: RuntimeHarnessObservation,
    ) -> Result<(), RuntimeHarnessListenerError> {
        self.tx.try_send(observation).map_err(|error| match error {
            mpsc::error::TrySendError::Full(_) => RuntimeHarnessListenerError::QueueFull,
            mpsc::error::TrySendError::Closed(_) => RuntimeHarnessListenerError::Closed,
        })
    }
}

fn spawn_slack_stream_listener(
    delivery_client: Arc<dyn SlackDeliveryClient>,
    handle: SlackStreamHandle,
) -> (
    Arc<dyn RuntimeHarnessListener>,
    tokio::task::JoinHandle<Result<SlackStreamTaskResult, SlackDeliveryError>>,
) {
    let (tx, mut rx) = mpsc::channel(256);
    let listener: Arc<dyn RuntimeHarnessListener> = Arc::new(SlackRuntimeStreamListener { tx });
    let task = tokio::spawn(async move {
        let mut pending = String::new();
        let mut streamed_text = false;
        let mut stopped = false;

        while let Some(observation) = rx.recv().await {
            let RuntimeHarnessObservation::Event(envelope) = observation else {
                continue;
            };
            match envelope.event {
                RuntimeEvent::AnswerTextDelta { delta, .. } => {
                    pending.push_str(&delta);
                    streamed_text = true;
                    if should_flush_slack_stream_buffer(&pending) {
                        delivery_client.append_stream(&handle, &pending).await?;
                        pending.clear();
                    }
                }
                RuntimeEvent::TurnEnded { termination, .. } => {
                    if !pending.is_empty() {
                        delivery_client.append_stream(&handle, &pending).await?;
                        pending.clear();
                    }
                    let final_note = if matches!(termination, TerminationReason::Final) {
                        None
                    } else {
                        streamed_text = true;
                        Some("\n\n_Arka hit an error while replying._")
                    };
                    delivery_client.stop_stream(&handle, final_note).await?;
                    stopped = true;
                    break;
                }
                _ => {}
            }
        }

        if !stopped {
            if !pending.is_empty() {
                delivery_client.append_stream(&handle, &pending).await?;
            }
            delivery_client.stop_stream(&handle, None).await?;
        }

        Ok(SlackStreamTaskResult { streamed_text })
    });
    (listener, task)
}

fn should_flush_slack_stream_buffer(buffer: &str) -> bool {
    buffer.len() >= 320
        || buffer.ends_with('\n')
        || buffer.ends_with(". ")
        || buffer.ends_with("? ")
        || buffer.ends_with("! ")
}

async fn deliver_slack_text<R, S>(
    service: &SessionService<R, S>,
    adapter: &SlackChannelAdapter,
    delivery_client: &dyn SlackDeliveryClient,
    target: &ChannelDeliveryTarget,
    session_id: Option<SessionId>,
    text: String,
) -> Result<(), SlackDeliveryError>
where
    R: TurnRunner + Send + Sync + 'static,
    S: ConversationStore + Send + Sync + 'static,
{
    let outbound = crate::types::ChannelResponseEnvelope {
        session_id: session_id.clone().unwrap_or_default(),
        kind: crate::types::OutboundMessageKind::Reply,
        text,
        delivery_target: Some(target.clone()),
    };
    let rendered = adapter
        .render(&outbound)
        .map_err(|error| SlackDeliveryError::Api(error.to_string()))?;
    delivery_client.post_message(target, &rendered).await?;
    if let Some(session_id) = session_id {
        service.emit_channel_delivery(session_id, ChannelKind::Slack, DeliveryStatus::Sent);
    }
    Ok(())
}

const ARKA_LOGO_SVG: &str = include_str!("../../../assets/arka-logo.svg");

const DEBUG_HISTORY_PAGE: &str = r###"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Debug History</title>
  <link rel="icon" type="image/svg+xml" href="{{ARKA_FAVICON}}" />
  <style>
    :root {
      --bg: #f6f1e8;
      --paper: #fffaf0;
      --ink: #1c1b18;
      --muted: #6f685c;
      --line: #d8cdbd;
      --accent: #0f5b50;
      --accent-soft: #dff2ee;
      --warn: #8b3d2f;
      --good: #245c35;
      --shadow: 0 10px 30px rgba(28, 27, 24, 0.08);
      --mono: "SFMono-Regular", ui-monospace, Menlo, monospace;
      --sans: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: var(--sans);
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15, 91, 80, 0.12), transparent 32%),
        radial-gradient(circle at top right, rgba(139, 61, 47, 0.10), transparent 28%),
        linear-gradient(180deg, #fbf7f0 0%, var(--bg) 100%);
    }
    .page {
      max-width: 1200px;
      margin: 0 auto;
      padding: 28px 20px 60px;
    }
    .hero {
      background: linear-gradient(135deg, rgba(255,250,240,0.94), rgba(248,240,227,0.96));
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 24px;
      box-shadow: var(--shadow);
      margin-bottom: 22px;
    }
    .brand-lockup {
      display: flex;
      gap: 18px;
      align-items: center;
      margin-bottom: 18px;
    }
    .brand-mark {
      flex: 0 0 auto;
      width: clamp(148px, 20vw, 220px);
    }
    .brand-mark svg {
      display: block;
      width: 100%;
      height: auto;
    }
    .brand-copy {
      display: grid;
      gap: 6px;
    }
    .eyebrow {
      font-size: 0.78rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--accent);
    }
    .hero h1 {
      margin: 0 0 8px;
      font-size: clamp(2rem, 4vw, 3rem);
      line-height: 1;
      letter-spacing: -0.04em;
    }
    .hero p { margin: 0; color: var(--muted); font-size: 1rem; }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      justify-content: space-between;
      margin: 18px 0;
    }
    .crumbs {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      font-size: 0.95rem;
      color: var(--muted);
    }
    .crumbs a {
      color: var(--accent);
      text-decoration: none;
      border-bottom: 1px solid transparent;
    }
    .crumbs a:hover { border-color: currentColor; }
    button, .button {
      border: 1px solid var(--line);
      background: var(--paper);
      color: var(--ink);
      border-radius: 999px;
      padding: 10px 14px;
      font: inherit;
      cursor: pointer;
      box-shadow: 0 3px 10px rgba(28, 27, 24, 0.05);
    }
    button.primary, .button.primary {
      background: var(--accent);
      color: white;
      border-color: transparent;
    }
    input[type="number"] {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 10px 14px;
      background: var(--paper);
      font: inherit;
      width: 90px;
    }
    .panel, .card, .event {
      background: rgba(255,250,240,0.88);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: var(--shadow);
    }
    .panel {
      padding: 18px;
      margin-bottom: 16px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 12px;
    }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px 18px;
      margin-bottom: 16px;
    }
    .summary-block {
      min-width: 0;
    }
    .summary-stack {
      display: grid;
      gap: 10px;
    }
    .summary-copy {
      display: grid;
      gap: 12px;
      margin-top: 8px;
    }
    .summary-copy-block {
      padding-top: 12px;
      border-top: 1px dashed var(--line);
    }
    .summary-text {
      font-size: 1rem;
      line-height: 1.55;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .card {
      padding: 16px;
      transition: transform 180ms ease, border-color 180ms ease;
    }
    .card:hover {
      transform: translateY(-2px);
      border-color: var(--accent);
    }
    .label {
      font-size: 0.76rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 6px;
    }
    .value {
      font-size: 1.02rem;
      line-height: 1.35;
      word-break: break-word;
    }
    .meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }
    .tag {
      border-radius: 999px;
      padding: 4px 10px;
      background: rgba(15, 91, 80, 0.10);
      color: var(--accent);
      font-size: 0.82rem;
    }
    .tag.warn { background: rgba(139, 61, 47, 0.10); color: var(--warn); }
    .tag.good { background: rgba(36, 92, 53, 0.10); color: var(--good); }
    .list {
      display: grid;
      gap: 14px;
    }
    .step-group {
      margin: 0;
      border: 1px solid rgba(15, 91, 80, 0.18);
      border-radius: 18px;
      background: rgba(15, 91, 80, 0.05);
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.55);
      overflow: hidden;
    }
    .step-group summary {
      list-style: none;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 16px;
      cursor: pointer;
      border-top: 0;
    }
    .step-group summary::-webkit-details-marker {
      display: none;
    }
    .step-group summary::before {
      content: "▸";
      color: var(--accent);
      font-size: 0.9rem;
      transform: translateY(-1px);
    }
    .step-group[open] summary::before {
      content: "▾";
    }
    .step-group-body {
      display: grid;
      gap: 12px;
      padding: 0 14px 14px;
      border-top: 1px dashed rgba(15, 91, 80, 0.18);
    }
    .step-group-head {
      display: flex;
      flex-wrap: wrap;
      align-items: baseline;
      gap: 10px;
    }
    .step-group-title {
      font-size: 1rem;
      font-weight: 700;
      color: var(--ink);
    }
    .step-group-sub {
      color: var(--muted);
      font-size: 0.9rem;
    }
    .step-group-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: flex-end;
      margin-left: auto;
    }
    .event {
      padding: 16px;
      display: grid;
      gap: 12px;
    }
    .merged-items {
      display: grid;
      gap: 12px;
    }
    .merged-item {
      display: grid;
      gap: 8px;
    }
    .merged-item + .merged-item {
      border-top: 1px dashed var(--line);
      padding-top: 12px;
    }
    .merged-phase-list {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .merged-phase-blocks {
      display: grid;
      gap: 10px;
      margin-top: 10px;
    }
    .event-head {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: baseline;
      justify-content: space-between;
    }
    .event-title {
      font-size: 1.05rem;
      font-weight: 600;
    }
    .event-sub {
      color: var(--muted);
      font-size: 0.92rem;
    }
    details {
      border-top: 1px dashed var(--line);
      padding-top: 10px;
    }
    summary {
      cursor: pointer;
      color: var(--accent);
      font-weight: 600;
    }
    pre {
      margin: 10px 0 0;
      background: #1b1a17;
      color: #f8f3e8;
      border-radius: 14px;
      padding: 14px;
      overflow-x: auto;
      overflow-y: auto;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      word-break: break-word;
      font-family: var(--mono);
      font-size: 0.84rem;
      line-height: 1.45;
    }
    .muted { color: var(--muted); }
    .empty, .error {
      padding: 24px;
      text-align: center;
      color: var(--muted);
    }
    .error { color: var(--warn); }
    .inline-actions { display: flex; gap: 8px; flex-wrap: wrap; }
    .filters-panel {
      display: grid;
      gap: 14px;
      background: rgba(15, 91, 80, 0.04);
      border-style: solid;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.55);
    }
    .filters-kicker {
      font-size: 0.74rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 6px;
    }
    .filters-head {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      justify-content: space-between;
    }
    .filters-summary {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      padding: 10px 12px;
      border: 1px dashed var(--line);
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.42);
    }
    .filters-body[hidden] {
      display: none;
    }
    .filters-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }
    .filters-button {
      padding: 8px 12px;
      box-shadow: none;
      background: rgba(255, 255, 255, 0.72);
    }
    .filters-button.primary {
      background: rgba(15, 91, 80, 0.10);
      color: var(--accent);
      border-color: rgba(15, 91, 80, 0.18);
    }
    .filter-tag {
      border-radius: 999px;
      padding: 5px 10px;
      background: rgba(15, 91, 80, 0.09);
      color: var(--accent);
      font-size: 0.84rem;
      border: 1px solid rgba(15, 91, 80, 0.10);
    }
    .filter-grid {
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }
    .filter-group {
      margin: 0;
      padding: 0;
      border: 0;
      min-width: 0;
    }
    .filter-group legend {
      padding: 0;
      margin-bottom: 8px;
      font-size: 0.8rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .filter-options {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .filter-option {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: var(--paper);
      cursor: pointer;
      font-size: 0.92rem;
    }
    .filter-option input {
      margin: 0;
      accent-color: var(--accent);
    }
    .filter-empty {
      color: var(--muted);
      font-size: 0.92rem;
    }
    @media (max-width: 720px) {
      .page { padding-inline: 14px; }
      .hero { padding: 18px; border-radius: 18px; }
      .panel, .card, .event { border-radius: 16px; }
      .brand-lockup { align-items: flex-start; flex-direction: column; }
    }
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <div class="brand-lockup">
        <div class="brand-mark" aria-hidden="true">{{ARKA_LOGO}}</div>
        <div class="brand-copy">
          <div class="eyebrow">Arka Control Plane</div>
          <h1>Debug History</h1>
          <p>Browse persisted runtime debug turns, harness events, and raw transport artifacts.</p>
        </div>
      </div>
    </section>
    <div class="toolbar">
      <nav class="crumbs" id="crumbs"></nav>
      <div class="inline-actions">
        <label>
          <span class="muted">Recent limit</span>
          <input id="limitInput" type="number" min="1" max="200" value="50" />
        </label>
        <button id="refreshButton" class="primary">Refresh</button>
      </div>
    </div>
    <section id="content" class="list"></section>
  </main>
  <script>
    const content = document.getElementById("content");
    const crumbs = document.getElementById("crumbs");
    const limitInput = document.getElementById("limitInput");
    const refreshButton = document.getElementById("refreshButton");

    refreshButton.addEventListener("click", () => renderRoute());
    window.addEventListener("hashchange", renderRoute);

    function routeState() {
      const rawHash = window.location.hash.replace(/^#/, "");
      const [pathPart, queryPart = ""] = rawHash.split("?");
      const cleanedPath = pathPart.replace(/^\/?/, "");
      return {
        parts: cleanedPath ? cleanedPath.split("/") : [],
        params: new URLSearchParams(queryPart),
      };
    }

    function buildHash(parts, params = new URLSearchParams()) {
      const path = `#/${parts.map((part) => encodeURIComponent(part)).join("/")}`;
      const query = params.toString();
      return query ? `${path}?${query}` : path;
    }

    function navigateToHash(nextHash) {
      if (window.location.hash === nextHash) {
        renderRoute();
        return;
      }
      window.location.hash = nextHash;
    }

    function setCrumbs(items) {
      crumbs.innerHTML = items.map((item, index) => {
        if (item.href) {
          return `<a href="${item.href}">${escapeHtml(item.label)}</a>`;
        }
        return `<span>${escapeHtml(item.label)}</span>`;
      }).join("<span> / </span>");
    }

    function renderEmpty(message) {
      content.innerHTML = `<div class="panel empty">${escapeHtml(message)}</div>`;
    }

    function renderError(message) {
      content.innerHTML = `<div class="panel error">${escapeHtml(message)}</div>`;
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    function formatTimestamp(value) {
      if (!value) return "n/a";
      const date = new Date(value);
      return Number.isNaN(date.getTime())
        ? value
        : date.toLocaleString([], {
            year: "numeric",
            month: "2-digit",
            day: "2-digit",
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
            fractionalSecondDigits: 3,
            hour12: false,
          });
    }

    function formatUsage(usage) {
      if (!usage) return "n/a";
      return `${usage.input_tokens ?? 0} in / ${usage.cached_tokens ?? 0} cached / ${usage.output_tokens ?? 0} out / ${usage.total_tokens ?? 0} total`;
    }

    function formatLatency(payload) {
      if (!payload || !payload.latency) return null;
      const secs = Number(payload.latency.secs ?? 0);
      const nanos = Number(payload.latency.nanos ?? 0);
      const ms = Math.round((secs * 1000) + (nanos / 1_000_000));
      return `${ms}ms`;
    }

    function legacyEventVariantKey(eventType) {
      return String(eventType || "")
        .split("_")
        .filter(Boolean)
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
        .join("");
    }

    function normalizeEventPayload(event) {
      const payload = event?.payload || {};
      if (!payload || Array.isArray(payload) || typeof payload !== "object") return payload;
      const legacyKey = legacyEventVariantKey(event?.event_type);
      const wrapped = payload[legacyKey];
      if (wrapped && typeof wrapped === "object" && !Array.isArray(wrapped)) {
        return wrapped;
      }
      return payload;
    }

    function formatExecutor(executor) {
      if (!executor) return null;
      if (executor.kind === "subagent") {
        return executor.subagent_type
          ? `${executor.display_name} (${executor.subagent_type})`
          : executor.display_name;
      }
      return executor.display_name || "Main Agent";
    }

    function describeEvent(event) {
      const payload = normalizeEventPayload(event);
      switch (event.event_type) {
        case "turn_started": return "Turn execution started";
        case "step_started": return `Step ${payload.step_number ?? "?"} started`;
        case "prompt_built": return "Prompt snapshot assembled";
        case "handoff_to_subagent":
          return `Handing off to sub-agent${payload.subagent_type ? ` (${payload.subagent_type})` : ""}`;
        case "model_called": return "Querying model";
        case "model_responded":
          return `Model responded${formatLatency(payload) ? ` in ${formatLatency(payload)}` : ""}`;
        case "handoff_to_main_agent":
          return `Returning to main agent${payload.subagent_type ? ` from ${payload.subagent_type}` : ""}${payload.status ? ` (${payload.status})` : ""}`;
        case "mcp_called":
          return `MCP tool call: ${payload.server_name ?? "server"}/${payload.tool_name ?? "tool"}`;
        case "mcp_responded":
          return `${payload.was_error ? "MCP error" : "MCP response"} from ${payload.server_name ?? "server"}/${payload.tool_name ?? "tool"}`;
        case "step_ended": return "Step ended";
        case "turn_ended":
          return `Turn ended (${payload.termination ?? "unknown"})`;
        default:
          return event.event_type;
      }
    }

    function describeArtifact(artifact) {
      switch (artifact.artifact_kind) {
        case "model_request": return "Model request payload";
        case "model_response": return "Model response payload";
        case "model_error": return "Model transport/provider error";
        case "mcp_request": return "MCP request payload";
        case "mcp_response": return "MCP response payload";
        case "mcp_error": return "MCP error payload";
        default:
          return artifact.artifact_kind;
      }
    }

    function executorScopeKey(executor) {
      if (!executor) return "main";
      if (executor.kind === "subagent") {
        return `subagent:${executor.display_name || ""}:${executor.subagent_type || ""}`;
      }
      return "main";
    }

    function timelineItemContext(item) {
      if (item.type === "group") {
        const first = item.items?.[0];
        return first ? timelineItemContext(first) : {
          turn_id: "",
          step_number: "",
          step_id: "",
          executor: null,
        };
      }
      if (item.type === "event") {
        const payload = normalizeEventPayload(item.event);
        return {
          turn_id: item.event.turn_id || "",
          step_number: item.event.step_number ?? payload?.step_number ?? "",
          step_id: item.event.step_id ?? payload?.step_id ?? "",
          executor: item.event.executor || payload?.executor || null,
        };
      }
      return {
        turn_id: item.artifact.turn_id || "",
        step_number: item.artifact.step_number ?? "",
        step_id: item.artifact.step_id ?? "",
        executor: item.artifact.executor || null,
      };
    }

    function timelineItemGroupKind(item) {
      if (item.type === "event" && ["prompt_built", "model_called", "model_responded"].includes(item.event.event_type)) {
        return "model";
      }
      if (item.type === "artifact" && ["model_request", "model_response", "model_error"].includes(item.artifact.artifact_kind)) {
        return "model";
      }
      return null;
    }

    function timelineItemGroupKey(item) {
      const kind = timelineItemGroupKind(item);
      if (!kind) return null;
      const context = timelineItemContext(item);
      return [
        kind,
        context.turn_id,
        context.step_number,
        context.step_id,
        executorScopeKey(context.executor),
      ].join("|");
    }

    function describeTimelineGroup(group) {
      const hasRequestPhase = group.items.some((item) =>
        item.type === "event"
          ? ["prompt_built", "model_called"].includes(item.event.event_type)
          : item.artifact.artifact_kind === "model_request"
      );
      const hasResponsePhase = group.items.some((item) =>
        item.type === "event"
          ? item.event.event_type === "model_responded"
          : ["model_response", "model_error"].includes(item.artifact.artifact_kind)
      );
      if (hasRequestPhase && hasResponsePhase) return "Model interaction";
      if (hasRequestPhase) return "Model request";
      if (hasResponsePhase) return "Model response";
      return "Model activity";
    }

    function timelineItemPhase(item) {
      if (item.type === "event") {
        if (["prompt_built", "model_called"].includes(item.event.event_type)) return "request";
        if (item.event.event_type === "model_responded") return "response";
        return null;
      }
      if (item.type === "artifact") {
        if (item.artifact.artifact_kind === "model_request") return "request";
        if (["model_response", "model_error"].includes(item.artifact.artifact_kind)) return "response";
      }
      return null;
    }

    function phaseLabel(phase) {
      switch (phase) {
        case "request": return "Model request";
        case "response": return "Model response";
        default: return "Model activity";
      }
    }

    function describeTimelineGroupSubtitle(group) {
      const phases = Array.from(
        new Set(group.items.map((item) => timelineItemPhase(item)).filter(Boolean))
      );
      if (phases.length === 0) {
        return `${group.items.length} merged timeline item${group.items.length === 1 ? "" : "s"}`;
      }
      return `${phases.length} condensed phase${phases.length === 1 ? "" : "s"}`;
    }

    function renderGroupedTimelinePhase(phase, items) {
      if (!items.length) return "";
      const labels = Array.from(
        new Set(
          items.map((item) => item.type === "event" ? describeEvent(item.event) : describeArtifact(item.artifact))
        )
      );
      const detailsLabel = phase === "request" ? "Raw request JSON" : "Raw response JSON";
      const phaseOccurredAt = items[0]?.occurred_at;
      return `
        <section class="merged-item">
          <div class="event-head">
            <div>
              <div class="event-title">${escapeHtml(phaseLabel(phase))}</div>
              <div class="event-sub">${escapeHtml(labels.join(" · "))}</div>
            </div>
            <div class="event-sub">${escapeHtml(formatTimestamp(phaseOccurredAt))}</div>
          </div>
          <div class="merged-phase-list">
            ${items.map((item) => item.type === "event"
              ? `<span class="tag">event #${item.event.event_index}</span>`
              : `<span class="tag warn">artifact #${item.artifact.artifact_index}</span>`
            ).join("")}
          </div>
          <details>
            <summary>${escapeHtml(`${detailsLabel} (${items.length})`)}</summary>
            <div class="merged-phase-blocks">
              ${items.map((item) => item.type === "event"
                ? `
                  <section>
                    <div class="event-sub">${escapeHtml(`${item.event.event_type} · event #${item.event.event_index}`)}</div>
                    <pre>${escapeHtml(JSON.stringify(normalizeEventPayload(item.event), null, 2))}</pre>
                  </section>
                `
                : `
                  <section>
                    <div class="event-sub">${escapeHtml(`${item.artifact.artifact_kind} · artifact #${item.artifact.artifact_index}`)}</div>
                    <pre>${escapeHtml(JSON.stringify(item.artifact.payload, null, 2))}</pre>
                  </section>
                `
              ).join("")}
            </div>
          </details>
        </section>
      `;
    }

    function buildTimeline(turn) {
      const eventItems = (turn.events || []).map((event) => ({
        type: "event",
        order: event.event_index ?? 0,
        occurred_at: event.occurred_at,
        event,
      }));
      const artifactItems = (turn.raw_artifacts || []).map((artifact) => ({
        type: "artifact",
        order: artifact.artifact_index ?? 0,
        occurred_at: artifact.occurred_at,
        artifact,
      }));
      const sortedItems = [...eventItems, ...artifactItems].sort((left, right) => {
        const leftTs = Date.parse(left.occurred_at || "") || 0;
        const rightTs = Date.parse(right.occurred_at || "") || 0;
        if (leftTs !== rightTs) return leftTs - rightTs;
        if (left.type !== right.type) return left.type === "event" ? -1 : 1;
        return left.order - right.order;
      });

      const groupedItems = [];
      let pendingGroup = null;

      function flushPendingGroup() {
        if (!pendingGroup) return;
        if (pendingGroup.items.length > 1) {
          groupedItems.push(pendingGroup);
        } else {
          groupedItems.push(pendingGroup.items[0]);
        }
        pendingGroup = null;
      }

      for (const item of sortedItems) {
        const groupKey = timelineItemGroupKey(item);
        if (!groupKey) {
          flushPendingGroup();
          groupedItems.push(item);
          continue;
        }
        if (!pendingGroup || pendingGroup.group_key !== groupKey) {
          flushPendingGroup();
          pendingGroup = {
            type: "group",
            group_kind: timelineItemGroupKind(item),
            group_key: groupKey,
            occurred_at: item.occurred_at,
            items: [item],
          };
          continue;
        }
        pendingGroup.items.push(item);
      }

      flushPendingGroup();
      return groupedItems;
    }

    function renderEventItem(item) {
      const payload = normalizeEventPayload(item.event);
      const executor = formatExecutor(item.event.executor || payload?.executor);
      return `
        <article class="event">
          <div class="event-head">
            <div>
              <div class="event-title">${escapeHtml(describeEvent(item.event))}</div>
              <div class="event-sub">${escapeHtml(item.event.event_type)} · event #${item.event.event_index}</div>
            </div>
            <div class="event-sub">${escapeHtml(formatTimestamp(item.event.occurred_at))}</div>
          </div>
          <div class="meta">
            <span class="tag">event</span>
            ${executor ? `<span class="tag">${escapeHtml(executor)}</span>` : ""}
            ${item.event.turn_id ? `<span class="tag">turn ${escapeHtml(item.event.turn_id)}</span>` : ""}
            ${formatStepTag(item.event.step_number, item.event.step_id) ? `<span class="tag">${escapeHtml(formatStepTag(item.event.step_number, item.event.step_id))}</span>` : ""}
            ${payload?.usage ? `<span class="tag">${escapeHtml(formatUsage(payload.usage))}</span>` : ""}
            ${formatLatency(payload) ? `<span class="tag">${escapeHtml(formatLatency(payload))}</span>` : ""}
          </div>
          <details>
            <summary>Raw event JSON</summary>
            <pre>${escapeHtml(JSON.stringify(payload, null, 2))}</pre>
          </details>
        </article>
      `;
    }

    function renderArtifactItem(item) {
      return `
        <article class="event">
          <div class="event-head">
            <div>
              <div class="event-title">${escapeHtml(describeArtifact(item.artifact))}</div>
              <div class="event-sub">${escapeHtml(item.artifact.artifact_kind)} · artifact #${item.artifact.artifact_index}</div>
            </div>
            <div class="event-sub">${escapeHtml(formatTimestamp(item.artifact.occurred_at))}</div>
          </div>
          <div class="meta">
            <span class="tag warn">artifact</span>
            <span class="tag">${escapeHtml(item.artifact.source)}</span>
            ${formatExecutor(item.artifact.executor) ? `<span class="tag">${escapeHtml(formatExecutor(item.artifact.executor))}</span>` : ""}
            ${item.artifact.turn_id ? `<span class="tag">turn ${escapeHtml(item.artifact.turn_id)}</span>` : ""}
            ${formatStepTag(item.artifact.step_number, item.artifact.step_id) ? `<span class="tag">${escapeHtml(formatStepTag(item.artifact.step_number, item.artifact.step_id))}</span>` : ""}
            ${item.artifact.summary ? `<span class="tag">${escapeHtml(item.artifact.summary)}</span>` : ""}
          </div>
          <details>
            <summary>Raw artifact JSON</summary>
            <pre>${escapeHtml(JSON.stringify(item.artifact.payload, null, 2))}</pre>
          </details>
        </article>
      `;
    }

    function renderGroupedTimelineItem(group) {
      const first = group.items[0];
      const context = timelineItemContext(first);
      const executor = formatExecutor(context.executor);
      const stepTag = formatStepTag(context.step_number, context.step_id);
      const artifactSources = Array.from(
        new Set(
          group.items
            .filter((item) => item.type === "artifact")
            .map((item) => item.artifact.source)
            .filter(Boolean)
        )
      );
      return `
        <article class="event">
          <div class="event-head">
            <div>
              <div class="event-title">${escapeHtml(describeTimelineGroup(group))}</div>
              <div class="event-sub">${escapeHtml(describeTimelineGroupSubtitle(group))}</div>
            </div>
            <div class="event-sub">${escapeHtml(formatTimestamp(group.occurred_at))}</div>
          </div>
          <div class="meta">
            <span class="tag">merged</span>
            <span class="tag">${escapeHtml(group.group_kind)}</span>
            ${executor ? `<span class="tag">${escapeHtml(executor)}</span>` : ""}
            ${context.turn_id ? `<span class="tag">turn ${escapeHtml(context.turn_id)}</span>` : ""}
            ${stepTag ? `<span class="tag">${escapeHtml(stepTag)}</span>` : ""}
            ${artifactSources.map((source) => `<span class="tag">${escapeHtml(source)}</span>`).join("")}
          </div>
          <div class="merged-items">
            ${renderGroupedTimelinePhase("request", group.items.filter((item) => timelineItemPhase(item) === "request"))}
            ${renderGroupedTimelinePhase("response", group.items.filter((item) => timelineItemPhase(item) === "response"))}
          </div>
        </article>
      `;
    }

    function renderTimelineItem(item) {
      if (item.type === "group") return renderGroupedTimelineItem(item);
      if (item.type === "event") return renderEventItem(item);
      return renderArtifactItem(item);
    }

    function timelineSectionKey(item) {
      const context = timelineItemContext(item);
      if (context.step_number != null && context.step_number !== "") return `step:${context.step_number}`;
      if (context.step_id) return `step-id:${context.step_id}`;
      return "turn-level";
    }

    function timelineSectionLabel(item) {
      const context = timelineItemContext(item);
      if (context.step_number != null && context.step_number !== "") return `Step ${context.step_number}`;
      if (context.step_id) return `Step ${context.step_id}`;
      return "Turn-level events";
    }

    function buildTimelineSections(timelineItems) {
      const stepSections = new Map();
      const stepOrder = [];
      const leadTurnItems = [];
      const tailTurnItems = [];
      let sawStepSection = false;

      for (const item of timelineItems) {
        const key = timelineSectionKey(item);
        if (key === "turn-level") {
          if (sawStepSection) {
            tailTurnItems.push(item);
          } else {
            leadTurnItems.push(item);
          }
          continue;
        }

        sawStepSection = true;
        if (!stepSections.has(key)) {
          stepSections.set(key, {
            key,
            label: timelineSectionLabel(item),
            items: [],
          });
          stepOrder.push(key);
        }
        stepSections.get(key).items.push(item);
      }

      const sections = [];
      if (leadTurnItems.length) {
        sections.push({
          key: "turn-level:lead",
          label: "Turn lifecycle",
          items: leadTurnItems,
        });
      }
      for (const key of stepOrder) {
        sections.push(stepSections.get(key));
      }
      if (tailTurnItems.length) {
        sections.push({
          key: "turn-level:tail",
          label: "Turn completion",
          items: tailTurnItems,
        });
      }
      return sections;
    }

    function renderTimelineSection(section) {
      const sample = section.items[0];
      const context = timelineItemContext(sample);
      const executors = Array.from(
        new Set(
          section.items
            .map((item) => {
              if (item.type === "group") return formatExecutor(timelineItemContext(item.items[0]).executor);
              return formatExecutor(timelineItemContext(item).executor);
            })
            .filter(Boolean)
        )
      );
      const tag = section.key === "turn-level"
        ? "turn-level"
        : formatStepTag(context.step_number, context.step_id) || section.label.toLowerCase();
      return `
        <details class="step-group">
          <summary>
            <div class="step-group-head">
              <div class="step-group-title">${escapeHtml(section.label)}</div>
              <div class="step-group-sub">${section.items.length} timeline item${section.items.length === 1 ? "" : "s"}</div>
            </div>
            <div class="step-group-meta">
              <span class="tag">${escapeHtml(tag)}</span>
              ${executors.map((executor) => `<span class="tag">${escapeHtml(executor)}</span>`).join("")}
            </div>
          </summary>
          <div class="step-group-body">
            ${section.items.map((item) => renderTimelineItem(item)).join("")}
          </div>
        </details>
      `;
    }

    function selectedFilterValues(params, key) {
      return new Set(
        String(params.get(key) || "")
          .split(",")
          .map((value) => value.trim())
          .filter(Boolean)
      );
    }

    function readTimelineFilterState(params) {
      return {
        event_types: selectedFilterValues(params, "event_types"),
        artifact_kinds: selectedFilterValues(params, "artifact_kinds"),
        executors: selectedFilterValues(params, "executors"),
        steps: selectedFilterValues(params, "steps"),
      };
    }

    function buildTimelineFilterParams(filterState) {
      const params = new URLSearchParams();
      for (const [key, values] of Object.entries(filterState)) {
        const selected = [...values];
        if (selected.length) {
          params.set(key, selected.join(","));
        }
      }
      return params;
    }

    function hasActiveTimelineFilters(filterState) {
      return Object.values(filterState).some((values) => values.size > 0);
    }

    function titleizeFilterKey(value) {
      return String(value || "")
        .split("_")
        .filter(Boolean)
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
        .join(" ");
    }

    function formatStepTag(stepNumber, stepId) {
      if (stepNumber != null) return `step ${stepNumber}`;
      if (stepId) return `step ${stepId}`;
      return null;
    }

    function renderFilterOptions(name, options, selectedValues) {
      if (!options.length) {
        return `<div class="filter-empty">No filter values available.</div>`;
      }
      return `
        <div class="filter-options">
          ${options.map((option) => `
            <label class="filter-option">
              <input
                type="checkbox"
                name="${escapeHtml(name)}"
                value="${escapeHtml(option.key)}"
                ${selectedValues.has(option.key) ? "checked" : ""}
              />
              <span>${escapeHtml(option.label)}</span>
            </label>
          `).join("")}
        </div>
      `;
    }

    function renderTimelineFilters(turn, filterState, resultCount) {
      const activeTags = [];
      const pushSelectedOptions = (options, selected) => {
        const byKey = new Map(options.map((option) => [option.key, option.label]));
        for (const key of selected) {
          activeTags.push(byKey.get(key) || key);
        }
      };
      pushSelectedOptions(
        (turn.available_filters.event_types || []).map((value) => ({ key: value, label: titleizeFilterKey(value) })),
        filterState.event_types,
      );
      pushSelectedOptions(
        (turn.available_filters.artifact_kinds || []).map((value) => ({ key: value, label: titleizeFilterKey(value) })),
        filterState.artifact_kinds,
      );
      pushSelectedOptions(turn.available_filters.executors || [], filterState.executors);
      pushSelectedOptions(turn.available_filters.steps || [], filterState.steps);
      const isCollapsed = !hasActiveTimelineFilters(filterState);

      return `
        <section class="panel filters-panel">
          <div class="filters-head">
            <div>
              <div class="filters-kicker">Search And Filter</div>
              <div class="value">${resultCount} matching timeline item${resultCount === 1 ? "" : "s"}</div>
            </div>
            <div class="filters-actions">
              <button id="toggleFiltersButton" class="filters-button primary" type="button">${isCollapsed ? "Show filters" : "Hide filters"}</button>
              ${hasActiveTimelineFilters(filterState) ? `<button id="clearFiltersButton" class="filters-button" type="button">Clear filters</button>` : ""}
            </div>
          </div>
          <div class="filters-summary">
            ${activeTags.length ? activeTags.map((tag) => `<span class="filter-tag">${escapeHtml(tag)}</span>`).join("") : `<span class="muted">No filters applied.</span>`}
          </div>
          <div id="filtersBody" class="filters-body" ${isCollapsed ? "hidden" : ""}>
            <form id="timelineFiltersForm" class="filter-grid">
              <fieldset class="filter-group">
                <legend>Event Types</legend>
                ${renderFilterOptions(
                  "event_types",
                  (turn.available_filters.event_types || []).map((value) => ({ key: value, label: titleizeFilterKey(value) })),
                  filterState.event_types,
                )}
              </fieldset>
              <fieldset class="filter-group">
                <legend>Artifact Kinds</legend>
                ${renderFilterOptions(
                  "artifact_kinds",
                  (turn.available_filters.artifact_kinds || []).map((value) => ({ key: value, label: titleizeFilterKey(value) })),
                  filterState.artifact_kinds,
                )}
              </fieldset>
              <fieldset class="filter-group">
                <legend>Executors</legend>
                ${renderFilterOptions("executors", turn.available_filters.executors || [], filterState.executors)}
              </fieldset>
              <fieldset class="filter-group">
                <legend>Steps</legend>
                ${renderFilterOptions("steps", turn.available_filters.steps || [], filterState.steps)}
              </fieldset>
            </form>
          </div>
        </section>
      `;
    }

    function bindTimelineFilterControls(turnId) {
      const form = document.getElementById("timelineFiltersForm");
      if (form) {
        form.addEventListener("change", () => {
          const formData = new FormData(form);
          const nextFilterState = {
            event_types: new Set(formData.getAll("event_types").map(String)),
            artifact_kinds: new Set(formData.getAll("artifact_kinds").map(String)),
            executors: new Set(formData.getAll("executors").map(String)),
            steps: new Set(formData.getAll("steps").map(String)),
          };
          navigateToHash(buildHash(["turn", turnId], buildTimelineFilterParams(nextFilterState)));
        });
      }

      const clearButton = document.getElementById("clearFiltersButton");
      if (clearButton) {
        clearButton.addEventListener("click", () => {
          navigateToHash(buildHash(["turn", turnId]));
        });
      }

      const toggleButton = document.getElementById("toggleFiltersButton");
      const filtersBody = document.getElementById("filtersBody");
      if (toggleButton && filtersBody) {
        toggleButton.addEventListener("click", () => {
          const nextHidden = !filtersBody.hasAttribute("hidden");
          if (nextHidden) {
            filtersBody.setAttribute("hidden", "");
            toggleButton.textContent = "Show filters";
          } else {
            filtersBody.removeAttribute("hidden");
            toggleButton.textContent = "Hide filters";
          }
        });
      }
    }

    async function fetchJson(path) {
      const response = await fetch(path, { headers: { "accept": "application/json" } });
      const body = await response.text();
      let data = null;
      try { data = body ? JSON.parse(body) : null; } catch (_) {}
      if (!response.ok) {
        throw new Error(data?.error || body || `Request failed: ${response.status}`);
      }
      return data;
    }

    async function renderSessions() {
      setCrumbs([{ label: "Recent Sessions" }]);
      const limit = Math.max(1, Math.min(200, Number(limitInput.value) || 50));
      const sessions = await fetchJson(`/debug/history/sessions?limit=${limit}`);
      if (!sessions.length) {
        renderEmpty("No persisted debug turns found.");
        return;
      }
      content.innerHTML = sessions.map((session) => `
        <article class="card">
          <div class="label">Session</div>
          <div class="value"><a href="#/session/${encodeURIComponent(session.session_id)}">${escapeHtml(session.session_id)}</a></div>
          <div class="meta">
            <span class="tag">turn ${session.latest_turn_number}</span>
            <span class="tag ${session.status === "completed" ? "good" : session.status === "running" ? "" : "warn"}">${escapeHtml(session.status)}</span>
            <span class="tag">${escapeHtml(session.model_name)}</span>
            <span class="tag ${session.termination === "Final" ? "good" : "warn"}">${escapeHtml(session.termination)}</span>
            <span class="tag">${session.elapsed_ms}ms</span>
          </div>
          <p class="muted">${escapeHtml(formatTimestamp(session.completed_at))}</p>
          <p>${escapeHtml(session.final_text_preview || session.error_message || "No final text preview.")}</p>
          <div class="inline-actions">
            <a class="button" href="#/turn/${encodeURIComponent(session.turn_id)}">Open latest turn</a>
          </div>
        </article>
      `).join("");
    }

    async function renderSession(sessionId) {
      setCrumbs([
        { label: "Recent Sessions", href: "#/" },
        { label: sessionId }
      ]);
      const turns = await fetchJson(`/debug/history/sessions/${encodeURIComponent(sessionId)}`);
      if (!turns.length) {
        renderEmpty("No persisted turns found for this session.");
        return;
      }
      content.innerHTML = `
        <section class="panel">
          <div class="label">Session</div>
          <div class="value">${escapeHtml(sessionId)}</div>
          <p class="muted">Recent persisted turns for this session.</p>
        </section>
        <section class="list">
          ${turns.map((turn) => `
            <article class="card">
              <div class="label">Turn ${turn.turn_number}</div>
              <div class="value"><a href="#/turn/${encodeURIComponent(turn.turn_id)}">${escapeHtml(turn.turn_id)}</a></div>
              <div class="meta">
                <span class="tag ${turn.status === "completed" ? "good" : turn.status === "running" ? "" : "warn"}">${escapeHtml(turn.status)}</span>
                <span class="tag">${escapeHtml(turn.model_name)}</span>
                <span class="tag ${turn.termination === "Final" ? "good" : "warn"}">${escapeHtml(turn.termination)}</span>
                <span class="tag">${turn.elapsed_ms}ms</span>
              </div>
              <p class="muted">${escapeHtml(formatTimestamp(turn.ended_at))}</p>
              <p>Usage: ${escapeHtml(formatUsage(turn.usage))}</p>
              ${turn.error_message ? `<p><strong>Error:</strong> ${escapeHtml(turn.error_message)}</p>` : ""}
            </article>
          `).join("")}
        </section>
      `;
    }

    async function renderTurn(turnId, routeParams) {
      const filterState = readTimelineFilterState(routeParams);
      const requestParams = buildTimelineFilterParams(filterState);
      const query = requestParams.toString();
      const turn = await fetchJson(`/debug/history/turns/${encodeURIComponent(turnId)}${query ? `?${query}` : ""}`);
      setCrumbs([
        { label: "Recent Sessions", href: "#/" },
        { label: turn.session_id, href: `#/session/${encodeURIComponent(turn.session_id)}` },
        { label: `turn ${turnId}` }
      ]);
      const summary = `
        <section class="panel">
          <div class="summary-grid">
            <div class="summary-block"><div class="label">Session</div><div class="value">${escapeHtml(turn.session_id)}</div></div>
            <div class="summary-block"><div class="label">Turn</div><div class="value">${turn.turn_number}</div></div>
            <div class="summary-block"><div class="label">Turn ID</div><div class="value">${escapeHtml(turn.turn_id)}</div></div>
            <div class="summary-block"><div class="label">Status</div><div class="value">${escapeHtml(turn.status)}</div></div>
            <div class="summary-block"><div class="label">Model</div><div class="value">${escapeHtml(turn.model_name)}</div></div>
            <div class="summary-block"><div class="label">Elapsed</div><div class="value">${turn.elapsed_ms}ms</div></div>
            <div class="summary-block summary-stack">
              <div>
                <div class="label">Started</div>
                <div class="value">${escapeHtml(formatTimestamp(turn.started_at))}</div>
              </div>
              <div>
                <div class="label">Ended</div>
                <div class="value">${escapeHtml(formatTimestamp(turn.ended_at))}</div>
              </div>
            </div>
          </div>
          <div class="summary-copy">
            <div class="summary-copy-block">
              <div class="label">Usage</div>
              <div class="summary-text">${escapeHtml(formatUsage(turn.usage))}</div>
            </div>
            <div class="summary-copy-block">
              <div class="label">Final Text</div>
              <div class="summary-text">${escapeHtml(turn.final_text || "n/a")}</div>
            </div>
            ${turn.error_message ? `
              <div class="summary-copy-block">
                <div class="label">Error</div>
                <div class="summary-text">${escapeHtml(turn.error_message)}</div>
              </div>
            ` : ""}
          </div>
          <div class="meta">
            <span class="tag">${escapeHtml(turn.system_prompt_path)}</span>
            <span class="tag">${escapeHtml(turn.registry_path)}</span>
            <span class="tag">${escapeHtml(turn.subagent_registry_path)}</span>
          </div>
          <details>
            <summary>Turn record snapshot</summary>
            <pre>${escapeHtml(JSON.stringify(turn.turn_record, null, 2))}</pre>
          </details>
        </section>
      `;
      const timelineItems = buildTimeline(turn);
      const timelineSections = buildTimelineSections(timelineItems);
      const filters = renderTimelineFilters(turn, filterState, timelineItems.length);
      const timeline = timelineSections.length
        ? timelineSections.map((section) => renderTimelineSection(section)).join("")
        : `<div class="panel empty">${hasActiveTimelineFilters(filterState) ? "No matching timeline items found for the selected filters." : "No persisted events or raw artifacts found for this turn."}</div>`;
      content.innerHTML = summary + filters + `<section class="list">${timeline}</section>`;
      bindTimelineFilterControls(turnId);
    }

    async function renderRoute() {
      const { parts, params } = routeState();
      try {
        if (parts.length === 0) {
          await renderSessions();
        } else if (parts[0] === "session" && parts[1]) {
          await renderSession(decodeURIComponent(parts[1]));
        } else if (parts[0] === "turn" && parts[1]) {
          await renderTurn(decodeURIComponent(parts[1]), params);
        } else {
          renderError("Unknown route.");
        }
      } catch (error) {
        renderError(error.message || String(error));
      }
    }

    renderRoute();
  </script>
</body>
</html>
"###;

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::{
        DebugHistoryTurnQuery, SlackStreamHandle, parse_debug_history_turn_filters,
        slack_stop_stream_body,
    };

    #[test]
    fn parses_debug_history_turn_filters_from_csv_query() {
        let filters = parse_debug_history_turn_filters(DebugHistoryTurnQuery {
            event_types: Some("model_called,mcp_responded".to_owned()),
            artifact_kinds: Some("model_request".to_owned()),
            executors: Some("main,subagent:tool-executor".to_owned()),
            steps: Some("1,none".to_owned()),
        })
        .expect("filters should parse");

        assert_eq!(
            filters.event_types,
            BTreeSet::from(["mcp_responded".to_owned(), "model_called".to_owned()])
        );
        assert_eq!(
            filters.artifact_kinds,
            BTreeSet::from(["model_request".to_owned()])
        );
        assert_eq!(
            filters.executors,
            BTreeSet::from(["main".to_owned(), "subagent:tool-executor".to_owned()])
        );
        assert_eq!(
            filters.steps,
            BTreeSet::from(["1".to_owned(), "none".to_owned()])
        );
    }

    #[test]
    fn rejects_invalid_step_filter_values() {
        let error = parse_debug_history_turn_filters(DebugHistoryTurnQuery {
            steps: Some("0,bad".to_owned()),
            ..DebugHistoryTurnQuery::default()
        })
        .expect_err("invalid step filters should fail");

        match error {
            super::ApiError::BadRequest(message) => {
                assert!(
                    message.contains("invalid step filter") || message.contains("positive integer")
                );
            }
            other => panic!("expected bad request, got {other:?}"),
        }
    }

    #[test]
    fn stop_stream_body_omits_markdown_text_when_absent() {
        let body = slack_stop_stream_body(
            &SlackStreamHandle {
                channel: "C123".to_owned(),
                ts: "1743076800.000100".to_owned(),
            },
            None,
        );

        assert_eq!(
            body.get("channel").and_then(serde_json::Value::as_str),
            Some("C123")
        );
        assert_eq!(
            body.get("ts").and_then(serde_json::Value::as_str),
            Some("1743076800.000100")
        );
        assert!(body.get("markdown_text").is_none());
    }
}
