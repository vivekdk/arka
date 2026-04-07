//! Stateful WhatsApp gateway support.
//!
//! The current implementation provides the control-plane shape needed for an
//! OpenClaw-style connector: login state, persistent gateway status, DM access
//! policy, and an outbound worker. The transport itself remains pluggable.

use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, mpsc};
use tracing::{info, warn};
use uuid::Uuid;

use crate::{
    adapters::{ChannelAdapter, ChannelAdapterError, WhatsAppChannelAdapter},
    service::SessionService,
    types::{
        ChannelDeliveryTarget, ChannelDispatchResult, ChannelKind, DeliveryStatus,
        ReceiveWhatsAppMessageRequest, ReceiveWhatsAppMessageResponse, SessionId,
        StartWhatsAppLoginResponse, WhatsAppDmPolicy, WhatsAppGatewayConnectionState,
        WhatsAppGatewayStatus, WhatsAppWebhookPayload,
    },
};

#[derive(Debug, thiserror::Error)]
pub enum WhatsAppGatewayError {
    #[error("whatsapp gateway is not ready; complete login first")]
    NotReady,
    #[error("sender `{0}` is not allowed by the current dm policy")]
    SenderBlocked(String),
    #[error("whatsapp login session was not found")]
    LoginSessionNotFound,
    #[error("failed to persist whatsapp gateway state at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to serialize whatsapp gateway state at {path}: {source}")]
    Serialize {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("failed to deserialize whatsapp gateway state at {path}: {source}")]
    Deserialize {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("whatsapp delivery failed: {0}")]
    Delivery(#[from] WhatsAppDeliveryError),
    #[error("whatsapp payload is invalid: {0}")]
    InvalidPayload(String),
    #[error("whatsapp operation is not supported by the configured transport: {0}")]
    UnsupportedOperation(String),
}

#[derive(Clone)]
pub struct WhatsAppConnector {
    pub account_id: String,
    pub dm_policy: WhatsAppDmPolicy,
    pub allow_from: Vec<String>,
    pub delivery_client: Arc<dyn WhatsAppDeliveryClient>,
    pub control_client: Option<Arc<dyn WhatsAppControlClient>>,
    pub event_queue_capacity: usize,
    pub state_path: PathBuf,
}

#[async_trait]
pub trait WhatsAppDeliveryClient: Send + Sync {
    async fn send_text(
        &self,
        account_id: &str,
        target: &ChannelDeliveryTarget,
        text: &str,
    ) -> Result<(), WhatsAppDeliveryError>;
}

#[async_trait]
pub trait WhatsAppControlClient: Send + Sync {
    async fn status(
        &self,
        account_id: &str,
    ) -> Result<WhatsAppControlStatus, WhatsAppDeliveryError>;

    async fn start_login(
        &self,
        account_id: &str,
    ) -> Result<StartWhatsAppLoginResponse, WhatsAppDeliveryError>;

    async fn complete_login(
        &self,
        account_id: &str,
        login_session_id: &str,
    ) -> Result<WhatsAppControlStatus, WhatsAppDeliveryError>;

    async fn logout(
        &self,
        account_id: &str,
    ) -> Result<WhatsAppControlStatus, WhatsAppDeliveryError>;
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WhatsAppControlStatus {
    pub connection_state: WhatsAppGatewayConnectionState,
    pub active_login_session_id: Option<String>,
    pub last_error: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct LoggingWhatsAppDeliveryClient;

#[async_trait]
impl WhatsAppDeliveryClient for LoggingWhatsAppDeliveryClient {
    async fn send_text(
        &self,
        account_id: &str,
        target: &ChannelDeliveryTarget,
        text: &str,
    ) -> Result<(), WhatsAppDeliveryError> {
        info!(
            account_id,
            conversation = %target.external_conversation_id,
            user = %target.external_user_id,
            text,
            "whatsapp outbound delivery emitted"
        );
        Ok(())
    }
}

#[derive(Clone)]
pub struct ReqwestWhatsAppWebBridgeClient {
    http: reqwest::Client,
    base_url: String,
}

impl ReqwestWhatsAppWebBridgeClient {
    pub fn new(base_url: Option<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: base_url.unwrap_or_else(|| "http://127.0.0.1:8091".to_owned()),
        }
    }

    fn endpoint(&self, path: &str) -> String {
        format!("{}/{}", self.base_url.trim_end_matches('/'), path)
    }
}

#[async_trait]
impl WhatsAppDeliveryClient for ReqwestWhatsAppWebBridgeClient {
    async fn send_text(
        &self,
        account_id: &str,
        target: &ChannelDeliveryTarget,
        text: &str,
    ) -> Result<(), WhatsAppDeliveryError> {
        let response = self
            .http
            .post(self.endpoint("messages"))
            .json(&serde_json::json!({
                "account_id": account_id,
                "to": target.external_user_id,
                "text": text,
            }))
            .send()
            .await
            .map_err(|error| WhatsAppDeliveryError::Transport(error.to_string()))?;
        if response.status().is_success() {
            Ok(())
        } else {
            Err(WhatsAppDeliveryError::Transport(format!(
                "bridge returned {}",
                response.status()
            )))
        }
    }
}

#[async_trait]
impl WhatsAppControlClient for ReqwestWhatsAppWebBridgeClient {
    async fn status(
        &self,
        account_id: &str,
    ) -> Result<WhatsAppControlStatus, WhatsAppDeliveryError> {
        let response = self
            .http
            .get(self.endpoint(&format!("status?account_id={account_id}")))
            .send()
            .await
            .map_err(|error| WhatsAppDeliveryError::Transport(error.to_string()))?;
        response
            .error_for_status()
            .map_err(|error| WhatsAppDeliveryError::Transport(error.to_string()))?
            .json::<WhatsAppControlStatus>()
            .await
            .map_err(|error| WhatsAppDeliveryError::Transport(error.to_string()))
    }

    async fn start_login(
        &self,
        account_id: &str,
    ) -> Result<StartWhatsAppLoginResponse, WhatsAppDeliveryError> {
        let response = self
            .http
            .post(self.endpoint("login/start"))
            .json(&serde_json::json!({ "account_id": account_id }))
            .send()
            .await
            .map_err(|error| WhatsAppDeliveryError::Transport(error.to_string()))?;
        response
            .error_for_status()
            .map_err(|error| WhatsAppDeliveryError::Transport(error.to_string()))?
            .json::<StartWhatsAppLoginResponse>()
            .await
            .map_err(|error| WhatsAppDeliveryError::Transport(error.to_string()))
    }

    async fn complete_login(
        &self,
        account_id: &str,
        login_session_id: &str,
    ) -> Result<WhatsAppControlStatus, WhatsAppDeliveryError> {
        let response = self
            .http
            .post(self.endpoint("login/complete"))
            .json(&serde_json::json!({
                "account_id": account_id,
                "login_session_id": login_session_id,
            }))
            .send()
            .await
            .map_err(|error| WhatsAppDeliveryError::Transport(error.to_string()))?;
        response
            .error_for_status()
            .map_err(|error| WhatsAppDeliveryError::Transport(error.to_string()))?
            .json::<WhatsAppControlStatus>()
            .await
            .map_err(|error| WhatsAppDeliveryError::Transport(error.to_string()))
    }

    async fn logout(
        &self,
        account_id: &str,
    ) -> Result<WhatsAppControlStatus, WhatsAppDeliveryError> {
        let response = self
            .http
            .post(self.endpoint("logout"))
            .json(&serde_json::json!({ "account_id": account_id }))
            .send()
            .await
            .map_err(|error| WhatsAppDeliveryError::Transport(error.to_string()))?;
        response
            .error_for_status()
            .map_err(|error| WhatsAppDeliveryError::Transport(error.to_string()))?
            .json::<WhatsAppControlStatus>()
            .await
            .map_err(|error| WhatsAppDeliveryError::Transport(error.to_string()))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum WhatsAppDeliveryError {
    #[error("invalid whatsapp delivery target: {0}")]
    InvalidTarget(String),
    #[error("transport error: {0}")]
    Transport(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct PersistedWhatsAppGatewayState {
    account_id: String,
    connection_state: WhatsAppGatewayConnectionState,
    dm_policy: WhatsAppDmPolicy,
    allow_from: Vec<String>,
    active_login_session_id: Option<String>,
    last_error: Option<String>,
}

struct WhatsAppGatewayRuntimeState {
    persisted: PersistedWhatsAppGatewayState,
    pending_outbound: usize,
}

#[derive(Clone, Debug)]
struct WhatsAppOutboundMessage {
    session_id: SessionId,
    target: ChannelDeliveryTarget,
    text: String,
}

#[derive(Clone)]
pub struct WhatsAppGatewayHandle {
    adapter: WhatsAppChannelAdapter,
    state_path: PathBuf,
    state: Arc<Mutex<WhatsAppGatewayRuntimeState>>,
    outbound_tx: mpsc::Sender<WhatsAppOutboundMessage>,
    account_id: String,
    control_client: Option<Arc<dyn WhatsAppControlClient>>,
}

impl WhatsAppGatewayHandle {
    pub fn spawn<R, S>(
        service: Arc<SessionService<R, S>>,
        connector: WhatsAppConnector,
    ) -> Result<Self, WhatsAppGatewayError>
    where
        R: crate::runner::TurnRunner + Send + Sync + 'static,
        S: crate::service::ConversationStore + Send + Sync + 'static,
    {
        let account_id = connector.account_id.clone();
        let persisted = load_gateway_state(
            connector.state_path.as_path(),
            &account_id,
            connector.dm_policy,
            &connector.allow_from,
        )?;
        let state = Arc::new(Mutex::new(WhatsAppGatewayRuntimeState {
            persisted,
            pending_outbound: 0,
        }));
        let (tx, rx) = mpsc::channel(connector.event_queue_capacity.max(1));
        tokio::spawn(run_whatsapp_worker(
            service,
            account_id.clone(),
            connector.delivery_client,
            connector.state_path.clone(),
            Arc::clone(&state),
            rx,
        ));
        Ok(Self {
            adapter: WhatsAppChannelAdapter,
            state_path: connector.state_path,
            state,
            outbound_tx: tx,
            account_id,
            control_client: connector.control_client,
        })
    }

    pub async fn status(&self) -> WhatsAppGatewayStatus {
        if let Some(control_client) = self.control_client.as_ref() {
            if let Ok(remote) = control_client.status(&self.account_id).await {
                let mut state = self.state.lock().await;
                state.persisted.connection_state = remote.connection_state;
                state.persisted.active_login_session_id = remote.active_login_session_id;
                state.persisted.last_error = remote.last_error;
                let _ = persist_gateway_state(self.state_path.as_path(), &state.persisted);
                return build_status(&state);
            }
        }
        let state = self.state.lock().await;
        build_status(&state)
    }

    pub async fn start_login(&self) -> Result<StartWhatsAppLoginResponse, WhatsAppGatewayError> {
        if let Some(control_client) = self.control_client.as_ref() {
            let response = control_client
                .start_login(&self.account_id)
                .await
                .map_err(WhatsAppGatewayError::Delivery)?;
            let mut state = self.state.lock().await;
            state.persisted.connection_state = WhatsAppGatewayConnectionState::NeedsLogin;
            state.persisted.active_login_session_id = Some(response.login_session_id.clone());
            state.persisted.last_error = None;
            persist_gateway_state(self.state_path.as_path(), &state.persisted)?;
            return Ok(response);
        }
        let mut state = self.state.lock().await;
        let login_session_id = Uuid::new_v4().to_string();
        state.persisted.connection_state = WhatsAppGatewayConnectionState::NeedsLogin;
        state.persisted.active_login_session_id = Some(login_session_id.clone());
        state.persisted.last_error = None;
        persist_gateway_state(self.state_path.as_path(), &state.persisted)?;
        Ok(StartWhatsAppLoginResponse {
            account_id: state.persisted.account_id.clone(),
            login_session_id: login_session_id.clone(),
            qr_code: format!("whatsapp://pair/{login_session_id}"),
        })
    }

    pub async fn complete_login(&self, login_session_id: &str) -> Result<(), WhatsAppGatewayError> {
        if let Some(control_client) = self.control_client.as_ref() {
            let remote = control_client
                .complete_login(&self.account_id, login_session_id)
                .await
                .map_err(WhatsAppGatewayError::Delivery)?;
            let mut state = self.state.lock().await;
            state.persisted.connection_state = remote.connection_state;
            state.persisted.active_login_session_id = remote.active_login_session_id;
            state.persisted.last_error = remote.last_error;
            persist_gateway_state(self.state_path.as_path(), &state.persisted)?;
            return Ok(());
        }
        let mut state = self.state.lock().await;
        if state.persisted.active_login_session_id.as_deref() != Some(login_session_id) {
            return Err(WhatsAppGatewayError::LoginSessionNotFound);
        }
        state.persisted.connection_state = WhatsAppGatewayConnectionState::Ready;
        state.persisted.active_login_session_id = None;
        state.persisted.last_error = None;
        persist_gateway_state(self.state_path.as_path(), &state.persisted)
    }

    pub async fn logout(&self) -> Result<WhatsAppGatewayStatus, WhatsAppGatewayError> {
        if let Some(control_client) = self.control_client.as_ref() {
            let remote = control_client
                .logout(&self.account_id)
                .await
                .map_err(WhatsAppGatewayError::Delivery)?;
            let mut state = self.state.lock().await;
            state.persisted.connection_state = remote.connection_state;
            state.persisted.active_login_session_id = remote.active_login_session_id;
            state.persisted.last_error = remote.last_error;
            persist_gateway_state(self.state_path.as_path(), &state.persisted)?;
            return Ok(build_status(&state));
        }
        let mut state = self.state.lock().await;
        state.persisted.connection_state = WhatsAppGatewayConnectionState::NeedsLogin;
        state.persisted.active_login_session_id = None;
        state.persisted.last_error = None;
        persist_gateway_state(self.state_path.as_path(), &state.persisted)?;
        Ok(build_status(&state))
    }

    pub async fn dispatch_inbound<R, S>(
        &self,
        service: &SessionService<R, S>,
        request: ReceiveWhatsAppMessageRequest,
    ) -> Result<ReceiveWhatsAppMessageResponse, WhatsAppGatewayError>
    where
        R: crate::runner::TurnRunner + Send + Sync + 'static,
        S: crate::service::ConversationStore + Send + Sync + 'static,
    {
        self.ensure_ready_and_allowed(&request.from_user_id).await?;
        let payload = normalized_payload(request);
        let envelope = self.adapter.ingest(payload).map_err(map_adapter_error)?;
        let dispatch = service
            .dispatch_envelope(envelope)
            .await
            .map_err(|error| WhatsAppGatewayError::InvalidPayload(error.to_string()))?;
        self.queue_outbound(dispatch).await
    }

    pub async fn queue_dispatch_result(
        &self,
        dispatch: ChannelDispatchResult,
    ) -> Result<ReceiveWhatsAppMessageResponse, WhatsAppGatewayError> {
        let queued = self.queue_outbound(dispatch).await?;
        Ok(ReceiveWhatsAppMessageResponse {
            session: queued.session,
            queued_outbound: queued.queued_outbound,
            was_duplicate: queued.was_duplicate,
        })
    }

    async fn queue_outbound(
        &self,
        dispatch: ChannelDispatchResult,
    ) -> Result<ReceiveWhatsAppMessageResponse, WhatsAppGatewayError> {
        let mut queued_outbound = 0usize;
        for outbound in &dispatch.outbound {
            let Some(target) = outbound.delivery_target.clone() else {
                continue;
            };
            let message = WhatsAppOutboundMessage {
                session_id: dispatch.session.session_id.clone(),
                target,
                text: self.adapter.render(outbound).map_err(map_adapter_error)?,
            };
            {
                let mut state = self.state.lock().await;
                state.pending_outbound += 1;
                persist_gateway_state(self.state_path.as_path(), &state.persisted)?;
            }
            if let Err(error) = self.outbound_tx.send(message).await {
                let mut state = self.state.lock().await;
                state.pending_outbound = state.pending_outbound.saturating_sub(1);
                state.persisted.last_error = Some(format!("worker unavailable: {error}"));
                persist_gateway_state(self.state_path.as_path(), &state.persisted)?;
                return Err(WhatsAppGatewayError::InvalidPayload(
                    "whatsapp worker is unavailable".to_owned(),
                ));
            }
            queued_outbound += 1;
        }
        Ok(ReceiveWhatsAppMessageResponse {
            session: dispatch.session,
            queued_outbound,
            was_duplicate: dispatch.was_duplicate,
        })
    }

    async fn ensure_ready_and_allowed(
        &self,
        from_user_id: &str,
    ) -> Result<(), WhatsAppGatewayError> {
        if let Some(control_client) = self.control_client.as_ref() {
            if let Ok(remote) = control_client.status(&self.account_id).await {
                let mut state = self.state.lock().await;
                state.persisted.connection_state = remote.connection_state;
                state.persisted.active_login_session_id = remote.active_login_session_id;
                state.persisted.last_error = remote.last_error;
                let _ = persist_gateway_state(self.state_path.as_path(), &state.persisted);
            }
        }
        let state = self.state.lock().await;
        if !matches!(
            state.persisted.connection_state,
            WhatsAppGatewayConnectionState::Ready
        ) {
            return Err(WhatsAppGatewayError::NotReady);
        }
        if matches!(state.persisted.dm_policy, WhatsAppDmPolicy::Allowlist)
            && !state
                .persisted
                .allow_from
                .iter()
                .any(|known| known == from_user_id)
        {
            return Err(WhatsAppGatewayError::SenderBlocked(from_user_id.to_owned()));
        }
        Ok(())
    }
}

async fn run_whatsapp_worker<R, S>(
    service: Arc<SessionService<R, S>>,
    account_id: String,
    delivery_client: Arc<dyn WhatsAppDeliveryClient>,
    state_path: PathBuf,
    state: Arc<Mutex<WhatsAppGatewayRuntimeState>>,
    mut rx: mpsc::Receiver<WhatsAppOutboundMessage>,
) where
    R: crate::runner::TurnRunner + Send + Sync + 'static,
    S: crate::service::ConversationStore + Send + Sync + 'static,
{
    while let Some(message) = rx.recv().await {
        let result = delivery_client
            .send_text(&account_id, &message.target, &message.text)
            .await;
        let mut state_guard = state.lock().await;
        state_guard.pending_outbound = state_guard.pending_outbound.saturating_sub(1);
        match result {
            Ok(()) => {
                state_guard.persisted.last_error = None;
                let _ = persist_gateway_state(state_path.as_path(), &state_guard.persisted);
                service.emit_channel_delivery(
                    message.session_id,
                    ChannelKind::WhatsApp,
                    DeliveryStatus::Sent,
                );
            }
            Err(error) => {
                state_guard.persisted.last_error = Some(error.to_string());
                let _ = persist_gateway_state(state_path.as_path(), &state_guard.persisted);
                warn!(error = %error, "whatsapp outbound delivery failed");
                service.emit_channel_delivery(
                    message.session_id,
                    ChannelKind::WhatsApp,
                    DeliveryStatus::Failed,
                );
            }
        }
    }
}

fn normalized_payload(request: ReceiveWhatsAppMessageRequest) -> WhatsAppWebhookPayload {
    let mut text = request.text;
    if let Some(quoted_text) = request.quoted_text {
        let quoted_message_id = request
            .quoted_message_id
            .unwrap_or_else(|| "quoted-message".to_owned());
        text = format!(
            "Quoted reply ({quoted_message_id}): {quoted_text}\n\n{text}",
            text = text
        );
    }
    WhatsAppWebhookPayload {
        message_id: request.message_id,
        conversation_id: request.conversation_id,
        from_user_id: request.from_user_id,
        text,
    }
}

fn build_status(state: &WhatsAppGatewayRuntimeState) -> WhatsAppGatewayStatus {
    WhatsAppGatewayStatus {
        account_id: state.persisted.account_id.clone(),
        connection_state: state.persisted.connection_state,
        dm_policy: state.persisted.dm_policy,
        allow_from: state.persisted.allow_from.clone(),
        pending_outbound: state.pending_outbound,
        active_login_session_id: state.persisted.active_login_session_id.clone(),
        last_error: state.persisted.last_error.clone(),
    }
}

fn load_gateway_state(
    path: &Path,
    account_id: &str,
    dm_policy: WhatsAppDmPolicy,
    allow_from: &[String],
) -> Result<PersistedWhatsAppGatewayState, WhatsAppGatewayError> {
    if path.exists() {
        let payload = fs::read(path).map_err(|source| WhatsAppGatewayError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let mut state = serde_json::from_slice::<PersistedWhatsAppGatewayState>(&payload).map_err(
            |source| WhatsAppGatewayError::Deserialize {
                path: path.to_path_buf(),
                source,
            },
        )?;
        state.account_id = account_id.to_owned();
        state.dm_policy = dm_policy;
        state.allow_from = dedupe_allow_from(allow_from);
        Ok(state)
    } else {
        let state = PersistedWhatsAppGatewayState {
            account_id: account_id.to_owned(),
            connection_state: WhatsAppGatewayConnectionState::NeedsLogin,
            dm_policy,
            allow_from: dedupe_allow_from(allow_from),
            active_login_session_id: None,
            last_error: None,
        };
        persist_gateway_state(path, &state)?;
        Ok(state)
    }
}

fn persist_gateway_state(
    path: &Path,
    state: &PersistedWhatsAppGatewayState,
) -> Result<(), WhatsAppGatewayError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| WhatsAppGatewayError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    let payload =
        serde_json::to_vec_pretty(state).map_err(|source| WhatsAppGatewayError::Serialize {
            path: path.to_path_buf(),
            source,
        })?;
    fs::write(path, payload).map_err(|source| WhatsAppGatewayError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn dedupe_allow_from(allow_from: &[String]) -> Vec<String> {
    allow_from
        .iter()
        .filter_map(|value| {
            let trimmed = value.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_owned())
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn map_adapter_error(error: ChannelAdapterError) -> WhatsAppGatewayError {
    match error {
        ChannelAdapterError::InvalidPayload(message)
        | ChannelAdapterError::NotConfigured(message) => {
            WhatsAppGatewayError::InvalidPayload(message)
        }
    }
}
