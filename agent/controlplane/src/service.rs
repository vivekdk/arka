//! In-memory session orchestration service.
//!
//! This service is the core of the channel architecture. It maps external
//! channels onto internal sessions, persists messages and approvals, invokes
//! the runtime turn runner, and broadcasts session events.

use std::{
    collections::{HashMap, HashSet},
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::SystemTime,
};

use agent_runtime::{
    ConversationMessage, ConversationRole, MessageRecord, ResponseClient, ResponseFormat,
    ResponseTarget, TurnRecord,
};
use async_trait::async_trait;
use serde::{Serialize, de::DeserializeOwned};
use thiserror::Error;
use tokio::sync::{Mutex, broadcast};
use tracing::{error, warn};
use uuid::Uuid;

use crate::{
    observability::{RuntimeHarnessListener, SseRuntimeHarnessListener},
    runner::{
        GeneratedArtifact, GeneratedArtifactKind, TurnRunner, TurnRunnerError, TurnRunnerInput,
        TurnRunnerOutput,
    },
    types::{
        ApprovalDecision, ApprovalId, ApprovalRequestRecord, ApprovalState, ChannelBinding,
        ChannelDeliveryTarget, ChannelDispatchResult, ChannelEnvelope, ChannelIntent, ChannelKind,
        ChannelLocalImageAttachment, ChannelResponseAttachment, ChannelResponseEnvelope,
        CreateSessionRequest, DeliveryStatus, OutboundMessageKind, SessionEvent, SessionId,
        SessionMessage, SessionMessageId, SessionMessageRole, SessionRecord, SessionStatus,
        SubmitApprovalRequest, TurnRecordSummary,
    },
};

const RECENT_COMPUTED_RESULTS_LIMIT: usize = 6;

/// Store abstraction so the session layer can later move beyond in-memory
/// persistence without changing orchestration logic.
#[async_trait]
pub trait ConversationStore: Send + Sync {
    /// Creates and stores a new session.
    async fn create_session(
        &self,
        bindings: Vec<ChannelBinding>,
    ) -> Result<SessionRecord, ConversationStoreError>;

    /// Gets a session by ID.
    async fn get_session(&self, session_id: &SessionId) -> Option<SessionRecord>;

    /// Finds a session by external channel binding.
    async fn find_by_binding(&self, binding: &ChannelBinding) -> Option<SessionRecord>;

    /// Upserts a binding onto an existing session.
    async fn attach_binding(
        &self,
        session_id: &SessionId,
        binding: ChannelBinding,
    ) -> Result<(), ConversationStoreError>;

    /// Removes a binding from an existing session when it is no longer active.
    async fn detach_binding(
        &self,
        session_id: &SessionId,
        binding: &ChannelBinding,
    ) -> Result<(), ConversationStoreError>;

    /// Stores one session message.
    async fn append_message(&self, message: SessionMessage) -> Result<(), ConversationStoreError>;

    /// Lists messages for one session in insertion order.
    async fn list_messages(&self, session_id: &SessionId) -> Vec<SessionMessage>;

    /// Stores reusable computed results emitted during one runtime turn.
    async fn append_turn_trace(
        &self,
        session_id: &SessionId,
        turn: &TurnRecord,
    ) -> Result<(), ConversationStoreError>;

    /// Lists recent computed results from prior turns in insertion order.
    async fn list_recent_computed_results(
        &self,
        session_id: &SessionId,
        limit: usize,
    ) -> Vec<MessageRecord>;

    /// Updates the session status and optional turn summary.
    async fn update_session(
        &self,
        session_id: &SessionId,
        status: SessionStatus,
        last_turn: Option<TurnRecordSummary>,
    ) -> Result<Option<SessionRecord>, ConversationStoreError>;

    /// Records one idempotency key if unseen.
    async fn record_idempotency(
        &self,
        idempotency_key: &str,
    ) -> Result<bool, ConversationStoreError>;

    /// Stores a pending approval request.
    async fn create_approval(
        &self,
        approval: ApprovalRequestRecord,
    ) -> Result<(), ConversationStoreError>;

    /// Gets one approval request.
    async fn get_approval(
        &self,
        session_id: &SessionId,
        approval_id: &ApprovalId,
    ) -> Option<ApprovalRequestRecord>;

    /// Updates one approval request state.
    async fn update_approval(
        &self,
        session_id: &SessionId,
        approval_id: &ApprovalId,
        state: ApprovalState,
    ) -> Result<Option<ApprovalRequestRecord>, ConversationStoreError>;
}

#[derive(Debug, Error)]
pub enum ConversationStoreError {
    #[error("failed to access conversation store path {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to serialize conversation store record for {path}: {source}")]
    Serialize {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("failed to deserialize conversation store record at {path}:{line}: {source}")]
    Deserialize {
        path: PathBuf,
        line: usize,
        #[source]
        source: serde_json::Error,
    },
    #[error("conversation store is missing session snapshot at {path}")]
    MissingSessionSnapshot { path: PathBuf },
}

#[derive(Default)]
struct InMemoryState {
    // Session-scoped records are split by concern so the default in-memory
    // store mirrors the abstractions required by a future persistent backend.
    sessions: HashMap<SessionId, SessionRecord>,
    bindings: HashMap<ChannelBinding, SessionId>,
    messages: HashMap<SessionId, Vec<SessionMessage>>,
    computed_results: HashMap<SessionId, Vec<MessageRecord>>,
    approvals: HashMap<(SessionId, ApprovalId), ApprovalRequestRecord>,
    idempotency_keys: HashSet<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
struct PersistedIdempotencyKey {
    key: String,
}

/// In-memory conversation store used by tests and the default server.
#[derive(Default)]
pub struct InMemoryConversationStore {
    inner: Mutex<InMemoryState>,
}

#[async_trait]
impl ConversationStore for InMemoryConversationStore {
    async fn create_session(
        &self,
        bindings: Vec<ChannelBinding>,
    ) -> Result<SessionRecord, ConversationStoreError> {
        let mut inner = self.inner.lock().await;
        let session = SessionRecord {
            session_id: SessionId::new(),
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            status: SessionStatus::Idle,
            bindings: bindings.clone(),
            last_turn: None,
        };
        for binding in bindings {
            inner.bindings.insert(binding, session.session_id.clone());
        }
        inner
            .sessions
            .insert(session.session_id.clone(), session.clone());
        inner
            .messages
            .entry(session.session_id.clone())
            .or_insert_with(Vec::new);
        Ok(session)
    }

    async fn get_session(&self, session_id: &SessionId) -> Option<SessionRecord> {
        self.inner.lock().await.sessions.get(session_id).cloned()
    }

    async fn find_by_binding(&self, binding: &ChannelBinding) -> Option<SessionRecord> {
        let inner = self.inner.lock().await;
        inner
            .bindings
            .get(binding)
            .and_then(|session_id| inner.sessions.get(session_id))
            .cloned()
    }

    async fn attach_binding(
        &self,
        session_id: &SessionId,
        binding: ChannelBinding,
    ) -> Result<(), ConversationStoreError> {
        let mut inner = self.inner.lock().await;
        inner.bindings.insert(binding.clone(), session_id.clone());
        if let Some(session) = inner.sessions.get_mut(session_id) {
            if !session.bindings.contains(&binding) {
                session.bindings.push(binding);
                session.updated_at = SystemTime::now();
            }
        }
        Ok(())
    }

    async fn detach_binding(
        &self,
        session_id: &SessionId,
        binding: &ChannelBinding,
    ) -> Result<(), ConversationStoreError> {
        let mut inner = self.inner.lock().await;
        if inner.bindings.get(binding) == Some(session_id) {
            inner.bindings.remove(binding);
        }
        if let Some(session) = inner.sessions.get_mut(session_id) {
            session.bindings.retain(|known| known != binding);
            session.updated_at = SystemTime::now();
        }
        Ok(())
    }

    async fn append_message(&self, message: SessionMessage) -> Result<(), ConversationStoreError> {
        let mut inner = self.inner.lock().await;
        inner
            .messages
            .entry(message.session_id.clone())
            .or_default()
            .push(message.clone());
        if let Some(session) = inner.sessions.get_mut(&message.session_id) {
            session.updated_at = SystemTime::now();
        }
        Ok(())
    }

    async fn list_messages(&self, session_id: &SessionId) -> Vec<SessionMessage> {
        self.inner
            .lock()
            .await
            .messages
            .get(session_id)
            .cloned()
            .unwrap_or_default()
    }

    async fn append_turn_trace(
        &self,
        session_id: &SessionId,
        turn: &TurnRecord,
    ) -> Result<(), ConversationStoreError> {
        let mut inner = self.inner.lock().await;
        let computed = computed_results_from_turn(turn);
        if computed.is_empty() {
            return Ok(());
        }
        inner
            .computed_results
            .entry(session_id.clone())
            .or_default()
            .extend(computed);
        Ok(())
    }

    async fn list_recent_computed_results(
        &self,
        session_id: &SessionId,
        limit: usize,
    ) -> Vec<MessageRecord> {
        let inner = self.inner.lock().await;
        let Some(messages) = inner.computed_results.get(session_id) else {
            return Vec::new();
        };
        let start = messages.len().saturating_sub(limit);
        messages[start..].to_vec()
    }

    async fn update_session(
        &self,
        session_id: &SessionId,
        status: SessionStatus,
        last_turn: Option<TurnRecordSummary>,
    ) -> Result<Option<SessionRecord>, ConversationStoreError> {
        let mut inner = self.inner.lock().await;
        let Some(session) = inner.sessions.get_mut(session_id) else {
            return Ok(None);
        };
        session.status = status;
        session.updated_at = SystemTime::now();
        if last_turn.is_some() {
            session.last_turn = last_turn;
        }
        Ok(Some(session.clone()))
    }

    async fn record_idempotency(
        &self,
        idempotency_key: &str,
    ) -> Result<bool, ConversationStoreError> {
        let mut inner = self.inner.lock().await;
        Ok(inner.idempotency_keys.insert(idempotency_key.to_owned()))
    }

    async fn create_approval(
        &self,
        approval: ApprovalRequestRecord,
    ) -> Result<(), ConversationStoreError> {
        let mut inner = self.inner.lock().await;
        inner.approvals.insert(
            (approval.session_id.clone(), approval.approval_id.clone()),
            approval,
        );
        Ok(())
    }

    async fn get_approval(
        &self,
        session_id: &SessionId,
        approval_id: &ApprovalId,
    ) -> Option<ApprovalRequestRecord> {
        self.inner
            .lock()
            .await
            .approvals
            .get(&(session_id.clone(), approval_id.clone()))
            .cloned()
    }

    async fn update_approval(
        &self,
        session_id: &SessionId,
        approval_id: &ApprovalId,
        state: ApprovalState,
    ) -> Result<Option<ApprovalRequestRecord>, ConversationStoreError> {
        let mut inner = self.inner.lock().await;
        let Some(approval) = inner
            .approvals
            .get_mut(&(session_id.clone(), approval_id.clone()))
        else {
            return Ok(None);
        };
        approval.state = state;
        approval.resolved_at = Some(SystemTime::now());
        Ok(Some(approval.clone()))
    }
}

/// JSONL-backed conversation store used by the default server.
pub struct JsonlConversationStore {
    root_dir: PathBuf,
    inner: Mutex<InMemoryState>,
}

impl JsonlConversationStore {
    pub fn open(root_dir: impl AsRef<Path>) -> Result<Self, ConversationStoreError> {
        let root_dir = root_dir.as_ref().to_path_buf();
        create_dir_all(root_dir.as_path())?;
        create_dir_all(root_dir.join("sessions").as_path())?;
        Ok(Self {
            root_dir,
            inner: Mutex::new(InMemoryState::default()),
        })
    }

    fn sessions_dir(&self) -> PathBuf {
        self.root_dir.join("sessions")
    }

    fn session_dir(&self, session_id: &SessionId) -> PathBuf {
        self.sessions_dir().join(session_id.to_string())
    }

    fn session_snapshot_path(&self, session_id: &SessionId) -> PathBuf {
        self.session_dir(session_id).join("session.json")
    }

    fn messages_path(&self, session_id: &SessionId) -> PathBuf {
        self.session_dir(session_id).join("messages.jsonl")
    }

    fn approvals_path(&self, session_id: &SessionId) -> PathBuf {
        self.session_dir(session_id).join("approvals.jsonl")
    }

    fn computed_results_path(&self, session_id: &SessionId) -> PathBuf {
        self.session_dir(session_id).join("computed_results.jsonl")
    }

    fn idempotency_path(&self) -> PathBuf {
        self.root_dir.join("idempotency_keys.jsonl")
    }

    fn ensure_session_loaded(
        &self,
        inner: &mut InMemoryState,
        session_id: &SessionId,
    ) -> Result<(), ConversationStoreError> {
        if inner.sessions.contains_key(session_id) {
            return Ok(());
        }

        let session_path = self.session_snapshot_path(session_id);
        if !session_path.exists() {
            return Ok(());
        }

        let mut session: SessionRecord = read_json_file(session_path.as_path())?;
        if matches!(session.status, SessionStatus::Running) {
            session.status = SessionStatus::Interrupted;
            write_json_file(session_path.as_path(), &session)?;
        }

        let messages_path = self.messages_path(session_id);
        let messages = if messages_path.exists() {
            read_jsonl_file::<SessionMessage>(messages_path.as_path())?
        } else {
            Vec::new()
        };

        let approvals_path = self.approvals_path(session_id);
        let approvals = if approvals_path.exists() {
            read_jsonl_file::<ApprovalRequestRecord>(approvals_path.as_path())?
        } else {
            Vec::new()
        };
        let computed_results_path = self.computed_results_path(session_id);
        let computed_results = if computed_results_path.exists() {
            read_jsonl_file::<MessageRecord>(computed_results_path.as_path())?
        } else {
            Vec::new()
        };

        for binding in &session.bindings {
            inner.bindings.insert(binding.clone(), session_id.clone());
        }
        for approval in approvals {
            inner.approvals.insert(
                (approval.session_id.clone(), approval.approval_id.clone()),
                approval,
            );
        }
        inner.messages.insert(session_id.clone(), messages);
        inner
            .computed_results
            .insert(session_id.clone(), computed_results);
        inner.sessions.insert(session_id.clone(), session);
        Ok(())
    }

    fn ensure_session_loaded_for_binding(
        &self,
        inner: &mut InMemoryState,
        binding: &ChannelBinding,
    ) -> Result<Option<SessionId>, ConversationStoreError> {
        if let Some(session_id) = inner.bindings.get(binding).cloned() {
            return Ok(Some(session_id));
        }

        for entry in read_dir_entries(self.sessions_dir().as_path())? {
            if !entry
                .file_type()
                .map_err(|source| ConversationStoreError::Io {
                    path: entry.path(),
                    source,
                })?
                .is_dir()
            {
                continue;
            }

            let session_path = entry.path().join("session.json");
            if !session_path.exists() {
                continue;
            }

            let session: SessionRecord = read_json_file(session_path.as_path())?;
            if session
                .bindings
                .iter()
                .any(|candidate| candidate == binding)
            {
                let session_id = session.session_id.clone();
                self.ensure_session_loaded(inner, &session_id)?;
                return Ok(Some(session_id));
            }
        }

        Ok(None)
    }

    fn idempotency_seen_on_disk(
        &self,
        idempotency_key: &str,
    ) -> Result<bool, ConversationStoreError> {
        let path = self.idempotency_path();
        if !path.exists() {
            return Ok(false);
        }
        let file = File::open(&path).map_err(|source| ConversationStoreError::Io {
            path: path.clone(),
            source,
        })?;
        for (index, line) in BufReader::new(file).lines().enumerate() {
            let line = line.map_err(|source| ConversationStoreError::Io {
                path: path.clone(),
                source,
            })?;
            if line.trim().is_empty() {
                continue;
            }
            let entry =
                serde_json::from_str::<PersistedIdempotencyKey>(&line).map_err(|source| {
                    ConversationStoreError::Deserialize {
                        path: path.clone(),
                        line: index + 1,
                        source,
                    }
                })?;
            if entry.key == idempotency_key {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

#[async_trait]
impl ConversationStore for JsonlConversationStore {
    async fn create_session(
        &self,
        bindings: Vec<ChannelBinding>,
    ) -> Result<SessionRecord, ConversationStoreError> {
        let mut inner = self.inner.lock().await;
        let session = SessionRecord {
            session_id: SessionId::new(),
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            status: SessionStatus::Idle,
            bindings: bindings.clone(),
            last_turn: None,
        };

        create_dir_all(self.session_dir(&session.session_id).as_path())?;
        write_json_file(
            self.session_snapshot_path(&session.session_id).as_path(),
            &session,
        )?;

        for binding in bindings {
            inner.bindings.insert(binding, session.session_id.clone());
        }
        inner
            .sessions
            .insert(session.session_id.clone(), session.clone());
        inner
            .messages
            .insert(session.session_id.clone(), Vec::new());
        inner
            .computed_results
            .insert(session.session_id.clone(), Vec::new());
        Ok(session)
    }

    async fn get_session(&self, session_id: &SessionId) -> Option<SessionRecord> {
        let mut inner = self.inner.lock().await;
        if let Err(err) = self.ensure_session_loaded(&mut inner, session_id) {
            error!(session_id = %session_id, error = %err, "failed to lazily load session");
            return None;
        }
        inner.sessions.get(session_id).cloned()
    }

    async fn find_by_binding(&self, binding: &ChannelBinding) -> Option<SessionRecord> {
        let mut inner = self.inner.lock().await;
        let session_id = match self.ensure_session_loaded_for_binding(&mut inner, binding) {
            Ok(session_id) => session_id,
            Err(err) => {
                error!(error = %err, "failed to lazily load session by binding");
                return None;
            }
        }?;
        inner.sessions.get(&session_id).cloned()
    }

    async fn attach_binding(
        &self,
        session_id: &SessionId,
        binding: ChannelBinding,
    ) -> Result<(), ConversationStoreError> {
        let mut inner = self.inner.lock().await;
        self.ensure_session_loaded(&mut inner, session_id)?;
        let Some(session) = inner.sessions.get_mut(session_id) else {
            return Ok(());
        };
        let should_persist = if !session.bindings.contains(&binding) {
            session.bindings.push(binding.clone());
            session.updated_at = SystemTime::now();
            true
        } else {
            false
        };
        let session_snapshot = if should_persist {
            Some(session.clone())
        } else {
            None
        };
        let _ = session;
        inner.bindings.insert(binding.clone(), session_id.clone());
        if let Some(session_snapshot) = session_snapshot {
            write_json_file(
                self.session_snapshot_path(session_id).as_path(),
                &session_snapshot,
            )?;
        }
        Ok(())
    }

    async fn detach_binding(
        &self,
        session_id: &SessionId,
        binding: &ChannelBinding,
    ) -> Result<(), ConversationStoreError> {
        let mut inner = self.inner.lock().await;
        self.ensure_session_loaded(&mut inner, session_id)?;
        if inner.bindings.get(binding) == Some(session_id) {
            inner.bindings.remove(binding);
        }
        if let Some(session) = inner.sessions.get_mut(session_id) {
            session.bindings.retain(|known| known != binding);
            session.updated_at = SystemTime::now();
            write_json_file(self.session_snapshot_path(session_id).as_path(), session)?;
        }
        Ok(())
    }

    async fn append_message(&self, message: SessionMessage) -> Result<(), ConversationStoreError> {
        let mut inner = self.inner.lock().await;
        self.ensure_session_loaded(&mut inner, &message.session_id)?;
        append_jsonl_file(self.messages_path(&message.session_id).as_path(), &message)?;
        inner
            .messages
            .entry(message.session_id.clone())
            .or_default()
            .push(message.clone());
        if let Some(session) = inner.sessions.get_mut(&message.session_id) {
            session.updated_at = SystemTime::now();
            write_json_file(
                self.session_snapshot_path(&message.session_id).as_path(),
                session,
            )?;
        }
        Ok(())
    }

    async fn list_messages(&self, session_id: &SessionId) -> Vec<SessionMessage> {
        let mut inner = self.inner.lock().await;
        if let Err(err) = self.ensure_session_loaded(&mut inner, session_id) {
            error!(session_id = %session_id, error = %err, "failed to lazily load session messages");
            return Vec::new();
        }
        inner.messages.get(session_id).cloned().unwrap_or_default()
    }

    async fn append_turn_trace(
        &self,
        session_id: &SessionId,
        turn: &TurnRecord,
    ) -> Result<(), ConversationStoreError> {
        let mut inner = self.inner.lock().await;
        self.ensure_session_loaded(&mut inner, session_id)?;
        let computed = computed_results_from_turn(turn);
        if computed.is_empty() {
            return Ok(());
        }
        for message in &computed {
            append_jsonl_file(self.computed_results_path(session_id).as_path(), message)?;
        }
        inner
            .computed_results
            .entry(session_id.clone())
            .or_default()
            .extend(computed);
        Ok(())
    }

    async fn list_recent_computed_results(
        &self,
        session_id: &SessionId,
        limit: usize,
    ) -> Vec<MessageRecord> {
        let mut inner = self.inner.lock().await;
        if let Err(err) = self.ensure_session_loaded(&mut inner, session_id) {
            error!(session_id = %session_id, error = %err, "failed to lazily load session computed results");
            return Vec::new();
        }
        let Some(messages) = inner.computed_results.get(session_id) else {
            return Vec::new();
        };
        let start = messages.len().saturating_sub(limit);
        messages[start..].to_vec()
    }

    async fn update_session(
        &self,
        session_id: &SessionId,
        status: SessionStatus,
        last_turn: Option<TurnRecordSummary>,
    ) -> Result<Option<SessionRecord>, ConversationStoreError> {
        let mut inner = self.inner.lock().await;
        self.ensure_session_loaded(&mut inner, session_id)?;
        let Some(session) = inner.sessions.get_mut(session_id) else {
            return Ok(None);
        };
        session.status = status;
        session.updated_at = SystemTime::now();
        if last_turn.is_some() {
            session.last_turn = last_turn;
        }
        write_json_file(self.session_snapshot_path(session_id).as_path(), session)?;
        Ok(Some(session.clone()))
    }

    async fn record_idempotency(
        &self,
        idempotency_key: &str,
    ) -> Result<bool, ConversationStoreError> {
        let mut inner = self.inner.lock().await;
        if inner.idempotency_keys.contains(idempotency_key) {
            return Ok(false);
        }
        if self.idempotency_seen_on_disk(idempotency_key)? {
            inner.idempotency_keys.insert(idempotency_key.to_owned());
            return Ok(false);
        }
        let persisted = PersistedIdempotencyKey {
            key: idempotency_key.to_owned(),
        };
        append_jsonl_file(self.idempotency_path().as_path(), &persisted)?;
        inner.idempotency_keys.insert(idempotency_key.to_owned());
        Ok(true)
    }

    async fn create_approval(
        &self,
        approval: ApprovalRequestRecord,
    ) -> Result<(), ConversationStoreError> {
        let mut inner = self.inner.lock().await;
        self.ensure_session_loaded(&mut inner, &approval.session_id)?;
        append_jsonl_file(
            self.approvals_path(&approval.session_id).as_path(),
            &approval,
        )?;
        inner.approvals.insert(
            (approval.session_id.clone(), approval.approval_id.clone()),
            approval,
        );
        Ok(())
    }

    async fn get_approval(
        &self,
        session_id: &SessionId,
        approval_id: &ApprovalId,
    ) -> Option<ApprovalRequestRecord> {
        let mut inner = self.inner.lock().await;
        if let Err(err) = self.ensure_session_loaded(&mut inner, session_id) {
            error!(session_id = %session_id, approval_id = %approval_id, error = %err, "failed to lazily load approval");
            return None;
        }
        inner
            .approvals
            .get(&(session_id.clone(), approval_id.clone()))
            .cloned()
    }

    async fn update_approval(
        &self,
        session_id: &SessionId,
        approval_id: &ApprovalId,
        state: ApprovalState,
    ) -> Result<Option<ApprovalRequestRecord>, ConversationStoreError> {
        let mut inner = self.inner.lock().await;
        self.ensure_session_loaded(&mut inner, session_id)?;
        let Some(approval) = inner
            .approvals
            .get_mut(&(session_id.clone(), approval_id.clone()))
        else {
            return Ok(None);
        };
        approval.state = state;
        approval.resolved_at = Some(SystemTime::now());
        append_jsonl_file(self.approvals_path(session_id).as_path(), approval)?;
        Ok(Some(approval.clone()))
    }
}

fn read_dir_entries(path: &Path) -> Result<Vec<fs::DirEntry>, ConversationStoreError> {
    let read_dir = fs::read_dir(path).map_err(|source| ConversationStoreError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    read_dir
        .collect::<Result<Vec<_>, _>>()
        .map_err(|source| ConversationStoreError::Io {
            path: path.to_path_buf(),
            source,
        })
}

fn create_dir_all(path: &Path) -> Result<(), ConversationStoreError> {
    fs::create_dir_all(path).map_err(|source| ConversationStoreError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn read_json_file<T: DeserializeOwned>(path: &Path) -> Result<T, ConversationStoreError> {
    let file = File::open(path).map_err(|source| ConversationStoreError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_reader(file).map_err(|source| ConversationStoreError::Deserialize {
        path: path.to_path_buf(),
        line: source.line(),
        source,
    })
}

fn read_jsonl_file<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>, ConversationStoreError> {
    let file = File::open(path).map_err(|source| ConversationStoreError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let reader = BufReader::new(file);
    let mut records = Vec::new();
    for (index, line) in reader.lines().enumerate() {
        let line = line.map_err(|source| ConversationStoreError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let record = serde_json::from_str::<T>(&line).map_err(|source| {
            ConversationStoreError::Deserialize {
                path: path.to_path_buf(),
                line: index + 1,
                source,
            }
        })?;
        records.push(record);
    }
    Ok(records)
}

fn write_json_file<T: Serialize>(path: &Path, value: &T) -> Result<(), ConversationStoreError> {
    if let Some(parent) = path.parent() {
        create_dir_all(parent)?;
    }
    let temp_path = path.with_extension(format!("tmp-{}", Uuid::new_v4()));
    let payload =
        serde_json::to_vec_pretty(value).map_err(|source| ConversationStoreError::Serialize {
            path: path.to_path_buf(),
            source,
        })?;
    fs::write(&temp_path, payload).map_err(|source| ConversationStoreError::Io {
        path: temp_path.clone(),
        source,
    })?;
    fs::rename(&temp_path, path).map_err(|source| ConversationStoreError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(())
}

fn append_jsonl_file<T: Serialize>(path: &Path, value: &T) -> Result<(), ConversationStoreError> {
    if let Some(parent) = path.parent() {
        create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|source| ConversationStoreError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    serde_json::to_writer(&mut file, value).map_err(|source| {
        ConversationStoreError::Serialize {
            path: path.to_path_buf(),
            source,
        }
    })?;
    file.write_all(b"\n")
        .map_err(|source| ConversationStoreError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    file.flush().map_err(|source| ConversationStoreError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(())
}

fn computed_results_from_turn(turn: &TurnRecord) -> Vec<MessageRecord> {
    turn.messages
        .iter()
        .filter(|message| matches!(message, MessageRecord::McpResult(_)))
        .cloned()
        .collect()
}

/// Session orchestration service shared by HTTP routes and channel adapters.
pub struct SessionService<R, S> {
    runner: Arc<R>,
    store: Arc<S>,
    /// Broadcast channel used to fan out typed session events to APIs and CLIs.
    events_tx: broadcast::Sender<SessionEvent>,
    runtime_harness_listeners: Vec<Arc<dyn RuntimeHarnessListener>>,
}

impl<R, S> Clone for SessionService<R, S> {
    fn clone(&self) -> Self {
        Self {
            runner: Arc::clone(&self.runner),
            store: Arc::clone(&self.store),
            events_tx: self.events_tx.clone(),
            runtime_harness_listeners: self.runtime_harness_listeners.clone(),
        }
    }
}

impl<R, S> SessionService<R, S>
where
    R: TurnRunner,
    S: ConversationStore,
{
    /// Creates a new session service with an in-memory broadcast event stream.
    pub fn new(runner: R, store: S) -> Self {
        let (events_tx, _) = broadcast::channel(512);
        Self {
            runner: Arc::new(runner),
            store: Arc::new(store),
            runtime_harness_listeners: vec![SseRuntimeHarnessListener::new(events_tx.clone(), 512)],
            events_tx,
        }
    }

    /// Registers one runtime harness listener used for future turns.
    pub fn with_runtime_harness_listener(
        mut self,
        listener: Arc<dyn RuntimeHarnessListener>,
    ) -> Self {
        self.runtime_harness_listeners.push(listener);
        self
    }

    /// Subscribes to session events for streaming APIs.
    pub fn subscribe(&self) -> broadcast::Receiver<SessionEvent> {
        self.events_tx.subscribe()
    }

    /// Shuts down runner-owned resources before the control-plane process exits.
    pub async fn shutdown(&self) -> Result<(), ControlPlaneError> {
        self.runner.shutdown().await?;
        Ok(())
    }

    /// Creates a session optionally bound to an external channel and eagerly
    /// prepares any runner state needed before the first user turn.
    pub async fn create_session(
        &self,
        request: CreateSessionRequest,
    ) -> Result<SessionRecord, ControlPlaneError> {
        let bindings = request
            .channel
            .zip(
                request
                    .external_conversation_id
                    .zip(request.external_user_id),
            )
            .map(|(channel, (external_conversation_id, external_user_id))| {
                vec![ChannelBinding {
                    channel,
                    external_workspace_id: None,
                    external_conversation_id,
                    external_channel_id: None,
                    external_thread_id: None,
                    external_user_id,
                }]
            })
            .unwrap_or_default();
        self.create_session_with_bindings(bindings).await
    }

    async fn create_session_with_bindings(
        &self,
        bindings: Vec<ChannelBinding>,
    ) -> Result<SessionRecord, ControlPlaneError> {
        let session = self.store.create_session(bindings).await?;
        if let Err(error) = self.runner.prepare_session(&session.session_id).await {
            error!(
                session_id = %session.session_id,
                error = %error,
                "session failed during MCP initialization"
            );
            return Err(error.into());
        }
        self.emit(SessionEvent::SessionCreated {
            session: session.clone(),
        });
        Ok(session)
    }

    /// Returns one session by ID.
    pub async fn get_session(&self, session_id: &SessionId) -> Option<SessionRecord> {
        self.store.get_session(session_id).await
    }

    /// Lists persisted messages for one session.
    pub async fn get_messages(&self, session_id: &SessionId) -> Vec<SessionMessage> {
        self.store.list_messages(session_id).await
    }

    /// Finds one session by an active channel binding.
    pub async fn find_session_by_binding(&self, binding: &ChannelBinding) -> Option<SessionRecord> {
        self.store.find_by_binding(binding).await
    }

    /// Emits a typed delivery outcome after a connector attempts to send.
    pub fn emit_channel_delivery(
        &self,
        session_id: SessionId,
        channel: ChannelKind,
        status: DeliveryStatus,
    ) {
        self.emit(SessionEvent::ChannelDeliveryAttempted {
            session_id,
            channel,
            status,
        });
    }

    /// Sends a text message to an already-known session without requiring a
    /// channel binding lookup.
    pub async fn send_session_text(
        &self,
        session_id: &SessionId,
        text: String,
        idempotency_key: String,
        channel: ChannelKind,
        response_target: ResponseTarget,
    ) -> Result<ChannelDispatchResult, ControlPlaneError> {
        self.send_session_text_with_runtime_listeners(
            session_id,
            text,
            idempotency_key,
            channel,
            response_target,
            Vec::new(),
        )
        .await
    }

    pub async fn send_session_text_with_runtime_listeners(
        &self,
        session_id: &SessionId,
        text: String,
        idempotency_key: String,
        channel: ChannelKind,
        response_target: ResponseTarget,
        extra_runtime_harness_listeners: Vec<Arc<dyn RuntimeHarnessListener>>,
    ) -> Result<ChannelDispatchResult, ControlPlaneError> {
        if !self.store.record_idempotency(&idempotency_key).await? {
            let session = self
                .store
                .get_session(session_id)
                .await
                .ok_or(ControlPlaneError::SessionNotFound)?;
            warn!(
                session_id = %session.session_id,
                idempotency_key = %idempotency_key,
                "duplicate message ignored"
            );
            return Ok(ChannelDispatchResult {
                session,
                outbound: Vec::new(),
                approval: None,
                was_duplicate: true,
            });
        }
        self.handle_user_text(
            session_id.clone(),
            text,
            channel,
            response_target,
            None,
            None,
            extra_runtime_harness_listeners,
        )
        .await
    }

    /// Creates a pending approval request.
    pub async fn create_approval_request(
        &self,
        session_id: &SessionId,
        prompt: impl Into<String>,
    ) -> Result<ApprovalRequestRecord, ControlPlaneError> {
        let _session = self
            .store
            .get_session(session_id)
            .await
            .ok_or(ControlPlaneError::SessionNotFound)?;
        let approval = ApprovalRequestRecord {
            approval_id: ApprovalId::new(),
            session_id: session_id.clone(),
            prompt: prompt.into(),
            state: ApprovalState::Pending,
            created_at: SystemTime::now(),
            resolved_at: None,
        };
        self.store.create_approval(approval.clone()).await?;
        let _ = self
            .store
            .update_session(session_id, SessionStatus::WaitingForApproval, None)
            .await?;
        self.emit(SessionEvent::ApprovalRequested {
            session_id: session_id.clone(),
            approval: approval.clone(),
        });
        Ok(approval)
    }

    /// Applies an approval decision to a pending approval request.
    pub async fn submit_approval(
        &self,
        session_id: &SessionId,
        approval_id: &ApprovalId,
        request: SubmitApprovalRequest,
    ) -> Result<ApprovalRequestRecord, ControlPlaneError> {
        let current = self
            .store
            .get_approval(session_id, approval_id)
            .await
            .ok_or(ControlPlaneError::ApprovalNotFound)?;
        if current.state != ApprovalState::Pending {
            return Err(ControlPlaneError::ApprovalNotPending);
        }

        let new_state = match request.decision {
            ApprovalDecision::Approve => ApprovalState::Approved,
            ApprovalDecision::Reject => ApprovalState::Rejected,
        };
        let approval = self
            .store
            .update_approval(session_id, approval_id, new_state)
            .await?
            .ok_or(ControlPlaneError::ApprovalNotFound)?;
        let new_status = match request.decision {
            ApprovalDecision::Approve => SessionStatus::Idle,
            ApprovalDecision::Reject => SessionStatus::Interrupted,
        };
        let _ = self
            .store
            .update_session(session_id, new_status, None)
            .await?;
        self.emit(SessionEvent::ApprovalResolved {
            session_id: session_id.clone(),
            approval: approval.clone(),
        });
        Ok(approval)
    }

    /// Processes one normalized envelope from any channel.
    pub async fn dispatch_envelope(
        &self,
        envelope: ChannelEnvelope,
    ) -> Result<ChannelDispatchResult, ControlPlaneError> {
        self.dispatch_envelope_with_runtime_listeners(envelope, Vec::new())
            .await
    }

    pub async fn dispatch_envelope_with_runtime_listeners(
        &self,
        envelope: ChannelEnvelope,
        extra_runtime_harness_listeners: Vec<Arc<dyn RuntimeHarnessListener>>,
    ) -> Result<ChannelDispatchResult, ControlPlaneError> {
        let binding = ChannelBinding {
            channel: envelope.channel,
            external_workspace_id: envelope.external_workspace_id.clone(),
            external_conversation_id: envelope.external_conversation_id.clone(),
            external_channel_id: envelope.external_channel_id.clone(),
            external_thread_id: envelope.external_thread_id.clone(),
            external_user_id: envelope.external_user_id.clone(),
        };

        if !self
            .store
            .record_idempotency(&envelope.idempotency_key)
            .await?
        {
            let session = self
                .store
                .find_by_binding(&binding)
                .await
                .ok_or(ControlPlaneError::SessionNotFound)?;
            warn!(
                session_id = %session.session_id,
                channel = ?binding.channel,
                idempotency_key = %envelope.idempotency_key,
                "duplicate channel envelope ignored"
            );
            return Ok(ChannelDispatchResult {
                session,
                outbound: Vec::new(),
                approval: None,
                was_duplicate: true,
            });
        }

        let delivery_target = delivery_target_for_envelope(&envelope);
        let existing_session = self.store.find_by_binding(&binding).await;

        match envelope.intent {
            ChannelIntent::UserText { text } => {
                let session = match existing_session {
                    Some(session) => session,
                    None => {
                        self.create_session_with_bindings(vec![binding.clone()])
                            .await?
                    }
                };
                self.store
                    .attach_binding(&session.session_id, binding)
                    .await?;
                self.handle_user_text(
                    session.session_id,
                    text,
                    envelope.channel,
                    response_target_for_channel(envelope.channel),
                    envelope.external_message_id,
                    delivery_target,
                    extra_runtime_harness_listeners,
                )
                .await
            }
            ChannelIntent::ResetSession => {
                if let Some(session) = existing_session.as_ref() {
                    self.store
                        .detach_binding(&session.session_id, &binding)
                        .await?;
                }
                let session = self
                    .create_session_with_bindings(vec![binding.clone()])
                    .await?;
                self.store
                    .attach_binding(&session.session_id, binding)
                    .await?;
                Ok(ChannelDispatchResult {
                    session: session.clone(),
                    outbound: vec![ChannelResponseEnvelope {
                        session_id: session.session_id.clone(),
                        kind: OutboundMessageKind::Status,
                        text: "started a new session for this thread".to_owned(),
                        attachment: None,
                        delivery_target,
                    }],
                    approval: None,
                    was_duplicate: false,
                })
            }
            ChannelIntent::ApprovalResponse {
                approval_id,
                decision,
            } => {
                let session = existing_session.ok_or(ControlPlaneError::SessionNotFound)?;
                let approval = self
                    .submit_approval(
                        &session.session_id,
                        &approval_id,
                        SubmitApprovalRequest { decision },
                    )
                    .await?;
                let session = self
                    .store
                    .get_session(&session.session_id)
                    .await
                    .ok_or(ControlPlaneError::SessionNotFound)?;
                Ok(ChannelDispatchResult {
                    session: session.clone(),
                    outbound: vec![ChannelResponseEnvelope {
                        session_id: session.session_id.clone(),
                        kind: OutboundMessageKind::Status,
                        text: format!("approval {} recorded", approval.approval_id),
                        attachment: None,
                        delivery_target,
                    }],
                    approval: Some(approval),
                    was_duplicate: false,
                })
            }
            ChannelIntent::Interrupt => {
                let session = existing_session.ok_or(ControlPlaneError::SessionNotFound)?;
                let session = self
                    .store
                    .update_session(&session.session_id, SessionStatus::Interrupted, None)
                    .await?
                    .ok_or(ControlPlaneError::SessionNotFound)?;
                Ok(ChannelDispatchResult {
                    session: session.clone(),
                    outbound: vec![ChannelResponseEnvelope {
                        session_id: session.session_id,
                        kind: OutboundMessageKind::Status,
                        text: "session interrupted".to_owned(),
                        attachment: None,
                        delivery_target,
                    }],
                    approval: None,
                    was_duplicate: false,
                })
            }
            ChannelIntent::Resume => {
                let session = existing_session.ok_or(ControlPlaneError::SessionNotFound)?;
                let session = self
                    .store
                    .update_session(&session.session_id, SessionStatus::Idle, None)
                    .await?
                    .ok_or(ControlPlaneError::SessionNotFound)?;
                Ok(ChannelDispatchResult {
                    session: session.clone(),
                    outbound: vec![ChannelResponseEnvelope {
                        session_id: session.session_id,
                        kind: OutboundMessageKind::Status,
                        text: "session resumed".to_owned(),
                        attachment: None,
                        delivery_target,
                    }],
                    approval: None,
                    was_duplicate: false,
                })
            }
            ChannelIntent::StatusRequest => {
                let session = match existing_session {
                    Some(session) => session,
                    None => {
                        self.create_session_with_bindings(vec![binding.clone()])
                            .await?
                    }
                };
                self.store
                    .attach_binding(&session.session_id, binding)
                    .await?;
                let session = self
                    .store
                    .get_session(&session.session_id)
                    .await
                    .ok_or(ControlPlaneError::SessionNotFound)?;
                Ok(ChannelDispatchResult {
                    outbound: vec![ChannelResponseEnvelope {
                        session_id: session.session_id.clone(),
                        kind: OutboundMessageKind::Status,
                        text: format!("session status: {:?}", session.status),
                        attachment: None,
                        delivery_target,
                    }],
                    session,
                    approval: None,
                    was_duplicate: false,
                })
            }
        }
    }

    async fn handle_user_text(
        &self,
        session_id: SessionId,
        text: String,
        channel: ChannelKind,
        response_target: ResponseTarget,
        external_message_id: Option<String>,
        delivery_target: Option<ChannelDeliveryTarget>,
        extra_runtime_harness_listeners: Vec<Arc<dyn RuntimeHarnessListener>>,
    ) -> Result<ChannelDispatchResult, ControlPlaneError> {
        let history = self.store.list_messages(&session_id).await;
        let user_message = SessionMessage {
            message_id: SessionMessageId::new(),
            session_id: session_id.clone(),
            role: SessionMessageRole::User,
            content: text.clone(),
            created_at: SystemTime::now(),
            channel: Some(channel),
            external_message_id,
        };
        self.store.append_message(user_message.clone()).await?;
        self.emit(SessionEvent::SessionMessageReceived {
            session_id: session_id.clone(),
            message: user_message,
        });
        self.emit(SessionEvent::TurnQueued {
            session_id: session_id.clone(),
        });
        self.store
            .update_session(&session_id, SessionStatus::Running, None)
            .await?;
        self.emit(SessionEvent::TurnStarted {
            session_id: session_id.clone(),
        });

        let prior_user_turns = history
            .iter()
            .filter(|message| message.role == SessionMessageRole::User)
            .count() as u32;
        let turn_number = prior_user_turns.saturating_add(1);
        let conversation_history = history
            .into_iter()
            .filter_map(|message| match message.role {
                SessionMessageRole::User => Some(ConversationMessage {
                    timestamp: message.created_at,
                    role: ConversationRole::User,
                    content: message.content,
                }),
                SessionMessageRole::Assistant => Some(ConversationMessage {
                    timestamp: message.created_at,
                    role: ConversationRole::Assistant,
                    content: message.content,
                }),
                SessionMessageRole::System => None,
            })
            .collect::<Vec<_>>();
        let recent_session_messages = self
            .store
            .list_recent_computed_results(&session_id, RECENT_COMPUTED_RESULTS_LIMIT)
            .await;

        let turn = self
            .runner
            .run_turn(TurnRunnerInput {
                session_id: session_id.clone(),
                turn_number,
                conversation_history,
                recent_session_messages,
                user_message: text,
                response_target,
                runtime_harness_listeners: self
                    .runtime_harness_listeners
                    .iter()
                    .cloned()
                    .chain(extra_runtime_harness_listeners.into_iter())
                    .collect(),
            })
            .await;

        match turn {
            Ok(TurnRunnerOutput {
                final_text,
                display_text,
                model_name,
                elapsed_ms,
                termination,
                usage,
                events: _events,
                turn,
                generated_artifacts,
            }) => {
                self.store.append_turn_trace(&session_id, &turn).await?;
                let assistant_message = SessionMessage {
                    message_id: SessionMessageId::new(),
                    session_id: session_id.clone(),
                    role: SessionMessageRole::Assistant,
                    content: final_text.clone(),
                    created_at: SystemTime::now(),
                    channel: Some(channel),
                    external_message_id: None,
                };
                self.store.append_message(assistant_message.clone()).await?;
                let summary = TurnRecordSummary {
                    turn_number,
                    model_name,
                    elapsed_ms,
                    final_text: final_text.clone(),
                    termination: termination.clone(),
                    usage,
                    completed_at: SystemTime::now(),
                };
                let new_status = if matches!(termination, agent_runtime::TerminationReason::Final) {
                    SessionStatus::Completed
                } else {
                    SessionStatus::Failed
                };
                let session = self
                    .store
                    .update_session(&session_id, new_status, Some(summary.clone()))
                    .await?
                    .ok_or(ControlPlaneError::SessionNotFound)?;
                self.emit(SessionEvent::TurnCompleted {
                    session_id: session_id.clone(),
                    summary,
                });
                Ok(ChannelDispatchResult {
                    session: session.clone(),
                    outbound: vec![ChannelResponseEnvelope {
                        session_id,
                        kind: OutboundMessageKind::Reply,
                        text: display_text,
                        attachment: slack_generated_attachment(channel, &generated_artifacts),
                        delivery_target,
                    }],
                    approval: None,
                    was_duplicate: false,
                })
            }
            Err(error) => {
                error!(
                    session_id = %session_id,
                    channel = ?channel,
                    error = %error,
                    "server turn failed"
                );
                let session = self
                    .store
                    .update_session(&session_id, SessionStatus::Failed, None)
                    .await?
                    .ok_or(ControlPlaneError::SessionNotFound)?;
                Ok(ChannelDispatchResult {
                    session,
                    outbound: vec![ChannelResponseEnvelope {
                        session_id,
                        kind: OutboundMessageKind::Error,
                        text: error.to_string(),
                        attachment: None,
                        delivery_target,
                    }],
                    approval: None,
                    was_duplicate: false,
                })
            }
        }
    }

    fn emit(&self, event: SessionEvent) {
        let _ = self.events_tx.send(event);
    }
}

fn delivery_target_for_envelope(envelope: &ChannelEnvelope) -> Option<ChannelDeliveryTarget> {
    if matches!(envelope.channel, ChannelKind::Api | ChannelKind::Cli) {
        return None;
    }
    Some(ChannelDeliveryTarget {
        channel: envelope.channel,
        external_workspace_id: envelope.external_workspace_id.clone(),
        external_conversation_id: envelope.external_conversation_id.clone(),
        external_channel_id: envelope.external_channel_id.clone(),
        external_thread_id: envelope.external_thread_id.clone(),
        external_user_id: envelope.external_user_id.clone(),
    })
}

fn response_target_for_channel(channel: ChannelKind) -> ResponseTarget {
    match channel {
        ChannelKind::Api => ResponseTarget {
            client: ResponseClient::Api,
            format: ResponseFormat::PlainText,
        },
        ChannelKind::Cli => ResponseTarget {
            client: ResponseClient::Cli,
            format: ResponseFormat::Markdown,
        },
        ChannelKind::Slack => ResponseTarget {
            client: ResponseClient::Slack,
            format: ResponseFormat::SlackMrkdwn,
        },
        ChannelKind::WhatsApp => ResponseTarget {
            client: ResponseClient::WhatsApp,
            format: ResponseFormat::WhatsAppText,
        },
    }
}

fn slack_generated_attachment(
    channel: ChannelKind,
    generated_artifacts: &[GeneratedArtifact],
) -> Option<ChannelResponseAttachment> {
    if !matches!(channel, ChannelKind::Slack) {
        return None;
    }
    let artifact = generated_artifacts
        .iter()
        .rev()
        .find(|artifact| matches!(artifact.kind, GeneratedArtifactKind::Image))?;
    Some(ChannelResponseAttachment::LocalImage(
        ChannelLocalImageAttachment {
            file_name: artifact.file_name.clone(),
            alt_text: humanize_artifact_name(&artifact.file_name),
            mime_type: artifact.mime_type.clone(),
            caption: None,
            local_path: Some(artifact.path.display().to_string()),
        },
    ))
}

fn humanize_artifact_name(file_name: &str) -> String {
    let stem = Path::new(file_name)
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or(file_name);
    let mut label = String::with_capacity(stem.len());
    let mut previous_was_space = false;
    for ch in stem.chars() {
        let mapped = if ch == '_' || ch == '-' { ' ' } else { ch };
        if mapped == ' ' {
            if !previous_was_space {
                label.push(mapped);
            }
            previous_was_space = true;
        } else {
            label.push(mapped);
            previous_was_space = false;
        }
    }
    if label.trim().is_empty() {
        "Generated chart".to_owned()
    } else {
        label
    }
}

#[cfg(test)]
mod tests {
    use std::time::SystemTime;

    use agent_runtime::{
        MessageId, StepRecord, TerminationReason, TurnId, UsageSummary,
        state::{McpCapabilityTarget, McpResultMessageRecord, UserMessageRecord},
    };
    use mcp_metadata::CapabilityKind;

    use super::*;

    #[test]
    fn computed_results_from_turn_keeps_only_mcp_results() {
        let turn = TurnRecord {
            turn_id: TurnId::new(),
            started_at: SystemTime::now(),
            ended_at: SystemTime::now(),
            steps: Vec::<StepRecord>::new(),
            messages: vec![
                MessageRecord::User(UserMessageRecord {
                    message_id: MessageId::new(),
                    timestamp: SystemTime::now(),
                    content: "show weekly users".to_owned(),
                }),
                MessageRecord::McpResult(McpResultMessageRecord {
                    message_id: MessageId::new(),
                    timestamp: SystemTime::now(),
                    target: McpCapabilityTarget {
                        server_name: agent_runtime::ServerName::new("ex-vol")
                            .expect("valid server"),
                        capability_kind: CapabilityKind::Tool,
                        capability_id: "run_select_query".to_owned(),
                    },
                    result_summary: "{\"rows\":[[\"2026-04-05\",8]]}".to_owned(),
                    error: None,
                }),
            ],
            final_text: Some("done".to_owned()),
            termination: TerminationReason::Final,
            usage: UsageSummary::default(),
        };

        let computed = computed_results_from_turn(&turn);

        assert_eq!(computed.len(), 1);
        assert!(matches!(computed[0], MessageRecord::McpResult(_)));
    }

    #[tokio::test]
    async fn in_memory_store_returns_recent_computed_results_in_order() {
        let store = InMemoryConversationStore::default();
        let session = store
            .create_session(Vec::new())
            .await
            .expect("session should create");
        let first = sample_turn("2026-03-29", 14);
        let second = sample_turn("2026-04-05", 8);

        store
            .append_turn_trace(&session.session_id, &first)
            .await
            .expect("first turn should persist");
        store
            .append_turn_trace(&session.session_id, &second)
            .await
            .expect("second turn should persist");

        let recent = store
            .list_recent_computed_results(&session.session_id, 2)
            .await;

        assert_eq!(recent.len(), 2);
        assert!(recent[0].summary_line().contains("2026-03-29"));
        assert!(recent[1].summary_line().contains("2026-04-05"));
    }

    fn sample_turn(week_start: &str, new_users: u32) -> TurnRecord {
        TurnRecord {
            turn_id: TurnId::new(),
            started_at: SystemTime::now(),
            ended_at: SystemTime::now(),
            steps: Vec::<StepRecord>::new(),
            messages: vec![MessageRecord::McpResult(McpResultMessageRecord {
                message_id: MessageId::new(),
                timestamp: SystemTime::now(),
                target: McpCapabilityTarget {
                    server_name: agent_runtime::ServerName::new("ex-vol").expect("valid server"),
                    capability_kind: CapabilityKind::Tool,
                    capability_id: "run_select_query".to_owned(),
                },
                result_summary: format!(
                    "{{\"columns\":[\"week_start\",\"new_users\"],\"rows\":[[\"{week_start}\",{new_users}]]}}"
                ),
                error: None,
            })],
            final_text: Some("done".to_owned()),
            termination: TerminationReason::Final,
            usage: UsageSummary::default(),
        }
    }
}

/// Control-plane level failures.
#[derive(Debug, Error)]
pub enum ControlPlaneError {
    #[error("session not found")]
    SessionNotFound,
    #[error("approval not found")]
    ApprovalNotFound,
    #[error("approval is not pending")]
    ApprovalNotPending,
    #[error("conversation store failed: {0}")]
    Store(#[from] ConversationStoreError),
    #[error("turn runner failed: {0}")]
    Runner(#[from] TurnRunnerError),
}
