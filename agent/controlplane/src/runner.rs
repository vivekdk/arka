//! Turn-runner abstraction used by the control plane.
//!
//! The control plane owns sessions and channels, but it delegates one user turn
//! at a time to a runner so the existing runtime can remain single-turn.

use std::{
    collections::HashMap,
    fs,
    path::Path,
    path::PathBuf,
    sync::Arc,
    time::{Instant, SystemTime},
};

use crate::{
    observability::{
        DebugTurnStatus, RuntimeHarnessFanoutSink, RuntimeHarnessListener, TurnDebugMetadata,
        TurnDebugSnapshot, enabled_server_names, path_display,
    },
    types::SessionId,
};
use agent_runtime::{
    AgentRuntime, ConversationMessage, EventSink, McpSession, MessageRecord, ModelConfig,
    ResponseTarget, RuntimeError as AgentRuntimeError, RuntimeEvent, RuntimeExecutor,
    RuntimeLimits, ServerName, TerminationReason, TurnId, TurnRecord, UsageSummary,
};
use async_trait::async_trait;
use thiserror::Error;
use tokio::sync::Mutex;

/// Input passed from the control plane into one runtime turn.
#[derive(Clone)]
pub struct TurnRunnerInput {
    /// Session that owns the turn.
    pub session_id: SessionId,
    /// One-based turn number inside the session.
    pub turn_number: u32,
    /// Prior conversation history normalized into runtime records.
    pub conversation_history: Vec<ConversationMessage>,
    /// Recent persisted runtime messages from prior turns in this session.
    pub recent_session_messages: Vec<MessageRecord>,
    /// Fresh user text for the turn.
    pub user_message: String,
    /// Resolved client and formatting target for the turn reply.
    pub response_target: ResponseTarget,
    /// Runtime harness listeners for this turn.
    pub runtime_harness_listeners: Vec<Arc<dyn RuntimeHarnessListener>>,
}

/// Output returned from one runtime turn.
#[derive(Clone, Debug, PartialEq)]
pub struct TurnRunnerOutput {
    /// Final assistant text returned by the runtime.
    pub final_text: String,
    /// Client-formatted assistant text returned by the runtime.
    pub display_text: String,
    /// Model name used for the turn.
    pub model_name: String,
    /// Total elapsed wall-clock time in milliseconds.
    pub elapsed_ms: u64,
    /// Why the turn ended.
    pub termination: agent_runtime::TerminationReason,
    /// Aggregate token usage for the turn.
    pub usage: agent_runtime::UsageSummary,
    /// Runtime lifecycle events emitted during execution.
    pub events: Vec<agent_runtime::RuntimeEvent>,
    /// Canonical runtime turn record.
    pub turn: agent_runtime::TurnRecord,
    /// New local artifacts produced during the turn inside the session workspace.
    pub generated_artifacts: Vec<GeneratedArtifact>,
}

/// Kind of local artifact generated during a runtime turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeneratedArtifactKind {
    Image,
    Document,
}

/// One local artifact generated during a runtime turn.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeneratedArtifact {
    pub kind: GeneratedArtifactKind,
    pub path: PathBuf,
    pub file_name: String,
    pub mime_type: Option<String>,
}

/// Static runtime configuration used by the control plane for every turn.
#[derive(Clone, Debug, PartialEq)]
pub struct RuntimeExecutionConfig {
    /// Prompt template path used for every turn.
    pub system_prompt_path: PathBuf,
    /// Root directory under which per-session workspaces are created.
    pub workspace_root: PathBuf,
    /// MCP registry path used to prepare sessions.
    pub registry_path: PathBuf,
    /// Sub-agent registry path used to render prompts and delegate execution.
    pub subagent_registry_path: PathBuf,
    /// Optional tool-policy overlay config path.
    pub tool_policy_path: Option<PathBuf>,
    /// Optional allowlist of MCP servers enabled for all turns.
    pub enabled_servers: Option<Vec<ServerName>>,
    /// Runtime safety and timeout limits.
    pub limits: RuntimeLimits,
    /// Model selection and provider options used for every turn.
    pub model_config: ModelConfig,
}

/// Trait implemented by any component capable of executing one runtime turn.
#[async_trait]
pub trait TurnRunner: Send + Sync {
    /// Ensures a specific control-plane session has any runner-owned state
    /// allocated before the first turn begins.
    ///
    /// For the runtime-backed implementation this eagerly creates and caches
    /// the MCP connection set associated with the session, which moves server
    /// startup, handshake, and tool discovery out of the first user message.
    async fn prepare_session(&self, _session_id: &SessionId) -> Result<(), TurnRunnerError> {
        Ok(())
    }

    /// Releases any runner-owned resources during control-plane shutdown.
    ///
    /// The runtime-backed implementation drops all cached MCP sessions so each
    /// connection can clean up its child process before the server exits.
    async fn shutdown(&self) -> Result<(), TurnRunnerError> {
        Ok(())
    }

    /// Runs one user turn using normalized conversation history.
    async fn run_turn(&self, input: TurnRunnerInput) -> Result<TurnRunnerOutput, TurnRunnerError>;
}

/// Runtime-backed turn runner that delegates to the existing `AgentRuntime`.
pub struct RuntimeTurnRunner<A> {
    runtime: Arc<AgentRuntime<A>>,
    config: RuntimeExecutionConfig,
    /// Cached MCP sessions keyed by control-plane session.
    session_mcp: Arc<Mutex<HashMap<SessionId, Arc<Mutex<McpSession>>>>>,
}

impl<A> RuntimeTurnRunner<A>
where
    A: agent_runtime::model::ModelAdapter,
{
    /// Wraps an `AgentRuntime` with static execution defaults.
    pub fn new(runtime: AgentRuntime<A>, config: RuntimeExecutionConfig) -> Self {
        Self {
            runtime: Arc::new(runtime),
            config,
            session_mcp: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    async fn get_or_prepare_mcp_session(
        &self,
        session_id: &SessionId,
    ) -> Result<Arc<Mutex<McpSession>>, TurnRunnerError> {
        if let Some(existing) = self.session_mcp.lock().await.get(session_id).cloned() {
            return Ok(existing);
        }

        let prepared = self
            .runtime
            .prepare_mcp_session(
                &self.config.registry_path,
                self.config.enabled_servers.as_deref(),
            )
            .await?;
        let prepared = Arc::new(Mutex::new(prepared));
        let mut sessions = self.session_mcp.lock().await;
        // If another task prepared the same session concurrently, keep the
        // existing entry and drop the redundant prepared value.
        Ok(sessions
            .entry(session_id.clone())
            .or_insert_with(|| prepared.clone())
            .clone())
    }
}

#[async_trait]
impl<A> TurnRunner for RuntimeTurnRunner<A>
where
    A: agent_runtime::model::ModelAdapter,
{
    async fn prepare_session(&self, session_id: &SessionId) -> Result<(), TurnRunnerError> {
        let _ = prepare_session_workspace_at(&self.config.workspace_root, session_id)?;
        let _ = self.get_or_prepare_mcp_session(session_id).await?;
        Ok(())
    }

    async fn shutdown(&self) -> Result<(), TurnRunnerError> {
        self.session_mcp.lock().await.clear();
        Ok(())
    }

    async fn run_turn(&self, input: TurnRunnerInput) -> Result<TurnRunnerOutput, TurnRunnerError> {
        let session_mcp = self.get_or_prepare_mcp_session(&input.session_id).await?;
        let mut session_mcp = session_mcp.lock().await;
        let turn_started_at = SystemTime::now();
        let started = Instant::now();
        let working_directory =
            prepare_session_workspace_at(&self.config.workspace_root, &input.session_id)?;
        let request = agent_runtime::RunRequest {
            system_prompt_path: self.config.system_prompt_path.clone(),
            working_directory: working_directory.clone(),
            conversation_history: input.conversation_history,
            recent_session_messages: input.recent_session_messages,
            user_message: input.user_message,
            response_target: input.response_target,
            registry_path: self.config.registry_path.clone(),
            subagent_registry_path: self.config.subagent_registry_path.clone(),
            tool_policy_path: self.config.tool_policy_path.clone(),
            enabled_servers: self.config.enabled_servers.clone(),
            limits: self.config.limits.clone(),
            model_config: self.config.model_config.clone(),
        };
        let turn_debug_metadata = TurnDebugMetadata {
            model_name: self.config.model_config.model_name.clone(),
            system_prompt_path: path_display(&self.config.system_prompt_path),
            registry_path: path_display(&self.config.registry_path),
            subagent_registry_path: path_display(&self.config.subagent_registry_path),
            enabled_servers: enabled_server_names(self.config.enabled_servers.as_deref()),
        };
        let mut sink = RuntimeHarnessFanoutSink::new(
            input.session_id.clone(),
            input.turn_number,
            turn_debug_metadata,
            input.runtime_harness_listeners.clone(),
        );
        let outcome = self
            .runtime
            .run_turn_with_mcp_session_and_sink(request, &mut session_mcp, &mut sink)
            .await;
        let elapsed_ms = started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
        match outcome {
            Ok(outcome) => {
                let turn = outcome.turn.clone();
                let snapshot = TurnDebugSnapshot {
                    session_id: input.session_id.clone(),
                    turn_number: input.turn_number,
                    status: final_status_for_termination(&outcome.termination),
                    model_name: self.config.model_config.model_name.clone(),
                    elapsed_ms,
                    error_message: None,
                    system_prompt_path: path_display(&self.config.system_prompt_path),
                    registry_path: path_display(&self.config.registry_path),
                    subagent_registry_path: path_display(&self.config.subagent_registry_path),
                    enabled_servers: enabled_server_names(self.config.enabled_servers.as_deref()),
                    turn: turn.clone(),
                };
                sink.emit_turn_snapshot(snapshot);
                let events = sink.into_events();
                let generated_artifacts =
                    collect_generated_artifacts(&working_directory, turn_started_at)?;
                Ok(TurnRunnerOutput {
                    final_text: outcome.final_text,
                    display_text: outcome.display_text,
                    model_name: self.config.model_config.model_name.clone(),
                    elapsed_ms,
                    termination: outcome.termination,
                    usage: outcome.usage,
                    events,
                    turn,
                    generated_artifacts,
                })
            }
            Err(error) => {
                emit_failed_turn_snapshot(
                    &mut sink,
                    &input.session_id,
                    input.turn_number,
                    &self.config,
                    elapsed_ms,
                    &error,
                );
                Err(error.into())
            }
        }
    }
}

fn emit_failed_turn_snapshot(
    sink: &mut RuntimeHarnessFanoutSink,
    session_id: &SessionId,
    turn_number: u32,
    config: &RuntimeExecutionConfig,
    elapsed_ms: u64,
    error: &AgentRuntimeError,
) {
    let Some(turn_id) = sink.current_turn_id() else {
        return;
    };
    let termination = failure_termination(error);
    let usage = sink.accumulated_usage();
    let ended_at = SystemTime::now();
    sink.record(RuntimeEvent::TurnEnded {
        turn_id: turn_id.clone(),
        executor: RuntimeExecutor::main_agent(),
        at: ended_at,
        termination: termination.clone(),
        usage,
    });
    let snapshot = TurnDebugSnapshot {
        session_id: session_id.clone(),
        turn_number,
        status: DebugTurnStatus::Failed,
        model_name: config.model_config.model_name.clone(),
        elapsed_ms,
        error_message: Some(error.to_string()),
        system_prompt_path: path_display(&config.system_prompt_path),
        registry_path: path_display(&config.registry_path),
        subagent_registry_path: path_display(&config.subagent_registry_path),
        enabled_servers: enabled_server_names(config.enabled_servers.as_deref()),
        turn: failed_turn_record(
            turn_id,
            sink.turn_started_at().unwrap_or(ended_at),
            ended_at,
            termination,
            usage,
        ),
    };
    sink.emit_turn_snapshot(snapshot);
}

fn final_status_for_termination(termination: &TerminationReason) -> DebugTurnStatus {
    if matches!(termination, TerminationReason::Final) {
        DebugTurnStatus::Completed
    } else {
        DebugTurnStatus::Failed
    }
}

fn failed_turn_record(
    turn_id: TurnId,
    started_at: SystemTime,
    ended_at: SystemTime,
    termination: TerminationReason,
    usage: UsageSummary,
) -> TurnRecord {
    TurnRecord {
        turn_id,
        started_at,
        ended_at,
        steps: Vec::new(),
        messages: Vec::new(),
        final_text: None,
        termination,
        usage,
    }
}

fn failure_termination(error: &AgentRuntimeError) -> TerminationReason {
    match error {
        AgentRuntimeError::Timeout(_) => TerminationReason::Timeout,
        AgentRuntimeError::Validation(_) => TerminationReason::ValidationError,
        _ => TerminationReason::RuntimeError,
    }
}

/// Runner-level failure modes surfaced to the control plane.
#[derive(Debug, Error)]
pub enum TurnRunnerError {
    #[error("runtime turn failed: {0}")]
    Runtime(#[from] AgentRuntimeError),
    #[error("failed to prepare session workspace: {0}")]
    WorkspaceIo(#[from] std::io::Error),
}

fn prepare_session_workspace_at(
    workspace_root: &Path,
    session_id: &SessionId,
) -> Result<PathBuf, std::io::Error> {
    let workspace = session_workspace_path(workspace_root, session_id);
    fs::create_dir_all(workspace.join("scripts"))?;
    fs::create_dir_all(workspace.join("outputs"))?;
    Ok(workspace)
}

fn session_workspace_path(workspace_root: &Path, session_id: &SessionId) -> PathBuf {
    workspace_root.join(session_id.to_string())
}

fn collect_generated_artifacts(
    working_directory: &Path,
    since: SystemTime,
) -> Result<Vec<GeneratedArtifact>, std::io::Error> {
    let outputs_dir = working_directory.join("outputs");
    let mut artifacts = Vec::new();
    for entry in fs::read_dir(outputs_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !entry.file_type()?.is_file() {
            continue;
        }
        let metadata = entry.metadata()?;
        let modified = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);
        if modified < since {
            continue;
        }
        let Some((kind, mime_type)) = generated_artifact_kind(&path) else {
            continue;
        };
        let Some(file_name) = path
            .file_name()
            .and_then(|value| value.to_str())
            .map(ToOwned::to_owned)
        else {
            continue;
        };
        artifacts.push((
            modified,
            GeneratedArtifact {
                kind,
                path,
                file_name,
                mime_type: Some(mime_type.to_owned()),
            },
        ));
    }
    artifacts.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(artifacts
        .into_iter()
        .map(|(_, artifact)| artifact)
        .collect())
}

fn generated_artifact_kind(path: &Path) -> Option<(GeneratedArtifactKind, &'static str)> {
    let extension = path.extension()?.to_str()?.to_ascii_lowercase();
    match extension.as_str() {
        "png" => Some((GeneratedArtifactKind::Image, "image/png")),
        "jpg" | "jpeg" => Some((GeneratedArtifactKind::Image, "image/jpeg")),
        "gif" => Some((GeneratedArtifactKind::Image, "image/gif")),
        "webp" => Some((GeneratedArtifactKind::Image, "image/webp")),
        "pdf" => Some((GeneratedArtifactKind::Document, "application/pdf")),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    use crate::types::SessionId;

    use super::{
        GeneratedArtifactKind, collect_generated_artifacts, prepare_session_workspace_at,
        session_workspace_path,
    };

    #[test]
    fn session_workspace_path_uses_session_id_under_workspace_root() {
        let workspace_root = PathBuf::from("/tmp/arka-workspaces");
        let session_id = SessionId::new();

        assert_eq!(
            session_workspace_path(&workspace_root, &session_id),
            workspace_root.join(session_id.to_string())
        );
    }

    #[test]
    fn prepare_session_workspace_creates_session_scripts_and_outputs_dirs() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        let workspace_root = std::env::temp_dir().join(format!(
            "agent-controlplane-session-workspace-test-{}-{unique}",
            std::process::id()
        ));
        let session_id = SessionId::new();

        let workspace = prepare_session_workspace_at(&workspace_root, &session_id)
            .expect("session workspace should be prepared");

        assert_eq!(workspace, workspace_root.join(session_id.to_string()));
        assert!(workspace.is_dir());
        assert!(workspace.join("scripts").is_dir());
        assert!(workspace.join("outputs").is_dir());

        fs::remove_dir_all(&workspace_root).expect("temp directory should be removable");
    }

    #[test]
    fn prepare_session_workspace_isolates_each_session_directory() {
        let workspace_root = PathBuf::from("/tmp/arka-workspaces");
        let first = SessionId::new();
        let second = SessionId::new();

        assert_ne!(
            session_workspace_path(&workspace_root, &first),
            session_workspace_path(&workspace_root, &second)
        );
    }

    #[test]
    fn collect_generated_artifacts_reads_recent_output_images() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        let workspace_root = std::env::temp_dir().join(format!(
            "agent-controlplane-generated-artifacts-test-{}-{unique}",
            std::process::id()
        ));
        let session_id = SessionId::new();
        let workspace = prepare_session_workspace_at(&workspace_root, &session_id)
            .expect("session workspace should be prepared");
        let image_path = workspace.join("outputs").join("chart.png");
        fs::write(&image_path, b"png-bytes").expect("image output should write");

        let artifacts =
            collect_generated_artifacts(&workspace, SystemTime::UNIX_EPOCH).expect("scan works");

        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0].kind, GeneratedArtifactKind::Image);
        assert_eq!(artifacts[0].file_name, "chart.png");
        assert_eq!(artifacts[0].mime_type.as_deref(), Some("image/png"));

        fs::remove_dir_all(&workspace_root).expect("temp directory should be removable");
    }
}
