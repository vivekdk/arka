//! Runtime harness listener fanout and debug persistence.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    path::Path,
    sync::Arc,
    time::{Duration, SystemTime},
};

use agent_runtime::{
    EventSink, RuntimeDebugSink, RuntimeEvent, RuntimeExecutor, RuntimeRawArtifact,
    RuntimeRawArtifactKind, ServerName, TurnRecord, UsageSummary,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use time::{OffsetDateTime, format_description::well_known::Rfc3339};
use tokio::sync::{broadcast, mpsc};
use tokio_postgres::{Client, NoTls};
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::{SessionEvent, SessionId};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum RuntimeHarnessObservation {
    Event(RuntimeHarnessEventEnvelope),
    RawArtifact(RuntimeHarnessRawArtifactEnvelope),
    TurnSnapshot(TurnDebugSnapshot),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimeHarnessEventEnvelope {
    pub session_id: SessionId,
    pub turn_number: u32,
    pub event_index: u32,
    pub event: RuntimeEvent,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TurnDebugSnapshot {
    pub session_id: SessionId,
    pub turn_number: u32,
    pub status: DebugTurnStatus,
    pub model_name: String,
    pub elapsed_ms: u64,
    pub error_message: Option<String>,
    pub system_prompt_path: String,
    pub registry_path: String,
    pub subagent_registry_path: String,
    pub enabled_servers: Option<Vec<String>>,
    pub turn: TurnRecord,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DebugTurnStatus {
    Running,
    Completed,
    Failed,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TurnDebugMetadata {
    pub model_name: String,
    pub system_prompt_path: String,
    pub registry_path: String,
    pub subagent_registry_path: String,
    pub enabled_servers: Option<Vec<String>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimeHarnessRawArtifactEnvelope {
    pub session_id: SessionId,
    pub turn_number: u32,
    pub artifact_index: u32,
    pub artifact: RuntimeRawArtifact,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DebugHistorySessionSummary {
    pub session_id: String,
    pub latest_turn_number: u32,
    pub turn_id: String,
    pub status: String,
    pub model_name: String,
    pub termination: String,
    pub elapsed_ms: u64,
    pub completed_at: String,
    pub final_text_preview: Option<String>,
    pub error_message: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DebugHistoryTurnSummary {
    pub session_id: String,
    pub turn_number: u32,
    pub turn_id: String,
    pub status: String,
    pub model_name: String,
    pub termination: String,
    pub started_at: String,
    pub ended_at: String,
    pub elapsed_ms: u64,
    pub usage: UsageSummary,
    pub error_message: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DebugHistoryEventRow {
    pub event_index: u32,
    pub event_type: String,
    pub occurred_at: String,
    pub turn_id: Option<String>,
    pub step_id: Option<String>,
    pub step_number: Option<u32>,
    pub executor: Option<RuntimeExecutor>,
    pub payload: Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DebugHistoryRawArtifactRow {
    pub artifact_index: u32,
    pub artifact_kind: String,
    pub source: String,
    pub summary: Option<String>,
    pub occurred_at: String,
    pub turn_id: String,
    pub step_id: Option<String>,
    pub step_number: Option<u32>,
    pub executor: Option<RuntimeExecutor>,
    pub payload: Value,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DebugHistoryFilterOption {
    pub key: String,
    pub label: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DebugHistoryAvailableFilters {
    pub event_types: Vec<String>,
    pub artifact_kinds: Vec<String>,
    pub executors: Vec<DebugHistoryFilterOption>,
    pub steps: Vec<DebugHistoryFilterOption>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DebugHistoryTurnDetail {
    pub session_id: String,
    pub turn_number: u32,
    pub turn_id: String,
    pub status: String,
    pub model_name: String,
    pub termination: String,
    pub started_at: String,
    pub ended_at: String,
    pub elapsed_ms: u64,
    pub usage: UsageSummary,
    pub final_text: Option<String>,
    pub error_message: Option<String>,
    pub system_prompt_path: String,
    pub registry_path: String,
    pub subagent_registry_path: String,
    pub enabled_servers: Option<Vec<String>>,
    pub turn_record: TurnRecord,
    pub available_filters: DebugHistoryAvailableFilters,
    pub events: Vec<DebugHistoryEventRow>,
    pub raw_artifacts: Vec<DebugHistoryRawArtifactRow>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DebugHistoryTurnFilters {
    pub event_types: BTreeSet<String>,
    pub artifact_kinds: BTreeSet<String>,
    pub executors: BTreeSet<String>,
    pub steps: BTreeSet<String>,
}

#[derive(Clone)]
pub struct DebugHistoryStore {
    client: Arc<Client>,
}

#[derive(Debug, Default)]
pub struct ConsoleRuntimeHarnessListener;

pub trait RuntimeHarnessListener: Send + Sync {
    fn try_observe(
        &self,
        observation: RuntimeHarnessObservation,
    ) -> Result<(), RuntimeHarnessListenerError>;
}

#[derive(Debug, Error)]
pub enum RuntimeHarnessListenerError {
    #[error("listener queue is full")]
    QueueFull,
    #[error("listener is closed")]
    Closed,
    #[error("listener setup failed: {0}")]
    Setup(String),
}

#[derive(Clone)]
pub struct RuntimeHarnessFanoutSink {
    session_id: SessionId,
    turn_number: u32,
    event_index: u32,
    artifact_index: u32,
    current_turn_id: Option<agent_runtime::TurnId>,
    turn_started_at: Option<SystemTime>,
    accumulated_usage: UsageSummary,
    metadata: TurnDebugMetadata,
    events: Vec<RuntimeEvent>,
    listeners: Vec<Arc<dyn RuntimeHarnessListener>>,
}

impl RuntimeHarnessFanoutSink {
    pub fn new(
        session_id: SessionId,
        turn_number: u32,
        metadata: TurnDebugMetadata,
        listeners: Vec<Arc<dyn RuntimeHarnessListener>>,
    ) -> Self {
        Self {
            session_id,
            turn_number,
            event_index: 0,
            artifact_index: 0,
            current_turn_id: None,
            turn_started_at: None,
            accumulated_usage: UsageSummary::default(),
            metadata,
            events: Vec::new(),
            listeners,
        }
    }

    pub fn into_events(self) -> Vec<RuntimeEvent> {
        self.events
    }

    pub fn current_turn_id(&self) -> Option<agent_runtime::TurnId> {
        self.current_turn_id.clone()
    }

    pub fn turn_started_at(&self) -> Option<SystemTime> {
        self.turn_started_at
    }

    pub fn accumulated_usage(&self) -> UsageSummary {
        self.accumulated_usage
    }

    pub fn emit_turn_snapshot(&self, snapshot: TurnDebugSnapshot) {
        self.dispatch(RuntimeHarnessObservation::TurnSnapshot(snapshot));
    }

    fn dispatch(&self, observation: RuntimeHarnessObservation) {
        for listener in &self.listeners {
            if let Err(error) = listener.try_observe(observation.clone()) {
                warn!(error = %error, "runtime harness listener dropped observation");
            }
        }
    }
}

impl ConsoleRuntimeHarnessListener {
    pub fn new() -> Arc<dyn RuntimeHarnessListener> {
        Arc::new(Self)
    }
}

impl RuntimeHarnessListener for ConsoleRuntimeHarnessListener {
    fn try_observe(
        &self,
        observation: RuntimeHarnessObservation,
    ) -> Result<(), RuntimeHarnessListenerError> {
        if let RuntimeHarnessObservation::Event(envelope) = observation {
            log_runtime_event_to_console(&envelope);
        }
        Ok(())
    }
}

impl EventSink for RuntimeHarnessFanoutSink {
    fn record(&mut self, event: RuntimeEvent) {
        match &event {
            RuntimeEvent::TurnStarted { turn_id, at, .. } => {
                self.current_turn_id = Some(turn_id.clone());
                self.turn_started_at = Some(*at);
                self.emit_turn_snapshot(TurnDebugSnapshot {
                    session_id: self.session_id.clone(),
                    turn_number: self.turn_number,
                    status: DebugTurnStatus::Running,
                    model_name: self.metadata.model_name.clone(),
                    elapsed_ms: 0,
                    error_message: None,
                    system_prompt_path: self.metadata.system_prompt_path.clone(),
                    registry_path: self.metadata.registry_path.clone(),
                    subagent_registry_path: self.metadata.subagent_registry_path.clone(),
                    enabled_servers: self.metadata.enabled_servers.clone(),
                    turn: provisional_turn_record(turn_id.clone(), *at),
                });
            }
            RuntimeEvent::ModelResponded { usage, .. } => {
                self.accumulated_usage.add_assign(*usage);
            }
            _ => {}
        }
        self.event_index += 1;
        self.events.push(event.clone());
        self.dispatch(RuntimeHarnessObservation::Event(
            RuntimeHarnessEventEnvelope {
                session_id: self.session_id.clone(),
                turn_number: self.turn_number,
                event_index: self.event_index,
                event,
            },
        ));
    }
}

impl RuntimeDebugSink for RuntimeHarnessFanoutSink {
    fn record_raw_artifact(&mut self, artifact: RuntimeRawArtifact) {
        self.artifact_index += 1;
        self.dispatch(RuntimeHarnessObservation::RawArtifact(
            RuntimeHarnessRawArtifactEnvelope {
                session_id: self.session_id.clone(),
                turn_number: self.turn_number,
                artifact_index: self.artifact_index,
                artifact,
            },
        ));
    }
}

#[derive(Debug)]
pub struct SseRuntimeHarnessListener {
    tx: mpsc::Sender<RuntimeHarnessObservation>,
}

impl SseRuntimeHarnessListener {
    pub fn new(
        events_tx: broadcast::Sender<SessionEvent>,
        queue_capacity: usize,
    ) -> Arc<dyn RuntimeHarnessListener> {
        let (tx, mut rx) = mpsc::channel(queue_capacity.max(1));
        tokio::spawn(async move {
            while let Some(observation) = rx.recv().await {
                if let RuntimeHarnessObservation::Event(envelope) = observation {
                    let _ = events_tx.send(SessionEvent::RuntimeEvent {
                        session_id: envelope.session_id,
                        event: envelope.event,
                    });
                }
            }
        });
        Arc::new(Self { tx })
    }
}

impl RuntimeHarnessListener for SseRuntimeHarnessListener {
    fn try_observe(
        &self,
        observation: RuntimeHarnessObservation,
    ) -> Result<(), RuntimeHarnessListenerError> {
        self.tx.try_send(observation).map_err(map_try_send_error)
    }
}

#[derive(Debug)]
pub struct PostgresRuntimeDebugListener {
    tx: mpsc::Sender<RuntimeHarnessObservation>,
}

impl PostgresRuntimeDebugListener {
    pub async fn connect(
        connection_string: &str,
        queue_capacity: usize,
    ) -> Result<Arc<dyn RuntimeHarnessListener>, RuntimeHarnessListenerError> {
        let (client, connection) = tokio_postgres::connect(connection_string, NoTls)
            .await
            .map_err(|error| RuntimeHarnessListenerError::Setup(error.to_string()))?;
        tokio::spawn(async move {
            if let Err(error) = connection.await {
                warn!(error = %error, "runtime debug postgres connection exited");
            }
        });
        setup_schema(&client)
            .await
            .map_err(|error| RuntimeHarnessListenerError::Setup(error.to_string()))?;

        let (tx, mut rx) = mpsc::channel(queue_capacity.max(1));
        tokio::spawn(async move {
            while let Some(observation) = rx.recv().await {
                if let Err(error) = persist_observation(&client, observation).await {
                    warn!(error = %error, "runtime debug postgres persistence failed");
                }
            }
        });

        Ok(Arc::new(Self { tx }))
    }
}

impl DebugHistoryStore {
    pub async fn connect(connection_string: &str) -> Result<Self, RuntimeHarnessListenerError> {
        let (client, connection) = tokio_postgres::connect(connection_string, NoTls)
            .await
            .map_err(|error| RuntimeHarnessListenerError::Setup(error.to_string()))?;
        tokio::spawn(async move {
            if let Err(error) = connection.await {
                warn!(error = %error, "debug history postgres connection exited");
            }
        });
        setup_schema(&client)
            .await
            .map_err(|error| RuntimeHarnessListenerError::Setup(error.to_string()))?;
        Ok(Self {
            client: Arc::new(client),
        })
    }

    pub async fn list_recent_sessions(
        &self,
        limit: usize,
    ) -> Result<Vec<DebugHistorySessionSummary>, RuntimeHarnessListenerError> {
        let rows = self
            .client
            .query(
                r#"
                select *
                from (
                    select distinct on (session_id)
                        session_id,
                        turn_number,
                        turn_id,
                        status,
                        model_name,
                        termination,
                        elapsed_ms,
                        ended_at,
                        final_text,
                        error_message
                    from runtime_debug_turns
                    order by session_id, ended_at desc
                ) latest
                order by ended_at desc
                limit $1
                "#,
                &[&(limit.min(i64::MAX as usize) as i64)],
            )
            .await
            .map_err(|error| RuntimeHarnessListenerError::Setup(error.to_string()))?;
        rows.into_iter().map(map_session_summary_row).collect()
    }

    pub async fn list_session_turns(
        &self,
        session_id: Uuid,
    ) -> Result<Vec<DebugHistoryTurnSummary>, RuntimeHarnessListenerError> {
        let rows = self
            .client
            .query(
                r#"
                    select
                        session_id,
                        turn_number,
                        turn_id,
                        status,
                        model_name,
                        termination,
                        started_at,
                    ended_at,
                    elapsed_ms,
                    usage,
                    error_message
                from runtime_debug_turns
                where session_id = $1
                order by turn_number asc
                "#,
                &[&session_id],
            )
            .await
            .map_err(|error| RuntimeHarnessListenerError::Setup(error.to_string()))?;
        rows.into_iter().map(map_turn_summary_row).collect()
    }

    pub async fn get_turn_detail(
        &self,
        turn_id: Uuid,
        filters: &DebugHistoryTurnFilters,
    ) -> Result<Option<DebugHistoryTurnDetail>, RuntimeHarnessListenerError> {
        let turn_row = self
            .client
            .query_opt(
                r#"
                select
                    session_id,
                    turn_number,
                    turn_id,
                    status,
                    model_name,
                    termination,
                    started_at,
                    ended_at,
                    elapsed_ms,
                    usage,
                    final_text,
                    error_message,
                    system_prompt_path,
                    registry_path,
                    subagent_registry_path,
                    enabled_servers,
                    turn_record
                from runtime_debug_turns
                where turn_id = $1
                "#,
                &[&turn_id],
            )
            .await
            .map_err(|error| RuntimeHarnessListenerError::Setup(error.to_string()))?;
        let Some(turn_row) = turn_row else {
            return Ok(None);
        };
        let event_rows = self
            .client
            .query(
                r#"
                select
                    event_index,
                    event_type,
                    occurred_at,
                    turn_id,
                    step_id,
                    executor,
                    payload
                from runtime_debug_events
                where turn_id = $1
                order by event_index asc
                "#,
                &[&turn_id],
            )
            .await
            .map_err(|error| RuntimeHarnessListenerError::Setup(error.to_string()))?;
        let raw_artifact_rows = self
            .client
            .query(
                r#"
                select
                    artifact_index,
                    artifact_kind,
                    source,
                    summary,
                    occurred_at,
                    turn_id,
                    step_id,
                    executor,
                    payload
                from runtime_debug_raw_artifacts
                where turn_id = $1
                order by artifact_index asc
                "#,
                &[&turn_id],
            )
            .await
            .map_err(|error| RuntimeHarnessListenerError::Setup(error.to_string()))?;

        let usage = parse_usage(turn_row.get("usage"))?;
        let enabled_servers = parse_enabled_servers(turn_row.get("enabled_servers"))?;
        let turn_record = parse_turn_record(turn_row.get("turn_record"))?;
        let mut events = event_rows
            .into_iter()
            .map(map_event_row)
            .collect::<Result<Vec<_>, _>>()?;
        let mut raw_artifacts = raw_artifact_rows
            .into_iter()
            .map(map_raw_artifact_row)
            .collect::<Result<Vec<_>, _>>()?;
        let step_numbers = build_step_number_lookup(&turn_record);
        annotate_event_step_numbers(&mut events, &step_numbers);
        annotate_raw_artifact_step_numbers(&mut raw_artifacts, &step_numbers);
        let available_filters = build_available_filters(&events, &raw_artifacts);
        apply_turn_filters(&mut events, &mut raw_artifacts, filters);

        Ok(Some(DebugHistoryTurnDetail {
            session_id: turn_row.get::<_, Uuid>("session_id").to_string(),
            turn_number: turn_row.get::<_, i32>("turn_number") as u32,
            turn_id: turn_row.get::<_, Uuid>("turn_id").to_string(),
            status: turn_row.get("status"),
            model_name: turn_row.get("model_name"),
            termination: turn_row.get("termination"),
            started_at: format_offset_datetime(turn_row.get("started_at")),
            ended_at: format_offset_datetime(turn_row.get("ended_at")),
            elapsed_ms: turn_row.get::<_, i64>("elapsed_ms") as u64,
            usage,
            final_text: turn_row.get("final_text"),
            error_message: turn_row.get("error_message"),
            system_prompt_path: turn_row.get("system_prompt_path"),
            registry_path: turn_row.get("registry_path"),
            subagent_registry_path: turn_row.get("subagent_registry_path"),
            enabled_servers,
            turn_record,
            available_filters,
            events,
            raw_artifacts,
        }))
    }
}

impl RuntimeHarnessListener for PostgresRuntimeDebugListener {
    fn try_observe(
        &self,
        observation: RuntimeHarnessObservation,
    ) -> Result<(), RuntimeHarnessListenerError> {
        self.tx.try_send(observation).map_err(map_try_send_error)
    }
}

fn map_try_send_error(
    error: mpsc::error::TrySendError<RuntimeHarnessObservation>,
) -> RuntimeHarnessListenerError {
    match error {
        mpsc::error::TrySendError::Full(_) => RuntimeHarnessListenerError::QueueFull,
        mpsc::error::TrySendError::Closed(_) => RuntimeHarnessListenerError::Closed,
    }
}

async fn setup_schema(client: &Client) -> Result<(), tokio_postgres::Error> {
    client
        .batch_execute(
            r#"
            create table if not exists runtime_debug_events (
                id bigserial primary key,
                session_id uuid not null,
                turn_number integer not null,
                turn_id uuid not null,
                step_id uuid null,
                event_index integer not null,
                event_type text not null,
                occurred_at timestamptz not null,
                payload jsonb not null,
                executor jsonb null,
                created_at timestamptz not null default now()
            );
            create index if not exists runtime_debug_events_session_idx
                on runtime_debug_events (session_id, id);
            create index if not exists runtime_debug_events_turn_idx
                on runtime_debug_events (turn_id, event_index);

            create table if not exists runtime_debug_turns (
                id bigserial primary key,
                session_id uuid not null,
                turn_number integer not null,
                turn_id uuid not null unique,
                status text not null default 'completed',
                model_name text not null,
                termination text not null,
                started_at timestamptz not null,
                ended_at timestamptz not null,
                elapsed_ms bigint not null,
                usage jsonb not null,
                final_text text null,
                error_message text null,
                system_prompt_path text not null,
                registry_path text not null,
                subagent_registry_path text not null,
                enabled_servers jsonb null,
                turn_record jsonb not null,
                created_at timestamptz not null default now()
            );
            create index if not exists runtime_debug_turns_session_idx
                on runtime_debug_turns (session_id, turn_number desc);
            create index if not exists runtime_debug_turns_turn_idx
                on runtime_debug_turns (turn_id);

            create table if not exists runtime_debug_raw_artifacts (
                id bigserial primary key,
                session_id uuid not null,
                turn_number integer not null,
                turn_id uuid not null,
                step_id uuid null,
                artifact_index integer not null,
                artifact_kind text not null,
                source text not null,
                summary text null,
                occurred_at timestamptz not null,
                executor jsonb null,
                payload jsonb not null,
                created_at timestamptz not null default now()
            );
            create index if not exists runtime_debug_raw_artifacts_session_idx
                on runtime_debug_raw_artifacts (session_id, created_at);
            create index if not exists runtime_debug_raw_artifacts_turn_idx
                on runtime_debug_raw_artifacts (turn_id, artifact_index);

            alter table runtime_debug_events
                add column if not exists executor jsonb null;
            alter table runtime_debug_raw_artifacts
                add column if not exists executor jsonb null;
            alter table runtime_debug_events
                drop column if exists run_id;
            alter table runtime_debug_turns
                add column if not exists status text not null default 'completed';
            alter table runtime_debug_turns
                add column if not exists error_message text null;
            update runtime_debug_turns
                set status = 'completed'
                where status is null;
            alter table runtime_debug_turns
                drop column if exists run_id;
            alter table runtime_debug_raw_artifacts
                drop column if exists run_id;
            drop index if exists runtime_debug_events_run_idx;
            drop index if exists runtime_debug_turns_run_idx;
            drop index if exists runtime_debug_raw_artifacts_run_idx;
            "#,
        )
        .await
}

async fn persist_observation(
    client: &Client,
    observation: RuntimeHarnessObservation,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    match observation {
        RuntimeHarnessObservation::Event(envelope) => {
            let payload = runtime_event_payload(&envelope.event);
            let turn_id = runtime_event_turn_id(&envelope.event);
            let step_id = runtime_event_step_id(&envelope.event);
            let occurred_at = OffsetDateTime::from(runtime_event_at(&envelope.event));
            let executor = serde_json::to_value(runtime_event_executor(&envelope.event))?;
            client
                .execute(
                    r#"
                    insert into runtime_debug_events (
                        session_id,
                        turn_number,
                        turn_id,
                        step_id,
                        event_index,
                        event_type,
                        occurred_at,
                        executor,
                        payload
                    ) values (
                        $1::uuid,
                        $2,
                        $3::uuid,
                        $4::uuid,
                        $5,
                        $6,
                        $7,
                        $8::jsonb,
                        $9::jsonb
                    )
                    "#,
                    &[
                        &envelope.session_id.as_uuid(),
                        &(envelope.turn_number as i32),
                        &turn_id,
                        &step_id,
                        &(envelope.event_index as i32),
                        &runtime_event_type(&envelope.event),
                        &occurred_at,
                        &executor,
                        &payload,
                    ],
                )
                .await?;
        }
        RuntimeHarnessObservation::RawArtifact(envelope) => {
            let executor = serde_json::to_value(&envelope.artifact.executor)?;
            client
                .execute(
                    r#"
                    insert into runtime_debug_raw_artifacts (
                        session_id,
                        turn_number,
                        turn_id,
                        step_id,
                        artifact_index,
                        artifact_kind,
                        source,
                        summary,
                        occurred_at,
                        executor,
                        payload
                    ) values (
                        $1::uuid,
                        $2,
                        $3::uuid,
                        $4::uuid,
                        $5,
                        $6,
                        $7,
                        $8,
                        $9,
                        $10::jsonb,
                        $11::jsonb
                    )
                    "#,
                    &[
                        &envelope.session_id.as_uuid(),
                        &(envelope.turn_number as i32),
                        &envelope.artifact.turn_id.as_uuid(),
                        &envelope
                            .artifact
                            .step_id
                            .as_ref()
                            .map(|value| value.as_uuid()),
                        &(envelope.artifact_index as i32),
                        &raw_artifact_kind(&envelope.artifact.kind),
                        &envelope.artifact.source,
                        &envelope.artifact.summary,
                        &OffsetDateTime::from(envelope.artifact.occurred_at),
                        &executor,
                        &envelope.artifact.payload,
                    ],
                )
                .await?;
        }
        RuntimeHarnessObservation::TurnSnapshot(snapshot) => {
            let enabled_servers = snapshot
                .enabled_servers
                .as_ref()
                .map(serde_json::to_value)
                .transpose()?;
            let usage = serde_json::to_value(snapshot.turn.usage)?;
            let turn_record = serde_json::to_value(&snapshot.turn)?;
            client
                .execute(
                    r#"
                    insert into runtime_debug_turns (
                        session_id,
                        turn_number,
                        turn_id,
                        status,
                        model_name,
                        termination,
                        started_at,
                        ended_at,
                        elapsed_ms,
                        usage,
                        final_text,
                        error_message,
                        system_prompt_path,
                        registry_path,
                        subagent_registry_path,
                        enabled_servers,
                        turn_record
                    ) values (
                        $1::uuid,
                        $2,
                        $3::uuid,
                        $4,
                        $5,
                        $6,
                        $7,
                        $8,
                        $9,
                        $10::jsonb,
                        $11,
                        $12,
                        $13,
                        $14,
                        $15,
                        $16::jsonb,
                        $17::jsonb
                    )
                    on conflict (turn_id) do update set
                        turn_number = excluded.turn_number,
                        status = excluded.status,
                        model_name = excluded.model_name,
                        termination = excluded.termination,
                        started_at = excluded.started_at,
                        ended_at = excluded.ended_at,
                        elapsed_ms = excluded.elapsed_ms,
                        usage = excluded.usage,
                        final_text = excluded.final_text,
                        error_message = excluded.error_message,
                        system_prompt_path = excluded.system_prompt_path,
                        registry_path = excluded.registry_path,
                        subagent_registry_path = excluded.subagent_registry_path,
                        enabled_servers = excluded.enabled_servers,
                        turn_record = excluded.turn_record
                    "#,
                    &[
                        &snapshot.session_id.as_uuid(),
                        &(snapshot.turn_number as i32),
                        &snapshot.turn.turn_id.as_uuid(),
                        &debug_turn_status_label(snapshot.status),
                        &snapshot.model_name,
                        &format!("{:?}", snapshot.turn.termination),
                        &OffsetDateTime::from(snapshot.turn.started_at),
                        &OffsetDateTime::from(snapshot.turn.ended_at),
                        &(snapshot.elapsed_ms as i64),
                        &usage,
                        &snapshot.turn.final_text,
                        &snapshot.error_message,
                        &snapshot.system_prompt_path,
                        &snapshot.registry_path,
                        &snapshot.subagent_registry_path,
                        &enabled_servers,
                        &turn_record,
                    ],
                )
                .await?;
        }
    }
    Ok(())
}

fn runtime_event_type(event: &RuntimeEvent) -> &'static str {
    match event {
        RuntimeEvent::TurnStarted { .. } => "turn_started",
        RuntimeEvent::StepStarted { .. } => "step_started",
        RuntimeEvent::PromptBuilt { .. } => "prompt_built",
        RuntimeEvent::HandoffToSubagent { .. } => "handoff_to_subagent",
        RuntimeEvent::ModelCalled { .. } => "model_called",
        RuntimeEvent::ModelResponded { .. } => "model_responded",
        RuntimeEvent::AnswerRenderStarted { .. } => "answer_render_started",
        RuntimeEvent::AnswerTextDelta { .. } => "answer_text_delta",
        RuntimeEvent::AnswerRenderCompleted { .. } => "answer_render_completed",
        RuntimeEvent::AnswerRenderFailed { .. } => "answer_render_failed",
        RuntimeEvent::HandoffToMainAgent { .. } => "handoff_to_main_agent",
        RuntimeEvent::ToolMaskEvaluated { .. } => "tool_mask_evaluated",
        RuntimeEvent::McpCalled { .. } => "mcp_called",
        RuntimeEvent::McpResponded { .. } => "mcp_responded",
        RuntimeEvent::LocalToolCalled { .. } => "local_tool_called",
        RuntimeEvent::LocalToolResponded { .. } => "local_tool_responded",
        RuntimeEvent::StepEnded { .. } => "step_ended",
        RuntimeEvent::TurnEnded { .. } => "turn_ended",
    }
}

fn log_runtime_event_to_console(envelope: &RuntimeHarnessEventEnvelope) {
    let event_type = runtime_event_type(&envelope.event);
    match &envelope.event {
        RuntimeEvent::TurnStarted { executor, .. } => {
            info!(
                session_id = %envelope.session_id,
                turn_number = envelope.turn_number,
                event = event_type,
                executor = %executor_filter_label(executor),
                "turn started"
            );
        }
        RuntimeEvent::StepStarted {
            step_id,
            step_number,
            executor,
            ..
        } => {
            info!(
                session_id = %envelope.session_id,
                turn_number = envelope.turn_number,
                step_id = %step_id,
                step_number,
                event = event_type,
                executor = %executor_filter_label(executor),
                "step started"
            );
        }
        RuntimeEvent::HandoffToSubagent {
            step_id,
            executor,
            subagent_type,
            goal,
            target,
            ..
        } => {
            info!(
                session_id = %envelope.session_id,
                turn_number = envelope.turn_number,
                step_id = %step_id,
                event = event_type,
                executor = %executor_filter_label(executor),
                subagent_type = %subagent_type,
                target = %target_summary(target),
                goal = %truncate_for_log(goal, 160),
                "main agent handed off to subagent"
            );
        }
        RuntimeEvent::ModelCalled {
            step_id, executor, ..
        } => {
            info!(
                session_id = %envelope.session_id,
                turn_number = envelope.turn_number,
                step_id = %step_id,
                event = event_type,
                executor = %executor_filter_label(executor),
                "calling model"
            );
        }
        RuntimeEvent::ModelResponded {
            step_id,
            latency,
            usage,
            executor,
            ..
        } => {
            info!(
                session_id = %envelope.session_id,
                turn_number = envelope.turn_number,
                step_id = %step_id,
                event = event_type,
                executor = %executor_filter_label(executor),
                latency_ms = duration_ms(*latency),
                input_tokens = usage.input_tokens,
                cached_tokens = usage.cached_tokens,
                output_tokens = usage.output_tokens,
                total_tokens = usage.total_tokens,
                "model responded"
            );
        }
        RuntimeEvent::HandoffToMainAgent {
            step_id,
            executor,
            subagent_type,
            status,
            ..
        } => {
            info!(
                session_id = %envelope.session_id,
                turn_number = envelope.turn_number,
                step_id = %step_id,
                event = event_type,
                executor = %executor_filter_label(executor),
                subagent_type = %subagent_type,
                status = %status,
                "subagent returned control to main agent"
            );
        }
        RuntimeEvent::McpCalled {
            step_id,
            server_name,
            tool_name,
            executor,
            ..
        } => {
            info!(
                session_id = %envelope.session_id,
                turn_number = envelope.turn_number,
                step_id = %step_id,
                event = event_type,
                executor = %executor_filter_label(executor),
                server_name = %server_name,
                tool_name = %tool_name,
                "calling MCP tool"
            );
        }
        RuntimeEvent::McpResponded {
            step_id,
            server_name,
            tool_name,
            executor,
            latency,
            was_error,
            result_summary,
            error: mcp_error,
            ..
        } => {
            let log_message = if *was_error {
                "MCP tool returned error"
            } else {
                "MCP tool responded"
            };
            if *was_error {
                warn!(
                    session_id = %envelope.session_id,
                    turn_number = envelope.turn_number,
                    step_id = %step_id,
                    event = event_type,
                    executor = %executor_filter_label(executor),
                    server_name = %server_name,
                    tool_name = %tool_name,
                    latency_ms = duration_ms(*latency),
                    result_summary = %truncate_for_log(result_summary, 160),
                    error = ?mcp_error,
                    "{log_message}"
                );
            } else {
                info!(
                    session_id = %envelope.session_id,
                    turn_number = envelope.turn_number,
                    step_id = %step_id,
                    event = event_type,
                    executor = %executor_filter_label(executor),
                    server_name = %server_name,
                    tool_name = %tool_name,
                    latency_ms = duration_ms(*latency),
                    result_summary = %truncate_for_log(result_summary, 160),
                    "{log_message}"
                );
            }
        }
        RuntimeEvent::LocalToolCalled {
            step_id,
            tool_name,
            executor,
            ..
        } => {
            info!(
                session_id = %envelope.session_id,
                turn_number = envelope.turn_number,
                step_id = %step_id,
                event = event_type,
                executor = %executor_filter_label(executor),
                tool_name = %tool_name,
                "calling local tool"
            );
        }
        RuntimeEvent::LocalToolResponded {
            step_id,
            tool_name,
            executor,
            latency,
            was_error,
            result_summary,
            error: tool_error,
            ..
        } => {
            let log_message = if *was_error {
                "local tool returned error"
            } else {
                "local tool responded"
            };
            if *was_error {
                warn!(
                    session_id = %envelope.session_id,
                    turn_number = envelope.turn_number,
                    step_id = %step_id,
                    event = event_type,
                    executor = %executor_filter_label(executor),
                    tool_name = %tool_name,
                    latency_ms = duration_ms(*latency),
                    result_summary = %truncate_for_log(result_summary, 160),
                    error = ?tool_error,
                    "{log_message}"
                );
            } else {
                info!(
                    session_id = %envelope.session_id,
                    turn_number = envelope.turn_number,
                    step_id = %step_id,
                    event = event_type,
                    executor = %executor_filter_label(executor),
                    tool_name = %tool_name,
                    latency_ms = duration_ms(*latency),
                    result_summary = %truncate_for_log(result_summary, 160),
                    "{log_message}"
                );
            }
        }
        RuntimeEvent::AnswerRenderFailed {
            step_id,
            executor,
            error: render_error,
            ..
        } => {
            error!(
                session_id = %envelope.session_id,
                turn_number = envelope.turn_number,
                step_id = %step_id,
                event = event_type,
                executor = %executor_filter_label(executor),
                error = %render_error,
                "answer render failed"
            );
        }
        RuntimeEvent::TurnEnded {
            executor,
            termination,
            usage,
            ..
        } => {
            info!(
                session_id = %envelope.session_id,
                turn_number = envelope.turn_number,
                event = event_type,
                executor = %executor_filter_label(executor),
                termination = ?termination,
                input_tokens = usage.input_tokens,
                cached_tokens = usage.cached_tokens,
                output_tokens = usage.output_tokens,
                total_tokens = usage.total_tokens,
                "turn ended"
            );
        }
        RuntimeEvent::PromptBuilt { .. }
        | RuntimeEvent::AnswerRenderStarted { .. }
        | RuntimeEvent::AnswerTextDelta { .. }
        | RuntimeEvent::AnswerRenderCompleted { .. }
        | RuntimeEvent::ToolMaskEvaluated { .. }
        | RuntimeEvent::StepEnded { .. } => {}
    }
}

fn duration_ms(duration: Duration) -> u128 {
    duration.as_millis()
}

fn truncate_for_log(text: &str, max_chars: usize) -> String {
    let mut normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.chars().count() > max_chars {
        normalized = normalized.chars().take(max_chars).collect::<String>() + "...";
    }
    normalized
}

fn target_summary(target: &agent_runtime::state::DelegationTarget) -> String {
    truncate_for_log(&target.summary(), 120)
}

fn debug_turn_status_label(status: DebugTurnStatus) -> &'static str {
    match status {
        DebugTurnStatus::Running => "running",
        DebugTurnStatus::Completed => "completed",
        DebugTurnStatus::Failed => "failed",
    }
}

fn provisional_turn_record(turn_id: agent_runtime::TurnId, started_at: SystemTime) -> TurnRecord {
    TurnRecord {
        turn_id,
        started_at,
        ended_at: started_at,
        steps: Vec::new(),
        messages: Vec::new(),
        final_text: None,
        termination: agent_runtime::TerminationReason::RuntimeError,
        usage: UsageSummary::default(),
    }
}

fn runtime_event_payload(event: &RuntimeEvent) -> Value {
    match event {
        RuntimeEvent::TurnStarted { executor, .. } => serde_json::json!({
            "executor": executor,
        }),
        RuntimeEvent::StepStarted {
            step_number,
            executor,
            ..
        } => serde_json::json!({
            "step_number": step_number,
            "executor": executor,
        }),
        RuntimeEvent::PromptBuilt { executor, .. } => serde_json::json!({
            "executor": executor,
        }),
        RuntimeEvent::HandoffToSubagent {
            executor,
            subagent_type,
            goal,
            target,
            ..
        } => serde_json::json!({
            "executor": executor,
            "subagent_type": subagent_type,
            "goal": goal,
            "target": target,
        }),
        RuntimeEvent::ModelCalled { executor, .. } => serde_json::json!({
            "executor": executor,
        }),
        RuntimeEvent::ModelResponded {
            latency,
            usage,
            executor,
            ..
        } => serde_json::json!({
            "latency": latency,
            "usage": usage,
            "executor": executor,
        }),
        RuntimeEvent::AnswerRenderStarted { executor, .. } => serde_json::json!({
            "executor": executor,
        }),
        RuntimeEvent::AnswerTextDelta {
            delta, executor, ..
        } => serde_json::json!({
            "delta": delta,
            "executor": executor,
        }),
        RuntimeEvent::AnswerRenderCompleted { executor, .. } => serde_json::json!({
            "executor": executor,
        }),
        RuntimeEvent::AnswerRenderFailed {
            error, executor, ..
        } => serde_json::json!({
            "error": error,
            "executor": executor,
        }),
        RuntimeEvent::HandoffToMainAgent {
            executor,
            subagent_type,
            status,
            ..
        } => serde_json::json!({
            "executor": executor,
            "subagent_type": subagent_type,
            "status": status,
        }),
        RuntimeEvent::ToolMaskEvaluated {
            executor,
            enforcement_mode,
            allowed_tool_ids,
            denied_tool_ids,
            decisions,
            ..
        } => serde_json::json!({
            "executor": executor,
            "enforcement_mode": enforcement_mode,
            "allowed_tool_ids": allowed_tool_ids,
            "denied_tool_ids": denied_tool_ids,
            "decisions": decisions,
        }),
        RuntimeEvent::McpCalled {
            server_name,
            tool_name,
            executor,
            ..
        } => serde_json::json!({
            "server_name": server_name,
            "tool_name": tool_name,
            "executor": executor,
        }),
        RuntimeEvent::McpResponded {
            server_name,
            tool_name,
            latency,
            was_error,
            result_summary,
            error,
            response_payload,
            executor,
            ..
        } => serde_json::json!({
            "server_name": server_name,
            "tool_name": tool_name,
            "latency": latency,
            "was_error": was_error,
            "result_summary": result_summary,
            "error": error,
            "response_payload": response_payload,
            "executor": executor,
        }),
        RuntimeEvent::LocalToolCalled {
            tool_name,
            executor,
            ..
        } => serde_json::json!({
            "tool_name": tool_name,
            "executor": executor,
        }),
        RuntimeEvent::LocalToolResponded {
            tool_name,
            latency,
            was_error,
            result_summary,
            error,
            response_payload,
            executor,
            ..
        } => serde_json::json!({
            "tool_name": tool_name,
            "latency": latency,
            "was_error": was_error,
            "result_summary": result_summary,
            "error": error,
            "response_payload": response_payload,
            "executor": executor,
        }),
        RuntimeEvent::StepEnded { executor, .. } => serde_json::json!({
            "executor": executor,
        }),
        RuntimeEvent::TurnEnded {
            termination,
            usage,
            executor,
            ..
        } => serde_json::json!({
            "termination": termination,
            "usage": usage,
            "executor": executor,
        }),
    }
}

fn runtime_event_executor(event: &RuntimeEvent) -> &RuntimeExecutor {
    match event {
        RuntimeEvent::TurnStarted { executor, .. }
        | RuntimeEvent::StepStarted { executor, .. }
        | RuntimeEvent::PromptBuilt { executor, .. }
        | RuntimeEvent::HandoffToSubagent { executor, .. }
        | RuntimeEvent::ModelCalled { executor, .. }
        | RuntimeEvent::ModelResponded { executor, .. }
        | RuntimeEvent::AnswerRenderStarted { executor, .. }
        | RuntimeEvent::AnswerTextDelta { executor, .. }
        | RuntimeEvent::AnswerRenderCompleted { executor, .. }
        | RuntimeEvent::AnswerRenderFailed { executor, .. }
        | RuntimeEvent::HandoffToMainAgent { executor, .. }
        | RuntimeEvent::ToolMaskEvaluated { executor, .. }
        | RuntimeEvent::McpCalled { executor, .. }
        | RuntimeEvent::McpResponded { executor, .. }
        | RuntimeEvent::LocalToolCalled { executor, .. }
        | RuntimeEvent::LocalToolResponded { executor, .. }
        | RuntimeEvent::StepEnded { executor, .. }
        | RuntimeEvent::TurnEnded { executor, .. } => executor,
    }
}

fn runtime_event_turn_id(event: &RuntimeEvent) -> Uuid {
    match event {
        RuntimeEvent::TurnStarted { turn_id, .. }
        | RuntimeEvent::StepStarted { turn_id, .. }
        | RuntimeEvent::PromptBuilt { turn_id, .. }
        | RuntimeEvent::HandoffToSubagent { turn_id, .. }
        | RuntimeEvent::ModelCalled { turn_id, .. }
        | RuntimeEvent::ModelResponded { turn_id, .. }
        | RuntimeEvent::AnswerRenderStarted { turn_id, .. }
        | RuntimeEvent::AnswerTextDelta { turn_id, .. }
        | RuntimeEvent::AnswerRenderCompleted { turn_id, .. }
        | RuntimeEvent::AnswerRenderFailed { turn_id, .. }
        | RuntimeEvent::HandoffToMainAgent { turn_id, .. }
        | RuntimeEvent::ToolMaskEvaluated { turn_id, .. }
        | RuntimeEvent::McpCalled { turn_id, .. }
        | RuntimeEvent::McpResponded { turn_id, .. }
        | RuntimeEvent::LocalToolCalled { turn_id, .. }
        | RuntimeEvent::LocalToolResponded { turn_id, .. }
        | RuntimeEvent::StepEnded { turn_id, .. }
        | RuntimeEvent::TurnEnded { turn_id, .. } => turn_id.as_uuid(),
    }
}

fn runtime_event_step_id(event: &RuntimeEvent) -> Option<Uuid> {
    match event {
        RuntimeEvent::StepStarted { step_id, .. }
        | RuntimeEvent::PromptBuilt { step_id, .. }
        | RuntimeEvent::HandoffToSubagent { step_id, .. }
        | RuntimeEvent::ModelCalled { step_id, .. }
        | RuntimeEvent::ModelResponded { step_id, .. }
        | RuntimeEvent::AnswerRenderStarted { step_id, .. }
        | RuntimeEvent::AnswerTextDelta { step_id, .. }
        | RuntimeEvent::AnswerRenderCompleted { step_id, .. }
        | RuntimeEvent::AnswerRenderFailed { step_id, .. }
        | RuntimeEvent::HandoffToMainAgent { step_id, .. }
        | RuntimeEvent::ToolMaskEvaluated { step_id, .. }
        | RuntimeEvent::McpCalled { step_id, .. }
        | RuntimeEvent::McpResponded { step_id, .. }
        | RuntimeEvent::LocalToolCalled { step_id, .. }
        | RuntimeEvent::LocalToolResponded { step_id, .. }
        | RuntimeEvent::StepEnded { step_id, .. } => Some(step_id.as_uuid()),
        RuntimeEvent::TurnStarted { .. } | RuntimeEvent::TurnEnded { .. } => None,
    }
}

fn runtime_event_at(event: &RuntimeEvent) -> SystemTime {
    match event {
        RuntimeEvent::TurnStarted { at, .. }
        | RuntimeEvent::StepStarted { at, .. }
        | RuntimeEvent::PromptBuilt { at, .. }
        | RuntimeEvent::HandoffToSubagent { at, .. }
        | RuntimeEvent::ModelCalled { at, .. }
        | RuntimeEvent::ModelResponded { at, .. }
        | RuntimeEvent::AnswerRenderStarted { at, .. }
        | RuntimeEvent::AnswerTextDelta { at, .. }
        | RuntimeEvent::AnswerRenderCompleted { at, .. }
        | RuntimeEvent::AnswerRenderFailed { at, .. }
        | RuntimeEvent::HandoffToMainAgent { at, .. }
        | RuntimeEvent::ToolMaskEvaluated { at, .. }
        | RuntimeEvent::McpCalled { at, .. }
        | RuntimeEvent::McpResponded { at, .. }
        | RuntimeEvent::LocalToolCalled { at, .. }
        | RuntimeEvent::LocalToolResponded { at, .. }
        | RuntimeEvent::StepEnded { at, .. }
        | RuntimeEvent::TurnEnded { at, .. } => *at,
    }
}

fn raw_artifact_kind(kind: &RuntimeRawArtifactKind) -> &'static str {
    match kind {
        RuntimeRawArtifactKind::ModelRequest => "model_request",
        RuntimeRawArtifactKind::ModelResponse => "model_response",
        RuntimeRawArtifactKind::ModelError => "model_error",
        RuntimeRawArtifactKind::McpRequest => "mcp_request",
        RuntimeRawArtifactKind::McpResponse => "mcp_response",
        RuntimeRawArtifactKind::McpError => "mcp_error",
        RuntimeRawArtifactKind::LocalToolRequest => "local_tool_request",
        RuntimeRawArtifactKind::LocalToolResponse => "local_tool_response",
        RuntimeRawArtifactKind::LocalToolError => "local_tool_error",
        RuntimeRawArtifactKind::PolicyDecision => "policy_decision",
    }
}

pub fn enabled_server_names(enabled_servers: Option<&[ServerName]>) -> Option<Vec<String>> {
    enabled_servers.map(|servers| {
        servers
            .iter()
            .map(ServerName::to_string)
            .collect::<Vec<_>>()
    })
}

pub fn path_display(path: &Path) -> String {
    path.display().to_string()
}

fn map_session_summary_row(
    row: tokio_postgres::Row,
) -> Result<DebugHistorySessionSummary, RuntimeHarnessListenerError> {
    let final_text: Option<String> = row.get("final_text");
    Ok(DebugHistorySessionSummary {
        session_id: row.get::<_, Uuid>("session_id").to_string(),
        latest_turn_number: row.get::<_, i32>("turn_number") as u32,
        turn_id: row.get::<_, Uuid>("turn_id").to_string(),
        status: row.get("status"),
        model_name: row.get("model_name"),
        termination: row.get("termination"),
        elapsed_ms: row.get::<_, i64>("elapsed_ms") as u64,
        completed_at: format_offset_datetime(row.get("ended_at")),
        final_text_preview: final_text,
        error_message: row.get("error_message"),
    })
}

fn map_turn_summary_row(
    row: tokio_postgres::Row,
) -> Result<DebugHistoryTurnSummary, RuntimeHarnessListenerError> {
    Ok(DebugHistoryTurnSummary {
        session_id: row.get::<_, Uuid>("session_id").to_string(),
        turn_number: row.get::<_, i32>("turn_number") as u32,
        turn_id: row.get::<_, Uuid>("turn_id").to_string(),
        status: row.get("status"),
        model_name: row.get("model_name"),
        termination: row.get("termination"),
        started_at: format_offset_datetime(row.get("started_at")),
        ended_at: format_offset_datetime(row.get("ended_at")),
        elapsed_ms: row.get::<_, i64>("elapsed_ms") as u64,
        usage: parse_usage(row.get("usage"))?,
        error_message: row.get("error_message"),
    })
}

fn map_event_row(
    row: tokio_postgres::Row,
) -> Result<DebugHistoryEventRow, RuntimeHarnessListenerError> {
    Ok(DebugHistoryEventRow {
        event_index: row.get::<_, i32>("event_index") as u32,
        event_type: row.get("event_type"),
        occurred_at: format_offset_datetime(row.get("occurred_at")),
        turn_id: row
            .get::<_, Option<Uuid>>("turn_id")
            .map(|value| value.to_string()),
        step_id: row
            .get::<_, Option<Uuid>>("step_id")
            .map(|value| value.to_string()),
        step_number: None,
        executor: parse_runtime_executor(row.get("executor"))?,
        payload: row.get("payload"),
    })
}

fn map_raw_artifact_row(
    row: tokio_postgres::Row,
) -> Result<DebugHistoryRawArtifactRow, RuntimeHarnessListenerError> {
    Ok(DebugHistoryRawArtifactRow {
        artifact_index: row.get::<_, i32>("artifact_index") as u32,
        artifact_kind: row.get("artifact_kind"),
        source: row.get("source"),
        summary: row.get("summary"),
        occurred_at: format_offset_datetime(row.get("occurred_at")),
        turn_id: row.get::<_, Uuid>("turn_id").to_string(),
        step_id: row
            .get::<_, Option<Uuid>>("step_id")
            .map(|value| value.to_string()),
        step_number: None,
        executor: parse_runtime_executor(row.get("executor"))?,
        payload: row.get("payload"),
    })
}

fn build_step_number_lookup(turn_record: &TurnRecord) -> HashMap<String, u32> {
    turn_record
        .steps
        .iter()
        .map(|step| (step.step_id.to_string(), step.step_number))
        .collect()
}

fn annotate_event_step_numbers(
    events: &mut [DebugHistoryEventRow],
    step_numbers: &HashMap<String, u32>,
) {
    for event in events {
        event.step_number = event
            .step_id
            .as_ref()
            .and_then(|step_id| step_numbers.get(step_id))
            .copied();
    }
}

fn annotate_raw_artifact_step_numbers(
    raw_artifacts: &mut [DebugHistoryRawArtifactRow],
    step_numbers: &HashMap<String, u32>,
) {
    for artifact in raw_artifacts {
        artifact.step_number = artifact
            .step_id
            .as_ref()
            .and_then(|step_id| step_numbers.get(step_id))
            .copied();
    }
}

fn build_available_filters(
    events: &[DebugHistoryEventRow],
    raw_artifacts: &[DebugHistoryRawArtifactRow],
) -> DebugHistoryAvailableFilters {
    let event_types = events
        .iter()
        .map(|event| event.event_type.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let artifact_kinds = raw_artifacts
        .iter()
        .map(|artifact| artifact.artifact_kind.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    let mut executors = BTreeMap::new();
    for executor in events.iter().filter_map(|event| event.executor.as_ref()) {
        executors.insert(
            executor_filter_key(executor),
            executor_filter_label(executor),
        );
    }
    for executor in raw_artifacts
        .iter()
        .filter_map(|artifact| artifact.executor.as_ref())
    {
        executors.insert(
            executor_filter_key(executor),
            executor_filter_label(executor),
        );
    }
    let mut executor_options = executors
        .into_iter()
        .map(|(key, label)| DebugHistoryFilterOption { key, label })
        .collect::<Vec<_>>();
    executor_options.sort_by(|left, right| {
        executor_filter_sort_rank(&left.key)
            .cmp(&executor_filter_sort_rank(&right.key))
            .then_with(|| left.label.cmp(&right.label))
    });

    let mut step_numbers = BTreeSet::new();
    let mut has_no_step = false;
    for event in events {
        if let Some(step_number) = event.step_number {
            step_numbers.insert(step_number);
        } else {
            has_no_step = true;
        }
    }
    for artifact in raw_artifacts {
        if let Some(step_number) = artifact.step_number {
            step_numbers.insert(step_number);
        } else {
            has_no_step = true;
        }
    }
    let mut steps = step_numbers
        .into_iter()
        .map(|step_number| DebugHistoryFilterOption {
            key: step_number.to_string(),
            label: format!("Step {step_number}"),
        })
        .collect::<Vec<_>>();
    if has_no_step {
        steps.push(DebugHistoryFilterOption {
            key: "none".to_owned(),
            label: "No step".to_owned(),
        });
    }

    DebugHistoryAvailableFilters {
        event_types,
        artifact_kinds,
        executors: executor_options,
        steps,
    }
}

fn apply_turn_filters(
    events: &mut Vec<DebugHistoryEventRow>,
    raw_artifacts: &mut Vec<DebugHistoryRawArtifactRow>,
    filters: &DebugHistoryTurnFilters,
) {
    let type_filters_active = !filters.event_types.is_empty() || !filters.artifact_kinds.is_empty();
    events.retain(|event| {
        matches_event_type_filter(event, filters, type_filters_active)
            && matches_executor_filter(event.executor.as_ref(), &filters.executors)
            && matches_step_filter(event.step_number, &filters.steps)
    });
    raw_artifacts.retain(|artifact| {
        matches_artifact_type_filter(artifact, filters, type_filters_active)
            && matches_executor_filter(artifact.executor.as_ref(), &filters.executors)
            && matches_step_filter(artifact.step_number, &filters.steps)
    });
}

fn matches_event_type_filter(
    event: &DebugHistoryEventRow,
    filters: &DebugHistoryTurnFilters,
    type_filters_active: bool,
) -> bool {
    if !type_filters_active {
        true
    } else {
        filters.event_types.contains(&event.event_type)
    }
}

fn matches_artifact_type_filter(
    artifact: &DebugHistoryRawArtifactRow,
    filters: &DebugHistoryTurnFilters,
    type_filters_active: bool,
) -> bool {
    if !type_filters_active {
        true
    } else {
        filters.artifact_kinds.contains(&artifact.artifact_kind)
    }
}

fn matches_executor_filter(
    executor: Option<&RuntimeExecutor>,
    selected_executors: &BTreeSet<String>,
) -> bool {
    if selected_executors.is_empty() {
        return true;
    }
    executor
        .map(executor_filter_key)
        .is_some_and(|key| selected_executors.contains(&key))
}

fn matches_step_filter(step_number: Option<u32>, selected_steps: &BTreeSet<String>) -> bool {
    if selected_steps.is_empty() {
        return true;
    }
    match step_number {
        Some(step_number) => selected_steps.contains(&step_number.to_string()),
        None => selected_steps.contains("none"),
    }
}

fn executor_filter_key(executor: &RuntimeExecutor) -> String {
    match executor.kind {
        agent_runtime::RuntimeExecutorKind::MainAgent => "main".to_owned(),
        agent_runtime::RuntimeExecutorKind::Subagent => format!(
            "subagent:{}",
            executor
                .subagent_type
                .clone()
                .unwrap_or_else(|| executor.display_name.clone())
        ),
    }
}

fn executor_filter_label(executor: &RuntimeExecutor) -> String {
    match (&executor.kind, executor.subagent_type.as_deref()) {
        (agent_runtime::RuntimeExecutorKind::MainAgent, _) => executor.display_name.clone(),
        (agent_runtime::RuntimeExecutorKind::Subagent, Some(subagent_type)) => {
            format!("{} ({subagent_type})", executor.display_name)
        }
        (agent_runtime::RuntimeExecutorKind::Subagent, None) => executor.display_name.clone(),
    }
}

fn executor_filter_sort_rank(key: &str) -> u8 {
    if key == "main" { 0 } else { 1 }
}

fn parse_usage(value: Value) -> Result<UsageSummary, RuntimeHarnessListenerError> {
    serde_json::from_value(value)
        .map_err(|error| RuntimeHarnessListenerError::Setup(error.to_string()))
}

fn parse_turn_record(value: Value) -> Result<TurnRecord, RuntimeHarnessListenerError> {
    serde_json::from_value(value)
        .map_err(|error| RuntimeHarnessListenerError::Setup(error.to_string()))
}

fn parse_enabled_servers(
    value: Option<Value>,
) -> Result<Option<Vec<String>>, RuntimeHarnessListenerError> {
    value
        .map(|json| {
            serde_json::from_value(json)
                .map_err(|error| RuntimeHarnessListenerError::Setup(error.to_string()))
        })
        .transpose()
}

fn parse_runtime_executor(
    value: Option<Value>,
) -> Result<Option<RuntimeExecutor>, RuntimeHarnessListenerError> {
    value
        .map(|json| {
            serde_json::from_value(json)
                .map_err(|error| RuntimeHarnessListenerError::Setup(error.to_string()))
        })
        .transpose()
}

fn format_offset_datetime(value: OffsetDateTime) -> String {
    value
        .format(&Rfc3339)
        .unwrap_or_else(|_| value.unix_timestamp().to_string())
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, Mutex},
        time::SystemTime,
    };

    use agent_runtime::{
        PromptSnapshot, RuntimeExecutor, StepId, StepOutcomeKind, StepRecord, TerminationReason,
        TurnId, TurnRecord, UsageSummary,
    };

    use super::*;

    #[derive(Default)]
    struct RecordingListener {
        observations: Mutex<Vec<RuntimeHarnessObservation>>,
    }

    impl RuntimeHarnessListener for RecordingListener {
        fn try_observe(
            &self,
            observation: RuntimeHarnessObservation,
        ) -> Result<(), RuntimeHarnessListenerError> {
            self.observations
                .lock()
                .expect("observations should lock")
                .push(observation);
            Ok(())
        }
    }

    #[test]
    fn fanout_sink_records_events_and_assigns_event_index() {
        let listener = Arc::new(RecordingListener::default());
        let mut sink = RuntimeHarnessFanoutSink::new(
            SessionId::new(),
            3,
            TurnDebugMetadata {
                model_name: "gpt-5.4".to_owned(),
                system_prompt_path: "config/prompt.md".to_owned(),
                registry_path: "config/mcp_servers.json".to_owned(),
                subagent_registry_path: "config/subagents.json".to_owned(),
                enabled_servers: None,
            },
            vec![listener.clone() as Arc<dyn RuntimeHarnessListener>],
        );
        let event = RuntimeEvent::TurnStarted {
            turn_id: TurnId::new(),
            executor: RuntimeExecutor::main_agent(),
            at: SystemTime::now(),
        };

        sink.record(event.clone());

        let recorded = sink.into_events();
        assert_eq!(recorded, vec![event.clone()]);
        let observations = listener
            .observations
            .lock()
            .expect("observations should lock");
        assert_eq!(observations.len(), 2);
        assert!(matches!(
            &observations[0],
            RuntimeHarnessObservation::TurnSnapshot(TurnDebugSnapshot {
                status: DebugTurnStatus::Running,
                turn_number: 3,
                ..
            })
        ));
        assert!(matches!(
            &observations[1],
            RuntimeHarnessObservation::Event(RuntimeHarnessEventEnvelope {
                turn_number: 3,
                event_index: 1,
                event: observed,
                ..
            }) if observed == &event
        ));
    }

    #[tokio::test]
    async fn sse_listener_forwards_runtime_events_to_session_stream() {
        let (events_tx, mut events_rx) = broadcast::channel(8);
        let listener = SseRuntimeHarnessListener::new(events_tx, 8);
        let session_id = SessionId::new();
        let event = RuntimeEvent::TurnStarted {
            turn_id: TurnId::new(),
            executor: RuntimeExecutor::main_agent(),
            at: SystemTime::now(),
        };

        listener
            .try_observe(RuntimeHarnessObservation::Event(
                RuntimeHarnessEventEnvelope {
                    session_id: session_id.clone(),
                    turn_number: 1,
                    event_index: 1,
                    event: event.clone(),
                },
            ))
            .expect("listener should accept event");

        let forwarded = events_rx.recv().await.expect("event should be forwarded");
        assert!(matches!(
            forwarded,
            SessionEvent::RuntimeEvent {
                session_id: forwarded_session_id,
                event: forwarded_event,
            } if forwarded_session_id == session_id && forwarded_event == event
        ));
    }

    #[test]
    fn event_payload_is_flattened_and_includes_executor() {
        let event = RuntimeEvent::McpResponded {
            turn_id: TurnId::new(),
            step_id: agent_runtime::StepId::new(),
            server_name: "postgres".to_owned(),
            tool_name: "list_schemas".to_owned(),
            executor: RuntimeExecutor::subagent("Tool Executor", "tool-executor"),
            at: SystemTime::now(),
            latency: std::time::Duration::from_millis(25),
            was_error: false,
            result_summary: "summary".to_owned(),
            error: None,
            response_payload: serde_json::json!({
                "result": [{"schema_name": "public"}],
            }),
        };

        let payload = runtime_event_payload(&event);
        assert_eq!(payload["server_name"], "postgres");
        assert_eq!(payload["tool_name"], "list_schemas");
        assert_eq!(payload["executor"]["display_name"], "Tool Executor");
        assert_eq!(
            payload["response_payload"]["result"][0]["schema_name"],
            "public"
        );
        assert!(payload.get("McpResponded").is_none());
    }

    #[test]
    fn available_filters_include_steps_executors_and_none_step() {
        let main = RuntimeExecutor::main_agent();
        let subagent = RuntimeExecutor::subagent("Tool Executor", "tool-executor");
        let events = vec![
            DebugHistoryEventRow {
                event_index: 1,
                event_type: "step_started".to_owned(),
                occurred_at: "2026-03-26T00:00:00Z".to_owned(),
                turn_id: Some(TurnId::new().to_string()),
                step_id: Some(StepId::new().to_string()),
                step_number: Some(1),
                executor: Some(main.clone()),
                payload: serde_json::json!({}),
            },
            DebugHistoryEventRow {
                event_index: 2,
                event_type: "turn_started".to_owned(),
                occurred_at: "2026-03-26T00:00:01Z".to_owned(),
                turn_id: Some(TurnId::new().to_string()),
                step_id: None,
                step_number: None,
                executor: Some(main),
                payload: serde_json::json!({}),
            },
        ];
        let artifacts = vec![DebugHistoryRawArtifactRow {
            artifact_index: 1,
            artifact_kind: "mcp_response".to_owned(),
            source: "mcp_client".to_owned(),
            summary: None,
            occurred_at: "2026-03-26T00:00:02Z".to_owned(),
            turn_id: TurnId::new().to_string(),
            step_id: Some(StepId::new().to_string()),
            step_number: Some(2),
            executor: Some(subagent),
            payload: serde_json::json!({}),
        }];

        let filters = build_available_filters(&events, &artifacts);

        assert_eq!(filters.event_types, vec!["step_started", "turn_started"]);
        assert_eq!(filters.artifact_kinds, vec!["mcp_response"]);
        assert_eq!(filters.executors[0].key, "main");
        assert_eq!(filters.executors[1].key, "subagent:tool-executor");
        assert_eq!(
            filters.steps,
            vec![
                DebugHistoryFilterOption {
                    key: "1".to_owned(),
                    label: "Step 1".to_owned()
                },
                DebugHistoryFilterOption {
                    key: "2".to_owned(),
                    label: "Step 2".to_owned()
                },
                DebugHistoryFilterOption {
                    key: "none".to_owned(),
                    label: "No step".to_owned()
                }
            ]
        );
    }

    #[test]
    fn turn_filters_use_and_across_groups_and_or_within_group() {
        let turn_record = sample_turn_record();
        let step_lookup = build_step_number_lookup(&turn_record);
        let step_one = turn_record.steps[0].step_id.to_string();
        let step_two = turn_record.steps[1].step_id.to_string();
        let turn_id = turn_record.turn_id.to_string();

        let mut events = vec![
            DebugHistoryEventRow {
                event_index: 1,
                event_type: "model_called".to_owned(),
                occurred_at: "2026-03-26T00:00:00Z".to_owned(),
                turn_id: Some(turn_id.clone()),
                step_id: Some(step_one),
                step_number: None,
                executor: Some(RuntimeExecutor::main_agent()),
                payload: serde_json::json!({}),
            },
            DebugHistoryEventRow {
                event_index: 2,
                event_type: "mcp_responded".to_owned(),
                occurred_at: "2026-03-26T00:00:01Z".to_owned(),
                turn_id: Some(turn_id.clone()),
                step_id: Some(step_two.clone()),
                step_number: None,
                executor: Some(RuntimeExecutor::subagent("Tool Executor", "tool-executor")),
                payload: serde_json::json!({}),
            },
        ];
        let mut artifacts = vec![
            DebugHistoryRawArtifactRow {
                artifact_index: 1,
                artifact_kind: "model_request".to_owned(),
                source: "openai_responses".to_owned(),
                summary: None,
                occurred_at: "2026-03-26T00:00:02Z".to_owned(),
                turn_id: turn_id.clone(),
                step_id: Some(step_two),
                step_number: None,
                executor: Some(RuntimeExecutor::subagent("Tool Executor", "tool-executor")),
                payload: serde_json::json!({}),
            },
            DebugHistoryRawArtifactRow {
                artifact_index: 2,
                artifact_kind: "mcp_error".to_owned(),
                source: "mcp_client".to_owned(),
                summary: None,
                occurred_at: "2026-03-26T00:00:03Z".to_owned(),
                turn_id,
                step_id: None,
                step_number: None,
                executor: Some(RuntimeExecutor::main_agent()),
                payload: serde_json::json!({}),
            },
        ];

        annotate_event_step_numbers(&mut events, &step_lookup);
        annotate_raw_artifact_step_numbers(&mut artifacts, &step_lookup);

        let filters = DebugHistoryTurnFilters {
            event_types: ["mcp_responded".to_owned()].into_iter().collect(),
            artifact_kinds: ["model_request".to_owned()].into_iter().collect(),
            executors: ["subagent:tool-executor".to_owned()].into_iter().collect(),
            steps: ["2".to_owned()].into_iter().collect(),
        };

        apply_turn_filters(&mut events, &mut artifacts, &filters);

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "mcp_responded");
        assert_eq!(events[0].step_number, Some(2));
        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0].artifact_kind, "model_request");
        assert_eq!(artifacts[0].step_number, Some(2));
    }

    fn sample_turn_record() -> TurnRecord {
        TurnRecord {
            turn_id: TurnId::new(),
            started_at: SystemTime::now(),
            ended_at: SystemTime::now(),
            steps: vec![
                StepRecord {
                    step_id: StepId::new(),
                    step_number: 1,
                    started_at: SystemTime::now(),
                    ended_at: SystemTime::now(),
                    prompt: PromptSnapshot {
                        rendered: "step one".to_owned(),
                        sections: Vec::new(),
                    },
                    decision: None,
                    messages: Vec::new(),
                    outcome: StepOutcomeKind::Continue,
                    usage: UsageSummary::default(),
                },
                StepRecord {
                    step_id: StepId::new(),
                    step_number: 2,
                    started_at: SystemTime::now(),
                    ended_at: SystemTime::now(),
                    prompt: PromptSnapshot {
                        rendered: "step two".to_owned(),
                        sections: Vec::new(),
                    },
                    decision: None,
                    messages: Vec::new(),
                    outcome: StepOutcomeKind::Final,
                    usage: UsageSummary::default(),
                },
            ],
            messages: Vec::new(),
            final_text: Some("done".to_owned()),
            termination: TerminationReason::Final,
            usage: UsageSummary::default(),
        }
    }
}
