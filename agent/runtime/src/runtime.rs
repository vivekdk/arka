//! Agent runtime implementation.

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant, SystemTime},
};

use crate::{
    events::{
        EventSink, InMemoryEventSink, RuntimeDebugSink, RuntimeEvent, RuntimeExecutor,
        RuntimeRawArtifact, RuntimeRawArtifactKind,
    },
    ids::{MessageId, StepId, TurnId},
    model::{
        FinalAnswerRenderRequest, FinalAnswerRenderResponse, FinalAnswerStreamSink, ModelAdapter,
        ModelAdapterArtifact, ModelAdapterArtifactKind, ModelAdapterDebugSink, ModelAdapterError,
        ModelStepDecision, SubagentDecision, SubagentStepRequest,
    },
    prompt::{PromptAssembler, PromptRenderError, load_and_render_system_prompt},
    state::{
        LlmMessageRecord, McpCallMessageRecord, McpCapability, McpCapabilityTarget,
        McpResultMessageRecord, MessageRecord, RunRequest, RuntimeLimits, ServerName,
        StepOutcomeKind, StepRecord, TerminationReason, TurnOutcome, TurnRecord, UsageSummary,
        UserMessageRecord,
    },
    subagent::{ConfiguredSubagent, SubagentConfigError, SubagentRegistry, load_subagent_prompt},
};
use mcp_client::{
    ClientInfo, McpClient, McpClientError, McpConnection, McpResourceReadResult, McpToolCallResult,
    McpToolName,
};
use mcp_config::{ConfigError, McpRegistry, McpServerConfig};
use mcp_metadata::{
    CapabilityKind, FullResourceMetadata, FullToolMetadata, McpFullCatalog, McpMinimalCatalog,
    artifact_paths, load_full_catalog, load_minimal_catalog,
};
use thiserror::Error;
use tokio::time::timeout;
use tracing::{info, warn};

/// Runtime entrypoint parameterized by a provider-specific model adapter.
pub struct AgentRuntime<A> {
    model_adapter: A,
    prompt_assembler: PromptAssembler,
}

/// Reusable MCP server state that can be kept alive across multiple turns.
#[derive(Debug)]
pub struct McpSession {
    servers: HashMap<ServerName, PreparedServer>,
}

#[derive(Debug)]
struct PreparedServer {
    config: McpServerConfig,
    minimal_catalog: McpMinimalCatalog,
    full_catalog: McpFullCatalog,
    full_catalog_markdown: String,
    connection: Option<McpConnection>,
}

#[derive(Clone, Debug)]
struct CapturedMcpResult {
    result_summary: String,
    error: Option<String>,
    response_payload: serde_json::Value,
    was_error: bool,
}

impl CapturedMcpResult {
    fn artifact_payload(&self, target: &McpCapabilityTarget) -> serde_json::Value {
        serde_json::json!({
            "target": {
                "server_name": target.server_name.to_string(),
                "capability_kind": format!("{:?}", target.capability_kind),
                "capability_id": target.capability_id,
            },
            "result_summary": self.result_summary,
            "error": self.error,
            "response_payload": self.response_payload,
        })
    }
}

fn subagent_executor(configured: &ConfiguredSubagent) -> RuntimeExecutor {
    RuntimeExecutor::subagent(
        configured.display_name.clone(),
        configured.subagent_type.clone(),
    )
}

fn subagent_display_name(subagent_type: &str) -> String {
    subagent_type
        .split('-')
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => {
                    let mut word = first.to_uppercase().collect::<String>();
                    word.push_str(chars.as_str());
                    word
                }
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

impl<A> AgentRuntime<A> {
    pub fn new(model_adapter: A) -> Self {
        Self {
            model_adapter,
            prompt_assembler: PromptAssembler,
        }
    }
}

impl<A> AgentRuntime<A>
where
    A: ModelAdapter,
{
    pub async fn run_turn(&self, request: RunRequest) -> Result<TurnOutcome, RuntimeError> {
        let mut sink = InMemoryEventSink::default();
        let mut mcp_session = self
            .prepare_mcp_session(&request.registry_path, request.enabled_servers.as_deref())
            .await?;
        let outcome = self
            .run_turn_with_sink_and_mcp_session(request, &mut mcp_session, &mut sink)
            .await?;
        Ok(TurnOutcome {
            events: sink.into_events(),
            ..outcome
        })
    }

    pub async fn run_turn_with_sink(
        &self,
        request: RunRequest,
        sink: &mut dyn RuntimeDebugSink,
    ) -> Result<TurnOutcome, RuntimeError> {
        let mut mcp_session = self
            .prepare_mcp_session(&request.registry_path, request.enabled_servers.as_deref())
            .await?;
        self.run_turn_with_sink_and_mcp_session(request, &mut mcp_session, sink)
            .await
    }

    pub async fn prepare_mcp_session(
        &self,
        registry_path: &Path,
        enabled_servers: Option<&[ServerName]>,
    ) -> Result<McpSession, RuntimeError> {
        Ok(McpSession {
            servers: prepare_servers(registry_path, enabled_servers)?,
        })
    }

    pub async fn run_turn_with_mcp_session(
        &self,
        request: RunRequest,
        mcp_session: &mut McpSession,
    ) -> Result<TurnOutcome, RuntimeError> {
        let mut sink = InMemoryEventSink::default();
        let outcome = self
            .run_turn_with_sink_and_mcp_session(request, mcp_session, &mut sink)
            .await?;
        Ok(TurnOutcome {
            events: sink.into_events(),
            ..outcome
        })
    }

    pub async fn run_turn_with_mcp_session_and_sink(
        &self,
        request: RunRequest,
        mcp_session: &mut McpSession,
        sink: &mut dyn RuntimeDebugSink,
    ) -> Result<TurnOutcome, RuntimeError> {
        self.run_turn_with_sink_and_mcp_session(request, mcp_session, sink)
            .await
    }

    async fn run_turn_with_sink_and_mcp_session(
        &self,
        request: RunRequest,
        mcp_session: &mut McpSession,
        sink: &mut dyn RuntimeDebugSink,
    ) -> Result<TurnOutcome, RuntimeError> {
        validate_limits(&request)?;

        let subagent_registry = SubagentRegistry::load_from_path(&request.subagent_registry_path)?;
        let subagent_cards = subagent_registry.enabled_cards();
        let main_executor = RuntimeExecutor::main_agent();

        let turn_id = TurnId::new();
        let turn_started_at = SystemTime::now();
        sink.record(RuntimeEvent::TurnStarted {
            turn_id: turn_id.clone(),
            executor: main_executor.clone(),
            at: turn_started_at,
        });

        let mut usage = UsageSummary::default();
        let user_message = MessageRecord::User(UserMessageRecord {
            message_id: MessageId::new(),
            timestamp: SystemTime::now(),
            content: request.user_message.clone(),
        });
        let mut all_messages = vec![user_message.clone()];
        let mut prompt_messages = vec![user_message];
        let mut steps = Vec::new();
        let turn_start = Instant::now();

        let capabilities = build_capabilities(&mcp_session.servers);
        let system_prompt = load_and_render_system_prompt(
            &request.system_prompt_path,
            &capabilities,
            &subagent_cards,
            &request.response_target,
        )?;

        for step_number in 1..=request.limits.max_steps_per_turn {
            ensure_turn_time_remaining(turn_start, request.limits.turn_timeout)?;

            let step_id = StepId::new();
            let step_started_at = SystemTime::now();
            sink.record(RuntimeEvent::StepStarted {
                turn_id: turn_id.clone(),
                step_id: step_id.clone(),
                step_number,
                executor: main_executor.clone(),
                at: step_started_at,
            });

            let prompt = self.prompt_assembler.build(
                &system_prompt,
                &request.conversation_history,
                &prompt_messages,
            );
            sink.record(RuntimeEvent::PromptBuilt {
                turn_id: turn_id.clone(),
                step_id: step_id.clone(),
                executor: main_executor.clone(),
                at: SystemTime::now(),
            });

            let (decision, step_usage) = self
                .generate_validated_step(
                    step_number,
                    prompt.clone(),
                    request.model_config.clone(),
                    turn_start,
                    request.limits.turn_timeout,
                    sink,
                    &turn_id,
                    &step_id,
                )
                .await?;
            usage.add_assign(step_usage);

            let mut step_messages = Vec::new();
            match decision.clone() {
                ModelStepDecision::Final { content } => {
                    let rendered = self
                        .render_final_answer(
                            &request,
                            &prompt_messages,
                            request.model_config.clone(),
                            &content,
                            sink,
                            &turn_id,
                            &step_id,
                            &main_executor,
                        )
                        .await?;
                    usage.add_assign(rendered.usage);
                    let message = MessageRecord::Llm(LlmMessageRecord {
                        message_id: MessageId::new(),
                        timestamp: SystemTime::now(),
                        content: rendered.canonical_text.clone(),
                    });
                    step_messages.push(message.clone());
                    all_messages.push(message);
                    prompt_messages
                        .push(step_messages.last().expect("message just pushed").clone());

                    let step = StepRecord {
                        step_id: step_id.clone(),
                        step_number,
                        started_at: step_started_at,
                        ended_at: SystemTime::now(),
                        prompt,
                        decision: Some(decision),
                        messages: step_messages,
                        outcome: StepOutcomeKind::Final,
                        usage: step_usage,
                    };
                    steps.push(step);
                    return finish_turn(
                        sink,
                        turn_id,
                        turn_started_at,
                        SystemTime::now(),
                        steps,
                        all_messages,
                        usage,
                        rendered.canonical_text,
                        rendered.display_text,
                        TerminationReason::Final,
                        &main_executor,
                    );
                }
                ModelStepDecision::DelegateSubagent { delegation } => {
                    let llm_message = MessageRecord::Llm(LlmMessageRecord {
                        message_id: MessageId::new(),
                        timestamp: SystemTime::now(),
                        content: format!(
                            "delegating to sub-agent `{}` for target `{}`",
                            delegation.subagent_type, delegation.target.capability_id
                        ),
                    });
                    step_messages.push(llm_message.clone());
                    all_messages.push(llm_message);
                    prompt_messages
                        .push(step_messages.last().expect("message just pushed").clone());

                    let subagent_call =
                        MessageRecord::SubAgentCall(crate::state::SubAgentCallMessageRecord {
                            message_id: MessageId::new(),
                            timestamp: SystemTime::now(),
                            subagent_type: delegation.subagent_type.clone(),
                            goal: delegation.goal.clone(),
                            target: delegation.target.clone(),
                        });
                    step_messages.push(subagent_call.clone());
                    all_messages.push(subagent_call);
                    prompt_messages
                        .push(step_messages.last().expect("message just pushed").clone());
                    sink.record(RuntimeEvent::HandoffToSubagent {
                        turn_id: turn_id.clone(),
                        step_id: step_id.clone(),
                        executor: main_executor.clone(),
                        at: SystemTime::now(),
                        subagent_type: delegation.subagent_type.clone(),
                        goal: delegation.goal.clone(),
                        target: delegation.target.clone(),
                    });

                    let (subagent_result_message, maybe_execution) = self
                        .run_subagent(
                            &request,
                            &subagent_registry,
                            mcp_session,
                            &delegation.subagent_type,
                            &delegation.target,
                            &delegation.goal,
                            turn_start,
                            sink,
                            &turn_id,
                            &step_id,
                        )
                        .await?;
                    if let MessageRecord::SubAgentResult(record) = &subagent_result_message {
                        sink.record(RuntimeEvent::HandoffToMainAgent {
                            turn_id: turn_id.clone(),
                            step_id: step_id.clone(),
                            executor: RuntimeExecutor::subagent(
                                subagent_display_name(&record.subagent_type),
                                record.subagent_type.clone(),
                            ),
                            at: SystemTime::now(),
                            subagent_type: record.subagent_type.clone(),
                            status: record.status.clone(),
                        });
                    }
                    usage.add_assign(maybe_execution.usage);
                    for trace_message in maybe_execution.trace_messages {
                        step_messages.push(trace_message.clone());
                        all_messages.push(trace_message);
                    }
                    step_messages.push(subagent_result_message.clone());
                    all_messages.push(subagent_result_message.clone());
                    prompt_messages.push(subagent_result_message);

                    let step = StepRecord {
                        step_id: step_id.clone(),
                        step_number,
                        started_at: step_started_at,
                        ended_at: SystemTime::now(),
                        prompt,
                        decision: Some(decision),
                        messages: step_messages,
                        outcome: StepOutcomeKind::Continue,
                        usage: usage_delta(step_usage, maybe_execution.usage),
                    };
                    steps.push(step);
                }
            }

            sink.record(RuntimeEvent::StepEnded {
                turn_id: turn_id.clone(),
                step_id: step_id.clone(),
                executor: main_executor.clone(),
                at: SystemTime::now(),
            });
        }

        finish_turn(
            sink,
            turn_id,
            turn_started_at,
            SystemTime::now(),
            steps,
            all_messages,
            usage,
            String::new(),
            String::new(),
            TerminationReason::MaxStepsReached,
            &main_executor,
        )
    }

    async fn generate_validated_step(
        &self,
        step_number: u32,
        prompt: crate::state::PromptSnapshot,
        model_config: crate::state::ModelConfig,
        turn_start: Instant,
        turn_timeout: Duration,
        sink: &mut dyn RuntimeDebugSink,
        turn_id: &TurnId,
        step_id: &StepId,
    ) -> Result<(ModelStepDecision, UsageSummary), RuntimeError> {
        let now = SystemTime::now();
        let executor = RuntimeExecutor::main_agent();
        sink.record(RuntimeEvent::ModelCalled {
            turn_id: turn_id.clone(),
            step_id: step_id.clone(),
            executor: executor.clone(),
            at: now,
        });
        let remaining = remaining_turn_budget(turn_start, turn_timeout)?;
        let model_started = Instant::now();
        let mut debug_sink = BoundModelDebugSink::new(sink, turn_id, step_id, executor.clone());
        let response = timeout(
            remaining,
            self.model_adapter.generate_step(
                crate::model::ModelStepRequest {
                    step_number,
                    prompt,
                    model_config,
                },
                Some(&mut debug_sink),
            ),
        )
        .await
        .map_err(|_| RuntimeError::Timeout("model step timed out".to_owned()))?
        .map_err(RuntimeError::Model)?;

        sink.record(RuntimeEvent::ModelResponded {
            turn_id: turn_id.clone(),
            step_id: step_id.clone(),
            at: SystemTime::now(),
            latency: model_started.elapsed(),
            executor,
            usage: response.usage,
        });
        Ok((response.decision, response.usage))
    }

    async fn run_subagent(
        &self,
        request: &RunRequest,
        registry: &SubagentRegistry,
        mcp_session: &mut McpSession,
        subagent_type: &str,
        target: &McpCapabilityTarget,
        goal: &str,
        turn_start: Instant,
        sink: &mut dyn RuntimeDebugSink,
        turn_id: &TurnId,
        step_id: &StepId,
    ) -> Result<(MessageRecord, DelegatedExecutionOutcome), RuntimeError> {
        let configured = registry.get_enabled(subagent_type)?;
        let base_prompt = build_subagent_prompt(
            &request.subagent_registry_path,
            configured,
            mcp_session,
            target,
            goal,
            &request.user_message,
        )?;
        let model_config = configured
            .model_name
            .clone()
            .map(crate::state::ModelConfig::new)
            .unwrap_or_else(|| request.model_config.clone());
        let executor = subagent_executor(configured);
        let mut usage = UsageSummary::default();
        let mut trace_messages = Vec::new();
        let mut subagent_messages = Vec::new();
        let mut executed_action_count = 0u32;

        for subagent_step_number in 1..=request.limits.max_subagent_steps_per_invocation {
            ensure_turn_time_remaining(turn_start, request.limits.turn_timeout)?;
            let prompt = build_subagent_loop_prompt(
                &base_prompt,
                &subagent_messages,
                subagent_step_number,
                &request.limits,
            );

            let remaining = remaining_turn_budget(turn_start, request.limits.turn_timeout)?;
            sink.record(RuntimeEvent::ModelCalled {
                turn_id: turn_id.clone(),
                step_id: step_id.clone(),
                executor: executor.clone(),
                at: SystemTime::now(),
            });
            let started = Instant::now();
            let mut debug_sink = BoundModelDebugSink::new(sink, turn_id, step_id, executor.clone());
            let response = timeout(
                remaining,
                self.model_adapter.generate_subagent_step(
                    SubagentStepRequest {
                        subagent_type: subagent_type.to_owned(),
                        prompt,
                        model_config: model_config.clone(),
                    },
                    Some(&mut debug_sink),
                ),
            )
            .await
            .map_err(|_| RuntimeError::Timeout("sub-agent step timed out".to_owned()))?
            .map_err(RuntimeError::Model)?;
            usage.add_assign(response.usage);
            sink.record(RuntimeEvent::ModelResponded {
                turn_id: turn_id.clone(),
                step_id: step_id.clone(),
                at: SystemTime::now(),
                latency: started.elapsed(),
                executor: executor.clone(),
                usage: response.usage,
            });
            match response.decision {
                SubagentDecision::Done { summary } => {
                    return Ok((
                        subagent_result_message(
                            subagent_type,
                            "completed",
                            executed_action_count,
                            summary,
                        ),
                        DelegatedExecutionOutcome {
                            usage,
                            trace_messages,
                        },
                    ));
                }
                SubagentDecision::Partial { summary, reason } => {
                    return Ok((
                        subagent_result_message(
                            subagent_type,
                            "partial",
                            executed_action_count,
                            format_partial_detail(&summary, &reason),
                        ),
                        DelegatedExecutionOutcome {
                            usage,
                            trace_messages,
                        },
                    ));
                }
                SubagentDecision::CannotExecute { reason } => {
                    let status = if executed_action_count > 0 {
                        "partial"
                    } else {
                        "cannot_execute"
                    };
                    return Ok((
                        subagent_result_message(
                            subagent_type,
                            status,
                            executed_action_count,
                            reason,
                        ),
                        DelegatedExecutionOutcome {
                            usage,
                            trace_messages,
                        },
                    ));
                }
                SubagentDecision::ToolCall {
                    server_name,
                    tool_name,
                    arguments,
                } => {
                    if executed_action_count >= request.limits.max_subagent_mcp_calls_per_invocation
                    {
                        return Ok((
                            subagent_budget_result(
                                subagent_type,
                                executed_action_count,
                                "sub-agent exhausted its MCP action budget".to_owned(),
                            ),
                            DelegatedExecutionOutcome {
                                usage,
                                trace_messages,
                            },
                        ));
                    }

                    let call_target = match validate_subagent_target(
                        &mcp_session.servers,
                        target,
                        &server_name,
                        CapabilityKind::Tool,
                        &tool_name,
                    ) {
                        Ok(call_target) => call_target,
                        Err(reason) => {
                            return Ok((
                                subagent_policy_result(
                                    subagent_type,
                                    executed_action_count,
                                    reason,
                                ),
                                DelegatedExecutionOutcome {
                                    usage,
                                    trace_messages,
                                },
                            ));
                        }
                    };

                    let call_message = MessageRecord::McpCall(McpCallMessageRecord {
                        message_id: MessageId::new(),
                        timestamp: SystemTime::now(),
                        target: call_target.clone(),
                        arguments: arguments.clone(),
                    });
                    sink.record(RuntimeEvent::McpCalled {
                        turn_id: turn_id.clone(),
                        step_id: step_id.clone(),
                        server_name: call_target.server_name.to_string(),
                        tool_name: call_target.capability_id.clone(),
                        executor: executor.clone(),
                        at: SystemTime::now(),
                    });
                    sink.record_raw_artifact(RuntimeRawArtifact {
                        turn_id: turn_id.clone(),
                        step_id: Some(step_id.clone()),
                        occurred_at: SystemTime::now(),
                        kind: RuntimeRawArtifactKind::McpRequest,
                        source: "mcp_client".to_owned(),
                        executor: executor.clone(),
                        summary: Some(format!(
                            "tool call {}::{}",
                            call_target.server_name, call_target.capability_id
                        )),
                        payload: serde_json::json!({
                            "server_name": call_target.server_name.to_string(),
                            "tool_name": call_target.capability_id.clone(),
                            "arguments": arguments.clone(),
                        }),
                    });
                    let started = Instant::now();
                    let result = execute_tool_call(
                        &mut mcp_session.servers,
                        &call_target.server_name,
                        &tool_name,
                        arguments,
                        request.limits.mcp_call_timeout,
                    )
                    .await;
                    let captured = capture_tool_result(result);
                    let result_message = tool_result_message(call_target.clone(), &captured);
                    let was_error = captured.was_error;
                    sink.record_raw_artifact(RuntimeRawArtifact {
                        turn_id: turn_id.clone(),
                        step_id: Some(step_id.clone()),
                        occurred_at: SystemTime::now(),
                        kind: if was_error {
                            RuntimeRawArtifactKind::McpError
                        } else {
                            RuntimeRawArtifactKind::McpResponse
                        },
                        source: "mcp_client".to_owned(),
                        executor: executor.clone(),
                        summary: Some(format!(
                            "{} {}::{}",
                            if was_error {
                                "tool error"
                            } else {
                                "tool response"
                            },
                            call_target.server_name,
                            call_target.capability_id
                        )),
                        payload: captured.artifact_payload(&call_target),
                    });
                    sink.record(RuntimeEvent::McpResponded {
                        turn_id: turn_id.clone(),
                        step_id: step_id.clone(),
                        server_name: call_target.server_name.to_string(),
                        tool_name: call_target.capability_id.clone(),
                        executor: executor.clone(),
                        at: SystemTime::now(),
                        latency: started.elapsed(),
                        was_error,
                        result_summary: captured.result_summary.clone(),
                        error: captured.error.clone(),
                        response_payload: captured.response_payload.clone(),
                    });

                    executed_action_count += 1;
                    trace_messages.push(call_message.clone());
                    trace_messages.push(result_message.clone());
                    subagent_messages.push(call_message);
                    subagent_messages.push(result_message);
                }
                SubagentDecision::ResourceRead {
                    server_name,
                    resource_uri,
                } => {
                    if executed_action_count >= request.limits.max_subagent_mcp_calls_per_invocation
                    {
                        return Ok((
                            subagent_budget_result(
                                subagent_type,
                                executed_action_count,
                                "sub-agent exhausted its MCP action budget".to_owned(),
                            ),
                            DelegatedExecutionOutcome {
                                usage,
                                trace_messages,
                            },
                        ));
                    }

                    let call_target = match validate_subagent_target(
                        &mcp_session.servers,
                        target,
                        &server_name,
                        CapabilityKind::Resource,
                        &resource_uri,
                    ) {
                        Ok(call_target) => call_target,
                        Err(reason) => {
                            return Ok((
                                subagent_policy_result(
                                    subagent_type,
                                    executed_action_count,
                                    reason,
                                ),
                                DelegatedExecutionOutcome {
                                    usage,
                                    trace_messages,
                                },
                            ));
                        }
                    };

                    let call_message = MessageRecord::McpCall(McpCallMessageRecord {
                        message_id: MessageId::new(),
                        timestamp: SystemTime::now(),
                        target: call_target.clone(),
                        arguments: serde_json::Value::Null,
                    });
                    sink.record(RuntimeEvent::McpCalled {
                        turn_id: turn_id.clone(),
                        step_id: step_id.clone(),
                        server_name: call_target.server_name.to_string(),
                        tool_name: call_target.capability_id.clone(),
                        executor: executor.clone(),
                        at: SystemTime::now(),
                    });
                    sink.record_raw_artifact(RuntimeRawArtifact {
                        turn_id: turn_id.clone(),
                        step_id: Some(step_id.clone()),
                        occurred_at: SystemTime::now(),
                        kind: RuntimeRawArtifactKind::McpRequest,
                        source: "mcp_client".to_owned(),
                        executor: executor.clone(),
                        summary: Some(format!(
                            "resource read {}::{}",
                            call_target.server_name, call_target.capability_id
                        )),
                        payload: serde_json::json!({
                            "server_name": call_target.server_name.to_string(),
                            "resource_uri": call_target.capability_id.clone(),
                        }),
                    });
                    let started = Instant::now();
                    let result = execute_resource_read(
                        &mut mcp_session.servers,
                        &call_target.server_name,
                        &resource_uri,
                        request.limits.mcp_call_timeout,
                    )
                    .await;
                    let captured = capture_resource_result(result);
                    let result_message = resource_result_message(call_target.clone(), &captured);
                    let was_error = captured.was_error;
                    sink.record_raw_artifact(RuntimeRawArtifact {
                        turn_id: turn_id.clone(),
                        step_id: Some(step_id.clone()),
                        occurred_at: SystemTime::now(),
                        kind: if was_error {
                            RuntimeRawArtifactKind::McpError
                        } else {
                            RuntimeRawArtifactKind::McpResponse
                        },
                        source: "mcp_client".to_owned(),
                        executor: executor.clone(),
                        summary: Some(format!(
                            "{} {}::{}",
                            if was_error {
                                "resource error"
                            } else {
                                "resource response"
                            },
                            call_target.server_name,
                            call_target.capability_id
                        )),
                        payload: captured.artifact_payload(&call_target),
                    });
                    sink.record(RuntimeEvent::McpResponded {
                        turn_id: turn_id.clone(),
                        step_id: step_id.clone(),
                        server_name: call_target.server_name.to_string(),
                        tool_name: call_target.capability_id.clone(),
                        executor: executor.clone(),
                        at: SystemTime::now(),
                        latency: started.elapsed(),
                        was_error,
                        result_summary: captured.result_summary.clone(),
                        error: captured.error.clone(),
                        response_payload: captured.response_payload.clone(),
                    });

                    executed_action_count += 1;
                    trace_messages.push(call_message.clone());
                    trace_messages.push(result_message.clone());
                    subagent_messages.push(call_message);
                    subagent_messages.push(result_message);
                }
            }
        }

        Ok((
            subagent_budget_result(
                subagent_type,
                executed_action_count,
                "sub-agent exhausted its reasoning budget before reaching a terminal result"
                    .to_owned(),
            ),
            DelegatedExecutionOutcome {
                usage,
                trace_messages,
            },
        ))
    }

    async fn render_final_answer(
        &self,
        request: &RunRequest,
        prompt_messages: &[MessageRecord],
        model_config: crate::state::ModelConfig,
        answer_brief: &str,
        sink: &mut dyn RuntimeDebugSink,
        turn_id: &TurnId,
        step_id: &StepId,
        executor: &RuntimeExecutor,
    ) -> Result<FinalAnswerRenderResponse, RuntimeError> {
        sink.record(RuntimeEvent::AnswerRenderStarted {
            turn_id: turn_id.clone(),
            step_id: step_id.clone(),
            executor: executor.clone(),
            at: SystemTime::now(),
        });
        let prompt = build_final_answer_render_prompt(
            &request.user_message,
            &request.conversation_history,
            prompt_messages,
            answer_brief,
            &request.response_target,
        );
        let started = Instant::now();
        let mut stream_sink = RuntimeAnswerStreamSink {
            turn_id: turn_id.clone(),
            step_id: step_id.clone(),
            executor: executor.clone(),
            event_sink: sink,
        };
        let result = self
            .model_adapter
            .render_final_answer(
                FinalAnswerRenderRequest {
                    model_config,
                    prompt,
                    answer_brief: answer_brief.to_owned(),
                    response_target: request.response_target.clone(),
                },
                Some(&mut stream_sink),
                None,
            )
            .await
            .map_err(RuntimeError::Model);
        match result {
            Ok(response) => {
                sink.record(RuntimeEvent::AnswerRenderCompleted {
                    turn_id: turn_id.clone(),
                    step_id: step_id.clone(),
                    executor: executor.clone(),
                    at: SystemTime::now(),
                });
                sink.record(RuntimeEvent::ModelResponded {
                    turn_id: turn_id.clone(),
                    step_id: step_id.clone(),
                    at: SystemTime::now(),
                    latency: started.elapsed(),
                    executor: executor.clone(),
                    usage: response.usage,
                });
                Ok(response)
            }
            Err(error) => {
                sink.record(RuntimeEvent::AnswerRenderFailed {
                    turn_id: turn_id.clone(),
                    step_id: step_id.clone(),
                    executor: executor.clone(),
                    at: SystemTime::now(),
                    error: error.to_string(),
                });
                Err(error)
            }
        }
    }
}

struct RuntimeAnswerStreamSink<'a> {
    turn_id: TurnId,
    step_id: StepId,
    executor: RuntimeExecutor,
    event_sink: &'a mut dyn RuntimeDebugSink,
}

impl FinalAnswerStreamSink for RuntimeAnswerStreamSink<'_> {
    fn record_text_delta(&mut self, delta: &str) {
        if delta.is_empty() {
            return;
        }
        self.event_sink.record(RuntimeEvent::AnswerTextDelta {
            turn_id: self.turn_id.clone(),
            step_id: self.step_id.clone(),
            executor: self.executor.clone(),
            at: SystemTime::now(),
            delta: delta.to_owned(),
        });
    }
}

#[derive(Default)]
struct DelegatedExecutionOutcome {
    usage: UsageSummary,
    trace_messages: Vec<MessageRecord>,
}

struct BoundModelDebugSink<'a> {
    sink: &'a mut dyn RuntimeDebugSink,
    turn_id: &'a TurnId,
    step_id: &'a StepId,
    executor: RuntimeExecutor,
}

impl<'a> BoundModelDebugSink<'a> {
    fn new(
        sink: &'a mut dyn RuntimeDebugSink,
        turn_id: &'a TurnId,
        step_id: &'a StepId,
        executor: RuntimeExecutor,
    ) -> Self {
        Self {
            sink,
            turn_id,
            step_id,
            executor,
        }
    }
}

impl ModelAdapterDebugSink for BoundModelDebugSink<'_> {
    fn record_model_artifact(&mut self, artifact: ModelAdapterArtifact) {
        self.sink.record_raw_artifact(RuntimeRawArtifact {
            turn_id: self.turn_id.clone(),
            step_id: Some(self.step_id.clone()),
            occurred_at: SystemTime::now(),
            kind: match artifact.kind {
                ModelAdapterArtifactKind::Request => RuntimeRawArtifactKind::ModelRequest,
                ModelAdapterArtifactKind::Response => RuntimeRawArtifactKind::ModelResponse,
                ModelAdapterArtifactKind::Error => RuntimeRawArtifactKind::ModelError,
            },
            source: artifact.source,
            executor: self.executor.clone(),
            summary: artifact.summary,
            payload: artifact.payload,
        });
    }
}

fn prepare_servers(
    registry_path: &Path,
    enabled_servers: Option<&[ServerName]>,
) -> Result<HashMap<ServerName, PreparedServer>, RuntimeError> {
    let registry = McpRegistry::load_from_path(registry_path)?;
    let selected = select_servers(&registry, enabled_servers)?;
    let metadata_dir = artifact_paths(Path::new(mcp_metadata::DEFAULT_METADATA_DIR), "").dir;

    let mut prepared = HashMap::new();
    for name in selected {
        let config = registry.get(name.as_str())?.clone();
        let paths = artifact_paths(&metadata_dir, name.as_str());
        let minimal_catalog = load_minimal_catalog(&paths.minimal_path)?;
        let full_catalog = load_full_catalog(&paths.full_path)?;
        let full_catalog_markdown = fs::read_to_string(&paths.full_path).map_err(|source| {
            mcp_metadata::MetadataError::Read {
                path: paths.full_path.clone(),
                source,
            }
        })?;
        prepared.insert(
            name,
            PreparedServer {
                config,
                minimal_catalog,
                full_catalog,
                full_catalog_markdown,
                connection: None,
            },
        );
    }

    Ok(prepared)
}

fn build_capabilities(servers: &HashMap<ServerName, PreparedServer>) -> Vec<McpCapability> {
    let mut capabilities = Vec::new();
    for (server_name, server) in servers {
        for tool in &server.minimal_catalog.tools {
            capabilities.push(McpCapability {
                server_name: server_name.clone(),
                server_description: server.minimal_catalog.server.description.clone(),
                kind: CapabilityKind::Tool,
                capability_id: tool.name.clone(),
                title: tool.title.clone(),
                description: tool.description.clone(),
            });
        }
        for resource in &server.minimal_catalog.resources {
            capabilities.push(McpCapability {
                server_name: server_name.clone(),
                server_description: server.minimal_catalog.server.description.clone(),
                kind: CapabilityKind::Resource,
                capability_id: resource.uri.clone(),
                title: resource.title.clone(),
                description: resource.description.clone(),
            });
        }
    }
    capabilities.sort_by(|left, right| {
        left.server_name
            .as_str()
            .cmp(right.server_name.as_str())
            .then_with(|| left.capability_id.cmp(&right.capability_id))
    });
    capabilities
}

fn build_subagent_prompt(
    registry_path: &Path,
    configured: &ConfiguredSubagent,
    mcp_session: &McpSession,
    target: &McpCapabilityTarget,
    goal: &str,
    user_message: &str,
) -> Result<String, RuntimeError> {
    let registry_dir = registry_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let server = mcp_session
        .servers
        .get(&target.server_name)
        .ok_or_else(|| RuntimeError::UnknownServer(target.server_name.to_string()))?;
    let template = load_subagent_prompt(&registry_dir, configured, &server.full_catalog_markdown)?;

    let capability_block = match target.capability_kind {
        CapabilityKind::Tool => {
            let tool = lookup_tool(&server.full_catalog, &target.capability_id)?;
            format!(
                "Selected tool:\n- server: {}\n- name: {}\n- title: {}\n- description: {}\n- input_schema: {}\n",
                target.server_name,
                tool.name,
                tool.title.as_deref().unwrap_or("None"),
                tool.description.as_deref().unwrap_or("None"),
                tool.input_schema
            )
        }
        CapabilityKind::Resource => {
            let resource = lookup_resource(&server.full_catalog, &target.capability_id)?;
            format!(
                "Selected resource:\n- server: {}\n- uri: {}\n- title: {}\n- description: {}\n- mime_type: {}\n",
                target.server_name,
                resource.uri,
                resource.title.as_deref().unwrap_or("None"),
                resource.description.as_deref().unwrap_or("None"),
                resource.mime_type.as_deref().unwrap_or("None"),
            )
        }
    };

    Ok(format!(
        "{template}\n\n## Delegation Context\n- Goal: {goal}\n- User request: {user_message}\n- Selected capability kind: {:?}\n- Selected capability id: {}\n\n{capability_block}",
        target.capability_kind, target.capability_id
    ))
}

fn build_subagent_loop_prompt(
    base_prompt: &str,
    subagent_messages: &[MessageRecord],
    subagent_step_number: u32,
    limits: &RuntimeLimits,
) -> String {
    let prior_context = if subagent_messages.is_empty() {
        "No delegated MCP actions have been executed yet.".to_owned()
    } else {
        subagent_messages
            .iter()
            .map(MessageRecord::summary_line)
            .collect::<Vec<_>>()
            .join("\n")
    };

    format!(
        "{base_prompt}\n\n## Sub-agent Loop State\n- Iteration: {subagent_step_number}\n- Max sub-agent iterations: {}\n- Max delegated MCP actions: {}\n\n## Delegated Execution History\n{prior_context}",
        limits.max_subagent_steps_per_invocation, limits.max_subagent_mcp_calls_per_invocation,
    )
}

fn validate_subagent_target(
    servers: &HashMap<ServerName, PreparedServer>,
    selected_target: &McpCapabilityTarget,
    requested_server_name: &ServerName,
    capability_kind: CapabilityKind,
    capability_id: &str,
) -> Result<McpCapabilityTarget, String> {
    if requested_server_name != &selected_target.server_name {
        return Err(format!(
            "sub-agent may only use MCP server `{}` during this delegation, but requested `{}`",
            selected_target.server_name, requested_server_name
        ));
    }

    let server = servers
        .get(requested_server_name)
        .ok_or_else(|| format!("unknown MCP server `{requested_server_name}` requested"))?;
    match capability_kind {
        CapabilityKind::Tool => {
            lookup_tool(&server.full_catalog, capability_id).map_err(|error| error.to_string())?;
        }
        CapabilityKind::Resource => {
            lookup_resource(&server.full_catalog, capability_id)
                .map_err(|error| error.to_string())?;
        }
    }

    Ok(McpCapabilityTarget {
        server_name: requested_server_name.clone(),
        capability_kind,
        capability_id: capability_id.to_owned(),
    })
}

fn subagent_result_message(
    subagent_type: &str,
    status: &str,
    executed_action_count: u32,
    detail: String,
) -> MessageRecord {
    MessageRecord::SubAgentResult(crate::state::SubAgentResultMessageRecord {
        message_id: MessageId::new(),
        timestamp: SystemTime::now(),
        subagent_type: subagent_type.to_owned(),
        status: status.to_owned(),
        executed_action_count,
        detail,
    })
}

fn subagent_budget_result(
    subagent_type: &str,
    executed_action_count: u32,
    reason: String,
) -> MessageRecord {
    let status = if executed_action_count > 0 {
        "partial"
    } else {
        "cannot_execute"
    };
    subagent_result_message(subagent_type, status, executed_action_count, reason)
}

fn subagent_policy_result(
    subagent_type: &str,
    executed_action_count: u32,
    reason: String,
) -> MessageRecord {
    let status = if executed_action_count > 0 {
        "partial"
    } else {
        "cannot_execute"
    };
    subagent_result_message(subagent_type, status, executed_action_count, reason)
}

fn format_partial_detail(summary: &str, reason: &str) -> String {
    if summary.trim().is_empty() {
        reason.to_owned()
    } else if reason.trim().is_empty() {
        summary.to_owned()
    } else {
        format!("{summary}\nStopped because: {reason}")
    }
}

async fn execute_tool_call(
    servers: &mut HashMap<ServerName, PreparedServer>,
    server_name: &ServerName,
    tool_name: &str,
    arguments: serde_json::Value,
    timeout_duration: Duration,
) -> Result<McpToolCallResult, RuntimeError> {
    let server = servers
        .get_mut(server_name)
        .ok_or_else(|| RuntimeError::UnknownServer(server_name.to_string()))?;
    lookup_tool(&server.full_catalog, tool_name)?;
    ensure_connection(server).await?;
    info!(
        server = %server_name,
        tool = tool_name,
        timeout_ms = timeout_duration.as_millis() as u64,
        "starting MCP tool call"
    );
    let result = timeout(
        timeout_duration,
        server
            .connection
            .as_ref()
            .expect("connection should exist")
            .call_tool(
                &McpToolName::new(tool_name.to_owned()).map_err(RuntimeError::McpClient)?,
                arguments,
            ),
    )
    .await;
    match result {
        Ok(Ok(value)) => {
            info!(
                server = %server_name,
                tool = tool_name,
                "completed MCP tool call"
            );
            Ok(value)
        }
        Ok(Err(error)) => {
            warn!(
                server = %server_name,
                tool = tool_name,
                error = %error,
                "MCP tool call failed; resetting connection"
            );
            server.connection.take();
            Err(RuntimeError::McpClient(error))
        }
        Err(_) => {
            warn!(
                server = %server_name,
                tool = tool_name,
                timeout_ms = timeout_duration.as_millis() as u64,
                "MCP tool call timed out; resetting connection"
            );
            server.connection.take();
            Err(RuntimeError::Timeout("MCP call timed out".to_owned()))
        }
    }
}

async fn execute_resource_read(
    servers: &mut HashMap<ServerName, PreparedServer>,
    server_name: &ServerName,
    resource_uri: &str,
    timeout_duration: Duration,
) -> Result<McpResourceReadResult, RuntimeError> {
    let server = servers
        .get_mut(server_name)
        .ok_or_else(|| RuntimeError::UnknownServer(server_name.to_string()))?;
    lookup_resource(&server.full_catalog, resource_uri)?;
    ensure_connection(server).await?;
    info!(
        server = %server_name,
        resource_uri,
        timeout_ms = timeout_duration.as_millis() as u64,
        "starting MCP resource read"
    );
    let result = timeout(
        timeout_duration,
        server
            .connection
            .as_ref()
            .expect("connection should exist")
            .read_resource(resource_uri),
    )
    .await;
    match result {
        Ok(Ok(value)) => {
            info!(
                server = %server_name,
                resource_uri,
                "completed MCP resource read"
            );
            Ok(value)
        }
        Ok(Err(error)) => {
            warn!(
                server = %server_name,
                resource_uri,
                error = %error,
                "MCP resource read failed; resetting connection"
            );
            server.connection.take();
            Err(RuntimeError::McpClient(error))
        }
        Err(_) => {
            warn!(
                server = %server_name,
                resource_uri,
                timeout_ms = timeout_duration.as_millis() as u64,
                "MCP resource read timed out; resetting connection"
            );
            server.connection.take();
            Err(RuntimeError::Timeout(
                "MCP resource read timed out".to_owned(),
            ))
        }
    }
}

async fn ensure_connection(server: &mut PreparedServer) -> Result<(), RuntimeError> {
    if server.connection.is_some() {
        return Ok(());
    }

    let connection = McpClient::connect(&server.config).await?;
    connection
        .initialize(ClientInfo::new("agent-runtime", env!("CARGO_PKG_VERSION")))
        .await?;
    connection.notify_initialized().await?;
    server.connection = Some(connection);
    Ok(())
}

fn lookup_tool<'a>(
    catalog: &'a McpFullCatalog,
    tool_name: &str,
) -> Result<&'a FullToolMetadata, RuntimeError> {
    catalog
        .tools
        .iter()
        .find(|tool| tool.name == tool_name)
        .ok_or_else(|| RuntimeError::UnknownCapability {
            server_name: catalog.server.logical_name.clone(),
            capability_kind: CapabilityKind::Tool,
            capability_id: tool_name.to_owned(),
        })
}

fn lookup_resource<'a>(
    catalog: &'a McpFullCatalog,
    resource_uri: &str,
) -> Result<&'a FullResourceMetadata, RuntimeError> {
    catalog
        .resources
        .iter()
        .find(|resource| resource.uri == resource_uri)
        .ok_or_else(|| RuntimeError::UnknownCapability {
            server_name: catalog.server.logical_name.clone(),
            capability_kind: CapabilityKind::Resource,
            capability_id: resource_uri.to_owned(),
        })
}

fn tool_result_message(target: McpCapabilityTarget, captured: &CapturedMcpResult) -> MessageRecord {
    MessageRecord::McpResult(McpResultMessageRecord {
        message_id: MessageId::new(),
        timestamp: SystemTime::now(),
        target,
        result_summary: captured.result_summary.clone(),
        error: captured.error.clone(),
    })
}

fn resource_result_message(
    target: McpCapabilityTarget,
    captured: &CapturedMcpResult,
) -> MessageRecord {
    MessageRecord::McpResult(McpResultMessageRecord {
        message_id: MessageId::new(),
        timestamp: SystemTime::now(),
        target,
        result_summary: captured.result_summary.clone(),
        error: captured.error.clone(),
    })
}

fn capture_tool_result(result: Result<McpToolCallResult, RuntimeError>) -> CapturedMcpResult {
    match result {
        Ok(result) => CapturedMcpResult {
            result_summary: summarize_tool_result(&result),
            error: if result.is_error {
                Some("tool returned MCP error payload".to_owned())
            } else {
                None
            },
            response_payload: serde_json::to_value(&result).unwrap_or_else(|error| {
                serde_json::json!({
                    "serialization_error": error.to_string(),
                })
            }),
            was_error: result.is_error,
        },
        Err(error) => CapturedMcpResult {
            result_summary: error.to_string(),
            error: Some(error.to_string()),
            response_payload: serde_json::json!({
                "runtime_error": error.to_string(),
            }),
            was_error: true,
        },
    }
}

fn capture_resource_result(
    result: Result<McpResourceReadResult, RuntimeError>,
) -> CapturedMcpResult {
    match result {
        Ok(result) => CapturedMcpResult {
            result_summary: summarize_resource_result(&result),
            error: None,
            response_payload: serde_json::to_value(&result).unwrap_or_else(|error| {
                serde_json::json!({
                    "serialization_error": error.to_string(),
                })
            }),
            was_error: false,
        },
        Err(error) => CapturedMcpResult {
            result_summary: error.to_string(),
            error: Some(error.to_string()),
            response_payload: serde_json::json!({
                "runtime_error": error.to_string(),
            }),
            was_error: true,
        },
    }
}

fn summarize_tool_result(result: &McpToolCallResult) -> String {
    if let Some(structured_content) = &result.structured_content {
        return structured_content.to_string();
    }
    if result.content.is_empty() {
        "no content".to_owned()
    } else {
        format!("{:?}", result.content)
    }
}

fn summarize_resource_result(result: &McpResourceReadResult) -> String {
    if result.contents.is_empty() {
        "no resource contents".to_owned()
    } else {
        format!("{:?}", result.contents)
    }
}

fn finish_turn(
    sink: &mut dyn EventSink,
    turn_id: TurnId,
    turn_started_at: SystemTime,
    turn_ended_at: SystemTime,
    steps: Vec<StepRecord>,
    all_messages: Vec<MessageRecord>,
    usage: UsageSummary,
    final_text: String,
    display_text: String,
    termination: TerminationReason,
    executor: &RuntimeExecutor,
) -> Result<TurnOutcome, RuntimeError> {
    sink.record(RuntimeEvent::TurnEnded {
        turn_id: turn_id.clone(),
        executor: executor.clone(),
        at: turn_ended_at,
        termination: termination.clone(),
        usage,
    });

    let turn = TurnRecord {
        turn_id,
        started_at: turn_started_at,
        ended_at: turn_ended_at,
        steps,
        messages: all_messages,
        final_text: if final_text.is_empty() {
            None
        } else {
            Some(final_text.clone())
        },
        termination: termination.clone(),
        usage,
    };

    Ok(TurnOutcome {
        final_text,
        display_text,
        turn,
        events: Vec::new(),
        usage,
        termination,
    })
}

fn build_final_answer_render_prompt(
    user_message: &str,
    conversation_history: &[crate::state::ConversationMessage],
    prompt_messages: &[MessageRecord],
    answer_brief: &str,
    response_target: &crate::state::ResponseTarget,
) -> String {
    let client = match response_target.client {
        crate::state::ResponseClient::Api => "api",
        crate::state::ResponseClient::Cli => "cli",
        crate::state::ResponseClient::Slack => "slack",
        crate::state::ResponseClient::WhatsApp => "whatsapp",
    };
    let format = match response_target.format {
        crate::state::ResponseFormat::PlainText => "plain_text",
        crate::state::ResponseFormat::Markdown => "markdown",
        crate::state::ResponseFormat::SlackMrkdwn => "slack_mrkdwn",
        crate::state::ResponseFormat::WhatsAppText => "whatsapp_text",
    };
    let format_rules = match response_target.format {
        crate::state::ResponseFormat::PlainText => {
            "- display_text must be plain text only.\n- Avoid Markdown syntax, tables, and fenced code blocks."
        }
        crate::state::ResponseFormat::Markdown => {
            "- display_text may use normal Markdown.\n- Keep formatting concise and readable."
        }
        crate::state::ResponseFormat::SlackMrkdwn => {
            "- display_text must use Slack mrkdwn only.\n- Use only Slack-compatible emphasis, lists, quotes, and code blocks."
        }
        crate::state::ResponseFormat::WhatsAppText => {
            "- display_text must be easy to read in WhatsApp.\n- Prefer short paragraphs and simple lists.\n- Avoid tables and complex Markdown."
        }
    };
    let mut prompt = String::from(
        "You are rendering the final user-facing assistant answer.\n\
Return only valid JSON with this shape:\n\
{\"canonical_text\":\"...\",\"display_text\":\"...\"}\n\
Preserve the facts, caveats, and conclusions from the answer brief.\n\
Do not reveal chain-of-thought or internal reasoning.\n\
canonical_text must be channel-neutral and semantically complete.\n\
display_text must preserve the same meaning while matching the target client format.\n\n",
    );
    prompt.push_str("Response target:\n");
    prompt.push_str(&format!("Client: {client}\nFormat: {format}\n{format_rules}\n\n"));
    prompt.push_str("User message:\n");
    prompt.push_str(user_message);
    prompt.push_str("\n\nConversation history:\n");
    for message in conversation_history {
        prompt.push_str("- ");
        prompt.push_str(match message.role {
            crate::state::ConversationRole::User => "user: ",
            crate::state::ConversationRole::Assistant => "assistant: ",
        });
        prompt.push_str(&message.content);
        prompt.push('\n');
    }
    prompt.push_str("\nCurrent turn trace:\n");
    for message in prompt_messages {
        prompt.push_str("- ");
        prompt.push_str(&format!("{message:?}"));
        prompt.push('\n');
    }
    prompt.push_str("\nAnswer brief:\n");
    prompt.push_str(answer_brief);
    prompt
}

fn usage_delta(step_usage: UsageSummary, extra_usage: UsageSummary) -> UsageSummary {
    let mut total = step_usage;
    total.add_assign(extra_usage);
    total
}

fn validate_limits(request: &RunRequest) -> Result<(), RuntimeError> {
    if request.limits.max_steps_per_turn == 0 {
        return Err(RuntimeError::Validation(
            "max_steps_per_turn must be at least 1".to_owned(),
        ));
    }
    if request.limits.max_mcp_calls_per_step == 0 {
        return Err(RuntimeError::Validation(
            "max_mcp_calls_per_step must be at least 1".to_owned(),
        ));
    }
    if request.limits.max_subagent_steps_per_invocation == 0 {
        return Err(RuntimeError::Validation(
            "max_subagent_steps_per_invocation must be at least 1".to_owned(),
        ));
    }
    if request.limits.max_subagent_mcp_calls_per_invocation == 0 {
        return Err(RuntimeError::Validation(
            "max_subagent_mcp_calls_per_invocation must be at least 1".to_owned(),
        ));
    }
    Ok(())
}

fn remaining_turn_budget(
    turn_start: Instant,
    turn_timeout: Duration,
) -> Result<Duration, RuntimeError> {
    turn_timeout
        .checked_sub(turn_start.elapsed())
        .ok_or_else(|| RuntimeError::Timeout("turn exceeded configured time budget".to_owned()))
}

fn ensure_turn_time_remaining(
    turn_start: Instant,
    turn_timeout: Duration,
) -> Result<(), RuntimeError> {
    let _ = remaining_turn_budget(turn_start, turn_timeout)?;
    Ok(())
}

fn select_servers(
    registry: &McpRegistry,
    enabled_servers: Option<&[ServerName]>,
) -> Result<Vec<ServerName>, RuntimeError> {
    match enabled_servers {
        Some(enabled) => enabled
            .iter()
            .map(|name| {
                registry.get(name.as_str())?;
                Ok(name.clone())
            })
            .collect(),
        None => registry
            .servers
            .iter()
            .map(|server| ServerName::new(server.name.clone()).map_err(RuntimeError::State))
            .collect(),
    }
}

/// Runtime failure modes.
#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("runtime validation failed: {0}")]
    Validation(String),
    #[error("runtime timed out: {0}")]
    Timeout(String),
    #[error("runtime model interaction failed: {0}")]
    Model(#[from] ModelAdapterError),
    #[error("MCP registry failed: {0}")]
    Registry(#[from] ConfigError),
    #[error("MCP client failed: {0}")]
    McpClient(#[from] McpClientError),
    #[error("prompt rendering failed: {0}")]
    Prompt(#[from] PromptRenderError),
    #[error("state validation failed: {0}")]
    State(#[from] crate::state::StateValidationError),
    #[error("sub-agent config failed: {0}")]
    SubagentConfig(#[from] SubagentConfigError),
    #[error("MCP metadata failed: {0}")]
    Metadata(#[from] mcp_metadata::MetadataError),
    #[error("unknown MCP server requested by model: {0}")]
    UnknownServer(String),
    #[error(
        "unknown MCP capability `{capability_id}` ({capability_kind:?}) requested on server `{server_name}`"
    )]
    UnknownCapability {
        server_name: String,
        capability_kind: CapabilityKind,
        capability_id: String,
    },
}
