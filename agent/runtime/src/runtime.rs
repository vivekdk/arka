//! Agent runtime implementation.

use std::{
    collections::{HashMap, HashSet},
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
        ExecutionHandoff, FinalAnswerRenderRequest, FinalAnswerRenderResponse, ModelAdapter,
        ModelAdapterArtifact, ModelAdapterArtifactKind, ModelAdapterDebugSink, ModelAdapterError,
        ModelStepDecision, PlanningOutcome, SubagentDecision, SubagentStepRequest, TurnPhase,
    },
    policy::{ToolPolicyContext, ToolPolicyEngine, ToolPolicyPhase},
    prompt::{
        PromptAssembler, PromptRenderError, TodoPromptContext, TurnPolicyPromptContext,
        load_and_render_system_prompt, render_execution_handoff, render_todo_context,
        render_turn_policy_context,
    },
    state::{
        DelegationTarget, LlmMessageRecord, LocalToolCallMessageRecord,
        LocalToolResultMessageRecord, McpCallMessageRecord, McpCapability, McpCapabilityTarget,
        McpResultMessageRecord, MessageRecord, RunRequest, RuntimeLimits, ServerName,
        StepOutcomeKind, StepRecord, TerminationReason, TurnOutcome, TurnRecord, UsageSummary,
        UserMessageRecord,
    },
    subagent::{ConfiguredSubagent, SubagentConfigError, SubagentRegistry, load_subagent_prompt},
    todo::{
        GENERIC_STARTER_TODO, MANDATORY_TODO_GENERATE_HTML, MANDATORY_TODO_OPEN_HTML, TodoError,
        TodoExecutor, TodoItem, TodoList,
    },
    tools::{LocalToolContext, ToolDescriptor, builtin_local_tool_catalog, execute_local_tool},
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
use serde_json::Value;
use thiserror::Error;
use tokio::time::timeout;
use tracing::{info, warn};

const MAX_TOTAL_SUBAGENT_MCP_ERRORS: u32 = 3;
const MAX_TOTAL_SUBAGENT_LOCAL_TOOL_ERRORS: u32 = 4;
const POSTGRES_SCHEMA_RESOURCE_DISABLED_REASON: &str = "postgres schema resource reads are disabled for postgres-mcp servers; use the query tool with simple SELECT-based discovery instead";

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
        let request = request;
        validate_limits(&request)?;

        let subagent_registry = SubagentRegistry::load_from_path(&request.subagent_registry_path)?;
        let subagent_cards = subagent_registry.enabled_cards();
        let local_tool_catalog = builtin_local_tool_catalog();
        let policy_overlay = ToolPolicyEngine::load_overlay(request.tool_policy_path.as_deref());
        let tool_policy_engine = match policy_overlay {
            Ok(overlay) => ToolPolicyEngine::new(overlay),
            Err(error) => {
                warn!(error = %error, "tool policy overlay failed to load; continuing with defaults");
                ToolPolicyEngine::new(None)
            }
        };
        let main_executor = RuntimeExecutor::main_agent();

        let turn_id = TurnId::new();
        let turn_directory = turn_directory(&request.working_directory, &turn_id);
        fs::create_dir_all(&turn_directory).map_err(|error| {
            RuntimeError::Io(format!(
                "failed to prepare turn workspace `{}`: {error}",
                turn_directory.display()
            ))
        })?;
        let todo_path = todo_file_path(&request.working_directory, &turn_id);
        let html_output_path = deterministic_html_output_path(&request.working_directory, &turn_id);
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
        let mut planning_prompt_messages = vec![user_message.clone()];
        let mut execution_prompt_messages = vec![user_message];
        let mut steps = Vec::new();
        let turn_start = Instant::now();
        let mut phase = TurnPhase::Planning;
        let mut execution_handoff: Option<ExecutionHandoff> = None;

        let capabilities = build_capabilities(&mcp_session.servers);

        for step_number in 1..=request.limits.max_steps_per_turn {
            ensure_turn_time_remaining(turn_start, request.limits.turn_timeout)?;
            reconcile_mandatory_todos_from_turn_trace(
                &todo_path,
                &html_output_path,
                &request.working_directory,
                &all_messages,
            )?;
            reconcile_active_todo_from_turn_trace(&todo_path, &all_messages)?;
            reconcile_pending_mcp_todos_from_turn_trace(&todo_path, &all_messages)?;

            let step_id = StepId::new();
            let step_started_at = SystemTime::now();
            sink.record(RuntimeEvent::StepStarted {
                turn_id: turn_id.clone(),
                step_id: step_id.clone(),
                step_number,
                executor: main_executor.clone(),
                at: step_started_at,
            });

            let turn_policy_context = TurnPolicyPromptContext {
                phase,
                html_output_path: html_output_path.clone(),
                force_todo_file: request.require_todos,
                execution_todo_required: execution_handoff
                    .as_ref()
                    .map(|handoff| handoff.todo_required),
            };
            let system_prompt = load_and_render_system_prompt(
                &request.system_prompt_path,
                phase,
                &request.working_directory,
                &capabilities,
                &subagent_cards,
                &local_tool_catalog,
                &request.response_target,
            )?;
            let current_prompt_messages = match phase {
                TurnPhase::Planning => &planning_prompt_messages,
                TurnPhase::Execution => &execution_prompt_messages,
            };
            let prompt = self.prompt_assembler.build(
                &system_prompt,
                &request.conversation_history,
                &request.recent_session_messages,
                current_prompt_messages,
                &turn_policy_context,
                execution_handoff.as_ref(),
                load_todo_prompt_context(&todo_path)?.as_ref(),
            );
            sink.record(RuntimeEvent::PromptBuilt {
                turn_id: turn_id.clone(),
                step_id: step_id.clone(),
                executor: main_executor.clone(),
                at: SystemTime::now(),
            });
            info!(
                turn_id = %turn_id,
                step_id = %step_id,
                step_number,
                executor = %main_executor.display_name,
                prompt_chars = prompt.rendered.chars().count(),
                prompt_sections = prompt.sections.len(),
                conversation_history_messages = request.conversation_history.len(),
                recent_session_results = request.recent_session_messages.len(),
                current_turn_messages = current_prompt_messages.len(),
                todo_present = load_todo_prompt_context(&todo_path)?.is_some(),
                section_summary = %prompt_section_summary(&prompt),
                "prompt built"
            );

            let generated_step = self
                .generate_validated_step(
                    step_number,
                    phase,
                    prompt.clone(),
                    request.model_config.clone(),
                    local_tool_catalog.clone(),
                    turn_start,
                    request.limits.turn_timeout,
                    sink,
                    &turn_id,
                    &step_id,
                )
                .await;
            let (decision, step_usage) = match generated_step {
                Ok(generated) => generated,
                Err(RuntimeError::Model(ModelAdapterError::InvalidDecision(error))) => {
                    let feedback = MessageRecord::Llm(LlmMessageRecord {
                        message_id: MessageId::new(),
                        timestamp: SystemTime::now(),
                        content: invalid_main_decision_feedback(&error, phase, &subagent_registry),
                    });
                    all_messages.push(feedback.clone());
                    push_prompt_message_for_phase(
                        phase,
                        &mut planning_prompt_messages,
                        &mut execution_prompt_messages,
                        feedback.clone(),
                    );

                    let step = StepRecord {
                        step_id: step_id.clone(),
                        step_number,
                        started_at: step_started_at,
                        ended_at: SystemTime::now(),
                        prompt,
                        decision: None,
                        messages: vec![feedback],
                        outcome: StepOutcomeKind::Continue,
                        usage: UsageSummary::default(),
                    };
                    steps.push(step);
                    sink.record(RuntimeEvent::StepEnded {
                        turn_id: turn_id.clone(),
                        step_id: step_id.clone(),
                        executor: main_executor.clone(),
                        at: SystemTime::now(),
                    });
                    continue;
                }
                Err(error) => return Err(error),
            };
            usage.add_assign(step_usage);
            info!(
                turn_id = %turn_id,
                step_id = %step_id,
                step_number,
                executor = %main_executor.display_name,
                model_decision = %model_step_decision_summary(&decision),
                input_tokens = step_usage.input_tokens,
                cached_tokens = step_usage.cached_tokens,
                output_tokens = step_usage.output_tokens,
                total_tokens = step_usage.total_tokens,
                "model decision received"
            );

            let mut step_messages = Vec::new();
            match decision.clone() {
                ModelStepDecision::PlanningComplete { outcome } => {
                    if phase != TurnPhase::Planning {
                        let feedback = MessageRecord::Llm(LlmMessageRecord {
                            message_id: MessageId::new(),
                            timestamp: SystemTime::now(),
                            content: "Planning is already complete for this turn. Continue execution or return the final answer instead of emitting `planning_complete` again.".to_owned(),
                        });
                        step_messages.push(feedback.clone());
                        all_messages.push(feedback.clone());
                        push_prompt_message_for_phase(
                            phase,
                            &mut planning_prompt_messages,
                            &mut execution_prompt_messages,
                            feedback,
                        );

                        let step = StepRecord {
                            step_id: step_id.clone(),
                            step_number,
                            started_at: step_started_at,
                            ended_at: SystemTime::now(),
                            prompt,
                            decision: Some(decision),
                            messages: step_messages,
                            outcome: StepOutcomeKind::Continue,
                            usage: step_usage,
                        };
                        steps.push(step);
                        sink.record(RuntimeEvent::StepEnded {
                            turn_id: turn_id.clone(),
                            step_id: step_id.clone(),
                            executor: main_executor.clone(),
                            at: SystemTime::now(),
                        });
                        continue;
                    }

                    let Some(outcome) =
                        normalize_planning_outcome(outcome, request.require_todos, &todo_path)?
                    else {
                        let feedback = MessageRecord::Llm(LlmMessageRecord {
                            message_id: MessageId::new(),
                            timestamp: SystemTime::now(),
                            content: "Planning is not complete yet. If execution needs a todo file, `planning_complete` must include concrete ordered `todo_items`. Continue discovery or emit a complete planning outcome.".to_owned(),
                        });
                        step_messages.push(feedback.clone());
                        all_messages.push(feedback.clone());
                        push_prompt_message_for_phase(
                            phase,
                            &mut planning_prompt_messages,
                            &mut execution_prompt_messages,
                            feedback,
                        );

                        let step = StepRecord {
                            step_id: step_id.clone(),
                            step_number,
                            started_at: step_started_at,
                            ended_at: SystemTime::now(),
                            prompt,
                            decision: Some(decision),
                            messages: step_messages,
                            outcome: StepOutcomeKind::Continue,
                            usage: step_usage,
                        };
                        steps.push(step);
                        sink.record(RuntimeEvent::StepEnded {
                            turn_id: turn_id.clone(),
                            step_id: step_id.clone(),
                            executor: main_executor.clone(),
                            at: SystemTime::now(),
                        });
                        continue;
                    };

                    if let Some(feedback_content) =
                        planning_todo_items_feedback(&outcome, request.require_todos, &todo_path)?
                    {
                        let feedback = MessageRecord::Llm(LlmMessageRecord {
                            message_id: MessageId::new(),
                            timestamp: SystemTime::now(),
                            content: feedback_content,
                        });
                        step_messages.push(feedback.clone());
                        all_messages.push(feedback.clone());
                        push_prompt_message_for_phase(
                            phase,
                            &mut planning_prompt_messages,
                            &mut execution_prompt_messages,
                            feedback,
                        );

                        let step = StepRecord {
                            step_id: step_id.clone(),
                            step_number,
                            started_at: step_started_at,
                            ended_at: SystemTime::now(),
                            prompt,
                            decision: Some(decision),
                            messages: step_messages,
                            outcome: StepOutcomeKind::Continue,
                            usage: step_usage,
                        };
                        steps.push(step);
                        sink.record(RuntimeEvent::StepEnded {
                            turn_id: turn_id.clone(),
                            step_id: step_id.clone(),
                            executor: main_executor.clone(),
                            at: SystemTime::now(),
                        });
                        continue;
                    }

                    let handoff =
                        materialize_planning_outcome(&outcome, &todo_path, request.require_todos)?;
                    let transition_message = MessageRecord::Llm(LlmMessageRecord {
                        message_id: MessageId::new(),
                        timestamp: SystemTime::now(),
                        content: format!(
                            "planning complete; transitioning to execution\n{}",
                            render_execution_handoff(&handoff)
                        ),
                    });
                    step_messages.push(transition_message.clone());
                    all_messages.push(transition_message.clone());
                    planning_prompt_messages.push(transition_message);
                    execution_handoff = Some(handoff);
                    phase = TurnPhase::Execution;
                    execution_prompt_messages = vec![all_messages[0].clone()];

                    let step = StepRecord {
                        step_id: step_id.clone(),
                        step_number,
                        started_at: step_started_at,
                        ended_at: SystemTime::now(),
                        prompt,
                        decision: Some(decision),
                        messages: step_messages,
                        outcome: StepOutcomeKind::Continue,
                        usage: step_usage,
                    };
                    steps.push(step);
                    sink.record(RuntimeEvent::StepEnded {
                        turn_id: turn_id.clone(),
                        step_id: step_id.clone(),
                        executor: main_executor.clone(),
                        at: SystemTime::now(),
                    });
                    continue;
                }
                ModelStepDecision::Final { content } => {
                    if phase == TurnPhase::Planning {
                        if let Some(handoff) = infer_execution_handoff_from_legacy_planning(
                            &content,
                            request.require_todos,
                            &todo_path,
                        )? {
                            let transition_message = MessageRecord::Llm(LlmMessageRecord {
                                message_id: MessageId::new(),
                                timestamp: SystemTime::now(),
                                content: format!(
                                    "planning complete; transitioning to execution\n{}",
                                    render_execution_handoff(&handoff)
                                ),
                            });
                            step_messages.push(transition_message.clone());
                            all_messages.push(transition_message.clone());
                            planning_prompt_messages.push(transition_message);
                            execution_handoff = Some(handoff);
                            phase = TurnPhase::Execution;
                            execution_prompt_messages = vec![all_messages[0].clone()];

                            let step = StepRecord {
                                step_id: step_id.clone(),
                                step_number,
                                started_at: step_started_at,
                                ended_at: SystemTime::now(),
                                prompt,
                                decision: Some(decision),
                                messages: step_messages,
                                outcome: StepOutcomeKind::Continue,
                                usage: step_usage,
                            };
                            steps.push(step);
                            sink.record(RuntimeEvent::StepEnded {
                                turn_id: turn_id.clone(),
                                step_id: step_id.clone(),
                                executor: main_executor.clone(),
                                at: SystemTime::now(),
                            });
                            continue;
                        }

                        let rendered = if should_skip_final_answer_render(&content) {
                            FinalAnswerRenderResponse {
                                canonical_text: content.clone(),
                                display_text: content.clone(),
                                usage: UsageSummary::default(),
                            }
                        } else {
                            self.render_final_answer(
                                &request,
                                &planning_prompt_messages,
                                request.model_config.clone(),
                                &content,
                                &turn_policy_context,
                                load_todo_prompt_context(&todo_path)?.as_ref(),
                                sink,
                                &turn_id,
                                &step_id,
                                &main_executor,
                            )
                            .await?
                        };
                        usage.add_assign(rendered.usage);
                        let message = MessageRecord::Llm(LlmMessageRecord {
                            message_id: MessageId::new(),
                            timestamp: SystemTime::now(),
                            content: rendered.canonical_text.clone(),
                        });
                        step_messages.push(message.clone());
                        all_messages.push(message.clone());
                        planning_prompt_messages.push(message);

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
                    if let Some(feedback) = generic_starter_replan_feedback(&todo_path)? {
                        let feedback = MessageRecord::Llm(LlmMessageRecord {
                            message_id: MessageId::new(),
                            timestamp: SystemTime::now(),
                            content: feedback,
                        });
                        step_messages.push(feedback.clone());
                        all_messages.push(feedback);
                        execution_prompt_messages
                            .push(step_messages.last().expect("message just pushed").clone());

                        let step = StepRecord {
                            step_id: step_id.clone(),
                            step_number,
                            started_at: step_started_at,
                            ended_at: SystemTime::now(),
                            prompt,
                            decision: Some(decision),
                            messages: step_messages,
                            outcome: StepOutcomeKind::Continue,
                            usage: step_usage,
                        };
                        steps.push(step);
                        sink.record(RuntimeEvent::StepEnded {
                            turn_id: turn_id.clone(),
                            step_id: step_id.clone(),
                            executor: main_executor.clone(),
                            at: SystemTime::now(),
                        });
                        continue;
                    }
                    if let Some(todo_list) = load_todo_list_if_present(&todo_path)? {
                        if let Some((item_index, item)) = todo_list.next_actionable() {
                            if !allow_blocker_final_for_failed_todo(
                                &todo_list,
                                &content,
                                &all_messages,
                            ) {
                                match auto_advance_summary_todo_from_final_content(
                                    &todo_path,
                                    item_index,
                                    &item.text,
                                    &content,
                                    &html_output_path,
                                )? {
                                    FinalTodoAdvance::AdvancedWithFeedback(feedback_content) => {
                                        let content_message =
                                            MessageRecord::Llm(LlmMessageRecord {
                                                message_id: MessageId::new(),
                                                timestamp: SystemTime::now(),
                                                content: content.clone(),
                                            });
                                        step_messages.push(content_message.clone());
                                        all_messages.push(content_message);
                                        execution_prompt_messages.push(
                                            step_messages
                                                .last()
                                                .expect("message just pushed")
                                                .clone(),
                                        );

                                        let feedback = MessageRecord::Llm(LlmMessageRecord {
                                            message_id: MessageId::new(),
                                            timestamp: SystemTime::now(),
                                            content: feedback_content,
                                        });
                                        step_messages.push(feedback.clone());
                                        all_messages.push(feedback);
                                        execution_prompt_messages.push(
                                            step_messages
                                                .last()
                                                .expect("message just pushed")
                                                .clone(),
                                        );

                                        let step = StepRecord {
                                            step_id: step_id.clone(),
                                            step_number,
                                            started_at: step_started_at,
                                            ended_at: SystemTime::now(),
                                            prompt,
                                            decision: Some(decision),
                                            messages: step_messages,
                                            outcome: StepOutcomeKind::Continue,
                                            usage: step_usage,
                                        };
                                        steps.push(step);
                                        sink.record(RuntimeEvent::StepEnded {
                                            turn_id: turn_id.clone(),
                                            step_id: step_id.clone(),
                                            executor: main_executor.clone(),
                                            at: SystemTime::now(),
                                        });
                                        continue;
                                    }
                                    FinalTodoAdvance::AdvancedAllDone => {}
                                    FinalTodoAdvance::Noop => {
                                        let feedback = MessageRecord::Llm(LlmMessageRecord {
                                            message_id: MessageId::new(),
                                            timestamp: SystemTime::now(),
                                            content: incomplete_todo_final_feedback(
                                                item_index,
                                                item.status.as_str(),
                                                &item.text,
                                                &html_output_path,
                                            ),
                                        });
                                        step_messages.push(feedback.clone());
                                        all_messages.push(feedback);
                                        execution_prompt_messages.push(
                                            step_messages
                                                .last()
                                                .expect("message just pushed")
                                                .clone(),
                                        );

                                        let step = StepRecord {
                                            step_id: step_id.clone(),
                                            step_number,
                                            started_at: step_started_at,
                                            ended_at: SystemTime::now(),
                                            prompt,
                                            decision: Some(decision),
                                            messages: step_messages,
                                            outcome: StepOutcomeKind::Continue,
                                            usage: step_usage,
                                        };
                                        steps.push(step);
                                        sink.record(RuntimeEvent::StepEnded {
                                            turn_id: turn_id.clone(),
                                            step_id: step_id.clone(),
                                            executor: main_executor.clone(),
                                            at: SystemTime::now(),
                                        });
                                        continue;
                                    }
                                }
                            }
                        }
                    }
                    let rendered = if should_skip_final_answer_render(&content) {
                        FinalAnswerRenderResponse {
                            canonical_text: content.clone(),
                            display_text: content.clone(),
                            usage: UsageSummary::default(),
                        }
                    } else {
                        self.render_final_answer(
                            &request,
                            &execution_prompt_messages,
                            request.model_config.clone(),
                            &content,
                            &turn_policy_context,
                            load_todo_prompt_context(&todo_path)?.as_ref(),
                            sink,
                            &turn_id,
                            &step_id,
                            &main_executor,
                        )
                        .await?
                    };
                    usage.add_assign(rendered.usage);
                    let message = MessageRecord::Llm(LlmMessageRecord {
                        message_id: MessageId::new(),
                        timestamp: SystemTime::now(),
                        content: rendered.canonical_text.clone(),
                    });
                    step_messages.push(message.clone());
                    all_messages.push(message);
                    execution_prompt_messages
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
                    let mut delegation_phase = phase;
                    if should_auto_transition_to_execution(
                        phase,
                        request.require_todos,
                        &todo_path,
                        &delegation,
                    ) {
                        let handoff = ExecutionHandoff {
                            todo_required: false,
                            answer_brief: String::new(),
                            summary: "Planning inferred a direct execution path from the requested delegated action.".to_owned(),
                            selected_sources: Vec::new(),
                            key_facts: Vec::new(),
                            execution_strategy: delegation.goal.clone(),
                            risks_and_constraints: Vec::new(),
                            todo_path: None,
                        };
                        let transition_message = MessageRecord::Llm(LlmMessageRecord {
                            message_id: MessageId::new(),
                            timestamp: SystemTime::now(),
                            content: format!(
                                "planning complete; transitioning to execution\n{}",
                                render_execution_handoff(&handoff)
                            ),
                        });
                        step_messages.push(transition_message.clone());
                        all_messages.push(transition_message.clone());
                        planning_prompt_messages.push(transition_message);
                        execution_handoff = Some(handoff);
                        phase = TurnPhase::Execution;
                        delegation_phase = TurnPhase::Execution;
                        execution_prompt_messages = vec![all_messages[0].clone()];
                    }
                    let Some(configured_subagent) =
                        validate_delegation_subagent(&subagent_registry, &delegation)
                    else {
                        let feedback = MessageRecord::Llm(LlmMessageRecord {
                            message_id: MessageId::new(),
                            timestamp: SystemTime::now(),
                            content: invalid_subagent_feedback(&subagent_registry, &delegation),
                        });
                        step_messages.push(feedback.clone());
                        all_messages.push(feedback);
                        push_prompt_message_for_phase(
                            delegation_phase,
                            &mut planning_prompt_messages,
                            &mut execution_prompt_messages,
                            step_messages.last().expect("message just pushed").clone(),
                        );

                        let step = StepRecord {
                            step_id: step_id.clone(),
                            step_number,
                            started_at: step_started_at,
                            ended_at: SystemTime::now(),
                            prompt,
                            decision: Some(decision),
                            messages: step_messages,
                            outcome: StepOutcomeKind::Continue,
                            usage: step_usage,
                        };
                        steps.push(step);
                        sink.record(RuntimeEvent::StepEnded {
                            turn_id: turn_id.clone(),
                            step_id: step_id.clone(),
                            executor: main_executor.clone(),
                            at: SystemTime::now(),
                        });
                        continue;
                    };
                    let resolved_subagent_type = configured_subagent.subagent_type.clone();

                    if delegation_phase == TurnPhase::Execution {
                        if let Some(feedback) = generic_starter_replan_feedback_for_delegation(
                            &todo_path,
                            &delegation,
                            &resolved_subagent_type,
                        )? {
                            let feedback = MessageRecord::Llm(LlmMessageRecord {
                                message_id: MessageId::new(),
                                timestamp: SystemTime::now(),
                                content: feedback,
                            });
                            step_messages.push(feedback.clone());
                            all_messages.push(feedback);
                            execution_prompt_messages
                                .push(step_messages.last().expect("message just pushed").clone());

                            let step = StepRecord {
                                step_id: step_id.clone(),
                                step_number,
                                started_at: step_started_at,
                                ended_at: SystemTime::now(),
                                prompt,
                                decision: Some(decision),
                                messages: step_messages,
                                outcome: StepOutcomeKind::Continue,
                                usage: step_usage,
                            };
                            steps.push(step);
                            sink.record(RuntimeEvent::StepEnded {
                                turn_id: turn_id.clone(),
                                step_id: step_id.clone(),
                                executor: main_executor.clone(),
                                at: SystemTime::now(),
                            });
                            continue;
                        }
                        if let Some(feedback) = concrete_plan_replan_feedback_for_delegation(
                            &todo_path,
                            &delegation,
                            &resolved_subagent_type,
                        )? {
                            let feedback = MessageRecord::Llm(LlmMessageRecord {
                                message_id: MessageId::new(),
                                timestamp: SystemTime::now(),
                                content: feedback,
                            });
                            step_messages.push(feedback.clone());
                            all_messages.push(feedback);
                            execution_prompt_messages
                                .push(step_messages.last().expect("message just pushed").clone());

                            let step = StepRecord {
                                step_id: step_id.clone(),
                                step_number,
                                started_at: step_started_at,
                                ended_at: SystemTime::now(),
                                prompt,
                                decision: Some(decision),
                                messages: step_messages,
                                outcome: StepOutcomeKind::Continue,
                                usage: step_usage,
                            };
                            steps.push(step);
                            sink.record(RuntimeEvent::StepEnded {
                                turn_id: turn_id.clone(),
                                step_id: step_id.clone(),
                                executor: main_executor.clone(),
                                at: SystemTime::now(),
                            });
                            continue;
                        }
                        if let Some(feedback) = repeated_blocked_mcp_delegation_feedback(
                            &all_messages,
                            &delegation,
                            &resolved_subagent_type,
                        ) {
                            let feedback = MessageRecord::Llm(LlmMessageRecord {
                                message_id: MessageId::new(),
                                timestamp: SystemTime::now(),
                                content: feedback,
                            });
                            step_messages.push(feedback.clone());
                            all_messages.push(feedback);
                            execution_prompt_messages
                                .push(step_messages.last().expect("message just pushed").clone());

                            let step = StepRecord {
                                step_id: step_id.clone(),
                                step_number,
                                started_at: step_started_at,
                                ended_at: SystemTime::now(),
                                prompt,
                                decision: Some(decision),
                                messages: step_messages,
                                outcome: StepOutcomeKind::Continue,
                                usage: step_usage,
                            };
                            steps.push(step);
                            sink.record(RuntimeEvent::StepEnded {
                                turn_id: turn_id.clone(),
                                step_id: step_id.clone(),
                                executor: main_executor.clone(),
                                at: SystemTime::now(),
                            });
                            continue;
                        }
                        if let Some(feedback) = mcp_collection_feedback_for_delegation(
                            &todo_path,
                            &delegation,
                            &resolved_subagent_type,
                            !mcp_session.servers.is_empty(),
                        )? {
                            let feedback = MessageRecord::Llm(LlmMessageRecord {
                                message_id: MessageId::new(),
                                timestamp: SystemTime::now(),
                                content: feedback,
                            });
                            step_messages.push(feedback.clone());
                            all_messages.push(feedback);
                            execution_prompt_messages
                                .push(step_messages.last().expect("message just pushed").clone());

                            let step = StepRecord {
                                step_id: step_id.clone(),
                                step_number,
                                started_at: step_started_at,
                                ended_at: SystemTime::now(),
                                prompt,
                                decision: Some(decision),
                                messages: step_messages,
                                outcome: StepOutcomeKind::Continue,
                                usage: step_usage,
                            };
                            steps.push(step);
                            sink.record(RuntimeEvent::StepEnded {
                                turn_id: turn_id.clone(),
                                step_id: step_id.clone(),
                                executor: main_executor.clone(),
                                at: SystemTime::now(),
                            });
                            continue;
                        }
                    }
                    if let Some(feedback) = disabled_mcp_capability_feedback_for_delegation(
                        mcp_session,
                        &delegation,
                        &resolved_subagent_type,
                    )? {
                        let feedback = MessageRecord::Llm(LlmMessageRecord {
                            message_id: MessageId::new(),
                            timestamp: SystemTime::now(),
                            content: feedback,
                        });
                        step_messages.push(feedback.clone());
                        all_messages.push(feedback);
                        push_prompt_message_for_phase(
                            delegation_phase,
                            &mut planning_prompt_messages,
                            &mut execution_prompt_messages,
                            step_messages.last().expect("message just pushed").clone(),
                        );

                        let step = StepRecord {
                            step_id: step_id.clone(),
                            step_number,
                            started_at: step_started_at,
                            ended_at: SystemTime::now(),
                            prompt,
                            decision: Some(decision),
                            messages: step_messages,
                            outcome: StepOutcomeKind::Continue,
                            usage: step_usage,
                        };
                        steps.push(step);
                        sink.record(RuntimeEvent::StepEnded {
                            turn_id: turn_id.clone(),
                            step_id: step_id.clone(),
                            executor: main_executor.clone(),
                            at: SystemTime::now(),
                        });
                        continue;
                    }
                    if delegation_phase == TurnPhase::Execution {
                        if let Some(feedback) = html_todo_feedback_for_delegation(
                            &todo_path,
                            &delegation,
                            &html_output_path,
                        )? {
                            let feedback = MessageRecord::Llm(LlmMessageRecord {
                                message_id: MessageId::new(),
                                timestamp: SystemTime::now(),
                                content: feedback,
                            });
                            step_messages.push(feedback.clone());
                            all_messages.push(feedback);
                            execution_prompt_messages
                                .push(step_messages.last().expect("message just pushed").clone());

                            let step = StepRecord {
                                step_id: step_id.clone(),
                                step_number,
                                started_at: step_started_at,
                                ended_at: SystemTime::now(),
                                prompt,
                                decision: Some(decision),
                                messages: step_messages,
                                outcome: StepOutcomeKind::Continue,
                                usage: step_usage,
                            };
                            steps.push(step);
                            sink.record(RuntimeEvent::StepEnded {
                                turn_id: turn_id.clone(),
                                step_id: step_id.clone(),
                                executor: main_executor.clone(),
                                at: SystemTime::now(),
                            });
                            continue;
                        }
                    }
                    let llm_message = MessageRecord::Llm(LlmMessageRecord {
                        message_id: MessageId::new(),
                        timestamp: SystemTime::now(),
                        content: format!(
                            "delegating to sub-agent `{}` for target `{}`",
                            resolved_subagent_type,
                            delegation.target.summary()
                        ),
                    });
                    step_messages.push(llm_message.clone());
                    all_messages.push(llm_message);
                    push_prompt_message_for_phase(
                        delegation_phase,
                        &mut planning_prompt_messages,
                        &mut execution_prompt_messages,
                        step_messages.last().expect("message just pushed").clone(),
                    );

                    let subagent_call =
                        MessageRecord::SubAgentCall(crate::state::SubAgentCallMessageRecord {
                            message_id: MessageId::new(),
                            timestamp: SystemTime::now(),
                            subagent_type: resolved_subagent_type.clone(),
                            goal: delegation.goal.clone(),
                            target: delegation.target.clone(),
                        });
                    step_messages.push(subagent_call.clone());
                    all_messages.push(subagent_call);
                    push_prompt_message_for_phase(
                        delegation_phase,
                        &mut planning_prompt_messages,
                        &mut execution_prompt_messages,
                        step_messages.last().expect("message just pushed").clone(),
                    );
                    sink.record(RuntimeEvent::HandoffToSubagent {
                        turn_id: turn_id.clone(),
                        step_id: step_id.clone(),
                        executor: main_executor.clone(),
                        at: SystemTime::now(),
                        subagent_type: resolved_subagent_type.clone(),
                        goal: delegation.goal.clone(),
                        target: delegation.target.clone(),
                    });
                    if delegation_phase == TurnPhase::Execution {
                        sync_mcp_todo_before_delegation(
                            &todo_path,
                            &delegation,
                            &resolved_subagent_type,
                        )?;
                    }

                    let (subagent_result_message, maybe_execution) = self
                        .run_subagent(
                            &request,
                            &tool_policy_engine,
                            &subagent_registry,
                            mcp_session,
                            delegation_phase,
                            execution_handoff.as_ref(),
                            &resolved_subagent_type,
                            &delegation.target,
                            &delegation.goal,
                            &all_messages,
                            turn_start,
                            sink,
                            &turn_id,
                            &step_id,
                        )
                        .await?;
                    if let MessageRecord::SubAgentResult(record) = &subagent_result_message {
                        if delegation_phase == TurnPhase::Execution {
                            sync_mcp_todo_after_delegation(&todo_path, record)?;
                        }
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
                        all_messages.push(trace_message.clone());
                    }
                    step_messages.push(subagent_result_message.clone());
                    all_messages.push(subagent_result_message.clone());
                    push_prompt_message_for_phase(
                        delegation_phase,
                        &mut planning_prompt_messages,
                        &mut execution_prompt_messages,
                        subagent_result_message,
                    );

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
        phase: TurnPhase,
        prompt: crate::state::PromptSnapshot,
        model_config: crate::state::ModelConfig,
        registered_tools: Vec<ToolDescriptor>,
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
                    phase,
                    prompt,
                    model_config,
                    registered_tools,
                    tool_mask_plan: crate::policy::ToolMaskPlan::terminal_only(
                        "main agent does not directly execute tools",
                    ),
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
        tool_policy_engine: &ToolPolicyEngine,
        registry: &SubagentRegistry,
        mcp_session: &mut McpSession,
        phase: TurnPhase,
        execution_handoff: Option<&ExecutionHandoff>,
        subagent_type: &str,
        target: &DelegationTarget,
        goal: &str,
        current_turn_messages: &[MessageRecord],
        turn_start: Instant,
        sink: &mut dyn RuntimeDebugSink,
        turn_id: &TurnId,
        step_id: &StepId,
    ) -> Result<(MessageRecord, DelegatedExecutionOutcome), RuntimeError> {
        let subagent_context_messages = request
            .recent_session_messages
            .iter()
            .cloned()
            .chain(current_turn_messages.iter().cloned())
            .collect::<Vec<_>>();
        let configured = registry.get_enabled(subagent_type)?;
        let todo_path = todo_file_path(&request.working_directory, turn_id);
        let html_output_path = deterministic_html_output_path(&request.working_directory, turn_id);
        let turn_policy_context = TurnPolicyPromptContext {
            phase,
            html_output_path: html_output_path.clone(),
            force_todo_file: request.require_todos,
            execution_todo_required: if phase == TurnPhase::Planning {
                None
            } else {
                Some(todo_path.exists())
            },
        };
        let base_prompt = build_subagent_prompt(
            &request.subagent_registry_path,
            configured,
            mcp_session,
            phase,
            target,
            goal,
            &request.user_message,
            &request.working_directory,
            &todo_path,
            &turn_policy_context,
            execution_handoff,
            &subagent_context_messages,
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
        let mut total_mcp_error_count = 0u32;
        let mut total_local_tool_error_count = 0u32;
        let mut mcp_action_counts = HashMap::<String, u32>::new();
        let mut last_tool_mask_plan =
            crate::policy::ToolMaskPlan::terminal_only("no delegated tools evaluated yet");

        for subagent_step_number in 1..=request.limits.max_subagent_steps_per_invocation {
            ensure_turn_time_remaining(turn_start, request.limits.turn_timeout)?;
            let registered_tools = build_subagent_tool_catalog(mcp_session, target);
            let tool_mask_plan = match registered_tools {
                Ok(registered_tools) => {
                    let filtered_tools = filter_subagent_tools_for_phase(
                        phase,
                        configured.subagent_type.as_str(),
                        registered_tools,
                    );
                    let plan = tool_policy_engine.evaluate(
                        &ToolPolicyContext {
                            executor: configured.subagent_type.clone(),
                            response_client: request.response_target.client,
                            phase: if phase == TurnPhase::Planning {
                                ToolPolicyPhase::DelegatedPlanning
                            } else {
                                ToolPolicyPhase::DelegatedExecution
                            },
                            environment: None,
                            working_directory: request.working_directory.clone(),
                        },
                        &filtered_tools,
                    );
                    emit_tool_mask_debug(sink, turn_id, step_id, &executor, &plan, &filtered_tools);
                    (filtered_tools, plan)
                }
                Err(error) => {
                    let plan = crate::policy::ToolMaskPlan::terminal_only(error.to_string());
                    emit_tool_mask_debug(sink, turn_id, step_id, &executor, &plan, &[]);
                    (Vec::new(), plan)
                }
            };
            let registered_tools = tool_mask_plan.0;
            let tool_mask_plan = tool_mask_plan.1;
            last_tool_mask_plan = tool_mask_plan.clone();
            let prompt = build_subagent_loop_prompt(
                &base_prompt,
                &subagent_messages,
                subagent_step_number,
                &request.limits,
            );
            info!(
                turn_id = %turn_id,
                step_id = %step_id,
                subagent_type = %subagent_type,
                subagent_step_number,
                executor = %executor.display_name,
                prompt_chars = prompt.chars().count(),
                base_prompt_chars = base_prompt.chars().count(),
                execution_history_messages = subagent_messages.len(),
                registered_tools = registered_tools.len(),
                allowed_mcp_tools = tool_mask_plan.allowed_mcp_tools.len(),
                allowed_mcp_resources = tool_mask_plan.allowed_mcp_resources.len(),
                allowed_local_tools = tool_mask_plan.allowed_local_tools.len(),
                "subagent prompt built"
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
                        registered_tools: registered_tools.clone(),
                        tool_mask_plan: tool_mask_plan.clone(),
                    },
                    Some(&mut debug_sink),
                ),
            )
            .await
            .map_err(|_| RuntimeError::Timeout("sub-agent step timed out".to_owned()))?;
            let response = match response {
                Ok(response) => response,
                Err(ModelAdapterError::InvalidDecision(error)) => {
                    return Ok((
                        subagent_result_message(
                            subagent_type,
                            "partial",
                            executed_action_count,
                            format!(
                                "sub-agent returned invalid structured output and was stopped early: {error}"
                            ),
                            Some(tool_mask_plan),
                        ),
                        DelegatedExecutionOutcome {
                            usage,
                            trace_messages,
                        },
                    ));
                }
                Err(error) => return Err(RuntimeError::Model(error)),
            };
            usage.add_assign(response.usage);
            sink.record(RuntimeEvent::ModelResponded {
                turn_id: turn_id.clone(),
                step_id: step_id.clone(),
                at: SystemTime::now(),
                latency: started.elapsed(),
                executor: executor.clone(),
                usage: response.usage,
            });
            info!(
                turn_id = %turn_id,
                step_id = %step_id,
                subagent_type = %subagent_type,
                subagent_step_number,
                executor = %executor.display_name,
                decision = %subagent_decision_summary(&response.decision),
                input_tokens = response.usage.input_tokens,
                cached_tokens = response.usage.cached_tokens,
                output_tokens = response.usage.output_tokens,
                total_tokens = response.usage.total_tokens,
                "subagent model decision received"
            );
            match response.decision {
                SubagentDecision::Done { summary } => {
                    return Ok((
                        subagent_result_message(
                            subagent_type,
                            "completed",
                            executed_action_count,
                            summary,
                            Some(tool_mask_plan),
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
                            Some(tool_mask_plan),
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
                            Some(tool_mask_plan),
                        ),
                        DelegatedExecutionOutcome {
                            usage,
                            trace_messages,
                        },
                    ));
                }
                SubagentDecision::McpToolCall {
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
                                Some(tool_mask_plan),
                            ),
                            DelegatedExecutionOutcome {
                                usage,
                                trace_messages,
                            },
                        ));
                    }

                    let call_target = match validate_masked_mcp_call(
                        &mcp_session.servers,
                        &tool_mask_plan,
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
                                    Some(tool_mask_plan),
                                ),
                                DelegatedExecutionOutcome {
                                    usage,
                                    trace_messages,
                                },
                            ));
                        }
                    };

                    let sanitized = match sanitize_mcp_tool_arguments(
                        &registered_tools,
                        &call_target,
                        &arguments,
                    ) {
                        Ok(result) => {
                            emit_mcp_argument_validation_debug(
                                sink,
                                turn_id,
                                step_id,
                                &executor,
                                &call_target,
                                &result,
                                &arguments,
                            );
                            result
                        }
                        Err(error) => {
                            total_mcp_error_count += 1;
                            let validation = SanitizedMcpArguments::rejected(error.clone());
                            emit_mcp_argument_validation_debug(
                                sink,
                                turn_id,
                                step_id,
                                &executor,
                                &call_target,
                                &validation,
                                &arguments,
                            );
                            let validation_message =
                                MessageRecord::McpResult(McpResultMessageRecord {
                                    message_id: MessageId::new(),
                                    timestamp: SystemTime::now(),
                                    target: call_target.clone(),
                                    result_summary: error.clone(),
                                    error: Some(error),
                                });
                            trace_messages.push(validation_message.clone());
                            subagent_messages.push(validation_message);
                            if total_mcp_error_count >= MAX_TOTAL_SUBAGENT_MCP_ERRORS {
                                return Ok((
                                    subagent_result_message(
                                        subagent_type,
                                        "partial",
                                        executed_action_count,
                                        repeated_invalid_mcp_argument_detail(
                                            &call_target,
                                            executed_action_count,
                                            total_mcp_error_count,
                                        ),
                                        Some(tool_mask_plan),
                                    ),
                                    DelegatedExecutionOutcome {
                                        usage,
                                        trace_messages,
                                    },
                                ));
                            }
                            continue;
                        }
                    };
                    let sanitized_arguments = sanitized.arguments.clone().unwrap_or(Value::Null);

                    let action_signature = format!(
                        "tool::{}::{}::{}",
                        call_target.server_name,
                        call_target.capability_id,
                        canonicalize_action_arguments(&sanitized_arguments)
                    );
                    let action_count = mcp_action_counts
                        .entry(action_signature)
                        .and_modify(|count| *count += 1)
                        .or_insert(1);
                    if *action_count > request.limits.max_duplicate_mcp_calls_per_invocation {
                        let duplicate_message = duplicate_mcp_action_feedback_message(
                            &call_target,
                            "tool call",
                            request.limits.max_duplicate_mcp_calls_per_invocation,
                        );
                        total_mcp_error_count += 1;
                        trace_messages.push(duplicate_message.clone());
                        subagent_messages.push(duplicate_message);
                        if total_mcp_error_count >= MAX_TOTAL_SUBAGENT_MCP_ERRORS {
                            return Ok((
                                subagent_result_message(
                                    subagent_type,
                                    "partial",
                                    executed_action_count,
                                    format!(
                                        "repeated duplicate MCP tool calls were blocked on server `{}` capability `{}`; use earlier results or choose a different next action",
                                        call_target.server_name, call_target.capability_id
                                    ),
                                    Some(tool_mask_plan),
                                ),
                                DelegatedExecutionOutcome {
                                    usage,
                                    trace_messages,
                                },
                            ));
                        }
                        continue;
                    }

                    let call_message = MessageRecord::McpCall(McpCallMessageRecord {
                        message_id: MessageId::new(),
                        timestamp: SystemTime::now(),
                        target: call_target.clone(),
                        arguments: sanitized_arguments.clone(),
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
                            "arguments": sanitized_arguments.clone(),
                        }),
                    });
                    let started = Instant::now();
                    let result = execute_tool_call(
                        &mut mcp_session.servers,
                        &call_target.server_name,
                        &tool_name,
                        sanitized_arguments,
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
                    if was_error {
                        total_mcp_error_count += 1;
                        if total_mcp_error_count >= MAX_TOTAL_SUBAGENT_MCP_ERRORS {
                            return Ok((
                                subagent_result_message(
                                    subagent_type,
                                    "partial",
                                    executed_action_count,
                                    repeated_mcp_failure_detail(
                                        &call_target,
                                        executed_action_count,
                                        total_mcp_error_count,
                                        &captured,
                                    ),
                                    Some(tool_mask_plan),
                                ),
                                DelegatedExecutionOutcome {
                                    usage,
                                    trace_messages,
                                },
                            ));
                        }
                    }
                }
                SubagentDecision::McpResourceRead {
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
                                Some(tool_mask_plan),
                            ),
                            DelegatedExecutionOutcome {
                                usage,
                                trace_messages,
                            },
                        ));
                    }

                    let call_target = match validate_masked_mcp_call(
                        &mcp_session.servers,
                        &tool_mask_plan,
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
                                    Some(tool_mask_plan),
                                ),
                                DelegatedExecutionOutcome {
                                    usage,
                                    trace_messages,
                                },
                            ));
                        }
                    };

                    let action_signature = format!(
                        "resource::{}::{}",
                        call_target.server_name, call_target.capability_id
                    );
                    let action_count = mcp_action_counts
                        .entry(action_signature)
                        .and_modify(|count| *count += 1)
                        .or_insert(1);
                    if *action_count > request.limits.max_duplicate_mcp_calls_per_invocation {
                        let duplicate_message = duplicate_mcp_action_feedback_message(
                            &call_target,
                            "resource read",
                            request.limits.max_duplicate_mcp_calls_per_invocation,
                        );
                        total_mcp_error_count += 1;
                        trace_messages.push(duplicate_message.clone());
                        subagent_messages.push(duplicate_message);
                        if total_mcp_error_count >= MAX_TOTAL_SUBAGENT_MCP_ERRORS {
                            return Ok((
                                subagent_result_message(
                                    subagent_type,
                                    "partial",
                                    executed_action_count,
                                    format!(
                                        "repeated duplicate MCP resource reads were blocked on server `{}` capability `{}`; use earlier results or choose a different next action",
                                        call_target.server_name, call_target.capability_id
                                    ),
                                    Some(tool_mask_plan),
                                ),
                                DelegatedExecutionOutcome {
                                    usage,
                                    trace_messages,
                                },
                            ));
                        }
                        continue;
                    }

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
                    if was_error {
                        total_mcp_error_count += 1;
                        if total_mcp_error_count >= MAX_TOTAL_SUBAGENT_MCP_ERRORS {
                            return Ok((
                                subagent_result_message(
                                    subagent_type,
                                    "partial",
                                    executed_action_count,
                                    repeated_mcp_failure_detail(
                                        &call_target,
                                        executed_action_count,
                                        total_mcp_error_count,
                                        &captured,
                                    ),
                                    Some(tool_mask_plan),
                                ),
                                DelegatedExecutionOutcome {
                                    usage,
                                    trace_messages,
                                },
                            ));
                        }
                    }
                }
                SubagentDecision::LocalToolCall {
                    tool_name,
                    arguments,
                } => {
                    if let Some(detail) =
                        tool_executor_mcp_todo_partial_feedback(todo_path.as_path(), &tool_name)?
                    {
                        return Ok((
                            subagent_result_message(
                                subagent_type,
                                "partial",
                                executed_action_count,
                                detail,
                                Some(tool_mask_plan),
                            ),
                            DelegatedExecutionOutcome {
                                usage,
                                trace_messages,
                            },
                        ));
                    }
                    if executed_action_count >= request.limits.max_subagent_mcp_calls_per_invocation
                    {
                        return Ok((
                            subagent_budget_result(
                                subagent_type,
                                executed_action_count,
                                "sub-agent exhausted its action budget".to_owned(),
                                Some(tool_mask_plan),
                            ),
                            DelegatedExecutionOutcome {
                                usage,
                                trace_messages,
                            },
                        ));
                    }

                    if !tool_mask_plan
                        .allowed_local_tools
                        .iter()
                        .any(|candidate| candidate == tool_name.as_str())
                    {
                        return Ok((
                            subagent_policy_result(
                                subagent_type,
                                executed_action_count,
                                format!("local tool `{}` is not allowed in this step", tool_name),
                                Some(tool_mask_plan),
                            ),
                            DelegatedExecutionOutcome {
                                usage,
                                trace_messages,
                            },
                        ));
                    }

                    let call_message = MessageRecord::LocalToolCall(LocalToolCallMessageRecord {
                        message_id: MessageId::new(),
                        timestamp: SystemTime::now(),
                        tool_name: tool_name.clone(),
                        arguments: arguments.clone(),
                    });
                    sink.record(RuntimeEvent::LocalToolCalled {
                        turn_id: turn_id.clone(),
                        step_id: step_id.clone(),
                        tool_name: tool_name.to_string(),
                        executor: executor.clone(),
                        at: SystemTime::now(),
                    });
                    sink.record_raw_artifact(RuntimeRawArtifact {
                        turn_id: turn_id.clone(),
                        step_id: Some(step_id.clone()),
                        occurred_at: SystemTime::now(),
                        kind: RuntimeRawArtifactKind::LocalToolRequest,
                        source: "local_tool_runtime".to_owned(),
                        executor: executor.clone(),
                        summary: Some(format!("local tool call {}", tool_name)),
                        payload: serde_json::json!({
                            "tool_name": tool_name.to_string(),
                            "arguments": arguments.clone(),
                        }),
                    });
                    let started = Instant::now();
                    let remaining = remaining_turn_budget(turn_start, request.limits.turn_timeout)?;
                    let result = execute_local_tool(
                        &tool_name,
                        &arguments,
                        &LocalToolContext {
                            working_directory: request.working_directory.clone(),
                            turn_directory: turn_directory(&request.working_directory, turn_id),
                            todo_path: todo_file_path(&request.working_directory, turn_id),
                            html_output_path: deterministic_html_output_path(
                                &request.working_directory,
                                turn_id,
                            ),
                        },
                        remaining,
                    )
                    .await;
                    let (result_message, was_error, response_payload, result_summary, error) =
                        match result {
                            Ok(result) => {
                                let was_error =
                                    result.status != "ok" || result.error.as_ref().is_some();
                                (
                                    MessageRecord::LocalToolResult(LocalToolResultMessageRecord {
                                        message_id: MessageId::new(),
                                        timestamp: SystemTime::now(),
                                        tool_name: tool_name.clone(),
                                        status: result.status.clone(),
                                        result_summary: result.summary.clone(),
                                        error: result.error.clone(),
                                    }),
                                    was_error,
                                    serde_json::to_value(&result).unwrap_or_else(
                                        |serialization_error| {
                                            serde_json::json!({
                                                "serialization_error": serialization_error.to_string(),
                                            })
                                        },
                                    ),
                                    result.summary,
                                    result.error,
                                )
                            }
                            Err(error) => (
                                MessageRecord::LocalToolResult(LocalToolResultMessageRecord {
                                    message_id: MessageId::new(),
                                    timestamp: SystemTime::now(),
                                    tool_name: tool_name.clone(),
                                    status: "error".to_owned(),
                                    result_summary: error.to_string(),
                                    error: Some(error.to_string()),
                                }),
                                true,
                                serde_json::json!({
                                    "runtime_error": error.to_string(),
                                }),
                                error.to_string(),
                                Some(error.to_string()),
                            ),
                        };
                    sink.record_raw_artifact(RuntimeRawArtifact {
                        turn_id: turn_id.clone(),
                        step_id: Some(step_id.clone()),
                        occurred_at: SystemTime::now(),
                        kind: if was_error {
                            RuntimeRawArtifactKind::LocalToolError
                        } else {
                            RuntimeRawArtifactKind::LocalToolResponse
                        },
                        source: "local_tool_runtime".to_owned(),
                        executor: executor.clone(),
                        summary: Some(format!(
                            "{} local tool {}",
                            if was_error { "error" } else { "response" },
                            tool_name
                        )),
                        payload: response_payload.clone(),
                    });
                    sink.record(RuntimeEvent::LocalToolResponded {
                        turn_id: turn_id.clone(),
                        step_id: step_id.clone(),
                        tool_name: tool_name.to_string(),
                        executor: executor.clone(),
                        at: SystemTime::now(),
                        latency: started.elapsed(),
                        was_error,
                        result_summary: result_summary.clone(),
                        error: error.clone(),
                        response_payload: response_payload.clone(),
                    });

                    executed_action_count += 1;
                    trace_messages.push(call_message.clone());
                    trace_messages.push(result_message.clone());
                    subagent_messages.push(call_message);
                    subagent_messages.push(result_message);
                    if was_error {
                        if let Some(detail) = immediate_partial_detail_for_local_tool_error(
                            &tool_name,
                            &result_summary,
                        ) {
                            return Ok((
                                subagent_result_message(
                                    subagent_type,
                                    "partial",
                                    executed_action_count,
                                    detail,
                                    Some(tool_mask_plan),
                                ),
                                DelegatedExecutionOutcome {
                                    usage,
                                    trace_messages,
                                },
                            ));
                        }
                        total_local_tool_error_count += 1;
                        if total_local_tool_error_count >= MAX_TOTAL_SUBAGENT_LOCAL_TOOL_ERRORS {
                            return Ok((
                                subagent_result_message(
                                    subagent_type,
                                    "partial",
                                    executed_action_count,
                                    repeated_local_tool_failure_detail(
                                        &tool_name,
                                        executed_action_count,
                                        total_local_tool_error_count,
                                        &result_summary,
                                    ),
                                    Some(tool_mask_plan),
                                ),
                                DelegatedExecutionOutcome {
                                    usage,
                                    trace_messages,
                                },
                            ));
                        }
                    }
                }
            }
        }

        Ok((
            subagent_budget_result(
                subagent_type,
                executed_action_count,
                "sub-agent exhausted its reasoning budget before reaching a terminal result"
                    .to_owned(),
                Some(last_tool_mask_plan),
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
        turn_policy_context: &TurnPolicyPromptContext,
        todo_context: Option<&TodoPromptContext>,
        sink: &mut dyn RuntimeDebugSink,
        turn_id: &TurnId,
        step_id: &StepId,
        executor: &RuntimeExecutor,
    ) -> Result<FinalAnswerRenderResponse, RuntimeError> {
        sink.record(RuntimeEvent::ModelCalled {
            turn_id: turn_id.clone(),
            step_id: step_id.clone(),
            executor: executor.clone(),
            at: SystemTime::now(),
        });
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
            turn_policy_context,
            todo_context,
        );
        let started = Instant::now();
        let mut debug_sink = BoundModelDebugSink::new(sink, turn_id, step_id, executor.clone());
        let result = self
            .model_adapter
            .render_final_answer(
                FinalAnswerRenderRequest {
                    model_config,
                    prompt,
                    answer_brief: answer_brief.to_owned(),
                    response_target: request.response_target.clone(),
                },
                None,
                Some(&mut debug_sink),
            )
            .await
            .map_err(RuntimeError::Model);
        match result {
            Ok(response) => {
                sink.record(RuntimeEvent::AnswerTextDelta {
                    turn_id: turn_id.clone(),
                    step_id: step_id.clone(),
                    executor: executor.clone(),
                    at: SystemTime::now(),
                    delta: response.display_text.clone(),
                });
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

fn push_prompt_message_for_phase(
    phase: TurnPhase,
    planning_prompt_messages: &mut Vec<MessageRecord>,
    execution_prompt_messages: &mut Vec<MessageRecord>,
    message: MessageRecord,
) {
    match phase {
        TurnPhase::Planning => planning_prompt_messages.push(message),
        TurnPhase::Execution => execution_prompt_messages.push(message),
    }
}

fn normalize_planning_outcome(
    mut outcome: PlanningOutcome,
    force_todo_file: bool,
    todo_path: &Path,
) -> Result<Option<PlanningOutcome>, RuntimeError> {
    outcome.planning_complete = true;
    outcome.todo_required |= force_todo_file || todo_path.exists();
    if outcome.planning_summary.trim().is_empty() {
        outcome.planning_summary = if outcome.todo_required {
            "Planning identified a concrete execution workflow and requires a todo file.".to_owned()
        } else {
            "Planning determined this turn can execute without a todo file.".to_owned()
        };
    }
    if outcome.execution_strategy.trim().is_empty() {
        outcome.execution_strategy = outcome.planning_summary.clone();
    }
    if outcome.todo_required && !todo_path.exists() && outcome.todo_items.is_empty() {
        return Ok(None);
    }
    Ok(Some(outcome))
}

fn planning_todo_items_feedback(
    outcome: &PlanningOutcome,
    force_todo_file: bool,
    todo_path: &Path,
) -> Result<Option<String>, RuntimeError> {
    let todo_required = outcome.todo_required || force_todo_file || todo_path.exists();
    if !todo_required || todo_path.exists() {
        return Ok(None);
    }
    match TodoList::initialize(&outcome.todo_items) {
        Ok(_) => Ok(None),
        Err(TodoError::Validation(message)) if message.contains("executor hint") => {
            Ok(Some(format!(
                "The `planning_complete.todo_items` are invalid: {message}. Re-emit `planning_complete` with every todo item prefixed by exactly one semantic executor hint: `[mcp-executor]` for MCP/database/API-backed discovery and queries, `[tool-executor]` for local scripts, charts, HTML generation, and path printing, or `[main-agent]` for synthesis/final-answer steps. Do not guess from keywords at runtime; choose the executor from the actual work location."
            )))
        }
        Err(error) => Ok(Some(format!(
            "The `planning_complete.todo_items` are invalid: {error}. Re-emit `planning_complete` with a valid concrete ordered todo plan."
        ))),
    }
}

fn materialize_planning_outcome(
    outcome: &PlanningOutcome,
    todo_path: &Path,
    force_todo_file: bool,
) -> Result<ExecutionHandoff, RuntimeError> {
    let todo_required = outcome.todo_required || force_todo_file || todo_path.exists();
    if todo_required && !todo_path.exists() {
        TodoList::initialize(&outcome.todo_items)?.save_to_path(todo_path)?;
    }
    let mut normalized = outcome.clone();
    normalized.todo_required = todo_required;
    Ok(normalized.into_execution_handoff(todo_required.then(|| todo_path.to_path_buf())))
}

fn infer_execution_handoff_from_legacy_planning(
    summary: &str,
    force_todo_file: bool,
    todo_path: &Path,
) -> Result<Option<ExecutionHandoff>, RuntimeError> {
    let todo_required = force_todo_file || todo_path.exists();
    if force_todo_file && !todo_path.exists() {
        return Ok(None);
    }
    if todo_required {
        let todo_list = TodoList::load_from_path(todo_path)?;
        let todo_items = todo_list.items.into_iter().map(|item| item.text).collect();
        let outcome = PlanningOutcome {
            planning_complete: true,
            answer_brief: String::new(),
            todo_required: true,
            planning_summary: summary.to_owned(),
            selected_sources: Vec::new(),
            discovered_facts: Vec::new(),
            execution_strategy: summary.to_owned(),
            todo_items,
            risks_and_constraints: Vec::new(),
        };
        return Ok(Some(materialize_planning_outcome(
            &outcome,
            todo_path,
            force_todo_file,
        )?));
    }
    Ok(None)
}

fn should_auto_transition_to_execution(
    phase: TurnPhase,
    force_todo_file: bool,
    todo_path: &Path,
    delegation: &crate::model::SubagentDelegationRequest,
) -> bool {
    phase == TurnPhase::Planning
        && !force_todo_file
        && !todo_path.exists()
        && !is_todo_planning_goal(&delegation.goal)
}

fn filter_subagent_tools_for_phase(
    phase: TurnPhase,
    subagent_type: &str,
    tools: Vec<ToolDescriptor>,
) -> Vec<ToolDescriptor> {
    if phase != TurnPhase::Planning || subagent_type != "tool-executor" {
        return tools;
    }
    tools
        .into_iter()
        .filter(|tool| matches!(tool.name.as_str(), "glob" | "read_file" | "write_todos"))
        .collect()
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

fn turn_directory(working_directory: &Path, turn_id: &TurnId) -> PathBuf {
    working_directory.join(turn_id.to_string())
}

fn todo_file_path(working_directory: &Path, turn_id: &TurnId) -> PathBuf {
    turn_directory(working_directory, turn_id).join("todos.txt")
}

fn deterministic_html_output_path(working_directory: &Path, turn_id: &TurnId) -> PathBuf {
    working_directory
        .join("outputs")
        .join(format!("{}-report.html", turn_id))
}

fn load_todo_prompt_context(todo_path: &Path) -> Result<Option<TodoPromptContext>, RuntimeError> {
    let Some(todo_list) = load_todo_list_if_present(todo_path)? else {
        return Ok(None);
    };
    Ok(Some(TodoPromptContext {
        todo_path: todo_path.to_path_buf(),
        todo_contents: todo_list.render(),
        next_actionable: todo_list.next_actionable().map(|(index, item)| {
            if let Some(executor_hint) = item.executor_hint {
                format!(
                    "{index}. [{}] [{}] {}",
                    item.status.as_str(),
                    executor_hint,
                    item.text
                )
            } else {
                format!("{index}. [{}] {}", item.status.as_str(), item.text)
            }
        }),
    }))
}

fn load_todo_list_if_present(todo_path: &Path) -> Result<Option<TodoList>, RuntimeError> {
    if !todo_path.exists() {
        return Ok(None);
    }
    Ok(Some(TodoList::load_from_path(todo_path)?))
}

fn generic_starter_replan_feedback(todo_path: &Path) -> Result<Option<String>, RuntimeError> {
    if !todo_starts_with_generic_scaffold(todo_path)? {
        return Ok(None);
    }
    Ok(Some(
        "The current todo plan is still the generic starter scaffold. Before substantive work or a final answer, delegate `tool-executor` with local tools scope and use `write_todos` `replan_pending_suffix` to replace it with concrete ordered todos that match this turn. Prefix every new todo with `[mcp-executor]`, `[tool-executor]`, or `[main-agent]`. That delegation is planning-only: once the todo plan is rewritten, return control immediately instead of starting execution.".to_owned(),
    ))
}

fn generic_starter_replan_feedback_for_delegation(
    todo_path: &Path,
    delegation: &crate::model::SubagentDelegationRequest,
    resolved_subagent_type: &str,
) -> Result<Option<String>, RuntimeError> {
    if !todo_starts_with_generic_scaffold(todo_path)? {
        return Ok(None);
    }
    if resolved_subagent_type == "tool-executor"
        && matches!(delegation.target, DelegationTarget::LocalToolsScope(_))
        && is_todo_planning_goal(&delegation.goal)
    {
        return Ok(None);
    }
    generic_starter_replan_feedback(todo_path)
}

fn concrete_plan_replan_feedback_for_delegation(
    todo_path: &Path,
    delegation: &crate::model::SubagentDelegationRequest,
    resolved_subagent_type: &str,
) -> Result<Option<String>, RuntimeError> {
    if resolved_subagent_type != "tool-executor"
        || !matches!(delegation.target, DelegationTarget::LocalToolsScope(_))
        || !is_todo_planning_goal(&delegation.goal)
    {
        return Ok(None);
    }
    let Some(todo_list) = load_todo_list_if_present(todo_path)? else {
        return Ok(None);
    };
    if matches!(
        todo_list.next_actionable(),
        Some((_, item)) if item.text == GENERIC_STARTER_TODO
    ) || todo_list
        .items
        .iter()
        .any(|item| item.status == crate::todo::TodoStatus::Failed)
    {
        return Ok(None);
    }
    let Some((item_index, item)) = todo_list.next_actionable() else {
        return Ok(None);
    };
    Ok(Some(format!(
        "A concrete todo plan already exists and execution is underway. Do not replan the todo list again yet. Continue the current todo instead: {item_index}. [{}] {}. Only use planning-only replanning again if the starter scaffold is still present or a todo has failed.",
        item.status.as_str(),
        item.text
    )))
}

fn is_todo_planning_goal(goal: &str) -> bool {
    let goal = goal.to_lowercase();
    let mentions_planning =
        goal.contains("todo") || goal.contains("replan") || goal.contains("plan");
    let mentions_execution = [
        "execute", "complete", "finish", "continue", "run", "perform", "workflow",
    ]
    .iter()
    .any(|term| goal.contains(term));
    mentions_planning && !goal.contains("html") && !goal.contains("browser") && !mentions_execution
}

fn mcp_collection_feedback_for_delegation(
    todo_path: &Path,
    delegation: &crate::model::SubagentDelegationRequest,
    resolved_subagent_type: &str,
    mcp_available: bool,
) -> Result<Option<String>, RuntimeError> {
    if !mcp_available
        || resolved_subagent_type != "tool-executor"
        || !matches!(delegation.target, DelegationTarget::LocalToolsScope(_))
    {
        return Ok(None);
    }
    let Some(todo_list) = load_todo_list_if_present(todo_path)? else {
        return Ok(None);
    };
    let Some((item_index, item)) = todo_list.next_actionable() else {
        return Ok(None);
    };
    if todo_item_prefers_mcp_executor(item) {
        return Ok(Some(format!(
            "The current next actionable todo is MCP-style data discovery or collection work and should not be delegated to `tool-executor`: {item_index}. [{}] {}. Delegate `mcp-executor` with server scope to inspect/query the data first, then use `tool-executor` later for local scripts, HTML generation, and printing the report path.",
            item.status.as_str(),
            item.text
        )));
    }
    Ok(None)
}

fn tool_executor_mcp_todo_partial_feedback(
    todo_path: &Path,
    tool_name: &crate::state::LocalToolName,
) -> Result<Option<String>, RuntimeError> {
    if tool_name.as_str() == "write_todos" {
        return Ok(None);
    }
    let Some(todo_list) = load_todo_list_if_present(todo_path)? else {
        return Ok(None);
    };
    let Some((item_index, item)) = todo_list.next_actionable() else {
        return Ok(None);
    };
    if todo_item_prefers_mcp_executor(item) {
        return Ok(Some(format!(
            "The next actionable todo belongs to MCP-backed data discovery or collection, not local workspace execution: {item_index}. [{}] {}. Return control so the main agent can delegate `mcp-executor` instead of continuing with local tools.",
            item.status.as_str(),
            item.text
        )));
    }
    Ok(None)
}

fn disabled_mcp_capability_feedback_for_delegation(
    mcp_session: &McpSession,
    delegation: &crate::model::SubagentDelegationRequest,
    resolved_subagent_type: &str,
) -> Result<Option<String>, RuntimeError> {
    if resolved_subagent_type != "mcp-executor" {
        return Ok(None);
    }
    let DelegationTarget::McpCapability(target) = &delegation.target else {
        return Ok(None);
    };
    let server = mcp_session
        .servers
        .get(&target.server_name)
        .ok_or_else(|| RuntimeError::UnknownServer(target.server_name.to_string()))?;
    let Some(reason) =
        disabled_mcp_capability_reason(server, &target.capability_kind, &target.capability_id)
    else {
        return Ok(None);
    };

    let scope_hint = match target.capability_kind {
        CapabilityKind::Resource => format!(
            "Do not delegate `mcp-executor` to this resource again. Delegate `mcp-executor` with `mcp_server_scope` for server `{}` instead so it can use the allowed tools on that server.",
            target.server_name
        ),
        CapabilityKind::Tool => format!(
            "Do not delegate `mcp-executor` to this capability again. Choose a different allowed MCP target on server `{}`.",
            target.server_name
        ),
    };
    let server_guidance = render_mcp_server_guidance(server)
        .replace('\n', " ")
        .trim()
        .to_owned();

    Ok(Some(format!(
        "The selected MCP capability cannot be used: {reason}. {scope_hint} {server_guidance}"
    )))
}

fn todo_item_prefers_mcp_executor(item: &TodoItem) -> bool {
    item.executor_hint == Some(TodoExecutor::McpExecutor)
}

fn sync_mcp_todo_before_delegation(
    todo_path: &Path,
    _delegation: &crate::model::SubagentDelegationRequest,
    resolved_subagent_type: &str,
) -> Result<(), RuntimeError> {
    if resolved_subagent_type != "mcp-executor" {
        return Ok(());
    }
    let Some(mut todo_list) = load_todo_list_if_present(todo_path)? else {
        return Ok(());
    };
    let Some((item_index, item)) = todo_list.next_actionable() else {
        return Ok(());
    };
    if matches!(
        item.status,
        crate::todo::TodoStatus::Pending | crate::todo::TodoStatus::Failed
    ) && todo_item_prefers_mcp_executor(item)
    {
        todo_list.set_status(item_index, crate::todo::TodoStatus::InProgress)?;
        todo_list.save_to_path(todo_path)?;
    }
    Ok(())
}

fn sync_mcp_todo_after_delegation(
    todo_path: &Path,
    record: &crate::state::SubAgentResultMessageRecord,
) -> Result<(), RuntimeError> {
    if record.subagent_type != "mcp-executor" {
        return Ok(());
    }
    let Some(mut todo_list) = load_todo_list_if_present(todo_path)? else {
        return Ok(());
    };
    let Some((item_index, item)) = todo_list.next_actionable() else {
        return Ok(());
    };
    if item.status != crate::todo::TodoStatus::InProgress || !todo_item_prefers_mcp_executor(item) {
        return Ok(());
    }
    match record.status.as_str() {
        "completed" => {
            todo_list.set_status(item_index, crate::todo::TodoStatus::Completed)?;
            todo_list.save_to_path(todo_path)?;
        }
        "partial" => {
            if mcp_partial_indicates_blocking_failure(&record.detail) {
                todo_list.set_status(item_index, crate::todo::TodoStatus::Failed)?;
                todo_list.save_to_path(todo_path)?;
            } else if record.executed_action_count > 0
                && todo_can_complete_from_partial_mcp_collection(&item.text)
            {
                todo_list.set_status(item_index, crate::todo::TodoStatus::Completed)?;
                todo_list.save_to_path(todo_path)?;
            }
        }
        "cannot_execute" => {
            todo_list.set_status(item_index, crate::todo::TodoStatus::Failed)?;
            todo_list.save_to_path(todo_path)?;
        }
        _ => {}
    }
    Ok(())
}

fn repeated_blocked_mcp_delegation_feedback(
    messages: &[MessageRecord],
    delegation: &crate::model::SubagentDelegationRequest,
    resolved_subagent_type: &str,
) -> Option<String> {
    if resolved_subagent_type != "mcp-executor" {
        return None;
    }
    let mut last_result: Option<&crate::state::SubAgentResultMessageRecord> = None;
    let mut last_call: Option<&crate::state::SubAgentCallMessageRecord> = None;
    for message in messages.iter().rev() {
        match message {
            MessageRecord::SubAgentResult(record)
                if last_result.is_none() && record.subagent_type == "mcp-executor" =>
            {
                last_result = Some(record);
            }
            MessageRecord::SubAgentCall(record)
                if last_result.is_some()
                    && last_call.is_none()
                    && record.subagent_type == "mcp-executor" =>
            {
                last_call = Some(record);
                break;
            }
            _ => {}
        }
    }
    let (last_result, last_call) = match (last_result, last_call) {
        (Some(result), Some(call)) => (result, call),
        _ => return None,
    };
    if !matches!(last_result.status.as_str(), "partial" | "cannot_execute")
        || !mcp_partial_indicates_blocking_failure(&last_result.detail)
        || last_call.target.summary() != delegation.target.summary()
        || !goals_are_materially_similar(&last_call.goal, &delegation.goal)
    {
        return None;
    }
    Some(format!(
        "The previous `mcp-executor` attempt for target `{}` already stopped because this MCP path is blocked: {} Do not delegate the same blocked MCP path again in this turn. Either replan around the failed todo if a todo file is present, switch to a materially different next step, or return a blocker summary with the confirmed findings gathered so far.",
        delegation.target.summary(),
        truncate_for_log(&last_result.detail, 220)
    ))
}

fn validate_delegation_subagent<'a>(
    registry: &'a SubagentRegistry,
    delegation: &crate::model::SubagentDelegationRequest,
) -> Option<&'a ConfiguredSubagent> {
    registry.get_enabled(delegation.subagent_type.trim()).ok()
}

fn invalid_main_decision_feedback(
    error: &str,
    phase: TurnPhase,
    registry: &SubagentRegistry,
) -> String {
    let valid_subagents = registry
        .enabled_cards()
        .into_iter()
        .map(|card| format!("`{}`", card.subagent_type))
        .collect::<Vec<_>>();
    let valid_subagents = if valid_subagents.is_empty() {
        "No enabled sub-agents are configured.".to_owned()
    } else {
        format!(
            "Valid sub-agents for this turn are: {}.",
            valid_subagents.join(", ")
        )
    };
    let phase_options = match phase {
        TurnPhase::Planning => {
            "In planning, emit `planning_complete`, `final`, or a valid `delegate_subagent`."
        }
        TurnPhase::Execution => "In execution, emit `final` or a valid `delegate_subagent`.",
    };
    format!(
        "The previous structured decision was invalid: {error}. Re-emit exactly one valid structured decision for the current phase. {phase_options} If you emit `delegate_subagent`, include both `subagent_type` and `target`. {valid_subagents} Use target `{{\"kind\":\"local_tools_scope\",\"value\":{{\"scope\":\"workspace\"}}}}` for local tools, scripts, files, reports, or html-path-print work. Use target `{{\"kind\":\"mcp_server_scope\",\"value\":{{\"server_name\":\"<server>\"}}}}` or `{{\"kind\":\"mcp_capability\",\"value\":{{\"server_name\":\"<server>\",\"capability_kind\":\"tool\",\"capability_id\":\"<tool>\"}}}}` for MCP-backed data discovery and queries."
    )
}

fn invalid_subagent_feedback(
    registry: &SubagentRegistry,
    delegation: &crate::model::SubagentDelegationRequest,
) -> String {
    let requested = delegation.subagent_type.trim();
    let valid_subagents = registry
        .enabled_cards()
        .into_iter()
        .map(|card| format!("`{}`", card.subagent_type))
        .collect::<Vec<_>>();
    let valid_subagents = if valid_subagents.is_empty() {
        "No enabled sub-agents are configured.".to_owned()
    } else {
        format!(
            "Valid sub-agents for this turn are: {}.",
            valid_subagents.join(", ")
        )
    };
    let target_hint = match delegation.target {
        DelegationTarget::LocalToolsScope(_) => {
            "For local workspace inspection, Python scripts, report generation, and html-path-print work, use `tool-executor`."
        }
        DelegationTarget::McpCapability(_) | DelegationTarget::McpServerScope(_) => {
            "For MCP discovery, schema inspection, counts, ranges, and data collection, use `mcp-executor`."
        }
    };
    if requested.is_empty() {
        format!(
            "The `delegate_subagent` decision is invalid because `type` is blank. Re-emit `delegate_subagent` with a configured sub-agent type. {valid_subagents} {target_hint}"
        )
    } else {
        format!(
            "The requested sub-agent `{requested}` is not configured or is disabled. Re-emit `delegate_subagent` with one of the configured sub-agent types. {valid_subagents} {target_hint}"
        )
    }
}

fn mcp_partial_indicates_blocking_failure(detail: &str) -> bool {
    let detail = detail.to_lowercase();
    detail.contains("repeated mcp failures forced an early stop")
        || detail.contains("repeated invalid mcp tool arguments forced an early stop")
        || detail.contains("sub-agent exhausted its mcp action budget")
        || detail
            .contains("sub-agent exhausted its reasoning budget before reaching a terminal result")
        || contains_postgres_transaction_session_error(&detail)
        || detail.contains("duplicate-call")
}

fn should_cycle_mcp_connection_after_success(server: &PreparedServer, tool_name: &str) -> bool {
    server.full_catalog.server.protocol_name == "postgres-mcp" && tool_name == "query"
}

fn should_retry_postgres_query_session_error(
    server: &PreparedServer,
    tool_name: &str,
    result_summary: &str,
) -> bool {
    should_cycle_mcp_connection_after_success(server, tool_name)
        && contains_postgres_transaction_session_error(&result_summary.to_lowercase())
}

fn contains_postgres_transaction_session_error(text: &str) -> bool {
    text.contains("set_session cannot be used inside a transaction")
        || text.contains("transaction/session")
}

fn goals_are_materially_similar(left: &str, right: &str) -> bool {
    let left_tokens = normalized_goal_tokens(left);
    let right_tokens = normalized_goal_tokens(right);
    if left_tokens.is_empty() || right_tokens.is_empty() {
        return false;
    }
    let overlap = left_tokens.intersection(&right_tokens).count();
    overlap >= 3
}

fn normalized_goal_tokens(goal: &str) -> HashSet<String> {
    const STOPWORDS: &[&str] = &[
        "the", "and", "for", "with", "from", "that", "this", "then", "into", "using", "use", "via",
        "after", "before", "your", "their", "them", "into", "through", "need", "build", "complete",
        "report", "analysis",
    ];
    goal.split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter_map(|token| {
            let token = token.trim().to_ascii_lowercase();
            if token.len() < 4 || STOPWORDS.contains(&token.as_str()) {
                None
            } else {
                Some(token)
            }
        })
        .collect()
}

fn todo_can_complete_from_partial_mcp_collection(text: &str) -> bool {
    let text = text.to_lowercase();
    let hybrid_collection_prefixes = [
        "compute or extract",
        "compute or query",
        "compute or inspect",
    ];
    let collection_prefixes = [
        "query ",
        "query or inspect",
        "inspect ",
        "inspect or query",
        "collect ",
        "load ",
        "discover ",
        "confirm ",
        "prepare ",
        "fetch ",
        "extract or materialize",
    ];
    let starts_as_collection_step = hybrid_collection_prefixes
        .iter()
        .any(|prefix| text.starts_with(prefix))
        || collection_prefixes
            .iter()
            .any(|prefix| text.starts_with(prefix));
    let is_follow_on_analysis = (text.starts_with("compute")
        && !hybrid_collection_prefixes
            .iter()
            .any(|prefix| text.starts_with(prefix)))
        || text.starts_with("identify")
        || text.starts_with("derive")
        || text.starts_with("validate")
        || text.starts_with("analy")
        || text.starts_with("generate")
        || text.starts_with("open ")
        || text.contains("html")
        || text.contains("browser")
        || text.contains("chart")
        || text.contains("table");
    starts_as_collection_step && !is_follow_on_analysis
}

fn todo_starts_with_generic_scaffold(todo_path: &Path) -> Result<bool, RuntimeError> {
    let Some(todo_list) = load_todo_list_if_present(todo_path)? else {
        return Ok(false);
    };
    Ok(matches!(
        todo_list.next_actionable(),
        Some((1, item)) if item.text == GENERIC_STARTER_TODO
    ))
}

fn reconcile_mandatory_todos_from_turn_trace(
    todo_path: &Path,
    html_output_path: &Path,
    working_directory: &Path,
    messages: &[MessageRecord],
) -> Result<(), RuntimeError> {
    let Some(mut todos) = load_todo_list_if_present(todo_path)? else {
        return Ok(());
    };

    let html_written = observed_html_write(messages, html_output_path);
    let html_path_printed = observed_html_path_print(messages, html_output_path, working_directory);
    let mut changed = false;

    if html_written {
        changed |= auto_complete_next_todo_if_matches(&mut todos, MANDATORY_TODO_GENERATE_HTML)?;
    }
    if html_path_printed {
        changed |= auto_complete_next_todo_if_matches(&mut todos, MANDATORY_TODO_OPEN_HTML)?;
    }

    if changed {
        todos.save_to_path(todo_path)?;
    }

    Ok(())
}

fn reconcile_active_todo_from_turn_trace(
    todo_path: &Path,
    messages: &[MessageRecord],
) -> Result<(), RuntimeError> {
    let Some(mut todos) = load_todo_list_if_present(todo_path)? else {
        return Ok(());
    };
    let Some((item_index, item)) = todos.next_actionable() else {
        return Ok(());
    };
    if item.status != crate::todo::TodoStatus::InProgress
        || item.text == GENERIC_STARTER_TODO
        || item.text == MANDATORY_TODO_GENERATE_HTML
        || item.text == MANDATORY_TODO_OPEN_HTML
    {
        return Ok(());
    }

    let Some(last_subagent_result) = messages.iter().rev().find_map(|message| match message {
        MessageRecord::SubAgentResult(record) => Some(record),
        _ => None,
    }) else {
        return Ok(());
    };

    let should_complete = match last_subagent_result.status.as_str() {
        "completed" => true,
        "partial" => {
            last_subagent_result.subagent_type == "tool-executor"
                && todo_can_complete_from_partial_local_inspection(&item.text)
        }
        _ => false,
    };

    if !should_complete {
        return Ok(());
    }

    todos.set_status(item_index, crate::todo::TodoStatus::Completed)?;
    todos.save_to_path(todo_path)?;
    Ok(())
}

fn reconcile_pending_mcp_todos_from_turn_trace(
    todo_path: &Path,
    messages: &[MessageRecord],
) -> Result<(), RuntimeError> {
    let Some(mut todos) = load_todo_list_if_present(todo_path)? else {
        return Ok(());
    };
    let Some(segment) = latest_mcp_execution_segment(messages) else {
        return Ok(());
    };
    if segment.result.executed_action_count == 0 {
        return Ok(());
    }

    let status = segment.result.status.as_str();
    if status == "cannot_execute"
        || (status == "partial" && mcp_partial_indicates_blocking_failure(&segment.result.detail))
    {
        return Ok(());
    }

    let evidence = render_mcp_execution_evidence(&segment);
    let mut changed = false;

    loop {
        let Some((item_index, item)) = todos.next_actionable() else {
            break;
        };
        if item.status != crate::todo::TodoStatus::Pending || !todo_item_prefers_mcp_executor(item)
        {
            break;
        }

        let should_complete = match status {
            "completed" => mcp_execution_evidence_satisfies_todo(&evidence, &item.text),
            "partial" => {
                todo_can_complete_from_partial_mcp_collection(&item.text)
                    && mcp_execution_evidence_satisfies_todo(&evidence, &item.text)
            }
            _ => false,
        };
        if !should_complete {
            break;
        }

        todos.set_status(item_index, crate::todo::TodoStatus::InProgress)?;
        todos.set_status(item_index, crate::todo::TodoStatus::Completed)?;
        changed = true;
    }

    if changed {
        todos.save_to_path(todo_path)?;
    }

    Ok(())
}

struct LatestMcpExecutionSegment<'a> {
    messages: &'a [MessageRecord],
    result: &'a crate::state::SubAgentResultMessageRecord,
}

fn latest_mcp_execution_segment(
    messages: &[MessageRecord],
) -> Option<LatestMcpExecutionSegment<'_>> {
    let result_index = messages.iter().rposition(|message| {
        matches!(
            message,
            MessageRecord::SubAgentResult(record) if record.subagent_type == "mcp-executor"
        )
    })?;
    let MessageRecord::SubAgentResult(result) = &messages[result_index] else {
        return None;
    };
    let call_index = messages[..result_index].iter().rposition(|message| {
        matches!(
            message,
            MessageRecord::SubAgentCall(record) if record.subagent_type == "mcp-executor"
        )
    })?;
    Some(LatestMcpExecutionSegment {
        messages: &messages[call_index..=result_index],
        result,
    })
}

fn render_mcp_execution_evidence(segment: &LatestMcpExecutionSegment<'_>) -> String {
    segment
        .messages
        .iter()
        .map(|message| match message {
            MessageRecord::SubAgentCall(record) => {
                format!(
                    "subagent goal {} target {}",
                    record.goal,
                    record.target.summary()
                )
            }
            MessageRecord::McpCall(record) => format!(
                "mcp call target {} args {}",
                record.target.capability_id, record.arguments
            ),
            MessageRecord::McpResult(record) => {
                format!("mcp result {}", record.result_summary)
            }
            MessageRecord::SubAgentResult(record) => {
                format!("subagent detail {}", record.detail)
            }
            _ => String::new(),
        })
        .filter(|chunk| !chunk.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn mcp_execution_evidence_satisfies_todo(evidence: &str, todo_text: &str) -> bool {
    let evidence_tokens = normalized_runtime_tokens(evidence);
    let todo_tokens = normalized_runtime_tokens(todo_text);
    if evidence_tokens.is_empty() || todo_tokens.is_empty() {
        return false;
    }

    let overlap = todo_tokens.intersection(&evidence_tokens).count();
    let required_overlap = if todo_tokens.len() >= 10 {
        4
    } else if todo_tokens.len() >= 6 {
        3
    } else {
        2
    };

    overlap >= required_overlap
}

fn normalized_runtime_tokens(text: &str) -> HashSet<String> {
    const STOPWORDS: &[&str] = &[
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "then",
        "into",
        "using",
        "use",
        "via",
        "after",
        "before",
        "your",
        "their",
        "them",
        "need",
        "build",
        "perform",
        "brief",
        "compact",
        "exact",
        "useful",
        "couple",
        "without",
        "overextending",
        "scope",
        "across",
        "table",
        "tables",
    ];
    text.split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter_map(|token| {
            let token = token.trim().to_ascii_lowercase();
            if token.len() < 4 || STOPWORDS.contains(&token.as_str()) {
                None
            } else {
                Some(token)
            }
        })
        .collect()
}

fn incomplete_todo_final_feedback(
    item_index: usize,
    item_status: &str,
    item_text: &str,
    html_output_path: &Path,
) -> String {
    if item_text == MANDATORY_TODO_GENERATE_HTML {
        return format!(
            "Cannot finish the turn yet because the mandatory deliverable is still incomplete. The todo file exists and is already loaded in the Current Turn Todo Plan; do not claim that the todo file is missing. Next actionable todo: {item_index}. [{item_status}] {item_text}. Emit a `delegate_subagent` decision now with `subagent_type: \"tool-executor\"` and target `{{\"kind\":\"local_tools_scope\",\"value\":{{\"scope\":\"workspace\"}}}}` to generate the deterministic HTML report at `{}`. Then continue the turn instead of returning a final answer. Do not ask `tool-executor` to print the report path in the same delegation; return control first and print it only when the next actionable todo becomes `{}`.",
            html_output_path.display(),
            MANDATORY_TODO_OPEN_HTML
        );
    }
    if item_text == MANDATORY_TODO_OPEN_HTML {
        return format!(
            "Cannot finish the turn yet because the report path still needs to be printed. The todo file exists and is already loaded in the Current Turn Todo Plan; do not claim that the todo file is missing. Next actionable todo: {item_index}. [{item_status}] {item_text}. Emit a `delegate_subagent` decision now with `subagent_type: \"tool-executor\"` and target `{{\"kind\":\"local_tools_scope\",\"value\":{{\"scope\":\"workspace\"}}}}` to print the generated HTML file path `{}` with local `bash`, then continue instead of returning a final answer.",
            html_output_path.display()
        );
    }
    format!(
        "Cannot finish the turn yet because the todo plan is incomplete. Next actionable todo: {item_index}. [{item_status}] {item_text}. Continue execution instead of returning a final answer."
    )
}

enum FinalTodoAdvance {
    Noop,
    AdvancedWithFeedback(String),
    AdvancedAllDone,
}

fn auto_advance_summary_todo_from_final_content(
    todo_path: &Path,
    item_index: usize,
    item_text: &str,
    content: &str,
    html_output_path: &Path,
) -> Result<FinalTodoAdvance, RuntimeError> {
    if !todo_text_looks_like_summary_step(item_text)
        || content_looks_like_blocker_summary(content)
        || content.trim().len() < 60
    {
        return Ok(FinalTodoAdvance::Noop);
    }

    let mut todo_list = match load_todo_list_if_present(todo_path)? {
        Some(todo_list) => todo_list,
        None => return Ok(FinalTodoAdvance::Noop),
    };
    let Some((current_index, current_item)) = todo_list.next_actionable() else {
        return Ok(FinalTodoAdvance::Noop);
    };
    if current_index != item_index || current_item.text != item_text {
        return Ok(FinalTodoAdvance::Noop);
    }

    todo_list
        .set_status(item_index, crate::todo::TodoStatus::InProgress)
        .map_err(RuntimeError::Todo)?;
    todo_list
        .set_status(item_index, crate::todo::TodoStatus::Completed)
        .map_err(RuntimeError::Todo)?;
    todo_list
        .save_to_path(todo_path)
        .map_err(RuntimeError::Todo)?;

    let feedback = todo_list.next_actionable().map(|(next_index, next_item)| {
        format!(
            "The previous answer draft already satisfied todo {item_index} and has been recorded for this turn. Do not repeat it. {}",
            incomplete_todo_final_feedback(
                next_index,
                next_item.status.as_str(),
                &next_item.text,
                html_output_path,
            )
        )
    });

    Ok(match feedback {
        Some(content) => FinalTodoAdvance::AdvancedWithFeedback(content),
        None => FinalTodoAdvance::AdvancedAllDone,
    })
}

fn todo_text_looks_like_summary_step(item_text: &str) -> bool {
    if item_text == MANDATORY_TODO_GENERATE_HTML || item_text == MANDATORY_TODO_OPEN_HTML {
        return false;
    }

    let normalized = item_text.to_ascii_lowercase();
    let padded = format!(" {normalized} ");
    let summary_markers = [
        "return ",
        "summary",
        "summarize",
        "summarise",
        "report",
        "business-readable",
        "business readable",
        "business-style",
        "business style",
        "concise",
        "tell the user",
        "provide",
        "answer",
    ];
    let execution_markers = [
        " inspect ",
        " query ",
        " compute ",
        " calculate ",
        " find ",
        " check ",
        " collect ",
        " extract ",
        " load ",
        " run ",
        "generate an output html",
        "open the generated output html",
    ];

    summary_markers.iter().any(|marker| padded.contains(marker))
        && !execution_markers
            .iter()
            .any(|marker| padded.contains(marker))
}

fn allow_blocker_final_for_failed_todo(
    todo_list: &TodoList,
    content: &str,
    messages: &[MessageRecord],
) -> bool {
    let Some((_, item)) = todo_list.next_actionable() else {
        return false;
    };
    if item.status != crate::todo::TodoStatus::Failed {
        return false;
    }
    if !content_looks_like_blocker_summary(content) {
        return false;
    }

    messages
        .iter()
        .rev()
        .any(message_indicates_execution_blocker)
}

fn content_looks_like_blocker_summary(content: &str) -> bool {
    let normalized = content.to_lowercase();
    let blocker_terms = [
        "i'm blocked",
        "i am blocked",
        "cannot continue",
        "can't continue",
        "unable to continue",
        "need a different",
        "different execution route",
        "fresh execution path",
        "alternate allowed",
        "blocked mcp path",
        "duplicate query",
    ];
    blocker_terms.iter().any(|term| normalized.contains(term))
}

fn message_indicates_execution_blocker(message: &MessageRecord) -> bool {
    match message {
        MessageRecord::SubAgentResult(record) => {
            if !matches!(record.status.as_str(), "partial" | "cannot_execute") {
                return false;
            }
            let detail = record.detail.to_lowercase();
            detail.contains("blocked")
                || detail.contains("duplicate")
                || detail.contains("cannot execute")
                || detail.contains("repeated mcp failures")
                || detail.contains("forced an early stop")
        }
        MessageRecord::Llm(record) => {
            let content = record.content.to_lowercase();
            content.contains("do not delegate the same blocked mcp path again")
                || content.contains("selected mcp capability cannot be used")
                || content.contains("the previous `mcp-executor` attempt")
                || content.contains("blocked mcp path")
        }
        _ => false,
    }
}

fn todo_can_complete_from_partial_local_inspection(text: &str) -> bool {
    let text = text.to_lowercase();
    let mentions_local_inspection = text.contains("inspect available local workspace inputs")
        || text.contains("inspect local workspace inputs")
        || text.contains("inspect available local inputs")
        || text.contains("inspect the local workspace")
        || text.contains("search workspace inputs")
        || text.contains("inspect workspace inputs");
    let is_substantive_data_step = text.contains("extract")
        || text.contains("materialize")
        || text.contains("compute")
        || text.contains("validate")
        || text.contains("generate")
        || text.contains("html")
        || text.contains("chart")
        || text.contains("table");
    mentions_local_inspection && !is_substantive_data_step
}

fn html_todo_feedback_for_delegation(
    todo_path: &Path,
    delegation: &crate::model::SubagentDelegationRequest,
    html_output_path: &Path,
) -> Result<Option<String>, RuntimeError> {
    let Some(todo_list) = load_todo_list_if_present(todo_path)? else {
        return Ok(None);
    };
    let Some((item_index, item)) = todo_list.next_actionable() else {
        return Ok(None);
    };
    let goal = delegation.goal.to_lowercase();
    let is_local_tool_delegation = delegation.subagent_type == "tool-executor"
        && matches!(delegation.target, DelegationTarget::LocalToolsScope(_));

    if todo_matches_generate_html(item.text.as_str()) && !is_local_tool_delegation {
        return Ok(Some(format!(
            "The current next actionable todo is the HTML-generation step: {item_index}. [{}] {}. Do not delegate MCP/data work or any other broader workflow now. Emit `delegate_subagent` with `subagent_type: \"tool-executor\"` and target `{{\"kind\":\"local_tools_scope\",\"value\":{{\"scope\":\"workspace\"}}}}` to generate only the deterministic HTML report at `{}`. Return control after generating it; print the report path only when the next actionable todo becomes `{}`.",
            item.status.as_str(),
            item.text,
            html_output_path.display(),
            MANDATORY_TODO_OPEN_HTML
        )));
    }

    if todo_matches_generate_html(item.text.as_str()) && goal_mentions_html_open(&goal) {
        return Ok(Some(format!(
            "The current next actionable todo is only the HTML-generation step: {item_index}. [{}] {}. Do not delegate broader analysis work. Do not bundle the html-path-print step into the same `tool-executor` delegation. Delegate only the deterministic HTML report generation at `{}` first, return control immediately, and print its path only when the next actionable todo becomes `{}`.",
            item.status.as_str(),
            item.text,
            html_output_path.display(),
            MANDATORY_TODO_OPEN_HTML
        )));
    }

    if todo_matches_generate_html(item.text.as_str())
        && (!goal_mentions_html_generation(&goal) || goal_mentions_data_execution(&goal))
    {
        return Ok(Some(format!(
            "The current next actionable todo is only the HTML-generation step: {item_index}. [{}] {}. Do not delegate broader analysis work. Do not bundle the html-path-print step into the same `tool-executor` delegation. Delegate only the deterministic HTML report generation at `{}` first, return control immediately, and print its path only when the next actionable todo becomes `{}`.",
            item.status.as_str(),
            item.text,
            html_output_path.display(),
            MANDATORY_TODO_OPEN_HTML
        )));
    }

    if todo_matches_open_html(item.text.as_str()) && !is_local_tool_delegation {
        return Ok(Some(format!(
            "The current next actionable todo is the html-path-print step: {item_index}. [{}] {}. The deterministic HTML report already exists at `{}`. Do not delegate MCP/data work or broader analysis now. Emit `delegate_subagent` with `subagent_type: \"tool-executor\"` and target `{{\"kind\":\"local_tools_scope\",\"value\":{{\"scope\":\"workspace\"}}}}`, then use local `bash` with `printf '%s\\n' {}` and return control.",
            item.status.as_str(),
            item.text,
            html_output_path.display(),
            html_output_path.display()
        )));
    }

    if todo_matches_open_html(item.text.as_str()) && goal_mentions_html_generation(&goal) {
        return Ok(Some(format!(
            "The current next actionable todo is already the html-path-print step: {item_index}. [{}] {}. Do not regenerate the HTML report in this delegation. Only print the existing deterministic HTML report path `{}` by delegating `tool-executor` and using local `bash` with `printf '%s\\n' {}`.",
            item.status.as_str(),
            item.text,
            html_output_path.display(),
            html_output_path.display()
        )));
    }
    if todo_matches_open_html(item.text.as_str()) && !goal_is_open_only(&goal) {
        return Ok(Some(format!(
            "The current next actionable todo is already the html-path-print step: {item_index}. [{}] {}. Do not delegate any broader `tool-executor` work here. Only print the existing deterministic HTML report path `{}` by delegating `tool-executor` and using local `bash` with `printf '%s\\n' {}`, then return control.",
            item.status.as_str(),
            item.text,
            html_output_path.display(),
            html_output_path.display()
        )));
    }

    Ok(None)
}

#[cfg(test)]
fn should_require_todos_for_turn(
    user_message: &str,
    conversation_history: &[crate::state::ConversationMessage],
) -> bool {
    !(is_simple_factual_turn(user_message)
        || is_simple_factual_follow_up_turn(user_message, conversation_history))
}

#[cfg(test)]
fn is_simple_factual_turn(user_message: &str) -> bool {
    let normalized = user_message
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    if normalized.is_empty() || normalized.len() > 160 {
        return false;
    }
    if user_message
        .lines()
        .filter(|line| !line.trim().is_empty())
        .count()
        > 2
    {
        return false;
    }

    let disqualifying_terms = [
        "analysis",
        "analyze",
        "compare",
        "breakdown",
        "trend",
        "chart",
        "graph",
        "plot",
        "table",
        "visual",
        "dashboard",
        "report",
        "html",
        "browser",
        "csv",
        "file",
        "script",
    ];
    if disqualifying_terms
        .iter()
        .any(|term| normalized.contains(term))
    {
        return false;
    }

    let factual_prefixes = [
        "who ",
        "what ",
        "when ",
        "where ",
        "which ",
        "is ",
        "are ",
        "was ",
        "were ",
        "did ",
        "does ",
        "do ",
        "can ",
        "could ",
        "how many ",
        "how much ",
        "how old ",
        "tell me ",
    ];
    if factual_prefixes
        .iter()
        .any(|prefix| normalized.starts_with(prefix))
    {
        return true;
    }

    normalized.ends_with('?') && normalized.split_whitespace().count() <= 12
}

#[cfg(test)]
fn is_simple_factual_follow_up_turn(
    user_message: &str,
    conversation_history: &[crate::state::ConversationMessage],
) -> bool {
    if !is_short_contextual_follow_up(user_message) {
        return false;
    }

    conversation_history
        .iter()
        .rev()
        .find(|message| matches!(message.role, crate::state::ConversationRole::User))
        .map(|message| is_simple_factual_turn(&message.content))
        .unwrap_or(false)
}

#[cfg(test)]
fn is_short_contextual_follow_up(user_message: &str) -> bool {
    let normalized = user_message
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    if normalized.is_empty() {
        return false;
    }

    let trimmed = normalized
        .trim_matches(|ch: char| !ch.is_alphanumeric())
        .trim()
        .to_owned();
    if trimmed.is_empty() {
        return false;
    }
    if trimmed.len() > 40 && !is_short_metric_clarification_follow_up(&trimmed) {
        return false;
    }

    let first_token = trimmed
        .split(|ch: char| ch.is_whitespace() || [',', '.', '!', '?', ';', ':'].contains(&ch))
        .find(|token| !token.is_empty())
        .unwrap_or("");
    if matches!(
        first_token,
        "yes" | "yeah" | "yep" | "ok" | "okay" | "sure" | "retry"
    ) {
        return true;
    }

    ["try again", "go ahead", "please do", "do it", "continue"]
        .iter()
        .any(|phrase| trimmed.starts_with(phrase))
        || is_short_metric_clarification_follow_up(&trimmed)
}

#[cfg(test)]
fn is_short_metric_clarification_follow_up(user_message: &str) -> bool {
    if user_message.len() > 90 || user_message.contains('?') {
        return false;
    }

    let subject_prefixes = [
        "best ",
        "top ",
        "most ",
        "highest ",
        "lowest ",
        "fielder ",
        "bowler ",
        "batter ",
        "metric ",
        "fielding ",
        "bowling ",
        "batting ",
    ];
    if !subject_prefixes
        .iter()
        .any(|prefix| user_message.starts_with(prefix))
    {
        return false;
    }

    let clarification_markers = [
        " i mean ",
        " means ",
        " should mean ",
        " is somebody who ",
        " is the player who ",
        " is whoever ",
        " based on ",
    ];
    clarification_markers
        .iter()
        .any(|marker| user_message.contains(marker))
}

fn goal_mentions_html_open(goal: &str) -> bool {
    let mentions_legacy_open = goal.contains("open")
        && (goal.contains("browser")
            || goal.contains("html")
            || goal.contains("report")
            || goal.contains("page"));
    let mentions_path_print = (goal.contains("print")
        || goal.contains("display")
        || goal.contains("show")
        || goal.contains("return"))
        && goal.contains("path")
        && (goal.contains("html") || goal.contains("report") || goal.contains("file"));
    mentions_legacy_open || mentions_path_print
}

fn todo_matches_generate_html(text: &str) -> bool {
    text == MANDATORY_TODO_GENERATE_HTML
        || (text.to_lowercase().contains("html")
            && text.to_lowercase().contains("chart")
            && text.to_lowercase().contains("table"))
}

fn todo_matches_open_html(text: &str) -> bool {
    let text = text.to_lowercase();
    text == MANDATORY_TODO_OPEN_HTML.to_lowercase()
        || ((text.contains("print")
            || text.contains("display")
            || text.contains("show")
            || text.contains("return"))
            && text.contains("path")
            && (text.contains("html") || text.contains("report") || text.contains("file")))
        || (text.contains("open") && text.contains("html") && text.contains("browser"))
}

fn goal_mentions_html_generation(goal: &str) -> bool {
    let mentions_generation = ["generate", "create", "build", "render", "produce", "write"]
        .iter()
        .any(|verb| goal.contains(verb));
    let mentions_completion_as_html_generation =
        (goal.contains("complete") || goal.contains("finish")) && goal.contains("html");
    let mentions_html_target =
        goal.contains("html") || goal.contains("report") || goal.contains("page");
    (mentions_generation && mentions_html_target) || mentions_completion_as_html_generation
}

fn goal_mentions_data_execution(goal: &str) -> bool {
    [
        "calculate",
        "collect",
        "compute",
        "database",
        "determine",
        "discover",
        "extract",
        "inspect",
        "query",
        "sql",
    ]
    .iter()
    .any(|term| goal.contains(term))
}

fn goal_is_open_only(goal: &str) -> bool {
    if !goal_mentions_html_open(goal) {
        return false;
    }
    let disallowed_terms = [
        "generate",
        "create",
        "build",
        "render",
        "produce",
        "write",
        "answer",
        "deliverable",
        "report deliverable",
        "complete",
        "summarize",
        "chart",
        "table",
        "analy",
    ];
    !disallowed_terms.iter().any(|term| goal.contains(term))
}

fn auto_complete_next_todo_if_matches(
    todos: &mut TodoList,
    expected_text: &str,
) -> Result<bool, RuntimeError> {
    let Some((item_index, item)) = todos.next_actionable() else {
        return Ok(false);
    };
    if item.text != expected_text {
        return Ok(false);
    }

    todos.set_status(item_index, crate::todo::TodoStatus::InProgress)?;
    todos.set_status(item_index, crate::todo::TodoStatus::Completed)?;
    Ok(true)
}

fn observed_html_write(messages: &[MessageRecord], html_output_path: &Path) -> bool {
    if html_output_path.exists() {
        return html_report_file_is_acceptable(html_output_path);
    }

    let html_path = html_output_path.display().to_string();
    let html_file_name = html_output_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    messages.iter().any(|message| match message {
        MessageRecord::LocalToolResult(record) if record.tool_name.as_str() == "write_file" => {
            record.status == "ok" && record.result_summary.contains(&html_path)
        }
        MessageRecord::LocalToolResult(record) if record.tool_name.as_str() == "bash" => {
            record.status == "ok"
                && record.result_summary.contains("exit_code: 0")
                && (record.result_summary.contains(&html_path)
                    || (!html_file_name.is_empty()
                        && record.result_summary.contains(html_file_name)))
        }
        _ => false,
    })
}

fn html_report_file_is_acceptable(html_output_path: &Path) -> bool {
    std::fs::read_to_string(html_output_path)
        .map(|content| {
            let normalized = content.to_ascii_lowercase();
            normalized.contains("<html")
                && normalized.contains("</html")
                && !normalized.contains("placeholder")
        })
        .unwrap_or(false)
}

fn observed_html_path_print(
    messages: &[MessageRecord],
    html_output_path: &Path,
    working_directory: &Path,
) -> bool {
    if !html_report_file_is_acceptable(html_output_path) {
        return false;
    }
    let absolute = html_output_path.display().to_string();
    let relative = html_output_path
        .strip_prefix(working_directory)
        .ok()
        .map(|path| path.display().to_string());

    messages.iter().any(|message| match message {
        MessageRecord::LocalToolResult(record) if record.tool_name.as_str() == "bash" => {
            if record.status != "ok" || !record.result_summary.contains("exit_code: 0") {
                return false;
            }
            let printed_absolute =
                bash_result_printed_path(record.result_summary.as_str(), &absolute);
            let printed_relative = relative.as_ref().is_some_and(|path| {
                bash_result_printed_path(record.result_summary.as_str(), path)
                    || bash_result_printed_path(
                        record.result_summary.as_str(),
                        &format!("./{path}"),
                    )
            });
            printed_absolute || printed_relative
        }
        _ => false,
    })
}

fn bash_result_printed_path(summary: &str, path: &str) -> bool {
    summary
        .split_once("\nstdout:\n")
        .map(|(_, stdout)| stdout.lines().any(|line| line.trim() == path))
        .unwrap_or(false)
}

fn build_subagent_prompt(
    registry_path: &Path,
    configured: &ConfiguredSubagent,
    mcp_session: &McpSession,
    phase: TurnPhase,
    target: &DelegationTarget,
    goal: &str,
    user_message: &str,
    working_directory: &Path,
    todo_path: &Path,
    turn_policy_context: &TurnPolicyPromptContext,
    execution_handoff: Option<&ExecutionHandoff>,
    current_turn_messages: &[MessageRecord],
) -> Result<String, RuntimeError> {
    let registry_dir = registry_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let turn_policy_block = format!(
        "\n## Turn Policy\n{}\n",
        render_turn_policy_context(turn_policy_context)
    );
    let execution_handoff_block = if phase == TurnPhase::Execution {
        execution_handoff
            .map(|handoff| {
                format!(
                    "\n## Execution Handoff\n{}\n",
                    render_execution_handoff(handoff)
                )
            })
            .unwrap_or_default()
    } else {
        String::new()
    };
    let todo_block = load_todo_prompt_context(todo_path)?
        .map(|todo_context| {
            format!(
                "\n## Current Turn Todo Plan\n{}\n- Replanning rule: rewrite only the future pending suffix when the current plan is insufficient.\n",
                render_todo_context(&todo_context)
            )
        })
        .unwrap_or_default();
    match target {
        DelegationTarget::McpCapability(target) => {
            let server = mcp_session
                .servers
                .get(&target.server_name)
                .ok_or_else(|| RuntimeError::UnknownServer(target.server_name.to_string()))?;
            let template = load_subagent_prompt(
                &registry_dir,
                configured,
                phase,
                &server.full_catalog_markdown,
            )?;
            let recent_results_block =
                render_recent_mcp_results_context_for_mcp(current_turn_messages);
            let confirmed_tables_block =
                render_confirmed_table_context(current_turn_messages, &target.server_name);
            let server_guidance_block = render_mcp_server_guidance(server);

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
                "{template}\n\n## Delegation Context\n- Goal: {goal}\n- User request: {user_message}\n- Selected target: {}\n- Execution scope: only the selected MCP capability\n- Local tools: unavailable\n{server_guidance_block}{turn_policy_block}{execution_handoff_block}{todo_block}\n{}\n{}\n{capability_block}",
                target.capability_id, confirmed_tables_block, recent_results_block,
            ))
        }
        DelegationTarget::McpServerScope(target) => {
            let server = mcp_session
                .servers
                .get(&target.server_name)
                .ok_or_else(|| RuntimeError::UnknownServer(target.server_name.to_string()))?;
            let template = load_subagent_prompt(
                &registry_dir,
                configured,
                phase,
                &server.full_catalog_markdown,
            )?;
            let recent_results_block =
                render_recent_mcp_results_context_for_mcp(current_turn_messages);
            let confirmed_tables_block =
                render_confirmed_table_context(current_turn_messages, &target.server_name);
            let server_guidance_block = render_mcp_server_guidance(server);

            Ok(format!(
                "{template}\n\n## Delegation Context\n- Goal: {goal}\n- User request: {user_message}\n- Selected server: {}\n- Execution scope: any allowed MCP tool or resource on this server only\n- Local tools: unavailable\n- Do not switch to a different MCP server\n{server_guidance_block}{turn_policy_block}{execution_handoff_block}{todo_block}\n{}\n{}\n",
                target.server_name, confirmed_tables_block, recent_results_block,
            ))
        }
        DelegationTarget::LocalToolsScope(scope) => {
            let template =
                load_subagent_prompt(&registry_dir, configured, phase, "Local tool catalog")?;
            let tools_block = builtin_local_tool_catalog()
                .into_iter()
                .map(|tool| {
                    format!(
                        "- Tool: {}\n  Description: {}\n  Input schema: {}",
                        tool.name, tool.description, tool.input_schema
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            let recent_results_block = render_recent_mcp_results_context(current_turn_messages);
            let planning_only_block = if configured.subagent_type == "tool-executor"
                && is_todo_planning_goal(goal)
            {
                "- This delegation is todo-planning only. Update the todo plan with `write_todos`, prefixing every new todo with `[mcp-executor]`, `[tool-executor]`, or `[main-agent]`, then return `done` immediately.\n- Do not inspect unrelated workspace data, do not begin executing the planned analysis, and do not mark later execution todos in progress or completed in this delegation.\n"
            } else {
                ""
            };
            let explicit_html_todo_guidance = if configured.subagent_type == "tool-executor" {
                load_todo_prompt_context(todo_path)?
                    .and_then(|todo_context| todo_context.next_actionable)
                    .map(|next| {
                        if next.contains(MANDATORY_TODO_GENERATE_HTML) {
                            format!(
                                "- The current actionable todo is the HTML-generation step. Write the final deterministic HTML report directly at `{}` and return control; do not print the report path in this delegation. Avoid fragile intermediate report-builder scripts unless necessary. If you do use Python/pandas to assemble the HTML, avoid `itertuples()` field aliases for columns with spaces or symbols; use dictionaries, `to_dict(orient=\"records\")`, or explicit column indexing instead.\n",
                                turn_policy_context.html_output_path.display()
                            )
                        } else if next.contains(MANDATORY_TODO_OPEN_HTML) {
                            format!(
                                "- The current actionable todo is the html-path-print step. Execute it with local `bash` using `printf '%s\\n' {}`. Do not say that no local tool action is available for this step.\n",
                                turn_policy_context.html_output_path.display()
                            )
                        } else {
                            String::new()
                        }
                    })
                    .unwrap_or_default()
            } else {
                String::new()
            };
            Ok(format!(
                "{template}\n\n## Delegation Context\n- Goal: {goal}\n- User request: {user_message}\n- Local scope: {}\n- Working directory: {}\n- Deterministic HTML output path: {}\n- If todos are optional and no todo file exists, complete the HTML report and print the generated HTML file path for analysis/reporting work before finishing.\n- When todo context is present, update only the current todo item, and replan only the future pending suffix.\n{explicit_html_todo_guidance}{planning_only_block}{turn_policy_block}{execution_handoff_block}{todo_block}\n{}\n## Local Tools\n{}",
                scope.scope,
                working_directory.display(),
                turn_policy_context.html_output_path.display(),
                recent_results_block,
                tools_block
            ))
        }
    }
}

fn render_recent_mcp_results_context(messages: &[MessageRecord]) -> String {
    const MAX_RECENT_RESULTS: usize = 3;
    const MAX_RESULT_SUMMARY_CHARS: usize = 1_500;

    let recent_results = messages
        .iter()
        .rev()
        .filter_map(|message| match message {
            MessageRecord::McpResult(record) => Some(record),
            _ => None,
        })
        .take(MAX_RECENT_RESULTS)
        .collect::<Vec<_>>();

    if recent_results.is_empty() {
        return "## Recent Computed Results\nNo recent MCP-derived results are available in this turn. If a visualization is needed, compute or fetch the data first, then write it to a local file before plotting.\n".to_owned();
    }

    let rendered_results = recent_results
        .into_iter()
        .rev()
        .map(|record| {
            format!(
                "- server={} kind={:?} target={} error={}\n  result_summary:\n{}",
                record.target.server_name,
                record.target.capability_kind,
                record.target.capability_id,
                record.error.is_some(),
                indent_block(&truncate_for_prompt(
                    &record.result_summary,
                    MAX_RESULT_SUMMARY_CHARS
                ))
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "## Recent Computed Results\nUse these recent MCP-derived results as available analysis context. If you need a chart or follow-up calculation, first write the relevant rows into a local file under `outputs/` or `scripts/`, then build the visualization from that file.\n{}\n",
        rendered_results
    )
}

fn render_recent_mcp_results_context_for_mcp(messages: &[MessageRecord]) -> String {
    const MAX_RECENT_RESULTS: usize = 3;
    const MAX_RESULT_SUMMARY_CHARS: usize = 1_500;

    let recent_results = messages
        .iter()
        .rev()
        .filter_map(|message| match message {
            MessageRecord::McpResult(record) => Some(record),
            _ => None,
        })
        .take(MAX_RECENT_RESULTS)
        .collect::<Vec<_>>();

    if recent_results.is_empty() {
        return "## Recent Computed Results\nNo recent MCP-derived results are available yet for this server. If the exact target table is not already confirmed, start with discovery.\n".to_owned();
    }

    let rendered_results = recent_results
        .into_iter()
        .rev()
        .map(|record| {
            format!(
                "- server={} kind={:?} target={} error={}\n  result_summary:\n{}",
                record.target.server_name,
                record.target.capability_kind,
                record.target.capability_id,
                record.error.is_some(),
                indent_block(&truncate_for_prompt(
                    &record.result_summary,
                    MAX_RESULT_SUMMARY_CHARS
                ))
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "## Recent Computed Results\nUse these recent MCP-derived results as server-side analysis context. Prefer direct `run_query` only when they already confirm the exact target table; otherwise continue with discovery.\n{}\n",
        rendered_results
    )
}

fn render_confirmed_table_context(messages: &[MessageRecord], server_name: &ServerName) -> String {
    let confirmed_tables = collect_confirmed_tables(messages, server_name);
    if confirmed_tables.is_empty() {
        return "## Confirmed Tables\nNo exact table has been confirmed yet for this server. Prefer discovery first unless the user explicitly named the table.\n".to_owned();
    }

    let rendered = confirmed_tables
        .into_iter()
        .map(|table| format!("- {table}"))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "## Confirmed Tables\nThese exact tables were already confirmed by successful MCP query context in this session. Prefer direct `run_query` only when one of these tables clearly matches the delegated goal.\n{}\n",
        rendered
    )
}

fn collect_confirmed_tables(messages: &[MessageRecord], server_name: &ServerName) -> Vec<String> {
    let mut confirmed = Vec::new();
    let mut pending_queries = HashMap::<String, Vec<String>>::new();

    for message in messages {
        match message {
            MessageRecord::McpCall(record)
                if record.target.server_name == *server_name
                    && record.target.capability_kind == CapabilityKind::Tool
                    && record.target.capability_id == "run_query" =>
            {
                if let Some(query) = record
                    .arguments
                    .get("query")
                    .and_then(serde_json::Value::as_str)
                {
                    pending_queries.insert(query.to_owned(), extract_table_references(query));
                }
            }
            MessageRecord::McpResult(record)
                if record.target.server_name == *server_name
                    && record.target.capability_kind == CapabilityKind::Tool
                    && record.target.capability_id == "run_query"
                    && record.error.is_none() =>
            {
                for tables in pending_queries.values() {
                    for table in tables {
                        if !confirmed.iter().any(|existing| existing == table) {
                            confirmed.push(table.clone());
                        }
                    }
                }
                pending_queries.clear();
            }
            _ => {}
        }
    }

    confirmed
}

fn extract_table_references(query: &str) -> Vec<String> {
    let flattened = query.replace(['\n', '\r', '\t'], " ");
    let normalized = flattened
        .split_whitespace()
        .map(|token| {
            token.trim_matches(|c: char| matches!(c, ',' | ';' | '(' | ')' | '"' | '\'' | '`'))
        })
        .collect::<Vec<_>>();
    let mut tables = Vec::new();

    for window in normalized.windows(2) {
        let keyword = window[0].to_ascii_lowercase();
        if matches!(keyword.as_str(), "from" | "join") {
            let candidate = window[1];
            if candidate.contains('.') && !tables.iter().any(|existing| existing == candidate) {
                tables.push(candidate.to_owned());
            }
        }
    }

    tables
}

fn truncate_for_prompt(input: &str, max_chars: usize) -> String {
    if input.chars().count() <= max_chars {
        return input.to_owned();
    }

    let mut truncated = input.chars().take(max_chars).collect::<String>();
    truncated.push_str("...");
    truncated
}

fn indent_block(text: &str) -> String {
    text.lines()
        .map(|line| format!("    {line}"))
        .collect::<Vec<_>>()
        .join("\n")
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

fn build_subagent_tool_catalog(
    mcp_session: &McpSession,
    target: &DelegationTarget,
) -> Result<Vec<ToolDescriptor>, RuntimeError> {
    match target {
        DelegationTarget::McpCapability(target) => {
            let server = mcp_session
                .servers
                .get(&target.server_name)
                .ok_or_else(|| RuntimeError::UnknownServer(target.server_name.to_string()))?;
            let mut tools = Vec::with_capacity(1);
            match target.capability_kind {
                CapabilityKind::Tool => {
                    let tool = lookup_tool(&server.full_catalog, &target.capability_id)?;
                    tools.push(ToolDescriptor::mcp_tool(
                        &target.server_name,
                        &tool.name,
                        tool.title.as_deref(),
                        tool.description.as_deref(),
                        tool.input_schema.clone(),
                    ));
                }
                CapabilityKind::Resource => {
                    if let Some(reason) = disabled_mcp_capability_reason(
                        server,
                        &CapabilityKind::Resource,
                        &target.capability_id,
                    ) {
                        return Err(RuntimeError::Validation(reason));
                    }
                    let resource = lookup_resource(&server.full_catalog, &target.capability_id)?;
                    tools.push(ToolDescriptor::mcp_resource(
                        &target.server_name,
                        &resource.uri,
                        resource.title.as_deref(),
                        resource.description.as_deref(),
                    ));
                }
            }
            Ok(tools)
        }
        DelegationTarget::McpServerScope(target) => {
            let server = mcp_session
                .servers
                .get(&target.server_name)
                .ok_or_else(|| RuntimeError::UnknownServer(target.server_name.to_string()))?;
            let mut tools = Vec::new();
            for tool in &server.full_catalog.tools {
                tools.push(ToolDescriptor::mcp_tool(
                    &target.server_name,
                    &tool.name,
                    tool.title.as_deref(),
                    tool.description.as_deref(),
                    tool.input_schema.clone(),
                ));
            }
            for resource in &server.full_catalog.resources {
                if disabled_mcp_capability_reason(server, &CapabilityKind::Resource, &resource.uri)
                    .is_some()
                {
                    continue;
                }
                tools.push(ToolDescriptor::mcp_resource(
                    &target.server_name,
                    &resource.uri,
                    resource.title.as_deref(),
                    resource.description.as_deref(),
                ));
            }
            Ok(tools)
        }
        DelegationTarget::LocalToolsScope(_) => Ok(builtin_local_tool_catalog()),
    }
}

#[derive(Clone, Debug, PartialEq)]
struct SanitizedMcpArguments {
    arguments: Option<Value>,
    dropped_keys: Vec<String>,
    rejection_reason: Option<String>,
}

impl SanitizedMcpArguments {
    fn rejected(reason: String) -> Self {
        Self {
            arguments: None,
            dropped_keys: Vec::new(),
            rejection_reason: Some(reason),
        }
    }
}

fn sanitize_mcp_tool_arguments(
    registered_tools: &[ToolDescriptor],
    target: &McpCapabilityTarget,
    arguments: &Value,
) -> Result<SanitizedMcpArguments, String> {
    let tool = registered_tools
        .iter()
        .find(|tool| {
            tool.family == crate::tools::ToolFamily::McpTool
                && tool.server_name.as_ref() == Some(&target.server_name)
                && tool.name == target.capability_id
        })
        .ok_or_else(|| {
            format!(
                "no registered MCP tool descriptor found for `{}` on server `{}`",
                target.capability_id, target.server_name
            )
        })?;

    sanitize_object_arguments(&tool.input_schema, arguments)
}

fn sanitize_object_arguments(
    schema: &Value,
    arguments: &Value,
) -> Result<SanitizedMcpArguments, String> {
    let Value::Object(argument_map) = arguments else {
        return Err("MCP tool arguments must be a JSON object".to_owned());
    };

    let Value::Object(schema_map) = schema else {
        return Ok(SanitizedMcpArguments {
            arguments: Some(arguments.clone()),
            dropped_keys: Vec::new(),
            rejection_reason: None,
        });
    };

    let properties = schema_map
        .get("properties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let required = schema_map
        .get("required")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();

    let mut sanitized = serde_json::Map::new();
    let mut dropped_keys = Vec::new();
    let mut present_keys = argument_map.keys().cloned().collect::<Vec<_>>();
    present_keys.sort();

    for key in present_keys {
        let value = argument_map
            .get(&key)
            .cloned()
            .ok_or_else(|| format!("missing argument value for `{key}`"))?;
        let Some(property_schema) = properties.get(&key) else {
            dropped_keys.push(key);
            continue;
        };

        if !json_value_matches_schema(&value, property_schema) {
            return Err(format!(
                "argument `{}` does not match the selected tool schema",
                key
            ));
        }
        sanitized.insert(key, value);
    }

    for key in required {
        if !sanitized.contains_key(key) {
            return Err(format!(
                "required argument `{}` is missing for the selected tool",
                key
            ));
        }
    }

    Ok(SanitizedMcpArguments {
        arguments: Some(Value::Object(sanitized)),
        dropped_keys,
        rejection_reason: None,
    })
}

fn json_value_matches_schema(value: &Value, schema: &Value) -> bool {
    match schema {
        Value::Object(map) => {
            if let Some(any_of) = map.get("anyOf").and_then(Value::as_array) {
                let matches_any_variant = any_of
                    .iter()
                    .any(|variant| json_value_matches_schema(value, variant));
                return matches_any_variant;
            }

            match map.get("type").and_then(Value::as_str) {
                Some("string") => value.is_string(),
                Some("integer") => value.as_i64().is_some() || value.as_u64().is_some(),
                Some("number") => value.as_f64().is_some(),
                Some("boolean") => value.is_boolean(),
                Some("object") => value.is_object(),
                Some("array") => value.is_array(),
                Some("null") => value.is_null(),
                Some(_) | None => true,
            }
        }
        _ => true,
    }
}

fn emit_mcp_argument_validation_debug(
    sink: &mut dyn RuntimeDebugSink,
    turn_id: &TurnId,
    step_id: &StepId,
    executor: &RuntimeExecutor,
    target: &McpCapabilityTarget,
    sanitized: &SanitizedMcpArguments,
    raw_arguments: &Value,
) {
    if sanitized.dropped_keys.is_empty() && sanitized.rejection_reason.is_none() {
        return;
    }

    let summary = if let Some(reason) = &sanitized.rejection_reason {
        format!(
            "rejected MCP arguments for {}::{}",
            target.server_name, target.capability_id
        )
        .to_owned()
            + &format!(" ({reason})")
    } else {
        format!(
            "sanitized MCP arguments for {}::{}",
            target.server_name, target.capability_id
        )
    };

    sink.record_raw_artifact(RuntimeRawArtifact {
        turn_id: turn_id.clone(),
        step_id: Some(step_id.clone()),
        occurred_at: SystemTime::now(),
        kind: RuntimeRawArtifactKind::PolicyDecision,
        source: "mcp_argument_validator".to_owned(),
        executor: executor.clone(),
        summary: Some(summary),
        payload: serde_json::json!({
            "server_name": target.server_name.to_string(),
            "tool_name": target.capability_id,
            "raw_arguments": raw_arguments,
            "sanitized_arguments": sanitized.arguments,
            "dropped_keys": sanitized.dropped_keys,
            "rejection_reason": sanitized.rejection_reason,
        }),
    });
}

fn canonicalize_action_arguments(arguments: &serde_json::Value) -> String {
    let mut value = arguments.clone();
    canonicalize_json_value(&mut value);
    value.to_string()
}

fn prompt_section_summary(prompt: &crate::state::PromptSnapshot) -> String {
    prompt
        .sections
        .iter()
        .map(|section| format!("{}:{}c", section.title, section.content.chars().count()))
        .collect::<Vec<_>>()
        .join(", ")
}

fn model_step_decision_summary(decision: &ModelStepDecision) -> String {
    match decision {
        ModelStepDecision::PlanningComplete { outcome } => format!(
            "planning_complete: todo_required={}, summary={}",
            outcome.todo_required,
            truncate_for_log(&outcome.planning_summary, 120)
        ),
        ModelStepDecision::Final { content } => {
            format!("final_answer: {}", truncate_for_log(content, 160))
        }
        ModelStepDecision::DelegateSubagent { delegation } => format!(
            "delegate_subagent: type={}, target={}, goal={}",
            delegation.subagent_type,
            delegation.target.summary(),
            truncate_for_log(&delegation.goal, 120)
        ),
    }
}

fn subagent_decision_summary(decision: &SubagentDecision) -> String {
    match decision {
        SubagentDecision::Done { summary } => {
            format!("done: {}", truncate_for_log(summary, 160))
        }
        SubagentDecision::Partial { summary, reason } => format!(
            "partial: summary={}, reason={}",
            truncate_for_log(summary, 120),
            truncate_for_log(reason, 120)
        ),
        SubagentDecision::CannotExecute { reason } => {
            format!("cannot_execute: {}", truncate_for_log(reason, 160))
        }
        SubagentDecision::McpToolCall {
            server_name,
            tool_name,
            ..
        } => format!("mcp_tool_call: {server_name}/{tool_name}"),
        SubagentDecision::McpResourceRead {
            server_name,
            resource_uri,
        } => format!(
            "mcp_resource_read: {server_name}/{}",
            truncate_for_log(resource_uri, 120)
        ),
        SubagentDecision::LocalToolCall { tool_name, .. } => {
            format!("local_tool_call: {tool_name}")
        }
    }
}

fn truncate_for_log(text: &str, max_chars: usize) -> String {
    let mut normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.chars().count() > max_chars {
        normalized = normalized.chars().take(max_chars).collect::<String>() + "...";
    }
    normalized
}

fn canonicalize_json_value(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            let mut entries = map
                .iter()
                .map(|(key, value)| {
                    let mut normalized = value.clone();
                    canonicalize_json_value(&mut normalized);
                    (key.clone(), normalized)
                })
                .collect::<Vec<_>>();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            map.clear();
            for (key, normalized) in entries {
                map.insert(key, normalized);
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                canonicalize_json_value(value);
            }
        }
        _ => {}
    }
}

fn should_skip_final_answer_render(answer_brief: &str) -> bool {
    let trimmed = answer_brief.trim();
    if trimmed.is_empty() || trimmed.chars().count() > 500 {
        return false;
    }
    if trimmed.contains("```") {
        return false;
    }
    !trimmed.lines().any(|line| {
        let line = line.trim();
        line.starts_with('|') || line.ends_with('|')
    })
}

fn validate_masked_mcp_call(
    servers: &HashMap<ServerName, PreparedServer>,
    tool_mask_plan: &crate::policy::ToolMaskPlan,
    requested_server_name: &ServerName,
    capability_kind: CapabilityKind,
    capability_id: &str,
) -> Result<McpCapabilityTarget, String> {
    let is_allowed = match capability_kind {
        CapabilityKind::Tool => tool_mask_plan.allowed_mcp_tools.iter().any(|tool| {
            tool.server_name == requested_server_name.to_string() && tool.tool_name == capability_id
        }),
        CapabilityKind::Resource => tool_mask_plan.allowed_mcp_resources.iter().any(|resource| {
            resource.server_name == requested_server_name.to_string()
                && resource.resource_uri == capability_id
        }),
    };
    if !is_allowed {
        return Err(format!(
            "MCP {} `{}` on server `{}` is not allowed in this step",
            match capability_kind {
                CapabilityKind::Tool => "tool",
                CapabilityKind::Resource => "resource",
            },
            capability_id,
            requested_server_name
        ));
    }

    let server = servers
        .get(requested_server_name)
        .ok_or_else(|| format!("unknown MCP server `{requested_server_name}` requested"))?;
    if let Some(reason) = disabled_mcp_capability_reason(server, &capability_kind, capability_id) {
        return Err(reason);
    }
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

fn disabled_mcp_capability_reason(
    server: &PreparedServer,
    capability_kind: &CapabilityKind,
    capability_id: &str,
) -> Option<String> {
    if capability_kind == &CapabilityKind::Resource
        && server.full_catalog.server.protocol_name == "postgres-mcp"
        && capability_id.starts_with("postgres://")
        && capability_id.ends_with("/schema")
    {
        return Some(POSTGRES_SCHEMA_RESOURCE_DISABLED_REASON.to_owned());
    }
    None
}

fn render_mcp_server_guidance(server: &PreparedServer) -> String {
    if server.full_catalog.server.protocol_name == "postgres-mcp" {
        return "- Do not use `postgres://.../schema` resource reads on this server; use the `query` tool with simple SELECT-based discovery instead.\n- If MCP calls keep failing, stop and return `partial` instead of retrying similar actions indefinitely.\n".to_owned();
    }
    String::new()
}

fn repeated_mcp_failure_detail(
    target: &McpCapabilityTarget,
    executed_action_count: u32,
    total_mcp_error_count: u32,
    captured: &CapturedMcpResult,
) -> String {
    let last_error = captured
        .error
        .as_deref()
        .unwrap_or(&captured.result_summary);
    format!(
        "repeated MCP failures forced an early stop after {total_mcp_error_count} errors and {executed_action_count} delegated MCP actions. Last failure on server `{}` capability `{}`: {last_error}",
        target.server_name, target.capability_id
    )
}

fn repeated_invalid_mcp_argument_detail(
    target: &McpCapabilityTarget,
    executed_action_count: u32,
    total_mcp_error_count: u32,
) -> String {
    format!(
        "repeated invalid MCP tool arguments forced an early stop after {total_mcp_error_count} validation errors and {executed_action_count} delegated MCP actions. Last validation failure on server `{}` capability `{}`.",
        target.server_name, target.capability_id
    )
}

fn repeated_local_tool_failure_detail(
    tool_name: &crate::state::LocalToolName,
    executed_action_count: u32,
    total_local_tool_error_count: u32,
    last_error: &str,
) -> String {
    format!(
        "repeated local tool failures forced an early stop after {total_local_tool_error_count} errors and {executed_action_count} delegated actions. Last failure on local tool `{}`: {last_error}",
        tool_name
    )
}

fn immediate_partial_detail_for_local_tool_error(
    tool_name: &crate::state::LocalToolName,
    error_text: &str,
) -> Option<String> {
    if tool_name.as_str() != "bash" {
        return None;
    }
    if error_text.contains(
        "cannot be opened while the current actionable todo is `Generate a clear, engaging data blog post with narrative writing, charts, and tables, using deep analysis in Python with pandas and numpy to surface insights.`",
    ) {
        return Some(
            "The delegated run tried to open the deterministic HTML report before completing the HTML-generation todo. Return control, then generate the report first and print its path only on the next todo step.".to_owned(),
        );
    }
    if error_text.contains(
        "cannot be written from bash while the current actionable todo is `Print the path of the generated HTML file.`",
    ) || error_text.contains(
        "cannot be written while the current actionable todo is `Print the path of the generated HTML file.`",
    ) {
        return Some(
            "The delegated run tried to regenerate the deterministic HTML report even though only the html-path-print todo remains. Return control and print the existing report path instead of writing it again.".to_owned(),
        );
    }
    if error_text.contains("cannot be written from bash while the current actionable todo is")
        || error_text.contains("cannot be written while the current actionable todo is")
    {
        return Some(
            "The delegated run tried to write the deterministic HTML report out of todo order. Return control and resume from the current actionable todo instead of skipping ahead.".to_owned(),
        );
    }
    None
}

fn subagent_result_message(
    subagent_type: &str,
    status: &str,
    executed_action_count: u32,
    detail: String,
    tool_mask: Option<crate::policy::ToolMaskPlan>,
) -> MessageRecord {
    MessageRecord::SubAgentResult(crate::state::SubAgentResultMessageRecord {
        message_id: MessageId::new(),
        timestamp: SystemTime::now(),
        subagent_type: subagent_type.to_owned(),
        status: status.to_owned(),
        executed_action_count,
        detail,
        tool_mask,
    })
}

fn subagent_budget_result(
    subagent_type: &str,
    executed_action_count: u32,
    reason: String,
    tool_mask: Option<crate::policy::ToolMaskPlan>,
) -> MessageRecord {
    let status = if executed_action_count > 0 {
        "partial"
    } else {
        "cannot_execute"
    };
    subagent_result_message(
        subagent_type,
        status,
        executed_action_count,
        reason,
        tool_mask,
    )
}

fn subagent_policy_result(
    subagent_type: &str,
    executed_action_count: u32,
    reason: String,
    tool_mask: Option<crate::policy::ToolMaskPlan>,
) -> MessageRecord {
    let status = if executed_action_count > 0 {
        "partial"
    } else {
        "cannot_execute"
    };
    subagent_result_message(
        subagent_type,
        status,
        executed_action_count,
        reason,
        tool_mask,
    )
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

fn emit_tool_mask_debug(
    sink: &mut dyn RuntimeDebugSink,
    turn_id: &TurnId,
    step_id: &StepId,
    executor: &RuntimeExecutor,
    plan: &crate::policy::ToolMaskPlan,
    registered_tools: &[ToolDescriptor],
) {
    sink.record(RuntimeEvent::ToolMaskEvaluated {
        turn_id: turn_id.clone(),
        step_id: step_id.clone(),
        executor: executor.clone(),
        at: SystemTime::now(),
        enforcement_mode: plan.enforcement_mode,
        allowed_tool_ids: plan.allowed_tool_ids.clone(),
        denied_tool_ids: plan.denied_tool_ids.clone(),
        decisions: plan.decisions.clone(),
    });
    sink.record_raw_artifact(RuntimeRawArtifact {
        turn_id: turn_id.clone(),
        step_id: Some(step_id.clone()),
        occurred_at: SystemTime::now(),
        kind: RuntimeRawArtifactKind::PolicyDecision,
        source: "tool_policy_engine".to_owned(),
        executor: executor.clone(),
        summary: Some(format!(
            "tool policy evaluated ({} allowed / {} denied)",
            plan.allowed_tool_ids.len(),
            plan.denied_tool_ids.len()
        )),
        payload: serde_json::json!({
            "plan": plan,
            "registered_tools": registered_tools,
        }),
    });
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
    let should_cycle_success = should_cycle_mcp_connection_after_success(server, tool_name);
    let mut retried_postgres_session_error = false;
    loop {
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
                    arguments.clone(),
                ),
        )
        .await;
        match result {
            Ok(Ok(value)) => {
                if value.is_error {
                    let result_summary = summarize_tool_result(&value);
                    if !retried_postgres_session_error
                        && should_retry_postgres_query_session_error(
                            server,
                            tool_name,
                            &result_summary,
                        )
                    {
                        warn!(
                            server = %server_name,
                            tool = tool_name,
                            "postgres MCP query hit a transaction/session error; resetting connection and retrying once"
                        );
                        server.connection.take();
                        retried_postgres_session_error = true;
                        continue;
                    }
                    warn!(
                        server = %server_name,
                        tool = tool_name,
                        "MCP tool returned isError=true; resetting connection"
                    );
                    server.connection.take();
                } else if should_cycle_success {
                    info!(
                        server = %server_name,
                        tool = tool_name,
                        "completed MCP tool call; cycling connection for postgres session hygiene"
                    );
                    server.connection.take();
                } else {
                    info!(
                        server = %server_name,
                        tool = tool_name,
                        "completed MCP tool call"
                    );
                }
                return Ok(value);
            }
            Ok(Err(error)) => {
                warn!(
                    server = %server_name,
                    tool = tool_name,
                    error = %error,
                    "MCP tool call failed; resetting connection"
                );
                server.connection.take();
                return Err(RuntimeError::McpClient(error));
            }
            Err(_) => {
                warn!(
                    server = %server_name,
                    tool = tool_name,
                    timeout_ms = timeout_duration.as_millis() as u64,
                    "MCP tool call timed out; resetting connection"
                );
                server.connection.take();
                return Err(RuntimeError::Timeout("MCP call timed out".to_owned()));
            }
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

fn duplicate_mcp_action_feedback_message(
    target: &McpCapabilityTarget,
    action_kind: &str,
    duplicate_threshold: u32,
) -> MessageRecord {
    let result_summary = format!(
        "duplicate MCP {action_kind} `{}` on server `{}` exceeded duplicate threshold {duplicate_threshold} and was blocked before execution. Use earlier results for this exact action, or choose a different MCP action with materially different arguments.",
        target.capability_id, target.server_name
    );
    MessageRecord::McpResult(McpResultMessageRecord {
        message_id: MessageId::new(),
        timestamp: SystemTime::now(),
        target: target.clone(),
        result_summary: result_summary.clone(),
        error: Some(result_summary),
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
    turn_policy_context: &TurnPolicyPromptContext,
    todo_context: Option<&TodoPromptContext>,
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
    prompt.push_str(&format!(
        "Client: {client}\nFormat: {format}\n{format_rules}\n\n"
    ));
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
    prompt.push_str("\nTurn policy:\n");
    prompt.push_str(&render_turn_policy_context(turn_policy_context));
    prompt.push('\n');
    if let Some(todo_context) = todo_context {
        prompt.push_str("\nCurrent turn todo plan:\n");
        prompt.push_str(&render_todo_context(todo_context));
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
    if request.limits.max_duplicate_mcp_calls_per_invocation == 0 {
        return Err(RuntimeError::Validation(
            "max_duplicate_mcp_calls_per_invocation must be at least 1".to_owned(),
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

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        path::{Path, PathBuf},
        time::SystemTime,
    };

    use mcp_config::{McpRegistry, McpServerConfig, McpTransportConfig};
    use mcp_metadata::CapabilityKind;
    use mcp_metadata::{
        CURRENT_SCHEMA_VERSION, FullResourceMetadata, FullToolMetadata, McpCapabilityFamilies,
        McpCapabilityFamilySummary, McpFullCatalog, McpMinimalCatalog, McpServerMetadata,
        MinimalResourceMetadata, MinimalToolMetadata,
    };

    use crate::{
        ids::MessageId,
        model::{
            ExecutionHandoff, PlannedSource, PlannedSourceKind, PlanningFact, PlanningOutcome,
            TurnPhase,
        },
        prompt::TurnPolicyPromptContext,
        state::{
            ConversationMessage, ConversationRole, DelegationTarget, LocalToolName,
            LocalToolResultMessageRecord, LocalToolsScopeTarget, McpCallMessageRecord,
            McpCapabilityTarget, McpResultMessageRecord, McpServerScopeTarget, MessageRecord,
            ServerName, SubAgentCallMessageRecord, SubAgentResultMessageRecord,
        },
        subagent::{ConfiguredSubagent, SubagentRegistry},
        todo::{
            GENERIC_STARTER_TODO, MANDATORY_TODO_GENERATE_HTML, MANDATORY_TODO_OPEN_HTML, TodoList,
            TodoStatus,
        },
    };

    use super::{
        FinalTodoAdvance, McpSession, POSTGRES_SCHEMA_RESOURCE_DISABLED_REASON, PreparedServer,
        auto_advance_summary_todo_from_final_content, build_subagent_prompt,
        build_subagent_tool_catalog, disabled_mcp_capability_feedback_for_delegation,
        disabled_mcp_capability_reason, generic_starter_replan_feedback_for_delegation,
        html_todo_feedback_for_delegation, immediate_partial_detail_for_local_tool_error,
        incomplete_todo_final_feedback, invalid_subagent_feedback,
        mcp_collection_feedback_for_delegation, planning_todo_items_feedback,
        reconcile_active_todo_from_turn_trace, reconcile_mandatory_todos_from_turn_trace,
        reconcile_pending_mcp_todos_from_turn_trace, render_confirmed_table_context,
        render_recent_mcp_results_context, sanitize_mcp_tool_arguments,
        should_cycle_mcp_connection_after_success, should_require_todos_for_turn,
        should_retry_postgres_query_session_error, sync_mcp_todo_after_delegation,
        sync_mcp_todo_before_delegation, tool_executor_mcp_todo_partial_feedback,
        validate_delegation_subagent,
    };

    #[test]
    fn planning_todo_items_feedback_rejects_unhinted_new_todos_without_failing_turn() {
        let todo_path = PathBuf::from("/tmp/nonexistent-planning-todos.txt");
        let outcome = PlanningOutcome {
            planning_complete: true,
            answer_brief: String::new(),
            todo_required: true,
            planning_summary: "Plan a customer user analysis.".to_owned(),
            selected_sources: vec![],
            discovered_facts: vec![],
            execution_strategy: "Inspect the users table, analyze, then report.".to_owned(),
            todo_items: vec![
                "Inspect the customer users schema and tiny sample.".to_owned(),
                "[main-agent] Summarize findings.".to_owned(),
            ],
            risks_and_constraints: vec![],
        };

        let feedback = planning_todo_items_feedback(&outcome, false, &todo_path)
            .expect("feedback should compute")
            .expect("unhinted todo should request feedback");

        assert!(feedback.contains("planning_complete.todo_items"));
        assert!(feedback.contains("[mcp-executor]"));
        assert!(feedback.contains("[tool-executor]"));
        assert!(feedback.contains("[main-agent]"));
    }

    #[test]
    fn planning_todo_items_feedback_accepts_hinted_new_todos() {
        let todo_path = PathBuf::from("/tmp/nonexistent-planning-todos.txt");
        let outcome = PlanningOutcome {
            planning_complete: true,
            answer_brief: String::new(),
            todo_required: true,
            planning_summary: "Plan a customer user analysis.".to_owned(),
            selected_sources: vec![],
            discovered_facts: vec![],
            execution_strategy: "Inspect the users table, analyze, then report.".to_owned(),
            todo_items: vec![
                "[mcp-executor] Inspect the customer users schema and tiny sample.".to_owned(),
                "[main-agent] Summarize findings.".to_owned(),
            ],
            risks_and_constraints: vec![],
        };

        let feedback = planning_todo_items_feedback(&outcome, false, &todo_path)
            .expect("feedback should compute");

        assert!(feedback.is_none());
    }

    #[test]
    fn recent_mcp_results_context_renders_recent_results_and_guidance() {
        let messages = vec![MessageRecord::McpResult(McpResultMessageRecord {
            message_id: MessageId::new(),
            timestamp: SystemTime::now(),
            target: McpCapabilityTarget {
                server_name: ServerName::new("ex-vol").expect("valid server"),
                capability_kind: CapabilityKind::Tool,
                capability_id: "run_select_query".to_owned(),
            },
            result_summary: "{\"columns\":[\"week_start\",\"new_users\"],\"rows\":[[\"2026-03-30\",14],[\"2026-04-06\",8]]}".to_owned(),
            error: None,
        })];

        let rendered = render_recent_mcp_results_context(&messages);

        assert!(rendered.contains("## Recent Computed Results"));
        assert!(rendered.contains("server=ex-vol"));
        assert!(rendered.contains("target=run_select_query"));
        assert!(rendered.contains("\"week_start\""));
        assert!(rendered.contains("write the relevant rows into a local file"));
    }

    #[test]
    fn local_tools_subagent_prompt_includes_recent_mcp_results() {
        let registry_path = workspace_root().join("config").join("subagents.json");
        let configured = ConfiguredSubagent {
            subagent_type: "tool-executor".to_owned(),
            display_name: "Tool Executor".to_owned(),
            purpose: "Complete delegated local tool work".to_owned(),
            when_to_use: "For local scripts".to_owned(),
            target_requirements: "local scope".to_owned(),
            result_summary: "Returns local tool work".to_owned(),
            prompt_path: PathBuf::from("subagents/tool-executor.prompt.md"),
            enabled: true,
            model_name: None,
        };
        let messages = vec![MessageRecord::McpResult(McpResultMessageRecord {
            message_id: MessageId::new(),
            timestamp: SystemTime::now(),
            target: McpCapabilityTarget {
                server_name: ServerName::new("ex-vol").expect("valid server"),
                capability_kind: CapabilityKind::Tool,
                capability_id: "run_select_query".to_owned(),
            },
            result_summary: "{\"columns\":[\"week_start\",\"new_users\"],\"rows\":[[\"2026-03-30\",14],[\"2026-04-06\",8]]}".to_owned(),
            error: None,
        })];

        let prompt = build_subagent_prompt(
            &registry_path,
            &configured,
            &McpSession {
                servers: HashMap::new(),
            },
            TurnPhase::Execution,
            &DelegationTarget::LocalToolsScope(LocalToolsScopeTarget {
                scope: "workspace".to_owned(),
            }),
            "Create a weekly new users chart",
            "give me a visualization for this",
            Path::new("/tmp/arka-session"),
            Path::new("/tmp/arka-session/turn-1/todos.txt"),
            &TurnPolicyPromptContext {
                phase: TurnPhase::Execution,
                html_output_path: PathBuf::from("/tmp/arka-session/outputs/turn-1-report.html"),
                force_todo_file: false,
                execution_todo_required: Some(false),
            },
            None,
            &messages,
        )
        .expect("prompt should build");

        assert!(prompt.contains("## Recent Computed Results"));
        assert!(prompt.contains("\"new_users\""));
        assert!(prompt.contains("Assume `pandas` and `numpy` are available"));
        assert!(prompt.contains("## Local Tools"));
    }

    #[test]
    fn execution_subagent_prompt_includes_execution_handoff_and_todo_context() {
        let registry_path = workspace_root().join("config").join("subagents.json");
        let configured = ConfiguredSubagent {
            subagent_type: "tool-executor".to_owned(),
            display_name: "Tool Executor".to_owned(),
            purpose: "Complete delegated local tool work".to_owned(),
            when_to_use: "For local scripts".to_owned(),
            target_requirements: "local scope".to_owned(),
            result_summary: "Returns local tool work".to_owned(),
            prompt_path: PathBuf::from("subagents/tool-executor.prompt.md"),
            enabled: true,
            model_name: None,
        };
        let temp_dir = std::env::temp_dir().join(format!(
            "agent-runtime-handoff-prompt-test-{}",
            std::process::id()
        ));
        let turn_dir = temp_dir.join("turn-1");
        std::fs::create_dir_all(&turn_dir).expect("turn dir should exist");
        let todo_path = turn_dir.join("todos.txt");
        TodoList::initialize(&[
            "[mcp-executor] Analyze weekly active users from analytics.events.".to_owned(),
        ])
        .expect("todo init should succeed")
        .save_to_path(&todo_path)
        .expect("todo file should write");
        let handoff = ExecutionHandoff {
            todo_required: true,
            answer_brief: String::new(),
            summary: "Planning selected the analytics event stream for a weekly activity report."
                .to_owned(),
            selected_sources: vec![PlannedSource {
                source_kind: PlannedSourceKind::McpServer,
                source_id: "postgres-mcp:analytics".to_owned(),
                rationale: "Contains the analytics.events table needed for activity metrics."
                    .to_owned(),
            }],
            key_facts: vec![PlanningFact {
                fact: "analytics.events has created_at and user_id fields.".to_owned(),
                evidence_source_ids: vec!["postgres-mcp:analytics".to_owned()],
            }],
            execution_strategy:
                "Query weekly user counts, materialize rows locally, then render the HTML report."
                    .to_owned(),
            risks_and_constraints: vec![
                "Use only completed events when filtering status.".to_owned(),
            ],
            todo_path: Some(todo_path.clone()),
        };

        let execution_prompt = build_subagent_prompt(
            &registry_path,
            &configured,
            &McpSession {
                servers: HashMap::new(),
            },
            TurnPhase::Execution,
            &DelegationTarget::LocalToolsScope(LocalToolsScopeTarget {
                scope: "workspace".to_owned(),
            }),
            "Create the weekly activity report",
            "Show weekly active users",
            &temp_dir,
            &todo_path,
            &TurnPolicyPromptContext {
                phase: TurnPhase::Execution,
                html_output_path: temp_dir.join("outputs").join("turn-1-report.html"),
                force_todo_file: false,
                execution_todo_required: Some(true),
            },
            Some(&handoff),
            &[],
        )
        .expect("execution prompt should build");

        assert!(execution_prompt.contains("## Execution Handoff"));
        assert!(execution_prompt.contains("weekly activity report"));
        assert!(execution_prompt.contains("postgres-mcp:analytics"));
        assert!(execution_prompt.contains("created_at and user_id"));
        assert!(execution_prompt.contains("Query weekly user counts"));
        assert!(execution_prompt.contains("Use only completed events"));
        assert!(execution_prompt.contains("## Current Turn Todo Plan"));
        assert!(execution_prompt.contains("Analyze weekly active users"));

        let planning_prompt = build_subagent_prompt(
            &registry_path,
            &configured,
            &McpSession {
                servers: HashMap::new(),
            },
            TurnPhase::Planning,
            &DelegationTarget::LocalToolsScope(LocalToolsScopeTarget {
                scope: "workspace".to_owned(),
            }),
            "Inspect local inputs",
            "Show weekly active users",
            &temp_dir,
            &todo_path,
            &TurnPolicyPromptContext {
                phase: TurnPhase::Planning,
                html_output_path: temp_dir.join("outputs").join("turn-1-report.html"),
                force_todo_file: false,
                execution_todo_required: None,
            },
            Some(&handoff),
            &[],
        )
        .expect("planning prompt should build");

        assert!(!planning_prompt.contains("## Execution Handoff"));
    }

    #[test]
    fn local_tools_subagent_prompt_warns_not_to_open_during_html_generation_todo() {
        let registry_path = workspace_root().join("config").join("subagents.json");
        let configured = ConfiguredSubagent {
            subagent_type: "tool-executor".to_owned(),
            display_name: "Tool Executor".to_owned(),
            purpose: "Complete delegated local tool work".to_owned(),
            when_to_use: "For local scripts".to_owned(),
            target_requirements: "local scope".to_owned(),
            result_summary: "Returns local tool work".to_owned(),
            prompt_path: PathBuf::from("subagents/tool-executor.prompt.md"),
            enabled: true,
            model_name: None,
        };
        let temp_dir = std::env::temp_dir().join(format!(
            "agent-runtime-html-generation-prompt-test-{}",
            std::process::id()
        ));
        let turn_dir = temp_dir.join("turn-1");
        std::fs::create_dir_all(&turn_dir).expect("turn dir should exist");
        let todo_path = turn_dir.join("todos.txt");
        TodoList::parse(
            "1. [completed] Compute the IPL 2025 balance metrics.\n2. [in_progress] Generate a clear, engaging data blog post with narrative writing, charts, and tables, using deep analysis in Python with pandas and numpy to surface insights.\n3. [pending] Print the path of the generated HTML file.\n",
        )
        .expect("seed todos should parse")
        .save_to_path(&todo_path)
        .expect("todo file should write");

        let prompt = build_subagent_prompt(
            &registry_path,
            &configured,
            &McpSession {
                servers: HashMap::new(),
            },
            TurnPhase::Execution,
            &DelegationTarget::LocalToolsScope(LocalToolsScopeTarget {
                scope: "workspace".to_owned(),
            }),
            "Generate the deterministic HTML report",
            "Show the most balanced IPL 2025 side",
            &temp_dir,
            &todo_path,
            &TurnPolicyPromptContext {
                phase: TurnPhase::Execution,
                html_output_path: temp_dir.join("outputs").join("turn-1-report.html"),
                force_todo_file: false,
                execution_todo_required: Some(true),
            },
            None,
            &[],
        )
        .expect("prompt should build");

        assert!(prompt.contains("current actionable todo is the HTML-generation step"));
        assert!(prompt.contains("Write the final deterministic HTML report directly"));
        assert!(prompt.contains("Avoid fragile intermediate report-builder scripts"));
        assert!(prompt.contains("avoid `itertuples()` field aliases"));
        assert!(prompt.contains("do not print the report path in this delegation"));
    }

    #[test]
    fn reconcile_mandatory_todos_marks_html_and_path_print_steps_completed() {
        let working_directory = test_fixtures_root();
        let turn_dir = working_directory.join("turn-1");
        std::fs::create_dir_all(&turn_dir).expect("turn dir should exist");
        let todo_path = turn_dir.join("todos.txt");
        let html_output_path = working_directory.join("outputs").join("turn-1-report.html");
        std::fs::create_dir_all(
            html_output_path
                .parent()
                .expect("html output parent should exist"),
        )
        .expect("outputs dir should exist");
        std::fs::write(
            &html_output_path,
            "<html><body><table><tr><td>report</td></tr></table></body></html>",
        )
        .expect("html report should exist");

        let mut todos = TodoList::initialize(&[
            "[mcp-executor] Determine Virat Kohli's IPL 2025 runs.".to_owned(),
        ])
        .expect("todo init should succeed");
        todos
            .set_status(1, crate::todo::TodoStatus::InProgress)
            .expect("item 1 should enter in_progress");
        todos
            .set_status(1, crate::todo::TodoStatus::Completed)
            .expect("item 1 should complete");
        todos
            .save_to_path(&todo_path)
            .expect("todo file should write");

        let messages = vec![
            MessageRecord::LocalToolResult(LocalToolResultMessageRecord {
                message_id: MessageId::new(),
                timestamp: SystemTime::now(),
                tool_name: LocalToolName::new("write_file").expect("valid tool"),
                status: "ok".to_owned(),
                result_summary: format!("wrote 123 bytes to {}", html_output_path.display()),
                error: None,
            }),
            MessageRecord::LocalToolResult(LocalToolResultMessageRecord {
                message_id: MessageId::new(),
                timestamp: SystemTime::now(),
                tool_name: LocalToolName::new("bash").expect("valid tool"),
                status: "ok".to_owned(),
                result_summary: "command: printf '%s\\n' outputs/turn-1-report.html\nexit_code: 0\nstdout:\noutputs/turn-1-report.html\n".to_owned(),
                error: None,
            }),
        ];

        reconcile_mandatory_todos_from_turn_trace(
            &todo_path,
            &html_output_path,
            &working_directory,
            &messages,
        )
        .expect("reconciliation should succeed");

        let reconciled = TodoList::load_from_path(&todo_path).expect("todo file should reload");
        assert_eq!(reconciled.items[1].text, MANDATORY_TODO_GENERATE_HTML);
        assert_eq!(
            reconciled.items[1].status,
            crate::todo::TodoStatus::Completed
        );
        assert_eq!(reconciled.items[2].text, MANDATORY_TODO_OPEN_HTML);
        assert_eq!(
            reconciled.items[2].status,
            crate::todo::TodoStatus::Completed
        );
    }

    #[test]
    fn reconcile_mandatory_todos_marks_path_print_completed_for_absolute_bash_stdout() {
        let working_directory = test_fixtures_root();
        let turn_dir = working_directory.join("turn-quoted-open-reconcile");
        std::fs::create_dir_all(&turn_dir).expect("turn dir should exist");
        let todo_path = turn_dir.join("todos.txt");
        let html_output_path = working_directory
            .join("outputs")
            .join("turn-quoted-open-reconcile-report.html");
        std::fs::create_dir_all(
            html_output_path
                .parent()
                .expect("html output parent should exist"),
        )
        .expect("outputs dir should exist");
        std::fs::write(
            &html_output_path,
            "<html><body><table><tr><td>report</td></tr></table></body></html>",
        )
        .expect("html report should exist");

        let mut todos = TodoList::initialize(&["[main-agent] Prepare the report.".to_owned()])
            .expect("todo init should succeed");
        todos
            .set_status(1, crate::todo::TodoStatus::InProgress)
            .expect("item 1 should enter in_progress");
        todos
            .set_status(1, crate::todo::TodoStatus::Completed)
            .expect("item 1 should complete");
        todos
            .set_status(2, crate::todo::TodoStatus::InProgress)
            .expect("html generation should enter in_progress");
        todos
            .set_status(2, crate::todo::TodoStatus::Completed)
            .expect("html generation should complete");
        todos
            .save_to_path(&todo_path)
            .expect("todo file should write");

        let messages = vec![MessageRecord::LocalToolResult(
            LocalToolResultMessageRecord {
                message_id: MessageId::new(),
                timestamp: SystemTime::now(),
                tool_name: LocalToolName::new("bash").expect("valid tool"),
                status: "ok".to_owned(),
                result_summary: format!(
                    "command: printf '%s\\n' '{}'\nexit_code: 0\nstdout:\n{}\n",
                    html_output_path.display(),
                    html_output_path.display()
                ),
                error: None,
            },
        )];

        reconcile_mandatory_todos_from_turn_trace(
            &todo_path,
            &html_output_path,
            &working_directory,
            &messages,
        )
        .expect("reconciliation should succeed");

        let reconciled = TodoList::load_from_path(&todo_path).expect("todo file should reload");
        assert_eq!(reconciled.items[2].text, MANDATORY_TODO_OPEN_HTML);
        assert_eq!(
            reconciled.items[2].status,
            crate::todo::TodoStatus::Completed
        );
    }

    #[test]
    fn reconcile_mandatory_todos_marks_html_completed_when_report_was_written_via_bash() {
        let working_directory = test_fixtures_root();
        let turn_dir = working_directory.join("turn-1-bash-html");
        std::fs::create_dir_all(&turn_dir).expect("turn dir should exist");
        let todo_path = turn_dir.join("todos.txt");
        let html_output_path = working_directory
            .join("outputs")
            .join("turn-1-bash-html-report.html");
        std::fs::create_dir_all(
            html_output_path
                .parent()
                .expect("html output parent should exist"),
        )
        .expect("outputs dir should exist");
        std::fs::write(&html_output_path, "<html><body>report</body></html>")
            .expect("html report should exist");

        let mut todos =
            TodoList::initialize(&["[mcp-executor] Determine Dhoni's IPL 2025 runs.".to_owned()])
                .expect("todo init should succeed");
        todos
            .set_status(1, crate::todo::TodoStatus::InProgress)
            .expect("item 1 should enter in_progress");
        todos
            .set_status(1, crate::todo::TodoStatus::Completed)
            .expect("item 1 should complete");
        todos
            .save_to_path(&todo_path)
            .expect("todo file should write");

        let messages = vec![
            MessageRecord::LocalToolResult(LocalToolResultMessageRecord {
                message_id: MessageId::new(),
                timestamp: SystemTime::now(),
                tool_name: LocalToolName::new("bash").expect("valid tool"),
                status: "ok".to_owned(),
                result_summary: format!(
                    "command: python3 - <<'PY'\n# wrote {}\nexit_code: 0",
                    html_output_path.display()
                ),
                error: None,
            }),
            MessageRecord::LocalToolResult(LocalToolResultMessageRecord {
                message_id: MessageId::new(),
                timestamp: SystemTime::now(),
                tool_name: LocalToolName::new("bash").expect("valid tool"),
                status: "ok".to_owned(),
                result_summary:
                    "command: printf '%s\\n' outputs/turn-1-bash-html-report.html\nexit_code: 0\nstdout:\noutputs/turn-1-bash-html-report.html\n"
                        .to_owned(),
                error: None,
            }),
        ];

        reconcile_mandatory_todos_from_turn_trace(
            &todo_path,
            &html_output_path,
            &working_directory,
            &messages,
        )
        .expect("reconciliation should succeed");

        let reconciled = TodoList::load_from_path(&todo_path).expect("todo file should reload");
        assert_eq!(reconciled.items[1].text, MANDATORY_TODO_GENERATE_HTML);
        assert_eq!(
            reconciled.items[1].status,
            crate::todo::TodoStatus::Completed
        );
        assert_eq!(reconciled.items[2].text, MANDATORY_TODO_OPEN_HTML);
        assert_eq!(
            reconciled.items[2].status,
            crate::todo::TodoStatus::Completed
        );
    }

    #[test]
    fn reconcile_active_todo_marks_in_progress_item_completed_after_subagent_success() {
        let working_directory = test_fixtures_root();
        let turn_dir = working_directory.join("turn-active-reconcile");
        std::fs::create_dir_all(&turn_dir).expect("turn dir should exist");
        let todo_path = turn_dir.join("todos.txt");

        let todos = TodoList::parse(
            "1. [in_progress] Validate the extracted runs against the available source and prepare a concise summary.\n2. [pending] Generate a clear, engaging data blog post with narrative writing, charts, and tables, using deep analysis in Python with pandas and numpy to surface insights.\n3. [pending] Print the path of the generated HTML file.\n",
        )
        .expect("seed todos should parse");
        todos
            .save_to_path(&todo_path)
            .expect("todo file should write");

        let messages = vec![MessageRecord::SubAgentResult(SubAgentResultMessageRecord {
            message_id: MessageId::new(),
            timestamp: SystemTime::now(),
            subagent_type: "tool-executor".to_owned(),
            status: "completed".to_owned(),
            executed_action_count: 3,
            detail: "Validated the answer and wrote the report".to_owned(),
            tool_mask: None,
        })];

        reconcile_active_todo_from_turn_trace(&todo_path, &messages)
            .expect("reconciliation should succeed");

        let reconciled = TodoList::load_from_path(&todo_path).expect("todo file should reload");
        assert_eq!(
            reconciled.items[0].status,
            crate::todo::TodoStatus::Completed
        );
        assert_eq!(reconciled.items[1].status, crate::todo::TodoStatus::Pending);
    }

    #[test]
    fn reconcile_active_todo_marks_local_inspection_todo_completed_after_partial_tool_executor() {
        let working_directory = test_fixtures_root();
        let turn_dir = working_directory.join("turn-active-partial-inspection");
        std::fs::create_dir_all(&turn_dir).expect("turn dir should exist");
        let todo_path = turn_dir.join("todos.txt");

        let todos = TodoList::parse(
            "1. [in_progress] Understand the user request and inspect available local workspace inputs for the MS Dhoni IPL 2025 breakdown.\n2. [pending] Extract or materialize the relevant IPL 2025/MS Dhoni data into local outputs for reproducible analysis.\n3. [pending] Generate a clear, engaging data blog post with narrative writing, charts, and tables, using deep analysis in Python with pandas and numpy to surface insights.\n4. [pending] Print the path of the generated HTML file.\n",
        )
        .expect("seed todos should parse");
        todos
            .save_to_path(&todo_path)
            .expect("todo file should write");

        let messages = vec![MessageRecord::SubAgentResult(SubAgentResultMessageRecord {
            message_id: MessageId::new(),
            timestamp: SystemTime::now(),
            subagent_type: "tool-executor".to_owned(),
            status: "partial".to_owned(),
            executed_action_count: 1,
            detail:
                "Inspected local workspace inputs and returned control for MCP-backed data collection."
                    .to_owned(),
            tool_mask: None,
        })];

        reconcile_active_todo_from_turn_trace(&todo_path, &messages)
            .expect("reconciliation should succeed");

        let reconciled = TodoList::load_from_path(&todo_path).expect("todo file should reload");
        assert_eq!(
            reconciled.items[0].status,
            crate::todo::TodoStatus::Completed
        );
        assert_eq!(reconciled.items[1].status, crate::todo::TodoStatus::Pending);
    }

    #[test]
    fn reconcile_active_todo_does_not_complete_generic_starter_scaffold() {
        let working_directory = test_fixtures_root();
        let turn_dir = working_directory.join("turn-active-generic-starter");
        std::fs::create_dir_all(&turn_dir).expect("turn dir should exist");
        let todo_path = turn_dir.join("todos.txt");

        let todos = TodoList::parse(
            "1. [in_progress] Understand and complete the user request.\n2. [pending] Generate a clear, engaging data blog post with narrative writing, charts, and tables, using deep analysis in Python with pandas and numpy to surface insights.\n3. [pending] Print the path of the generated HTML file.\n",
        )
        .expect("seed todos should parse");
        todos
            .save_to_path(&todo_path)
            .expect("todo file should write");

        let messages = vec![MessageRecord::SubAgentResult(SubAgentResultMessageRecord {
            message_id: MessageId::new(),
            timestamp: SystemTime::now(),
            subagent_type: "tool-executor".to_owned(),
            status: "completed".to_owned(),
            executed_action_count: 1,
            detail: "Finished delegated work".to_owned(),
            tool_mask: None,
        })];

        reconcile_active_todo_from_turn_trace(&todo_path, &messages)
            .expect("reconciliation should succeed without mutating the starter");

        let reconciled = TodoList::load_from_path(&todo_path).expect("todo file should reload");
        assert_eq!(
            reconciled.items[0].status,
            crate::todo::TodoStatus::InProgress
        );
    }

    #[test]
    fn reconcile_pending_mcp_todos_advances_multiple_grounded_collection_steps() {
        let working_directory = test_fixtures_root();
        let turn_dir = working_directory.join("turn-mcp-trace-reconcile");
        std::fs::create_dir_all(&turn_dir).expect("turn dir should exist");
        let todo_path = turn_dir.join("todos.txt");

        let todos = TodoList::parse(
            "1. [completed] [mcp-executor] Inspect the IPL schema for players, deliveries, matches, and innings to identify the exact fields needed for batting analysis.\n2. [pending] [mcp-executor] Confirm Virat Kohli's player identifier/name in the players table and verify whether batting events are recorded in deliveries or innings.\n3. [pending] [mcp-executor] Run a compact set of SQL queries to compute his overall IPL batting summary, season-wise run totals, dismissals, strike rate, and high-level consistency indicators.\n4. [pending] [main-agent] If useful from the data, add a brief comparison against a couple of career milestones or recent-season trends without overextending the scope.\n5. [pending] [tool-executor] Generate a clear, engaging data blog post with narrative writing, charts, and tables, using deep analysis in Python with pandas and numpy to surface insights.\n6. [pending] [tool-executor] Print the path of the generated HTML file.\n",
        )
        .expect("seed todos should parse");
        todos
            .save_to_path(&todo_path)
            .expect("todo file should write");

        let server_name = ServerName::new("ipl").expect("valid server");
        let messages = vec![
            MessageRecord::SubAgentCall(SubAgentCallMessageRecord {
                message_id: MessageId::new(),
                timestamp: SystemTime::now(),
                subagent_type: "mcp-executor".to_owned(),
                goal: "Analyze Virat Kohli's IPL batting performance using the IPL database."
                    .to_owned(),
                target: DelegationTarget::McpServerScope(McpServerScopeTarget {
                    server_name: server_name.clone(),
                }),
            }),
            MessageRecord::McpCall(McpCallMessageRecord {
                message_id: MessageId::new(),
                timestamp: SystemTime::now(),
                target: McpCapabilityTarget {
                    server_name: server_name.clone(),
                    capability_kind: CapabilityKind::Tool,
                    capability_id: "query".to_owned(),
                },
                arguments: serde_json::json!({
                    "sql": "select player_id, player_name from players where player_name ilike '%Kohli%';"
                }),
            }),
            MessageRecord::McpResult(McpResultMessageRecord {
                message_id: MessageId::new(),
                timestamp: SystemTime::now(),
                target: McpCapabilityTarget {
                    server_name: server_name.clone(),
                    capability_kind: CapabilityKind::Tool,
                    capability_id: "query".to_owned(),
                },
                result_summary:
                    "Confirmed Virat Kohli as player_id 135 and verified batting events are available in deliveries plus innings summaries."
                        .to_owned(),
                error: None,
            }),
            MessageRecord::McpCall(McpCallMessageRecord {
                message_id: MessageId::new(),
                timestamp: SystemTime::now(),
                target: McpCapabilityTarget {
                    server_name: server_name.clone(),
                    capability_kind: CapabilityKind::Tool,
                    capability_id: "query".to_owned(),
                },
                arguments: serde_json::json!({
                    "sql": "select season, runs, dismissals, strike_rate from virat_kohli_ipl_summary;"
                }),
            }),
            MessageRecord::McpResult(McpResultMessageRecord {
                message_id: MessageId::new(),
                timestamp: SystemTime::now(),
                target: McpCapabilityTarget {
                    server_name: server_name.clone(),
                    capability_kind: CapabilityKind::Tool,
                    capability_id: "query".to_owned(),
                },
                result_summary:
                    "Computed Virat Kohli's overall IPL batting summary, season-wise run totals, dismissals, and strike rate."
                        .to_owned(),
                error: None,
            }),
            MessageRecord::SubAgentResult(SubAgentResultMessageRecord {
                message_id: MessageId::new(),
                timestamp: SystemTime::now(),
                subagent_type: "mcp-executor".to_owned(),
                status: "completed".to_owned(),
                executed_action_count: 2,
                detail: "Virat Kohli identified as player_id 135. Overall IPL batting summary and season-wise totals were computed.".to_owned(),
                tool_mask: None,
            }),
        ];

        reconcile_pending_mcp_todos_from_turn_trace(&todo_path, &messages)
            .expect("reconciliation should succeed");

        let reconciled = TodoList::load_from_path(&todo_path).expect("todo file should reload");
        assert_eq!(
            reconciled.items[1].status,
            crate::todo::TodoStatus::Completed
        );
        assert_eq!(
            reconciled.items[2].status,
            crate::todo::TodoStatus::Completed
        );
        assert_eq!(reconciled.items[3].status, crate::todo::TodoStatus::Pending);
    }

    #[test]
    fn reconcile_pending_mcp_todos_advances_season_encoding_confirmation() {
        let working_directory = test_fixtures_root();
        let turn_dir = working_directory.join("turn-mcp-trace-season-confirm");
        std::fs::create_dir_all(&turn_dir).expect("turn dir should exist");
        let todo_path = turn_dir.join("todos.txt");

        let todos = TodoList::parse(
            "1. [completed] [mcp-executor] Inspect the IPL schema for matches, match_teams, innings, deliveries, and teams.\n2. [pending] [mcp-executor] Confirm how IPL 2025 is encoded in the matches data and whether there are any season/date filters needed.\n3. [pending] [mcp-executor] Compute a concrete team-balance metric for IPL 2025 from the available data.\n4. [pending] [main-agent] Rank all sides by the chosen balance metric and identify the top side.\n5. [pending] [tool-executor] Generate a clear, engaging data blog post with narrative writing, charts, and tables, using deep analysis in Python with pandas and numpy to surface insights.\n6. [pending] [tool-executor] Print the path of the generated HTML file.\n",
        )
        .expect("seed todos should parse");
        todos
            .save_to_path(&todo_path)
            .expect("todo file should write");

        let server_name = ServerName::new("ipl").expect("valid server");
        let messages = vec![
            MessageRecord::SubAgentCall(SubAgentCallMessageRecord {
                message_id: MessageId::new(),
                timestamp: SystemTime::now(),
                subagent_type: "mcp-executor".to_owned(),
                goal: "Confirm how IPL 2025 is encoded in the matches data and whether any season/date filters are needed.".to_owned(),
                target: DelegationTarget::McpServerScope(McpServerScopeTarget {
                    server_name: server_name.clone(),
                }),
            }),
            MessageRecord::McpCall(McpCallMessageRecord {
                message_id: MessageId::new(),
                timestamp: SystemTime::now(),
                target: McpCapabilityTarget {
                    server_name: server_name.clone(),
                    capability_kind: CapabilityKind::Tool,
                    capability_id: "query".to_owned(),
                },
                arguments: serde_json::json!({
                    "sql": "select season, count(*) as matches, min(match_date) as first_date, max(match_date) as last_date from matches where season = '2025' group by season;"
                }),
            }),
            MessageRecord::McpResult(McpResultMessageRecord {
                message_id: MessageId::new(),
                timestamp: SystemTime::now(),
                target: McpCapabilityTarget {
                    server_name: server_name.clone(),
                    capability_kind: CapabilityKind::Tool,
                    capability_id: "query".to_owned(),
                },
                result_summary:
                    "Confirmed IPL season 2025 is encoded as season='2025' with 74 matches from 2025-03-22 through 2025-06-03."
                        .to_owned(),
                error: None,
            }),
            MessageRecord::SubAgentResult(SubAgentResultMessageRecord {
                message_id: MessageId::new(),
                timestamp: SystemTime::now(),
                subagent_type: "mcp-executor".to_owned(),
                status: "completed".to_owned(),
                executed_action_count: 1,
                detail: "Confirmed season/date filter for IPL 2025.".to_owned(),
                tool_mask: None,
            }),
        ];

        reconcile_pending_mcp_todos_from_turn_trace(&todo_path, &messages)
            .expect("reconciliation should succeed");

        let reconciled = TodoList::load_from_path(&todo_path).expect("todo file should reload");
        assert_eq!(reconciled.items[1].status, TodoStatus::Completed);
        assert_eq!(reconciled.items[2].status, TodoStatus::Pending);
    }

    #[test]
    fn incomplete_todo_final_feedback_is_specific_for_html_tail_steps() {
        let html_output_path = PathBuf::from("/tmp/arka-session/outputs/turn-1-report.html");

        let html_feedback = incomplete_todo_final_feedback(
            3,
            "pending",
            MANDATORY_TODO_GENERATE_HTML,
            &html_output_path,
        );
        assert!(html_feedback.contains("mandatory deliverable is still incomplete"));
        assert!(html_feedback.contains("todo file exists and is already loaded"));
        assert!(html_feedback.contains("subagent_type: \"tool-executor\""));
        assert!(html_feedback.contains("Emit a `delegate_subagent` decision now"));
        assert!(html_feedback.contains("\"scope\":\"workspace\""));
        assert!(html_feedback.contains("/tmp/arka-session/outputs/turn-1-report.html"));
        assert!(html_feedback.contains(
            "Do not ask `tool-executor` to print the report path in the same delegation"
        ));

        let open_feedback = incomplete_todo_final_feedback(
            4,
            "pending",
            MANDATORY_TODO_OPEN_HTML,
            &html_output_path,
        );
        assert!(open_feedback.contains("report path still needs to be printed"));
        assert!(open_feedback.contains("todo file exists and is already loaded"));
        assert!(open_feedback.contains("Emit a `delegate_subagent` decision now"));
        assert!(open_feedback.contains("print the generated HTML file path"));
    }

    #[test]
    fn auto_advance_summary_todo_from_final_content_marks_summary_step_completed() {
        let working_directory = test_fixtures_root();
        let turn_dir = working_directory.join("turn-auto-advance-summary");
        std::fs::create_dir_all(turn_dir.join("outputs")).expect("outputs dir should exist");
        let todo_path = turn_dir.join("todos.txt");
        let html_output_path = turn_dir.join("outputs").join("turn-1-report.html");

        let todos = TodoList::parse(
            "1. [completed] Inspect schema\n2. [completed] Compute captaincy record\n3. [pending] Return the computed record in a concise cricket summary with any caveats if captaincy is inferred from a specific field.\n4. [pending] Generate a clear, engaging data blog post with narrative writing, charts, and tables, using deep analysis in Python with pandas and numpy to surface insights.\n5. [pending] Print the path of the generated HTML file.\n",
        )
        .expect("seed todos should parse");
        todos
            .save_to_path(&todo_path)
            .expect("todo file should write");

        let content = "Rohit Sharma's IPL captaincy record is 228 matches as captain, with 125 wins, 99 losses, 4 ties, and 0 no-results, for a 54.82% win rate. Caveat: captaincy is inferred from the available captain-designation field in the match data.";
        let outcome = auto_advance_summary_todo_from_final_content(
            &todo_path,
            3,
            "Return the computed record in a concise cricket summary with any caveats if captaincy is inferred from a specific field.",
            content,
            &html_output_path,
        )
        .expect("auto-advance should succeed");

        let feedback = match outcome {
            FinalTodoAdvance::AdvancedWithFeedback(feedback) => feedback,
            _ => panic!("summary todo should advance to the HTML step"),
        };
        assert!(feedback.contains("satisfied todo 3"));
        assert!(feedback.contains(MANDATORY_TODO_GENERATE_HTML));

        let reconciled = TodoList::load_from_path(&todo_path).expect("todo file should reload");
        assert_eq!(
            reconciled.items[2].status,
            crate::todo::TodoStatus::Completed
        );
        assert_eq!(reconciled.items[3].status, crate::todo::TodoStatus::Pending);
    }

    #[test]
    fn auto_advance_summary_todo_from_final_content_does_not_advance_compute_step() {
        let working_directory = test_fixtures_root();
        let turn_dir = working_directory.join("turn-auto-advance-compute");
        std::fs::create_dir_all(turn_dir.join("outputs")).expect("outputs dir should exist");
        let todo_path = turn_dir.join("todos.txt");
        let html_output_path = turn_dir.join("outputs").join("turn-1-report.html");

        let todos = TodoList::parse(
            "1. [completed] Inspect schema\n2. [pending] Compute Rohit Sharma's IPL captaincy record from the confirmed fields.\n3. [pending] Return the computed record in a concise cricket summary.\n",
        )
        .expect("seed todos should parse");
        todos
            .save_to_path(&todo_path)
            .expect("todo file should write");

        let content = "Rohit Sharma's IPL captaincy record is 228 matches as captain, with 125 wins and 99 losses.";
        let outcome = auto_advance_summary_todo_from_final_content(
            &todo_path,
            2,
            "Compute Rohit Sharma's IPL captaincy record from the confirmed fields.",
            content,
            &html_output_path,
        )
        .expect("auto-advance should succeed");

        assert!(matches!(outcome, FinalTodoAdvance::Noop));
        let reconciled = TodoList::load_from_path(&todo_path).expect("todo file should reload");
        assert_eq!(reconciled.items[1].status, crate::todo::TodoStatus::Pending);
    }

    #[test]
    fn immediate_partial_detail_is_generated_for_out_of_order_html_open() {
        let detail = immediate_partial_detail_for_local_tool_error(
            &LocalToolName::new("bash").expect("valid tool"),
            "the deterministic HTML report `/tmp/arka/turn-report.html` cannot be opened while the current actionable todo is `Generate a clear, engaging data blog post with narrative writing, charts, and tables, using deep analysis in Python with pandas and numpy to surface insights.`; finish todos in order",
        )
        .expect("detail should be returned");
        assert!(detail.contains("generate the report first"));
        assert!(detail.contains("next todo step"));
    }

    #[test]
    fn immediate_partial_detail_is_specific_for_html_rewrite_during_open_step() {
        let detail = immediate_partial_detail_for_local_tool_error(
            &LocalToolName::new("bash").expect("valid tool"),
            "the deterministic HTML report `/tmp/arka/turn-report.html` cannot be written from bash while the current actionable todo is `Print the path of the generated HTML file.`; finish todos in order",
        )
        .expect("detail should be returned");
        assert!(detail.contains("html-path-print todo remains"));
        assert!(detail.contains("print the existing report path instead of writing it again"));
    }

    #[test]
    fn html_todo_feedback_blocks_bundled_generate_and_open_goal() {
        let working_directory = test_fixtures_root();
        let turn_dir = working_directory.join("turn-active-reconcile");
        std::fs::create_dir_all(&turn_dir).expect("turn dir should exist");
        let todo_path = turn_dir.join("todos.txt");
        let html_output_path = turn_dir.join("outputs").join("turn-1-report.html");

        let todos = TodoList::parse(
            "1. [completed] Fetch the answer\n2. [in_progress] Generate a clear, engaging data blog post with narrative writing, charts, and tables, using deep analysis in Python with pandas and numpy to surface insights.\n3. [pending] Print the path of the generated HTML file.\n",
        )
        .expect("seed todos should parse");
        todos
            .save_to_path(&todo_path)
            .expect("todo file should write");

        let feedback = html_todo_feedback_for_delegation(
            &todo_path,
            &crate::model::SubagentDelegationRequest {
                subagent_type: "tool-executor".to_owned(),
                target: DelegationTarget::LocalToolsScope(
                    LocalToolsScopeTarget::working_directory(),
                ),
                goal:
                    "Generate the deterministic HTML report and print the generated HTML file path."
                        .to_owned(),
            },
            &html_output_path,
        )
        .expect("feedback should compute");

        let feedback = feedback.expect("feedback should be present");
        assert!(feedback.contains("Do not bundle the html-path-print step"));
        assert!(feedback.contains("Print the path of the generated HTML file."));
    }

    #[test]
    fn html_todo_feedback_blocks_broad_workflow_goal_during_html_generation() {
        let working_directory = test_fixtures_root();
        let turn_dir = working_directory.join("turn-active-html-broad-goal");
        std::fs::create_dir_all(&turn_dir).expect("turn dir should exist");
        let todo_path = turn_dir.join("todos.txt");
        let html_output_path = turn_dir.join("outputs").join("turn-1-report.html");

        let todos = TodoList::parse(
            "1. [completed] Compute IPL 2025 balance metrics\n2. [in_progress] Generate a clear, engaging data blog post with narrative writing, charts, and tables, using deep analysis in Python with pandas and numpy to surface insights.\n3. [pending] Print the path of the generated HTML file.\n",
        )
        .expect("seed todos should parse");
        todos
            .save_to_path(&todo_path)
            .expect("todo file should write");

        let feedback = html_todo_feedback_for_delegation(
            &todo_path,
            &crate::model::SubagentDelegationRequest {
                subagent_type: "tool-executor".to_owned(),
                target: DelegationTarget::LocalToolsScope(
                    LocalToolsScopeTarget::working_directory(),
                ),
                goal: "Determine the most balanced IPL 2025 side using grounded database analysis and complete the required report workflow."
                    .to_owned(),
            },
            &html_output_path,
        )
        .expect("feedback should compute");

        let feedback = feedback.expect("feedback should be present");
        assert!(feedback.contains("only the HTML-generation step"));
        assert!(feedback.contains("Do not delegate broader analysis work"));
        assert!(feedback.contains("Delegate only the deterministic HTML report generation"));
    }

    #[test]
    fn html_todo_feedback_allows_report_goal_about_existing_analysis() {
        let working_directory = test_fixtures_root();
        let turn_dir = working_directory.join("turn-active-html-report-analysis");
        std::fs::create_dir_all(&turn_dir).expect("turn dir should exist");
        let todo_path = turn_dir.join("todos.txt");
        let html_output_path = turn_dir.join("outputs").join("turn-1-report.html");

        let todos = TodoList::parse(
            "1. [completed] Compute users-table metrics\n2. [in_progress] Generate a clear, engaging data blog post with narrative writing, charts, and tables, using deep analysis in Python with pandas and numpy to surface insights.\n3. [pending] Print the path of the generated HTML file.\n",
        )
        .expect("seed todos should parse");
        todos
            .save_to_path(&todo_path)
            .expect("todo file should write");

        let feedback = html_todo_feedback_for_delegation(
            &todo_path,
            &crate::model::SubagentDelegationRequest {
                subagent_type: "tool-executor".to_owned(),
                target: DelegationTarget::LocalToolsScope(
                    LocalToolsScopeTarget::working_directory(),
                ),
                goal: "Generate the required HTML report for the users-only customer analysis."
                    .to_owned(),
            },
            &html_output_path,
        )
        .expect("feedback should compute");

        assert_eq!(feedback, None);
    }

    #[test]
    fn html_todo_feedback_allows_report_goal_about_users_table_analysis() {
        let working_directory = test_fixtures_root();
        let turn_dir = working_directory.join("turn-active-html-users-table-analysis");
        std::fs::create_dir_all(&turn_dir).expect("turn dir should exist");
        let todo_path = turn_dir.join("todos.txt");
        let html_output_path = turn_dir.join("outputs").join("turn-1-report.html");

        let todos = TodoList::parse(
            "1. [completed] Compute users-table metrics\n2. [in_progress] Generate a clear, engaging data blog post with narrative writing, charts, and tables, using deep analysis in Python with pandas and numpy to surface insights.\n3. [pending] Print the path of the generated HTML file.\n",
        )
        .expect("seed todos should parse");
        todos
            .save_to_path(&todo_path)
            .expect("todo file should write");

        let feedback = html_todo_feedback_for_delegation(
            &todo_path,
            &crate::model::SubagentDelegationRequest {
                subagent_type: "tool-executor".to_owned(),
                target: DelegationTarget::LocalToolsScope(
                    LocalToolsScopeTarget::working_directory(),
                ),
                goal: "Create the required HTML report for the customer users-table analysis."
                    .to_owned(),
            },
            &html_output_path,
        )
        .expect("feedback should compute");

        assert_eq!(feedback, None);
    }

    #[test]
    fn html_todo_feedback_allows_complete_html_deliverable_goal() {
        let working_directory = test_fixtures_root();
        let turn_dir = working_directory.join("turn-active-html-complete-deliverable");
        std::fs::create_dir_all(&turn_dir).expect("turn dir should exist");
        let todo_path = turn_dir.join("todos.txt");
        let html_output_path = turn_dir.join("outputs").join("turn-1-report.html");

        let todos = TodoList::parse(
            "1. [completed] Compute users-table metrics\n2. [pending] Generate a clear, engaging data blog post with narrative writing, charts, and tables, using deep analysis in Python with pandas and numpy to surface insights.\n3. [pending] Print the path of the generated HTML file.\n",
        )
        .expect("seed todos should parse");
        todos
            .save_to_path(&todo_path)
            .expect("todo file should write");

        let feedback = html_todo_feedback_for_delegation(
            &todo_path,
            &crate::model::SubagentDelegationRequest {
                subagent_type: "tool-executor".to_owned(),
                target: DelegationTarget::LocalToolsScope(
                    LocalToolsScopeTarget::working_directory(),
                ),
                goal:
                    "Complete the required HTML deliverable for the customer users-only analysis."
                        .to_owned(),
            },
            &html_output_path,
        )
        .expect("feedback should compute");

        assert_eq!(feedback, None);
    }

    #[test]
    fn html_todo_feedback_blocks_query_goal_during_html_generation() {
        let working_directory = test_fixtures_root();
        let turn_dir = working_directory.join("turn-active-html-query-goal");
        std::fs::create_dir_all(&turn_dir).expect("turn dir should exist");
        let todo_path = turn_dir.join("todos.txt");
        let html_output_path = turn_dir.join("outputs").join("turn-1-report.html");

        let todos = TodoList::parse(
            "1. [completed] Compute users-table metrics\n2. [in_progress] Generate a clear, engaging data blog post with narrative writing, charts, and tables, using deep analysis in Python with pandas and numpy to surface insights.\n3. [pending] Print the path of the generated HTML file.\n",
        )
        .expect("seed todos should parse");
        todos
            .save_to_path(&todo_path)
            .expect("todo file should write");

        let feedback = html_todo_feedback_for_delegation(
            &todo_path,
            &crate::model::SubagentDelegationRequest {
                subagent_type: "tool-executor".to_owned(),
                target: DelegationTarget::LocalToolsScope(
                    LocalToolsScopeTarget::working_directory(),
                ),
                goal: "Query users again and create the HTML report.".to_owned(),
            },
            &html_output_path,
        )
        .expect("feedback should compute");

        let feedback = feedback.expect("feedback should be present");
        assert!(feedback.contains("Do not delegate broader analysis work"));
        assert!(feedback.contains("Delegate only the deterministic HTML report generation"));
    }

    #[test]
    fn html_todo_feedback_blocks_non_open_only_goal_when_open_todo_is_active() {
        let working_directory = test_fixtures_root();
        let turn_dir = working_directory.join("turn-active-open-only");
        std::fs::create_dir_all(&turn_dir).expect("turn dir should exist");
        let todo_path = turn_dir.join("todos.txt");
        let html_output_path = turn_dir.join("outputs").join("turn-1-report.html");

        let todos = TodoList::parse(
            "1. [completed] Fetch the answer\n2. [completed] Generate a clear, engaging data blog post with narrative writing, charts, and tables, using deep analysis in Python with pandas and numpy to surface insights.\n3. [pending] Print the path of the generated HTML file.\n",
        )
        .expect("seed todos should parse");
        todos
            .save_to_path(&todo_path)
            .expect("todo file should write");

        let feedback = html_todo_feedback_for_delegation(
            &todo_path,
            &crate::model::SubagentDelegationRequest {
                subagent_type: "tool-executor".to_owned(),
                target: DelegationTarget::LocalToolsScope(LocalToolsScopeTarget::working_directory()),
                goal: "Answer the user's IPL stat query and complete the required report deliverables."
                    .to_owned(),
            },
            &html_output_path,
        )
        .expect("feedback should compute");

        let feedback = feedback.expect("feedback should be present");
        assert!(feedback.contains("Only print the existing deterministic HTML report path"));
        assert!(feedback.contains("Do not delegate any broader `tool-executor` work"));
        assert!(feedback.contains("using local `bash` with `printf"));
    }

    #[test]
    fn html_todo_feedback_blocks_mcp_delegation_when_open_todo_is_active() {
        let working_directory = test_fixtures_root();
        let turn_dir = working_directory.join("turn-active-open-mcp-block");
        std::fs::create_dir_all(&turn_dir).expect("turn dir should exist");
        let todo_path = turn_dir.join("todos.txt");
        let html_output_path = turn_dir.join("outputs").join("turn-1-report.html");

        let todos = TodoList::parse(
            "1. [completed] Fetch the answer\n2. [completed] Generate a clear, engaging data blog post with narrative writing, charts, and tables, using deep analysis in Python with pandas and numpy to surface insights.\n3. [pending] Print the path of the generated HTML file.\n",
        )
        .expect("seed todos should parse");
        todos
            .save_to_path(&todo_path)
            .expect("todo file should write");

        let feedback = html_todo_feedback_for_delegation(
            &todo_path,
            &crate::model::SubagentDelegationRequest {
                subagent_type: "mcp-executor".to_owned(),
                target: DelegationTarget::McpServerScope(McpServerScopeTarget {
                    server_name: ServerName::new("ipl").expect("valid server"),
                }),
                goal: "Complete the IPL 2025 balance analysis and required HTML report delivery."
                    .to_owned(),
            },
            &html_output_path,
        )
        .expect("feedback should compute");

        let feedback = feedback.expect("feedback should be present");
        assert!(feedback.contains("html-path-print step"));
        assert!(feedback.contains("Do not delegate MCP/data work"));
        assert!(feedback.contains("subagent_type: \"tool-executor\""));
        assert!(feedback.contains("printf"));
    }

    #[test]
    fn simple_factual_turns_skip_required_todos() {
        assert!(!should_require_todos_for_turn(
            "How many runs did Ajinkya Rahane score in 2025?",
            &[],
        ));
        assert!(should_require_todos_for_turn(
            "Analyze CSK batting trends in 2025 and generate a chart.",
            &[],
        ));
    }

    #[test]
    fn simple_factual_follow_up_turns_skip_required_todos() {
        let conversation_history = vec![
            ConversationMessage {
                timestamp: SystemTime::now(),
                role: ConversationRole::User,
                content: "How many runs did Ajinkya Rahane score in 2025?".to_owned(),
            },
            ConversationMessage {
                timestamp: SystemTime::now(),
                role: ConversationRole::Assistant,
                content: "If you mean IPL 2025, I can look it up now.".to_owned(),
            },
        ];

        assert!(!should_require_todos_for_turn(
            "yes, in IPL",
            &conversation_history,
        ));
        assert!(!should_require_todos_for_turn(
            "retry",
            &conversation_history
        ));
        assert!(!should_require_todos_for_turn(
            "best fielder is somebody who has taken the most catches",
            &conversation_history,
        ));
        assert!(should_require_todos_for_turn(
            "yes",
            &[ConversationMessage {
                timestamp: SystemTime::now(),
                role: ConversationRole::User,
                content: "Analyze RCB batting trends in IPL 2025 and generate a chart.".to_owned(),
            }],
        ));
        assert!(should_require_todos_for_turn(
            "best fielder is somebody who has taken the most catches",
            &[ConversationMessage {
                timestamp: SystemTime::now(),
                role: ConversationRole::User,
                content: "Analyze IPL 2025 fielding performance and generate a report.".to_owned(),
            }],
        ));
    }

    #[test]
    fn generic_starter_replan_blocks_bundled_plan_and_execute_goal() {
        let working_directory = test_fixtures_root();
        let turn_dir = working_directory.join("turn-generic-starter-bundled-goal");
        std::fs::create_dir_all(&turn_dir).expect("turn dir should exist");
        let todo_path = turn_dir.join("todos.txt");
        TodoList::initialize(&[GENERIC_STARTER_TODO.to_owned()])
            .expect("todo init should succeed")
            .save_to_path(&todo_path)
            .expect("todo file should write");

        let feedback = generic_starter_replan_feedback_for_delegation(
            &todo_path,
            &crate::model::SubagentDelegationRequest {
                subagent_type: "tool-executor".to_owned(),
                target: DelegationTarget::LocalToolsScope(LocalToolsScopeTarget::working_directory()),
                goal: "Replace the generic todo scaffold and execute a comprehensive Hardik Pandya performance report workflow.".to_owned(),
            },
            "tool-executor",
        )
        .expect("feedback should compute");

        let feedback = feedback.expect("feedback should be present");
        assert!(feedback.contains("planning-only"));
        assert!(feedback.contains("return control immediately"));
    }

    #[test]
    fn validate_delegation_subagent_accepts_display_name_and_returns_canonical_type() {
        let registry = test_subagent_registry();
        let configured = validate_delegation_subagent(
            &registry,
            &crate::model::SubagentDelegationRequest {
                subagent_type: "Tool Executor".to_owned(),
                target: DelegationTarget::LocalToolsScope(
                    LocalToolsScopeTarget::working_directory(),
                ),
                goal: "Inspect workspace files".to_owned(),
            },
        )
        .expect("display-name delegation should resolve");

        assert_eq!(configured.subagent_type, "tool-executor");
        assert_eq!(configured.display_name, "Tool Executor");
    }

    #[test]
    fn invalid_subagent_feedback_is_actionable_for_blank_type() {
        let registry = test_subagent_registry();
        let feedback = invalid_subagent_feedback(
            &registry,
            &crate::model::SubagentDelegationRequest {
                subagent_type: "   ".to_owned(),
                target: DelegationTarget::LocalToolsScope(
                    LocalToolsScopeTarget::working_directory(),
                ),
                goal: "Inspect the workspace and generate a report".to_owned(),
            },
        );

        assert!(feedback.contains("`type` is blank"));
        assert!(feedback.contains("`tool-executor`"));
        assert!(feedback.contains("`mcp-executor`"));
        assert!(feedback.contains("local workspace inspection"));
    }

    #[test]
    fn mcp_collection_feedback_blocks_tool_executor_for_data_collection_todos() {
        let working_directory = test_fixtures_root();
        let turn_dir = working_directory.join("turn-mcp-routing");
        std::fs::create_dir_all(&turn_dir).expect("turn dir should exist");
        let todo_path = turn_dir.join("todos.txt");
        let todos = TodoList::parse(
            "1. [completed] [main-agent] Define scope\n2. [pending] [mcp-executor] Collect and prepare CSK IPL 2025 season data and supporting context\n3. [pending] [main-agent] Compute insights\n4. [pending] [tool-executor] Generate a clear, engaging data blog post with narrative writing, charts, and tables, using deep analysis in Python with pandas and numpy to surface insights.\n5. [pending] [tool-executor] Print the path of the generated HTML file.\n",
        )
        .expect("seed todos should parse");
        todos
            .save_to_path(&todo_path)
            .expect("todo file should write");

        let feedback = mcp_collection_feedback_for_delegation(
            &todo_path,
            &crate::model::SubagentDelegationRequest {
                subagent_type: "tool-executor".to_owned(),
                target: DelegationTarget::LocalToolsScope(
                    LocalToolsScopeTarget::working_directory(),
                ),
                goal: "Collect the season data locally".to_owned(),
            },
            "tool-executor",
            true,
        )
        .expect("feedback should compute");

        let feedback = feedback.expect("feedback should be present");
        assert!(feedback.contains("should not be delegated to `tool-executor`"));
        assert!(feedback.contains("Delegate `mcp-executor` with server scope"));
    }

    #[test]
    fn mcp_collection_feedback_blocks_workspace_input_extraction_todos() {
        let working_directory = test_fixtures_root();
        let turn_dir = working_directory.join("turn-mcp-routing-workspace-inputs");
        std::fs::create_dir_all(&turn_dir).expect("turn dir should exist");
        let todo_path = turn_dir.join("todos.txt");
        let todos = TodoList::parse(
            "1. [completed] [main-agent] Clarify the user ask\n2. [pending] [mcp-executor] If needed, search workspace inputs for match/stat data and extract Dhoni's IPL 2025 run total.\n3. [pending] [tool-executor] Generate a clear, engaging data blog post with narrative writing, charts, and tables, using deep analysis in Python with pandas and numpy to surface insights.\n4. [pending] [tool-executor] Print the path of the generated HTML file.\n",
        )
        .expect("seed todos should parse");
        todos
            .save_to_path(&todo_path)
            .expect("todo file should write");

        let feedback = mcp_collection_feedback_for_delegation(
            &todo_path,
            &crate::model::SubagentDelegationRequest {
                subagent_type: "tool-executor".to_owned(),
                target: DelegationTarget::LocalToolsScope(
                    LocalToolsScopeTarget::working_directory(),
                ),
                goal: "Search local inputs for the Dhoni run total".to_owned(),
            },
            "tool-executor",
            true,
        )
        .expect("feedback should compute");

        let feedback = feedback.expect("feedback should be present");
        assert!(feedback.contains("should not be delegated to `tool-executor`"));
        assert!(feedback.contains("Delegate `mcp-executor` with server scope"));
    }

    #[test]
    fn mcp_collection_feedback_blocks_tool_executor_for_team_level_compute_todo() {
        let working_directory = test_fixtures_root();
        let turn_dir = working_directory.join("turn-mcp-routing-team-compute");
        std::fs::create_dir_all(&turn_dir).expect("turn dir should exist");
        let todo_path = turn_dir.join("todos.txt");
        let todos = TodoList::parse(
            "1. [completed] [mcp-executor] Discover IPL schemas\n2. [completed] [mcp-executor] Sample IPL 2025 rows\n3. [completed] [main-agent] Choose a concrete balance metric\n4. [in_progress] [mcp-executor] Run the full IPL 2025 team-level computation using the chosen metric.\n5. [pending] [main-agent] Prepare a concise business-style answer naming the most balanced side and explaining the metric used.\n6. [pending] [tool-executor] Generate a clear, engaging data blog post with narrative writing, charts, and tables, using deep analysis in Python with pandas and numpy to surface insights.\n7. [pending] [tool-executor] Print the path of the generated HTML file.\n",
        )
        .expect("seed todos should parse");
        todos
            .save_to_path(&todo_path)
            .expect("todo file should write");

        let feedback = mcp_collection_feedback_for_delegation(
            &todo_path,
            &crate::model::SubagentDelegationRequest {
                subagent_type: "tool-executor".to_owned(),
                target: DelegationTarget::LocalToolsScope(LocalToolsScopeTarget::working_directory()),
                goal: "Determine the most balanced IPL 2025 side using a defensible team-level metric.".to_owned(),
            },
            "tool-executor",
            true,
        )
        .expect("feedback should compute");

        let feedback = feedback.expect("feedback should be present");
        assert!(feedback.contains("should not be delegated to `tool-executor`"));
        assert!(feedback.contains("Delegate `mcp-executor` with server scope"));
        assert!(feedback.contains("Run the full IPL 2025 team-level computation"));
    }

    #[test]
    fn disabled_postgres_schema_resource_feedback_redirects_to_server_scope() {
        let server_name = ServerName::new("ipl").expect("valid server");
        let schema_uri = test_postgres_schema_resource_uri(&server_name);
        let feedback = disabled_mcp_capability_feedback_for_delegation(
            &fake_postgres_mcp_session(&server_name),
            &crate::model::SubagentDelegationRequest {
                subagent_type: "mcp-executor".to_owned(),
                target: DelegationTarget::McpCapability(McpCapabilityTarget {
                    server_name: server_name.clone(),
                    capability_kind: CapabilityKind::Resource,
                    capability_id: schema_uri,
                }),
                goal: "Inspect IPL matches schema and analyze RCB's 2025 performance.".to_owned(),
            },
            "mcp-executor",
        )
        .expect("feedback should compute");

        let feedback = feedback.expect("feedback should be present");
        assert!(feedback.contains("postgres schema resource reads are disabled"));
        assert!(feedback.contains("mcp_server_scope"));
        assert!(feedback.contains("query tool with simple SELECT-based discovery"));
    }

    #[test]
    fn tool_executor_mcp_todo_partial_feedback_blocks_non_todo_tools() {
        let working_directory = test_fixtures_root();
        let turn_dir = working_directory.join("turn-tool-executor-mcp-gate");
        std::fs::create_dir_all(&turn_dir).expect("turn dir should exist");
        let todo_path = turn_dir.join("todos.txt");
        let todos = TodoList::parse(
            "1. [completed] [main-agent] Define scope\n2. [pending] [mcp-executor] Load and inspect the CSK/IPL 2025 datasets, confirming schema, coverage, and quality.\n3. [pending] [main-agent] Perform analysis\n4. [pending] [tool-executor] Generate a clear, engaging data blog post with narrative writing, charts, and tables, using deep analysis in Python with pandas and numpy to surface insights.\n5. [pending] [tool-executor] Print the path of the generated HTML file.\n",
        )
        .expect("seed todos should parse");
        todos
            .save_to_path(&todo_path)
            .expect("todo file should write");

        let feedback = tool_executor_mcp_todo_partial_feedback(
            &todo_path,
            &LocalToolName::new("bash").expect("valid tool"),
        )
        .expect("feedback should compute");

        let feedback = feedback.expect("feedback should be present");
        assert!(feedback.contains("belongs to MCP-backed data discovery or collection"));
        assert!(feedback.contains("delegate `mcp-executor`"));
    }

    #[test]
    fn sync_mcp_todo_before_and_after_delegation_updates_matching_todo() {
        let working_directory = test_fixtures_root();
        let turn_dir = working_directory.join("turn-mcp-sync");
        std::fs::create_dir_all(&turn_dir).expect("turn dir should exist");
        let todo_path = turn_dir.join("todos.txt");
        let todos = TodoList::parse(
            "1. [completed] [main-agent] Define scope\n2. [pending] [mcp-executor] Collect and prepare CSK IPL 2025 season data and supporting context\n3. [pending] [main-agent] Perform analysis\n4. [pending] [tool-executor] Generate a clear, engaging data blog post with narrative writing, charts, and tables, using deep analysis in Python with pandas and numpy to surface insights.\n5. [pending] [tool-executor] Print the path of the generated HTML file.\n",
        )
        .expect("seed todos should parse");
        todos
            .save_to_path(&todo_path)
            .expect("todo file should write");

        sync_mcp_todo_before_delegation(
            &todo_path,
            &crate::model::SubagentDelegationRequest {
                subagent_type: "mcp-executor".to_owned(),
                target: DelegationTarget::McpServerScope(McpServerScopeTarget {
                    server_name: ServerName::new("fake").expect("valid server"),
                }),
                goal: "Collect season data".to_owned(),
            },
            "mcp-executor",
        )
        .expect("pre-sync should succeed");

        let in_progress = TodoList::load_from_path(&todo_path).expect("todo file should reload");
        assert_eq!(in_progress.items[1].status, TodoStatus::InProgress);

        sync_mcp_todo_after_delegation(
            &todo_path,
            &SubAgentResultMessageRecord {
                message_id: MessageId::new(),
                timestamp: SystemTime::now(),
                subagent_type: "mcp-executor".to_owned(),
                status: "completed".to_owned(),
                executed_action_count: 2,
                detail: "Collected season data".to_owned(),
                tool_mask: None,
            },
        )
        .expect("post-sync should succeed");

        let completed = TodoList::load_from_path(&todo_path).expect("todo file should reload");
        assert_eq!(completed.items[1].status, TodoStatus::Completed);
    }

    #[test]
    fn sync_mcp_todo_updates_season_encoding_confirmation_todo() {
        let working_directory = test_fixtures_root();
        let turn_dir = working_directory.join("turn-mcp-sync-season-confirm");
        std::fs::create_dir_all(&turn_dir).expect("turn dir should exist");
        let todo_path = turn_dir.join("todos.txt");
        let todos = TodoList::parse(
            "1. [completed] [mcp-executor] Inspect the IPL schema for matches, match_teams, innings, deliveries, and teams.\n2. [pending] [mcp-executor] Confirm how IPL 2025 is encoded in the matches data and whether there are any season/date filters needed.\n3. [pending] [mcp-executor] Compute a concrete team-balance metric for IPL 2025 from the available data.\n4. [pending] [tool-executor] Generate a clear, engaging data blog post with narrative writing, charts, and tables, using deep analysis in Python with pandas and numpy to surface insights.\n5. [pending] [tool-executor] Print the path of the generated HTML file.\n",
        )
        .expect("seed todos should parse");
        todos
            .save_to_path(&todo_path)
            .expect("todo file should write");

        sync_mcp_todo_before_delegation(
            &todo_path,
            &crate::model::SubagentDelegationRequest {
                subagent_type: "mcp-executor".to_owned(),
                target: DelegationTarget::McpServerScope(McpServerScopeTarget {
                    server_name: ServerName::new("ipl").expect("valid server"),
                }),
                goal: "Confirm how IPL 2025 is encoded in the matches data.".to_owned(),
            },
            "mcp-executor",
        )
        .expect("pre-sync should succeed");

        let in_progress = TodoList::load_from_path(&todo_path).expect("todo file should reload");
        assert_eq!(in_progress.items[1].status, TodoStatus::InProgress);

        sync_mcp_todo_after_delegation(
            &todo_path,
            &SubAgentResultMessageRecord {
                message_id: MessageId::new(),
                timestamp: SystemTime::now(),
                subagent_type: "mcp-executor".to_owned(),
                status: "completed".to_owned(),
                executed_action_count: 1,
                detail: "Confirmed season 2025 has 74 matches from 2025-03-22 to 2025-06-03."
                    .to_owned(),
                tool_mask: None,
            },
        )
        .expect("post-sync should succeed");

        let completed = TodoList::load_from_path(&todo_path).expect("todo file should reload");
        assert_eq!(completed.items[1].status, TodoStatus::Completed);
        assert_eq!(completed.items[2].status, TodoStatus::Pending);
    }

    #[test]
    fn sync_mcp_todo_retries_failed_timestamp_trend_todo() {
        let working_directory = test_fixtures_root();
        let turn_dir = working_directory.join("turn-mcp-sync-timestamp-trend");
        std::fs::create_dir_all(&turn_dir).expect("turn dir should exist");
        let todo_path = turn_dir.join("todos.txt");
        let todos = TodoList::parse(
            "1. [completed] [mcp-executor] Inspect the customer users table fields.\n2. [failed] [mcp-executor] Check signup or activity timestamps in the users table for simple trends if those fields exist.\n3. [pending] [main-agent] Prepare a concise business-language summary.\n4. [pending] [tool-executor] Generate a clear, engaging data blog post with narrative writing, charts, and tables, using deep analysis in Python with pandas and numpy to surface insights.\n5. [pending] [tool-executor] Print the path of the generated HTML file.\n",
        )
        .expect("seed todos should parse");
        todos
            .save_to_path(&todo_path)
            .expect("todo file should write");

        sync_mcp_todo_before_delegation(
            &todo_path,
            &crate::model::SubagentDelegationRequest {
                subagent_type: "mcp-executor".to_owned(),
                target: DelegationTarget::McpServerScope(McpServerScopeTarget {
                    server_name: ServerName::new("analytics").expect("valid server"),
                }),
                goal: "Check signup and activity timestamp trends in the users table.".to_owned(),
            },
            "mcp-executor",
        )
        .expect("pre-sync should succeed");

        let in_progress = TodoList::load_from_path(&todo_path).expect("todo file should reload");
        assert_eq!(in_progress.items[1].status, TodoStatus::InProgress);

        sync_mcp_todo_after_delegation(
            &todo_path,
            &SubAgentResultMessageRecord {
                message_id: MessageId::new(),
                timestamp: SystemTime::now(),
                subagent_type: "mcp-executor".to_owned(),
                status: "completed".to_owned(),
                executed_action_count: 1,
                detail: "Computed signup trend rows from created_at.".to_owned(),
                tool_mask: None,
            },
        )
        .expect("post-sync should succeed");

        let completed = TodoList::load_from_path(&todo_path).expect("todo file should reload");
        assert_eq!(completed.items[1].status, TodoStatus::Completed);
        assert_eq!(completed.items[2].status, TodoStatus::Pending);
    }

    #[test]
    fn sync_mcp_todo_marks_collection_step_completed_after_partial_result() {
        let working_directory = test_fixtures_root();
        let turn_dir = working_directory.join("turn-mcp-sync-partial");
        std::fs::create_dir_all(&turn_dir).expect("turn dir should exist");
        let todo_path = turn_dir.join("todos.txt");
        let todos = TodoList::parse(
            "1. [completed] [main-agent] Understand the requirements\n2. [in_progress] [mcp-executor] Query or inspect the IPL 2025 match results needed to compute team net run rate for the full season.\n3. [pending] [main-agent] Compute each team's NRR and identify the highest-NRR team for 2025.\n4. [pending] [tool-executor] Generate a clear, engaging data blog post with narrative writing, charts, and tables, using deep analysis in Python with pandas and numpy to surface insights.\n5. [pending] [tool-executor] Print the path of the generated HTML file.\n",
        )
        .expect("seed todos should parse");
        todos
            .save_to_path(&todo_path)
            .expect("todo file should write");

        sync_mcp_todo_after_delegation(
            &todo_path,
            &SubAgentResultMessageRecord {
                message_id: MessageId::new(),
                timestamp: SystemTime::now(),
                subagent_type: "mcp-executor".to_owned(),
                status: "partial".to_owned(),
                executed_action_count: 2,
                detail: "Fetched the season match results and returning control for computation."
                    .to_owned(),
                tool_mask: None,
            },
        )
        .expect("post-sync should succeed");

        let completed = TodoList::load_from_path(&todo_path).expect("todo file should reload");
        assert_eq!(completed.items[1].status, TodoStatus::Completed);
        assert_eq!(completed.items[2].status, TodoStatus::Pending);
    }

    #[test]
    fn sync_mcp_todo_marks_failed_after_blocking_partial_result() {
        let working_directory = test_fixtures_root();
        let turn_dir = working_directory.join("turn-mcp-sync-blocked-partial");
        std::fs::create_dir_all(&turn_dir).expect("turn dir should exist");
        let todo_path = turn_dir.join("todos.txt");
        let todos = TodoList::parse(
            "1. [completed] [mcp-executor] Confirm the IPL 2025 season scope.\n2. [in_progress] [mcp-executor] Run targeted SQL aggregations for IPL 2025 team wins and venue distribution.\n3. [pending] [tool-executor] Generate a clear, engaging data blog post with narrative writing, charts, and tables, using deep analysis in Python with pandas and numpy to surface insights.\n4. [pending] [tool-executor] Print the path of the generated HTML file.\n",
        )
        .expect("seed todos should parse");
        todos
            .save_to_path(&todo_path)
            .expect("todo file should write");

        sync_mcp_todo_after_delegation(
            &todo_path,
            &SubAgentResultMessageRecord {
                message_id: MessageId::new(),
                timestamp: SystemTime::now(),
                subagent_type: "mcp-executor".to_owned(),
                status: "partial".to_owned(),
                executed_action_count: 2,
                detail:
                    "repeated MCP failures forced an early stop after 2 errors and 2 delegated MCP actions. Last failure on server `ipl` capability `query`: set_session cannot be used inside a transaction"
                        .to_owned(),
                tool_mask: None,
            },
        )
        .expect("post-sync should succeed");

        let failed = TodoList::load_from_path(&todo_path).expect("todo file should reload");
        assert_eq!(failed.items[1].status, TodoStatus::Failed);
        assert_eq!(failed.items[2].status, TodoStatus::Pending);
    }

    #[test]
    fn postgres_query_calls_cycle_connection_after_success() {
        let server_name = ServerName::new("ipl").expect("valid server");
        let session = fake_postgres_mcp_session(&server_name);
        let server = session
            .servers
            .get(&server_name)
            .expect("postgres server should exist");

        assert!(should_cycle_mcp_connection_after_success(server, "query"));
        assert!(!should_cycle_mcp_connection_after_success(
            server,
            "other_tool"
        ));
    }

    #[test]
    fn postgres_query_session_errors_are_retryable() {
        let server_name = ServerName::new("ipl").expect("valid server");
        let session = fake_postgres_mcp_session(&server_name);
        let server = session
            .servers
            .get(&server_name)
            .expect("postgres server should exist");

        assert!(should_retry_postgres_query_session_error(
            server,
            "query",
            "[Text { text: \"Query execution failed: set_session cannot be used inside a transaction\" }]"
        ));
        assert!(!should_retry_postgres_query_session_error(
            server,
            "query",
            "[Text { text: \"permission denied for table players\" }]"
        ));
    }

    #[test]
    fn sync_mcp_todo_marks_hybrid_compute_or_extract_step_completed_after_partial_result() {
        let working_directory = test_fixtures_root();
        let turn_dir = working_directory.join("turn-mcp-sync-hybrid-partial");
        std::fs::create_dir_all(&turn_dir).expect("turn dir should exist");
        let todo_path = turn_dir.join("todos.txt");
        let todos = TodoList::parse(
            "1. [completed] [main-agent] Determine the evidence and data source needed to answer who was the best bowler in IPL 2025.\n2. [in_progress] [mcp-executor] Compute or extract the relevant bowling performance metrics for the 2025 season.\n3. [pending] [tool-executor] Generate a clear, engaging data blog post with narrative writing, charts, and tables, using deep analysis in Python with pandas and numpy to surface insights.\n4. [pending] [tool-executor] Print the path of the generated HTML file.\n",
        )
        .expect("seed todos should parse");
        todos
            .save_to_path(&todo_path)
            .expect("todo file should write");

        sync_mcp_todo_after_delegation(
            &todo_path,
            &SubAgentResultMessageRecord {
                message_id: MessageId::new(),
                timestamp: SystemTime::now(),
                subagent_type: "mcp-executor".to_owned(),
                status: "partial".to_owned(),
                executed_action_count: 1,
                detail: "Computed the IPL 2025 bowling ranking and returned control.".to_owned(),
                tool_mask: None,
            },
        )
        .expect("post-sync should succeed");

        let completed = TodoList::load_from_path(&todo_path).expect("todo file should reload");
        assert_eq!(completed.items[1].status, TodoStatus::Completed);
        assert_eq!(completed.items[2].status, TodoStatus::Pending);
    }

    #[test]
    fn sync_mcp_todo_revives_failed_step_before_successful_retry() {
        let working_directory = test_fixtures_root();
        let turn_dir = working_directory.join("turn-mcp-sync-retry");
        std::fs::create_dir_all(&turn_dir).expect("turn dir should exist");
        let todo_path = turn_dir.join("todos.txt");
        let todos = TodoList::parse(
            "1. [completed] [main-agent] Clarify the IPL 2025 bowling metric and identify the source data available in the workspace.\n2. [failed] [mcp-executor] Compute the leading bowler(s) using the available dataset and derive the ranking table.\n3. [pending] [tool-executor] Generate a clear, engaging data blog post with narrative writing, charts, and tables, using deep analysis in Python with pandas and numpy to surface insights.\n4. [pending] [tool-executor] Print the path of the generated HTML file.\n",
        )
        .expect("seed todos should parse");
        todos
            .save_to_path(&todo_path)
            .expect("todo file should write");

        sync_mcp_todo_before_delegation(
            &todo_path,
            &crate::model::SubagentDelegationRequest {
                subagent_type: "mcp-executor".to_owned(),
                target: DelegationTarget::McpServerScope(McpServerScopeTarget {
                    server_name: ServerName::new("ipl").expect("valid server"),
                }),
                goal: "Resolve the failed IPL 2025 bowler-ranking step.".to_owned(),
            },
            "mcp-executor",
        )
        .expect("pre-sync should succeed");

        let retried = TodoList::load_from_path(&todo_path).expect("todo file should reload");
        assert_eq!(retried.items[1].status, TodoStatus::InProgress);

        sync_mcp_todo_after_delegation(
            &todo_path,
            &SubAgentResultMessageRecord {
                message_id: MessageId::new(),
                timestamp: SystemTime::now(),
                subagent_type: "mcp-executor".to_owned(),
                status: "completed".to_owned(),
                executed_action_count: 1,
                detail: "Computed the bowler ranking successfully on retry.".to_owned(),
                tool_mask: None,
            },
        )
        .expect("post-sync should succeed");

        let completed = TodoList::load_from_path(&todo_path).expect("todo file should reload");
        assert_eq!(completed.items[1].status, TodoStatus::Completed);
        assert_eq!(completed.items[2].status, TodoStatus::Pending);
    }

    fn test_subagent_registry() -> SubagentRegistry {
        SubagentRegistry {
            subagents: vec![
                ConfiguredSubagent {
                    subagent_type: "mcp-executor".to_owned(),
                    display_name: "Mcp Executor".to_owned(),
                    purpose: "Run MCP work".to_owned(),
                    when_to_use: "When MCP is needed".to_owned(),
                    target_requirements: "mcp target".to_owned(),
                    result_summary: "Returns MCP results".to_owned(),
                    prompt_path: PathBuf::from("subagents/mcp-executor.prompt.md"),
                    enabled: true,
                    model_name: None,
                },
                ConfiguredSubagent {
                    subagent_type: "tool-executor".to_owned(),
                    display_name: "Tool Executor".to_owned(),
                    purpose: "Run local tool work".to_owned(),
                    when_to_use: "When local tools are needed".to_owned(),
                    target_requirements: "local tools scope".to_owned(),
                    result_summary: "Returns local tool results".to_owned(),
                    prompt_path: PathBuf::from("subagents/tool-executor.prompt.md"),
                    enabled: true,
                    model_name: None,
                },
            ],
        }
    }

    fn fake_mcp_session(server_name: &ServerName) -> McpSession {
        McpSession {
            servers: HashMap::from([(
                server_name.clone(),
                PreparedServer {
                    config: McpServerConfig {
                        name: server_name.to_string(),
                        transport: None,
                        command: String::new(),
                        args: Vec::new(),
                        env: HashMap::new(),
                        description: None,
                    },
                    minimal_catalog: McpMinimalCatalog {
                        schema_version: CURRENT_SCHEMA_VERSION,
                        server: McpServerMetadata {
                            logical_name: server_name.to_string(),
                            ..Default::default()
                        },
                        capability_families: McpCapabilityFamilies {
                            tools: McpCapabilityFamilySummary {
                                supported: true,
                                count: 2,
                            },
                            resources: McpCapabilityFamilySummary {
                                supported: true,
                                count: 1,
                            },
                        },
                        tools: vec![
                            MinimalToolMetadata {
                                name: "list_tables".to_owned(),
                                ..Default::default()
                            },
                            MinimalToolMetadata {
                                name: "run_query".to_owned(),
                                ..Default::default()
                            },
                        ],
                        resources: vec![MinimalResourceMetadata {
                            uri: "schema://users".to_owned(),
                            ..Default::default()
                        }],
                    },
                    full_catalog: McpFullCatalog {
                        schema_version: CURRENT_SCHEMA_VERSION,
                        server: McpServerMetadata {
                            logical_name: server_name.to_string(),
                            ..Default::default()
                        },
                        capability_families: McpCapabilityFamilies {
                            tools: McpCapabilityFamilySummary {
                                supported: true,
                                count: 2,
                            },
                            resources: McpCapabilityFamilySummary {
                                supported: true,
                                count: 1,
                            },
                        },
                        tools: vec![
                            FullToolMetadata {
                                name: "list_tables".to_owned(),
                                input_schema: serde_json::json!({
                                    "type": "object",
                                    "required": ["database"],
                                    "properties": {
                                        "database": { "type": "string" }
                                    }
                                }),
                                ..Default::default()
                            },
                            FullToolMetadata {
                                name: "run_query".to_owned(),
                                input_schema: serde_json::json!({
                                    "type": "object",
                                    "required": ["query"],
                                    "properties": {
                                        "query": { "type": "string" }
                                    }
                                }),
                                ..Default::default()
                            },
                        ],
                        resources: vec![FullResourceMetadata {
                            uri: "schema://users".to_owned(),
                            ..Default::default()
                        }],
                        extensions: serde_json::Value::Null,
                    },
                    full_catalog_markdown: "# MCP Full: ex-vol\n\n- tool: list_tables\n- tool: run_query\n- resource: schema://users".to_owned(),
                    connection: None,
                },
            )]),
        }
    }

    fn fake_postgres_mcp_session(server_name: &ServerName) -> McpSession {
        let config = test_postgres_server_config(server_name);
        let schema_uri = test_postgres_schema_resource_uri(server_name);
        McpSession {
            servers: HashMap::from([(
                server_name.clone(),
                PreparedServer {
                    config,
                    minimal_catalog: McpMinimalCatalog {
                        schema_version: CURRENT_SCHEMA_VERSION,
                        server: McpServerMetadata {
                            logical_name: server_name.to_string(),
                            protocol_name: "postgres-mcp".to_owned(),
                            ..Default::default()
                        },
                        capability_families: McpCapabilityFamilies {
                            tools: McpCapabilityFamilySummary {
                                supported: true,
                                count: 1,
                            },
                            resources: McpCapabilityFamilySummary {
                                supported: true,
                                count: 1,
                            },
                        },
                        tools: vec![MinimalToolMetadata {
                            name: "query".to_owned(),
                            ..Default::default()
                        }],
                        resources: vec![MinimalResourceMetadata {
                            uri: schema_uri.clone(),
                            ..Default::default()
                        }],
                    },
                    full_catalog: McpFullCatalog {
                        schema_version: CURRENT_SCHEMA_VERSION,
                        server: McpServerMetadata {
                            logical_name: server_name.to_string(),
                            protocol_name: "postgres-mcp".to_owned(),
                            ..Default::default()
                        },
                        capability_families: McpCapabilityFamilies {
                            tools: McpCapabilityFamilySummary {
                                supported: true,
                                count: 1,
                            },
                            resources: McpCapabilityFamilySummary {
                                supported: true,
                                count: 1,
                            },
                        },
                        tools: vec![FullToolMetadata {
                            name: "query".to_owned(),
                            input_schema: serde_json::json!({
                                "type": "object",
                                "required": ["sql"],
                                "properties": {
                                    "sql": { "type": "string" }
                                }
                            }),
                            ..Default::default()
                        }],
                        resources: vec![FullResourceMetadata {
                            uri: schema_uri.clone(),
                            ..Default::default()
                        }],
                        extensions: serde_json::Value::Null,
                    },
                    full_catalog_markdown: format!(
                        "# MCP Full: {server_name}\n\n- tool: query\n- resource: {schema_uri}"
                    ),
                    connection: None,
                },
            )]),
        }
    }

    #[test]
    fn mcp_server_scope_prompt_includes_server_scope_and_no_local_tools() {
        let registry_path = workspace_root().join("config").join("subagents.json");
        let configured = ConfiguredSubagent {
            subagent_type: "mcp-executor".to_owned(),
            display_name: "Mcp Executor".to_owned(),
            purpose: "Complete delegated MCP work".to_owned(),
            when_to_use: "For one-server MCP tasks".to_owned(),
            target_requirements: "mcp server scope".to_owned(),
            result_summary: "Executes delegated MCP work".to_owned(),
            prompt_path: PathBuf::from("subagents/mcp-executor.prompt.md"),
            enabled: true,
            model_name: None,
        };
        let server_name = ServerName::new("ex-vol").expect("valid server");

        let prompt = build_subagent_prompt(
            &registry_path,
            &configured,
            &fake_mcp_session(&server_name),
            TurnPhase::Execution,
            &DelegationTarget::McpServerScope(McpServerScopeTarget {
                server_name: server_name.clone(),
            }),
            "Inspect users",
            "How many users are there?",
            &test_fixtures_root(),
            &test_fixtures_root().join("turn-1").join("todos.txt"),
            &TurnPolicyPromptContext {
                phase: TurnPhase::Execution,
                html_output_path: test_fixtures_root()
                    .join("outputs")
                    .join("turn-1-report.html"),
                force_todo_file: false,
                execution_todo_required: Some(false),
            },
            None,
            &[],
        )
        .expect("prompt should build");

        assert!(prompt.contains("# MCP Full: ex-vol"));
        assert!(prompt.contains("Selected server: ex-vol"));
        assert!(
            prompt
                .contains("Execution scope: any allowed MCP tool or resource on this server only")
        );
        assert!(prompt.contains("Local tools: unavailable"));
    }

    #[test]
    fn confirmed_table_context_uses_successful_recent_queries() {
        let server_name = ServerName::new("vol").expect("valid server");
        let rendered = render_confirmed_table_context(
            &[
                MessageRecord::McpCall(McpCallMessageRecord {
                    message_id: MessageId::new(),
                    timestamp: SystemTime::now(),
                    target: McpCapabilityTarget {
                        server_name: server_name.clone(),
                        capability_kind: CapabilityKind::Tool,
                        capability_id: "run_query".to_owned(),
                    },
                    arguments: serde_json::json!({
                        "query": "SELECT count(*) FROM analytics_db.users"
                    }),
                }),
                MessageRecord::McpResult(McpResultMessageRecord {
                    message_id: MessageId::new(),
                    timestamp: SystemTime::now(),
                    target: McpCapabilityTarget {
                        server_name: server_name.clone(),
                        capability_kind: CapabilityKind::Tool,
                        capability_id: "run_query".to_owned(),
                    },
                    result_summary: "{\"columns\":[\"count\"],\"rows\":[[10]]}".to_owned(),
                    error: None,
                }),
            ],
            &server_name,
        );

        assert!(rendered.contains("analytics_db.users"));
    }

    #[test]
    fn mcp_capability_subagent_catalog_only_exposes_selected_tool() {
        let server_name = ServerName::new("ex-vol").expect("valid server");
        let target = DelegationTarget::McpCapability(McpCapabilityTarget {
            server_name: server_name.clone(),
            capability_kind: CapabilityKind::Tool,
            capability_id: "run_query".to_owned(),
        });
        let tools = build_subagent_tool_catalog(&fake_mcp_session(&server_name), &target)
            .expect("catalog should build");

        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "run_query");
    }

    #[test]
    fn mcp_server_scope_catalog_exposes_full_server_tools_and_resources() {
        let server_name = ServerName::new("ex-vol").expect("valid server");
        let tools = build_subagent_tool_catalog(
            &fake_mcp_session(&server_name),
            &DelegationTarget::McpServerScope(McpServerScopeTarget { server_name }),
        )
        .expect("catalog should build");

        assert_eq!(tools.len(), 3);
        assert!(tools.iter().any(|tool| tool.name == "list_tables"));
        assert!(tools.iter().any(|tool| tool.name == "run_query"));
        assert!(tools.iter().any(|tool| tool.name == "schema://users"));
    }

    #[test]
    fn postgres_mcp_server_scope_catalog_filters_schema_resources() {
        let server_name = ServerName::new("ipl").expect("valid server");
        let schema_uri = test_postgres_schema_resource_uri(&server_name);
        let tools = build_subagent_tool_catalog(
            &fake_postgres_mcp_session(&server_name),
            &DelegationTarget::McpServerScope(McpServerScopeTarget { server_name }),
        )
        .expect("catalog should build");

        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "query");
        assert!(!tools.iter().any(|tool| tool.name == schema_uri));
    }

    #[test]
    fn postgres_mcp_schema_resources_are_marked_disabled() {
        let server_name = ServerName::new("ipl").expect("valid server");
        let schema_uri = test_postgres_schema_resource_uri(&server_name);
        let session = fake_postgres_mcp_session(&server_name);
        let server = session
            .servers
            .get(&server_name)
            .expect("server should exist");

        let reason =
            disabled_mcp_capability_reason(server, &CapabilityKind::Resource, schema_uri.as_str());

        assert_eq!(
            reason.as_deref(),
            Some(POSTGRES_SCHEMA_RESOURCE_DISABLED_REASON)
        );
    }

    #[test]
    fn sanitize_mcp_tool_arguments_drops_extraneous_keys() {
        let server_name = ServerName::new("ex-vol").expect("valid server");
        let tools = build_subagent_tool_catalog(
            &fake_mcp_session(&server_name),
            &DelegationTarget::McpServerScope(McpServerScopeTarget {
                server_name: server_name.clone(),
            }),
        )
        .expect("catalog should build");

        let sanitized = sanitize_mcp_tool_arguments(
            &tools,
            &McpCapabilityTarget {
                server_name,
                capability_kind: CapabilityKind::Tool,
                capability_id: "run_query".to_owned(),
            },
            &serde_json::json!({
                "query": "select 1",
                "database": "analytics_db"
            }),
        )
        .expect("arguments should sanitize");

        assert_eq!(
            sanitized.arguments,
            Some(serde_json::json!({"query": "select 1"}))
        );
        assert_eq!(sanitized.dropped_keys, vec!["database".to_owned()]);
    }

    #[test]
    fn sanitize_mcp_tool_arguments_rejects_missing_required_keys() {
        let server_name = ServerName::new("ex-vol").expect("valid server");
        let tools = build_subagent_tool_catalog(
            &fake_mcp_session(&server_name),
            &DelegationTarget::McpServerScope(McpServerScopeTarget {
                server_name: server_name.clone(),
            }),
        )
        .expect("catalog should build");

        let error = sanitize_mcp_tool_arguments(
            &tools,
            &McpCapabilityTarget {
                server_name,
                capability_kind: CapabilityKind::Tool,
                capability_id: "run_query".to_owned(),
            },
            &serde_json::json!({}),
        )
        .expect_err("missing required query should fail");

        assert!(error.contains("required argument `query` is missing"));
    }

    fn workspace_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .expect("workspace root should resolve")
    }

    fn test_fixtures_root() -> PathBuf {
        workspace_root()
            .join("agent/runtime/tests/fixtures")
            .canonicalize()
            .expect("runtime test fixtures root should resolve")
    }

    fn test_postgres_registry() -> McpRegistry {
        McpRegistry::load_from_path(
            test_fixtures_root()
                .join("config")
                .join("postgres_mcp_servers.json"),
        )
        .expect("postgres test registry should load")
    }

    fn test_postgres_server_config(server_name: &ServerName) -> McpServerConfig {
        test_postgres_registry()
            .get(server_name.as_str())
            .expect("postgres test server should exist")
            .clone()
    }

    fn test_postgres_schema_resource_uri(server_name: &ServerName) -> String {
        let config = test_postgres_server_config(server_name);
        let connection_uri = postgres_connection_uri_from_config(&config)
            .expect("postgres test server config should include a connection URI");
        let base_uri = connection_uri
            .strip_prefix("postgresql://")
            .map(|rest| format!("postgres://{rest}"))
            .unwrap_or(connection_uri);
        format!("{}/matches/schema", base_uri.trim_end_matches('/'))
    }

    fn postgres_connection_uri_from_config(config: &McpServerConfig) -> Option<String> {
        match config.resolved_transport() {
            McpTransportConfig::Stdio { args, .. } => args
                .into_iter()
                .rev()
                .find(|arg| arg.starts_with("postgresql://") || arg.starts_with("postgres://")),
            McpTransportConfig::StreamableHttp { .. } => None,
        }
    }
}

/// Runtime failure modes.
#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("runtime validation failed: {0}")]
    Validation(String),
    #[error("runtime timed out: {0}")]
    Timeout(String),
    #[error("runtime I/O failed: {0}")]
    Io(String),
    #[error("runtime model interaction failed: {0}")]
    Model(#[from] ModelAdapterError),
    #[error("MCP registry failed: {0}")]
    Registry(#[from] ConfigError),
    #[error("MCP client failed: {0}")]
    McpClient(#[from] McpClientError),
    #[error("prompt rendering failed: {0}")]
    Prompt(#[from] PromptRenderError),
    #[error("todo state failed: {0}")]
    Todo(#[from] crate::todo::TodoError),
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
