//! Prompt assembly for the MCP-first runtime.
//!
//! The runtime keeps records as the source of truth, then derives the prompt
//! snapshot for each step from that state. This prevents provider prompt text
//! from becoming the canonical runtime state representation.

use std::{
    collections::BTreeMap,
    fmt::Write,
    fs,
    path::{Path, PathBuf},
};

use thiserror::Error;
use time::{OffsetDateTime, UtcOffset};

use crate::model::{ExecutionHandoff, TurnPhase};
use crate::state::{
    ConversationMessage, McpCapability, MessageRecord, PromptSection, PromptSnapshot,
    ResponseClient, ResponseFormat, ResponseTarget, SubagentCard, format_system_time,
};
use crate::tools::ToolDescriptor;

const WORKING_DIRECTORY_TAG: &str = "<dynamic variable: working_directory>";
const CURRENT_DATE_TIME_TAG: &str = "<dynamic variable: current_date and time>";
const AVAILABLE_MCPS_TAG: &str = "<dynamic variable: available MCPs>";
const AVAILABLE_SUBAGENTS_TAG: &str = "<dynamic variable: available sub-agents>";
const AVAILABLE_TOOLS_TAG: &str = "<dynamic variable: available tools>";
const RESPONSE_TARGET_TAG: &str = "<dynamic variable: response target>";

/// Prompt template loading and rendering failures.
#[derive(Debug, Error)]
pub enum PromptRenderError {
    #[error("failed to read system prompt file `{path}`: {source}")]
    ReadSystemPrompt {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

/// Builds prompt snapshots from canonical runtime state.
#[derive(Clone, Debug, Default)]
pub struct PromptAssembler;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TurnPolicyPromptContext {
    pub phase: TurnPhase,
    pub html_output_path: PathBuf,
    pub force_todo_file: bool,
    pub execution_todo_required: Option<bool>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TodoPromptContext {
    pub todo_path: PathBuf,
    pub todo_contents: String,
    pub next_actionable: Option<String>,
}

impl PromptAssembler {
    /// Renders the step prompt using the fully rendered system prompt and the
    /// current conversation state for this turn.
    pub fn build(
        &self,
        system_prompt: &str,
        conversation_history: &[ConversationMessage],
        recent_session_messages: &[MessageRecord],
        current_turn_messages: &[MessageRecord],
        turn_policy_context: &TurnPolicyPromptContext,
        execution_handoff: Option<&ExecutionHandoff>,
        todo_context: Option<&TodoPromptContext>,
    ) -> PromptSnapshot {
        let mut sections = Vec::new();

        sections.push(PromptSection {
            title: "System Prompt".to_owned(),
            content: system_prompt.trim().to_owned(),
        });
        sections.push(PromptSection {
            title: "Conversation History".to_owned(),
            content: render_history(conversation_history),
        });
        if !recent_session_messages.is_empty() {
            sections.push(PromptSection {
                title: "Recent Session Computed Results".to_owned(),
                content: render_recent_session_results(recent_session_messages),
            });
        }
        sections.push(PromptSection {
            title: "Turn Policy".to_owned(),
            content: render_turn_policy_context(turn_policy_context),
        });
        if let Some(execution_handoff) = execution_handoff {
            sections.push(PromptSection {
                title: "Execution Handoff".to_owned(),
                content: render_execution_handoff(execution_handoff),
            });
        }
        if let Some(todo_context) = todo_context {
            sections.push(PromptSection {
                title: "Current Turn Todo Plan".to_owned(),
                content: render_todo_context(todo_context),
            });
        }
        sections.push(PromptSection {
            title: "Current Turn Context".to_owned(),
            content: render_turn_context(current_turn_messages),
        });

        let rendered = sections
            .iter()
            .map(|section| format!("## {}\n{}\n", section.title, section.content))
            .collect::<Vec<_>>()
            .join("\n");

        PromptSnapshot { rendered, sections }
    }
}

pub fn render_turn_policy_context(turn_policy_context: &TurnPolicyPromptContext) -> String {
    let phase_summary = match turn_policy_context.phase {
        TurnPhase::Planning => {
            "Current phase: planning\nPlanning goal: inspect enough context to build a concrete execution plan. You may discover sources and sample lightly, but do not perform substantive analysis or final delivery work."
        }
        TurnPhase::Execution => {
            "Current phase: execution\nExecution goal: follow the execution handoff and complete the work. Do not redo planning unless recovery is required."
        }
    };
    let todo_summary = match turn_policy_context.execution_todo_required {
        Some(required) => format!("Execution todo file required: {required}"),
        None => "Execution todo file required: planning has not decided yet".to_owned(),
    };
    format!(
        "{phase_summary}\nDeployment force-todo override: {}\n{todo_summary}\nDeterministic HTML output: {}\nIf a todo file is required for execution, it must exist before substantive execution begins.\nDefault rule: if this turn performs analysis, reporting, transformation, or visualization work in execution, generate the HTML report and open it in the browser before finishing.\nSkip is allowed only for very simple factual replies with no meaningful transformation or reporting.",
        turn_policy_context.force_todo_file,
        turn_policy_context.html_output_path.display()
    )
}

/// Reads the prompt template from disk and replaces supported dynamic tags.
pub fn load_and_render_system_prompt(
    path: &Path,
    phase: TurnPhase,
    working_directory: &Path,
    mcp_capabilities: &[McpCapability],
    subagent_cards: &[SubagentCard],
    available_tools: &[ToolDescriptor],
    response_target: &ResponseTarget,
) -> Result<String, PromptRenderError> {
    let template = load_phase_prompt_template(path, phase)?;

    render_system_prompt(
        &template,
        working_directory,
        mcp_capabilities,
        subagent_cards,
        available_tools,
        response_target,
    )
}

fn load_phase_prompt_template(path: &Path, phase: TurnPhase) -> Result<String, PromptRenderError> {
    let Some(config_dir) = path.parent() else {
        return fs::read_to_string(path).map_err(|source| PromptRenderError::ReadSystemPrompt {
            path: path.to_path_buf(),
            source,
        });
    };
    let base_path = config_dir.join("prompt.base.md");
    let phase_path = config_dir.join(match phase {
        TurnPhase::Planning => "prompt.planning.md",
        TurnPhase::Execution => "prompt.execution.md",
    });
    if base_path.exists() && phase_path.exists() {
        let base = fs::read_to_string(&base_path).map_err(|source| {
            PromptRenderError::ReadSystemPrompt {
                path: base_path.clone(),
                source,
            }
        })?;
        let phase_template = fs::read_to_string(&phase_path).map_err(|source| {
            PromptRenderError::ReadSystemPrompt {
                path: phase_path.clone(),
                source,
            }
        })?;
        return Ok(format!("{base}\n\n{phase_template}"));
    }

    fs::read_to_string(path).map_err(|source| PromptRenderError::ReadSystemPrompt {
        path: path.to_path_buf(),
        source,
    })
}

fn render_system_prompt(
    template: &str,
    working_directory: &Path,
    mcp_capabilities: &[McpCapability],
    subagent_cards: &[SubagentCard],
    available_tools: &[ToolDescriptor],
    response_target: &ResponseTarget,
) -> Result<String, PromptRenderError> {
    // Dynamic prompt tags are expanded late so each turn sees the current
    // working directory, local date, and enabled MCP catalog.
    let working_directory = working_directory.display().to_string();
    let current_date = render_current_date();

    Ok(template
        .replace(WORKING_DIRECTORY_TAG, &working_directory)
        .replace(CURRENT_DATE_TIME_TAG, &current_date)
        .replace(AVAILABLE_MCPS_TAG, &render_available_mcps(mcp_capabilities))
        .replace(
            AVAILABLE_SUBAGENTS_TAG,
            &render_available_subagents(subagent_cards),
        )
        .replace(
            AVAILABLE_TOOLS_TAG,
            &render_available_tools(available_tools),
        )
        .replace(
            RESPONSE_TARGET_TAG,
            &render_response_target(response_target),
        ))
}

fn render_current_date() -> String {
    let offset = UtcOffset::current_local_offset().unwrap_or(UtcOffset::UTC);
    OffsetDateTime::now_utc()
        .to_offset(offset)
        .date()
        .to_string()
}

fn render_history(history: &[ConversationMessage]) -> String {
    if history.is_empty() {
        return "No prior conversation history.".to_owned();
    }

    history
        .iter()
        .map(|message| {
            format!(
                "- [{}] {}: {}",
                format_system_time(message.timestamp),
                message.role,
                message.content
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_turn_context(messages: &[MessageRecord]) -> String {
    if messages.is_empty() {
        return "No messages have been recorded for the current turn yet.".to_owned();
    }

    messages
        .iter()
        .map(MessageRecord::summary_line)
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_recent_session_results(messages: &[MessageRecord]) -> String {
    let rendered = messages
        .iter()
        .filter(|message| matches!(message, MessageRecord::McpResult(_)))
        .map(MessageRecord::summary_line)
        .collect::<Vec<_>>();

    if rendered.is_empty() {
        "No recent computed results from prior turns are available.".to_owned()
    } else {
        rendered.join("\n")
    }
}

pub fn render_todo_context(todo_context: &TodoPromptContext) -> String {
    let next_actionable = todo_context
        .next_actionable
        .as_deref()
        .unwrap_or("All todo items are completed.");
    format!(
        "Todo file: {}\nNext actionable todo: {}\n\nTodo contents:\n{}",
        todo_context.todo_path.display(),
        next_actionable,
        todo_context.todo_contents
    )
}

pub fn render_execution_handoff(handoff: &ExecutionHandoff) -> String {
    let answer_brief = if handoff.answer_brief.trim().is_empty() {
        "None".to_owned()
    } else {
        handoff.answer_brief.clone()
    };
    let selected_sources = if handoff.selected_sources.is_empty() {
        "No specific sources were selected during planning.".to_owned()
    } else {
        handoff
            .selected_sources
            .iter()
            .map(|source| {
                format!(
                    "- kind={:?} id={} rationale={}",
                    source.source_kind, source.source_id, source.rationale
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    let key_facts = if handoff.key_facts.is_empty() {
        "No planning facts were recorded.".to_owned()
    } else {
        handoff
            .key_facts
            .iter()
            .map(|fact| {
                let evidence = if fact.evidence_source_ids.is_empty() {
                    "none".to_owned()
                } else {
                    fact.evidence_source_ids.join(", ")
                };
                format!("- fact={} evidence={}", fact.fact, evidence)
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    let risks = if handoff.risks_and_constraints.is_empty() {
        "None recorded.".to_owned()
    } else {
        handoff
            .risks_and_constraints
            .iter()
            .map(|risk| format!("- {risk}"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let todo_path = handoff
        .todo_path
        .as_ref()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "None".to_owned());
    format!(
        "Summary: {}\nTodo required: {}\nTodo path: {}\nPlanned answer brief: {}\nExecution strategy: {}\n\nSelected sources:\n{}\n\nKey facts:\n{}\n\nRisks and constraints:\n{}",
        handoff.summary,
        handoff.todo_required,
        todo_path,
        answer_brief,
        handoff.execution_strategy,
        selected_sources,
        key_facts,
        risks,
    )
}

fn render_available_mcps(capabilities: &[McpCapability]) -> String {
    if capabilities.is_empty() {
        return "None".to_owned();
    }

    let mut grouped: BTreeMap<&str, Vec<&McpCapability>> = BTreeMap::new();
    for capability in capabilities {
        grouped
            .entry(capability.server_name.as_str())
            .or_default()
            .push(capability);
    }

    let mut rendered = String::new();
    for (index, (server_name, capabilities)) in grouped.iter().enumerate() {
        if index > 0 {
            rendered.push('\n');
            rendered.push('\n');
        }

        let _ = writeln!(rendered, "Server: {server_name}");
        let server_description = capabilities
            .iter()
            .find_map(|capability| capability.server_description.as_deref())
            .unwrap_or("None");
        let _ = writeln!(rendered, "Server Description: {server_description}");
        for capability in capabilities {
            let _ = writeln!(rendered, "- Kind: {:?}", capability.kind);
            let _ = writeln!(rendered, "  Capability: {}", capability.capability_id);
            let _ = writeln!(
                rendered,
                "  Title: {}",
                capability.title.as_deref().unwrap_or("None")
            );
            let _ = writeln!(
                rendered,
                "  Description: {}",
                capability.description.as_deref().unwrap_or("None")
            );
        }
    }

    rendered.trim_end().to_owned()
}

fn render_available_subagents(cards: &[SubagentCard]) -> String {
    if cards.is_empty() {
        return "None".to_owned();
    }

    cards
        .iter()
        .map(|card| {
            format!(
                "- Type: {}\n  Display name: {}\n  Purpose: {}\n  When to use: {}\n  Target requirements: {}\n  Result: {}",
                card.subagent_type,
                card.display_name,
                card.purpose,
                card.when_to_use,
                card.target_requirements,
                card.result_summary
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn render_available_tools(tools: &[ToolDescriptor]) -> String {
    if tools.is_empty() {
        return "No runtime-managed tools are currently enabled.".to_owned();
    }

    "Tool availability is managed by the runtime harness and policy engine. Use only the tools exposed at execution time."
        .to_owned()
}

fn render_response_target(target: &ResponseTarget) -> String {
    let client = match target.client {
        ResponseClient::Api => "api",
        ResponseClient::Cli => "cli",
        ResponseClient::Slack => "slack",
        ResponseClient::WhatsApp => "whatsapp",
    };
    let format = match target.format {
        ResponseFormat::PlainText => "plain_text",
        ResponseFormat::Markdown => "markdown",
        ResponseFormat::SlackMrkdwn => "slack_mrkdwn",
        ResponseFormat::WhatsAppText => "whatsapp_text",
    };
    let rules = match target.format {
        ResponseFormat::PlainText => {
            "- Final reply must be plain text.\n- Do not use Markdown syntax, tables, or fenced code blocks.\n- Keep structure with short paragraphs or numbered lists only when needed."
        }
        ResponseFormat::Markdown => {
            "- Final reply may use normal Markdown.\n- Use headings, bullets, and fenced code blocks only when they improve readability.\n- Keep the writing concise."
        }
        ResponseFormat::SlackMrkdwn => {
            "- Final reply must use Slack mrkdwn only.\n- Use Slack-compatible emphasis, bullets, quotes, and code blocks.\n- Do not use HTML or unsupported Markdown features."
        }
        ResponseFormat::WhatsAppText => {
            "- Final reply must be easy to read in WhatsApp.\n- Prefer short paragraphs and simple numbered or dashed lists.\n- Avoid tables, HTML, and complex Markdown."
        }
    };
    format!("Client: {client}\nFormat: {format}\nRules:\n{rules}")
}

#[cfg(test)]
mod tests {
    //! Prompt-rendering regressions for dynamic tags and canonical sections.

    use std::path::PathBuf;

    use crate::{
        ids::MessageId,
        model::TurnPhase,
        prompt::{TodoPromptContext, TurnPolicyPromptContext, load_and_render_system_prompt},
        state::{
            ConversationMessage, ConversationRole, LlmMessageRecord, McpCapability,
            McpCapabilityTarget, McpResultMessageRecord, MessageRecord, ResponseClient,
            ResponseFormat, ResponseTarget, ServerName, SubagentCard, format_system_time,
        },
        tools::builtin_local_tool_catalog,
    };

    use super::PromptAssembler;

    #[test]
    fn prompt_keeps_system_prompt_and_turn_state_only() {
        let assembler = PromptAssembler;
        let prompt = assembler.build(
            "Be precise.",
            &[ConversationMessage {
                timestamp: std::time::SystemTime::UNIX_EPOCH,
                role: ConversationRole::User,
                content: "hello".to_owned(),
            }],
            &[],
            &[MessageRecord::Llm(LlmMessageRecord {
                message_id: MessageId::new(),
                timestamp: std::time::SystemTime::now(),
                content: "thinking".to_owned(),
            })],
            &TurnPolicyPromptContext {
                phase: TurnPhase::Planning,
                html_output_path: PathBuf::from("/tmp/session/outputs/turn-report.html"),
                force_todo_file: false,
                execution_todo_required: None,
            },
            None,
            None,
        );

        assert!(prompt.rendered.contains("## System Prompt"));
        assert!(prompt.rendered.contains("## Conversation History"));
        assert!(prompt.rendered.contains("## Turn Policy"));
        assert!(prompt.rendered.contains("## Current Turn Context"));
        assert!(!prompt.rendered.contains("## MCP Capabilities"));
        assert!(!prompt.rendered.contains("## Sub-agents"));
        assert!(!prompt.rendered.contains("## Local Tools (inactive)"));
        assert!(
            prompt
                .rendered
                .contains(&format_system_time(std::time::SystemTime::UNIX_EPOCH))
        );
    }

    #[test]
    fn prompt_includes_recent_session_computed_results_section_when_present() {
        let assembler = PromptAssembler;
        let prompt = assembler.build(
            "Be precise.",
            &[],
            &[MessageRecord::McpResult(McpResultMessageRecord {
                message_id: MessageId::new(),
                timestamp: std::time::SystemTime::now(),
                target: McpCapabilityTarget {
                    server_name: ServerName::new("ex-vol").expect("valid server"),
                    capability_kind: mcp_metadata::CapabilityKind::Tool,
                    capability_id: "run_select_query".to_owned(),
                },
                result_summary: "{\"rows\":[[\"2026-04-05\",8]]}".to_owned(),
                error: None,
            })],
            &[],
            &TurnPolicyPromptContext {
                phase: TurnPhase::Planning,
                html_output_path: PathBuf::from("/tmp/session/outputs/turn-report.html"),
                force_todo_file: false,
                execution_todo_required: None,
            },
            None,
            None,
        );

        assert!(
            prompt
                .rendered
                .contains("## Recent Session Computed Results")
        );
        assert!(prompt.rendered.contains("run_select_query"));
        assert!(prompt.rendered.contains("2026-04-05"));
    }

    #[test]
    fn prompt_includes_todo_section_when_present() {
        let assembler = PromptAssembler;
        let prompt = assembler.build(
            "Be precise.",
            &[],
            &[],
            &[],
            &TurnPolicyPromptContext {
                phase: TurnPhase::Execution,
                html_output_path: PathBuf::from("/tmp/session/outputs/turn-report.html"),
                force_todo_file: false,
                execution_todo_required: Some(true),
            },
            None,
            Some(&TodoPromptContext {
                todo_path: PathBuf::from("/tmp/session/turn/todos.txt"),
                todo_contents: "1. [pending] Inspect data".to_owned(),
                next_actionable: Some("1. [pending] Inspect data".to_owned()),
            }),
        );

        assert!(prompt.rendered.contains("## Current Turn Todo Plan"));
        assert!(prompt.rendered.contains("Inspect data"));
        assert!(prompt.rendered.contains("turn-report.html"));
    }

    #[test]
    fn prompt_includes_turn_policy_without_todo_context() {
        let assembler = PromptAssembler;
        let prompt = assembler.build(
            "Be precise.",
            &[],
            &[],
            &[],
            &TurnPolicyPromptContext {
                phase: TurnPhase::Execution,
                html_output_path: PathBuf::from("/tmp/session/outputs/turn-report.html"),
                force_todo_file: true,
                execution_todo_required: Some(true),
            },
            None,
            None,
        );

        assert!(prompt.rendered.contains("## Turn Policy"));
        assert!(prompt.rendered.contains("turn-report.html"));
        assert!(
            prompt
                .rendered
                .contains("Deployment force-todo override: true")
        );
        assert!(
            prompt
                .rendered
                .contains("Default rule: if this turn performs analysis")
        );
    }

    #[test]
    fn system_prompt_template_replaces_supported_dynamic_tags() {
        let temp_dir =
            std::env::temp_dir().join(format!("agent-runtime-prompt-test-{}", std::process::id()));
        std::fs::create_dir_all(&temp_dir).expect("temp dir should exist");
        let template_path = temp_dir.join("prompt.md");
        std::fs::write(
            &template_path,
            "CWD: <dynamic variable: working_directory>\nDate: <dynamic variable: current_date and time>\nMCPs:\n<dynamic variable: available MCPs>\nSub-agents:\n<dynamic variable: available sub-agents>\nTools:\n<dynamic variable: available tools>\nResponse target:\n<dynamic variable: response target>\nUnknown: <dynamic variable: future>",
        )
        .expect("template should write");

        let rendered = load_and_render_system_prompt(
            &template_path,
            TurnPhase::Planning,
            &temp_dir,
            &[McpCapability {
                server_name: ServerName::new("sqlite").expect("valid server"),
                server_description: Some("SQLite workspace access".to_owned()),
                kind: mcp_metadata::CapabilityKind::Tool,
                capability_id: "run-sql".to_owned(),
                title: Some("Run SQL".to_owned()),
                description: Some("Execute SQL".to_owned()),
            }],
            &[SubagentCard {
                subagent_type: "tool-executor".to_owned(),
                display_name: "Tool Executor".to_owned(),
                purpose: "Build executable MCP actions".to_owned(),
                when_to_use: "After selecting an MCP capability".to_owned(),
                target_requirements: "server_name, capability_kind, capability_id".to_owned(),
                result_summary: "Returns a tool call or resource read".to_owned(),
            }],
            &builtin_local_tool_catalog(),
            &ResponseTarget {
                client: ResponseClient::Slack,
                format: ResponseFormat::SlackMrkdwn,
            },
        )
        .expect("prompt should render");

        let cwd_line = rendered
            .lines()
            .find(|line| line.starts_with("CWD: "))
            .expect("cwd line should render");
        assert_eq!(cwd_line, format!("CWD: {}", temp_dir.display()));

        let date_line = rendered
            .lines()
            .find(|line| line.starts_with("Date: "))
            .expect("date line should render");
        let date_value = date_line.trim_start_matches("Date: ");
        assert_eq!(date_value.len(), 10);
        assert!(
            date_value
                .chars()
                .enumerate()
                .all(|(index, ch)| match index {
                    4 | 7 => ch == '-',
                    _ => ch.is_ascii_digit(),
                })
        );
        assert!(!date_value.contains('T'));
        assert!(rendered.contains("Server: sqlite"));
        assert!(rendered.contains("Server Description: SQLite workspace access"));
        assert!(rendered.contains("Capability: run-sql"));
        assert!(rendered.contains("Description: Execute SQL"));
        assert!(rendered.contains("Sub-agents:"));
        assert!(rendered.contains("Type: tool-executor"));
        assert!(rendered.contains(
            "Tools:\nTool availability is managed by the runtime harness and policy engine. Use only the tools exposed at execution time."
        ));
        assert!(rendered.contains("Client: slack"));
        assert!(rendered.contains("Format: slack_mrkdwn"));
        assert!(rendered.contains("Unknown: <dynamic variable: future>"));
        assert!(!rendered.contains("<dynamic variable: working_directory>"));
        assert!(!rendered.contains("<dynamic variable: current_date and time>"));
        assert!(!rendered.contains("<dynamic variable: available MCPs>"));
        assert!(!rendered.contains("<dynamic variable: available tools>"));
        assert!(!rendered.contains("<dynamic variable: response target>"));
        assert!(!rendered.contains("Tool: read_file"));
    }
}
