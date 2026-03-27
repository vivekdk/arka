//! Prompt assembly for the MCP-first runtime.
//!
//! The runtime keeps records as the source of truth, then derives the prompt
//! snapshot for each step from that state. This prevents provider prompt text
//! from becoming the canonical runtime state representation.

use std::{
    collections::BTreeMap,
    env,
    fmt::Write,
    fs,
    path::{Path, PathBuf},
};

use thiserror::Error;
use time::{OffsetDateTime, UtcOffset, format_description::well_known::Rfc3339};

use crate::state::{
    ConversationMessage, McpCapability, MessageRecord, PromptSection, PromptSnapshot,
    ResponseClient, ResponseFormat, ResponseTarget, SubagentCard, format_system_time,
};

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
    #[error("failed to determine current working directory: {0}")]
    CurrentWorkingDirectory(#[source] std::io::Error),
    #[error("failed to format current date and time: {0}")]
    CurrentDateTime(#[source] time::error::Format),
}

/// Builds prompt snapshots from canonical runtime state.
#[derive(Clone, Debug, Default)]
pub struct PromptAssembler;

impl PromptAssembler {
    /// Renders the step prompt using the fully rendered system prompt and the
    /// current conversation state for this turn.
    pub fn build(
        &self,
        system_prompt: &str,
        conversation_history: &[ConversationMessage],
        current_turn_messages: &[MessageRecord],
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

/// Reads the prompt template from disk and replaces supported dynamic tags.
pub fn load_and_render_system_prompt(
    path: &Path,
    mcp_capabilities: &[McpCapability],
    subagent_cards: &[SubagentCard],
    response_target: &ResponseTarget,
) -> Result<String, PromptRenderError> {
    let template =
        fs::read_to_string(path).map_err(|source| PromptRenderError::ReadSystemPrompt {
            path: path.to_path_buf(),
            source,
        })?;

    render_system_prompt(&template, mcp_capabilities, subagent_cards, response_target)
}

fn render_system_prompt(
    template: &str,
    mcp_capabilities: &[McpCapability],
    subagent_cards: &[SubagentCard],
    response_target: &ResponseTarget,
) -> Result<String, PromptRenderError> {
    // Dynamic prompt tags are expanded late so each turn sees the current
    // working directory, local time, and enabled MCP catalog.
    let working_directory = env::current_dir()
        .map_err(PromptRenderError::CurrentWorkingDirectory)?
        .display()
        .to_string();
    let current_date_time = render_current_date_time()?;

    Ok(template
        .replace(WORKING_DIRECTORY_TAG, &working_directory)
        .replace(CURRENT_DATE_TIME_TAG, &current_date_time)
        .replace(AVAILABLE_MCPS_TAG, &render_available_mcps(mcp_capabilities))
        .replace(
            AVAILABLE_SUBAGENTS_TAG,
            &render_available_subagents(subagent_cards),
        )
        .replace(AVAILABLE_TOOLS_TAG, "None")
        .replace(RESPONSE_TARGET_TAG, &render_response_target(response_target)))
}

fn render_current_date_time() -> Result<String, PromptRenderError> {
    let offset = UtcOffset::current_local_offset().unwrap_or(UtcOffset::UTC);
    OffsetDateTime::now_utc()
        .to_offset(offset)
        .format(&Rfc3339)
        .map_err(PromptRenderError::CurrentDateTime)
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

    use crate::{
        ids::MessageId,
        prompt::load_and_render_system_prompt,
        state::{
            ConversationMessage, ConversationRole, LlmMessageRecord, McpCapability, MessageRecord,
            ResponseClient, ResponseFormat, ResponseTarget, ServerName, SubagentCard,
            format_system_time,
        },
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
            &[MessageRecord::Llm(LlmMessageRecord {
                message_id: MessageId::new(),
                timestamp: std::time::SystemTime::now(),
                content: "thinking".to_owned(),
            })],
        );

        assert!(prompt.rendered.contains("## System Prompt"));
        assert!(prompt.rendered.contains("## Conversation History"));
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
            &ResponseTarget {
                client: ResponseClient::Slack,
                format: ResponseFormat::SlackMrkdwn,
            },
        )
        .expect("prompt should render");

        assert!(rendered.contains("CWD: "));
        assert!(rendered.contains("Date: "));
        assert!(rendered.contains("Server: sqlite"));
        assert!(rendered.contains("Server Description: SQLite workspace access"));
        assert!(rendered.contains("Capability: run-sql"));
        assert!(rendered.contains("Description: Execute SQL"));
        assert!(rendered.contains("Sub-agents:"));
        assert!(rendered.contains("Type: tool-executor"));
        assert!(rendered.contains("Tools:\nNone"));
        assert!(rendered.contains("Client: slack"));
        assert!(rendered.contains("Format: slack_mrkdwn"));
        assert!(rendered.contains("Unknown: <dynamic variable: future>"));
        assert!(!rendered.contains("<dynamic variable: working_directory>"));
        assert!(!rendered.contains("<dynamic variable: current_date and time>"));
        assert!(!rendered.contains("<dynamic variable: available MCPs>"));
        assert!(!rendered.contains("<dynamic variable: available tools>"));
        assert!(!rendered.contains("<dynamic variable: response target>"));
    }
}
