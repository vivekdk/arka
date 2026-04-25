//! Sub-agent registry loading and prompt helpers.

use std::{
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::model::TurnPhase;
use crate::state::SubagentCard;

const MCP_SERVER_DETAILS_TAG: &str = "<dynamic variable: MCP server details>";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SubagentRegistry {
    #[serde(default)]
    pub subagents: Vec<ConfiguredSubagent>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConfiguredSubagent {
    #[serde(rename = "type")]
    pub subagent_type: String,
    pub display_name: String,
    pub purpose: String,
    pub when_to_use: String,
    pub target_requirements: String,
    pub result_summary: String,
    pub prompt_path: PathBuf,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub model_name: Option<String>,
}

impl ConfiguredSubagent {
    pub fn card(&self) -> SubagentCard {
        SubagentCard {
            subagent_type: self.subagent_type.clone(),
            display_name: self.display_name.clone(),
            purpose: self.purpose.clone(),
            when_to_use: self.when_to_use.clone(),
            target_requirements: self.target_requirements.clone(),
            result_summary: self.result_summary.clone(),
        }
    }
}

#[derive(Debug, Error)]
pub enum SubagentConfigError {
    #[error("failed to read sub-agent registry `{path}`: {source}")]
    ReadRegistry {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse sub-agent registry `{path}`: {source}")]
    ParseRegistry {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("sub-agent `{subagent_type}` is not configured")]
    UnknownSubagent { subagent_type: String },
    #[error("failed to read sub-agent prompt `{path}`: {source}")]
    ReadPrompt {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

impl SubagentRegistry {
    pub fn load_from_path(path: &Path) -> Result<Self, SubagentConfigError> {
        let raw = fs::read_to_string(path).map_err(|source| SubagentConfigError::ReadRegistry {
            path: path.to_path_buf(),
            source,
        })?;
        serde_json::from_str(&raw).map_err(|source| SubagentConfigError::ParseRegistry {
            path: path.to_path_buf(),
            source,
        })
    }

    pub fn enabled_cards(&self) -> Vec<SubagentCard> {
        let mut cards = self
            .subagents
            .iter()
            .filter(|subagent| subagent.enabled)
            .map(ConfiguredSubagent::card)
            .collect::<Vec<_>>();
        cards.sort_by(|left, right| left.subagent_type.cmp(&right.subagent_type));
        cards
    }

    pub fn get_enabled(
        &self,
        subagent_type: &str,
    ) -> Result<&ConfiguredSubagent, SubagentConfigError> {
        self.subagents
            .iter()
            .find(|subagent| {
                subagent.enabled && matches_subagent_identifier(subagent, subagent_type)
            })
            .ok_or_else(|| SubagentConfigError::UnknownSubagent {
                subagent_type: subagent_type.to_owned(),
            })
    }
}

fn matches_subagent_identifier(configured: &ConfiguredSubagent, requested: &str) -> bool {
    configured.subagent_type == requested
        || configured.display_name == requested
        || canonicalize_subagent_identifier(&configured.subagent_type)
            == canonicalize_subagent_identifier(requested)
        || canonicalize_subagent_identifier(&configured.display_name)
            == canonicalize_subagent_identifier(requested)
}

fn canonicalize_subagent_identifier(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(|ch| ch.to_lowercase())
        .collect()
}

pub fn load_subagent_prompt(
    base_dir: &Path,
    configured: &ConfiguredSubagent,
    phase: TurnPhase,
    mcp_server_details: &str,
) -> Result<String, SubagentConfigError> {
    let prompt_path = resolve_prompt_path(base_dir, configured, phase);
    fs::read_to_string(&prompt_path)
        .map_err(|source| SubagentConfigError::ReadPrompt {
            path: prompt_path,
            source,
        })
        .map(|template| render_subagent_prompt(&template, mcp_server_details))
}

fn resolve_prompt_path(
    base_dir: &Path,
    configured: &ConfiguredSubagent,
    phase: TurnPhase,
) -> PathBuf {
    let default_path = if configured.prompt_path.is_absolute() {
        configured.prompt_path.clone()
    } else {
        base_dir.join(&configured.prompt_path)
    };
    let Some(file_name) = default_path.file_name().and_then(|value| value.to_str()) else {
        return default_path;
    };
    if let Some(prefix) = file_name.strip_suffix(".prompt.md") {
        let phase_path =
            default_path.with_file_name(format!("{prefix}.{}.prompt.md", phase.as_str()));
        if phase_path.exists() {
            return phase_path;
        }
    }
    default_path
}

fn render_subagent_prompt(template: &str, mcp_server_details: &str) -> String {
    template.replace(MCP_SERVER_DETAILS_TAG, mcp_server_details)
}

fn default_true() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use crate::model::TurnPhase;

    use super::{
        ConfiguredSubagent, SubagentConfigError, SubagentRegistry, load_subagent_prompt,
        render_subagent_prompt,
    };

    #[test]
    fn subagent_prompt_replaces_mcp_server_details_tag() {
        let rendered = render_subagent_prompt(
            "Header\n<dynamic variable: MCP server details>\nFooter",
            "# MCP Full: postgres",
        );

        assert!(rendered.contains("Header"));
        assert!(rendered.contains("# MCP Full: postgres"));
        assert!(rendered.contains("Footer"));
        assert!(!rendered.contains("<dynamic variable: MCP server details>"));
    }

    #[test]
    fn load_subagent_prompt_renders_actual_tool_executor_template() {
        let base_dir = workspace_root().join("config");
        let configured = ConfiguredSubagent {
            subagent_type: "tool-executor".to_owned(),
            display_name: "Tool Executor".to_owned(),
            purpose: "Build executable MCP actions".to_owned(),
            when_to_use: "After selecting an MCP capability".to_owned(),
            target_requirements: "server_name, capability_kind, capability_id".to_owned(),
            result_summary: "Returns a tool call or resource read".to_owned(),
            prompt_path: PathBuf::from("subagents/tool-executor.prompt.md"),
            enabled: true,
            model_name: None,
        };

        let rendered = load_subagent_prompt(
            &base_dir,
            &configured,
            TurnPhase::Execution,
            "---\nserver:\n  logical_name: postgres\n---\n\n# MCP Full: postgres",
        )
        .expect("tool-executor prompt should load");

        assert!(rendered.contains("You are the `tool-executor` sub-agent."));
        assert!(
            rendered
                .contains("Treat Python script authoring and execution as a first-class workflow")
        );
        assert!(rendered.contains("Assume `pandas` and `numpy` are available"));
        assert!(rendered.contains("When current-turn todo context is present"));
        assert!(rendered.contains("Return `done` once the delegated goal has been completed"));
        assert!(!rendered.contains("<dynamic variable: MCP server details>"));
    }

    #[test]
    fn load_subagent_prompt_renders_actual_mcp_executor_template() {
        let base_dir = workspace_root().join("config");
        let configured = ConfiguredSubagent {
            subagent_type: "mcp-executor".to_owned(),
            display_name: "Mcp Executor".to_owned(),
            purpose: "Run delegated MCP work".to_owned(),
            when_to_use: "When MCP work is needed".to_owned(),
            target_requirements: "mcp target".to_owned(),
            result_summary:
                "Returns MCP tool or resource actions until the delegated goal is complete"
                    .to_owned(),
            prompt_path: PathBuf::from("subagents/mcp-executor.prompt.md"),
            enabled: true,
            model_name: None,
        };

        let rendered = load_subagent_prompt(
            &base_dir,
            &configured,
            TurnPhase::Execution,
            "---\nserver:\n  logical_name: ipl\n  protocol_name: postgres-mcp\n---\n\n# MCP Full: ipl",
        )
        .expect("mcp-executor prompt should load");

        assert!(rendered.contains("You are the `mcp-executor` sub-agent."));
        assert!(rendered.contains("You may use any allowed MCP tools or resources"));
        assert!(rendered.contains("Prefer simpler valid `SELECT` queries"));
        assert!(rendered.contains("stop and return `partial`"));
        assert!(!rendered.contains("<dynamic variable: MCP server details>"));
    }

    #[test]
    fn get_enabled_accepts_display_name_and_normalized_identifier() {
        let registry = SubagentRegistry {
            subagents: vec![ConfiguredSubagent {
                subagent_type: "mcp-executor".to_owned(),
                display_name: "Mcp Executor".to_owned(),
                purpose: "Run MCP work".to_owned(),
                when_to_use: "When an MCP action is needed".to_owned(),
                target_requirements: "mcp capability".to_owned(),
                result_summary: "Executes delegated MCP work".to_owned(),
                prompt_path: PathBuf::from("subagents/mcp-executor.prompt.md"),
                enabled: true,
                model_name: None,
            }],
        };

        assert_eq!(
            registry
                .get_enabled("mcp-executor")
                .expect("canonical type should resolve")
                .display_name,
            "Mcp Executor"
        );
        assert_eq!(
            registry
                .get_enabled("Mcp Executor")
                .expect("display name should resolve")
                .subagent_type,
            "mcp-executor"
        );
        assert_eq!(
            registry
                .get_enabled("MCP Executor")
                .expect("normalized display name should resolve")
                .subagent_type,
            "mcp-executor"
        );
    }

    #[test]
    fn get_enabled_still_rejects_unknown_identifier() {
        let registry = SubagentRegistry {
            subagents: vec![ConfiguredSubagent {
                subagent_type: "tool-executor".to_owned(),
                display_name: "Tool Executor".to_owned(),
                purpose: "Run local tool work".to_owned(),
                when_to_use: "When local tools are needed".to_owned(),
                target_requirements: "local tools scope".to_owned(),
                result_summary: "Executes delegated local tool work".to_owned(),
                prompt_path: PathBuf::from("subagents/tool-executor.prompt.md"),
                enabled: true,
                model_name: None,
            }],
        };

        let error = registry
            .get_enabled("missing-subagent")
            .expect_err("unknown identifier should fail");

        assert!(matches!(
            error,
            SubagentConfigError::UnknownSubagent { subagent_type }
                if subagent_type == "missing-subagent"
        ));
    }

    fn workspace_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .expect("workspace root should resolve")
    }
}
