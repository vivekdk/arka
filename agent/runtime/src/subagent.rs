//! Sub-agent registry loading and prompt helpers.

use std::{
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

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
            .find(|subagent| subagent.enabled && subagent.subagent_type == subagent_type)
            .ok_or_else(|| SubagentConfigError::UnknownSubagent {
                subagent_type: subagent_type.to_owned(),
            })
    }
}

pub fn load_subagent_prompt(
    base_dir: &Path,
    configured: &ConfiguredSubagent,
    mcp_server_details: &str,
) -> Result<String, SubagentConfigError> {
    let prompt_path = resolve_prompt_path(base_dir, configured);
    fs::read_to_string(&prompt_path)
        .map_err(|source| SubagentConfigError::ReadPrompt {
            path: prompt_path,
            source,
        })
        .map(|template| render_subagent_prompt(&template, mcp_server_details))
}

fn resolve_prompt_path(base_dir: &Path, configured: &ConfiguredSubagent) -> PathBuf {
    if configured.prompt_path.is_absolute() {
        configured.prompt_path.clone()
    } else {
        base_dir.join(&configured.prompt_path)
    }
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

    use super::{ConfiguredSubagent, load_subagent_prompt, render_subagent_prompt};

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
            "---\nserver:\n  logical_name: postgres\n---\n\n# MCP Full: postgres",
        )
        .expect("tool-executor prompt should load");

        assert!(rendered.contains("You are the `tool-executor` sub-agent."));
        assert!(rendered.contains("`read_file` reads the full UTF-8 contents of one file."));
        assert!(rendered.contains("`write_file` writes the full UTF-8 contents of one file."));
        assert!(rendered.contains(
            "`edit_file` replaces exactly one matching `old_text` block with `new_text` in a UTF-8 file."
        ));
        assert!(rendered.contains(
            "Use `read_file` before mutating when the current file contents are not already known."
        ));
        assert!(rendered.contains(
            "If `edit_file` cannot find the requested text, read the file again and reassess before continuing."
        ));
        assert!(rendered.contains("The delegated prompt will include the working directory and the available local tools."));
        assert!(!rendered.contains("<dynamic variable: MCP server details>"));
    }

    fn workspace_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .expect("workspace root should resolve")
    }
}
