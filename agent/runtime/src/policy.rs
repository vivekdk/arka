//! Tool policy evaluation and adapter-facing mask plans.

use std::{
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::{
    state::ResponseClient,
    tools::{ToolDescriptor, ToolFamily},
};

/// Runtime phase used by tool policy rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolPolicyPhase {
    MainStep,
    DelegatedPlanning,
    DelegatedExecution,
}

/// Policy action applied when a rule matches one tool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolPolicyRuleAction {
    Allow,
    Deny,
}

/// Provider enforcement mode used for one evaluated tool mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolMaskEnforcementMode {
    DecodeTimeMask,
    AdapterFallback,
    TerminalOnly,
}

/// One MCP tool explicitly permitted by the mask.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllowedMcpTool {
    pub server_name: String,
    pub tool_name: String,
}

/// One MCP resource explicitly permitted by the mask.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllowedMcpResource {
    pub server_name: String,
    pub resource_uri: String,
}

/// Debuggable allow/deny outcome for one candidate tool.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolMaskDecision {
    pub tool_id: String,
    pub allowed: bool,
    pub reason: String,
}

/// Adapter-facing evaluated tool mask for one step.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolMaskPlan {
    pub enforcement_mode: ToolMaskEnforcementMode,
    pub allowed_tool_ids: Vec<String>,
    pub denied_tool_ids: Vec<String>,
    #[serde(default)]
    pub decisions: Vec<ToolMaskDecision>,
    #[serde(default)]
    pub allowed_local_tools: Vec<String>,
    #[serde(default)]
    pub allowed_mcp_tools: Vec<AllowedMcpTool>,
    #[serde(default)]
    pub allowed_mcp_resources: Vec<AllowedMcpResource>,
}

impl ToolMaskPlan {
    pub fn terminal_only(reason: impl Into<String>) -> Self {
        Self {
            enforcement_mode: ToolMaskEnforcementMode::TerminalOnly,
            allowed_tool_ids: Vec::new(),
            denied_tool_ids: Vec::new(),
            decisions: vec![ToolMaskDecision {
                tool_id: "terminal_only".to_owned(),
                allowed: false,
                reason: reason.into(),
            }],
            allowed_local_tools: Vec::new(),
            allowed_mcp_tools: Vec::new(),
            allowed_mcp_resources: Vec::new(),
        }
    }
}

/// Runtime context matched against tool policy rules.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolPolicyContext {
    pub executor: String,
    pub response_client: ResponseClient,
    pub phase: ToolPolicyPhase,
    pub environment: Option<String>,
    pub working_directory: PathBuf,
}

/// One configurable allow/deny rule.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolPolicyRule {
    pub action: ToolPolicyRuleAction,
    #[serde(default)]
    pub executors: Vec<String>,
    #[serde(default)]
    pub tool_ids: Vec<String>,
    #[serde(default)]
    pub families: Vec<ToolFamily>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub response_clients: Vec<ResponseClient>,
    #[serde(default)]
    pub phases: Vec<ToolPolicyPhase>,
    #[serde(default)]
    pub environments: Vec<String>,
    #[serde(default)]
    pub reason: Option<String>,
}

impl ToolPolicyRule {
    fn matches(&self, context: &ToolPolicyContext, tool: &ToolDescriptor) -> bool {
        matches_optional_string(&self.executors, &context.executor)
            && matches_optional_string(&self.tool_ids, &tool.tool_id)
            && matches_optional_enum(&self.families, &tool.family)
            && matches_optional_tags(&self.tags, &tool.tags)
            && matches_optional_enum(&self.response_clients, &context.response_client)
            && matches_optional_enum(&self.phases, &context.phase)
            && matches_optional_string_option(&self.environments, context.environment.as_deref())
    }

    fn decision_reason(&self) -> String {
        self.reason
            .clone()
            .unwrap_or_else(|| format!("matched {:?} rule", self.action))
    }
}

/// Overlay file loaded from JSON config.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolPolicyOverlay {
    #[serde(default)]
    pub rules: Vec<ToolPolicyRule>,
}

/// Errors while loading a JSON tool policy overlay.
#[derive(Debug, thiserror::Error)]
pub enum ToolPolicyLoadError {
    #[error("failed to read tool policy `{path}`: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse tool policy `{path}`: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
}

/// Pure evaluator for runtime tool availability.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ToolPolicyEngine {
    default_rules: Vec<ToolPolicyRule>,
    overlay_rules: Vec<ToolPolicyRule>,
}

impl ToolPolicyEngine {
    pub fn new(overlay: Option<ToolPolicyOverlay>) -> Self {
        Self {
            default_rules: default_policy_rules(),
            overlay_rules: overlay.unwrap_or_default().rules,
        }
    }

    pub fn load_overlay(
        path: Option<&Path>,
    ) -> Result<Option<ToolPolicyOverlay>, ToolPolicyLoadError> {
        let Some(path) = path else {
            return Ok(None);
        };
        if !path.exists() {
            return Ok(None);
        }
        let raw = fs::read_to_string(path).map_err(|source| ToolPolicyLoadError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        let overlay = serde_json::from_str(&raw).map_err(|source| ToolPolicyLoadError::Parse {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(Some(overlay))
    }

    pub fn evaluate(
        &self,
        context: &ToolPolicyContext,
        candidates: &[ToolDescriptor],
    ) -> ToolMaskPlan {
        let mut allowed_tool_ids = Vec::new();
        let mut denied_tool_ids = Vec::new();
        let mut allowed_local_tools = Vec::new();
        let mut allowed_mcp_tools = Vec::new();
        let mut allowed_mcp_resources = Vec::new();
        let mut decisions = Vec::new();

        for tool in candidates {
            let mut allowed = true;
            let mut reason = "allowed by structural scope".to_owned();
            for rule in self
                .default_rules
                .iter()
                .chain(self.overlay_rules.iter())
                .filter(|rule| rule.matches(context, tool))
            {
                allowed = matches!(rule.action, ToolPolicyRuleAction::Allow);
                reason = rule.decision_reason();
            }

            if allowed {
                allowed_tool_ids.push(tool.tool_id.clone());
                match tool.family {
                    ToolFamily::Local => allowed_local_tools.push(tool.name.clone()),
                    ToolFamily::McpTool => {
                        allowed_mcp_tools.push(AllowedMcpTool {
                            server_name: tool
                                .server_name
                                .as_ref()
                                .map(ToString::to_string)
                                .unwrap_or_default(),
                            tool_name: tool.name.clone(),
                        });
                    }
                    ToolFamily::McpResource => {
                        allowed_mcp_resources.push(AllowedMcpResource {
                            server_name: tool
                                .server_name
                                .as_ref()
                                .map(ToString::to_string)
                                .unwrap_or_default(),
                            resource_uri: tool.name.clone(),
                        });
                    }
                }
            } else {
                denied_tool_ids.push(tool.tool_id.clone());
            }

            decisions.push(ToolMaskDecision {
                tool_id: tool.tool_id.clone(),
                allowed,
                reason,
            });
        }

        ToolMaskPlan {
            enforcement_mode: if allowed_tool_ids.is_empty() {
                ToolMaskEnforcementMode::TerminalOnly
            } else {
                ToolMaskEnforcementMode::AdapterFallback
            },
            allowed_tool_ids,
            denied_tool_ids,
            decisions,
            allowed_local_tools,
            allowed_mcp_tools,
            allowed_mcp_resources,
        }
    }
}

fn default_policy_rules() -> Vec<ToolPolicyRule> {
    vec![
        ToolPolicyRule {
            action: ToolPolicyRuleAction::Deny,
            executors: vec!["tool-executor".to_owned()],
            tool_ids: Vec::new(),
            families: Vec::new(),
            tags: vec!["file_write".to_owned()],
            response_clients: vec![ResponseClient::WhatsApp],
            phases: vec![ToolPolicyPhase::DelegatedExecution],
            environments: Vec::new(),
            reason: Some("file writes are disabled for WhatsApp sessions by default".to_owned()),
        },
        ToolPolicyRule {
            action: ToolPolicyRuleAction::Deny,
            executors: vec!["tool-executor".to_owned()],
            tool_ids: Vec::new(),
            families: Vec::new(),
            tags: vec!["command_exec".to_owned()],
            response_clients: vec![ResponseClient::WhatsApp],
            phases: vec![ToolPolicyPhase::DelegatedExecution],
            environments: Vec::new(),
            reason: Some(
                "command execution is disabled for WhatsApp sessions by default".to_owned(),
            ),
        },
    ]
}

fn matches_optional_string(filters: &[String], candidate: &str) -> bool {
    filters.is_empty() || filters.iter().any(|value| value == candidate)
}

fn matches_optional_string_option(filters: &[String], candidate: Option<&str>) -> bool {
    filters.is_empty()
        || candidate
            .map(|candidate| filters.iter().any(|value| value == candidate))
            .unwrap_or(false)
}

fn matches_optional_enum<T: PartialEq>(filters: &[T], candidate: &T) -> bool {
    filters.is_empty() || filters.iter().any(|value| value == candidate)
}

fn matches_optional_tags(filters: &[String], tags: &[String]) -> bool {
    filters.is_empty()
        || filters
            .iter()
            .any(|filter| tags.iter().any(|tag| tag == filter))
}

#[cfg(test)]
mod tests {
    use super::{
        ToolMaskEnforcementMode, ToolPolicyContext, ToolPolicyEngine, ToolPolicyOverlay,
        ToolPolicyPhase, ToolPolicyRule, ToolPolicyRuleAction,
    };
    use crate::{
        state::ResponseClient,
        tools::{ToolDescriptor, builtin_local_tool_catalog},
    };

    #[test]
    fn overlay_rule_can_deny_one_tool() {
        let engine = ToolPolicyEngine::new(Some(ToolPolicyOverlay {
            rules: vec![ToolPolicyRule {
                action: ToolPolicyRuleAction::Deny,
                executors: vec!["tool-executor".to_owned()],
                tool_ids: vec!["local.write_file".to_owned()],
                families: Vec::new(),
                tags: Vec::new(),
                response_clients: Vec::new(),
                phases: vec![ToolPolicyPhase::DelegatedExecution],
                environments: Vec::new(),
                reason: Some("test deny".to_owned()),
            }],
        }));

        let plan = engine.evaluate(
            &ToolPolicyContext {
                executor: "tool-executor".to_owned(),
                response_client: ResponseClient::Cli,
                phase: ToolPolicyPhase::DelegatedExecution,
                environment: None,
                working_directory: std::env::temp_dir(),
            },
            &builtin_local_tool_catalog(),
        );

        assert!(plan.allowed_local_tools.contains(&"read_file".to_owned()));
        assert!(!plan.allowed_local_tools.contains(&"write_file".to_owned()));
        assert_eq!(
            plan.enforcement_mode,
            ToolMaskEnforcementMode::AdapterFallback
        );
    }

    #[test]
    fn whatsapp_denies_command_exec_by_default() {
        let plan = ToolPolicyEngine::new(None).evaluate(
            &ToolPolicyContext {
                executor: "tool-executor".to_owned(),
                response_client: ResponseClient::WhatsApp,
                phase: ToolPolicyPhase::DelegatedExecution,
                environment: None,
                working_directory: std::env::temp_dir(),
            },
            &builtin_local_tool_catalog(),
        );

        assert!(plan.allowed_local_tools.contains(&"read_file".to_owned()));
        assert!(plan.allowed_local_tools.contains(&"glob".to_owned()));
        assert!(!plan.allowed_local_tools.contains(&"write_file".to_owned()));
        assert!(!plan.allowed_local_tools.contains(&"bash".to_owned()));
    }

    #[test]
    fn empty_candidates_becomes_terminal_only() {
        let plan = ToolPolicyEngine::new(None).evaluate(
            &ToolPolicyContext {
                executor: "tool-executor".to_owned(),
                response_client: ResponseClient::Cli,
                phase: ToolPolicyPhase::DelegatedExecution,
                environment: None,
                working_directory: std::env::temp_dir(),
            },
            &Vec::<ToolDescriptor>::new(),
        );

        assert_eq!(plan.enforcement_mode, ToolMaskEnforcementMode::TerminalOnly);
        assert!(plan.allowed_tool_ids.is_empty());
    }
}
