//! Loading, normalizing, and validating the MCP registry.
//!
//! The registry is intentionally small and JSON-based. This crate owns:
//! - backwards compatibility with the original stdio-only schema
//! - validation rules the runtime depends on before connecting to servers
//! - persistence helpers used by the interactive CLI when editing registry entries

use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fmt, fs, io,
    path::Path,
};

use serde::{Deserialize, Serialize};

/// Launch configuration for a single MCP server entry.
///
/// The current registry format is intentionally small. It is only concerned
/// with what is necessary to find a named server and spawn it locally.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct McpServerConfig {
    /// Unique logical server name used everywhere else in the workspace.
    pub name: String,
    /// Preferred transport shape in the current registry schema.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transport: Option<McpTransportConfig>,
    /// Legacy stdio command retained for backwards-compatible decoding.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub command: String,
    /// Legacy stdio arguments retained for backwards-compatible decoding.
    #[serde(default)]
    pub args: Vec<String>,
    /// Legacy stdio environment retained for backwards-compatible decoding.
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// Optional human-authored description surfaced to operators and prompts.
    #[serde(default)]
    pub description: Option<String>,
}

/// Transport options supported by the registry and client.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum McpTransportConfig {
    Stdio {
        /// Executable or shell entrypoint for the MCP server process.
        command: String,
        /// Command-line arguments passed to the stdio process.
        #[serde(default)]
        args: Vec<String>,
        /// Extra environment variables injected into the child process.
        #[serde(default)]
        env: HashMap<String, String>,
    },
    StreamableHttp {
        /// Remote Streamable HTTP endpoint for the MCP server.
        url: String,
        /// Additional HTTP headers, commonly used for auth.
        #[serde(default)]
        headers: HashMap<String, String>,
    },
}

/// In-memory representation of the JSON registry file.
///
/// The harness loads this once, validates it eagerly, and then does name-based
/// lookup when it needs to connect to a specific MCP server.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct McpRegistry {
    #[serde(default)]
    pub servers: Vec<McpServerConfig>,
}

impl McpRegistry {
    /// Loads, parses, and validates the registry from disk.
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let raw = fs::read_to_string(path).map_err(ConfigError::Io)?;
        Self::from_json_str(&raw)
    }

    /// Parses raw JSON into a registry and applies runtime validation rules.
    ///
    /// JSON parsing alone only checks structure and field types. The follow-up
    /// validation step enforces semantic requirements such as unique names.
    pub fn from_json_str(raw: &str) -> Result<Self, ConfigError> {
        let registry = serde_json::from_str::<Self>(raw).map_err(ConfigError::Json)?;
        registry.validate()?;
        Ok(registry)
    }

    /// Resolves one configured server by its unique `name`.
    pub fn get(&self, name: &str) -> Result<&McpServerConfig, ConfigError> {
        self.servers
            .iter()
            .find(|server| server.name == name)
            .ok_or_else(|| ConfigError::UnknownServer(name.to_owned()))
    }

    /// Inserts a new server or replaces an existing one with the same name.
    pub fn upsert_server(&mut self, server: McpServerConfig) -> bool {
        if let Some(existing) = self
            .servers
            .iter_mut()
            .find(|existing| existing.name == server.name)
        {
            *existing = server;
            true
        } else {
            self.servers.push(server);
            false
        }
    }

    /// Removes a configured server by name.
    pub fn remove_server(&mut self, name: &str) -> Result<McpServerConfig, ConfigError> {
        let index = self
            .servers
            .iter()
            .position(|server| server.name == name)
            .ok_or_else(|| ConfigError::UnknownServer(name.to_owned()))?;
        Ok(self.servers.remove(index))
    }

    /// Persists the registry as pretty JSON after normalizing transport fields.
    pub fn save_to_path(&self, path: impl AsRef<Path>) -> Result<(), ConfigError> {
        let path = path.as_ref();
        let normalized = self.normalized();
        normalized.validate()?;
        let body = serde_json::to_string_pretty(&normalized).map_err(ConfigError::Json)?;

        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent).map_err(ConfigError::Io)?;
        }

        let file_name = path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("mcp_servers.json");
        let temp_path = path.with_file_name(format!(".{file_name}.tmp"));
        fs::write(&temp_path, body).map_err(ConfigError::Io)?;
        fs::rename(&temp_path, path).map_err(ConfigError::Io)
    }

    /// Enforces the invariants the runtime depends on.
    ///
    /// This is where we reject blank lookup keys, blank commands, and name
    /// collisions that would otherwise make server selection ambiguous.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let mut seen = HashSet::new();

        for server in &self.servers {
            if server.name.trim().is_empty() {
                return Err(ConfigError::EmptyField {
                    server_name: None,
                    field: "name",
                });
            }

            validate_transport(server)?;

            // Server names are the registry lookup key, so collisions are a config error.
            if !seen.insert(server.name.clone()) {
                return Err(ConfigError::DuplicateServerName(server.name.clone()));
            }
        }

        Ok(())
    }

    fn normalized(&self) -> Self {
        Self {
            servers: self
                .servers
                .iter()
                .map(McpServerConfig::normalized)
                .collect(),
        }
    }
}

/// Registry-level loading and validation failures.
#[derive(Debug)]
pub enum ConfigError {
    /// Reading the registry file from disk failed.
    Io(io::Error),
    /// JSON parsing failed before semantic validation could run.
    Json(serde_json::Error),
    /// Multiple entries advertised the same logical server name.
    DuplicateServerName(String),
    /// A required field was present but blank after trimming.
    EmptyField {
        server_name: Option<String>,
        field: &'static str,
    },
    /// A caller requested a server name that is not present in the registry.
    UnknownServer(String),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "failed to read MCP registry: {err}"),
            Self::Json(err) => write!(f, "failed to parse JSON registry: {err}"),
            Self::DuplicateServerName(name) => {
                write!(f, "duplicate MCP server name `{name}` in registry")
            }
            Self::EmptyField {
                server_name: Some(server_name),
                field,
            } => write!(f, "MCP server `{server_name}` has an empty `{field}` field"),
            Self::EmptyField {
                server_name: None,
                field,
            } => write!(
                f,
                "MCP registry contains a server with an empty `{field}` field"
            ),
            Self::UnknownServer(name) => write!(f, "unknown MCP server `{name}`"),
        }
    }
}

impl Error for ConfigError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::Json(err) => Some(err),
            Self::DuplicateServerName(_) | Self::EmptyField { .. } | Self::UnknownServer(_) => None,
        }
    }
}

impl McpServerConfig {
    /// Resolves the transport configuration while preserving backwards
    /// compatibility with the original stdio-only schema.
    pub fn resolved_transport(&self) -> McpTransportConfig {
        self.transport
            .clone()
            .unwrap_or_else(|| McpTransportConfig::Stdio {
                command: self.command.clone(),
                args: self.args.clone(),
                env: self.env.clone(),
            })
    }

    fn normalized(&self) -> Self {
        let mut normalized = self.clone();
        // Persist the canonical transport form so reads and writes converge on
        // the same schema even if the input used legacy top-level fields.
        normalized.transport = Some(self.resolved_transport());
        normalized.command.clear();
        normalized.args.clear();
        normalized.env.clear();
        normalized
    }
}

fn validate_transport(server: &McpServerConfig) -> Result<(), ConfigError> {
    match server.resolved_transport() {
        McpTransportConfig::Stdio { command, env, .. } => {
            if command.trim().is_empty() {
                return Err(ConfigError::EmptyField {
                    server_name: Some(server.name.clone()),
                    field: "command",
                });
            }

            if env.keys().any(|key| key.trim().is_empty()) {
                return Err(ConfigError::EmptyField {
                    server_name: Some(server.name.clone()),
                    field: "env key",
                });
            }
        }
        McpTransportConfig::StreamableHttp { url, headers } => {
            if url.trim().is_empty() || !looks_like_http_url(&url) {
                return Err(ConfigError::EmptyField {
                    server_name: Some(server.name.clone()),
                    field: "transport.url",
                });
            }

            if headers.keys().any(|key| key.trim().is_empty()) {
                return Err(ConfigError::EmptyField {
                    server_name: Some(server.name.clone()),
                    field: "transport.headers key",
                });
            }
        }
    }

    Ok(())
}

fn looks_like_http_url(value: &str) -> bool {
    let trimmed = value.trim();
    trimmed
        .strip_prefix("http://")
        .or_else(|| trimmed.strip_prefix("https://"))
        .is_some_and(|rest| !rest.trim().is_empty())
}

#[cfg(test)]
mod tests {
    //! Regression tests for registry parsing, validation, and persistence.

    use std::{
        collections::HashMap,
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::{ConfigError, McpRegistry, McpServerConfig, McpTransportConfig};

    #[test]
    fn registry_loads_valid_json() {
        let registry = McpRegistry::from_json_str(
            r#"{
                "servers": [
                    {
                        "name": "local-sql",
                        "command": "uvx",
                        "args": ["mcp-server-sqlite"],
                        "env": {
                            "SQLITE_PATH": "local.db"
                        }
                    }
                ]
            }"#,
        )
        .expect("registry should parse");

        let server = registry.get("local-sql").expect("server must exist");
        assert_eq!(server.command, "uvx");
        assert_eq!(server.args, vec!["mcp-server-sqlite"]);
        assert_eq!(
            server.env.get("SQLITE_PATH").map(String::as_str),
            Some("local.db")
        );
    }

    #[test]
    fn registry_loads_explicit_stdio_transport_json() {
        let registry = McpRegistry::from_json_str(
            r#"{
                "servers": [
                    {
                        "name": "local-sql",
                        "transport": {
                            "type": "stdio",
                            "command": "uvx",
                            "args": ["mcp-server-sqlite"],
                            "env": {
                                "SQLITE_PATH": "local.db"
                            }
                        }
                    }
                ]
            }"#,
        )
        .expect("registry should parse");

        let server = registry.get("local-sql").expect("server must exist");
        assert_eq!(
            server.resolved_transport(),
            McpTransportConfig::Stdio {
                command: "uvx".to_owned(),
                args: vec!["mcp-server-sqlite".to_owned()],
                env: HashMap::from([("SQLITE_PATH".to_owned(), "local.db".to_owned())]),
            }
        );
    }

    #[test]
    fn registry_loads_streamable_http_transport_json() {
        let registry = McpRegistry::from_json_str(
            r#"{
                "servers": [
                    {
                        "name": "remote-sql",
                        "transport": {
                            "type": "streamable_http",
                            "url": "https://example.com/mcp",
                            "headers": {
                                "Authorization": "Bearer token"
                            }
                        }
                    }
                ]
            }"#,
        )
        .expect("registry should parse");

        let server = registry.get("remote-sql").expect("server must exist");
        assert_eq!(
            server.resolved_transport(),
            McpTransportConfig::StreamableHttp {
                url: "https://example.com/mcp".to_owned(),
                headers: HashMap::from([("Authorization".to_owned(), "Bearer token".to_owned())]),
            }
        );
    }

    #[test]
    fn registry_rejects_duplicate_names() {
        let error = McpRegistry::from_json_str(
            r#"{
                "servers": [
                    { "name": "dup", "command": "first", "args": [] },
                    { "name": "dup", "command": "second", "args": [] }
                ]
            }"#,
        )
        .expect_err("duplicate names must fail");

        assert!(matches!(error, ConfigError::DuplicateServerName(name) if name == "dup"));
    }

    #[test]
    fn registry_rejects_empty_command() {
        let error = McpRegistry::from_json_str(
            r#"{
                "servers": [
                    { "name": "broken", "command": "   ", "args": [] }
                ]
            }"#,
        )
        .expect_err("empty command must fail");

        assert!(matches!(
            error,
            ConfigError::EmptyField {
                server_name: Some(name),
                field: "command",
            } if name == "broken"
        ));
    }

    #[test]
    fn registry_rejects_invalid_streamable_http_url() {
        let error = McpRegistry::from_json_str(
            r#"{
                "servers": [
                    {
                        "name": "broken",
                        "transport": {
                            "type": "streamable_http",
                            "url": "not-a-url"
                        }
                    }
                ]
            }"#,
        )
        .expect_err("invalid url must fail");

        assert!(matches!(
            error,
            ConfigError::EmptyField {
                server_name: Some(name),
                field: "transport.url",
            } if name == "broken"
        ));
    }

    #[test]
    fn registry_rejects_blank_env_keys() {
        let error = McpRegistry::from_json_str(
            r#"{
                "servers": [
                    {
                        "name": "broken",
                        "command": "uvx",
                        "args": ["mcp-server-sqlite"],
                        "env": {
                            "   ": "value"
                        }
                    }
                ]
            }"#,
        )
        .expect_err("blank env keys must fail");

        assert!(matches!(
            error,
            ConfigError::EmptyField {
                server_name: Some(name),
                field: "env key",
            } if name == "broken"
        ));
    }

    #[test]
    fn registry_rejects_blank_http_header_keys() {
        let error = McpRegistry::from_json_str(
            r#"{
                "servers": [
                    {
                        "name": "broken",
                        "transport": {
                            "type": "streamable_http",
                            "url": "https://example.com/mcp",
                            "headers": {
                                "   ": "value"
                            }
                        }
                    }
                ]
            }"#,
        )
        .expect_err("blank header keys must fail");

        assert!(matches!(
            error,
            ConfigError::EmptyField {
                server_name: Some(name),
                field: "transport.headers key",
            } if name == "broken"
        ));
    }

    #[test]
    fn registry_rejects_unknown_server_lookup() {
        let registry = McpRegistry::from_json_str(r#"{ "servers": [] }"#).expect("valid config");

        let error = registry
            .get("missing")
            .expect_err("missing lookup must fail");
        assert!(matches!(error, ConfigError::UnknownServer(name) if name == "missing"));
    }

    #[test]
    fn registry_rejects_removed_semantic_context_field() {
        let error = McpRegistry::from_json_str(
            r#"{
                "servers": [
                    {
                        "name": "crm",
                        "command": "uvx",
                        "semantic_context": {
                            "injection": "progressive"
                        }
                    }
                ]
            }"#,
        )
        .expect_err("removed semantic_context field must fail");

        match error {
            ConfigError::Json(error) => {
                assert!(
                    error.to_string().contains("semantic_context"),
                    "expected unknown-field error mentioning semantic_context, got: {error}"
                );
            }
            other => panic!("expected JSON error for unknown semantic_context field, got {other}"),
        }
    }

    #[test]
    fn registry_upsert_replaces_existing_server_in_place() {
        let mut registry = McpRegistry::from_json_str(
            r#"{
                "servers": [
                    { "name": "crm", "command": "uvx" },
                    { "name": "analytics", "command": "uvx" }
                ]
            }"#,
        )
        .expect("config should parse");

        let replaced = registry.upsert_server(McpServerConfig {
            name: "crm".to_owned(),
            transport: Some(McpTransportConfig::StreamableHttp {
                url: "https://example.com/mcp".to_owned(),
                headers: HashMap::new(),
            }),
            command: String::new(),
            args: Vec::new(),
            env: HashMap::new(),
            description: Some("CRM".to_owned()),
        });

        assert!(replaced);
        assert_eq!(registry.servers[0].description.as_deref(), Some("CRM"));
        assert_eq!(registry.servers[1].name, "analytics");
    }

    #[test]
    fn registry_save_normalizes_to_explicit_transport_shape() {
        let registry = McpRegistry::from_json_str(
            r#"{
                "servers": [
                    {
                        "name": "crm",
                        "command": "uvx",
                        "args": ["mcp-server-sqlite"]
                    }
                ]
            }"#,
        )
        .expect("config should parse");
        let temp_dir = temp_dir("registry-save");
        let path = temp_dir.join("mcp_servers.json");

        registry.save_to_path(&path).expect("save should succeed");

        let saved = McpRegistry::load_from_path(&path).expect("saved registry should reload");
        assert_eq!(saved.servers.len(), 1);
        assert!(matches!(
            saved.servers[0].transport,
            Some(McpTransportConfig::Stdio { .. })
        ));
        assert!(saved.servers[0].command.is_empty());
    }

    #[test]
    fn registry_remove_deletes_named_entry() {
        let mut registry = McpRegistry::from_json_str(
            r#"{
                "servers": [
                    { "name": "crm", "command": "uvx" },
                    { "name": "analytics", "command": "uvx" }
                ]
            }"#,
        )
        .expect("config should parse");

        let removed = registry
            .remove_server("crm")
            .expect("server should be removed");
        assert_eq!(removed.name, "crm");
        assert_eq!(registry.servers.len(), 1);
        assert_eq!(registry.servers[0].name, "analytics");
    }

    fn temp_dir(prefix: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be valid")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("mcp-config-{prefix}-{unique}"));
        fs::create_dir_all(&path).expect("temp dir create");
        path
    }
}
