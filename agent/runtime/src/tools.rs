//! Built-in tool registry and local tool execution helpers.

use std::{
    fs,
    path::{Component, Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::state::{LocalToolName, ServerName};

/// Static classification for one executable tool capability.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolFamily {
    Local,
    McpTool,
    McpResource,
}

/// One registered tool or resource the harness can reason about.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolDescriptor {
    pub tool_id: String,
    pub name: String,
    pub display_name: String,
    pub description: String,
    pub family: ToolFamily,
    pub executor: String,
    #[serde(default)]
    pub tags: Vec<String>,
    pub input_schema: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_name: Option<ServerName>,
}

impl ToolDescriptor {
    pub fn local(
        name: &str,
        display_name: &str,
        description: &str,
        tags: Vec<&str>,
        input_schema: Value,
    ) -> Self {
        Self {
            tool_id: format!("local.{name}"),
            name: name.to_owned(),
            display_name: display_name.to_owned(),
            description: description.to_owned(),
            family: ToolFamily::Local,
            executor: "tool-executor".to_owned(),
            tags: tags.into_iter().map(str::to_owned).collect(),
            input_schema,
            server_name: None,
        }
    }

    pub fn mcp_tool(
        server_name: &ServerName,
        name: &str,
        display_name: Option<&str>,
        description: Option<&str>,
        input_schema: Value,
    ) -> Self {
        Self {
            tool_id: format!("mcp.{}.tool.{name}", server_name.as_str()),
            name: name.to_owned(),
            display_name: display_name.unwrap_or(name).to_owned(),
            description: description.unwrap_or("No description").to_owned(),
            family: ToolFamily::McpTool,
            executor: "mcp-executor".to_owned(),
            tags: vec!["mcp".to_owned()],
            input_schema,
            server_name: Some(server_name.clone()),
        }
    }

    pub fn mcp_resource(
        server_name: &ServerName,
        uri: &str,
        display_name: Option<&str>,
        description: Option<&str>,
    ) -> Self {
        Self {
            tool_id: format!("mcp.{}.resource.{uri}", server_name.as_str()),
            name: uri.to_owned(),
            display_name: display_name.unwrap_or(uri).to_owned(),
            description: description.unwrap_or("No description").to_owned(),
            family: ToolFamily::McpResource,
            executor: "mcp-executor".to_owned(),
            tags: vec!["mcp".to_owned(), "resource_read".to_owned()],
            input_schema: Value::Null,
            server_name: Some(server_name.clone()),
        }
    }
}

/// Standardized result envelope produced by one local tool execution.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCallResultEnvelope {
    pub status: String,
    pub summary: String,
    pub payload: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LocalToolExecutionError {
    InvalidArguments(String),
    UnknownTool(String),
    Path(String),
    Io(String),
    Utf8(String),
}

impl std::fmt::Display for LocalToolExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidArguments(message)
            | Self::UnknownTool(message)
            | Self::Path(message)
            | Self::Io(message)
            | Self::Utf8(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for LocalToolExecutionError {}

#[derive(Debug, Deserialize)]
struct ReadFileArgs {
    path: String,
}

#[derive(Debug, Deserialize)]
struct WriteFileArgs {
    path: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct EditFileArgs {
    path: String,
    old_text: String,
    new_text: String,
}

pub fn builtin_local_tool_catalog() -> Vec<ToolDescriptor> {
    vec![
        ToolDescriptor::local(
            "read_file",
            "Read File",
            "Read the full UTF-8 contents of one file inside the working directory.",
            vec!["file_read"],
            json!({
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": { "type": "string", "minLength": 1 }
                },
                "additionalProperties": false
            }),
        ),
        ToolDescriptor::local(
            "write_file",
            "Write File",
            "Write full UTF-8 contents to one file inside the working directory.",
            vec!["file_write"],
            json!({
                "type": "object",
                "required": ["path", "content"],
                "properties": {
                    "path": { "type": "string", "minLength": 1 },
                    "content": { "type": "string" }
                },
                "additionalProperties": false
            }),
        ),
        ToolDescriptor::local(
            "edit_file",
            "Edit File",
            "Replace exactly one matching text block in a UTF-8 file inside the working directory.",
            vec!["file_write"],
            json!({
                "type": "object",
                "required": ["path", "old_text", "new_text"],
                "properties": {
                    "path": { "type": "string", "minLength": 1 },
                    "old_text": { "type": "string" },
                    "new_text": { "type": "string" }
                },
                "additionalProperties": false
            }),
        ),
    ]
}

pub fn execute_local_tool(
    tool_name: &LocalToolName,
    arguments: &Value,
    working_directory: &Path,
) -> Result<ToolCallResultEnvelope, LocalToolExecutionError> {
    match tool_name.as_str() {
        "read_file" => {
            let args: ReadFileArgs = serde_json::from_value(arguments.clone())
                .map_err(|error| LocalToolExecutionError::InvalidArguments(error.to_string()))?;
            execute_read_file(&args, working_directory)
        }
        "write_file" => {
            let args: WriteFileArgs = serde_json::from_value(arguments.clone())
                .map_err(|error| LocalToolExecutionError::InvalidArguments(error.to_string()))?;
            execute_write_file(&args, working_directory)
        }
        "edit_file" => {
            let args: EditFileArgs = serde_json::from_value(arguments.clone())
                .map_err(|error| LocalToolExecutionError::InvalidArguments(error.to_string()))?;
            execute_edit_file(&args, working_directory)
        }
        other => Err(LocalToolExecutionError::UnknownTool(format!(
            "unknown local tool `{other}`"
        ))),
    }
}

fn execute_read_file(
    args: &ReadFileArgs,
    working_directory: &Path,
) -> Result<ToolCallResultEnvelope, LocalToolExecutionError> {
    let path = resolve_tool_path(working_directory, &args.path, true)?;
    let bytes = fs::read(&path).map_err(|error| {
        LocalToolExecutionError::Io(format!("failed to read `{}`: {error}", path.display()))
    })?;
    let content = String::from_utf8(bytes).map_err(|error| {
        LocalToolExecutionError::Utf8(format!(
            "file `{}` is not valid UTF-8: {error}",
            path.display()
        ))
    })?;
    Ok(ToolCallResultEnvelope {
        status: "ok".to_owned(),
        summary: format!("read {} bytes from {}", content.len(), path.display()),
        payload: json!({
            "path": path.display().to_string(),
            "content": content,
        }),
        error: None,
    })
}

fn execute_write_file(
    args: &WriteFileArgs,
    working_directory: &Path,
) -> Result<ToolCallResultEnvelope, LocalToolExecutionError> {
    let path = resolve_tool_path(working_directory, &args.path, false)?;
    let parent = path.parent().ok_or_else(|| {
        LocalToolExecutionError::Path(format!(
            "file `{}` does not have a writable parent directory",
            path.display()
        ))
    })?;
    if !parent.exists() {
        return Err(LocalToolExecutionError::Path(format!(
            "parent directory `{}` does not exist",
            parent.display()
        )));
    }
    fs::write(&path, args.content.as_bytes()).map_err(|error| {
        LocalToolExecutionError::Io(format!("failed to write `{}`: {error}", path.display()))
    })?;
    Ok(ToolCallResultEnvelope {
        status: "ok".to_owned(),
        summary: format!("wrote {} bytes to {}", args.content.len(), path.display()),
        payload: json!({
            "path": path.display().to_string(),
            "bytes_written": args.content.len(),
        }),
        error: None,
    })
}

fn execute_edit_file(
    args: &EditFileArgs,
    working_directory: &Path,
) -> Result<ToolCallResultEnvelope, LocalToolExecutionError> {
    let path = resolve_tool_path(working_directory, &args.path, true)?;
    let bytes = fs::read(&path).map_err(|error| {
        LocalToolExecutionError::Io(format!("failed to read `{}`: {error}", path.display()))
    })?;
    let content = String::from_utf8(bytes).map_err(|error| {
        LocalToolExecutionError::Utf8(format!(
            "file `{}` is not valid UTF-8: {error}",
            path.display()
        ))
    })?;
    let match_count = content.match_indices(&args.old_text).count();
    if match_count == 0 {
        return Err(LocalToolExecutionError::Path(format!(
            "edit_file could not find the requested text in `{}`",
            path.display()
        )));
    }
    if match_count > 1 {
        return Err(LocalToolExecutionError::Path(format!(
            "edit_file found multiple matching blocks in `{}`",
            path.display()
        )));
    }
    let updated = content.replacen(&args.old_text, &args.new_text, 1);
    fs::write(&path, updated.as_bytes()).map_err(|error| {
        LocalToolExecutionError::Io(format!("failed to write `{}`: {error}", path.display()))
    })?;
    Ok(ToolCallResultEnvelope {
        status: "ok".to_owned(),
        summary: format!("edited {}", path.display()),
        payload: json!({
            "path": path.display().to_string(),
            "replacements": 1,
        }),
        error: None,
    })
}

fn resolve_tool_path(
    working_directory: &Path,
    raw_path: &str,
    must_exist: bool,
) -> Result<PathBuf, LocalToolExecutionError> {
    let root = working_directory.canonicalize().map_err(|error| {
        LocalToolExecutionError::Path(format!(
            "failed to resolve working directory `{}`: {error}",
            working_directory.display()
        ))
    })?;
    let provided = PathBuf::from(raw_path);
    let joined = if provided.is_absolute() {
        provided
    } else {
        root.join(provided)
    };
    let normalized = normalize_path(&joined);

    let resolved = if must_exist || normalized.exists() {
        normalized.canonicalize().map_err(|error| {
            LocalToolExecutionError::Path(format!(
                "failed to resolve path `{}`: {error}",
                normalized.display()
            ))
        })?
    } else {
        let parent = normalized.parent().ok_or_else(|| {
            LocalToolExecutionError::Path(format!(
                "path `{}` does not have a parent directory",
                normalized.display()
            ))
        })?;
        let resolved_parent = parent.canonicalize().map_err(|error| {
            LocalToolExecutionError::Path(format!(
                "failed to resolve parent directory `{}`: {error}",
                parent.display()
            ))
        })?;
        let file_name = normalized.file_name().ok_or_else(|| {
            LocalToolExecutionError::Path(format!(
                "path `{}` must point to a file",
                normalized.display()
            ))
        })?;
        resolved_parent.join(file_name)
    };

    if !resolved.starts_with(&root) {
        return Err(LocalToolExecutionError::Path(format!(
            "path `{}` escapes the working directory `{}`",
            resolved.display(),
            root.display()
        )));
    }

    Ok(resolved)
}

fn normalize_path(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    normalized
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use serde_json::json;

    use super::{builtin_local_tool_catalog, execute_local_tool};
    use crate::state::LocalToolName;

    #[test]
    fn builtin_catalog_contains_three_local_tools() {
        let tools = builtin_local_tool_catalog();
        assert_eq!(tools.len(), 3);
        assert_eq!(tools[0].tool_id, "local.read_file");
        assert_eq!(tools[1].tool_id, "local.write_file");
        assert_eq!(tools[2].tool_id, "local.edit_file");
    }

    #[test]
    fn write_file_creates_file_inside_workspace() {
        let temp_dir = temp_dir("tool-write");
        let result = execute_local_tool(
            &LocalToolName::new("write_file").expect("valid tool"),
            &json!({"path":"notes.txt","content":"hello"}),
            &temp_dir,
        )
        .expect("write should succeed");
        assert_eq!(result.status, "ok");
        assert_eq!(
            std::fs::read_to_string(temp_dir.join("notes.txt")).expect("file should exist"),
            "hello"
        );
    }

    #[test]
    fn edit_file_requires_exact_single_match() {
        let temp_dir = temp_dir("tool-edit");
        std::fs::write(temp_dir.join("notes.txt"), "hello\nhello\n").expect("seed file");
        let error = execute_local_tool(
            &LocalToolName::new("edit_file").expect("valid tool"),
            &json!({"path":"notes.txt","old_text":"hello","new_text":"bye"}),
            &temp_dir,
        )
        .expect_err("edit should fail");
        assert!(error.to_string().contains("multiple"));
    }

    fn temp_dir(prefix: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should move forward")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("ai-data-analyst-{prefix}-{unique}"));
        std::fs::create_dir_all(&path).expect("temp dir should exist");
        path
    }
}
