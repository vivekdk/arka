//! Built-in tool registry and local tool execution helpers.

use std::{
    fs,
    path::{Component, Path, PathBuf},
    process::Stdio,
    time::Duration,
};

use glob::{MatchOptions, Pattern, glob_with};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncRead, AsyncReadExt},
    process::Command,
    task::JoinHandle,
    time::timeout,
};

use crate::state::{LocalToolName, ServerName};

const MAX_GLOB_MATCHES: usize = 1_000;
const MAX_SHELL_OUTPUT_BYTES: usize = 32 * 1024;

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

#[derive(Debug, Deserialize)]
struct GlobArgs {
    pattern: String,
    #[serde(default)]
    excludes: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct BashArgs {
    command: String,
    #[serde(default)]
    timeout_ms: Option<u64>,
}

#[derive(Debug)]
struct CapturedStream {
    text: String,
    truncated: bool,
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
        ToolDescriptor::local(
            "glob",
            "Glob",
            "Find workspace-relative file or directory paths matching a glob pattern, with optional exclude globs.",
            vec!["file_discovery"],
            json!({
                "type": "object",
                "required": ["pattern"],
                "properties": {
                    "pattern": { "type": "string", "minLength": 1 },
                    "excludes": {
                        "type": "array",
                        "items": { "type": "string", "minLength": 1 }
                    }
                },
                "additionalProperties": false
            }),
        ),
        ToolDescriptor::local(
            "bash",
            "Bash",
            "Run one non-interactive bash command inside the working directory and capture exit code, stdout, and stderr.",
            vec!["command_exec"],
            json!({
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": { "type": "string", "minLength": 1 },
                    "timeout_ms": { "type": "integer", "minimum": 1 }
                },
                "additionalProperties": false
            }),
        ),
    ]
}

pub async fn execute_local_tool(
    tool_name: &LocalToolName,
    arguments: &Value,
    working_directory: &Path,
    remaining_turn_budget: Duration,
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
        "glob" => {
            let args: GlobArgs = serde_json::from_value(arguments.clone())
                .map_err(|error| LocalToolExecutionError::InvalidArguments(error.to_string()))?;
            execute_glob(&args, working_directory)
        }
        "bash" => {
            let args: BashArgs = serde_json::from_value(arguments.clone())
                .map_err(|error| LocalToolExecutionError::InvalidArguments(error.to_string()))?;
            execute_bash(&args, working_directory, remaining_turn_budget).await
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
        summary: format!("read file {}\n{}", path.display(), content),
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

fn execute_glob(
    args: &GlobArgs,
    working_directory: &Path,
) -> Result<ToolCallResultEnvelope, LocalToolExecutionError> {
    validate_workspace_glob(&args.pattern)?;
    for exclude in &args.excludes {
        validate_workspace_glob(exclude)?;
    }

    let root = working_directory.canonicalize().map_err(|error| {
        LocalToolExecutionError::Path(format!(
            "failed to resolve working directory `{}`: {error}",
            working_directory.display()
        ))
    })?;
    let match_options = MatchOptions {
        case_sensitive: true,
        require_literal_separator: true,
        require_literal_leading_dot: false,
    };
    let absolute_pattern = root.join(&args.pattern).to_string_lossy().into_owned();
    let exclude_patterns = args
        .excludes
        .iter()
        .map(|pattern| {
            Pattern::new(&root.join(pattern).to_string_lossy()).map_err(|error| {
                LocalToolExecutionError::InvalidArguments(format!(
                    "invalid exclude glob `{pattern}`: {error}"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut matches = Vec::new();
    let mut truncated = false;
    for entry in glob_with(&absolute_pattern, match_options).map_err(|error| {
        LocalToolExecutionError::InvalidArguments(format!(
            "invalid glob pattern `{}`: {error}",
            args.pattern
        ))
    })? {
        let path = entry.map_err(|error| {
            LocalToolExecutionError::Io(format!(
                "failed to read match for glob `{}`: {error}",
                args.pattern
            ))
        })?;
        if exclude_patterns
            .iter()
            .any(|pattern| pattern.matches_path_with(&path, match_options))
        {
            continue;
        }
        let resolved = path.canonicalize().map_err(|error| {
            LocalToolExecutionError::Path(format!(
                "failed to resolve glob match `{}`: {error}",
                path.display()
            ))
        })?;
        if !resolved.starts_with(&root) {
            continue;
        }
        let mut relative = path
            .strip_prefix(&root)
            .map_err(|error| {
                LocalToolExecutionError::Path(format!(
                    "failed to convert glob match `{}` to a workspace-relative path: {error}",
                    path.display()
                ))
            })?
            .to_string_lossy()
            .replace('\\', "/");
        let metadata = fs::metadata(&path).map_err(|error| {
            LocalToolExecutionError::Io(format!(
                "failed to inspect glob match `{}`: {error}",
                path.display()
            ))
        })?;
        if metadata.is_dir() && !relative.ends_with('/') {
            relative.push('/');
        }
        if matches.len() >= MAX_GLOB_MATCHES {
            truncated = true;
            break;
        }
        matches.push(relative);
    }

    matches.sort();
    matches.dedup();

    let mut summary = format!(
        "matched {} path{} for glob `{}`",
        matches.len(),
        if matches.len() == 1 { "" } else { "s" },
        args.pattern
    );
    if !args.excludes.is_empty() {
        summary.push_str(&format!("\nexcludes: {}", args.excludes.join(", ")));
    }
    if !matches.is_empty() {
        summary.push('\n');
        summary.push_str(&matches.join("\n"));
    }
    if truncated {
        summary.push_str(&format!(
            "\nresults truncated after {} matches",
            MAX_GLOB_MATCHES
        ));
    }

    Ok(ToolCallResultEnvelope {
        status: "ok".to_owned(),
        summary,
        payload: json!({
            "matches": matches,
        }),
        error: None,
    })
}

async fn execute_bash(
    args: &BashArgs,
    working_directory: &Path,
    remaining_turn_budget: Duration,
) -> Result<ToolCallResultEnvelope, LocalToolExecutionError> {
    let root = working_directory.canonicalize().map_err(|error| {
        LocalToolExecutionError::Path(format!(
            "failed to resolve working directory `{}`: {error}",
            working_directory.display()
        ))
    })?;
    let effective_timeout = args
        .timeout_ms
        .map(Duration::from_millis)
        .map(|requested| requested.min(remaining_turn_budget))
        .unwrap_or(remaining_turn_budget);
    if effective_timeout.is_zero() {
        return Err(LocalToolExecutionError::Io(
            "bash command has no remaining execution time".to_owned(),
        ));
    }

    let mut child = Command::new("bash")
        .arg("-lc")
        .arg(&args.command)
        .current_dir(&root)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            LocalToolExecutionError::Io(format!(
                "failed to spawn bash for `{}`: {error}",
                args.command
            ))
        })?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| LocalToolExecutionError::Io("failed to capture bash stdout".to_owned()))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| LocalToolExecutionError::Io("failed to capture bash stderr".to_owned()))?;
    let stdout_task = tokio::spawn(capture_stream(stdout, "stdout"));
    let stderr_task = tokio::spawn(capture_stream(stderr, "stderr"));

    let status = match timeout(effective_timeout, child.wait()).await {
        Ok(wait_result) => wait_result.map_err(|error| {
            LocalToolExecutionError::Io(format!(
                "failed to wait for bash command `{}`: {error}",
                args.command
            ))
        })?,
        Err(_) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            let _ = join_capture(stdout_task, "stdout").await;
            let _ = join_capture(stderr_task, "stderr").await;
            return Err(LocalToolExecutionError::Io(format!(
                "bash command timed out after {} ms",
                effective_timeout.as_millis()
            )));
        }
    };

    let stdout = join_capture(stdout_task, "stdout").await?;
    let stderr = join_capture(stderr_task, "stderr").await?;
    let exit_code = status.code().unwrap_or(-1);

    let mut summary = format!("command: {}\nexit_code: {}", args.command, exit_code);
    if !stdout.text.is_empty() {
        summary.push_str("\nstdout:\n");
        summary.push_str(&stdout.text);
    }
    if !stderr.text.is_empty() {
        summary.push_str("\nstderr:\n");
        summary.push_str(&stderr.text);
    }
    if stdout.truncated || stderr.truncated {
        summary.push_str(&format!(
            "\noutput truncated to {} bytes per stream",
            MAX_SHELL_OUTPUT_BYTES
        ));
    }

    Ok(ToolCallResultEnvelope {
        status: "ok".to_owned(),
        summary,
        payload: json!({
            "command": args.command,
            "exit_code": exit_code,
            "stdout": stdout.text,
            "stderr": stderr.text,
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

fn validate_workspace_glob(raw_pattern: &str) -> Result<(), LocalToolExecutionError> {
    if raw_pattern.trim().is_empty() {
        return Err(LocalToolExecutionError::InvalidArguments(
            "glob pattern cannot be blank".to_owned(),
        ));
    }
    if Path::new(raw_pattern).is_absolute() {
        return Err(LocalToolExecutionError::Path(format!(
            "glob pattern `{raw_pattern}` must be workspace-relative"
        )));
    }
    if raw_pattern
        .split('/')
        .any(|segment| !segment.is_empty() && segment == "..")
    {
        return Err(LocalToolExecutionError::Path(format!(
            "glob pattern `{raw_pattern}` cannot traverse parent directories"
        )));
    }
    Ok(())
}

async fn capture_stream<R>(
    mut reader: R,
    label: &'static str,
) -> Result<CapturedStream, LocalToolExecutionError>
where
    R: AsyncRead + Unpin + Send + 'static,
{
    let mut buffer = [0u8; 4096];
    let mut kept = Vec::new();
    let mut truncated = false;
    loop {
        let read = reader.read(&mut buffer).await.map_err(|error| {
            LocalToolExecutionError::Io(format!("failed to read bash {label}: {error}"))
        })?;
        if read == 0 {
            break;
        }
        if kept.len() < MAX_SHELL_OUTPUT_BYTES {
            let remaining = MAX_SHELL_OUTPUT_BYTES - kept.len();
            let copy_len = remaining.min(read);
            kept.extend_from_slice(&buffer[..copy_len]);
            if copy_len < read {
                truncated = true;
            }
        } else {
            truncated = true;
        }
    }
    Ok(CapturedStream {
        text: String::from_utf8_lossy(&kept).into_owned(),
        truncated,
    })
}

async fn join_capture(
    task: JoinHandle<Result<CapturedStream, LocalToolExecutionError>>,
    label: &'static str,
) -> Result<CapturedStream, LocalToolExecutionError> {
    task.await.map_err(|error| {
        LocalToolExecutionError::Io(format!("bash {label} task failed: {error}"))
    })?
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use serde_json::json;

    use super::{builtin_local_tool_catalog, execute_local_tool};
    use crate::state::LocalToolName;

    #[test]
    fn builtin_catalog_contains_five_local_tools() {
        let tools = builtin_local_tool_catalog();
        assert_eq!(tools.len(), 5);
        assert_eq!(tools[0].tool_id, "local.read_file");
        assert_eq!(tools[1].tool_id, "local.write_file");
        assert_eq!(tools[2].tool_id, "local.edit_file");
        assert_eq!(tools[3].tool_id, "local.glob");
        assert_eq!(tools[4].tool_id, "local.bash");
    }

    #[tokio::test]
    async fn write_file_creates_file_inside_workspace() {
        let temp_dir = temp_dir("tool-write");
        let result = execute_local_tool(
            &LocalToolName::new("write_file").expect("valid tool"),
            &json!({"path":"notes.txt","content":"hello"}),
            &temp_dir,
            Duration::from_secs(5),
        )
        .await
        .expect("write should succeed");
        assert_eq!(result.status, "ok");
        assert_eq!(
            std::fs::read_to_string(temp_dir.join("notes.txt")).expect("file should exist"),
            "hello"
        );
    }

    #[tokio::test]
    async fn read_file_summary_includes_contents() {
        let temp_dir = temp_dir("tool-read");
        std::fs::write(temp_dir.join("notes.txt"), "hello\nworld\n").expect("seed file");
        let result = execute_local_tool(
            &LocalToolName::new("read_file").expect("valid tool"),
            &json!({"path":"notes.txt"}),
            &temp_dir,
            Duration::from_secs(5),
        )
        .await
        .expect("read should succeed");
        assert!(result.summary.contains("hello"));
        assert!(result.summary.contains("world"));
    }

    #[tokio::test]
    async fn edit_file_requires_exact_single_match() {
        let temp_dir = temp_dir("tool-edit");
        std::fs::write(temp_dir.join("notes.txt"), "hello\nhello\n").expect("seed file");
        let error = execute_local_tool(
            &LocalToolName::new("edit_file").expect("valid tool"),
            &json!({"path":"notes.txt","old_text":"hello","new_text":"bye"}),
            &temp_dir,
            Duration::from_secs(5),
        )
        .await
        .expect_err("edit should fail");
        assert!(error.to_string().contains("multiple"));
    }

    #[tokio::test]
    async fn glob_returns_sorted_relative_matches_and_excludes() {
        let temp_dir = temp_dir("tool-glob");
        std::fs::create_dir_all(temp_dir.join("src/nested")).expect("nested dir");
        std::fs::write(temp_dir.join("src/lib.rs"), "").expect("lib file");
        std::fs::write(temp_dir.join("src/nested/mod.rs"), "").expect("mod file");
        std::fs::create_dir_all(temp_dir.join("src/ignore")).expect("ignore dir");
        std::fs::write(temp_dir.join("src/ignore/skip.rs"), "").expect("skip file");

        let result = execute_local_tool(
            &LocalToolName::new("glob").expect("valid tool"),
            &json!({"pattern":"src/**/*.rs","excludes":["src/ignore/**"]}),
            &temp_dir,
            Duration::from_secs(5),
        )
        .await
        .expect("glob should succeed");

        assert_eq!(
            result.payload["matches"],
            json!(["src/lib.rs", "src/nested/mod.rs"])
        );
        assert!(result.summary.contains("src/lib.rs"));
        assert!(!result.summary.contains("src/ignore/skip.rs"));
    }

    #[tokio::test]
    async fn glob_rejects_parent_traversal() {
        let temp_dir = temp_dir("tool-glob-traversal");
        let error = execute_local_tool(
            &LocalToolName::new("glob").expect("valid tool"),
            &json!({"pattern":"../*.rs"}),
            &temp_dir,
            Duration::from_secs(5),
        )
        .await
        .expect_err("glob should reject traversal");
        assert!(
            error
                .to_string()
                .contains("cannot traverse parent directories")
        );
    }

    #[tokio::test]
    async fn bash_returns_exit_code_stdout_and_stderr() {
        let temp_dir = temp_dir("tool-bash");
        let result = execute_local_tool(
            &LocalToolName::new("bash").expect("valid tool"),
            &json!({"command":"printf 'hello'; printf 'oops' >&2; exit 7"}),
            &temp_dir,
            Duration::from_secs(5),
        )
        .await
        .expect("bash should succeed");

        assert_eq!(result.payload["exit_code"], json!(7));
        assert_eq!(result.payload["stdout"], json!("hello"));
        assert_eq!(result.payload["stderr"], json!("oops"));
        assert!(result.summary.contains("exit_code: 7"));
    }

    #[tokio::test]
    async fn bash_honors_timeout() {
        let temp_dir = temp_dir("tool-bash-timeout");
        let error = execute_local_tool(
            &LocalToolName::new("bash").expect("valid tool"),
            &json!({"command":"sleep 1","timeout_ms":10}),
            &temp_dir,
            Duration::from_secs(5),
        )
        .await
        .expect_err("bash should time out");
        assert!(error.to_string().contains("timed out"));
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
