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

use crate::{
    state::{LocalToolName, ServerName},
    todo::{
        MANDATORY_TODO_GENERATE_HTML, MANDATORY_TODO_OPEN_HTML, TodoError, TodoList, TodoStatus,
    },
};

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

#[derive(Debug, Deserialize)]
struct WriteTodosArgs {
    operation: String,
    #[serde(default)]
    items: Option<Vec<String>>,
    #[serde(default)]
    item_index: Option<usize>,
    #[serde(default)]
    status: Option<TodoStatus>,
}

#[derive(Debug)]
struct CapturedStream {
    text: String,
    truncated: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalToolContext {
    pub working_directory: PathBuf,
    pub turn_directory: PathBuf,
    pub todo_path: PathBuf,
    pub html_output_path: PathBuf,
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
        ToolDescriptor::local(
            "write_todos",
            "Write Todos",
            "Create or update the current turn's todo list at `.arka/tmp/<session_id>/<turn_id>/todos.txt`. New todo items must begin with an executor hint: `[mcp-executor]`, `[tool-executor]`, or `[main-agent]`.",
            vec!["file_write"],
            json!({
                "type": "object",
                "required": ["operation"],
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["initialize", "set_status", "replan_pending_suffix"]
                    },
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Todo text prefixed with an executor hint, for example `[mcp-executor] Query the users table`."
                        }
                    },
                    "item_index": { "type": "integer", "minimum": 1 },
                    "status": {
                        "type": "string",
                        "enum": ["in_progress", "completed", "failed"]
                    }
                },
                "additionalProperties": false
            }),
        ),
    ]
}

pub async fn execute_local_tool(
    tool_name: &LocalToolName,
    arguments: &Value,
    context: &LocalToolContext,
    remaining_turn_budget: Duration,
) -> Result<ToolCallResultEnvelope, LocalToolExecutionError> {
    match tool_name.as_str() {
        "read_file" => {
            let args: ReadFileArgs = serde_json::from_value(arguments.clone())
                .map_err(|error| LocalToolExecutionError::InvalidArguments(error.to_string()))?;
            execute_read_file(&args, &context.working_directory)
        }
        "write_file" => {
            let args: WriteFileArgs = serde_json::from_value(arguments.clone())
                .map_err(|error| LocalToolExecutionError::InvalidArguments(error.to_string()))?;
            execute_write_file(&args, context)
        }
        "edit_file" => {
            let args: EditFileArgs = serde_json::from_value(arguments.clone())
                .map_err(|error| LocalToolExecutionError::InvalidArguments(error.to_string()))?;
            execute_edit_file(&args, context)
        }
        "glob" => {
            let args: GlobArgs = serde_json::from_value(arguments.clone())
                .map_err(|error| LocalToolExecutionError::InvalidArguments(error.to_string()))?;
            execute_glob(&args, &context.working_directory)
        }
        "bash" => {
            let args: BashArgs = serde_json::from_value(arguments.clone())
                .map_err(|error| LocalToolExecutionError::InvalidArguments(error.to_string()))?;
            execute_bash(&args, context, remaining_turn_budget).await
        }
        "write_todos" => {
            let args: WriteTodosArgs = serde_json::from_value(arguments.clone())
                .map_err(|error| LocalToolExecutionError::InvalidArguments(error.to_string()))?;
            execute_write_todos(&args, context)
        }
        other => Err(LocalToolExecutionError::UnknownTool(format!(
            "unknown local tool `{other}`"
        ))),
    }
}

fn execute_write_todos(
    args: &WriteTodosArgs,
    context: &LocalToolContext,
) -> Result<ToolCallResultEnvelope, LocalToolExecutionError> {
    match args.operation.as_str() {
        "initialize" => {
            if context.todo_path.exists() {
                return Err(LocalToolExecutionError::Path(format!(
                    "todo file `{}` already exists for this turn",
                    context.todo_path.display()
                )));
            }
            let items = args.items.as_ref().ok_or_else(|| {
                LocalToolExecutionError::InvalidArguments(
                    "write_todos initialize requires `items`".to_owned(),
                )
            })?;
            let todos = TodoList::initialize(items).map_err(map_todo_error)?;
            todos
                .save_to_path(&context.todo_path)
                .map_err(map_todo_error)?;
            Ok(todo_result_envelope(
                "initialized todo list",
                &context.todo_path,
                &todos,
            ))
        }
        "set_status" => {
            let item_index = args.item_index.ok_or_else(|| {
                LocalToolExecutionError::InvalidArguments(
                    "write_todos set_status requires `item_index`".to_owned(),
                )
            })?;
            let status = args.status.ok_or_else(|| {
                LocalToolExecutionError::InvalidArguments(
                    "write_todos set_status requires `status`".to_owned(),
                )
            })?;
            let mut todos = TodoList::load_from_path(&context.todo_path).map_err(map_todo_error)?;
            todos
                .set_status(item_index, status)
                .map_err(map_todo_error)?;
            reject_invalid_mandatory_todo_completion(&todos, item_index, status, context)?;
            todos
                .save_to_path(&context.todo_path)
                .map_err(map_todo_error)?;
            Ok(todo_result_envelope(
                &format!("updated todo item {} to {}", item_index, status),
                &context.todo_path,
                &todos,
            ))
        }
        "replan_pending_suffix" => {
            let items = args.items.as_ref().ok_or_else(|| {
                LocalToolExecutionError::InvalidArguments(
                    "write_todos replan_pending_suffix requires `items`".to_owned(),
                )
            })?;
            let mut todos = TodoList::load_from_path(&context.todo_path).map_err(map_todo_error)?;
            todos.replan_pending_suffix(items).map_err(map_todo_error)?;
            todos
                .save_to_path(&context.todo_path)
                .map_err(map_todo_error)?;
            Ok(todo_result_envelope(
                "replanned pending todo suffix",
                &context.todo_path,
                &todos,
            ))
        }
        other => Err(LocalToolExecutionError::InvalidArguments(format!(
            "unknown write_todos operation `{other}`"
        ))),
    }
}

fn todo_result_envelope(
    action: &str,
    todo_path: &Path,
    todos: &TodoList,
) -> ToolCallResultEnvelope {
    let next_actionable = todos
        .next_actionable()
        .map(|(index, item)| {
            json!({
                "item_index": index,
                "status": item.status.as_str(),
                "executor_hint": item.executor_hint.map(|executor| executor.as_str()),
                "text": item.text,
            })
        })
        .unwrap_or(Value::Null);
    ToolCallResultEnvelope {
        status: "ok".to_owned(),
        summary: format!(
            "{action}\npath: {}\n{}",
            todo_path.display(),
            todos.render()
        ),
        payload: json!({
            "path": todo_path.display().to_string(),
            "todos": todos.items,
            "next_actionable": next_actionable,
        }),
        error: None,
    }
}

fn map_todo_error(error: TodoError) -> LocalToolExecutionError {
    match error {
        TodoError::Parse(message) | TodoError::Validation(message) => {
            LocalToolExecutionError::InvalidArguments(message)
        }
        TodoError::Read { .. } | TodoError::Write { .. } => {
            LocalToolExecutionError::Io(error.to_string())
        }
    }
}

fn reject_invalid_mandatory_todo_completion(
    todos: &TodoList,
    item_index: usize,
    status: TodoStatus,
    context: &LocalToolContext,
) -> Result<(), LocalToolExecutionError> {
    if status != TodoStatus::Completed {
        return Ok(());
    }
    let Some(item) = item_index
        .checked_sub(1)
        .and_then(|zero_based| todos.items.get(zero_based))
    else {
        return Ok(());
    };
    if item.text == MANDATORY_TODO_GENERATE_HTML && !html_report_file_is_acceptable(context) {
        return Err(LocalToolExecutionError::Path(format!(
            "cannot mark the HTML-generation todo completed before a non-placeholder deterministic HTML report exists at `{}`",
            context.html_output_path.display()
        )));
    }
    if item.text == MANDATORY_TODO_OPEN_HTML && !html_report_file_is_acceptable(context) {
        return Err(LocalToolExecutionError::Path(format!(
            "cannot mark the html-path-print todo completed because the deterministic HTML report is missing or still a placeholder at `{}`",
            context.html_output_path.display()
        )));
    }
    Ok(())
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
    context: &LocalToolContext,
) -> Result<ToolCallResultEnvelope, LocalToolExecutionError> {
    let path = resolve_tool_path(&context.working_directory, &args.path, false)?;
    reject_protected_todo_mutation(&path, context)?;
    reject_out_of_order_html_write(&path, context)?;
    if tool_paths_equivalent(&path, &context.html_output_path)?
        && !html_report_content_is_acceptable(&args.content)
    {
        return Err(LocalToolExecutionError::Path(format!(
            "the deterministic HTML report `{}` cannot be a placeholder; include the grounded report content before completing the HTML-generation todo",
            context.html_output_path.display()
        )));
    }
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
    context: &LocalToolContext,
) -> Result<ToolCallResultEnvelope, LocalToolExecutionError> {
    let path = resolve_tool_path(&context.working_directory, &args.path, true)?;
    reject_protected_todo_mutation(&path, context)?;
    reject_out_of_order_html_write(&path, context)?;
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
    context: &LocalToolContext,
    remaining_turn_budget: Duration,
) -> Result<ToolCallResultEnvelope, LocalToolExecutionError> {
    if bash_command_mentions_protected_todo(&args.command, context) {
        return Err(LocalToolExecutionError::Path(format!(
            "bash cannot read or modify the current turn todo file `{}`; use write_todos for mutations and read_file for inspection",
            context.todo_path.display()
        )));
    }
    reject_out_of_order_html_bash(&args.command, context)?;
    let root = context.working_directory.canonicalize().map_err(|error| {
        LocalToolExecutionError::Path(format!(
            "failed to resolve working directory `{}`: {error}",
            context.working_directory.display()
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
    if status.success()
        && bash_command_mentions_html_path(&args.command, context)
        && context.html_output_path.exists()
        && !html_report_file_is_acceptable(context)
    {
        return Err(LocalToolExecutionError::Path(format!(
            "the deterministic HTML report `{}` cannot be a placeholder; include the grounded report content before completing the HTML-generation todo",
            context.html_output_path.display()
        )));
    }

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

    let status_text = if status.success() { "ok" } else { "error" };
    let error = if status.success() {
        None
    } else {
        Some(format!("bash command exited with status {}", exit_code))
    };

    Ok(ToolCallResultEnvelope {
        status: status_text.to_owned(),
        summary,
        payload: json!({
            "command": args.command,
            "exit_code": exit_code,
            "stdout": stdout.text,
            "stderr": stderr.text,
        }),
        error,
    })
}

fn reject_protected_todo_mutation(
    path: &Path,
    context: &LocalToolContext,
) -> Result<(), LocalToolExecutionError> {
    if tool_paths_equivalent(path, &context.todo_path)? {
        return Err(LocalToolExecutionError::Path(format!(
            "the current turn todo file `{}` may only be modified through write_todos",
            context.todo_path.display()
        )));
    }
    Ok(())
}

fn bash_command_mentions_protected_todo(command: &str, context: &LocalToolContext) -> bool {
    let absolute = context.todo_path.display().to_string();
    let relative_to_working_directory = context
        .todo_path
        .strip_prefix(&context.working_directory)
        .ok()
        .map(|path| path.display().to_string());

    command.contains(&absolute)
        || relative_to_working_directory
            .as_ref()
            .is_some_and(|path| command.contains(path) || command.contains(&format!("./{path}")))
}

fn bash_command_mentions_html_path(command: &str, context: &LocalToolContext) -> bool {
    let html_absolute = context.html_output_path.display().to_string();
    let html_relative = context
        .html_output_path
        .strip_prefix(&context.working_directory)
        .ok()
        .map(|path| path.display().to_string());
    command.contains(&html_absolute)
        || html_relative
            .as_ref()
            .is_some_and(|path| command.contains(path) || command.contains(&format!("./{path}")))
}

fn reject_out_of_order_html_write(
    path: &Path,
    context: &LocalToolContext,
) -> Result<(), LocalToolExecutionError> {
    if !tool_paths_equivalent(path, &context.html_output_path)? {
        return Ok(());
    }
    let Some(todo_text) = load_next_actionable_todo_text(context)? else {
        return Ok(());
    };
    if todo_text != MANDATORY_TODO_GENERATE_HTML {
        return Err(LocalToolExecutionError::Path(format!(
            "the deterministic HTML report `{}` cannot be written while the current actionable todo is `{}`; finish todos in order",
            context.html_output_path.display(),
            todo_text
        )));
    }
    Ok(())
}

fn tool_paths_equivalent(left: &Path, right: &Path) -> Result<bool, LocalToolExecutionError> {
    let left = canonicalize_for_tool_guard(left)?;
    let right = canonicalize_for_tool_guard(right)?;
    Ok(left == right)
}

fn canonicalize_for_tool_guard(path: &Path) -> Result<PathBuf, LocalToolExecutionError> {
    let normalized = normalize_path(path);
    if normalized.exists() {
        return normalized.canonicalize().map_err(|error| {
            LocalToolExecutionError::Path(format!(
                "failed to resolve path `{}`: {error}",
                normalized.display()
            ))
        });
    }
    let parent = normalized.parent().ok_or_else(|| {
        LocalToolExecutionError::Path(format!(
            "path `{}` does not have a parent directory",
            normalized.display()
        ))
    })?;
    let resolved_parent = match parent.canonicalize() {
        Ok(path) => path,
        Err(_) => return Ok(normalized),
    };
    let file_name = normalized.file_name().ok_or_else(|| {
        LocalToolExecutionError::Path(format!(
            "path `{}` must point to a file",
            normalized.display()
        ))
    })?;
    Ok(resolved_parent.join(file_name))
}

fn reject_out_of_order_html_bash(
    command: &str,
    context: &LocalToolContext,
) -> Result<(), LocalToolExecutionError> {
    let Some(todo_text) = load_next_actionable_todo_text(context)? else {
        return Ok(());
    };
    let html_absolute = context.html_output_path.display().to_string();
    let html_relative = context
        .html_output_path
        .strip_prefix(&context.working_directory)
        .ok()
        .map(|path| path.display().to_string());
    if !bash_command_mentions_html_path(command, context) {
        return Ok(());
    }

    let opens_html = bash_command_opens_path(command, &html_absolute)
        || html_relative.as_ref().is_some_and(|path| {
            bash_command_opens_path(command, path)
                || bash_command_opens_path(command, &format!("./{path}"))
        });

    if opens_html {
        return Err(LocalToolExecutionError::Path(format!(
            "the deterministic HTML report `{}` should not be opened in the browser; print the generated HTML file path instead",
            context.html_output_path.display()
        )));
    } else if todo_text == MANDATORY_TODO_OPEN_HTML {
        if !html_report_file_is_acceptable(context) {
            return Err(LocalToolExecutionError::Path(format!(
                "the deterministic HTML report `{}` cannot have its path printed because it is missing or still a placeholder; finish the HTML-generation todo first",
                context.html_output_path.display()
            )));
        }
        let prints_html_path = bash_command_prints_path(command, &html_absolute)
            || html_relative.as_ref().is_some_and(|path| {
                bash_command_prints_path(command, path)
                    || bash_command_prints_path(command, &format!("./{path}"))
            });
        if prints_html_path {
            return Ok(());
        }
        return Err(LocalToolExecutionError::Path(format!(
            "the deterministic HTML report `{}` must be handled by printing its path while the current actionable todo is `{}`",
            context.html_output_path.display(),
            todo_text
        )));
    } else if todo_text != MANDATORY_TODO_GENERATE_HTML {
        return Err(LocalToolExecutionError::Path(format!(
            "the deterministic HTML report `{}` cannot be written from bash while the current actionable todo is `{}`; finish todos in order",
            context.html_output_path.display(),
            todo_text
        )));
    }
    Ok(())
}

fn bash_command_opens_path(command: &str, path: &str) -> bool {
    [
        format!("open {path}"),
        format!("open '{path}'"),
        format!("open \"{path}\""),
    ]
    .iter()
    .any(|pattern| command.contains(pattern))
}

fn bash_command_prints_path(command: &str, path: &str) -> bool {
    [
        format!("echo {path}"),
        format!("echo '{path}'"),
        format!("echo \"{path}\""),
        format!("printf '%s\\n' {path}"),
        format!("printf '%s\\n' '{path}'"),
        format!("printf '%s\\n' \"{path}\""),
        format!("printf \"%s\\n\" {path}"),
        format!("printf \"%s\\n\" '{path}'"),
        format!("printf \"%s\\n\" \"{path}\""),
        format!("printf '{path}\\n'"),
        format!("printf \"{path}\\n\""),
    ]
    .iter()
    .any(|pattern| command.contains(pattern))
}

fn html_report_file_is_acceptable(context: &LocalToolContext) -> bool {
    fs::read_to_string(&context.html_output_path)
        .map(|content| html_report_content_is_acceptable(&content))
        .unwrap_or(false)
}

fn html_report_content_is_acceptable(content: &str) -> bool {
    let normalized = content.to_ascii_lowercase();
    normalized.contains("<html")
        && normalized.contains("</html")
        && !normalized.contains("placeholder")
}

fn load_next_actionable_todo_text(
    context: &LocalToolContext,
) -> Result<Option<String>, LocalToolExecutionError> {
    if !context.todo_path.exists() {
        return Ok(None);
    }
    let todos = TodoList::load_from_path(&context.todo_path).map_err(map_todo_error)?;
    Ok(todos.next_actionable().map(|(_, item)| item.text.clone()))
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

    use super::{LocalToolContext, builtin_local_tool_catalog, execute_local_tool};
    use crate::state::LocalToolName;
    use crate::todo::{MANDATORY_TODO_GENERATE_HTML, MANDATORY_TODO_OPEN_HTML};

    #[test]
    fn builtin_catalog_contains_six_local_tools() {
        let tools = builtin_local_tool_catalog();
        assert_eq!(tools.len(), 6);
        assert_eq!(tools[0].tool_id, "local.read_file");
        assert_eq!(tools[1].tool_id, "local.write_file");
        assert_eq!(tools[2].tool_id, "local.edit_file");
        assert_eq!(tools[3].tool_id, "local.glob");
        assert_eq!(tools[4].tool_id, "local.bash");
        assert_eq!(tools[5].tool_id, "local.write_todos");
    }

    #[tokio::test]
    async fn write_file_creates_file_inside_workspace() {
        let temp_dir = temp_dir("tool-write");
        let result = execute_local_tool(
            &LocalToolName::new("write_file").expect("valid tool"),
            &json!({"path":"notes.txt","content":"hello"}),
            &local_tool_context(&temp_dir),
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
            &local_tool_context(&temp_dir),
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
            &local_tool_context(&temp_dir),
            Duration::from_secs(5),
        )
        .await
        .expect_err("edit should fail");
        assert!(error.to_string().contains("multiple"));
    }

    #[tokio::test]
    async fn write_file_cannot_mutate_current_turn_todo_file() {
        let temp_dir = temp_dir("tool-write-protected-todo");
        let context = local_tool_context(&temp_dir);
        std::fs::write(&context.todo_path, "1. [pending] Task").expect("seed todo file");
        let error = execute_local_tool(
            &LocalToolName::new("write_file").expect("valid tool"),
            &json!({"path":"turn-123/todos.txt","content":"overwritten"}),
            &context,
            Duration::from_secs(5),
        )
        .await
        .expect_err("write_file should not mutate todo file");
        assert!(
            error
                .to_string()
                .contains("may only be modified through write_todos")
        );
    }

    #[tokio::test]
    async fn edit_file_cannot_mutate_current_turn_todo_file() {
        let temp_dir = temp_dir("tool-edit-protected-todo");
        let context = local_tool_context(&temp_dir);
        std::fs::write(&context.todo_path, "1. [pending] Task").expect("seed todo file");
        let error = execute_local_tool(
            &LocalToolName::new("edit_file").expect("valid tool"),
            &json!({"path":"turn-123/todos.txt","old_text":"Task","new_text":"Changed"}),
            &context,
            Duration::from_secs(5),
        )
        .await
        .expect_err("edit_file should not mutate todo file");
        assert!(
            error
                .to_string()
                .contains("may only be modified through write_todos")
        );
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
            &local_tool_context(&temp_dir),
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
            &local_tool_context(&temp_dir),
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
            &local_tool_context(&temp_dir),
            Duration::from_secs(5),
        )
        .await
        .expect("bash should succeed");

        assert_eq!(result.status, "error");
        assert_eq!(result.payload["exit_code"], json!(7));
        assert_eq!(result.payload["stdout"], json!("hello"));
        assert_eq!(result.payload["stderr"], json!("oops"));
        assert_eq!(
            result.error,
            Some("bash command exited with status 7".to_owned())
        );
        assert!(result.summary.contains("exit_code: 7"));
    }

    #[tokio::test]
    async fn bash_honors_timeout() {
        let temp_dir = temp_dir("tool-bash-timeout");
        let error = execute_local_tool(
            &LocalToolName::new("bash").expect("valid tool"),
            &json!({"command":"sleep 1","timeout_ms":10}),
            &local_tool_context(&temp_dir),
            Duration::from_secs(5),
        )
        .await
        .expect_err("bash should time out");
        assert!(error.to_string().contains("timed out"));
    }

    #[tokio::test]
    async fn bash_cannot_touch_current_turn_todo_file() {
        let temp_dir = temp_dir("tool-bash-protected-todo");
        let context = local_tool_context(&temp_dir);
        std::fs::write(&context.todo_path, "1. [pending] Task").expect("seed todo file");
        let error = execute_local_tool(
            &LocalToolName::new("bash").expect("valid tool"),
            &json!({"command":"python3 - <<'PY'\nfrom pathlib import Path\np = Path('turn-123/todos.txt')\np.write_text('mutated')\nPY"}),
            &context,
            Duration::from_secs(5),
        )
        .await
        .expect_err("bash should not touch todo file");
        assert!(
            error
                .to_string()
                .contains("bash cannot read or modify the current turn todo file")
        );
    }

    #[tokio::test]
    async fn write_file_cannot_write_deterministic_html_before_html_todo_is_current() {
        let temp_dir = temp_dir("tool-write-html-out-of-order");
        let context = local_tool_context(&temp_dir);
        std::fs::create_dir_all(temp_dir.join("outputs")).expect("outputs dir");
        std::fs::write(
            &context.todo_path,
            "1. [in_progress] Compute the ranked batter impact metric.\n2. [pending] Generate a clear, engaging data blog post with narrative writing, charts, and tables, using deep analysis in Python with pandas and numpy to surface insights.\n3. [pending] Print the path of the generated HTML file.\n",
        )
        .expect("seed todo file");

        let error = execute_local_tool(
            &LocalToolName::new("write_file").expect("valid tool"),
            &json!({"path":"outputs/turn-123-report.html","content":"<html></html>"}),
            &context,
            Duration::from_secs(5),
        )
        .await
        .expect_err("write_file should reject out-of-order html write");
        assert!(
            error
                .to_string()
                .contains("cannot be written while the current actionable todo is")
        );
    }

    #[tokio::test]
    async fn bash_cannot_write_deterministic_html_before_html_todo_is_current() {
        let temp_dir = temp_dir("tool-bash-html-out-of-order");
        let context = local_tool_context(&temp_dir);
        std::fs::create_dir_all(temp_dir.join("outputs")).expect("outputs dir");
        std::fs::write(
            &context.todo_path,
            "1. [in_progress] Compute the ranked batter impact metric.\n2. [pending] Generate a clear, engaging data blog post with narrative writing, charts, and tables, using deep analysis in Python with pandas and numpy to surface insights.\n3. [pending] Print the path of the generated HTML file.\n",
        )
        .expect("seed todo file");

        let error = execute_local_tool(
            &LocalToolName::new("bash").expect("valid tool"),
            &json!({"command":"printf '<html></html>' > outputs/turn-123-report.html"}),
            &context,
            Duration::from_secs(5),
        )
        .await
        .expect_err("bash should reject out-of-order html write");
        assert!(
            error
                .to_string()
                .contains("cannot be written from bash while the current actionable todo is")
        );
    }

    #[tokio::test]
    async fn bash_cannot_open_deterministic_html_before_open_todo_is_current() {
        let temp_dir = temp_dir("tool-bash-open-html-out-of-order");
        let context = local_tool_context(&temp_dir);
        std::fs::create_dir_all(temp_dir.join("outputs")).expect("outputs dir");
        std::fs::write(
            &context.todo_path,
            "1. [pending] Generate a clear, engaging data blog post with narrative writing, charts, and tables, using deep analysis in Python with pandas and numpy to surface insights.\n2. [pending] Print the path of the generated HTML file.\n",
        )
        .expect("seed todo file");

        let error = execute_local_tool(
            &LocalToolName::new("bash").expect("valid tool"),
            &json!({"command":"open outputs/turn-123-report.html"}),
            &context,
            Duration::from_secs(5),
        )
        .await
        .expect_err("bash should reject out-of-order html open");
        assert!(
            error
                .to_string()
                .contains("should not be opened in the browser")
        );
    }

    #[tokio::test]
    async fn bash_treats_quoted_open_as_open_not_write() {
        let temp_dir = temp_dir("tool-bash-quoted-open-html-out-of-order");
        let context = local_tool_context(&temp_dir);
        std::fs::create_dir_all(temp_dir.join("outputs")).expect("outputs dir");
        std::fs::write(
            &context.todo_path,
            "1. [pending] Generate a clear, engaging data blog post with narrative writing, charts, and tables, using deep analysis in Python with pandas and numpy to surface insights.\n2. [pending] Print the path of the generated HTML file.\n",
        )
        .expect("seed todo file");

        let error = execute_local_tool(
            &LocalToolName::new("bash").expect("valid tool"),
            &json!({"command":"open 'outputs/turn-123-report.html'"}),
            &context,
            Duration::from_secs(5),
        )
        .await
        .expect_err("quoted bash open should still be recognized as open");
        let message = error.to_string();
        assert!(message.contains("should not be opened in the browser"));
        assert!(!message.contains("cannot be written from bash"));
    }

    #[tokio::test]
    async fn write_todos_initializes_turn_file_and_appends_mandatory_steps() {
        let temp_dir = temp_dir("tool-write-todos-init");
        let context = local_tool_context(&temp_dir);
        let result = execute_local_tool(
            &LocalToolName::new("write_todos").expect("valid tool"),
            &json!({"operation":"initialize","items":["[mcp-executor] Inspect source data","[main-agent] Compute metrics"]}),
            &context,
            Duration::from_secs(5),
        )
        .await
        .expect("write_todos initialize should succeed");

        let raw = std::fs::read_to_string(&context.todo_path).expect("todo file should exist");
        assert!(raw.contains("[pending] [mcp-executor] Inspect source data"));
        assert!(raw.contains(
            "[pending] [tool-executor] Generate a clear, engaging data blog post with narrative writing, charts, and tables, using deep analysis in Python with pandas and numpy to surface insights."
        ));
        assert_eq!(
            result.payload["next_actionable"]["executor_hint"].as_str(),
            Some("mcp-executor")
        );
        assert!(result.summary.contains("initialized todo list"));
    }

    #[tokio::test]
    async fn write_todos_enforces_ordered_status_updates() {
        let temp_dir = temp_dir("tool-write-todos-status");
        let context = local_tool_context(&temp_dir);
        execute_local_tool(
            &LocalToolName::new("write_todos").expect("valid tool"),
            &json!({"operation":"initialize","items":["[mcp-executor] Inspect source data"]}),
            &context,
            Duration::from_secs(5),
        )
        .await
        .expect("initialize should succeed");

        execute_local_tool(
            &LocalToolName::new("write_todos").expect("valid tool"),
            &json!({"operation":"set_status","item_index":1,"status":"in_progress"}),
            &context,
            Duration::from_secs(5),
        )
        .await
        .expect("pending -> in_progress should succeed");

        let error = execute_local_tool(
            &LocalToolName::new("write_todos").expect("valid tool"),
            &json!({"operation":"set_status","item_index":2,"status":"completed"}),
            &context,
            Duration::from_secs(5),
        )
        .await
        .expect_err("cannot complete a later todo");
        assert!(error.to_string().contains("before item 1"));

        let result = execute_local_tool(
            &LocalToolName::new("write_todos").expect("valid tool"),
            &json!({"operation":"set_status","item_index":1,"status":"failed"}),
            &context,
            Duration::from_secs(5),
        )
        .await
        .expect("in_progress -> failed should succeed");
        assert!(
            result
                .summary
                .contains("[failed] [mcp-executor] Inspect source data")
        );
    }

    #[tokio::test]
    async fn write_todos_cannot_complete_html_generation_without_report() {
        let temp_dir = temp_dir("tool-write-todos-html-report-required");
        let context = local_tool_context(&temp_dir);
        std::fs::write(
            &context.todo_path,
            format!("1. [in_progress] {MANDATORY_TODO_GENERATE_HTML}"),
        )
        .expect("seed todo file");

        let error = execute_local_tool(
            &LocalToolName::new("write_todos").expect("valid tool"),
            &json!({"operation":"set_status","item_index":1,"status":"completed"}),
            &context,
            Duration::from_secs(5),
        )
        .await
        .expect_err("html todo completion requires a report file");
        assert!(
            error
                .to_string()
                .contains("before a non-placeholder deterministic HTML report exists")
        );
    }

    #[tokio::test]
    async fn write_file_rejects_placeholder_html_report() {
        let temp_dir = temp_dir("tool-write-file-placeholder-html");
        let context = local_tool_context(&temp_dir);
        std::fs::create_dir_all(
            context
                .html_output_path
                .parent()
                .expect("html output parent should exist"),
        )
        .expect("output dir should exist");
        std::fs::write(
            &context.todo_path,
            format!("1. [in_progress] {MANDATORY_TODO_GENERATE_HTML}"),
        )
        .expect("seed todo file");

        let error = execute_local_tool(
            &LocalToolName::new("write_file").expect("valid tool"),
            &json!({
                "path": "outputs/turn-123-report.html",
                "content": "<html><body><p>Report placeholder.</p></body></html>"
            }),
            &context,
            Duration::from_secs(5),
        )
        .await
        .expect_err("placeholder report should be rejected");
        assert!(error.to_string().contains("cannot be a placeholder"));
    }

    #[tokio::test]
    async fn bash_cannot_create_missing_report_during_open_todo() {
        let temp_dir = temp_dir("tool-bash-open-cannot-create-html");
        let context = local_tool_context(&temp_dir);
        std::fs::create_dir_all(
            context
                .html_output_path
                .parent()
                .expect("html output parent should exist"),
        )
        .expect("output dir should exist");
        std::fs::write(
            &context.todo_path,
            format!(
                "1. [completed] {MANDATORY_TODO_GENERATE_HTML}\n2. [in_progress] {MANDATORY_TODO_OPEN_HTML}"
            ),
        )
        .expect("seed todo file");

        let command = format!("printf '%s\\n' {}", context.html_output_path.display());
        let error = execute_local_tool(
            &LocalToolName::new("bash").expect("valid tool"),
            &json!({"command": command}),
            &context,
            Duration::from_secs(5),
        )
        .await
        .expect_err("path-print step cannot proceed with a missing html report");
        assert!(
            error.to_string().contains(
                "cannot have its path printed because it is missing or still a placeholder"
            )
        );
    }

    #[tokio::test]
    async fn write_todos_replans_only_pending_suffix() {
        let temp_dir = temp_dir("tool-write-todos-replan");
        let context = local_tool_context(&temp_dir);
        execute_local_tool(
            &LocalToolName::new("write_todos").expect("valid tool"),
            &json!({"operation":"initialize","items":["[mcp-executor] Inspect source data","[main-agent] Analyze outliers"]}),
            &context,
            Duration::from_secs(5),
        )
        .await
        .expect("initialize should succeed");
        execute_local_tool(
            &LocalToolName::new("write_todos").expect("valid tool"),
            &json!({"operation":"set_status","item_index":1,"status":"in_progress"}),
            &context,
            Duration::from_secs(5),
        )
        .await
        .expect("item 1 should enter progress");
        execute_local_tool(
            &LocalToolName::new("write_todos").expect("valid tool"),
            &json!({"operation":"set_status","item_index":1,"status":"completed"}),
            &context,
            Duration::from_secs(5),
        )
        .await
        .expect("item 1 should complete");
        execute_local_tool(
            &LocalToolName::new("write_todos").expect("valid tool"),
            &json!({"operation":"set_status","item_index":2,"status":"in_progress"}),
            &context,
            Duration::from_secs(5),
        )
        .await
        .expect("item 2 should enter progress");
        execute_local_tool(
            &LocalToolName::new("write_todos").expect("valid tool"),
            &json!({"operation":"set_status","item_index":2,"status":"failed"}),
            &context,
            Duration::from_secs(5),
        )
        .await
        .expect("item 2 should fail before replanning");

        let result = execute_local_tool(
            &LocalToolName::new("write_todos").expect("valid tool"),
            &json!({"operation":"replan_pending_suffix","items":["[main-agent] Analyze cohorts instead"]}),
            &context,
            Duration::from_secs(5),
        )
        .await
        .expect("replan should succeed");

        let raw = std::fs::read_to_string(&context.todo_path).expect("todo file should exist");
        assert!(raw.contains("[completed] [mcp-executor] Inspect source data"));
        assert!(raw.contains("[pending] [main-agent] Analyze cohorts instead"));
        assert!(
            raw.contains("[pending] [tool-executor] Print the path of the generated HTML file.")
        );
        assert!(result.summary.contains("replanned pending todo suffix"));
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

    fn local_tool_context(working_directory: &PathBuf) -> LocalToolContext {
        let turn_directory = working_directory.join("turn-123");
        std::fs::create_dir_all(&turn_directory).expect("turn directory should exist");
        LocalToolContext {
            working_directory: working_directory.clone(),
            turn_directory: turn_directory.clone(),
            todo_path: turn_directory.join("todos.txt"),
            html_output_path: working_directory
                .join("outputs")
                .join("turn-123-report.html"),
        }
    }
}
