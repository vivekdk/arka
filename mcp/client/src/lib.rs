//! MCP transport client shared by the runtime and inspection CLI.
//!
//! This crate owns:
//! - transport setup for stdio and Streamable HTTP servers
//! - JSON-RPC request/response correlation
//! - MCP handshake and tool discovery helpers

mod codec;

use std::{
    collections::HashMap,
    error::Error,
    fmt, io,
    process::Stdio,
    sync::{
        Arc, Mutex as StdMutex,
        atomic::{AtomicU64, Ordering},
    },
};

use mcp_config::{McpServerConfig, McpTransportConfig};
use reqwest::{
    Client as HttpClient, StatusCode,
    header::{ACCEPT, CONTENT_TYPE, HeaderName, HeaderValue},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncWriteExt, BufReader},
    process::{Child, ChildStdout, Command},
    sync::{Mutex, oneshot},
    task::JoinHandle,
    time::{Duration, sleep},
};

use crate::codec::{encode_newline_message, read_message};

/// Latest protocol version advertised during `initialize`.
pub const SUPPORTED_PROTOCOL_VERSION: &str = "2025-11-25";

/// Protocol revisions this client can negotiate successfully today.
///
/// MCP version negotiation allows the server to return a different supported
/// revision than the one requested during `initialize`.
pub const SUPPORTED_PROTOCOL_VERSIONS: &[&str] =
    &["2025-03-26", "2025-06-18", SUPPORTED_PROTOCOL_VERSION];

/// Factory for establishing MCP connections.
///
/// The type has no state of its own; it exists to make connection creation
/// explicit and keep the live connection object focused on runtime behavior.
pub struct McpClient;

impl McpClient {
    /// Connects to a configured MCP server using its resolved transport.
    pub async fn connect(config: &McpServerConfig) -> Result<McpConnection, McpClientError> {
        match config.resolved_transport() {
            McpTransportConfig::Stdio { command, args, env } => {
                Self::connect_stdio_process(command, args, env).await
            }
            McpTransportConfig::StreamableHttp { url, headers } => {
                Self::connect_streamable_http(url, headers).await
            }
        }
    }

    /// Spawns a local MCP server and wires its stdio to a live connection.
    ///
    /// `stderr` is inherited so server-side startup failures are still visible
    /// during development and CLI-based inspection.
    pub async fn connect_stdio(config: &McpServerConfig) -> Result<McpConnection, McpClientError> {
        match config.resolved_transport() {
            McpTransportConfig::Stdio { command, args, env } => {
                Self::connect_stdio_process(command, args, env).await
            }
            McpTransportConfig::StreamableHttp { .. } => Err(McpClientError::Transport(
                "expected stdio MCP transport".to_owned(),
            )),
        }
    }

    async fn connect_stdio_process(
        command: String,
        args: Vec<String>,
        env: HashMap<String, String>,
    ) -> Result<McpConnection, McpClientError> {
        let mut child = Command::new(&command);
        child
            .args(&args)
            .envs(&env)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());

        let mut child = child.spawn().map_err(McpClientError::Spawn)?;
        let stdin = child
            .stdin
            .take()
            .ok_or(McpClientError::MissingPipe("stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or(McpClientError::MissingPipe("stdout"))?;

        let pending = Arc::new(Mutex::new(HashMap::new()));
        let reader_task = spawn_reader(stdout, Arc::clone(&pending));

        Ok(McpConnection {
            transport: ConnectionTransport::Stdio(StdioTransport {
                child: StdMutex::new(child),
                stdin: Mutex::new(stdin),
                pending,
                reader_task,
            }),
            next_id: AtomicU64::new(1),
        })
    }

    async fn connect_streamable_http(
        url: String,
        headers: HashMap<String, String>,
    ) -> Result<McpConnection, McpClientError> {
        let client = HttpClient::builder()
            .build()
            .map_err(McpClientError::Http)?;
        Ok(McpConnection {
            transport: ConnectionTransport::StreamableHttp(StreamableHttpTransport {
                client,
                url,
                headers,
                session_id: Mutex::new(None),
                protocol_version: Mutex::new(None),
            }),
            next_id: AtomicU64::new(1),
        })
    }
}

/// Live connection to one MCP server.
///
/// The request/response API is transport-neutral; the underlying transport
/// handles only message delivery and raw JSON-RPC response extraction.
#[derive(Debug)]
pub struct McpConnection {
    transport: ConnectionTransport,
    next_id: AtomicU64,
}

#[derive(Debug)]
enum ConnectionTransport {
    Stdio(StdioTransport),
    StreamableHttp(StreamableHttpTransport),
}

#[derive(Debug)]
struct StdioTransport {
    child: StdMutex<Child>,
    stdin: Mutex<tokio::process::ChildStdin>,
    pending: Arc<Mutex<HashMap<u64, PendingSender>>>,
    reader_task: JoinHandle<()>,
}

#[derive(Debug)]
struct StreamableHttpTransport {
    client: HttpClient,
    url: String,
    headers: HashMap<String, String>,
    session_id: Mutex<Option<String>>,
    protocol_version: Mutex<Option<String>>,
}

impl McpConnection {
    /// Performs the initial MCP handshake.
    ///
    /// The client sends its identity and latest supported protocol version, then
    /// validates that the server-selected version is one this client supports.
    pub async fn initialize(
        &self,
        client_info: ClientInfo,
    ) -> Result<McpInitializeResult, McpClientError> {
        let params = json!({
            "protocolVersion": SUPPORTED_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": client_info,
        });

        let response: McpInitializeResult = self.request("initialize", Some(params)).await?;
        if !supports_protocol_version(&response.protocol_version) {
            return Err(McpClientError::UnsupportedProtocolVersion(
                response.protocol_version,
            ));
        }

        if let ConnectionTransport::StreamableHttp(transport) = &self.transport {
            transport
                .set_protocol_version(response.protocol_version.clone())
                .await;
        }

        Ok(response)
    }

    pub async fn notify_initialized(&self) -> Result<(), McpClientError> {
        self.send_notification("notifications/initialized", None)
            .await
    }

    /// Fetches the server's currently advertised tool catalog.
    pub async fn list_tools(&self) -> Result<McpToolsListResult, McpClientError> {
        self.request("tools/list", Some(json!({}))).await
    }

    /// Fetches the server's currently advertised resource catalog.
    pub async fn list_resources(&self) -> Result<McpResourcesListResult, McpClientError> {
        self.request("resources/list", Some(json!({}))).await
    }

    /// Executes one MCP tool on the connected server.
    ///
    /// Tool calls remain strongly identified by name even though the MCP wire
    /// format uses raw JSON payloads at the protocol boundary.
    pub async fn call_tool(
        &self,
        tool_name: &McpToolName,
        arguments: Value,
    ) -> Result<McpToolCallResult, McpClientError> {
        self.request(
            "tools/call",
            Some(json!({
                "name": tool_name.as_str(),
                "arguments": arguments,
            })),
        )
        .await
    }

    /// Reads one MCP resource by URI.
    pub async fn read_resource(
        &self,
        resource_uri: &str,
    ) -> Result<McpResourceReadResult, McpClientError> {
        self.request(
            "resources/read",
            Some(json!({
                "uri": resource_uri,
            })),
        )
        .await
    }

    /// Sends a JSON-RPC request and waits for the matching response.
    async fn request<T>(&self, method: &str, params: Option<Value>) -> Result<T, McpClientError>
    where
        T: DeserializeOwned,
    {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let request = JsonRpcRequest {
            jsonrpc: "2.0",
            id,
            method: method.to_owned(),
            params,
        };
        let body = serde_json::to_value(request).map_err(McpClientError::Json)?;

        let response = match &self.transport {
            ConnectionTransport::Stdio(transport) => transport.send_request(id, &body).await?,
            ConnectionTransport::StreamableHttp(transport) => {
                transport.send_request(id, method, &body).await?
            }
        };

        serde_json::from_value(response).map_err(McpClientError::Json)
    }

    /// Sends a JSON-RPC notification, which has no response path.
    async fn send_notification(
        &self,
        method: &str,
        params: Option<Value>,
    ) -> Result<(), McpClientError> {
        let notification = JsonRpcNotification {
            jsonrpc: "2.0",
            method: method.to_owned(),
            params,
        };
        let body = serde_json::to_value(notification).map_err(McpClientError::Json)?;

        match &self.transport {
            ConnectionTransport::Stdio(transport) => transport.send_notification(&body).await,
            ConnectionTransport::StreamableHttp(transport) => {
                transport.send_notification(method, &body).await
            }
        }
    }

    /// Performs best-effort transport shutdown for temporary connections.
    ///
    /// For stdio transports this follows the MCP shutdown guidance more closely:
    /// close stdin first, allow the child a brief chance to exit, then kill only
    /// if the process is still running.
    pub async fn close(&mut self) -> Result<(), McpClientError> {
        match &mut self.transport {
            ConnectionTransport::Stdio(transport) => transport.close().await,
            ConnectionTransport::StreamableHttp(_) => Ok(()),
        }
    }
}

impl Drop for McpConnection {
    fn drop(&mut self) {
        if let ConnectionTransport::Stdio(transport) = &mut self.transport {
            transport.reader_task.abort();
            if let Ok(mut child) = transport.child.lock() {
                let _ = child.start_kill();
            }
        }
    }
}

impl StdioTransport {
    async fn close(&mut self) -> Result<(), McpClientError> {
        {
            let mut stdin = self.stdin.lock().await;
            let _ = stdin.shutdown().await;
        }

        for _ in 0..20 {
            let exited = if let Ok(mut child) = self.child.lock() {
                child.try_wait().map_err(McpClientError::Io)?.is_some()
            } else {
                false
            };
            if exited {
                self.reader_task.abort();
                return Ok(());
            }
            sleep(Duration::from_millis(10)).await;
        }

        self.reader_task.abort();
        if let Ok(mut child) = self.child.lock() {
            let _ = child.start_kill();
        }
        Ok(())
    }

    async fn send_request(&self, id: u64, body: &Value) -> Result<Value, McpClientError> {
        let message = encode_newline_message(body).map_err(McpClientError::Json)?;
        let (tx, rx) = oneshot::channel();

        self.pending.lock().await.insert(id, tx);

        {
            let mut stdin = self.stdin.lock().await;
            if let Err(err) = stdin.write_all(&message).await {
                self.pending.lock().await.remove(&id);
                return Err(McpClientError::Io(err));
            }
            if let Err(err) = stdin.flush().await {
                self.pending.lock().await.remove(&id);
                return Err(McpClientError::Io(err));
            }
        }

        match rx.await {
            Ok(response) => response,
            Err(_) => Err(McpClientError::ConnectionClosed),
        }
    }

    async fn send_notification(&self, body: &Value) -> Result<(), McpClientError> {
        let message = encode_newline_message(body).map_err(McpClientError::Json)?;
        let mut stdin = self.stdin.lock().await;
        stdin
            .write_all(&message)
            .await
            .map_err(McpClientError::Io)?;
        stdin.flush().await.map_err(McpClientError::Io)?;
        Ok(())
    }
}

impl StreamableHttpTransport {
    async fn send_request(
        &self,
        expected_id: u64,
        method: &str,
        body: &Value,
    ) -> Result<Value, McpClientError> {
        let protocol_version = self.protocol_version.lock().await.clone();
        let session_id = self.session_id.lock().await.clone();
        let session_header_present = session_id.is_some();

        let mut request = self
            .client
            .post(&self.url)
            .header(ACCEPT, "application/json, text/event-stream")
            .header(CONTENT_TYPE, "application/json");

        request = apply_header_map(request, &self.headers)?;
        if let Some(protocol_version) = protocol_version {
            request = request.header("MCP-Protocol-Version", protocol_version);
        }
        if let Some(session_id) = session_id {
            request = request.header("Mcp-Session-Id", session_id);
        }

        let response = request
            .json(body)
            .send()
            .await
            .map_err(McpClientError::Http)?;

        if response.status() == StatusCode::NOT_FOUND && session_header_present {
            return Err(McpClientError::SessionExpired);
        }

        let status = response.status();
        let response_headers = response.headers().clone();
        if !status.is_success() {
            let body_text = response.text().await.map_err(McpClientError::Http)?;
            return Err(McpClientError::HttpStatus {
                status: status.as_u16(),
                body: body_text,
            });
        }

        if status == StatusCode::ACCEPTED {
            return Err(McpClientError::Transport(
                "server returned HTTP 202 for a request that expects a JSON-RPC response"
                    .to_owned(),
            ));
        }

        let session_id_from_response = if method == "initialize" {
            response_headers
                .get("Mcp-Session-Id")
                .and_then(|value| value.to_str().ok())
                .map(str::to_owned)
        } else {
            None
        };

        let content_type = response_headers
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_owned();

        let result = if content_type.contains("text/event-stream") {
            let body_text = response.text().await.map_err(McpClientError::Http)?;
            extract_expected_response_from_sse(&body_text, expected_id)?
        } else if content_type.contains("application/json") || content_type.is_empty() {
            let message = response
                .json::<Value>()
                .await
                .map_err(McpClientError::Http)?;
            extract_expected_response(message, expected_id)?
        } else {
            return Err(McpClientError::Transport(format!(
                "unsupported HTTP response content type `{content_type}`"
            )));
        };

        if let Some(session_id) = session_id_from_response {
            *self.session_id.lock().await = Some(session_id);
        }

        Ok(result)
    }

    async fn send_notification(&self, _method: &str, body: &Value) -> Result<(), McpClientError> {
        let protocol_version = self.protocol_version.lock().await.clone();
        let session_id = self.session_id.lock().await.clone();

        let mut request = self
            .client
            .post(&self.url)
            .header(ACCEPT, "application/json, text/event-stream")
            .header(CONTENT_TYPE, "application/json");

        request = apply_header_map(request, &self.headers)?;
        if let Some(protocol_version) = protocol_version {
            request = request.header("MCP-Protocol-Version", protocol_version);
        }
        if let Some(session_id) = session_id {
            request = request.header("Mcp-Session-Id", session_id);
        }

        let response = request
            .json(body)
            .send()
            .await
            .map_err(McpClientError::Http)?;

        let status = response.status();
        if status == StatusCode::NOT_FOUND {
            return Err(McpClientError::SessionExpired);
        }
        if status == StatusCode::ACCEPTED || status == StatusCode::NO_CONTENT {
            return Ok(());
        }
        if !status.is_success() {
            let body_text = response.text().await.map_err(McpClientError::Http)?;
            return Err(McpClientError::HttpStatus {
                status: status.as_u16(),
                body: body_text,
            });
        }

        Ok(())
    }

    async fn set_protocol_version(&self, version: String) {
        *self.protocol_version.lock().await = Some(version);
    }
}

fn supports_protocol_version(version: &str) -> bool {
    SUPPORTED_PROTOCOL_VERSIONS.contains(&version)
}

fn apply_header_map(
    mut request: reqwest::RequestBuilder,
    headers: &HashMap<String, String>,
) -> Result<reqwest::RequestBuilder, McpClientError> {
    for (name, value) in headers {
        let header_name = HeaderName::from_bytes(name.as_bytes()).map_err(|err| {
            McpClientError::Transport(format!("invalid HTTP header name `{name}`: {err}"))
        })?;
        let header_value = HeaderValue::from_str(value).map_err(|err| {
            McpClientError::Transport(format!("invalid HTTP header value for `{name}`: {err}"))
        })?;
        request = request.header(header_name, header_value);
    }

    Ok(request)
}

/// Identity sent to the server during `initialize`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ClientInfo {
    /// Client name advertised during the MCP initialize handshake.
    pub name: String,
    /// Client version advertised during the MCP initialize handshake.
    pub version: String,
}

impl ClientInfo {
    /// Convenience constructor for the small identity payload required by MCP.
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
        }
    }
}

/// Result of the MCP `initialize` handshake.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct McpInitializeResult {
    /// Negotiated protocol version returned by the server.
    pub protocol_version: String,
    /// Raw MCP capability map returned by the server.
    pub capabilities: Value,
    /// Human-readable server identity block.
    pub server_info: ServerInfo,
    /// Optional free-form instructions returned during initialization.
    #[serde(default)]
    pub instructions: Option<String>,
}

/// Human-readable server identity returned by the MCP handshake.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct ServerInfo {
    /// Stable server implementation name.
    pub name: String,
    /// Server version string.
    pub version: String,
    /// Optional display title separate from the stable name.
    #[serde(default)]
    pub title: Option<String>,
}

/// Result of one MCP `tools/list` request.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct McpToolsListResult {
    /// Current page of tool descriptors.
    pub tools: Vec<McpToolDescriptor>,
    /// Optional pagination cursor returned by some servers.
    #[serde(default)]
    pub next_cursor: Option<String>,
}

/// One tool descriptor advertised by an MCP server.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct McpToolDescriptor {
    /// Stable tool identifier used in `tools/call`.
    pub name: McpToolName,
    /// Optional human-friendly title.
    #[serde(default)]
    pub title: Option<String>,
    /// Free-form tool description surfaced to callers and prompts.
    #[serde(default)]
    pub description: Option<String>,
    /// JSON schema describing the accepted argument payload.
    #[serde(default)]
    pub input_schema: Value,
}

/// Result of one MCP `resources/list` request.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct McpResourcesListResult {
    /// Current page of resource descriptors.
    #[serde(default)]
    pub resources: Vec<McpResourceDescriptor>,
    /// Optional pagination cursor returned by some servers.
    #[serde(default)]
    pub next_cursor: Option<String>,
}

/// One resource descriptor advertised by an MCP server.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct McpResourceDescriptor {
    /// Stable resource URI used in `resources/read`.
    pub uri: String,
    /// Optional stable name separate from the URI.
    #[serde(default)]
    pub name: Option<String>,
    /// Optional human-friendly title.
    #[serde(default)]
    pub title: Option<String>,
    /// Free-form description surfaced to callers and prompts.
    #[serde(default)]
    pub description: Option<String>,
    /// Optional MIME type associated with reads.
    #[serde(default)]
    pub mime_type: Option<String>,
    /// Provider-specific annotations or metadata.
    #[serde(default)]
    pub annotations: Option<Value>,
}

/// Stable identifier for a tool exposed through MCP.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct McpToolName(String);

impl McpToolName {
    /// Creates a tool name after rejecting blank identifiers.
    pub fn new(value: impl Into<String>) -> Result<Self, McpClientError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(McpClientError::InvalidToolName);
        }
        Ok(Self(value))
    }

    /// Borrows the normalized tool name.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for McpToolName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Result payload from one MCP `tools/call` request.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct McpToolCallResult {
    /// Ordered content fragments returned by the tool call.
    #[serde(default)]
    pub content: Vec<McpToolContent>,
    /// Optional structured result payload separate from human-readable content.
    #[serde(default)]
    pub structured_content: Option<Value>,
    /// Whether the tool considered the call an application-level error.
    #[serde(default)]
    pub is_error: bool,
}

/// Result payload from one MCP `resources/read` request.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct McpResourceReadResult {
    /// Ordered resource content fragments returned by the read.
    #[serde(default)]
    pub contents: Vec<Value>,
}

/// One content item returned by an MCP tool call.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum McpToolContent {
    Text {
        /// Plain text content emitted by the tool.
        text: String,
    },
    Image {
        /// Base64-encoded image payload.
        data: String,
        /// MIME type associated with `data`.
        mime_type: String,
        /// Optional provider-specific image annotations.
        #[serde(default)]
        annotations: Option<Value>,
    },
    Resource {
        /// Arbitrary resource payload returned by the tool.
        resource: Value,
    },
}

/// Top-level error type for process launch, transport, and JSON-RPC failures.
#[derive(Debug)]
pub enum McpClientError {
    /// Failed to spawn a configured stdio child process.
    Spawn(io::Error),
    /// Expected a piped stdio stream from the child but it was unavailable.
    MissingPipe(&'static str),
    /// Generic I/O failure while talking to the transport.
    Io(io::Error),
    /// HTTP client-level failure before a provider response was received.
    Http(reqwest::Error),
    /// Non-success HTTP response from a Streamable HTTP server.
    HttpStatus { status: u16, body: String },
    /// JSON serialization or deserialization failed.
    Json(serde_json::Error),
    /// Tool identifiers must not be blank.
    InvalidToolName,
    /// Transport closed while requests were still pending.
    ConnectionClosed,
    /// Streamable HTTP server discarded the current session.
    SessionExpired,
    /// Other transport-level failure that does not map to a richer variant.
    Transport(String),
    /// JSON-RPC application error returned by the server.
    Rpc {
        code: i64,
        message: String,
        data: Option<Value>,
    },
    /// Handshake completed but negotiated a protocol version this client does not support.
    UnsupportedProtocolVersion(String),
}

impl fmt::Display for McpClientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spawn(err) => write!(f, "failed to spawn MCP server process: {err}"),
            Self::MissingPipe(pipe) => write!(f, "child process is missing a piped {pipe} stream"),
            Self::Io(err) => write!(f, "transport I/O failed: {err}"),
            Self::Http(err) => write!(f, "HTTP transport failed: {err}"),
            Self::HttpStatus { status, body } if body.trim().is_empty() => {
                write!(f, "HTTP transport returned status {status}")
            }
            Self::HttpStatus { status, body } => {
                write!(f, "HTTP transport returned status {status}: {body}")
            }
            Self::Json(err) => write!(f, "invalid JSON-RPC payload: {err}"),
            Self::InvalidToolName => write!(f, "MCP tool name cannot be blank"),
            Self::ConnectionClosed => {
                write!(f, "transport closed before a response was received")
            }
            Self::SessionExpired => write!(f, "HTTP MCP session expired; re-initialize"),
            Self::Transport(message) => write!(f, "transport failed: {message}"),
            Self::Rpc { code, message, .. } => {
                write!(f, "MCP server returned JSON-RPC error {code}: {message}")
            }
            Self::UnsupportedProtocolVersion(version) => {
                write!(
                    f,
                    "server negotiated unsupported protocol version `{version}`"
                )
            }
        }
    }
}

impl Error for McpClientError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Spawn(err) => Some(err),
            Self::Io(err) => Some(err),
            Self::Http(err) => Some(err),
            Self::Json(err) => Some(err),
            Self::MissingPipe(_)
            | Self::HttpStatus { .. }
            | Self::InvalidToolName
            | Self::ConnectionClosed
            | Self::SessionExpired
            | Self::Transport(_)
            | Self::Rpc { .. }
            | Self::UnsupportedProtocolVersion(_) => None,
        }
    }
}

type PendingSender = oneshot::Sender<Result<Value, McpClientError>>;

/// Spawns the background reader that owns message decoding for the connection.
///
/// Any unreadable or closed transport is treated as terminal and is broadcast
/// to all pending requests.
fn spawn_reader(
    stdout: ChildStdout,
    pending: Arc<Mutex<HashMap<u64, PendingSender>>>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut reader = BufReader::new(stdout);

        loop {
            match read_message(&mut reader).await {
                Ok(Some(message)) => {
                    if let Some((id, response)) = parse_response(message) {
                        if let Some(sender) = pending.lock().await.remove(&id) {
                            let _ = sender.send(response);
                        }
                    }
                }
                Ok(None) => {
                    fail_pending(&pending, PendingFailure::Closed).await;
                    break;
                }
                Err(err) => {
                    fail_pending(&pending, PendingFailure::Transport(err.to_string())).await;
                    break;
                }
            }
        }
    })
}

/// Converts a raw server message into a `(request_id, result)` pair.
///
/// Notifications and any other message without an `id` are ignored because
/// they are not responses to a waiting caller.
fn parse_response(message: Value) -> Option<(u64, Result<Value, McpClientError>)> {
    let id = response_id(message.get("id")?)?;

    if let Some(error) = message.get("error") {
        let error = serde_json::from_value::<RpcError>(error.clone()).ok()?;
        return Some((
            id,
            Err(McpClientError::Rpc {
                code: error.code,
                message: error.message,
                data: error.data,
            }),
        ));
    }

    let result = message.get("result")?.clone();
    Some((id, Ok(result)))
}

fn extract_expected_response(message: Value, expected_id: u64) -> Result<Value, McpClientError> {
    if let Some(result) = match_expected_response(message, expected_id) {
        return result;
    }

    Err(McpClientError::Transport(format!(
        "response payload did not contain JSON-RPC response id `{expected_id}`"
    )))
}

fn extract_expected_response_from_sse(
    body: &str,
    expected_id: u64,
) -> Result<Value, McpClientError> {
    for message in parse_sse_json_messages(body)? {
        if let Some(result) = match_expected_response(message, expected_id) {
            return result;
        }
    }

    Err(McpClientError::ConnectionClosed)
}

fn match_expected_response(
    message: Value,
    expected_id: u64,
) -> Option<Result<Value, McpClientError>> {
    if let Some((id, response)) = parse_response(message.clone()) {
        if id == expected_id {
            return Some(response);
        }
    }

    let batch = message.as_array()?;
    for entry in batch {
        if let Some((id, response)) = parse_response(entry.clone()) {
            if id == expected_id {
                return Some(response);
            }
        }
    }

    None
}

fn parse_sse_json_messages(body: &str) -> Result<Vec<Value>, McpClientError> {
    let normalized = body.replace("\r\n", "\n");
    let mut messages = Vec::new();

    for frame in normalized.split("\n\n") {
        let mut data_lines = Vec::new();
        for line in frame.lines() {
            if let Some(data) = line.strip_prefix("data:") {
                data_lines.push(data.trim_start());
            }
        }

        if data_lines.is_empty() {
            continue;
        }

        let payload = data_lines.join("\n");
        if payload.trim().is_empty() {
            continue;
        }

        messages.push(serde_json::from_str::<Value>(&payload).map_err(McpClientError::Json)?);
    }

    Ok(messages)
}

/// Normalizes the JSON-RPC `id` field into the local `u64` representation.
fn response_id(value: &Value) -> Option<u64> {
    value
        .as_u64()
        .or_else(|| value.as_i64().filter(|id| *id >= 0).map(|id| id as u64))
}

/// Resolves every in-flight request with the same terminal transport failure.
async fn fail_pending(pending: &Arc<Mutex<HashMap<u64, PendingSender>>>, failure: PendingFailure) {
    let mut pending = pending.lock().await;
    for (_, sender) in pending.drain() {
        let error = match &failure {
            PendingFailure::Closed => McpClientError::ConnectionClosed,
            PendingFailure::Transport(message) => McpClientError::Transport(message.clone()),
        };
        let _ = sender.send(Err(error));
    }
}

enum PendingFailure {
    Closed,
    Transport(String),
}

#[derive(Serialize)]
struct JsonRpcRequest {
    jsonrpc: &'static str,
    id: u64,
    method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<Value>,
}

#[derive(Serialize)]
struct JsonRpcNotification {
    jsonrpc: &'static str,
    method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<Value>,
}

#[derive(Deserialize)]
struct RpcError {
    code: i64,
    message: String,
    #[serde(default)]
    data: Option<Value>,
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use axum::{
        Router,
        extract::State,
        http::{HeaderMap as AxumHeaderMap, HeaderValue as AxumHeaderValue, StatusCode},
        response::IntoResponse,
        routing::post,
    };
    use serde_json::{Value, json};
    use tokio::{
        io::{AsyncWriteExt, BufReader, duplex},
        net::TcpListener,
        sync::Mutex as TokioMutex,
    };

    use super::{
        CONTENT_TYPE, ClientInfo, McpClient, McpClientError, McpToolCallResult, McpToolContent,
        McpToolName,
    };

    #[test]
    fn rejects_blank_mcp_tool_names() {
        let error = McpToolName::new("  ").expect_err("blank names must fail");
        assert!(matches!(error, super::McpClientError::InvalidToolName));
    }

    #[test]
    fn deserializes_tool_call_result_payloads() {
        let payload = json!({
            "content": [
                {
                    "type": "text",
                    "text": "rows returned: 3"
                }
            ],
            "structuredContent": {
                "rowCount": 3
            },
            "isError": false
        });

        let result: McpToolCallResult =
            serde_json::from_value(payload).expect("payload should deserialize");
        assert!(!result.is_error);
        assert_eq!(result.structured_content, Some(json!({"rowCount": 3})));
        assert_eq!(
            result.content,
            vec![McpToolContent::Text {
                text: "rows returned: 3".to_owned()
            }]
        );
    }

    #[tokio::test]
    async fn reads_newline_delimited_messages() {
        let message = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": { "ok": true }
        });
        let (mut writer, reader) = duplex(1024);

        writer
            .write_all(&crate::codec::encode_newline_message(&message).expect("message encoding"))
            .await
            .expect("write should succeed");
        drop(writer);

        let mut reader = BufReader::new(reader);
        let parsed = crate::codec::read_message(&mut reader)
            .await
            .expect("read should succeed")
            .expect("message should exist");

        assert_eq!(parsed, message);
    }

    #[tokio::test]
    async fn reads_content_length_messages() {
        let message = json!({
            "jsonrpc": "2.0",
            "id": 7,
            "result": { "toolCount": 2 }
        });
        let (mut writer, reader) = duplex(1024);

        writer
            .write_all(
                &crate::codec::encode_content_length_message(&message).expect("message encoding"),
            )
            .await
            .expect("write should succeed");
        drop(writer);

        let mut reader = BufReader::new(reader);
        let parsed = crate::codec::read_message(&mut reader)
            .await
            .expect("read should succeed")
            .expect("message should exist");

        assert_eq!(parsed, message);
    }

    #[tokio::test]
    async fn streamable_http_reuses_initialize_and_tool_listing_flow() {
        let state = HttpTestState::default();
        let Some(listener) = bind_test_listener().await else {
            return;
        };
        let addr = listener.local_addr().expect("addr should exist");
        let server = tokio::spawn(axum::serve(listener, app(state.clone())).into_future());

        let config = mcp_config::McpServerConfig {
            name: "remote".to_owned(),
            transport: Some(mcp_config::McpTransportConfig::StreamableHttp {
                url: format!("http://{addr}/mcp"),
                headers: HashMap::new(),
            }),
            command: String::new(),
            args: Vec::new(),
            env: HashMap::new(),
            description: None,
        };

        let connection = McpClient::connect(&config)
            .await
            .expect("connection should succeed");
        let initialize_result = connection
            .initialize(ClientInfo::new("test-client", "1.0.0"))
            .await
            .expect("initialize should succeed");
        connection
            .notify_initialized()
            .await
            .expect("initialized notification should succeed");
        let tools = connection
            .list_tools()
            .await
            .expect("tools/list should succeed");

        assert_eq!(initialize_result.server_info.name, "fake-http-server");
        assert_eq!(tools.tools.len(), 2);

        let recorded = state.recorded_headers.lock().await.clone();
        assert_eq!(
            recorded,
            vec![
                RecordedHeaders {
                    method: "initialize".to_owned(),
                    protocol_version: None,
                    session_id: None,
                },
                RecordedHeaders {
                    method: "notifications/initialized".to_owned(),
                    protocol_version: Some("2025-11-25".to_owned()),
                    session_id: Some("session-123".to_owned()),
                },
                RecordedHeaders {
                    method: "tools/list".to_owned(),
                    protocol_version: Some("2025-11-25".to_owned()),
                    session_id: Some("session-123".to_owned()),
                },
            ]
        );

        server.abort();
    }

    #[tokio::test]
    async fn streamable_http_accepts_sse_responses() {
        let state = HttpTestState {
            respond_with_sse: true,
            ..HttpTestState::default()
        };
        let Some(listener) = bind_test_listener().await else {
            return;
        };
        let addr = listener.local_addr().expect("addr should exist");
        let server = tokio::spawn(axum::serve(listener, app(state)).into_future());

        let config = mcp_config::McpServerConfig {
            name: "remote".to_owned(),
            transport: Some(mcp_config::McpTransportConfig::StreamableHttp {
                url: format!("http://{addr}/mcp"),
                headers: HashMap::new(),
            }),
            command: String::new(),
            args: Vec::new(),
            env: HashMap::new(),
            description: None,
        };

        let connection = McpClient::connect(&config)
            .await
            .expect("connection should succeed");
        connection
            .initialize(ClientInfo::new("test-client", "1.0.0"))
            .await
            .expect("initialize should succeed");
        connection
            .notify_initialized()
            .await
            .expect("initialized notification should succeed");
        let tools = connection
            .list_tools()
            .await
            .expect("tools/list should succeed");
        assert_eq!(tools.tools.len(), 2);

        server.abort();
    }

    #[tokio::test]
    async fn initialize_accepts_older_supported_protocol_and_reuses_it_for_headers() {
        let state = HttpTestState {
            initialize_protocol_version: "2025-03-26".to_owned(),
            ..HttpTestState::default()
        };
        let Some(listener) = bind_test_listener().await else {
            return;
        };
        let addr = listener.local_addr().expect("addr should exist");
        let server = tokio::spawn(axum::serve(listener, app(state.clone())).into_future());

        let config = mcp_config::McpServerConfig {
            name: "remote".to_owned(),
            transport: Some(mcp_config::McpTransportConfig::StreamableHttp {
                url: format!("http://{addr}/mcp"),
                headers: HashMap::new(),
            }),
            command: String::new(),
            args: Vec::new(),
            env: HashMap::new(),
            description: None,
        };

        let connection = McpClient::connect(&config)
            .await
            .expect("connection should succeed");
        let initialize_result = connection
            .initialize(ClientInfo::new("test-client", "1.0.0"))
            .await
            .expect("initialize should succeed");
        connection
            .notify_initialized()
            .await
            .expect("initialized notification should succeed");
        connection
            .list_tools()
            .await
            .expect("tools/list should succeed");

        assert_eq!(initialize_result.protocol_version, "2025-03-26");

        let recorded = state.recorded_headers.lock().await.clone();
        assert_eq!(
            recorded,
            vec![
                RecordedHeaders {
                    method: "initialize".to_owned(),
                    protocol_version: None,
                    session_id: None,
                },
                RecordedHeaders {
                    method: "notifications/initialized".to_owned(),
                    protocol_version: Some("2025-03-26".to_owned()),
                    session_id: Some("session-123".to_owned()),
                },
                RecordedHeaders {
                    method: "tools/list".to_owned(),
                    protocol_version: Some("2025-03-26".to_owned()),
                    session_id: Some("session-123".to_owned()),
                },
            ]
        );

        server.abort();
    }

    #[tokio::test]
    async fn initialize_rejects_negotiated_protocols_outside_supported_set() {
        let state = HttpTestState {
            initialize_protocol_version: "1999-01-01".to_owned(),
            ..HttpTestState::default()
        };
        let Some(listener) = bind_test_listener().await else {
            return;
        };
        let addr = listener.local_addr().expect("addr should exist");
        let server = tokio::spawn(axum::serve(listener, app(state)).into_future());

        let config = mcp_config::McpServerConfig {
            name: "remote".to_owned(),
            transport: Some(mcp_config::McpTransportConfig::StreamableHttp {
                url: format!("http://{addr}/mcp"),
                headers: HashMap::new(),
            }),
            command: String::new(),
            args: Vec::new(),
            env: HashMap::new(),
            description: None,
        };

        let connection = McpClient::connect(&config)
            .await
            .expect("connection should succeed");
        let error = connection
            .initialize(ClientInfo::new("test-client", "1.0.0"))
            .await
            .expect_err("initialize should reject unsupported protocol");

        assert!(matches!(
            error,
            McpClientError::UnsupportedProtocolVersion(version) if version == "1999-01-01"
        ));

        server.abort();
    }

    #[derive(Clone, Debug, Default, PartialEq, Eq)]
    struct RecordedHeaders {
        method: String,
        protocol_version: Option<String>,
        session_id: Option<String>,
    }

    #[derive(Clone, Debug)]
    struct HttpTestState {
        recorded_headers: std::sync::Arc<TokioMutex<Vec<RecordedHeaders>>>,
        respond_with_sse: bool,
        initialize_protocol_version: String,
    }

    impl Default for HttpTestState {
        fn default() -> Self {
            Self {
                recorded_headers: std::sync::Arc::new(TokioMutex::new(Vec::new())),
                respond_with_sse: false,
                initialize_protocol_version: super::SUPPORTED_PROTOCOL_VERSION.to_owned(),
            }
        }
    }

    async fn bind_test_listener() -> Option<TcpListener> {
        match TcpListener::bind("127.0.0.1:0").await {
            Ok(listener) => Some(listener),
            Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => None,
            Err(error) => panic!("listener should bind: {error}"),
        }
    }

    fn app(state: HttpTestState) -> Router {
        Router::new()
            .route("/mcp", post(handle_mcp))
            .with_state(state)
    }

    async fn handle_mcp(
        State(state): State<HttpTestState>,
        headers: AxumHeaderMap,
        axum::Json(payload): axum::Json<Value>,
    ) -> impl IntoResponse {
        let method = payload
            .get("method")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        state.recorded_headers.lock().await.push(RecordedHeaders {
            method: method.clone(),
            protocol_version: headers
                .get("MCP-Protocol-Version")
                .and_then(|value| value.to_str().ok())
                .map(str::to_owned),
            session_id: headers
                .get("Mcp-Session-Id")
                .and_then(|value| value.to_str().ok())
                .map(str::to_owned),
        });

        if method == "notifications/initialized" {
            return (StatusCode::ACCEPTED, AxumHeaderMap::new(), String::new()).into_response();
        }

        let id = payload.get("id").cloned().expect("request id should exist");
        let body = match method.as_str() {
            "initialize" => json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "protocolVersion": state.initialize_protocol_version.clone(),
                    "capabilities": {},
                    "serverInfo": {
                        "name": "fake-http-server",
                        "version": "1.0.0"
                    }
                }
            }),
            "tools/list" => json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "tools": [
                        {
                            "name": "run-sql",
                            "description": "Execute SQL",
                            "inputSchema": {
                                "type": "object"
                            }
                        },
                        {
                            "name": "describe-table",
                            "description": "Describe a table",
                            "inputSchema": {
                                "type": "object"
                            }
                        }
                    ]
                }
            }),
            _ => json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {
                    "code": -32601,
                    "message": format!("unknown method `{method}`")
                }
            }),
        };

        let mut response_headers = AxumHeaderMap::new();
        if method == "initialize" {
            response_headers.insert(
                "Mcp-Session-Id",
                AxumHeaderValue::from_static("session-123"),
            );
        }

        if state.respond_with_sse && method == "tools/list" {
            response_headers.insert(
                CONTENT_TYPE,
                AxumHeaderValue::from_static("text/event-stream"),
            );
            let body = format!("event: message\ndata: {}\n\n", body);
            return (StatusCode::OK, response_headers, body).into_response();
        }

        response_headers.insert(
            CONTENT_TYPE,
            AxumHeaderValue::from_static("application/json"),
        );
        (StatusCode::OK, response_headers, body.to_string()).into_response()
    }
}
