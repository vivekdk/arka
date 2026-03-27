//! Terminal client for the control-plane server.
//!
//! The binary supports both a chat-oriented interactive mode and focused
//! subcommands for session management, SSE watching, and registry editing.

use std::{
    env,
    future::Future,
    io::{self, BufRead, IsTerminal, Write},
    path::PathBuf,
    process,
    time::Duration,
};

use agent_controlplane::{
    ApprovalDecision, ApprovalRequestRecord, ApprovalState, ChannelKind, CreateSessionRequest,
    CompleteWhatsAppLoginRequest, ReceiveWhatsAppMessageRequest, ReceiveWhatsAppMessageResponse,
    SendSessionMessageRequest, SendSessionMessageResponse, SessionEvent, SessionId, SessionRecord,
    SessionStatus, StartWhatsAppLoginResponse, SubmitApprovalRequest, TurnRecordSummary,
    WhatsAppGatewayStatus,
};
use agent_runtime::{RuntimeEvent, TerminationReason};
use futures_util::StreamExt;
use mcp_client::{
    ClientInfo, McpClient, McpClientError, McpInitializeResult, McpResourceDescriptor,
    McpToolDescriptor,
};
use mcp_config::{McpRegistry, McpServerConfig, McpTransportConfig};
use mcp_metadata::{
    CURRENT_SCHEMA_VERSION, DEFAULT_METADATA_DIR, FullResourceMetadata, FullToolMetadata,
    McpCapabilityFamilies, McpCapabilityFamilySummary, McpFullCatalog, McpMinimalCatalog,
    McpServerMetadata, MinimalResourceMetadata, MinimalToolMetadata, artifact_paths,
    load_full_catalog, load_minimal_catalog, write_catalogs,
};
use reqwest::StatusCode;
use rustyline::{DefaultEditor, error::ReadlineError};
use tokio::sync::oneshot;
use uuid::Uuid;

const CONNECT_RETRY_COUNT: u32 = 5;
const INITIAL_BACKOFF_MS: u64 = 500;
const SESSION_START_TIMEOUT_SECS: u64 = 30;
const REQUEST_TIMEOUT_SECS: u64 = 60;
const APP_NAME: &str = "Arka";
const APP_TAGLINE: &str = "Data analyst assistant";
const USER_LABEL: &str = "You";
const TURN_SPINNER_FRAMES: [&str; 4] = ["-", "\\", "|", "/"];
const MCP_REMOTE_PACKAGE_PREFIX: &str = "mcp-remote";

type CliResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("error: {error}");
        process::exit(1);
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut args = env::args();
    let bin = args
        .next()
        .unwrap_or_else(|| "agent-channel-cli".to_owned());
    let theme = CliTheme::detect();
    let base_url =
        env::var("AGENT_API_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:8080".to_owned());
    let command = args.next();
    let session_client = build_http_client(Some(Duration::from_secs(SESSION_START_TIMEOUT_SECS)))?;
    let request_client = build_http_client(Some(Duration::from_secs(REQUEST_TIMEOUT_SECS)))?;
    let stream_client = build_http_client(None)?;

    match command.as_deref() {
        None => run_chat_mode(&session_client, &request_client, &stream_client, &base_url).await?,
        Some("session") => match args.next().as_deref() {
            Some("start") => {
                let session = create_cli_session(&session_client, &base_url).await?;
                println!("{}", serde_json::to_string(&session)?);
            }
            Some("watch") => {
                let session_id = args.next().ok_or_else(|| usage(&bin))?;
                stream_session_events(&stream_client, &base_url, &session_id, false).await?;
            }
            _ => return Err(usage(&bin).into()),
        },
        Some("message") => match args.next().as_deref() {
            Some("send") => {
                let session_id = args.next().ok_or_else(|| usage(&bin))?;
                let text = args.collect::<Vec<_>>().join(" ");
                if text.trim().is_empty() {
                    return Err("message text cannot be empty".into());
                }
                let response = send_message_with_feedback(
                    &theme,
                    &request_client,
                    &base_url,
                    &session_id,
                    text,
                )
                .await?;
                print_chat_response(&theme, &response);
            }
            _ => return Err(usage(&bin).into()),
        },
        Some("approve") => {
            let session_id = args.next().ok_or_else(|| usage(&bin))?;
            let approval_id = args.next().ok_or_else(|| usage(&bin))?;
            let decision = match args.next().as_deref() {
                Some("approve") => ApprovalDecision::Approve,
                Some("reject") => ApprovalDecision::Reject,
                _ => return Err(usage(&bin).into()),
            };
            let approval = submit_approval(
                &request_client,
                &base_url,
                &session_id,
                &approval_id,
                decision,
            )
            .await?;
            println!("{}", render_approval_resolution(&theme, &approval));
        }
        Some("whatsapp") => match args.next().as_deref() {
            Some("status") => {
                let status = fetch_whatsapp_status(&request_client, &base_url).await?;
                print_whatsapp_status(&theme, &status);
            }
            Some("login") => {
                let response = start_whatsapp_login(&request_client, &base_url).await?;
                print_whatsapp_login(&theme, &response);
            }
            Some("complete-login") => {
                let login_session_id = args.next().ok_or_else(|| usage(&bin))?;
                let status = complete_whatsapp_login(
                    &request_client,
                    &base_url,
                    &login_session_id,
                )
                .await?;
                print_whatsapp_status(&theme, &status);
            }
            Some("logout") => {
                let status = logout_whatsapp(&request_client, &base_url).await?;
                print_whatsapp_status(&theme, &status);
            }
            Some("receive") => {
                let from_user_id = args.next().ok_or_else(|| usage(&bin))?;
                let text = args.collect::<Vec<_>>().join(" ");
                if text.trim().is_empty() {
                    return Err("whatsapp message text cannot be empty".into());
                }
                let response = send_whatsapp_message(
                    &request_client,
                    &base_url,
                    ReceiveWhatsAppMessageRequest {
                        message_id: Uuid::new_v4().to_string(),
                        account_id: None,
                        conversation_id: from_user_id.clone(),
                        from_user_id,
                        text,
                        quoted_message_id: None,
                        quoted_text: None,
                    },
                )
                .await?;
                print_whatsapp_receive(&theme, &response);
            }
            _ => return Err(usage(&bin).into()),
        },
        Some("repl") => {
            run_chat_mode(&session_client, &request_client, &stream_client, &base_url).await?
        }
        Some("help") | Some("--help") | Some("-h") => {
            println!("{}", usage(&bin));
        }
        Some(_) => return Err(usage(&bin).into()),
    }

    Ok(())
}

async fn run_chat_mode(
    session_client: &reqwest::Client,
    request_client: &reqwest::Client,
    stream_client: &reqwest::Client,
    base_url: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let theme = CliTheme::detect();
    let session = create_cli_session(session_client, base_url).await?;
    println!("{}", render_welcome_banner(&theme, &session, base_url));
    println!(
        "{}",
        render_notice(
            &theme,
            NoticeTone::Info,
            "session",
            &format!("started {}", session.session_id),
        )
    );
    println!(
        "{}",
        render_notice(
            &theme,
            NoticeTone::Muted,
            "repl",
            "type messages directly. /help for commands.",
        )
    );

    if io::stdin().is_terminal() && io::stdout().is_terminal() {
        run_interactive_chat_loop(
            &theme,
            session_client,
            request_client,
            stream_client,
            base_url,
            session,
        )
        .await?;
    } else {
        run_line_chat_loop(
            &theme,
            session_client,
            request_client,
            stream_client,
            base_url,
            session,
        )
        .await?;
    }

    Ok(())
}

async fn run_interactive_chat_loop(
    theme: &CliTheme,
    session_client: &reqwest::Client,
    request_client: &reqwest::Client,
    stream_client: &reqwest::Client,
    base_url: &str,
    mut session: SessionRecord,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut editor = DefaultEditor::new()
        .map_err(|error| format!("failed to initialize interactive input: {error}"))?;

    loop {
        match editor.readline(readline_prompt()) {
            Ok(line) => {
                let input = line.trim();
                if input.is_empty() {
                    continue;
                }

                let _ = editor.add_history_entry(input);
                clear_previous_line()?;
                match handle_chat_input(
                    theme,
                    session_client,
                    request_client,
                    stream_client,
                    base_url,
                    &mut session,
                    input,
                    true,
                )
                .await?
                {
                    ChatLoopAction::Continue => {}
                    ChatLoopAction::Exit => break,
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!(
                    "{}",
                    render_notice(theme, NoticeTone::Muted, "repl", "exiting")
                );
                break;
            }
            Err(ReadlineError::Eof) => break,
            Err(error) => return Err(format!("interactive input failed: {error}").into()),
        }
    }

    Ok(())
}

async fn run_line_chat_loop(
    theme: &CliTheme,
    session_client: &reqwest::Client,
    request_client: &reqwest::Client,
    stream_client: &reqwest::Client,
    base_url: &str,
    mut session: SessionRecord,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    print_prompt(theme);
    io::stdout().flush()?;

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = line?;
        let input = line.trim();
        if input.is_empty() {
            print_prompt(theme);
            io::stdout().flush()?;
            continue;
        }

        match handle_chat_input(
            theme,
            session_client,
            request_client,
            stream_client,
            base_url,
            &mut session,
            input,
            false,
        )
        .await?
        {
            ChatLoopAction::Continue => {}
            ChatLoopAction::Exit => break,
        }

        print_prompt(theme);
        io::stdout().flush()?;
    }

    Ok(())
}

async fn handle_chat_input(
    theme: &CliTheme,
    session_client: &reqwest::Client,
    request_client: &reqwest::Client,
    stream_client: &reqwest::Client,
    base_url: &str,
    session: &mut SessionRecord,
    input: &str,
    echo_user_message: bool,
) -> Result<ChatLoopAction, Box<dyn std::error::Error + Send + Sync>> {
    match parse_chat_command(input) {
        ChatCommand::Help => print_repl_help(theme),
        ChatCommand::New => {
            *session = create_cli_session(session_client, base_url).await?;
            println!(
                "{}",
                render_notice(
                    theme,
                    NoticeTone::Info,
                    "session",
                    &format!("started {}", session.session_id),
                )
            );
        }
        ChatCommand::Session => println!(
            "{}",
            render_notice(
                theme,
                NoticeTone::Info,
                "session",
                &session.session_id.to_string(),
            )
        ),
        ChatCommand::Status => {
            let session_record =
                fetch_session(request_client, base_url, &session.session_id).await?;
            print_status(theme, &session_record);
        }
        ChatCommand::Watch => {
            stream_session_events(
                stream_client,
                base_url,
                &session.session_id.to_string(),
                false,
            )
            .await?;
        }
        ChatCommand::Approve {
            approval_id,
            decision,
        } => {
            let approval = submit_approval(
                request_client,
                base_url,
                &session.session_id.to_string(),
                &approval_id,
                decision,
            )
            .await?;
            println!("{}", render_approval_resolution(theme, &approval));
        }
        ChatCommand::Mcp(command) => handle_mcp_command(theme, command).await?,
        ChatCommand::Exit => return Ok(ChatLoopAction::Exit),
        ChatCommand::Unknown(command) => {
            println!(
                "{}",
                render_notice(
                    theme,
                    NoticeTone::Error,
                    "error",
                    &format!("unknown command `{command}`"),
                )
            );
            print_repl_help(theme);
        }
        ChatCommand::SendMessage(text) => {
            if echo_user_message {
                print_user_message(theme, &text);
            }

            let response = send_message_with_feedback(
                theme,
                request_client,
                base_url,
                &session.session_id.to_string(),
                text,
            )
            .await?;
            print_chat_response(theme, &response);
        }
    }

    Ok(ChatLoopAction::Continue)
}

fn build_http_client(
    timeout: Option<Duration>,
) -> Result<reqwest::Client, Box<dyn std::error::Error + Send + Sync>> {
    let mut builder = reqwest::Client::builder();
    if let Some(timeout) = timeout {
        builder = builder.timeout(timeout);
    }
    Ok(builder.build()?)
}

async fn create_cli_session(
    client: &reqwest::Client,
    base_url: &str,
) -> Result<SessionRecord, Box<dyn std::error::Error + Send + Sync>> {
    let session = send_with_retry(base_url, || async {
        client
            .post(format!("{base_url}/sessions"))
            .json(&CreateSessionRequest {
                channel: Some(ChannelKind::Cli),
                external_conversation_id: None,
                external_user_id: None,
            })
            .send()
            .await?
            .error_for_status()?
            .json::<SessionRecord>()
            .await
    })
    .await?;
    Ok(session)
}

async fn fetch_session(
    client: &reqwest::Client,
    base_url: &str,
    session_id: &SessionId,
) -> Result<SessionRecord, Box<dyn std::error::Error + Send + Sync>> {
    Ok(send_with_retry(base_url, || async {
        client
            .get(format!("{base_url}/sessions/{session_id}"))
            .send()
            .await?
            .error_for_status()?
            .json::<SessionRecord>()
            .await
    })
    .await?)
}

async fn send_message(
    client: &reqwest::Client,
    base_url: &str,
    session_id: &str,
    text: String,
) -> Result<SendSessionMessageResponse, Box<dyn std::error::Error + Send + Sync>> {
    Ok(send_with_retry(base_url, || async {
        client
            .post(format!("{base_url}/sessions/{session_id}/messages"))
            .json(&SendSessionMessageRequest {
                text: text.clone(),
                idempotency_key: None,
                response_format: None,
            })
            .send()
            .await?
            .error_for_status()?
            .json::<SendSessionMessageResponse>()
            .await
    })
    .await?)
}

async fn send_message_with_feedback(
    theme: &CliTheme,
    client: &reqwest::Client,
    base_url: &str,
    session_id: &str,
    text: String,
) -> Result<SendSessionMessageResponse, Box<dyn std::error::Error + Send + Sync>> {
    let spinner = TurnSpinner::start(*theme, "Thinking")?;
    let response = send_message(client, base_url, session_id, text).await;
    spinner.stop().await?;
    response
}

async fn submit_approval(
    client: &reqwest::Client,
    base_url: &str,
    session_id: &str,
    approval_id: &str,
    decision: ApprovalDecision,
) -> Result<ApprovalRequestRecord, Box<dyn std::error::Error + Send + Sync>> {
    let response = send_with_retry(base_url, || async {
        client
            .post(format!(
                "{base_url}/sessions/{session_id}/approvals/{approval_id}"
            ))
            .json(&SubmitApprovalRequest { decision })
            .send()
            .await
    })
    .await?;
    if response.status() == StatusCode::NOT_FOUND {
        return Err("approval not found".into());
    }
    Ok(response
        .error_for_status()?
        .json::<ApprovalRequestRecord>()
        .await?)
}

async fn fetch_whatsapp_status(
    client: &reqwest::Client,
    base_url: &str,
) -> CliResult<WhatsAppGatewayStatus> {
    let response = send_with_retry(base_url, || async {
        client
            .get(format!(
                "{}/channels/whatsapp/status",
                base_url.trim_end_matches('/')
            ))
            .send()
            .await
    })
    .await?;
    Ok(response
        .error_for_status()?
        .json::<WhatsAppGatewayStatus>()
        .await?)
}

async fn start_whatsapp_login(
    client: &reqwest::Client,
    base_url: &str,
) -> CliResult<StartWhatsAppLoginResponse> {
    let response = send_with_retry(base_url, || async {
        client
            .post(format!(
                "{}/channels/whatsapp/login/start",
                base_url.trim_end_matches('/')
            ))
            .send()
            .await
    })
    .await?;
    Ok(response
        .error_for_status()?
        .json::<StartWhatsAppLoginResponse>()
        .await?)
}

async fn complete_whatsapp_login(
    client: &reqwest::Client,
    base_url: &str,
    login_session_id: &str,
) -> CliResult<WhatsAppGatewayStatus> {
    let response = send_with_retry(base_url, || async {
        client
            .post(format!(
                "{}/channels/whatsapp/login/complete",
                base_url.trim_end_matches('/')
            ))
            .json(&CompleteWhatsAppLoginRequest {
                login_session_id: login_session_id.to_owned(),
            })
            .send()
            .await
    })
    .await?;
    Ok(response
        .error_for_status()?
        .json::<WhatsAppGatewayStatus>()
        .await?)
}

async fn logout_whatsapp(
    client: &reqwest::Client,
    base_url: &str,
) -> CliResult<WhatsAppGatewayStatus> {
    let response = send_with_retry(base_url, || async {
        client
            .post(format!(
                "{}/channels/whatsapp/logout",
                base_url.trim_end_matches('/')
            ))
            .send()
            .await
    })
    .await?;
    Ok(response
        .error_for_status()?
        .json::<WhatsAppGatewayStatus>()
        .await?)
}

async fn send_whatsapp_message(
    client: &reqwest::Client,
    base_url: &str,
    request: ReceiveWhatsAppMessageRequest,
) -> CliResult<ReceiveWhatsAppMessageResponse> {
    let response = send_with_retry(base_url, || {
        let request = request.clone();
        async move {
            client
                .post(format!(
                    "{}/channels/whatsapp/inbound",
                    base_url.trim_end_matches('/')
                ))
                .json(&request)
                .send()
                .await
        }
    })
    .await?;
    Ok(response
        .error_for_status()?
        .json::<ReceiveWhatsAppMessageResponse>()
        .await?)
}

async fn stream_session_events(
    client: &reqwest::Client,
    base_url: &str,
    session_id: &str,
    raw: bool,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let response = send_with_retry(base_url, || async {
        client
            .get(format!("{base_url}/sessions/{session_id}/events"))
            .send()
            .await?
            .error_for_status()
    })
    .await?;
    let mut stream = response.bytes_stream();
    let mut buffer = String::new();
    let theme = CliTheme::detect();
    let mut render_state = SessionEventRenderState::default();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        buffer.push_str(&String::from_utf8_lossy(&chunk));

        while let Some((frame, rest)) = split_sse_frame(&buffer) {
            let frame = frame.to_owned();
            let rest = rest.to_owned();
            buffer = rest;
            if let Some(data) = parse_sse_data(&frame) {
                if raw {
                    println!("{data}");
                } else if let Ok(event) = serde_json::from_str::<SessionEvent>(data) {
                    if let Some(lines) =
                        render_session_event_with_theme(&theme, &mut render_state, &event)
                    {
                        for line in lines {
                            println!("{line}");
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

async fn send_with_retry<T, F, Fut>(
    base_url: &str,
    mut operation: F,
) -> Result<T, Box<dyn std::error::Error + Send + Sync>>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, reqwest::Error>>,
{
    let mut backoff = Duration::from_millis(INITIAL_BACKOFF_MS);
    let mut last_error = None;

    for attempt in 0..=CONNECT_RETRY_COUNT {
        match operation().await {
            Ok(value) => return Ok(value),
            Err(error) if should_retry_request(&error) && attempt < CONNECT_RETRY_COUNT => {
                eprintln!(
                    "status: control-plane server unavailable at {base_url}; retrying in {}ms ({}/{})",
                    backoff.as_millis(),
                    attempt + 1,
                    CONNECT_RETRY_COUNT
                );
                last_error = Some(error);
                tokio::time::sleep(backoff).await;
                backoff = backoff.saturating_mul(2);
            }
            Err(error) => {
                let error: Box<dyn std::error::Error + Send + Sync> =
                    if should_retry_request(&error) {
                        format!(
                        "control-plane server unavailable at {base_url} after {} retries: {error}",
                        CONNECT_RETRY_COUNT
                    )
                    .into()
                    } else {
                        error.into()
                    };
                return Err(error);
            }
        }
    }

    Err(format!(
        "control-plane server unavailable at {base_url} after {} retries: {}",
        CONNECT_RETRY_COUNT,
        last_error
            .map(|error| error.to_string())
            .unwrap_or_else(|| "unknown error".to_owned())
    )
    .into())
}

fn should_retry_request(error: &reqwest::Error) -> bool {
    error.is_connect()
}

fn split_sse_frame(buffer: &str) -> Option<(&str, &str)> {
    buffer
        .find("\n\n")
        .map(|index| (&buffer[..index], &buffer[index + 2..]))
}

fn parse_sse_data(frame: &str) -> Option<&str> {
    frame.lines().find_map(|line| line.strip_prefix("data: "))
}

#[derive(Default)]
struct SessionEventRenderState {
    turn_open: bool,
}

fn render_session_event_with_theme(
    theme: &CliTheme,
    state: &mut SessionEventRenderState,
    event: &SessionEvent,
) -> Option<Vec<String>> {
    match event {
        SessionEvent::TurnQueued { .. } => Some(vec![render_notice(
            theme,
            NoticeTone::Muted,
            "turn",
            "queued",
        )]),
        SessionEvent::TurnStarted { .. } => {
            state.turn_open = true;
            Some(vec![render_group_header(theme, "Live Turn", "executing")])
        }
        SessionEvent::TurnCompleted { summary, .. } => {
            state.turn_open = false;
            Some(vec![render_turn_summary(
                theme,
                APP_NAME,
                &summary.final_text,
                Some(summary),
            )])
        }
        SessionEvent::ApprovalRequested { approval, .. } => {
            let mut lines = Vec::new();
            if state.turn_open {
                lines.push(render_timeline_event(
                    theme,
                    NoticeTone::Warning,
                    "approval",
                    "decision required",
                ));
            }
            lines.push(render_card(
                theme,
                CardTone::Warning,
                "Approval",
                Some(&format!("{} pending", approval.approval_id)),
                &[approval.prompt.clone()],
            ));
            Some(lines)
        }
        SessionEvent::ApprovalResolved { approval, .. } => {
            Some(vec![render_approval_resolution(theme, approval)])
        }
        SessionEvent::RuntimeEvent { event, .. } => {
            render_runtime_event_with_theme(theme, state.turn_open, event).map(|line| vec![line])
        }
        SessionEvent::ChannelDeliveryAttempted { status, .. } => Some(vec![if state.turn_open {
            render_timeline_event(
                theme,
                if matches!(status, agent_controlplane::DeliveryStatus::Failed) {
                    NoticeTone::Error
                } else {
                    NoticeTone::Success
                },
                "delivery",
                &format!("{}", format_delivery_status(*status)),
            )
        } else {
            render_notice(
                theme,
                if matches!(status, agent_controlplane::DeliveryStatus::Failed) {
                    NoticeTone::Error
                } else {
                    NoticeTone::Success
                },
                "delivery",
                &format!("{}", format_delivery_status(*status)),
            )
        }]),
        SessionEvent::SessionCreated { .. } | SessionEvent::SessionMessageReceived { .. } => None,
    }
}

fn render_runtime_event_with_theme(
    theme: &CliTheme,
    grouped: bool,
    event: &RuntimeEvent,
) -> Option<String> {
    match event {
        RuntimeEvent::HandoffToSubagent { subagent_type, .. } => Some(render_event_line(
            theme,
            grouped,
            NoticeTone::Info,
            "handoff",
            &format!("main agent -> {subagent_type}"),
        )),
        RuntimeEvent::McpCalled {
            server_name,
            tool_name,
            ..
        } => Some(render_event_line(
            theme,
            grouped,
            NoticeTone::Info,
            "mcp",
            &format!("calling {server_name}/{tool_name}"),
        )),
        RuntimeEvent::McpResponded {
            server_name,
            tool_name,
            was_error,
            ..
        } => Some(render_event_line(
            theme,
            grouped,
            if *was_error {
                NoticeTone::Error
            } else {
                NoticeTone::Success
            },
            "mcp",
            &format!(
                "{} {server_name}/{tool_name}",
                if *was_error { "error" } else { "done" }
            ),
        )),
        RuntimeEvent::TurnEnded { termination, .. } => Some(render_event_line(
            theme,
            grouped,
            NoticeTone::Muted,
            "status",
            &format!("turn ended ({})", format_termination(termination)),
        )),
        RuntimeEvent::AnswerRenderStarted { .. } => Some(render_event_line(
            theme,
            grouped,
            NoticeTone::Info,
            "answer",
            "drafting final answer",
        )),
        RuntimeEvent::AnswerRenderFailed { error, .. } => Some(render_event_line(
            theme,
            grouped,
            NoticeTone::Error,
            "answer",
            &format!("render failed ({error})"),
        )),
        RuntimeEvent::ModelCalled { .. } => Some(render_event_line(
            theme,
            grouped,
            NoticeTone::Info,
            "model",
            "querying model",
        )),
        RuntimeEvent::ModelResponded { latency, .. } => Some(render_event_line(
            theme,
            grouped,
            NoticeTone::Success,
            "model",
            &format!("responded in {}", format_elapsed_ms(latency.as_millis())),
        )),
        RuntimeEvent::HandoffToMainAgent {
            subagent_type,
            status,
            ..
        } => Some(render_event_line(
            theme,
            grouped,
            NoticeTone::Success,
            "handoff",
            &format!("{subagent_type} -> main agent ({status})"),
        )),
        RuntimeEvent::AnswerTextDelta { .. } | RuntimeEvent::AnswerRenderCompleted { .. } => None,
        _ => None,
    }
}

fn print_chat_response(theme: &CliTheme, response: &SendSessionMessageResponse) {
    if let Some(outbound) = response.result.outbound.first() {
        if let Some(summary) = response.result.session.last_turn.as_ref() {
            println!(
                "{}",
                render_turn_summary(theme, APP_NAME, &outbound.text, Some(summary))
            );
        } else {
            println!(
                "{}",
                render_turn_summary(theme, APP_NAME, &outbound.text, None)
            );
        }
    } else {
        println!(
            "{}",
            render_notice(theme, NoticeTone::Muted, "assistant", "no reply")
        );
    }
}

fn print_user_message(theme: &CliTheme, text: &str) {
    println!("{}", render_user_message(theme, text));
}

fn print_status(theme: &CliTheme, session: &SessionRecord) {
    let mut lines = vec![
        format_kv(theme, "Session", &session.session_id.to_string()),
        format_kv(theme, "Status", &format_status(session.status)),
    ];
    if let Some(last_turn) = &session.last_turn {
        lines.push(format_kv(theme, "Turn", &last_turn.turn_number.to_string()));
        lines.push(format_kv(
            theme,
            "Last Result",
            &format_termination(&last_turn.termination),
        ));
        lines.push(format_kv(theme, "Model", &last_turn.model_name));
        lines.push(format_kv(
            theme,
            "Duration",
            &format_elapsed_ms(u128::from(last_turn.elapsed_ms)),
        ));
        lines.push(format_kv(
            theme,
            "Tokens",
            &last_turn.usage.total_tokens.to_string(),
        ));
    } else {
        lines.push(format_kv(theme, "Turn", "0"));
    }
    println!(
        "{}",
        render_card(theme, CardTone::Status, "Session Status", None, &lines)
    );
}

fn print_whatsapp_status(theme: &CliTheme, status: &WhatsAppGatewayStatus) {
    let allow_from = if status.allow_from.is_empty() {
        "none".to_owned()
    } else {
        status.allow_from.join(", ")
    };
    let lines = vec![
        format_kv(theme, "Account", &status.account_id),
        format_kv(theme, "State", &format!("{:?}", status.connection_state).to_lowercase()),
        format_kv(theme, "DM Policy", &format!("{:?}", status.dm_policy).to_lowercase()),
        format_kv(theme, "Allow From", &allow_from),
        format_kv(
            theme,
            "Queued Outbound",
            &status.pending_outbound.to_string(),
        ),
        format_kv(
            theme,
            "Pending Login",
            status.active_login_session_id.as_deref().unwrap_or("none"),
        ),
        format_kv(
            theme,
            "Last Error",
            status.last_error.as_deref().unwrap_or("none"),
        ),
    ];
    println!(
        "{}",
        render_card(theme, CardTone::Status, "WhatsApp Gateway", None, &lines)
    );
}

fn print_whatsapp_login(theme: &CliTheme, response: &StartWhatsAppLoginResponse) {
    println!(
        "{}",
        render_card(
            theme,
            CardTone::Status,
            "WhatsApp Login",
            Some(&response.account_id),
            &[
                format!("Login Session: {}", response.login_session_id),
                format!("QR: {}", response.qr_code),
                "Complete the login with `whatsapp complete-login <id>` once paired.".to_owned(),
            ],
        )
    );
    if response.qr_code != "already connected" {
        println!();
        println!("{}", response.qr_code);
    }
}

fn print_whatsapp_receive(theme: &CliTheme, response: &ReceiveWhatsAppMessageResponse) {
    println!(
        "{}",
        render_card(
            theme,
            CardTone::Assistant,
            "WhatsApp Dispatch",
            Some(&response.session.session_id.to_string()),
            &[
                format!("Queued outbound: {}", response.queued_outbound),
                format!("Duplicate ignored: {}", response.was_duplicate),
                format!("Session status: {}", format_status(response.session.status)),
            ],
        )
    );
}

fn print_repl_help(theme: &CliTheme) {
    println!(
        "{}",
        render_card(
            theme,
            CardTone::Status,
            "Commands",
            None,
            &[
                "Chat".to_owned(),
                "Type any message to start a turn".to_owned(),
                "/help                         Show this help".to_owned(),
                "/new                          Start a new session".to_owned(),
                "/exit | \\exit                Leave the CLI".to_owned(),
                String::new(),
                "Session".to_owned(),
                "/session                      Print the current session id".to_owned(),
                "/status                       Show compact session status".to_owned(),
                "/watch                        Stream detailed live session events".to_owned(),
                String::new(),
                "Approvals".to_owned(),
                "/approve <id> <approve|reject> Resolve a pending approval".to_owned(),
                String::new(),
                "MCP".to_owned(),
                "/mcp                          Manage MCP registry entries".to_owned(),
            ],
        )
    );
}

async fn handle_mcp_command(theme: &CliTheme, command: McpCommand) -> CliResult<()> {
    let registry_path = resolve_registry_path();
    match command {
        McpCommand::Help => print_mcp_help(theme, &registry_path),
        McpCommand::List => print_mcp_list(theme, &load_registry(&registry_path)?, &registry_path),
        McpCommand::Show(name) => {
            let registry = load_registry(&registry_path)?;
            let server = registry.get(&name)?;
            print_mcp_show(theme, server, &registry_path)?;
            print_mcp_metadata_summary(theme, &name)?;
        }
        McpCommand::Add => {
            ensure_interactive_mcp(theme, "add")?;
            let mut registry = load_registry(&registry_path)?;
            let server = prompt_new_mcp_server(theme, &registry)?;
            refresh_mcp_metadata(&server).await?;
            registry.upsert_server(server.clone());
            registry.save_to_path(&registry_path)?;
            print_mcp_saved_notice(theme, "saved", &server.name, &registry_path);
        }
        McpCommand::Edit(name) => {
            ensure_interactive_mcp(theme, "edit")?;
            let mut registry = load_registry(&registry_path)?;
            let existing = registry.get(&name)?.clone();
            let updated = prompt_edit_mcp_server(theme, &existing)?;
            refresh_mcp_metadata(&updated).await?;
            registry.upsert_server(updated.clone());
            registry.save_to_path(&registry_path)?;
            print_mcp_saved_notice(theme, "updated", &updated.name, &registry_path);
        }
        McpCommand::Remove(name) => {
            ensure_interactive_mcp(theme, "remove")?;
            let mut registry = load_registry(&registry_path)?;
            let existing = registry.get(&name)?.clone();
            let confirmation = prompt_line(
                theme,
                &format!("Type REMOVE to delete `{}`", existing.name),
                None,
            )?;
            if confirmation != "REMOVE" {
                println!(
                    "{}",
                    render_notice(theme, NoticeTone::Muted, "mcp", "remove cancelled")
                );
                return Ok(());
            }
            delete_mcp_metadata(&existing.name)?;
            registry.remove_server(&name)?;
            registry.save_to_path(&registry_path)?;
            print_mcp_saved_notice(theme, "removed", &existing.name, &registry_path);
        }
    }

    Ok(())
}

fn resolve_registry_path() -> PathBuf {
    env::var("MCP_REGISTRY_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("config/mcp_servers.json"))
}

fn load_registry(path: &PathBuf) -> CliResult<McpRegistry> {
    if path.exists() {
        Ok(McpRegistry::load_from_path(path)?)
    } else {
        Ok(McpRegistry::default())
    }
}

fn ensure_interactive_mcp(theme: &CliTheme, action: &str) -> CliResult<()> {
    if theme.interactive {
        return Ok(());
    }

    Err(format!("/mcp {action} requires an interactive terminal").into())
}

fn print_mcp_help(theme: &CliTheme, registry_path: &PathBuf) {
    println!(
        "{}",
        render_card(
            theme,
            CardTone::Status,
            "MCP Commands",
            None,
            &[
                format!("Registry: {}", registry_path.display()),
                "/mcp list                     List configured MCP servers".to_owned(),
                "/mcp show <name>              Show one MCP entry".to_owned(),
                "/mcp add                      Add a new MCP entry".to_owned(),
                "/mcp edit <name>              Edit an existing MCP entry".to_owned(),
                "/mcp remove <name>            Remove an MCP entry".to_owned(),
            ],
        )
    );
}

fn print_mcp_list(theme: &CliTheme, registry: &McpRegistry, registry_path: &PathBuf) {
    let mut lines = vec![format!("Registry: {}", registry_path.display())];
    if registry.servers.is_empty() {
        lines.push("No MCP servers configured".to_owned());
    } else {
        let mut entries = registry
            .servers
            .iter()
            .map(|server| {
                format!(
                    "{} ({})",
                    server.name,
                    transport_label(&server.resolved_transport())
                )
            })
            .collect::<Vec<_>>();
        entries.sort();
        lines.extend(entries);
    }

    println!(
        "{}",
        render_card(theme, CardTone::Status, "MCP Registry", None, &lines)
    );
}

fn print_mcp_show(
    theme: &CliTheme,
    server: &McpServerConfig,
    registry_path: &PathBuf,
) -> CliResult<()> {
    let body = serde_json::to_string_pretty(&normalized_server(server))?;
    println!(
        "{}",
        render_card(
            theme,
            CardTone::Status,
            &format!("MCP {}", server.name),
            None,
            &[format!("Registry: {}", registry_path.display()), body],
        )
    );
    Ok(())
}

fn print_mcp_metadata_summary(theme: &CliTheme, name: &str) -> CliResult<()> {
    let paths = artifact_paths(PathBuf::from(DEFAULT_METADATA_DIR).as_path(), name);
    let minimal = load_minimal_catalog(&paths.minimal_path)?;
    let full = load_full_catalog(&paths.full_path)?;
    let lines = vec![
        format!("Minimal: {}", paths.minimal_path.display()),
        format!("Full: {}", paths.full_path.display()),
        format!("Server version: {}", minimal.server.version),
        format!("Tools: {}", minimal.tools.len()),
        format!("Resources: {}", minimal.resources.len()),
        format!("Full extensions: {}", full.extensions),
    ];
    println!(
        "{}",
        render_card(theme, CardTone::Status, "MCP Metadata", None, &lines)
    );
    Ok(())
}

fn print_mcp_saved_notice(theme: &CliTheme, action: &str, name: &str, registry_path: &PathBuf) {
    println!(
        "{}",
        render_notice(
            theme,
            NoticeTone::Success,
            "mcp",
            &format!("{action} `{name}` in {}", registry_path.display()),
        )
    );
    println!(
        "{}",
        render_notice(
            theme,
            NoticeTone::Muted,
            "mcp",
            "changes apply to newly created sessions only",
        )
    );
}

async fn refresh_mcp_metadata(server: &McpServerConfig) -> CliResult<()> {
    let mut connection = McpClient::connect(server).await?;
    let result: CliResult<()> = async {
        let initialize = connection
            .initialize(ClientInfo::new(
                "agent-channel-cli",
                env!("CARGO_PKG_VERSION"),
            ))
            .await?;
        connection.notify_initialized().await?;
        let mut tools_supported = server_supports_capability(&initialize, "tools");
        let tools = if tools_supported {
            match connection.list_tools().await {
                Ok(result) => result.tools,
                Err(McpClientError::Rpc { code: -32601, .. }) => {
                    tools_supported = false;
                    Vec::new()
                }
                Err(error) => return Err(error.into()),
            }
        } else {
            Vec::new()
        };
        let mut resources_supported = server_supports_capability(&initialize, "resources");
        let resources = if resources_supported {
            match connection.list_resources().await {
                Ok(result) => result.resources,
                Err(McpClientError::Rpc { code: -32601, .. }) => {
                    resources_supported = false;
                    Vec::new()
                }
                Err(error) => return Err(error.into()),
            }
        } else {
            Vec::new()
        };
        let (minimal, full) = build_metadata_catalogs(
            server,
            &initialize,
            tools_supported,
            &tools,
            resources_supported,
            &resources,
        );
        let paths = artifact_paths(PathBuf::from(DEFAULT_METADATA_DIR).as_path(), &server.name);
        write_catalogs(&paths, &minimal, &full)?;
        Ok(())
    }
    .await;

    let close_result = connection.close().await;
    let result = match (result, close_result) {
        (Err(error), _) => Err(error),
        (Ok(()), Ok(())) => Ok(()),
        (Ok(()), Err(error)) => Err(error.into()),
    };

    if let Err(error) = result {
        if uses_mcp_remote_wrapper(server) {
            return Err(format_mcp_remote_refresh_error(server, &error.to_string()).into());
        }
        return Err(error);
    }

    Ok(())
}

fn server_supports_capability(initialize: &McpInitializeResult, family: &str) -> bool {
    initialize
        .capabilities
        .get(family)
        .is_some_and(|value| !value.is_null())
}

fn delete_mcp_metadata(name: &str) -> CliResult<()> {
    let paths = artifact_paths(PathBuf::from(DEFAULT_METADATA_DIR).as_path(), name);
    if paths.minimal_path.exists() {
        std::fs::remove_file(&paths.minimal_path)?;
    }
    if paths.full_path.exists() {
        std::fs::remove_file(&paths.full_path)?;
    }
    Ok(())
}

fn uses_mcp_remote_wrapper(server: &McpServerConfig) -> bool {
    match server.resolved_transport() {
        McpTransportConfig::Stdio { command, args, .. } => is_mcp_remote_stdio(&command, &args),
        McpTransportConfig::StreamableHttp { .. } => false,
    }
}

fn is_mcp_remote_stdio(command: &str, args: &[String]) -> bool {
    if command.trim().is_empty() {
        return false;
    }

    args.iter().any(|arg| {
        arg == MCP_REMOTE_PACKAGE_PREFIX
            || arg.starts_with(&format!("{MCP_REMOTE_PACKAGE_PREFIX}@"))
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct McpRemoteDefaults {
    command: String,
    package: String,
    include_npx_yes: bool,
    remote_url: Option<String>,
    extra_args: Vec<String>,
    env: std::collections::HashMap<String, String>,
}

fn mcp_remote_defaults(existing: Option<&McpTransportConfig>) -> McpRemoteDefaults {
    let Some(McpTransportConfig::Stdio { command, args, env }) = existing else {
        return McpRemoteDefaults {
            command: "npx".to_owned(),
            package: MCP_REMOTE_PACKAGE_PREFIX.to_owned(),
            include_npx_yes: true,
            remote_url: None,
            extra_args: Vec::new(),
            env: std::collections::HashMap::new(),
        };
    };

    if !is_mcp_remote_stdio(command, args) {
        return McpRemoteDefaults {
            command: command.clone(),
            package: MCP_REMOTE_PACKAGE_PREFIX.to_owned(),
            include_npx_yes: command == "npx",
            remote_url: None,
            extra_args: Vec::new(),
            env: env.clone(),
        };
    }

    let mut index = 0usize;
    let include_npx_yes = command == "npx" && args.first().is_some_and(|value| value == "-y");
    if include_npx_yes {
        index += 1;
    }

    let package = args
        .get(index)
        .cloned()
        .unwrap_or_else(|| MCP_REMOTE_PACKAGE_PREFIX.to_owned());
    let remote_url = args.get(index + 1).cloned();
    let extra_args = args.iter().skip(index + 2).cloned().collect();

    McpRemoteDefaults {
        command: command.clone(),
        package,
        include_npx_yes,
        remote_url,
        extra_args,
        env: env.clone(),
    }
}

fn mcp_remote_url(server: &McpServerConfig) -> Option<String> {
    let McpTransportConfig::Stdio { command, args, .. } = server.resolved_transport() else {
        return None;
    };
    if !is_mcp_remote_stdio(&command, &args) {
        return None;
    }
    mcp_remote_defaults(Some(&McpTransportConfig::Stdio {
        command,
        args,
        env: std::collections::HashMap::new(),
    }))
    .remote_url
}

fn format_mcp_remote_refresh_error(server: &McpServerConfig, error: &str) -> String {
    let mut lines = vec![
        format!(
            "failed to refresh metadata for `{}` via mcp-remote: {error}",
            server.name
        ),
        "mcp-remote owns the OAuth flow; Arka is only hosting it as a stdio MCP server.".to_owned(),
        "Wrapper troubleshooting:".to_owned(),
    ];

    if matches!(
        server.resolved_transport(),
        McpTransportConfig::Stdio { command, ref args, .. }
            if command == "npx" && !args.iter().any(|arg| arg == "-y")
    ) {
        lines.push(
            "- Add `-y` to the npx args so package-install prompts cannot block startup."
                .to_owned(),
        );
    }
    lines.push(
        "- Add `--debug` to the mcp-remote args and inspect `~/.mcp-auth/*_debug.log`.".to_owned(),
    );
    lines.push(
        "- Clear stale wrapper state with `rm -rf ~/.mcp-auth` if auth looks wedged.".to_owned(),
    );
    if let Some(remote_url) = mcp_remote_url(server) {
        lines.push(format!(
            "- Smoke-test outside Arka with `npx -p mcp-remote@latest mcp-remote-client {remote_url}`."
        ));
    }

    lines.join("\n")
}

fn build_metadata_catalogs(
    server: &McpServerConfig,
    initialize: &McpInitializeResult,
    tools_supported: bool,
    tools: &[McpToolDescriptor],
    resources_supported: bool,
    resources: &[McpResourceDescriptor],
) -> (McpMinimalCatalog, McpFullCatalog) {
    let server_metadata = McpServerMetadata {
        logical_name: server.name.clone(),
        protocol_name: initialize.server_info.name.clone(),
        title: initialize.server_info.title.clone(),
        version: initialize.server_info.version.clone(),
        description: server.description.clone(),
        instructions_summary: initialize.instructions.clone(),
    };
    let capability_families = McpCapabilityFamilies {
        tools: McpCapabilityFamilySummary {
            supported: tools_supported,
            count: tools.len(),
        },
        resources: McpCapabilityFamilySummary {
            supported: resources_supported,
            count: resources.len(),
        },
    };
    let minimal_tools = tools
        .iter()
        .map(|tool| MinimalToolMetadata {
            name: tool.name.to_string(),
            title: tool.title.clone(),
            description: tool.description.clone(),
        })
        .collect::<Vec<_>>();
    let full_tools = tools
        .iter()
        .map(|tool| FullToolMetadata {
            name: tool.name.to_string(),
            title: tool.title.clone(),
            description: tool.description.clone(),
            input_schema: tool.input_schema.clone(),
        })
        .collect::<Vec<_>>();
    let minimal_resources = resources
        .iter()
        .map(|resource| MinimalResourceMetadata {
            uri: resource.uri.clone(),
            name: resource.name.clone(),
            title: resource.title.clone(),
            description: resource.description.clone(),
            mime_type: resource.mime_type.clone(),
        })
        .collect::<Vec<_>>();
    let full_resources = resources
        .iter()
        .map(|resource| FullResourceMetadata {
            uri: resource.uri.clone(),
            name: resource.name.clone(),
            title: resource.title.clone(),
            description: resource.description.clone(),
            mime_type: resource.mime_type.clone(),
            annotations: resource.annotations.clone(),
        })
        .collect::<Vec<_>>();

    (
        McpMinimalCatalog {
            schema_version: CURRENT_SCHEMA_VERSION,
            server: server_metadata.clone(),
            capability_families: capability_families.clone(),
            tools: minimal_tools,
            resources: minimal_resources,
        },
        McpFullCatalog {
            schema_version: CURRENT_SCHEMA_VERSION,
            server: server_metadata,
            capability_families,
            tools: full_tools,
            resources: full_resources,
            extensions: serde_json::json!({}),
        },
    )
}

fn prompt_new_mcp_server(theme: &CliTheme, registry: &McpRegistry) -> CliResult<McpServerConfig> {
    let name = loop {
        let candidate = prompt_line(theme, "MCP name", None)?;
        if candidate.trim().is_empty() {
            println!(
                "{}",
                render_notice(theme, NoticeTone::Error, "mcp", "name cannot be blank")
            );
            continue;
        }
        if registry
            .servers
            .iter()
            .any(|server| server.name == candidate)
        {
            println!(
                "{}",
                render_notice(theme, NoticeTone::Error, "mcp", "name already exists")
            );
            continue;
        }
        break candidate;
    };

    build_mcp_server(theme, name, None)
}

fn prompt_edit_mcp_server(
    theme: &CliTheme,
    existing: &McpServerConfig,
) -> CliResult<McpServerConfig> {
    build_mcp_server(theme, existing.name.clone(), Some(existing))
}

fn build_mcp_server(
    theme: &CliTheme,
    name: String,
    existing: Option<&McpServerConfig>,
) -> CliResult<McpServerConfig> {
    let current_transport = existing.map(McpServerConfig::resolved_transport);
    let transport = match choose_transport_kind(theme, current_transport.as_ref())? {
        TransportKind::Stdio => build_stdio_transport(theme, current_transport.as_ref())?,
        TransportKind::StreamableHttp => {
            let current_url = current_transport
                .as_ref()
                .and_then(|transport| match transport {
                    McpTransportConfig::StreamableHttp { url, .. } => Some(url.as_str()),
                    _ => None,
                });
            let current_headers =
                current_transport
                    .as_ref()
                    .and_then(|transport| match transport {
                        McpTransportConfig::StreamableHttp { headers, .. } => Some(headers),
                        _ => None,
                    });
            McpTransportConfig::StreamableHttp {
                url: prompt_required_line(theme, "URL", current_url)?,
                headers: prompt_key_value_map(theme, "Headers", current_headers)?,
            }
        }
    };

    Ok(McpServerConfig {
        name,
        transport: Some(transport),
        command: String::new(),
        args: Vec::new(),
        env: std::collections::HashMap::new(),
        description: prompt_optional_line(
            theme,
            "Description",
            existing.and_then(|server| server.description.as_deref()),
        )?,
    })
}

fn build_stdio_transport(
    theme: &CliTheme,
    current_transport: Option<&McpTransportConfig>,
) -> CliResult<McpTransportConfig> {
    let default_mode = match current_transport {
        Some(McpTransportConfig::Stdio { command, args, .. })
            if is_mcp_remote_stdio(command, args) =>
        {
            "mcp_remote"
        }
        _ => "plain",
    };
    let mode = prompt_stdio_mode(theme, default_mode)?;

    match mode.as_str() {
        "mcp_remote" => prompt_mcp_remote_transport(theme, current_transport),
        _ => prompt_plain_stdio_transport(theme, current_transport),
    }
}

fn prompt_stdio_mode(theme: &CliTheme, default: &str) -> CliResult<String> {
    loop {
        let answer = prompt_line(theme, "Stdio mode (plain/mcp_remote)", Some(default))?;
        match answer.as_str() {
            "plain" | "mcp_remote" => return Ok(answer),
            _ => println!(
                "{}",
                render_notice(
                    theme,
                    NoticeTone::Error,
                    "mcp",
                    "enter `plain` or `mcp_remote`",
                )
            ),
        }
    }
}

fn prompt_plain_stdio_transport(
    theme: &CliTheme,
    current_transport: Option<&McpTransportConfig>,
) -> CliResult<McpTransportConfig> {
    let current_command = current_transport.and_then(|transport| match transport {
        McpTransportConfig::Stdio { command, .. } => Some(command.as_str()),
        _ => None,
    });
    let current_args = current_transport.and_then(|transport| match transport {
        McpTransportConfig::Stdio { args, .. } => Some(args),
        _ => None,
    });
    let current_env = current_transport.and_then(|transport| match transport {
        McpTransportConfig::Stdio { env, .. } => Some(env),
        _ => None,
    });

    Ok(McpTransportConfig::Stdio {
        command: prompt_required_line(theme, "Command", current_command)?,
        args: prompt_string_list(theme, "Arguments", current_args)?,
        env: prompt_key_value_map(theme, "Environment", current_env)?,
    })
}

fn prompt_mcp_remote_transport(
    theme: &CliTheme,
    current_transport: Option<&McpTransportConfig>,
) -> CliResult<McpTransportConfig> {
    let defaults = mcp_remote_defaults(current_transport);
    println!(
        "{}",
        render_notice(
            theme,
            NoticeTone::Info,
            "mcp",
            "mcp-remote wrapper mode: Arka will save a stdio config and let mcp-remote handle OAuth.",
        )
    );
    println!(
        "{}",
        render_notice(
            theme,
            NoticeTone::Muted,
            "mcp",
            "Common extra args: `--debug`, `--resource`, `--auth-timeout`, `--host`, `--header`, `--static-oauth-client-metadata`, `--static-oauth-client-info`.",
        )
    );

    let command = prompt_required_line(theme, "Command", Some(&defaults.command))?;
    let package_label = if command == "npx" {
        "Package"
    } else {
        "Wrapper entrypoint"
    };
    let package = prompt_required_line(theme, package_label, Some(&defaults.package))?;
    let remote_url = prompt_required_line(theme, "Remote URL", defaults.remote_url.as_deref())?;
    let include_npx_yes = if command == "npx" {
        prompt_yes_no(
            theme,
            "Add `-y` so npx install prompts cannot block startup",
            defaults.include_npx_yes,
        )?
    } else {
        false
    };
    let extra_args = prompt_string_list(theme, "Extra args", Some(&defaults.extra_args))?;
    let env = prompt_key_value_map(theme, "Environment", Some(&defaults.env))?;

    let mut args = Vec::new();
    if command == "npx" && include_npx_yes {
        args.push("-y".to_owned());
    }
    args.push(package);
    args.push(remote_url);
    args.extend(extra_args);

    Ok(McpTransportConfig::Stdio { command, args, env })
}

fn choose_transport_kind(
    theme: &CliTheme,
    current: Option<&McpTransportConfig>,
) -> CliResult<TransportKind> {
    let default = match current {
        Some(McpTransportConfig::StreamableHttp { .. }) => "streamable_http",
        _ => "stdio",
    };
    loop {
        let answer = prompt_line(
            theme,
            "Transport type (stdio/streamable_http)",
            Some(default),
        )?;
        match answer.as_str() {
            "stdio" => return Ok(TransportKind::Stdio),
            "streamable_http" | "http" => return Ok(TransportKind::StreamableHttp),
            _ => println!(
                "{}",
                render_notice(
                    theme,
                    NoticeTone::Error,
                    "mcp",
                    "enter `stdio` or `streamable_http`",
                )
            ),
        }
    }
}

fn prompt_required_line(theme: &CliTheme, label: &str, current: Option<&str>) -> CliResult<String> {
    loop {
        let answer = prompt_line(theme, label, current)?;
        if !answer.trim().is_empty() {
            return Ok(answer);
        }
        println!(
            "{}",
            render_notice(
                theme,
                NoticeTone::Error,
                "mcp",
                &format!("{label} cannot be blank")
            )
        );
    }
}

fn prompt_optional_line(
    theme: &CliTheme,
    label: &str,
    current: Option<&str>,
) -> CliResult<Option<String>> {
    let prompt = match current {
        Some(value) => format!("{label} [{value}] (blank keeps, `-` clears)"),
        None => label.to_owned(),
    };
    let answer = prompt_line(theme, &prompt, None)?;
    if answer == "-" {
        return Ok(None);
    }
    if answer.is_empty() {
        return Ok(current.map(str::to_owned));
    }
    Ok(Some(answer))
}

fn prompt_yes_no(theme: &CliTheme, label: &str, default: bool) -> CliResult<bool> {
    let default_text = if default { "y" } else { "n" };
    loop {
        let answer = prompt_line(theme, &format!("{label} (y/n)"), Some(default_text))?;
        match answer.as_str() {
            "y" | "Y" => return Ok(true),
            "n" | "N" => return Ok(false),
            _ => println!(
                "{}",
                render_notice(theme, NoticeTone::Error, "mcp", "enter `y` or `n`")
            ),
        }
    }
}

fn prompt_string_list(
    theme: &CliTheme,
    label: &str,
    current: Option<&Vec<String>>,
) -> CliResult<Vec<String>> {
    if let Some(current) = current {
        if !current.is_empty() {
            let keep = prompt_line(
                theme,
                &format!("{label}: keep current values? (y/n)"),
                Some("y"),
            )?;
            if keep.eq_ignore_ascii_case("y") {
                return Ok(current.clone());
            }
        }
    }

    let mut values = Vec::new();
    loop {
        let entry = prompt_line(
            theme,
            &format!("{label} item {} (blank to finish)", values.len() + 1),
            None,
        )?;
        if entry.is_empty() {
            break;
        }
        values.push(entry);
    }
    Ok(values)
}

fn prompt_key_value_map(
    theme: &CliTheme,
    label: &str,
    current: Option<&std::collections::HashMap<String, String>>,
) -> CliResult<std::collections::HashMap<String, String>> {
    if let Some(current) = current {
        if !current.is_empty() {
            let keep = prompt_line(
                theme,
                &format!("{label}: keep current values? (y/n)"),
                Some("y"),
            )?;
            if keep.eq_ignore_ascii_case("y") {
                return Ok(current.clone());
            }
        }
    }

    let mut values = std::collections::HashMap::new();
    loop {
        let key = prompt_line(
            theme,
            &format!("{label} key {} (blank to finish)", values.len() + 1),
            None,
        )?;
        if key.is_empty() {
            break;
        }
        let value = prompt_line(theme, &format!("{label} value for `{key}`"), None)?;
        values.insert(key, value);
    }
    Ok(values)
}

fn prompt_line(theme: &CliTheme, label: &str, default: Option<&str>) -> CliResult<String> {
    let prompt = match default {
        Some(default) if !default.is_empty() => format!("{label} [{default}]: "),
        _ => format!("{label}: "),
    };

    let input = if io::stdin().is_terminal() && io::stdout().is_terminal() {
        let mut editor = DefaultEditor::new()
            .map_err(|error| format!("failed to initialize prompt input: {error}"))?;
        match editor.readline(&theme.accent_soft(&prompt)) {
            Ok(line) => line.trim().to_owned(),
            Err(ReadlineError::Interrupted | ReadlineError::Eof) => {
                return Err("interactive input cancelled".into());
            }
            Err(error) => return Err(format!("interactive input failed: {error}").into()),
        }
    } else {
        print!("{}", theme.accent_soft(&prompt));
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        input.trim().to_owned()
    };

    if input.is_empty() {
        Ok(default.unwrap_or_default().to_owned())
    } else {
        Ok(input)
    }
}

fn normalized_server(server: &McpServerConfig) -> McpServerConfig {
    McpServerConfig {
        name: server.name.clone(),
        transport: Some(server.resolved_transport()),
        command: String::new(),
        args: Vec::new(),
        env: std::collections::HashMap::new(),
        description: server.description.clone(),
    }
}

fn transport_label(transport: &McpTransportConfig) -> &'static str {
    match transport {
        McpTransportConfig::Stdio { .. } => "stdio",
        McpTransportConfig::StreamableHttp { .. } => "streamable_http",
    }
}

#[derive(Clone, Copy)]
enum TransportKind {
    Stdio,
    StreamableHttp,
}

#[derive(Clone, Copy)]
struct CliTheme {
    color: bool,
    interactive: bool,
    presentation: CliPresentation,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum VisualMode {
    Rich,
    Minimal,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MetadataDensity {
    Compact,
    Balanced,
    Verbose,
}

#[derive(Clone, Copy)]
struct CliPresentation {
    visual_mode: VisualMode,
    metadata_density: MetadataDensity,
    transcript_width_override: Option<usize>,
}

impl CliTheme {
    fn detect() -> Self {
        let interactive = io::stdout().is_terminal();
        let visual_mode = match env::var("ARKA_CLI_VISUAL_MODE") {
            Ok(value) if value.eq_ignore_ascii_case("minimal") => VisualMode::Minimal,
            _ if interactive => VisualMode::Rich,
            _ => VisualMode::Minimal,
        };
        let metadata_density = match env::var("ARKA_CLI_METADATA_DENSITY") {
            Ok(value) if value.eq_ignore_ascii_case("balanced") => MetadataDensity::Balanced,
            Ok(value) if value.eq_ignore_ascii_case("verbose") => MetadataDensity::Verbose,
            _ => MetadataDensity::Compact,
        };
        let transcript_width_override = env::var("ARKA_CLI_WIDTH")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|width| *width >= 48);
        Self {
            color: interactive && env::var_os("NO_COLOR").is_none(),
            interactive,
            presentation: CliPresentation {
                visual_mode,
                metadata_density,
                transcript_width_override,
            },
        }
    }

    #[cfg(test)]
    fn plain() -> Self {
        Self {
            color: false,
            interactive: false,
            presentation: CliPresentation {
                visual_mode: VisualMode::Minimal,
                metadata_density: MetadataDensity::Compact,
                transcript_width_override: None,
            },
        }
    }

    fn paint(&self, text: &str, code: &str) -> String {
        if self.color {
            format!("\u{1b}[{code}m{text}\u{1b}[0m")
        } else {
            text.to_owned()
        }
    }

    fn bold(&self, text: &str) -> String {
        self.paint(text, "1")
    }

    fn strong(&self, text: &str) -> String {
        self.paint(text, "1;38;5;255")
    }

    fn muted(&self, text: &str) -> String {
        self.paint(text, "38;5;244")
    }

    fn accent(&self, text: &str) -> String {
        self.paint(text, "1;38;5;81")
    }

    fn accent_soft(&self, text: &str) -> String {
        self.paint(text, "38;5;117")
    }

    fn success(&self, text: &str) -> String {
        self.paint(text, "1;38;5;114")
    }

    fn warning(&self, text: &str) -> String {
        self.paint(text, "1;38;5;221")
    }

    fn error(&self, text: &str) -> String {
        self.paint(text, "1;38;5;210")
    }

    fn interactive(&self) -> bool {
        self.interactive
    }

    fn metadata_density(&self) -> MetadataDensity {
        self.presentation.metadata_density
    }

    fn transcript_width(&self) -> usize {
        self.presentation
            .transcript_width_override
            .unwrap_or_else(|| default_card_width(84))
    }

    fn card_width(&self) -> usize {
        self.presentation
            .transcript_width_override
            .unwrap_or_else(|| default_card_width(82))
    }

    fn uses_rich_borders(&self) -> bool {
        self.presentation.visual_mode == VisualMode::Rich
    }
}

#[derive(Clone, Copy)]
enum CardTone {
    Assistant,
    User,
    Status,
    Warning,
}

#[derive(Clone, Copy)]
enum NoticeTone {
    Info,
    Success,
    Warning,
    Error,
    Muted,
}

struct TurnSpinner {
    handle: Option<tokio::task::JoinHandle<io::Result<()>>>,
    stop_tx: Option<oneshot::Sender<()>>,
}

impl TurnSpinner {
    fn start(theme: CliTheme, message: &str) -> io::Result<Self> {
        if !theme.interactive() {
            println!(
                "{}",
                render_notice(&theme, NoticeTone::Info, "status", &message.to_lowercase())
            );
            return Ok(Self {
                handle: None,
                stop_tx: None,
            });
        }

        render_progress_frame(&theme, TURN_SPINNER_FRAMES[0], message)?;

        let (stop_tx, mut stop_rx) = oneshot::channel();
        let message = message.to_owned();
        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(100));
            let mut frame_index = 1usize;

            loop {
                tokio::select! {
                    _ = &mut stop_rx => {
                        clear_current_line()?;
                        return Ok(());
                    }
                    _ = interval.tick() => {
                        render_progress_frame(
                            &theme,
                            TURN_SPINNER_FRAMES[frame_index % TURN_SPINNER_FRAMES.len()],
                            &message,
                        )?;
                        frame_index += 1;
                    }
                }
            }
        });

        Ok(Self {
            handle: Some(handle),
            stop_tx: Some(stop_tx),
        })
    }

    async fn stop(mut self) -> io::Result<()> {
        if let Some(stop_tx) = self.stop_tx.take() {
            let _ = stop_tx.send(());
        }
        if let Some(handle) = self.handle.take() {
            handle
                .await
                .map_err(|error| io::Error::other(format!("spinner task failed: {error}")))??;
        }
        Ok(())
    }
}

fn render_progress_frame(theme: &CliTheme, frame: &str, message: &str) -> io::Result<()> {
    print!("\r\x1b[2K{} {}", theme.accent(frame), theme.strong(message));
    io::stdout().flush()
}

fn clear_current_line() -> io::Result<()> {
    print!("\r\x1b[2K");
    io::stdout().flush()
}

fn clear_previous_line() -> io::Result<()> {
    print!("\x1b[1A\r\x1b[2K");
    io::stdout().flush()
}

fn render_turn_summary(
    theme: &CliTheme,
    title: &str,
    body: &str,
    summary: Option<&TurnRecordSummary>,
) -> String {
    let width = theme.transcript_width().saturating_sub(4);
    let meta = summary.map(|summary| match theme.metadata_density() {
        MetadataDensity::Compact | MetadataDensity::Balanced => {
            format!(
                "Turn {}  {}",
                summary.turn_number,
                format_turn_meta_line(summary)
            )
        }
        MetadataDensity::Verbose => format!(
            "Turn {}  {}  Completed: {:?}",
            summary.turn_number,
            format_turn_meta_line(summary),
            summary.completed_at
        ),
    });
    let body_lines = render_markdownish_text(theme, body, width);
    render_card(
        theme,
        CardTone::Assistant,
        title,
        meta.as_deref(),
        &body_lines,
    )
}

fn render_approval_resolution(theme: &CliTheme, approval: &ApprovalRequestRecord) -> String {
    let tone = match approval.state {
        ApprovalState::Approved => CardTone::Assistant,
        ApprovalState::Rejected => CardTone::Warning,
        ApprovalState::Pending | ApprovalState::Expired | ApprovalState::Cancelled => {
            CardTone::Status
        }
    };
    render_card(
        theme,
        tone,
        "Approval",
        Some(&approval.approval_id.to_string()),
        &[
            format!("State • {}", format_approval_state(approval.state)),
            approval.prompt.clone(),
        ],
    )
}

fn render_welcome_banner(theme: &CliTheme, session: &SessionRecord, base_url: &str) -> String {
    let lines = vec![
        APP_TAGLINE.to_owned(),
        format!("Session: {}", session.session_id),
        format!("API: {base_url}"),
        "Type a message or run /help".to_owned(),
        "Quick actions: /new  /status  /watch  /mcp  /exit".to_owned(),
    ];
    render_card(
        theme,
        CardTone::Status,
        APP_NAME,
        Some("Interactive CLI"),
        &lines,
    )
}

fn render_user_message(theme: &CliTheme, text: &str) -> String {
    let width = theme.transcript_width().saturating_sub(4);
    let lines = render_markdownish_text(theme, text, width);
    render_card(theme, CardTone::User, USER_LABEL, None, &lines)
}

fn render_card(
    theme: &CliTheme,
    tone: CardTone,
    title: &str,
    meta: Option<&str>,
    lines: &[String],
) -> String {
    let width = theme.card_width();
    let inner_width = width.saturating_sub(2);
    let (top_left, horizontal, top_right, vertical, bottom_left, bottom_right) =
        if theme.uses_rich_borders() {
            ("╭", "─", "╮", "│", "╰", "╯")
        } else {
            ("+", "-", "+", "|", "+", "+")
        };
    let title_plain = format!(" {title} ");
    let fill = horizontal.repeat(inner_width.saturating_sub(title_plain.chars().count()));
    let title_styled = match tone {
        CardTone::Assistant => theme.accent(title),
        CardTone::User => theme.strong(title),
        CardTone::Status => theme.bold(title),
        CardTone::Warning => theme.warning(title),
    };
    let title_styled = format!(" {} ", title_styled);

    let mut rendered = vec![format!(
        "{}{}{}{}",
        style_card_border(theme, tone, top_left),
        title_styled,
        style_card_border(theme, tone, &fill),
        style_card_border(theme, tone, top_right),
    )];
    if let Some(meta) = meta {
        for line in wrap_text(meta, inner_width.saturating_sub(2)) {
            rendered.push(format_card_line(
                theme,
                tone,
                vertical,
                &theme.muted(&line),
                line.chars().count(),
                inner_width,
            ));
        }
        if !lines.is_empty() {
            rendered.push(format_card_line(theme, tone, vertical, "", 0, inner_width));
        }
    }
    if lines.is_empty() {
        rendered.push(format_card_line(theme, tone, vertical, "", 0, inner_width));
    } else {
        for line in lines {
            let wrapped = wrap_text(line, inner_width.saturating_sub(2));
            for wrapped_line in wrapped {
                let formatted =
                    if wrapped_line.starts_with('/') || is_section_heading(&wrapped_line) {
                        wrapped_line.clone()
                    } else {
                        style_kv_line(theme, &wrapped_line)
                    };
                rendered.push(format_card_line(
                    theme,
                    tone,
                    vertical,
                    &formatted,
                    wrapped_line.chars().count(),
                    inner_width,
                ));
            }
        }
    }
    rendered.push(format!(
        "{}{}{}",
        style_card_border(theme, tone, bottom_left),
        style_card_border(theme, tone, &horizontal.repeat(inner_width)),
        style_card_border(theme, tone, bottom_right),
    ));
    rendered.join("\n")
}

fn render_group_header(theme: &CliTheme, title: &str, status: &str) -> String {
    format!(
        "{} {} {}",
        theme.accent("●"),
        theme.strong(title),
        theme.muted(status)
    )
}

fn render_event_line(
    theme: &CliTheme,
    grouped: bool,
    tone: NoticeTone,
    label: &str,
    message: &str,
) -> String {
    if grouped {
        render_timeline_event(theme, tone, label, message)
    } else {
        render_notice(theme, tone, label, message)
    }
}

fn render_timeline_event(theme: &CliTheme, tone: NoticeTone, label: &str, message: &str) -> String {
    format!(
        "  {} {} {}",
        theme.muted("•"),
        style_notice_label(theme, tone, label),
        message
    )
}

fn render_notice(theme: &CliTheme, tone: NoticeTone, label: &str, message: &str) -> String {
    format!("{} {}", style_notice_label(theme, tone, label), message)
}

fn style_notice_label(theme: &CliTheme, tone: NoticeTone, label: &str) -> String {
    let label = format!("[{label}]");
    match tone {
        NoticeTone::Info => theme.accent_soft(&label),
        NoticeTone::Success => theme.success(&label),
        NoticeTone::Warning => theme.warning(&label),
        NoticeTone::Error => theme.error(&label),
        NoticeTone::Muted => theme.muted(&label),
    }
}

fn format_kv(theme: &CliTheme, label: &str, value: &str) -> String {
    format!("{} {}", theme.bold(&format!("{label}:")), value)
}

fn format_status(status: SessionStatus) -> String {
    match status {
        SessionStatus::Idle => "Idle",
        SessionStatus::Running => "Running",
        SessionStatus::WaitingForApproval => "Waiting for approval",
        SessionStatus::Interrupted => "Interrupted",
        SessionStatus::Failed => "Failed",
        SessionStatus::Completed => "Completed",
    }
    .to_owned()
}

fn format_termination(termination: &TerminationReason) -> &'static str {
    match termination {
        TerminationReason::Final => "Final",
        TerminationReason::MaxStepsReached => "Max steps reached",
        TerminationReason::Timeout => "Timeout",
        TerminationReason::ValidationError => "Validation error",
        TerminationReason::RuntimeError => "Runtime error",
    }
}

fn format_approval_state(state: ApprovalState) -> &'static str {
    match state {
        ApprovalState::Pending => "Pending",
        ApprovalState::Approved => "Approved",
        ApprovalState::Rejected => "Rejected",
        ApprovalState::Expired => "Expired",
        ApprovalState::Cancelled => "Cancelled",
    }
}

fn format_delivery_status(status: agent_controlplane::DeliveryStatus) -> &'static str {
    match status {
        agent_controlplane::DeliveryStatus::Pending => "Pending",
        agent_controlplane::DeliveryStatus::Sent => "Sent",
        agent_controlplane::DeliveryStatus::Failed => "Failed",
    }
}

fn format_elapsed_ms(elapsed_ms: u128) -> String {
    if elapsed_ms < 1_000 {
        format!("{elapsed_ms}ms")
    } else if elapsed_ms < 10_000 {
        format!("{:.2}s", elapsed_ms as f64 / 1_000.0)
    } else {
        format!("{:.1}s", elapsed_ms as f64 / 1_000.0)
    }
}

fn default_card_width(fallback: usize) -> usize {
    let columns = env::var("COLUMNS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(fallback + 4);
    columns.saturating_sub(4).clamp(56, fallback)
}

fn format_count(value: u32) -> String {
    let digits = value.to_string();
    let mut formatted = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, ch) in digits.chars().rev().enumerate() {
        if index > 0 && index % 3 == 0 {
            formatted.push(',');
        }
        formatted.push(ch);
    }
    formatted.chars().rev().collect()
}

fn format_turn_meta_line(summary: &TurnRecordSummary) -> String {
    let usage = if summary.usage.cached_tokens > 0 {
        format!(
            "{} in ({} cached) · {} out · {} total",
            format_count(summary.usage.input_tokens),
            format_count(summary.usage.cached_tokens),
            format_count(summary.usage.output_tokens),
            format_count(summary.usage.total_tokens),
        )
    } else {
        format!(
            "{} in · {} out · {} total",
            format_count(summary.usage.input_tokens),
            format_count(summary.usage.output_tokens),
            format_count(summary.usage.total_tokens),
        )
    };
    format!(
        "{} · {} · {} · {}",
        summary.model_name,
        format_elapsed_ms(u128::from(summary.elapsed_ms)),
        format_termination(&summary.termination),
        usage,
    )
}

fn render_markdownish_text(theme: &CliTheme, text: &str, width: usize) -> Vec<String> {
    let mut rendered = Vec::new();
    let mut in_code_block = false;
    for raw_line in text.lines() {
        let trimmed = raw_line.trim_end();
        if trimmed.starts_with("```") {
            in_code_block = !in_code_block;
            if in_code_block {
                let language = trimmed.trim_start_matches('`').trim();
                if !language.is_empty() {
                    rendered.push(theme.muted(&format!("[{language}]")));
                }
            }
            continue;
        }

        if in_code_block {
            if trimmed.is_empty() {
                rendered.push(String::new());
            } else {
                for part in split_long_word(trimmed, width.saturating_sub(2).max(12)) {
                    rendered.push(format!("  {}", theme.muted(&part)));
                }
            }
            continue;
        }

        if trimmed.is_empty() {
            if rendered.last().is_some_and(|line| !line.is_empty()) {
                rendered.push(String::new());
            }
            continue;
        }

        if let Some((prefix, content)) = parse_list_marker(trimmed) {
            rendered.extend(wrap_with_prefix(prefix, content, width));
            continue;
        }

        rendered.extend(wrap_text(trimmed, width));
    }

    while rendered.last().is_some_and(|line| line.is_empty()) {
        rendered.pop();
    }

    rendered
}

fn parse_list_marker(line: &str) -> Option<(&str, &str)> {
    if let Some(content) = line.strip_prefix("- ").or_else(|| line.strip_prefix("* ")) {
        return Some(("•", content.trim_start()));
    }

    let digits = line.chars().take_while(|ch| ch.is_ascii_digit()).count();
    if digits > 0 {
        let marker_end = digits + 2;
        if line.chars().nth(digits) == Some('.') && line.chars().nth(digits + 1) == Some(' ') {
            return Some((&line[..digits + 1], line[marker_end..].trim_start()));
        }
    }

    None
}

fn wrap_with_prefix(prefix: &str, text: &str, width: usize) -> Vec<String> {
    let prefix_width = prefix.chars().count() + 1;
    let content_width = width.saturating_sub(prefix_width).max(12);
    let wrapped = wrap_text(text, content_width);
    wrapped
        .into_iter()
        .enumerate()
        .map(|(index, line)| {
            if index == 0 {
                format!("{prefix} {line}")
            } else {
                format!("{} {}", " ".repeat(prefix.chars().count()), line)
            }
        })
        .collect()
}

fn is_section_heading(line: &str) -> bool {
    !line.contains(':')
        && !line.starts_with('/')
        && !line.starts_with(' ')
        && !line.is_empty()
        && line
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == ' ' || ch == '&')
}

fn wrap_text(text: &str, width: usize) -> Vec<String> {
    if text.is_empty() || width == 0 {
        return vec![String::new()];
    }

    let mut wrapped = Vec::new();
    for raw_line in text.lines() {
        if raw_line.trim().is_empty() {
            wrapped.push(String::new());
            continue;
        }

        let mut current = String::new();
        for word in raw_line.split_whitespace() {
            let next_len = if current.is_empty() {
                word.chars().count()
            } else {
                current.chars().count() + 1 + word.chars().count()
            };

            if next_len <= width {
                if !current.is_empty() {
                    current.push(' ');
                }
                current.push_str(word);
                continue;
            }

            if !current.is_empty() {
                wrapped.push(current);
                current = String::new();
            }

            if word.chars().count() <= width {
                current.push_str(word);
            } else {
                wrapped.extend(split_long_word(word, width));
            }
        }

        if !current.is_empty() {
            wrapped.push(current);
        }
    }

    if wrapped.is_empty() {
        wrapped.push(String::new());
    }

    wrapped
}

fn split_long_word(word: &str, width: usize) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    for ch in word.chars() {
        if current.chars().count() == width {
            parts.push(current);
            current = String::new();
        }
        current.push(ch);
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts
}

fn style_card_border(theme: &CliTheme, tone: CardTone, text: &str) -> String {
    match tone {
        CardTone::Assistant | CardTone::Status => theme.muted(text),
        CardTone::User => theme.success(text),
        CardTone::Warning => theme.warning(text),
    }
}

fn format_card_line(
    theme: &CliTheme,
    tone: CardTone,
    vertical: &str,
    content: &str,
    visible_width: usize,
    inner_width: usize,
) -> String {
    let padding = " ".repeat(inner_width.saturating_sub(visible_width + 2));
    format!(
        "{} {}{} {}",
        style_card_border(theme, tone, vertical),
        content,
        padding,
        style_card_border(theme, tone, vertical)
    )
}

fn style_kv_line(theme: &CliTheme, line: &str) -> String {
    let Some((label, value)) = line.split_once(':') else {
        return line.to_owned();
    };
    format!("{}:{}", theme.bold(label), value)
}

fn print_prompt(theme: &CliTheme) {
    print!("{} ", theme.accent(prompt_text()));
}

fn prompt_text() -> &'static str {
    "You>"
}

fn readline_prompt() -> &'static str {
    "You> "
}

enum ChatCommand {
    Help,
    New,
    Session,
    Status,
    Watch,
    Mcp(McpCommand),
    Approve {
        approval_id: String,
        decision: ApprovalDecision,
    },
    Exit,
    SendMessage(String),
    Unknown(String),
}

fn parse_chat_command(input: &str) -> ChatCommand {
    if !input.starts_with('/') && !input.starts_with('\\') {
        return ChatCommand::SendMessage(input.to_owned());
    }

    let parts = input.split_whitespace().collect::<Vec<_>>();
    match parts.as_slice() {
        ["/help"] => ChatCommand::Help,
        ["/new"] => ChatCommand::New,
        ["/session"] => ChatCommand::Session,
        ["/status"] => ChatCommand::Status,
        ["/watch"] => ChatCommand::Watch,
        ["/mcp"] | ["/mcp", "help"] => ChatCommand::Mcp(McpCommand::Help),
        ["/mcp", "list"] => ChatCommand::Mcp(McpCommand::List),
        ["/mcp", "show", name] => ChatCommand::Mcp(McpCommand::Show((*name).to_owned())),
        ["/mcp", "add"] => ChatCommand::Mcp(McpCommand::Add),
        ["/mcp", "edit", name] => ChatCommand::Mcp(McpCommand::Edit((*name).to_owned())),
        ["/mcp", "remove", name] => ChatCommand::Mcp(McpCommand::Remove((*name).to_owned())),
        ["/exit"] | ["\\exit"] => ChatCommand::Exit,
        ["/approve", approval_id, "approve"] => ChatCommand::Approve {
            approval_id: (*approval_id).to_owned(),
            decision: ApprovalDecision::Approve,
        },
        ["/approve", approval_id, "reject"] => ChatCommand::Approve {
            approval_id: (*approval_id).to_owned(),
            decision: ApprovalDecision::Reject,
        },
        [command, ..] => ChatCommand::Unknown(command.trim_start_matches(['/', '\\']).to_owned()),
        _ => ChatCommand::Unknown("".to_owned()),
    }
}

enum ChatLoopAction {
    Continue,
    Exit,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum McpCommand {
    Help,
    List,
    Show(String),
    Add,
    Edit(String),
    Remove(String),
}

fn usage(bin: &str) -> String {
    format!(
        "Usage:\n  {bin}\n  {bin} repl\n  {bin} session start\n  {bin} session watch <session_id>\n  {bin} message send <session_id> <text>\n  {bin} approve <session_id> <approval_id> <approve|reject>\n  {bin} whatsapp status\n  {bin} whatsapp login\n  {bin} whatsapp complete-login <login_session_id>\n  {bin} whatsapp logout\n  {bin} whatsapp receive <from_user_id> <text>\n  {bin} help"
    )
}

#[cfg(test)]
mod tests {
    use agent_controlplane::{
        ApprovalDecision, ApprovalId, ApprovalRequestRecord, DeliveryStatus, SessionEvent,
        SessionStatus, TurnRecordSummary,
    };
    use agent_runtime::{
        RuntimeEvent, RuntimeExecutor, StepId, TerminationReason, TurnId, UsageSummary,
    };
    use mcp_client::McpInitializeResult;
    use mcp_config::McpServerConfig;
    use serde_json::json;

    use super::{
        CONNECT_RETRY_COUNT, ChatCommand, CliTheme, INITIAL_BACKOFF_MS, McpCommand,
        SessionEventRenderState, build_metadata_catalogs, format_mcp_remote_refresh_error,
        format_turn_meta_line, is_mcp_remote_stdio, mcp_remote_defaults, parse_chat_command,
        parse_sse_data, prompt_text, render_markdownish_text, render_runtime_event_with_theme,
        render_session_event_with_theme, render_user_message, send_with_retry,
        server_supports_capability, should_retry_request, split_sse_frame,
    };

    #[test]
    fn parses_basic_chat_commands() {
        assert!(matches!(parse_chat_command("/help"), ChatCommand::Help));
        assert!(matches!(parse_chat_command("/new"), ChatCommand::New));
        assert!(matches!(
            parse_chat_command("/session"),
            ChatCommand::Session
        ));
        assert!(matches!(parse_chat_command("/status"), ChatCommand::Status));
        assert!(matches!(parse_chat_command("/watch"), ChatCommand::Watch));
        assert!(matches!(
            parse_chat_command("/mcp"),
            ChatCommand::Mcp(McpCommand::Help)
        ));
        assert!(matches!(
            parse_chat_command("/mcp list"),
            ChatCommand::Mcp(McpCommand::List)
        ));
        assert!(matches!(
            parse_chat_command("/mcp show crm"),
            ChatCommand::Mcp(McpCommand::Show(name)) if name == "crm"
        ));
        assert!(matches!(
            parse_chat_command("/mcp add"),
            ChatCommand::Mcp(McpCommand::Add)
        ));
        assert!(matches!(
            parse_chat_command("/mcp edit crm"),
            ChatCommand::Mcp(McpCommand::Edit(name)) if name == "crm"
        ));
        assert!(matches!(
            parse_chat_command("/mcp remove crm"),
            ChatCommand::Mcp(McpCommand::Remove(name)) if name == "crm"
        ));
        assert!(matches!(parse_chat_command("/exit"), ChatCommand::Exit));
        assert!(matches!(parse_chat_command("\\exit"), ChatCommand::Exit));
        assert!(matches!(
            parse_chat_command("/approve abc approve"),
            ChatCommand::Approve { approval_id, decision: ApprovalDecision::Approve }
            if approval_id == "abc"
        ));
        assert!(matches!(
            parse_chat_command("hello"),
            ChatCommand::SendMessage(text) if text == "hello"
        ));
        assert!(matches!(
            parse_chat_command("/nope"),
            ChatCommand::Unknown(command) if command == "nope"
        ));
    }

    #[test]
    fn renders_user_prompt_and_card() {
        let theme = CliTheme::plain();

        assert_eq!(prompt_text(), "You>");

        let card = render_user_message(&theme, "show latest insights");
        assert!(card.contains("You"));
        assert!(card.contains("show latest insights"));
        assert!(!card.contains("Arka"));
    }

    #[test]
    fn metadata_catalogs_keep_full_initialize_instructions() {
        let server = McpServerConfig {
            name: "crm".to_owned(),
            transport: None,
            command: "fake-server".to_owned(),
            args: vec!["--stdio".to_owned()],
            env: Default::default(),
            description: Some("CRM server".to_owned()),
        };
        let initialize: McpInitializeResult = serde_json::from_value(json!({
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "serverInfo": {
                "name": "fake-crm",
                "version": "1.0.0",
                "title": "Fake CRM"
            },
            "instructions": "x".repeat(300)
        }))
        .expect("initialize payload should parse");
        let tools = vec![
            serde_json::from_value(json!({
                "name": "run-query",
                "title": "Run Query",
                "description": "Execute a query",
                "inputSchema": {"type": "object"}
            }))
            .expect("tool descriptor should parse"),
        ];

        let (minimal, full) =
            build_metadata_catalogs(&server, &initialize, true, &tools, false, &[]);

        assert_eq!(minimal.server.instructions_summary, initialize.instructions);
        assert_eq!(full.server.instructions_summary, initialize.instructions);
    }

    #[test]
    fn metadata_catalogs_track_supported_capability_families_independently_of_counts() {
        let server = McpServerConfig {
            name: "crm".to_owned(),
            transport: None,
            command: "fake-server".to_owned(),
            args: vec!["--stdio".to_owned()],
            env: Default::default(),
            description: None,
        };
        let initialize: McpInitializeResult = serde_json::from_value(json!({
            "protocolVersion": "2025-11-25",
            "capabilities": {
                "tools": {},
                "resources": {}
            },
            "serverInfo": {
                "name": "fake-crm",
                "version": "1.0.0"
            }
        }))
        .expect("initialize payload should parse");

        let (minimal, full) = build_metadata_catalogs(&server, &initialize, true, &[], true, &[]);

        assert!(minimal.capability_families.tools.supported);
        assert_eq!(minimal.capability_families.tools.count, 0);
        assert!(minimal.capability_families.resources.supported);
        assert_eq!(minimal.capability_families.resources.count, 0);
        assert_eq!(full.capability_families, minimal.capability_families);
    }

    #[test]
    fn detects_capabilities_from_initialize_payload() {
        let initialize: McpInitializeResult = serde_json::from_value(json!({
            "protocolVersion": "2025-11-25",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "fake-crm",
                "version": "1.0.0"
            }
        }))
        .expect("initialize payload should parse");

        assert!(server_supports_capability(&initialize, "tools"));
        assert!(!server_supports_capability(&initialize, "resources"));
    }

    #[test]
    fn detects_mcp_remote_stdio_wrappers() {
        assert!(is_mcp_remote_stdio(
            "npx",
            &[
                "-y".to_owned(),
                "mcp-remote".to_owned(),
                "https://example.com/mcp".to_owned(),
            ]
        ));
        assert!(is_mcp_remote_stdio(
            "npx",
            &[
                "mcp-remote@latest".to_owned(),
                "https://example.com/mcp".to_owned(),
            ]
        ));
        assert!(!is_mcp_remote_stdio(
            "uvx",
            &["mcp-server-sqlite".to_owned()]
        ));
    }

    #[test]
    fn extracts_mcp_remote_defaults_from_existing_transport() {
        let transport = mcp_config::McpTransportConfig::Stdio {
            command: "npx".to_owned(),
            args: vec![
                "-y".to_owned(),
                "mcp-remote@latest".to_owned(),
                "https://remote.example.com/mcp".to_owned(),
                "--debug".to_owned(),
                "--resource".to_owned(),
                "tenant-1".to_owned(),
            ],
            env: std::collections::HashMap::from([(
                "HTTPS_PROXY".to_owned(),
                "http://127.0.0.1:3128".to_owned(),
            )]),
        };

        let defaults = mcp_remote_defaults(Some(&transport));
        assert_eq!(defaults.command, "npx");
        assert_eq!(defaults.package, "mcp-remote@latest");
        assert!(defaults.include_npx_yes);
        assert_eq!(
            defaults.remote_url.as_deref(),
            Some("https://remote.example.com/mcp")
        );
        assert_eq!(
            defaults.extra_args,
            vec![
                "--debug".to_owned(),
                "--resource".to_owned(),
                "tenant-1".to_owned(),
            ]
        );
        assert_eq!(
            defaults.env.get("HTTPS_PROXY").map(String::as_str),
            Some("http://127.0.0.1:3128")
        );
    }

    #[test]
    fn formats_wrapper_specific_refresh_guidance() {
        let server = McpServerConfig {
            name: "kite".to_owned(),
            transport: Some(mcp_config::McpTransportConfig::Stdio {
                command: "npx".to_owned(),
                args: vec![
                    "mcp-remote".to_owned(),
                    "https://mcp.kite.trade/mcp".to_owned(),
                ],
                env: Default::default(),
            }),
            command: String::new(),
            args: Vec::new(),
            env: Default::default(),
            description: None,
        };

        let rendered = format_mcp_remote_refresh_error(&server, "boom");
        assert!(rendered.contains("failed to refresh metadata for `kite` via mcp-remote"));
        assert!(rendered.contains("Add `-y`"));
        assert!(rendered.contains("`~/.mcp-auth/*_debug.log`"));
        assert!(rendered.contains("mcp-remote-client https://mcp.kite.trade/mcp"));
    }

    #[test]
    fn parses_sse_data_frames() {
        let buffer = "event: message\ndata: {\"type\":\"turn_started\"}\n\nrest";
        let (frame, rest) = split_sse_frame(buffer).expect("frame should split");
        assert_eq!(rest, "rest");
        assert_eq!(parse_sse_data(frame), Some("{\"type\":\"turn_started\"}"));
    }

    #[test]
    fn renders_runtime_and_session_events() {
        let theme = CliTheme::plain();
        let mut state = SessionEventRenderState::default();
        let runtime = RuntimeEvent::McpCalled {
            turn_id: TurnId::new(),
            step_id: StepId::new(),
            server_name: "postgres".to_owned(),
            tool_name: "run-sql".to_owned(),
            executor: RuntimeExecutor::main_agent(),
            at: std::time::SystemTime::now(),
        };
        let runtime_line = render_runtime_event_with_theme(&theme, false, &runtime)
            .expect("runtime should render");
        assert!(runtime_line.contains("mcp"));
        assert!(runtime_line.contains("calling postgres/run-sql"));

        let turn_started = SessionEvent::TurnStarted {
            session_id: agent_controlplane::SessionId::new(),
        };
        let turn_started_lines = render_session_event_with_theme(&theme, &mut state, &turn_started)
            .expect("turn started should render");
        assert!(turn_started_lines[0].contains("Live Turn"));

        let session_event = SessionEvent::TurnCompleted {
            session_id: agent_controlplane::SessionId::new(),
            summary: TurnRecordSummary {
                turn_number: 3,
                model_name: "gpt-5.4".to_owned(),
                elapsed_ms: 2400,
                final_text: "done".to_owned(),
                termination: TerminationReason::Final,
                usage: UsageSummary {
                    input_tokens: 1,
                    cached_tokens: 0,
                    output_tokens: 1,
                    total_tokens: 2,
                },
                completed_at: std::time::SystemTime::now(),
            },
        };
        let session_card = render_session_event_with_theme(&theme, &mut state, &session_event)
            .expect("session event should render");
        let transcript = session_card.join("\n");
        assert!(transcript.contains("Arka"));
        assert!(transcript.contains("Turn 3"));
        assert!(transcript.contains("gpt-5.4 · 2.40s · Final · 1 in · 1 out · 2 total"));
        assert!(transcript.contains("done"));

        let approval = SessionEvent::ApprovalRequested {
            session_id: agent_controlplane::SessionId::new(),
            approval: ApprovalRequestRecord {
                approval_id: ApprovalId::new(),
                session_id: agent_controlplane::SessionId::new(),
                prompt: "Need approval".to_owned(),
                state: agent_controlplane::ApprovalState::Pending,
                created_at: std::time::SystemTime::now(),
                resolved_at: None,
            },
        };
        assert!(
            render_session_event_with_theme(&theme, &mut state, &approval)
                .expect("approval should render")
                .iter()
                .any(|line| line.contains("Approval"))
        );

        let delivery = SessionEvent::ChannelDeliveryAttempted {
            session_id: agent_controlplane::SessionId::new(),
            channel: agent_controlplane::ChannelKind::Cli,
            status: DeliveryStatus::Sent,
        };
        let delivery_line = render_session_event_with_theme(&theme, &mut state, &delivery)
            .expect("delivery should render");
        assert!(delivery_line[0].contains("delivery"));
        assert!(delivery_line[0].contains("Sent"));

        let ignored = SessionEvent::SessionCreated {
            session: agent_controlplane::SessionRecord {
                session_id: agent_controlplane::SessionId::new(),
                created_at: std::time::SystemTime::now(),
                updated_at: std::time::SystemTime::now(),
                status: SessionStatus::Idle,
                bindings: Vec::new(),
                last_turn: None,
            },
        };
        assert_eq!(
            render_session_event_with_theme(&theme, &mut state, &ignored),
            None
        );
    }

    #[test]
    fn renders_markdownish_lists_and_code_blocks() {
        let theme = CliTheme::plain();
        let rendered = render_markdownish_text(
            &theme,
            "Summary\n\n1. first item\n- second item\n\n```sql\nselect * from table;\n```",
            48,
        )
        .join("\n");

        assert!(rendered.contains("Summary"));
        assert!(rendered.contains("1. first item"));
        assert!(rendered.contains("• second item"));
        assert!(rendered.contains("select * from table;"));
    }

    #[test]
    fn format_turn_meta_line_includes_cached_tokens_when_present() {
        let summary = TurnRecordSummary {
            turn_number: 7,
            model_name: "gpt-5.4-mini".to_owned(),
            elapsed_ms: 16_700,
            final_text: "done".to_owned(),
            termination: TerminationReason::Final,
            usage: UsageSummary {
                input_tokens: 356_788,
                cached_tokens: 350_000,
                output_tokens: 657,
                total_tokens: 357_445,
            },
            completed_at: std::time::SystemTime::now(),
        };

        assert_eq!(
            format_turn_meta_line(&summary),
            "gpt-5.4-mini · 16.7s · Final · 356,788 in (350,000 cached) · 657 out · 357,445 total"
        );
    }

    #[tokio::test]
    async fn retries_connect_errors_until_success() {
        let attempts = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let attempts_for_closure = attempts.clone();

        let result = send_with_retry("http://127.0.0.1:8080", move || {
            let attempts = attempts_for_closure.clone();
            async move {
                let current = attempts.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                if current <= 2 {
                    let temp_client = reqwest::Client::new();
                    temp_client
                        .get("http://127.0.0.1:9")
                        .send()
                        .await
                        .map(|_| "unexpected")
                } else {
                    Ok("ok")
                }
            }
        })
        .await
        .expect("operation should eventually succeed");

        assert_eq!(result, "ok");
        assert_eq!(attempts.load(std::sync::atomic::Ordering::SeqCst), 3);
    }

    #[test]
    fn only_connect_errors_are_retried() {
        let client = reqwest::Client::new();
        let error = tokio::runtime::Runtime::new()
            .expect("runtime should build")
            .block_on(async { client.get("http://127.0.0.1:9").send().await })
            .expect_err("request should fail");
        assert!(should_retry_request(&error));
        assert_eq!(CONNECT_RETRY_COUNT, 5);
        assert_eq!(INITIAL_BACKOFF_MS, 500);
    }
}
