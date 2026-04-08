//! Control-plane HTTP server entrypoint.
//!
//! This binary wires together environment loading, the OpenAI-backed runtime,
//! the session service, and the Axum router used by API and channel clients.

use std::{
    env, fs,
    path::{Path, PathBuf},
    str::FromStr,
    time::Duration,
};

use agent_controlplane::{
    DebugHistoryStore, JsonlConversationStore, LoggingWhatsAppDeliveryClient,
    PostgresRuntimeDebugListener, ReqwestSlackDeliveryClient, ReqwestWhatsAppWebBridgeClient,
    RuntimeExecutionConfig, RuntimeTurnRunner, SlackConnector, WhatsAppConnector, WhatsAppDmPolicy,
    router_with_channels,
};
use agent_openai::OpenAiModelAdapter;
use agent_runtime::{AgentRuntime, ModelConfig, RuntimeLimits, ServerName};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Load local development configuration from `.env` when present while
    // leaving already-exported environment variables untouched.
    let _ = dotenvy::dotenv();
    init_logging();

    let api_key = env::var("OPENAI_API_KEY")?;
    let model_name = env::var("MODEL_NAME").unwrap_or_else(|_| "gpt-5.4".to_owned());
    let system_prompt_path = load_system_prompt_path();
    let workspace_root = prepare_runtime_workspace_root()?;
    let registry_path =
        env::var("MCP_REGISTRY_PATH").unwrap_or_else(|_| "config/mcp_servers.json".to_owned());
    let subagent_registry_path =
        env::var("SUBAGENT_REGISTRY_PATH").unwrap_or_else(|_| "config/subagents.json".to_owned());
    let tool_policy_path = env::var("TOOL_POLICY_PATH").ok().map(PathBuf::from);
    let server_names = env::var("ENABLED_MCP_SERVERS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter(|part| !part.trim().is_empty())
                .map(ServerName::from_str)
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?;
    let bind_addr = env::var("BIND_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_owned());
    let session_store_dir =
        env::var("SESSION_STORE_DIR").unwrap_or_else(|_| "data/sessions".to_owned());
    let runtime_debug_listener_queue_capacity = env::var("RUNTIME_DEBUG_LISTENER_QUEUE_CAPACITY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(512);
    let runtime_debug_postgres_dsn = env::var("RUNTIME_DEBUG_POSTGRES_DSN").ok();
    let runtime_debug_postgres_enabled = env::var("RUNTIME_DEBUG_POSTGRES_ENABLED")
        .ok()
        .map(|value| value == "true")
        .unwrap_or(runtime_debug_postgres_dsn.is_some());
    let slack_bot_token = env::var("SLACK_BOT_TOKEN").ok();
    let slack_signing_secret = env::var("SLACK_SIGNING_SECRET").ok();
    let slack_api_base_url = env::var("SLACK_API_BASE_URL").ok();
    let slack_event_queue_capacity = env::var("SLACK_EVENT_QUEUE_CAPACITY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(256);
    let whatsapp_gateway_enabled = read_env_bool("WHATSAPP_GATEWAY_ENABLED", false);
    let whatsapp_local_web_enabled = read_env_bool("WHATSAPP_LOCAL_WEB_ENABLED", false);
    let whatsapp_account_id =
        env::var("WHATSAPP_ACCOUNT_ID").unwrap_or_else(|_| "default".to_owned());
    let whatsapp_event_queue_capacity = env::var("WHATSAPP_EVENT_QUEUE_CAPACITY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(256);
    let whatsapp_dm_policy = match env::var("WHATSAPP_DM_POLICY")
        .unwrap_or_else(|_| "allowlist".to_owned())
        .as_str()
    {
        "open" => WhatsAppDmPolicy::Open,
        _ => WhatsAppDmPolicy::Allowlist,
    };
    let whatsapp_allow_from = env::var("WHATSAPP_DM_ALLOW_FROM")
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|part| {
                    let trimmed = part.trim();
                    (!trimmed.is_empty()).then(|| trimmed.to_owned())
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let whatsapp_bridge_base_url = env::var("WHATSAPP_BRIDGE_BASE_URL").ok();
    let runtime_limits = RuntimeLimits {
        max_steps_per_turn: read_env_u32("RUNTIME_MAX_STEPS_PER_TURN", 8),
        max_mcp_calls_per_step: read_env_u32("RUNTIME_MAX_MCP_CALLS_PER_STEP", 4),
        max_subagent_steps_per_invocation: read_env_u32(
            "RUNTIME_MAX_SUBAGENT_STEPS_PER_INVOCATION",
            8,
        ),
        max_subagent_mcp_calls_per_invocation: read_env_u32(
            "RUNTIME_MAX_SUBAGENT_MCP_CALLS_PER_INVOCATION",
            4,
        ),
        turn_timeout: Duration::from_secs(read_env_u64("RUNTIME_TURN_TIMEOUT_SECS", 180)),
        mcp_call_timeout: Duration::from_secs(read_env_u64("RUNTIME_MCP_CALL_TIMEOUT_SECS", 10)),
        ..RuntimeLimits::default()
    };
    let adapter = OpenAiModelAdapter::new(api_key)?;
    let runtime = AgentRuntime::new(adapter);
    let runner = RuntimeTurnRunner::new(
        runtime,
        RuntimeExecutionConfig {
            system_prompt_path,
            workspace_root,
            registry_path: registry_path.into(),
            subagent_registry_path: subagent_registry_path.into(),
            tool_policy_path,
            enabled_servers: server_names,
            limits: runtime_limits,
            model_config: ModelConfig::new(model_name),
        },
    );
    let store = JsonlConversationStore::open(&session_store_dir)?;
    let mut service = agent_controlplane::SessionService::new(runner, store);
    let mut debug_history_store = None;
    if runtime_debug_postgres_enabled {
        let dsn = runtime_debug_postgres_dsn.ok_or(
            "RUNTIME_DEBUG_POSTGRES_DSN must be set when Postgres debug listener is enabled",
        )?;
        debug_history_store = Some(DebugHistoryStore::connect(&dsn).await?);
        let listener =
            PostgresRuntimeDebugListener::connect(&dsn, runtime_debug_listener_queue_capacity)
                .await?;
        service = service.with_runtime_harness_listener(listener);
    }
    let slack_connector =
        slack_bot_token
            .zip(slack_signing_secret)
            .map(|(bot_token, signing_secret)| SlackConnector {
                signing_secret,
                delivery_client: std::sync::Arc::new(ReqwestSlackDeliveryClient::new(
                    bot_token,
                    slack_api_base_url,
                )),
                event_queue_capacity: slack_event_queue_capacity,
            });
    let whatsapp_connector = whatsapp_gateway_enabled.then(|| {
        let (delivery_client, control_client) = if whatsapp_local_web_enabled {
            let client = std::sync::Arc::new(ReqwestWhatsAppWebBridgeClient::new(
                whatsapp_bridge_base_url,
            ));
            (
                client.clone() as std::sync::Arc<dyn agent_controlplane::WhatsAppDeliveryClient>,
                Some(client as std::sync::Arc<dyn agent_controlplane::WhatsAppControlClient>),
            )
        } else {
            (
                std::sync::Arc::new(LoggingWhatsAppDeliveryClient)
                    as std::sync::Arc<dyn agent_controlplane::WhatsAppDeliveryClient>,
                None,
            )
        };
        WhatsAppConnector {
            account_id: whatsapp_account_id,
            dm_policy: whatsapp_dm_policy,
            allow_from: whatsapp_allow_from,
            delivery_client,
            control_client,
            event_queue_capacity: whatsapp_event_queue_capacity,
            state_path: PathBuf::from(&session_store_dir)
                .join("channels")
                .join("whatsapp-gateway.json"),
        }
    });
    let app = router_with_channels(
        service.clone(),
        debug_history_store,
        slack_connector,
        whatsapp_connector,
    );
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    service.shutdown().await?;
    Ok(())
}

fn init_logging() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("warn,tower_http=warn"));

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .compact()
        .init();
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl-C shutdown handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM shutdown handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

fn load_system_prompt_path() -> PathBuf {
    env::var("SYSTEM_PROMPT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("config/prompt.md"))
}

fn prepare_runtime_workspace_root() -> Result<PathBuf, std::io::Error> {
    let current_dir = env::current_dir()?;
    prepare_runtime_workspace_root_at(&current_dir)
}

fn prepare_runtime_workspace_root_at(current_dir: &Path) -> Result<PathBuf, std::io::Error> {
    let workspace_root = runtime_workspace_root_path(current_dir);
    fs::create_dir_all(&workspace_root)?;
    Ok(workspace_root)
}

fn runtime_workspace_root_path(current_dir: &Path) -> PathBuf {
    current_dir.join(".arka").join("tmp")
}

fn read_env_u64(name: &str, default: u64) -> u64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(default)
}

fn read_env_u32(name: &str, default: u32) -> u32 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(default)
}

fn read_env_bool(name: &str, default: bool) -> bool {
    env::var(name)
        .ok()
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::{prepare_runtime_workspace_root_at, runtime_workspace_root_path};

    #[test]
    fn runtime_workspace_root_path_uses_arka_tmp() {
        let current_dir = PathBuf::from("/tmp/arka-project");
        assert_eq!(
            runtime_workspace_root_path(&current_dir),
            current_dir.join(".arka").join("tmp")
        );
    }

    #[test]
    fn prepare_runtime_workspace_root_creates_missing_directory() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        let sandbox_root = std::env::temp_dir().join(format!(
            "agent-controlplane-working-directory-test-{}-{unique}",
            std::process::id()
        ));
        let expected = sandbox_root.join(".arka").join("tmp");
        if sandbox_root.exists() {
            fs::remove_dir_all(&sandbox_root).expect("old temp directory should be removable");
        }

        let prepared = prepare_runtime_workspace_root_at(&sandbox_root)
            .expect("workspace root should be prepared");

        assert_eq!(prepared, expected);
        assert!(prepared.is_dir());

        fs::remove_dir_all(&sandbox_root).expect("temp directory should be removable");
    }
}
