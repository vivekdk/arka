//! End-to-end runtime integration tests for stored MCP metadata and sub-agent delegation.

use std::{
    collections::VecDeque,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use agent_runtime::{
    AgentRuntime, ConversationMessage, ConversationRole, LocalToolName, MessageRecord, ModelConfig,
    ResponseClient, ResponseFormat, ResponseTarget, RunRequest, RuntimeError, RuntimeEvent,
    RuntimeLimits, ServerName, TerminationReason,
    model::{
        ModelAdapter, ModelAdapterError, ModelAdapterResponse, ModelStepDecision, ModelStepRequest,
        SubagentAdapterResponse, SubagentDecision, SubagentDelegationRequest, SubagentStepRequest,
    },
    state::{DelegationTarget, LocalToolsScopeTarget, McpCapabilityTarget, McpServerScopeTarget},
};
use async_trait::async_trait;
use mcp_metadata::{
    CURRENT_SCHEMA_VERSION, DEFAULT_METADATA_DIR, FullResourceMetadata, FullToolMetadata,
    McpCapabilityFamilies, McpCapabilityFamilySummary, McpFullCatalog, McpMinimalCatalog,
    McpServerMetadata, MinimalResourceMetadata, MinimalToolMetadata, artifact_paths,
    write_catalogs,
};
use serde_json::json;

const FAKE_SERVER_BIN: &str = env!("CARGO_BIN_EXE_fake-runtime-mcp-server");

fn default_response_target() -> ResponseTarget {
    ResponseTarget {
        client: ResponseClient::Cli,
        format: ResponseFormat::Markdown,
    }
}

fn metadata_test_lock() -> &'static tokio::sync::Mutex<()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
}

#[tokio::test]
async fn runtime_delegates_tool_executor_and_executes_tool() {
    let _metadata_guard = metadata_test_lock().lock().await;
    cleanup_mcp_metadata("fake");
    let prompt_log = Arc::new(Mutex::new(Vec::new()));
    let adapter = FakeModelAdapter::new(
        vec![
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::DelegateSubagent {
                    delegation: SubagentDelegationRequest {
                        subagent_type: "mcp-executor".to_owned(),
                        target: DelegationTarget::McpCapability(McpCapabilityTarget {
                            server_name: ServerName::new("fake").expect("valid server"),
                            capability_kind: mcp_metadata::CapabilityKind::Tool,
                            capability_id: "run-sql".to_owned(),
                        }),
                        goal: "Count rows".to_owned(),
                    },
                },
                usage: token_usage(4, 3),
            }),
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::Final {
                    content: "Final answer after delegated MCP".to_owned(),
                },
                usage: token_usage(2, 4),
            }),
        ],
        vec![
            Ok(SubagentAdapterResponse {
                decision: SubagentDecision::McpToolCall {
                    server_name: ServerName::new("fake").expect("valid server"),
                    tool_name: "run-sql".to_owned(),
                    arguments: json!({"query": "select 1"}),
                },
                usage: token_usage(2, 2),
            }),
            Ok(SubagentAdapterResponse {
                decision: SubagentDecision::Done {
                    summary: "Counted rows successfully".to_owned(),
                },
                usage: token_usage(1, 1),
            }),
        ],
        Arc::clone(&prompt_log),
        Duration::from_millis(0),
    );

    let runtime = AgentRuntime::new(adapter);
    let temp_dir = temp_dir("runtime-delegate-tool");
    let registry_path = write_registry(&temp_dir, FAKE_SERVER_BIN, Vec::new());
    write_mcp_metadata(&temp_dir, "fake");
    let prompt_path = write_prompt(
        &temp_dir,
        "MCPs:\n<dynamic variable: available MCPs>\nSub-agents:\n<dynamic variable: available sub-agents>\nTools:\n<dynamic variable: available tools>",
    );
    let subagent_registry_path = write_subagent_registry(&temp_dir);

    let outcome = runtime
        .run_turn(RunRequest {
            system_prompt_path: prompt_path,
            working_directory: temp_dir.clone(),
            conversation_history: vec![ConversationMessage {
                timestamp: SystemTime::now(),
                role: ConversationRole::Assistant,
                content: "Prior answer".to_owned(),
            }],
            recent_session_messages: vec![],
            user_message: "How many rows?".to_owned(),
            response_target: default_response_target(),
            registry_path,
            subagent_registry_path,
            tool_policy_path: None,
            enabled_servers: Some(vec![ServerName::new("fake").expect("valid server")]),
            limits: RuntimeLimits::default(),
            model_config: ModelConfig::new("fake-model"),
            require_todos: false,
        })
        .await
        .expect("turn should succeed");

    assert_eq!(outcome.final_text, "Final answer after delegated MCP");
    assert_eq!(outcome.termination, TerminationReason::Final);
    assert!(
        outcome
            .turn
            .messages
            .iter()
            .any(|message| matches!(message, MessageRecord::SubAgentCall(_)))
    );
    assert!(outcome.turn.messages.iter().any(|message| matches!(
        message,
        MessageRecord::SubAgentResult(record)
            if record.status == "completed"
                && record.executed_action_count == 1
                && record.detail.contains("Counted rows successfully")
    )));
    assert!(
        outcome
            .turn
            .messages
            .iter()
            .any(|message| matches!(message, MessageRecord::McpCall(_)))
    );
    assert!(
        outcome
            .turn
            .messages
            .iter()
            .any(|message| matches!(message, MessageRecord::McpResult(_)))
    );
    assert!(outcome.events.iter().any(|event| matches!(
        event,
        RuntimeEvent::HandoffToSubagent {
            subagent_type,
            goal,
            ..
        } if subagent_type == "mcp-executor" && goal == "Count rows"
    )));
    assert!(outcome.events.iter().any(|event| matches!(
        event,
        RuntimeEvent::HandoffToMainAgent {
            subagent_type,
            status,
            ..
        } if subagent_type == "mcp-executor" && status == "completed"
    )));

    let prompts = prompt_log.lock().expect("prompt log should lock");
    assert!(prompts[0].contains("available MCPs") || prompts[0].contains("Server: fake"));
    assert!(prompts[0].contains("mcp-executor"));
}

#[tokio::test]
async fn runtime_records_full_mcp_payload_and_subagent_executor_on_events() {
    let _metadata_guard = metadata_test_lock().lock().await;
    cleanup_mcp_metadata("fake");
    let prompt_log = Arc::new(Mutex::new(Vec::new()));
    let adapter = FakeModelAdapter::new(
        vec![
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::DelegateSubagent {
                    delegation: SubagentDelegationRequest {
                        subagent_type: "mcp-executor".to_owned(),
                        target: DelegationTarget::McpCapability(McpCapabilityTarget {
                            server_name: ServerName::new("fake").expect("valid server"),
                            capability_kind: mcp_metadata::CapabilityKind::Tool,
                            capability_id: "preview_leads".to_owned(),
                        }),
                        goal: "Preview leads".to_owned(),
                    },
                },
                usage: token_usage(2, 2),
            }),
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::Final {
                    content: "Previewed leads".to_owned(),
                },
                usage: token_usage(1, 1),
            }),
        ],
        vec![Ok(SubagentAdapterResponse {
            decision: SubagentDecision::McpToolCall {
                server_name: ServerName::new("fake").expect("valid server"),
                tool_name: "preview_leads".to_owned(),
                arguments: json!({"limit": 2}),
            },
            usage: token_usage(1, 1),
        })],
        Arc::clone(&prompt_log),
        Duration::from_millis(0),
    );
    let runtime = AgentRuntime::new(adapter);
    let temp_dir = temp_dir("runtime-full-mcp-payload");
    let registry_path = write_registry(&temp_dir, FAKE_SERVER_BIN, Vec::new());
    write_mcp_metadata(&temp_dir, "fake");

    let outcome = runtime
        .run_turn(RunRequest {
            system_prompt_path: write_prompt(
                &temp_dir,
                "MCPs:\n<dynamic variable: available MCPs>\nSub-agents:\n<dynamic variable: available sub-agents>",
            ),
            working_directory: temp_dir.clone(),
            conversation_history: vec![],
            recent_session_messages: vec![],
            user_message: "Show sample leads".to_owned(),
            response_target: default_response_target(),
            registry_path,
            subagent_registry_path: write_subagent_registry(&temp_dir),
            tool_policy_path: None,
            enabled_servers: Some(vec![ServerName::new("fake").expect("valid server")]),
            limits: RuntimeLimits::default(),
            model_config: ModelConfig::new("fake-model"),
            require_todos: false,
        })
        .await
        .expect("turn should succeed");

    let responded = outcome
        .events
        .iter()
        .find_map(|event| match event {
            RuntimeEvent::McpResponded {
                tool_name,
                executor,
                result_summary,
                response_payload,
                ..
            } if tool_name == "preview_leads" => Some((executor, result_summary, response_payload)),
            _ => None,
        })
        .expect("mcp responded event should be present");

    assert_eq!(responded.0.display_name, "Mcp Executor");
    assert_eq!(responded.0.subagent_type.as_deref(), Some("mcp-executor"));
    assert!(
        responded.2.to_string().len() > responded.1.len(),
        "debug payload should retain more detail than the transcript summary"
    );
    assert_eq!(
        responded.2["structuredContent"]["rows"][0]["user_id"],
        "usr_123456789"
    );
    assert_eq!(
        responded.2["structuredContent"]["rows"][1]["user_id"],
        "usr_987654321"
    );
    assert!(responded.1.contains("usr_123456789"));
    assert!(responded.1.contains("usr_987654321"));
    assert!(!responded.1.contains("..."));
}

#[tokio::test]
async fn delegated_subagent_follow_up_prompt_keeps_full_mcp_result_text() {
    let _metadata_guard = metadata_test_lock().lock().await;
    cleanup_mcp_metadata("fake");
    let prompt_log = Arc::new(Mutex::new(Vec::new()));
    let adapter = FakeModelAdapter::new(
        vec![
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::DelegateSubagent {
                    delegation: SubagentDelegationRequest {
                        subagent_type: "mcp-executor".to_owned(),
                        target: DelegationTarget::McpCapability(McpCapabilityTarget {
                            server_name: ServerName::new("fake").expect("valid server"),
                            capability_kind: mcp_metadata::CapabilityKind::Tool,
                            capability_id: "preview_leads".to_owned(),
                        }),
                        goal: "Preview leads".to_owned(),
                    },
                },
                usage: token_usage(2, 2),
            }),
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::Final {
                    content: "Preview complete".to_owned(),
                },
                usage: token_usage(1, 1),
            }),
        ],
        vec![
            Ok(SubagentAdapterResponse {
                decision: SubagentDecision::McpToolCall {
                    server_name: ServerName::new("fake").expect("valid server"),
                    tool_name: "preview_leads".to_owned(),
                    arguments: json!({"limit": 2}),
                },
                usage: token_usage(1, 1),
            }),
            Ok(SubagentAdapterResponse {
                decision: SubagentDecision::Done {
                    summary: "Previewed leads successfully".to_owned(),
                },
                usage: token_usage(1, 1),
            }),
        ],
        Arc::clone(&prompt_log),
        Duration::from_millis(0),
    );
    let runtime = AgentRuntime::new(adapter);
    let temp_dir = temp_dir("runtime-subagent-full-history");
    let registry_path = write_registry(&temp_dir, FAKE_SERVER_BIN, Vec::new());
    write_mcp_metadata(&temp_dir, "fake");

    runtime
        .run_turn(RunRequest {
            system_prompt_path: write_prompt(
                &temp_dir,
                "MCPs:\n<dynamic variable: available MCPs>\nSub-agents:\n<dynamic variable: available sub-agents>",
            ),
            working_directory: temp_dir.clone(),
            conversation_history: vec![],
            recent_session_messages: vec![],
            user_message: "Preview the latest leads".to_owned(),
            response_target: default_response_target(),
            registry_path,
            subagent_registry_path: write_subagent_registry(&temp_dir),
            tool_policy_path: None,
            enabled_servers: Some(vec![ServerName::new("fake").expect("valid server")]),
            limits: RuntimeLimits::default(),
            model_config: ModelConfig::new("fake-model"),
            require_todos: false,
        })
        .await
        .expect("turn should succeed");

    let prompts = prompt_log.lock().expect("prompt log should lock");
    let follow_up_prompt = prompts
        .iter()
        .find(|prompt| {
            prompt.contains("## Delegated Execution History")
                && prompt.contains("mcp_result")
                && prompt.contains("usr_123456789")
        })
        .expect("subagent follow-up prompt should include MCP result history");
    assert!(follow_up_prompt.contains("usr_987654321"));
    assert!(!follow_up_prompt.contains("..."));
}

#[tokio::test]
async fn prepared_session_skips_discovery_until_first_execution() {
    let _metadata_guard = metadata_test_lock().lock().await;
    cleanup_mcp_metadata("fake");
    let prompt_log = Arc::new(Mutex::new(Vec::new()));
    let adapter = FakeModelAdapter::new(
        vec![
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::DelegateSubagent {
                    delegation: SubagentDelegationRequest {
                        subagent_type: "mcp-executor".to_owned(),
                        target: DelegationTarget::McpCapability(McpCapabilityTarget {
                            server_name: ServerName::new("fake").expect("valid server"),
                            capability_kind: mcp_metadata::CapabilityKind::Tool,
                            capability_id: "run-sql".to_owned(),
                        }),
                        goal: "Run a query".to_owned(),
                    },
                },
                usage: token_usage(2, 2),
            }),
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::Final {
                    content: "done".to_owned(),
                },
                usage: token_usage(1, 1),
            }),
        ],
        vec![
            Ok(SubagentAdapterResponse {
                decision: SubagentDecision::McpToolCall {
                    server_name: ServerName::new("fake").expect("valid server"),
                    tool_name: "run-sql".to_owned(),
                    arguments: json!({"query": "select 1"}),
                },
                usage: token_usage(1, 1),
            }),
            Ok(SubagentAdapterResponse {
                decision: SubagentDecision::Done {
                    summary: "Query executed".to_owned(),
                },
                usage: token_usage(1, 1),
            }),
        ],
        prompt_log,
        Duration::from_millis(0),
    );
    let runtime = AgentRuntime::new(adapter);
    let temp_dir = temp_dir("runtime-lazy-connect");
    let log_path = temp_dir.join("mcp-log.txt");
    let registry_path = write_registry(
        &temp_dir,
        FAKE_SERVER_BIN,
        vec![
            "--log-file".to_owned(),
            log_path.to_str().expect("utf-8 path").to_owned(),
        ],
    );
    write_mcp_metadata(&temp_dir, "fake");
    let subagent_registry_path = write_subagent_registry(&temp_dir);
    let mut session = runtime
        .prepare_mcp_session(
            &registry_path,
            Some(&[ServerName::new("fake").expect("valid server")]),
        )
        .await
        .expect("session should prepare");

    assert!(
        !log_path.exists(),
        "session prep should not connect to the MCP"
    );

    runtime
        .run_turn_with_mcp_session(
            RunRequest {
                system_prompt_path: write_prompt(&temp_dir, "MCPs:\n<dynamic variable: available MCPs>\nSub-agents:\n<dynamic variable: available sub-agents>"),
                working_directory: temp_dir.clone(),
                conversation_history: vec![],
                recent_session_messages: vec![],
                user_message: "Run it".to_owned(),
                response_target: default_response_target(),
                registry_path,
                subagent_registry_path,
                tool_policy_path: None,
                enabled_servers: Some(vec![ServerName::new("fake").expect("valid server")]),
                limits: RuntimeLimits::default(),
                model_config: ModelConfig::new("fake-model"),
            require_todos: false,
            },
            &mut session,
        )
        .await
        .expect("turn should succeed");

    let log = fs::read_to_string(log_path).expect("log file should exist");
    assert!(log.contains("initialize"));
    assert!(log.contains("tools/call"));
    assert!(!log.contains("tools/list"));
}

#[tokio::test]
async fn runtime_reads_resource_via_tool_executor() {
    let _metadata_guard = metadata_test_lock().lock().await;
    cleanup_mcp_metadata("fake");
    let adapter = FakeModelAdapter::new(
        vec![
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::DelegateSubagent {
                    delegation: SubagentDelegationRequest {
                        subagent_type: "mcp-executor".to_owned(),
                        target: DelegationTarget::McpCapability(McpCapabilityTarget {
                            server_name: ServerName::new("fake").expect("valid server"),
                            capability_kind: mcp_metadata::CapabilityKind::Resource,
                            capability_id: "crm://dashboards/main".to_owned(),
                        }),
                        goal: "Read the dashboard".to_owned(),
                    },
                },
                usage: token_usage(2, 2),
            }),
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::Final {
                    content: "resource answer".to_owned(),
                },
                usage: token_usage(1, 1),
            }),
        ],
        vec![
            Ok(SubagentAdapterResponse {
                decision: SubagentDecision::McpResourceRead {
                    server_name: ServerName::new("fake").expect("valid server"),
                    resource_uri: "crm://dashboards/main".to_owned(),
                },
                usage: token_usage(1, 1),
            }),
            Ok(SubagentAdapterResponse {
                decision: SubagentDecision::Done {
                    summary: "Dashboard loaded".to_owned(),
                },
                usage: token_usage(1, 1),
            }),
        ],
        Arc::new(Mutex::new(Vec::new())),
        Duration::from_millis(0),
    );
    let runtime = AgentRuntime::new(adapter);
    let temp_dir = temp_dir("runtime-resource-read");
    let registry_path = write_registry(&temp_dir, FAKE_SERVER_BIN, Vec::new());
    write_mcp_metadata(&temp_dir, "fake");
    let outcome = runtime
        .run_turn(RunRequest {
            system_prompt_path: write_prompt(&temp_dir, "MCPs:\n<dynamic variable: available MCPs>\nSub-agents:\n<dynamic variable: available sub-agents>"),
            working_directory: temp_dir.clone(),
            conversation_history: vec![],
            recent_session_messages: vec![],
            user_message: "Show the dashboard".to_owned(),
            response_target: default_response_target(),
            registry_path,
            subagent_registry_path: write_subagent_registry(&temp_dir),
            tool_policy_path: None,
            enabled_servers: Some(vec![ServerName::new("fake").expect("valid server")]),
            limits: RuntimeLimits::default(),
            model_config: ModelConfig::new("fake-model"),
            require_todos: false,
        })
        .await
        .expect("turn should succeed");

    assert_eq!(outcome.final_text, "resource answer");
    assert!(
        outcome
            .turn
            .messages
            .iter()
            .any(|message| matches!(message, MessageRecord::McpResult(record) if record.result_summary.contains("dashboard")))
    );
    assert!(outcome.turn.messages.iter().any(|message| matches!(
        message,
        MessageRecord::SubAgentResult(record)
            if record.status == "completed"
                && record.executed_action_count == 1
                && record.detail.contains("Dashboard loaded")
    )));
}

#[tokio::test]
async fn delegated_subagent_can_take_multiple_mcp_steps_without_leaking_trace_to_main_prompt() {
    let _metadata_guard = metadata_test_lock().lock().await;
    cleanup_mcp_metadata("fake");
    let prompt_log = Arc::new(Mutex::new(Vec::new()));
    let adapter = FakeModelAdapter::new(
        vec![
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::DelegateSubagent {
                    delegation: SubagentDelegationRequest {
                        subagent_type: "mcp-executor".to_owned(),
                        target: DelegationTarget::McpCapability(McpCapabilityTarget {
                            server_name: ServerName::new("fake").expect("valid server"),
                            capability_kind: mcp_metadata::CapabilityKind::Tool,
                            capability_id: "run-sql".to_owned(),
                        }),
                        goal: "Find the final count".to_owned(),
                    },
                },
                usage: token_usage(2, 2),
            }),
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::Final {
                    content: "final after delegated loop".to_owned(),
                },
                usage: token_usage(1, 1),
            }),
        ],
        vec![
            Ok(SubagentAdapterResponse {
                decision: SubagentDecision::McpToolCall {
                    server_name: ServerName::new("fake").expect("valid server"),
                    tool_name: "run-sql".to_owned(),
                    arguments: json!({"query": "select count(*) from one"}),
                },
                usage: token_usage(1, 1),
            }),
            Ok(SubagentAdapterResponse {
                decision: SubagentDecision::McpToolCall {
                    server_name: ServerName::new("fake").expect("valid server"),
                    tool_name: "run-sql".to_owned(),
                    arguments: json!({"query": "select count(*) from two"}),
                },
                usage: token_usage(1, 1),
            }),
            Ok(SubagentAdapterResponse {
                decision: SubagentDecision::Done {
                    summary: "Completed delegated count workflow".to_owned(),
                },
                usage: token_usage(1, 1),
            }),
        ],
        Arc::clone(&prompt_log),
        Duration::from_millis(0),
    );

    let runtime = AgentRuntime::new(adapter);
    let temp_dir = temp_dir("runtime-delegated-loop");
    let registry_path = write_registry(&temp_dir, FAKE_SERVER_BIN, Vec::new());
    write_mcp_metadata(&temp_dir, "fake");
    let outcome = runtime
        .run_turn(RunRequest {
            system_prompt_path: write_prompt(&temp_dir, "MCPs:\n<dynamic variable: available MCPs>\nSub-agents:\n<dynamic variable: available sub-agents>"),
            working_directory: temp_dir.clone(),
            conversation_history: vec![],
            recent_session_messages: vec![],
            user_message: "Get me the final count".to_owned(),
            response_target: default_response_target(),
            registry_path,
            subagent_registry_path: write_subagent_registry(&temp_dir),
            tool_policy_path: None,
            enabled_servers: Some(vec![ServerName::new("fake").expect("valid server")]),
            limits: RuntimeLimits::default(),
            model_config: ModelConfig::new("fake-model"),
            require_todos: false,
        })
        .await
        .expect("turn should succeed");

    let prompts = prompt_log.lock().expect("prompt log should lock");
    let follow_up_prompt = prompts.last().expect("follow-up prompt should exist");
    assert!(follow_up_prompt.contains("sub_agent_result"));
    assert!(!follow_up_prompt.contains("mcp_call"));
    assert!(!follow_up_prompt.contains("mcp_result"));
    assert!(
        outcome
            .turn
            .messages
            .iter()
            .filter(|message| matches!(message, MessageRecord::McpCall(_)))
            .count()
            == 2
    );
    assert!(
        outcome
            .turn
            .messages
            .iter()
            .filter(|message| matches!(message, MessageRecord::McpResult(_)))
            .count()
            == 2
    );
}

#[tokio::test]
async fn runtime_server_scoped_mcp_subagent_can_use_multiple_tools_in_one_handoff() {
    let _metadata_guard = metadata_test_lock().lock().await;
    cleanup_mcp_metadata("fake");
    let adapter = FakeModelAdapter::new(
        vec![
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::DelegateSubagent {
                    delegation: SubagentDelegationRequest {
                        subagent_type: "mcp-executor".to_owned(),
                        target: DelegationTarget::McpServerScope(McpServerScopeTarget {
                            server_name: ServerName::new("fake").expect("valid server"),
                        }),
                        goal: "Inspect the fake server and compute the final answer".to_owned(),
                    },
                },
                usage: token_usage(2, 2),
            }),
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::Final {
                    content: "found the answer".to_owned(),
                },
                usage: token_usage(1, 1),
            }),
        ],
        vec![
            Ok(SubagentAdapterResponse {
                decision: SubagentDecision::McpToolCall {
                    server_name: ServerName::new("fake").expect("valid server"),
                    tool_name: "preview_leads".to_owned(),
                    arguments: json!({"limit": 2}),
                },
                usage: token_usage(1, 1),
            }),
            Ok(SubagentAdapterResponse {
                decision: SubagentDecision::McpToolCall {
                    server_name: ServerName::new("fake").expect("valid server"),
                    tool_name: "run-sql".to_owned(),
                    arguments: json!({"query": "select count(*) from leads"}),
                },
                usage: token_usage(1, 1),
            }),
            Ok(SubagentAdapterResponse {
                decision: SubagentDecision::Done {
                    summary: "Completed one-server MCP workflow".to_owned(),
                },
                usage: token_usage(1, 1),
            }),
        ],
        Arc::new(Mutex::new(Vec::new())),
        Duration::from_millis(0),
    );
    let runtime = AgentRuntime::new(adapter);
    let temp_dir = temp_dir("runtime-server-scope");
    let registry_path = write_registry(&temp_dir, FAKE_SERVER_BIN, Vec::new());
    write_mcp_metadata(&temp_dir, "fake");

    let outcome = runtime
        .run_turn(RunRequest {
            system_prompt_path: write_prompt(&temp_dir, "MCPs:\n<dynamic variable: available MCPs>\nSub-agents:\n<dynamic variable: available sub-agents>"),
            working_directory: temp_dir.clone(),
            conversation_history: vec![],
            recent_session_messages: vec![],
            user_message: "Find the answer".to_owned(),
            response_target: default_response_target(),
            registry_path,
            subagent_registry_path: write_subagent_registry(&temp_dir),
            tool_policy_path: None,
            enabled_servers: Some(vec![ServerName::new("fake").expect("valid server")]),
            limits: RuntimeLimits::default(),
            model_config: ModelConfig::new("fake-model"),
            require_todos: false,
        })
        .await
        .expect("turn should succeed");

    let tool_results = outcome
        .turn
        .messages
        .iter()
        .filter_map(|message| match message {
            MessageRecord::McpResult(record) => Some(record.target.capability_id.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(tool_results, vec!["preview_leads", "run-sql"]);
    assert!(outcome.turn.messages.iter().any(|message| matches!(
        message,
        MessageRecord::SubAgentResult(record)
            if record.status == "completed"
                && record.executed_action_count == 2
                && record.detail.contains("Completed one-server MCP workflow")
    )));
}

#[tokio::test]
async fn runtime_restricts_mcp_subagent_to_selected_capability() {
    let _metadata_guard = metadata_test_lock().lock().await;
    cleanup_mcp_metadata("fake");
    let adapter = FakeModelAdapter::new(
        vec![
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::DelegateSubagent {
                    delegation: SubagentDelegationRequest {
                        subagent_type: "mcp-executor".to_owned(),
                        target: DelegationTarget::McpCapability(McpCapabilityTarget {
                            server_name: ServerName::new("fake").expect("valid server"),
                            capability_kind: mcp_metadata::CapabilityKind::Tool,
                            capability_id: "run-sql".to_owned(),
                        }),
                        goal: "Recover from a poisoned MCP session".to_owned(),
                    },
                },
                usage: token_usage(2, 2),
            }),
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::Final {
                    content: "recovered".to_owned(),
                },
                usage: token_usage(1, 1),
            }),
        ],
        vec![
            Ok(SubagentAdapterResponse {
                decision: SubagentDecision::McpToolCall {
                    server_name: ServerName::new("fake").expect("valid server"),
                    tool_name: "run-sql".to_owned(),
                    arguments: json!({"query": "select 1"}),
                },
                usage: token_usage(1, 1),
            }),
            Ok(SubagentAdapterResponse {
                decision: SubagentDecision::McpToolCall {
                    server_name: ServerName::new("fake").expect("valid server"),
                    tool_name: "fail-tool".to_owned(),
                    arguments: json!({}),
                },
                usage: token_usage(1, 1),
            }),
            Ok(SubagentAdapterResponse {
                decision: SubagentDecision::McpToolCall {
                    server_name: ServerName::new("fake").expect("valid server"),
                    tool_name: "run-sql".to_owned(),
                    arguments: json!({"query": "select 2"}),
                },
                usage: token_usage(1, 1),
            }),
            Ok(SubagentAdapterResponse {
                decision: SubagentDecision::Done {
                    summary: "Recovered after reconnect".to_owned(),
                },
                usage: token_usage(1, 1),
            }),
        ],
        Arc::new(Mutex::new(Vec::new())),
        Duration::from_millis(0),
    );
    let runtime = AgentRuntime::new(adapter);
    let temp_dir = temp_dir("runtime-reset-on-tool-error");
    let log_path = temp_dir.join("mcp-log.txt");
    let registry_path = write_registry(
        &temp_dir,
        FAKE_SERVER_BIN,
        vec![
            "--log-file".to_owned(),
            log_path.to_str().expect("utf-8 path").to_owned(),
            "--tool-mode".to_owned(),
            "poison-after-tool-error".to_owned(),
        ],
    );
    write_mcp_metadata(&temp_dir, "fake");

    let outcome = runtime
        .run_turn(RunRequest {
            system_prompt_path: write_prompt(&temp_dir, "MCPs:\n<dynamic variable: available MCPs>\nSub-agents:\n<dynamic variable: available sub-agents>"),
            working_directory: temp_dir.clone(),
            conversation_history: vec![],
            recent_session_messages: vec![],
            user_message: "Run the recovery flow".to_owned(),
            response_target: default_response_target(),
            registry_path,
            subagent_registry_path: write_subagent_registry(&temp_dir),
            tool_policy_path: None,
            enabled_servers: Some(vec![ServerName::new("fake").expect("valid server")]),
            limits: RuntimeLimits::default(),
            model_config: ModelConfig::new("fake-model"),
            require_todos: false,
        })
        .await
        .expect("turn should succeed");

    assert_eq!(outcome.final_text, "recovered");
    let tool_results = outcome
        .turn
        .messages
        .iter()
        .filter_map(|message| match message {
            MessageRecord::McpResult(record) => Some((
                record.target.capability_id.as_str().to_owned(),
                record.error.is_some(),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        tool_results,
        vec![("run-sql".to_owned(), false)],
        "delegated MCP execution should stay pinned to the selected capability"
    );
}

#[tokio::test]
async fn runtime_stops_mcp_subagent_after_repeated_mcp_errors() {
    let _metadata_guard = metadata_test_lock().lock().await;
    cleanup_mcp_metadata("fake");
    let adapter = FakeModelAdapter::new(
        vec![
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::DelegateSubagent {
                    delegation: SubagentDelegationRequest {
                        subagent_type: "mcp-executor".to_owned(),
                        target: DelegationTarget::McpServerScope(McpServerScopeTarget {
                            server_name: ServerName::new("fake").expect("valid server"),
                        }),
                        goal: "Collect season data".to_owned(),
                    },
                },
                usage: token_usage(1, 1),
            }),
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::Final {
                    content: "stopped after repeated MCP errors".to_owned(),
                },
                usage: token_usage(1, 1),
            }),
        ],
        vec![
            Ok(SubagentAdapterResponse {
                decision: SubagentDecision::McpToolCall {
                    server_name: ServerName::new("fake").expect("valid server"),
                    tool_name: "run-sql".to_owned(),
                    arguments: json!({"query": "select 1"}),
                },
                usage: token_usage(1, 1),
            }),
            Ok(SubagentAdapterResponse {
                decision: SubagentDecision::McpToolCall {
                    server_name: ServerName::new("fake").expect("valid server"),
                    tool_name: "run-sql".to_owned(),
                    arguments: json!({"query": "select 2"}),
                },
                usage: token_usage(1, 1),
            }),
            Ok(SubagentAdapterResponse {
                decision: SubagentDecision::McpToolCall {
                    server_name: ServerName::new("fake").expect("valid server"),
                    tool_name: "run-sql".to_owned(),
                    arguments: json!({"query": "select 3"}),
                },
                usage: token_usage(1, 1),
            }),
        ],
        Arc::new(Mutex::new(Vec::new())),
        Duration::from_millis(0),
    );
    let runtime = AgentRuntime::new(adapter);
    let temp_dir = temp_dir("runtime-mcp-error-circuit-breaker");
    let registry_path = write_registry(
        &temp_dir,
        FAKE_SERVER_BIN,
        vec!["--tool-mode".to_owned(), "always-tool-error".to_owned()],
    );
    write_mcp_metadata(&temp_dir, "fake");

    let outcome = runtime
        .run_turn(RunRequest {
            system_prompt_path: write_prompt(
                &temp_dir,
                "MCPs:\n<dynamic variable: available MCPs>\nSub-agents:\n<dynamic variable: available sub-agents>",
            ),
            working_directory: temp_dir.clone(),
            conversation_history: vec![],
            recent_session_messages: vec![],
            user_message: "Do the full analysis".to_owned(),
            response_target: default_response_target(),
            registry_path,
            subagent_registry_path: write_subagent_registry(&temp_dir),
            tool_policy_path: None,
            enabled_servers: Some(vec![ServerName::new("fake").expect("valid server")]),
            limits: RuntimeLimits::default(),
            model_config: ModelConfig::new("fake-model"),
            require_todos: false,
        })
        .await
        .expect("turn should succeed");

    assert_eq!(outcome.final_text, "stopped after repeated MCP errors");
    let mcp_results = outcome
        .turn
        .messages
        .iter()
        .filter(|message| matches!(message, MessageRecord::McpResult(_)))
        .count();
    assert_eq!(mcp_results, 3);
    assert!(outcome.turn.messages.iter().any(|message| matches!(
        message,
        MessageRecord::SubAgentResult(record)
            if record.status == "partial"
                && record.detail.contains("repeated MCP failures forced an early stop")
                && record.executed_action_count == 3
    )));
}

#[tokio::test]
async fn runtime_stops_tool_executor_after_repeated_local_tool_errors() {
    let _metadata_guard = metadata_test_lock().lock().await;
    cleanup_mcp_metadata("fake");
    let adapter = FakeModelAdapter::new(
        vec![
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::DelegateSubagent {
                    delegation: SubagentDelegationRequest {
                        subagent_type: "tool-executor".to_owned(),
                        target: DelegationTarget::LocalToolsScope(
                            LocalToolsScopeTarget::working_directory(),
                        ),
                        goal: "Do the local CSK analysis".to_owned(),
                    },
                },
                usage: token_usage(1, 1),
            }),
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::Final {
                    content: "stopped after repeated local tool errors".to_owned(),
                },
                usage: token_usage(1, 1),
            }),
        ],
        vec![
            Ok(SubagentAdapterResponse {
                decision: SubagentDecision::LocalToolCall {
                    tool_name: LocalToolName::new("write_todos").expect("valid tool"),
                    arguments: json!({}),
                },
                usage: token_usage(1, 1),
            }),
            Ok(SubagentAdapterResponse {
                decision: SubagentDecision::LocalToolCall {
                    tool_name: LocalToolName::new("glob").expect("valid tool"),
                    arguments: json!({}),
                },
                usage: token_usage(1, 1),
            }),
            Ok(SubagentAdapterResponse {
                decision: SubagentDecision::LocalToolCall {
                    tool_name: LocalToolName::new("bash").expect("valid tool"),
                    arguments: json!({}),
                },
                usage: token_usage(1, 1),
            }),
            Ok(SubagentAdapterResponse {
                decision: SubagentDecision::LocalToolCall {
                    tool_name: LocalToolName::new("write_todos").expect("valid tool"),
                    arguments: json!({"status":"completed"}),
                },
                usage: token_usage(1, 1),
            }),
        ],
        Arc::new(Mutex::new(Vec::new())),
        Duration::from_millis(0),
    );
    let runtime = AgentRuntime::new(adapter);
    let temp_dir = temp_dir("runtime-local-tool-error-circuit-breaker");
    let registry_path = write_registry(&temp_dir, FAKE_SERVER_BIN, Vec::new());
    write_mcp_metadata(&temp_dir, "fake");

    let outcome = runtime
        .run_turn(RunRequest {
            system_prompt_path: write_prompt(
                &temp_dir,
                "Sub-agents:\n<dynamic variable: available sub-agents>\nTools:\n<dynamic variable: available tools>",
            ),
            working_directory: temp_dir.clone(),
            conversation_history: vec![],
            recent_session_messages: vec![],
            user_message: "Do the analysis".to_owned(),
            response_target: default_response_target(),
            registry_path,
            subagent_registry_path: write_subagent_registry(&temp_dir),
            tool_policy_path: None,
            enabled_servers: Some(vec![ServerName::new("fake").expect("valid server")]),
            limits: RuntimeLimits::default(),
            model_config: ModelConfig::new("fake-model"),
            require_todos: false,
        })
        .await
        .expect("turn should succeed");

    assert_eq!(outcome.final_text, "stopped after repeated local tool errors");
    let local_tool_results = outcome
        .turn
        .messages
        .iter()
        .filter(|message| matches!(message, MessageRecord::LocalToolResult(_)))
        .count();
    assert_eq!(local_tool_results, 4);
    assert!(outcome.turn.messages.iter().any(|message| matches!(
        message,
        MessageRecord::SubAgentResult(record)
            if record.status == "partial"
                && record.detail.contains("repeated local tool failures forced an early stop")
                && record.executed_action_count == 4
    )));
}

#[tokio::test]
async fn runtime_treats_invalid_subagent_structured_output_as_partial_instead_of_failing_turn() {
    let _metadata_guard = metadata_test_lock().lock().await;
    cleanup_mcp_metadata("fake");
    let adapter = FakeModelAdapter::new(
        vec![
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::DelegateSubagent {
                    delegation: SubagentDelegationRequest {
                        subagent_type: "tool-executor".to_owned(),
                        target: DelegationTarget::LocalToolsScope(
                            LocalToolsScopeTarget::working_directory(),
                        ),
                        goal: "Do the local CSK analysis".to_owned(),
                    },
                },
                usage: token_usage(1, 1),
            }),
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::Final {
                    content: "continued after invalid subagent output".to_owned(),
                },
                usage: token_usage(1, 1),
            }),
        ],
        vec![Err(ModelAdapterError::InvalidDecision(
            "local tool name cannot be blank".to_owned(),
        ))],
        Arc::new(Mutex::new(Vec::new())),
        Duration::from_millis(0),
    );
    let runtime = AgentRuntime::new(adapter);
    let temp_dir = temp_dir("runtime-invalid-subagent-structured-output");
    let registry_path = write_registry(&temp_dir, FAKE_SERVER_BIN, Vec::new());
    write_mcp_metadata(&temp_dir, "fake");

    let outcome = runtime
        .run_turn(RunRequest {
            system_prompt_path: write_prompt(
                &temp_dir,
                "Sub-agents:\n<dynamic variable: available sub-agents>\nTools:\n<dynamic variable: available tools>",
            ),
            working_directory: temp_dir.clone(),
            conversation_history: vec![],
            recent_session_messages: vec![],
            user_message: "Do the analysis".to_owned(),
            response_target: default_response_target(),
            registry_path,
            subagent_registry_path: write_subagent_registry(&temp_dir),
            tool_policy_path: None,
            enabled_servers: Some(vec![ServerName::new("fake").expect("valid server")]),
            limits: RuntimeLimits::default(),
            model_config: ModelConfig::new("fake-model"),
            require_todos: false,
        })
        .await
        .expect("turn should succeed");

    assert_eq!(outcome.final_text, "continued after invalid subagent output");
    assert!(outcome.turn.messages.iter().any(|message| matches!(
        message,
        MessageRecord::SubAgentResult(record)
            if record.status == "partial"
                && record.executed_action_count == 0
                && record.detail.contains("invalid structured output")
                && record.detail.contains("local tool name cannot be blank")
    )));
}

#[tokio::test]
async fn runtime_blocks_duplicate_mcp_calls_with_identical_arguments() {
    let _metadata_guard = metadata_test_lock().lock().await;
    cleanup_mcp_metadata("fake");
    let adapter = FakeModelAdapter::new(
        vec![
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::DelegateSubagent {
                    delegation: SubagentDelegationRequest {
                        subagent_type: "mcp-executor".to_owned(),
                        target: DelegationTarget::McpServerScope(McpServerScopeTarget {
                            server_name: ServerName::new("fake").expect("valid server"),
                        }),
                        goal: "Avoid duplicate tool calls".to_owned(),
                    },
                },
                usage: token_usage(2, 2),
            }),
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::Final {
                    content: "duplicate blocked".to_owned(),
                },
                usage: token_usage(1, 1),
            }),
        ],
        vec![
            Ok(SubagentAdapterResponse {
                decision: SubagentDecision::McpToolCall {
                    server_name: ServerName::new("fake").expect("valid server"),
                    tool_name: "preview_leads".to_owned(),
                    arguments: json!({"limit": 2}),
                },
                usage: token_usage(1, 1),
            }),
            Ok(SubagentAdapterResponse {
                decision: SubagentDecision::McpToolCall {
                    server_name: ServerName::new("fake").expect("valid server"),
                    tool_name: "preview_leads".to_owned(),
                    arguments: json!({"limit": 2}),
                },
                usage: token_usage(1, 1),
            }),
        ],
        Arc::new(Mutex::new(Vec::new())),
        Duration::from_millis(0),
    );
    let runtime = AgentRuntime::new(adapter);
    let temp_dir = temp_dir("runtime-duplicate-mcp");
    let registry_path = write_registry(&temp_dir, FAKE_SERVER_BIN, Vec::new());
    write_mcp_metadata(&temp_dir, "fake");

    let outcome = runtime
        .run_turn(RunRequest {
            system_prompt_path: write_prompt(&temp_dir, "MCPs:\n<dynamic variable: available MCPs>\nSub-agents:\n<dynamic variable: available sub-agents>"),
            working_directory: temp_dir.clone(),
            conversation_history: vec![],
            recent_session_messages: vec![],
            user_message: "Avoid duplicates".to_owned(),
            response_target: default_response_target(),
            registry_path,
            subagent_registry_path: write_subagent_registry(&temp_dir),
            tool_policy_path: None,
            enabled_servers: Some(vec![ServerName::new("fake").expect("valid server")]),
            limits: RuntimeLimits::default(),
            model_config: ModelConfig::new("fake-model"),
            require_todos: false,
        })
        .await
        .expect("turn should succeed");

    let tool_results = outcome
        .turn
        .messages
        .iter()
        .filter(|message| matches!(message, MessageRecord::McpResult(_)))
        .count();
    assert_eq!(tool_results, 1);
    assert!(outcome.turn.messages.iter().any(|message| matches!(
        message,
        MessageRecord::SubAgentResult(record)
            if record.status == "partial"
                && record.detail.contains("duplicate MCP tool call")
    )));
}

#[tokio::test]
async fn runtime_sanitizes_hybrid_mcp_arguments_before_execution() {
    let _metadata_guard = metadata_test_lock().lock().await;
    cleanup_mcp_metadata("fake");
    let adapter = FakeModelAdapter::new(
        vec![
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::DelegateSubagent {
                    delegation: SubagentDelegationRequest {
                        subagent_type: "mcp-executor".to_owned(),
                        target: DelegationTarget::McpServerScope(McpServerScopeTarget {
                            server_name: ServerName::new("fake").expect("valid server"),
                        }),
                        goal: "Run the direct query".to_owned(),
                    },
                },
                usage: token_usage(2, 2),
            }),
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::Final {
                    content: "sanitized".to_owned(),
                },
                usage: token_usage(1, 1),
            }),
        ],
        vec![
            Ok(SubagentAdapterResponse {
                decision: SubagentDecision::McpToolCall {
                    server_name: ServerName::new("fake").expect("valid server"),
                    tool_name: "run-sql".to_owned(),
                    arguments: json!({
                        "query": "select 1",
                        "database": "fake"
                    }),
                },
                usage: token_usage(1, 1),
            }),
            Ok(SubagentAdapterResponse {
                decision: SubagentDecision::Done {
                    summary: "Query executed after sanitization".to_owned(),
                },
                usage: token_usage(1, 1),
            }),
        ],
        Arc::new(Mutex::new(Vec::new())),
        Duration::from_millis(0),
    );
    let runtime = AgentRuntime::new(adapter);
    let temp_dir = temp_dir("runtime-sanitize-hybrid-mcp");
    let registry_path = write_registry(&temp_dir, FAKE_SERVER_BIN, Vec::new());
    write_mcp_metadata(&temp_dir, "fake");

    let outcome = runtime
        .run_turn(RunRequest {
            system_prompt_path: write_prompt(
                &temp_dir,
                "MCPs:\n<dynamic variable: available MCPs>\nSub-agents:\n<dynamic variable: available sub-agents>",
            ),
            working_directory: temp_dir.clone(),
            conversation_history: vec![],
            recent_session_messages: vec![],
            user_message: "Run the query".to_owned(),
            response_target: default_response_target(),
            registry_path,
            subagent_registry_path: write_subagent_registry(&temp_dir),
            tool_policy_path: None,
            enabled_servers: Some(vec![ServerName::new("fake").expect("valid server")]),
            limits: RuntimeLimits::default(),
            model_config: ModelConfig::new("fake-model"),
            require_todos: false,
        })
        .await
        .expect("turn should succeed");

    let mcp_call = outcome
        .turn
        .messages
        .iter()
        .find_map(|message| match message {
            MessageRecord::McpCall(record) => Some(record),
            _ => None,
        })
        .expect("sanitized call should be recorded");
    assert_eq!(mcp_call.arguments, json!({"query": "select 1"}));
    assert!(outcome.turn.messages.iter().any(|message| matches!(
        message,
        MessageRecord::McpResult(record)
            if record.error.is_none() && record.target.capability_id == "run-sql"
    )));
}

#[tokio::test]
async fn runtime_mcp_prompt_includes_confirmed_tables_from_recent_session_context() {
    let _metadata_guard = metadata_test_lock().lock().await;
    cleanup_mcp_metadata("fake");
    let prompt_log = Arc::new(Mutex::new(Vec::new()));
    let adapter = FakeModelAdapter::new(
        vec![
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::DelegateSubagent {
                    delegation: SubagentDelegationRequest {
                        subagent_type: "mcp-executor".to_owned(),
                        target: DelegationTarget::McpServerScope(McpServerScopeTarget {
                            server_name: ServerName::new("fake").expect("valid server"),
                        }),
                        goal: "Use the confirmed table".to_owned(),
                    },
                },
                usage: token_usage(2, 2),
            }),
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::Final {
                    content: "done".to_owned(),
                },
                usage: token_usage(1, 1),
            }),
        ],
        vec![Ok(SubagentAdapterResponse {
            decision: SubagentDecision::Done {
                summary: "Used confirmed table context".to_owned(),
            },
            usage: token_usage(1, 1),
        })],
        Arc::clone(&prompt_log),
        Duration::from_millis(0),
    );
    let runtime = AgentRuntime::new(adapter);
    let temp_dir = temp_dir("runtime-confirmed-table-context");
    let registry_path = write_registry(&temp_dir, FAKE_SERVER_BIN, Vec::new());
    write_mcp_metadata(&temp_dir, "fake");

    runtime
        .run_turn(RunRequest {
            system_prompt_path: write_prompt(
                &temp_dir,
                "MCPs:\n<dynamic variable: available MCPs>\nSub-agents:\n<dynamic variable: available sub-agents>",
            ),
            working_directory: temp_dir.clone(),
            conversation_history: vec![],
            recent_session_messages: vec![
                MessageRecord::McpCall(agent_runtime::state::McpCallMessageRecord {
                    message_id: agent_runtime::MessageId::new(),
                    timestamp: SystemTime::now(),
                    target: McpCapabilityTarget {
                        server_name: ServerName::new("fake").expect("valid server"),
                        capability_kind: mcp_metadata::CapabilityKind::Tool,
                        capability_id: "run_query".to_owned(),
                    },
                    arguments: json!({"query": "select count(*) from fake.users"}),
                }),
                MessageRecord::McpResult(agent_runtime::state::McpResultMessageRecord {
                    message_id: agent_runtime::MessageId::new(),
                    timestamp: SystemTime::now(),
                    target: McpCapabilityTarget {
                        server_name: ServerName::new("fake").expect("valid server"),
                        capability_kind: mcp_metadata::CapabilityKind::Tool,
                        capability_id: "run_query".to_owned(),
                    },
                    result_summary: "{\"columns\":[\"count\"],\"rows\":[[1]]}".to_owned(),
                    error: None,
                }),
            ],
            user_message: "How many users are there?".to_owned(),
            response_target: default_response_target(),
            registry_path,
            subagent_registry_path: write_subagent_registry(&temp_dir),
            tool_policy_path: None,
            enabled_servers: Some(vec![ServerName::new("fake").expect("valid server")]),
            limits: RuntimeLimits::default(),
            model_config: ModelConfig::new("fake-model"),
            require_todos: false,
        })
        .await
        .expect("turn should succeed");

    let prompts = prompt_log.lock().expect("prompt log should lock");
    let subagent_prompt = prompts
        .iter()
        .find(|prompt| prompt.contains("## Confirmed Tables"))
        .expect("subagent prompt should include confirmed tables");
    assert!(subagent_prompt.contains("fake.users"));
    assert!(subagent_prompt.contains("Prefer direct `run_query`"));
}

#[tokio::test]
async fn runtime_mcp_executor_cannot_use_local_tools() {
    let _metadata_guard = metadata_test_lock().lock().await;
    cleanup_mcp_metadata("fake");
    let adapter = FakeModelAdapter::new(
        vec![
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::DelegateSubagent {
                    delegation: SubagentDelegationRequest {
                        subagent_type: "mcp-executor".to_owned(),
                        target: DelegationTarget::McpServerScope(McpServerScopeTarget {
                            server_name: ServerName::new("fake").expect("valid server"),
                        }),
                        goal: "Try a forbidden local tool".to_owned(),
                    },
                },
                usage: token_usage(2, 2),
            }),
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::Final {
                    content: "blocked".to_owned(),
                },
                usage: token_usage(1, 1),
            }),
        ],
        vec![Ok(SubagentAdapterResponse {
            decision: SubagentDecision::LocalToolCall {
                tool_name: LocalToolName::new("write_file").expect("valid tool"),
                arguments: json!({"path":"outputs/result.txt","content":"hi"}),
            },
            usage: token_usage(1, 1),
        })],
        Arc::new(Mutex::new(Vec::new())),
        Duration::from_millis(0),
    );
    let runtime = AgentRuntime::new(adapter);
    let temp_dir = temp_dir("runtime-mcp-no-local-tools");
    let registry_path = write_registry(&temp_dir, FAKE_SERVER_BIN, Vec::new());
    write_mcp_metadata(&temp_dir, "fake");

    let outcome = runtime
        .run_turn(RunRequest {
            system_prompt_path: write_prompt(&temp_dir, "MCPs:\n<dynamic variable: available MCPs>\nSub-agents:\n<dynamic variable: available sub-agents>"),
            working_directory: temp_dir.clone(),
            conversation_history: vec![],
            recent_session_messages: vec![],
            user_message: "Try the forbidden path".to_owned(),
            response_target: default_response_target(),
            registry_path,
            subagent_registry_path: write_subagent_registry(&temp_dir),
            tool_policy_path: None,
            enabled_servers: Some(vec![ServerName::new("fake").expect("valid server")]),
            limits: RuntimeLimits::default(),
            model_config: ModelConfig::new("fake-model"),
            require_todos: false,
        })
        .await
        .expect("turn should succeed");

    assert!(outcome.turn.messages.iter().any(|message| matches!(
        message,
        MessageRecord::SubAgentResult(record)
            if record.status == "cannot_execute"
                && record.detail.contains("local tool `write_file` is not allowed")
    )));
}

#[tokio::test]
async fn runtime_tool_executor_cannot_use_mcp_tools() {
    let _metadata_guard = metadata_test_lock().lock().await;
    cleanup_mcp_metadata("fake");
    let adapter = FakeModelAdapter::new(
        vec![
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::DelegateSubagent {
                    delegation: SubagentDelegationRequest {
                        subagent_type: "tool-executor".to_owned(),
                        target: DelegationTarget::LocalToolsScope(
                            LocalToolsScopeTarget::working_directory(),
                        ),
                        goal: "Try a forbidden MCP tool".to_owned(),
                    },
                },
                usage: token_usage(2, 2),
            }),
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::Final {
                    content: "blocked".to_owned(),
                },
                usage: token_usage(1, 1),
            }),
        ],
        vec![Ok(SubagentAdapterResponse {
            decision: SubagentDecision::McpToolCall {
                server_name: ServerName::new("fake").expect("valid server"),
                tool_name: "run-sql".to_owned(),
                arguments: json!({"query":"select 1"}),
            },
            usage: token_usage(1, 1),
        })],
        Arc::new(Mutex::new(Vec::new())),
        Duration::from_millis(0),
    );
    let runtime = AgentRuntime::new(adapter);
    let temp_dir = temp_dir("runtime-tool-no-mcp");
    let registry_path = write_registry(&temp_dir, FAKE_SERVER_BIN, Vec::new());
    write_mcp_metadata(&temp_dir, "fake");

    let outcome = runtime
        .run_turn(RunRequest {
            system_prompt_path: write_prompt(
                &temp_dir,
                "Sub-agents:\n<dynamic variable: available sub-agents>\nTools:\n<dynamic variable: available tools>",
            ),
            working_directory: temp_dir.clone(),
            conversation_history: vec![],
            recent_session_messages: vec![],
            user_message: "Try the forbidden MCP path".to_owned(),
            response_target: default_response_target(),
            registry_path,
            subagent_registry_path: write_subagent_registry(&temp_dir),
            tool_policy_path: None,
            enabled_servers: Some(vec![ServerName::new("fake").expect("valid server")]),
            limits: RuntimeLimits::default(),
            model_config: ModelConfig::new("fake-model"),
            require_todos: false,
        })
        .await
        .expect("turn should succeed");

    assert!(outcome.turn.messages.iter().any(|message| matches!(
        message,
        MessageRecord::SubAgentResult(record)
            if record.status == "cannot_execute"
                && record.detail.contains("MCP tool `run-sql` on server `fake` is not allowed")
    )));
}

#[tokio::test]
async fn runtime_delegates_tool_executor_and_executes_glob() {
    let _metadata_guard = metadata_test_lock().lock().await;
    cleanup_mcp_metadata("fake");
    let prompt_log = Arc::new(Mutex::new(Vec::new()));
    let adapter = FakeModelAdapter::new(
        vec![
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::DelegateSubagent {
                    delegation: SubagentDelegationRequest {
                        subagent_type: "tool-executor".to_owned(),
                        target: DelegationTarget::LocalToolsScope(
                            LocalToolsScopeTarget::working_directory(),
                        ),
                        goal: "Find Rust source files".to_owned(),
                    },
                },
                usage: token_usage(2, 2),
            }),
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::Final {
                    content: "glob complete".to_owned(),
                },
                usage: token_usage(1, 1),
            }),
        ],
        vec![
            Ok(SubagentAdapterResponse {
                decision: SubagentDecision::LocalToolCall {
                    tool_name: LocalToolName::new("glob").expect("valid tool"),
                    arguments: json!({"pattern":"src/**/*.rs"}),
                },
                usage: token_usage(1, 1),
            }),
            Ok(SubagentAdapterResponse {
                decision: SubagentDecision::Done {
                    summary: "Found Rust files".to_owned(),
                },
                usage: token_usage(1, 1),
            }),
        ],
        Arc::clone(&prompt_log),
        Duration::from_millis(0),
    );

    let runtime = AgentRuntime::new(adapter);
    let temp_dir = temp_dir("runtime-tool-glob");
    fs::create_dir_all(temp_dir.join("src/nested")).expect("nested src dir");
    fs::write(temp_dir.join("src/lib.rs"), "pub fn a() {}\n").expect("lib file");
    fs::write(temp_dir.join("src/nested/mod.rs"), "pub fn b() {}\n").expect("mod file");
    let registry_path = write_registry(&temp_dir, FAKE_SERVER_BIN, Vec::new());
    write_mcp_metadata(&temp_dir, "fake");

    let outcome = runtime
        .run_turn(RunRequest {
            system_prompt_path: write_prompt(
                &temp_dir,
                "Sub-agents:\n<dynamic variable: available sub-agents>\nTools:\n<dynamic variable: available tools>",
            ),
            working_directory: temp_dir.clone(),
            conversation_history: vec![],
            recent_session_messages: vec![],
            user_message: "Find Rust files".to_owned(),
            response_target: default_response_target(),
            registry_path,
            subagent_registry_path: write_subagent_registry(&temp_dir),
            tool_policy_path: None,
            enabled_servers: Some(vec![ServerName::new("fake").expect("valid server")]),
            limits: RuntimeLimits::default(),
            model_config: ModelConfig::new("fake-model"),
            require_todos: false,
        })
        .await
        .expect("turn should succeed");

    assert_eq!(outcome.final_text, "glob complete");
    assert!(outcome.turn.messages.iter().any(|message| matches!(
        message,
        MessageRecord::LocalToolCall(record) if record.tool_name.as_str() == "glob"
    )));
    assert!(outcome.turn.messages.iter().any(|message| matches!(
        message,
        MessageRecord::LocalToolResult(record)
            if record.tool_name.as_str() == "glob"
                && record.result_summary.contains("src/lib.rs")
                && record.result_summary.contains("src/nested/mod.rs")
    )));
    assert!(outcome.events.iter().any(|event| matches!(
        event,
        RuntimeEvent::LocalToolResponded {
            tool_name,
            executor,
            result_summary,
            ..
        } if tool_name == "glob"
            && executor.subagent_type.as_deref() == Some("tool-executor")
            && result_summary.contains("src/lib.rs")
    )));
}

#[tokio::test]
async fn complex_turn_creates_todo_file_and_injects_follow_up_prompt_context() {
    let _metadata_guard = metadata_test_lock().lock().await;
    cleanup_mcp_metadata("fake");
    let prompt_log = Arc::new(Mutex::new(Vec::new()));
    let adapter = FakeModelAdapter::new(
        vec![
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::DelegateSubagent {
                    delegation: SubagentDelegationRequest {
                        subagent_type: "tool-executor".to_owned(),
                        target: DelegationTarget::LocalToolsScope(
                            LocalToolsScopeTarget::working_directory(),
                        ),
                        goal: "Create a complex-turn todo plan".to_owned(),
                    },
                },
                usage: token_usage(2, 2),
            }),
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::Final {
                    content: "todo planning complete".to_owned(),
                },
                usage: token_usage(1, 1),
            }),
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::Final {
                    content: "todo planning complete".to_owned(),
                },
                usage: token_usage(1, 1),
            }),
        ],
        vec![
            Ok(SubagentAdapterResponse {
                decision: SubagentDecision::LocalToolCall {
                    tool_name: LocalToolName::new("write_todos").expect("valid tool"),
                    arguments: json!({
                        "operation":"initialize",
                        "items":["Inspect the source data","Compute summary metrics"]
                    }),
                },
                usage: token_usage(1, 1),
            }),
            Ok(SubagentAdapterResponse {
                decision: SubagentDecision::Done {
                    summary: "Initialized complex-turn todos".to_owned(),
                },
                usage: token_usage(1, 1),
            }),
        ],
        Arc::clone(&prompt_log),
        Duration::from_millis(0),
    );

    let runtime = AgentRuntime::new(adapter);
    let temp_dir = temp_dir("runtime-complex-todo");
    let registry_path = write_registry(&temp_dir, FAKE_SERVER_BIN, Vec::new());
    write_mcp_metadata(&temp_dir, "fake");

    let outcome = runtime
        .run_turn(RunRequest {
            system_prompt_path: write_prompt(
                &temp_dir,
                "Sub-agents:\n<dynamic variable: available sub-agents>\nTools:\n<dynamic variable: available tools>",
            ),
            working_directory: temp_dir.clone(),
            conversation_history: vec![],
            recent_session_messages: vec![],
            user_message: "Perform a complex analysis".to_owned(),
            response_target: default_response_target(),
            registry_path,
            subagent_registry_path: write_subagent_registry(&temp_dir),
            tool_policy_path: None,
            enabled_servers: Some(vec![ServerName::new("fake").expect("valid server")]),
            limits: RuntimeLimits {
                max_steps_per_turn: 3,
                ..RuntimeLimits::default()
            },
            model_config: ModelConfig::new("fake-model"),
            require_todos: false,
        })
        .await
        .expect("turn should succeed");

    let todo_path = temp_dir
        .join(outcome.turn.turn_id.to_string())
        .join("todos.txt");
    let todo_text = fs::read_to_string(&todo_path).expect("todo file should exist");
    assert!(todo_text.contains("[pending] Inspect the source data"));
    assert!(todo_text.contains("[pending] Generate an output HTML page with charts and tables."));
    assert!(todo_text.contains("[pending] Open the generated output HTML page in the browser."));

    let prompts = prompt_log.lock().expect("prompt log should lock");
    assert!(prompts.iter().any(|prompt| {
        prompt.contains("## Current Turn Todo Plan")
            && prompt.contains("Inspect the source data")
            && prompt.contains("Deterministic HTML output")
    }));
    assert!(prompts.iter().any(|prompt| {
        prompt.contains("This delegation is todo-planning only.")
            && prompt.contains("return `done` immediately")
            && prompt.contains("do not begin executing the planned analysis")
    }));
}

#[tokio::test]
async fn non_todo_analysis_turn_still_gets_turn_policy_guidance() {
    let _metadata_guard = metadata_test_lock().lock().await;
    cleanup_mcp_metadata("fake");
    let prompt_log = Arc::new(Mutex::new(Vec::new()));
    let adapter = FakeModelAdapter::new(
        vec![Ok(ModelAdapterResponse {
            decision: ModelStepDecision::Final {
                content: "Analysis complete".to_owned(),
            },
            usage: token_usage(1, 1),
        })],
        vec![],
        Arc::clone(&prompt_log),
        Duration::from_millis(0),
    );
    let runtime = AgentRuntime::new(adapter);
    let temp_dir = temp_dir("runtime-analysis-html-guidance");
    let registry_path = write_registry(&temp_dir, FAKE_SERVER_BIN, Vec::new());
    write_mcp_metadata(&temp_dir, "fake");

    let outcome = runtime
        .run_turn(RunRequest {
            system_prompt_path: write_prompt(&temp_dir, "Be precise."),
            working_directory: temp_dir.clone(),
            conversation_history: vec![],
            recent_session_messages: vec![],
            user_message: "Analyze the uploaded dataset".to_owned(),
            response_target: default_response_target(),
            registry_path,
            subagent_registry_path: write_subagent_registry(&temp_dir),
            tool_policy_path: None,
            enabled_servers: Some(vec![ServerName::new("fake").expect("valid server")]),
            limits: RuntimeLimits::default(),
            model_config: ModelConfig::new("fake-model"),
            require_todos: false,
        })
        .await
        .expect("turn should succeed");

    let todo_path = temp_dir
        .join(outcome.turn.turn_id.to_string())
        .join("todos.txt");
    assert!(
        !todo_path.exists(),
        "non-todo analysis turn should not force todos.txt"
    );

    let prompts = prompt_log.lock().expect("prompt log should lock");
    assert!(prompts.iter().any(|prompt| {
        prompt.contains("## Turn Policy")
            && prompt.contains("Todos required: false")
            && prompt.contains("Deterministic HTML output:")
            && prompt.contains("generate the HTML report and open it in the browser")
    }));
}

#[tokio::test]
async fn simple_turn_does_not_create_todo_file() {
    let _metadata_guard = metadata_test_lock().lock().await;
    cleanup_mcp_metadata("fake");
    let prompt_log = Arc::new(Mutex::new(Vec::new()));
    let adapter = FakeModelAdapter::new(
        vec![Ok(ModelAdapterResponse {
            decision: ModelStepDecision::Final {
                content: "Simple answer".to_owned(),
            },
            usage: token_usage(1, 1),
        })],
        vec![],
        Arc::clone(&prompt_log),
        Duration::from_millis(0),
    );
    let runtime = AgentRuntime::new(adapter);
    let temp_dir = temp_dir("runtime-simple-no-todo");
    let registry_path = write_registry(&temp_dir, FAKE_SERVER_BIN, Vec::new());
    write_mcp_metadata(&temp_dir, "fake");

    let outcome = runtime
        .run_turn(RunRequest {
            system_prompt_path: write_prompt(&temp_dir, "Be concise."),
            working_directory: temp_dir.clone(),
            conversation_history: vec![],
            recent_session_messages: vec![],
            user_message: "What is a mean?".to_owned(),
            response_target: default_response_target(),
            registry_path,
            subagent_registry_path: write_subagent_registry(&temp_dir),
            tool_policy_path: None,
            enabled_servers: Some(vec![ServerName::new("fake").expect("valid server")]),
            limits: RuntimeLimits::default(),
            model_config: ModelConfig::new("fake-model"),
            require_todos: false,
        })
        .await
        .expect("turn should succeed");

    let todo_path = temp_dir
        .join(outcome.turn.turn_id.to_string())
        .join("todos.txt");
    assert!(
        !todo_path.exists(),
        "simple turn should not create todos.txt"
    );

    let prompts = prompt_log.lock().expect("prompt log should lock");
    assert!(prompts.iter().any(|prompt| {
        prompt.contains("## Turn Policy")
            && prompt.contains("Todos required: false")
            && prompt.contains("Skip is allowed only for very simple factual replies")
    }));
}

#[tokio::test]
async fn required_todos_mode_creates_starter_todo_file_for_simple_turns() {
    let _metadata_guard = metadata_test_lock().lock().await;
    cleanup_mcp_metadata("fake");
    let prompt_log = Arc::new(Mutex::new(Vec::new()));
    let adapter = FakeModelAdapter::new(
        vec![
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::Final {
                    content: "Simple answer".to_owned(),
                },
                usage: token_usage(1, 1),
            }),
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::Final {
                    content: "Simple answer".to_owned(),
                },
                usage: token_usage(1, 1),
            }),
        ],
        vec![],
        Arc::clone(&prompt_log),
        Duration::from_millis(0),
    );
    let runtime = AgentRuntime::new(adapter);
    let temp_dir = temp_dir("runtime-simple-required-todos");
    let registry_path = write_registry(&temp_dir, FAKE_SERVER_BIN, Vec::new());
    write_mcp_metadata(&temp_dir, "fake");

    let outcome = runtime
        .run_turn(RunRequest {
            system_prompt_path: write_prompt(&temp_dir, "Be concise."),
            working_directory: temp_dir.clone(),
            conversation_history: vec![],
            recent_session_messages: vec![],
            user_message: "What is a mean?".to_owned(),
            response_target: default_response_target(),
            registry_path,
            subagent_registry_path: write_subagent_registry(&temp_dir),
            tool_policy_path: None,
            enabled_servers: Some(vec![ServerName::new("fake").expect("valid server")]),
            limits: RuntimeLimits {
                max_steps_per_turn: 2,
                ..RuntimeLimits::default()
            },
            model_config: ModelConfig::new("fake-model"),
            require_todos: true,
        })
        .await
        .expect("turn should succeed");

    let todo_path = temp_dir
        .join(outcome.turn.turn_id.to_string())
        .join("todos.txt");
    let todo_text = fs::read_to_string(&todo_path).expect("required mode should create todos.txt");
    assert!(todo_text.contains("[pending] Understand and complete the user request."));
    assert!(todo_text.contains("[pending] Generate an output HTML page with charts and tables."));
    assert!(todo_text.contains("[pending] Open the generated output HTML page in the browser."));

    let prompts = prompt_log.lock().expect("prompt log should lock");
    assert!(prompts.iter().any(|prompt| {
        prompt.contains("## Turn Policy")
            && prompt.contains("Todos required: true")
            && prompt.contains("## Current Turn Todo Plan")
            && prompt.contains("Understand and complete the user request.")
    }));
}

#[tokio::test]
async fn runtime_rejects_premature_final_when_todos_are_incomplete() {
    let _metadata_guard = metadata_test_lock().lock().await;
    cleanup_mcp_metadata("fake");
    let prompt_log = Arc::new(Mutex::new(Vec::new()));
    let adapter = FakeModelAdapter::new(
        vec![
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::Final {
                    content: "Virat Kohli scored 657 runs.".to_owned(),
                },
                usage: token_usage(1, 1),
            }),
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::Final {
                    content: "Virat Kohli scored 657 runs.".to_owned(),
                },
                usage: token_usage(1, 1),
            }),
        ],
        vec![],
        Arc::clone(&prompt_log),
        Duration::from_millis(0),
    );
    let runtime = AgentRuntime::new(adapter);
    let temp_dir = temp_dir("runtime-reject-premature-final");
    let registry_path = write_registry(&temp_dir, FAKE_SERVER_BIN, Vec::new());
    write_mcp_metadata(&temp_dir, "fake");

    let outcome = runtime
        .run_turn(RunRequest {
            system_prompt_path: write_prompt(&temp_dir, "Be concise."),
            working_directory: temp_dir.clone(),
            conversation_history: vec![],
            recent_session_messages: vec![],
            user_message: "How many runs did Virat Kohli score in IPL 2025?".to_owned(),
            response_target: default_response_target(),
            registry_path,
            subagent_registry_path: write_subagent_registry(&temp_dir),
            tool_policy_path: None,
            enabled_servers: Some(vec![ServerName::new("fake").expect("valid server")]),
            limits: RuntimeLimits {
                max_steps_per_turn: 2,
                ..RuntimeLimits::default()
            },
            model_config: ModelConfig::new("fake-model"),
            require_todos: true,
        })
        .await
        .expect("turn should succeed");

    assert_eq!(outcome.termination, TerminationReason::MaxStepsReached);
    assert!(outcome.final_text.is_empty());
    assert!(outcome.turn.steps.len() >= 2);
    assert_eq!(
        outcome.turn.steps[0].outcome,
        agent_runtime::StepOutcomeKind::Continue
    );
    assert!(outcome.turn.messages.iter().any(|message| matches!(
        message,
        MessageRecord::Llm(record)
            if record.content.contains("generic starter scaffold")
    )));

    let prompts = prompt_log.lock().expect("prompt log should lock");
    assert!(prompts.len() >= 2);
    assert!(prompts[1].contains("generic starter scaffold"));
    assert!(prompts[1].contains("replan_pending_suffix"));
    assert!(prompts[1].contains("return control immediately"));
}

#[tokio::test]
async fn runtime_blocks_substantive_delegation_until_generic_scaffold_is_replanned() {
    let _metadata_guard = metadata_test_lock().lock().await;
    cleanup_mcp_metadata("fake");
    let prompt_log = Arc::new(Mutex::new(Vec::new()));
    let adapter = FakeModelAdapter::new(
        vec![
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::DelegateSubagent {
                    delegation: SubagentDelegationRequest {
                        subagent_type: "mcp-executor".to_owned(),
                        target: DelegationTarget::McpServerScope(McpServerScopeTarget {
                            server_name: ServerName::new("fake").expect("valid server"),
                        }),
                        goal: "Inspect the database and compute the season analysis".to_owned(),
                    },
                },
                usage: token_usage(1, 1),
            }),
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::Final {
                    content: "unused".to_owned(),
                },
                usage: token_usage(1, 1),
            }),
        ],
        vec![],
        Arc::clone(&prompt_log),
        Duration::from_millis(0),
    );
    let runtime = AgentRuntime::new(adapter);
    let temp_dir = temp_dir("runtime-block-generic-scaffold");
    let registry_path = write_registry(&temp_dir, FAKE_SERVER_BIN, Vec::new());
    write_mcp_metadata(&temp_dir, "fake");

    let outcome = runtime
        .run_turn(RunRequest {
            system_prompt_path: write_prompt(&temp_dir, "Be concise."),
            working_directory: temp_dir.clone(),
            conversation_history: vec![],
            recent_session_messages: vec![],
            user_message: "Do a full season analysis for CSK in IPL 2025".to_owned(),
            response_target: default_response_target(),
            registry_path,
            subagent_registry_path: write_subagent_registry(&temp_dir),
            tool_policy_path: None,
            enabled_servers: Some(vec![ServerName::new("fake").expect("valid server")]),
            limits: RuntimeLimits {
                max_steps_per_turn: 2,
                ..RuntimeLimits::default()
            },
            model_config: ModelConfig::new("fake-model"),
            require_todos: true,
        })
        .await
        .expect("turn should succeed");

    assert_eq!(outcome.termination, TerminationReason::MaxStepsReached);
    assert!(
        !outcome
            .events
            .iter()
            .any(|event| matches!(event, RuntimeEvent::HandoffToSubagent { subagent_type, .. } if subagent_type == "mcp-executor"))
    );
    assert!(outcome.turn.messages.iter().any(|message| matches!(
        message,
        MessageRecord::Llm(record)
            if record.content.contains("generic starter scaffold")
                && record.content.contains("replan_pending_suffix")
    )));

    let prompts = prompt_log.lock().expect("prompt log should lock");
    assert!(prompts.len() >= 2);
    assert!(prompts[1].contains("generic starter scaffold"));
}

#[tokio::test]
async fn runtime_blocks_replanning_again_after_concrete_plan_exists() {
    let _metadata_guard = metadata_test_lock().lock().await;
    cleanup_mcp_metadata("fake");
    let prompt_log = Arc::new(Mutex::new(Vec::new()));
    let adapter = FakeModelAdapter::new(
        vec![
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::DelegateSubagent {
                    delegation: SubagentDelegationRequest {
                        subagent_type: "tool-executor".to_owned(),
                        target: DelegationTarget::LocalToolsScope(
                            LocalToolsScopeTarget::working_directory(),
                        ),
                        goal: "Create a concrete todo plan for the CSK analysis".to_owned(),
                    },
                },
                usage: token_usage(1, 1),
            }),
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::DelegateSubagent {
                    delegation: SubagentDelegationRequest {
                        subagent_type: "tool-executor".to_owned(),
                        target: DelegationTarget::LocalToolsScope(
                            LocalToolsScopeTarget::working_directory(),
                        ),
                        goal: "Replan the todo list again before doing any work".to_owned(),
                    },
                },
                usage: token_usage(1, 1),
            }),
        ],
        vec![
            Ok(SubagentAdapterResponse {
                decision: SubagentDecision::LocalToolCall {
                    tool_name: LocalToolName::new("write_todos").expect("valid tool"),
                    arguments: json!({
                        "operation": "replan_pending_suffix",
                        "items": [
                            "Define the CSK IPL 2025 analysis scope and key questions to answer",
                            "Collect and prepare CSK IPL 2025 season data and supporting context",
                            "Compute team and player insights with tables and charts"
                        ]
                    }),
                },
                usage: token_usage(1, 1),
            }),
            Ok(SubagentAdapterResponse {
                decision: SubagentDecision::Done {
                    summary: "Concrete todo plan created".to_owned(),
                },
                usage: token_usage(1, 1),
            }),
        ],
        Arc::clone(&prompt_log),
        Duration::from_millis(0),
    );
    let runtime = AgentRuntime::new(adapter);
    let temp_dir = temp_dir("runtime-block-repeat-replan");
    let registry_path = write_registry(&temp_dir, FAKE_SERVER_BIN, Vec::new());
    write_mcp_metadata(&temp_dir, "fake");

    let outcome = runtime
        .run_turn(RunRequest {
            system_prompt_path: write_prompt(&temp_dir, "Be concise."),
            working_directory: temp_dir.clone(),
            conversation_history: vec![],
            recent_session_messages: vec![],
            user_message: "Do a full season analysis for CSK in IPL 2025".to_owned(),
            response_target: default_response_target(),
            registry_path,
            subagent_registry_path: write_subagent_registry(&temp_dir),
            tool_policy_path: None,
            enabled_servers: Some(vec![ServerName::new("fake").expect("valid server")]),
            limits: RuntimeLimits {
                max_steps_per_turn: 2,
                ..RuntimeLimits::default()
            },
            model_config: ModelConfig::new("fake-model"),
            require_todos: true,
        })
        .await
        .expect("turn should succeed");

    assert_eq!(outcome.termination, TerminationReason::MaxStepsReached);
    let handoffs = outcome
        .events
        .iter()
        .filter(|event| {
            matches!(
                event,
                RuntimeEvent::HandoffToSubagent { subagent_type, .. } if subagent_type == "tool-executor"
            )
        })
        .count();
    assert_eq!(handoffs, 1);
    assert!(outcome.turn.messages.iter().any(|message| matches!(
        message,
        MessageRecord::Llm(record)
            if record.content.contains("Do not replan the todo list again yet")
    )));
}

#[tokio::test]
async fn delegated_tool_follow_up_prompt_keeps_bash_output() {
    let _metadata_guard = metadata_test_lock().lock().await;
    cleanup_mcp_metadata("fake");
    let prompt_log = Arc::new(Mutex::new(Vec::new()));
    let adapter = FakeModelAdapter::new(
        vec![
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::DelegateSubagent {
                    delegation: SubagentDelegationRequest {
                        subagent_type: "tool-executor".to_owned(),
                        target: DelegationTarget::LocalToolsScope(
                            LocalToolsScopeTarget::working_directory(),
                        ),
                        goal: "Inspect bash output".to_owned(),
                    },
                },
                usage: token_usage(2, 2),
            }),
            Ok(ModelAdapterResponse {
                decision: ModelStepDecision::Final {
                    content: "bash complete".to_owned(),
                },
                usage: token_usage(1, 1),
            }),
        ],
        vec![
            Ok(SubagentAdapterResponse {
                decision: SubagentDecision::LocalToolCall {
                    tool_name: LocalToolName::new("bash").expect("valid tool"),
                    arguments: json!({"command":"printf 'stdout-ok'; printf 'stderr-ok' >&2"}),
                },
                usage: token_usage(1, 1),
            }),
            Ok(SubagentAdapterResponse {
                decision: SubagentDecision::Done {
                    summary: "Bash inspected".to_owned(),
                },
                usage: token_usage(1, 1),
            }),
        ],
        Arc::clone(&prompt_log),
        Duration::from_millis(0),
    );

    let runtime = AgentRuntime::new(adapter);
    let temp_dir = temp_dir("runtime-tool-bash");
    let registry_path = write_registry(&temp_dir, FAKE_SERVER_BIN, Vec::new());
    write_mcp_metadata(&temp_dir, "fake");

    runtime
        .run_turn(RunRequest {
            system_prompt_path: write_prompt(
                &temp_dir,
                "Sub-agents:\n<dynamic variable: available sub-agents>\nTools:\n<dynamic variable: available tools>",
            ),
            working_directory: temp_dir.clone(),
            conversation_history: vec![],
            recent_session_messages: vec![],
            user_message: "Run a bash check".to_owned(),
            response_target: default_response_target(),
            registry_path,
            subagent_registry_path: write_subagent_registry(&temp_dir),
            tool_policy_path: None,
            enabled_servers: Some(vec![ServerName::new("fake").expect("valid server")]),
            limits: RuntimeLimits::default(),
            model_config: ModelConfig::new("fake-model"),
            require_todos: false,
        })
        .await
        .expect("turn should succeed");

    let prompts = prompt_log.lock().expect("prompt log should lock");
    let follow_up_prompt = prompts
        .iter()
        .find(|prompt| {
            prompt.contains("## Delegated Execution History")
                && prompt.contains("tool_result: local_tool=bash")
                && prompt.contains("stdout-ok")
        })
        .expect("subagent follow-up prompt should include bash output");
    assert!(follow_up_prompt.contains("stderr-ok"));
}

#[tokio::test]
async fn runtime_fails_when_metadata_is_missing() {
    let _metadata_guard = metadata_test_lock().lock().await;
    cleanup_mcp_metadata("missing-meta");
    let adapter = FakeModelAdapter::new(
        vec![Ok(ModelAdapterResponse {
            decision: ModelStepDecision::Final {
                content: "unused".to_owned(),
            },
            usage: token_usage(1, 1),
        })],
        Vec::new(),
        Arc::new(Mutex::new(Vec::new())),
        Duration::from_millis(0),
    );
    let runtime = AgentRuntime::new(adapter);
    let temp_dir = temp_dir("runtime-missing-metadata");
    let registry_path =
        write_registry_named(&temp_dir, "missing-meta", FAKE_SERVER_BIN, Vec::new());

    let error = runtime
        .run_turn(RunRequest {
            system_prompt_path: write_prompt(&temp_dir, "Be precise."),
            working_directory: temp_dir.clone(),
            conversation_history: vec![],
            recent_session_messages: vec![],
            user_message: "Hello".to_owned(),
            response_target: default_response_target(),
            registry_path,
            subagent_registry_path: write_subagent_registry(&temp_dir),
            tool_policy_path: None,
            enabled_servers: Some(vec![ServerName::new("missing-meta").expect("valid server")]),
            limits: RuntimeLimits::default(),
            model_config: ModelConfig::new("fake-model"),
            require_todos: false,
        })
        .await
        .expect_err("missing metadata must fail");

    assert!(matches!(error, RuntimeError::Metadata(_)));
}

#[tokio::test]
async fn runtime_skips_final_render_for_short_factual_answers() {
    let _metadata_guard = metadata_test_lock().lock().await;
    cleanup_mcp_metadata("fake");
    let adapter = FakeModelAdapter::new(
        vec![Ok(ModelAdapterResponse {
            decision: ModelStepDecision::Final {
                content: "There are 42 rows.".to_owned(),
            },
            usage: token_usage(1, 1),
        })],
        Vec::new(),
        Arc::new(Mutex::new(Vec::new())),
        Duration::from_millis(0),
    );
    let runtime = AgentRuntime::new(adapter);
    let temp_dir = temp_dir("runtime-short-answer-fast-path");
    let registry_path = write_registry(&temp_dir, FAKE_SERVER_BIN, Vec::new());
    write_mcp_metadata(&temp_dir, "fake");

    let outcome = runtime
        .run_turn(RunRequest {
            system_prompt_path: write_prompt(&temp_dir, "Be concise."),
            working_directory: temp_dir.clone(),
            conversation_history: vec![],
            recent_session_messages: vec![],
            user_message: "How many rows?".to_owned(),
            response_target: default_response_target(),
            registry_path,
            subagent_registry_path: write_subagent_registry(&temp_dir),
            tool_policy_path: None,
            enabled_servers: Some(vec![ServerName::new("fake").expect("valid server")]),
            limits: RuntimeLimits::default(),
            model_config: ModelConfig::new("fake-model"),
            require_todos: false,
        })
        .await
        .expect("turn should succeed");

    assert_eq!(outcome.final_text, "There are 42 rows.");
    assert!(!outcome.events.iter().any(|event| matches!(
        event,
        RuntimeEvent::AnswerRenderStarted { .. } | RuntimeEvent::AnswerRenderCompleted { .. }
    )));
}

#[tokio::test]
async fn runtime_tracks_model_events_for_rendered_answers() {
    let _metadata_guard = metadata_test_lock().lock().await;
    cleanup_mcp_metadata("fake");
    let adapter = FakeModelAdapter::new(
        vec![Ok(ModelAdapterResponse {
            decision: ModelStepDecision::Final {
                content: "A".repeat(600),
            },
            usage: token_usage(1, 1),
        })],
        Vec::new(),
        Arc::new(Mutex::new(Vec::new())),
        Duration::from_millis(0),
    );
    let runtime = AgentRuntime::new(adapter);
    let temp_dir = temp_dir("runtime-rendered-answer-events");
    let registry_path = write_registry(&temp_dir, FAKE_SERVER_BIN, Vec::new());
    write_mcp_metadata(&temp_dir, "fake");

    let outcome = runtime
        .run_turn(RunRequest {
            system_prompt_path: write_prompt(&temp_dir, "Be precise."),
            working_directory: temp_dir.clone(),
            conversation_history: vec![],
            recent_session_messages: vec![],
            user_message: "Explain the result".to_owned(),
            response_target: default_response_target(),
            registry_path,
            subagent_registry_path: write_subagent_registry(&temp_dir),
            tool_policy_path: None,
            enabled_servers: Some(vec![ServerName::new("fake").expect("valid server")]),
            limits: RuntimeLimits::default(),
            model_config: ModelConfig::new("fake-model"),
            require_todos: false,
        })
        .await
        .expect("turn should succeed");

    let model_called = outcome
        .events
        .iter()
        .filter(|event| matches!(event, RuntimeEvent::ModelCalled { .. }))
        .count();
    let model_responded = outcome
        .events
        .iter()
        .filter(|event| matches!(event, RuntimeEvent::ModelResponded { .. }))
        .count();
    assert_eq!(model_called, model_responded);
    assert!(
        outcome
            .events
            .iter()
            .any(|event| matches!(event, RuntimeEvent::AnswerRenderStarted { .. }))
    );
}

fn write_mcp_metadata(_temp_dir: &Path, logical_name: &str) {
    cleanup_mcp_metadata(logical_name);
    let current_dir = std::env::current_dir().expect("cwd should resolve");
    let metadata_dir = current_dir.join(DEFAULT_METADATA_DIR);
    fs::create_dir_all(&metadata_dir).expect("metadata dir should exist");
    let paths = artifact_paths(&metadata_dir, logical_name);
    let families = McpCapabilityFamilies {
        tools: McpCapabilityFamilySummary {
            supported: true,
            count: 3,
        },
        resources: McpCapabilityFamilySummary {
            supported: true,
            count: 1,
        },
    };
    let server = McpServerMetadata {
        logical_name: logical_name.to_owned(),
        protocol_name: "fake-runtime-server".to_owned(),
        title: Some("Fake Runtime".to_owned()),
        version: "1.0.0".to_owned(),
        description: Some("Test MCP".to_owned()),
        instructions_summary: Some("Use the selected capability only.".to_owned()),
    };
    let minimal = McpMinimalCatalog {
        schema_version: CURRENT_SCHEMA_VERSION,
        server: server.clone(),
        capability_families: families.clone(),
        tools: vec![
            MinimalToolMetadata {
                name: "run-sql".to_owned(),
                title: Some("Run SQL".to_owned()),
                description: Some("Execute a SQL query".to_owned()),
            },
            MinimalToolMetadata {
                name: "fail-tool".to_owned(),
                title: Some("Fail Tool".to_owned()),
                description: Some("Return an MCP-level error payload".to_owned()),
            },
            MinimalToolMetadata {
                name: "preview_leads".to_owned(),
                title: Some("Preview Leads".to_owned()),
                description: Some("Preview sample leads rows".to_owned()),
            },
        ],
        resources: vec![MinimalResourceMetadata {
            uri: "crm://dashboards/main".to_owned(),
            name: Some("main_dashboard".to_owned()),
            title: Some("Main Dashboard".to_owned()),
            description: Some("Primary dashboard".to_owned()),
            mime_type: Some("application/json".to_owned()),
        }],
    };
    let full = McpFullCatalog {
        schema_version: CURRENT_SCHEMA_VERSION,
        server,
        capability_families: families,
        tools: vec![
            FullToolMetadata {
                name: "run-sql".to_owned(),
                title: Some("Run SQL".to_owned()),
                description: Some("Execute a SQL query".to_owned()),
                input_schema: json!({"type":"object","properties":{"query":{"type":"string"}}}),
            },
            FullToolMetadata {
                name: "fail-tool".to_owned(),
                title: Some("Fail Tool".to_owned()),
                description: Some("Return an MCP-level error payload".to_owned()),
                input_schema: json!({"type":"object"}),
            },
            FullToolMetadata {
                name: "preview_leads".to_owned(),
                title: Some("Preview Leads".to_owned()),
                description: Some("Preview sample leads rows".to_owned()),
                input_schema: json!({"type":"object","properties":{"limit":{"type":"integer"}}}),
            },
        ],
        resources: vec![FullResourceMetadata {
            uri: "crm://dashboards/main".to_owned(),
            name: Some("main_dashboard".to_owned()),
            title: Some("Main Dashboard".to_owned()),
            description: Some("Primary dashboard".to_owned()),
            mime_type: Some("application/json".to_owned()),
            annotations: None,
        }],
        extensions: json!({}),
    };
    write_catalogs(&paths, &minimal, &full).expect("metadata should write");
}

fn cleanup_mcp_metadata(logical_name: &str) {
    let current_dir = std::env::current_dir().expect("cwd should resolve");
    let metadata_dir = current_dir.join(DEFAULT_METADATA_DIR);
    let paths = artifact_paths(&metadata_dir, logical_name);
    let _ = fs::remove_file(paths.minimal_path);
    let _ = fs::remove_file(paths.full_path);
}

fn write_subagent_registry(temp_dir: &Path) -> PathBuf {
    let prompt_dir = temp_dir.join("subagents");
    fs::create_dir_all(&prompt_dir).expect("subagent dir should exist");
    let prompt_path = prompt_dir.join("mcp-executor.prompt.md");
    fs::write(
        &prompt_path,
        "You are the mcp-executor. Return exactly one executable action or cannot_execute.",
    )
    .expect("subagent prompt should write");
    let tool_prompt_path = prompt_dir.join("tool-executor.prompt.md");
    fs::write(
        &tool_prompt_path,
        "You are the tool-executor. Return exactly one executable local tool action or cannot_execute.",
    )
    .expect("tool subagent prompt should write");
    let registry_path = temp_dir.join("subagents.json");
    fs::write(
        &registry_path,
        json!({
            "subagents": [
                {
                    "type": "mcp-executor",
                    "display_name": "Mcp Executor",
                    "purpose": "Complete one bounded MCP task on a single server",
                    "when_to_use": "After selecting a capability or server scope",
                    "target_requirements": "mcp_capability or mcp_server_scope",
                    "result_summary": "Returns MCP tool or resource actions until the delegated goal is complete",
                    "prompt_path": "subagents/mcp-executor.prompt.md",
                    "enabled": true
                },
                {
                    "type": "tool-executor",
                    "display_name": "Tool Executor",
                    "purpose": "Complete one bounded local-tools task",
                    "when_to_use": "After selecting the local tools scope",
                    "target_requirements": "local_tools_scope",
                    "result_summary": "Returns local tool actions until the delegated goal is complete",
                    "prompt_path": "subagents/tool-executor.prompt.md",
                    "enabled": true
                }
            ]
        })
        .to_string(),
    )
    .expect("subagent registry should write");
    registry_path
}

fn token_usage(input_tokens: u32, output_tokens: u32) -> agent_runtime::UsageSummary {
    agent_runtime::UsageSummary {
        input_tokens,
        cached_tokens: 0,
        output_tokens,
        total_tokens: input_tokens + output_tokens,
    }
}

fn write_registry(temp_dir: &Path, server_bin: &str, args: Vec<String>) -> PathBuf {
    write_registry_named(temp_dir, "fake", server_bin, args)
}

fn write_registry_named(
    temp_dir: &Path,
    server_name: &str,
    server_bin: &str,
    args: Vec<String>,
) -> PathBuf {
    let config_path = temp_dir.join("mcp_servers.json");
    fs::write(
        &config_path,
        json!({
            "servers": [
                {
                    "name": server_name,
                    "command": server_bin,
                    "args": args
                }
            ]
        })
        .to_string(),
    )
    .expect("temp config write");
    config_path
}

fn write_prompt(temp_dir: &Path, contents: &str) -> PathBuf {
    let prompt_path = temp_dir.join("prompt.md");
    fs::write(&prompt_path, contents).expect("prompt write");
    prompt_path
}

fn temp_dir(prefix: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock should be valid")
        .as_nanos();
    let path = std::env::temp_dir().join(format!("agent-runtime-{prefix}-{unique}"));
    fs::create_dir_all(&path).expect("temp dir create");
    path
}

#[derive(Debug)]
struct FakeModelAdapter {
    responses: Arc<Mutex<VecDeque<Result<ModelAdapterResponse, ModelAdapterError>>>>,
    subagent_responses: Arc<Mutex<VecDeque<Result<SubagentAdapterResponse, ModelAdapterError>>>>,
    prompt_log: Arc<Mutex<Vec<String>>>,
    delay: Duration,
}

impl FakeModelAdapter {
    fn new(
        responses: Vec<Result<ModelAdapterResponse, ModelAdapterError>>,
        subagent_responses: Vec<Result<SubagentAdapterResponse, ModelAdapterError>>,
        prompt_log: Arc<Mutex<Vec<String>>>,
        delay: Duration,
    ) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses.into())),
            subagent_responses: Arc::new(Mutex::new(subagent_responses.into())),
            prompt_log,
            delay,
        }
    }
}

#[async_trait]
impl ModelAdapter for FakeModelAdapter {
    async fn generate_step(
        &self,
        request: ModelStepRequest,
        _debug_sink: Option<&mut dyn agent_runtime::model::ModelAdapterDebugSink>,
    ) -> Result<ModelAdapterResponse, ModelAdapterError> {
        self.prompt_log
            .lock()
            .expect("prompt log should lock")
            .push(request.prompt.rendered);

        if !self.delay.is_zero() {
            tokio::time::sleep(self.delay).await;
        }

        self.responses
            .lock()
            .expect("responses queue should lock")
            .pop_front()
            .expect("a test response should be queued")
    }

    async fn generate_subagent_step(
        &self,
        request: SubagentStepRequest,
        _debug_sink: Option<&mut dyn agent_runtime::model::ModelAdapterDebugSink>,
    ) -> Result<SubagentAdapterResponse, ModelAdapterError> {
        self.prompt_log
            .lock()
            .expect("prompt log should lock")
            .push(request.prompt);

        self.subagent_responses
            .lock()
            .expect("subagent responses queue should lock")
            .pop_front()
            .unwrap_or_else(|| {
                Ok(SubagentAdapterResponse {
                    decision: SubagentDecision::CannotExecute {
                        reason: "no subagent response queued".to_owned(),
                    },
                    usage: token_usage(0, 0),
                })
            })
    }
}
