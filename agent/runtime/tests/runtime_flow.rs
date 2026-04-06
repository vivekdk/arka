//! End-to-end runtime integration tests for stored MCP metadata and sub-agent delegation.

use std::{
    collections::VecDeque,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use agent_runtime::{
    AgentRuntime, ConversationMessage, ConversationRole, MessageRecord, ModelConfig,
    ResponseClient, ResponseFormat, ResponseTarget, RunRequest, RuntimeError, RuntimeEvent,
    RuntimeLimits, ServerName, TerminationReason,
    model::{
        ModelAdapter, ModelAdapterError, ModelAdapterResponse, ModelStepDecision, ModelStepRequest,
        SubagentAdapterResponse, SubagentDecision, SubagentDelegationRequest, SubagentStepRequest,
    },
    state::{DelegationTarget, McpCapabilityTarget},
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

#[tokio::test]
async fn runtime_delegates_tool_executor_and_executes_tool() {
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
            user_message: "How many rows?".to_owned(),
            response_target: default_response_target(),
            registry_path,
            subagent_registry_path,
            tool_policy_path: None,
            enabled_servers: Some(vec![ServerName::new("fake").expect("valid server")]),
            limits: RuntimeLimits::default(),
            model_config: ModelConfig::new("fake-model"),
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
            user_message: "Show sample leads".to_owned(),
            response_target: default_response_target(),
            registry_path,
            subagent_registry_path: write_subagent_registry(&temp_dir),
            tool_policy_path: None,
            enabled_servers: Some(vec![ServerName::new("fake").expect("valid server")]),
            limits: RuntimeLimits::default(),
            model_config: ModelConfig::new("fake-model"),
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
            user_message: "Preview the latest leads".to_owned(),
            response_target: default_response_target(),
            registry_path,
            subagent_registry_path: write_subagent_registry(&temp_dir),
            tool_policy_path: None,
            enabled_servers: Some(vec![ServerName::new("fake").expect("valid server")]),
            limits: RuntimeLimits::default(),
            model_config: ModelConfig::new("fake-model"),
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
                user_message: "Run it".to_owned(),
                response_target: default_response_target(),
                registry_path,
                subagent_registry_path,
                tool_policy_path: None,
                enabled_servers: Some(vec![ServerName::new("fake").expect("valid server")]),
                limits: RuntimeLimits::default(),
                model_config: ModelConfig::new("fake-model"),
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
            user_message: "Show the dashboard".to_owned(),
            response_target: default_response_target(),
            registry_path,
            subagent_registry_path: write_subagent_registry(&temp_dir),
            tool_policy_path: None,
            enabled_servers: Some(vec![ServerName::new("fake").expect("valid server")]),
            limits: RuntimeLimits::default(),
            model_config: ModelConfig::new("fake-model"),
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
            user_message: "Get me the final count".to_owned(),
            response_target: default_response_target(),
            registry_path,
            subagent_registry_path: write_subagent_registry(&temp_dir),
            tool_policy_path: None,
            enabled_servers: Some(vec![ServerName::new("fake").expect("valid server")]),
            limits: RuntimeLimits::default(),
            model_config: ModelConfig::new("fake-model"),
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
async fn runtime_fails_when_metadata_is_missing() {
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
            user_message: "Hello".to_owned(),
            response_target: default_response_target(),
            registry_path,
            subagent_registry_path: write_subagent_registry(&temp_dir),
            tool_policy_path: None,
            enabled_servers: Some(vec![ServerName::new("missing-meta").expect("valid server")]),
            limits: RuntimeLimits::default(),
            model_config: ModelConfig::new("fake-model"),
        })
        .await
        .expect_err("missing metadata must fail");

    assert!(matches!(error, RuntimeError::Metadata(_)));
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
            count: 2,
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
    let registry_path = temp_dir.join("subagents.json");
    fs::write(
        &registry_path,
        json!({
            "subagents": [
                {
                    "type": "mcp-executor",
                    "display_name": "Mcp Executor",
                    "purpose": "Build the final MCP action",
                    "when_to_use": "After selecting a capability",
                    "target_requirements": "server_name, capability_kind, capability_id",
                    "result_summary": "Returns one tool call or resource read",
                    "prompt_path": "subagents/mcp-executor.prompt.md",
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
