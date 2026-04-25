//! Integration tests for the HTTP control-plane API and webhook flows.

use std::{
    collections::VecDeque,
    fs,
    path::PathBuf,
    process::Command,
    sync::{Arc, Mutex},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use agent_controlplane::{
    ApiResponseFormat, ApprovalDecision, ChannelBinding, ChannelDeliveryTarget, ChannelEnvelope,
    ChannelIntent, ChannelKind, ConversationStore, CreateSessionRequest, InMemoryConversationStore,
    JsonlConversationStore, RuntimeHarnessEventEnvelope, RuntimeHarnessObservation,
    SendSessionMessageRequest, SessionService, SlackConnector, SlackDeliveryClient,
    SlackFileUpload, SlackMessagePayload, SubmitApprovalRequest, TurnRunner, TurnRunnerInput,
    TurnRunnerOutput, WhatsAppConnector, WhatsAppDeliveryClient, WhatsAppDeliveryError,
    WhatsAppDmPolicy, WhatsAppGatewayStatus, WhatsAppMessagePayload, WhatsAppWebhookPayload,
    api::SlackStreamHandle, router, router_with_channels, router_with_slack,
};
use agent_runtime::{
    ConversationRole, ResponseClient, ResponseFormat, ResponseTarget, RuntimeError, RuntimeEvent,
    RuntimeExecutor, StepId, TerminationReason, TurnId, TurnRecord, UsageSummary,
};
use async_trait::async_trait;
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use tower::util::ServiceExt;
use uuid::Uuid;

fn cli_response_target() -> ResponseTarget {
    ResponseTarget {
        client: ResponseClient::Cli,
        format: ResponseFormat::Markdown,
    }
}

#[tokio::test]
async fn api_creates_session_and_processes_message() {
    let app = test_router(vec![Ok(turn_output("hello back"))]);

    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/sessions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&CreateSessionRequest {
                        channel: Some(ChannelKind::Api),
                        external_conversation_id: Some("api-thread-1".to_owned()),
                        external_user_id: Some("user-1".to_owned()),
                    })
                    .expect("request serialization"),
                ))
                .expect("request should build"),
        )
        .await
        .expect("request should succeed");
    assert_eq!(create.status(), StatusCode::OK);
    let created: agent_controlplane::SessionRecord = serde_json::from_slice(
        &axum::body::to_bytes(create.into_body(), usize::MAX)
            .await
            .expect("body should read"),
    )
    .expect("session should deserialize");

    let send = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/sessions/{}/messages", created.session_id))
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&SendSessionMessageRequest {
                        text: "hi".to_owned(),
                        idempotency_key: Some("msg-1".to_owned()),
                        response_format: None,
                    })
                    .expect("request serialization"),
                ))
                .expect("request should build"),
        )
        .await
        .expect("request should succeed");
    assert_eq!(send.status(), StatusCode::OK);
    let sent: agent_controlplane::SendSessionMessageResponse = serde_json::from_slice(
        &axum::body::to_bytes(send.into_body(), usize::MAX)
            .await
            .expect("body should read"),
    )
    .expect("response should deserialize");

    assert_eq!(sent.result.outbound.len(), 1);
    assert_eq!(sent.result.outbound[0].text, "hello back");
    assert_eq!(
        sent.result.session.status,
        agent_controlplane::SessionStatus::Completed
    );
    let last_turn = sent
        .result
        .session
        .last_turn
        .as_ref()
        .expect("last turn should be present");
    assert_eq!(last_turn.turn_number, 1);
    assert_eq!(last_turn.model_name, "gpt-5.4");
    assert_eq!(last_turn.elapsed_ms, 1234);
}

#[tokio::test]
async fn api_maps_default_and_override_response_formats() {
    let seen_targets = Arc::new(Mutex::new(Vec::new()));
    let app = router(SessionService::new(
        CapturingTurnRunner::new(Arc::clone(&seen_targets)),
        InMemoryConversationStore::default(),
    ));

    let create = post_json(
        &app,
        "/sessions",
        &CreateSessionRequest {
            channel: Some(ChannelKind::Api),
            external_conversation_id: Some("api-thread-2".to_owned()),
            external_user_id: Some("user-2".to_owned()),
        },
    )
    .await;
    let created: agent_controlplane::SessionRecord =
        serde_json::from_slice(&create).expect("session should deserialize");

    post_json(
        &app,
        &format!("/sessions/{}/messages", created.session_id),
        &SendSessionMessageRequest {
            text: "plain".to_owned(),
            idempotency_key: Some("api-default".to_owned()),
            response_format: None,
        },
    )
    .await;
    post_json(
        &app,
        &format!("/sessions/{}/messages", created.session_id),
        &SendSessionMessageRequest {
            text: "markdown".to_owned(),
            idempotency_key: Some("api-markdown".to_owned()),
            response_format: Some(ApiResponseFormat::Markdown),
        },
    )
    .await;

    assert_eq!(
        *seen_targets.lock().expect("seen targets should lock"),
        vec![
            ResponseTarget {
                client: ResponseClient::Api,
                format: ResponseFormat::PlainText,
            },
            ResponseTarget {
                client: ResponseClient::Api,
                format: ResponseFormat::Markdown,
            },
        ]
    );
}

#[tokio::test]
async fn slack_webhook_is_idempotent_and_session_scoped() {
    let service = SessionService::new(
        FakeTurnRunner::new(vec![Ok(turn_output("from slack"))]),
        InMemoryConversationStore::default(),
    );
    let deliveries = Arc::new(Mutex::new(Vec::new()));
    let app = router_with_slack(
        service.clone(),
        None,
        Some(SlackConnector {
            signing_secret: "secret".to_owned(),
            delivery_client: Arc::new(FakeSlackDeliveryClient::new(Arc::clone(&deliveries))),
            event_queue_capacity: 8,
        }),
    );

    let payload = br#"{
        "type": "event_callback",
        "team_id": "T1",
        "event_id": "event-1",
        "event": {
            "type": "app_mention",
            "user": "user-1",
            "channel": "channel-1",
            "text": "<@B1> hello",
            "thread_ts": "thread-1",
            "ts": "thread-1"
        }
    }"#;

    let first = post_signed_slack_json(&app, payload, "secret").await;
    assert_eq!(first.status(), StatusCode::OK);
    wait_for_delivery_count(&deliveries, 1).await;

    let binding = ChannelBinding {
        channel: ChannelKind::Slack,
        external_workspace_id: Some("T1".to_owned()),
        external_conversation_id: "thread-1".to_owned(),
        external_channel_id: Some("channel-1".to_owned()),
        external_thread_id: Some("thread-1".to_owned()),
        external_user_id: "user-1".to_owned(),
    };
    let first_session = service
        .find_session_by_binding(&binding)
        .await
        .expect("slack session should exist");

    let second = post_signed_slack_json(&app, payload, "secret").await;
    assert_eq!(second.status(), StatusCode::OK);
    tokio::time::sleep(Duration::from_millis(100)).await;

    let delivery_log = deliveries.lock().expect("delivery log should lock");
    assert_eq!(delivery_log.len(), 1);
    assert_eq!(delivery_log[0].payload.text, "<@user-1> from slack");
    drop(delivery_log);

    let second_session = service
        .find_session_by_binding(&binding)
        .await
        .expect("slack session should still exist");
    assert_eq!(first_session.session_id, second_session.session_id);
}

#[tokio::test]
async fn slack_webhook_streams_reply_when_client_supports_streaming() {
    let service = SessionService::new(
        StreamingFakeTurnRunner::new("from slack streamed"),
        InMemoryConversationStore::default(),
    );
    let deliveries = Arc::new(Mutex::new(Vec::new()));
    let stream_ops = Arc::new(Mutex::new(Vec::new()));
    let app = router_with_slack(
        service,
        None,
        Some(SlackConnector {
            signing_secret: "secret".to_owned(),
            delivery_client: Arc::new(FakeSlackDeliveryClient::with_streams(
                Arc::clone(&deliveries),
                Arc::clone(&stream_ops),
            )),
            event_queue_capacity: 8,
        }),
    );

    let payload = br#"{
        "type": "event_callback",
        "team_id": "T1",
        "event_id": "event-stream-1",
        "event": {
            "type": "app_mention",
            "user": "user-1",
            "channel": "channel-1",
            "text": "<@B1> hello",
            "thread_ts": "thread-1",
            "ts": "thread-1"
        }
    }"#;

    let response = post_signed_slack_json(&app, payload, "secret").await;
    assert_eq!(response.status(), StatusCode::OK);
    wait_for_stream_op_count(&stream_ops, 3).await;

    let messages = deliveries.lock().expect("delivery log should lock");
    assert!(messages.is_empty());
    drop(messages);

    let stream_ops = stream_ops.lock().expect("stream log should lock");
    assert_eq!(
        stream_ops.as_slice(),
        &[
            FakeSlackStreamOp::Start {
                target: ChannelDeliveryTarget {
                    channel: ChannelKind::Slack,
                    external_workspace_id: Some("T1".to_owned()),
                    external_conversation_id: "thread-1".to_owned(),
                    external_channel_id: Some("channel-1".to_owned()),
                    external_thread_id: Some("thread-1".to_owned()),
                    external_user_id: "user-1".to_owned(),
                },
                text: "<@user-1> ".to_owned(),
            },
            FakeSlackStreamOp::Append {
                channel: "channel-1".to_owned(),
                ts: "stream-ts-1".to_owned(),
                text: "from slack streamed".to_owned(),
            },
            FakeSlackStreamOp::Stop {
                channel: "channel-1".to_owned(),
                ts: "stream-ts-1".to_owned(),
                text: None,
            },
        ]
    );
}

#[tokio::test]
async fn slack_webhook_uploads_generated_chart_as_file() {
    let deliveries = Arc::new(Mutex::new(Vec::new()));
    let uploads = Arc::new(Mutex::new(Vec::new()));
    let chart_dir = unique_test_store_dir("slack-chart");
    fs::create_dir_all(&chart_dir).expect("chart directory should exist");
    let chart_path = chart_dir.join("chart.png");
    fs::write(&chart_path, b"png-bytes").expect("chart image should write");

    let mut output = turn_output("chart ready");
    output.generated_artifacts = vec![agent_controlplane::runner::GeneratedArtifact {
        kind: agent_controlplane::runner::GeneratedArtifactKind::Image,
        path: chart_path.clone(),
        file_name: "chart.png".to_owned(),
        mime_type: Some("image/png".to_owned()),
    }];
    let service = SessionService::new(
        FakeTurnRunner::new(vec![Ok(output)]),
        InMemoryConversationStore::default(),
    );
    let app = router_with_slack(
        service,
        None,
        Some(SlackConnector {
            signing_secret: "secret".to_owned(),
            delivery_client: Arc::new(FakeSlackDeliveryClient::with_uploads(
                Arc::clone(&deliveries),
                Arc::clone(&uploads),
            )),
            event_queue_capacity: 8,
        }),
    );

    let payload = br#"{
        "type": "event_callback",
        "team_id": "T1",
        "event_id": "event-upload-1",
        "event": {
            "type": "app_mention",
            "user": "user-1",
            "channel": "channel-1",
            "text": "<@B1> make chart",
            "thread_ts": "thread-1",
            "ts": "thread-1"
        }
    }"#;

    let response = post_signed_slack_json(&app, payload, "secret").await;
    assert_eq!(response.status(), StatusCode::OK);
    wait_for_slack_upload_count(&uploads, 1).await;

    assert!(
        deliveries
            .lock()
            .expect("delivery log should lock")
            .is_empty()
    );
    let uploads = uploads.lock().expect("upload log should lock");
    assert_eq!(uploads[0].upload.file_name, "chart.png");
    assert_eq!(uploads[0].upload.mime_type.as_deref(), Some("image/png"));
    assert_eq!(
        uploads[0].upload.initial_comment.as_deref(),
        Some("<@user-1> chart ready")
    );
    assert_eq!(uploads[0].upload.bytes, b"png-bytes");

    cleanup_test_store_dir(&chart_dir);
}

#[tokio::test]
async fn whatsapp_webhook_creates_session_and_reply() {
    let app = test_router(vec![Ok(turn_output("from whatsapp"))]);
    let payload = WhatsAppWebhookPayload {
        message_id: "wa-1".to_owned(),
        conversation_id: "conversation-1".to_owned(),
        from_user_id: "user-1".to_owned(),
        text: "hello".to_owned(),
    };
    let body = post_json(&app, "/channels/whatsapp/webhook", &payload).await;
    let result: agent_controlplane::ReceiveWhatsAppMessageResponse =
        serde_json::from_slice(&body).expect("response");
    assert_eq!(result.queued_outbound, 0);
    assert!(!result.was_duplicate);
}

#[tokio::test]
async fn whatsapp_gateway_login_status_and_delivery_flow() {
    let service = SessionService::new(
        FakeTurnRunner::new(vec![Ok(turn_output("from whatsapp gateway"))]),
        InMemoryConversationStore::default(),
    );
    let deliveries = Arc::new(Mutex::new(Vec::new()));
    let state_path = unique_test_store_dir("whatsapp-state").join("gateway.json");
    let app = router_with_channels(
        service,
        None,
        None,
        Some(WhatsAppConnector {
            account_id: "acct-1".to_owned(),
            dm_policy: WhatsAppDmPolicy::Allowlist,
            allow_from: vec!["user-1".to_owned()],
            delivery_client: Arc::new(FakeWhatsAppDeliveryClient::new(Arc::clone(&deliveries))),
            control_client: None,
            event_queue_capacity: 8,
            state_path: state_path.clone(),
        }),
    );

    let initial_status: WhatsAppGatewayStatus = get_json(&app, "/channels/whatsapp/status").await;
    assert_eq!(
        initial_status.connection_state,
        agent_controlplane::WhatsAppGatewayConnectionState::NeedsLogin
    );

    let login: agent_controlplane::StartWhatsAppLoginResponse = post_json_value(
        &app,
        "/channels/whatsapp/login/start",
        serde_json::json!({}),
    )
    .await;
    let ready_status: WhatsAppGatewayStatus = post_json_value(
        &app,
        "/channels/whatsapp/login/complete",
        serde_json::json!({ "login_session_id": login.login_session_id }),
    )
    .await;
    assert_eq!(
        ready_status.connection_state,
        agent_controlplane::WhatsAppGatewayConnectionState::Ready
    );

    let response: agent_controlplane::ReceiveWhatsAppMessageResponse = post_json_value(
        &app,
        "/channels/whatsapp/inbound",
        serde_json::json!({
            "message_id": "wa-gateway-1",
            "conversation_id": "user-1",
            "from_user_id": "user-1",
            "text": "hello"
        }),
    )
    .await;
    assert_eq!(response.queued_outbound, 1);
    wait_for_whatsapp_delivery_count(&deliveries, 1).await;
    assert_eq!(
        deliveries.lock().expect("delivery log should lock")[0].payload,
        WhatsAppMessagePayload::Text {
            text: "from whatsapp gateway".to_owned()
        }
    );

    cleanup_test_store_dir(&state_path.parent().expect("state parent").to_path_buf());
}

#[tokio::test]
async fn whatsapp_gateway_rejects_blocked_sender() {
    let state_path = unique_test_store_dir("whatsapp-blocked").join("gateway.json");
    let app = router_with_channels(
        SessionService::new(
            FakeTurnRunner::new(vec![Ok(turn_output("unused"))]),
            InMemoryConversationStore::default(),
        ),
        None,
        None,
        Some(WhatsAppConnector {
            account_id: "acct-1".to_owned(),
            dm_policy: WhatsAppDmPolicy::Allowlist,
            allow_from: vec!["user-1".to_owned()],
            delivery_client: Arc::new(FakeWhatsAppDeliveryClient::new(Arc::new(Mutex::new(
                Vec::new(),
            )))),
            control_client: None,
            event_queue_capacity: 8,
            state_path: state_path.clone(),
        }),
    );

    let login: agent_controlplane::StartWhatsAppLoginResponse = post_json_value(
        &app,
        "/channels/whatsapp/login/start",
        serde_json::json!({}),
    )
    .await;
    let _: WhatsAppGatewayStatus = post_json_value(
        &app,
        "/channels/whatsapp/login/complete",
        serde_json::json!({ "login_session_id": login.login_session_id }),
    )
    .await;

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/channels/whatsapp/inbound")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&serde_json::json!({
                        "message_id": "wa-blocked-1",
                        "conversation_id": "blocked-user",
                        "from_user_id": "blocked-user",
                        "text": "hello"
                    }))
                    .expect("request serialization"),
                ))
                .expect("request should build"),
        )
        .await
        .expect("request should succeed");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    cleanup_test_store_dir(&state_path.parent().expect("state parent").to_path_buf());
}

#[tokio::test]
async fn approval_endpoint_resolves_pending_approval() {
    let service = SessionService::new(
        FakeTurnRunner::new(vec![Ok(turn_output("unused"))]),
        InMemoryConversationStore::default(),
    );
    let session = service
        .create_session(CreateSessionRequest {
            channel: None,
            external_conversation_id: None,
            external_user_id: None,
        })
        .await;
    let session = session.expect("session should be created");
    let approval = service
        .create_approval_request(&session.session_id, "Approve?")
        .await
        .expect("approval should be created");
    let app = router(service);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!(
                    "/sessions/{}/approvals/{}",
                    session.session_id, approval.approval_id
                ))
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&SubmitApprovalRequest {
                        decision: ApprovalDecision::Approve,
                    })
                    .expect("request serialization"),
                ))
                .expect("request should build"),
        )
        .await
        .expect("request should succeed");
    assert_eq!(response.status(), StatusCode::OK);
    let approval: agent_controlplane::ApprovalRequestRecord = serde_json::from_slice(
        &axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should read"),
    )
    .expect("approval should deserialize");
    assert_eq!(approval.state, agent_controlplane::ApprovalState::Approved);
}

#[test]
fn cli_prints_usage_for_help_command() {
    let output = Command::new(env!("CARGO_BIN_EXE_cli"))
        .arg("help")
        .output()
        .expect("cli should execute");

    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("Usage:"));
}

#[tokio::test]
async fn debug_history_page_renders_static_html() {
    let app = test_router(vec![]);

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/debug/history")
                .body(Body::empty())
                .expect("request should build"),
        )
        .await
        .expect("request should succeed");
    assert_eq!(response.status(), StatusCode::OK);
    let body = String::from_utf8(
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should read")
            .to_vec(),
    )
    .expect("html should be utf-8");
    assert!(body.contains("Debug History"));
    assert!(body.contains("Arka"));
    assert!(body.contains("cached"));
    assert!(body.contains("View markdown"));
    assert!(body.contains("Formatted markdown"));
    assert!(body.contains("View JSON"));
    assert!(body.contains("Formatted JSON"));
    assert!(body.contains("metadata?.phase"));
    assert!(body.contains("buildStepNumberLookup"));
}

#[tokio::test]
async fn debug_history_sessions_endpoint_reports_when_store_is_unavailable() {
    let app = test_router(vec![]);

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/debug/history/sessions")
                .body(Body::empty())
                .expect("request should build"),
        )
        .await
        .expect("request should succeed");
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn create_session_eagerly_prepares_runner_state() {
    let runner = FakeTurnRunner::new(vec![]);
    let prepare_calls = Arc::clone(&runner.prepare_calls);
    let service = SessionService::new(runner, InMemoryConversationStore::default());

    let session = service
        .create_session(CreateSessionRequest {
            channel: Some(ChannelKind::Api),
            external_conversation_id: Some("api-thread-prepare".to_owned()),
            external_user_id: Some("user-prepare".to_owned()),
        })
        .await
        .expect("session should be created");

    let prepared = prepare_calls.lock().expect("prepare calls should lock");
    assert_eq!(prepared.as_slice(), &[session.session_id]);
}

#[tokio::test]
async fn shutdown_releases_runner_resources() {
    let runner = FakeTurnRunner::new(vec![]);
    let shutdown_calls = Arc::clone(&runner.shutdown_calls);
    let service = SessionService::new(runner, InMemoryConversationStore::default());

    service.shutdown().await.expect("shutdown should succeed");

    assert_eq!(
        *shutdown_calls.lock().expect("shutdown calls should lock"),
        1
    );
}

#[tokio::test]
async fn session_turn_number_increments_across_completed_replies() {
    let service = SessionService::new(
        FakeTurnRunner::new(vec![Ok(turn_output("first")), Ok(turn_output("second"))]),
        InMemoryConversationStore::default(),
    );

    let session = service
        .create_session(CreateSessionRequest {
            channel: None,
            external_conversation_id: None,
            external_user_id: None,
        })
        .await
        .expect("session should be created");

    let first = service
        .send_session_text(
            &session.session_id,
            "hello".to_owned(),
            "msg-turn-1".to_owned(),
            ChannelKind::Cli,
            cli_response_target(),
        )
        .await
        .expect("first turn should succeed");
    assert_eq!(
        first
            .session
            .last_turn
            .as_ref()
            .expect("first last turn should exist")
            .turn_number,
        1
    );

    let second = service
        .send_session_text(
            &session.session_id,
            "again".to_owned(),
            "msg-turn-2".to_owned(),
            ChannelKind::Cli,
            cli_response_target(),
        )
        .await
        .expect("second turn should succeed");
    assert_eq!(
        second
            .session
            .last_turn
            .as_ref()
            .expect("second last turn should exist")
            .turn_number,
        2
    );
}

#[tokio::test]
async fn jsonl_store_recovers_history_and_continues_turn_numbers() {
    let store_dir = unique_test_store_dir("resume");
    let service = SessionService::new(
        FakeTurnRunner::new(vec![Ok(turn_output("first answer"))]),
        JsonlConversationStore::open(&store_dir).expect("jsonl store should open"),
    );

    let session = service
        .create_session(CreateSessionRequest {
            channel: None,
            external_conversation_id: None,
            external_user_id: None,
        })
        .await
        .expect("session should be created");
    service
        .send_session_text(
            &session.session_id,
            "hello".to_owned(),
            "jsonl-msg-1".to_owned(),
            ChannelKind::Cli,
            cli_response_target(),
        )
        .await
        .expect("first persisted turn should succeed");

    let reopened = SessionService::new(
        FakeTurnRunner::new(vec![Ok(turn_output("second answer"))]),
        JsonlConversationStore::open(&store_dir).expect("jsonl store should reopen"),
    );
    let recovered_messages = reopened.get_messages(&session.session_id).await;
    assert_eq!(recovered_messages.len(), 2);
    assert_eq!(recovered_messages[0].content, "hello");
    assert_eq!(recovered_messages[1].content, "first answer");

    let second = reopened
        .send_session_text(
            &session.session_id,
            "again".to_owned(),
            "jsonl-msg-2".to_owned(),
            ChannelKind::Cli,
            cli_response_target(),
        )
        .await
        .expect("reopened turn should succeed");
    assert_eq!(
        second
            .session
            .last_turn
            .as_ref()
            .expect("last turn should exist after reopen")
            .turn_number,
        2
    );

    cleanup_test_store_dir(&store_dir);
}

#[tokio::test]
async fn jsonl_store_recovers_bindings_and_idempotency() {
    let store_dir = unique_test_store_dir("binding");
    let binding = ChannelBinding {
        channel: ChannelKind::Slack,
        external_workspace_id: Some("T1".to_owned()),
        external_conversation_id: "thread-1".to_owned(),
        external_channel_id: Some("channel-1".to_owned()),
        external_thread_id: Some("thread-1".to_owned()),
        external_user_id: "user-1".to_owned(),
    };
    let service = SessionService::new(
        FakeTurnRunner::new(vec![Ok(turn_output("from slack"))]),
        JsonlConversationStore::open(&store_dir).expect("jsonl store should open"),
    );

    let first = service
        .dispatch_envelope(ChannelEnvelope {
            channel: ChannelKind::Slack,
            external_workspace_id: binding.external_workspace_id.clone(),
            external_conversation_id: binding.external_conversation_id.clone(),
            external_channel_id: binding.external_channel_id.clone(),
            external_thread_id: binding.external_thread_id.clone(),
            external_user_id: binding.external_user_id.clone(),
            external_message_id: Some("msg-1".to_owned()),
            idempotency_key: "slack-event-1".to_owned(),
            occurred_at: SystemTime::now(),
            intent: ChannelIntent::UserText {
                text: "hello".to_owned(),
            },
        })
        .await
        .expect("first envelope should succeed");
    assert!(!first.was_duplicate);

    let reopened = SessionService::new(
        FakeTurnRunner::new(Vec::new()),
        JsonlConversationStore::open(&store_dir).expect("jsonl store should reopen"),
    );
    let recovered = reopened
        .find_session_by_binding(&binding)
        .await
        .expect("binding should survive restart");
    assert_eq!(recovered.session_id, first.session.session_id);

    let duplicate = reopened
        .dispatch_envelope(ChannelEnvelope {
            channel: ChannelKind::Slack,
            external_workspace_id: binding.external_workspace_id.clone(),
            external_conversation_id: binding.external_conversation_id.clone(),
            external_channel_id: binding.external_channel_id.clone(),
            external_thread_id: binding.external_thread_id.clone(),
            external_user_id: binding.external_user_id.clone(),
            external_message_id: Some("msg-1".to_owned()),
            idempotency_key: "slack-event-1".to_owned(),
            occurred_at: SystemTime::now(),
            intent: ChannelIntent::UserText {
                text: "hello".to_owned(),
            },
        })
        .await
        .expect("duplicate envelope should be ignored");
    assert!(duplicate.was_duplicate);
    assert!(duplicate.outbound.is_empty());
    assert_eq!(duplicate.session.session_id, recovered.session_id);

    cleanup_test_store_dir(&store_dir);
}

#[tokio::test]
async fn jsonl_store_restores_running_sessions_as_interrupted() {
    let store_dir = unique_test_store_dir("status");
    let store = JsonlConversationStore::open(&store_dir).expect("jsonl store should open");
    let session = store
        .create_session(Vec::new())
        .await
        .expect("session should be created");
    let updated = store
        .update_session(
            &session.session_id,
            agent_controlplane::SessionStatus::Running,
            None,
        )
        .await
        .expect("session update should persist")
        .expect("session should still exist");
    assert_eq!(updated.status, agent_controlplane::SessionStatus::Running);

    let reopened = JsonlConversationStore::open(&store_dir).expect("jsonl store should reopen");
    let recovered = reopened
        .get_session(&session.session_id)
        .await
        .expect("session should reload");
    assert_eq!(
        recovered.status,
        agent_controlplane::SessionStatus::Interrupted
    );

    cleanup_test_store_dir(&store_dir);
}

#[tokio::test]
async fn session_turn_number_advances_after_failed_turns() {
    let service = SessionService::new(
        FakeTurnRunner::new(vec![
            Err(agent_controlplane::runner::TurnRunnerError::Runtime(
                RuntimeError::Timeout("sub-agent step timed out".to_owned()),
            )),
            Ok(turn_output("second")),
        ]),
        InMemoryConversationStore::default(),
    );

    let session = service
        .create_session(CreateSessionRequest {
            channel: None,
            external_conversation_id: None,
            external_user_id: None,
        })
        .await
        .expect("session should be created");

    let first = service
        .send_session_text(
            &session.session_id,
            "hello".to_owned(),
            "msg-failed-turn-1".to_owned(),
            ChannelKind::Cli,
            cli_response_target(),
        )
        .await
        .expect("failed turn should still return a dispatch result");
    assert_eq!(
        first.session.status,
        agent_controlplane::SessionStatus::Failed
    );
    assert!(first.session.last_turn.is_none());

    let second = service
        .send_session_text(
            &session.session_id,
            "again".to_owned(),
            "msg-failed-turn-2".to_owned(),
            ChannelKind::Cli,
            cli_response_target(),
        )
        .await
        .expect("second turn should succeed");
    assert_eq!(
        second
            .session
            .last_turn
            .as_ref()
            .expect("second last turn should exist")
            .turn_number,
        2
    );
}

fn test_router(
    outputs: Vec<Result<TurnRunnerOutput, agent_controlplane::runner::TurnRunnerError>>,
) -> axum::Router {
    router(SessionService::new(
        FakeTurnRunner::new(outputs),
        InMemoryConversationStore::default(),
    ))
}

fn unique_test_store_dir(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!("agent-controlplane-{label}-{}", Uuid::new_v4()))
}

fn cleanup_test_store_dir(path: &PathBuf) {
    let _ = fs::remove_dir_all(path);
}

fn turn_output(final_text: &str) -> TurnRunnerOutput {
    let turn_id = agent_runtime::TurnId::new();
    let completed_at = SystemTime::now();
    TurnRunnerOutput {
        final_text: final_text.to_owned(),
        display_text: final_text.to_owned(),
        model_name: "gpt-5.4".to_owned(),
        elapsed_ms: 1234,
        termination: TerminationReason::Final,
        usage: UsageSummary {
            input_tokens: 1,
            cached_tokens: 0,
            output_tokens: 1,
            total_tokens: 2,
        },
        events: vec![RuntimeEvent::TurnEnded {
            turn_id: turn_id.clone(),
            executor: RuntimeExecutor::main_agent(),
            at: completed_at,
            termination: TerminationReason::Final,
            usage: UsageSummary {
                input_tokens: 1,
                cached_tokens: 0,
                output_tokens: 1,
                total_tokens: 2,
            },
        }],
        turn: TurnRecord {
            turn_id,
            started_at: completed_at,
            ended_at: completed_at,
            steps: Vec::new(),
            messages: Vec::new(),
            final_text: Some(final_text.to_owned()),
            termination: TerminationReason::Final,
            usage: UsageSummary {
                input_tokens: 1,
                cached_tokens: 0,
                output_tokens: 1,
                total_tokens: 2,
            },
        },
        generated_artifacts: Vec::new(),
    }
}

async fn post_json<T: serde::Serialize>(app: &axum::Router, uri: &str, body: &T) -> Vec<u8> {
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(body).expect("request serialization"),
                ))
                .expect("request should build"),
        )
        .await
        .expect("request should succeed");
    assert_eq!(response.status(), StatusCode::OK);
    axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body should read")
        .to_vec()
}

async fn post_json_value<T: serde::de::DeserializeOwned>(
    app: &axum::Router,
    uri: &str,
    body: serde_json::Value,
) -> T {
    serde_json::from_slice(&post_json(app, uri, &body).await).expect("response should deserialize")
}

async fn get_json<T: serde::de::DeserializeOwned>(app: &axum::Router, uri: &str) -> T {
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(uri)
                .body(Body::empty())
                .expect("request should build"),
        )
        .await
        .expect("request should succeed");
    assert_eq!(response.status(), StatusCode::OK);
    serde_json::from_slice(
        &axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should read"),
    )
    .expect("response should deserialize")
}

async fn post_signed_slack_json(
    app: &axum::Router,
    body: &[u8],
    secret: &str,
) -> axum::http::Response<Body> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("current time should be after epoch")
        .as_secs() as i64;
    let signature = slack_signature(secret, body, timestamp);
    app.clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/channels/slack/events")
                .header("content-type", "application/json")
                .header("x-slack-request-timestamp", timestamp.to_string())
                .header("x-slack-signature", signature)
                .body(Body::from(body.to_vec()))
                .expect("request should build"),
        )
        .await
        .expect("request should succeed")
}

fn slack_signature(secret: &str, body: &[u8], timestamp: i64) -> String {
    let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes()).expect("mac should initialize");
    mac.update(format!("v0:{timestamp}:{}", String::from_utf8_lossy(body)).as_bytes());
    format!("v0={:x}", mac.finalize().into_bytes())
}

async fn wait_for_delivery_count(deliveries: &Arc<Mutex<Vec<FakeSlackMessage>>>, expected: usize) {
    for _ in 0..20 {
        if deliveries.lock().expect("delivery log should lock").len() >= expected {
            return;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("timed out waiting for slack delivery");
}

async fn wait_for_whatsapp_delivery_count(
    deliveries: &Arc<Mutex<Vec<FakeWhatsAppMessage>>>,
    expected: usize,
) {
    for _ in 0..20 {
        if deliveries.lock().expect("delivery log should lock").len() >= expected {
            return;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("timed out waiting for whatsapp delivery");
}

async fn wait_for_stream_op_count(
    stream_ops: &Arc<Mutex<Vec<FakeSlackStreamOp>>>,
    expected: usize,
) {
    for _ in 0..20 {
        if stream_ops.lock().expect("stream log should lock").len() >= expected {
            return;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("timed out waiting for slack stream ops");
}

async fn wait_for_slack_upload_count(uploads: &Arc<Mutex<Vec<FakeSlackUpload>>>, expected: usize) {
    for _ in 0..20 {
        if uploads.lock().expect("upload log should lock").len() >= expected {
            return;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("timed out waiting for slack uploads");
}

#[derive(Debug)]
struct FakeTurnRunner {
    outputs:
        Arc<Mutex<VecDeque<Result<TurnRunnerOutput, agent_controlplane::runner::TurnRunnerError>>>>,
    prepare_calls: Arc<Mutex<Vec<agent_controlplane::SessionId>>>,
    shutdown_calls: Arc<Mutex<u32>>,
}

struct CapturingTurnRunner {
    seen_targets: Arc<Mutex<Vec<ResponseTarget>>>,
}

#[derive(Clone, Debug, PartialEq)]
struct FakeSlackMessage {
    target: ChannelDeliveryTarget,
    payload: SlackMessagePayload,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum FakeSlackStreamOp {
    Start {
        target: ChannelDeliveryTarget,
        text: String,
    },
    Append {
        channel: String,
        ts: String,
        text: String,
    },
    Stop {
        channel: String,
        ts: String,
        text: Option<String>,
    },
}

#[derive(Debug)]
struct FakeSlackDeliveryClient {
    deliveries: Arc<Mutex<Vec<FakeSlackMessage>>>,
    stream_ops: Option<Arc<Mutex<Vec<FakeSlackStreamOp>>>>,
    uploads: Arc<Mutex<Vec<FakeSlackUpload>>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FakeWhatsAppMessage {
    account_id: String,
    target: ChannelDeliveryTarget,
    payload: WhatsAppMessagePayload,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FakeSlackUpload {
    target: ChannelDeliveryTarget,
    upload: SlackFileUpload,
}

#[derive(Debug)]
struct FakeWhatsAppDeliveryClient {
    deliveries: Arc<Mutex<Vec<FakeWhatsAppMessage>>>,
}

impl FakeWhatsAppDeliveryClient {
    fn new(deliveries: Arc<Mutex<Vec<FakeWhatsAppMessage>>>) -> Self {
        Self { deliveries }
    }
}

impl FakeSlackDeliveryClient {
    fn new(deliveries: Arc<Mutex<Vec<FakeSlackMessage>>>) -> Self {
        Self {
            deliveries,
            stream_ops: None,
            uploads: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn with_streams(
        deliveries: Arc<Mutex<Vec<FakeSlackMessage>>>,
        stream_ops: Arc<Mutex<Vec<FakeSlackStreamOp>>>,
    ) -> Self {
        Self {
            deliveries,
            stream_ops: Some(stream_ops),
            uploads: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn with_uploads(
        deliveries: Arc<Mutex<Vec<FakeSlackMessage>>>,
        uploads: Arc<Mutex<Vec<FakeSlackUpload>>>,
    ) -> Self {
        Self {
            deliveries,
            stream_ops: None,
            uploads,
        }
    }
}

#[async_trait]
impl SlackDeliveryClient for FakeSlackDeliveryClient {
    async fn post_message(
        &self,
        target: &ChannelDeliveryTarget,
        message: &SlackMessagePayload,
    ) -> Result<(), agent_controlplane::SlackDeliveryError> {
        self.deliveries
            .lock()
            .expect("delivery log should lock")
            .push(FakeSlackMessage {
                target: target.clone(),
                payload: message.clone(),
            });
        Ok(())
    }

    async fn start_stream(
        &self,
        target: &ChannelDeliveryTarget,
        markdown_text: &str,
    ) -> Result<SlackStreamHandle, agent_controlplane::SlackDeliveryError> {
        let stream_ops = self.stream_ops.as_ref().ok_or_else(|| {
            agent_controlplane::SlackDeliveryError::Api(
                "streaming not configured in fake client".to_owned(),
            )
        })?;
        stream_ops
            .lock()
            .expect("stream log should lock")
            .push(FakeSlackStreamOp::Start {
                target: target.clone(),
                text: markdown_text.to_owned(),
            });
        Ok(SlackStreamHandle {
            channel: target
                .external_channel_id
                .clone()
                .expect("slack target should include channel"),
            ts: "stream-ts-1".to_owned(),
        })
    }

    async fn append_stream(
        &self,
        handle: &SlackStreamHandle,
        markdown_text: &str,
    ) -> Result<(), agent_controlplane::SlackDeliveryError> {
        let stream_ops = self.stream_ops.as_ref().ok_or_else(|| {
            agent_controlplane::SlackDeliveryError::Api(
                "streaming not configured in fake client".to_owned(),
            )
        })?;
        stream_ops
            .lock()
            .expect("stream log should lock")
            .push(FakeSlackStreamOp::Append {
                channel: handle.channel.clone(),
                ts: handle.ts.clone(),
                text: markdown_text.to_owned(),
            });
        Ok(())
    }

    async fn stop_stream(
        &self,
        handle: &SlackStreamHandle,
        markdown_text: Option<&str>,
    ) -> Result<(), agent_controlplane::SlackDeliveryError> {
        let stream_ops = self.stream_ops.as_ref().ok_or_else(|| {
            agent_controlplane::SlackDeliveryError::Api(
                "streaming not configured in fake client".to_owned(),
            )
        })?;
        stream_ops
            .lock()
            .expect("stream log should lock")
            .push(FakeSlackStreamOp::Stop {
                channel: handle.channel.clone(),
                ts: handle.ts.clone(),
                text: markdown_text.map(ToOwned::to_owned),
            });
        Ok(())
    }

    async fn share_file(
        &self,
        target: &ChannelDeliveryTarget,
        upload: &SlackFileUpload,
    ) -> Result<(), agent_controlplane::SlackDeliveryError> {
        self.uploads
            .lock()
            .expect("upload log should lock")
            .push(FakeSlackUpload {
                target: target.clone(),
                upload: upload.clone(),
            });
        Ok(())
    }
}

#[async_trait]
impl WhatsAppDeliveryClient for FakeWhatsAppDeliveryClient {
    async fn send_message(
        &self,
        account_id: &str,
        target: &ChannelDeliveryTarget,
        message: &WhatsAppMessagePayload,
    ) -> Result<(), WhatsAppDeliveryError> {
        self.deliveries
            .lock()
            .expect("delivery log should lock")
            .push(FakeWhatsAppMessage {
                account_id: account_id.to_owned(),
                target: target.clone(),
                payload: message.clone(),
            });
        Ok(())
    }
}

#[derive(Debug, Default)]
struct StreamingFakeTurnRunner {
    reply_text: String,
}

impl StreamingFakeTurnRunner {
    fn new(reply_text: &str) -> Self {
        Self {
            reply_text: reply_text.to_owned(),
        }
    }
}

impl CapturingTurnRunner {
    fn new(seen_targets: Arc<Mutex<Vec<ResponseTarget>>>) -> Self {
        Self { seen_targets }
    }
}

#[async_trait]
impl TurnRunner for StreamingFakeTurnRunner {
    async fn run_turn(
        &self,
        input: TurnRunnerInput,
    ) -> Result<TurnRunnerOutput, agent_controlplane::runner::TurnRunnerError> {
        let turn_id = TurnId::new();
        let step_id = StepId::new();
        let executor = RuntimeExecutor::main_agent();
        for (index, event) in [
            RuntimeEvent::AnswerRenderStarted {
                turn_id: turn_id.clone(),
                step_id: step_id.clone(),
                executor: executor.clone(),
                at: SystemTime::now(),
            },
            RuntimeEvent::AnswerTextDelta {
                turn_id: turn_id.clone(),
                step_id: step_id.clone(),
                executor: executor.clone(),
                at: SystemTime::now(),
                delta: self.reply_text.clone(),
            },
            RuntimeEvent::TurnEnded {
                turn_id,
                executor,
                at: SystemTime::now(),
                termination: TerminationReason::Final,
                usage: UsageSummary::default(),
            },
        ]
        .into_iter()
        .enumerate()
        {
            let observation = RuntimeHarnessObservation::Event(RuntimeHarnessEventEnvelope {
                session_id: input.session_id.clone(),
                turn_number: input.turn_number,
                event_index: (index + 1) as u32,
                event,
            });
            for listener in &input.runtime_harness_listeners {
                listener
                    .try_observe(observation.clone())
                    .expect("stream observation should be accepted");
            }
        }
        Ok(turn_output(&self.reply_text))
    }
}

impl FakeTurnRunner {
    fn new(
        outputs: Vec<Result<TurnRunnerOutput, agent_controlplane::runner::TurnRunnerError>>,
    ) -> Self {
        Self {
            outputs: Arc::new(Mutex::new(outputs.into())),
            prepare_calls: Arc::new(Mutex::new(Vec::new())),
            shutdown_calls: Arc::new(Mutex::new(0)),
        }
    }
}

#[async_trait]
impl TurnRunner for FakeTurnRunner {
    async fn prepare_session(
        &self,
        session_id: &agent_controlplane::SessionId,
    ) -> Result<(), agent_controlplane::runner::TurnRunnerError> {
        self.prepare_calls
            .lock()
            .expect("prepare calls should lock")
            .push(session_id.clone());
        Ok(())
    }

    async fn shutdown(&self) -> Result<(), agent_controlplane::runner::TurnRunnerError> {
        let mut shutdown_calls = self
            .shutdown_calls
            .lock()
            .expect("shutdown calls should lock");
        *shutdown_calls += 1;
        Ok(())
    }

    async fn run_turn(
        &self,
        input: TurnRunnerInput,
    ) -> Result<TurnRunnerOutput, agent_controlplane::runner::TurnRunnerError> {
        let _ = &input.session_id;
        let _ = input.turn_number;
        let _ = input.runtime_harness_listeners.len();
        if let Some(first) = input.conversation_history.first() {
            assert!(matches!(
                first.role,
                ConversationRole::User | ConversationRole::Assistant
            ));
            let _ = &first.content;
        }
        self.outputs
            .lock()
            .expect("outputs should lock")
            .pop_front()
            .unwrap_or_else(|| Ok(turn_output("default")))
    }
}

#[async_trait]
impl TurnRunner for CapturingTurnRunner {
    async fn run_turn(
        &self,
        input: TurnRunnerInput,
    ) -> Result<TurnRunnerOutput, agent_controlplane::runner::TurnRunnerError> {
        self.seen_targets
            .lock()
            .expect("seen targets should lock")
            .push(input.response_target);
        Ok(turn_output("captured"))
    }
}
