//! Channel adapter abstractions and concrete Slack/WhatsApp normalization.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::http::HeaderMap;
use hmac::{Hmac, Mac};
use serde::Deserialize;
use sha2::Sha256;
use thiserror::Error;

use crate::types::{
    ChannelBinding, ChannelDeliveryTarget, ChannelEnvelope, ChannelIntent, ChannelKind,
    ChannelResponseEnvelope, OutboundMessageKind, SlackWebhookPayload, WhatsAppWebhookPayload,
};

type HmacSha256 = Hmac<Sha256>;

const SLACK_SIGNATURE_VERSION: &str = "v0";
const SLACK_MAX_REQUEST_AGE_SECS: i64 = 60 * 5;

/// Channel adapter contract used to normalize inbound traffic and render
/// outbound envelopes for delivery.
pub trait ChannelAdapter {
    /// Normalizes one external payload into a channel-neutral envelope.
    type Inbound;

    /// Converts one channel-specific payload into a normalized envelope.
    fn ingest(&self, inbound: Self::Inbound) -> Result<ChannelEnvelope, ChannelAdapterError>;

    /// Renders one normalized outbound response into channel-specific text.
    fn render(&self, outbound: &ChannelResponseEnvelope) -> Result<String, ChannelAdapterError>;
}

/// Result of parsing one real Slack HTTP request.
#[derive(Clone, Debug, PartialEq)]
pub enum SlackIngressResult {
    /// Slack URL verification challenge payload.
    UrlVerification { challenge: String },
    /// Request should be acknowledged with no further processing.
    AckOnly,
    /// Parsed event to be processed asynchronously.
    Dispatch(SlackDispatchRequest),
}

/// One verified Slack event ready for worker processing.
#[derive(Clone, Debug, PartialEq)]
pub struct SlackDispatchRequest {
    /// Route key used for session lookup and rebinding.
    pub binding: ChannelBinding,
    /// Delivery target for outbound Slack replies.
    pub delivery_target: ChannelDeliveryTarget,
    /// Whether the route must already have an active session.
    pub requires_existing_session: bool,
    /// Optional immediate reply that should bypass the runtime.
    pub immediate_text: Option<String>,
    /// Normalized control-plane envelope to dispatch.
    pub envelope: ChannelEnvelope,
}

/// Adapter for Slack thread and mention events.
#[derive(Clone, Debug, Default)]
pub struct SlackChannelAdapter {
    signing_secret: Option<String>,
}

impl SlackChannelAdapter {
    pub fn new(signing_secret: Option<String>) -> Self {
        Self { signing_secret }
    }

    pub fn ingest_http_request(
        &self,
        headers: &HeaderMap,
        body: &[u8],
    ) -> Result<SlackIngressResult, ChannelAdapterError> {
        self.verify_request(headers, body)?;
        self.parse_http_request(body)
    }

    fn parse_http_request(&self, body: &[u8]) -> Result<SlackIngressResult, ChannelAdapterError> {
        let payload: SlackEventRequest = serde_json::from_slice(body)
            .map_err(|error| ChannelAdapterError::InvalidPayload(error.to_string()))?;
        match payload.request_type.as_str() {
            "url_verification" => Ok(SlackIngressResult::UrlVerification {
                challenge: payload.challenge.ok_or_else(|| {
                    ChannelAdapterError::InvalidPayload(
                        "slack url verification request is missing challenge".to_owned(),
                    )
                })?,
            }),
            "event_callback" => {
                let team_id = payload.team_id.ok_or_else(|| {
                    ChannelAdapterError::InvalidPayload(
                        "slack event callback is missing team_id".to_owned(),
                    )
                })?;
                let event_id = payload.event_id.ok_or_else(|| {
                    ChannelAdapterError::InvalidPayload(
                        "slack event callback is missing event_id".to_owned(),
                    )
                })?;
                let event = payload.event.ok_or_else(|| {
                    ChannelAdapterError::InvalidPayload(
                        "slack event callback is missing event payload".to_owned(),
                    )
                })?;
                self.parse_event(team_id, event_id, event)
            }
            other => Err(ChannelAdapterError::InvalidPayload(format!(
                "unsupported slack request type `{other}`"
            ))),
        }
    }

    fn parse_event(
        &self,
        team_id: String,
        event_id: String,
        event: SlackEventPayload,
    ) -> Result<SlackIngressResult, ChannelAdapterError> {
        if event.subtype.is_some() || event.bot_id.is_some() {
            return Ok(SlackIngressResult::AckOnly);
        }

        let user_id = match event.user {
            Some(user_id) if !user_id.trim().is_empty() => user_id,
            _ => return Ok(SlackIngressResult::AckOnly),
        };
        let channel_id = event.channel.ok_or_else(|| {
            ChannelAdapterError::InvalidPayload("slack event is missing channel".to_owned())
        })?;
        let message_ts = event.ts.ok_or_else(|| {
            ChannelAdapterError::InvalidPayload("slack event is missing ts".to_owned())
        })?;
        let thread_root_ts = event
            .thread_ts
            .clone()
            .unwrap_or_else(|| message_ts.clone());
        let binding = ChannelBinding {
            channel: ChannelKind::Slack,
            external_workspace_id: Some(team_id.clone()),
            external_conversation_id: thread_root_ts.clone(),
            external_channel_id: Some(channel_id.clone()),
            external_thread_id: Some(thread_root_ts.clone()),
            external_user_id: user_id.clone(),
        };
        let delivery_target = ChannelDeliveryTarget {
            channel: ChannelKind::Slack,
            external_workspace_id: Some(team_id),
            external_conversation_id: thread_root_ts.clone(),
            external_channel_id: Some(channel_id),
            external_thread_id: Some(thread_root_ts.clone()),
            external_user_id: user_id.clone(),
        };
        let raw_text = match event.text {
            Some(text) if !text.trim().is_empty() => text,
            _ => return Ok(SlackIngressResult::AckOnly),
        };

        match event.event_type.as_str() {
            "app_mention" => {
                let text = strip_leading_slack_mentions(&raw_text).trim().to_owned();
                if text.is_empty() {
                    return Ok(SlackIngressResult::Dispatch(SlackDispatchRequest {
                        binding: binding.clone(),
                        delivery_target,
                        requires_existing_session: false,
                        immediate_text: Some(
                            "Ask a question after mentioning me, or use `@app /new` to reset the session."
                                .to_owned(),
                        ),
                        envelope: ChannelEnvelope {
                            channel: ChannelKind::Slack,
                            external_workspace_id: binding.external_workspace_id.clone(),
                            external_conversation_id: thread_root_ts,
                            external_channel_id: binding.external_channel_id.clone(),
                            external_thread_id: binding.external_thread_id.clone(),
                            external_user_id: user_id,
                            external_message_id: Some(event_id.clone()),
                            idempotency_key: event_id,
                            occurred_at: SystemTime::now(),
                            intent: ChannelIntent::StatusRequest,
                        },
                    }));
                }
                let intent = if text == "/new" {
                    ChannelIntent::ResetSession
                } else {
                    ChannelIntent::UserText { text }
                };
                Ok(SlackIngressResult::Dispatch(SlackDispatchRequest {
                    binding: binding.clone(),
                    delivery_target,
                    requires_existing_session: false,
                    immediate_text: None,
                    envelope: ChannelEnvelope {
                        channel: ChannelKind::Slack,
                        external_workspace_id: binding.external_workspace_id.clone(),
                        external_conversation_id: thread_root_ts,
                        external_channel_id: binding.external_channel_id.clone(),
                        external_thread_id: binding.external_thread_id.clone(),
                        external_user_id: user_id,
                        external_message_id: Some(event_id.clone()),
                        idempotency_key: event_id,
                        occurred_at: SystemTime::now(),
                        intent,
                    },
                }))
            }
            "message" => Ok(SlackIngressResult::Dispatch(SlackDispatchRequest {
                binding: binding.clone(),
                delivery_target,
                requires_existing_session: true,
                immediate_text: None,
                envelope: ChannelEnvelope {
                    channel: ChannelKind::Slack,
                    external_workspace_id: binding.external_workspace_id.clone(),
                    external_conversation_id: thread_root_ts,
                    external_channel_id: binding.external_channel_id.clone(),
                    external_thread_id: binding.external_thread_id.clone(),
                    external_user_id: user_id,
                    external_message_id: Some(event_id.clone()),
                    idempotency_key: event_id,
                    occurred_at: SystemTime::now(),
                    intent: ChannelIntent::UserText { text: raw_text },
                },
            })),
            _ => Ok(SlackIngressResult::AckOnly),
        }
    }

    fn verify_request(&self, headers: &HeaderMap, body: &[u8]) -> Result<(), ChannelAdapterError> {
        let signing_secret = self.signing_secret.as_ref().ok_or_else(|| {
            ChannelAdapterError::NotConfigured("slack is not configured".to_owned())
        })?;
        let timestamp = header_value(headers, "x-slack-request-timestamp")?;
        let signature = header_value(headers, "x-slack-signature")?;
        let now_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::from_secs(0))
            .as_secs() as i64;
        verify_slack_signature(signing_secret, body, &timestamp, &signature, now_secs)
    }
}

impl ChannelAdapter for SlackChannelAdapter {
    type Inbound = SlackWebhookPayload;

    fn ingest(&self, inbound: SlackWebhookPayload) -> Result<ChannelEnvelope, ChannelAdapterError> {
        let thread_root_ts = inbound
            .thread_ts
            .clone()
            .unwrap_or_else(|| inbound.channel_id.clone());
        Ok(ChannelEnvelope {
            channel: ChannelKind::Slack,
            external_workspace_id: None,
            external_conversation_id: thread_root_ts.clone(),
            external_channel_id: Some(inbound.channel_id),
            external_thread_id: Some(thread_root_ts),
            external_user_id: inbound.user_id,
            external_message_id: Some(inbound.event_id.clone()),
            idempotency_key: inbound.event_id,
            occurred_at: SystemTime::now(),
            intent: ChannelIntent::UserText { text: inbound.text },
        })
    }

    fn render(&self, outbound: &ChannelResponseEnvelope) -> Result<String, ChannelAdapterError> {
        let target = outbound.delivery_target.as_ref().ok_or_else(|| {
            ChannelAdapterError::InvalidPayload(
                "slack outbound message is missing delivery target".to_owned(),
            )
        })?;
        let prefix = format!("<@{}> ", target.external_user_id);
        let body = match outbound.kind {
            OutboundMessageKind::Reply
            | OutboundMessageKind::Status
            | OutboundMessageKind::ApprovalPrompt
            | OutboundMessageKind::Error => outbound.text.clone(),
        };
        Ok(format!("{prefix}{body}"))
    }
}

/// Adapter for WhatsApp business webhooks.
#[derive(Clone, Debug, Default)]
pub struct WhatsAppChannelAdapter;

impl ChannelAdapter for WhatsAppChannelAdapter {
    type Inbound = WhatsAppWebhookPayload;

    fn ingest(
        &self,
        inbound: WhatsAppWebhookPayload,
    ) -> Result<ChannelEnvelope, ChannelAdapterError> {
        Ok(ChannelEnvelope {
            channel: ChannelKind::WhatsApp,
            external_workspace_id: None,
            external_conversation_id: inbound.conversation_id,
            external_channel_id: None,
            external_thread_id: None,
            external_user_id: inbound.from_user_id,
            external_message_id: Some(inbound.message_id.clone()),
            idempotency_key: inbound.message_id,
            occurred_at: SystemTime::now(),
            intent: ChannelIntent::UserText { text: inbound.text },
        })
    }

    fn render(&self, outbound: &ChannelResponseEnvelope) -> Result<String, ChannelAdapterError> {
        Ok(outbound.text.clone())
    }
}

fn header_value(headers: &HeaderMap, key: &'static str) -> Result<String, ChannelAdapterError> {
    headers
        .get(key)
        .ok_or_else(|| ChannelAdapterError::InvalidPayload(format!("missing header `{key}`")))?
        .to_str()
        .map(|value| value.to_owned())
        .map_err(|_| ChannelAdapterError::InvalidPayload(format!("invalid header `{key}`")))
}

fn verify_slack_signature(
    signing_secret: &str,
    body: &[u8],
    timestamp: &str,
    signature: &str,
    now_secs: i64,
) -> Result<(), ChannelAdapterError> {
    let request_ts = timestamp.parse::<i64>().map_err(|_| {
        ChannelAdapterError::InvalidPayload(
            "slack request timestamp is not a valid integer".to_owned(),
        )
    })?;
    if (now_secs - request_ts).abs() > SLACK_MAX_REQUEST_AGE_SECS {
        return Err(ChannelAdapterError::InvalidPayload(
            "slack request timestamp is too old".to_owned(),
        ));
    }
    let signed_payload = format!(
        "{SLACK_SIGNATURE_VERSION}:{timestamp}:{}",
        String::from_utf8_lossy(body)
    );
    let mut mac = HmacSha256::new_from_slice(signing_secret.as_bytes()).map_err(|_| {
        ChannelAdapterError::InvalidPayload("invalid slack signing secret".to_owned())
    })?;
    mac.update(signed_payload.as_bytes());
    let expected = signature
        .strip_prefix(&format!("{SLACK_SIGNATURE_VERSION}="))
        .ok_or_else(|| {
            ChannelAdapterError::InvalidPayload(
                "slack signature does not use the expected version".to_owned(),
            )
        })?;
    let expected = decode_hex(expected)?;
    mac.verify_slice(&expected).map_err(|_| {
        ChannelAdapterError::InvalidPayload("slack request signature is invalid".to_owned())
    })
}

fn decode_hex(value: &str) -> Result<Vec<u8>, ChannelAdapterError> {
    if value.len() % 2 != 0 {
        return Err(ChannelAdapterError::InvalidPayload(
            "slack signature hex is malformed".to_owned(),
        ));
    }
    let mut bytes = Vec::with_capacity(value.len() / 2);
    for chunk in value.as_bytes().chunks(2) {
        let pair = std::str::from_utf8(chunk).map_err(|_| {
            ChannelAdapterError::InvalidPayload("slack signature hex is malformed".to_owned())
        })?;
        let byte = u8::from_str_radix(pair, 16).map_err(|_| {
            ChannelAdapterError::InvalidPayload("slack signature hex is malformed".to_owned())
        })?;
        bytes.push(byte);
    }
    Ok(bytes)
}

fn strip_leading_slack_mentions(text: &str) -> &str {
    let trimmed = text.trim_start();
    let mut remainder = trimmed;
    loop {
        let next = remainder.trim_start();
        if !next.starts_with("<@") {
            return next;
        }
        let Some(end) = next.find('>') else {
            return trimmed;
        };
        remainder = &next[end + 1..];
    }
}

#[derive(Debug, Deserialize)]
struct SlackEventRequest {
    #[serde(rename = "type")]
    request_type: String,
    challenge: Option<String>,
    team_id: Option<String>,
    event_id: Option<String>,
    event: Option<SlackEventPayload>,
}

#[derive(Clone, Debug, Deserialize)]
struct SlackEventPayload {
    #[serde(rename = "type")]
    event_type: String,
    subtype: Option<String>,
    bot_id: Option<String>,
    user: Option<String>,
    channel: Option<String>,
    text: Option<String>,
    thread_ts: Option<String>,
    ts: Option<String>,
}

/// Channel adapter failures.
#[derive(Debug, Error)]
pub enum ChannelAdapterError {
    #[error("channel payload is invalid: {0}")]
    InvalidPayload(String),
    #[error("channel adapter is not configured: {0}")]
    NotConfigured(String),
}

#[cfg(test)]
mod tests {
    use axum::http::HeaderValue;
    use hmac::Mac;

    use super::{
        ChannelAdapter, SlackChannelAdapter, SlackIngressResult, WhatsAppChannelAdapter,
        verify_slack_signature,
    };
    use crate::types::{
        ChannelDeliveryTarget, ChannelKind, ChannelResponseEnvelope, OutboundMessageKind, SessionId,
    };

    #[test]
    fn whatsapp_render_keeps_full_outbound_text() {
        let adapter = WhatsAppChannelAdapter;
        let text = "x".repeat(600);
        let outbound = ChannelResponseEnvelope {
            session_id: SessionId::new(),
            kind: OutboundMessageKind::Reply,
            text: text.clone(),
            attachment: None,
            delivery_target: None,
        };

        let rendered = adapter
            .render(&outbound)
            .expect("whatsapp render should succeed");

        assert_eq!(rendered, text);
        assert!(!rendered.ends_with("..."));
    }

    #[test]
    fn slack_render_prefixes_target_user() {
        let adapter = SlackChannelAdapter::default();
        let outbound = ChannelResponseEnvelope {
            session_id: SessionId::new(),
            kind: OutboundMessageKind::Reply,
            text: "hello".to_owned(),
            attachment: None,
            delivery_target: Some(ChannelDeliveryTarget {
                channel: ChannelKind::Slack,
                external_workspace_id: Some("T1".to_owned()),
                external_conversation_id: "thread-1".to_owned(),
                external_channel_id: Some("C1".to_owned()),
                external_thread_id: Some("thread-1".to_owned()),
                external_user_id: "U1".to_owned(),
            }),
        };

        let rendered = adapter
            .render(&outbound)
            .expect("slack render should succeed");

        assert_eq!(rendered, "<@U1> hello");
    }

    #[test]
    fn slack_ingest_parses_app_mention_into_reset() {
        let adapter = SlackChannelAdapter::new(Some("secret".to_owned()));
        let body = br#"{
            "type": "event_callback",
            "team_id": "T1",
            "event_id": "Ev1",
            "event": {
                "type": "app_mention",
                "user": "U1",
                "channel": "C1",
                "text": "<@B1> /new",
                "thread_ts": "1700000000.000100",
                "ts": "1700000001.000200"
            }
        }"#;
        let headers = signed_headers("secret", body, current_timestamp());

        let result = adapter
            .ingest_http_request(&headers, body)
            .expect("slack request should parse");

        let SlackIngressResult::Dispatch(dispatch) = result else {
            panic!("expected dispatch");
        };
        assert_eq!(
            dispatch.envelope.intent,
            crate::types::ChannelIntent::ResetSession
        );
        assert!(!dispatch.requires_existing_session);
        assert_eq!(dispatch.binding.external_user_id, "U1");
        assert_eq!(
            dispatch.binding.external_conversation_id,
            "1700000000.000100"
        );
    }

    #[test]
    fn slack_ingest_requires_existing_session_for_plain_thread_reply() {
        let adapter = SlackChannelAdapter::new(Some("secret".to_owned()));
        let body = br#"{
            "type": "event_callback",
            "team_id": "T1",
            "event_id": "Ev2",
            "event": {
                "type": "message",
                "user": "U1",
                "channel": "C1",
                "text": "follow up",
                "thread_ts": "1700000000.000100",
                "ts": "1700000001.000200"
            }
        }"#;
        let headers = signed_headers("secret", body, current_timestamp());

        let result = adapter
            .ingest_http_request(&headers, body)
            .expect("slack request should parse");

        let SlackIngressResult::Dispatch(dispatch) = result else {
            panic!("expected dispatch");
        };
        assert!(dispatch.requires_existing_session);
    }

    #[test]
    fn slack_verification_rejects_stale_timestamp() {
        let body = br#"{"type":"url_verification","challenge":"test"}"#;
        let signature = sign("secret", body, 1_700_000_000);
        let error = verify_slack_signature("secret", body, "1700000000", &signature, 1_700_050_000)
            .expect_err("stale request should be rejected");
        assert!(error.to_string().contains("too old"));
    }

    fn signed_headers(secret: &str, body: &[u8], timestamp: i64) -> axum::http::HeaderMap {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(
            "x-slack-request-timestamp",
            HeaderValue::from_str(&timestamp.to_string()).expect("timestamp header"),
        );
        headers.insert(
            "x-slack-signature",
            HeaderValue::from_str(&sign(secret, body, timestamp)).expect("signature header"),
        );
        headers
    }

    fn sign(secret: &str, body: &[u8], timestamp: i64) -> String {
        let signed_payload = format!("v0:{timestamp}:{}", String::from_utf8_lossy(body));
        let mut mac =
            super::HmacSha256::new_from_slice(secret.as_bytes()).expect("mac should initialize");
        mac.update(signed_payload.as_bytes());
        format!("v0={:x}", mac.finalize().into_bytes())
    }

    fn current_timestamp() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("current time should be after epoch")
            .as_secs() as i64
    }
}
