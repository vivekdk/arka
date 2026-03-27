//! Low-level MCP message framing helpers.
//!
//! The transport layer currently writes newline-delimited JSON over stdio, but
//! this codec also accepts `Content-Length` framing so the client remains
//! tolerant of older or more protocol-strict servers.

use serde_json::Value;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncReadExt};

use crate::McpClientError;

/// Test-only helper that encodes one JSON message using `Content-Length`
/// framing.
#[cfg(test)]
pub(crate) fn encode_content_length_message(value: &Value) -> Result<Vec<u8>, serde_json::Error> {
    let body = serde_json::to_vec(value)?;
    let mut message = format!("Content-Length: {}\r\n\r\n", body.len()).into_bytes();
    message.extend_from_slice(&body);
    Ok(message)
}

/// Encodes one JSON message as newline-delimited stdio output.
pub fn encode_newline_message(value: &Value) -> Result<Vec<u8>, serde_json::Error> {
    let mut body = serde_json::to_vec(value)?;
    body.push(b'\n');
    Ok(body)
}

/// Reads exactly one MCP message from a buffered stream.
///
/// The reader accepts both newline-delimited JSON and `Content-Length` frames.
/// Returning `Ok(None)` indicates EOF before another message began.
pub async fn read_message<R>(reader: &mut R) -> Result<Option<Value>, McpClientError>
where
    R: AsyncBufRead + Unpin,
{
    loop {
        let mut line = Vec::new();
        let bytes_read = reader
            .read_until(b'\n', &mut line)
            .await
            .map_err(McpClientError::Io)?;

        if bytes_read == 0 {
            return Ok(None);
        }

        let line = trim_line_endings(&line);
        if line.is_empty() {
            continue;
        }

        let line_text = std::str::from_utf8(line).map_err(|err| {
            McpClientError::Transport(format!("invalid UTF-8 in MCP frame: {err}"))
        })?;
        let trimmed = line_text.trim_start();

        // MCP stdio is newline-delimited today, but the reader also accepts
        // Content-Length framing so the transport layer stays tolerant.
        if trimmed.starts_with('{') || trimmed.starts_with('[') {
            let value = serde_json::from_str::<Value>(trimmed).map_err(McpClientError::Json)?;
            return Ok(Some(value));
        }

        if line_text
            .to_ascii_lowercase()
            .starts_with("content-length:")
        {
            // After the length header, consume any remaining headers until the
            // blank separator line, then read exactly the declared body bytes.
            let content_length = parse_content_length(line_text)?;

            loop {
                let mut header_line = Vec::new();
                reader
                    .read_until(b'\n', &mut header_line)
                    .await
                    .map_err(McpClientError::Io)?;
                if header_line.is_empty() {
                    return Err(McpClientError::Transport(
                        "unexpected EOF while reading MCP headers".to_owned(),
                    ));
                }

                if trim_line_endings(&header_line).is_empty() {
                    break;
                }
            }

            let mut body = vec![0; content_length];
            reader
                .read_exact(&mut body)
                .await
                .map_err(McpClientError::Io)?;
            let value = serde_json::from_slice::<Value>(&body).map_err(McpClientError::Json)?;
            return Ok(Some(value));
        }

        return Err(McpClientError::Transport(format!(
            "unsupported MCP stdio frame header `{line_text}`"
        )));
    }
}

fn parse_content_length(header: &str) -> Result<usize, McpClientError> {
    let (_, raw_length) = header.split_once(':').ok_or_else(|| {
        McpClientError::Transport("missing `:` in Content-Length header".to_owned())
    })?;

    raw_length
        .trim()
        .parse::<usize>()
        .map_err(|err| McpClientError::Transport(format!("invalid Content-Length header: {err}")))
}

fn trim_line_endings(line: &[u8]) -> &[u8] {
    let mut end = line.len();
    while end > 0 && matches!(line[end - 1], b'\n' | b'\r') {
        end -= 1;
    }
    &line[..end]
}

#[cfg(test)]
mod tests {
    //! Framing regressions for newline-delimited and `Content-Length` messages.

    use serde_json::json;
    use tokio::io::{AsyncWriteExt, BufReader, duplex};

    use super::{encode_content_length_message, encode_newline_message, read_message};

    #[tokio::test]
    async fn reads_newline_delimited_messages() {
        let message = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": { "ok": true }
        });
        let (mut writer, reader) = duplex(1024);

        writer
            .write_all(&encode_newline_message(&message).expect("message encoding"))
            .await
            .expect("write should succeed");
        drop(writer);

        let mut reader = BufReader::new(reader);
        let parsed = read_message(&mut reader)
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
            .write_all(&encode_content_length_message(&message).expect("message encoding"))
            .await
            .expect("write should succeed");
        drop(writer);

        let mut reader = BufReader::new(reader);
        let parsed = read_message(&mut reader)
            .await
            .expect("read should succeed")
            .expect("message should exist");

        assert_eq!(parsed, message);
    }
}
