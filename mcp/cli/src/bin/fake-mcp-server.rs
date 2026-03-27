//! Tiny fake MCP server used by `mcp-cli` integration tests.
//!
//! The process speaks newline-delimited JSON-RPC over stdio and exposes a
//! fixed tool catalog so tests can validate client handshake and env passing.

use std::io::{self, BufRead, Write};

use serde_json::{Value, json};

fn main() {
    let mut require_env = None;
    for arg in std::env::args().skip(1) {
        if arg == "--exit-immediately" {
            return;
        }
        if let Some(value) = arg.strip_prefix("--require-env=") {
            require_env = Some(value.to_owned());
        }
    }

    if let Some(var_name) = require_env {
        if std::env::var_os(&var_name).is_none() {
            return;
        }
    }

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = io::BufReader::new(stdin.lock());
    let mut writer = io::BufWriter::new(stdout.lock());
    let mut line = String::new();

    loop {
        line.clear();
        let bytes_read = reader
            .read_line(&mut line)
            .expect("stdin should be readable");
        if bytes_read == 0 {
            break;
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let message = serde_json::from_str::<Value>(trimmed).expect("valid JSON-RPC request");
        let Some(method) = message.get("method").and_then(Value::as_str) else {
            continue;
        };

        match method {
            "initialize" => respond(
                &mut writer,
                message.get("id").cloned().expect("initialize request id"),
                json!({
                    "protocolVersion": "2025-11-25",
                    "capabilities": {
                        "tools": {
                            "listChanged": false
                        },
                        "resources": {
                            "listChanged": false
                        }
                    },
                    "serverInfo": {
                        "name": "fake-server",
                        "version": "1.0.0"
                    }
                }),
            ),
            "notifications/initialized" => {}
            "tools/list" => respond(
                &mut writer,
                message.get("id").cloned().expect("tools/list request id"),
                json!({
                    "tools": [
                        {
                            "name": "run-sql",
                            "description": "Execute a SQL query",
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
                }),
            ),
            "resources/list" => respond(
                &mut writer,
                message
                    .get("id")
                    .cloned()
                    .expect("resources/list request id"),
                json!({
                    "resources": [
                        {
                            "uri": "fake://docs/overview",
                            "name": "overview",
                            "title": "Overview",
                            "description": "Overview resource",
                            "mimeType": "text/plain"
                        }
                    ]
                }),
            ),
            "resources/read" => respond(
                &mut writer,
                message
                    .get("id")
                    .cloned()
                    .expect("resources/read request id"),
                json!({
                    "contents": [
                        {
                            "uri": "fake://docs/overview",
                            "mimeType": "text/plain",
                            "text": "fake resource body"
                        }
                    ]
                }),
            ),
            "tools/call" => {
                let id = message.get("id").cloned().expect("tools/call request id");
                let params = message.get("params").expect("tools/call params");
                let tool_name = params
                    .get("name")
                    .and_then(Value::as_str)
                    .expect("tool name must be a string");

                match tool_name {
                    "run-sql" => respond(
                        &mut writer,
                        id,
                        json!({
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
                        }),
                    ),
                    "describe-table" => respond(
                        &mut writer,
                        id,
                        json!({
                            "content": [
                                {
                                    "type": "text",
                                    "text": "columns: id, name"
                                }
                            ],
                            "structuredContent": {
                                "columns": ["id", "name"]
                            },
                            "isError": false
                        }),
                    ),
                    "fail-tool" => respond(
                        &mut writer,
                        id,
                        json!({
                            "content": [
                                {
                                    "type": "text",
                                    "text": "simulated MCP tool failure"
                                }
                            ],
                            "structuredContent": {
                                "reason": "simulated"
                            },
                            "isError": true
                        }),
                    ),
                    _ => {
                        let response = json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "error": {
                                "code": -32602,
                                "message": format!("unknown tool `{tool_name}`")
                            }
                        });
                        writeln!(writer, "{response}").expect("error response write");
                        writer.flush().expect("writer flush");
                    }
                }
            }
            _ => {
                if let Some(id) = message.get("id").cloned() {
                    let response = json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": {
                            "code": -32601,
                            "message": format!("unknown method `{method}`")
                        }
                    });
                    writeln!(writer, "{response}").expect("error response write");
                    writer.flush().expect("writer flush");
                }
            }
        }
    }
}

fn respond(writer: &mut impl Write, id: Value, result: Value) {
    let response = json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result
    });

    writeln!(writer, "{response}").expect("response write");
    writer.flush().expect("writer flush");
}
