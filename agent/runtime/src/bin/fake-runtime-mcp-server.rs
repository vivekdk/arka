//! Fake MCP server used by runtime integration tests.
//!
//! The binary exposes a controllable tool catalog and optional request logging
//! so tests can verify bootstrap probing, cache behavior, and session reuse.

use std::io::{self, BufRead, Write};
use std::{env, fs::OpenOptions};

use serde_json::{Value, json};

fn main() {
    let mut args = env::args().skip(1);
    let mut log_file = None;
    let mut tool_mode = "default".to_owned();
    let mut poisoned = false;
    while let Some(arg) = args.next() {
        if arg == "--log-file" {
            log_file = args.next();
        } else if arg == "--tool-mode" {
            tool_mode = args.next().unwrap_or_else(|| "default".to_owned());
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
        log_request(log_file.as_deref(), method);

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
                        "name": "fake-runtime-server",
                        "version": "1.0.0"
                    }
                }),
            ),
            "notifications/initialized" => {}
            "tools/list" => respond(
                &mut writer,
                message.get("id").cloned().expect("tools/list request id"),
                json!({ "tools": tool_catalog(&tool_mode) }),
            ),
            "resources/list" => respond(
                &mut writer,
                message
                    .get("id")
                    .cloned()
                    .expect("resources/list request id"),
                json!({ "resources": resource_catalog() }),
            ),
            "resources/read" => {
                let id = message
                    .get("id")
                    .cloned()
                    .expect("resources/read request id");
                let params = message.get("params").expect("resources/read params");
                let uri = params
                    .get("uri")
                    .and_then(Value::as_str)
                    .expect("resource uri must be a string");
                respond(
                    &mut writer,
                    id,
                    json!({
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": "application/json",
                                "text": "{\"dashboard\":\"main\",\"rows\":3}"
                            }
                        ]
                    }),
                )
            }
            "tools/call" => {
                let id = message.get("id").cloned().expect("tools/call request id");
                let params = message.get("params").expect("tools/call params");
                let tool_name = params
                    .get("name")
                    .and_then(Value::as_str)
                    .expect("tool name must be a string");
                log_tool_call(log_file.as_deref(), tool_name);

                match tool_name {
                    "run-sql" => {
                        if tool_mode == "poison-after-tool-error" && poisoned {
                            respond(
                                &mut writer,
                                id,
                                json!({
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": "connection is poisoned until the process restarts"
                                        }
                                    ],
                                    "structuredContent": {
                                        "reason": "poisoned_connection"
                                    },
                                    "isError": true
                                }),
                            );
                        } else {
                            respond(
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
                            );
                        }
                    }
                    "fail-tool" => {
                        if tool_mode == "poison-after-tool-error" {
                            poisoned = true;
                        }
                        respond(
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
                        );
                    }
                    "list_tables" => respond(
                        &mut writer,
                        id,
                        json!({
                            "content": [
                                {
                                    "type": "text",
                                    "text": "tables: leads, lead_stages, users"
                                }
                            ],
                            "structuredContent": {
                                "tables": ["leads", "lead_stages", "users"]
                            },
                            "isError": false
                        }),
                    ),
                    "describe_leads" => respond(
                        &mut writer,
                        id,
                        json!({
                            "content": [
                                {
                                    "type": "text",
                                    "text": "leads are stored in the leads table and use stage = NOT_CONTACTED for untouched leads"
                                }
                            ],
                            "structuredContent": {
                                "entity": "leads",
                                "table": "leads",
                                "stage_column": "stage",
                                "not_contacted_value": "NOT_CONTACTED"
                            },
                            "isError": false
                        }),
                    ),
                    "preview_leads" => respond(
                        &mut writer,
                        id,
                        json!({
                            "content": [
                                {
                                    "type": "text",
                                    "text": "sample rows include stage NOT_CONTACTED and CONTACTED"
                                }
                            ],
                            "structuredContent": {
                                "rows": [
                                    {
                                        "name": "Alice",
                                        "email": "alice@example.com",
                                        "stage": "NOT_CONTACTED",
                                        "user_id": "usr_123456789"
                                    },
                                    {
                                        "name": "Bob",
                                        "email": "bob@example.com",
                                        "stage": "CONTACTED",
                                        "user_id": "usr_987654321"
                                    }
                                ]
                            },
                            "isError": false
                        }),
                    ),
                    "delete_leads" => respond(
                        &mut writer,
                        id,
                        json!({
                            "content": [
                                {
                                    "type": "text",
                                    "text": "delete should never be auto-called"
                                }
                            ],
                            "structuredContent": {
                                "ok": true
                            },
                            "isError": false
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

fn tool_catalog(tool_mode: &str) -> Vec<Value> {
    if tool_mode == "metadata-only" {
        return vec![
            json!({
                "name": "delete_leads",
                "description": "Delete lead records",
                "inputSchema": {
                    "type": "object"
                }
            }),
            json!({
                "name": "run-sql",
                "description": "Execute a SQL query",
                "inputSchema": {
                    "type": "object"
                }
            }),
        ];
    }

    vec![
        json!({
            "name": "run-sql",
            "description": "Execute a SQL query",
            "inputSchema": {
                "type": "object"
            }
        }),
        json!({
            "name": "fail-tool",
            "description": "Return an MCP-level error payload",
            "inputSchema": {
                "type": "object"
            }
        }),
        json!({
            "name": "list_tables",
            "description": "List tables for semantic bootstrap",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": { "type": "integer" }
                }
            }
        }),
        json!({
            "name": "describe_leads",
            "description": "Describe the leads business object",
            "inputSchema": {
                "type": "object"
            }
        }),
        json!({
            "name": "preview_leads",
            "description": "Preview sample lead rows",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": { "type": "integer" }
                }
            }
        }),
        json!({
            "name": "delete_leads",
            "description": "Delete lead records",
            "inputSchema": {
                "type": "object"
            }
        }),
    ]
}

fn resource_catalog() -> Vec<Value> {
    vec![json!({
        "uri": "crm://dashboards/main",
        "name": "main_dashboard",
        "title": "Main Dashboard",
        "description": "Primary CRM dashboard snapshot",
        "mimeType": "application/json"
    })]
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

fn log_request(path: Option<&str>, method: &str) {
    let Some(path) = path else {
        return;
    };
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("log file should open");
    writeln!(file, "{method}").expect("log file should write");
}

fn log_tool_call(path: Option<&str>, tool_name: &str) {
    let Some(path) = path else {
        return;
    };
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("log file should open");
    writeln!(file, "tool:{tool_name}").expect("log file should write");
}
