//! Integration tests for the `mcp-cli inspect` command.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use serde_json::json;

const CLI_BIN: &str = env!("CARGO_BIN_EXE_mcp-cli");
const FAKE_SERVER_BIN: &str = env!("CARGO_BIN_EXE_fake-mcp-server");

#[test]
fn inspect_lists_tools_from_the_configured_server() {
    let temp_dir = temp_dir("inspect-success");
    let config_path = write_registry(
        &temp_dir,
        json!({
            "servers": [
                {
                    "name": "fake",
                    "command": FAKE_SERVER_BIN,
                    "args": []
                }
            ]
        }),
    );

    let output = run_cli(&config_path, "fake");

    assert!(
        output.status().success(),
        "stderr: {}",
        output.stderr_text()
    );
    assert!(output.stdout_text().contains("Server: fake"));
    assert!(
        output
            .stdout_text()
            .contains("Protocol version: 2025-11-25")
    );
    assert!(
        output
            .stdout_text()
            .contains("Server info: fake-server 1.0.0")
    );
    assert!(output.stdout_text().contains("- run-sql"));
    assert!(output.stdout_text().contains("- describe-table"));
}

#[test]
fn inspect_fails_for_unknown_server_name() {
    let temp_dir = temp_dir("inspect-unknown-server");
    let config_path = write_registry(
        &temp_dir,
        json!({
            "servers": [
                {
                    "name": "fake",
                    "command": FAKE_SERVER_BIN,
                    "args": []
                }
            ]
        }),
    );

    let output = run_cli(&config_path, "missing");

    assert!(!output.status().success());
    assert!(
        output
            .stderr_text()
            .contains("unknown MCP server `missing`")
    );
}

#[test]
fn inspect_fails_for_invalid_json() {
    let temp_dir = temp_dir("inspect-invalid-json");
    let config_path = temp_dir.join("mcp_servers.json");
    fs::write(&config_path, "{invalid json").expect("temp config write");

    let output = run_cli(&config_path, "fake");

    assert!(!output.status().success());
    assert!(
        output
            .stderr_text()
            .contains("failed to parse JSON registry")
    );
}

#[test]
fn inspect_fails_when_the_server_command_is_missing() {
    let temp_dir = temp_dir("inspect-missing-command");
    let config_path = write_registry(
        &temp_dir,
        json!({
            "servers": [
                {
                    "name": "missing-command",
                    "command": "/definitely/missing/mcp-server",
                    "args": []
                }
            ]
        }),
    );

    let output = run_cli(&config_path, "missing-command");

    assert!(!output.status().success());
    assert!(
        output
            .stderr_text()
            .contains("failed to spawn MCP server process")
    );
}

#[test]
fn inspect_fails_when_the_server_exits_before_handshake() {
    let temp_dir = temp_dir("inspect-handshake-failure");
    let config_path = write_registry(
        &temp_dir,
        json!({
            "servers": [
                {
                    "name": "early-exit",
                    "command": FAKE_SERVER_BIN,
                    "args": ["--exit-immediately"]
                }
            ]
        }),
    );

    let output = run_cli(&config_path, "early-exit");

    assert!(!output.status().success());
    assert!(output.stderr_text().contains("transport closed"));
}

#[test]
fn inspect_passes_configured_env_to_the_server_process() {
    let temp_dir = temp_dir("inspect-env");
    let config_path = write_registry(
        &temp_dir,
        json!({
            "servers": [
                {
                    "name": "env-aware",
                    "command": FAKE_SERVER_BIN,
                    "args": ["--require-env=MCP_TEST_TOKEN"],
                    "env": {
                        "MCP_TEST_TOKEN": "present"
                    }
                }
            ]
        }),
    );

    let output = run_cli(&config_path, "env-aware");

    assert!(
        output.status().success(),
        "stderr: {}",
        output.stderr_text()
    );
    assert!(output.stdout_text().contains("Server: env-aware"));
}

fn write_registry(temp_dir: &Path, payload: serde_json::Value) -> PathBuf {
    let config_path = temp_dir.join("mcp_servers.json");
    fs::write(&config_path, payload.to_string()).expect("temp config write");
    config_path
}

fn run_cli(config_path: &Path, server_name: &str) -> OutputExt {
    let output = Command::new(CLI_BIN)
        .args([
            "inspect",
            "--server",
            server_name,
            "--config",
            config_path.to_str().expect("utf-8 path"),
        ])
        .output()
        .expect("CLI should run");

    OutputExt(output)
}

fn temp_dir(prefix: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock should be valid")
        .as_nanos();
    let path = std::env::temp_dir().join(format!("ai-data-analyst-{prefix}-{unique}"));
    fs::create_dir_all(&path).expect("temp dir create");
    path
}

struct OutputExt(std::process::Output);

impl OutputExt {
    fn status(&self) -> &std::process::ExitStatus {
        &self.0.status
    }

    fn stdout_text(&self) -> String {
        String::from_utf8(self.0.stdout.clone()).expect("stdout should be utf-8")
    }

    fn stderr_text(&self) -> String {
        String::from_utf8(self.0.stderr.clone()).expect("stderr should be utf-8")
    }
}
