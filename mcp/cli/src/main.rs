//! Small CLI for verifying that a configured MCP server can be reached.
//!
//! The binary intentionally stays minimal so it can act as a transport smoke
//! test independent of the larger runtime and control-plane stack.

use std::{env, error::Error, path::PathBuf, process};

use mcp_client::{ClientInfo, McpClient};
use mcp_config::McpRegistry;

const DEFAULT_CONFIG_PATH: &str = "config/mcp_servers.json";

#[tokio::main]
async fn main() {
    if let Err(err) = run().await {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

/// Runs the end-to-end inspection flow for one configured server.
async fn run() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args = parse_args(env::args())?;
    let registry = McpRegistry::load_from_path(&args.config_path)?;
    let server = registry.get(&args.server_name)?;
    let connection = McpClient::connect(server).await?;
    // Reuse the CLI package metadata as the MCP client identity.
    let initialize_result = connection
        .initialize(ClientInfo::new(
            env!("CARGO_PKG_NAME"),
            env!("CARGO_PKG_VERSION"),
        ))
        .await?;
    connection.notify_initialized().await?;
    let tools = connection.list_tools().await?;

    println!("Server: {}", args.server_name);
    println!("Protocol version: {}", initialize_result.protocol_version);
    println!(
        "Server info: {} {}",
        initialize_result.server_info.name, initialize_result.server_info.version
    );
    if tools.tools.is_empty() {
        println!("Tools: none");
    } else {
        println!("Tools:");
        for tool in tools.tools {
            println!("- {}", tool.name);
        }
    }

    Ok(())
}

/// Parses the intentionally small CLI surface without a dependency-heavy parser.
fn parse_args(args: impl IntoIterator<Item = String>) -> Result<InspectArgs, String> {
    let mut args = args.into_iter();
    let bin = args.next().unwrap_or_else(|| "mcp-cli".to_owned());
    let command = args.next().ok_or_else(|| usage(&bin))?;

    if command != "inspect" {
        return Err(usage(&bin));
    }

    let mut server_name = None;
    let mut config_path = PathBuf::from(DEFAULT_CONFIG_PATH);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--server" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("missing value for `--server`\n\n{}", usage(&bin)))?;
                server_name = Some(value);
            }
            "--config" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("missing value for `--config`\n\n{}", usage(&bin)))?;
                config_path = PathBuf::from(value);
            }
            "--help" | "-h" => return Err(usage(&bin)),
            other => {
                return Err(format!("unexpected argument `{other}`\n\n{}", usage(&bin)));
            }
        }
    }

    let server_name = server_name.ok_or_else(|| {
        format!(
            "missing required `--server <name>` argument\n\n{}",
            usage(&bin)
        )
    })?;

    Ok(InspectArgs {
        server_name,
        config_path,
    })
}

/// Produces a compact usage string for invalid input and `--help`.
fn usage(bin: &str) -> String {
    format!(
        "Usage:\n  {bin} inspect --server <name> [--config <path>]\n\nDefault config path: {DEFAULT_CONFIG_PATH}"
    )
}

/// Parsed inputs for the `inspect` command.
#[derive(Debug)]
struct InspectArgs {
    server_name: String,
    config_path: PathBuf,
}
