#!/usr/bin/env python3
"""
Patched PostgreSQL MCP server wrapper.

The upstream `postgres-mcp-server` package rejects any query that does not
literally start with `SELECT`. That incorrectly blocks read-only CTE queries
such as `WITH ... SELECT ...`. This wrapper preserves the original read-only
execution model while allowing top-level `WITH` statements and rejecting
multiple statements in one request.
"""

import asyncio
import json
import logging
import sys
from typing import Any, Optional
from urllib.parse import urlparse, urlunparse

import psycopg2.pool
from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server
from psycopg2.extras import RealDictCursor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _strip_leading_comments(sql: str) -> str:
    idx = 0
    length = len(sql)
    while idx < length:
        while idx < length and sql[idx].isspace():
            idx += 1
        if sql.startswith("--", idx):
            newline = sql.find("\n", idx + 2)
            if newline == -1:
                return ""
            idx = newline + 1
            continue
        if sql.startswith("/*", idx):
            block_end = sql.find("*/", idx + 2)
            if block_end == -1:
                return ""
            idx = block_end + 2
            continue
        break
    return sql[idx:]


def _has_multiple_statements(sql: str) -> bool:
    in_single_quote = False
    in_double_quote = False
    in_line_comment = False
    in_block_comment = False
    trailing_after_semicolon = False
    idx = 0
    length = len(sql)

    while idx < length:
        ch = sql[idx]
        nxt = sql[idx + 1] if idx + 1 < length else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            idx += 1
            continue
        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                idx += 2
            else:
                idx += 1
            continue
        if in_single_quote:
            if ch == "'" and nxt == "'":
                idx += 2
                continue
            if ch == "'":
                in_single_quote = False
            idx += 1
            continue
        if in_double_quote:
            if ch == '"':
                in_double_quote = False
            idx += 1
            continue

        if ch == "-" and nxt == "-":
            in_line_comment = True
            idx += 2
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            idx += 2
            continue
        if ch == "'":
            in_single_quote = True
            idx += 1
            continue
        if ch == '"':
            in_double_quote = True
            idx += 1
            continue
        if ch == ";":
            trailing_after_semicolon = True
            idx += 1
            continue
        if trailing_after_semicolon and not ch.isspace():
            return True
        idx += 1

    return False


def is_allowed_read_only_query(sql: str) -> bool:
    stripped = _strip_leading_comments(sql)
    if not stripped:
        return False
    if _has_multiple_statements(stripped):
        return False

    normalized = stripped.lstrip().upper()
    return normalized.startswith("SELECT") or normalized.startswith("WITH")


class PostgreSQLMCPServer:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.connection_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
        self.schema_path = "schema"
        self.parsed_url = urlparse(database_url)
        self._setup_connection_pool()
        self.resource_base_url = self._create_resource_base_url()
        self.server = Server("postgres-mcp")
        self._setup_handlers()

    def _setup_connection_pool(self) -> None:
        conn_params = {
            "host": self.parsed_url.hostname,
            "port": self.parsed_url.port or 5432,
            "database": self.parsed_url.path.lstrip("/"),
            "user": self.parsed_url.username,
            "password": self.parsed_url.password,
            "cursor_factory": RealDictCursor,
        }
        if "sslmode" not in self.database_url:
            conn_params["sslmode"] = "prefer"

        self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            **conn_params,
        )
        logger.info("PostgreSQL connection pool created successfully")

    def _create_resource_base_url(self) -> str:
        safe_url = urlunparse(
            (
                "postgres",
                f"{self.parsed_url.username}@{self.parsed_url.hostname}:{self.parsed_url.port or 5432}",
                self.parsed_url.path,
                self.parsed_url.params,
                self.parsed_url.query,
                self.parsed_url.fragment,
            )
        )
        return safe_url

    def _setup_handlers(self) -> None:
        @self.server.list_resources()
        async def list_resources() -> list[types.Resource]:
            conn = None
            try:
                conn = self.connection_pool.getconn()
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT table_name
                        FROM information_schema.tables
                        WHERE table_schema = 'public'
                        ORDER BY table_name
                        """
                    )
                    tables = cursor.fetchall()

                return [
                    types.Resource(
                        uri=f"{self.resource_base_url}/{table['table_name']}/{self.schema_path}",
                        name=f"\"{table['table_name']}\" database schema",
                        mimeType="application/json",
                    )
                    for table in tables
                ]
            finally:
                if conn:
                    self.connection_pool.putconn(conn)

        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            conn = None
            try:
                parts = uri.split("/")
                if len(parts) < 2:
                    raise ValueError(f"Invalid resource URI: {uri}")

                table_name = parts[-2]
                conn = self.connection_pool.getconn()
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT column_name, data_type, is_nullable
                        FROM information_schema.columns
                        WHERE table_schema = 'public' AND table_name = %s
                        ORDER BY ordinal_position
                        """,
                        (table_name,),
                    )
                    columns = cursor.fetchall()

                return json.dumps(
                    {
                        "table_name": table_name,
                        "columns": list(columns),
                    },
                    indent=2,
                )
            finally:
                if conn:
                    self.connection_pool.putconn(conn)

        @self.server.list_tools()
        async def list_tools() -> list[types.Tool]:
            return [
                types.Tool(
                    name="query",
                    description="Run a read-only SQL query against the PostgreSQL database",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "sql": {
                                "type": "string",
                                "description": "A single read-only SQL statement. Top-level SELECT and WITH ... SELECT are allowed.",
                            }
                        },
                        "required": ["sql"],
                    },
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
            if name != "query":
                raise ValueError(f"Unknown tool: {name}")

            sql_query = arguments.get("sql")
            if not sql_query:
                raise ValueError("SQL query is required")
            if not is_allowed_read_only_query(sql_query):
                raise ValueError(
                    "Only single-statement read-only SELECT queries are allowed. "
                    "Top-level SELECT and WITH ... SELECT are supported."
                )

            conn = None
            try:
                conn = self.connection_pool.getconn()
                conn.set_session(readonly=True)

                with conn.cursor() as cursor:
                    cursor.execute("BEGIN TRANSACTION READ ONLY")
                    cursor.execute(sql_query)
                    results = cursor.fetchall()
                    cursor.execute("ROLLBACK")

                result_data = [dict(row) for row in results]
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(result_data, indent=2, default=str),
                    )
                ]
            except Exception as exc:
                logger.error("Error executing query: %s", exc)
                if conn:
                    try:
                        with conn.cursor() as cursor:
                            cursor.execute("ROLLBACK")
                    except Exception:
                        pass
                raise ValueError(f"Query execution failed: {exc}")
            finally:
                if conn:
                    self.connection_pool.putconn(conn)

    async def run(self) -> None:
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )


async def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/postgres_mcp_server_wrapper.py <postgresql://...>", file=sys.stderr)
        sys.exit(1)

    server = PostgreSQLMCPServer(sys.argv[1])
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
