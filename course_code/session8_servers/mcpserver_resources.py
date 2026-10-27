"""An MCP server with two RESOURCES: read-only information a client can fetch.

Each resource has an address (a URI) such as "resource://greeting" or "data://config".
Test it without Claude:       python test_client.py mcpserver_resources.py
"""

from fastmcp import FastMCP

mcp = FastMCP(name="DataServer")


# A resource that returns text
@mcp.resource("resource://greeting")
def get_greeting() -> str:
    """Provides a simple greeting message."""
    return "Hello from FastMCP Resources!"


# A resource that returns JSON (FastMCP turns the dictionary into JSON for us)
@mcp.resource("data://config")
def get_config() -> dict:
    """Provides application configuration as JSON."""
    return {
        "theme": "dark",
        "version": "1.2.0",
        "features": ["tools", "resources"],
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
