"""An MCP server with three TOOLS and one RESOURCE.

Tools are actions the model can choose to call. A resource is read-only information.
Run it on its own:            python mcpserver_tools.py      (it waits for a client; stop with Ctrl+C)
Test it without Claude:       python test_client.py mcpserver_tools.py
"""

from fastmcp import FastMCP

# Create the MCP server
mcp = FastMCP("Simple Math Server")


@mcp.tool()
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together.

    Args:
        a: First number
        b: Second number
    """
    return a + b


@mcp.tool()
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers.

    Args:
        a: First number
        b: Second number
    """
    return a * b


@mcp.tool()
def greet(name: str) -> str:
    """Greet a person by name.

    Args:
        name: The person's name
    """
    return f"Hello, {name}! Welcome to the MCP server."


# A resource: information a client can read (not an action)
@mcp.resource("config://server")
def get_server_config() -> dict:
    """Get server configuration information."""
    return {
        "name": "Simple Math Server",
        "version": "1.0.0",
        "description": "A simple MCP server for basic math operations",
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")       # stdio: the client starts this file and talks to it directly
