"""An MCP CLIENT that tests a server without Claude Desktop.

It starts the server file, connects to it over stdio (exactly as Claude Desktop does),
lists everything the server offers, and tries each part.

Run:   python test_client.py mcpserver_tools.py
       python test_client.py mcpserver_resources.py
       python test_client.py mcpserver_prompt.py
"""

import asyncio
import sys
from pathlib import Path

from fastmcp import Client


async def test(server_file):
    async with Client(Path(server_file)) as client:    # starts the server file and connects to it over stdio
        tools = await client.list_tools()
        print(f"\nTOOLS ({len(tools)}):")
        for tool in tools:
            print(f"  {tool.name}: {tool.description.splitlines()[0]}")
        if len(tools) > 0 and tools[0].name == "add_numbers":
            result = await client.call_tool("add_numbers", {"a": 2, "b": 3})
            print("  calling add_numbers(2, 3) ->", result.data)

        resources = await client.list_resources()
        print(f"\nRESOURCES ({len(resources)}):")
        for resource in resources:
            content = await client.read_resource(str(resource.uri))
            print(f"  {resource.uri} -> {content[0].text}")

        prompts = await client.list_prompts()
        print(f"\nPROMPTS ({len(prompts)}):")
        for prompt in prompts:
            names = []
            for argument in prompt.arguments or []:
                names.append(argument.name)
            print(f"  {prompt.name}  (arguments: {', '.join(names)})")
        if len(prompts) > 0 and prompts[0].name == "explain_topic":
            rendered = await client.get_prompt("explain_topic", {"topic": "recursion"})
            print("  explain_topic('recursion') ->", rendered.messages[0].content.text)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_client.py <server file>, for example: python test_client.py mcpserver_tools.py")
    else:
        asyncio.run(test(sys.argv[1]))
