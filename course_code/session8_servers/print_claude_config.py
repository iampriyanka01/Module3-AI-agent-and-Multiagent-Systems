"""Print the Claude Desktop configuration for THIS computer.

It fills in the exact Python you're using (your virtual environment) and the full path to each server,
which is where most setup mistakes happen. Copy the output into Claude Desktop:
Settings > Developer > Edit Config, then restart Claude Desktop completely.

Run:   python print_claude_config.py
"""

import json
import os
import sys

folder = os.path.dirname(os.path.abspath(__file__))
config = {"mcpServers": {
    "math-server": {"command": sys.executable, "args": [os.path.join(folder, "mcpserver_tools.py")]},
    "data-server": {"command": sys.executable, "args": [os.path.join(folder, "mcpserver_resources.py")]},
    "prompt-server": {"command": sys.executable, "args": [os.path.join(folder, "mcpserver_prompt.py")]},
}}
print(json.dumps(config, indent=2))
