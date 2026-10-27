"""
Engine for the Protocols lab (Agents, Session 8: AI Communication Protocols & Building an MCP Server,
Part 2: MCP with CrewAI and A2A, and the post-read on Agent-to-Agent communication).

Plain Python, so every lab runs with NO API key and NO network:
  * MiniMCP: a tiny, readable MCP server that speaks real JSON-RPC 2.0 messages (tools/list, tools/call,
    resources/read, prompts/get, standard error codes). Same decorators as FastMCP, so the code looks familiar.
  * a static checker for FastMCP server files (reads your code with `ast`, never runs it)
  * a checker for Claude Desktop's claude_desktop_config.json
  * CrewAI's MCP tool-name rule (copied from CrewAI 1.15.30, adapted from python-slugify, MIT licence)
  * A2A: Agent Cards, structured messages with schema checks, task states, and tracing with correlation IDs
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import random
import re
import typing
import unicodedata


# ----------------------------------------------------------------------------------------
# Optional live backend (Gemini). Demo mode never touches it.
# ----------------------------------------------------------------------------------------
class Backend:
    def __init__(self, api_key: str = "", chat_model: str = "gemini-3.5-flash-lite", rate_limit: bool = True):
        self.live, self.error, self.last_source, self.llm = False, "", "demo", None
        if api_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                kwargs = {}
                if rate_limit:
                    from langchain_core.rate_limiters import InMemoryRateLimiter
                    kwargs["rate_limiter"] = InMemoryRateLimiter(requests_per_second=4 / 60, check_every_n_seconds=0.5, max_bucket_size=1)
                self.llm = ChatGoogleGenerativeAI(model=chat_model, google_api_key=api_key, **kwargs)
                self.live = True
            except Exception as exc:
                self.error = "Could not start Gemini: " + str(exc)

    def ask(self, system: str, request: str, demo) -> str:
        if self.live:
            try:
                from langchain_core.messages import HumanMessage, SystemMessage
                out = self.llm.invoke([SystemMessage(content=system), HumanMessage(content=request)])
                self.last_source = "gemini"
                text = getattr(out, "text", None)
                return text if isinstance(text, str) else str(out.content)
            except Exception as exc:
                self.error, self.last_source = "Gemini call failed, showing the demo answer: " + str(exc), "demo (fallback)"
        else:
            self.last_source = "demo"
        return demo() if callable(demo) else demo


# ----------------------------------------------------------------------------------------
# 1. Why protocols
# ----------------------------------------------------------------------------------------
PROTOCOL_WHY = [
    {"Reason": "Consistency", "What it means": "Every message has the same structure and meaning", "Example": "A 'data request' always uses one format, so any agent can read it"},
    {"Reason": "Reliability & error handling", "What it means": "Rules for lost or corrupted messages", "Example": "Retries, acknowledgements, standard error codes"},
    {"Reason": "Interoperability", "What it means": "Systems in different languages and platforms work together", "Example": "A Python MCP client talks to a Node.js MCP server"},
    {"Reason": "Security & permissions", "What it means": "Who may send what, and when", "Example": "Only authenticated clients may send 'write' requests"},
    {"Reason": "Scalability & coordination", "What it means": "Data flows between many parts without conflicts", "Example": "Tools, resources and prompts interact smoothly"},
]

CALLING_VS_MCP = [
    {"Term": "Function calling", "What it is": "A feature of one LLM's API: the model outputs JSON naming a function; your code runs it", "Analogy": "One brand of soda"},
    {"Term": "Tool calling", "What it is": "The general idea frameworks use: a tool can be any function or API (Sessions 1–7)", "Analogy": "Soda in general"},
    {"Term": "MCP", "What it is": "A STANDARD for packaging tools, data and prompts in servers so ANY MCP client can use them", "Analogy": "USB-C: one plug for every device"},
]

MCP_PRINCIPLES = [
    {"Principle": "Structured communication", "In practice": "Every message is JSON. Real MCP uses JSON-RPC 2.0: jsonrpc, id, method, params → result or error"},
    {"Principle": "Separation of roles", "In practice": "Client sends requests and receives responses; server offers tools/resources/prompts and answers"},
    {"Principle": "Explicit context handling", "In practice": "Sessions keep multi-turn work together (in real MCP the HOST holds the conversation)"},
    {"Principle": "Deterministic action calls", "In practice": "Each tool declares exact parameters (a JSON schema) and returns structured results"},
    {"Principle": "Error & fallback management", "In practice": "Standard error codes (invalid request, unknown method, tool failure); clients retry or fall back"},
    {"Principle": "Protocol versioning", "In practice": "Both sides agree a protocol version during the initialize handshake"},
]


# ----------------------------------------------------------------------------------------
# 2. MiniMCP: a tiny MCP server that handles real JSON-RPC messages
# ----------------------------------------------------------------------------------------
_JSON_TYPES = {float: "number", int: "integer", str: "string", bool: "boolean", dict: "object", list: "array"}
PROTOCOL_VERSION = "2026-07-28"


def _schema_for(fn) -> dict:
    sig = inspect.signature(fn)
    hints = typing.get_type_hints(fn)
    props, required = {}, []
    for name, p in sig.parameters.items():
        t = hints.get(name)
        props[name] = {"type": _JSON_TYPES.get(typing.get_origin(t) or t, "string")} if t else {}
        if p.default is inspect.Parameter.empty:
            required.append(name)
        else:
            props[name]["default"] = p.default
    return {"type": "object", "properties": props, "required": required}


class MiniMCP:
    """A teaching copy of what FastMCP does for you. Decorate functions; send it JSON-RPC requests with handle()."""

    def __init__(self, name: str):
        self.name, self.tools, self.resources, self.prompts = name, {}, {}, {}

    def tool(self, fn=None):
        def register(f):
            self.tools[f.__name__] = f
            return f
        return register(fn) if callable(fn) else register          # works as @mcp.tool and @mcp.tool()

    def resource(self, uri: str):
        def register(f):
            self.resources[uri] = f
            return f
        return register

    def prompt(self, fn=None):
        def register(f):
            self.prompts[f.__name__] = f
            return f
        return register(fn) if callable(fn) else register

    @staticmethod
    def _err(rid, code, message):
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}

    def handle(self, raw) -> dict:
        """One request in, one response out. raw may be a dict or a JSON string."""
        if isinstance(raw, str):
            try:
                req = json.loads(raw)
            except Exception as exc:
                return self._err(None, -32700, "Parse error: " + str(exc))
        else:
            req = raw
        if not isinstance(req, dict) or req.get("jsonrpc") != "2.0" or "method" not in req:
            return self._err(req.get("id") if isinstance(req, dict) else None, -32600, "Invalid Request: needs \"jsonrpc\": \"2.0\" and a \"method\"")
        rid, method, params = req.get("id"), req["method"], req.get("params") or {}
        ok = lambda result: {"jsonrpc": "2.0", "id": rid, "result": result}
        if method == "initialize":
            return ok({"protocolVersion": PROTOCOL_VERSION, "serverInfo": {"name": self.name, "version": "1.0.0"},
                       "capabilities": {"tools": {}, "resources": {}, "prompts": {}}})
        if method == "tools/list":
            return ok({"tools": [{"name": n, "description": (inspect.getdoc(f) or "").split("\n")[0], "inputSchema": _schema_for(f)} for n, f in self.tools.items()]})
        if method == "tools/call":
            name, args = params.get("name"), params.get("arguments") or {}
            # Like FastMCP 4: problems with a tool call come back as a RESULT flagged isError, so the model can read them and recover
            fail = lambda text: ok({"content": [{"type": "text", "text": text}], "isError": True})
            if name not in self.tools:
                return fail("Unknown tool: '%s'" % name)
            schema = _schema_for(self.tools[name])
            missing = [r for r in schema["required"] if r not in args]
            if missing:
                return fail("Invalid arguments for tool '%s': missing required argument %s" % (name, ", ".join(missing)))
            try:
                fixed = {}
                for k, v in args.items():
                    t = schema["properties"].get(k, {}).get("type")
                    fixed[k] = float(v) if t == "number" else int(v) if t == "integer" else v
                out = self.tools[name](**fixed)
                return ok({"content": [{"type": "text", "text": str(out)}], "isError": False})
            except Exception as exc:
                return fail("Error calling tool '%s': %s" % (name, exc))
        if method == "resources/list":
            return ok({"resources": [{"uri": u, "name": f.__name__, "description": (inspect.getdoc(f) or "")} for u, f in self.resources.items()]})
        if method == "resources/read":
            uri = params.get("uri")
            if uri not in self.resources:
                return self._err(rid, -32002, "Resource not found: %s" % uri)
            val = self.resources[uri]()
            text, mime = (json.dumps(val), "application/json") if isinstance(val, (dict, list)) else (str(val), "text/plain")
            return ok({"contents": [{"uri": uri, "mimeType": mime, "text": text}]})
        if method == "prompts/list":
            out = []
            for n, f in self.prompts.items():
                sig = inspect.signature(f)
                out.append({"name": n, "description": inspect.getdoc(f) or "",
                            "arguments": [{"name": a, "required": p.default is inspect.Parameter.empty} for a, p in sig.parameters.items()]})
            return ok({"prompts": out})
        if method == "prompts/get":
            name, args = params.get("name"), params.get("arguments") or {}
            if name not in self.prompts:
                return self._err(rid, -32602, "Unknown prompt: %s" % name)
            try:
                val = self.prompts[name](**args)
            except TypeError as exc:
                return self._err(rid, -32602, "Invalid params: %s" % exc)
            msgs = val if isinstance(val, list) else [("user", val)]
            return ok({"messages": [{"role": r, "content": {"type": "text", "text": t}} for r, t in msgs]})
        return self._err(rid, -32601, "Method not found: %s" % method)


def course_server() -> MiniMCP:
    """The three course servers (mcpserver_tools.py, mcpserver_resources.py, mcpserver_prompt.py) as one MiniMCP."""
    mcp = MiniMCP("Simple Math Server")

    @mcp.tool()
    def add_numbers(a: float, b: float) -> float:
        """Add two numbers together."""
        return a + b

    @mcp.tool()
    def multiply_numbers(a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b

    @mcp.tool()
    def greet(name: str) -> str:
        """Greet a person by name."""
        return "Hello, %s! Welcome to the MCP server." % name

    @mcp.tool()
    def divide_numbers(a: float, b: float) -> float:
        """Divide a by b (added for this lab, to show a tool failure)."""
        return a / b

    @mcp.resource("config://server")
    def get_server_config() -> dict:
        """Get server configuration information."""
        return {"name": "Simple Math Server", "version": "1.0.0", "description": "A simple MCP server for basic math operations"}

    @mcp.resource("resource://greeting")
    def get_greeting() -> str:
        """Provides a simple greeting message."""
        return "Hello from FastMCP Resources!"

    @mcp.resource("data://config")
    def get_config() -> dict:
        """Provides application configuration as JSON."""
        return {"theme": "dark", "version": "1.2.0", "features": ["tools", "resources"]}

    @mcp.prompt
    def explain_topic(topic: str) -> str:
        """Asks the model to explain a study topic."""
        return "Explain the concept of '%s' in simple terms with a short example." % topic

    @mcp.prompt
    def generate_quiz(topic: str, difficulty: str = "easy") -> list:
        """Asks the model to create a few quiz questions for a topic."""
        return [("user", "I'm studying %s. Can you quiz me?" % topic), ("assistant", "Of course! What difficulty would you like?"),
                ("user", "%s. Please give 3 questions, each with 4 options, and say which answer is correct." % difficulty)]

    @mcp.prompt
    def summarize_notes(notes: str) -> str:
        """Summarizes long notes into a short recap."""
        return "Summarize the following notes into 3 bullet points for quick revision:\n\n%s" % notes

    return mcp


REQUEST_EXAMPLES = {
    "initialize (the handshake)": {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": PROTOCOL_VERSION, "clientInfo": {"name": "Claude Desktop"}}},
    "tools/list": {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    "tools/call add_numbers(2, 3)": {"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "add_numbers", "arguments": {"a": 2, "b": 3}}},
    "tools/call divide_numbers(1, 0) (tool fails)": {"jsonrpc": "2.0", "id": 8, "method": "tools/call", "params": {"name": "divide_numbers", "arguments": {"a": 1, "b": 0}}},
    "tools/call with a missing argument": {"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": {"name": "add_numbers", "arguments": {"a": 2}}},
    "resources/list": {"jsonrpc": "2.0", "id": 3, "method": "resources/list"},
    "resources/read data://config": {"jsonrpc": "2.0", "id": 4, "method": "resources/read", "params": {"uri": "data://config"}},
    "prompts/list": {"jsonrpc": "2.0", "id": 5, "method": "prompts/list"},
    "prompts/get generate_quiz": {"jsonrpc": "2.0", "id": 6, "method": "prompts/get", "params": {"name": "generate_quiz", "arguments": {"topic": "MCP", "difficulty": "medium"}}},
    "an unknown method": {"jsonrpc": "2.0", "id": 10, "method": "tools/run"},
}

SERVER_FEATURES = [
    {"Feature": "Tools", "What it is": "Actions the model can call: calculations, API calls, writing to a database", "Examples": "Search flights, send messages, create calendar events",
     "Who controls it": "The MODEL (decides when to call)", "FastMCP": "@mcp.tool"},
    {"Feature": "Resources", "What it is": "Read-only context data", "Examples": "Files, knowledge bases, a database schema, a calendar",
     "Who controls it": "The APPLICATION (decides what to load)", "FastMCP": '@mcp.resource("data://config")'},
    {"Feature": "Prompts", "What it is": "Ready-made instruction templates", "Examples": "Plan a vacation, summarize my meetings, draft an email",
     "Who controls it": "The USER (picks one)", "FastMCP": "@mcp.prompt"},
]


def host_turn(user_message: str, server: MiniMCP, backend: Backend | None = None) -> dict:
    """What a host (e.g. Claude Desktop) does with one user message: list tools, let its LLM choose, call over MCP, answer."""
    steps = []
    listed = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]
    steps.append(("client → server", {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}))
    steps.append(("server → client", {"tools": [t["name"] for t in listed]}))

    def demo_choice():
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", user_message)]
        low = user_message.lower()
        if len(nums) >= 2 and any(w in low for w in ("multiply", "times", "product", "×", "*")):
            return json.dumps({"tool": "multiply_numbers", "arguments": {"a": nums[0], "b": nums[1]}})
        if len(nums) >= 2 and any(w in low for w in ("divide", "divided", "/")):
            return json.dumps({"tool": "divide_numbers", "arguments": {"a": nums[0], "b": nums[1]}})
        if len(nums) >= 2 and any(w in low for w in ("add", "plus", "sum", "+", "total")):
            return json.dumps({"tool": "add_numbers", "arguments": {"a": nums[0], "b": nums[1]}})
        m = re.search(r"(?:greet|say hello to|welcome)\s+([A-Z][a-z]+)", user_message, re.I)
        if m:
            return json.dumps({"tool": "greet", "arguments": {"name": m.group(1).title()}})
        return json.dumps({"tool": None, "answer": "I can answer that without a tool."})

    system = ("You are the LLM inside an MCP host. Available tools (JSON): %s\nReply ONLY with JSON: "
              "{\"tool\": name, \"arguments\": {...}} to call a tool, or {\"tool\": null, \"answer\": \"...\"}." % json.dumps(listed))
    raw = (backend.ask(system, user_message, demo_choice) if backend else demo_choice())
    try:
        choice = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
    except Exception:
        choice = json.loads(demo_choice())
    steps.append(("host's LLM decides", choice))
    if not choice.get("tool"):
        return {"steps": steps, "answer": choice.get("answer", raw), "called": None}
    req = {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": choice["tool"], "arguments": choice.get("arguments", {})}}
    steps.append(("client → server", req))
    resp = server.handle(req)
    steps.append(("server → client", resp))
    if "error" in resp:
        answer = "The server rejected the request: %s" % resp["error"]["message"]
    elif resp["result"]["isError"]:
        answer = "The tool reported a problem (%s), so the model explains it instead of inventing a number." % resp["result"]["content"][0]["text"]
    else:
        answer = "The answer is %s (computed by the server's %s tool)." % (resp["result"]["content"][0]["text"], choice["tool"])
    return {"steps": steps, "answer": answer, "called": choice["tool"]}


# ----------------------------------------------------------------------------------------
# 3. Static checker for a FastMCP server file (never runs the code)
# ----------------------------------------------------------------------------------------
def _decorator_kind(dec, servers):
    target = dec.func if isinstance(dec, ast.Call) else dec
    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id in servers and target.attr in ("tool", "resource", "prompt"):
        uri = None
        if target.attr == "resource" and isinstance(dec, ast.Call) and dec.args and isinstance(dec.args[0], ast.Constant):
            uri = dec.args[0].value
        return target.attr, uri, target.value.id
    return None, None, None


def inspect_server_code(src: str) -> dict:
    """What would this FastMCP file register, and what is likely to go wrong? Reads the code; never executes it."""
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return {"ok": False, "error": "Syntax error on line %s: %s" % (exc.lineno, exc.msg), "items": [], "findings": []}
    servers, server_names, findings, items = set(), {}, [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("fastmcp.prompts.prompt"):
            findings.append(("❌", "Old import `from %s import ...` no longer exists in FastMCP 4. Use `from fastmcp.prompts import Message`." % node.module))
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            f = node.value.func
            fname = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
            if fname == "FastMCP":
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        servers.add(t.id)
                        arg = node.value.args[0].value if node.value.args and isinstance(node.value.args[0], ast.Constant) else next(
                            (k.value.value for k in node.value.keywords if k.arg == "name" and isinstance(k.value, ast.Constant)), "?")
                        server_names[t.id] = arg
    if not servers:
        findings.append(("❌", "No `mcp = FastMCP(\"...\")` found, so there is no server to register anything with."))
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        kind, uri, srv = None, None, None
        for d in node.decorator_list:
            kind, uri, srv = _decorator_kind(d, servers)
            if kind:
                break
        doc = ast.get_docstring(node)
        params = [a.arg for a in node.args.args]
        untyped = [a.arg for a in node.args.args if a.annotation is None]
        if kind:
            name = uri if kind == "resource" else node.name
            items.append({"Registered as": kind, "Name / URI": name, "Function": node.name + "(" + ", ".join(params) + ")",
                          "Description the model sees": (doc or "").split("\n")[0] or "(none!)"})
            if not doc:
                findings.append(("⚠️", "`%s` has no docstring. The model chooses tools by their description, so it may never pick this one." % node.name))
            if kind == "tool" and untyped:
                findings.append(("⚠️", "`%s`: parameters without type hints (%s). The input schema can't tell the model what type to send." % (node.name, ", ".join(untyped))))
            if kind == "prompt" and node.returns is not None:
                ret = ast.unparse(node.returns)
                if ret in ("Message", "PromptMessage"):
                    findings.append(("❌", "Prompt `%s` returns a single %s. FastMCP 4 rejects that: return a str or a list[Message]." % (node.name, ret)))
            if kind == "resource" and not uri:
                findings.append(("❌", "`%s` uses @resource without a URI, e.g. @mcp.resource(\"data://config\")." % node.name))
        else:
            if node.name.startswith("_"):
                findings.append(("✅", "`%s` is not decorated, so it stays a private helper: the client can't see or call it." % node.name))
            elif node.name != "main":
                findings.append(("❌", "`%s` is NOT registered: it has no @mcp.tool (or @mcp.resource / @mcp.prompt) decorator, so clients will never see it. "
                                       "The decorator, not the name, decides what is exposed." % node.name))
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id == "print":
                findings.append(("⚠️", "`%s` calls print(). In a stdio server, stdout IS the message channel, so stray prints corrupt the JSON-RPC stream. "
                                       "Log to stderr instead (print(..., file=sys.stderr) or logging)." % node.name))
                break
    run_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "run"
                 and isinstance(n.func.value, ast.Name) and n.func.value.id in servers]
    transport = "stdio (default)"
    if not run_calls:
        findings.append(("⚠️", "No `mcp.run()`: running `python server.py` will start nothing. (Fine if another program imports this server.)"))
    else:
        for k in run_calls[0].keywords:
            if k.arg == "transport" and isinstance(k.value, ast.Constant):
                transport = k.value.value
        if transport in ("http", "streamable-http", "sse"):
            findings.append(("ℹ️", "transport=\"%s\": the server listens on a network port. Claude Desktop's command/args config expects stdio, so connect to it by URL instead." % transport))
        guarded = any(isinstance(n, ast.If) and "__name__" in ast.unparse(n.test) and any(c is run_calls[0] for c in ast.walk(n)) for n in tree.body)
        if not guarded:
            findings.append(("⚠️", "`mcp.run()` isn't inside `if __name__ == \"__main__\":`, so importing this file (e.g. in a test) would start the server."))
    counts = {k: sum(1 for i in items if i["Registered as"] == k) for k in ("tool", "resource", "prompt")}
    return {"ok": True, "items": items, "findings": findings, "transport": transport, "counts": counts,
            "server": ", ".join("%s (%s)" % (v, k) for k, v in server_names.items())}


SERVER_SNIPPETS = {
    "Course: mcpserver_tools.py": '''from fastmcp import FastMCP

mcp = FastMCP("Simple Math Server")


@mcp.tool()
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


@mcp.tool()
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@mcp.tool()
def greet(name: str) -> str:
    """Greet a person by name."""
    return f"Hello, {name}! Welcome to the MCP server."


@mcp.resource("config://server")
def get_server_config() -> dict:
    """Get server configuration information."""
    return {"name": "Simple Math Server", "version": "1.0.0"}


if __name__ == "__main__":
    mcp.run(transport="stdio")
''',
    "🐞 Exam: the unregistered weather tool": '''from fastmcp import FastMCP

mcp = FastMCP("Weather Server")

# This function is not being registered
def get_current_temp(location: str) -> float:
    """Gets the current temperature for a location."""
    # ... implementation ...
    return 72.0

def _log_to_database(data: str):
    """Internal logging function."""
    # ... implementation ...
    pass

mcp.run()
''',
    "🐞 Old prompt server (fails on FastMCP 4)": '''from fastmcp import FastMCP
from fastmcp.prompts.prompt import Message, PromptMessage, TextContent

mcp = FastMCP(name="PromptServer")

@mcp.prompt
def explain_topic(topic: str) -> PromptMessage:
    """Asks the model to explain a study topic."""
    return PromptMessage(role="user", content=TextContent(type="text", text=f"Explain {topic}"))

if __name__ == "__main__":
    mcp.run(transport="stdio")
''',
    "🐞 Debug prints and missing docs": '''from fastmcp import FastMCP

mcp = FastMCP("Travel Tools")

@mcp.tool
def get_weather_forecast(place):
    print("looking up", place)
    return f"Sunny in {place}"

mcp.run(transport="http")
''',
}


# ----------------------------------------------------------------------------------------
# 4. Claude Desktop config checker
# ----------------------------------------------------------------------------------------
def check_claude_config(text: str, platform: str = "macOS / Linux") -> dict:
    """Check a claude_desktop_config.json. Returns findings as (level, message) plus the parsed servers."""
    findings, servers = [], []
    dupes = []

    def hook(pairs):
        seen = set()
        for k, _ in pairs:
            if k in seen:
                dupes.append(k)
            seen.add(k)
        return dict(pairs)

    try:
        data = json.loads(text, object_pairs_hook=hook)
    except json.JSONDecodeError as exc:
        msg = "Not valid JSON (line %d, column %d): %s." % (exc.lineno, exc.colno, exc.msg)
        line = text.splitlines()[exc.lineno - 1] if 0 < exc.lineno <= len(text.splitlines()) else ""
        hints = []
        if re.search(r'[A-Za-z]:\\[^\\]', text) or "Invalid \\escape" in exc.msg:
            hints.append("Windows paths need double backslashes (C:\\\\Users\\\\...) or forward slashes (C:/Users/...).")
        if re.search(r",\s*[}\]]", text):
            hints.append("JSON doesn't allow a trailing comma before } or ].")
        if "//" in text or "#" in text:
            hints.append("JSON has no comments: remove // or # lines.")
        if "“" in text or "”" in text or "'" in line:
            hints.append("Use straight double quotes \" (not ' or curly quotes).")
        findings.append(("❌", msg + (" " + " ".join(hints) if hints else "")))
        return {"ok": False, "findings": findings, "servers": servers}
    for d in dupes:
        findings.append(("⚠️", "The key \"%s\" appears twice; JSON keeps only the LAST one, so the first server silently disappears." % d))
    if not isinstance(data, dict) or "mcpServers" not in data:
        findings.append(("❌", "The top level must contain \"mcpServers\": { ... }. Check the spelling and capitalisation."))
        return {"ok": False, "findings": findings, "servers": servers}
    ms = data["mcpServers"]
    if not isinstance(ms, dict) or not ms:
        findings.append(("❌", "\"mcpServers\" must be an object with at least one server."))
        return {"ok": False, "findings": findings, "servers": servers}
    for name, cfg in ms.items():
        row = {"Server": name, "command": "", "args": "", "Transport": "stdio"}
        if not isinstance(cfg, dict):
            findings.append(("❌", "%s: each server must be an object with \"command\" and \"args\"." % name))
            continue
        if "url" in cfg and "command" not in cfg:
            row["Transport"] = "remote (URL)"
            findings.append(("ℹ️", "%s: a remote server given by URL. Remote servers use Streamable HTTP; in Claude apps they are usually added as connectors, not with command/args." % name))
            servers.append(row)
            continue
        cmd, args = cfg.get("command"), cfg.get("args", [])
        row["command"], row["args"] = str(cmd), json.dumps(args)
        if not cmd:
            findings.append(("❌", "%s: \"command\" is missing: the program Claude Desktop should start (your Python)." % name))
        elif isinstance(cmd, str):
            base = re.split(r"[\\/]", cmd)[-1]
            if base in ("python", "python3", "python.exe", "py") and not re.search(r"[\\/]", cmd):
                findings.append(("⚠️", "%s: command is just \"%s\". Claude Desktop may find a different Python than your virtual environment, without FastMCP installed. "
                                       "Use the full path to your venv's Python (print_claude_config.py prints it)." % (name, cmd)))
            elif not (cmd.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", cmd)) and base not in ("uv", "uvx", "npx", "node", "fastmcp"):
                findings.append(("⚠️", "%s: command \"%s\" is a relative path. Claude Desktop doesn't start in your project folder, so use an absolute path." % (name, cmd)))
            if platform.startswith("Windows") and cmd.startswith("/"):
                findings.append(("⚠️", "%s: that looks like a Mac/Linux path, but you chose Windows (e.g. C:\\\\Users\\\\you\\\\proj\\\\.venv\\\\Scripts\\\\python.exe)." % name))
            if platform.startswith("mac") and re.match(r"^[A-Za-z]:", cmd):
                findings.append(("⚠️", "%s: that looks like a Windows path, but you chose macOS / Linux." % name))
        if not isinstance(args, list):
            findings.append(("❌", "%s: \"args\" must be a LIST of strings, e.g. [\"/path/server.py\"], not a single string." % name))
            args = [str(args)]
        flat = " ".join(str(a) for a in args).lower()
        if re.search(r"--transport[ =](http|sse|streamable-http)|transport=(http|sse)", flat):
            findings.append(("❌", "%s: the args ask the server to start in HTTP mode. A command/args entry means Claude Desktop launches the server and talks over STDIO, "
                                   "so the two sides use different transports → \"Connection Error\". Remove the transport flags." % name))
            row["Transport"] = "MISMATCH: client stdio, server http"
        scripts = [a for a in args if str(a).endswith(".py")]
        if not scripts and cmd and "python" in str(cmd).lower():
            findings.append(("⚠️", "%s: no .py file in args. Python needs the path to your server script." % name))
        for s in scripts:
            if not (str(s).startswith("/") or re.match(r"^[A-Za-z]:[\\/]", str(s))):
                findings.append(("⚠️", "%s: the script path \"%s\" is relative. Use the full path (run pwd in its folder)." % (name, s)))
        env = cfg.get("env") or {}
        for k, v in env.items():
            if re.search(r"key|token|secret", k, re.I) and v:
                findings.append(("ℹ️", "%s: %s is set in the config file. That works, but anyone who can read this file can read the key." % (name, k)))
        servers.append(row)
    if not any(f[0] == "❌" for f in findings):
        findings.append(("✅", "Valid. Save the file, then QUIT Claude Desktop completely (Cmd+Q, or from the Windows system tray) and reopen it. "
                               "Check Settings > Developer: each server should show as running."))
    return {"ok": not any(f[0] == "❌" for f in findings), "findings": findings, "servers": servers}


CONFIG_EXAMPLES = {
    "Course example (macOS)": '''{
  "mcpServers": {
    "math-server": {
      "command": "/Users/codingninjas/Desktop/practice/myenv/bin/python3",
      "args": ["/Users/codingninjas/Desktop/practice/fastmcpserver.py"]
    }
  }
}''',
    "🐞 Exam: the protocol mismatch paradox": '''{
  "mcpServers": {
    "local-analyzer": {
      "command": "/usr/bin/python3",
      "args": ["/home/dev/analyzer.py", "--transport", "http"]
    }
  }
}''',
    "🐞 Windows paths with single backslashes": '''{
  "mcpServers": {
    "math-server": {
      "command": "C:\\Users\\priya\\proj\\.venv\\Scripts\\python.exe",
      "args": ["C:\\Users\\priya\\proj\\servers\\mcpserver_tools.py"]
    }
  }
}'''.replace("\\\\", "\\"),
    "🐞 Bare python, relative path, trailing comma": '''{
  "mcpServers": {
    "math-server": {
      "command": "python",
      "args": ["servers/mcpserver_tools.py"],
    }
  }
}''',
    "Three course servers (print_claude_config.py output, Windows)": '''{
  "mcpServers": {
    "math-server": {"command": "C:\\\\Users\\\\priya\\\\session8\\\\.venv\\\\Scripts\\\\python.exe", "args": ["C:\\\\Users\\\\priya\\\\session8\\\\servers\\\\mcpserver_tools.py"]},
    "data-server": {"command": "C:\\\\Users\\\\priya\\\\session8\\\\.venv\\\\Scripts\\\\python.exe", "args": ["C:\\\\Users\\\\priya\\\\session8\\\\servers\\\\mcpserver_resources.py"]},
    "prompt-server": {"command": "C:\\\\Users\\\\priya\\\\session8\\\\.venv\\\\Scripts\\\\python.exe", "args": ["C:\\\\Users\\\\priya\\\\session8\\\\servers\\\\mcpserver_prompt.py"]}
  }
}'''.replace("\\\\\\\\", "\\\\"),
}


# ----------------------------------------------------------------------------------------
# 5. Client features: sampling, roots, elicitation (course analogy vs the specification)
# ----------------------------------------------------------------------------------------
CLIENT_FEATURES = [
    {"Feature": "Sampling", "Specification (and the PDF's table)": "The SERVER asks the client's LLM for a completion, so the server needs no model of its own",
     "Example": "A travel server sends 5 flights and asks the LLM to pick the best one", "Restaurant analogy (course quizzes)": "'Looking at the menu': discovering what the server can do"},
    {"Feature": "Roots", "Specification (and the PDF's table)": "The CLIENT tells the server which folders/files it may work with (a security boundary)",
     "Example": "The server may read only the user's calendar folder", "Restaurant analogy (course quizzes)": "'Lunch or dinner menu?': the context or location to start from"},
    {"Feature": "Elicitation", "Specification (and the PDF's table)": "The SERVER asks the USER, through the client, for missing information mid-task",
     "Example": "'Window or aisle seat?' to finish a booking", "Restaurant analogy (course quizzes)": "'Placing your order': the concrete request"},
]

CLIENT_SCENARIOS_SPEC = [
    ("A booking server pauses and asks the user: 'Window or aisle?'", "Elicitation"),
    ("A server with no model of its own asks the host's LLM to rank five flights.", "Sampling"),
    ("Claude Desktop tells a file server it may only work inside /corporate_filings/2024/Q4/.", "Roots"),
    ("A server needs the user's phone number to finish a booking and asks for it.", "Elicitation"),
    ("A summarising server asks the client's model to write the summary text.", "Sampling"),
]

TRANSPORTS = [
    {"": "How", "STDIO": "The client STARTS the server as a child process; messages go through stdin/stdout", "Streamable HTTP": "The server runs as a web service; clients send HTTP POSTs (with optional streaming)"},
    {"": "Network", "STDIO": "None needed (works air-gapped)", "Streamable HTTP": "LAN, WAN, cloud"},
    {"": "Clients", "STDIO": "One (the process that started it)", "Streamable HTTP": "Many, concurrently"},
    {"": "Use for", "STDIO": "Local tools, testing, Claude Desktop with command/args", "Streamable HTTP": "Hosted, shared or production servers"},
    {"": "FastMCP", "STDIO": 'mcp.run()  or  mcp.run(transport="stdio")', "Streamable HTTP": 'mcp.run(transport="http", port=8000)'},
    {"": "CrewAI", "STDIO": "MCPServerStdio(command=..., args=[...])", "Streamable HTTP": "MCPServerHTTP(url=..., headers={...})"},
]


# ----------------------------------------------------------------------------------------
# 6. CrewAI + MCP: tool names (rule copied from CrewAI 1.15.30)
# ----------------------------------------------------------------------------------------
def sanitize_tool_name(name: str, max_length: int = 64) -> str:
    """CrewAI's tool-name cleaner (adapted from python-slugify by Val Neekman, MIT licence)."""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z])([A-Z])", r"\1_\2", name).lower()
    name = re.sub(r"[\'\"]+", "", name)
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if len(name) > max_length:
        suffix = "_" + hashlib.sha256(name.encode()).hexdigest()[:8]
        name = name[: max_length - len(suffix)].rstrip("_") + suffix
    return name


def crewai_tool_name(command: str, args: list, tool: str) -> dict:
    server = command + "_" + "_".join(args)
    raw = server + "_" + tool
    clean = sanitize_tool_name(raw)
    return {"raw": raw, "name": clean, "length": len(clean), "cut": not clean.endswith(sanitize_tool_name(tool))}


# ----------------------------------------------------------------------------------------
# 7. A2A: Agent Cards, messages, task states, tracing
# ----------------------------------------------------------------------------------------
AGENT_CARDS = {
    "calendar_agent": {"agent_id": "calendar_agent", "description": "Handles calendar events and scheduling tasks.",
                       "capabilities": ["get_free_slots", "book_meeting"], "communication_channel": "mcp",
                       "schemas": {"input": {"meeting_topic": "string", "preferred_time": "string"}, "output": {"status": "string", "scheduled_time": "string"}}},
    "notification_agent": {"agent_id": "notification_agent", "description": "Emails and messages people about confirmed events.",
                           "capabilities": ["send_email"], "communication_channel": "http",
                           "schemas": {"input": {"to": "string", "subject": "string", "body": "string"}, "output": {"status": "string"}}},
    "flight_agent": {"agent_id": "flight_agent", "description": "Finds and compares flights within India for couples.",
                     "capabilities": ["search_flights"], "communication_channel": "http",
                     "schemas": {"input": {"from_city": "string", "to_city": "string", "date": "string"}, "output": {"status": "string", "options": "array"}}},
    "hotel_agent": {"agent_id": "hotel_agent", "description": "Finds hotels near a landmark or airport.",
                    "capabilities": ["search_hotels"], "communication_channel": "websocket",
                    "schemas": {"input": {"near": "string", "nights": "integer"}, "output": {"status": "string", "options": "array"}}},
}

A2A_SPEC_CARD = {
    "name": "DilSeTrips Flight Agent",
    "description": "Finds and compares flights within India for couples.",
    "version": "1.0.0",
    "supportedInterfaces": [{"url": "https://agents.dilsetrips.example/a2a", "protocolBinding": "JSONRPC"}],
    "capabilities": {"streaming": True, "pushNotifications": False},
    "defaultInputModes": ["text/plain"],
    "defaultOutputModes": ["text/plain", "application/json"],
    "skills": [{"id": "search_flights", "name": "Search flights",
                "description": "Finds direct or one-stop flights between Indian cities, with prices in rupees.",
                "examples": ["Find a direct flight from Mumbai to Goa on 20 December"]}],
}

_PY = {"string": str, "integer": int, "array": list, "number": (int, float), "boolean": bool}


def discover(intent: str) -> list:
    return [c["agent_id"] for c in AGENT_CARDS.values() if intent in c["capabilities"]]


def send_a2a(sender: str, receiver: str, intent: str, payload: dict) -> dict:
    """Validate a message against the receiver's Agent Card, then let the receiver answer (scripted)."""
    card = AGENT_CARDS.get(receiver)
    msg = {"sender": sender, "receiver": receiver, "intent": intent, "payload": payload}
    problems = []
    if not card:
        return {"message": msg, "problems": ["No Agent Card for '%s': nobody to talk to." % receiver], "response": None}
    if intent not in card["capabilities"]:
        problems.append("'%s' can't do '%s'. Its card lists: %s." % (receiver, intent, ", ".join(card["capabilities"])))
    schema = card["schemas"]["input"]
    for field, typ in schema.items():
        if field not in payload:
            problems.append("Missing field '%s' (%s)." % (field, typ))
        elif not isinstance(payload[field], _PY[typ]):
            problems.append("Field '%s' should be %s, got %s." % (field, typ, type(payload[field]).__name__))
    for field in payload:
        if field not in schema:
            problems.append("Unexpected field '%s': the receiver's schema doesn't know it (a typo, or 'hallucination drift')." % field)
    if problems:
        return {"message": msg, "problems": problems, "response": {"status": "rejected", "reason": problems[0]}}
    if intent == "book_meeting":
        t = str(payload["preferred_time"])
        resp = {"status": "success", "scheduled_time": t.replace("4 PM", "4:15 PM") if "4 PM" in t else t}
    elif intent == "get_free_slots":
        resp = {"status": "success", "scheduled_time": "Tomorrow 11:00 AM, 4:15 PM"}
    elif intent == "send_email":
        resp = {"status": "sent"}
    elif intent == "search_flights":
        resp = {"status": "success", "options": [{"flight": "6E-512", "price_inr": 4890}, {"flight": "AI-661", "price_inr": 5320}]}
    else:
        resp = {"status": "success", "options": [{"hotel": "Airport Residency", "price_inr": 3200}]}
    return {"message": msg, "problems": [], "response": resp}


TASK_STATES = {
    "submitted": "Received, not started yet", "working": "In progress",
    "input-required": "Paused: the agent needs more information (like elicitation in MCP)",
    "auth-required": "Paused: the user must sign in or authorise something",
    "completed": "Finished successfully (terminal)", "failed": "Something went wrong (terminal)",
    "canceled": "Stopped on request (terminal)", "rejected": "The agent declined the task (terminal)",
}
TERMINAL = {"completed", "failed", "canceled", "rejected"}
TRANSITIONS = {
    "submitted": ["working", "rejected", "canceled"],
    "working": ["input-required", "auth-required", "completed", "failed", "canceled"],
    "input-required": ["working", "canceled"],
    "auth-required": ["working", "canceled"],
}

A2A_PATTERNS = [
    {"Pattern": "Hierarchical", "Analogy": "A manager and a team", "How": "A supervisor agent delegates to workers, collects and combines results", "Good for": "Central control; modular and scalable"},
    {"Pattern": "Collaborative", "Analogy": "A brainstorming team", "How": "Equals review and refine each other's outputs (write → critique → refine)", "Good for": "Diverse reasoning, higher quality"},
    {"Pattern": "Pipeline", "Analogy": "An assembly line", "How": "Each agent does one step and passes it on (extract → clean → summarise → visualise)", "Good for": "Clear, linear data processing"},
]

MCP_VS_A2A = [
    {"": "Connects", "MCP": "An agent to its TOOLS and DATA", "A2A": "An agent to OTHER AGENTS"},
    {"": "Direction", "MCP": "Vertical: an agent reaching down to its tools", "A2A": "Horizontal: agents working as peers"},
    {"": "Layer", "MCP": "Inner layer: inside each agent", "A2A": "Outer layer: between agents"},
    {"": "You discover...", "MCP": "A server's tools, resources and prompts (tools/list)", "A2A": "An agent's skills, from its Agent Card"},
    {"": "A request is...", "MCP": "One tool call, usually quick", "A2A": "A task that may run long, with states"},
    {"": "Each side sees...", "MCP": "The tool's inputs and outputs", "A2A": "Only results: each agent stays opaque"},
    {"": "Analogy", "MCP": "A worker using a wrench or a calculator", "A2A": "Specialists talking to finish a project"},
    {"": "Example", "MCP": "Your planner calls a weather tool", "A2A": "Your planner asks an airline's agent to book"},
]

A2A_CHALLENGES = [
    {"Challenge": "The 'Narrow Agent' problem", "What goes wrong": "Agents take on too many jobs; roles blur and communication gets confusing", "Analogy": "One hire as developer, designer and HR", "Best practice": "Cohesive agents: 'do one thing, and do it well'"},
    {"Challenge": "Orchestration overhead", "What goes wrong": "Coordination cost grows fast; the orchestrator becomes a bottleneck", "Analogy": "A manager coordinating 20 people across time zones", "Best practice": "Keep orchestrator logic lean; limit depth"},
    {"Challenge": "Consistency & state", "What goes wrong": "Stateless agents share and modify one context; it drifts out of sync", "Analogy": "Cooks adding to one pot without talking", "Best practice": "Pass context explicitly; use a shared context store"},
    {"Challenge": "Debugging conversations", "What goes wrong": "Tracing one request across many agents is hard", "Analogy": "A group chat where people forget to reply", "Best practice": "A session_id / correlation_id on every message"},
]


def trace_requests(n_users: int, fail_hop: str, with_ids: bool, seed: int = 4) -> dict:
    """Several user requests flow Coordinator → Flight/Hotel agents → their MCP tools, interleaved in one log.
    One hop fails for ONE request. With correlation IDs you can filter the log to that request's path."""
    rng = random.Random(seed)
    hops = [("coordinator", "flight_agent", "A2A"), ("flight_agent", "flights_db", "MCP"), ("coordinator", "hotel_agent", "A2A"),
            ("hotel_agent", "maps_api", "MCP"), ("coordinator", "user", "reply")]
    victim = rng.randrange(n_users)
    log, t = [], 0
    streams = []
    for u in range(n_users):
        cid = "req-%04x" % rng.randrange(16 ** 4)
        streams.append((u, cid, list(hops)))
    pending = [list(s[2]) for s in streams]
    while any(pending):
        u = rng.choice([i for i, p in enumerate(pending) if p])
        frm, to, kind = pending[u].pop(0)
        t += rng.randint(5, 40)
        failed = (u == victim and "%s → %s" % (frm, to) == fail_hop)
        log.append({"time (ms)": t, "correlation_id": streams[u][1] if with_ids else "", "from": frm, "to": to, "via": kind,
                    "status": "❌ timeout" if failed else "ok", "_user": u})
        if failed:
            pending[u] = [("coordinator", "user", "reply")]
    return {"log": log, "victim_id": streams[victim][1], "victim": victim}


# ----------------------------------------------------------------------------------------
# 8. Quiz
# ----------------------------------------------------------------------------------------
QUIZ = [
    ("Which TWO roles does MCP separate?", ["Host and model", "Client (sends requests) and server (offers tools, resources, prompts)", "Agent and agent"], 1, "Client initiates; server answers. The host is the app that contains clients."),
    ("Which components are essential for an MCP SERVER? (exam: Server Essentials)", ["Tools, prompts, resources", "Tools and a session manager", "Only tools"], 0,
     "The session manager belongs to the CLIENT. Tools do, resources show, prompts guide."),
    ("Who controls each server feature?", ["Tools → user, resources → model, prompts → app", "Tools → model, resources → application, prompts → user", "All → model"], 1, "The model acts, the app supplies, the user chooses."),
    ("In FastMCP, what makes a function visible to clients?", ["A name without an underscore", "The @mcp.tool decorator", "A docstring"], 1,
     "No decorator, no tool. With get_current_temp undecorated and _log_to_database private, the client finds an EMPTY tool list."),
    ("Claude Desktop launches a server with command/args, but args include --transport http. Result?", ["Works over HTTP", "Connection error: the client expects stdio, the server starts in HTTP mode", "Claude passes args wrongly"], 1,
     "command/args = stdio. Remove the HTTP flags (and the machine was air-gapped anyway)."),
    ("Which transport fits a single-user, local, no-network setup?", ["STDIO", "Streamable HTTP", "WebSocket"], 0, "Lightweight, no network setup. HTTP is for remote, multi-client, production."),
    ("Specification: what is SAMPLING?", ["The client checks what a server can do", "The server asks the client's LLM for a completion", "Token selection settings"], 1,
     "The PDF's table and the spec agree. The restaurant analogy ('look at the menu') describes tool discovery, which is what course mapping quizzes use."),
    ("Specification: what are ROOTS?", ["Folders/files the client lets the server work with", "The first message", "The server's tool list"], 0, "A security boundary set by the client."),
    ("Specification: what is ELICITATION?", ["Prompt engineering", "The server asking the user for missing information mid-task", "A tool call"], 1, "E.g. 'window or aisle?'"),
    ("Real MCP messages use...", ["XML", "JSON-RPC 2.0 (jsonrpc, id, method, params → result or error)", "Plain text"], 1, "The id links a reply to its request."),
    ("A tool raises an exception. How does a FastMCP server report it?", ["It crashes", "A result with isError: true, so the model sees the problem", "It returns an empty string"], 1,
     "Checked on FastMCP 4.1.4: exceptions, missing arguments and unknown tools all come back as results flagged isError. Protocol-level problems (an unknown method, invalid JSON) use 'error' with a code."),
    ("In CrewAI, how do you give an agent the tools of an MCP server?", ["Write @tool wrappers", "Agent(..., mcps=[MCPServerStdio(command=..., args=[...])])", "crew.add_server()"], 1,
     "CrewAI starts the server, lists its tools and hands them to the agent."),
    ("A2A connects...", ["An agent to its tools", "Agents to other agents", "A model to its weights"], 1, "MCP inside agents, A2A between agents."),
    ("What does an Agent Card describe?", ["The agent's prompt and memory", "Who the agent is, what it can do, how to reach it, and its input/output schema", "Its billing plan"], 1,
     "Discovery. Agents stay opaque: the card never reveals internal prompts or tools."),
    ("Why did a senior engineer reject MCP for a manager-less, peer-to-peer drone swarm?", ["Transport too slow", "MCP's client-server model leans centralized; the swarm needs distributed peer coordination", "MCP has no error handling"], 1,
     "An architectural-philosophy mismatch. Peer negotiation is A2A territory."),
    ("Agents keep misreading each other's outputs. Best practice?", ["Bigger models", "Strict shared schemas (JSON Schema / Pydantic)", "More agents"], 1, "It reduces 'hallucination drift'."),
    ("A request fails somewhere across five agents. What makes it traceable?", ["Longer prompts", "A correlation_id / session_id on every message", "Fewer logs"], 1, "Like a parcel tracking number."),
]
