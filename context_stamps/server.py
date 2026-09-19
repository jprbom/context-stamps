"""Optional MCP stdio tools. Explicit calls only; no host transcript interception."""


def serve(memory, *, allow_writes: bool = False) -> None:
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("Context Stamps")

    # Async handlers run these short local operations on the owning event-loop thread.
    # This preserves SQLite connection affinity. Large indexes need a worker-owned store.
    async def remember(source: str, text: str, dependencies: dict[str, str] | None = None) -> dict:
        """Store a text chunk with a stable source ID and optional dependency versions."""
        return memory.add(text, source=source, dependencies=dependencies)

    @server.tool()
    async def recall(query: str, limit: int = 5, revisions: dict[str, str] | None = None) -> list[dict]:
        """Find related stored text; report whether source versions were checked."""
        return memory.recall(query, limit=limit, revisions=revisions)

    @server.tool()
    async def pack(query: str, byte_budget: int = 2048, revisions: dict[str, str] | None = None) -> dict:
        """Return original text and decisions under a UTF-8 byte budget, not model tokens."""
        return memory.pack(query, token_budget=byte_budget, revisions=revisions).to_dict()

    @server.tool()
    async def inspect_source(source: str) -> dict:
        """Fetch the stored source text and its metadata; check the stale flag before reuse."""
        return memory.get(source) or {"error": "source not found"}

    async def invalidate(source: str) -> dict:
        """Mark a source and declared dependents stale without deleting the original text."""
        return {"invalidated": memory.invalidate(source)}

    async def forget(source: str) -> dict:
        """Delete one source and invalidate declared dependents. Not forensic secure erasure."""
        return {"deleted": memory.forget(source)}

    if allow_writes:
        for operation in (remember, invalidate, forget):
            server.tool()(operation)

    server.run(transport="stdio")
