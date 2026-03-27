# LM Studio Optimization MCP Server
# This module is standalone — it mounts alongside proxy.py, not inside it.
# Usage: python mcp_server.py  (runs on port 8090 to avoid conflict with proxy.py port 8080)
from typing import Any
import httpx
import json

LMSTUDIO_BASE = "http://192.168.1.12:1234"

TOOLS = [
    {
        "name": "complete",
        "description": "Streaming chat completion through the optimization bridge",
        "input_schema": {
            "model": "string",
            "messages": "array",
            "mode": "fast|think|architect",
            "stream": "boolean"
        }
    },
    {
        "name": "embed",
        "description": "Generate embeddings for text chunks",
        "input_schema": {"texts": "array of strings", "model": "string"}
    },
    {
        "name": "rerank",
        "description": "Rerank context chunks by relevance to a query",
        "input_schema": {"query": "string", "chunks": "array of strings", "top_k": "number"}
    },
    {
        "name": "list_models",
        "description": "List available models on LM Studio",
        "input_schema": {}
    },
    {
        "name": "model_stats",
        "description": "Get loaded model details (VRAM, context length, state)",
        "input_schema": {}
    }
]

def get_tools():
    return [{"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]} for t in TOOLS]

async def call_tool(name: str, arguments: dict) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        if name == "list_models":
            r = await client.get(f"{LMSTUDIO_BASE}/api/v1/models")
            return r.json()
        elif name == "model_stats":
            r = await client.get(f"{LMSTUDIO_BASE}/api/v1/models")
            data = r.json()
            models = data.get("data", []) if isinstance(data, dict) else data
            loaded = [m for m in models if m.get("state") == "loaded"]
            return {"loaded": loaded}
        elif name == "embed":
            r = await client.post(
                f"{LMSTUDIO_BASE}/v1/embeddings",
                json={"input": arguments.get("texts", []), "model": arguments.get("model", "nomic-embed-text")}
            )
            return r.json()
        elif name == "complete":
            r = await client.post(
                f"{LMSTUDIO_BASE}/v1/chat/completions",
                json={
                    "model": arguments.get("model", "qwen3.5-4b"),
                    "messages": arguments.get("messages", []),
                    "stream": arguments.get("stream", False)
                }
            )
            return r.json()
        elif name == "rerank":
            query = arguments.get("query", "")
            chunks = arguments.get("chunks", [])
            if not query or not chunks:
                return {"reranked": chunks}
            embed_resp = await client.post(
                f"{LMSTUDIO_BASE}/v1/embeddings",
                json={"input": [query] + chunks}
            )
            if embed_resp.status_code != 200:
                return {"reranked": chunks[:3]}
            data = embed_resp.json()
            embedding_list = data.get("data", data)
            if isinstance(embedding_list, list) and len(embedding_list) > 1:
                query_emb = embedding_list[0].get("embedding", []) if isinstance(embedding_list[0], dict) else []
                chunk_embs = embedding_list[1:]
                scored = []
                for i, emb_item in enumerate(chunk_embs):
                    emb = emb_item.get("embedding", []) if isinstance(emb_item, dict) else []
                    score = sum(a * b for a, b in zip(query_emb, emb)) if query_emb and emb else 0
                    scored.append((score, i, chunks[i]))
                scored.sort(reverse=True)
                top_k = arguments.get("top_k", 3)
                return {"reranked": [c for _, _, c in scored[:top_k]]}
            return {"reranked": chunks[:3]}
    return {"error": "unknown_tool"}

if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI

    app = FastAPI(title="LM Studio MCP Server")

    @app.get("/tools")
    async def list_tools():
        return {"tools": get_tools()}

    @app.post("/tools/call")
    async def invoke_tool(body: dict):
        name = body.get("name", "")
        arguments = body.get("arguments", {})
        result = await call_tool(name, arguments)
        return {"result": result}

    print("MCP tool server running on http://0.0.0.0:8090")
    uvicorn.run(app, host="0.0.0.0", port=8090)
