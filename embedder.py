import httpx
import numpy as np
from typing import List, Optional
import structlog
from lmstudio_embedder import LMStudioPythonEmbedder

logger = structlog.get_logger()

LMSTUDIO_BASE = "http://"
DEFAULT_EMBED_MODEL = "text-embedding-qwen3-embedding-4b"


def cosine_score(a: List[float], b: List[float]) -> float:
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


class Embedder:
    def __init__(self, base_url: str = LMSTUDIO_BASE, model: str = DEFAULT_EMBED_MODEL, add_eos_token: bool = True):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.add_eos_token = add_eos_token
        # Extract host:port for Python API
        host_port = base_url.replace("http://", "").replace("https://", "").rstrip("/")
        self.python_embedder = LMStudioPythonEmbedder(
            lmstudio_address=host_port,
            model_name=model,
            add_eos_token=add_eos_token,
        )

    async def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        # Try Python API first (with EOS support)
        try:
            embeddings = await self.python_embedder.embed(texts, add_eos=self.add_eos_token)
            if embeddings:
                return embeddings
        except Exception as e:
            logger.debug("python_api_embed_fallback", error=str(e))
        
        # Fallback to HTTP API
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                for path in ("/v1/embeddings", "/api/v1/embeddings", "/embeddings"):
                    resp = await client.post(
                        f"{self.base_url}{path}",
                        json={"input": texts, "model": self.model}
                    )
                    if resp.status_code != 200:
                        continue
                    data = resp.json()
                    items = data.get("data", data) if isinstance(data, dict) else data
                    if isinstance(items, list) and len(items) > 0:
                        first = items[0]
                        if isinstance(first, dict) and "embedding" in first:
                            return [item["embedding"] for item in items]
                        if isinstance(first, list):
                            return items
                logger.error("embed_http_error", model=self.model, body="No compatible embeddings endpoint succeeded")
                return []
        except Exception as e:
            logger.error("embed_failed", error=str(e))
            return []

    async def embed_query(self, query: str) -> List[float]:
        results = await self.embed([query])
        return results[0] if results else []

    async def embed_chunks(self, chunks: List[str]) -> List[List[float]]:
        return await self.embed(chunks)

    async def rerank(
        self,
        query: str,
        chunks: List[str],
        top_k: int = 3,
        method: str = "cosine"
    ) -> List[dict]:
        if not query or not chunks:
            return [{"chunk": c, "score": 0.0} for c in chunks[:top_k]]

        query_emb = await self.embed_query(query)
        chunk_embs = await self.embed_chunks(chunks)

        if not query_emb or not chunk_embs:
            return [{"chunk": c, "score": 0.0} for c in chunks[:top_k]]

        scored = []
        for i, chunk_emb in enumerate(chunk_embs):
            if isinstance(chunk_emb, list) and chunk_emb:
                score = cosine_score(query_emb, chunk_emb)
            else:
                score = 0.0
            scored.append({"chunk": chunks[i], "score": float(score)})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


if __name__ == "__main__":
    import asyncio

    async def test():
        e = Embedder()
        texts = ["Hello world", "Python is great", "LLMs are powerful"]
        emb = await e.embed(texts)
        print(f"Embedded {len(emb)} texts, dim={len(emb[0]) if emb else 0}")

        results = await e.rerank("Tell me about programming", texts, top_k=2)
        for r in results:
            print(f"  score={r['score']:.4f}  chunk={r['chunk']}")

    asyncio.run(test())
