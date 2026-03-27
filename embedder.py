import httpx
import numpy as np
from typing import List, Optional
import structlog

logger = structlog.get_logger()

LMSTUDIO_BASE = "http://192.168.1.12:1234"
DEFAULT_EMBED_MODEL = "nomic-embed-text-v1.5"


def cosine_score(a: List[float], b: List[float]) -> float:
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


class Embedder:
    def __init__(self, base_url: str = LMSTUDIO_BASE, model: str = DEFAULT_EMBED_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{self.base_url}/v1/embeddings",
                    json={"input": texts, "model": self.model}
                )
                if resp.status_code != 200:
                    logger.error("embed_http_error", status=resp.status_code, body=resp.text)
                    return []
                data = resp.json()
                items = data.get("data", data)
                if isinstance(items, list) and len(items) > 0:
                    first = items[0]
                    if isinstance(first, dict) and "embedding" in first:
                        return [item["embedding"] for item in items]
                    elif isinstance(first, list):
                        return items
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
