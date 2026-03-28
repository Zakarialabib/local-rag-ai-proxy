import os
import glob
import json
from typing import List, Optional
import structlog

logger = structlog.get_logger()

LMSTUDIO_BASE = "http://192.168.1.12:1234"
DEFAULT_RERANK_MODEL = "qwen.qwen3-reranker-4b"

MODEL_DIR = os.path.join(os.path.expanduser("~"), ".cache", "lm-studio", "models")


def find_model_path(model_id: str) -> Optional[str]:
    model_dir = MODEL_DIR
    if not os.path.exists(model_dir):
        return None
    model_id_lower = model_id.lower().replace("/", "_")
    for root, dirs, files in os.walk(model_dir):
        for f in files:
            f_lower = f.lower()
            if model_id_lower in f_lower or model_id.replace("/", "_").lower() in f_lower:
                gguf_candidates = [fi for fi in files if fi.endswith(".gguf")]
                if gguf_candidates:
                    return os.path.join(root, gguf_candidates[0])
                return root
    return None


class CrossEncoderReranker:
    def __init__(
        self,
        model_id: str = DEFAULT_RERANK_MODEL,
        base_url: str = LMSTUDIO_BASE,
        device: str = "cpu"
    ):
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.device = device
        self._pipeline = None

    def _load_pipeline(self):
        try:
            from sentence_transformers import CrossEncoder
            path = find_model_path(self.model_id)
            if path and os.path.isdir(path):
                self._pipeline = CrossEncoder(path, device=self.device)
            elif path and path.endswith(".gguf"):
                self._pipeline = CrossEncoder(path, device=self.device, model_kwargs={"gguf_file": os.path.basename(path)})
            else:
                self._pipeline = CrossEncoder(self.model_id, device=self.device)
            logger.info("reranker_loaded", model=self.model_id)
        except ImportError:
            logger.warning("sentence_transformers_not_installed", model=self.model_id)
            self._pipeline = None
        except Exception as e:
            logger.error("reranker_load_failed", model=self.model_id, error=str(e))
            self._pipeline = None

    async def rerank(
        self,
        query: str,
        chunks: List[str],
        top_k: int = 3
    ) -> List[dict]:
        if not chunks:
            return []
        if self._pipeline is None:
            self._load_pipeline()

        if self._pipeline is None:
            return [{"chunk": c, "score": 0.0} for c in chunks[:top_k]]

        try:
            pairs = [[query, chunk] for chunk in chunks]
            scores = self._pipeline.predict(pairs)
            if isinstance(scores, (list, tuple)):
                scored = [{"chunk": chunks[i], "score": float(scores[i])} for i in range(len(chunks))]
            else:
                scored = [{"chunk": chunks[0], "score": float(scores)}]
            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[:top_k]
        except Exception as e:
            logger.error("rerank_failed", error=str(e))
            return [{"chunk": c, "score": 0.0} for c in chunks[:top_k]]


class LMStudioReranker:
    def __init__(self, model_id: str = DEFAULT_RERANK_MODEL, base_url: str = LMSTUDIO_BASE):
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")

    async def rerank(self, query: str, chunks: List[str], top_k: int = 3) -> List[dict]:
        if not chunks:
            return []
        try:
            import httpx
            async with httpx.AsyncClient(timeout=30) as client:
                for path in ("/v1/rerank", "/api/v1/rerank", "/rerank"):
                    resp = await client.post(
                        f"{self.base_url}{path}",
                        json={
                            "model": self.model_id,
                            "query": query,
                            "documents": chunks,
                            "top_k": top_k
                        }
                    )
                    if resp.status_code != 200:
                        continue
                    data = resp.json()
                    results = data.get("results", data) if isinstance(data, dict) else data
                    if isinstance(results, list):
                        normalized = []
                        for index, item in enumerate(results):
                            if isinstance(item, dict):
                                normalized.append({
                                    "chunk": item.get("document") or item.get("text") or item.get("chunk") or chunks[index],
                                    "score": float(item.get("relevance_score", item.get("score", 0))),
                                })
                        if normalized:
                            return normalized[:top_k]
                logger.warning("rerank_http_failed", model=self.model_id)
                return [{"chunk": c, "score": 0.0} for c in chunks[:top_k]]
        except Exception as e:
            logger.error("lmstudio_rerank_failed", error=str(e))
            return [{"chunk": c, "score": 0.0} for c in chunks[:top_k]]


async def get_reranker(type_: str = "lmstudio", **kwargs):
    if type_ == "cross-encoder":
        return CrossEncoderReranker(**kwargs)
    return LMStudioReranker(**kwargs)


if __name__ == "__main__":
    import asyncio

    async def test():
        r = await get_reranker("lmstudio")
        results = await r.rerank(
            "How does authentication work?",
            [
                "JWT tokens are stored in localStorage",
                "OAuth2 uses redirect URIs for auth",
                "The API uses bearer tokens in headers"
            ],
            top_k=2
        )
        for r in results:
            print(f"  score={r['score']:.4f}  chunk={r['chunk']}")

    asyncio.run(test())
