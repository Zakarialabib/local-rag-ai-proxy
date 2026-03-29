import os
import glob
import json
import httpx
from typing import List, Optional, Dict, Literal, Any
import structlog

logger = structlog.get_logger()

LMSTUDIO_BASE = "http://192.168.1.12:1234"
DEFAULT_RERANK_MODEL = "qwen.qwen3-reranker-4b"

MODEL_DIR = os.path.join(os.path.expanduser("~"), ".cache", "lm-studio", "models")

# Task-specific instructions for reranking
RERANK_INSTRUCTIONS = {
    "retrieval": "Rank the documents by how relevant they are to the query.",
    "semantic_similarity": "Rank by semantic similarity to the query.",
    "relevance": "Rank by relevance to the query.",
    "coherence": "Rank by coherence and relevance.",
    "code_search": "Rank code snippets by relevance to the query.",
    "question_answering": "Rank answers by how well they answer the question.",
    "passage_retrieval": "Rank passages by their relevance to the query.",
    "paraphrase": "Rank paraphrases by semantic equivalence to the original.",
}


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
    """Local reranker using sentence-transformers (CrossEncoder)."""
    
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
            logger.info("reranker_loaded_local", model=self.model_id)
        except ImportError:
            logger.warning("sentence_transformers_not_installed", hint="pip install sentence-transformers")
            self._pipeline = None
        except Exception as e:
            logger.error("reranker_load_failed", model=self.model_id, error=str(e))
            self._pipeline = None

    async def rerank(
        self,
        query: str,
        chunks: List[str],
        top_k: int = 3,
        instruction: Optional[str] = None,
    ) -> List[dict]:
        if not chunks:
            return []
        if self._pipeline is None:
            self._load_pipeline()

        if self._pipeline is None:
            return [{"chunk": c, "score": 0.0, "index": i} for i, c in enumerate(chunks[:top_k])]

        try:
            # Note: instruction is ignored by standard CrossEncoder pipeline
            pairs = [[query, chunk] for chunk in chunks]
            scores = self._pipeline.predict(pairs)
            scored = []
            if isinstance(scores, (list, tuple, np.ndarray)):
                for i, score in enumerate(scores):
                    scored.append({"chunk": chunks[i], "score": float(score), "index": i})
            else:
                scored.append({"chunk": chunks[0], "score": float(scores), "index": 0})
                
            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[:top_k]
        except Exception as e:
            logger.error("rerank_local_failed", error=str(e))
            return [{"chunk": c, "score": 0.0, "index": i} for i, c in enumerate(chunks[:top_k])]


class LMStudioReranker:
    """Optimized LM Studio API Reranker with instruction and batching support."""
    
    def __init__(
        self,
        model_id: str = DEFAULT_RERANK_MODEL,
        base_url: str = LMSTUDIO_BASE,
        default_instruction: str = "retrieval"
    ):
        self.model_id = model_id
        self.base_url = (base_url or "http://127.0.0.1:1234").rstrip("/")
        if not self.base_url.startswith("http"):
            self.base_url = f"http://{self.base_url}"
        self.default_instruction = default_instruction
        self.timeout = 60

    async def rerank(
        self,
        query: str,
        chunks: List[str],
        top_k: int = 3,
        instruction: Optional[str] = None,
        batch_size: int = 32,
    ) -> List[dict]:
        """
        Rerank chunks using LM Studio API with task-specific instructions.
        """
        if not chunks:
            return []
        if not query:
            return [{"chunk": c, "score": 0.0, "index": i} for i, c in enumerate(chunks[:top_k])]

        active_instruction = instruction or self.default_instruction
        inst_text = RERANK_INSTRUCTIONS.get(active_instruction, active_instruction)

        # Batch tasks for concurrent execution
        batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
        
        async def process_batch(client: httpx.AsyncClient, b_chunks: List[str], b_start: int) -> List[dict]:
            payload = {
                "model": self.model_id,
                "query": query,
                "documents": b_chunks,
                "top_k": len(b_chunks),
                "instruction": inst_text,
            }
            # Try multiple endpoints
            for path in ("/v1/rerank", "/api/v1/rerank", "/rerank"):
                try:
                    resp = await client.post(f"{self.base_url}{path}", json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        results = data.get("results", data)
                        if not isinstance(results, list):
                            continue
                            
                        batch_results = []
                        for i, item in enumerate(results):
                            if isinstance(item, dict):
                                score = float(item.get("relevance_score", item.get("score", 0)))
                                chunk = item.get("document") or item.get("text") or b_chunks[i]
                            else:
                                score = float(item)
                                chunk = b_chunks[i]
                                
                            batch_results.append({
                                "chunk": chunk,
                                "score": score,
                                "index": b_start + i
                            })
                        return batch_results
                except Exception:
                    continue
            
            logger.warning("rerank_batch_failed", batch_start=b_start, model=self.model_id)
            return [{"chunk": c, "score": 0.0, "index": b_start + i} for i, c in enumerate(b_chunks)]

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                tasks = [process_batch(client, b, i * batch_size) for i, b in enumerate(batches)]
                all_batches = await asyncio.gather(*tasks)
                
            all_results = [item for sublist in all_batches for item in sublist]
            all_results.sort(key=lambda x: x["score"], reverse=True)
            return all_results[:top_k]

        except Exception as e:
            logger.error("rerank_api_exception", error=str(e))
            return [{"chunk": c, "score": 0.0, "index": i} for i, c in enumerate(chunks[:top_k])]

    async def rerank_advanced(
        self,
        query: str,
        chunks: List[str],
        top_k: int = 3,
        instruction: Optional[str] = None,
        threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """Provides results with additional metadata and filtering."""
        results = await self.rerank(query, chunks, top_k, instruction)
        filtered = [r for r in results if r["score"] >= threshold]
        
        return {
            "results": filtered or results,
            "meta": {
                "total": len(chunks),
                "top_score": results[0]["score"] if results else 0,
                "model": self.model_id,
            }
        }


async def get_reranker(type_: str = "api", **kwargs):
    """
    Factory to retrieve a reranker.
    'api' or 'lmstudio' -> LMStudioReranker
    'local' or 'cross-encoder' -> CrossEncoderReranker
    """
    if type_ in ("local", "cross-encoder"):
        return CrossEncoderReranker(**kwargs)
    return LMStudioReranker(**kwargs)


if __name__ == "__main__":
    import asyncio

    async def test():
        r = await get_reranker("api")
        print("Testing LM Studio Reranker API...")
        results = await r.rerank(
            "How does machine learning work?",
            [
                "Machine learning is a subset of AI that uses data to train models.",
                "Pizza is a popular Italian dish.",
                "Deep learning uses neural networks for complex tasks."
            ],
            top_k=2
        )
        for i, res in enumerate(results, 1):
            print(f"  {i}. Score: {res['score']:.4f} | {res['chunk']}")

    asyncio.run(test())
