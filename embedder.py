import httpx
import numpy as np
from typing import List, Optional, Dict, Literal
import structlog
import os

logger = structlog.get_logger()

LMSTUDIO_BASE = "http://"
DEFAULT_EMBED_MODEL = "text-embedding-qwen3-embedding-4b"

# Task-specific instructions (from Qwen3 documentation)
TASK_INSTRUCTIONS = {
    "retrieval": "Represent the text for retrieval purposes.",
    "retrieval_query": "Represent the query for retrieval purposes.",
    "clustering": "Represent the text for clustering purposes.",
    "classification": "Represent the text for classification purposes.",
    "semantic_search": "Represent the text for semantic search.",
    "similarity": "Represent the text for similarity computation.",
    "reranking": "Represent the text for reranking purposes.",
    "code": "Represent the code snippet for semantic understanding.",
    "question": "Represent the question for question answering.",
    "answer": "Represent the answer for question answering.",
}

# MRL (Matryoshka) dimensions - embed_dim can be 256, 512, 1024, 1536, 2560 for 4B model
MRL_DIMENSIONS = {
    "compact": 256,      # Very compact, faster but lower quality
    "compressed": 512,   # Balanced compression
    "standard": 1024,    # Default quality
    "high": 1536,        # Higher quality
    "full": 2560,        # Full dimension (default for 4B model)
}


def cosine_score(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


class Embedder:
    """
    Optimized LM Studio Embedder with Qwen3 best practices.
    
    Features:
    - Instruction-aware embeddings (task-specific prompts)
    - MRL (Matryoshka) support for flexible embedding dimensions
    - EOS token handling (recommended for Qwen3)
    - Multi-endpoint fallback and error recovery
    """
    
    def __init__(
        self,
        base_url: str = LMSTUDIO_BASE,
        model: str = DEFAULT_EMBED_MODEL,
        add_eos_token: bool = True,
        embed_dim: Optional[int] = None,
        default_instruction: str = "retrieval",
    ):
        self.base_url = (base_url or "http://127.0.0.1:1234").rstrip("/")
        if not self.base_url.startswith("http"):
            self.base_url = f"http://{self.base_url}"
        self.model = model
        self.add_eos_token = add_eos_token
        self.embed_dim = embed_dim
        self.default_instruction = default_instruction
        self.timeout = 60
        self._cache: Dict[str, List[float]] = {}
        self._cache_limit = 1000

    def _preprocess_text(self, text: str, instruction: Optional[str] = None) -> str:
        """Add EOS token and/or instruction if configured."""
        processed = text.strip()
        
        # Add instruction if provided or default
        active_instruction = instruction or self.default_instruction
        if active_instruction and active_instruction in TASK_INSTRUCTIONS:
            inst_text = TASK_INSTRUCTIONS[active_instruction]
            processed = f"{inst_text}\n{processed}"
        elif active_instruction:
             processed = f"{active_instruction}\n{processed}"

        # Add EOS token if configured (Qwen3 uses </s>)
        if self.add_eos_token:
            processed = processed + " </s>"
            
        return processed

    async def embed(
        self,
        texts: List[str],
        instruction: Optional[str] = None,
        embed_dim: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Embed a list of texts with optional instruction and dimension constraint.
        """
        if not texts:
            return []

        active_dim = embed_dim or self.embed_dim
        
        # 1. Check cache for each text
        results = [None] * len(texts)
        missing_indices = []
        missing_processed_texts = []
        
        for i, text in enumerate(texts):
            processed = self._preprocess_text(text, instruction)
            cache_key = f"{active_dim}:{processed}"
            if cache_key in self._cache:
                results[i] = self._cache[cache_key]
            else:
                missing_indices.append(i)
                missing_processed_texts.append(processed)
        
        if not missing_processed_texts:
            return results

        # 2. Get missing embeddings from API
        payload = {
            "input": missing_processed_texts,
            "model": self.model,
        }
        if active_dim:
            payload["embed_dim"] = active_dim

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                for path in ("/v1/embeddings", "/api/v1/embeddings", "/embeddings"):
                    try:
                        resp = await client.post(f"{self.base_url}{path}", json=payload)
                        if resp.status_code == 200:
                            data = resp.json()
                            batch_results = self._extract_embeddings(data, embed_dim=active_dim)
                            
                            if not batch_results:
                                continue

                            # Update cache and merge results
                            for i, emb in enumerate(batch_results):
                                if i >= len(missing_indices): break
                                idx = missing_indices[i]
                                results[idx] = emb
                                
                                # Store in LRU cache
                                proc_text = missing_processed_texts[i]
                                cache_key = f"{active_dim}:{proc_text}"
                                if len(self._cache) >= self._cache_limit:
                                    self._cache.pop(next(iter(self._cache)))
                                self._cache[cache_key] = emb
                                
                            return [r for r in results if r is not None]
                    except Exception as e:
                        logger.debug(f"Path {path} failed: {e}")
                        continue
                
                logger.error("embed_failed", model=self.model, base_url=self.base_url)
                return [r for r in results if r is not None]
        except Exception as e:
            logger.error("embed_exception", error=str(e))
            return [r for r in results if r is not None]

    def _extract_embeddings(self, data: Dict[str, Any], embed_dim: Optional[int] = None) -> List[List[float]]:
        """Extract embeddings from various potential response formats and apply MRL slicing."""
        items = data.get("data", data) if isinstance(data, dict) else data
        raw_list = []
        if isinstance(items, list) and len(items) > 0:
            first = items[0]
            if isinstance(first, dict) and "embedding" in first:
                raw_list = [item["embedding"] for item in items]
            elif isinstance(first, list):
                raw_list = items
        
        if not raw_list:
            return []

        # Apply MRL (Matryoshka) slicing if requested
        active_dim = embed_dim or self.embed_dim
        if active_dim:
            sliced_list = []
            for emb in raw_list:
                # Qwen supports MRL by simple slicing
                sliced = emb[:active_dim]
                # Re-normalize for cosine similarity
                arr = np.array(sliced, dtype=np.float32)
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
                sliced_list.append(arr.tolist())
            return sliced_list
            
        return raw_list

    async def embed_query(self, query: str, instruction: str = "retrieval_query") -> List[float]:
        """Embed a single query, typically using a more specific query instruction."""
        results = await self.embed([query], instruction=instruction)
        return results[0] if results else []

    async def embed_chunks(self, chunks: List[str], instruction: str = "retrieval") -> List[List[float]]:
        """Embed document chunks."""
        return await self.embed(chunks, instruction=instruction)

    async def rerank(
        self,
        query: str,
        chunks: List[str],
        top_k: int = 3,
        instruction: str = "retrieval",
    ) -> List[dict]:
        """
        Rerank chunks based on cosine similarity of their embeddings.
        This provides a high-quality fallback or alternative to a dedicated reranker.
        """
        if not query or not chunks:
            return [{"chunk": c, "score": 0.0} for c in chunks[:top_k]]

        try:
            query_emb = await self.embed_query(query, instruction=f"{instruction}_query" if f"{instruction}_query" in TASK_INSTRUCTIONS else instruction)
            chunk_embs = await self.embed_chunks(chunks, instruction=instruction)

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
        except Exception as e:
            logger.error("rerank_by_embedding_failed", error=str(e))
            return [{"chunk": c, "score": 0.0} for c in chunks[:top_k]]


if __name__ == "__main__":
    import asyncio

    async def test():
        e = Embedder()
        print("Testing basic embedding...")
        texts = ["Hello world", "Python is great for AI", "LM Studio is powerful"]
        emb = await e.embed(texts)
        print(f"✓ Embedded {len(emb)} texts, dim={len(emb[0]) if emb else 0}")

        print("\nTesting instruction-aware reranking...")
        results = await e.rerank("Tell me about machine learning tools", texts, top_k=2)
        for r in results:
            print(f"  Score: {r['score']:.4f} | Chunk: {r['chunk']}")

    asyncio.run(test())
