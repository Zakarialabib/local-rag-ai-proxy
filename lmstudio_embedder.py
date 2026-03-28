"""LMStudio Python API-based Embedder with EOS token handling."""

import httpx
import numpy as np
from typing import List, Optional, Dict, Any
import structlog
import json

logger = structlog.get_logger()


class LMStudioPythonEmbedder:
    """
    Direct LMStudio Python API embedder with EOS token support.
    
    Uses LMStudio's native Python inference instead of going through HTTP.
    Handles EOS (end-of-sequence) token automatically for better embedding quality.
    """

    def __init__(
        self,
        lmstudio_address: str = "127.0.0.1:1234",
        model_name: str = "text-embedding-qwen3-embedding-4b",
        add_eos_token: bool = True,
        request_timeout: float = 60.0,
    ):
        """
        Initialize LMStudio embedder.
        
        Args:
            lmstudio_address: LMStudio server address (host:port)
            model_name: Model name/ID in LMStudio
            add_eos_token: Whether to add EOS token (recommended for Qwen3 embedding)
            request_timeout: Request timeout in seconds
        """
        self.lmstudio_address = lmstudio_address
        self.model_name = model_name
        self.add_eos_token = add_eos_token
        self.request_timeout = request_timeout
        self.base_url = f"http://{lmstudio_address}"
        
        logger.info(
            "embedder_init",
            model=model_name,
            add_eos_token=add_eos_token,
            base_url=self.base_url
        )

    def _preprocess_text(self, text: str) -> str:
        """Add EOS token if configured."""
        if not self.add_eos_token:
            return text
        # Qwen3 uses </s> as EOS token marker
        return text.rstrip() + " </s>"

    async def embed(self, texts: List[str], add_eos: Optional[bool] = None) -> List[List[float]]:
        """
        Embed texts using LMStudio Python API.
        
        Args:
            texts: List of texts to embed
            add_eos: Override add_eos_token setting for this call
            
        Returns:
            List of embeddings (vectors)
        """
        if not texts:
            return []
        
        add_eos_token = add_eos if add_eos is not None else self.add_eos_token
        
        try:
            # Preprocess texts with EOS if needed
            processed_texts = [self._preprocess_text(t) if add_eos_token else t for t in texts]
            
            logger.info(
                "embed_start",
                text_count=len(texts),
                add_eos=add_eos_token,
            )
            
            # Try OpenAI-compatible endpoint first (recommended)
            async with httpx.AsyncClient(timeout=self.request_timeout) as client:
                # Try /v1/embeddings (OpenAI-compatible)
                for endpoint in ["/v1/embeddings", "/api/v1/embeddings"]:
                    try:
                        response = await client.post(
                            f"{self.base_url}{endpoint}",
                            json={
                                "input": processed_texts,
                                "model": self.model_name,
                            },
                            timeout=self.request_timeout,
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            embeddings = self._extract_embeddings(data)
                            
                            if embeddings:
                                logger.info(
                                    "embed_success",
                                    endpoint=endpoint,
                                    text_count=len(texts),
                                    embed_dim=len(embeddings[0]) if embeddings else 0,
                                    add_eos=add_eos_token,
                                )
                                return embeddings
                    except Exception as e:
                        logger.debug("embed_endpoint_failed", endpoint=endpoint, error=str(e))
                        continue
                
                # Fallback error
                logger.error(
                    "embed_all_endpoints_failed",
                    model=self.model_name,
                    tried_endpoints=["/v1/embeddings", "/api/v1/embeddings"]
                )
                return []
                
        except Exception as e:
            logger.error("embed_exception", error=str(e), model=self.model_name)
            return []

    def _extract_embeddings(self, response_data: Dict[str, Any]) -> List[List[float]]:
        """Extract embeddings from LMStudio response."""
        if isinstance(response_data, dict):
            # OpenAI-compatible format
            if "data" in response_data:
                data = response_data["data"]
                if isinstance(data, list) and len(data) > 0:
                    if isinstance(data[0], dict) and "embedding" in data[0]:
                        return [item["embedding"] for item in data]
                    if isinstance(data[0], list):
                        return data
            # Fallback: direct embedding format
            if "embedding" in response_data:
                return [response_data["embedding"]]
        
        # Try list format directly
        if isinstance(response_data, list):
            if len(response_data) > 0 and isinstance(response_data[0], list):
                return response_data
        
        return []

    async def embed_query(self, query: str) -> List[float]:
        """Embed a single query."""
        results = await self.embed([query])
        return results[0] if results else []

    async def embed_chunks(self, chunks: List[str]) -> List[List[float]]:
        """Embed multiple chunks."""
        return await self.embed(chunks)

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    async def rerank_by_similarity(
        self,
        query: str,
        chunks: List[str],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Rerank chunks by embedding similarity to query.
        Fallback when reranker is unavailable.
        """
        if not query or not chunks:
            return [{"chunk": c, "score": 0.0} for c in chunks[:top_k]]

        try:
            query_emb = await self.embed_query(query)
            chunk_embs = await self.embed_chunks(chunks)

            if not query_emb or not chunk_embs:
                logger.warning("rerank_no_embeddings", query_len=len(query), chunk_count=len(chunks))
                return [{"chunk": c, "score": 0.0} for c in chunks[:top_k]]

            scored = []
            for i, chunk_emb in enumerate(chunk_embs):
                if isinstance(chunk_emb, list) and chunk_emb:
                    score = self._cosine_similarity(query_emb, chunk_emb)
                else:
                    score = 0.0
                scored.append({"chunk": chunks[i], "score": float(score)})

            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[:top_k]
        except Exception as e:
            logger.error("rerank_failed", error=str(e))
            return [{"chunk": c, "score": 0.0} for c in chunks[:top_k]]


# Backward compatibility: expose as Embedder
class Embedder(LMStudioPythonEmbedder):
    """Backward-compatible Embedder class (alias for LMStudioPythonEmbedder)."""
    pass


if __name__ == "__main__":
    import asyncio

    async def test():
        # Test LMStudio Python embedder with EOS token
        embedder = LMStudioPythonEmbedder(
            lmstudio_address="127.0.0.1:1234",
            model_name="text-embedding-qwen3-embedding-4b",
            add_eos_token=True,  # EOS token enabled
        )
        
        texts = [
            "What is artificial intelligence?",
            "Machine learning is a subset of AI",
            "Python is a programming language",
        ]
        
        print("Embedding texts with EOS token...")
        embeddings = await embedder.embed(texts)
        print(f"✓ Embedded {len(embeddings)} texts, dimension: {len(embeddings[0]) if embeddings else 0}")
        
        if embeddings:
            query = "Tell me about AI"
            results = await embedder.rerank_by_similarity(query, texts, top_k=2)
            print(f"\nRanked results for '{query}':")
            for result in results:
                print(f"  Score: {result['score']:.4f} | {result['chunk'][:50]}...")

    asyncio.run(test())
