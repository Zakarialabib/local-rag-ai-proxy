"""
Optimized Qwen3 Embedding Integration with Instruction-Aware Support

Based on Qwen3 Embedding best practices:
- Instruction-aware embeddings for task-specific performance
- MRL (Matryoshka) support for flexible embedding dimensions
- Dual-encoder architecture optimization
- Multi-language and multi-domain support
"""

import httpx
import numpy as np
from typing import List, Optional, Dict, Literal
import structlog

logger = structlog.get_logger()

LMSTUDIO_BASE = "http://192.168.1.12:1234"
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


class QwenEmbedder:
    """
    Optimized Qwen3 Embedding client with instruction-aware support.
    
    Key features:
    - Task-specific instructions for better semantic understanding
    - MRL (Matryoshka) support for flexible embedding dimensions
    - Batch embedding with efficient HTTP pooling
    - Comprehensive error logging and recovery
    """
    
    def __init__(
        self,
        base_url: str = LMSTUDIO_BASE,
        model: str = DEFAULT_EMBED_MODEL,
        embed_dim: Optional[int] = None,
        default_instruction: str = "retrieval",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.embed_dim = embed_dim  # None = use model's default (2560 for 4B)
        self.default_instruction = default_instruction
        self.timeout = 60  # Increased for large batch embeddings

    async def _embed_with_instruction(
        self,
        texts: List[str],
        instruction: str = "retrieval",
        instruction_type: Literal["query", "document"] = "document",
    ) -> List[List[float]]:
        """
        Embed texts with task-specific instruction.
        
        Args:
            texts: List of texts to embed
            instruction: Task name (retrieval, clustering, etc.) or custom instruction
            instruction_type: For retrieval only - "query" or "document"
        
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Build full instruction
        if instruction in TASK_INSTRUCTIONS:
            full_instruction = TASK_INSTRUCTIONS[instruction]
            # For retrieval, append query/document type
            if instruction in ("retrieval", "similarity") and instruction_type == "query":
                full_instruction = TASK_INSTRUCTIONS.get("retrieval_query", full_instruction)
        else:
            full_instruction = instruction  # Use custom instruction as-is

        payload = {
            "input": texts,
            "model": self.model,
            "instruction": full_instruction,
        }

        # Add MRL dimension if specified
        if self.embed_dim:
            payload["embed_dim"] = self.embed_dim

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                for path in ("/v1/embeddings", "/api/v1/embeddings", "/embeddings"):
                    try:
                        resp = await client.post(
                            f"{self.base_url}{path}",
                            json=payload,
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            items = data.get("data", data) if isinstance(data, dict) else data
                            
                            if isinstance(items, list) and len(items) > 0:
                                first = items[0]
                                if isinstance(first, dict) and "embedding" in first:
                                    embeddings = [item["embedding"] for item in items]
                                    logger.debug(
                                        "embeddings_success",
                                        count=len(embeddings),
                                        dims=len(embeddings[0]) if embeddings else 0,
                                        instruction=instruction,
                                        endpoint=path,
                                    )
                                    return embeddings
                                if isinstance(first, list):
                                    logger.debug(
                                        "embeddings_success",
                                        count=len(items),
                                        dims=len(first),
                                        instruction=instruction,
                                        endpoint=path,
                                    )
                                    return items
                    except httpx.HTTPError as e:
                        logger.debug("embed_endpoint_failed", path=path, error=str(e))
                        continue

                logger.error(
                    "embed_all_endpoints_failed",
                    model=self.model,
                    count=len(texts),
                    instruction=instruction,
                )
                return []

        except Exception as e:
            logger.error(
                "embed_exception",
                error=str(e),
                instruction=instruction,
                count=len(texts),
            )
            return []

    async def embed(
        self,
        texts: List[str],
        instruction: Optional[str] = None,
        instruction_type: Literal["query", "document"] = "document",
    ) -> List[List[float]]:
        """
        Embed texts with optional task-specific instruction.
        
        Args:
            texts: List of texts to embed
            instruction: Task name (retrieval, clustering) or None for default
            instruction_type: "query" or "document" for retrieval instruction
        
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        instruction = instruction or self.default_instruction
        return await self._embed_with_instruction(texts, instruction, instruction_type)

    async def embed_query(
        self,
        query: str,
        instruction: str = "retrieval",
    ) -> List[float]:
        """
        Embed a query with retrieval-specific instruction.
        
        Queries often benefit from shorter embeddings (MRL) and specific phrasing.
        """
        results = await self._embed_with_instruction(
            [query],
            instruction=instruction,
            instruction_type="query",
        )
        return results[0] if results else []

    async def embed_documents(
        self,
        documents: List[str],
        instruction: str = "retrieval",
    ) -> List[List[float]]:
        """
        Embed documents with document-specific instruction.
        
        Documents are longer texts that benefit from full embedding dimensions.
        """
        if not documents:
            return []
        return await self._embed_with_instruction(
            documents,
            instruction=instruction,
            instruction_type="document",
        )

    async def embed_chunks(
        self,
        chunks: List[str],
        instruction: Optional[str] = None,
    ) -> List[List[float]]:
        """Embed chunks with optional instruction (alias for embed_documents)."""
        instruction = instruction or self.default_instruction
        return await self.embed_documents(chunks, instruction=instruction)

    async def rerank_with_embeddings(
        self,
        query: str,
        chunks: List[str],
        top_k: int = 3,
        instruction: str = "retrieval",
    ) -> List[Dict]:
        """
        Rerank chunks using cosine similarity of embeddings.
        
        This is a fallback for when LMStudio's cross-encoder reranker isn't available.
        For better results, use LMStudioReranker with instruction support.
        """
        if not query or not chunks:
            return [{"chunk": c, "score": 0.0} for c in chunks[:top_k]]

        try:
            # Embed query with query-specific instruction
            query_emb = await self.embed_query(query, instruction=instruction)
            
            # Embed documents with document-specific instruction
            chunk_embs = await self.embed_documents(chunks, instruction=instruction)

            if not query_emb or not chunk_embs:
                return [{"chunk": c, "score": 0.0} for c in chunks[:top_k]]

            # Score and rank
            scored = []
            for i, chunk_emb in enumerate(chunk_embs):
                if isinstance(chunk_emb, list) and chunk_emb:
                    score = cosine_score(query_emb, chunk_emb)
                else:
                    score = 0.0
                scored.append({
                    "chunk": chunks[i],
                    "score": float(score),
                    "index": i,
                })

            scored.sort(key=lambda x: x["score"], reverse=True)
            
            logger.info(
                "rerank_embeddings_complete",
                count=len(chunks),
                top_k=top_k,
                top_score=scored[0]["score"] if scored else 0.0,
            )
            
            return scored[:top_k]

        except Exception as e:
            logger.error(
                "rerank_embeddings_failed",
                error=str(e),
                count=len(chunks),
            )
            return [{"chunk": c, "score": 0.0} for c in chunks[:top_k]]


# Backward compatibility
Embedder = QwenEmbedder


if __name__ == "__main__":
    import asyncio

    async def test():
        """Test optimized embedding with instructions."""
        embedder = QwenEmbedder(
            base_url=LMSTUDIO_BASE,
            model=DEFAULT_EMBED_MODEL,
            embed_dim=MRL_DIMENSIONS["standard"],  # Use 1024-dim embeddings
            default_instruction="retrieval",
        )

        # Test 1: Retrieve documents
        print("\n=== Test 1: Retrieval Task ===")
        query = "How does machine learning work?"
        documents = [
            "Machine learning is a subset of AI that enables systems to learn from data.",
            "Neural networks are inspired by biological neurons.",
            "Python is a popular programming language.",
        ]

        query_emb = await embedder.embed_query(query, instruction="retrieval")
        print(f"Query embedding: {len(query_emb)} dimensions")

        doc_embs = await embedder.embed_documents(documents, instruction="retrieval")
        print(f"Document embeddings: {len(doc_embs)} documents, {len(doc_embs[0])} dimensions")

        # Test 2: Rerank
        print("\n=== Test 2: Reranking ===")
        reranked = await embedder.rerank_with_embeddings(
            query,
            documents,
            top_k=2,
            instruction="retrieval",
        )
        for item in reranked:
            print(f"  Score: {item['score']:.4f} | Chunk: {item['chunk'][:50]}...")

    asyncio.run(test())
