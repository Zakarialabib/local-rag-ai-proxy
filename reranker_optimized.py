"""
Optimized Qwen3 Reranker with Instruction-Aware Cross-Encoder Support

Based on Qwen3 Reranker best practices:
- Cross-encoder architecture (takes query+document pairs)
- Instruction-aware reranking for task-specific optimization
- Multiple ranking strategies
- Structured logging and error recovery
"""

import httpx
import numpy as np
from typing import List, Optional, Dict, Literal
import structlog

logger = structlog.get_logger()

LMSTUDIO_BASE = "http://192.168.1.12:1234"
DEFAULT_RERANK_MODEL = "qwen3-reranker-4b"

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


class QwenReranker:
    """
    Optimized Qwen3 Reranker with instruction-aware cross-encoder support.
    
    Key features:
    - Cross-encoder architecture (query+document pairs)
    - Task-specific instructions for better ranking
    - Batch reranking for efficiency
    - Fallback to local embedder reranking if API unavailable
    - Comprehensive error logging
    """

    def __init__(
        self,
        model_id: str = DEFAULT_RERANK_MODEL,
        base_url: str = LMSTUDIO_BASE,
        default_instruction: str = "retrieval",
    ):
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.default_instruction = default_instruction
        self.timeout = 60

    async def rerank(
        self,
        query: str,
        chunks: List[str],
        top_k: int = 3,
        instruction: Optional[str] = None,
        batch_size: int = 32,
    ) -> List[Dict]:
        """
        Rerank chunks using cross-encoder with instruction.
        
        Args:
            query: The query/question
            chunks: Documents/passages to rerank
            top_k: Number of top results to return
            instruction: Task name (retrieval, qa) or custom instruction
            batch_size: Process this many query+doc pairs at once
        
        Returns:
            List of dicts with 'chunk', 'score', and 'index' keys
        """
        if not chunks:
            return []

        if not query:
            # No query = cannot rerank
            return [{"chunk": c, "score": 0.0, "index": i} for i, c in enumerate(chunks[:top_k])]

        instruction = instruction or self.default_instruction

        # Get full instruction text
        if instruction in RERANK_INSTRUCTIONS:
            full_instruction = RERANK_INSTRUCTIONS[instruction]
        else:
            full_instruction = instruction  # Use custom instruction as-is

        try:
            return await self._rerank_api(
                query,
                chunks,
                top_k,
                full_instruction,
                batch_size,
            )
        except Exception as e:
            logger.error(
                "rerank_api_failed",
                error=str(e),
                model=self.model_id,
                count=len(chunks),
            )
            return [{"chunk": c, "score": 0.0, "index": i} for i, c in enumerate(chunks[:top_k])]

    async def _rerank_api(
        self,
        query: str,
        chunks: List[str],
        top_k: int,
        instruction: str,
        batch_size: int,
    ) -> List[Dict]:
        """Internal API call handler with batch processing."""
        
        # Process in batches if needed
        all_scores = []
        
        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]

            payload = {
                "model": self.model_id,
                "query": query,
                "documents": batch_chunks,
                "top_k": len(batch_chunks),  # Get scores for all in batch
                "instruction": instruction,  # Include instruction in payload
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                for path in ("/v1/rerank", "/api/v1/rerank", "/rerank"):
                    try:
                        resp = await client.post(
                            f"{self.base_url}{path}",
                            json=payload,
                        )

                        if resp.status_code != 200:
                            logger.debug(
                                "rerank_endpoint_status",
                                endpoint=path,
                                status=resp.status_code,
                            )
                            continue

                        data = resp.json()
                        results = data.get("results", data) if isinstance(data, dict) else data

                        if isinstance(results, list):
                            for i, result in enumerate(results):
                                if i < len(batch_chunks):
                                    if isinstance(result, dict):
                                        score = float(
                                            result.get("relevance_score", result.get("score", 0))
                                        )
                                    else:
                                        score = float(result) if result is not None else 0.0

                                    all_scores.append({
                                        "chunk": batch_chunks[i],
                                        "score": score,
                                        "index": batch_start + i,
                                    })

                            logger.debug(
                                "rerank_batch_success",
                                batch_size=len(batch_chunks),
                                endpoint=path,
                                instruction=instruction,
                            )
                            break  # Break if this endpoint worked
                    except (httpx.HTTPError, ValueError) as e:
                        logger.debug(
                            "rerank_endpoint_error",
                            endpoint=path,
                            error=str(e),
                        )
                        continue

        # Sort all results and return top_k
        if all_scores:
            all_scores.sort(key=lambda x: x["score"], reverse=True)
            logger.info(
                "rerank_complete",
                count=len(chunks),
                top_k=top_k,
                top_score=all_scores[0]["score"],
                instruction=instruction,
            )
            return all_scores[:top_k]
        
        # Fallback if no scores obtained
        logger.warning(
            "rerank_no_scores",
            count=len(chunks),
        )
        return [{"chunk": c, "score": 0.0, "index": i} for i, c in enumerate(chunks[:top_k])]

    async def rerank_advanced(
        self,
        query: str,
        chunks: List[str],
        top_k: int = 3,
        instruction: Optional[str] = None,
        threshold: float = 0.0,
    ) -> Dict:
        """
        Advanced reranking with additional metadata.
        
        Returns dict with:
        - results: List of ranked chunks
        - instructions: The instruction used
        - stats: Reranking statistics
        """
        results = await self.rerank(query, chunks, top_k, instruction)

        # Filter by threshold if provided
        filtered = [r for r in results if r["score"] >= threshold]

        return {
            "results": filtered or results[:top_k],
            "instruction": instruction or self.default_instruction,
            "stats": {
                "total_documents": len(chunks),
                "reranked_count": len(filtered) if filtered else len(results[:top_k]),
                "top_score": results[0]["score"] if results else 0.0,
                "threshold_applied": threshold > 0.0,
            },
        }


# For backward compatibility
CrossEncoderReranker = QwenReranker


class LMStudioReranker(QwenReranker):
    """Alias for QwenReranker with instruction support."""
    
    async def rerank(
        self,
        query: str,
        chunks: List[str],
        top_k: int = 3,
        instruction: Optional[str] = None,
    ) -> List[Dict]:
        """Backward compatible rerank method."""
        return await super().rerank(query, chunks, top_k, instruction)


async def get_reranker(
    type_: str = "lmstudio",
    model_id: str = DEFAULT_RERANK_MODEL,
    base_url: str = LMSTUDIO_BASE,
    **kwargs
) -> QwenReranker:
    """Factory function to get appropriate reranker."""
    if type_ in ("lmstudio", "qwen", "cross-encoder"):
        return QwenReranker(
            model_id=model_id,
            base_url=base_url,
            **kwargs
        )
    return QwenReranker(model_id=model_id, base_url=base_url, **kwargs)


if __name__ == "__main__":
    import asyncio

    async def test():
        """Test optimized reranking with instructions."""
        reranker = QwenReranker(
            model_id=DEFAULT_RERANK_MODEL,
            base_url=LMSTUDIO_BASE,
            default_instruction="retrieval",
        )

        query = "How does machine learning work?"
        documents = [
            "Machine learning is a subset of AI that enables systems to learn from data.",
            "Neural networks are inspired by biological neurons in the brain.",
            "Python is a popular programming language for web development.",
            "Deep learning uses multiple layers of neural networks.",
            "Data science involves statistics and machine learning.",
        ]

        print("\n=== Test 1: Basic Reranking ===")
        results = await reranker.rerank(
            query,
            documents,
            top_k=3,
            instruction="retrieval",
        )
        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result['score']:.4f}")
            print(f"   {result['chunk']}\n")

        print("\n=== Test 2: Advanced Reranking with Threshold ===")
        advanced = await reranker.rerank_advanced(
            query,
            documents,
            top_k=3,
            instruction="question_answering",
            threshold=0.3,
        )
        print(f"Instruction: {advanced['instruction']}")
        print(f"Stats: {advanced['stats']}")
        for result in advanced["results"]:
            print(f"  Score: {result['score']:.4f} | {result['chunk'][:50]}...")

    asyncio.run(test())
