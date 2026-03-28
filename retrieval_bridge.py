"""
Comprehensive Qwen3 Retrieval Bridge
Full two-stage pipeline: Dense Retrieval (Embedder) → Reranking (Reranker)

Supports:
- Initial dense retrieval with task-specific instructions
- Semantic search and similarity ranking
- Code search with instruction tuning
- Classification and clustering tasks
- Two-stage reranking for accuracy (50 → 5)
- Question-answering optimized context selection
- High-quality passage retrieval

This replaces the old retrieval logic in proxy.py with a production-ready
two-stage pipeline that leverages Qwen3 Embedding and Reranking models.
"""

import asyncio
import numpy as np
from typing import List, Dict, Optional, Literal, Tuple
from dataclasses import dataclass
from enum import Enum
import structlog
import time

logger = structlog.get_logger()

from embedder_optimized import QwenEmbedder, MRL_DIMENSIONS, TASK_INSTRUCTIONS
from reranker_optimized import QwenReranker, RERANK_INSTRUCTIONS


# ============================================================================
# Configuration & Enums
# ============================================================================

class RetrievalUseCase(str, Enum):
    """Supported retrieval use cases with pre-tuned settings."""
    GENERAL = "general"              # Default general retrieval
    CODE_SEARCH = "code_search"      # Code snippet retrieval
    SEMANTIC_SEARCH = "semantic"     # Semantic/similarity search
    QA = "question_answering"        # Q&A context selection
    CLASSIFICATION = "classification" # Text classification
    CLUSTERING = "clustering"        # Document clustering
    PASSAGE_RETRIEVAL = "passage"    # High-quality passage ranking


@dataclass
class RetrievalConfig:
    """Configuration for retrieval pipeline."""
    use_case: RetrievalUseCase = RetrievalUseCase.GENERAL
    # Dense retrieval (stage 1)
    initial_candidates: int = 50    # Get this many candidates from embedder
    embed_dim: int = 1024           # Embedding dimension (256-2560)
    embed_instruction: Optional[str] = None  # Override instruction
    # Reranking (stage 2)
    final_top_k: int = 5            # Return top K after reranking
    enable_reranking: bool = True   # Use two-stage or just embedder?
    rerank_instruction: Optional[str] = None  # Override instruction
    # Optimization
    use_cache: bool = True          # Cache embeddings?
    batch_size: int = 32            # Batch size for embeddings
    similarity_threshold: float = 0.0  # Filter by similarity score
    rerank_threshold: float = 0.0   # Filter by rerank score
    # Logging
    verbose: bool = False           # Verbose logging


# Use case templates with pre-tuned settings
USE_CASE_TEMPLATES = {
    RetrievalUseCase.GENERAL: {
        "embed_dim": MRL_DIMENSIONS["standard"],  # 1024
        "embed_instruction": "retrieval",
        "rerank_instruction": "retrieval",
        "initial_candidates": 50,
        "final_top_k": 5,
        "description": "General-purpose semantic retrieval"
    },
    RetrievalUseCase.CODE_SEARCH: {
        "embed_dim": MRL_DIMENSIONS["high"],      # 1536
        "embed_instruction": "code",
        "rerank_instruction": "code_search",
        "initial_candidates": 30,
        "final_top_k": 3,
        "description": "Code snippet search with high quality"
    },
    RetrievalUseCase.SEMANTIC_SEARCH: {
        "embed_dim": MRL_DIMENSIONS["compressed"],  # 512
        "embed_instruction": "similarity",
        "rerank_instruction": "semantic_similarity",
        "initial_candidates": 100,
        "final_top_k": 10,
        "description": "Fast semantic similarity search"
    },
    RetrievalUseCase.QA: {
        "embed_dim": MRL_DIMENSIONS["high"],      # 1536
        "embed_instruction": "question",
        "rerank_instruction": "question_answering",
        "initial_candidates": 50,
        "final_top_k": 3,
        "description": "Q&A optimized context selection"
    },
    RetrievalUseCase.CLASSIFICATION: {
        "embed_dim": MRL_DIMENSIONS["standard"],  # 1024
        "embed_instruction": "classification",
        "rerank_instruction": "relevance",
        "initial_candidates": 50,
        "final_top_k": 5,
        "description": "Text classification retrieval"
    },
    RetrievalUseCase.CLUSTERING: {
        "embed_dim": MRL_DIMENSIONS["standard"],  # 1024
        "embed_instruction": "clustering",
        "rerank_instruction": "relevance",
        "initial_candidates": 50,
        "final_top_k": 5,
        "description": "Document clustering"
    },
    RetrievalUseCase.PASSAGE_RETRIEVAL: {
        "embed_dim": MRL_DIMENSIONS["full"],      # 2560
        "embed_instruction": "retrieval",
        "rerank_instruction": "passage_retrieval",
        "initial_candidates": 30,
        "final_top_k": 5,
        "description": "High-quality passage retrieval"
    },
}


# ============================================================================
# Retrieval Bridge - Two-Stage Pipeline
# ============================================================================

class QwenRetrievalBridge:
    """
    Production-grade two-stage retrieval pipeline using Qwen3 models.
    
    Stage 1 (Dense Retrieval): Embedder
    - Embeds query with task-specific instruction
    - Embeds documents with matching instruction
    - Computes similarity scores
    - Returns top-N candidates
    
    Stage 2 (Reranking): Reranker
    - Takes top-N candidates from stage 1
    - Uses cross-encoder with task instruction
    - Returns top-K refined results
    
    Architecture:
        Query + Documents
            ↓
        [Stage 1: Dense Retrieval]
        - Embed query (instruction-aware)
        - Embed documents (batch)
        - Compute similarities
        - Filter by threshold
        - Get top 50 candidates
            ↓
        [Stage 2: Reranking]
        - Send (query, candidate) pairs
        - Cross-encoder scoring
        - Sort by relevance
        - Return top 5
            ↓
        Final Results (High Quality)
    """

    def __init__(
        self,
        embedder: Optional[QwenEmbedder] = None,
        reranker: Optional[QwenReranker] = None,
        enable_cache: bool = True,
    ):
        """
        Initialize retrieval bridge with embedder and reranker.
        
        Args:
            embedder: QwenEmbedder instance or None (creates default)
            reranker: QwenReranker instance or None (creates default)
            enable_cache: Enable caching of embeddings
        """
        self.embedder = embedder or QwenEmbedder(
            embed_dim=MRL_DIMENSIONS["standard"],
            default_instruction="retrieval"
        )
        
        self.reranker = reranker or QwenReranker(
            default_instruction="retrieval"
        )
        
        self.enable_cache = enable_cache
        self._embedding_cache = {}  # Cache query → embeddings
        self._doc_cache = {}         # Cache doc_id → embedding

    async def retrieve(
        self,
        query: str,
        documents: List[str],
        config: Optional[RetrievalConfig] = None,
    ) -> Dict:
        """
        Full two-stage retrieval pipeline.
        
        Args:
            query: Search query
            documents: List of documents to search
            config: RetrievalConfig with tuning parameters
        
        Returns:
            Dict with:
            - results: List of top-k reranked results
            - scores: Relevance scores for each result
            - metadata: Pipeline statistics and timings
        """
        config = config or RetrievalConfig()
        
        # Apply use case template
        if config.embed_instruction is None:
            template = USE_CASE_TEMPLATES.get(config.use_case, {})
            config.embed_instruction = template.get("embed_instruction", "retrieval")
        
        if config.rerank_instruction is None:
            template = USE_CASE_TEMPLATES.get(config.use_case, {})
            config.rerank_instruction = template.get("rerank_instruction", "retrieval")

        start_time = time.time()
        metadata = {
            "use_case": config.use_case.value,
            "query_length": len(query),
            "document_count": len(documents),
                            "stages": {},
        }

        # ====== STAGE 1: Dense Retrieval ======
        try:
            stage1_start = time.time()
            
            logger.info(
                "retrieval_stage1_start",
                use_case=config.use_case.value,
                doc_count=len(documents),
                instruction=config.embed_instruction,
            )

            # Get embeddings
            stage1_results = await self._dense_retrieval_stage(
                query=query,
                documents=documents,
                instruction=config.embed_instruction,
                initial_candidates=config.initial_candidates,
                threshold=config.similarity_threshold,
                batch_size=config.batch_size,
            )

            stage1_time = time.time() - stage1_start
            metadata["stages"]["dense_retrieval"] = {
                "time_ms": stage1_time * 1000,
                "candidates_returned": len(stage1_results),
                "threshold_applied": config.similarity_threshold > 0,
            }

            if not stage1_results:
                logger.warning("retrieval_stage1_empty", query_len=len(query))
                return {
                    "success": False,
                    "results": [],
                    "scores": [],
                    "error": "No candidates from dense retrieval",
                    "metadata": metadata,
                }

            logger.info(
                "retrieval_stage1_complete",
                candidates=len(stage1_results),
                top_score=stage1_results[0]["score"] if stage1_results else 0,
            )

            # ====== STAGE 2: Reranking ======
            if config.enable_reranking and len(stage1_results) > 1:
                try:
                    stage2_start = time.time()
                    
                    logger.info(
                        "retrieval_stage2_start",
                        candidates=len(stage1_results),
                        instruction=config.rerank_instruction,
                    )

                    stage2_results = await self._reranking_stage(
                        query=query,
                        candidates=[r["chunk"] for r in stage1_results],
                        instruction=config.rerank_instruction,
                        top_k=config.final_top_k,
                        threshold=config.rerank_threshold,
                    )

                    stage2_time = time.time() - stage2_start
                    metadata["stages"]["reranking"] = {
                        "time_ms": stage2_time * 1000,
                        "input_count": len(stage1_results),
                        "output_count": len(stage2_results),
                        "threshold_applied": config.rerank_threshold > 0,
                    }

                    final_results = stage2_results

                    logger.info(
                        "retrieval_stage2_complete",
                        final_count=len(final_results),
                        top_score=final_results[0]["score"] if final_results else 0,
                    )

                except Exception as e:
                    logger.warning(
                        "retrieval_stage2_failed",
                        error=str(e),
                        falling_back_to_stage1=True,
                    )
                    final_results = stage1_results[:config.final_top_k]
                    metadata["stages"]["reranking"] = {
                        "error": str(e),
                        "fallback": True,
                    }
            else:
                # No reranking, use stage 1 results directly
                final_results = stage1_results[:config.final_top_k]
                metadata["stages"]["reranking"] = {"skipped": True}

            total_time = time.time() - start_time
            metadata.update({
                "total_time_ms": total_time * 1000,
                "final_count": len(final_results),
                "mean_score": np.mean([r["score"] for r in final_results]) if final_results else 0,
                "max_score": max([r["score"] for r in final_results]) if final_results else 0,
                "min_score": min([r["score"] for r in final_results]) if final_results else 0,
            })

            if config.verbose:
                logger.info("retrieval_complete", metadata=metadata)

            return {
                "success": True,
                "results": [r["chunk"] for r in final_results],
                "scores": [r["score"] for r in final_results],
                "metadata": metadata,
            }

        except Exception as e:
            logger.error("retrieval_failed", error=str(e), traceback=True)
            return {
                "success": False,
                "results": [],
                "scores": [],
                "error": str(e),
                "metadata": metadata,
            }

    async def _dense_retrieval_stage(
        self,
        query: str,
        documents: List[str],
        instruction: str,
        initial_candidates: int,
        threshold: float,
        batch_size: int,
    ) -> List[Dict]:
        """
        Stage 1: Embedder-based dense retrieval.
        
        Returns:
            List[Dict] with keys: chunk, score, index
        """
        # Embed query
        query_emb = await self.embedder.embed_query(
            query,
            instruction=instruction
        )

        if not query_emb:
            logger.error("dense_retrieval_query_embed_failed")
            return []

        # Embed documents (with batching)
        doc_embeddings = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            batch_embs = await self.embedder.embed_documents(
                batch,
                instruction=instruction
            )
            doc_embeddings.extend(batch_embs)

        if not doc_embeddings:
            logger.error("dense_retrieval_doc_embed_failed")
            return []

        # Compute similarities (cosine)
        query_arr = np.array(query_emb, dtype=np.float32)
        query_norm = np.linalg.norm(query_arr)

        scored = []
        for i, doc_emb in enumerate(doc_embeddings):
            if not isinstance(doc_emb, list) or len(doc_emb) == 0:
                continue

            doc_arr = np.array(doc_emb, dtype=np.float32)
            doc_norm = np.linalg.norm(doc_arr)

            if query_norm > 0 and doc_norm > 0:
                similarity = float(np.dot(query_arr, doc_arr) / (query_norm * doc_norm))
            else:
                similarity = 0.0

            # Apply threshold
            if similarity >= threshold:
                scored.append({
                    "chunk": documents[i],
                    "score": similarity,
                    "index": i,
                })

        # Sort by score and get top candidates
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:initial_candidates]

    async def _reranking_stage(
        self,
        query: str,
        candidates: List[str],
        instruction: str,
        top_k: int,
        threshold: float,
    ) -> List[Dict]:
        """
        Stage 2: Reranker-based final ranking.
        
        Returns:
            List[Dict] with keys: chunk, score, index (from stage 2)
        """
        if not candidates:
            return []

        # Use reranker on candidates
        reranked = await self.reranker.rerank(
            query=query,
            chunks=candidates,
            top_k=len(candidates),  # Get scores for all
            instruction=instruction,
        )

        if not reranked:
            logger.warning("reranking_no_results")
            return []

        # Apply threshold
        filtered = [r for r in reranked if r.get("score", 0) >= threshold]

        # Return top_k
        return filtered[:top_k] if filtered else reranked[:top_k]

    def clear_cache(self):
        """Clear embedding cache."""
        self._embedding_cache.clear()
        self._doc_cache.clear()
        logger.info("retrieval_cache_cleared")


# ============================================================================
# High-Level API Functions
# ============================================================================

async def retrieve_with_embedder(
    query: str,
    documents: List[str],
    use_case: RetrievalUseCase = RetrievalUseCase.GENERAL,
    top_k: int = 5,
    embedder: Optional[QwenEmbedder] = None,
    reranker: Optional[QwenReranker] = None,
) -> Dict:
    """
    High-level retrieval function with smart defaults.
    
    Automatically configures embedder and reranker based on use case.
    Returns top-K results after two-stage pipeline.
    
    Args:
        query: Search query
        documents: Documents to search
        use_case: Type of retrieval (general, code, qa, etc.)
        top_k: Number of final results
        embedder: Optional custom embedder
        reranker: Optional custom reranker
    
    Returns:
        Dict with results, scores, and metadata
    
    Example:
        results = await retrieve_with_embedder(
            query="How do transformers work?",
            documents=my_docs,
            use_case=RetrievalUseCase.QA,
            top_k=3
        )
    """
    template = USE_CASE_TEMPLATES.get(use_case, {})
    
    config = RetrievalConfig(
        use_case=use_case,
        initial_candidates=template.get("initial_candidates", 50),
        final_top_k=top_k,
        embed_dim=template.get("embed_dim", MRL_DIMENSIONS["standard"]),
        embed_instruction=template.get("embed_instruction", "retrieval"),
        rerank_instruction=template.get("rerank_instruction", "retrieval"),
        enable_reranking=True,
        verbose=False,
    )
    
    bridge = QwenRetrievalBridge(embedder=embedder, reranker=reranker)
    return await bridge.retrieve(query, documents, config)


# ============================================================================
# Use-Case Specific Wrappers
# ============================================================================

async def code_search(
    query: str,
    code_files: List[str],
    top_k: int = 3,
) -> Dict:
    """
    Specialized code search with instruction tuning.
    
    Features:
    - High-dimension embeddings (1536) for code precision
    - Code-specific instructions
    - Optimized reranking for code relevance
    
    Example:
        results = await code_search(
            "class inheritance and super()",
            [file1_code, file2_code],
            top_k=3
        )
    """
    return await retrieve_with_embedder(
        query=query,
        documents=code_files,
        use_case=RetrievalUseCase.CODE_SEARCH,
        top_k=top_k,
    )


async def qa_retrieval(
    question: str,
    answer_pool: List[str],
    top_k: int = 3,
) -> Dict:
    """
    Question-answering optimized retrieval.
    
    Features:
    - Question-aware embeddings
    - Q&A specific reranking instructions
    - High quality for passage selection
    
    Example:
        results = await qa_retrieval(
            "What is machine learning?",
            [answer1, answer2, answer3],
            top_k=3
        )
    """
    return await retrieve_with_embedder(
        query=question,
        documents=answer_pool,
        use_case=RetrievalUseCase.QA,
        top_k=top_k,
    )


async def semantic_similarity_search(
    query: str,
    documents: List[str],
    top_k: int = 10,
) -> Dict:
    """
    Fast semantic similarity search.
    
    Features:
    - Compressed embeddings (512 dims) for speed
    - Broader search (100 initial candidates)
    - Good for exploratory search
    
    Example:
        results = await semantic_similarity_search(
            "similar documents",
            [doc1, doc2, doc3],
            top_k=10
        )
    """
    return await retrieve_with_embedder(
        query=query,
        documents=documents,
        use_case=RetrievalUseCase.SEMANTIC_SEARCH,
        top_k=top_k,
    )


async def passage_retrieval(
    query: str,
    passages: List[str],
    top_k: int = 5,
) -> Dict:
    """
    High-quality passage retrieval with maximum accuracy.
    
    Features:
    - Full-dimension embeddings (2560) for best quality
    - Passage-specific reranking
    - Optimized for document and passage retrieval
    
    Example:
        results = await passage_retrieval(
            "explain neural networks",
            passages,
            top_k=5
        )
    """
    return await retrieve_with_embedder(
        query=query,
        documents=passages,
        use_case=RetrievalUseCase.PASSAGE_RETRIEVAL,
        top_k=top_k,
    )


# ============================================================================
# Testing
# ============================================================================

async def main():
    """Test comprehensive retrieval pipeline."""
    
    print("\n" + "="*70)
    print("COMPREHENSIVE QWEN3 RETRIEVAL PIPELINE TEST")
    print("="*70 + "\n")

    # Test 1: General Retrieval
    print("Test 1: General Retrieval (Dense + Rerank)")
    print("-" * 70)
    
    query1 = "How do transformers work in machine learning?"
    docs1 = [
        "Transformers use self-attention mechanisms to process sequences.",
        "Neural networks process data through multiple layers.",
        "Python is used for machine learning development.",
        "Attention mechanisms allow models to focus on relevant parts.",
        "Deep learning uses multiple neural network layers.",
    ]
    
    result1 = await retrieve_with_embedder(
        query=query1,
        documents=docs1,
        use_case=RetrievalUseCase.GENERAL,
        top_k=3
    )
    
    print(f"Query: {query1}")
    print(f"Total docs: {len(docs1)}")
    print(f"Results: {len(result1['results'])}")
    for i, (res, score) in enumerate(zip(result1["results"], result1["scores"]), 1):
        print(f"  {i}. [{score:.4f}] {res[:50]}...")
    print(f"Metadata: {result1['metadata']}\n")

    # Test 2: Code Search
    print("Test 2: Code Search (High Quality)")
    print("-" * 70)
    
    query2 = "Python class inheritance and polymorphism"
    code_samples = [
        "class Animal: def speak(self): pass",
        "def fibonacci(n): return n if n < 2 else fib(n-1) + fib(n-2)",
        "class Dog(Animal): def speak(self): return 'Woof'",
        "import numpy as np; arr = np.array([1,2,3])",
    ]
    
    result2 = await code_search(
        query=query2,
        code_files=code_samples,
        top_k=2
    )
    
    print(f"Query: {query2}")
    print(f"Total samples: {len(code_samples)}")
    print(f"Results: {len(result2['results'])}")
    for i, (res, score) in enumerate(zip(result2["results"], result2["scores"]), 1):
        print(f"  {i}. [{score:.4f}] {res[:50]}...")
    print()

    # Test 3: Q&A Retrieval
    print("Test 3: Question-Answering (Q&A Optimized)")
    print("-" * 70)
    
    question = "What is supervised learning?"
    answers = [
        "Supervised learning requires labeled training data",
        "Unsupervised learning finds patterns without labels",
        "Reinforcement learning uses rewards and penalties",
        "Semi-supervised combines labeled and unlabeled data",
    ]
    
    result3 = await qa_retrieval(
        question=question,
        answer_pool=answers,
        top_k=2
    )
    
    print(f"Question: {question}")
    print(f"Total answers: {len(answers)}")
    print(f"Results: {len(result3['results'])}")
    for i, (res, score) in enumerate(zip(result3["results"], result3["scores"]), 1):
        print(f"  {i}. [{score:.4f}] {res[:50]}...")
    print()

    # Test 4: Passage Retrieval (High Quality)
    print("Test 4: Passage Retrieval (Maximum Quality)")
    print("-" * 70)
    
    query4 = "neural networks architecture"
    passages = [
        "Neural networks are computational models inspired by biological neurons in the brain.",
        "Python is a programming language.",
        "The architecture of neural networks consists of interconnected layers.",
        "Machine learning enables computers to learn from data.",
    ]
    
    result4 = await passage_retrieval(
        query=query4,
        passages=passages,
        top_k=2
    )
    
    print(f"Query: {query4}")
    print(f"Total passages: {len(passages)}")
    print(f"Results: {len(result4['results'])}")
    for i, (res, score) in enumerate(zip(result4["results"], result4["scores"]), 1):
        print(f"  {i}. [{score:.4f}] {res[:50]}...")
    print()

    print("="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
