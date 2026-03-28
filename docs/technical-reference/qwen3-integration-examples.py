"""
Qwen3 Integration Examples - Practical Code Patterns

Shows how to integrate optimized embedding and reranking
into your proxy and other parts of the system.
"""

import asyncio
from typing import List, Dict, Optional
import structlog

logger = structlog.get_logger()

# ============================================================================
# Pattern 1: Drop-in Replacement in Proxy
# ============================================================================

# OLD CODE (in proxy.py):
"""
from embedder import Embedder
from reranker import get_reranker

embed_engine = Embedder(model=DEFAULT_EMBED_MODEL)
rerank_engine = await get_reranker("lmstudio", model_id=DEFAULT_RERANK_MODEL)
"""

# NEW CODE (with optimization):
"""
from embedder_optimized import QwenEmbedder, MRL_DIMENSIONS
from reranker_optimized import QwenReranker

embed_engine = QwenEmbedder(
    embed_dim=MRL_DIMENSIONS["standard"],  # 1024
    default_instruction="retrieval"
)

rerank_engine = QwenReranker(
    default_instruction="retrieval"
)
"""


# ============================================================================
# Pattern 2: Enhanced Retrieval Bridge in Proxy
# ============================================================================

async def enhanced_retrieval_bridge(
    query: str,
    documents: List[str],
    top_k: int = 5,
    use_reranking: bool = True,
) -> Dict:
    """
    Enhanced retrieval with instructions and metadata tracking.
    
    Useful for: proxy.py bridge_retrieval() function
    """
    from embedder_optimized import QwenEmbedder, MRL_DIMENSIONS
    from reranker_optimized import QwenReranker

    embedder = QwenEmbedder(
        embed_dim=MRL_DIMENSIONS["standard"],
        default_instruction="retrieval"
    )
    
    reranker = QwenReranker(
        default_instruction="retrieval"
    )

    try:
        # Stage 1: Dense retrieval
        logger.info("retrieval_stage1_start", query_len=len(query), doc_count=len(documents))
        
        query_emb = await embedder.embed_query(query, instruction="retrieval")
        doc_embs = await embedder.embed_documents(documents, instruction="retrieval")

        # Compute similarity scores
        similarities = []
        for i, doc_emb in enumerate(doc_embs):
            # Simple cosine similarity (consider using numpy for large-scale)
            score = sum(a*b for a, b in zip(query_emb, doc_emb))
            similarities.append((i, score, documents[i]))

        # Sort by score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get initial candidates (broader set for reranking)
        initial_candidates = [doc for _, _, doc in similarities[:min(20, len(documents))]]
        
        logger.info("retrieval_stage1_complete", 
                   candidates=len(initial_candidates),
                   top_score=similarities[0][1])

        # Stage 2: Rerank if enabled
        if use_reranking and len(initial_candidates) > 0:
            logger.info("retrieval_stage2_start", candidates=len(initial_candidates))
            
            reranked = await reranker.rerank(
                query,
                initial_candidates,
                top_k=top_k,
                instruction="retrieval"
            )
            
            final_results = [r["chunk"] for r in reranked]
            
            logger.info("retrieval_stage2_complete",
                       final_count=len(final_results),
                       top_score=reranked[0]["score"] if reranked else 0)
        else:
            # No reranking, just use initial results
            final_results = [doc for _, _, doc in similarities[:top_k]]

        return {
            "success": True,
            "results": final_results,
            "count": len(final_results),
            "metadata": {
                "stage1_count": len(initial_candidates),
                "final_count": len(final_results),
                "reranking_used": use_reranking,
                "instruction": "retrieval",
            }
        }

    except Exception as e:
        logger.error("enhanced_retrieval_failed", error=str(e))
        return {
            "success": False,
            "results": [],
            "count": 0,
            "error": str(e),
        }


# ============================================================================
# Pattern 3: Context Manager for Resource Efficiency
# ============================================================================

class QwenRetrievalPipeline:
    """
    Manages Qwen embedding and reranking pipeline with caching.
    
    Useful for: Reusing in multiple endpoints or services
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        embedding_instruction: str = "retrieval",
        reranking_instruction: str = "retrieval",
        enable_caching: bool = True,
    ):
        from embedder_optimized import QwenEmbedder, MRL_DIMENSIONS
        from reranker_optimized import QwenReranker

        self.embedder = QwenEmbedder(
            embed_dim=embed_dim,
            default_instruction=embedding_instruction,
        )
        
        self.reranker = QwenReranker(
            default_instruction=reranking_instruction,
        )
        
        self.enable_caching = enable_caching
        self._cache = {}  # Simple dict cache

    async def retrieve_and_rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
        rerank: bool = True,
        cache_key: Optional[str] = None,
    ) -> List[Dict]:
        """
        Full retrieval pipeline with optional caching.
        
        Args:
            query: Search query
            documents: Documents to search
            top_k: Number of results
            rerank: Whether to rerank results
            cache_key: Optional cache key (e.g., document hash)
        """
        
        # Check cache
        cache_id = f"{cache_key}:{query}:{top_k}" if cache_key else None
        if self.enable_caching and cache_id and cache_id in self._cache:
            logger.info("cache_hit", cache_id=cache_id)
            return self._cache[cache_id]

        # Stage 1: Embedding-based retrieval
        query_emb = await self.embedder.embed_query(query)
        doc_embs = await self.embedder.embed_documents(documents)

        scores = []
        for i, doc_emb in enumerate(doc_embs):
            similarity = sum(a*b for a, b in zip(query_emb, doc_emb))
            scores.append({
                "chunk": documents[i],
                "score": similarity,
                "index": i,
            })

        scores.sort(key=lambda x: x["score"], reverse=True)
        initial_results = scores[:min(20, len(scores))]

        # Stage 2: Reranking
        if rerank and len(initial_results) > 0:
            candidates = [r["chunk"] for r in initial_results]
            reranked = await self.reranker.rerank(
                query,
                candidates,
                top_k=top_k,
                instruction="retrieval"
            )
            results = reranked[:top_k]
        else:
            results = initial_results[:top_k]

        # Cache results
        if self.enable_caching and cache_id:
            self._cache[cache_id] = results
            logger.info("cache_store", cache_id=cache_id)

        return results

    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()


# ============================================================================
# Pattern 4: Task-Specific Retrieval
# ============================================================================

async def code_search_retrieval(
    query: str,
    code_files: List[str],
    top_k: int = 5,
) -> List[Dict]:
    """
    Specialized retrieval for code search with appropriate instructions.
    
    Useful for: Agent mode code lookup in Continue IDE
    """
    from embedder_optimized import QwenEmbedder
    from reranker_optimized import QwenReranker

    embedder = QwenEmbedder(
        embed_dim=1536,  # Higher quality for code
        default_instruction="code"  # Code-specific instruction
    )
    
    reranker = QwenReranker(
        default_instruction="code_search"  # If available
    )

    # Retrieve
    query_emb = await embedder.embed_query(query, instruction="code")
    code_embs = await embedder.embed(code_files, instruction="code")

    # Score
    scores = []
    for i, code_emb in enumerate(code_embs):
        score = sum(a*b for a, b in zip(query_emb, code_emb))
        scores.append((i, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    candidates = [code_files[i] for i, _ in scores[:20]]

    # Rerank
    reranked = await reranker.rerank(
        query,
        candidates,
        top_k=top_k,
        instruction="code_search"
    )

    return reranked


async def qa_retrieval(
    question: str,
    answer_pool: List[str],
    top_k: int = 3,
) -> List[Dict]:
    """
    Specialized retrieval for question-answering.
    
    Useful for: QA-focused context selection
    """
    from embedder_optimized import QwenEmbedder
    from reranker_optimized import QwenReranker

    embedder = QwenEmbedder(
        embed_dim=1536,  # Higher quality
        default_instruction="question"
    )
    
    reranker = QwenReranker(
        default_instruction="question_answering"
    )

    # Dense retrieval with QA instruction
    query_emb = await embedder.embed_query(question, instruction="question")
    answer_embs = await embedder.embed(
        answer_pool,
        instruction="answer"
    )

    # Score
    scores = []
    for i, ans_emb in enumerate(answer_embs):
        score = sum(a*b for a, b in zip(query_emb, ans_emb))
        scores.append((i, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    candidates = [answer_pool[i] for i, _ in scores[:15]]

    # Rerank with QA-specific instruction
    reranked = await reranker.rerank(
        question,
        candidates,
        top_k=top_k,
        instruction="question_answering"
    )

    return reranked


# ============================================================================
# Pattern 5: Instruction Customization per Use Case
# ============================================================================

async def flexible_retrieval(
    query: str,
    documents: List[str],
    use_case: str = "retrieval",
    top_k: int = 5,
) -> List[Dict]:
    """
    Dynamically choose instructions based on use case.
    
    Useful for: General-purpose retrieval with custom instructions
    """
    from embedder_optimized import (
        QwenEmbedder,
        TASK_INSTRUCTIONS,
        MRL_DIMENSIONS
    )
    from reranker_optimized import (
        QwenReranker,
        RERANK_INSTRUCTIONS
    )

    # Map use case to appropriate dimensions and instructions
    use_case_config = {
        "retrieval": {
            "embed_dim": MRL_DIMENSIONS["standard"],
            "embed_instruction": "retrieval",
            "rerank_instruction": "retrieval",
            "description": "Standard semantic retrieval"
        },
        "code": {
            "embed_dim": MRL_DIMENSIONS["high"],
            "embed_instruction": "code",
            "rerank_instruction": "code_search",
            "description": "Code search with higher quality"
        },
        "qa": {
            "embed_dim": MRL_DIMENSIONS["high"],
            "embed_instruction": "question",
            "rerank_instruction": "question_answering",
            "description": "Q&A with optimized instructions"
        },
        "similarity": {
            "embed_dim": MRL_DIMENSIONS["compressed"],
            "embed_instruction": "similarity",
            "rerank_instruction": "relevance",
            "description": "Fast similarity with compressed dims"
        },
        "classification": {
            "embed_dim": MRL_DIMENSIONS["standard"],
            "embed_instruction": "classification",
            "rerank_instruction": "relevance",
            "description": "Classification task"
        },
    }

    config = use_case_config.get(use_case, use_case_config["retrieval"])
    
    logger.info("retrieval_use_case", 
                use_case=use_case,
                description=config["description"],
                embed_dim=config["embed_dim"])

    embedder = QwenEmbedder(
        embed_dim=config["embed_dim"],
        default_instruction=config["embed_instruction"]
    )
    
    reranker = QwenReranker(
        default_instruction=config["rerank_instruction"]
    )

    # Retrieve
    query_emb = await embedder.embed_query(query, instruction=config["embed_instruction"])
    doc_embs = await embedder.embed_documents(
        documents,
        instruction=config["embed_instruction"]
    )

    # Score
    scores = []
    for i, doc_emb in enumerate(doc_embs):
        score = sum(a*b for a, b in zip(query_emb, doc_emb))
        scores.append((i, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    candidates = [documents[i] for i, _ in scores[:min(20, len(scores))]]

    # Rerank
    if len(candidates) > 0:
        reranked = await reranker.rerank(
            query,
            candidates,
            top_k=top_k,
            instruction=config["rerank_instruction"]
        )
        return reranked

    return []


# ============================================================================
# Usage Examples
# ============================================================================

async def main():
    """Example usage of all patterns."""

    print("Pattern 1: Enhanced Retrieval Bridge")
    print("-" * 50)
    result = await enhanced_retrieval_bridge(
        query="How do transformers work?",
        documents=[
            "Transformers use self-attention mechanisms",
            "The internet connects computers globally",
            "Python is a programming language",
        ],
        top_k=2
    )
    print(f"Results: {len(result['results'])} documents")
    print()

    print("Pattern 2: Pipeline Context Manager")
    print("-" * 50)
    pipeline = QwenRetrievalPipeline(enable_caching=True)
    results = await pipeline.retrieve_and_rerank(
        query="machine learning algorithms",
        documents=[
            "Neural networks process data through layers",
            "Support vector machines find optimal boundaries",
            "Decision trees split data recursively",
        ],
        top_k=2,
        cache_key="ml_docs"
    )
    print(f"Results: {len(results)} documents (cached)")
    print()

    print("Pattern 3: Task-Specific Retrieval")
    print("-" * 50)
    qa_results = await qa_retrieval(
        question="What is supervised learning?",
        answer_pool=[
            "Supervised learning requires labeled data",
            "Reinforcement learning uses rewards",
            "Unsupervised learning finds patterns",
        ],
        top_k=2
    )
    print(f"Q&A Results: {len(qa_results)} answers\n")

    print("Pattern 4: Flexible Use-Case Routing")
    print("-" * 50)
    flex_results = await flexible_retrieval(
        query="def hello(): return 42",
        documents=[
            "Python functions use def keyword",
            "Java uses public static methods",
            "Decorators enhance function behavior",
        ],
        use_case="code",
        top_k=2
    )
    print(f"Code Search Results: {len(flex_results)} code snippets")


if __name__ == "__main__":
    asyncio.run(main())
