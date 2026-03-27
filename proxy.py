from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from benchmark import StreamingBenchmark
from contextlib import asynccontextmanager
import httpx
import asyncio
import time
import json
import sys
from typing import Dict, Any, Optional, List
from pathlib import Path
import os
from hardware_detector import HardwareDetector
from engine import RecommendationEngine
from models import HardwareProfile
from tuner import AdaptiveTuner
from cache_manager import CacheManager
from domain_router import DomainRouter
from context_manager import ContextManager
from embedder import Embedder
from reranker import get_reranker, LMStudioReranker
import logging
import structlog
from datetime import datetime

EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "bge-reranker-v2-m3")
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))
# from exporters import export_preset_json  # You'll need to create this

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
structlog.configure(
    processors=[
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=False,
)
logger = structlog.get_logger("proxy")


# Cache to track recent requests
recent_requests = {}

def handle_request(request):
    global recent_requests
    request_key = (request.method, request.url.path, request.url.query)

    current_time = datetime.now()
    for k, last_time in list(recent_requests.items()):
        if (current_time - last_time).total_seconds() > 30:
            recent_requests.pop(k, None)

    if request_key in recent_requests:
        last_time = recent_requests[request_key]
        if (current_time - last_time).total_seconds() < 5:
            return

    logger.debug("http_request", method=request.method, path=request.url.path, query=request.url.query)
    recent_requests[request_key] = current_time
    return

class OptimizingProxy:
    def __init__(self):
        self.lmstudio_base = os.getenv("LMSTUDIO_BASE_URL", "http://192.168.1.12:1234").rstrip("/")
        self.hardware: Optional[HardwareProfile] = None
        self.detector = HardwareDetector()
        self.engine: Optional[RecommendationEngine] = None
        self.tuner = AdaptiveTuner(self.lmstudio_base)
        self.cache = CacheManager()
        self.router = DomainRouter()
        self.context = ContextManager()
        self.embedder = Embedder(base_url=self.lmstudio_base, model=EMBED_MODEL)
        self._reranker: Optional[LMStudioReranker] = None
        
    async def get_hardware(self) -> HardwareProfile:
        if not self.hardware:
            loop = asyncio.get_event_loop()
            self.hardware = await loop.run_in_executor(None, self.detector.detect)
            self.engine = RecommendationEngine(self.hardware)
        return self.hardware

    async def optimize_request(self, body: Dict[str, Any], domain: str) -> Dict[str, Any]:
        hw = await self.get_hardware()
        model_id = body.get("model", "default")
        
        # 1. Prune conversation history to reduce token usage
        if "messages" in body:
            original_count = len(body["messages"])
            body["messages"] = self.context.prune_conversation_history(body["messages"])
            if original_count > len(body["messages"]):
                logger.info("context_pruned", original=original_count, pruned=len(body["messages"]), domain=domain)
                          
            # Context compression hook
            if self.context.should_compress(body["messages"]):
                body["messages"] = self.context.compress_context(body["messages"])
                logger.info("context_compressed")
        
        # 2. Detect and inject domain-specific system prompt
        if "messages" in body:
            body["messages"] = self.router.optimize_prompts(body["messages"], model_id, domain)
        
        # 3. Hardware-aware context limits
        vram = hw.gpu_vram_gb or 0
        if vram >= 24: max_ctx = 32768
        elif vram >= 12: max_ctx = 16384
        elif vram >= 8: max_ctx = 8192
        else: max_ctx = 4096
        
        body["max_tokens"] = min(body.get("max_tokens", 2048), max_ctx)
        
        # 4. Apply domain-specific generation settings
        if domain == "code":
            body["temperature"] = min(body.get("temperature", 0.7), 0.3)
        elif domain == "reasoning":
            body["temperature"] = min(body.get("temperature", 0.7), 0.4)
        elif domain == "business":
            body["temperature"] = min(body.get("temperature", 0.7), 0.5)
            body["max_tokens"] = min(body.get("max_tokens", 1024), 512)

        # 5. Routing mode — Fast / Think / Architect
        extra = body.get("extra_body", {})
        mode = extra.get("mode", None)
        if mode == "think":
            body["temperature"] = min(body.get("temperature", 0.7), 0.4)
        elif mode == "architect":
            body["temperature"] = min(body.get("temperature", 0.7), 0.3)

        # 6. Context enrichment — rerank + embed provided chunks or URL fetches
        context_docs = extra.get("context_docs", [])
        if context_docs:
            chunks: List[str] = []
            for doc in context_docs:
                if isinstance(doc, dict):
                    if doc.get("url"):
                        try:
                            async with httpx.AsyncClient(timeout=10) as c:
                                r = await c.get(doc["url"])
                                if r.status_code == 200:
                                    chunks.append(r.text[:4000])
                        except Exception as e:
                            logger.warning("url_fetch_failed", url=doc["url"], error=str(e))
                    elif doc.get("chunk"):
                        chunks.append(str(doc["chunk"])[:2000])
                    elif doc.get("text"):
                        chunks.append(str(doc["text"])[:2000])
                    elif doc.get("content"):
                        chunks.append(str(doc["content"])[:2000])
                elif isinstance(doc, str):
                    chunks.append(doc[:2000])

            if chunks:
                if self._reranker is None:
                    self._reranker = LMStudioReranker(base_url=self.lmstudio_base, model_id=RERANK_MODEL)
                try:
                    user_prompt = next(
                        (m["content"] for m in body.get("messages", []) if m.get("role") == "user"),
                        ""
                    )
                    reranked = await self._reranker.rerank(user_prompt, chunks, top_k=RERANK_TOP_K)
                    if reranked:
                        ctx_text = "\n---\n".join(
                            f"[Context {i+1}] {r['chunk']}" for i, r in enumerate(reranked)
                        )
                        body["messages"].insert(
                            1,
                            {"role": "system", "content": f"Relevant context:\n{ctx_text}"}
                        )
                        logger.info("context_reranked", count=len(reranked))
                except Exception as e:
                    logger.warning("rerank_failed", error=str(e))

        return body
                
proxy = OptimizingProxy()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await proxy.get_hardware()
    asyncio.create_task(proxy.tuner.monitor_and_tune())

    hw = proxy.hardware
    lm = proxy.lmstudio_base
    loaded = proxy.tuner.last_stats.get("loaded_models", [])

    logger.info("proxy_startup_complete")
    logger.info("proxy_diagnostics",
                lmstudio=lm,
                gpu=hw.gpu_name or "none",
                vram_gb=hw.gpu_vram_gb or "unknown",
                cuda=hw.cuda_version or "none",
                platform=hw.platform,
                cpu_cores=hw.cpu_cores,
                ram_gb=hw.system_ram_gb,
                loaded_models=loaded)
    yield
    logger.info("proxy_shutdown_complete")

app = FastAPI(title="LM Studio Optimization Proxy (V4.1)", lifespan=lifespan)

@app.middleware("http")
async def request_logger(request: Request, call_next):
    handle_request(request)
    return await call_next(request)

@app.get("/api/v1/models")
async def list_models_native():
    """Native LM Studio v1 model list — no caching, always fresh for tuner."""
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            resp = await client.get(f"{proxy.lmstudio_base}/api/v1/models")
            return resp.json()
        except Exception as e:
            logger.error("list_models_failed", error=str(e))
            raise HTTPException(status_code=502, detail=f"Failed to reach LM Studio: {str(e)}")

@app.get("/api/v1/model-stats")
async def get_model_stats():
    """Per-model VRAM/ctx stats from LM Studio's native API."""
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            resp = await client.get(f"{proxy.lmstudio_base}/api/v1/models")
            data = resp.json()
            models = data.get("data", []) if isinstance(data, dict) else data
            summary = []
            for m in models:
                summary.append({
                    "id": m.get("id", "unknown"),
                    "state": m.get("state", "unknown"),
                    "size": m.get("size", "unknown"),
                })
            return {"models": summary}
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

@app.post("/api/v1/embed")
async def embed_texts(body: dict):
    """
    Generate embeddings for a list of texts.
    Uses LM Studio's embedding model if available, otherwise nomic-embed-text via /v1/embeddings.
    """
    texts = body.get("texts", [])
    model = body.get("model", "nomic-embed-text")
    if not texts:
        return {"embeddings": []}

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.post(
                f"{proxy.lmstudio_base}/v1/embeddings",
                json={"input": texts, "model": model}
            )
            if resp.status_code != 200:
                return {"embeddings": [], "error": resp.text}
            result = resp.json()
            embeddings = result.get("data", result)
            return {"embeddings": embeddings}
        except Exception as e:
            logger.error("embed_failed", error=str(e))
            return {"embeddings": [], "error": str(e)}

@app.post("/api/v1/rerank")
async def rerank(body: dict):
    """
    Rerank context chunks by semantic relevance to a query.
    Embeds query + chunks, scores by cosine similarity, returns reordered list.
    """
    query = body.get("query", "")
    chunks = body.get("chunks", [])
    top_k = body.get("top_k", 3)
    embed_model = body.get("embed_model", "nomic-embed-text")

    if not query or not chunks:
        return {"reranked": chunks}

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            embed_resp = await client.post(
                f"{proxy.lmstudio_base}/v1/embeddings",
                json={"input": [query] + chunks, "model": embed_model}
            )
            if embed_resp.status_code != 200:
                return {"reranked": chunks[:top_k]}
            data = embed_resp.json()
            embedding_list = data.get("data", data)
            if isinstance(embedding_list, list) and len(embedding_list) > 1:
                query_emb = embedding_list[0].get("embedding", []) if isinstance(embedding_list[0], dict) else []
                chunk_embs = embedding_list[1:]
                scored = []
                for i, emb_item in enumerate(chunk_embs):
                    emb = emb_item.get("embedding", []) if isinstance(emb_item, dict) else []
                    score = sum(a * b for a, b in zip(query_emb, emb)) if query_emb and emb else 0
                    scored.append((score, i, chunks[i]))
                scored.sort(reverse=True)
                reranked = [c for _, _, c in scored[:top_k]]
                return {"reranked": reranked, "scores": [s for s, _, _ in scored[:top_k]]}
            return {"reranked": chunks[:top_k]}
        except Exception as e:
            logger.error("rerank_failed", error=str(e))
            return {"reranked": chunks[:top_k]}
async def list_models():
    """Proxy model listing to LM Studio with short-lived TTL cache"""
    now = time.time()
    if proxy.cache.model_list_cache and now < proxy.cache.model_list_expiry:
        return proxy.cache.model_list_cache
        
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{proxy.lmstudio_base}/api/v1/models")
            if resp.status_code != 200:
                resp = await client.get(f"{proxy.lmstudio_base}/v1/models")
            
            data = resp.json()
            proxy.cache.model_list_cache = data
            proxy.cache.model_list_expiry = now + 2.0
            return data
        except Exception as e:
            logger.error("list_models_failed", error=str(e))
            raise HTTPException(status_code=502, detail=f"Failed to reach LM Studio: {str(e)}")

@app.post("/v1/chat/completions")
async def proxy_chat(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()
    is_streaming = body.get("stream", False)
    
    # Detect domain
    user_content = ""
    for msg in body.get("messages", []):
        if msg.get("role") == "user":
            user_content += msg.get("content", "")
    domain = proxy.router.detect_domain(user_content)
    
    # Check response cache
    cache_key = proxy.cache.build_cache_key(body)
    if not is_streaming:
        cached = proxy.cache.response_cache.get(cache_key)
        if cached:
            logger.info("cache_hit", domain=domain)
            return cached
    
    # Optimize
    optimized_body = await proxy.optimize_request(body, domain)
 
    # Determine routing mode (from extra_body merged into body by OpenAI SDK)
    mode = body.pop("mode", None) or body.pop("extra_body", {}).get("mode", None)
    if mode == "think":
        optimized_body["temperature"] = min(optimized_body.get("temperature", 0.7), 0.4)
    elif mode == "architect":
        optimized_body["temperature"] = min(optimized_body.get("temperature", 0.7), 0.3)
        optimized_body["max_tokens"] = min(optimized_body.get("max_tokens", 2048), 8192)

    async def _parse_sse_event(line: str):
        line = line.strip()
        if line.startswith("event: "):
            return line[7:], None
        if line.startswith("data: "):
            return None, line[6:].strip()
        return None, None

    async def stream_response():
        full_chunks = []
        reasoning_chunks = []
        current_event = None

        async with httpx.AsyncClient(timeout=None) as client:
            try:
                # We forward tool-related parameters seamlessly to LM Studio
                # structured outputs (response_format) are also passed transparently.
                async with client.stream(
                    "POST",
                    f"{proxy.lmstudio_base}/v1/chat/completions",
                    json=optimized_body,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    async for raw_line in response.aiter_lines():
                        if not raw_line:
                            continue

                        stripped = raw_line.strip()
                        if not stripped:
                            continue

                        if stripped == "[DONE]":
                            break

                        if stripped.startswith("event: "):
                            current_event = stripped[7:].strip()
                            continue

                        if stripped.startswith("data: "):
                            data_str = stripped[6:].strip()
                            if not data_str:
                                current_event = None
                                continue

                            if data_str == "[DONE]":
                                break

                            try:
                                data = json.loads(data_str)
                            except json.JSONDecodeError:
                                current_event = None
                                continue

                            if not isinstance(data, dict):
                                current_event = None
                                continue

                            # Standard OpenAI stream behavior (no `event:` line emitted by LM Studio for these)
                            if current_event is None and "choices" in data:
                                choices = data.get("choices")
                                if choices and len(choices) > 0:
                                    delta = choices[0].get("delta", {})
                                    if isinstance(delta, dict):
                                        if delta.get("content") is not None:
                                            content = delta["content"]
                                            full_chunks.append(content)
                                            yield f"data: {json.dumps({'choices':[{'delta':{'content': content}}]})}\n".encode()
                                        if delta.get("reasoning_content") is not None:
                                            rc = delta["reasoning_content"]
                                            reasoning_chunks.append(rc)
                                            yield f"data: {json.dumps({'choices':[{'delta':{'reasoning_content': rc}}]})}\n".encode()
                                        if delta.get("tool_calls"):
                                            yield f"data: {json.dumps({'choices':[{'delta':{'tool_calls': delta['tool_calls']}}]})}\n".encode()
                                current_event = None
                                continue

                            # Route based on the precise LM Studio event type
                            if current_event == "chat.start":
                                # Nothing to yield specifically, but we could initialize state
                                pass

                            elif current_event == "message.delta":
                                choices = data.get("choices")
                                if choices and len(choices) > 0:
                                    delta = choices[0].get("delta", {})
                                    if isinstance(delta, dict):
                                        if delta.get("content") is not None:
                                            content = delta["content"]
                                            full_chunks.append(content)
                                            yield f"data: {json.dumps({'choices':[{'delta':{'content': content}}]})}\n".encode()
                                        if delta.get("reasoning_content") is not None:
                                            rc = delta["reasoning_content"]
                                            reasoning_chunks.append(rc)
                                            yield f"data: {json.dumps({'choices':[{'delta':{'reasoning_content': rc}}]})}\n".encode()
                                        if delta.get("tool_calls"):
                                            # Forward tool call deltas as they are
                                            yield f"data: {json.dumps({'choices':[{'delta':{'tool_calls': delta['tool_calls']}}]})}\n".encode()

                            elif current_event == "reasoning.delta":
                                # Native LM Studio reasoning format
                                rc = data.get("content", "")
                                if rc:
                                    reasoning_chunks.append(rc)
                                    yield f"data: {json.dumps({'choices':[{'delta':{'reasoning_content': rc}}]})}\n".encode()

                            elif current_event in ("tool_call.start", "tool_call.arguments", "tool_call.success", "tool_call.failure"):
                                # If using native LM Studio events instead of OpenAI format
                                # For OpenAI compatibility, we wrap these into tool_calls format
                                # Note: Ideally LM Studio /v1/chat/completions sends standard tool_calls
                                # This handles potential leakage of native events
                                pass

                            elif current_event == "chat.end":
                                break

                            current_event = None

            except Exception as e:
                logger.error("streaming_failed", error=str(e))
                yield json.dumps({"error": {"message": str(e), "type": "proxy_error"}}).encode()
                return

        yield b"data: [DONE]\n"
        
        # We don't cache tool calls or complex structured output yet, just simple text
        if not any(t in optimized_body for t in ["tools", "response_format"]):
            final_content = "".join(full_chunks)
            if final_content:
                formatted = proxy.router.format_output(final_content, domain)
                proxy.cache.response_cache.put(cache_key, {"content": formatted})

    if is_streaming:
        return StreamingResponse(stream_response(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            # Note: For non-streaming requests, LM Studio responds with the full payload
            # We must transparently pass through the response, including tools and structured outputs
            response = await client.post(
                f"{proxy.lmstudio_base}/v1/chat/completions",
                json=optimized_body,
                headers={"Content-Type": "application/json"}
            )
            if response.status_code != 200:
                return response.json()

            result = response.json()
            choices = result.get("choices") if isinstance(result, dict) else None
            if choices and len(choices) > 0:
                msg = choices[0].get("message", {})
                
                # Format standard text content if present
                content = msg.get("content", "") if isinstance(msg, dict) else ""
                if content and isinstance(content, str):
                    formatted = proxy.router.format_output(content, domain)
                    if isinstance(msg, dict):
                        msg["content"] = formatted
                
                # We don't modify tool_calls or structured output, we just pass them through
                proxy.cache.response_cache.put(cache_key, result)
            return result
        except Exception as e:
            logger.error("chat_proxy_failed", error=str(e))
            raise HTTPException(status_code=502, detail=f"LM Studio connection failed: {str(e)}")

@app.get("/api/v1/hardware")
async def get_hardware_status():
    hw = await proxy.get_hardware()
    return hw.model_dump()

@app.post("/api/v1/benchmark/{model_id}")
async def benchmark_model(model_id: str, background_tasks: BackgroundTasks):
    """Run comprehensive benchmark through proxy (async background task)"""
    async def run_benchmark():
        bench = StreamingBenchmark(
            base_url=proxy.lmstudio_base,  # Test LM Studio directly
            proxy_url=f"http://0.0.0.0:8080"  # Or test through self
        )
        results = await bench.run_full_benchmark(model_id)
        # Store results somewhere or log them
        logger.info("benchmark_completed", model=model_id, results=results)
    
    # Run in background so API doesn't hang
    background_tasks.add_task(run_benchmark)
    return {"status": "benchmark_started", "model": model_id}

@app.get("/api/v1/benchmark/compare")
async def compare_modes(model_id: str = "qwen3.5-4b"):
    """Quick streaming vs non-streaming comparison"""
    bench = StreamingBenchmark(base_url=proxy.lmstudio_base)
    comparison = await bench.compare_streaming_vs_nonstreaming(model_id)
    return comparison

@app.post("/api/v1/presets/generate")
async def generate_preset(
    model_id: str, 
    use_case: str = "balanced",
    params_b: float = 4.0
):
    """
    Generate optimized JSON preset for LM Studio based on detected hardware
    """
    hw = await proxy.get_hardware()
    
    if not proxy.engine:
        proxy.engine = RecommendationEngine(hw)
    
    # Get recommendations
    recommendations = proxy.engine.recommend(
        model_id=model_id,
        params_b=params_b,
        use_case=use_case
    )
    
    if not recommendations:
        raise HTTPException(status_code=400, detail="No valid configuration for this hardware")
    
    top_rec = recommendations[0]
    
    # Generate preset JSON (LM Studio format)
    preset = {
        "identifier": f"@local:{model_id.replace('/', '-')}-{use_case}",
        "name": f"{model_id} - {use_case.title()} (Auto)",
        "changed": True,
        "importedTimeStamp": int(time.time() * 1000),
        "operation": {
            "fields": [
                {"key": "llm.prediction.temperature", "value": top_rec.temperature},
                {"key": "llm.prediction.topPSampling", "value": {"checked": True, "value": top_rec.top_p}},
                {"key": "llm.prediction.topKSampling", "value": {"checked": True, "value": top_rec.top_k}},
                {"key": "llm.prediction.repeatPenalty", "value": {"checked": True, "value": top_rec.repeat_penalty}},
            ]
        },
        "load": {
            "fields": [
                {"key": "llm.load.contextLength", "value": top_rec.context_length},
                {"key": "llm.load.gpuOffload", "value": top_rec.gpu_layers},
                {"key": "llm.load.flashAttention", "value": top_rec.flash_attention and (hw.cuda_compute or 0) >= 7.5},
                {"key": "llm.load.useMmap", "value": top_rec.use_mmap},
            ]
        },
        "hardware": {
            "gpu": hw.gpu_name,
            "vram_gb": hw.gpu_vram_gb,
            "compute": hw.cuda_compute,
            "is_maxwell": hw.cuda_compute is not None and hw.cuda_compute < 6.0
        },
        "recommendation": {
            "quality_score": top_rec.quality_score,
            "estimated_vram_gb": top_rec.estimated_vram_gb,
            "backend": top_rec.inference_backend.value,
            "quantization": top_rec.quantization.value
        }
    }
    
    # Save to presets folder
    presets_dir = Path.home() / ".cache" / "lm-studio" / "presets"
    presets_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{model_id.replace('/', '_')}_{use_case}.json"
    filepath = presets_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(preset, f, indent=2)
    
    return {
        "preset": preset,
        "saved_to": str(filepath),
        "message": f"Preset saved. Import in LM Studio via 'Load Preset'"
    }

@app.get("/api/v1/presets/list")
async def list_presets():
    """List all generated presets"""
    presets_dir = Path.home() / ".cache" / "lm-studio" / "presets"
    if not presets_dir.exists():
        return {"presets": []}
    
    presets = []
    for f in presets_dir.glob("*.json"):
        presets.append({
            "name": f.stem,
            "path": str(f),
            "modified": f.stat().st_mtime
        })
    
    return {"presets": sorted(presets, key=lambda x: x["modified"], reverse=True)}

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("BRIDGE_HOST", "0.0.0.0")
    port = int(os.getenv("BRIDGE_PORT", "8080"))
    uvicorn.run(app, host=host, port=port)
