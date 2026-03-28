from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from benchmark import StreamingBenchmark
from contextlib import asynccontextmanager
import httpx
import asyncio
import time
import json
import re
import sys
from typing import Dict, Any, Optional
from pathlib import Path
import os
import uuid
from dotenv import load_dotenv
from urllib.parse import urlparse

load_dotenv()

from hardware_detector import HardwareDetector
from engine import RecommendationEngine
from models import HardwareProfile
from tuner import AdaptiveTuner
from cache_manager import CacheManager
from domain_router import DomainRouter
from context_manager import ContextManager
from lmstudio_bridge import LMStudioBridge, RetrievalConfig, ChunkingConfig
from exporters import ConfigExporter
from shared.ace import ACEOrchestrator, ACESessionStore
from shared.agent_orchestrator import AgentOrchestrator
from shared.agent_state import AgentSessionStore
from shared.agent_tools import AgentToolRegistry
from datetime import datetime
from logger_config import setup_logger, setup_structlog, log_api_error, error_to_dict

# Setup centralized structlog for keyword argument support
logger = setup_structlog("proxy")


def _clean_env_model(value: str, default: str) -> str:
    text = str(value or default).strip()
    match = re.search(r"model_key='([^']+)'", text)
    if match:
        return match.group(1)
    return text


def _clean_bridge_host(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "127.0.0.1"
    if "://" in text:
        parsed = urlparse(text)
        return parsed.hostname or "127.0.0.1"
    return text.split(":", 1)[0] or "127.0.0.1"


def _clean_bridge_port(value: str, default: int = 8080) -> int:
    text = str(value or "").strip()
    if not text:
        return default
    if "://" in text:
        parsed = urlparse(text)
        return parsed.port or default
    try:
        return int(text)
    except Exception:
        try:
            return int(text.rsplit(":", 1)[1])
        except Exception:
            return default

EMBED_MODEL = _clean_env_model(os.getenv("EMBED_MODEL"), "text-embedding-qwen3-embedding-4b")
RERANK_MODEL = _clean_env_model(os.getenv("RERANK_MODEL"), "qwen.qwen3-reranker-4b")
MAIN_MODEL = _clean_env_model(os.getenv("MAIN_MODEL"), "qwen3.5-4b")
REASONING_MODEL = _clean_env_model(os.getenv("REASONING_MODEL"), "qwen3.5-4b")
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "900"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "150"))
AUTO_LOAD_MODELS = os.getenv("AUTO_LOAD_MODELS", "true").lower() in {"1", "true", "yes", "on"}


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
        self.bridge = LMStudioBridge(
            base_url=self.lmstudio_base,
            embed_model=EMBED_MODEL,
            rerank_model=RERANK_MODEL,
            auto_load_models=AUTO_LOAD_MODELS,
        )
        
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
        low_resource = vram <= 8.5 or (hw.system_ram_gb or 0) <= 16.5
        
        body["max_tokens"] = min(body.get("max_tokens", 2048), max_ctx)
        if low_resource:
            body["max_tokens"] = min(body.get("max_tokens", 1024), 1024)

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
        retrieval_cfg = extra.get("retrieval", {}) if isinstance(extra, dict) else {}
        if isinstance(retrieval_cfg, dict) and low_resource:
            retrieval_cfg["top_k"] = min(int(retrieval_cfg.get("top_k", RERANK_TOP_K)), 3)
            retrieval_cfg["max_context_chars"] = min(int(retrieval_cfg.get("max_context_chars", 3200)), 2400)
            retrieval_cfg["max_chunks"] = min(int(retrieval_cfg.get("max_chunks", 64)), 24)
            retrieval_cfg["chunk_size"] = min(int(retrieval_cfg.get("chunk_size", DEFAULT_CHUNK_SIZE)), 700)
            retrieval_cfg["chunk_overlap"] = min(int(retrieval_cfg.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)), 100)
            extra["retrieval"] = retrieval_cfg
        mode = extra.get("mode", None)
        if mode == "think" or domain == "reasoning":
            body["temperature"] = min(body.get("temperature", 0.7), 0.4)
            if body.get("model") == "default" or body.get("model") == MAIN_MODEL:
                body["model"] = REASONING_MODEL
        elif mode == "architect":
            body["temperature"] = min(body.get("temperature", 0.7), 0.3)
            if low_resource:
                extra["mode"] = "fast"
            
        if body.get("model") == "default":
            body["model"] = MAIN_MODEL
        if body.get("model"):
            body["model"] = await self.bridge.resolve_model_id(body["model"])

        # 6. Context enrichment and model orchestration through the bridge
        try:
            body, bridge_meta = await self.bridge.enrich_chat_body(
                body,
                top_k=RERANK_TOP_K,
                default_chunk_size=DEFAULT_CHUNK_SIZE,
                default_chunk_overlap=DEFAULT_CHUNK_OVERLAP,
            )
            retrieval = bridge_meta.get("retrieval") if isinstance(bridge_meta, dict) else None
            if retrieval and retrieval.get("chunks"):
                logger.info(
                    "context_reranked",
                    count=len(retrieval["chunks"]),
                    sources=retrieval.get("sources", []),
                )
        except Exception as e:
            logger.warning("bridge_enrichment_failed", error=str(e))

        return body
                
proxy = OptimizingProxy()
BRIDGE_BASE_URL = f"http://{_clean_bridge_host(os.getenv('BRIDGE_HOST', '127.0.0.1'))}:{_clean_bridge_port(os.getenv('BRIDGE_PORT', '8080'))}"
agent_session_store = AgentSessionStore(Path(".gui_state") / "agent_sessions")
ace_session_store = ACESessionStore(Path(".gui_state") / "ace_sessions")
agent_tool_registry = AgentToolRegistry(root_dir=Path.cwd(), bridge_base=BRIDGE_BASE_URL)


async def _agent_llm_generate(model_id: str, messages: list[dict], options: Dict[str, Any]) -> Dict[str, Any]:
    body = {
        "model": model_id or MAIN_MODEL,
        "messages": messages,
        "stream": False,
        "max_tokens": 700,
        "temperature": 0.2,
        "extra_body": {"mode": options.get("mode", "fast")},
    }
    user_content = "\n".join(msg.get("content", "") for msg in messages if msg.get("role") == "user")
    domain = proxy.router.detect_domain(user_content)
    optimized_body = await proxy.optimize_request(body, domain)
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(
            f"{proxy.lmstudio_base}/v1/chat/completions",
            json=optimized_body,
            headers={"Content-Type": "application/json"},
        )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        result = response.json()
    choices = result.get("choices") if isinstance(result, dict) else None
    message = choices[0].get("message", {}) if choices else {}
    content = message.get("content", "") if isinstance(message, dict) else ""
    reasoning = message.get("reasoning_content", "") if isinstance(message, dict) else ""
    return {
        "content": content,
        "reasoning_content": reasoning,
        "raw": result,
        "prompt_stats": {
            "original_chars": sum(len(str(item.get("content", ""))) for item in messages),
            "condensed_chars": sum(len(str(item.get("content", ""))) for item in messages),
            "saved_chars": 0,
            "saved_percent": 0.0,
        },
    }


agent_orchestrator = AgentOrchestrator(
    session_store=agent_session_store,
    tool_registry=agent_tool_registry,
    llm_generate=_agent_llm_generate,
    workspace_root=Path.cwd(),
)


async def _ace_prepare_body(body: Dict[str, Any], domain: str) -> Dict[str, Any]:
    return await proxy.optimize_request(body, domain)


ace_orchestrator = ACEOrchestrator(
    session_store=ace_session_store,
    tool_registry=agent_tool_registry,
    lmstudio_base=proxy.lmstudio_base,
    prepare_body=_ace_prepare_body,
)

async def watch_env_file():
    env_path = Path(".env")
    last_mtime = 0
    if env_path.exists():
        last_mtime = env_path.stat().st_mtime
        
    while True:
        await asyncio.sleep(2)
        if env_path.exists():
            current_mtime = env_path.stat().st_mtime
            if current_mtime > last_mtime:
                last_mtime = current_mtime
                logger.info("env_file_changed", action="reloading_config")
                load_dotenv(override=True)
                
                global MAIN_MODEL, REASONING_MODEL, EMBED_MODEL, RERANK_MODEL, RERANK_TOP_K
                MAIN_MODEL = _clean_env_model(os.getenv("MAIN_MODEL"), MAIN_MODEL)
                REASONING_MODEL = _clean_env_model(os.getenv("REASONING_MODEL"), REASONING_MODEL)
                EMBED_MODEL = _clean_env_model(os.getenv("EMBED_MODEL"), EMBED_MODEL)
                RERANK_MODEL = _clean_env_model(os.getenv("RERANK_MODEL"), RERANK_MODEL)
                RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", str(RERANK_TOP_K)))
                
                # Update bridge explicitly
                proxy.bridge.embed_model = EMBED_MODEL
                proxy.bridge.rerank_model = RERANK_MODEL
                
                logger.info("config_reloaded", main=MAIN_MODEL, reasoning=REASONING_MODEL, embed=EMBED_MODEL, rerank=RERANK_MODEL)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await proxy.get_hardware()
    asyncio.create_task(proxy.tuner.monitor_and_tune())
    asyncio.create_task(watch_env_file())

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

# Configure CORS to allow requests from Flask dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8090", "http://127.0.0.1:8090", "http://localhost", "http://127.0.0.1"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def request_logger(request: Request, call_next):
    handle_request(request)
    return await call_next(request)

# OpenAI-compatible standard endpoints (v1/ prefix)
@app.get("/v1/models")
async def list_models_v1():
    """OpenAI-compatible v1 model list endpoint."""
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            resp = await client.get(f"{proxy.lmstudio_base}/api/v1/models")
            data = resp.json()
            return data
        except Exception as e:
            logger.error("list_models_failed", error=str(e))
            raise HTTPException(status_code=502, detail=f"Failed to reach LM Studio: {str(e)}")

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
    texts = body.get("texts", [])
    model = body.get("model", EMBED_MODEL)
    if not texts:
        return {"embeddings": []}
    try:
        embeddings = await proxy.bridge.embed_texts(texts, model=model)
        return {"embeddings": embeddings, "model": model}
    except Exception as e:
        logger.error("embed_failed", error=str(e))
        return {"embeddings": [], "error": str(e)}

@app.post("/v1/embeddings")
async def embed_texts_v1(body: dict):
    """OpenAI-compatible embeddings endpoint."""
    input_data = body.get("input", [])
    model = body.get("model", EMBED_MODEL)
    
    # Normalize input to list of strings
    if isinstance(input_data, str):
        texts = [input_data]
    else:
        texts = input_data if isinstance(input_data, list) else [str(input_data)]
    
    if not texts:
        return {"data": [], "model": model, "usage": {"prompt_tokens": 0, "total_tokens": 0}}
    
    try:
        embeddings = await proxy.bridge.embed_texts(texts, model=model)
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "index": i, "embedding": emb}
                for i, emb in enumerate(embeddings)
            ],
            "model": model,
            "usage": {
                "prompt_tokens": sum(len(t.split()) for t in texts),
                "total_tokens": sum(len(t.split()) for t in texts)
            }
        }
    except Exception as e:
        logger.error("embed_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/rerank")
async def rerank(body: dict):
    query = body.get("query", "")
    chunks = body.get("chunks", [])
    top_k = body.get("top_k", 3)

    if not query or not chunks:
        return {"reranked": chunks}

    try:
        reranked = await proxy.bridge.rerank_chunks(query, chunks, top_k=top_k)
        return {
            "reranked": [item["chunk"] for item in reranked],
            "results": reranked,
            "scores": [item["score"] for item in reranked],
        }
    except Exception as e:
        logger.error("rerank_failed", error=str(e))
        return {"reranked": chunks[:top_k], "error": str(e)}

@app.post("/api/v1/retrieve")
async def retrieve_context(body: dict):
    query = body.get("query", "")
    docs = body.get("docs", []) or body.get("context_docs", [])
    retrieval = body.get("retrieval", {})
    if not query or not docs:
        return {"chunks": [], "context_text": "", "sources": []}

    config = RetrievalConfig(
        top_k=int(retrieval.get("top_k", RERANK_TOP_K)),
        chunking=ChunkingConfig(
            chunk_size=int(retrieval.get("chunk_size", DEFAULT_CHUNK_SIZE)),
            chunk_overlap=int(retrieval.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)),
            max_chunks=int(retrieval.get("max_chunks", 64)),
            max_chunk_chars=int(retrieval.get("max_chunk_chars", 1600)),
        ),
        include_sources=bool(retrieval.get("include_sources", True)),
        max_context_chars=int(retrieval.get("max_context_chars", 6000)),
    )
    try:
        result = await proxy.bridge.build_retrieval_context(query, docs, config)
        return result
    except Exception as e:
        logger.error("retrieve_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/models/load")
async def load_model(body: dict):
    model_id = body.get("model") or body.get("model_id")
    if not model_id:
        raise HTTPException(status_code=400, detail="Missing model or model_id")
    try:
        resolved_model_id = await proxy.bridge.resolve_model_id(model_id)
        result = await proxy.bridge.load_model(
            resolved_model_id,
            context_length=body.get("context_length"),
            identifier=body.get("identifier"),
            gpu=body.get("gpu"),
            ttl=body.get("ttl"),
            eval_batch_size=body.get("eval_batch_size"),
            flash_attention=body.get("flash_attention"),
        )
        return {"status": "loaded", "model": resolved_model_id, "result": result}
    except Exception as e:
        logger.error("model_load_failed", model=model_id, error=str(e))
        raise HTTPException(status_code=502, detail=f"Failed to load model: {str(e)}")


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type in {"input_text", "output_text", "text"}:
                parts.append(item.get("text", ""))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        return str(content.get("text", ""))
    return ""


def _responses_input_to_messages(body: Dict[str, Any]) -> list:
    messages = []
    instructions = body.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": instructions})

    response_input = body.get("input", "")
    if isinstance(response_input, str):
        if response_input.strip():
            messages.append({"role": "user", "content": response_input})
        return messages

    if not isinstance(response_input, list):
        return messages

    for item in response_input:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            role = item.get("role", "user")
            text = _content_to_text(item.get("content", ""))
            if text:
                messages.append({"role": role, "content": text})
            continue
        if item.get("role"):
            text = _content_to_text(item.get("content", ""))
            if text:
                messages.append({"role": item["role"], "content": text})
            continue
        if item.get("type") in {"input_text", "text"}:
            text = item.get("text", "")
            if text:
                messages.append({"role": "user", "content": text})

    return messages


def _response_to_completion_body(body: Dict[str, Any]) -> Dict[str, Any]:
    extra_body = dict(body.get("extra_body", {}))
    for key in ("context_docs", "retrieval", "load_model", "mode", "embed_model", "rerank_model"):
        if body.get(key) is not None:
            extra_body[key] = body.get(key)

    reasoning = body.get("reasoning", {})
    if isinstance(reasoning, dict) and reasoning.get("effort") in {"medium", "high"} and "mode" not in extra_body:
        extra_body["mode"] = "think"

    completion_body: Dict[str, Any] = {
        "model": body.get("model", MAIN_MODEL),
        "messages": _responses_input_to_messages(body),
        "stream": bool(body.get("stream", False)),
        "extra_body": extra_body,
    }

    if body.get("max_output_tokens") is not None:
        completion_body["max_tokens"] = body["max_output_tokens"]
    if body.get("temperature") is not None:
        completion_body["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        completion_body["top_p"] = body["top_p"]
    if body.get("top_k") is not None:
        completion_body["top_k"] = body["top_k"]
    if body.get("repeat_penalty") is not None:
        completion_body["repeat_penalty"] = body["repeat_penalty"]
    if body.get("tools") is not None:
        completion_body["tools"] = body["tools"]
    if body.get("tool_choice") is not None:
        completion_body["tool_choice"] = body["tool_choice"]
    if body.get("response_format") is not None:
        completion_body["response_format"] = body["response_format"]

    return completion_body


def _completion_to_response(result: Dict[str, Any], request_body: Dict[str, Any], retrieval_sources: Optional[list] = None) -> Dict[str, Any]:
    created_at = int(time.time())
    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    choices = result.get("choices", []) if isinstance(result, dict) else []
    message = choices[0].get("message", {}) if choices else {}
    content = message.get("content", "") if isinstance(message, dict) else ""
    reasoning_content = message.get("reasoning_content", "") if isinstance(message, dict) else ""
    output_items = []
    content_items = []

    if reasoning_content:
        content_items.append({
            "type": "reasoning",
            "text": reasoning_content,
        })
    if content:
        content_items.append({
            "type": "output_text",
            "text": content,
            "annotations": [],
        })
    if content_items:
        output_items.append({
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": content_items,
        })

    response = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "model": request_body.get("model", MAIN_MODEL),
        "output": output_items,
        "output_text": content,
        "usage": result.get("usage"),
        "metadata": {
            "retrieval_sources": retrieval_sources or [],
            "lmstudio_raw_id": result.get("id"),
        },
    }
    if message.get("tool_calls"):
        response["tool_calls"] = message["tool_calls"]
    return response


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


@app.post("/v1/responses")
async def proxy_responses(request: Request):
    body = await request.json()
    if body.get("stream"):
        raise HTTPException(status_code=501, detail="Streaming /v1/responses is not implemented yet. Use /v1/chat/completions for streaming.")

    completion_body = _response_to_completion_body(body)
    if not completion_body.get("messages"):
        raise HTTPException(status_code=400, detail="Responses input did not produce any messages.")

    user_content = "\n".join(
        msg.get("content", "")
        for msg in completion_body.get("messages", [])
        if msg.get("role") == "user"
    )
    domain = proxy.router.detect_domain(user_content)
    optimized_body = await proxy.optimize_request(completion_body, domain)

    retrieval_sources = []
    extra_body = optimized_body.get("extra_body", {})
    if isinstance(extra_body, dict):
        retrieval_cfg = extra_body.get("retrieval", {})
        docs = extra_body.get("context_docs", [])
        if docs and user_content:
            try:
                retrieval_result = await proxy.bridge.build_retrieval_context(
                    user_content,
                    docs,
                    RetrievalConfig(
                        top_k=int(retrieval_cfg.get("top_k", RERANK_TOP_K)),
                        chunking=ChunkingConfig(
                            chunk_size=int(retrieval_cfg.get("chunk_size", DEFAULT_CHUNK_SIZE)),
                            chunk_overlap=int(retrieval_cfg.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)),
                            max_chunks=int(retrieval_cfg.get("max_chunks", 64)),
                            max_chunk_chars=int(retrieval_cfg.get("max_chunk_chars", 1600)),
                        ),
                        include_sources=bool(retrieval_cfg.get("include_sources", True)),
                        max_context_chars=int(retrieval_cfg.get("max_context_chars", 6000)),
                    ),
                )
                retrieval_sources = retrieval_result.get("sources", [])
            except Exception as exc:
                logger.warning("responses_retrieval_metadata_failed", error=str(exc))

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            response = await client.post(
                f"{proxy.lmstudio_base}/v1/chat/completions",
                json=optimized_body,
                headers={"Content-Type": "application/json"},
            )
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            result = response.json()
            choices = result.get("choices") if isinstance(result, dict) else None
            if choices and len(choices) > 0:
                msg = choices[0].get("message", {})
                content = msg.get("content", "") if isinstance(msg, dict) else ""
                if content and isinstance(content, str):
                    msg["content"] = proxy.router.format_output(content, domain)
            return _completion_to_response(result, body, retrieval_sources=retrieval_sources)
        except HTTPException:
            raise
        except Exception as e:
            logger.error("responses_proxy_failed", error=str(e))
            raise HTTPException(status_code=502, detail=f"LM Studio connection failed: {str(e)}")

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


@app.get("/v1/agent/sessions")
async def list_agent_sessions():
    return {"sessions": agent_orchestrator.list_sessions()}


@app.post("/v1/agent/sessions")
async def create_agent_session(body: dict):
    workflow = str(body.get("workflow") or "coding_sprint")
    tool_budget = int(body.get("tool_budget") or 6)
    role_map = {
        "main": _clean_env_model(body.get("main_model"), MAIN_MODEL),
        "reasoning": _clean_env_model(body.get("reasoning_model"), REASONING_MODEL),
        "embed": _clean_env_model(body.get("embed_model"), EMBED_MODEL),
        "rerank": _clean_env_model(body.get("rerank_model"), RERANK_MODEL),
    }
    session = agent_orchestrator.create_session(
        workflow=workflow,
        role_map=role_map,
        cwd=str(body.get("cwd") or Path.cwd()),
        tool_budget=tool_budget,
    )
    return {"session_id": session.id, "status": "ready", "workflow": workflow}


@app.post("/v1/agent/sessions/{session_id}/turn")
async def agent_turn(session_id: str, body: dict):
    return await agent_orchestrator.execute_turn(session_id, body)


@app.post("/v1/agent/sessions/{session_id}/tool-result")
async def agent_tool_result(session_id: str, body: dict):
    return await agent_orchestrator.submit_tool_result(session_id, body.get("result") or body)


@app.get("/v1/agent/sessions/{session_id}/state")
async def agent_session_state(session_id: str):
    return agent_orchestrator.get_session_state(session_id)


@app.post("/v1/agent/sessions/{session_id}/checkpoint/{checkpoint_id}/restore")
async def restore_agent_checkpoint(session_id: str, checkpoint_id: str):
    return agent_orchestrator.restore_checkpoint(session_id, checkpoint_id)


@app.post("/v1/agent/sessions/{session_id}/branch")
async def branch_agent_session(session_id: str, body: dict):
    checkpoint_id = str(body.get("from_checkpoint") or body.get("checkpoint_id") or "")
    if not checkpoint_id:
        raise HTTPException(status_code=400, detail="Missing checkpoint id")
    return agent_orchestrator.branch_session(session_id, checkpoint_id)


@app.post("/v1/agent/sessions/{session_id}/turn/stream")
async def agent_turn_stream(session_id: str, body: dict):
    async def event_stream():
        try:
            async for event in agent_orchestrator.execute_turn_stream(session_id, body):
                yield f"event: {event['type']}\ndata: {json.dumps(event['data'])}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/v1/ace/sessions")
async def list_ace_sessions():
    return {"sessions": ace_orchestrator.list_sessions()}


@app.post("/v1/ace/generate")
async def ace_generate(body: dict):
    async def event_stream():
        try:
            async for event in ace_orchestrator.generate(body):
                payload = {"session_id": event["session_id"], "data": event["data"], "continue": event["continue"]}
                yield f"event: {event['type']}\ndata: {json.dumps(payload)}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"

    headers = {"X-ACE-Version": "v3-active-context"}
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


@app.post("/v1/ace/select-option")
async def ace_select_option(body: dict):
    session_id = str(body.get("session_id") or "")
    option_id = str(body.get("option_id") or "")
    if not session_id or not option_id:
        raise HTTPException(status_code=400, detail="session_id and option_id are required")
    selection = ace_orchestrator.record_selection(session_id, option_id, str(body.get("label") or ""))
    return {"status": "selection_recorded", "session_id": session_id, "selection": selection}


@app.get("/v1/ace/sessions/{session_id}/trace")
async def get_ace_trace(session_id: str):
    return ace_orchestrator.get_trace(session_id)

@app.post("/api/v1/benchmark/{model_id}")
async def benchmark_model(model_id: str, background_tasks: BackgroundTasks):
    """Run comprehensive benchmark through proxy (async background task)"""
    async def run_benchmark():
        bench = StreamingBenchmark(
            base_url=proxy.lmstudio_base,  # Test LM Studio directly
            proxy_url=f"http://127.0.0.1:8080"  # Or test through self
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
    
    low_resource = (hw.gpu_vram_gb or 0) <= 8.5 or (hw.system_ram_gb or 0) <= 16.5
    profile = {
        "mode": "think" if top_rec.enable_thinking and not low_resource else "fast",
        "embed_model": EMBED_MODEL,
        "rerank_model": RERANK_MODEL,
        "chunk_size": 700 if low_resource else 900,
        "chunk_overlap": 100 if low_resource else 150,
        "retrieval_top_k": 3 if low_resource else RERANK_TOP_K,
        "max_context_chars": 2400 if low_resource else 6000,
    }
    hardware_meta = {
        "gpu": hw.gpu_name,
        "vram_gb": hw.gpu_vram_gb,
        "compute": hw.cuda_compute,
        "ram_gb": hw.system_ram_gb,
        "is_maxwell": hw.cuda_compute is not None and hw.cuda_compute < 6.0,
    }
    preset = ConfigExporter.build_preset_dict(
        top_rec,
        profile=profile,
        hardware=hardware_meta,
        name=f"{model_id} - {use_case.title()} (Auto)",
        identifier=f"@local:{model_id.replace('/', '-')}-{use_case}",
    )
    
    # Save to presets folder
    presets_dir = Path.home() / ".cache" / "lm-studio" / "config-presets"
    presets_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{model_id.replace('/', '_')}_{use_case}.preset.json"
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
    host = _clean_bridge_host(os.getenv("BRIDGE_HOST", "127.0.0.1"))
    port = _clean_bridge_port(os.getenv("BRIDGE_PORT", "8080"))
    uvicorn.run(app, host=host, port=port)
