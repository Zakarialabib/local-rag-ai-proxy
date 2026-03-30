import asyncio
import httpx
import json
import os
import re
import structlog
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from embedder import Embedder, cosine_score
from reranker import LMStudioReranker

logger = structlog.get_logger()


# =============================================
# Types
# =============================================

def clean_bridge_host(host: str) -> str:
    """Extract hostname from potentially dirty host string or URL."""
    if not host:
        return "127.0.0.1"
    if "://" in host:
        return urlparse(host).hostname or "127.0.0.1"
    return host.split(":", 1)[0]


def clean_bridge_port(port_val: Any) -> int:
    """Extract port integer from potentially dirty string or URL."""
    if not port_val:
        return 8080
    text = str(port_val)
    if "://" in text:
        parsed = urlparse(text)
        return parsed.port or 8080
    try:
        if ":" in text:
            return int(text.rsplit(":", 1)[1])
        return int(text)
    except (ValueError, IndexError):
        return 8080


@dataclass
class ChunkingConfig:
    chunk_size: int = 900
    chunk_overlap: int = 150
    max_chunks: int = 64
    max_chunk_chars: int = 1600


@dataclass
class RetrievalConfig:
    top_k: int = 4
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    include_sources: bool = True
    max_context_chars: int = 6000
    rerank_instruction: str = "retrieval"
    embed_instruction: str = "retrieval"
    embed_dim: Optional[int] = None


# =============================================
# Main Class
# =============================================

class LMStudioBridge:
    def __init__(
        self,
        base_url: str,
        embed_model: str,
        rerank_model: str,
        auto_load_models: bool = True,
        request_timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.embed_model = embed_model
        self.rerank_model = rerank_model
        self.auto_load_models = auto_load_models
        self.request_timeout = request_timeout
        self.embedder = Embedder(base_url=self.base_url, model=self.embed_model)
        self.reranker = LMStudioReranker(base_url=self.base_url, model_id=self.rerank_model)

    async def resolve_model_id(self, model_id: str) -> str:
        """Resolve model_id string to actual model ID returned by LM Studio."""
        if not model_id:
            return model_id
        try:
            models = await self.list_models()
            normalized_target = self.normalize_model_token(self.extract_model_key(model_id))
            best_match = None
            for model in models:
                candidates = {
                    model.get("id"), model.get("key"), model.get("model_key"),
                    model.get("display_name"), model.get("name"),
                }
                for candidate in candidates:
                    if not candidate:
                        continue
                    normalized = self.normalize_model_token(candidate)
                    if not normalized:
                        continue
                    if normalized == normalized_target:
                        return str(model.get("id") or model.get("key") or model.get("model_key") or candidate)
                    if normalized_target in normalized or normalized in normalized_target:
                        best_match = str(model.get("id") or model.get("key") or model.get("model_key") or candidate)
            return best_match or self.extract_model_key(model_id)
        except Exception as e:
            logger.exception("resolve_model_id_failed", model=model_id, error=str(e))
            return model_id

    async def list_models(self) -> List[Dict[str, Any]]:
        """List all models available on LM Studio server."""
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{self.base_url}/api/v1/models")
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                return payload.get("data") or payload.get("models") or []
            return payload

    async def get_loaded_models(self) -> List[Dict[str, Any]]:
        """Get models that are currently loaded."""
        models = await self.list_models()
        loaded = []
        for model in models:
            if model.get("state") == "loaded":
                loaded.append(model)
                continue
            loaded_instances = model.get("loaded_instances")
            if isinstance(loaded_instances, list) and loaded_instances:
                enriched = dict(model)
                enriched["state"] = "loaded"
                loaded.append(enriched)
        return loaded

    async def is_model_loaded(self, model_id: str) -> bool:
        """Check if a model is loaded."""
        target = self.normalize_model_token(self.extract_model_key(model_id))
        for model in await self.get_loaded_models():
            candidates = {
                model.get("id"), model.get("key"), model.get("model_key"),
                model.get("display_name"), model.get("identifier"),
                model.get("model"), model.get("name"), model.get("path"),
            }
            normalized = {self.normalize_model_token(value) for value in candidates if value}
            if target in normalized or any(target and item and (target in item or item in target) for item in normalized):
                return True
        return False

    @staticmethod
    def extract_model_key(value: Optional[str]) -> str:
        """Extract the core model key from string formats."""
        text = str(value or "").strip()
        match = re.search(r"model_key='([^']+)'", text)
        if match:
            return match.group(1)
        return text

    @staticmethod
    def normalize_model_token(value: Optional[str]) -> str:
        """Normalize model identifier for comparison."""
        text = LMStudioBridge.extract_model_key(value).lower().replace("\\", "/")
        text = text.rsplit("/", 1)[-1]
        text = text.rsplit(":", 1)[-1]
        text = re.sub(r"[^a-z0-9._-]+", "/", text)
        return text

    @staticmethod
    def extract_text_from_content(content: Any) -> str:
        """Extract plain text from OpenAI-style content (string OR list of parts)."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict) and (item.get("type") == "text" or "text" in item)
            )
        return str(content or "")

    async def load_model(
        self,
        model_id: str,
        *,
        context_length: Optional[int] = None,
        identifier: Optional[str] = None,
        gpu: Optional[str] = None,
        ttl: Optional[int] = None,
        eval_batch_size: Optional[int] = None,
        flash_attention: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Load a model into LM Studio."""
        resolved_model_id = await self.resolve_model_id(model_id)
        body: Dict[str, Any] = {"model": resolved_model_id}
        if context_length:
            body["context_length"] = context_length
        if identifier:
            body["identifier"] = identifier
        if gpu:
            body["gpu"] = gpu
        if ttl:
            body["ttl"] = ttl
        if eval_batch_size:
            body["eval_batch_size"] = eval_batch_size
        if flash_attention is not None:
            body["flash_attention"] = flash_attention

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(f"{self.base_url}/api/v1/models/load", json=body)
            if response.status_code >= 400:
                detail = response.text[:600]
                raise RuntimeError(f"LM Studio load failed for {resolved_model_id}: {detail}")
            data = response.json()
            logger.info("bridge_model_loaded", model=resolved_model_id, identifier=identifier, context_length=context_length)
            return data

    async def unload_model(self, model_id: str) -> bool:
        """Unload a model from LM Studio to free VRAM."""
        resolved_model_id = await self.resolve_model_id(model_id)
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                response = await client.post(f"{self.base_url}/api/v1/models/unload", json={"model": resolved_model_id})
                if response.status_code == 200:
                    logger.info("bridge_model_unloaded", model=resolved_model_id)
                    return True
                return False
            except Exception as e:
                logger.warning("bridge_unload_failed", model=resolved_model_id, error=str(e))
                return False

    async def ensure_model_loaded(
        self,
        model_id: str,
        *,
        context_length: Optional[int] = None,
        identifier: Optional[str] = None,
        gpu: Optional[str] = None,
        ttl: Optional[int] = None,
        eval_batch_size: Optional[int] = None,
        flash_attention: Optional[bool] = None,
    ) -> bool:
        """Ensure model is loaded, load it if necessary."""
        if not self.auto_load_models or not model_id:
            return False
        if await self.is_model_loaded(model_id):
            return False

        await self.load_model(
            model_id, context_length=context_length, identifier=identifier,
            gpu=gpu, ttl=ttl, eval_batch_size=eval_batch_size, flash_attention=flash_attention
        )
        return True

    async def embed_texts(
        self, 
        texts: List[str], 
        model: Optional[str] = None, 
        instruction: Optional[str] = None,
        embed_dim: Optional[int] = None
    ) -> List[List[float]]:
        """Embed texts using the specified model."""
        if model and model != self.embed_model:
            return await Embedder(base_url=self.base_url, model=model).embed(
                texts, instruction=instruction, embed_dim=embed_dim
            )
        return await self.embedder.embed(
            texts, instruction=instruction, embed_dim=embed_dim
        )

    async def rerank_chunks(
        self, 
        query: str, 
        chunks: List[str], 
        top_k: int,
        instruction: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Rerank chunks to optimize retrieval."""
        if not query or not chunks:
            return []

        reranked = await self.reranker.rerank(query, chunks, top_k=top_k, instruction=instruction)
        if reranked and any(item.get("score", 0) != 0 for item in reranked):
            return reranked

        # Fallback to embeddings if reranker fails
        query_embedding = await self.embedder.embed_query(query)
        chunk_embeddings = await self.embedder.embed_chunks(chunks)
        if not query_embedding or not chunk_embeddings:
            return [{"chunk": chunk, "score": 0.0} for chunk in chunks[:top_k]]

        scored: List[Dict[str, Any]] = []
        for index, embedding in enumerate(chunk_embeddings):
            score = cosine_score(query_embedding, embedding) if embedding else 0.0
            scored.append({"chunk": chunks[index], "score": float(score)})
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    async def build_retrieval_context(
        self,
        query: str,
        docs: Sequence[Any],
        config: Optional[RetrievalConfig] = None,
    ) -> Dict[str, Any]:
        """Build retrieval context for RAG."""
        retrieval = config or RetrievalConfig()
        normalized_docs = await self._normalize_docs(docs)
        chunk_records = self._chunk_documents(normalized_docs, retrieval.chunking)

        if not chunk_records:
            return {"chunks": [], "context_text": "", "sources": []}

        reranked = await self.rerank_chunks(
            query, [record["chunk"] for record in chunk_records], retrieval.top_k,
            instruction=retrieval.rerank_instruction
        )
        selected: List[Dict[str, Any]] = []
        for item in reranked:
            match = next((r for r in chunk_records if r["chunk"] == item["chunk"]), None)
            if not match:
                continue
            selected.append({
                "text": match["chunk"],
                "score": item.get("score", 0.0),
                "source": match["source"],
                "chunk_index": match["chunk_index"],
            })

        if not selected:
            selected = [
                {
                    "text": record["chunk"],
                    "score": 0.0,
                    "source": record["source"],
                    "chunk_index": record["chunk_index"],
                }
                for record in chunk_records[: retrieval.top_k]
            ]

        context_lines: List[str] = []
        running_chars = 0
        for idx, item in enumerate(selected, start=1):
            snippet = item["text"].strip()
            if not snippet:
                continue
            block = f"[Context {idx} | source={item['source']} | score={item['score']:.4f}]\n{snippet}"
            projected = running_chars + len(block)
            if projected > retrieval.max_context_chars and context_lines:
                break
            context_lines.append(block)
            running_chars = projected

        return {
            "chunks": selected,
            "context_text": "\n\n---\n\n".join(context_lines),
            "sources": [item["source"] for item in selected] if retrieval.include_sources else [],
        }

    async def enrich_chat_body(
        self,
        body: Dict[str, Any],
        *,
        top_k: int,
        default_chunk_size: int = 900,
        default_chunk_overlap: int = 150,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Enrich chat body with model loading and context injection."""
        model_id = body.get("model")
        extra = body.get("extra_body", {})
        retrieval_cfg = extra.get("retrieval", {})
        context_docs = extra.get("context_docs", [])
        if not isinstance(context_docs, list):
            context_docs = []

        load_cfg = extra.get("load_model", {})
        if model_id and (self.auto_load_models or load_cfg):
            await self.ensure_model_loaded(
                model_id, context_length=load_cfg.get("context_length"),
                identifier=load_cfg.get("identifier"), gpu=load_cfg.get("gpu"),
                ttl=load_cfg.get("ttl"), eval_batch_size=load_cfg.get("eval_batch_size"),
                flash_attention=load_cfg.get("flash_attention"),
            )

        for aux_model_key, model_name in (
            ("embed_model", retrieval_cfg.get("embed_model") or extra.get("embed_model")),
            ("rerank_model", retrieval_cfg.get("rerank_model") or extra.get("rerank_model")),
        ):
            if model_name:
                try:
                    await self.ensure_model_loaded(model_name, identifier=f"{aux_model_key}:{model_name}")
                except Exception as exc:
                    logger.warning("bridge_aux_model_load_failed", model=model_name, error=str(exc))

        if not context_docs:
            return body, {"retrieval": None}

        user_prompt = self._extract_last_user_message(body.get("messages", []))
        retrieval = RetrievalConfig(
            top_k=int(retrieval_cfg.get("top_k", top_k)),
            chunking=ChunkingConfig(
                chunk_size=int(retrieval_cfg.get("chunk_size", default_chunk_size)),
                chunk_overlap=int(retrieval_cfg.get("chunk_overlap", default_chunk_overlap)),
                max_chunks=int(retrieval_cfg.get("max_chunks", 64)),
                max_chunk_chars=int(retrieval_cfg.get("max_chunk_chars", 1600)),
            ),
            include_sources=bool(retrieval_cfg.get("include_sources", True)),
            max_context_chars=int(retrieval_cfg.get("max_context_chars", 6000)),
            rerank_instruction=retrieval_cfg.get("rerank_instruction", "retrieval"),
            embed_instruction=retrieval_cfg.get("embed_instruction", "retrieval"),
            embed_dim=retrieval_cfg.get("embed_dim"),
        )
        result = await self.build_retrieval_context(user_prompt, context_docs, retrieval)
        context_text = result["context_text"]
        if context_text:
            insertion = {
                "role": "system",
                "content": (
                    "Use the retrieved context below when it is relevant. "
                    "Prefer grounded answers, cite the source labels when useful, "
                    "and say when the provided context is insufficient.\n\n"
                    f"{context_text}"
                ),
            }
            messages = body.get("messages", [])
            insert_at = 1 if messages and messages[0].get("role") == "system" else 0
            messages.insert(insert_at, insertion)
            body["messages"] = messages
            logger.info("bridge_context_injected", chunks=len(result["chunks"]), sources=result["sources"])

        return body, {"retrieval": result}

    async def _normalize_docs(self, docs: Sequence[Any]) -> List[Dict[str, str]]:
        """Normalize documents concurrently using asyncio.gather."""
        tasks = [self._normalize_one_doc(doc, index) for index, doc in enumerate(docs)]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r]

    async def _normalize_one_doc(self, doc: Any, index: int) -> Optional[Dict[str, str]]:
        if isinstance(doc, str):
            if os.path.exists(doc):
                return self._read_local_path(doc)
            return {"source": f"inline:{index}", "text": doc}

        if not isinstance(doc, dict):
            return None

        if doc.get("text") or doc.get("content") or doc.get("chunk"):
            text = doc.get("text") or doc.get("content") or doc.get("chunk") or ""
            source = doc.get("source") or doc.get("title") or f"inline:{index}"
            return {"source": str(source), "text": str(text)}

        if doc.get("path"):
            return self._read_local_path(str(doc["path"]))

        if doc.get("url"):
            url = str(doc["url"])
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    response = await client.get(url)
                    response.raise_for_status()
                    return {"source": url, "text": response.text}
            except Exception as exc:
                logger.warning("bridge_url_fetch_failed", url=url, error=str(exc))
                return None

        return None

    def _read_local_path(self, path: str) -> Optional[Dict[str, str]]:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                return {"source": os.path.abspath(path), "text": handle.read()}
        except Exception as exc:
            logger.warning("bridge_local_doc_failed", path=path, error=str(exc))
            return None

    def _chunk_documents(self, docs: Sequence[Dict[str, str]], config: ChunkingConfig) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for doc in docs:
            chunks = self._chunk_text(doc["text"], config.chunk_size, config.chunk_overlap)
            for chunk_index, chunk in enumerate(chunks):
                clipped = chunk.strip()[: config.max_chunk_chars]
                if clipped:
                    records.append({
                        "source": doc["source"],
                        "chunk": clipped,
                        "chunk_index": chunk_index,
                    })
                if len(records) >= config.max_chunks:
                    return records
        return records

    def _chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        cleaned = " ".join(text.split())
        if not cleaned:
            return []
        if len(cleaned) <= chunk_size:
            return [cleaned]

        sentences = self._split_sentences(cleaned)
        if len(sentences) == 1:
            return self._fixed_chunks(cleaned, chunk_size, chunk_overlap)

        chunks: List[str] = []
        current = ""
        for sentence in sentences:
            candidate = sentence if not current else f"{current} {sentence}"
            if len(candidate) <= chunk_size:
                current = candidate
                continue
            if current:
                chunks.append(current)
            if len(sentence) > chunk_size:
                chunks.extend(self._fixed_chunks(sentence, chunk_size, chunk_overlap))
                current = ""
                continue
            if chunk_overlap > 0 and chunks:
                overlap_text = chunks[-1][-chunk_overlap:]
                current = f"{overlap_text} {sentence}".strip()
            else:
                current = sentence
        if current:
            chunks.append(current)
        return chunks

    def _fixed_chunks(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        if chunk_overlap >= chunk_size:
            chunk_overlap = max(0, chunk_size // 5)
        step = max(1, chunk_size - chunk_overlap)
        chunks = []
        for start in range(0, len(text), step):
            piece = text[start : start + chunk_size].strip()
            if piece:
                chunks.append(piece)
        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        sentence = []
        sentences = []
        for char in text:
            sentence.append(char)
            if char in ".!?:" :
                joined = "".join(sentence).strip()
                if joined:
                    sentences.append(joined)
                sentence = []
        remainder = "".join(sentence).strip()
        if remainder:
            sentences.append(remainder)
        return sentences or [text]

    def _extract_last_user_message(self, messages: Iterable[Dict[str, Any]]) -> str:
        for message in reversed(list(messages)):
            if message.get("role") != "user":
                continue
            return self.extract_text_from_content(message.get("content"))
        return ""

    def _serialize_content(self, content: Any) -> str:
        """Serialize content for sending to LM Studio."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return json.dumps(content)
        return str(content)

    async def openai_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = 0.6,
        top_p: Optional[float] = 0.95,
        max_tokens: Optional[int] = 4096,
    ) -> Dict[str, Any]:
        """Make an OpenAI-compatible chat completion request."""
        model_id = await self.resolve_model_id(model)
        enriched = await self.enrich_chat_body({"model": model_id, "messages": messages}, top_k=4)
        body, retrieval = enriched
        requests: List[Dict[str, Any]] = []

        for idx, message in enumerate(body["messages"]):
            content = self._serialize_content(message.get("content", ""))
            req = {"role": message.get("role", "user"), "content": content}

            tool_calls_to_apply = None
            msg = message.get("tool_calls")
            if msg is not None and isinstance(msg, list) and msg and isinstance(msg[0], dict):
                tool_calls_to_apply = {t.get("function", {}).get("name", "") for t in msg}
            func_call = message.get("function_call")
            if func_call is not None and isinstance(func_call, dict):
                tool_calls_to_apply = {func_call.get("name", "")}

            req["tool_choice"] = tool_calls_to_apply if tool_calls_to_apply else "auto"

            if "role" in message and message.get("role") == "tool":
                tool_call_id = None
                for t in body["messages"]:
                    if t.get("content") == content:
                        for tc in t.get("tool_calls", []):
                            if tc and isinstance(tc, dict):
                                for ft in tc.get("function", []):
                                    if isinstance(ft, dict) and ft.get("name") == t.get("tool_call_id"):
                                        tool_call_id = ft["name"]
                                        break
                                    if isinstance(ft, str) and ft == t.get("tool_call_id"):
                                        tool_call_id = ft
                                        break
                req["tool_call_id"] = tool_call_id
                body["messages"][idx] = req
                requests.append(req)
            else:
                body["messages"][idx] = req
                requests.append(req)

        body["messages"] = body["messages"][: max(1, len(body["messages"]) - len(requests))]

        extra = {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}
        for param in extra:
            if getattr(body, param, None) is not None:
                extra[param] = body[param]
        extra["stream"] = False

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json={"model": model_id, "messages": body["messages"], **extra}
            )
            if response.status_code >= 400:
                detail = response.text[:512]
                raise RuntimeError(f"LM Studio chat completion failed: {detail}")
            data = response.json()
            return data

    async def openai_chat_completion_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = 0.6,
        top_p: Optional[float] = 0.95,
        max_tokens: Optional[int] = 4096,
    ) -> Generator[Dict[str, Dict[str, Any]]]:
        """Generate streaming responses using SSE."""
        async for chunk in await self._stream_chat_completion(
            model, messages, temperature, top_p, max_tokens
        ):
            yield chunk

    async def _stream_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = 0.6,
        top_p: Optional[float] = 0.95,
        max_tokens: Optional[int] = 4096,
    ) -> Generator[Dict[str, Dict[str, Any]]]:
        """Internal async generator for SSE streaming responses."""
        model_id = await self.resolve_model_id(model)
        enriched = await self.enrich_chat_body({"model": model_id, "messages": messages}, top_k=4)
        body, retrieval = enriched
        requests: List[Dict[str, Any]] = []

        for idx, message in enumerate(body["messages"]):
            content = self._serialize_content(message.get("content", ""))
            req = {"role": message.get("role", "user"), "content": content}

            tool_calls_to_apply = None
            msg = message.get("tool_calls")
            if msg is not None and isinstance(msg, list) and msg and isinstance(msg[0], dict):
                tool_calls_to_apply = {t.get("function", {}).get("name", "") for t in msg}
            func_call = message.get("function_call")
            if func_call is not None and isinstance(func_call, dict):
                tool_calls_to_apply = {func_call.get("name", "")}

            req["tool_choice"] = tool_calls_to_apply if tool_calls_to_apply else "auto"

            if "role" in message and message.get("role") == "tool":
                tool_call_id = None
                for t in body["messages"]:
                    if t.get("content") == content:
                        for tc in t.get("tool_calls", []):
                            if tc and isinstance(tc, dict):
                                for ft in tc.get("function", []):
                                    if isinstance(ft, dict) and ft.get("name") == t.get("tool_call_id"):
                                        tool_call_id = ft["name"]
                                        break
                                    if isinstance(ft, str) and ft == t.get("tool_call_id"):
                                        tool_call_id = ft
                                        break
                req["tool_call_id"] = tool_call_id
                body["messages"][idx] = req
                requests.append(req)
            else:
                body["messages"][idx] = req
                requests.append(req)

        body["messages"] = body["messages"][: max(1, len(body["messages"]) - len(requests))]

        extra = {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}
        for param in extra:
            if getattr(body, param, None) is not None:
                extra[param] = body[param]
        extra["stream"] = True

        async with httpx.AsyncClient(timeout=60) as client:
            async for line in client.stream(
                f"{self.base_url}/v1/chat/completions",
                json={"model": model_id, "messages": body["messages"], **extra}
            ): 
                try: 
                    line_data = line.decode() 
                    break 
                except Exception as error: 
                    logger.error("stream_line_decode_error", error=str(error)); 
                 
                if line_data.startswith("data:") or line_data == ":" \
                    or not line_data.strip(): 
                    continue 
                 
                event_data = line_data[6:].strip() 
                if not event_data: 
                    continue 
                 
                try: 
                    data = json.loads(event_data) 
                except json.JSONDecodeError as json_err: 
                    logger.warning("stream_event_parse_failed", error=(str(json_err))) 
                    continue 
                 
                try: 
                    content = json.dumps(data) 
                except Exception as encode_err: 
                    logger.warning("stream_event_encode_failed", error=(str(encode_err))) 
                    return 
                 
                try: 
                    yield {"data": content} 
                except Exception as yield_err: 
                    logger.warning("stream_yield_failed", error=(str(yield_err))); 
                 
                if not data.get("has_content") or not data.get("choices") \
                    or data["choices"] and not data["choices"][0].get("content"): 
                    break 
                 
                if data.get("choices") and len(data["choices"]) > 0: 
                    content = data["choices"][0].get("content") or "" 
                    content = json.dumps(content).replace("\\n", "\\n") 
                    yield {"data": f"content: {content}"} 
                 
                if data.get("choices") and len(data["choices"]) > 0: 
                    tool_calls = data["choices"][0].get("tool_calls") 
                    if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0: 
                        tool_call_str = json.dumps(tool_calls).replace("\\n", "\\n") 
                        yield {"data": f"tool_calls: {tool_call_str}"} 
                 
                if data.get("choices") and len(data["choices"]) > 0: 
                    finish_reason = data["choices"][0].get("finish_reason") 
                    if finish_reason: 
                        yield {"data": f"finish_reason: {finish_reason}"} 
                 
                if data.get("choices") and len(data["choices"]) > 0: 
                    completion_data = data["choices"][0] 
                    if "message" in completion_data: 
                        msg = completion_data["message"] 
                        tools = msg.get("tools", {}) 
                        if tools and isinstance(tools, list) and len(tools) > 0: 
                            tools_str = json.dumps(tools).replace("\\n", "\\n") 
                            yield {"data": f"tools: {tools_str}"} 
                        
                    role = msg.get("role") 
                    if role and msg not in body["messages"]: 
                        body["messages"].insert(0, dict(msg)) 
                        requests.append(dict(msg)) 
                    
                    role = msg.get("role") 
                    if role and body["messages"] and body["messages"][0].get("role") != msg.get("role"): 
                        msg_copy = dict(msg) 
                        msg_copy["role"] = msg_get(
                                role)
                        if "tool_calls" in msg_copy:
                            for tc in msg_copy["tool_calls"]:
                                tc.copy()
                            msg_copy["tool_choice"] = tool_calls_to_apply if tool_calls_to_apply else "auto"
                        if "function_call" in msg_copy:
                            for fc in msg_copy["function_call"]:
                                fc.copy()
                            msg_copy["tool_choice"] = tool_calls_to_apply if tool_calls_to_apply else "auto"
                        msg_copy["role"] = msg_get(
                                role) if msg_copy.get("role") != "assistant" else "assistant"
                        if "content" in msg_copy:
                            msg_copy["content"] = self.extract_text_from_content(msg_copy.get("content"))
                        
                        body["messages"][0] = msg_copy
                        requests[0] = msg_copy