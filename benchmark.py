# benchmark.py - Complete LM Studio Benchmark with Proxy & Reasoning Detection
import asyncio
import httpx
import time
import json
import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlparse
import argparse
import os

class StopReason(Enum):
    COMPLETE = "complete"
    LENGTH = "max_tokens_reached"
    TIMEOUT = "timeout"
    ERROR = "error"
    TRUNCATED = "possibly_truncated"
    CANCELLED = "cancelled"

@dataclass
class ModelCapabilities:
    reasoning: bool = False
    tool_use: bool = False
    vision: bool = False
    context_urls: bool = False
    max_context: int = 4096
    recommended_quant: str = "Q4_K_M"

class StreamingBenchmark:
    def __init__(self, 
                 base_url: str = "http://0.0.0.0:8080", 
                 timeout: int = 60, 
                 proxy_url: Optional[str] = None):
        self.base_url = base_url
        self.timeout = timeout
        self.proxy_url = proxy_url
        
        # Configure client with proxy
        self.client_args = {}
        if proxy_url:
            self.client_args["proxy"] = proxy_url
            
        # Test prompts designed to expose specific capabilities
        self.test_prompts = {
            "short": "What is 2+2?",
            "code": "Write a complete Python function to sort a list using quicksort.",
            "long_reasoning": "Explain step by step how to solve the travelling salesman problem.",
            "json_output": 'Generate valid JSON: {"name": "test", "value": 123}',
            "context_url": "Fetch and summarize: https://httpbin.org/html",
            "tool_use": "Calculate the square root of 144. Use a tool if available.",
            "vision": "Describe what you see in https://picsum.photos/200/300",
            "thinking": "Think step by step: If a train travels 60km in 30 minutes, what's its speed?"
        }

    def get_client(self) -> httpx.AsyncClient:
        """Get configured HTTP client with optional proxy"""
        return httpx.AsyncClient(timeout=self.timeout, **self.client_args)

    async def check_server(self) -> Dict[str, Any]:
        async with self.get_client() as client:
            try:
                resp = await client.get(f"{self.base_url}/v1/models")
                return {"ok": resp.status_code == 200, "status_code": resp.status_code}
            except Exception as e:
                return {"ok": False, "error": str(e)}

    async def list_models(self) -> Dict[str, Any]:
        async with self.get_client() as client:
            for path in ["/v1/models", "/api/v1/models"]:
                try:
                    resp = await client.get(f"{self.base_url}{path}")
                    if resp.status_code != 200:
                        continue
                    data = resp.json()
                    if isinstance(data, dict) and isinstance(data.get("data"), list):
                        return {"ok": True, "source": path, "models": data["data"]}
                    if isinstance(data, list):
                        return {"ok": True, "source": path, "models": data}
                    return {"ok": True, "source": path, "raw": data, "models": []}
                except Exception:
                    continue
        return {"ok": False, "models": []}

    async def get_optional_hardware(self) -> Optional[Dict[str, Any]]:
        async with self.get_client() as client:
            for path in ["/api/v1/hardware", "/api/v1/system/hardware"]:
                try:
                    resp = await client.get(f"{self.base_url}{path}")
                    if resp.status_code == 200:
                        data = resp.json()
                        return {"source": path, "data": data}
                except Exception:
                    continue
        return None

    async def test_proxy_connectivity(self, test_url: str = "https://httpbin.org/get") -> Dict[str, Any]:
        """Test proxy connectivity and measure latency"""
        if not self.proxy_url:
            return {"status": "no_proxy_configured"}
            
        async with self.get_client() as client:
            start = time.time()
            try:
                resp = await client.get(test_url)
                return {
                    "status": "ok" if resp.status_code == 200 else f"http_{resp.status_code}",
                    "latency_ms": round((time.time() - start) * 1000, 2),
                    "proxy": self.proxy_url,
                    "headers": dict(resp.headers) if resp.status_code == 200 else None
                }
            except Exception as e:
                return {"status": "error", "error": str(e), "proxy": self.proxy_url}

    async def detect_capabilities(self, model_id: str) -> ModelCapabilities:
        """Detect model capabilities (reasoning, tools, vision, context)"""
        caps = ModelCapabilities()
        
        # Test 1: Reasoning detection (Qwen/DeepSeek style)
        reasoning_test = await self._quick_test(
            model_id, 
            "Show your reasoning: What is 15 * 23?",
            max_tokens=256
        )
        if reasoning_test.get("has_reasoning"):
            caps.reasoning = True
            # Extract reasoning format
            content = reasoning_test.get("content", "")
            if "<reasoning>" in content or "<think>" in content:
                caps.reasoning_format = "xml_tags"
            elif "reasoning_content" in reasoning_test.get("raw_response", {}):
                caps.reasoning_format = "separate_field"
        
        # Test 2: Tool use detection
        tool_test = await self._quick_test(
            model_id,
            "What is the weather in Paris? (Use a tool if available)",
            max_tokens=100
        )
        if tool_test.get("has_tool_calls"):
            caps.tool_use = True
        
        # Test 3: Vision detection
        vision_test = await self._quick_test(
            model_id,
            "Describe: https://picsum.photos/200/300",
            max_tokens=100
        )
        # If it describes the image rather than saying it can't see, it has vision
        if any(x in vision_test.get("content", "").lower() for x in ["image", "photo", "picture", "shows"]):
            caps.vision = True
        
        # Test 4: Context URL handling
        url_test = await self._quick_test(
            model_id,
            "Summarize: https://httpbin.org/html",
            max_tokens=150
        )
        content = url_test.get("content", "")
        # Check if it actually fetched (mentions html/httpbin) or hallucinated
        if any(x in content.lower() for x in ["html", "httpbin", "h1", "body"]):
            caps.context_urls = True
        
        # Determine max context based on model name
        if "qwen" in model_id.lower():
            caps.max_context = 32768 if "4b" in model_id.lower() else 131072
            caps.recommended_quant = "Q6_K" if "4b" in model_id.lower() else "Q4_K_M"
        elif "llama" in model_id.lower():
            caps.max_context = 8192 if "3.1" not in model_id else 131072
        elif "nemotron" in model_id.lower():
            caps.max_context = 4096
        
        return caps

    async def _quick_test(self, model_id: str, prompt: str, max_tokens: int = 100) -> Dict[str, Any]:
        """Quick single-turn test for capability detection"""
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        try:
            async with self.get_client() as client:
                resp = await client.post(f"{self.base_url}/v1/chat/completions", json=payload)
                if resp.status_code != 200:
                    return {"error": resp.status_code}
                
                data = resp.json()
                choice = data.get("choices", [{}])[0]
                message = choice.get("message", {})
                content = message.get("content", "")
                
                return {
                    "content": content,
                    "has_reasoning": "reasoning_content" in message or "<reasoning>" in content,
                    "has_tool_calls": "tool_calls" in message or "function_call" in message,
                    "raw_response": message
                }
        except Exception as e:
            return {"error": str(e)}

    async def benchmark_streaming(self, model_id: str, prompt_key: str = "code", 
                                 max_tokens: int = 512) -> Dict[str, Any]:
        """
        Comprehensive streaming benchmark with premature stop detection
        """
        prompt = self.test_prompts[prompt_key]
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": max_tokens,
            "stream": True,
            "stop": None  # Prevent early stopping
        }
        
        start_time = time.time()
        first_token_time = None
        last_token_time = None
        content_chunks = []
        reasoning_chunks = []
        token_times = []  # Track inter-token latency
        chunk_count = 0
        
        try:
            async with self.get_client() as client:
                async with client.stream("POST", f"{self.base_url}/v1/chat/completions", json=payload) as resp:
                    if resp.status_code != 200:
                        return self._error_result(model_id, f"HTTP {resp.status_code}")
                    
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(data_str)
                            choice = data.get("choices", [{}])[0]
                            delta = choice.get("delta", {})
                            finish_reason = choice.get("finish_reason")
                            
                            now = time.time()
                            
                            # Detect reasoning content (Qwen 3.5, DeepSeek)
                            if "reasoning_content" in delta:
                                reasoning_chunks.append(delta["reasoning_content"])
                                chunk_count += 1
                            elif "content" in delta:
                                if first_token_time is None:
                                    first_token_time = now
                                content_chunks.append(delta["content"])
                                token_times.append(now)
                                chunk_count += 1
                                last_token_time = now
                                
                        except json.JSONDecodeError:
                            continue
        
        except asyncio.TimeoutError:
            return self._error_result(model_id, "Timeout")
        except Exception as e:
            return self._error_result(model_id, str(e))
        
        end_time = time.time()
        full_content = "".join(content_chunks)
        reasoning_content = "".join(reasoning_chunks)
        
        # Calculate metrics
        total_duration = end_time - start_time
        ttft = (first_token_time - start_time) if first_token_time else 0
        generation_time = (last_token_time - first_token_time) if last_token_time and first_token_time else 0
        
        # Token estimation (since usage not in streaming)
        output_tokens = len(full_content.split()) * 1.3
        reasoning_tokens = len(reasoning_content.split()) if reasoning_content else 0
        
        # TPS calculation (generation phase only, excluding TTFT)
        tps = output_tokens / generation_time if generation_time > 0 else 0
        
        # Inter-token latency (jitter)
        if len(token_times) > 1:
            intervals = [token_times[i] - token_times[i-1] for i in range(1, len(token_times))]
            avg_interval = sum(intervals) / len(intervals)
            jitter = max(intervals) - min(intervals) if len(intervals) > 1 else 0
        else:
            avg_interval = 0
            jitter = 0
        
        # Detect stop reason
        stop_reason = self._classify_stop(full_content, finish_reason, total_duration)
        
        return {
            "model": model_id,
            "mode": "streaming",
            "prompt": prompt_key,
            "metrics": {
                "ttft_ms": round(ttft * 1000, 2),
                "generation_time_ms": round(generation_time * 1000, 2),
                "total_duration_ms": round(total_duration * 1000, 2),
                "tps": round(tps, 2),
                "tokens_generated": int(output_tokens),
                "reasoning_tokens": reasoning_tokens,
                "content_tokens": len(full_content.split()),
                "chunks_received": chunk_count,
                "avg_inter_token_ms": round(avg_interval * 1000, 2),
                "jitter_ms": round(jitter * 1000, 2),
                "stop_reason": stop_reason.value,
                "finish_reason": finish_reason
            },
            "content_analysis": {
                "total_length": len(full_content),
                "has_code_blocks": "```" in full_content,
                "has_json": "{" in full_content and "}" in full_content,
                "truncation_issues": self._get_truncation_issues(full_content),
                "reasoning_length": len(reasoning_content)
            },
            "sample": full_content[:300] + "..." if len(full_content) > 300 else full_content
        }

    async def benchmark_non_streaming(self, model_id: str, prompt_key: str = "code",
                                     max_tokens: int = 512) -> Dict[str, Any]:
        """Non-streaming baseline for comparison"""
        prompt = self.test_prompts[prompt_key]
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        start = time.time()
        
        try:
            async with self.get_client() as client:
                resp = await client.post(f"{self.base_url}/v1/chat/completions", json=payload)
                end = time.time()
                
                if resp.status_code != 200:
                    return self._error_result(model_id, f"HTTP {resp.status_code}", mode="non_streaming")
                
                data = resp.json()
                choice = data.get("choices", [{}])[0]
                message = choice.get("message", {})
                content = message.get("content", "")
                usage = data.get("usage", {})
                
                duration = end - start
                output_tokens = usage.get("completion_tokens", 0)
                
                return {
                    "model": model_id,
                    "mode": "non_streaming",
                    "prompt": prompt_key,
                    "metrics": {
                        "total_duration_ms": round(duration * 1000, 2),
                        "tps": round(output_tokens / duration, 2) if duration > 0 else 0,
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": output_tokens,
                        "total_tokens": usage.get("total_tokens", 0),
                        "stop_reason": choice.get("finish_reason", "unknown")
                    },
                    "content_analysis": {
                        "total_length": len(content),
                        "truncation_issues": self._get_truncation_issues(content)
                    },
                    "sample": content[:300] + "..." if len(content) > 300 else content
                }
        except Exception as e:
            return self._error_result(model_id, str(e), mode="non_streaming")

    def _classify_stop(self, content: str, finish_reason: Optional[str], duration: float) -> StopReason:
        """Classify why generation stopped"""
        if finish_reason == "length":
            return StopReason.LENGTH
        elif duration >= self.timeout - 5:
            return StopReason.TIMEOUT
        elif self._is_truncated(content):
            return StopReason.TRUNCATED
        elif finish_reason == "stop":
            return StopReason.COMPLETE
        else:
            return StopReason.ERROR

    def _is_truncated(self, content: str) -> bool:
        """Detect truncated output"""
        if not content:
            return True
        
        issues = self._get_truncation_issues(content)
        return len(issues) > 0

    def _get_truncation_issues(self, content: str) -> List[str]:
        """List specific truncation issues"""
        issues = []
        if content.count("```") % 2 != 0:
            issues.append("unclosed_code_block")
        if content.count("{") != content.count("}"):
            issues.append("unclosed_braces")
        if content.count("(") != content.count(")"):
            issues.append("unclosed_parens")
        if content.count("[") != content.count("]"):
            issues.append("unclosed_brackets")
        
        # Check for mid-sentence ending
        content_stripped = content.rstrip()
        if content_stripped and content_stripped[-1] not in ".!?;:\"')]}\\n":
            issues.append("mid_sentence")
        
        return issues

    def _error_result(self, model_id: str, error: str, mode: str = "streaming") -> Dict[str, Any]:
        return {
            "model": model_id,
            "mode": mode,
            "error": error,
            "metrics": {},
            "content_analysis": {}
        }

    async def compare_streaming_vs_nonstreaming(self, model_id: str, 
                                               prompt_key: str = "code") -> Dict[str, Any]:
        """Direct comparison of streaming vs non-streaming for same prompt"""
        print(f"  → Testing streaming...")
        stream_result = await self.benchmark_streaming(model_id, prompt_key)
        
        print(f"  → Testing non-streaming...")
        non_stream_result = await self.benchmark_non_streaming(model_id, prompt_key)
        
        # Calculate overhead
        stream_time = stream_result.get("metrics", {}).get("total_duration_ms", 0)
        non_stream_time = non_stream_result.get("metrics", {}).get("total_duration_ms", 0)
        overhead = stream_time - non_stream_time if stream_time > non_stream_time else 0
        
        return {
            "model": model_id,
            "prompt": prompt_key,
            "streaming": stream_result,
            "non_streaming": non_stream_result,
            "comparison": {
                "streaming_overhead_ms": round(overhead, 2),
                "ttft_ms": stream_result.get("metrics", {}).get("ttft_ms"),
                "streaming_tps": stream_result.get("metrics", {}).get("tps"),
                "non_streaming_tps": non_stream_result.get("metrics", {}).get("tps"),
                "recommendation": "non_streaming" if overhead > 5000 else "streaming"
            }
        }

    async def run_full_benchmark(self, model_id: str, prompt_key: str = "code"):
        """Complete benchmark suite"""
        print(f"\n🔬 Benchmarking {model_id}")
        print("=" * 60)

        server = await self.check_server()
        if not server.get("ok"):
            print(f"\n❌ Cannot reach LM Studio at {self.base_url}")
            if server.get("status_code") is not None:
                print(f"   HTTP: {server.get('status_code')}")
            if server.get("error"):
                print(f"   Error: {server.get('error')}")
            print("\n" + "=" * 60)
            return

        models = await self.list_models()
        if models.get("ok") and isinstance(models.get("models"), list):
            ids = []
            for m in models["models"]:
                if isinstance(m, dict) and m.get("id"):
                    ids.append(m["id"])
                elif isinstance(m, str):
                    ids.append(m)
            if ids:
                print(f"\n📦 Models ({models.get('source')}):")
                for mid in ids[:10]:
                    print(f"   - {mid}")
                if model_id not in ids:
                    model_id = ids[0]
                    print(f"\nℹ️ Using first available model: {model_id}")
        hw = await self.get_optional_hardware()
        if hw and isinstance(hw.get("data"), dict):
            data = hw["data"]
            gpu = data.get("gpu_name") or data.get("gpu") or data.get("name")
            vram = data.get("gpu_vram_gb") or data.get("vram_gb") or data.get("vram")
            platform = data.get("platform")
            print(f"\n🖥️ Hardware ({hw.get('source')}): {platform} | {gpu} | vram_gb={vram}")

        try:
            from model_discovery import get_model_path, extract_model_specs
            model_path = get_model_path(model_id)
            specs = extract_model_specs(model_path) if model_path else None
            if specs:
                max_pos = specs.get("max_position") or specs.get("max_context")
                print(f"\n🧩 Model Specs (local): type={specs.get('model_type')} layers={specs.get('num_layers')} heads={specs.get('num_heads')} kv_heads={specs.get('kv_heads')} ctx={max_pos}")
        except Exception:
            pass

        try:
            import re as _re
            from hardware_detector import HardwareDetector
            from engine import RecommendationEngine
            m = _re.search(r"(\d+(?:\.\d+)?)\s*b", model_id.lower())
            params_b = float(m.group(1)) if m else 4.0
            detector = HardwareDetector()
            profile = detector.detect()
            engine = RecommendationEngine(profile)
            recs = engine.recommend(model_id=model_id, params_b=params_b, use_case="balanced")
            if recs:
                top = recs[0]
                print(f"\n🎛️ Recommended: ctx={top.context_length} gpu_layers={top.gpu_layers} backend={top.inference_backend.value} quant={top.quantization.value} est_vram_gb={top.estimated_vram_gb}")
        except Exception:
            pass
        
        # 1. Proxy test
        if self.proxy_url:
            print("\n📡 Testing Proxy...")
            proxy_test = await self.test_proxy_connectivity()
            print(f"   Status: {proxy_test.get('status')}")
            print(f"   Latency: {proxy_test.get('latency_ms', 'N/A')}ms")
        
        # 2. Capability detection
        print("\n🧠 Detecting Capabilities...")
        caps = await self.detect_capabilities(model_id)
        print(f"   Reasoning: {'✅' if caps.reasoning else '❌'}")
        print(f"   Tool Use: {'✅' if caps.tool_use else '❌'}")
        print(f"   Vision: {'✅' if caps.vision else '❌'}")
        print(f"   Context URLs: {'✅' if caps.context_urls else '❌'}")
        print(f"   Max Context: {caps.max_context:,} tokens")
        print(f"   Recommended Quant: {caps.recommended_quant}")
        
        # 3. Streaming vs Non-streaming comparison
        print("\n⚡ Streaming vs Non-Streaming...")
        comparison = await self.compare_streaming_vs_nonstreaming(model_id, prompt_key)
        comp = comparison.get("comparison", {})
        print(f"   Streaming TTFT: {comp.get('ttft_ms', 'N/A')}ms")
        print(f"   Streaming TPS: {comp.get('streaming_tps', 'N/A')}")
        print(f"   Non-Streaming TPS: {comp.get('non_streaming_tps', 'N/A')}")
        print(f"   Overhead: {comp.get('streaming_overhead_ms', 'N/A')}ms")
        print(f"   Recommendation: Use {comp.get('recommendation', 'unknown')}")
        
        # 4. Test reasoning if available
        if caps.reasoning:
            print("\n🤔 Testing Reasoning Mode...")
            reasoning_test = await self.benchmark_streaming(model_id, "thinking", max_tokens=512)
            m = reasoning_test.get("metrics", {})
            print(f"   Reasoning tokens: {m.get('reasoning_tokens', 0)}")
            print(f"   Content tokens: {m.get('content_tokens', 0)}")
            print(f"   TPS: {m.get('tps', 'N/A')}")
        
        # 5. Test context URL
        print("\n🌐 Testing Context URL...")
        url_test = await self._quick_test(model_id, self.test_prompts["context_url"], max_tokens=200)
        if url_test.get("error"):
            print(f"   ❌ Error: {url_test['error']}")
        else:
            content = url_test.get("content", "")
            fetched = any(x in content.lower() for x in ["html", "httpbin", "h1"])
            print(f"   {'✅' if fetched else '⚠️'} URL Fetch: {'Success' if fetched else 'Hallucinated'}")
            print(f"   Length: {len(content)} chars")
        
        print("\n" + "=" * 60)

# Usage
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.getenv("LMSTUDIO_BASE_URL", "http://192.168.1.12:1234"))
    parser.add_argument("--model", default=os.getenv("MODEL_ID", "qwen3.5-4b"))
    parser.add_argument("--prompt", default=os.getenv("PROMPT_KEY", "code"))
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--connect-only", action="store_true")
    args = parser.parse_args()

    bench = StreamingBenchmark(base_url=args.base_url.rstrip("/"))
    if args.connect_only:
        server = await bench.check_server()
        print(json.dumps({"base_url": bench.base_url, "server": server}, indent=2))
        return
    if args.list_models:
        models = await bench.list_models()
        print(json.dumps({"base_url": bench.base_url, "models": models}, indent=2))
        return
    await bench.run_full_benchmark(args.model, prompt_key=args.prompt)

if __name__ == "__main__":
    asyncio.run(main())
