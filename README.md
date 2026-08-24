# Local Retrieval System Proxy 

Production-ready proxy for LMStudio with OpenAI-compatible API, two-stage retrieval, agent orchestration, and ACE context enhancement.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the proxy
python proxy.py

# Access API
curl http://localhost:8080/v1/models

# Web console
open http://localhost:8090
```

## 📚 Documentation

**All documentation has been reorganized and consolidated in the `docs/` folder.**

### Quick Links
- **[📖 Documentation Index](docs/_index.md)** - Start here for complete navigation
- **[Getting Started](docs/01-getting-started/)** - Installation and first run
- **[API Reference](docs/02-api-integration/)** - All endpoints and examples
- **[Retrieval System](docs/03-retrieval-system/)** - Two-stage retrieval pipeline
- **[Agent Development](docs/04-agent-development/)** - Building agentic AI systems
- **[Architecture](docs/05-architecture/)** - System design and components
- **[Optimization](docs/06-optimization/)** - Performance tuning
- **[Operations](docs/07-operations/)** - Deployment and monitoring
- **[Configuration](docs/08-configuration/)** - Configuration reference

## ✨ Features

- **OpenAI-Compatible API** - Drop-in replacement for `openai` SDK
- **Two-Stage Retrieval** - Embeddings + cross-encoder reranking
- **EOS Token Handling** - Automatic LMStudio tokenizer fixes
- **Agent Orchestration** - Multi-turn sessions with tool calling
- **ACE Context Engine** - Advanced context analysis and injection
- **Hardware Optimization** - Adaptive tuning based on GPU/CPU
- **Web Dashboard** - Flask console for management
- **Production Ready** - Logging, error handling, monitoring


## 🏗️ Architecture

```
┌─ Web Console (Flask @ :8090)
│  ├─ Dashboard
│  ├─ ACE session management
│  └─ Configuration UI
│
├─ FastAPI Bridge (@ :8080)
│  ├─ /v1/chat/completions (OpenAI-compatible)
│  ├─ /v1/embeddings (with EOS token)
│  ├─ /api/v1/retrieve (two-stage pipeline)
│  ├─ /v1/agent/* (agent sessions)
│  └─ /v1/ace/* (context enhancement)
│
└─ LMStudio Backend
   ├─ Reasoning model (qwen3.5-4b)
   ├─ Embedding model (text-embedding-qwen3)
   └─ Reranking model (qwen3-reranker)
```

## 🔧 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8+ | 3.11+ |
| VRAM | 4 GB | 8+ GB |
| RAM | 8 GB | 16+ GB |
| LMStudio | Latest | Latest |

## 📝 Configuration

### Environment Variables
```bash
# LMStudio connection
BRIDGE_HOST=127.0.0.1
BRIDGE_PORT=8080
LMSTUDIO_BASE_URL=http://192.168.1.12:1234

# Models (optional, auto-detected from LMStudio)
MAIN_MODEL=qwen3.5-4b
EMBED_MODEL=text-embedding-qwen3-embedding-4b
RERANK_MODEL=qwen.qwen3-reranker-4b

# Web console
WEB_PORT=8090

# Proxy server
PROXY_PORT=8080
```

### Load Models Automatically
```bash
AUTO_LOAD_MODELS=true
```

## 🎯 Common Tasks

### Generate Embeddings
```python
import openai
openai.api_base = "http://localhost:8080"

result = openai.Embedding.create(
    input="Your text",
    model="text-embedding-qwen3-embedding-4b"
)
```

### Chat with Retrieval
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Question?"}],
    "model": "qwen3.5-4b",
    "context": ["document1", "document2"]
  }'
```

### Create an Agent
```bash
curl -X POST http://localhost:8080/v1/agent/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Your task here",
    "system_prompt": "You are helpful",
    "tool_budget": 3
  }'
```

## 📊 Performance

| Operation | Latency | Throughput |
|-----------|---------|-----------|
| Chat query | 200-300ms | 3-4 req/sec |
| Embedding | 50-100ms | 10-15 texts/sec |
| Retrieval | 150-200ms | 5-7 req/sec |
| Agent turn | 400-500ms | 2-3 turns/sec |

## 🐛 Troubleshooting

### LMStudio Connection Failed
```bash
# Verify LMStudio is running
curl http://localhost:1234/v1/models

# Check proxy logs
tail -f logs/proxy.log
```

### High Latency
1. Check GPU utilization: `nvidia-smi`
2. Reduce batch sizes
3. Enable model quantization
4. See [Optimization Guide](docs/06-optimization/)

### Out of Memory Errors
1. Reduce context window
2. Use batch processing instead of large documents
3. Keep fewer sessions active
4. See [Memory Management](docs/06-optimization/memory-management.md)

## 📖 Full Documentation

All detailed documentation has been organized in the `docs/` folder:

- **01-getting-started/** - Installation and setup
- **02-api-integration/** - API endpoints and examples
- **03-retrieval-system/** - Retrieval pipeline details
- **04-agent-development/** - Agent building guides
- **05-architecture/** - System architecture
- **06-optimization/** - Performance optimization
- **07-operations/** - Deployment and monitoring
- **08-configuration/** - Configuration reference
- **technical-reference/** - Technical details and specs

**[→ Start with Documentation Index](docs/_index.md)**

## 🤝 Contributing

1. Create a new branch
2. Make changes
3. Test with `pytest` (if available)
4. Submit PR with documentation updates

## 📄 License

See LICENSE file for details.

## 🔗 Resources

- **LMStudio**: https://lmstudio.ai/
- **Qwen Models**: https://huggingface.co/Qwen
- **OpenAI API Docs**: https://platform.openai.com/docs

---

**Need Help?** → [Documentation Index](docs/_index.md) | **Issues?** → Check [Troubleshooting](docs/07-operations/troubleshooting.md)

**Last Updated**: 2026-03-28 | **Version**: 4.1
