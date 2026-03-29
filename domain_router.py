import structlog
import numpy as np
import os
import json
from typing import Dict, List, Any, Optional, Callable

logger = structlog.get_logger()

# Standard domain prompts for optimization
DOMAIN_SYSTEM_PROMPTS = {
    "code": "You are an expert programmer. Provide clean, well-commented code. Format code blocks properly.",
    "json": "You are a data architect. Always respond with valid JSON when requested. No markdown around JSON.",
    "data": "You are a data analyst. Provide structured insights with clear metrics and breakdowns.",
    "business": "You are a business analyst. Provide concise, actionable insights with key metrics.",
    "reasoning": "You are a logical reasoning assistant. Show your work step-by-step.",
    "tool_heavy": "You are an agent with access to powerful tools. Plan your tool usage carefully and explain your steps.",
}

# Load custom prompts
PROMPTS_FILE = os.path.join(os.path.dirname(__file__), "custom_prompts.json")
if os.path.exists(PROMPTS_FILE):
    try:
        with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
            custom_prompts = json.load(f)
            DOMAIN_SYSTEM_PROMPTS.update(custom_prompts)
            logger.info("loaded_custom_prompts", count=len(custom_prompts))
    except Exception as e:
        logger.error("failed_to_load_custom_prompts", error=str(e))

# Semantic signatures for cognitive modes
COGNITIVE_SIGNATURES = {
    "code": [
        "Write a python function to sort a list",
        "Debug this javascript error with async await",
        "Refactor the database schema to improve performance",
        "How does this class inherit from the base implementation",
        "Create a bash script to automate deployment",
        "Syntax error in my react component",
        "Implement a REST API endpoint using FastAPI"
    ],
    "reasoning": [
        "Why did the results change in the last step?",
        "Compare these two architectural approaches for scalability",
        "Explain the logical implication of this decision",
        "Analyze the pros and cons of using microservices",
        "What are the first principles of this problem?",
        "Break down this complex question into logical steps",
        "Evaluate the trade-offs between speed and safety"
    ],
    "tool_heavy": [
        "Search the web for the latest documentation on MCP",
        "Browse the latest news about Qwen 3.5 models",
        "Fetch the content of this URL and summarize it",
        "Run a code interpreter to calculate the result",
        "Analyze my local workspace files to find common patterns",
        "Use the terminal to list all files in the directory",
        "GitHub repository search for similar implementations"
    ],
    "vision": [
        "Describe what you see in this image",
        "What is happening in this screenshot?",
        "Analyze the UI layout of this mockup and suggest fixes",
        "Extract text from this picture and format it as JSON",
        "Identify the objects in this photo",
        "Does this image contain any PII or sensitive data?"
    ],
    "business": [
        "What was our revenue for the last quarter?",
        "Project the sales growth for the next year",
        "Generate a report on customer churn metrics",
        "Calculate the ROI of this marketing campaign",
        "Identify key performance indicators for the team",
        "Summarize the annual market share report"
    ]
}

class DomainRouter:
    def __init__(self):
        self.signature_embeddings: Dict[str, np.ndarray] = {}
        self.use_semantic = False
    async def detect_domain_semantic(self, text: str, embed_func: Optional[Callable]) -> str:
        """Detect the cognitive domain using embedding similarity."""
        if not embed_func:
            return "general"

        if not self.signature_embeddings:
            # Pre-embed all signatures (this usually happens once on first hit)
            try:
                for domain, sigs in COGNITIVE_SIGNATURES.items():
                    embs = await embed_func(sigs)
                    if embs:
                        self.signature_embeddings[domain] = np.mean(embs, axis=0)
                logger.info("cognitive_signatures_embedded", domains=list(self.signature_embeddings.keys()))
            except Exception as e:
                logger.warning("signature_embedding_failed", error=str(e))
                return "general"

        if not self.signature_embeddings:
            return "general"

        # Embed target text
        try:
            target_emb_list = await embed_func([text])
            if not target_emb_list:
                return "general"
            target_emb = target_emb_list[0]
            
            # Normalize target
            target_norm = target_emb / np.linalg.norm(target_emb)
            
            scores = {}
            for domain, sig_emb in self.signature_embeddings.items():
                # Normalize signature
                sig_norm = sig_emb / np.linalg.norm(sig_emb)
                # Cosine similarity
                scores[domain] = np.dot(target_norm, sig_norm)
            
            best_domain = max(scores, key=scores.get)
            if scores[best_domain] > 0.35: # Sane threshold
                logger.info("cognitive_route_detected", domain=best_domain, score=float(scores[best_domain]))
                return best_domain
        except Exception as e:
            logger.warning("semantic_route_failed", error=str(e))
        
        return "general"

    def detect_domain(self, text: str) -> str:
        """Fallback regex detection for legacy support."""
        # Note: We keep this for internal logic that doesn't use the async loop
        return "general"

    def optimize_prompts(self, messages: list, model_id: str, domain: str) -> list:
        is_reasoning = any(x in model_id.lower() for x in ["r1", "reasoning", "thought"])
        
        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system:
            prompt = DOMAIN_SYSTEM_PROMPTS.get(domain, "You are a helpful AI assistant.")
            if is_reasoning:
                prompt += " You should provide detailed reasoning in <thinking> blocks before answering."
            messages.insert(0, {"role": "system", "content": prompt})
        else:
            for i, msg in enumerate(messages):
                if msg.get("role") == "system":
                    existing = msg.get("content", "")
                    if domain != "general" and domain not in existing.lower():
                        msg["content"] = f"{existing}\n\n{DOMAIN_SYSTEM_PROMPTS.get(domain, '')}"
                    break
        
        return messages

    def format_output(self, text: str, domain: str) -> str:
        """Format output based on detected domain."""
        if domain == "code":
            if "```" not in text and any(x in text for x in ["function", "class", "def ", "import "]):
                text = "```\n" + text.strip() + "\n```"
        elif domain == "business":
            lines = text.split('\n')
            formatted = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith(('•', '-', '*', '1.', '2.', '3.')):
                    if len(line) > 20:
                        line = "• " + line
                formatted.append(line)
            text = '\n'.join(formatted)
        return text
