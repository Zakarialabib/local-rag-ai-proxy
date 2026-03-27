import re
import os
import json
import structlog
from typing import Dict, List, Any

logger = structlog.get_logger()

DOMAIN_PATTERNS = {
    "code": r"(?i)(code|programming|function|class|def |import |=>|->|{|}|syntax)",
    "json": r"(?i)(json|api|rest|graphql|endpoint|webhook)",
    "data": r"(?i)(data|database|sql|query|table|schema|record)",
    "business": r"(?i)(revenue|profit|customer|sales|metric|report|quarter|annual)",
    "reasoning": r"(?i)(why|how|explain|think|reason|analyze|compare)",
}

DOMAIN_SYSTEM_PROMPTS = {
    "code": "You are an expert programmer. Provide clean, well-commented code. Format code blocks properly.",
    "json": "You are a data architect. Always respond with valid JSON when requested. No markdown around JSON.",
    "data": "You are a data analyst. Provide structured insights with clear metrics and breakdowns.",
    "business": "You are a business analyst. Provide concise, actionable insights with key metrics.",
    "reasoning": "You are a logical reasoning assistant. Show your work step-by-step.",
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

class DomainRouter:
    def detect_domain(self, text: str) -> str:
        """Detect the domain of the input based on patterns."""
        scores = {}
        for domain, pattern in DOMAIN_PATTERNS.items():
            matches = len(re.findall(pattern, text))
            scores[domain] = matches
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
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
