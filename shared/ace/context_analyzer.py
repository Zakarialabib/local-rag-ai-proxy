"""
Intelligent system prompt analyzer for extracting agent characteristics.
Uses regex patterns, NLP heuristics, and domain recognition.
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class AgentProfile:
    """Extracted agent profile from system prompt."""
    role_type: str  # "researcher", "coder", "writer", "analyst", etc.
    domain: str     # primary area of expertise
    constraints: List[str]  # limitations, rules to follow
    goals: List[str]   # objectives the agent pursues
    tools_required: List[str]  # expected tool usage
    output_format: str  # preferred response format
    tone: str  # communication style


class SystemPromptAnalyzer:
    """Analyzes system prompts to extract agent characteristics."""
    
    ROLE_PATTERNS = {
        "researcher": r"(?:research|investigate|explore|discover|study)",
        "coder": r"(?:code|program|develop|build|implement|debug)",
        "writer": r"(?:write|compose|draft|author|create content)",
        "analyst": r"(?:analyze|evaluate|assess|interpret|summarize)",
        "advisor": r"(?:advise|consult|guide|mentor|coach)",
    }
    
    CONSTRAINT_PATTERNS = [
        r"must not\s+(?:be|do)\s+(\w+)",
        r"avoid(?:ing)?\s+(\w+)",
        r"never\s+(?:say|use|mention)\s+(\w+)",
        r"without\s+(?:permission|approval)\s+\(\w+\)",
    ]
    
    GOAL_PATTERNS = [
        r"(?:aim to|seek to|strive for|work towards)\s+(\w+)",
        r"focus on\s+(\w+)",
        r"ensure\s+(?:accurate|complete|comprehensive)\s+(\w+)",
    ]
    
    def analyze(self, system_prompt: str) -> AgentProfile:
        """Extract agent profile from system prompt."""
        normalized = self._normalize_prompt(system_prompt)
        
        # Extract role type
        role_type = "general"
        for agent_type, pattern in self.ROLE_PATTERNS.items():
            if re.search(pattern, normalized, re.IGNORECASE):
                role_type = agent_type
                break
        
        # Extract domain from context mentions
        domain = self._extract_domain(normalized)
        
        # Extract constraints
        constraints = []
        for pattern in self.CONSTRAINT_PATTERNS:
            matches = re.findall(pattern, normalized, re.IGNORECASE)
            constraints.extend(matches)
        
        # Extract goals
        goals = []
        for pattern in self.GOAL_PATTERNS:
            matches = re.findall(pattern, normalized, re.IGNORECASE)
            goals.extend(matches)
        
        # Detect output format hints
        output_format = "text"
        if any(keyword in normalized.lower() for keyword in 
                ["json", "xml", "table", "code block", "markdown"]):
            output_format = self._detect_output_format(normalized)
        
        # Detect tone
        tone = self._detect_tone(normalized)
        
        return AgentProfile(
            role_type=role_type,
            domain=domain or "general",
            constraints=constraints[:5],  # Top 5 constraints
            goals=goals[:3],  # Top 3 goals
            tools_required=self._inferred_tools(role_type),
            output_format=output_format,
            tone=tone,
        )
    
    def _normalize_prompt(self, prompt: str) -> str:
        """Clean and normalize the system prompt."""
        text = re.sub(r'\s+', ' ', prompt.strip())
        return text.lower()
    
    def _extract_domain(self, normalized: str) -> str:
        """Extract domain from context in system prompt."""
        domain_keywords = {
            "software": ["code", "program", "bug", "deploy", "api"],
            "science": ["experiment", "hypothesis", "data", "theory"],
            "business": ["strategy", "market", "revenue", "profit"],
            "education": ["student", "lesson", "curriculum", "exam"],
            "healthcare": ["patient", "diagnosis", "treatment", "medication"],
        }
        
        for domain, keywords in domain_keywords.items():
            if any(kw in normalized for kw in keywords):
                return domain
        
        # Fallback: extract from first few sentences
        sentences = re.split(r'[.!?]', normalized)[:2]
        combined = ' '.join(sentences)
        
        # Try to find a noun phrase that could be a domain
        words = re.findall(r'\b\w+\b', combined)
        if len(words) >= 3:
            return ' '.join(words[:3])
        
        return "general"
    
    def _detect_output_format(self, normalized: str) -> str:
        """Detect expected output format from prompt."""
        if re.search(r'json', normalized):
            return "json"
        elif re.search(r'markdown|md', normalized):
            return "markdown"
        elif re.search(r'table', normalized):
            return "table"
        else:
            return "text"
    
    def _inferred_tools(self, role_type: str) -> List[str]:
        """Infer likely tool requirements from agent type."""
        tool_map = {
            "researcher": ["search", "read", "write"],
            "coder": ["code_editor", "terminal", "git", "linter"],
            "writer": ["document_generator", "editor", "spell_checker"],
            "analyst": ["data_processor", "chart_generator", "report_writer"],
        }
        return tool_map.get(role_type, [])
    
    def _detect_tone(self, normalized: str) -> str:
        """Detect communication tone from system prompt."""
        if any(word in normalized for word in ["professional", "formal", "official"]):
            return "professional"
        elif any(word in normalized for word in ["casual", "friendly", "informal"]):
            return "casual"
        elif any(word in normalized for word in ["technical", "expert", "advanced"]):
            return "technical"
        elif any(word in normalized for word in ["simple", "beginner", "easy"]):
            return "simple"
        else:
            return "neutral"


# Singleton instance for global access
_analyzer = SystemPromptAnalyzer()

def analyze_system_prompt(system_prompt: str) -> AgentProfile:
    """Convenience function to analyze system prompt."""
    return _analyzer.analyze(system_prompt)