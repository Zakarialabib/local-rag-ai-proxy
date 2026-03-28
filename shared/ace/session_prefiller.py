"""
Intelligent session prefill engine that merges historical patterns
with system prompt analysis for optimal context engineering.
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from shared.ace.context_analyzer import AgentProfile
from shared.ace.form_generator import generate_context


@dataclass
class SessionHistory:
    """Cached session history for pattern extraction."""
    session_id: str
    turn_count: int
    avg_context_length: float
    common_topics: List[str]
    preferred_formats: List[str]
    intervention_rate: float


class SessionPrefiller:
    """Intelligently pre-fills ACE sessions based on history + analysis."""
    
    def __init__(self, session_store):
        self.session_store = session_store
        
    def prefetch(self, agent_profile: AgentProfile) -> Dict[str, Any]:
        """Generate prefilled session data for an agent."""
        # Generate fresh context template from system prompt
        context_template = generate_context(agent_profile)
        
        # Load historical patterns if available
        history_patterns = self._load_history_patterns()
        
        # Merge patterns with fresh analysis
        merged_context = {
            "template": context_template,
            "historical_patterns": history_patterns,
            "suggested_topics": self._suggest_topics(agent_profile),
            "intervention_triggers": self._generate_intervention_triggers(
                agent_profile.constraints
            ),
        }
        
        return merged_context
    
    def _load_history_patterns(self) -> Dict[str, Any]:
        """Load patterns from historical ACE sessions."""
        try:
            # Load all ace_sessions from .gui_state/ace_sessions
            import os
            session_dir = ".gui_state/ace_sessions"
            
            if not os.path.exists(session_dir):
                return {}
            
            history = {
                "total_sessions": 0,
                "avg_context_length": 0.0,
                "common_topics": [],
                "preferred_formats": [],
            }
            
            for filename in os.listdir(session_dir):
                if not filename.endswith(".json"):
                    continue
                
                filepath = os.path.join(session_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                    
                    history["total_sessions"] += 1
                    
                    # Extract context information
                    if "context" in session_data:
                        ctx = session_data["context"]
                        if isinstance(ctx, dict):
                            for field_name, value in ctx.get("fields", {}).items():
                                if value and len(value) > 0:
                                    history["avg_context_length"] += len(str(value))
                                    
                                    # Track topics
                                    topic_keywords = self._extract_topics(value)
                                    for topic in topic_keywords[:3]:
                                        if topic not in history["common_topics"]:
                                            history["common_topics"].append(topic)
                                    
                                    # Track formats
                                    format_value = ctx.get("format", "text")
                                    if format_value not in history["preferred_formats"]:
                                        history["preferred_formats"].append(format_value)
                except Exception:
                    continue
            
            return {
                "total_sessions": history["total_sessions"],
                "avg_context_length": round(history["avg_context_length"] / 
                                            max(1, history["total_sessions"]), 2),
                "common_topics": list(set(history["common_topics"]))[:5],
                "preferred_formats": list(set(history["preferred_formats"]))[:3],
            }
        except Exception:
            return {}
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topic keywords from context."""
        # Simple keyword extraction - can be enhanced with NLP
        topic_keywords = [
            "research", "analysis", "investigation", "study", 
            "development", "implementation", "design", "architecture",
            "optimization", "performance", "bug", "fix",
            "documentation", "reporting", "summary", "overview"
        ]
        
        text_lower = text.lower()
        topics = [kw for kw in topic_keywords if kw in text_lower]
        return topics[:3]
    
    def _suggest_topics(self, agent_profile: AgentProfile) -> List[str]:
        """Suggest topics based on agent domain and history."""
        suggestions = []
        
        # Domain-based suggestions
        domain_suggestions = {
            "researcher": ["current trends", "recent findings", "methodology comparison"],
            "coder": ["best practices", "common pitfalls", "performance optimization"],
            "writer": ["target audience", "tone guidelines", "content structure"],
            "analyst": ["key metrics", "trend identification", "causal analysis"],
        }
        
        domain_suggestions = domain_suggestions.get(agent_profile.domain, [])
        suggestions.extend(domain_suggestions)
        
        # History-based suggestions
        history_patterns = self._load_history_patterns()
        if history_patterns.get("common_topics"):
            suggestions.extend(history_patterns["common_topics"])
        
        return list(set(suggestions))[:5]
    
    def _generate_intervention_triggers(self, constraints: List[str]) -> Dict[str, Any]:
        """Generate intervention trigger conditions based on constraints."""
        triggers = {
            "checklist": [],
            "auto_pause": False,
            "format_validation": False,
        }
        
        # Parse constraint keywords for triggers
        constraint_keywords = {
            "must not": ["pause", "abort"],
            "without": ["validate", "verify"],
            "ensure": ["check", "confirm"],
        }
        
        for keyword, action in constraint_keywords.items():
            for constraint in constraints:
                if keyword in constraint.lower():
                    triggers["checklist"].append(action)
                    break
        
        return triggers


# Singleton instance
_prefiller = SessionPrefiller(None)  # Will be injected with session_store

def prefetch_session(agent_profile: AgentProfile, session_store=None) -> Dict[str, Any]:
    """Convenience function to prefetch session data."""
    _prefiller.session_store = session_store if session_store else None
    return _prefiller.prefetch(agent_profile)