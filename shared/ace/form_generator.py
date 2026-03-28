"""
Generates dynamic forms and templates based on agent analysis.
Creates context-aware fields and intelligent defaults.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from shared.ace.context_analyzer import AgentProfile

@dataclass
class ContextField:
    """A single context field for form generation."""
    name: str  # Field identifier
    label: str  # User-facing label
    type_: str  # "text", "select", "checkbox", "textarea"
    placeholder: Optional[str] = None
    default_value: Any = None
    options: List[Any] = field(default_factory=list)
    required: bool = True


@dataclass  
class ACEContextTemplate:
    """Complete context template for an agent session."""
    fields: List[ContextField]
    system_prompt_override: Optional[str] = None
    initial_context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class FormGenerator:
    """Generates dynamic forms based on agent analysis."""
    
    DEFAULT_FIELDS = [
        ContextField(name="topic", label="Topic or Question", type_="text"),
        ContextField(name="constraints", label="Key Constraints", type_="textarea"),
        ContextField(name="goals", label="Primary Goals", type_="textarea"),
        ContextField(name="format", label="Output Format", type_="select", 
                     options=["text", "json", "markdown", "table"]),
    ]
    
    ROLE_SPECIFIC_FIELDS = {
        "researcher": [
            ContextField(name="source_quality", label="Source Quality Level", 
                        type_="select", options=["high", "medium", "low"]),
            ContextField(name="depth", label="Analysis Depth", 
                        type_="select", options=["surface", "moderate", "deep"]),
        ],
        "coder": [
            ContextField(name="language", label="Programming Language", 
                        type_="select", options=["python", "javascript", "rust", "go", "c++"]),
            ContextField(name="style", label="Code Style", 
                        type_="select", options=["pep8", "google", "airbnb"]),
        ],
    }
    
    def generate(self, agent_profile: AgentProfile) -> ACEContextTemplate:
        """Generate context template from agent profile."""
        base_fields = self.DEFAULT_FIELDS.copy()
        
        # Add role-specific fields
        if agent_profile.role_type in self.ROLE_SPECIFIC_FIELDS:
            for field_def in self.ROLE_SPECIFIC_FIELDS[agent_profile.role_type]:
                field = ContextField(
                    name=field_def.name,
                    label=field_def.label,
                    type_=field_def.type_,
                    options=field_def.options if hasattr(field_def, 'options') else [],
                )
                base_fields.append(field)
        
        # Generate defaults from system prompt analysis
        defaults = self._generate_defaults(agent_profile)
        
        # Update fields with defaults
        for field in base_fields:
            if field.name in defaults:
                field.default_value = defaults[field.name]
                if hasattr(field, 'options') and field.name in defaults:
                    field.options = [str(opt) for opt in defaults[field.name]]
        
        # Create system prompt override if available
        system_override = None
        if agent_profile.role_type != "general":
            role_prefixes = {
                "researcher": "As a research assistant, ",
                "coder": "As a software engineer, ",
                "writer": "As a content writer, ",
                "analyst": "As an analyst, ",
            }
            system_override = role_prefixes.get(agent_profile.role_type) or ""
        
        return ACEContextTemplate(
            fields=base_fields,
            system_prompt_override=system_override,
            metadata={
                "agent_role": agent_profile.role_type,
                "domain": agent_profile.domain,
                "constraints_count": len(agent_profile.constraints),
            }
        )
    
    def _generate_defaults(self, profile: AgentProfile) -> Dict[str, Any]:
        """Generate default values from system prompt analysis."""
        defaults = {}
        
        # Generate topic placeholder from goals
        if profile.goals and len(profile.goals) > 0:
            defaults["topic"] = f"Regarding {profile.goals[0]}"
        
        # Generate constraints text
        if profile.constraints:
            defaults["constraints"] = "\n".join(f"- {c}" for c in profile.constraints[:3])
        
        # Format-specific defaults
        if profile.output_format == "json":
            defaults["format"] = "json"
        elif profile.output_format == "markdown":
            defaults["format"] = "markdown"
        
        return defaults


# Singleton instance
_generator = FormGenerator()

def generate_context(agent_profile: AgentProfile) -> ACEContextTemplate:
    """Convenience function to generate context template."""
    return _generator.generate(agent_profile)