"""Environment-based configuration for dependency-aware reasoning."""
import os

from pydantic import BaseModel, ConfigDict, Field


class ReasoningConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    enabled: bool = True
    max_depth: int = Field(default=3, ge=1, le=10)
    max_nodes: int = Field(default=10, ge=1, le=50)
    max_planning_attempts: int = Field(default=2, ge=1, le=5)
    max_execution_steps: int = Field(default=15, ge=1, le=100)
    max_parallel_nodes: int = Field(default=3, ge=1, le=10)
    timeout_seconds: float = Field(default=120, gt=0, le=3600)
    node_timeout_seconds: float = Field(default=60, gt=0, le=3600)
    max_node_attempts: int = Field(default=2, ge=1, le=3)
    recovery_enabled: bool = True
    max_recovery_attempts_per_node: int = Field(default=1, ge=0, le=3)
    max_total_recovery_attempts: int = Field(default=3, ge=0, le=10)
    tracing_enabled: bool = True
    # Optional LLM support checks are fallible and disabled by default.
    support_verifier: str = Field(default="none", pattern="^(none|llm)$")
    fallback_enabled: bool = False

    @classmethod
    def from_env(cls):
        return cls.model_validate({name: os.environ["REASONING_" + name.upper()]
                                   for name in cls.model_fields if "REASONING_" + name.upper() in os.environ})


def default_mode():
    mode = os.getenv("RAG_MODE", "baseline")
    if mode not in {"baseline", "focus", "tree"}:
        raise ValueError("RAG_MODE must be baseline, focus, or tree")
    return mode
