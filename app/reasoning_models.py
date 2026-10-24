"""Immutable analysis/plans and separate mutable execution results."""
import re
from string import Formatter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Constrained(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Entity(Constrained):
    name: str = Field(min_length=1, max_length=500)
    type: str = Field(min_length=1, max_length=100)


class InformationDependency(Constrained):
    entity: str
    depends_on: tuple[str, ...]


class StructuralAnalysis(Constrained):
    original_question: str
    core_query: str = Field(min_length=1, max_length=4000)
    known_entities: tuple[Entity, ...]
    unknown_entities: tuple[Entity, ...]
    query_type: Literal["direct", "parallel", "sequential", "mixed"]
    information_dependencies: tuple[InformationDependency, ...]
    requires_decomposition: bool


class PlanNode(Constrained):
    node_id: str = Field(pattern=r"^N\d+$")
    node_type: Literal["retrieval", "integration"]
    question_template: str = Field(min_length=1, max_length=4000)
    dependencies: tuple[str, ...] = ()


class ReasoningPlan(Constrained):
    original_question: str
    root_node_id: str
    nodes: dict[str, PlanNode]


class SemanticCheck(Constrained):
    valid: bool
    errors: tuple[str, ...]


class SupportQuote(Constrained):
    evidence_id: str
    quote: str = Field(min_length=1, max_length=18000)


class GroundedAnswer(Constrained):
    status: Literal["answered", "partial", "unanswerable"]
    answer: str = Field(max_length=6000, description="Evidence-only answer with separate citation labels, e.g. CVE-2099-4242 [E1].")
    # A concise verbatim value is required before a retrieval answer can be substituted.
    value: str = Field(max_length=1000, description="Concise verbatim answer/identifier copied from a supporting quote; empty for unanswerable or integration.")
    evidence_ids: tuple[str, ...] = Field(description="Bare IDs of actually supporting evidence, e.g. [\"E1\"], without brackets in the string.")
    supporting_quotes: tuple[SupportQuote, ...] = Field(description="Short exact excerpts from each cited source that support the answer and substitution value.")
    contradictions: tuple[str, ...] = ()


class SupportVerdict(Constrained):
    supported: bool
    reason: str


class RecoveryDecision(Constrained):
    can_retrieve_independently: bool
    query: str = Field(max_length=4000)
    reason: str


class NodeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node_id: str
    status: Literal["pending", "running", "answered", "partial", "unanswerable", "blocked", "failed"] = "pending"
    answer: str = ""
    value: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    unresolved_dependencies: list[str] = Field(default_factory=list)
    error: str | None = None
    failure_stage: str | None = None
    resolved_question: str | None = None
    substitutions: dict[str, str] = Field(default_factory=dict)
    attempts: int = 0
    recovery_attempts: int = 0
    seconds: float = 0
    contradictions: list[str] = Field(default_factory=list)
    supporting_quotes: list[SupportQuote] = Field(default_factory=list)


def placeholders(template):
    fields = []
    for _, name, spec, conversion in Formatter().parse(template):
        if name is not None:
            if not re.fullmatch(r"N\d+", name) or spec or conversion:
                raise ValueError("Templates only support plain {NodeID} placeholders")
            fields.append(name)
    return fields


def validate_analysis(analysis, question):
    if analysis.original_question != question:
        raise ValueError("Analysis changed original_question")
    for entity in analysis.known_entities:
        if entity.name not in question:
            raise ValueError(f"Known entity was not provided verbatim: {entity.name}")
    unknown = {e.name for e in analysis.unknown_entities}
    known = {e.name for e in analysis.known_entities}
    if len(unknown) != len(analysis.unknown_entities):
        raise ValueError("Unknown entity names must be unique")
    links = {}
    for link in analysis.information_dependencies:
        if link.entity not in unknown or not set(link.depends_on).issubset(unknown | known):
            raise ValueError(f"Dependency entity={link.entity!r} must name an unknown fact; depends_on={link.depends_on!r} "
                             f"must use declared unknown or provided known names: {sorted(unknown | known)}")
        if link.entity in links or len(set(link.depends_on)) != len(link.depends_on):
            raise ValueError("Duplicate information dependency")
        links[link.entity] = link.depends_on
    active, done = set(), set()
    def visit(name):
        if name in active:
            raise ValueError("Information dependencies contain a cycle")
        if name not in done:
            active.add(name)
            for dep in links.get(name, ()):
                visit(dep)
            active.remove(name); done.add(name)
    for name in unknown:
        visit(name)
    if (analysis.query_type == "direct") != (not analysis.requires_decomposition):
        raise ValueError("Direct/decomposition classification is inconsistent")
    if analysis.query_type in {"direct", "parallel"} and any(dep in unknown for deps in links.values() for dep in deps):
        raise ValueError("Independent questions cannot contain sequential dependencies")


def validate_plan(plan, config, question=None, analysis=None):
    """Deterministic checks must finish before any retrieval call."""
    if question is not None and plan.original_question != question:
        raise ValueError("Plan changed original_question")
    nodes = plan.nodes
    if not nodes or len(nodes) > config.max_nodes or len(nodes) > config.max_execution_steps:
        raise ValueError("Plan exceeds node-count or execution-step budget")
    if plan.root_node_id not in nodes:
        raise ValueError("Root node does not exist")
    if len({node.node_id for node in nodes.values()}) != len(nodes):
        raise ValueError("Node IDs must be unique")
    for key, node in nodes.items():
        if node.node_type not in {"retrieval", "integration"} or not node.question_template.strip():
            raise ValueError("Unsupported node type or empty question template")
        if key != node.node_id:
            raise ValueError("Node map keys must match node IDs")
        if len(node.dependencies) != len(set(node.dependencies)):
            raise ValueError("Duplicate dependency")
        if not set(node.dependencies).issubset(nodes):
            raise ValueError("Missing dependency node")
        if not set(placeholders(node.question_template)).issubset(node.dependencies):
            raise ValueError("Each placeholder must refer to a declared dependency")
        if node.node_type == "integration" and not node.dependencies:
            raise ValueError("Integration nodes require dependencies")
    depths, active = {}, set()
    def visit(key):
        if key in active:
            raise ValueError("Plan contains a cycle")
        if key not in depths:
            active.add(key)
            depths[key] = 1 + max((visit(dep) for dep in nodes[key].dependencies), default=0)
            active.remove(key)
        return depths[key]
    depth = visit(plan.root_node_id) - 1
    if len(depths) != len(nodes):
        raise ValueError("Plan contains unreachable nodes")
    if depth > config.max_depth:
        raise ValueError("Plan exceeds maximum dependency depth (edges from root to leaf)")
    if analysis and not analysis.requires_decomposition:
        if len(nodes) != 1 or nodes[plan.root_node_id].node_type != "retrieval":
            raise ValueError("Direct questions must use a single retrieval node")
    return {"valid": True, "node_count": len(nodes), "depth": depth, "reachable": sorted(depths)}


def resolve_question(node, results):
    substitutions = {}
    for dep in placeholders(node.question_template):
        result = results[dep]
        if result.status != "answered" or not result.value or not result.evidence_ids:
            raise ValueError(f"Dependency {dep} lacks a grounded substitution value")
        if (not any(" ".join(result.value.split()) in " ".join(q.quote.split()) for q in result.supporting_quotes)
                or re.search(r"[{}\[\]\r\n]", result.value)):
            raise ValueError(f"Dependency {dep} has an unverified substitution value")
        substitutions[dep] = result.value
    question = node.question_template.format_map(substitutions)
    if len(question) > 4000:
        raise ValueError("Resolved query exceeds the 4,000-character limit")
    return question, substitutions
