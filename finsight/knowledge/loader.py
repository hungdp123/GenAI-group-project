"""Load and validate the knowledge layer YAML. Owner: Person 5 (Hung)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from finsight.contracts.schemas import SubClaimType, KnowledgeRule

DEFAULT_PATH = Path(__file__).with_name("knowledge_layer.yaml")


class KnowledgeLayerError(ValueError):
    """Raised when the knowledge layer YAML is malformed. Fail loudly at startup, never mid-run."""


class KnowledgeLayer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    rules: list[KnowledgeRule]
    fallback: KnowledgeRule

    @model_validator(mode="after")
    def _check(self) -> "KnowledgeLayer":
        seen: set[SubClaimType] = set()
        for rule in self.rules:
            if rule.sub_claim_type in seen:
                raise ValueError(f"duplicate rule for {rule.sub_claim_type.value}")
            seen.add(rule.sub_claim_type)
            if rule.highest_verdict is None and not rule.refuse:
                raise ValueError(f"{rule.sub_claim_type.value}: highest_verdict is required unless refuse is true")
        return self

    def rule_for(self, sub_claim_type: SubClaimType) -> KnowledgeRule:
        """Rule for `sub_claim_type`, or the fallback row."""
        for rule in self.rules:
            if rule.sub_claim_type is sub_claim_type:
                return rule
        return self.fallback.model_copy(update={"sub_claim_type": sub_claim_type})


def load_knowledge_layer(path: str | Path | None = None) -> KnowledgeLayer:
    """Parse and validate the YAML. Unknown sub-claim types, verdicts, tiers or fields raise KnowledgeLayerError."""
    path = Path(path) if path else DEFAULT_PATH
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return KnowledgeLayer.model_validate(data)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise KnowledgeLayerError(f"Invalid knowledge layer {path}: {exc}") from exc


@lru_cache(maxsize=1)
def default_knowledge_layer() -> KnowledgeLayer:
    return load_knowledge_layer()
