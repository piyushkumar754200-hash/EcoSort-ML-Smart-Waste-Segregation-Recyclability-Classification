"""Taxonomy loader and startup validator for waste categories and recyclability rules."""
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
import yaml
from pydantic import BaseModel, Field, model_validator

REQUIRED_CATEGORIES = {"plastic", "paper", "metal", "glass", "organic"}
AllowedCategory = Literal["plastic", "paper", "metal", "glass", "organic"]
AllowedRecyclability = Literal["Recyclable", "Compostable", "Non-recyclable", "Special handling"]


class LabelTaxonomyEntry(BaseModel):
    category: str
    detected_item: str = Field(..., min_length=1)
    recyclability_class: str
    bin: str = Field(..., min_length=1)
    prep_steps: List[str] = Field(default_factory=list)
    disposal_tips: List[str] = Field(..., min_length=1)
    notes: str = ""


class PrimaryStreamInfo(BaseModel):
    name: str
    category: str
    recyclability_class: str
    bin: str
    badge_color: str = "slate"
    examples: List[str] = Field(default_factory=list)
    prep_steps: List[str] = Field(default_factory=list)
    contamination_guidance: List[str] = Field(default_factory=list)
    local_acceptance: str = ""


class SpecialWasteStreamInfo(BaseModel):
    name: str
    category: str = "special_waste"
    recyclability_class: str = "Special handling"
    recommended_collection: str
    warning: str
    safe_handling: List[str] = Field(default_factory=list)
    local_acceptance: str = ""


class MaterialDirectoryItem(BaseModel):
    id: str
    name: str
    aliases: List[str] = Field(default_factory=list)
    category: str
    primary_stream: str
    recyclability_class: str
    bin: str
    explanation: str
    prep_steps: List[str] = Field(default_factory=list)
    disposal_tips: List[str] = Field(default_factory=list)
    contamination_warnings: List[str] = Field(default_factory=list)
    local_acceptance: str = ""


class CommonMistakeItem(BaseModel):
    mistake: str
    consequence: str
    correct_action: str


class TaxonomyConfig(BaseModel):
    version: str
    disclaimer: str
    categories: List[str]
    recyclability_classes: List[str]
    primary_streams: Dict[str, PrimaryStreamInfo] = Field(default_factory=dict)
    special_waste_streams: Dict[str, SpecialWasteStreamInfo] = Field(default_factory=dict)
    material_directory: List[MaterialDirectoryItem] = Field(default_factory=list)
    common_mistakes: List[CommonMistakeItem] = Field(default_factory=list)
    labels: Dict[str, LabelTaxonomyEntry]

    @model_validator(mode="after")
    def verify_required_categories(self) -> "TaxonomyConfig":
        declared = set(self.categories)
        missing = REQUIRED_CATEGORIES - declared
        if missing:
            raise ValueError(f"Taxonomy missing required target categories: {sorted(missing)}")
        for req_cat in REQUIRED_CATEGORIES:
            if req_cat not in self.labels:
                raise ValueError(f"Taxonomy labels map must contain default entry for category '{req_cat}'")
        return self

    def get_entry(self, label: str) -> LabelTaxonomyEntry:
        normalized = label.strip().lower().replace(" ", "_").replace("-", "_")
        if normalized in self.labels:
            return self.labels[normalized]
        if label in self.labels:
            return self.labels[label]
        raise KeyError(f"Label '{label}' not found in taxonomy configuration")

    def search_material(self, query: str) -> Optional[MaterialDirectoryItem]:
        q = query.strip().lower()
        if not q:
            return None
        # 1. Exact match on id, name, or aliases
        for item in self.material_directory:
            if q == item.id.lower() or q == item.name.lower():
                return item
            if any(q == alias.lower() for alias in item.aliases):
                return item
        # 2. Substring match: query contains alias or alias contains query
        for item in self.material_directory:
            for alias in item.aliases:
                a = alias.lower()
                if a in q or q in a:
                    return item
            if q in item.name.lower() or item.name.lower() in q:
                return item
        return None


def load_taxonomy(yaml_path: str | Path) -> TaxonomyConfig:
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Taxonomy YAML file not found at: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("Taxonomy YAML root must be a mapping")
    return TaxonomyConfig.model_validate(raw)
