"""Pydantic v2 request and response schemas for the EcoSort ML API."""
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class TopKItem(BaseModel):
    label: str
    category: str
    probability: float = Field(..., ge=0.0, le=1.0)


class CandidateGuidanceItem(BaseModel):
    category: str
    detected_item: str
    recyclability_class: str
    bin: str
    disposal_tips: List[str]
    prep_steps: List[str]
    notes: str


class PredictionResponse(BaseModel):
    prediction_id: str
    status: Literal["ok", "uncertain"]
    detected_item: Optional[str] = None
    category: Optional[str] = None
    predicted_top1: Optional[str] = None
    recyclability_class: Literal["Recyclable", "Compostable", "Non-recyclable", "Special handling", "Unknown"]
    bin: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence_level: Literal["HIGH", "MEDIUM", "LOW"]
    top_k: List[TopKItem] = Field(default_factory=list)
    probabilities: Optional[Dict[str, float]] = None
    explanation: Optional[str] = None
    classification_method: Optional[str] = "image_model"
    disposal_tips: List[str] = Field(default_factory=list)
    prep_steps: List[str] = Field(default_factory=list)
    safe_handling_advice: List[str] = Field(default_factory=list)
    candidate_guidance: Optional[Dict[str, CandidateGuidanceItem]] = None
    notes: str
    retake_advice: Optional[str] = None
    quality_warnings: List[str] = Field(default_factory=list)
    entropy: Optional[float] = None
    is_near_uniform: bool = False
    inference_ms: float
    model_version: str
    is_smoke_test: bool = False
    is_demo: bool = False
    corrected_category: Optional[str] = None
    original_filename: str
    file_sha256: str
    image_width: int
    image_height: int
    has_thumbnail: bool
    created_at: datetime


class ClassifyTextRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=200)


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class BatchItemResult(BaseModel):
    filename: str
    success: bool
    prediction: Optional[PredictionResponse] = None
    error: Optional[ErrorDetail] = None


class BatchPredictionResponse(BaseModel):
    total: int
    succeeded: int
    failed: int
    results: List[BatchItemResult]


class PaginatedPredictionsResponse(BaseModel):
    items: List[PredictionResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class FeedbackRequest(BaseModel):
    corrected_category: Literal["plastic", "paper", "metal", "glass", "organic", "other"]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "model_not_loaded"]
    model_loaded: bool
    model_version: str
    db_connected: bool
    is_smoke_test: bool = False
    smoke_test_warning: Optional[str] = None


class ModelInfoResponse(BaseModel):
    model_loaded: bool
    model_version: str
    is_smoke_test: bool
    architecture: str
    classes: List[str]
    thresholds: Dict[str, float]
    temperature: float
    input_size: int
    model_card: Dict[str, Any]
    metrics: Optional[Dict[str, Any]] = None
    disclaimer: str


class CategoryStat(BaseModel):
    category: str
    count: int
    percentage: float
    avg_confidence: float


class RecyclabilityStat(BaseModel):
    recyclability_class: str
    count: int
    percentage: float


class TimeSeriesPoint(BaseModel):
    date: str
    count: int
    diverted: int
    uncertain: int


class ConfidenceBinStat(BaseModel):
    bucket: str
    count: int


class ItemFrequencyStat(BaseModel):
    item: str
    count: int
    category: str


class AnalyticsSummaryResponse(BaseModel):
    total_predictions: int
    average_confidence: float
    uncertain_count: int
    uncertain_rate: float
    recyclable_count: int
    compostable_count: int
    special_waste_count: int = 0
    recyclable_or_compostable_rate: float
    feedback_count: int
    primary_stream_counts: Dict[str, int] = Field(default_factory=dict)
    frequently_classified_items: List[ItemFrequencyStat] = Field(default_factory=list)
    category_distribution: List[CategoryStat]
    recyclability_distribution: List[RecyclabilityStat]
    time_series: List[TimeSeriesPoint]
    confidence_histogram: List[ConfidenceBinStat]
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    exclude_demo: bool = True
