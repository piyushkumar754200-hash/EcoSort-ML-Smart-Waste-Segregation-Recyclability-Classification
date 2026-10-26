"""FastAPI route handlers for waste classification, history, feedback, analytics, and CSV export."""
import asyncio
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from sqlalchemy import asc, case, desc, func, or_, select
from sqlalchemy.orm import Session

from app.config import settings
from app.core.security import (
    ImageValidationError,
    sanitize_csv_cell,
    save_sanitized_thumbnail,
    validate_and_sanitize_image,
)
from app.core.taxonomy import TaxonomyConfig, load_taxonomy
from app.db.session import get_db
from app.ml.inference import classifier_service
from app.models.prediction import Prediction
from app.schemas.api import (
    AnalyticsSummaryResponse,
    BatchItemResult,
    BatchPredictionResponse,
    CandidateGuidanceItem,
    ClassifyTextRequest,
    ErrorDetail,
    FeedbackRequest,
    HealthResponse,
    ModelInfoResponse,
    PaginatedPredictionsResponse,
    PredictionResponse,
    TopKItem,
)

router = APIRouter(prefix="/api")

_taxonomy_cache: Optional[TaxonomyConfig] = None

UNCERTAIN_SAFE_HANDLING_ADVICE: List[str] = [
    "Do not place an unverified item into a recycling or compost bin — material contamination can spoil an entire recycling batch.",
    "Check the item manually for a plastic resin code (#1–#7), metal stamp (ALU/FE), or packaging label.",
    "Retake the photo on a plain neutral surface in bright, even lighting, or select the confirmed material category below.",
]


def get_taxonomy() -> TaxonomyConfig:
    global _taxonomy_cache
    if _taxonomy_cache is None:
        _taxonomy_cache = load_taxonomy(settings.TAXONOMY_YAML_PATH)
    return _taxonomy_cache


def api_error_response(status_code: int, code: str, message: str, details: Optional[dict] = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "details": details or {},
            }
        },
    )


def _build_candidate_guidance(taxonomy: TaxonomyConfig) -> Dict[str, CandidateGuidanceItem]:
    out: Dict[str, CandidateGuidanceItem] = {}
    for cat_key in taxonomy.categories:
        entry = taxonomy.get_entry(cat_key)
        out[cat_key] = CandidateGuidanceItem(
            category=entry.category,
            detected_item=entry.detected_item,
            recyclability_class=entry.recyclability_class,
            bin=entry.bin,
            disposal_tips=entry.disposal_tips,
            prep_steps=entry.prep_steps,
            notes=entry.notes,
        )
    return out


def _row_to_response(
    row: Prediction,
    taxonomy: TaxonomyConfig,
    uncertainty_reason: Optional[str] = None,
) -> PredictionResponse:
    is_uncertain_unresolved = row.status == "uncertain" and not row.corrected_category
    candidate_map = _build_candidate_guidance(taxonomy)
    top_k_items = [TopKItem(**item) for item in (row.top_k or [])]
    is_near_uniform = bool(
        len(top_k_items) >= 3
        and (float(top_k_items[0].probability) - float(top_k_items[2].probability)) < 0.10
    )

    orig_name = row.original_filename or ""
    is_text_search = (
        orig_name.startswith("text:")
        or orig_name.startswith("catalog:")
        or orig_name.startswith("search:")
    )
    class_method = "text_search" if is_text_search else "image_model"

    dir_item = taxonomy.search_material(row.detected_item) if row.detected_item else None
    if not dir_item and is_text_search:
        query_term = orig_name.split(":", 1)[-1]
        dir_item = taxonomy.search_material(query_term)

    if is_uncertain_unresolved:
        default_retake = (
            "The model could not distinguish between these materials. Please retake the photo with a single item centered against a plain, neutral background."
            if is_near_uniform
            else "Confidence or top-2 class margin is below the safety threshold. Please retake the photo with the single waste item centered against a plain, neutral background in bright, even lighting."
        )
        retake_advice = uncertainty_reason or default_retake
        return PredictionResponse(
            prediction_id=row.id,
            status="uncertain",
            detected_item=None,
            category=None,
            predicted_top1=row.predicted_top1,
            recyclability_class="Unknown",
            bin=None,
            confidence=round(float(row.confidence), 4),
            confidence_level=row.confidence_level,  # type: ignore[arg-type]
            top_k=top_k_items,
            classification_method=class_method,
            explanation="Uncertain classification. Follow safe handling advice to prevent contamination.",
            disposal_tips=[],
            prep_steps=[],
            safe_handling_advice=UNCERTAIN_SAFE_HANDLING_ADVICE,
            candidate_guidance=candidate_map,
            notes="Prediction confidence or top-2 margin is below threshold; category-specific disposal instructions are withheld until the item is retaken or manually confirmed.",
            retake_advice=retake_advice,
            quality_warnings=row.quality_warnings or [],
            is_near_uniform=is_near_uniform,
            inference_ms=round(float(row.inference_ms), 2),
            model_version=row.model_version,
            is_smoke_test="SMOKE-TEST" in row.model_version.upper() or classifier_service.is_smoke_test,
            is_demo=bool(row.is_demo),
            corrected_category=row.corrected_category,
            original_filename=row.original_filename,
            file_sha256=row.file_sha256,
            image_width=row.image_width,
            image_height=row.image_height,
            has_thumbnail=bool(row.thumbnail_path),
            created_at=row.created_at,
        )

    effective_cat = row.corrected_category or row.category or row.predicted_top1 or (dir_item.category if dir_item else "plastic")
    try:
        entry = taxonomy.get_entry(effective_cat)
    except KeyError:
        entry = taxonomy.get_entry("plastic")

    detected_name = dir_item.name if dir_item else (row.detected_item or entry.detected_item)
    explanation = dir_item.explanation if dir_item else f"Classified as {effective_cat.capitalize()} waste."
    disposal_tips = dir_item.disposal_tips if (dir_item and dir_item.disposal_tips) else entry.disposal_tips
    prep_steps = dir_item.prep_steps if (dir_item and dir_item.prep_steps) else entry.prep_steps
    bin_name = row.bin or (dir_item.bin if dir_item else None) or entry.bin
    notes_text = f"{dir_item.explanation} (Local acceptance: {dir_item.local_acceptance})" if dir_item else entry.notes

    return PredictionResponse(
        prediction_id=row.id,
        status="uncertain" if row.status == "uncertain" else "ok",
        detected_item=detected_name,
        category=row.corrected_category or row.category or (dir_item.category if dir_item else entry.category),
        predicted_top1=row.predicted_top1 or row.category,
        recyclability_class=entry.recyclability_class if row.corrected_category else (row.recyclability_class or entry.recyclability_class),  # type: ignore[arg-type]
        bin=bin_name,
        confidence=round(float(row.confidence), 4),
        confidence_level=row.confidence_level,  # type: ignore[arg-type]
        top_k=[TopKItem(**item) for item in (row.top_k or [])],
        classification_method=class_method,
        explanation=explanation,
        disposal_tips=disposal_tips,
        prep_steps=prep_steps,
        safe_handling_advice=[],
        candidate_guidance=candidate_map,
        notes=notes_text,
        retake_advice=None,
        quality_warnings=row.quality_warnings or [],
        inference_ms=round(float(row.inference_ms), 2),
        model_version=row.model_version,
        is_smoke_test="SMOKE-TEST" in row.model_version.upper() or classifier_service.is_smoke_test,
        is_demo=bool(row.is_demo),
        corrected_category=row.corrected_category,
        original_filename=row.original_filename,
        file_sha256=row.file_sha256,
        image_width=row.image_width,
        image_height=row.image_height,
        has_thumbnail=bool(row.thumbnail_path),
        created_at=row.created_at,
    )


@router.get("/health", response_model=HealthResponse)
def get_health(db: Session = Depends(get_db)):
    db_ok = False
    try:
        db.execute(select(func.count(Prediction.id)))
        db_ok = True
    except Exception:
        db_ok = False

    loaded = classifier_service.ensure_loaded()
    smoke_warning = (
        "SMOKE-TEST, NOT FOR PRODUCTION: Model was trained on a minimal subset for pipeline verification."
        if (loaded and classifier_service.is_smoke_test)
        else None
    )
    status_str = "ok" if (loaded and db_ok) else ("model_not_loaded" if not loaded else "degraded")

    return HealthResponse(
        status=status_str,
        model_loaded=loaded,
        model_version=classifier_service.model_version if loaded else "model not loaded",
        db_connected=db_ok,
        is_smoke_test=classifier_service.is_smoke_test if loaded else False,
        smoke_test_warning=smoke_warning,
    )


@router.get("/model/info", response_model=ModelInfoResponse)
def get_model_info():
    loaded = classifier_service.ensure_loaded()
    taxonomy = get_taxonomy()
    metrics_data: Optional[Dict[str, Any]] = None
    metrics_path = Path(settings.METRICS_JSON_PATH)
    if metrics_path.exists():
        try:
            with metrics_path.open("r", encoding="utf-8") as f:
                metrics_data = json.load(f)
        except Exception:
            metrics_data = None

    card = classifier_service.model_card if loaded else {}
    return ModelInfoResponse(
        model_loaded=loaded,
        model_version=classifier_service.model_version if loaded else "model not loaded",
        is_smoke_test=classifier_service.is_smoke_test if loaded else False,
        architecture=card.get("architecture", "efficientnet_b0"),
        classes=classifier_service.class_names,
        thresholds={
            "high": settings.CONFIDENCE_HIGH,
            "medium": settings.CONFIDENCE_MIN,
            "min": settings.CONFIDENCE_MIN,
            "margin_min": settings.MARGIN_MIN,
        },
        temperature=classifier_service.temperature,
        input_size=classifier_service.image_size,
        model_card=card,
        metrics=metrics_data,
        disclaimer=taxonomy.disclaimer,
    )


async def _process_single_upload(
    upload: UploadFile,
    db: Session,
    taxonomy: TaxonomyConfig,
) -> PredictionResponse:
    raw_bytes = await upload.read()
    validated = validate_and_sanitize_image(raw_bytes, original_filename=upload.filename)

    inf_res = await asyncio.to_thread(classifier_service.predict, validated.clean_rgb_image)
    tax_entry = taxonomy.get_entry(inf_res.top_label)
    thumb_name = await asyncio.to_thread(save_sanitized_thumbnail, validated.clean_rgb_image)

    warnings = list(validated.quality_warnings)
    if inf_res.status == "uncertain":
        reason_prefix = inf_res.uncertainty_reason or f"Low prediction certainty ({inf_res.confidence * 100:.1f}%)."
        warnings.append(
            f"{reason_prefix} Top candidates: "
            + ", ".join(f"{item['label']} ({item['probability'] * 100:.1f}%)" for item in inf_res.top_k)
        )

    is_uncertain = inf_res.status == "uncertain"
    record = Prediction(
        original_filename=validated.sanitized_filename,
        file_sha256=validated.file_sha256,
        detected_item=None if is_uncertain else tax_entry.detected_item,
        category=None if is_uncertain else tax_entry.category,
        predicted_top1=tax_entry.category,
        recyclability_class="Unknown" if is_uncertain else tax_entry.recyclability_class,
        bin=None if is_uncertain else tax_entry.bin,
        confidence=inf_res.confidence,
        confidence_level=inf_res.confidence_level,
        status=inf_res.status,
        top_k=inf_res.top_k,
        quality_warnings=warnings,
        model_version=inf_res.model_version,
        inference_ms=inf_res.inference_ms,
        image_width=validated.width,
        image_height=validated.height,
        thumbnail_path=thumb_name,
        is_demo=False,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    resp = _row_to_response(record, taxonomy, uncertainty_reason=inf_res.uncertainty_reason)
    resp.probabilities = inf_res.probabilities
    return resp


@router.post("/predict", response_model=PredictionResponse)
@router.post("/classify/single", response_model=PredictionResponse)
async def predict_waste(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    db: Session = Depends(get_db),
):
    if not classifier_service.ensure_loaded():
        return api_error_response(
            status_code=503,
            code="MODEL_NOT_LOADED",
            message="Model not loaded. Train and export the ONNX model before requesting predictions.",
        )

    if file is None:
        return api_error_response(
            status_code=400,
            code="MISSING_FILE",
            message="Multipart form field 'file' is required.",
        )

    taxonomy = get_taxonomy()
    try:
        return await _process_single_upload(file, db, taxonomy)
    except ImageValidationError as exc:
        return api_error_response(exc.status_code, exc.code, exc.message, exc.details)


@router.post("/classify/text", response_model=PredictionResponse)
def classify_text_query(
    payload: ClassifyTextRequest,
    db: Session = Depends(get_db),
):
    query = payload.query.strip()
    if not query:
        return api_error_response(400, "EMPTY_QUERY", "Search query cannot be empty.")

    taxonomy = get_taxonomy()
    matched = taxonomy.search_material(query)

    if matched:
        record = Prediction(
            id=str(uuid.uuid4()),
            created_at=datetime.now(timezone.utc),
            original_filename=f"catalog:{matched.name}",
            file_sha256=hashlib.sha256(matched.name.encode("utf-8")).hexdigest(),
            detected_item=matched.name,
            category=matched.category,
            predicted_top1=matched.category,
            recyclability_class=matched.recyclability_class,
            bin=matched.bin,
            confidence=0.99,
            confidence_level="HIGH",
            status="ok",
            top_k=[{"category": matched.category, "label": matched.name, "probability": 0.99}],
            quality_warnings=[],
            model_version=f"taxonomy-{taxonomy.version}",
            inference_ms=1.2,
            image_width=0,
            image_height=0,
            thumbnail_path=None,
            is_demo=False,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        resp = _row_to_response(record, taxonomy)
        resp.explanation = matched.explanation
        resp.classification_method = "text_search"
        resp.prep_steps = matched.prep_steps
        resp.disposal_tips = matched.disposal_tips
        return resp

    # Safe handling fallback for unknown or ambiguous search
    record = Prediction(
        id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc),
        original_filename=f"search:{query[:60]}",
        file_sha256=hashlib.sha256(query.encode("utf-8")).hexdigest(),
        detected_item=None,
        category=None,
        predicted_top1=None,
        recyclability_class="Unknown",
        bin=None,
        confidence=0.0,
        confidence_level="LOW",
        status="uncertain",
        top_k=[],
        quality_warnings=[
            f"'{query}' was not found in the verified material catalog or special waste safety directory."
        ],
        model_version=f"taxonomy-{taxonomy.version}",
        inference_ms=1.0,
        image_width=0,
        image_height=0,
        thumbnail_path=None,
        is_demo=False,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    resp = _row_to_response(
        record,
        taxonomy,
        uncertainty_reason=f"'{query}' is ambiguous or unverified. Never place unverified items in curbside recycling or compost bins.",
    )
    resp.explanation = "Item is not listed in the verified catalog. Consult local municipal waste authorities or inspect the product package for standard recycling codes."
    resp.classification_method = "text_search"
    return resp


@router.get("/taxonomy")
def get_taxonomy_endpoint():
    tax = get_taxonomy()
    return {
        "version": tax.version,
        "disclaimer": tax.disclaimer,
        "categories": tax.categories,
        "recyclability_classes": tax.recyclability_classes,
        "primary_streams": {k: v.model_dump() for k, v in tax.primary_streams.items()},
        "special_waste_streams": {k: v.model_dump() for k, v in tax.special_waste_streams.items()},
        "material_directory": [m.model_dump() for m in tax.material_directory],
        "common_mistakes": [m.model_dump() for m in tax.common_mistakes],
    }


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_waste_batch(
    request: Request,
    files: Optional[List[UploadFile]] = File(default=None),
    db: Session = Depends(get_db),
):
    if not classifier_service.ensure_loaded():
        return api_error_response(
            status_code=503,
            code="MODEL_NOT_LOADED",
            message="Model not loaded. Train and export the ONNX model before requesting batch predictions.",
        )

    if not files or len(files) == 0:
        return api_error_response(
            status_code=400,
            code="EMPTY_BATCH",
            message="At least 1 image file must be provided in 'files'.",
        )

    if len(files) > 20:
        return api_error_response(
            status_code=400,
            code="BATCH_LIMIT_EXCEEDED",
            message="Batch predictions accept a maximum of 20 images per request.",
            details={"submitted_count": len(files), "max_batch_size": 20},
        )

    taxonomy = get_taxonomy()
    results: List[BatchItemResult] = []
    succeeded = 0
    failed = 0

    for upload in files:
        fname = upload.filename or "upload.jpg"
        try:
            pred = await _process_single_upload(upload, db, taxonomy)
            results.append(BatchItemResult(filename=fname, success=True, prediction=pred))
            succeeded += 1
        except ImageValidationError as exc:
            results.append(
                BatchItemResult(
                    filename=fname,
                    success=False,
                    error=ErrorDetail(code=exc.code, message=exc.message, details=exc.details),
                )
            )
            failed += 1
        except Exception as exc:
            results.append(
                BatchItemResult(
                    filename=fname,
                    success=False,
                    error=ErrorDetail(code="PROCESSING_ERROR", message=str(exc)),
                )
            )
            failed += 1

    return BatchPredictionResponse(
        total=len(files),
        succeeded=succeeded,
        failed=failed,
        results=results,
    )


def _apply_prediction_filters(
    stmt,
    category: Optional[str] = None,
    recyclability_class: Optional[str] = None,
    status: Optional[str] = None,
    min_confidence: Optional[float] = None,
    max_confidence: Optional[float] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    search: Optional[str] = None,
    include_demo: bool = True,
):
    if not include_demo:
        stmt = stmt.where(Prediction.is_demo.is_(False))
    if category:
        cat_clean = category.strip().lower()
        if cat_clean == "uncertain":
            stmt = stmt.where(Prediction.category.is_(None))
        else:
            stmt = stmt.where(
                or_(
                    Prediction.category == cat_clean,
                    Prediction.corrected_category == cat_clean,
                )
            )
    if recyclability_class:
        stmt = stmt.where(Prediction.recyclability_class == recyclability_class.strip())
    if status:
        stmt = stmt.where(Prediction.status == status.strip().lower())
    if min_confidence is not None:
        stmt = stmt.where(Prediction.confidence >= min_confidence)
    if max_confidence is not None:
        stmt = stmt.where(Prediction.confidence <= max_confidence)
    if date_from:
        try:
            dt_from = datetime.fromisoformat(date_from.replace("Z", "+00:00"))
            stmt = stmt.where(Prediction.created_at >= dt_from)
        except ValueError:
            pass
    if date_to:
        try:
            dt_to = datetime.fromisoformat(date_to.replace("Z", "+00:00"))
            if len(date_to) <= 10:
                dt_to = dt_to + timedelta(days=1)
            stmt = stmt.where(Prediction.created_at <= dt_to)
        except ValueError:
            pass
    if search and search.strip():
        q = f"%{search.strip()}%"
        stmt = stmt.where(
            or_(
                Prediction.original_filename.ilike(q),
                Prediction.detected_item.ilike(q),
                Prediction.category.ilike(q),
                Prediction.recyclability_class.ilike(q),
                Prediction.id.ilike(q),
            )
        )
    return stmt


@router.get("/predictions", response_model=PaginatedPredictionsResponse)
@router.get("/history", response_model=PaginatedPredictionsResponse)
def list_predictions(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=15, ge=1, le=100),
    sort_by: str = Query(default="created_at"),
    sort_order: str = Query(default="desc"),
    category: Optional[str] = Query(default=None),
    recyclability_class: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    min_confidence: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    max_confidence: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    date_from: Optional[str] = Query(default=None),
    date_to: Optional[str] = Query(default=None),
    search: Optional[str] = Query(default=None),
    include_demo: bool = Query(default=True),
    db: Session = Depends(get_db),
):
    base_stmt = _apply_prediction_filters(
        select(Prediction),
        category=category,
        recyclability_class=recyclability_class,
        status=status,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
        date_from=date_from,
        date_to=date_to,
        search=search,
        include_demo=include_demo,
    )

    count_stmt = select(func.count()).select_from(base_stmt.subquery())
    total = int(db.execute(count_stmt).scalar_one() or 0)

    sort_map = {
        "created_at": Prediction.created_at,
        "confidence": Prediction.confidence,
        "category": Prediction.category,
        "recyclability_class": Prediction.recyclability_class,
        "inference_ms": Prediction.inference_ms,
        "original_filename": Prediction.original_filename,
    }
    sort_col = sort_map.get(sort_by, Prediction.created_at)
    order_fn = asc if sort_order.lower() == "asc" else desc

    rows = (
        db.execute(
            base_stmt.order_by(order_fn(sort_col))
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        .scalars()
        .all()
    )

    taxonomy = get_taxonomy()
    total_pages = max(1, (total + page_size - 1) // page_size)
    return PaginatedPredictionsResponse(
        items=[_row_to_response(r, taxonomy) for r in rows],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get("/predictions/{prediction_id}", response_model=PredictionResponse)
def get_prediction_by_id(prediction_id: str, db: Session = Depends(get_db)):
    row = db.get(Prediction, prediction_id)
    if not row:
        return api_error_response(404, "NOT_FOUND", f"Prediction '{prediction_id}' not found.")
    return _row_to_response(row, get_taxonomy())


@router.delete("/predictions/{prediction_id}")
def delete_prediction_by_id(prediction_id: str, db: Session = Depends(get_db)):
    row = db.get(Prediction, prediction_id)
    if not row:
        return api_error_response(404, "NOT_FOUND", f"Prediction '{prediction_id}' not found.")

    if row.thumbnail_path:
        thumb_file = Path(settings.THUMBNAIL_DIR) / Path(row.thumbnail_path).name
        if thumb_file.exists():
            try:
                thumb_file.unlink()
            except OSError:
                pass

    db.delete(row)
    db.commit()
    return {"deleted": True, "prediction_id": prediction_id}


@router.delete("/predictions")
@router.delete("/history")
def bulk_delete_predictions(
    confirm: bool = Query(default=False),
    demo_only: bool = Query(default=False),
    db: Session = Depends(get_db),
):
    if not confirm:
        return api_error_response(
            400,
            "CONFIRMATION_REQUIRED",
            "Bulk deletion requires query parameter '?confirm=true'.",
        )

    stmt = select(Prediction)
    if demo_only:
        stmt = stmt.where(Prediction.is_demo.is_(True))
    rows = db.execute(stmt).scalars().all()
    deleted_count = len(rows)

    for row in rows:
        if row.thumbnail_path:
            thumb_file = Path(settings.THUMBNAIL_DIR) / Path(row.thumbnail_path).name
            if thumb_file.exists():
                try:
                    thumb_file.unlink()
                except OSError:
                    pass
        db.delete(row)

    db.commit()
    return {"deleted": True, "deleted_count": deleted_count, "demo_only": demo_only}


@router.get("/predictions/{prediction_id}/image")
def get_prediction_image(prediction_id: str, db: Session = Depends(get_db)):
    row = db.get(Prediction, prediction_id)
    if not row:
        return api_error_response(404, "NOT_FOUND", f"Prediction '{prediction_id}' not found.")
    if not row.thumbnail_path:
        return api_error_response(404, "IMAGE_NOT_STORED", "No thumbnail stored for this prediction.")

    safe_name = Path(row.thumbnail_path).name
    thumb_file = Path(settings.THUMBNAIL_DIR) / safe_name
    if not thumb_file.exists():
        return api_error_response(404, "IMAGE_FILE_MISSING", "Stored thumbnail file could not be found on disk.")

    return FileResponse(thumb_file, media_type="image/jpeg")


@router.post("/predictions/{prediction_id}/feedback", response_model=PredictionResponse)
def submit_prediction_feedback(
    prediction_id: str,
    payload: FeedbackRequest,
    db: Session = Depends(get_db),
):
    row = db.get(Prediction, prediction_id)
    if not row:
        return api_error_response(404, "NOT_FOUND", f"Prediction '{prediction_id}' not found.")

    taxonomy = get_taxonomy()
    row.corrected_category = payload.corrected_category
    if payload.corrected_category in taxonomy.categories:
        entry = taxonomy.get_entry(payload.corrected_category)
        row.category = entry.category
        row.detected_item = entry.detected_item
        row.recyclability_class = entry.recyclability_class
        row.bin = entry.bin
    db.commit()
    db.refresh(row)
    return _row_to_response(row, taxonomy)


@router.get("/analytics/summary", response_model=AnalyticsSummaryResponse)
def get_analytics_summary(
    date_from: Optional[str] = Query(default=None),
    date_to: Optional[str] = Query(default=None),
    include_demo: bool = Query(default=False),
    db: Session = Depends(get_db),
):
    filtered_stmt = _apply_prediction_filters(
        select(Prediction),
        date_from=date_from,
        date_to=date_to,
        include_demo=include_demo,
    )
    subq = filtered_stmt.subquery()

    kpi_stmt = select(
        func.count(subq.c.id).label("total"),
        func.coalesce(func.avg(subq.c.confidence), 0.0).label("avg_conf"),
        func.sum(case((subq.c.status == "uncertain", 1), else_=0)).label("uncertain_cnt"),
        func.sum(case((subq.c.recyclability_class == "Recyclable", 1), else_=0)).label("recyclable_cnt"),
        func.sum(case((subq.c.recyclability_class == "Compostable", 1), else_=0)).label("compostable_cnt"),
        func.sum(case((subq.c.category == "special_waste", 1), else_=0)).label("special_waste_cnt"),
        func.sum(case((subq.c.category == "plastic", 1), else_=0)).label("plastic_cnt"),
        func.sum(case((subq.c.category == "paper", 1), else_=0)).label("paper_cnt"),
        func.sum(case((subq.c.category == "metal", 1), else_=0)).label("metal_cnt"),
        func.sum(case((subq.c.category == "glass", 1), else_=0)).label("glass_cnt"),
        func.sum(case((subq.c.category == "organic", 1), else_=0)).label("organic_cnt"),
        func.sum(case((subq.c.corrected_category.is_not(None), 1), else_=0)).label("feedback_cnt"),
    )
    kpi_row = db.execute(kpi_stmt).one()

    total = int(kpi_row.total or 0)
    avg_conf = round(float(kpi_row.avg_conf or 0.0), 4)
    uncertain_cnt = int(kpi_row.uncertain_cnt or 0)
    recyclable_cnt = int(kpi_row.recyclable_cnt or 0)
    compostable_cnt = int(kpi_row.compostable_cnt or 0)
    special_waste_cnt = int(kpi_row.special_waste_cnt or 0)
    feedback_cnt = int(kpi_row.feedback_cnt or 0)

    uncertain_rate = round(uncertain_cnt / total, 4) if total > 0 else 0.0
    diversion_rate = round((recyclable_cnt + compostable_cnt) / total, 4) if total > 0 else 0.0

    cat_expr = func.coalesce(subq.c.corrected_category, subq.c.category, "uncertain")
    cat_stmt = (
        select(
            cat_expr.label("category"),
            func.count(subq.c.id).label("count"),
            func.avg(subq.c.confidence).label("avg_confidence"),
        )
        .group_by(cat_expr)
        .order_by(desc("count"))
    )
    cat_rows = db.execute(cat_stmt).all()
    category_distribution = [
        {
            "category": str(r.category or "uncertain"),
            "count": int(r.count),
            "percentage": round(int(r.count) / total * 100.0, 2) if total > 0 else 0.0,
            "avg_confidence": round(float(r.avg_confidence or 0.0), 4),
        }
        for r in cat_rows
    ]

    rec_stmt = (
        select(
            subq.c.recyclability_class,
            func.count(subq.c.id).label("count"),
        )
        .group_by(subq.c.recyclability_class)
        .order_by(desc("count"))
    )
    rec_rows = db.execute(rec_stmt).all()
    recyclability_distribution = [
        {
            "recyclability_class": str(r.recyclability_class or "Unknown"),
            "count": int(r.count),
            "percentage": round(int(r.count) / total * 100.0, 2) if total > 0 else 0.0,
        }
        for r in rec_rows
    ]

    freq_stmt = (
        select(
            func.coalesce(subq.c.detected_item, subq.c.original_filename, subq.c.category).label("item"),
            func.coalesce(subq.c.category, "uncertain").label("category"),
            func.count(subq.c.id).label("count"),
        )
        .where(subq.c.status != "uncertain")
        .group_by("item", "category")
        .order_by(desc("count"))
        .limit(10)
    )
    freq_rows = db.execute(freq_stmt).all()
    frequently_classified_items = [
        {
            "item": str(r.item).replace("catalog:", "").replace("text:", "").replace("search:", ""),
            "count": int(r.count),
            "category": str(r.category),
        }
        for r in freq_rows
    ]

    day_expr = func.substr(func.cast(subq.c.created_at, Prediction.id.type), 1, 10)
    ts_stmt = (
        select(
            day_expr.label("date"),
            func.count(subq.c.id).label("count"),
            func.sum(
                case(
                    (subq.c.recyclability_class.in_(["Recyclable", "Compostable"]), 1),
                    else_=0,
                )
            ).label("diverted"),
            func.sum(case((subq.c.status == "uncertain", 1), else_=0)).label("uncertain"),
        )
        .group_by(day_expr)
        .order_by(asc("date"))
    )
    ts_rows = db.execute(ts_stmt).all()
    time_series = [
        {
            "date": str(r.date),
            "count": int(r.count),
            "diverted": int(r.diverted or 0),
            "uncertain": int(r.uncertain or 0),
        }
        for r in ts_rows
    ]

    bins = [
        ("0.0-0.1", 0.0, 0.1),
        ("0.1-0.2", 0.1, 0.2),
        ("0.2-0.3", 0.2, 0.3),
        ("0.3-0.4", 0.3, 0.4),
        ("0.4-0.5", 0.4, 0.5),
        ("0.5-0.6", 0.5, 0.6),
        ("0.6-0.7", 0.6, 0.7),
        ("0.7-0.8", 0.7, 0.8),
        ("0.8-0.9", 0.8, 0.9),
        ("0.9-1.0", 0.9, 1.0001),
    ]
    hist_selects = [
        func.sum(
            case(
                ((subq.c.confidence >= low) & (subq.c.confidence < high), 1),
                else_=0,
            )
        ).label(label)
        for label, low, high in bins
    ]
    hist_row = db.execute(select(*hist_selects)).one()
    confidence_histogram = [
        {"bucket": label, "count": int(getattr(hist_row, label) or 0)}
        for label, _, _ in bins
    ]

    return AnalyticsSummaryResponse(
        total_predictions=total,
        average_confidence=avg_conf,
        uncertain_count=uncertain_cnt,
        uncertain_rate=uncertain_rate,
        recyclable_count=recyclable_cnt,
        compostable_count=compostable_cnt,
        special_waste_count=special_waste_cnt,
        recyclable_or_compostable_rate=diversion_rate,
        feedback_count=feedback_cnt,
        primary_stream_counts={
            "plastic": int(kpi_row.plastic_cnt or 0),
            "paper": int(kpi_row.paper_cnt or 0),
            "metal": int(kpi_row.metal_cnt or 0),
            "glass": int(kpi_row.glass_cnt or 0),
            "organic": int(kpi_row.organic_cnt or 0),
        },
        frequently_classified_items=frequently_classified_items,
        category_distribution=category_distribution,
        recyclability_distribution=recyclability_distribution,
        time_series=time_series,
        confidence_histogram=confidence_histogram,
        date_from=date_from,
        date_to=date_to,
        exclude_demo=not include_demo,
    )


@router.get("/export/csv")
def export_predictions_csv(
    category: Optional[str] = Query(default=None),
    recyclability_class: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    min_confidence: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    max_confidence: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    date_from: Optional[str] = Query(default=None),
    date_to: Optional[str] = Query(default=None),
    search: Optional[str] = Query(default=None),
    include_demo: bool = Query(default=True),
    bom: bool = Query(default=False),
    db: Session = Depends(get_db),
):
    stmt = _apply_prediction_filters(
        select(Prediction),
        category=category,
        recyclability_class=recyclability_class,
        status=status,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
        date_from=date_from,
        date_to=date_to,
        search=search,
        include_demo=include_demo,
    ).order_by(desc(Prediction.created_at))

    rows = db.execute(stmt).scalars().all()

    headers = [
        "prediction_id",
        "timestamp",
        "filename",
        "detected_item",
        "category",
        "recyclability_class",
        "confidence_score",
        "confidence_level",
        "status",
        "model_version",
    ]

    def generate_csv():
        buf = io.StringIO()
        if bom:
            buf.write("\ufeff")
        writer = csv.writer(buf, dialect="excel", quoting=csv.QUOTE_MINIMAL, lineterminator="\r\n")
        writer.writerow(headers)
        yield buf.getvalue()
        buf.seek(0)
        buf.truncate(0)

        for r in rows:
            ts_iso = (
                r.created_at.astimezone(timezone.utc).isoformat()
                if r.created_at.tzinfo
                else r.created_at.replace(tzinfo=timezone.utc).isoformat()
            )
            eff_cat = r.corrected_category or r.category or "Uncertain"
            eff_item = r.detected_item or ("Uncertain" if r.status == "uncertain" else eff_cat)
            writer.writerow(
                [
                    sanitize_csv_cell(r.id),
                    sanitize_csv_cell(ts_iso),
                    sanitize_csv_cell(r.original_filename),
                    sanitize_csv_cell(eff_item),
                    sanitize_csv_cell(eff_cat),
                    sanitize_csv_cell(r.recyclability_class or "Unknown"),
                    f"{float(r.confidence):.4f}",
                    sanitize_csv_cell(r.confidence_level),
                    sanitize_csv_cell(r.status),
                    sanitize_csv_cell(r.model_version),
                ]
            )
            yield buf.getvalue()
            buf.seek(0)
            buf.truncate(0)

    filename = f"ecosort_predictions_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        generate_csv(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
