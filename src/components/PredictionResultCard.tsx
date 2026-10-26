import React, { useState } from 'react';
import {
  AlertTriangle,
  Camera,
  CheckCircle2,
  Eye,
  FileText,
  HelpCircle,
  Leaf,
  Package,
  Recycle,
  ShieldAlert,
  Trash2,
  Wine,
  Wrench,
} from 'lucide-react';
import { CandidateGuidanceItem, PredictionRecord } from '../types/api';
import { apiClient } from '../api/client';

interface Props {
  prediction: PredictionRecord;
  previewUrl?: string | null;
  onFeedbackUpdated?: (updated: PredictionRecord) => void;
}

const CATEGORY_META: Record<
  string,
  { label: string; colorClass: string; icon: React.ComponentType<{ className?: string }> }
> = {
  plastic: {
    label: 'Plastic',
    colorClass: 'text-sky-700 dark:text-sky-300',
    icon: Package,
  },
  paper: {
    label: 'Paper & Fiber',
    colorClass: 'text-amber-700 dark:text-amber-300',
    icon: FileText,
  },
  metal: {
    label: 'Metal',
    colorClass: 'text-slate-700 dark:text-slate-300',
    icon: Wrench,
  },
  glass: {
    label: 'Glass',
    colorClass: 'text-teal-700 dark:text-teal-300',
    icon: Wine,
  },
  organic: {
    label: 'Organic',
    colorClass: 'text-emerald-700 dark:text-emerald-300',
    icon: Leaf,
  },
  special_waste: {
    label: 'Special / Hazardous Waste',
    colorClass: 'text-rose-600 dark:text-rose-400',
    icon: ShieldAlert,
  },
};

const FALLBACK_CANDIDATE_GUIDANCE: Record<string, CandidateGuidanceItem> = {
  plastic: {
    category: 'plastic',
    detected_item: 'Plastic Container / Bottle',
    recyclability_class: 'Recyclable',
    bin: 'Blue Recycling Bin (Plastics & Containers)',
    prep_steps: [
      'Empty all liquid or food residue completely.',
      'Give the container a quick rinse; remove loose plastic film wraps.',
      'Crush bottles lengthwise if possible to save bin capacity and replace the cap.',
    ],
    disposal_tips: [
      'Check for resin codes #1 (PET) or #2 (HDPE), which are universally accepted in curbside programs.',
      'Do not bag recyclables in plastic grocery bags; keep items loose in the bin.',
    ],
    notes: 'Disposal guidance is general; local rules vary.',
  },
  paper: {
    category: 'paper',
    detected_item: 'Paper / Cardboard Fiber',
    recyclability_class: 'Recyclable',
    bin: 'Blue Recycling Bin (Clean Paper & Cardboard)',
    prep_steps: [
      'Flatten all cardboard boxes and remove plastic packing tape or bubble wrap.',
      'Ensure paper is dry and free of grease or food stains.',
    ],
    disposal_tips: [
      'Keep paper and cardboard dry; wet or grease-soaked fiber (like greasy pizza box bottoms) belongs in compost or trash.',
      'Remove plastic windows from envelopes if required by your municipality.',
    ],
    notes: 'Disposal guidance is general; local rules vary.',
  },
  metal: {
    category: 'metal',
    detected_item: 'Aluminum / Steel Can or Metal Packaging',
    recyclability_class: 'Recyclable',
    bin: 'Blue Recycling Bin (Metals & Cans)',
    prep_steps: [
      'Empty and rinse food or beverage residue from cans.',
      'Ensure aerosol cans are completely empty before recycling.',
    ],
    disposal_tips: [
      'Aluminum and steel cans can be recycled indefinitely without loss of material quality.',
      'Roll clean aluminum foil into a ball at least 2 inches in diameter before placing in the recycling bin.',
    ],
    notes: 'Disposal guidance is general; local rules vary.',
  },
  glass: {
    category: 'glass',
    detected_item: 'Glass Bottle / Jar',
    recyclability_class: 'Recyclable',
    bin: 'Blue / Grey Glass Recycling Bin (Bottles & Jars)',
    prep_steps: [
      'Empty and rinse bottles or jars; paper labels can remain attached.',
      'Remove metal or plastic caps and sort them separately.',
    ],
    disposal_tips: [
      'Only container glass (bottles and jars) belongs in curbside glass recycling.',
      'Never place broken window glass, mirrors, drinkware, or ceramics in the recycling bin.',
    ],
    notes: 'Disposal guidance is general; local rules vary.',
  },
  organic: {
    category: 'organic',
    detected_item: 'Food Scraps / Yard Organics',
    recyclability_class: 'Compostable',
    bin: 'Green Organics / Compost Bin',
    prep_steps: [
      'Remove all produce stickers, rubber bands, plastic twist-ties, and packaging.',
      'Drain excess liquids before placing scraps into your kitchen caddy or green bin.',
    ],
    disposal_tips: [
      'Use only certified BPI compostable liners or paper bags if liners are permitted locally.',
      'Never include pet waste, treated wood, or synthetic plastics in the organics stream.',
    ],
    notes: 'Disposal guidance is general; local rules vary.',
  },
};

const DEFAULT_SAFE_HANDLING_ADVICE: string[] = [
  'Do not guess or place an unverified item into a recycling or compost bin — contamination can spoil an entire batch of recyclables.',
  'Inspect the item for a resin code (#1–#7), material stamp (ALU, FE, GL), or packaging label.',
  'If the material cannot be verified locally, place it in the general waste stream or retake a clearer photo.',
];

export const PredictionResultCard: React.FC<Props> = ({
  prediction,
  previewUrl,
  onFeedbackUpdated,
}) => {
  const [showFeedbackForm, setShowFeedbackForm] = useState(false);
  const [selectedCorrection, setSelectedCorrection] = useState<string>(
    prediction.corrected_category || prediction.category || prediction.predicted_top1 || 'plastic'
  );
  const [previewedCandidate, setPreviewedCandidate] = useState<string | null>(null);
  const [submittingFeedback, setSubmittingFeedback] = useState(false);
  const [feedbackError, setFeedbackError] = useState<string | null>(null);
  const [feedbackSuccess, setFeedbackSuccess] = useState(false);

  const effectiveCategory = prediction.corrected_category || prediction.category;
  const isUncertainUnresolved = prediction.status === 'uncertain' && !prediction.corrected_category;

  const catMeta = (effectiveCategory && CATEGORY_META[effectiveCategory]) || {
    label: effectiveCategory || 'Uncertain',
    colorClass: 'text-slate-700 dark:text-slate-300',
    icon: Recycle,
  };
  const CatIcon = catMeta.icon;

  const confPct = (prediction.confidence * 100).toFixed(1);

  const handleFeedbackSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmittingFeedback(true);
    setFeedbackError(null);
    try {
      const updated = await apiClient.submitFeedback(prediction.prediction_id, selectedCorrection);
      setFeedbackSuccess(true);
      setShowFeedbackForm(false);
      setPreviewedCandidate(null);
      onFeedbackUpdated?.(updated);
    } catch (err: any) {
      setFeedbackError(err?.message || 'Failed to save feedback.');
    } finally {
      setSubmittingFeedback(false);
    }
  };

  const getCandidateGuidance = (cat: string): CandidateGuidanceItem | null => {
    const key = cat.toLowerCase();
    if (prediction.candidate_guidance && prediction.candidate_guidance[key]) {
      return prediction.candidate_guidance[key];
    }
    return FALLBACK_CANDIDATE_GUIDANCE[key] || null;
  };

  const activeCandidatePreview = previewedCandidate ? getCandidateGuidance(previewedCandidate) : null;
  const confirmedGuidance =
    prediction.corrected_category && prediction.status === 'uncertain'
      ? getCandidateGuidance(prediction.corrected_category)
      : null;

  const safeAdvice =
    prediction.safe_handling_advice && prediction.safe_handling_advice.length > 0
      ? prediction.safe_handling_advice
      : DEFAULT_SAFE_HANDLING_ADVICE;

  const thumbSrc =
    previewUrl ||
    (prediction.has_thumbnail ? `/api/predictions/${prediction.prediction_id}/image` : null);

  return (
    <div
      className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 transition-opacity duration-150"
      data-testid="prediction-result-card"
    >
      <div className="flex flex-col md:flex-row gap-6 items-start">
        {thumbSrc && (
          <div className="w-full md:w-44 h-44 rounded-lg overflow-hidden bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 shrink-0 flex items-center justify-center">
            <img
              src={thumbSrc}
              alt={`Classified waste item: ${prediction.original_filename}`}
              referrerPolicy="no-referrer"
              className="w-full h-full object-cover"
            />
          </div>
        )}

        <div className="flex-1 min-w-0 w-full">
          <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500 dark:text-slate-400 mb-2">
            <span className="truncate max-w-[220px] font-medium text-slate-700 dark:text-slate-300">
              {prediction.original_filename}
            </span>
            <span aria-hidden="true">·</span>
            <span className="font-mono tabular-nums">{prediction.inference_ms.toFixed(1)} ms</span>
            <span aria-hidden="true">·</span>
            <span className="font-mono tabular-nums">
              {prediction.image_width}×{prediction.image_height}px
            </span>
            <span aria-hidden="true">·</span>
            <span className="font-mono tabular-nums">v{prediction.model_version}</span>
          </div>

          {isUncertainUnresolved ? (
            <div className="mb-3" data-testid="uncertain-state-banner">
              <div className="flex items-center gap-2 text-amber-700 dark:text-amber-400">
                <HelpCircle className="w-6 h-6 shrink-0" />
                <h3 className="text-2xl font-bold tracking-tight">
                  Uncertain Classification ({confPct}%)
                </h3>
              </div>
              {(prediction.is_near_uniform ||
                (prediction.top_k.length >= 3 &&
                  prediction.top_k[0].probability - prediction.top_k[2].probability < 0.10)) && (
                <p
                  className="text-sm font-semibold text-amber-800 dark:text-amber-300 mt-1.5"
                  data-testid="near-equal-probabilities-msg"
                >
                  The model could not distinguish between these materials.
                </p>
              )}
              <p className="text-sm text-slate-700 dark:text-slate-300 mt-1.5">
                {prediction.retake_advice ||
                  'Confidence or top-2 class margin is below the safety threshold. Category-specific disposal instructions are hidden to prevent stream contamination.'}
              </p>
              <p className="text-xs text-slate-600 dark:text-slate-400 mt-1">
                Reflective metal, transparent plastic and glass, and crushed or piled items can look similar under uneven lighting, busy backgrounds, or low resolution.
              </p>
            </div>
          ) : (
            <div className="mb-3">
              <div className="flex flex-wrap items-center gap-2.5">
                <CatIcon className={`w-6 h-6 shrink-0 ${catMeta.colorClass}`} />
                <h3 className="text-2xl font-bold tracking-tight capitalize text-slate-900 dark:text-slate-100">
                  {effectiveCategory === 'special_waste' ? 'Special Waste' : effectiveCategory}
                </h3>
                {prediction.classification_method === 'text_search' && (
                  <span className="text-[11px] px-2.5 py-0.5 rounded-full font-medium bg-emerald-100 text-emerald-800 dark:bg-emerald-950/70 dark:text-emerald-300 border border-emerald-300 dark:border-emerald-800">
                    Verified Catalog Match
                  </span>
                )}
              </div>
              {prediction.explanation && (
                <div className="mt-2.5 p-3 rounded-lg bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700/60 text-xs sm:text-sm text-slate-700 dark:text-slate-300 leading-relaxed">
                  <strong className="text-slate-900 dark:text-slate-100 font-semibold">Classification Rationale: </strong>
                  {prediction.explanation}
                </div>
              )}
              {effectiveCategory === 'special_waste' && (
                <div className="mt-2 p-3 rounded-lg bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-800 text-xs sm:text-sm text-rose-900 dark:text-rose-200 flex items-start gap-2.5">
                  <ShieldAlert className="w-4 h-4 text-rose-600 dark:text-rose-400 shrink-0 mt-0.5" />
                  <div>
                    <strong className="font-semibold">Special Waste Safety Mandate: </strong>
                    Never place batteries, e-waste, expired medicines, broken glass, or hazardous chemicals into ordinary recycling or compost carts. Follow authorized collection channels.
                  </div>
                </div>
              )}
              <div className="flex flex-wrap items-center gap-2 text-sm text-slate-600 dark:text-slate-300 mt-2.5">
                <span>
                  Detected item:{' '}
                  <strong className="font-semibold text-slate-900 dark:text-slate-100">
                    {confirmedGuidance?.detected_item || prediction.detected_item}
                  </strong>
                </span>
                <span aria-hidden="true">·</span>
                <span className="inline-flex items-center gap-1 font-semibold text-emerald-700 dark:text-emerald-400">
                  {(confirmedGuidance?.recyclability_class || prediction.recyclability_class) ===
                    'Recyclable' && <Recycle className="w-4 h-4" />}
                  {(confirmedGuidance?.recyclability_class || prediction.recyclability_class) ===
                    'Compostable' && <Leaf className="w-4 h-4" />}
                  {(confirmedGuidance?.recyclability_class || prediction.recyclability_class) ===
                    'Non-recyclable' && <Trash2 className="w-4 h-4 text-red-600" />}
                  {(confirmedGuidance?.recyclability_class || prediction.recyclability_class) ===
                    'Special handling' && <ShieldAlert className="w-4 h-4 text-amber-600" />}
                  {confirmedGuidance?.recyclability_class || prediction.recyclability_class}
                </span>
                <span aria-hidden="true">·</span>
                <span>
                  Target Bin:{' '}
                  <strong className="font-semibold text-slate-900 dark:text-slate-100">
                    {confirmedGuidance?.bin || prediction.bin}
                  </strong>
                </span>
              </div>
            </div>
          )}

          <div className="mt-4 pt-4 border-t border-slate-200 dark:border-slate-800">
            <div className="flex items-center justify-between text-xs mb-1.5">
              <span className="font-medium text-slate-700 dark:text-slate-300">
                Calibrated Confidence · Level:{' '}
                <strong
                  className={
                    prediction.confidence_level === 'HIGH'
                      ? 'text-emerald-700 dark:text-emerald-400'
                      : prediction.confidence_level === 'MEDIUM'
                      ? 'text-amber-700 dark:text-amber-400'
                      : 'text-red-700 dark:text-red-400'
                  }
                >
                  {prediction.confidence_level}
                </strong>
              </span>
              <span className="font-mono font-semibold tabular-nums text-slate-900 dark:text-slate-100">
                {confPct}%
              </span>
            </div>
            <div
              className="w-full h-2.5 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden"
              role="progressbar"
              aria-label="Calibrated confidence score"
              aria-valuenow={Math.round(prediction.confidence * 100)}
              aria-valuemin={0}
              aria-valuemax={100}
            >
              <div
                data-testid="confidence-progress-bar"
                className={`h-full origin-left transition-transform duration-200 ${
                  prediction.confidence_level === 'HIGH'
                    ? 'bg-emerald-600 dark:bg-emerald-500'
                    : prediction.confidence_level === 'MEDIUM'
                    ? 'bg-amber-600 dark:bg-amber-500'
                    : 'bg-red-600 dark:bg-red-500'
                }`}
                style={{ transform: `scaleX(${Math.min(Math.max(prediction.confidence, 0.02), 1)})` }}
              />
            </div>
          </div>

          <div className="mt-4">
            <p className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-2">
              {isUncertainUnresolved
                ? 'Top-3 Candidate Probabilities (Click a candidate to preview its guidance)'
                : 'Top-3 Candidate Probabilities'}
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
              {prediction.top_k.map((item, idx) => {
                const candidateCat = (item.category || item.label).toLowerCase();
                const isPreviewActive = previewedCandidate === candidateCat;
                return isUncertainUnresolved ? (
                  <button
                    key={item.label}
                    type="button"
                    data-testid={`candidate-preview-btn-${candidateCat}`}
                    onClick={() =>
                      setPreviewedCandidate(isPreviewActive ? null : candidateCat)
                    }
                    className={`px-3 py-2 rounded-lg border text-left transition-colors flex flex-col gap-1 ${
                      isPreviewActive
                        ? 'bg-amber-50 dark:bg-amber-950/40 border-amber-400 dark:border-amber-600'
                        : 'bg-slate-50 dark:bg-slate-800/60 border-slate-200/80 dark:border-slate-800 hover:border-slate-300 dark:hover:border-slate-700'
                    }`}
                  >
                    <div className="w-full flex items-center justify-between">
                      <span className="text-xs font-medium capitalize text-slate-800 dark:text-slate-200">
                        {idx + 1}. {item.label}
                      </span>
                      <span className="text-xs font-mono font-semibold tabular-nums text-slate-900 dark:text-slate-100">
                        {(item.probability * 100).toFixed(1)}%
                      </span>
                    </div>
                    <span className="inline-flex items-center gap-1 text-[11px] font-medium text-emerald-700 dark:text-emerald-400">
                      <Eye className="w-3 h-3" />
                      {isPreviewActive ? `Hide ${item.label} guidance` : `View ${item.label} guidance`}
                    </span>
                  </button>
                ) : (
                  <div
                    key={item.label}
                    className="px-3 py-2 rounded-lg bg-slate-50 dark:bg-slate-800/60 border border-slate-200/80 dark:border-slate-800 flex items-center justify-between"
                  >
                    <span className="text-xs font-medium capitalize text-slate-700 dark:text-slate-300">
                      {idx + 1}. {item.label}
                    </span>
                    <span className="text-xs font-mono font-semibold tabular-nums text-slate-900 dark:text-slate-100">
                      {(item.probability * 100).toFixed(1)}%
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {prediction.quality_warnings && prediction.quality_warnings.length > 0 && (
        <div
          className="mt-5 p-3.5 rounded-lg bg-amber-50/80 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-800/60 text-amber-900 dark:text-amber-200 text-xs space-y-1"
          role="alert"
          data-testid="quality-warnings-alert"
        >
          <div className="flex items-center gap-2 font-semibold">
            <AlertTriangle className="w-4 h-4 shrink-0 text-amber-600 dark:text-amber-400" />
            <span>Image Quality & Certainty Diagnostics</span>
          </div>
          <ul className="list-disc list-inside space-y-0.5 pl-1">
            {prediction.quality_warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </div>
      )}

      {isUncertainUnresolved ? (
        <div className="mt-5 pt-5 border-t border-slate-200 dark:border-slate-800 space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="p-4 rounded-lg bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-800">
              <div className="flex items-center gap-2 text-xs font-semibold text-slate-900 dark:text-slate-100 mb-2">
                <Camera className="w-4 h-4 text-amber-600 dark:text-amber-400" />
                <h4>How to Retake a Clearer Photo</h4>
              </div>
              <ul className="space-y-1.5 text-xs text-slate-600 dark:text-slate-300 list-disc list-inside">
                <li>Place a single item on a plain, neutral surface (avoid busy grass or patterned backgrounds).</li>
                <li>Use bright, even lighting to reduce harsh specular glare on transparent containers.</li>
                <li>Fill 60–80% of the camera frame (include cap, neck, and base) at &ge; 224 px resolution.</li>
              </ul>
            </div>

            <div className="p-4 rounded-lg bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-800">
              <div className="flex items-center gap-2 text-xs font-semibold text-slate-900 dark:text-slate-100 mb-2">
                <ShieldAlert className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
                <h4>Safe Unverified Handling Advice</h4>
              </div>
              <ul className="space-y-1.5 text-xs text-slate-600 dark:text-slate-300">
                {safeAdvice.map((advice, idx) => (
                  <li key={idx} className="flex items-start gap-2">
                    <span className="text-emerald-600 font-bold">&bull;</span>
                    <span>{advice}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {activeCandidatePreview && (
            <div
              className="p-4 rounded-lg bg-amber-50/90 dark:bg-amber-950/40 border border-amber-300 dark:border-amber-700 space-y-3"
              data-testid="candidate-guidance-preview"
            >
              <div className="flex items-center justify-between gap-2">
                <p className="text-xs font-semibold text-amber-900 dark:text-amber-200">
                  Candidate Preview ({activeCandidatePreview.category.toUpperCase()}) — Based on your selection, not the AI&apos;s prediction
                </p>
                <span className="text-xs font-medium text-amber-800 dark:text-amber-300">
                  Target Bin: {activeCandidatePreview.bin}
                </span>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-700 dark:text-slate-200">
                <div>
                  <h5 className="font-semibold mb-1">
                    Candidate Preparation ({activeCandidatePreview.category})
                  </h5>
                  <ul className="list-disc list-inside space-y-1">
                    {activeCandidatePreview.prep_steps.map((s, i) => (
                      <li key={i}>{s}</li>
                    ))}
                  </ul>
                </div>
                <div>
                  <h5 className="font-semibold mb-1">
                    Candidate Disposal Rules ({activeCandidatePreview.category})
                  </h5>
                  <ul className="list-disc list-inside space-y-1">
                    {activeCandidatePreview.disposal_tips.map((t, i) => (
                      <li key={i}>{t}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          )}

          <div className="p-4 rounded-lg bg-emerald-50/60 dark:bg-emerald-950/20 border border-emerald-200 dark:border-emerald-800/60 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <h4 className="text-xs font-semibold text-slate-900 dark:text-slate-100">
                Know what this item is? Choose the correct category
              </h4>
              <p className="text-xs text-slate-600 dark:text-slate-400">
                Confirming the material records ground-truth feedback and unlocks verified disposal instructions.
              </p>
            </div>
            <form onSubmit={handleFeedbackSubmit} className="flex flex-wrap items-center gap-2 shrink-0">
              <label htmlFor={`uncertain-corr-${prediction.prediction_id}`} className="sr-only">
                Choose the correct category
              </label>
              <select
                id={`uncertain-corr-${prediction.prediction_id}`}
                data-testid="uncertain-category-select"
                value={selectedCorrection}
                onChange={(e) => setSelectedCorrection(e.target.value)}
                className="px-2.5 py-1.5 text-xs rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100"
              >
                <option value="plastic">Plastic</option>
                <option value="paper">Paper</option>
                <option value="metal">Metal</option>
                <option value="glass">Glass</option>
                <option value="organic">Organic</option>
              </select>
              <button
                type="submit"
                disabled={submittingFeedback}
                data-testid="uncertain-confirm-category-btn"
                className="px-3 py-1.5 text-xs font-medium text-white bg-emerald-600 hover:bg-emerald-700 rounded transition-colors whitespace-nowrap shrink-0"
              >
                {submittingFeedback ? 'Saving...' : 'Confirm Category'}
              </button>
              {feedbackError && <span className="text-xs text-red-600">{feedbackError}</span>}
            </form>
          </div>
        </div>
      ) : (
        <div
          className="mt-5 pt-5 border-t border-slate-200 dark:border-slate-800 grid grid-cols-1 md:grid-cols-2 gap-6"
          data-testid="definite-guidance-section"
        >
          {((confirmedGuidance?.prep_steps && confirmedGuidance.prep_steps.length > 0) ||
            (prediction.prep_steps && prediction.prep_steps.length > 0)) && (
            <div>
              <h4 className="text-xs font-semibold text-slate-900 dark:text-slate-100 mb-2">
                Preparation Before Disposal
              </h4>
              <ul className="space-y-1.5 text-sm text-slate-600 dark:text-slate-300">
                {(confirmedGuidance?.prep_steps || prediction.prep_steps).map((step, i) => (
                  <li key={i} className="flex items-start gap-2">
                    <CheckCircle2 className="w-4 h-4 text-emerald-600 dark:text-emerald-400 shrink-0 mt-0.5" />
                    <span>{step}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div>
            <h4 className="text-xs font-semibold text-slate-900 dark:text-slate-100 mb-2">
              Disposal Guidance & Contamination Rules
            </h4>
            <ul className="space-y-1.5 text-sm text-slate-600 dark:text-slate-300">
              {(confirmedGuidance?.disposal_tips || prediction.disposal_tips).map((tip, i) => (
                <li key={i} className="flex items-start gap-2">
                  <Recycle className="w-4 h-4 text-slate-500 dark:text-slate-400 shrink-0 mt-0.5" />
                  <span>{tip}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}

      <div className="mt-5 pt-4 border-t border-slate-200 dark:border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs text-slate-500 dark:text-slate-400">
        <div>
          <span>
            Note: Disposal guidance is general; local municipal recycling rules vary.
          </span>
          {prediction.corrected_category && (
            <span className="ml-2 font-medium text-emerald-700 dark:text-emerald-400">
              · User correction recorded: <strong className="capitalize">{prediction.corrected_category}</strong>
            </span>
          )}
          {feedbackSuccess && !prediction.corrected_category && (
            <span className="ml-2 font-medium text-emerald-700 dark:text-emerald-400">
              · Feedback saved for retraining review
            </span>
          )}
        </div>

        <div>
          {!showFeedbackForm ? (
            <button
              type="button"
              onClick={() => setShowFeedbackForm(true)}
              className="text-xs font-medium text-emerald-700 dark:text-emerald-400 hover:underline whitespace-nowrap shrink-0"
            >
              Wrong? Correct it
            </button>
          ) : (
            <form onSubmit={handleFeedbackSubmit} className="flex flex-wrap items-center gap-2">
              <label htmlFor={`corr-${prediction.prediction_id}`} className="sr-only">
                Correct category
              </label>
              <select
                id={`corr-${prediction.prediction_id}`}
                value={selectedCorrection}
                onChange={(e) => setSelectedCorrection(e.target.value)}
                className="px-2.5 py-1 text-xs rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100"
              >
                <option value="plastic">Plastic</option>
                <option value="paper">Paper</option>
                <option value="metal">Metal</option>
                <option value="glass">Glass</option>
                <option value="organic">Organic</option>
                <option value="other">Other / Non-recyclable</option>
              </select>
              <button
                type="submit"
                disabled={submittingFeedback}
                className="px-3 py-1 text-xs font-medium text-white bg-emerald-600 hover:bg-emerald-700 rounded transition-colors whitespace-nowrap shrink-0"
              >
                {submittingFeedback ? 'Saving...' : 'Save Correction'}
              </button>
              <button
                type="button"
                onClick={() => setShowFeedbackForm(false)}
                className="px-2 py-1 text-xs text-slate-600 dark:text-slate-400 hover:text-slate-900 whitespace-nowrap shrink-0"
              >
                Cancel
              </button>
              {feedbackError && <span className="text-red-600">{feedbackError}</span>}
            </form>
          )}
        </div>
      </div>
    </div>
  );
};
