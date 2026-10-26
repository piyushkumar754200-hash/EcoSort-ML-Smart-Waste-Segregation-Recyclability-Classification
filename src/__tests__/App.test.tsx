import React from 'react';
import { describe, expect, it } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { buildFilterQueryString, apiClient } from '../api/client';
import { PredictionResultCard } from '../components/PredictionResultCard';
import { PredictionRecord } from '../types/api';

const sampleOkPrediction: PredictionRecord = {
  prediction_id: 'pred-1234',
  status: 'ok',
  detected_item: 'Plastic Container / Bottle',
  category: 'plastic',
  predicted_top1: 'plastic',
  recyclability_class: 'Recyclable',
  bin: 'Blue Recycling Bin (Containers & Plastics)',
  confidence: 0.9125,
  confidence_level: 'HIGH',
  top_k: [
    { label: 'plastic', category: 'plastic', probability: 0.9125 },
    { label: 'glass', category: 'glass', probability: 0.0525 },
    { label: 'metal', category: 'metal', probability: 0.035 },
  ],
  disposal_tips: ['Empty and rinse bottles before recycling.'],
  prep_steps: ['Keep caps screwed on tightly.'],
  notes: 'General plastic guidance.',
  quality_warnings: [],
  inference_ms: 14.5,
  model_version: '2.0.0+20261025.abcd1234',
  is_smoke_test: false,
  is_demo: false,
  corrected_category: null,
  original_filename: 'water_bottle.jpg',
  file_sha256: 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
  image_width: 512,
  image_height: 512,
  has_thumbnail: false,
  created_at: '2026-10-25T14:30:00Z',
};

const sampleUncertainPrediction: PredictionRecord = {
  prediction_id: 'pred-5678',
  status: 'uncertain',
  detected_item: null,
  category: null,
  predicted_top1: 'glass',
  recyclability_class: 'Unknown',
  bin: null,
  confidence: 0.442,
  confidence_level: 'LOW',
  top_k: [
    { label: 'glass', category: 'glass', probability: 0.442 },
    { label: 'metal', category: 'metal', probability: 0.198 },
    { label: 'plastic', category: 'plastic', probability: 0.181 },
  ],
  disposal_tips: [],
  prep_steps: [],
  safe_handling_advice: [
    'Do not place an unverified item into a recycling or compost stream.',
    'Check the item manually for a resin code (#1-#7) or packaging label.',
  ],
  notes: 'Prediction confidence or margin is below threshold.',
  retake_advice:
    'Top-1 confidence (44.2%) is below minimum threshold (50%). Retake the photo on a plain background.',
  quality_warnings: [
    'Image is small (194x259). Larger photos (>= 224 px on the shorter side) usually give better results.',
    'Dominant background color/texture detected around the edges.',
  ],
  inference_ms: 15.2,
  model_version: '2.0.0+20261025.abcd1234',
  is_smoke_test: false,
  is_demo: false,
  corrected_category: null,
  original_filename: 'images.jpeg',
  file_sha256: '361fc86ae97745c5b056350a5daed66e361fc86ae97745c5b056350a5daed66e',
  image_width: 194,
  image_height: 259,
  has_thumbnail: false,
  created_at: '2026-10-25T14:35:00Z',
};

describe('EcoSort ML Frontend Unit & Component Tests', () => {
  it('renders PredictionResultCard for definite (ok) classification with recyclability, tips, and HIGH green bar', () => {
    render(<PredictionResultCard prediction={sampleOkPrediction} />);
    expect(screen.getAllByText('91.3%').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('Recyclable')).toBeTruthy();
    expect(screen.getByText(/Blue Recycling Bin/i)).toBeTruthy();
    expect(screen.getByText(/Preparation Before Disposal/i)).toBeTruthy();
    expect(screen.getByText(/Disposal Guidance & Contamination Rules/i)).toBeTruthy();
    expect(screen.getByText(/Empty and rinse bottles/i)).toBeTruthy();

    const bar = screen.getByTestId('confidence-progress-bar');
    expect(bar.className).toContain('bg-emerald-600');
  });

  it('hides default disposal/preparation panels when status is uncertain and shows retake + safe handling advice', () => {
    render(<PredictionResultCard prediction={sampleUncertainPrediction} />);
    expect(screen.getByText(/Uncertain Classification \(44\.2%\)/i)).toBeTruthy();
    expect(screen.getByText(/How to Retake a Clearer Photo/i)).toBeTruthy();
    expect(screen.getByText(/Safe Unverified Handling Advice/i)).toBeTruthy();

    // Critical regression check: definite Preparation & Disposal panels MUST NOT be rendered when uncertain
    expect(screen.queryByText('Preparation Before Disposal')).toBeNull();
    expect(screen.queryByText('Disposal Guidance & Contamination Rules')).toBeNull();
    expect(screen.queryByTestId('definite-guidance-section')).toBeNull();

    // Verify LOW confidence bar uses semantic red
    const bar = screen.getByTestId('confidence-progress-bar');
    expect(bar.className).toContain('bg-red-600');

    // Verify low-res (194x259) warning is displayed
    expect(
      screen.getByText(/Image is small \(194x259\)\. Larger photos \(>= 224 px/i)
    ).toBeTruthy();
  });

  it('allows previewing a candidate category guidance with explicit non-AI warning banner when uncertain', () => {
    render(<PredictionResultCard prediction={sampleUncertainPrediction} />);
    const plasticCandidateBtn = screen.getByTestId('candidate-preview-btn-plastic');
    fireEvent.click(plasticCandidateBtn);

    const previewBox = screen.getByTestId('candidate-guidance-preview');
    expect(previewBox.textContent).toMatch(
      /Candidate Preview \(PLASTIC\) — Based on your selection, not the AI's prediction/i
    );
    expect(previewBox.textContent).toMatch(/Check for resin codes #1 \(PET\)/i);

    // Clicking again hides the preview
    fireEvent.click(plasticCandidateBtn);
    expect(screen.queryByTestId('candidate-guidance-preview')).toBeNull();
  });

  it('renders manual category confirmation control in uncertain state and opens footer correction form in ok state', () => {
    const { unmount } = render(<PredictionResultCard prediction={sampleUncertainPrediction} />);
    const select = screen.getByTestId('uncertain-category-select') as HTMLSelectElement;
    fireEvent.change(select, { target: { value: 'plastic' } });
    expect(select.value).toBe('plastic');
    expect(screen.getByTestId('uncertain-confirm-category-btn')).toBeTruthy();
    unmount();

    render(<PredictionResultCard prediction={sampleOkPrediction} />);
    const btn = screen.getByText(/Wrong\? Correct it/i);
    fireEvent.click(btn);
    expect(screen.getByText(/Save Correction/i)).toBeTruthy();
  });

  it('builds synchronized filter query strings for History and CSV Export', () => {
    const filters = {
      page: 2,
      page_size: 15,
      category: 'organic',
      recyclability_class: 'Compostable',
      status: 'ok',
      min_confidence: '0.80',
      search: 'peel',
      include_demo: false,
    };
    const qs = buildFilterQueryString(filters);
    expect(qs).toContain('category=organic');
    expect(qs).toContain('recyclability_class=Compostable');
    expect(qs).toContain('min_confidence=0.80');
    expect(qs).toContain('include_demo=false');

    const csvUrl = apiClient.getCsvExportUrl(filters);
    expect(csvUrl).toContain('/api/export/csv?');
    expect(csvUrl).toContain('category=organic');
    expect(csvUrl).not.toContain('page=2');
  });

  it('displays near-equal top-3 probability notice, crushed/piled item explanation, and pile quality warning on uncertain can-pile predictions', () => {
    const nearUniformPilePrediction: PredictionRecord = {
      ...sampleUncertainPrediction,
      prediction_id: 'pred-pile-999',
      predicted_top1: 'metal',
      confidence: 0.294,
      confidence_level: 'LOW',
      normalized_entropy: 0.86,
      top_k: [
        { label: 'metal', category: 'metal', probability: 0.294 },
        { label: 'plastic', category: 'plastic', probability: 0.285 },
        { label: 'organic', category: 'organic', probability: 0.275 },
      ],
      quality_warnings: [
        'This looks like multiple items or a pile. Photograph one item at a time for the most reliable result.',
      ],
      original_filename: 'metal_can_pile_01.jpg',
    };

    render(<PredictionResultCard prediction={nearUniformPilePrediction} />);

    // Must show near-equal material disambiguation notice
    expect(screen.getByTestId('near-equal-probabilities-msg').textContent).toMatch(
      /The model could not distinguish between these materials/i
    );

    // Must mention crushed or piled items in the uncertain explanation
    expect(screen.getByText(/crushed or piled items/i)).toBeTruthy();

    // Must render the honest pile/multi-item warning banner
    expect(
      screen.getByText(
        /This looks like multiple items or a pile\. Photograph one item at a time for the most reliable result\./i
      )
    ).toBeTruthy();

    // Must never render definite disposal or prep sections
    expect(screen.queryByTestId('definite-guidance-section')).toBeNull();
  });
});

