export type WasteCategory = 'plastic' | 'paper' | 'metal' | 'glass' | 'organic';

export type RecyclabilityClass =
  | 'Recyclable'
  | 'Compostable'
  | 'Non-recyclable'
  | 'Special handling'
  | 'Unknown';

export interface TopKItem {
  label: string;
  category: string;
  probability: number;
}

export interface CandidateGuidanceItem {
  category: string;
  detected_item: string;
  recyclability_class: string;
  bin: string;
  disposal_tips: string[];
  prep_steps: string[];
  notes: string;
}

export interface PredictionRecord {
  prediction_id: string;
  status: 'ok' | 'uncertain';
  detected_item: string | null;
  category: WasteCategory | string | null;
  predicted_top1?: string | null;
  recyclability_class: RecyclabilityClass | string;
  bin: string | null;
  confidence: number;
  confidence_level: 'HIGH' | 'MEDIUM' | 'LOW';
  top_k: TopKItem[];
  probabilities?: Record<string, number>;
  explanation?: string | null;
  classification_method?: string;
  disposal_tips: string[];
  prep_steps: string[];
  safe_handling_advice?: string[];
  candidate_guidance?: Record<string, CandidateGuidanceItem>;
  notes: string;
  retake_advice?: string | null;
  quality_warnings: string[];
  entropy?: number | null;
  normalized_entropy?: number | null;
  is_near_uniform?: boolean;
  inference_ms: number;
  model_version: string;
  is_smoke_test: boolean;
  is_demo: boolean;
  corrected_category?: string | null;
  original_filename: string;
  file_sha256: string;
  image_width: number;
  image_height: number;
  has_thumbnail: boolean;
  created_at: string;
}

export interface BatchItemResult {
  filename: string;
  success: boolean;
  prediction?: PredictionRecord | null;
  error?: {
    code: string;
    message: string;
    details?: Record<string, unknown>;
  } | null;
}

export interface BatchPredictionResponse {
  total: number;
  succeeded: number;
  failed: number;
  results: BatchItemResult[];
}

export interface PaginatedPredictions {
  items: PredictionRecord[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface HealthStatus {
  status: string;
  model_loaded: boolean;
  model_version: string;
  db_connected: boolean;
  is_smoke_test: boolean;
  smoke_test_warning?: string | null;
}

export interface PerClassMetric {
  category: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface ModelMetrics {
  selected_model: string;
  backbone: string;
  test_samples: number;
  accuracy: number;
  top_3_accuracy: number;
  macro_precision: number;
  macro_recall: number;
  macro_f1: number;
  weighted_precision: number;
  weighted_recall: number;
  weighted_f1: number;
  ece: number;
  temperature: number;
  val_macro_f1: number;
  per_class: PerClassMetric[];
  confusion_matrix: number[][];
  class_names: string[];
  reliability_bins: Array<{
    bin_low: number;
    bin_high: number;
    count: number;
    avg_confidence: number;
    avg_accuracy: number;
  }>;
  per_source: Record<
    string,
    {
      count: number;
      accuracy: number;
      macro_f1: number;
    }
  >;
  model_comparison: Array<{
    checkpoint: string;
    backbone: string;
    val_macro_f1: number;
    temperature: number;
    wall_clock_seconds: number;
  }>;
}

export interface ModelInfo {
  model_loaded: boolean;
  model_version: string;
  is_smoke_test: boolean;
  architecture: string;
  classes: string[];
  thresholds: {
    high: number;
    medium: number;
    min: number;
    margin_min: number;
  };
  temperature: number;
  input_size: number;
  model_card: Record<string, any>;
  metrics?: ModelMetrics | null;
  disclaimer: string;
}

export interface AnalyticsSummary {
  total_predictions: number;
  average_confidence: number;
  uncertain_count: number;
  uncertain_rate: number;
  recyclable_count: number;
  compostable_count: number;
  special_waste_count?: number;
  recyclable_or_compostable_rate: number;
  feedback_count: number;
  primary_stream_counts?: Record<string, number>;
  frequently_classified_items?: Array<{
    item: string;
    count: number;
    category: string;
  }>;
  category_distribution: Array<{
    category: string;
    count: number;
    percentage: number;
    avg_confidence: number;
  }>;
  recyclability_distribution: Array<{
    recyclability_class: string;
    count: number;
    percentage: number;
  }>;
  time_series: Array<{
    date: string;
    count: number;
    diverted: number;
    uncertain: number;
  }>;
  confidence_histogram: Array<{
    bucket: string;
    count: number;
  }>;
  date_from?: string | null;
  date_to?: string | null;
  exclude_demo: boolean;
}

export interface HistoryFilters {
  page?: number;
  page_size?: number;
  sort_by?: string;
  sort_order?: 'asc' | 'desc';
  category?: string;
  recyclability_class?: string;
  status?: string;
  min_confidence?: string;
  max_confidence?: string;
  date_from?: string;
  date_to?: string;
  search?: string;
  include_demo?: boolean;
}

export interface PrimaryStreamData {
  name: string;
  category: string;
  recyclability_class: string;
  bin: string;
  badge_color: string;
  examples: string[];
  prep_steps: string[];
  contamination_guidance: string[];
  local_acceptance: string;
}

export interface SpecialWasteData {
  name: string;
  category: string;
  recyclability_class: string;
  recommended_collection: string;
  warning: string;
  safe_handling: string[];
  local_acceptance: string;
}

export interface MaterialDirectoryItem {
  id: string;
  name: string;
  aliases: string[];
  category: string;
  primary_stream: string;
  recyclability_class: string;
  bin: string;
  explanation: string;
  prep_steps: string[];
  disposal_tips: string[];
  contamination_warnings: string[];
  local_acceptance: string;
}

export interface CommonMistakeData {
  mistake: string;
  consequence: string;
  correct_action: string;
}

export interface TaxonomyResponse {
  version: string;
  disclaimer: string;
  categories: string[];
  recyclability_classes: string[];
  primary_streams: Record<string, PrimaryStreamData>;
  special_waste_streams: Record<string, SpecialWasteData>;
  material_directory: MaterialDirectoryItem[];
  common_mistakes: CommonMistakeData[];
}
