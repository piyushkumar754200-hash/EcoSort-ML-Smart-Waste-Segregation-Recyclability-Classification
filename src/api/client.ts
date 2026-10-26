
import {
  AnalyticsSummary,
  BatchPredictionResponse,
  HealthStatus,
  HistoryFilters,
  ModelInfo,
  PaginatedPredictions,
  PredictionRecord,
  TaxonomyResponse,
} from '../types/api';

// Render backend URL from Vercel environment variables
const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || ''
).replace(/\/+$/, '');

function apiUrl(path: string): string {
  return `${API_BASE_URL}${path}`;
}

export class ApiClientError extends Error {
  status: number;
  code: string;
  details: Record<string, unknown>;

  constructor(
    status: number,
    code: string,
    message: string,
    details: Record<string, unknown> = {}
  ) {
    super(message);
    this.name = 'ApiClientError';
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let code = `HTTP_${res.status}`;
    let message = `Request failed with status ${res.status}`;
    let details: Record<string, unknown> = {};

    try {
      const data = await res.json();
      if (data?.error) {
        code = data.error.code || code;
        message = data.error.message || message;
        details = data.error.details || {};
      } else if (data?.detail) {
        message = String(data.detail);
      }
    } catch {
      if (res.status === 413) {
        message = 'Uploaded file is too large (max 10 MB).';
      } else if (res.status === 415) {
        message = 'Unsupported file type. Upload JPEG, PNG, or WebP.';
      } else if (res.status === 429) {
        message = 'Rate limit exceeded. Please wait a moment and retry.';
      } else if (res.status === 503) {
        message = 'Classification model is currently not loaded (503).';
      }
    }

    throw new ApiClientError(res.status, code, message, details);
  }

  return res.json() as Promise<T>;
}

export function buildFilterQueryString(filters: HistoryFilters): string {
  const params = new URLSearchParams();

  if (filters.page !== undefined) params.set('page', String(filters.page));
  if (filters.page_size !== undefined) {
    params.set('page_size', String(filters.page_size));
  }
  if (filters.sort_by) params.set('sort_by', filters.sort_by);
  if (filters.sort_order) params.set('sort_order', filters.sort_order);
  if (filters.category) params.set('category', filters.category);
  if (filters.recyclability_class) {
    params.set('recyclability_class', filters.recyclability_class);
  }
  if (filters.status) params.set('status', filters.status);

  if (filters.min_confidence !== undefined && filters.min_confidence !== '') {
    params.set('min_confidence', String(filters.min_confidence));
  }
  if (filters.max_confidence !== undefined && filters.max_confidence !== '') {
    params.set('max_confidence', String(filters.max_confidence));
  }
  if (filters.date_from) params.set('date_from', filters.date_from);
  if (filters.date_to) params.set('date_to', filters.date_to);
  if (filters.search?.trim()) params.set('search', filters.search.trim());

  if (filters.include_demo !== undefined) {
    params.set('include_demo', String(filters.include_demo));
  }

  return params.toString();
}

export const apiClient = {
  async getHealth(): Promise<HealthStatus> {
    const res = await fetch(apiUrl('/api/health'));
    return handleResponse<HealthStatus>(res);
  },

  async getModelInfo(): Promise<ModelInfo> {
    const res = await fetch(apiUrl('/api/model/info'));
    return handleResponse<ModelInfo>(res);
  },

  async predictSingle(file: File): Promise<PredictionRecord> {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch(apiUrl('/api/predict'), {
      method: 'POST',
      body: formData,
    });

    return handleResponse<PredictionRecord>(res);
  },

  async predictBatch(files: File[]): Promise<BatchPredictionResponse> {
    const formData = new FormData();

    for (const file of files) {
      formData.append('files', file);
    }

    const res = await fetch(apiUrl('/api/predict/batch'), {
      method: 'POST',
      body: formData,
    });

    return handleResponse<BatchPredictionResponse>(res);
  },

  async getPredictions(filters: HistoryFilters): Promise<PaginatedPredictions> {
    const qs = buildFilterQueryString(filters);
    const path = `/api/predictions${qs ? `?${qs}` : ''}`;
    const res = await fetch(apiUrl(path));

    return handleResponse<PaginatedPredictions>(res);
  },

  async getPredictionById(id: string): Promise<PredictionRecord> {
    const res = await fetch(
      apiUrl(`/api/predictions/${encodeURIComponent(id)}`)
    );

    return handleResponse<PredictionRecord>(res);
  },

  async deletePrediction(
    id: string
  ): Promise<{ deleted: boolean; prediction_id: string }> {
    const res = await fetch(
      apiUrl(`/api/predictions/${encodeURIComponent(id)}`),
      { method: 'DELETE' }
    );

    return handleResponse(res);
  },

  async bulkDeletePredictions(
    demoOnly: boolean = false
  ): Promise<{ deleted: boolean; deleted_count: number }> {
    const res = await fetch(
      apiUrl(`/api/predictions?confirm=true&demo_only=${demoOnly}`),
      { method: 'DELETE' }
    );

    return handleResponse(res);
  },

  async submitFeedback(
    id: string,
    correctedCategory: string,
    notes?: string
  ): Promise<PredictionRecord> {
    const res = await fetch(
      apiUrl(`/api/predictions/${encodeURIComponent(id)}/feedback`),
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          corrected_category: correctedCategory,
          notes: notes || undefined,
        }),
      }
    );

    return handleResponse<PredictionRecord>(res);
  },

  async getAnalyticsSummary(params: {
    date_from?: string;
    date_to?: string;
    include_demo?: boolean;
  }): Promise<AnalyticsSummary> {
    const search = new URLSearchParams();

    if (params.date_from) search.set('date_from', params.date_from);
    if (params.date_to) search.set('date_to', params.date_to);

    if (params.include_demo !== undefined) {
      search.set('include_demo', String(params.include_demo));
    }

    const qs = search.toString();
    const path = `/api/analytics/summary${qs ? `?${qs}` : ''}`;
    const res = await fetch(apiUrl(path));

    return handleResponse<AnalyticsSummary>(res);
  },

  async classifyText(query: string): Promise<PredictionRecord> {
    const res = await fetch(apiUrl('/api/classify/text'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    });

    return handleResponse<PredictionRecord>(res);
  },

  async getTaxonomy(): Promise<TaxonomyResponse> {
    const res = await fetch(apiUrl('/api/taxonomy'));
    return handleResponse<TaxonomyResponse>(res);
  },

  getCsvExportUrl(filters: HistoryFilters): string {
    const clone = { ...filters };
    delete clone.page;
    delete clone.page_size;

    const qs = buildFilterQueryString(clone);
    const path = `/api/export/csv${qs ? `?${qs}` : ''}`;

    return apiUrl(path);
  },
};
