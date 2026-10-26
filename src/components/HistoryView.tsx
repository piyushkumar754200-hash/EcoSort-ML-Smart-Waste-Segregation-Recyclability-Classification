import React, { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import {
  ChevronLeft,
  ChevronRight,
  Download,
  Eye,
  Filter,
  RotateCcw,
  Search,
  Trash2,
  X,
} from 'lucide-react';
import { apiClient } from '../api/client';
import { HistoryFilters, PredictionRecord } from '../types/api';
import { PredictionResultCard } from './PredictionResultCard';

interface Props {
  onNavigateClassify: () => void;
}

export const HistoryView: React.FC<Props> = ({ onNavigateClassify }) => {
  const queryClient = useQueryClient();

  const [filters, setFilters] = useState<HistoryFilters>({
    page: 1,
    page_size: 15,
    sort_by: 'created_at',
    sort_order: 'desc',
    category: '',
    recyclability_class: '',
    status: '',
    min_confidence: '',
    max_confidence: '',
    date_from: '',
    date_to: '',
    search: '',
    include_demo: true,
  });

  const [selectedRecord, setSelectedRecord] = useState<PredictionRecord | null>(null);
  const [confirmBulkClear, setConfirmBulkClear] = useState(false);

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['predictions', filters],
    queryFn: () => apiClient.getPredictions(filters),
  });

  const updateFilter = (patch: Partial<HistoryFilters>) => {
    setFilters((prev) => ({
      ...prev,
      ...patch,
      page: patch.page !== undefined ? patch.page : 1,
    }));
  };

  const resetFilters = () => {
    setFilters({
      page: 1,
      page_size: 15,
      sort_by: 'created_at',
      sort_order: 'desc',
      category: '',
      recyclability_class: '',
      status: '',
      min_confidence: '',
      max_confidence: '',
      date_from: '',
      date_to: '',
      search: '',
      include_demo: true,
    });
  };

  const handleDeleteSingle = async (id: string) => {
    await apiClient.deletePrediction(id);
    if (selectedRecord?.prediction_id === id) {
      setSelectedRecord(null);
    }
    queryClient.invalidateQueries({ queryKey: ['predictions'] });
    queryClient.invalidateQueries({ queryKey: ['analytics'] });
  };

  const handleBulkClear = async () => {
    await apiClient.bulkDeletePredictions(false);
    setConfirmBulkClear(false);
    setSelectedRecord(null);
    queryClient.invalidateQueries({ queryKey: ['predictions'] });
    queryClient.invalidateQueries({ queryKey: ['analytics'] });
  };

  const csvUrl = apiClient.getCsvExportUrl(filters);

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-200 dark:border-slate-800 pb-5">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900 dark:text-slate-100">
            Prediction History & Audit Log
          </h1>
          <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
            Browse, filter, inspect, and export persisted waste classifications from the relational database.
          </p>
        </div>

        <div className="flex items-center gap-3 shrink-0">
          <a
            href={csvUrl}
            download
            data-testid="export-csv-button"
            className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-emerald-600 hover:bg-emerald-700 rounded-lg transition-colors whitespace-nowrap shrink-0"
          >
            <Download className="w-4 h-4" />
            Export CSV
          </a>

          {!confirmBulkClear ? (
            <button
              type="button"
              onClick={() => setConfirmBulkClear(true)}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 text-sm font-medium text-slate-700 dark:text-slate-300 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 hover:border-red-400 hover:text-red-600 rounded-lg transition-colors whitespace-nowrap shrink-0"
            >
              <Trash2 className="w-4 h-4" />
              Clear All
            </button>
          ) : (
            <div className="flex items-center gap-2 bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-800 px-3 py-1.5 rounded-lg">
              <span className="text-xs font-medium text-red-800 dark:text-red-200">Confirm delete all?</span>
              <button
                type="button"
                onClick={handleBulkClear}
                className="px-2.5 py-1 text-xs font-semibold text-white bg-red-600 hover:bg-red-700 rounded whitespace-nowrap shrink-0"
              >
                Yes, Delete
              </button>
              <button
                type="button"
                onClick={() => setConfirmBulkClear(false)}
                className="px-2 py-1 text-xs text-slate-600 dark:text-slate-300 whitespace-nowrap shrink-0"
              >
                Cancel
              </button>
            </div>
          )}
        </div>
      </div>

      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-4 space-y-4">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
          <div className="relative lg:col-span-2">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={filters.search || ''}
              onChange={(e) => updateFilter({ search: e.target.value })}
              placeholder="Search filename, item, category, or ID..."
              aria-label="Search predictions"
              className="w-full pl-9 pr-3 py-2 text-sm rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100"
            />
          </div>

          <div>
            <select
              aria-label="Filter by category"
              value={filters.category || ''}
              onChange={(e) => updateFilter({ category: e.target.value })}
              className="w-full px-3 py-2 text-sm rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100"
            >
              <option value="">All Categories</option>
              <option value="plastic">Plastic</option>
              <option value="paper">Paper</option>
              <option value="metal">Metal</option>
              <option value="glass">Glass</option>
              <option value="organic">Organic</option>
              <option value="special_waste">Special Waste</option>
            </select>
          </div>

          <div>
            <select
              aria-label="Filter by recyclability class"
              value={filters.recyclability_class || ''}
              onChange={(e) => updateFilter({ recyclability_class: e.target.value })}
              className="w-full px-3 py-2 text-sm rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100"
            >
              <option value="">All Recyclability</option>
              <option value="Recyclable">Recyclable</option>
              <option value="Compostable">Compostable</option>
              <option value="Non-recyclable">Non-recyclable</option>
              <option value="Special handling">Special handling</option>
            </select>
          </div>

          <div>
            <select
              aria-label="Filter by status"
              value={filters.status || ''}
              onChange={(e) => updateFilter({ status: e.target.value })}
              className="w-full px-3 py-2 text-sm rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100"
            >
              <option value="">All Statuses</option>
              <option value="ok">Definite (ok)</option>
              <option value="uncertain">Uncertain (&lt; threshold)</option>
            </select>
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 pt-2 border-t border-slate-100 dark:border-slate-800/70 text-xs">
          <div className="flex flex-wrap items-center gap-3">
            <label className="flex items-center gap-1.5 text-slate-600 dark:text-slate-400">
              <span>Min Conf:</span>
              <input
                type="number"
                step="0.05"
                min="0"
                max="1"
                value={filters.min_confidence || ''}
                onChange={(e) => updateFilter({ min_confidence: e.target.value })}
                placeholder="0.0"
                className="w-16 px-2 py-1 rounded border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 font-mono tabular-nums"
              />
            </label>

            <label className="flex items-center gap-1.5 text-slate-600 dark:text-slate-400">
              <span>From:</span>
              <input
                type="date"
                value={filters.date_from || ''}
                onChange={(e) => updateFilter({ date_from: e.target.value })}
                className="px-2 py-1 rounded border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 font-mono"
              />
            </label>

            <label className="flex items-center gap-1.5 text-slate-600 dark:text-slate-400">
              <span>To:</span>
              <input
                type="date"
                value={filters.date_to || ''}
                onChange={(e) => updateFilter({ date_to: e.target.value })}
                className="px-2 py-1 rounded border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 font-mono"
              />
            </label>

            <label className="flex items-center gap-1.5 text-slate-600 dark:text-slate-400">
              <span>Sort:</span>
              <select
                value={`${filters.sort_by}:${filters.sort_order}`}
                onChange={(e) => {
                  const [sb, so] = e.target.value.split(':');
                  updateFilter({ sort_by: sb, sort_order: so as 'asc' | 'desc' });
                }}
                className="px-2 py-1 rounded border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950"
              >
                <option value="created_at:desc">Newest First</option>
                <option value="created_at:asc">Oldest First</option>
                <option value="confidence:desc">Highest Confidence</option>
                <option value="confidence:asc">Lowest Confidence</option>
              </select>
            </label>
          </div>

          <button
            type="button"
            onClick={resetFilters}
            className="inline-flex items-center gap-1 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100 whitespace-nowrap shrink-0"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Reset Filters
          </button>
        </div>
      </div>

      {isLoading ? (
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 space-y-3">
          {[...Array(5)].map((_, i) => (
            <div
              key={i}
              className="h-10 bg-slate-100 dark:bg-slate-800 rounded animate-pulse"
            />
          ))}
        </div>
      ) : isError ? (
        <div className="bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800 rounded-xl p-6 text-sm text-red-800 dark:text-red-200">
          Failed to load prediction history: {(error as any)?.message || 'Unknown error'}
        </div>
      ) : !data || data.items.length === 0 ? (
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-12 text-center">
          <Filter className="w-8 h-8 text-slate-400 mx-auto mb-3" />
          <h2 className="text-base font-semibold text-slate-900 dark:text-slate-100">
            No matching predictions found
          </h2>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 max-w-md mx-auto">
            Upload a waste image in the Classify tab or adjust your active search and filter criteria.
          </p>
          <button
            type="button"
            onClick={onNavigateClassify}
            className="mt-4 px-4 py-2 text-xs font-medium text-white bg-emerald-600 hover:bg-emerald-700 rounded-lg transition-colors whitespace-nowrap shrink-0"
          >
            Classify First Item
          </button>
        </div>
      ) : (
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-200 dark:border-slate-800 text-xs font-semibold text-slate-500 dark:text-slate-400 bg-slate-50/70 dark:bg-slate-950/50">
                  <th className="py-3 px-4">Item / Name</th>
                  <th className="py-3 px-4">Method</th>
                  <th className="py-3 px-4">Category</th>
                  <th className="py-3 px-4">Recyclability & Bin</th>
                  <th className="py-3 px-4 text-right">Confidence</th>
                  <th className="py-3 px-4">Status</th>
                  <th className="py-3 px-4">Timestamp</th>
                  <th className="py-3 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-slate-800 text-sm">
                {data.items.map((row) => {
                  const isTextSearch =
                    row.original_filename.startsWith('catalog:') ||
                    row.original_filename.startsWith('search:') ||
                    row.original_filename.startsWith('text:') ||
                    row.classification_method === 'text_search';

                  const displayName = isTextSearch
                    ? row.detected_item ||
                      row.original_filename
                        .replace('catalog:', '')
                        .replace('search:', '')
                        .replace('text:', '')
                    : row.original_filename;

                  return (
                    <tr
                      key={row.prediction_id}
                      className="hover:bg-slate-50/80 dark:hover:bg-slate-800/40 transition-colors"
                    >
                      <td className="py-2.5 px-4">
                        <div className="flex items-center gap-3">
                          {row.has_thumbnail ? (
                            <img
                              src={`/api/predictions/${row.prediction_id}/image`}
                              alt={displayName}
                              referrerPolicy="no-referrer"
                              className="w-9 h-9 rounded object-cover bg-slate-100 dark:bg-slate-800 shrink-0"
                            />
                          ) : (
                            <div className="w-9 h-9 rounded bg-slate-100 dark:bg-slate-800 shrink-0 flex items-center justify-center text-[10px] font-mono text-slate-400">
                              {isTextSearch ? 'CAT' : 'IMG'}
                            </div>
                          )}
                          <div className="min-w-0">
                            <p className="font-semibold text-slate-900 dark:text-slate-100 truncate max-w-[200px]">
                              {displayName}
                            </p>
                            <p className="text-xs text-slate-500 dark:text-slate-400">
                              {row.detected_item && !isTextSearch ? `Detected: ${row.detected_item}` : ''}
                              {row.is_demo && ' · Demo'}
                              {row.corrected_category && ` · Corrected: ${row.corrected_category}`}
                            </p>
                          </div>
                        </div>
                      </td>
                      <td className="py-2.5 px-4 text-xs font-medium">
                        {isTextSearch ? (
                          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-emerald-100 dark:bg-emerald-950/70 text-emerald-800 dark:text-emerald-300 font-semibold text-[11px] border border-emerald-300 dark:border-emerald-800">
                            Catalog Search
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-sky-100 dark:bg-sky-950/70 text-sky-800 dark:text-sky-300 font-semibold text-[11px] border border-sky-300 dark:border-sky-800">
                            ML Model
                          </span>
                        )}
                      </td>
                      <td className="py-2.5 px-4 capitalize font-semibold text-slate-800 dark:text-slate-200">
                        {row.corrected_category === 'special_waste' || row.category === 'special_waste'
                          ? 'Special Waste'
                          : row.corrected_category || row.category || 'Uncertain'}
                      </td>
                      <td className="py-2.5 px-4">
                        <p className="font-medium text-slate-900 dark:text-slate-100">
                          {row.recyclability_class}
                        </p>
                        <p className="text-xs text-slate-500 dark:text-slate-400 truncate max-w-[200px]">
                          {row.bin || 'Manual verification required'}
                        </p>
                      </td>
                      <td className="py-2.5 px-4 text-right font-mono tabular-nums font-semibold text-slate-900 dark:text-slate-100">
                        {isTextSearch ? (
                          <span className="text-xs font-sans text-emerald-600 dark:text-emerald-400">
                            Verified Match
                          </span>
                        ) : row.confidence !== undefined && row.confidence !== null ? (
                          `${(row.confidence * 100).toFixed(1)}%`
                        ) : (
                          'N/A'
                        )}
                      </td>
                      <td className="py-2.5 px-4 text-xs">
                        <span
                          className={
                            row.status === 'uncertain'
                              ? 'text-amber-700 dark:text-amber-400 font-medium'
                              : 'text-emerald-700 dark:text-emerald-400 font-medium'
                          }
                        >
                          {row.status === 'uncertain' ? 'Uncertain' : `OK (${row.confidence_level})`}
                        </span>
                      </td>
                      <td className="py-2.5 px-4 font-mono tabular-nums text-xs text-slate-500 dark:text-slate-400">
                        {new Date(row.created_at).toLocaleString()}
                      </td>
                    <td className="py-2.5 px-4 text-right">
                      <div className="inline-flex items-center gap-2">
                        <button
                          type="button"
                          onClick={() => setSelectedRecord(row)}
                          aria-label={`Inspect ${row.original_filename}`}
                          className="p-1.5 text-slate-500 hover:text-emerald-600 transition-colors"
                        >
                          <Eye className="w-4 h-4" />
                        </button>
                        <button
                          type="button"
                          onClick={() => handleDeleteSingle(row.prediction_id)}
                          aria-label={`Delete ${row.original_filename}`}
                          className="p-1.5 text-slate-500 hover:text-red-600 transition-colors"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
              </tbody>
            </table>
          </div>

          <div className="px-4 py-3 border-t border-slate-200 dark:border-slate-800 flex items-center justify-between text-xs text-slate-600 dark:text-slate-400">
            <span className="font-mono tabular-nums">
              Showing page {data.page} of {data.total_pages} ({data.total} total records)
            </span>
            <div className="flex items-center gap-2">
              <button
                type="button"
                disabled={data.page <= 1}
                onClick={() => updateFilter({ page: data.page - 1 })}
                className="inline-flex items-center gap-1 px-2.5 py-1 rounded border border-slate-200 dark:border-slate-800 disabled:opacity-40 whitespace-nowrap shrink-0"
              >
                <ChevronLeft className="w-3.5 h-3.5" />
                Prev
              </button>
              <button
                type="button"
                disabled={data.page >= data.total_pages}
                onClick={() => updateFilter({ page: data.page + 1 })}
                className="inline-flex items-center gap-1 px-2.5 py-1 rounded border border-slate-200 dark:border-slate-800 disabled:opacity-40 whitespace-nowrap shrink-0"
              >
                Next
                <ChevronRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        </div>
      )}

      {selectedRecord && (
        <div
          className="fixed inset-0 z-50 bg-black/60 flex items-center justify-end"
          role="dialog"
          aria-modal="true"
          aria-label="Prediction Details"
        >
          <div className="w-full max-w-2xl h-full bg-slate-50 dark:bg-slate-950 border-l border-slate-200 dark:border-slate-800 p-6 overflow-y-auto space-y-5">
            <div className="flex items-center justify-between border-b border-slate-200 dark:border-slate-800 pb-4">
              <div>
                <h2 className="text-lg font-bold text-slate-900 dark:text-slate-100">
                  Prediction Record Inspection
                </h2>
                <p className="text-xs font-mono text-slate-500">ID: {selectedRecord.prediction_id}</p>
              </div>
              <button
                type="button"
                onClick={() => setSelectedRecord(null)}
                aria-label="Close inspection drawer"
                className="p-2 rounded-lg text-slate-500 hover:text-slate-900 dark:hover:text-slate-100"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <PredictionResultCard
              prediction={selectedRecord}
              onFeedbackUpdated={(updated) => {
                setSelectedRecord(updated);
                queryClient.invalidateQueries({ queryKey: ['predictions'] });
              }}
            />

            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-4 text-xs font-mono space-y-1 text-slate-600 dark:text-slate-400">
              <p>SHA-256 Digest: {selectedRecord.file_sha256}</p>
              <p>Timestamp (UTC): {selectedRecord.created_at}</p>
              <p>Model Version: {selectedRecord.model_version}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
