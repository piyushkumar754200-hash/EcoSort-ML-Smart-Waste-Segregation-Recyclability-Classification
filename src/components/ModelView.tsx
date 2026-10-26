import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { AlertCircle, CheckCircle2, Cpu } from 'lucide-react';
import { apiClient } from '../api/client';

export const ModelView: React.FC = () => {
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['model-info'],
    queryFn: () => apiClient.getModelInfo(),
  });

  if (isLoading) {
    return (
      <div className="space-y-4">
        {[...Array(3)].map((_, i) => (
          <div
            key={i}
            className="h-32 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl animate-pulse"
          />
        ))}
      </div>
    );
  }

  if (isError || !data) {
    return (
      <div className="p-6 rounded-xl bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800 text-sm text-red-800 dark:text-red-200">
        Failed to load model telemetry: {(error as any)?.message || 'Unknown error'}
      </div>
    );
  }

  const metrics = data.metrics;
  const parity = data.model_card?.parity_check;

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="border-b border-slate-200 dark:border-slate-800 pb-5">
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900 dark:text-slate-100">
          Model Card & Evaluation Telemetry
        </h1>
        <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
          Live metadata, held-out test set evaluation metrics, confusion matrix, and PyTorch-to-ONNX
          numerical parity verification loaded from <code className="font-mono">artifacts/</code>.
        </p>
      </div>

      {/* Model Runtime Summary Strip */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
          <div className="flex items-center gap-3">
            <Cpu className="w-6 h-6 text-emerald-600 dark:text-emerald-400" />
            <div>
              <h2 className="text-lg font-bold text-slate-900 dark:text-slate-100">
                Architecture: {data.architecture} (ONNX Runtime CPU)
              </h2>
              <p className="text-xs text-slate-500 font-mono">
                Version: {data.model_version} · Input: {data.input_size}×{data.input_size} RGB ·
                Calibration Temp T={data.temperature.toFixed(4)}
              </p>
            </div>
          </div>

          <div className="text-xs font-medium">
            {data.model_loaded ? (
              <span className="inline-flex items-center gap-1.5 text-emerald-700 dark:text-emerald-400">
                <CheckCircle2 className="w-4 h-4" />
                ONNX Weights Loaded & Verified
              </span>
            ) : (
              <span className="inline-flex items-center gap-1.5 text-amber-700 dark:text-amber-400">
                <AlertCircle className="w-4 h-4" />
                Model Not Loaded
              </span>
            )}
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 pt-4 border-t border-slate-200 dark:border-slate-800 text-xs">
          <div>
            <p className="text-slate-500">High Confidence Threshold</p>
            <p className="font-mono font-semibold text-sm text-slate-900 dark:text-slate-100 mt-0.5">
              &ge; {(data.thresholds.high * 100).toFixed(0)}%
            </p>
          </div>
          <div>
            <p className="text-slate-500">Min Certainty Threshold</p>
            <p className="font-mono font-semibold text-sm text-slate-900 dark:text-slate-100 mt-0.5">
              &ge; {(data.thresholds.min * 100).toFixed(0)}%
            </p>
          </div>
          <div>
            <p className="text-slate-500">ONNX vs PyTorch Max Diff</p>
            <p className="font-mono font-semibold text-sm text-emerald-700 dark:text-emerald-400 mt-0.5">
              {parity ? `${Number(parity.max_abs_softmax_diff).toExponential(2)} (< 1e-4)` : 'N/A'}
            </p>
          </div>
          <div>
            <p className="text-slate-500">Weights SHA-256 Prefix</p>
            <p className="font-mono font-semibold text-sm text-slate-900 dark:text-slate-100 mt-0.5">
              {data.model_card?.weights_sha256
                ? `${String(data.model_card.weights_sha256).slice(0, 12)}...`
                : 'N/A'}
            </p>
          </div>
        </div>
      </div>

      {!metrics ? (
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-10 text-center">
          <AlertCircle className="w-8 h-8 text-amber-500 mx-auto mb-2" />
          <h3 className="text-base font-semibold text-slate-900 dark:text-slate-100">
            Metrics Unavailable
          </h3>
          <p className="text-xs text-slate-500 mt-1">
            Evaluation metrics artifact (<code className="font-mono">artifacts/metrics/metrics.json</code>) is
            not yet available. Run <code className="font-mono">python ml/scripts/evaluate.py</code> after
            training completes.
          </p>
        </div>
      ) : (
        <>
          {/* Held-Out Test Set KPIs */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-4">
              <p className="text-xs text-slate-500">Test Accuracy</p>
              <p className="text-2xl font-bold font-mono tabular-nums text-slate-900 dark:text-slate-100 mt-1">
                {(metrics.accuracy * 100).toFixed(2)}%
              </p>
            </div>
            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-4">
              <p className="text-xs text-slate-500">Test Macro-F1</p>
              <p className="text-2xl font-bold font-mono tabular-nums text-emerald-600 dark:text-emerald-400 mt-1">
                {metrics.macro_f1.toFixed(4)}
              </p>
            </div>
            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-4">
              <p className="text-xs text-slate-500">Weighted-F1</p>
              <p className="text-2xl font-bold font-mono tabular-nums text-slate-900 dark:text-slate-100 mt-1">
                {metrics.weighted_f1.toFixed(4)}
              </p>
            </div>
            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-4">
              <p className="text-xs text-slate-500">Top-3 Accuracy</p>
              <p className="text-2xl font-bold font-mono tabular-nums text-slate-900 dark:text-slate-100 mt-1">
                {(metrics.top_3_accuracy * 100).toFixed(2)}%
              </p>
            </div>
            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-4">
              <p className="text-xs text-slate-500">Calibration ECE</p>
              <p className="text-2xl font-bold font-mono tabular-nums text-slate-900 dark:text-slate-100 mt-1">
                {metrics.ece.toFixed(4)}
              </p>
            </div>
            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-4">
              <p className="text-xs text-slate-500">Held-Out Test Size</p>
              <p className="text-2xl font-bold font-mono tabular-nums text-slate-900 dark:text-slate-100 mt-1">
                {metrics.test_samples}
              </p>
            </div>
          </div>

          {/* Per-Class Performance & Confusion Matrix */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Per-Class Table */}
            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6">
              <h2 className="text-base font-bold text-slate-900 dark:text-slate-100 mb-4">
                Per-Category Test Set Performance
              </h2>
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-sm">
                  <thead>
                    <tr className="border-b border-slate-200 dark:border-slate-800 text-xs text-slate-500">
                      <th className="py-2 pr-4">Category</th>
                      <th className="py-2 px-3 text-right">Precision</th>
                      <th className="py-2 px-3 text-right">Recall</th>
                      <th className="py-2 px-3 text-right">F1-Score</th>
                      <th className="py-2 pl-3 text-right">Support</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
                    {metrics.per_class.map((row) => (
                      <tr key={row.category}>
                        <td className="py-2.5 pr-4 font-semibold capitalize text-slate-900 dark:text-slate-100">
                          {row.category}
                        </td>
                        <td className="py-2.5 px-3 text-right font-mono tabular-nums">
                          {(row.precision * 100).toFixed(1)}%
                        </td>
                        <td className="py-2.5 px-3 text-right font-mono tabular-nums">
                          {(row.recall * 100).toFixed(1)}%
                        </td>
                        <td className="py-2.5 px-3 text-right font-mono tabular-nums font-semibold text-emerald-700 dark:text-emerald-400">
                          {row.f1.toFixed(4)}
                        </td>
                        <td className="py-2.5 pl-3 text-right font-mono tabular-nums text-slate-500">
                          {row.support}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Confusion Matrix Grid */}
            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6">
              <h2 className="text-base font-bold text-slate-900 dark:text-slate-100 mb-4">
                Held-Out Test Set Confusion Matrix (True vs Predicted)
              </h2>
              <div className="overflow-x-auto">
                <table className="w-full border-collapse text-xs font-mono tabular-nums">
                  <thead>
                    <tr>
                      <th className="p-2 text-left text-slate-400">True \ Pred</th>
                      {metrics.class_names.map((c) => (
                        <th key={c} className="p-2 text-center capitalize text-slate-700 dark:text-slate-300">
                          {c}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {metrics.confusion_matrix.map((row, rIdx) => {
                      const rowSum = row.reduce((a, b) => a + b, 0) || 1;
                      return (
                        <tr key={rIdx} className="border-t border-slate-200 dark:border-slate-800">
                          <td className="p-2 font-sans font-semibold capitalize text-slate-800 dark:text-slate-200">
                            {metrics.class_names[rIdx]}
                          </td>
                          {row.map((val, cIdx) => {
                            const isDiag = rIdx === cIdx;
                            const intensity = val / rowSum;
                            return (
                              <td
                                key={cIdx}
                                className={`p-2 text-center rounded ${
                                  isDiag
                                    ? 'bg-emerald-600/15 dark:bg-emerald-500/20 font-bold text-emerald-900 dark:text-emerald-200'
                                    : val > 0
                                    ? 'text-slate-700 dark:text-slate-300'
                                    : 'text-slate-400 dark:text-slate-600'
                                }`}
                                title={`True ${metrics.class_names[rIdx]} predicted as ${metrics.class_names[cIdx]}: ${val} (${(intensity * 100).toFixed(1)}%)`}
                              >
                                {val}
                              </td>
                            );
                          })}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Backbone Comparison & Multi-Source Domain Evaluation */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {metrics.model_comparison && metrics.model_comparison.length > 0 && (
              <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6">
                <h2 className="text-base font-bold text-slate-900 dark:text-slate-100 mb-3">
                  Architecture Comparison (Selected by Validation Macro-F1)
                </h2>
                <table className="w-full text-left border-collapse text-sm">
                  <thead>
                    <tr className="border-b border-slate-200 dark:border-slate-800 text-xs text-slate-500">
                      <th className="py-2">Backbone</th>
                      <th className="py-2 text-right">Val Macro-F1</th>
                      <th className="py-2 text-right">Temp T</th>
                      <th className="py-2 text-right">Train Time</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
                    {metrics.model_comparison.map((mc) => (
                      <tr key={mc.checkpoint}>
                        <td className="py-2.5 font-medium text-slate-900 dark:text-slate-100">
                          {mc.backbone}
                        </td>
                        <td className="py-2.5 text-right font-mono tabular-nums font-semibold text-emerald-700 dark:text-emerald-400">
                          {mc.val_macro_f1.toFixed(4)}
                        </td>
                        <td className="py-2.5 text-right font-mono tabular-nums">
                          {mc.temperature.toFixed(4)}
                        </td>
                        <td className="py-2.5 text-right font-mono tabular-nums text-slate-500">
                          {mc.wall_clock_seconds.toFixed(1)}s
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {metrics.per_source && (
              <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6">
                <h2 className="text-base font-bold text-slate-900 dark:text-slate-100 mb-3">
                  Cross-Source Domain Generalization (Held-Out Test Set)
                </h2>
                <table className="w-full text-left border-collapse text-sm">
                  <thead>
                    <tr className="border-b border-slate-200 dark:border-slate-800 text-xs text-slate-500">
                      <th className="py-2">Dataset Source</th>
                      <th className="py-2 text-right">Test Images</th>
                      <th className="py-2 text-right">Accuracy</th>
                      <th className="py-2 text-right">Macro-F1</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
                    {Object.entries(metrics.per_source).map(([src, stats]) => (
                      <tr key={src}>
                        <td className="py-2.5 font-medium capitalize text-slate-900 dark:text-slate-100">
                          {src}
                        </td>
                        <td className="py-2.5 text-right font-mono tabular-nums">{stats.count}</td>
                        <td className="py-2.5 text-right font-mono tabular-nums">
                          {(stats.accuracy * 100).toFixed(2)}%
                        </td>
                        <td className="py-2.5 text-right font-mono tabular-nums font-semibold">
                          {stats.macro_f1.toFixed(4)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
};
