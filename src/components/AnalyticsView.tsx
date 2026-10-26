import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  BarChart3,
  CheckCircle2,
  Leaf,
  Package,
  RotateCcw,
  ShieldAlert,
  Sparkles,
  Wine,
  Wrench,
} from 'lucide-react';
import { apiClient } from '../api/client';

const CATEGORY_COLORS: Record<string, string> = {
  plastic: '#0284C7',
  paper: '#D97706',
  metal: '#475569',
  glass: '#0D9488',
  organic: '#059669',
  special_waste: '#E11D48',
  uncertain: '#94A3B8',
};

const RECYCLABILITY_COLORS: Record<string, string> = {
  Recyclable: '#059669',
  Compostable: '#10B981',
  'Special handling': '#D97706',
  'Non-recyclable': '#DC2626',
  Unknown: '#64748B',
};

interface Props {
  onNavigateClassify: () => void;
}

export const AnalyticsView: React.FC<Props> = ({ onNavigateClassify }) => {
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');
  const [includeDemo, setIncludeDemo] = useState(false);

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['analytics', dateFrom, dateTo, includeDemo],
    queryFn: () =>
      apiClient.getAnalyticsSummary({
        date_from: dateFrom || undefined,
        date_to: dateTo || undefined,
        include_demo: includeDemo,
      }),
  });

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => (
            <div
              key={i}
              className="h-28 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl animate-pulse"
            />
          ))}
        </div>
      </div>
    );
  }

  if (isError || !data) {
    return (
      <div className="p-6 rounded-xl bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800 text-sm text-red-800 dark:text-red-200">
        Failed to load analytics: {(error as any)?.message || 'Unknown error'}
      </div>
    );
  }

  const primaryCounts = data.primary_stream_counts || {
    plastic: 0,
    paper: 0,
    metal: 0,
    glass: 0,
    organic: 0,
  };
  const specialCount = data.special_waste_count || 0;
  const frequentItems = data.frequently_classified_items || [];

  return (
    <div className="space-y-8 max-w-5xl">
      {/* Header & Date Filters */}
      <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-4 border-b border-slate-200 dark:border-slate-800 pb-5">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900 dark:text-slate-100">
            Operational Waste & Diversion Analytics
          </h1>
          <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
            Real-time aggregations computed directly over persisted classification records.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3 text-xs">
          <label className="flex items-center gap-1.5 text-slate-600 dark:text-slate-400">
            <span>From:</span>
            <input
              type="date"
              value={dateFrom}
              onChange={(e) => setDateFrom(e.target.value)}
              className="px-2.5 py-1.5 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 font-mono"
            />
          </label>
          <label className="flex items-center gap-1.5 text-slate-600 dark:text-slate-400">
            <span>To:</span>
            <input
              type="date"
              value={dateTo}
              onChange={(e) => setDateTo(e.target.value)}
              className="px-2.5 py-1.5 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 font-mono"
            />
          </label>
          <label className="flex items-center gap-1.5 text-slate-600 dark:text-slate-400 cursor-pointer">
            <input
              type="checkbox"
              checked={includeDemo}
              onChange={(e) => setIncludeDemo(e.target.checked)}
              className="rounded border-slate-300"
            />
            <span>Include demo rows</span>
          </label>
          {(dateFrom || dateTo || includeDemo) && (
            <button
              type="button"
              onClick={() => {
                setDateFrom('');
                setDateTo('');
                setIncludeDemo(false);
              }}
              className="inline-flex items-center gap-1 text-slate-600 hover:text-slate-900 dark:text-slate-400 whitespace-nowrap shrink-0"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Reset
            </button>
          )}
        </div>
      </div>

      {/* Top KPI Cards (5 cards) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5">
          <p className="text-xs font-medium text-slate-500 dark:text-slate-400">Total Classified</p>
          <p className="text-3xl font-bold font-mono tabular-nums text-slate-900 dark:text-slate-100 mt-2">
            {data.total_predictions}
          </p>
          <p className="text-xs text-slate-500 mt-1">
            {data.feedback_count} user corrections
          </p>
        </div>

        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5">
          <p className="text-xs font-medium text-slate-500 dark:text-slate-400">
            Diversion Rate
          </p>
          <p className="text-3xl font-bold font-mono tabular-nums text-emerald-600 dark:text-emerald-400 mt-2">
            {(data.recyclable_or_compostable_rate * 100).toFixed(1)}%
          </p>
          <p className="text-xs text-slate-500 mt-1 font-mono tabular-nums">
            {data.recyclable_count} rec · {data.compostable_count} comp
          </p>
        </div>

        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5">
          <p className="text-xs font-medium text-slate-500 dark:text-slate-400">
            Special Waste Items
          </p>
          <p className="text-3xl font-bold font-mono tabular-nums text-rose-600 dark:text-rose-400 mt-2">
            {specialCount}
          </p>
          <p className="text-xs text-slate-500 mt-1">Hazardous / Non-curbside</p>
        </div>

        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5">
          <p className="text-xs font-medium text-slate-500 dark:text-slate-400">
            Mean Confidence
          </p>
          <p className="text-3xl font-bold font-mono tabular-nums text-slate-900 dark:text-slate-100 mt-2">
            {(data.average_confidence * 100).toFixed(1)}%
          </p>
          <p className="text-xs text-slate-500 mt-1">Calibrated probability</p>
        </div>

        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5">
          <p className="text-xs font-medium text-slate-500 dark:text-slate-400">
            Uncertain Flagged
          </p>
          <p className="text-3xl font-bold font-mono tabular-nums text-amber-600 dark:text-amber-400 mt-2">
            {(data.uncertain_rate * 100).toFixed(1)}%
          </p>
          <p className="text-xs text-slate-500 mt-1 font-mono tabular-nums">
            {data.uncertain_count} items withheld
          </p>
        </div>
      </div>

      {/* Primary Streams & Special Waste Stream Strip */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5 space-y-3">
        <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
          Stream Volume Distribution (5 Primary Streams + Special Waste)
        </h2>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 pt-1">
          <div className="p-3 rounded-lg bg-sky-50 dark:bg-sky-950/30 border border-sky-200 dark:border-sky-900/60">
            <span className="text-[11px] font-semibold text-sky-800 dark:text-sky-300">Plastic</span>
            <p className="text-xl font-bold font-mono text-slate-900 dark:text-slate-100 mt-1">
              {primaryCounts.plastic || 0}
            </p>
          </div>

          <div className="p-3 rounded-lg bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-900/60">
            <span className="text-[11px] font-semibold text-amber-800 dark:text-amber-300">Paper</span>
            <p className="text-xl font-bold font-mono text-slate-900 dark:text-slate-100 mt-1">
              {primaryCounts.paper || 0}
            </p>
          </div>

          <div className="p-3 rounded-lg bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700">
            <span className="text-[11px] font-semibold text-slate-700 dark:text-slate-300">Metal</span>
            <p className="text-xl font-bold font-mono text-slate-900 dark:text-slate-100 mt-1">
              {primaryCounts.metal || 0}
            </p>
          </div>

          <div className="p-3 rounded-lg bg-teal-50 dark:bg-teal-950/30 border border-teal-200 dark:border-teal-900/60">
            <span className="text-[11px] font-semibold text-teal-800 dark:text-teal-300">Glass</span>
            <p className="text-xl font-bold font-mono text-slate-900 dark:text-slate-100 mt-1">
              {primaryCounts.glass || 0}
            </p>
          </div>

          <div className="p-3 rounded-lg bg-emerald-50 dark:bg-emerald-950/30 border border-emerald-200 dark:border-emerald-900/60">
            <span className="text-[11px] font-semibold text-emerald-800 dark:text-emerald-300">Organic</span>
            <p className="text-xl font-bold font-mono text-slate-900 dark:text-slate-100 mt-1">
              {primaryCounts.organic || 0}
            </p>
          </div>

          <div className="p-3 rounded-lg bg-rose-50 dark:bg-rose-950/30 border border-rose-200 dark:border-rose-900/60">
            <span className="text-[11px] font-semibold text-rose-800 dark:text-rose-300">Special Waste</span>
            <p className="text-xl font-bold font-mono text-slate-900 dark:text-slate-100 mt-1">
              {specialCount}
            </p>
          </div>
        </div>
      </div>

      {data.total_predictions === 0 ? (
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-12 text-center">
          <BarChart3 className="w-8 h-8 text-slate-400 mx-auto mb-3" />
          <h2 className="text-base font-semibold text-slate-900 dark:text-slate-100">
            No Analytics Data Available for Selected Period
          </h2>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 max-w-md mx-auto">
            Classify waste images or search items to populate real database records and view category, recyclability, and confidence distributions.
          </p>
          <button
            type="button"
            onClick={onNavigateClassify}
            className="mt-4 px-4 py-2 text-xs font-medium text-white bg-emerald-600 hover:bg-emerald-700 rounded-lg transition-colors whitespace-nowrap shrink-0"
          >
            Classify Waste Item
          </button>
        </div>
      ) : (
        <>
          {/* Charts Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5">
              <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100 mb-4">
                Waste Stream Distribution by Category
              </h2>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data.category_distribution}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.25} />
                    <XAxis dataKey="category" tick={{ fontSize: 12 }} />
                    <YAxis allowDecimals={false} tick={{ fontSize: 12 }} />
                    <Tooltip />
                    <Bar dataKey="count" name="Items" radius={[4, 4, 0, 0]}>
                      {data.category_distribution.map((entry, idx) => (
                        <Cell
                          key={idx}
                          fill={CATEGORY_COLORS[entry.category] || '#059669'}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5">
              <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100 mb-4">
                Recyclability Class Breakdown
              </h2>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={data.recyclability_distribution}
                      dataKey="count"
                      nameKey="recyclability_class"
                      cx="50%"
                      cy="50%"
                      innerRadius={55}
                      outerRadius={85}
                      paddingAngle={3}
                    >
                      {data.recyclability_distribution.map((entry, idx) => (
                        <Cell
                          key={idx}
                          fill={RECYCLABILITY_COLORS[entry.recyclability_class] || '#64748B'}
                        />
                      ))}
                    </Pie>
                    <Tooltip />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5">
              <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100 mb-4">
                Daily Classification Volume & Diversion Trend
              </h2>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={data.time_series}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.25} />
                    <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                    <YAxis allowDecimals={false} tick={{ fontSize: 12 }} />
                    <Tooltip />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="count"
                      name="Total Classified"
                      stroke="#2563EB"
                      strokeWidth={2}
                    />
                    <Line
                      type="monotone"
                      dataKey="diverted"
                      name="Recyclable/Compostable"
                      stroke="#059669"
                      strokeWidth={2}
                    />
                    <Line
                      type="monotone"
                      dataKey="uncertain"
                      name="Uncertain"
                      stroke="#D97706"
                      strokeWidth={2}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5">
              <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100 mb-4">
                Calibrated Confidence Score Histogram (10 Bins)
              </h2>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data.confidence_histogram}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.25} />
                    <XAxis dataKey="bucket" tick={{ fontSize: 11 }} />
                    <YAxis allowDecimals={false} tick={{ fontSize: 12 }} />
                    <Tooltip />
                    <Bar dataKey="count" name="Predictions" fill="#059669" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Frequently Classified Items Section */}
          {frequentItems.length > 0 && (
            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5 space-y-3">
              <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                Most Frequently Classified Waste Materials
              </h2>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="border-b border-slate-200 dark:border-slate-800 text-slate-500 pb-2">
                      <th className="py-2 px-3">Rank</th>
                      <th className="py-2 px-3">Item Name</th>
                      <th className="py-2 px-3">Category</th>
                      <th className="py-2 px-3 text-right">Classified Count</th>
                      <th className="py-2 px-3 text-right">% of Total</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                    {frequentItems.map((item, i) => (
                      <tr key={i} className="hover:bg-slate-50/50 dark:hover:bg-slate-800/30">
                        <td className="py-2.5 px-3 font-mono font-semibold text-slate-400">
                          #{i + 1}
                        </td>
                        <td className="py-2.5 px-3 font-semibold text-slate-900 dark:text-slate-100">
                          {item.item}
                        </td>
                        <td className="py-2.5 px-3 capitalize">
                          <span
                            className="px-2 py-0.5 rounded-full text-[11px] font-medium"
                            style={{
                              backgroundColor: `${CATEGORY_COLORS[item.category] || '#64748B'}20`,
                              color: CATEGORY_COLORS[item.category] || '#64748B',
                            }}
                          >
                            {item.category === 'special_waste' ? 'Special Waste' : item.category}
                          </span>
                        </td>
                        <td className="py-2.5 px-3 text-right font-mono font-semibold tabular-nums">
                          {item.count}
                        </td>
                        <td className="py-2.5 px-3 text-right font-mono text-slate-500 tabular-nums">
                          {data.total_predictions > 0
                            ? `${((item.count / data.total_predictions) * 100).toFixed(1)}%`
                            : '0%'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};
