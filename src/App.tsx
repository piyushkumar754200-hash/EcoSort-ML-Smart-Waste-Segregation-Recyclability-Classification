import React, { useEffect, useState } from 'react';
import { QueryClient, QueryClientProvider, useQuery } from '@tanstack/react-query';
import { AlertTriangle, Moon, Recycle, Sun } from 'lucide-react';
import { apiClient } from './api/client';
import { AnalyticsView } from './components/AnalyticsView';
import { ClassifyView } from './components/ClassifyView';
import { ErrorBoundary } from './components/ErrorBoundary';
import { GuideView } from './components/GuideView';
import { HistoryView } from './components/HistoryView';
import { ModelView } from './components/ModelView';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

type ActiveTab = 'classify' | 'history' | 'analytics' | 'model' | 'guide';

const MainShell: React.FC = () => {
  const [activeTab, setActiveTab] = useState<ActiveTab>('classify');
  const [darkMode, setDarkMode] = useState<boolean>(true);

  useEffect(() => {
    const root = document.documentElement;
    if (darkMode) {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
  }, [darkMode]);

  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: () => apiClient.getHealth(),
    refetchInterval: 15000,
  });

  const navItems: Array<{ id: ActiveTab; label: string }> = [
    { id: 'classify', label: 'Classify' },
    { id: 'history', label: 'History' },
    { id: 'analytics', label: 'Analytics' },
    { id: 'model', label: 'Model' },
    { id: 'guide', label: 'Guide' },
  ];

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900 dark:bg-slate-950 dark:text-slate-100">
      {/* Strict 3-Zone Top Bar Contract */}
      <header className="sticky top-0 z-30 bg-white/95 dark:bg-slate-900/95 backdrop-blur border-b border-slate-200 dark:border-slate-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between gap-8">
          {/* Zone 1: Single text element Brand wordmark */}
          <a
            href="#classify"
            onClick={(e) => {
              e.preventDefault();
              setActiveTab('classify');
            }}
            className="text-lg font-bold tracking-tight text-slate-900 dark:text-slate-100 whitespace-nowrap shrink-0 font-display"
          >
            EcoSort ML
          </a>

          {/* Zone 2: 5 concise single-line text navigation links */}
          <nav aria-label="Main Navigation" className="flex items-center gap-5 sm:gap-7 overflow-x-auto">
            {navItems.map((item) => {
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => setActiveTab(item.id)}
                  className={`text-sm font-medium transition-colors whitespace-nowrap shrink-0 py-1 border-b-2 ${
                    isActive
                      ? 'border-emerald-600 text-slate-900 dark:text-slate-100 dark:border-emerald-400'
                      : 'border-transparent text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                  }`}
                >
                  {item.label}
                </button>
              );
            })}
          </nav>

          {/* Zone 3: 1 primary action */}
          <div className="flex items-center shrink-0">
            <button
              type="button"
              onClick={() => setDarkMode((d) => !d)}
              aria-label={darkMode ? 'Switch to light theme' : 'Switch to dark theme'}
              className="inline-flex items-center gap-2 px-3.5 py-1.5 text-xs font-medium text-slate-700 dark:text-slate-200 bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 rounded-lg transition-colors whitespace-nowrap shrink-0"
            >
              {darkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
              <span>{darkMode ? 'Light Theme' : 'Dark Theme'}</span>
            </button>
          </div>
        </div>
      </header>

      {/* Conditional Smoke-Test or Model Unloaded Banner */}
      {health?.is_smoke_test && (
        <div
          className="bg-amber-500 text-slate-950 px-4 py-2 text-xs font-semibold text-center"
          role="status"
        >
          SMOKE-TEST, NOT FOR PRODUCTION — The currently loaded model checkpoint is marked as a smoke-test
          artifact.
        </div>
      )}

      {health && !health.model_loaded && (
        <div
          className="bg-amber-50 dark:bg-amber-950/50 border-b border-amber-200 dark:border-amber-800 px-4 py-2.5 text-xs text-amber-900 dark:text-amber-200"
          role="alert"
        >
          <div className="max-w-7xl mx-auto flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0 text-amber-600" />
            <span>
              <strong>Model Not Loaded (HTTP 503):</strong> ONNX weights are currently being trained or have
              not been exported to <code className="font-mono">artifacts/models/model.onnx</code> yet.
            </span>
          </div>
        </div>
      )}

      {/* Main Content Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-8">
        <ErrorBoundary>
          {activeTab === 'classify' && <ClassifyView />}
          {activeTab === 'history' && (
            <HistoryView onNavigateClassify={() => setActiveTab('classify')} />
          )}
          {activeTab === 'analytics' && (
            <AnalyticsView onNavigateClassify={() => setActiveTab('classify')} />
          )}
          {activeTab === 'model' && <ModelView />}
          {activeTab === 'guide' && <GuideView />}
        </ErrorBoundary>
      </main>

      {/* Quiet Footer */}
      <footer className="border-t border-slate-200 dark:border-slate-800 py-5 px-4 sm:px-6 text-xs text-slate-500 dark:text-slate-400">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Recycle className="w-4 h-4 text-emerald-600" />
            <span>EcoSort ML — Smart Waste Segregation & Recyclability Classification</span>
          </div>
          <div>
            <span>
              Disposal guidance is general; verify local municipal recycling rules.
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <MainShell />
    </QueryClientProvider>
  );
}
