import React, { useRef, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import {
  AlertCircle,
  Camera,
  FileUp,
  Image as ImageIcon,
  Loader2,
  Search,
  Sparkles,
  Trash2,
  Upload,
  X,
} from 'lucide-react';
import { apiClient } from '../api/client';
import { BatchPredictionResponse, PredictionRecord } from '../types/api';
import { PredictionResultCard } from './PredictionResultCard';

const ALLOWED_MIME_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
const MAX_FILE_BYTES = 10 * 1024 * 1024;
const MAX_BATCH_FILES = 20;

const QUICK_TEST_ITEMS = [
  'Plastic Bottles',
  'Newspaper',
  'Cardboard',
  'Metal Cans',
  'Glass Jars',
  'Banana Peels',
  'Batteries',
  'Old Mobile Phones',
  'Expired Medicines',
  'Sanitary Pads',
  'Broken Glass',
];

const DEMO_PHOTOS = [
  {
    id: 'plastic_bottle',
    title: 'Plastic Bottle',
    subtitle: 'Clear PET beverage bottle',
    category: 'Plastic',
    badgeColor: 'bg-sky-100 text-sky-800 dark:bg-sky-950 dark:text-sky-300',
    url: '/demo/demo_plastic_bottle.jpg',
    filename: 'plastic_bottle_grass_01.jpg',
  },
  {
    id: 'metal_can_pile',
    title: 'Crushed Metal Cans',
    subtitle: 'Full-frame aluminum pile',
    category: 'Metal',
    badgeColor: 'bg-slate-200 text-slate-800 dark:bg-slate-800 dark:text-slate-200',
    url: '/demo/demo_metal_can_pile.jpg',
    filename: 'metal_can_pile_01.jpg',
  },
  {
    id: 'organic_peel',
    title: 'Organic Food Waste',
    subtitle: 'Compostable fruit scraps',
    category: 'Organic',
    badgeColor: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300',
    url: '/demo/demo_organic_peel.jpg',
    filename: 'field_organic_02_realwaste.jpg',
  },
  {
    id: 'paper_box',
    title: 'Cardboard Box',
    subtitle: 'Flattened shipping fiber',
    category: 'Paper',
    badgeColor: 'bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300',
    url: '/demo/demo_paper_box.jpg',
    filename: 'field_paper_02_realwaste.jpg',
  },
];

interface SelectedFileItem {
  file: File;
  previewUrl: string;
  validationError: string | null;
}

export const ClassifyView: React.FC = () => {
  const queryClient = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const cameraInputRef = useRef<HTMLInputElement | null>(null);

  // Mode: 'search' | 'upload'
  const [activeMode, setActiveMode] = useState<'search' | 'upload'>('search');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedFiles, setSelectedFiles] = useState<SelectedFileItem[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [loadingDemoId, setLoadingDemoId] = useState<string | null>(null);
  const [clientError, setClientError] = useState<string | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);

  const [singleResult, setSingleResult] = useState<{
    prediction: PredictionRecord;
    previewUrl?: string;
  } | null>(null);
  const [batchResult, setBatchResult] = useState<BatchPredictionResponse | null>(null);

  const validateFileClientSide = (file: File): string | null => {
    const ext = file.name.split('.').pop()?.toLowerCase() || '';
    const validType =
      ALLOWED_MIME_TYPES.includes(file.type) || ['jpg', 'jpeg', 'png', 'webp'].includes(ext);
    if (!validType) {
      return `Unsupported format "${file.name}". Only JPEG, PNG, and WebP images are accepted.`;
    }
    if (file.size === 0) {
      return `File "${file.name}" is empty (0 bytes).`;
    }
    if (file.size > MAX_FILE_BYTES) {
      return `File "${file.name}" (${(file.size / (1024 * 1024)).toFixed(1)} MB) exceeds the 10 MB limit.`;
    }
    return null;
  };

  const handleFilesAdded = (fileList: FileList | null) => {
    if (!fileList || fileList.length === 0) return;
    setClientError(null);
    setApiError(null);

    const incoming = Array.from(fileList);
    if (incoming.length > MAX_BATCH_FILES) {
      setClientError(`Batch upload accepts up to ${MAX_BATCH_FILES} images at once.`);
      return;
    }

    const mapped: SelectedFileItem[] = incoming.map((file) => {
      const err = validateFileClientSide(file);
      return {
        file,
        previewUrl: URL.createObjectURL(file),
        validationError: err,
      };
    });

    const firstInvalid = mapped.find((m) => m.validationError);
    if (firstInvalid && mapped.length === 1) {
      setClientError(firstInvalid.validationError);
    }

    setSelectedFiles(mapped);
  };

  const clearSelection = () => {
    selectedFiles.forEach((item) => URL.revokeObjectURL(item.previewUrl));
    setSelectedFiles([]);
    setClientError(null);
    setApiError(null);
  };

  const handleTextSearchSubmit = async (e?: React.FormEvent, customQuery?: string) => {
    if (e) e.preventDefault();
    const q = (customQuery ?? searchQuery).trim();
    if (!q) {
      setClientError('Please enter a waste item name to classify.');
      return;
    }

    setClientError(null);
    setApiError(null);
    setIsSubmitting(true);
    setSingleResult(null);
    setBatchResult(null);

    try {
      const pred = await apiClient.classifyText(q);
      setSingleResult({
        prediction: pred,
      });
      queryClient.invalidateQueries({ queryKey: ['predictions'] });
      queryClient.invalidateQueries({ queryKey: ['analytics'] });
    } catch (err: any) {
      setApiError(err?.message || 'Classification request failed.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleRunImageClassification = async () => {
    const validItems = selectedFiles.filter((f) => !f.validationError);
    if (validItems.length === 0) {
      setClientError('Please select at least one valid JPEG, PNG, or WebP image (<= 10 MB).');
      return;
    }

    setIsSubmitting(true);
    setApiError(null);
    setSingleResult(null);
    setBatchResult(null);

    try {
      if (selectedFiles.length === 1 && !selectedFiles[0].validationError) {
        const pred = await apiClient.predictSingle(selectedFiles[0].file);
        setSingleResult({
          prediction: pred,
          previewUrl: selectedFiles[0].previewUrl,
        });
      } else {
        const res = await apiClient.predictBatch(selectedFiles.map((s) => s.file));
        setBatchResult(res);
      }
      queryClient.invalidateQueries({ queryKey: ['predictions'] });
      queryClient.invalidateQueries({ queryKey: ['analytics'] });
    } catch (err: any) {
      setApiError(err?.message || 'Classification request failed.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleLoadDemoPhoto = async (demo: (typeof DEMO_PHOTOS)[0]) => {
    setIsSubmitting(true);
    setLoadingDemoId(demo.id);
    setClientError(null);
    setApiError(null);
    setSingleResult(null);
    setBatchResult(null);

    try {
      const res = await fetch(demo.url);
      if (!res.ok) {
        throw new Error(`Failed to load demo image: ${res.statusText}`);
      }
      const blob = await res.blob();
      const file = new File([blob], demo.filename, { type: 'image/jpeg' });

      setSelectedFiles([
        {
          file,
          previewUrl: demo.url,
          validationError: null,
        },
      ]);

      const pred = await apiClient.predictSingle(file);
      setSingleResult({
        prediction: pred,
        previewUrl: demo.url,
      });
      queryClient.invalidateQueries({ queryKey: ['predictions'] });
      queryClient.invalidateQueries({ queryKey: ['analytics'] });
    } catch (err: any) {
      setApiError(err?.message || 'Failed to classify demo photo.');
    } finally {
      setIsSubmitting(false);
      setLoadingDemoId(null);
    }
  };

  return (
    <div className="space-y-8 max-w-5xl">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 border-b border-slate-200 dark:border-slate-800 pb-5">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900 dark:text-slate-100">
            Smart Waste Segregation Classifier
          </h1>
          <p className="text-sm text-slate-600 dark:text-slate-400 mt-1 max-w-2xl">
            Classify items by name search or upload waste photographs. Get authoritative recycling streams,
            dustbin colors, preparation steps, contamination alerts, and disposal instructions.
          </p>
        </div>

        {/* Tab switch */}
        <div className="inline-flex rounded-lg p-1 bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 shrink-0">
          <button
            type="button"
            onClick={() => setActiveMode('search')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-all ${
              activeMode === 'search'
                ? 'bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 shadow-sm'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
            }`}
          >
            <span className="flex items-center gap-1.5">
              <Search className="w-3.5 h-3.5" />
              Item Name Search
            </span>
          </button>
          <button
            type="button"
            onClick={() => setActiveMode('upload')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-all ${
              activeMode === 'upload'
                ? 'bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 shadow-sm'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
            }`}
          >
            <span className="flex items-center gap-1.5">
              <Upload className="w-3.5 h-3.5" />
              Upload Image (ML Model)
            </span>
          </button>
        </div>
      </div>

      {/* Mode 1: Search by Waste Item Name */}
      {activeMode === 'search' && (
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 space-y-5">
          <div>
            <h2 className="text-base font-semibold text-slate-900 dark:text-slate-100 flex items-center gap-2">
              <Search className="w-4 h-4 text-emerald-600" />
              Search Waste Material by Item Name
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
              Look up common household items, packaging, compostables, and hazardous materials for instant disposal rules.
            </p>
          </div>

          <form onSubmit={handleTextSearchSubmit} className="flex gap-2">
            <div className="relative flex-1">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Enter item name (e.g. plastic bottle, newspaper, batteries, old phone, banana peel)..."
                className="w-full pl-4 pr-10 py-3 rounded-xl border border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent text-sm"
              />
              {searchQuery && (
                <button
                  type="button"
                  onClick={() => setSearchQuery('')}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
                >
                  <X className="w-4 h-4" />
                </button>
              )}
            </div>
            <button
              type="submit"
              disabled={isSubmitting}
              className="px-6 py-3 bg-emerald-600 hover:bg-emerald-700 disabled:opacity-50 text-white font-medium text-sm rounded-xl transition-colors flex items-center gap-2 shrink-0"
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Classifying...
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  Classify Item
                </>
              )}
            </button>
          </form>

          {/* Quick-select suggestion pills for all 11 test items */}
          <div className="pt-2">
            <p className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-2">
              Quick test items:
            </p>
            <div className="flex flex-wrap gap-2">
              {QUICK_TEST_ITEMS.map((item) => (
                <button
                  key={item}
                  type="button"
                  onClick={() => {
                    setSearchQuery(item);
                    handleTextSearchSubmit(undefined, item);
                  }}
                  className="px-3 py-1 text-xs rounded-full bg-slate-100 hover:bg-emerald-50 dark:bg-slate-800 dark:hover:bg-slate-700/80 text-slate-700 dark:text-slate-300 hover:text-emerald-700 dark:hover:text-emerald-300 border border-slate-200 dark:border-slate-700 transition-colors"
                >
                  {item}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Mode 2: Upload Image */}
      {activeMode === 'upload' && (
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6">
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setIsDragging(false);
              handleFilesAdded(e.dataTransfer.files);
            }}
            className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors ${
              isDragging
                ? 'border-emerald-500 bg-emerald-50/50 dark:bg-emerald-950/20'
                : 'border-slate-300 dark:border-slate-700 hover:border-emerald-500/70'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp"
              multiple
              onChange={(e) => handleFilesAdded(e.target.files)}
              className="hidden"
              data-testid="file-upload-input"
            />
            <input
              ref={cameraInputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp"
              capture="environment"
              onChange={(e) => handleFilesAdded(e.target.files)}
              className="hidden"
            />

            <div className="w-12 h-12 rounded-xl bg-emerald-50 dark:bg-emerald-950/60 text-emerald-600 dark:text-emerald-400 flex items-center justify-center mx-auto mb-4">
              <Upload className="w-6 h-6" />
            </div>

            <h2 className="text-base font-semibold text-slate-900 dark:text-slate-100">
              Drag and drop waste image(s) here
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
              Supports JPEG, PNG, and WebP up to 10 MB per file · Up to 20 images in batch mode
            </p>

            <div className="mt-5 flex flex-wrap items-center justify-center gap-3">
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-emerald-600 hover:bg-emerald-700 rounded-lg transition-colors whitespace-nowrap shrink-0"
              >
                <FileUp className="w-4 h-4" />
                Browse Images
              </button>
              <button
                type="button"
                onClick={() => cameraInputRef.current?.click()}
                className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-slate-700 dark:text-slate-200 bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 rounded-lg transition-colors whitespace-nowrap shrink-0"
              >
                <Camera className="w-4 h-4" />
                Capture Photo
              </button>
            </div>
          </div>

          {selectedFiles.length > 0 && (
            <div className="mt-6 space-y-4">
              <div className="flex items-center justify-between">
                <p className="text-xs font-semibold text-slate-700 dark:text-slate-300">
                  {selectedFiles.length} {selectedFiles.length === 1 ? 'image' : 'images'} selected
                </p>
                <button
                  type="button"
                  onClick={clearSelection}
                  className="text-xs text-red-600 hover:text-red-700 flex items-center gap-1"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  Clear all
                </button>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 gap-3">
                {selectedFiles.map((item, idx) => (
                  <div
                    key={idx}
                    className="relative group rounded-lg overflow-hidden border border-slate-200 dark:border-slate-800 bg-slate-100 dark:bg-slate-800 aspect-square"
                  >
                    <img
                      src={item.previewUrl}
                      alt={item.file.name}
                      className="w-full h-full object-cover"
                    />
                    {item.validationError && (
                      <div className="absolute inset-0 bg-red-950/80 p-2 flex items-center justify-center text-center">
                        <p className="text-[10px] text-red-200 line-clamp-3">
                          {item.validationError}
                        </p>
                      </div>
                    )}
                  </div>
                ))}
              </div>

              <div className="pt-2 flex justify-end">
                <button
                  type="button"
                  onClick={handleRunImageClassification}
                  disabled={isSubmitting}
                  className="px-6 py-2.5 bg-emerald-600 hover:bg-emerald-700 disabled:opacity-50 text-white font-medium text-sm rounded-lg transition-colors flex items-center gap-2"
                >
                  {isSubmitting ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Classifying Image...
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-4 h-4" />
                      Run ONNX Classification
                    </>
                  )}
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Demo Sample Photos (One-Click Test) */}
      <div className="bg-slate-50 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 rounded-xl p-5 space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800 dark:text-slate-200 flex items-center gap-1.5">
              <ImageIcon className="w-4 h-4 text-emerald-600" />
              Try a Demo Photo (One-Click ML Test)
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              Click any demo photo below to test the ONNX model with authentic waste imagery.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-1">
          {DEMO_PHOTOS.map((demo) => (
            <button
              key={demo.id}
              type="button"
              disabled={isSubmitting}
              onClick={() => handleLoadDemoPhoto(demo)}
              className="group relative rounded-xl overflow-hidden border border-slate-200 dark:border-slate-800 hover:border-emerald-500 dark:hover:border-emerald-500 bg-white dark:bg-slate-900 text-left transition-all hover:shadow-md disabled:opacity-50 p-2 flex flex-col justify-between"
            >
              <div className="aspect-[4/3] w-full rounded-lg overflow-hidden bg-slate-100 dark:bg-slate-800 relative mb-2">
                <img
                  src={demo.url}
                  alt={demo.title}
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
                />
                <span
                  className={`absolute top-1.5 left-1.5 px-2 py-0.5 rounded text-[10px] font-bold uppercase ${demo.badgeColor}`}
                >
                  {demo.category}
                </span>
                {loadingDemoId === demo.id && (
                  <div className="absolute inset-0 bg-slate-950/70 flex items-center justify-center text-white">
                    <Loader2 className="w-5 h-5 animate-spin" />
                  </div>
                )}
              </div>
              <div>
                <p className="text-xs font-bold text-slate-900 dark:text-slate-100 truncate">
                  {demo.title}
                </p>
                <p className="text-[11px] text-slate-500 dark:text-slate-400 truncate">
                  {demo.subtitle}
                </p>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Error alert */}
      {(clientError || apiError) && (
        <div
          className="p-4 rounded-lg bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800/70 text-red-800 dark:text-red-200 text-sm flex items-start gap-2.5"
          role="alert"
        >
          <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400 shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold">Classification Error</p>
            <p className="text-xs mt-0.5">{clientError || apiError}</p>
          </div>
        </div>
      )}

      {/* Single Result */}
      {singleResult && (
        <div className="space-y-4">
          <h2 className="text-lg font-bold text-slate-900 dark:text-slate-100">
            Classification Result
          </h2>
          <PredictionResultCard
            prediction={singleResult.prediction}
            previewUrl={singleResult.previewUrl}
            onFeedbackUpdated={(updated) =>
              setSingleResult((prev) => (prev ? { ...prev, prediction: updated } : null))
            }
          />
        </div>
      )}

      {/* Batch Results */}
      {batchResult && (
        <div className="space-y-6">
          <div className="flex items-center justify-between border-b border-slate-200 dark:border-slate-800 pb-3">
            <h2 className="text-lg font-bold text-slate-900 dark:text-slate-100">
              Batch Classification Results ({batchResult.succeeded}/{batchResult.total} Succeeded)
            </h2>
          </div>
          <div className="space-y-4">
            {batchResult.results.map((item, idx) => (
              <div
                key={idx}
                className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5"
              >
                <div className="flex items-center justify-between mb-3 text-xs text-slate-500 font-mono">
                  <span>File: {item.filename}</span>
                  <span className={item.success ? 'text-emerald-600' : 'text-red-600'}>
                    {item.success ? 'Success' : item.error?.message || 'Failed'}
                  </span>
                </div>
                {item.prediction && (
                  <PredictionResultCard
                    prediction={item.prediction}
                    onFeedbackUpdated={(updated) => {
                      setBatchResult((prev) => {
                        if (!prev) return null;
                        const copy = [...prev.results];
                        copy[idx] = { ...copy[idx], prediction: updated };
                        return { ...prev, results: copy };
                      });
                    }}
                  />
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
