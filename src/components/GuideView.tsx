import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  AlertOctagon,
  AlertTriangle,
  BatteryCharging,
  BookOpen,
  CheckCircle2,
  Cpu,
  Flame,
  Info,
  Leaf,
  Package,
  Pill,
  Recycle,
  Search,
  ShieldAlert,
  ShieldCheck,
  Trash2,
  Wine,
  Wrench,
  Zap,
} from 'lucide-react';
import { apiClient } from '../api/client';
import {
  CommonMistakeData,
  MaterialDirectoryItem,
  PrimaryStreamData,
  SpecialWasteData,
  TaxonomyResponse,
} from '../types/api';

const DEFAULT_PRIMARY_STREAMS: Record<string, PrimaryStreamData> = {
  plastic: {
    name: 'Plastic',
    category: 'plastic',
    recyclability_class: 'Recyclable',
    bin: 'Blue Recycling Bin (Containers & Plastics)',
    badge_color: 'blue',
    examples: [
      'Water bottles & beverage bottles (PET #1)',
      'Milk jugs & detergent bottles (HDPE #2)',
      'Rigid shampoo & lotion bottles',
      'Food storage tubs & yogurt containers (PP #5)',
    ],
    prep_steps: [
      'Empty out all remaining liquids and rinse food residue.',
      'Keep caps and lids screwed on tightly to avoid loss during sorting.',
      'Remove non-recyclable spray pumps or plastic wrap labels.',
    ],
    contamination_guidance: [
      'Do NOT put soft plastic bags, shrink wrap, chip packets, or bubble wrap in curbside bins (they jam sorting conveyors).',
      'No styrofoam/expanded polystyrene (#6), plastic straws, or disposable cutlery.',
      'Never bag recyclables—place all plastic items loose in the bin.',
    ],
    local_acceptance:
      'Curbside programs almost universally accept rigid plastics #1 (PET) and #2 (HDPE). Plastics #3 (PVC), #4 (LDPE), #5 (PP), and #7 (other) depend on local MRF optical sorting equipment.',
  },
  paper: {
    name: 'Paper & Cardboard',
    category: 'paper',
    recyclability_class: 'Recyclable',
    bin: 'Blue Recycling Bin (Mixed Paper & Fiber)',
    badge_color: 'amber',
    examples: [
      'Corrugated cardboard boxes & shipping cartons',
      'Newspapers, magazines, and flyers',
      'Clean office printer paper & mail',
      'Paperboard cereal and pantry boxes',
    ],
    prep_steps: [
      'Ensure all paper and cardboard are clean, dry, and free of food grease.',
      'Flatten all corrugated cardboard boxes to conserve bin space.',
      'Remove plastic shipping tape, bubble inserts, or wax liners.',
    ],
    contamination_guidance: [
      'Greasy pizza box bottoms, used napkins, paper towels, and sanitary tissues are contaminated and belong in compost or general waste.',
      'Thermal receipts contain BPA/BPS and must NOT be recycled.',
      'Wax-coated or foil-lined beverage cartons require specialized hydropulping.',
    ],
    local_acceptance:
      'Widely accepted curbside. Dry clean fiber is baled and pulped up to 5–7 times to create new paperboard and newsprint.',
  },
  metal: {
    name: 'Metal',
    category: 'metal',
    recyclability_class: 'Recyclable',
    bin: 'Blue Recycling Bin (Metals & Cans)',
    badge_color: 'slate',
    examples: [
      'Aluminum beverage and soda cans',
      'Tin and steel food cans (soup, tuna, pet food)',
      'Clean aluminum pie pans & baking trays',
      'Clean aluminum foil (balled up)',
      'Metal jar lids and bottle caps',
    ],
    prep_steps: [
      'Empty and rinse beverage and food cans.',
      'Push tin can lids inside the can or detach carefully.',
      'Scrunch aluminum foil into a firm ball at least 2 inches (5 cm) wide so automated sorting screens capture it.',
    ],
    contamination_guidance: [
      'Pressurized aerosol cans with leftover product are hazardous (recycle only when completely empty).',
      'Do NOT put scrap metal wires, coat hangers, metal pipes, or electronics in curbside bins.',
      'No paint cans with wet paint residues.',
    ],
    local_acceptance:
      'Metals have infinite recyclability without quality loss. Aluminum and ferrous steel are automatically separated via eddy-current separators and industrial magnets.',
  },
  glass: {
    name: 'Glass',
    category: 'glass',
    recyclability_class: 'Recyclable',
    bin: 'Blue Recycling Bin or Dedicated Glass Drop-Off',
    badge_color: 'teal',
    examples: [
      'Clear, green, and amber beverage bottles',
      'Glass food jars (pasta sauce, jam, condiments)',
      'Cosmetic & skincare glass containers (cleaned)',
    ],
    prep_steps: [
      'Empty out contents and give a quick rinse.',
      'Remove metal or plastic lids and recycle them in their respective streams.',
      'Paper labels can remain attached; they burn off during high-temperature kiln remelting.',
    ],
    contamination_guidance: [
      'Do NOT place broken window glass, mirrors, Pyrex cookware, ceramics, porcelain, or light bulbs in glass recycling (they have different chemical formulations and melting temperatures).',
      'Never place broken container glass loose in the bin if it poses laceration risks to collection staff.',
    ],
    local_acceptance:
      '100% recyclable indefinitely. Some municipalities require separate glass caddies or neighborhood drop-off igloos to prevent glass shards from contaminating paper bales.',
  },
  organic: {
    name: 'Organic Waste',
    category: 'organic',
    recyclability_class: 'Compostable',
    bin: 'Green Organics / Compost Bin',
    badge_color: 'emerald',
    examples: [
      'Fruit and vegetable peels, cores, and scraps (banana peels, citrus, apple cores)',
      'Coffee grounds, unbleached filters, and tea bags (plastic-free)',
      'Eggshells and nut shells',
      'Yard waste: leaves, grass clippings, weeds, small plant trimmings',
    ],
    prep_steps: [
      'Remove all plastic PLU produce stickers, twist ties, and rubber bands.',
      'Separate organic food matter from any plastic wrapping or packaging.',
      'Drain excess liquids into the sink before placing solids in the kitchen caddy.',
    ],
    contamination_guidance: [
      'No plastic bags, plastic wrap, or conventional packaging.',
      'Check local rules regarding cooked meat, bones, and dairy (accepted in industrial aerobic/anaerobic digesters, but avoided in backyard worm bins).',
      'Only certified BPI/EN 13432 compostable liners if permitted locally.',
    ],
    local_acceptance:
      'Organic waste diversion prevents anaerobic decomposition in landfills, directly eliminating fugitive methane emissions and generating nutrient-rich humus for soil regeneration.',
  },
};

export const GuideView: React.FC = () => {
  const { data: taxonomy } = useQuery<TaxonomyResponse>({
    queryKey: ['taxonomy'],
    queryFn: () => apiClient.getTaxonomy(),
  });

  const [dirSearch, setDirSearch] = useState('');
  const [activeCategoryTab, setActiveCategoryTab] = useState('all');

  const primaryStreams = taxonomy?.primary_streams || DEFAULT_PRIMARY_STREAMS;
  const specialStreams = taxonomy?.special_waste_streams || {};
  const materialDirectory = taxonomy?.material_directory || [];
  const commonMistakes = taxonomy?.common_mistakes || [];

  const filteredDirectory = materialDirectory.filter((item: MaterialDirectoryItem) => {
    const matchesTab =
      activeCategoryTab === 'all'
        ? true
        : activeCategoryTab === 'special_waste'
        ? item.category === 'special_waste'
        : item.primary_stream === activeCategoryTab;

    if (!matchesTab) return false;
    if (!dirSearch.trim()) return true;

    const q = dirSearch.toLowerCase().trim();
    return (
      item.name.toLowerCase().includes(q) ||
      item.explanation.toLowerCase().includes(q) ||
      item.bin.toLowerCase().includes(q) ||
      item.aliases.some((a) => a.toLowerCase().includes(q))
    );
  });

  return (
    <div className="space-y-10 max-w-5xl">
      {/* Page Header */}
      <div className="border-b border-slate-200 dark:border-slate-800 pb-5">
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900 dark:text-slate-100">
          Five-Stream Material Taxonomy & Safety Guide
        </h1>
        <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
          Complete disposal matrix, searchable material directory, special waste handling protocols, and contamination prevention rules.
        </p>
      </div>

      {/* Municipal Notice Banner */}
      <div className="bg-amber-50/80 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-800/70 rounded-xl p-5 flex items-start gap-3">
        <AlertTriangle className="w-5 h-5 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
        <div className="text-sm text-amber-950 dark:text-amber-200">
          <h2 className="font-semibold">Local Municipal Recycling Rules May Vary</h2>
          <p className="text-xs mt-1 leading-relaxed">
            {taxonomy?.disclaimer ||
              'Disposal recommendations, target bin designations, and preparation instructions represent general circular economy recovery standards. Always verify local resin code acceptance (#1–#7), special waste drop-off events, and composting parameters with your local municipal sanitation department.'}
          </p>
        </div>
      </div>

      {/* 1. Five-Stream Material Taxonomy & Disposal Matrix */}
      <section className="space-y-4">
        <div className="flex items-center gap-2">
          <BookOpen className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
          <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100">
            1. Five-Stream Material Taxonomy & Disposal Matrix
          </h2>
        </div>
        <p className="text-xs text-slate-500 dark:text-slate-400">
          The standard 5-stream curbside segregation taxonomy applied across automated optical sorting plants and municipal collection vehicles.
        </p>

        <div className="grid grid-cols-1 gap-4">
          {Object.entries(primaryStreams).map(([key, stream]) => {
            const getBorderColor = () => {
              switch (key) {
                case 'plastic':
                  return 'border-sky-500/30 dark:border-sky-500/40 bg-sky-50/20 dark:bg-sky-950/20';
                case 'paper':
                  return 'border-amber-500/30 dark:border-amber-500/40 bg-amber-50/20 dark:bg-amber-950/20';
                case 'metal':
                  return 'border-slate-500/30 dark:border-slate-500/40 bg-slate-50/20 dark:bg-slate-900/40';
                case 'glass':
                  return 'border-teal-500/30 dark:border-teal-500/40 bg-teal-50/20 dark:bg-teal-950/20';
                case 'organic':
                  return 'border-emerald-500/30 dark:border-emerald-500/40 bg-emerald-50/20 dark:bg-emerald-950/20';
                default:
                  return 'border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900';
              }
            };

            return (
              <div
                key={key}
                className={`rounded-xl border p-5 space-y-3 transition-all ${getBorderColor()}`}
              >
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-200/60 dark:border-slate-800 pb-3">
                  <div className="flex items-center gap-2.5">
                    {key === 'plastic' && <Package className="w-5 h-5 text-sky-600 dark:text-sky-400" />}
                    {key === 'paper' && <BookOpen className="w-5 h-5 text-amber-600 dark:text-amber-400" />}
                    {key === 'metal' && <Wrench className="w-5 h-5 text-slate-600 dark:text-slate-400" />}
                    {key === 'glass' && <Wine className="w-5 h-5 text-teal-600 dark:text-teal-400" />}
                    {key === 'organic' && <Leaf className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />}
                    <h3 className="text-lg font-bold text-slate-900 dark:text-slate-100">
                      {stream.name}
                    </h3>
                  </div>

                  <div className="flex flex-wrap items-center gap-2 text-xs">
                    <span className="px-2.5 py-0.5 rounded-full font-semibold bg-emerald-100 text-emerald-800 dark:bg-emerald-950/80 dark:text-emerald-300">
                      {stream.recyclability_class}
                    </span>
                    <span className="font-medium text-slate-700 dark:text-slate-300 bg-white dark:bg-slate-800 px-2.5 py-0.5 rounded-full border border-slate-200 dark:border-slate-700">
                      {stream.bin}
                    </span>
                  </div>
                </div>

                {/* Examples */}
                <div>
                  <p className="text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1">
                    Accepted Example Materials:
                  </p>
                  <div className="flex flex-wrap gap-1.5">
                    {stream.examples.map((ex, i) => (
                      <span
                        key={i}
                        className="text-xs px-2 py-0.5 rounded bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300"
                      >
                        {ex}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Preparation & Contamination 2-col */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs pt-2">
                  <div className="space-y-1.5 bg-white/70 dark:bg-slate-900/60 p-3 rounded-lg border border-slate-200/60 dark:border-slate-800">
                    <h4 className="font-semibold text-slate-900 dark:text-slate-100 flex items-center gap-1.5">
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                      Preparation Instructions
                    </h4>
                    <ul className="space-y-1 text-slate-600 dark:text-slate-300">
                      {stream.prep_steps.map((step, i) => (
                        <li key={i} className="flex items-start gap-1.5">
                          <span className="text-emerald-600 font-bold">&bull;</span>
                          <span>{step}</span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div className="space-y-1.5 bg-white/70 dark:bg-slate-900/60 p-3 rounded-lg border border-slate-200/60 dark:border-slate-800">
                    <h4 className="font-semibold text-rose-900 dark:text-rose-300 flex items-center gap-1.5">
                      <AlertOctagon className="w-3.5 h-3.5 text-rose-600" />
                      Contamination Warnings
                    </h4>
                    <ul className="space-y-1 text-slate-600 dark:text-slate-300">
                      {stream.contamination_guidance.map((c, i) => (
                        <li key={i} className="flex items-start gap-1.5">
                          <span className="text-rose-600 font-bold">&bull;</span>
                          <span>{c}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>

                {/* Local acceptance note */}
                <p className="text-[11px] text-slate-500 dark:text-slate-400 italic pt-1">
                  Local recovery note: {stream.local_acceptance}
                </p>
              </div>
            );
          })}
        </div>
      </section>

      {/* 2. Special Waste Safety Protocols (Non-Curbside / Hazardous) */}
      <section className="space-y-4">
        <div className="flex items-center gap-2">
          <ShieldAlert className="w-5 h-5 text-rose-600 dark:text-rose-400" />
          <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100">
            2. Special Waste Safety & Mandatory Hazardous Handling
          </h2>
        </div>
        <p className="text-xs text-slate-500 dark:text-slate-400">
          The following materials MUST NEVER be placed into curbside recycling or composting bins. Follow authorized collection channels to prevent facility fires, chemical hazards, and bio-pathogen exposure.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {Object.entries(specialStreams).map(([key, item]: [string, SpecialWasteData]) => {
            const getIcon = () => {
              switch (key) {
                case 'batteries':
                  return <BatteryCharging className="w-5 h-5 text-rose-600" />;
                case 'electronic_waste':
                  return <Cpu className="w-5 h-5 text-amber-600" />;
                case 'expired_medicines':
                  return <Pill className="w-5 h-5 text-indigo-600" />;
                case 'hazardous_chemicals':
                  return <Flame className="w-5 h-5 text-rose-600" />;
                case 'sanitary_pads':
                  return <Trash2 className="w-5 h-5 text-slate-600" />;
                case 'broken_glass':
                  return <AlertTriangle className="w-5 h-5 text-rose-600" />;
                case 'fluorescent_tubes':
                  return <Zap className="w-5 h-5 text-cyan-600" />;
                default:
                  return <ShieldAlert className="w-5 h-5 text-rose-600" />;
              }
            };

            return (
              <div
                key={key}
                className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5 space-y-3 flex flex-col justify-between"
              >
                <div className="space-y-2">
                  <div className="flex items-start gap-2.5">
                    <div className="p-2 rounded-lg bg-slate-100 dark:bg-slate-800 shrink-0 mt-0.5">
                      {getIcon()}
                    </div>
                    <div>
                      <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100">
                        {item.name}
                      </h3>
                      <p className="text-xs font-semibold text-rose-600 dark:text-rose-400">
                        Target: {item.recommended_collection}
                      </p>
                    </div>
                  </div>

                  {/* Warning Box */}
                  <div className="p-2.5 rounded bg-rose-50/80 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-900/60 text-[11px] text-rose-900 dark:text-rose-200 leading-snug">
                    <strong className="font-semibold">HAZARD WARNING: </strong>
                    {item.warning}
                  </div>

                  {/* Safe Handling Steps */}
                  <div className="text-xs space-y-1 pt-1">
                    <p className="font-semibold text-slate-800 dark:text-slate-200">
                      Safe Handling Instructions:
                    </p>
                    <ul className="space-y-1 text-slate-600 dark:text-slate-300">
                      {item.safe_handling.map((sh, idx) => (
                        <li key={idx} className="flex items-start gap-1.5">
                          <span className="text-emerald-600 font-bold">&bull;</span>
                          <span>{sh}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>

                <div className="pt-2 border-t border-slate-100 dark:border-slate-800 text-[11px] text-slate-500 dark:text-slate-400">
                  <strong className="text-slate-700 dark:text-slate-300">Authorized Channels: </strong>
                  {item.local_acceptance}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* 3. Searchable Material Directory */}
      <section className="space-y-4">
        <div className="flex items-center gap-2">
          <Search className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
          <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100">
            3. Searchable Material Directory
          </h2>
        </div>
        <p className="text-xs text-slate-500 dark:text-slate-400">
          Search specific everyday items to inspect their preparation rules, target bins, and contamination risks.
        </p>

        {/* Search Bar + Tabs */}
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-4 space-y-3">
          <div className="relative">
            <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={dirSearch}
              onChange={(e) => setDirSearch(e.target.value)}
              placeholder="Search directory (e.g. plastic bottle, metal can, batteries, pizza box, mirror)..."
              className="w-full pl-10 pr-4 py-2.5 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 text-xs sm:text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500"
            />
          </div>

          <div className="flex flex-wrap gap-1.5 text-xs">
            {['all', 'plastic', 'paper', 'metal', 'glass', 'organic', 'special_waste'].map((tab) => (
              <button
                key={tab}
                type="button"
                onClick={() => setActiveCategoryTab(tab)}
                className={`px-3 py-1 rounded-lg font-medium capitalize transition-colors ${
                  activeCategoryTab === tab
                    ? 'bg-emerald-600 text-white shadow-sm'
                    : 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                }`}
              >
                {tab === 'all'
                  ? 'All Materials'
                  : tab === 'special_waste'
                  ? 'Special Waste'
                  : tab}
              </button>
            ))}
          </div>
        </div>

        {/* Directory Items List */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {filteredDirectory.length === 0 ? (
            <div className="col-span-2 text-center py-8 text-xs text-slate-500 dark:text-slate-400">
              No materials match your search filter &quot;{dirSearch}&quot;.
            </div>
          ) : (
            filteredDirectory.map((item: MaterialDirectoryItem) => (
              <div
                key={item.id}
                className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-4 space-y-2.5"
              >
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <h3 className="font-bold text-sm text-slate-900 dark:text-slate-100">
                      {item.name}
                    </h3>
                    <p className="text-xs text-emerald-600 dark:text-emerald-400 font-medium">
                      {item.bin}
                    </p>
                  </div>
                  <span className="text-[11px] px-2 py-0.5 rounded-full font-semibold uppercase bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300">
                    {item.recyclability_class}
                  </span>
                </div>

                <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
                  {item.explanation}
                </p>

                {item.prep_steps.length > 0 && (
                  <div className="text-xs space-y-1">
                    <p className="font-semibold text-slate-800 dark:text-slate-200">
                      Prep:
                    </p>
                    <ul className="list-disc list-inside space-y-0.5 text-slate-600 dark:text-slate-400">
                      {item.prep_steps.map((p, i) => (
                        <li key={i}>{p}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {item.contamination_warnings.length > 0 && (
                  <p className="text-[11px] text-rose-600 dark:text-rose-400 font-medium">
                    Warning: {item.contamination_warnings.join(' ')}
                  </p>
                )}
              </div>
            ))
          )}
        </div>
      </section>

      {/* 4. Common Recycling Mistakes */}
      <section className="space-y-4">
        <div className="flex items-center gap-2">
          <AlertOctagon className="w-5 h-5 text-amber-600 dark:text-amber-400" />
          <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100">
            4. Common Recycling Mistakes & Facility Impacts
          </h2>
        </div>
        <p className="text-xs text-slate-500 dark:text-slate-400">
          Avoid these frequent contamination errors that cause sorting plant shutdowns and batch spoilage.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {commonMistakes.map((m: CommonMistakeData, idx: number) => (
            <div
              key={idx}
              className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5 space-y-2.5"
            >
              <div className="flex items-start gap-2 text-rose-700 dark:text-rose-400 font-bold text-xs sm:text-sm">
                <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                <h4>Mistake: {m.mistake}</h4>
              </div>

              <div className="text-xs text-slate-600 dark:text-slate-300">
                <strong className="text-slate-800 dark:text-slate-200">Consequence: </strong>
                {m.consequence}
              </div>

              <div className="p-2.5 rounded-lg bg-emerald-50/70 dark:bg-emerald-950/30 border border-emerald-200 dark:border-emerald-900 text-xs text-emerald-900 dark:text-emerald-200">
                <strong className="font-semibold">Correct Action: </strong>
                {m.correct_action}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* 5. Calibration & Limitations */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <section className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 space-y-3">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-5 h-5 text-emerald-600" />
            <h2 className="text-base font-bold text-slate-900 dark:text-slate-100">
              5. Calibration & Confidence Gating
            </h2>
          </div>
          <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
            Raw neural network logits are calibrated on held-out validation images using temperature scaling
            to minimize Expected Calibration Error (ECE). The system enforces 3 safety gates:
          </p>
          <ul className="space-y-1.5 text-xs font-mono text-slate-700 dark:text-slate-300">
            <li>• HIGH (&ge; 0.80): Definite classification; confident stream returned.</li>
            <li>• MEDIUM (0.50 – 0.80): Verified category with candidate breakdown.</li>
            <li>• LOW (&lt; 0.50) / Low Margin / High Entropy: Returns &quot;uncertain&quot; and safe handling advice.</li>
          </ul>
        </section>

        <section className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 space-y-3">
          <div className="flex items-center gap-2">
            <Info className="w-5 h-5 text-sky-600" />
            <h2 className="text-base font-bold text-slate-900 dark:text-slate-100">
              6. Known Limitations & Pile Handling
            </h2>
          </div>
          <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
            The closed-set 5-class vision backbone classifies single items best. Multi-item piles and out-of-distribution items are guarded by:
          </p>
          <ul className="space-y-1.5 text-xs text-slate-600 dark:text-slate-300">
            <li>• Adaptive multi-view tile inference and de-colourised grayscale checks.</li>
            <li>• Normalized Shannon entropy gating to flag near-uniform candidate outputs.</li>
            <li>• Blur variance and border saturation heuristics.</li>
          </ul>
        </section>
      </div>

      {/* Security & Privacy */}
      <section className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 space-y-2">
        <div className="flex items-center gap-2">
          <ShieldCheck className="w-5 h-5 text-emerald-600" />
          <h2 className="text-base font-bold text-slate-900 dark:text-slate-100">
            7. Image Security & Privacy Policy
          </h2>
        </div>
        <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
          All uploaded images are validated by magic bytes, bounded to 10 MB and 25 MP, and stripped of all EXIF/GPS metadata by reconstructing a clean pixel buffer before generating sanitized 256px thumbnails.
        </p>
      </section>
    </div>
  );
};
