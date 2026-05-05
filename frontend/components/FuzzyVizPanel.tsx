"use client";

import { useMemo } from "react";
import { motion } from "framer-motion";
import { FuzzyInputs, Priorities } from "@/types/listing";

// ── MF math (mirrors engine.py exactly) ──────────────────────────────────────

function trapmfVal(x: number, a: number, b: number, c: number, d: number): number {
  if (x < a || x > d) return 0;
  if (x >= b && x <= c) return 1;
  if (x >= a && x < b && b > a) return (x - a) / (b - a);
  if (x > c && x <= d && d > c) return (d - x) / (d - c);
  return 0;
}

function trimfVal(x: number, a: number, b: number, c: number): number {
  if (x < a || x > c) return 0;
  if (x === b) return 1;
  if (x > a && x < b && b > a) return (x - a) / (b - a);
  if (x > b && x < c && c > b) return (c - x) / (c - b);
  return 0;
}

function boost(w: number) { return Math.pow(w, 1.5); }
function avg(...vals: number[]) { return vals.reduce((s, v) => s + v, 0) / vals.length; }

// ── Output MF functions (mirrors engine.py) ───────────────────────────────────
const OUTPUT_MFS: Record<string, (x: number) => number> = {
  cop:    (x) => trimfVal(x, 0, 0, 25),
  dusuk:  (x) => trimfVal(x, 15, 35, 55),
  orta:   (x) => trimfVal(x, 45, 60, 75),
  yuksek: (x) => trimfVal(x, 65, 80, 90),
  efsane: (x) => trimfVal(x, 85, 100, 100),
};
const OUTPUT_PEAKS: Record<string, number> = { cop: 0, dusuk: 35, orta: 60, yuksek: 80, efsane: 100 };
const OUTPUT_COLORS: Record<string, string> = {
  cop: "#ef4444", dusuk: "#f97316", orta: "#eab308", yuksek: "#22c55e", efsane: "#6366f1",
};
const OUTPUT_LABELS: Record<string, string> = {
  cop: "çöp", dusuk: "düşük", orta: "orta", yuksek: "yüksek", efsane: "efsane",
};

// ── Proper Mamdani centroid (same universe as backend: 0-100, step 1) ────────
function mamdaniCentroid(activations: RuleActivation[]): number {
  const N = 101;
  let num = 0;
  let den = 0;
  for (let i = 0; i < N; i++) {
    let maxClip = 0;
    for (const act of activations) {
      if (act.strength <= 0) continue;
      const mfVal = OUTPUT_MFS[act.output](i);
      const clipped = Math.min(act.strength, mfVal);
      if (clipped > maxClip) maxClip = clipped;
    }
    num += i * maxClip;
    den += maxClip;
  }
  return den > 0 ? num / den : 50;
}

// ── Types ─────────────────────────────────────────────────────────────────────
interface RuleActivation {
  text: string;
  strength: number;
  output: string;
  color: string;
  isCombo?: boolean;
}

interface MFTerm {
  label: string;
  color: string;
  compute: (x: number) => number;
}

// ── MF Graph ──────────────────────────────────────────────────────────────────
function buildPolyPoints(
  compute: (x: number) => number,
  displayMin: number, displayMax: number,
  W: number, H: number, padX: number, padY: number
): string {
  const N = 200;
  const toX = (v: number) => padX + ((v - displayMin) / (displayMax - displayMin)) * W;
  const toY = (v: number) => padY + (1 - v) * H;
  const pts: string[] = [];
  for (let i = 0; i <= N; i++) {
    const x = displayMin + (i / N) * (displayMax - displayMin);
    pts.push(`${toX(x).toFixed(1)},${toY(compute(x)).toFixed(1)}`);
  }
  pts.push(`${toX(displayMax).toFixed(1)},${toY(0).toFixed(1)}`);
  pts.push(`${toX(displayMin).toFixed(1)},${toY(0).toFixed(1)}`);
  return pts.join(" ");
}

function MFGraph({ title, value, displayMin, displayMax, terms }: {
  title: string; value: number; displayMin: number; displayMax: number; terms: MFTerm[];
}) {
  const W = 270; const H = 60; const padX = 15; const padY = 8;
  const svgW = W + padX * 2; const svgH = H + padY + 20;
  const toX = (v: number) => padX + ((v - displayMin) / (displayMax - displayMin)) * W;
  const lineX = toX(Math.min(Math.max(value, displayMin), displayMax));

  return (
    <div className="bg-white rounded-xl border border-gray-100 p-3 shadow-sm">
      <div className="flex justify-between items-center mb-1">
        <span className="text-xs font-bold text-gray-700">{title}</span>
        <span className="text-xs font-extrabold text-navy-deep bg-amber-50 px-2 py-0.5 rounded border border-amber-200">
          {value.toFixed(1)}
        </span>
      </div>
      <svg viewBox={`0 0 ${svgW} ${svgH}`} className="w-full" style={{ height: 90 }}>
        {[0, 0.5, 1].map((v) => (
          <line key={v} x1={padX} x2={padX + W} y1={padY + (1 - v) * H} y2={padY + (1 - v) * H}
            stroke="#f1f5f9" strokeWidth="1" />
        ))}
        {terms.map((t) => (
          <polygon key={t.label}
            points={buildPolyPoints(t.compute, displayMin, displayMax, W, H, padX, padY)}
            fill={t.color} fillOpacity={0.15} stroke={t.color} strokeWidth="1.5" strokeLinejoin="round" />
        ))}
        <line x1={lineX} x2={lineX} y1={padY} y2={padY + H}
          stroke="#1e293b" strokeWidth="1.5" strokeDasharray="3,2" />
        <circle cx={lineX} cy={padY + H} r="3" fill="#1e293b" />
        {[displayMin, Math.round((displayMin + displayMax) / 2), displayMax].map((v) => (
          <text key={v} x={toX(v)} y={padY + H + 14} textAnchor="middle" fontSize="8" fill="#94a3b8">{v}</text>
        ))}
      </svg>
      <div className="flex flex-wrap gap-1.5 mt-1">
        {terms.map((t) => {
          const val = t.compute(value);
          return (
            <span key={t.label} className="text-[10px] font-bold px-2 py-0.5 rounded-full border"
              style={{ color: t.color, borderColor: t.color, backgroundColor: t.color + "15" }}>
              {t.label}: {val.toFixed(2)}
            </span>
          );
        })}
      </div>
    </div>
  );
}

// ── Rule Firing Table ─────────────────────────────────────────────────────────
function RuleRow({ a, strength, maxStr, idx }: { a: RuleActivation; strength: number; maxStr: number; idx: number }) {
  return (
    <div className="flex items-center gap-2 rounded-lg px-2 py-1 bg-indigo-50/60 border border-indigo-100">
      <div className="flex-1 min-w-0">
        <p className="text-[10px] text-gray-600 font-mono truncate">{a.text}</p>
        <div className="mt-0.5 h-1.5 bg-gray-100 rounded-full overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${(strength / maxStr) * 100}%` }}
            transition={{ duration: 0.8, delay: idx * 0.04 }}
            className="h-full rounded-full"
            style={{ backgroundColor: a.color }}
          />
        </div>
      </div>
      <span className="text-[10px] font-bold text-gray-700 w-10 text-right shrink-0">
        {strength.toFixed(3)}
      </span>
      <span className="text-[10px] font-black px-1.5 py-0.5 rounded shrink-0"
        style={{ color: a.color, backgroundColor: a.color + "20" }}>
        {OUTPUT_LABELS[a.output]} ({OUTPUT_PEAKS[a.output]})
      </span>
    </div>
  );
}

function RuleFiringTable({ activations }: { activations: RuleActivation[] }) {
  // Combo rules: always show all, sorted by strength
  const combo = [...activations]
    .filter((a) => a.isCombo)
    .sort((a, b) => b.strength - a.strength);
  const maxStr = Math.max(...combo.map((a) => a.strength), 0.01);

  return (
    <div className="bg-white rounded-xl border border-gray-100 p-3 shadow-sm">
      <div className="flex items-center gap-2 mb-1">
        <p className="text-xs font-bold text-gray-700">Kural Ateşleme</p>
        <span className="text-[8px] text-indigo-400 bg-indigo-50 border border-indigo-100 px-1.5 py-0.5 rounded font-bold">fiyat & konum & boyut & oda</span>
      </div>
      <p className="text-[10px] text-gray-400 mb-3">ort(fiyat, konum, boyut, oda üyeliği) × min(ağırlıklar) — her kural her ilanda ateşlenir</p>

      {combo.length > 0 && (
        <div className="space-y-1.5">
          {combo.map((a, i) => <RuleRow key={i} a={a} strength={a.strength} maxStr={maxStr} idx={i} />)}
        </div>
      )}

      {combo.every((a) => a.strength < 0.005) && (
        <p className="text-[10px] text-gray-400 italic mt-2">Hiçbir kombinasyon kuralı ateşlenmedi — tüm koşullar aynı anda sağlanmadı.</p>
      )}
    </div>
  );
}

// ── Defuzzification ───────────────────────────────────────────────────────────
function DefuzzViz({ activations, centroid, actualScore }: {
  activations: RuleActivation[]; centroid: number; actualScore: number;
}) {
  // Compute aggregated output at each of 5 output peaks (for bar chart)
  const outputKeys = ["cop", "dusuk", "orta", "yuksek", "efsane"];
  const barHeights: Record<string, number> = {};
  for (const key of outputKeys) {
    let maxClip = 0;
    for (const a of activations) {
      if (a.output !== key || a.strength <= 0) continue;
      const mfPeak = OUTPUT_MFS[key](OUTPUT_PEAKS[key]);
      const clipped = Math.min(a.strength, mfPeak);
      if (clipped > maxClip) maxClip = clipped;
    }
    barHeights[key] = maxClip;
  }
  const maxBar = Math.max(...Object.values(barHeights), 0.01);

  return (
    <div className="bg-navy-deep rounded-xl p-4 shadow-md border border-navy-light">
      <p className="text-xs font-bold text-gray-200 mb-1">Defuzzification — Mamdani Centroid</p>
      <p className="text-[10px] text-gray-400 mb-3">
        Her kural çıktı fonksiyonunu aktivasyon seviyesinde keser → alanların centroid'i alınır
      </p>

      {/* Output bars */}
      <div className="flex items-end gap-1 mb-3 h-12 px-2">
        {outputKeys.map((key) => {
          const h = barHeights[key] / maxBar;
          return (
            <div key={key} className="flex-1 flex flex-col items-center justify-end gap-0.5">
              <span className="text-[8px] text-gray-400">
                {barHeights[key] > 0.01 ? barHeights[key].toFixed(2) : ""}
              </span>
              <motion.div
                initial={{ height: 0 }}
                animate={{ height: `${h * 40}px` }}
                transition={{ duration: 0.6 }}
                className="w-full rounded-t"
                style={{ backgroundColor: OUTPUT_COLORS[key], opacity: h > 0.05 ? 0.85 : 0.15 }}
              />
              <span className="text-[8px] text-gray-500">{OUTPUT_LABELS[key]}</span>
            </div>
          );
        })}
      </div>

      {/* 0-100 scale with centroid marker */}
      <div className="relative mx-2 mb-4">
        <div className="h-1 bg-gray-600 rounded-full" />
        <motion.div
          initial={{ left: "50%" }}
          animate={{ left: `${centroid}%` }}
          transition={{ duration: 1, type: "spring" }}
          className="absolute -top-1.5 transform -translate-x-1/2"
        >
          <div className="w-3 h-3 bg-yellow-400 rounded-full border-2 border-white shadow" />
        </motion.div>
        <div className="flex justify-between text-[8px] text-gray-500 mt-1.5">
          <span>0</span><span>25</span><span>50</span><span>75</span><span>100</span>
        </div>
      </div>

      {/* Formula & result */}
      <div className="bg-black/20 rounded-lg p-2.5 font-mono text-[9px] text-gray-300 leading-5">
        <p>Centroid = Σ(x · agg(x)) / Σ(agg(x))   x ∈ [0, 100]</p>
        <p className="text-gray-400">agg(x) = max küme birleşimi (101 nokta)</p>
        <div className="mt-1 pt-1 border-t border-gray-600 flex items-center justify-between">
          <span className="text-gray-300">Hesaplanan centroid:</span>
          <span className="text-yellow-400 font-black text-sm">{centroid.toFixed(1)}</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-gray-300">Gerçek Scout Skoru:</span>
          <span className="text-green-400 font-black text-sm">%{actualScore.toFixed(1)}</span>
        </div>
      </div>
    </div>
  );
}

// ── Main Panel ────────────────────────────────────────────────────────────────
interface Props {
  fuzzyInputs: FuzzyInputs;
  priorities: Priorities;
  actualScore: number;
}

export default function FuzzyVizPanel({ fuzzyInputs: fi, priorities, actualScore }: Props) {
  const { activations, centroid } = useMemo(() => {
    const p = fi.price_suitability;
    const l = fi.location_score;
    const s = fi.size_suitability;
    const m = fi.room_match;

    const mPahali  = trapmfVal(p, 70, 70, 75, 83);
    const mMakul   = trimfVal(p, 75, 83, 92);
    const mUcuz    = trapmfVal(p, 87, 93, 100, 100);
    const mUzak    = trapmfVal(l, 0, 0, 30, 60);
    const mOrtaL   = trimfVal(l, 40, 60, 80);
    const mYakin   = trapmfVal(l, 70, 90, 100, 100);
    const mKucuk   = trapmfVal(s, 0, 0, 25, 50);
    const mIdeal   = trapmfVal(s, 40, 70, 100, 100);
    const mUyumsuz = trapmfVal(m, 0, 0, 2, 5);
    const mKismi   = trimfVal(m, 4, 6, 8);
    const mUyumlu  = trapmfVal(m, 7, 9, 10, 10);

    const wp = boost(priorities.price);
    const wl = boost(priorities.location);
    const ws = boost(priorities.size);
    const wm = boost(priorities.rooms);

    const acts: RuleActivation[] = [
      // Single-input rules
      { text: "fiyat['ucuz'] → score['yuksek']",    strength: mUcuz  * wp,        output: "yuksek", color: OUTPUT_COLORS.yuksek },
      { text: "fiyat['pahali'] → score['cop']",      strength: mPahali * wp,       output: "cop",    color: OUTPUT_COLORS.cop },
      { text: "fiyat['makul'] → score['orta']",      strength: mMakul  * wp,       output: "orta",   color: OUTPUT_COLORS.orta },
      { text: "konum['yakin'] → score['efsane']",    strength: mYakin  * wl * 1.3, output: "efsane", color: OUTPUT_COLORS.efsane },
      { text: "konum['uzak'] → score['dusuk']",      strength: mUzak   * wl,       output: "dusuk",  color: OUTPUT_COLORS.dusuk },
      { text: "konum['orta'] → score['orta']",       strength: mOrtaL  * wl,       output: "orta",   color: OUTPUT_COLORS.orta },
      { text: "boyut['ideal'] → score['yuksek']",    strength: mIdeal  * ws,       output: "yuksek", color: OUTPUT_COLORS.yuksek },
      { text: "boyut['kucuk'] → score['dusuk']",     strength: mKucuk  * ws,       output: "dusuk",  color: OUTPUT_COLORS.dusuk },
      { text: "oda['uyumlu'] → score['efsane']",     strength: mUyumlu * wm,       output: "efsane", color: OUTPUT_COLORS.efsane },
      { text: "oda['kismi'] → score['orta']",        strength: mKismi  * wm,       output: "orta",   color: OUTPUT_COLORS.orta },
      { text: "oda['uyumsuz'] → score['cop']",       strength: mUyumsuz * wm,      output: "cop",    color: OUTPUT_COLORS.cop },
      // Multi-input combination rules (4 girdi)
      // Kombinasyon kuralları — strength: 4 üyelik değerinin ortalaması × min(ağırlıklar)
      { text: "ucuz & yakın & ideal & uyumlu → efsane",         strength: avg(mUcuz,   mYakin, mIdeal, mUyumlu)  * Math.min(wp, wl, ws, wm), output: "efsane", color: OUTPUT_COLORS.efsane, isCombo: true },
      { text: "makul & yakın & ideal & kısmi oda → yüksek",     strength: avg(mMakul,  mYakin, mIdeal, mKismi)   * Math.min(wp, wl, ws, wm), output: "yuksek", color: OUTPUT_COLORS.yuksek, isCombo: true },
      { text: "ucuz & orta konum & ideal & kısmi oda → yüksek", strength: avg(mUcuz,   mOrtaL, mIdeal, mKismi)   * Math.min(wp, wl, ws, wm), output: "yuksek", color: OUTPUT_COLORS.yuksek, isCombo: true },
      { text: "makul & orta konum & ideal & kısmi oda → orta",  strength: avg(mMakul,  mOrtaL, mIdeal, mKismi)   * Math.min(wp, wl, ws, wm), output: "orta",   color: OUTPUT_COLORS.orta,   isCombo: true },
      { text: "pahalı & uzak & küçük & uyumsuz → çöp",          strength: avg(mPahali, mUzak,  mKucuk, mUyumsuz) * Math.min(wp, wl, ws, wm), output: "cop",    color: OUTPUT_COLORS.cop,    isCombo: true },
      { text: "pahalı & uzak & küçük & kısmi oda → düşük",      strength: avg(mPahali, mUzak,  mKucuk, mKismi)   * Math.min(wp, wl, ws, wm), output: "dusuk",  color: OUTPUT_COLORS.dusuk,  isCombo: true },
      // Çelişki çözümü (fiyat vs konum)
      ...(priorities.location >= priorities.price
        ? [{ text: "pahalı & yakın → orta",  strength: avg(mPahali, mYakin) * wl, output: "orta",  color: OUTPUT_COLORS.orta,  isCombo: true }]
        : [{ text: "pahalı & yakın → düşük", strength: avg(mPahali, mYakin) * wp, output: "dusuk", color: OUTPUT_COLORS.dusuk, isCombo: true }]
      ),
    ];

    const centroid = mamdaniCentroid(acts);
    return { activations: acts, centroid };
  }, [fi, priorities]);

  return (
    <div className="mt-3 space-y-3">
      <p className="text-[10px] text-gray-400 font-medium text-center">
        Mamdani bulanık mantık — fuzzification → kural ateşleme → defuzzification (centroid)
      </p>

      {/* MF Grafikleri */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <MFGraph title="Fiyat Uyumu" value={fi.price_suitability} displayMin={65} displayMax={100}
          terms={[
            { label: "pahalı", color: "#ef4444", compute: (x) => trapmfVal(x, 70, 70, 75, 83) },
            { label: "makul",  color: "#f97316", compute: (x) => trimfVal(x, 75, 83, 92) },
            { label: "ucuz",   color: "#22c55e", compute: (x) => trapmfVal(x, 87, 93, 100, 100) },
          ]} />
        <MFGraph title="Konum Skoru" value={fi.location_score} displayMin={0} displayMax={100}
          terms={[
            { label: "uzak",  color: "#ef4444", compute: (x) => trapmfVal(x, 0, 0, 30, 60) },
            { label: "orta",  color: "#f97316", compute: (x) => trimfVal(x, 40, 60, 80) },
            { label: "yakın", color: "#22c55e", compute: (x) => trapmfVal(x, 70, 90, 100, 100) },
          ]} />
        <MFGraph title="Boyut Uyumu" value={fi.size_suitability} displayMin={0} displayMax={100}
          terms={[
            { label: "küçük", color: "#ef4444", compute: (x) => trapmfVal(x, 0, 0, 25, 50) },
            { label: "ideal", color: "#22c55e", compute: (x) => trapmfVal(x, 40, 70, 100, 100) },
          ]} />
        <MFGraph title="Oda Uyumu" value={fi.room_match} displayMin={0} displayMax={10}
          terms={[
            { label: "uyumsuz", color: "#ef4444", compute: (x) => trapmfVal(x, 0, 0, 2, 5) },
            { label: "kısmi",   color: "#f97316", compute: (x) => trimfVal(x, 4, 6, 8) },
            { label: "uyumlu",  color: "#22c55e", compute: (x) => trapmfVal(x, 7, 9, 10, 10) },
          ]} />
      </div>

      {/* Kural Ateşleme */}
      <RuleFiringTable activations={activations} />

      {/* Defuzzification */}
      <DefuzzViz activations={activations} centroid={centroid} actualScore={actualScore} />
    </div>
  );
}
