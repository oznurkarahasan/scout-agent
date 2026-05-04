"use client";

import { useEffect, useState } from "react";
import { fetchCities, fetchDistricts } from "@/lib/api";
import { FilterState, Priorities } from "@/types/listing";
import { motion } from "framer-motion";
import { SlidersHorizontal, Search, Map, Banknote, Maximize, DoorOpen, Target } from "lucide-react";

interface Props {
  filters: FilterState;
  onChange: (f: FilterState) => void;
  onSearch: () => void;
  loading: boolean;
}

const ROOM_OPTIONS = ["1+1", "2+1", "3+1", "4+1"];

const PRIORITY_LABELS: Record<keyof Priorities, string> = {
  price: "Fiyat Uyumluluğu",
  location: "Konum Skoru",
  size: "m² Uyumu",
  quality: "İlan Görselleri/Kalite",
  llm: "LLM Skoru",
};

const VISIBLE_PRIORITY_KEYS: (keyof Priorities)[] = ["price", "location", "size", "quality"];

export default function FilterSidebar({ filters, onChange, onSearch, loading }: Props) {
  const [cities, setCities] = useState<string[]>([]);
  const [districts, setDistricts] = useState<string[]>([]);

  useEffect(() => {
    fetchCities().then(setCities);
  }, []);

  useEffect(() => {
    if (filters.city) {
      fetchDistricts(filters.city).then(setDistricts);
    }
  }, [filters.city]);

  const set = (patch: Partial<FilterState>) => onChange({ ...filters, ...patch });

  const toggleRoom = (room: string) => {
    if (room === "Hepsi") {
      set({ rooms: ["Hepsi"] });
      return;
    }
    const without = filters.rooms.filter((r) => r !== "Hepsi");
    const next = without.includes(room)
      ? without.filter((r) => r !== room)
      : [...without, room];
    set({ rooms: next.length ? next : ["Hepsi"] });
  };

  const setPriority = (key: keyof Priorities, val: number) =>
    set({ priorities: { ...filters.priorities, [key]: val } });

  const priceRange =
    filters.listing_type === "Satılık"
      ? { min: 500000, max: 20000000, step: 100000 }
      : filters.listing_type === "Kiralık"
      ? { min: 2000, max: 100000, step: 500 }
      : { min: 2000, max: 20000000, step: 5000 };

  return (
    <motion.aside 
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className="w-72 min-w-[18rem] max-h-[calc(100vh-2rem)] overflow-y-auto bg-white/80 backdrop-blur-md shadow-lg border border-gray-100 rounded-2xl p-6 flex flex-col gap-6 self-start sticky top-4 custom-scrollbar"
    >
      <div className="flex flex-col gap-2">
        <h1 className="text-xl font-bold text-slate-900 tracking-tight flex items-center gap-2">
          <SlidersHorizontal className="text-gold" size={20} /> Arama Filtreleri
        </h1>
        <button
          onClick={onSearch}
          className="w-full py-2.5 bg-slate-900 text-white rounded-xl font-bold hover:bg-slate-800 hover:shadow-lg hover:-translate-y-0.5 disabled:opacity-50 transition-all shadow-sm flex items-center justify-center gap-2"
        >
          {loading ? <Search className="animate-spin" size={18} /> : <Search size={18} />}
          {loading ? "Aranıyor..." : "Aramayı Başlat"}
        </button>
      </div>

      {/* Listing type */}
      <div>
        <p className="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-1.5"><Map size={16} className="text-gray-400" /> İlan Tipi</p>
        <div className="flex gap-2 bg-gray-50 p-1 rounded-lg border border-gray-100">
          {(["Hepsi", "Kiralık", "Satılık"] as const).map((t) => (
            <button
              key={t}
              onClick={() => set({ listing_type: t })}
              className={`flex-1 py-1.5 rounded-md text-sm font-medium transition-all ${
                filters.listing_type === t
                  ? "bg-white text-slate-900 shadow-sm border border-amber-500"
                  : "text-gray-500 hover:text-slate-900"
              }`}
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      {/* City & District */}
      <div className="flex flex-col gap-3">
        <div>
          <p className="text-sm font-semibold text-gray-700 mb-1.5">Şehir</p>
          <select
            value={filters.city}
            onChange={(e) => set({ city: e.target.value, district: "" })}
            className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm bg-gray-50 focus:bg-white focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-transparent transition-all"
          >
            {cities.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </div>

        <div>
          <p className="text-sm font-semibold text-gray-700 mb-1.5">İlçe</p>
          <select
            value={filters.district}
            onChange={(e) => set({ district: e.target.value })}
            className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm bg-gray-50 focus:bg-white focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-transparent transition-all"
          >
            <option value="">Tümü</option>
            {districts.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Price */}
      <div>
        <p className="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-1.5"><Banknote size={16} className="text-gray-400" /> Fiyat Aralığı (TL)</p>
        <div className="flex gap-3 items-center">
          <div className="flex-1 relative">
            <span className="absolute left-3 top-2.5 text-gray-400 text-sm">₺</span>
            <input
              type="number"
              min={0}
              value={filters.min_price || ""}
              onChange={(e) => set({ min_price: Number(e.target.value) })}
              placeholder="En Az"
              className="w-full border border-gray-200 rounded-lg pl-7 pr-3 py-2 text-sm bg-gray-50 focus:bg-white focus:outline-none focus:ring-2 focus:ring-amber-500 transition-all appearance-none"
            />
          </div>
          <span className="text-gray-400">-</span>
          <div className="flex-1 relative">
            <span className="absolute left-3 top-2.5 text-gray-400 text-sm">₺</span>
            <input
              type="number"
              min={0}
              value={filters.max_price || ""}
              onChange={(e) => set({ max_price: Number(e.target.value) })}
              placeholder="En Çok"
              className="w-full border border-gray-200 rounded-lg pl-7 pr-3 py-2 text-sm bg-gray-50 focus:bg-white focus:outline-none focus:ring-2 focus:ring-amber-500 transition-all appearance-none"
            />
          </div>
        </div>
      </div>

      {/* m2 */}
      <div>
        <p className="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-1.5"><Maximize size={16} className="text-gray-400" /> Büyüklük (m²)</p>
        <div className="flex gap-3 items-center">
          <div className="flex-1">
            <input
              type="number"
              min={0}
              value={filters.min_m2 || ""}
              onChange={(e) => set({ min_m2: Number(e.target.value) })}
              placeholder="Min"
              className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm bg-gray-50 focus:bg-white focus:outline-none focus:ring-2 focus:ring-amber-500 transition-all"
            />
          </div>
          <span className="text-gray-400">-</span>
          <div className="flex-1">
            <input
              type="number"
              min={0}
              value={filters.max_m2 || ""}
              onChange={(e) => set({ max_m2: Number(e.target.value) })}
              placeholder="Max"
              className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm bg-gray-50 focus:bg-white focus:outline-none focus:ring-2 focus:ring-amber-500 transition-all"
            />
          </div>
        </div>
      </div>

      {/* Rooms */}
      <div>
        <p className="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-1.5"><DoorOpen size={16} className="text-gray-400" /> Oda Sayısı</p>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => set({ rooms: ["Hepsi"] })}
            className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-all ${
              filters.rooms.includes("Hepsi")
                ? "bg-slate-900 text-amber-500 shadow-sm border border-slate-900"
                : "bg-gray-50 text-gray-600 hover:text-slate-900 hover:border-amber-500 border border-gray-200"
            }`}
          >
            Tümü
          </button>
          {ROOM_OPTIONS.map((r) => (
            <button
              key={r}
              onClick={() => toggleRoom(r)}
              className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-all ${
                filters.rooms.includes(r)
                  ? "bg-slate-900 text-amber-500 shadow-sm border border-slate-900"
                  : "bg-gray-50 text-gray-600 hover:text-slate-900 hover:border-amber-500 border border-gray-200"
              }`}
            >
              {r}
            </button>
          ))}
        </div>
      </div>

      <hr className="border-gray-100" />

      {/* Priorities */}
      <div>
        <p className="text-sm font-bold text-slate-900 mb-4 tracking-tight flex items-center gap-1.5"><Target size={18} className="text-gold" /> Öncelik Ağırlıkları</p>
        <div className="flex flex-col gap-4">
          {VISIBLE_PRIORITY_KEYS.map((key) => (
            <div key={key} className="flex flex-col gap-1.5">
              <div className="flex justify-between text-xs text-gray-600 font-medium">
                <span>{PRIORITY_LABELS[key]}</span>
                <span className="text-slate-900 bg-amber-500/10 text-amber-700 px-1.5 py-0.5 rounded font-bold">{filters.priorities[key].toFixed(1)}</span>
              </div>
              <input
                type="range"
                min={0}
                max={1}
                step={0.1}
                value={filters.priorities[key]}
                onChange={(e) => setPriority(key, Number(e.target.value))}
                className="w-full premium-slider"
                style={{
                  background: `linear-gradient(to right, #d97706 ${filters.priorities[key] * 100}%, #e2e8f0 ${filters.priorities[key] * 100}%)`
                }}
              />
            </div>
          ))}
        </div>
      </div>
    </motion.aside>
  );
}
