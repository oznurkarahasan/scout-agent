"use client";

import { useEffect, useState } from "react";
import { fetchCities, fetchDistricts } from "@/lib/api";
import { FilterState, Priorities } from "@/types/listing";

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
  llm: "Metin Analizi (LLM)",
};

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
    <aside className="w-72 min-w-[18rem] bg-white shadow-md rounded-xl p-5 flex flex-col gap-4 self-start sticky top-4">
      <h1 className="text-lg font-bold text-gray-800">🏢 Arama Filtreleri</h1>

      <button
        onClick={onSearch}
        disabled={loading}
        className="w-full py-2 bg-gray-900 text-white rounded-lg font-semibold hover:bg-gray-700 disabled:opacity-50"
      >
        {loading ? "Aranıyor..." : "🔍 Ara"}
      </button>

      {/* Listing type */}
      <div>
        <p className="text-sm font-semibold text-gray-600 mb-1">🏠 İlan Tipi</p>
        <div className="flex gap-2">
          {(["Hepsi", "Kiralık", "Satılık"] as const).map((t) => (
            <button
              key={t}
              onClick={() => set({ listing_type: t })}
              className={`px-3 py-1 rounded-full text-sm border ${
                filters.listing_type === t
                  ? "bg-gray-900 text-white border-gray-900"
                  : "border-gray-300 text-gray-700"
              }`}
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      {/* City */}
      <div>
        <p className="text-sm font-semibold text-gray-600 mb-1">📍 Şehir</p>
        <select
          value={filters.city}
          onChange={(e) => set({ city: e.target.value, district: "" })}
          className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
        >
          {cities.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      </div>

      {/* District */}
      <div>
        <p className="text-sm font-semibold text-gray-600 mb-1">🔍 İlçe</p>
        <select
          value={filters.district}
          onChange={(e) => set({ district: e.target.value })}
          className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
        >
          <option value="">Hepsi</option>
          {districts.map((d) => (
            <option key={d} value={d}>
              {d}
            </option>
          ))}
        </select>
      </div>

      {/* Price */}
      <div>
        <p className="text-sm font-semibold text-gray-600 mb-1">
          💰 Fiyat Aralığı — {filters.min_price.toLocaleString("tr")} -{" "}
          {filters.max_price.toLocaleString("tr")} TL
        </p>
        <div className="flex gap-2">
          <input
            type="range"
            min={priceRange.min}
            max={priceRange.max}
            step={priceRange.step}
            value={filters.min_price}
            onChange={(e) => set({ min_price: Number(e.target.value) })}
            className="w-full"
          />
          <input
            type="range"
            min={priceRange.min}
            max={priceRange.max}
            step={priceRange.step}
            value={filters.max_price}
            onChange={(e) => set({ max_price: Number(e.target.value) })}
            className="w-full"
          />
        </div>
      </div>

      {/* m2 */}
      <div>
        <p className="text-sm font-semibold text-gray-600 mb-1">
          📐 Büyüklük — {filters.min_m2} - {filters.max_m2} m²
        </p>
        <div className="flex gap-2">
          <input
            type="range"
            min={0}
            max={1000}
            step={5}
            value={filters.min_m2}
            onChange={(e) => set({ min_m2: Number(e.target.value) })}
            className="w-full"
          />
          <input
            type="range"
            min={0}
            max={1000}
            step={5}
            value={filters.max_m2}
            onChange={(e) => set({ max_m2: Number(e.target.value) })}
            className="w-full"
          />
        </div>
      </div>

      {/* Rooms */}
      <div>
        <p className="text-sm font-semibold text-gray-600 mb-1">🛏️ Oda Sayısı</p>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => set({ rooms: ["Hepsi"] })}
            className={`px-3 py-1 rounded-full text-sm border ${
              filters.rooms.includes("Hepsi")
                ? "bg-gray-900 text-white border-gray-900"
                : "border-gray-300 text-gray-700"
            }`}
          >
            Hepsi
          </button>
          {ROOM_OPTIONS.map((r) => (
            <button
              key={r}
              onClick={() => toggleRoom(r)}
              className={`px-3 py-1 rounded-full text-sm border ${
                filters.rooms.includes(r)
                  ? "bg-gray-900 text-white border-gray-900"
                  : "border-gray-300 text-gray-700"
              }`}
            >
              {r}
            </button>
          ))}
        </div>
      </div>

      <hr />

      {/* Priorities */}
      <div>
        <p className="text-sm font-bold text-gray-700 mb-2">🎯 Önceliklerim</p>
        {(Object.keys(PRIORITY_LABELS) as (keyof Priorities)[]).map((key) => (
          <div key={key} className="mb-2">
            <div className="flex justify-between text-xs text-gray-500 mb-0.5">
              <span>{PRIORITY_LABELS[key]}</span>
              <span>{filters.priorities[key].toFixed(1)}</span>
            </div>
            <input
              type="range"
              min={0}
              max={1}
              step={0.1}
              value={filters.priorities[key]}
              onChange={(e) => setPriority(key, Number(e.target.value))}
              className="w-full"
            />
          </div>
        ))}
      </div>
    </aside>
  );
}
