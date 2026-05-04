"use client";

import { useState } from "react";
import { Listing } from "@/types/listing";

interface Props {
  listing: Listing;
  minPrice: number;
  maxPrice: number;
  minM2: number;
  maxM2: number;
  targetCity: string;
  targetDistrict: string;
}

const SOURCE_COLORS: Record<string, string> = {
  sahibinden: "bg-yellow-300 text-black",
  hepsiemlak: "bg-red-600 text-white",
  emlakjet: "bg-blue-500 text-white",
};

function districtName(raw: string): string {
  let d = (raw || "").trim();
  if (d.includes(" - ")) d = d.split(" - ")[0].trim();
  d = d.split(/\s{2,}/)[0].trim();
  return d;
}

export default function ListingCard({
  listing,
  minPrice,
  maxPrice,
  minM2,
  maxM2,
  targetCity,
  targetDistrict,
}: Props) {
  const [open, setOpen] = useState(false);
  const score = listing.scout_score;
  const isGreen = score >= 50;
  const borderColor = isGreen ? "border-green-500" : "border-red-400";
  const badgeBg = isGreen ? "bg-green-100 text-green-800 border-green-300" : "bg-red-100 text-red-800 border-red-300";
  const sourceKey = listing.source.toLowerCase().replace(/\s/g, "");
  const sourceClass = SOURCE_COLORS[sourceKey] ?? "bg-gray-200 text-gray-800";

  const label =
    score > 80 ? "TAVSİYE EDİLEN" : isGreen ? "DEĞERLENDİRİLEBİLİR" : "DÜŞÜK UYUM";

  const llmText =
    typeof listing.llm_score === "number" ? `${listing.llm_score}/10` : "Henüz yok";

  const fi = listing.fuzzy_inputs;

  // Suitability ratios (aynı app.py calculate_suitability_ratios mantığı)
  const pSuit =
    listing.price >= minPrice && listing.price <= maxPrice
      ? 100
      : listing.price < minPrice
      ? 100
      : Math.max(0, 100 - ((listing.price - maxPrice) / maxPrice) * 200);

  const lScore =
    listing.city === targetCity
      ? targetDistrict &&
        targetDistrict.toLowerCase() === districtName(listing.district).toLowerCase()
        ? 10
        : 7
      : 2;

  const adM2 = listing.area_m2;
  const sSuit =
    adM2 >= minM2 && adM2 <= maxM2
      ? 100
      : Math.max(
          0,
          100 - (Math.min(Math.abs(adM2 - minM2), Math.abs(adM2 - maxM2)) / Math.max(1, minM2)) * 100
        );

  return (
    <div className={`bg-white rounded-xl shadow p-5 border-l-8 ${borderColor} mb-5`}>
      <div className="flex gap-4">
        {/* Sol taraf: bilgiler */}
        <div className="flex-1">
          <span className={`text-xs font-bold uppercase px-2 py-1 rounded ${sourceClass}`}>
            {listing.source}
          </span>
          <h2 className="text-xl font-bold text-gray-900 mt-2">{listing.title}</h2>
          <p className="text-xl font-semibold text-gray-700 mt-1">
            {listing.price.toLocaleString("tr")} TL
            {listing.listing_type === "Kiralık" && (
              <span className="text-sm text-gray-500"> / ay</span>
            )}{" "}
            | {listing.area_m2} m²
          </p>
          <p className="text-sm text-gray-500 mt-1">
            📍 {districtName(listing.district)}, {listing.city} | 🛏️{" "}
            {listing.room_count || "Bilinmiyor"} | 📅 {listing.days_since_posted} gün önce
          </p>
          <div className="flex flex-wrap gap-2 mt-3 text-xs">
            <span className="bg-gray-100 px-2 py-1 rounded-full">💰 Fiyat: %{pSuit.toFixed(0)}</span>
            <span className="bg-gray-100 px-2 py-1 rounded-full">📍 Konum: {lScore}/10</span>
            <span className="bg-gray-100 px-2 py-1 rounded-full">📐 Boyut: %{sSuit.toFixed(0)}</span>
            <span className="bg-gray-100 px-2 py-1 rounded-full">🧠 LLM: {llmText}</span>
          </div>
          <p className="text-sm text-gray-600 mt-3 leading-relaxed">
            {listing.description.slice(0, 220)}...
          </p>
        </div>

        {/* Sağ taraf: skor */}
        <div className="flex flex-col items-center justify-start border-l border-gray-100 pl-5 min-w-[120px]">
          <div className={`text-3xl font-extrabold px-4 py-2 rounded-lg border-2 ${badgeBg}`}>
            %{score.toFixed(1)}
          </div>
          <p className={`mt-2 text-xs font-bold ${isGreen ? "text-green-700" : "text-red-600"}`}>
            {label}
          </p>
          <a
            href={listing.url}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-4 w-full text-center py-2 bg-gray-900 text-white text-sm rounded-lg font-semibold hover:bg-gray-700"
          >
            İlana Git ↗
          </a>
        </div>
      </div>

      {/* Detay accordion */}
      <button
        onClick={() => setOpen(!open)}
        className="mt-4 text-xs text-gray-500 underline"
      >
        {open ? "▲ Gizle" : "▼ Scout Mantığı: Bu Puan Nasıl Hesaplandı?"}
      </button>

      {open && (
        <div className="mt-4 text-sm text-gray-700 space-y-3 border-t pt-4">
          <p className="text-xs text-gray-500">
            Bu skor basit bir ortalama değil; Mamdani bulanık mantık kuralları tüm girdileri birlikte değerlendirir.
          </p>

          <div>
            <strong>Fiyat yüzdesi: %{fi.price_suitability.toFixed(0)}</strong>
            <p>İlan fiyatı {listing.price.toLocaleString("tr")} TL. Seçilen aralık {minPrice.toLocaleString("tr")} - {maxPrice.toLocaleString("tr")} TL.</p>
            {listing.price >= minPrice && listing.price <= maxPrice
              ? <p>Fiyat aralık içinde — daha ucuz olanlar hafif avantajlı.</p>
              : listing.price < minPrice
              ? <p>Alt sınırın altında, uygun kabul edildi (%100).</p>
              : <p>Üst sınırın üstünde, uzaklaştıkça yüzde düşüyor.</p>}
          </div>

          <div>
            <strong>Konum skoru: {fi.location_score}/10</strong>
            <p>İlanın şehri: {listing.city}. Seçilen şehir: {targetCity}.</p>
            {targetDistrict && <p>Seçilen ilçe: {targetDistrict}.</p>}
          </div>

          <div>
            <strong>Boyut yüzdesi: %{fi.size_suitability.toFixed(0)}</strong>
            <p>{listing.area_m2} m² — seçilen aralık: {minM2} - {maxM2} m².</p>
          </div>

          <div>
            <strong>Kalite: {fi.listing_quality}/10 | Metin: {fi.llm_alignment}/10</strong>
            {typeof listing.llm_score === "number" && (
              <p>Gerçek LLM çıktısı: {listing.llm_score}/10.</p>
            )}
          </div>

          <div className="bg-gray-50 rounded-lg p-3">
            <p className="font-semibold">Nihai Skor</p>
            <p className="text-xs text-gray-500 mb-2">
              Mamdani bulanık mantık kuralları tüm girdileri birlikte değerlendirip tek bir son puan üretir.
            </p>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className={`h-2 rounded-full ${isGreen ? "bg-green-500" : "bg-red-400"}`}
                style={{ width: `${score}%` }}
              />
            </div>
            <p className="text-right text-xs mt-1 font-bold">%{score.toFixed(1)}</p>
          </div>
        </div>
      )}
    </div>
  );
}
