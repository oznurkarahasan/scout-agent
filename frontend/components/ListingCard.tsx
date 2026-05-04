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
          <h2 className="text-xl font-bold text-slate-900 mt-2">{listing.title}</h2>
          <p className="text-xl font-semibold text-gray-700 mt-1">
            {listing.price.toLocaleString("tr")} TL
            {listing.listing_type === "Kiralık" && (
              <span className="text-sm text-gray-500"> / ay</span>
            )}{" "}
            | {listing.area_m2} m²
          </p>
          <div className="flex items-center gap-2.5 text-sm text-gray-500 mt-1.5">
            <span>{districtName(listing.district)}, {listing.city}</span>
            <span className="text-gray-300">•</span>
            <span>{listing.room_count || "Bilinmiyor"} Oda</span>
            <span className="text-gray-300">•</span>
            <span>{listing.days_since_posted} gün önce</span>
          </div>
          <div className="flex flex-wrap gap-2 mt-4 text-xs font-medium text-slate-900">
            <span className="bg-amber-500/10 border border-amber-500/20 px-2.5 py-1 rounded-md">Fiyat Uyum: %{pSuit.toFixed(0)}</span>
            <span className="bg-amber-500/10 border border-amber-500/20 px-2.5 py-1 rounded-md">Konum: {lScore}/10</span>
            <span className="bg-amber-500/10 border border-amber-500/20 px-2.5 py-1 rounded-md">Boyut: %{sSuit.toFixed(0)}</span>
            <span className="bg-amber-500/10 border border-amber-500/20 px-2.5 py-1 rounded-md">LLM Skoru: {llmText}</span>
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
            className="mt-4 w-full text-center py-2 bg-slate-900 text-white text-sm rounded-lg font-semibold hover:bg-slate-800 transition-colors shadow-sm"
          >
            İlana Git ↗
          </a>
        </div>
      </div>

      {/* Detay accordion */}
      <button
        onClick={() => setOpen(!open)}
        className="mt-5 w-full flex items-center justify-between px-4 py-2.5 bg-gray-50 hover:bg-gray-100 rounded-lg transition-colors text-sm font-semibold text-navy-deep border border-gray-200"
      >
        <span>Scout Mantığı: Puan Hesaplama Özeti</span>
        <span className="text-gray-400 font-normal text-xs">{open ? "Gizle ▲" : "Göster ▼"}</span>
      </button>

      {open && (
        <div className="mt-3 bg-gray-50 border border-gray-100 rounded-xl p-5 space-y-4 shadow-inner">
          <p className="text-xs text-gray-500 font-medium text-center mb-2">
            Mamdani bulanık mantık motoru tüm girdileri birlikte değerlendirir.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="bg-white p-3.5 rounded-lg border border-gray-100 shadow-sm">
              <div className="flex justify-between items-center mb-1.5">
                <span className="text-xs font-bold text-gray-700">Fiyat Uyumu</span>
                <span className="text-xs font-extrabold text-navy-deep">%{fi.price_suitability.toFixed(0)}</span>
              </div>
              <div className="w-full bg-gray-100 rounded-full h-1.5 mb-2.5">
                <div className="bg-gold h-1.5 rounded-full" style={{ width: `${fi.price_suitability}%` }} />
              </div>
              <p className="text-[11px] text-gray-500 leading-tight">
                İlan: <strong>{listing.price.toLocaleString("tr")} ₺</strong><br />
                Hedef: {minPrice.toLocaleString("tr")} - {maxPrice.toLocaleString("tr")} ₺
              </p>
            </div>

            <div className="bg-white p-3.5 rounded-lg border border-gray-100 shadow-sm">
              <div className="flex justify-between items-center mb-1.5">
                <span className="text-xs font-bold text-gray-700">Boyut Uyumu</span>
                <span className="text-xs font-extrabold text-navy-deep">%{fi.size_suitability.toFixed(0)}</span>
              </div>
              <div className="w-full bg-gray-100 rounded-full h-1.5 mb-2.5">
                <div className="bg-gold h-1.5 rounded-full" style={{ width: `${fi.size_suitability}%` }} />
              </div>
              <p className="text-[11px] text-gray-500 leading-tight">
                İlan: <strong>{listing.area_m2} m²</strong><br />
                Hedef: {minM2} - {maxM2} m²
              </p>
            </div>

            <div className="bg-white p-3.5 rounded-lg border border-gray-100 shadow-sm">
              <div className="flex justify-between items-center mb-1.5">
                <span className="text-xs font-bold text-gray-700">Konum Skoru</span>
                <span className="text-xs font-extrabold text-navy-deep">{fi.location_score}/10</span>
              </div>
              <div className="w-full bg-gray-100 rounded-full h-1.5 mb-2.5">
                <div className="bg-navy-light h-1.5 rounded-full" style={{ width: `${fi.location_score * 10}%` }} />
              </div>
              <p className="text-[11px] text-gray-500 leading-tight">
                {listing.city} {targetDistrict && targetDistrict !== "Hepsi" ? ` / ${targetDistrict}` : ''}
              </p>
            </div>

            <div className="bg-white p-3.5 rounded-lg border border-gray-100 shadow-sm">
              <div className="flex justify-between items-center mb-1.5">
                <span className="text-xs font-bold text-gray-700">Kalite & Metin</span>
                <span className="text-xs font-extrabold text-navy-deep">{fi.listing_quality}/10 & {fi.llm_alignment}/10</span>
              </div>
              <p className="text-[11px] text-gray-500 leading-tight mt-2">
                Görsel ve açıklama kalitesine ek olarak yapay zeka (LLM) ile metin uygunluğu ölçülmüştür.
              </p>
            </div>
          </div>

          <div className="bg-navy-deep rounded-xl p-4 mt-2 text-white shadow-md flex items-center justify-between border border-navy-light">
            <div>
              <p className="text-sm font-semibold text-gray-100 tracking-wide">Nihai Scout Skoru</p>
              <p className="text-[11px] text-gray-400 mt-0.5">Bulanık mantık birleşimi sonucu</p>
            </div>
            <div className="text-right">
              <p className="text-3xl font-black text-gold">%{score.toFixed(1)}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
