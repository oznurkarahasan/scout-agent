"use client";

import { useState } from "react";
import { Listing } from "@/types/listing";
import { motion, AnimatePresence } from "framer-motion";
import { MapPin, BedDouble, Calendar, ChevronDown, ChevronUp, ExternalLink, ShieldCheck, Tag, Box, Info } from "lucide-react";
import FuzzyVizPanel from "@/components/FuzzyVizPanel";
import { Priorities } from "@/types/listing";

interface Props {
  listing: Listing;
  minPrice: number;
  maxPrice: number;
  minM2: number;
  maxM2: number;
  targetCity: string;
  targetDistrict: string;
  priorities: Priorities;
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
  priorities,
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

  const fi = listing.fuzzy_inputs;

  const pSuit = fi.price_suitability;
  const lScore = fi.location_score;
  const sSuit = fi.size_suitability;

  return (
    <motion.div 
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      whileHover={{ y: -4, transition: { duration: 0.2 } }}
      className={`bg-white rounded-2xl shadow-md hover:shadow-xl transition-shadow duration-300 p-6 border border-gray-100 border-l-[6px] ${borderColor} mb-6 relative overflow-hidden group`}
    >
      <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-bl from-navy-deep/5 to-transparent rounded-bl-full pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
      <div className="flex gap-4 relative z-10">
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
          <div className="flex items-center gap-4 text-sm text-gray-500 mt-3 font-medium bg-gray-50/50 p-2 rounded-lg inline-flex">
            <div className="flex items-center gap-1.5"><MapPin size={16} className="text-gray-400" /> <span>{districtName(listing.district)}, {listing.city}</span></div>
            <div className="w-1 h-1 bg-gray-300 rounded-full" />
            <div className="flex items-center gap-1.5"><BedDouble size={16} className="text-gray-400" /> <span>{listing.room_count || "Bilinmiyor"} Oda</span></div>
            <div className="w-1 h-1 bg-gray-300 rounded-full" />
            <div className="flex items-center gap-1.5"><Calendar size={16} className="text-gray-400" /> <span>{listing.days_since_posted} gün önce</span></div>
          </div>
          <div className="flex flex-wrap gap-2.5 mt-4 text-xs font-semibold text-navy-deep">
            <span className="flex items-center gap-1.5 bg-gold/10 border border-gold/20 px-3 py-1.5 rounded-lg shadow-sm">
              <Tag size={14} className="text-gold" /> Fiyat Uyum: %{pSuit.toFixed(0)}
            </span>
            <span className="flex items-center gap-1.5 bg-gold/10 border border-gold/20 px-3 py-1.5 rounded-lg shadow-sm">
              <MapPin size={14} className="text-gold" /> Konum: {lScore}/100
            </span>
            <span className="flex items-center gap-1.5 bg-gold/10 border border-gold/20 px-3 py-1.5 rounded-lg shadow-sm">
              <Box size={14} className="text-gold" /> Boyut: %{sSuit.toFixed(0)}
            </span>
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
          <p className={`mt-2 text-[10px] font-black uppercase tracking-wider text-center flex items-center justify-center gap-1 w-full ${isGreen ? "text-green-600" : "text-red-500"}`}>
            {isGreen && <ShieldCheck size={14} />}
            {label}
          </p>
          <a
            href={listing.url}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-4 w-full flex items-center justify-center gap-2 py-2 bg-slate-900 text-white text-sm rounded-lg font-semibold hover:bg-slate-800 transition-colors shadow-sm"
          >
            İlana Git <ExternalLink size={16} />
          </a>
        </div>
      </div>

      {/* Detay accordion */}
      <button
        onClick={() => setOpen(!open)}
        className="mt-5 w-full flex items-center justify-between px-5 py-3 bg-gray-50 hover:bg-gray-100 rounded-xl transition-colors text-sm font-bold text-navy-deep border border-gray-200"
      >
        <span className="flex items-center gap-2"><Info size={16} className="text-gold" /> Scout Mantığı: Puan Hesaplama Özeti</span>
        <span className="text-gray-400 font-normal text-xs flex items-center gap-1">
          {open ? <>Gizle <ChevronUp size={14} /></> : <>Göster <ChevronDown size={14} /></>}
        </span>
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: "easeInOut" }}
            className="overflow-hidden"
          >
            <FuzzyVizPanel fuzzyInputs={fi} priorities={priorities} actualScore={score} />
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
