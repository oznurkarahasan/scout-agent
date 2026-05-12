"use client";

import { useEffect, useRef, useState } from "react";
import FilterSidebar from "@/components/FilterSidebar";
import ListingCard from "@/components/ListingCard";
import { fetchCities, fetchListings } from "@/lib/api";
import { motion, AnimatePresence } from "framer-motion";
import { Loader2 } from "lucide-react";
import { FilterState, Listing } from "@/types/listing";

const DEFAULT_FILTERS: FilterState = {
  city: "",
  district: "",
  listing_type: "Hepsi",
  min_price: 10000,
  max_price: 30000,
  min_m2: 75,
  max_m2: 200,
  rooms: ["Hepsi"],
  priorities: {
    price: 0.9,
    location: 0.7,
    size: 0.6,
    rooms: 0.5,
  },
};

export default function Home() {
  const [filters, setFilters] = useState<FilterState>(DEFAULT_FILTERS);
  const [listings, setListings] = useState<Listing[]>([]);
  const [count, setCount] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const requestSeqRef = useRef(0);

  const search = async (f: FilterState) => {
    if (!f.city) return;
    const requestSeq = ++requestSeqRef.current;
    setLoading(true);
    try {
      const result = await fetchListings(f);
      if (requestSeq !== requestSeqRef.current) return;
      setListings(result.listings);
      setCount(result.count);
    } finally {
      if (requestSeq === requestSeqRef.current) {
        setLoading(false);
      }
    }
  };

  // İlk açılış: şehirleri çek, varsayılanı ata → otomatik arama tetiklenir
  useEffect(() => {
    fetchCities().then((cities) => {
      if (cities.length > 0) {
        setFilters((f) => ({ ...f, city: f.city || cities[0] }));
      }
    });
  }, []);

  // Filtreler değişince neredeyse anında otomatik ara
  useEffect(() => {
    if (!filters.city) return;
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => search(filters), 80);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [filters]);

  return (
    <div className="flex gap-6 p-6 max-w-screen-xl mx-auto">
      <FilterSidebar
        filters={filters}
        onChange={setFilters}
        onSearch={() => search(filters)}
        loading={loading}
      />

      <main className="flex-1">
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6"
        >
          <h1 className="text-3xl font-black text-transparent bg-clip-text bg-gradient-to-r from-navy-deep to-gold tracking-tight">Scout Agent: Gayrimenkul Asistanı</h1>
          {count !== null && (
            <p className="text-gray-500 mt-2 font-medium bg-white px-4 py-2 rounded-lg border border-gray-100 shadow-sm inline-block">
              <strong className="text-navy-deep">{filters.city}</strong> bölgesinde{" "}
              <strong className="text-navy-deep">{count}</strong> uygun ilan bulundu.
              <span className="text-sm text-gray-400 ml-3 pl-3 border-l border-gray-200">
                {filters.listing_type} |{" "}
                {filters.min_price.toLocaleString("tr")} -{" "}
                {filters.max_price.toLocaleString("tr")} ₺
              </span>
            </p>
          )}
        </motion.div>

        <AnimatePresence mode="wait">
          {loading ? (
            <motion.div 
              key="loader"
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              className="flex flex-col items-center justify-center py-20 text-navy-deep/60"
            >
              <Loader2 className="w-10 h-10 animate-spin text-gold mb-4" />
              <p className="font-semibold text-lg animate-pulse">En iyi ilanlar analiz ediliyor...</p>
            </motion.div>
          ) : count === 0 ? (
            <motion.div 
              key="empty"
              initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }}
              className="bg-amber-50 border border-amber-200 rounded-2xl p-8 text-amber-800 shadow-inner flex flex-col items-center text-center"
            >
              <span className="text-4xl mb-3">🔍</span>
              <h3 className="text-xl font-bold mb-2">İlan Bulunamadı</h3>
              <p className="font-medium">Aradığınız kriterlerde {filters.city} şehrinde ilan bulunamadı.<br/>Filtreleri esnetmeyi deneyin.</p>
            </motion.div>
          ) : (
            <motion.div key="list" className="space-y-6">
              {listings.map((listing) => (
            <ListingCard
              key={listing.id}
              listing={listing}
              minPrice={filters.min_price}
              maxPrice={filters.max_price}
              minM2={filters.min_m2}
              maxM2={filters.max_m2}
              targetCity={filters.city}
              targetDistrict={filters.district}
              priorities={filters.priorities}
            />
          ))}
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}
