"use client";

import { useEffect, useState } from "react";
import FilterSidebar from "@/components/FilterSidebar";
import ListingCard from "@/components/ListingCard";
import { fetchCities, fetchListings } from "@/lib/api";
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
    quality: 0.5,
    llm: 0.4,
  },
};

export default function Home() {
  const [filters, setFilters] = useState<FilterState>(DEFAULT_FILTERS);
  const [listings, setListings] = useState<Listing[]>([]);
  const [count, setCount] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);

  // İlk açılışta şehirleri çek ve varsayılan şehri ata
  useEffect(() => {
    fetchCities().then((cities) => {
      if (cities.length > 0) {
        setFilters((f) => ({ ...f, city: f.city || cities[0] }));
      }
    });
  }, []);

  const handleSearch = async () => {
    if (!filters.city) return;
    setLoading(true);
    setSearched(true);
    try {
      const result = await fetchListings(filters);
      setListings(result.listings);
      setCount(result.count);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex gap-6 p-6 max-w-screen-xl mx-auto">
      <FilterSidebar
        filters={filters}
        onChange={setFilters}
        onSearch={handleSearch}
        loading={loading}
      />

      <main className="flex-1">
        <div className="mb-4">
          <h1 className="text-2xl font-bold text-gray-900">🏹 Scout Agent: Zeki Emlak Bulucu</h1>
          {searched && count !== null && (
            <p className="text-gray-600 mt-1">
              <strong>{filters.city}</strong> bölgesinde{" "}
              <strong>{count}</strong> uygun ilan bulundu.
              <span className="text-sm text-gray-400 ml-2">
                {filters.listing_type} |{" "}
                {filters.min_price.toLocaleString("tr")} -{" "}
                {filters.max_price.toLocaleString("tr")} TL
              </span>
            </p>
          )}
        </div>

        {loading && (
          <div className="text-center py-20 text-gray-400 text-lg">Aranıyor...</div>
        )}

        {!loading && searched && listings.length === 0 && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-6 text-yellow-800">
            Aradığınız kriterlerde {filters.city} şehrinde ilan bulunamadı. Filtreleri esnetmeyi deneyin.
          </div>
        )}

        {!loading && !searched && (
          <div className="text-center py-20 text-gray-400">
            Filtreleri ayarlayıp <strong>Ara</strong> butonuna tıklayın.
          </div>
        )}

        {!loading &&
          listings.map((listing) => (
            <ListingCard
              key={listing.id}
              listing={listing}
              minPrice={filters.min_price}
              maxPrice={filters.max_price}
              minM2={filters.min_m2}
              maxM2={filters.max_m2}
              targetCity={filters.city}
              targetDistrict={filters.district}
            />
          ))}
      </main>
    </div>
  );
}
