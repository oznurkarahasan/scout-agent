import { FilterState, Listing } from "@/types/listing";

const BASE = "/api";

export async function fetchCities(): Promise<string[]> {
  const res = await fetch(`${BASE}/cities`, { cache: "no-store" });
  const data = await res.json();
  return data.cities;
}

export async function fetchDistricts(city: string): Promise<string[]> {
  const res = await fetch(`${BASE}/districts?city=${encodeURIComponent(city)}`, { cache: "no-store" });
  const data = await res.json();
  return data.districts;
}

export async function fetchListings(
  filters: FilterState
): Promise<{ count: number; listings: Listing[] }> {
  const rooms =
    filters.rooms.includes("Hepsi") || filters.rooms.length === 0
      ? "Hepsi"
      : filters.rooms.join(",");

  const params = new URLSearchParams({
    city: filters.city,
    district: filters.district,
    listing_type: filters.listing_type,
    min_price: String(filters.min_price),
    max_price: String(filters.max_price),
    min_m2: String(filters.min_m2),
    max_m2: String(filters.max_m2),
    rooms,
    priority_price: String(filters.priorities.price),
    priority_location: String(filters.priorities.location),
    priority_size: String(filters.priorities.size),
    priority_quality: String(filters.priorities.quality),
    priority_llm: String(filters.priorities.llm),
  });

  const res = await fetch(`${BASE}/listings?${params}`, { cache: "no-store" });
  return res.json();
}
