export interface FuzzyInputs {
  price_suitability: number;
  location_score: number;
  size_suitability: number;
  room_match: number;
}

export interface Listing {
  id: string;
  title: string;
  price: number;
  city: string;
  district: string;
  area_m2: number;
  room_count: string;
  description: string;
  image_count: number;
  days_since_posted: number;
  listing_type: string;
  source: string;
  url: string;
  publisher_type: string;
  llm_score?: number;
  scout_score: number;
  fuzzy_inputs: FuzzyInputs;
}

export interface Priorities {
  price: number;
  location: number;
  size: number;
  rooms: number;
}

export interface FilterState {
  city: string;
  district: string;
  listing_type: "Hepsi" | "Kiralık" | "Satılık";
  min_price: number;
  max_price: number;
  min_m2: number;
  max_m2: number;
  rooms: string[];
  priorities: Priorities;
}
