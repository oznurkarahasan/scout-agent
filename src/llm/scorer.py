from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

from .groq_client import GroqClient


@dataclass(frozen=True)
class UserPreferences:
    """
    Basit tercih modeli (ileride genişletilebilir).
    """

    free_text: str


def _clamp01(x: float) -> float:
    if math.isnan(x):
        return 0.0
    return max(0.0, min(1.0, float(x)))


def heuristic_llm_score(description: str, prefs: UserPreferences) -> float:
    """
    API anahtarı yokken çalışmayı bozmayacak deterministik bir skorlayıcı.
    Çok kaba bir yaklaşım: tercihlerdeki anahtar kelimeleri description'da arar.
    """
    desc = (description or "").casefold()
    tokens = [t.strip().casefold() for t in (prefs.free_text or "").split(",") if t.strip()]
    if not tokens:
        return 0.5

    hits = 0
    for t in tokens:
        if t and t in desc:
            hits += 1
    return _clamp01(0.2 + 0.8 * (hits / len(tokens)))


def groq_llm_score(description: str, prefs: UserPreferences, *, require_remote: bool = False) -> float:
    client = GroqClient.from_env()
    if client is None:
        if require_remote:
            raise RuntimeError("GROQ_API_KEY missing; remote LLM scoring is required.")
        return heuristic_llm_score(description, prefs)

    system = (
        "You are a strict scorer for Turkish real-estate listings. "
        "Return JSON only. Score must be a float between 0 and 1."
    )
    user = json.dumps(
        {
            "task": "Score how well the listing matches the user's preferences.",
            "user_preferences": prefs.free_text,
            "listing_description": description,
            "return": {"llm_score": "float in [0,1]"},
        },
        ensure_ascii=False,
    )
    try:
        out = client.chat_completions_json(system=system, user=user)
        return _clamp01(float(out.get("llm_score", 0.5)))
    except Exception:
        if require_remote:
            raise
        # Ağ/servis hatalarında sistemi bozmayalım; heuristic'e düş.
        return heuristic_llm_score(description, prefs)


def score_ads_with_llm(
    ads: list[dict],
    prefs: UserPreferences,
    *,
    use_remote_llm: bool | None = None,
) -> dict[str, float]:
    """
    Her ilan için {ad_id: llm_score} döndürür.

    use_remote_llm:
      - None: GROQ_API_KEY varsa Groq kullan, yoksa heuristic
      - True: zorla Groq (anahtar yoksa RuntimeError)
      - False: zorla heuristic
    """
    client_present = GroqClient.from_env() is not None
    if use_remote_llm is True and not client_present:
        raise RuntimeError("GROQ_API_KEY missing; cannot use remote LLM scoring.")

    scores: dict[str, float] = {}
    for ad in ads:
        ad_id = str(ad.get("id", ""))
        desc = str(ad.get("description", ""))
        if not ad_id:
            continue
        if use_remote_llm is False:
            scores[ad_id] = heuristic_llm_score(desc, prefs)
        else:
            scores[ad_id] = groq_llm_score(desc, prefs, require_remote=(use_remote_llm is True))
    return scores

