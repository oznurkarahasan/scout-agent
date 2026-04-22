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
        "You are a real-estate listing scorer for Turkish listings. "
        "Score how well a listing matches the user's preferences using partial credit. "
        "Requirements are PREFERENCES, not hard filters — partial matches should get middle scores. "
        "Return JSON only with a single key 'llm_score' as a float between 0.0 and 1.0.\n\n"
        "Scoring guide:\n"
        "  0.8-1.0 : All or nearly all preferences clearly present\n"
        "  0.5-0.8 : Most preferences met, 1-2 missing\n"
        "  0.3-0.5 : Some preferences met (e.g. correct room count but missing amenities)\n"
        "  0.1-0.3 : Few preferences met\n"
        "  0.0-0.1 : Nothing matches\n\n"
        "Important: never give 0.0 unless the listing is completely irrelevant. "
        "A listing with the correct room count but missing other features should score at least 0.3."
    )
    user = json.dumps(
        {
            "user_preferences": prefs.free_text,
            "listing": description,
            "return": {"llm_score": "float in [0.0, 1.0]"},
        },
        ensure_ascii=False,
    )
    try:
        out = client.chat_completions_json(system=system, user=user)
        return _clamp01(float(out.get("llm_score", 0.5)))
    except Exception:
        if require_remote:
            raise
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
        # Başlık + açıklama birleştir — model oda sayısını başlıktan okuyabilsin
        title = str(ad.get("title", ""))
        desc  = str(ad.get("description", ""))
        full_text = f"Başlık: {title}\nAçıklama: {desc}".strip()
        if not ad_id:
            continue
        if use_remote_llm is False:
            scores[ad_id] = heuristic_llm_score(full_text, prefs)
        else:
            scores[ad_id] = groq_llm_score(full_text, prefs, require_remote=(use_remote_llm is True))
    return scores

