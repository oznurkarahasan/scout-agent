from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GroqConfig:
    api_key: str
    model: str = "llama-3.1-8b-instant"
    base_url: str = "https://api.groq.com/openai/v1"
    timeout_s: int = 30
    max_retries: int = 2


class GroqClient:
    """
    Groq'un OpenAI uyumlu REST API'sini kullanan minimal Groq istemcisi.

    Bu deponun yalnızca standart kütüphanenin mevcut olduğu ortamlarda çalışabilmesi için
    `groq` python paketine sert bir bağımlılıktan kaçınıyoruz.
    """

    def __init__(self, config: GroqConfig):
        self._cfg = config

    @classmethod
    def from_env(cls) -> GroqClient | None:
        # Varsa .env dosyasını yükle (repo python-dotenv kullanır)
        try:
            from dotenv import load_dotenv  # type: ignore

            load_dotenv()
        except Exception:
            # Eğer python-dotenv mevcut değilse, mevcut ortam değişkenlerine geri dön.
            pass

        api_key = (os.getenv("GROQ_API_KEY") or "").strip()
        if not api_key:
            return None
        model = (os.getenv("GROQ_MODEL") or "llama-3.1-8b-instant").strip()
        base_url = (os.getenv("GROQ_BASE_URL") or "https://api.groq.com/openai/v1").strip()
        timeout_s = int(os.getenv("GROQ_TIMEOUT_S") or "30")
        max_retries = int(os.getenv("GROQ_MAX_RETRIES") or "2")
        return cls(GroqConfig(api_key=api_key, model=model, base_url=base_url, timeout_s=timeout_s, max_retries=max_retries))

    def chat_completions_json(self, *, system: str, user: str) -> dict[str, Any]:
        url = f"{self._cfg.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self._cfg.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {self._cfg.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                # Bazı WAF/CDN kurulumları, User-Agent (UA) olmayan istekleri engeller.
                "User-Agent": "scout-agent/0.1 (+https://localhost)",
            },
            method="POST",
        )

        last_err: Exception | None = None
        for attempt in range(self._cfg.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self._cfg.timeout_s) as resp:
                    data = resp.read().decode("utf-8")
                raw = json.loads(data)
                content = raw["choices"][0]["message"]["content"]
                return json.loads(content)
            except urllib.error.HTTPError as e:
                # Sunucu hata detaylarını dahil etmeye çalış (genellikle JSON).
                try:
                    body = e.read().decode("utf-8", errors="replace")
                except Exception:
                    body = "<no body>"
                last_err = RuntimeError(f"HTTP {e.code} {e.reason}: {body}")
                if attempt >= self._cfg.max_retries:
                    break
                time.sleep(0.5 * (attempt + 1))
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
                last_err = e
                if attempt >= self._cfg.max_retries:
                    break
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"Groq request failed after retries: {last_err}") from last_err

