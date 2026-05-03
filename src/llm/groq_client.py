import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """Sen bir emlak ilanı analiz uzmanısın. 
Sana verilen ilan açıklamasını analiz edip SADECE JSON formatında yanıt ver.
Başka hiçbir şey yazma, sadece JSON.

Çıktı formatı:
{
  "balkon": true/false,
  "esyali": true/false,
  "otopark": true/false,
  "asansor": true/false,
  "yeni_bina": true/false,
  "guvenlik": true/false,
  "genel_izlenim": "pozitif/notr/negatif",
  "llm_score": 0-10
}

llm_score hesaplama:
- Her pozitif özellik +1 puan
- genel_izlenim pozitif ise +2, notr ise +1, negatif ise 0
- Maksimum 10"""

def analyze_listing(description: str) -> dict:
    """İlan açıklamasını analiz eder, özellik skoru döner."""
    try:
        response = client.chat.completions.create(
           model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"İlan açıklaması:\n{description}"}
            ],
            temperature=0.1,
            max_tokens=300
        )
        
        raw = response.choices[0].message.content.strip()
        result = json.loads(raw)
        return result
        
    except json.JSONDecodeError:
        return {"llm_score": 5, "hata": "JSON parse edilemedi"}
    except Exception as e:
        return {"llm_score": 5, "hata": str(e)}