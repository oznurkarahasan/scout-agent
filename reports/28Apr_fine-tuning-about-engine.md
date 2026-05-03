# Aşama 1 — Kritik Hata Düzeltmeleri (28 Nisan 2026)

## 1. Bare Exception → Hata Ayıklanabilir Hale Getirildi

**Dosya:** `src/fuzzy/engine.py:121`

**Eski durum:** Tüm hatalar sessizce yakalanıyor, her zaman `0` döndürülüyordu. Input aralık dışına çıksa, NaN üretse veya defuzzification başarısız olsa anlaşılmıyordu.

**Yeni durum:** `Exception as e` ile yakalanıyor, hangi input'un hangi hatayı ürettiği Streamlit uyarısı olarak gösteriliyor.

---

## 2. Çelişen Kurallar Çözüldü

**Dosya:** `src/fuzzy/engine.py:96`

**Eski durum:** `price[pahali] & location[yakin]` antecedent'i için iki ayrı kural vardı — biri `score[dusuk]`, diğeri `score[orta]` sonucuna yönlendiriyordu. Her ikisi de aynı anda ateşlendiği için defuzzifier belirsiz bir ortalama alıyordu. Kullanıcı ağırlıklandırmasının bu senaryoda etkisi yoktu.

**Yeni durum:** Tek bir kural kalıyor. Kural içeriği kullanıcının öncelik slider'larına göre dinamik belirleniyor:
- Fiyat önceliği ≥ Konum önceliği → `score[dusuk]` (pahalı affedilmez)
- Konum önceliği > Fiyat önceliği → `score[orta]` (yakın konum pahalıyı kısmen telafi eder)

---

## 3. Hard Filter / Soft Filter Çakışması Giderildi

**Dosya:** `app.py:174`

**Eski durum:** Oda sayısı hem hard filter (ilan eleme) hem de fuzzy input (`m_score`) olarak hesaplanıyordu. Hard filter'dan geçen bir ilan için `m_score` hiçbir zaman `0` alamıyordu — yani fuzzy sistemi için anlamsız bir input üretiliyordu. Bulanık mantığın soft boundary amacı bozuluyordu.

**Yeni durum:** Hard filter kaldı, fuzzy input yeniden tanımlandı:

| Durum | m_score | Açıklama |
|---|---|---|
| "Hepsi" seçili | 5 | Nötr — fuzzy diğer kriterlere göre karar verir |
| Tek tercih, tam eşleşme | 10 | Kesin uyum |
| Birden fazla tercih içinde eşleşme | 7 | Kısmi esneklik |

`m_score = 0` artık üretilmiyor — uyumsuz ilanlar hard filter'da zaten eleniyor.

---

# Aşama 2 — Kural Tabanının Tamamlanması (28 Nisan 2026)

**Dosya:** `src/fuzzy/engine.py:51`

## Değişiklik Özeti

Kural sayısı **9 → 21**'e çıkarıldı. Eski sistemde "makul fiyat", "orta konum", "iyi kalite" ve "kismi oda uyumu" gibi orta linguistic değerler için hiç kural yoktu. Bu inputlar sisteme girdiğinde hiçbir kural ateşlenmiyordu — kullanıcı ağırlığı ne olursa olsun o kriter skoru etkileyemiyordu.

## Yeni Kural Yapısı

### Tekli Kurallar (15 adet)

Her input için tam 3 kural: `düşük → dusuk`, `orta → orta`, `yüksek → yuksek`. Artık her input her zaman en az bir kural ateşler.

```
price:    ucuz→yuksek  makul→orta  pahali→dusuk
location: yakin→yuksek orta→orta   uzak→dusuk
size:     ideal→yuksek kucuk→orta  buyuk→orta
quality:  mukemmel→yuksek iyi→orta  zayif→dusuk
room:     uyumlu→yuksek  kismi→orta uyumsuz→dusuk
```

`size[kucuk]` ve `size[buyuk]` → `orta` olarak güncellendi (eskiden `dusuk`). Biraz dışı bir daire kötü değil, sadece ideal değil.

### Kombinasyon Kuralları: "efsane" (3 adet)

Birden fazla kriter aynı anda iyiyse `efsane` tetiklenir. Tekli kurallarla bu sonuç hiç üretilemiyordu.

```
price[ucuz]    & location[yakin] → efsane
price[ucuz]    & size[ideal]     → efsane
location[yakin] & size[ideal]    → efsane
```

### Kombinasyon Kuralları: "cop" (2 adet)

Birden fazla kriter aynı anda kötüyse `cop` tetiklenir.

```
price[pahali] & location[uzak]  → cop
price[pahali] & quality[zayif]  → cop
```

### Etkileşim Kuralı (1 adet, Aşama 1'den taşındı)

```
price[pahali] & location[yakin] → dusuk (fiyat öncelikliyse) / orta (konum öncelikliyse)
```

## Ağırlıklandırma Güncellemeleri

| Tür | Eski | Yeni | Fark |
|---|---|---|---|
| Boost fonksiyonu | `w ** 1.5` | `w ** 2` | Daha keskin ayrışım: w=0.1→0.01, w=0.9→0.81 |
| Tekli kurallar | kriter ağırlığı | kriter ağırlığı | Değişmedi |
| "Efsane" kombinasyonlar | — | `√(w₁ × w₂)` | İkisi de yüksekse güçlü, biri düşükse zayıf |
| "Çöp" kombinasyonlar | — | `max(w₁, w₂)` | En katı kriter baskın olsun |

Geometrik ortalama kullanımı: Kullanıcı fiyat önceliğini 0.9, konum önceliğini 0.1 verdiyse `ucuz & yakin` kuralı `√(0.81 × 0.01) = 0.09` ağırlıkla ateşlenir — konum önemsizse efsane kombinasyon da önemsizleşir.

---

# Aşama 2 Ek — Kural Ağırlıklandırması Keşfi ve Input Scaling (28 Nisan 2026)

**Dosya:** `src/fuzzy/engine.py`, `tests/test_fuzzy_engine.py`

## Tespit: `rule.weight` scikit-fuzzy'de Çalışmıyor

Test yazıldıktan sonra Senaryo 2 başarısız oldu: farklı önceliklerle çalışan engine instance'ları **birebir aynı skoru** üretiyordu.

```
Ayrı engine — fiyat_önce: 59.5,  konum_önce: 59.5
Fark: 0.0 — weights çalışıyor mu: False
```

Yani projedeki şu satırlar başından beri etkisizdi:

```python
rules[-1].weight = w_p   # ← skfuzzy bu değeri inference'ta kullanmıyor
```

Kullanıcı slider'ı ne yaparsa yapsın, skor hiç değişmiyordu. Bu scikit-fuzzy'nin bilinen bir sınırlığı — `rule.weight` API'de var ama Mamdani inference'ı etkilemiyor.

## Çözüm: Input Scaling

Kural ağırlıkları yerine **inputlar önceliğe göre ölçeklendi**. Mantık:

> "Bu kriteri önemsemiyorsan, onu ortalama kabul ediyoruz."

```
scaled = neutral + (val - neutral) * priority
```

| priority | Etki |
|---|---|
| 1.0 | Input değişmez — kriter tam güçle çalışır |
| 0.5 | Input nötr merkeze %50 yaklaşır |
| 0.0 | Input = neutral — kriter tamamen etkisiz |

Kurallar sabit kalır (21 adet, bir kez derlenir). Öncelik değişince kural yapısı değil, engine'e verilen değerler değişir.

## Mimari Değişiklikler

| Bileşen | Eski | Yeni |
|---|---|---|
| Kural ağırlıkları | `rule.weight = w` (çalışmıyor) | Kaldırıldı |
| ControlSystem | Öncelik değişince yeniden derleniyor | Bir kez derlenir, cache'lenir |
| ControlSystemSimulation | Yeniden kullanılıyor (state kirlenmesi riski) | Her `compute()` çağrısında taze oluşturuluyor |
| Öncelik uygulaması | Kural katmanında (etkisiz) | Input katmanında (çalışıyor) |

## Test Sonuçları

```
Pahalı+yakın ilan:  fiyat_önce=%52.8   konum_önce=%68.2
Ucuz+uzak ilan:     fiyat_önce=%70.1   konum_önce=%52.9
Mükemmel ilan:      %81.1
Kötü ilan:          %35.7

8/8 test geçti
```

Aynı ilan, farklı öncelikte farklı skor alıyor — kullanıcı ağırlıklandırması artık gerçekten çalışıyor.
