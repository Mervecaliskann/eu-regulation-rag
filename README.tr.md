# AB Mevzuatı RAG Asistanı

[English](README.md) · **Türkçe**

Üç AB yönetmeliği hakkındaki soruları cevaplayan ve kullandığı maddeyi kaynak olarak gösteren bir RAG (retrieval-augmented generation) sistemi:

- **AB Yapay Zeka Yasası (AI Act)** (Yönetmelik 2024/1689, 144 sayfa)
- **GDPR**, kişisel verilerin korunması (Yönetmelik 2016/679, 88 sayfa)
- **DORA**, finans sektörü için dijital operasyonel dayanıklılık (Yönetmelik 2022/2554, 79 sayfa)

Asıl amaç sadece sistemi kurmak değil, her adımı **ölçmekti**: doğru madde ne sıklıkla bulunuyor, reranker işe yarıyor mu, cevaplar kaynaklara sadık kalıyor mu.

## Örnek çıktı

```text
Q: What is the right to be forgotten?   (Unutulma hakkı nedir?)

The "right to be forgotten" (right of erasure) allows a data subject to obtain from a controller
the deletion of personal data concerning them without undue delay, when certain conditions are met
... [3]. The controller must then erase the data and, if it has made the data public, take
reasonable steps ... to inform other controllers ... [5]. The right does not apply where erasure
would conflict with freedom of expression, legal obligations, public-health or research interests
... [5].

Sources:
  [1] GDPR - Article 17 - Right to erasure ('right to be forgotten') (p. 43)
  [2] GDPR - Recital 66 (p. 13)
  [3] GDPR - Article 17 - Right to erasure ('right to be forgotten') (p. 43)
  [4] GDPR - Recital 65 (p. 12)
  [5] GDPR - Article 17 - Right to erasure ('right to be forgotten') (p. 43)
```

Kapsam dışı bir soru:

```text
Q: What is the VAT rate for restaurants in Germany?   (Almanya'da restoranlarda KDV oranı?)
I could not find this in the provided regulations.
```

35 test sorusunun tüm cevapları ve hakem kararları [`eval_results.csv`](eval_results.csv) dosyasında.

## Nasıl çalışıyor?

```text
PDF (AI Act, GDPR, DORA)
  -> ingest.py       metni temizle, gerekçe / madde / ek olarak böl          -> chunks.jsonl (1.264 parça)
  -> build_index.py  her parçayı 384 sayılık vektöre çevir (bge-small)       -> index.faiss
Soru
  -> rag.py          1. arama  : FAISS soruya en yakın 20 parçayı bulur (bi-encoder)
                     2. rerank : cross-encoder soru ve parçayı birlikte okuyup en iyi 5'i seçer
                     3. cevap  : LLM SADECE bu 5 parçadan, [n] kaynaklı cevap yazar
  -> eval.py         35 test sorusu: Hit@1, Recall@5, Recall@20, MRR, faithfulness, reddetme
```

## Her dosya ne yapıyor?

### `ingest.py`: PDF → parçalar
PDF'leri pypdf ile okur, sayfa başlıklarını, sayfa numaralarını ve dipnotları atar. Metni **gerekçeler** (maddelerden önceki "neden" paragrafları), **maddeler** ve **ekler** olarak böler, uzun maddeleri parçalara ayırır.

```python
HEADING = re.compile(r"^(Article (\d+)|ANNEX ([IVXLC]+))\s*$", re.MULTILINE)  # SADECE "Article 12" yazan satır
if int(h.group(2)) == next_art:      # sadece beklenen sıradaki numarayı kabul et (1, 2, 3 ...)
    heads.append(h); next_art += 1
```
Sonuç: 1.264 parça. AI Act 180 gerekçe / 113 madde / 13 ek, GDPR 173 / 99 / 0, DORA 104 / 64 / 0.

### `build_index.py`: parçalar → vektörler
Her parçayı anlamını temsil eden 384 sayılık bir vektöre çevirir ve hepsini bir FAISS indeksine kaydeder.

```python
vectors = model.encode(texts, batch_size=32, normalize_embeddings=True)  # (1264, 384)
index = faiss.IndexFlatIP(384)     # normalize vektörlerde iç çarpım = kosinüs benzerliği
index.add(vectors)
```

### `rag.py`: soru → cevap
```python
candidates = search(question)                   # FAISS'ten 20 parça
best5 = rerank(question, candidates)            # cross-encoder her (soru, parça) çiftine puan verir
answer = llm(SYSTEM_PROMPT, best5, question)    # "SADECE numaralı kaynakları kullan, [n] yaz,
                                                #  yoksa 'I could not find this' de"
```

### `eval.py`: ölçüm
Her test sorusu için FAISS 20 adayı **bir kez** getirir. Aynı 20 aday iki şekilde sıralanır (FAISS sırası ve reranker sırası); böylece aradaki tek fark reranker olur. Sonra LLM reranker'ın seçtiği 5 parçadan cevap verir, ikinci bir model (hakem) her iddianın kaynaklarda olup olmadığını kontrol eder.

### `api.py`: web servisi
`rag.py`'yi saran FastAPI servisi; bkz. [Web API](#web-api-fastapi).

## Teknolojiler ve ayarlar

| Bölüm | Seçim | Değer | Neden |
|---|---|---|---|
| PDF okuma | pypdf + regex | | Metin tabanlı PDF'ler; başlık ve dipnotlar kalıplarla atıldı |
| Parçalama | madde / gerekçe / ek bazında | en fazla 1.500 karakter, 200 overlap | Madde, yasanın doğal anlam birimi. 1.500 karakter ≈ 330 token, modelin 512 token sınırına sığar. Overlap bölünen cümlenin bağlamını korur. |
| Parça başlığı | `GDPR - Article 17 - Right to erasure` | | Aramanın maddeyi başlığından da bulabilmesi için |
| Embedding | `BAAI/bge-small-en-v1.5` | 384 boyut, normalize | Ücretsiz, GPU'suz bir dizüstü bilgisayarda CPU ile çalışır |
| Vektör indeksi | FAISS `IndexFlatIP` | birebir arama | 1.264 vektör (~1,9 MB): birebir arama milisaniye sürer |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 20 → 5 | Soru ve parçayı birlikte okur; az RAM için küçük (~90 MB) |
| LLM | `openai/gpt-oss-120b` (Groq) | temperature 0 | Tekrarlanabilir cevap; kaynak gösteren ve bulamazsa reddeden prompt |
| Hakem | `openai/gpt-oss-20b` (Groq) | | Farklı bir model kontrol eder; model kendi cevabını notlamaz |
| API | FastAPI + uvicorn, Docker | | Modeller açılışta bir kez yüklenir; istekler otomatik kontrol edilir, doküman otomatik oluşur |

## Temel kavramlar

| Kavram | Anlamı |
|---|---|
| Chunk (parça) | Belgenin aranabilir küçük bir bölümü |
| Embedding | Anlamı gösteren sabit uzunlukta sayı listesi (burada 384); benzer metinlerin vektörleri birbirine yakındır |
| Kosinüs benzerliği | İki vektörün aralarındaki açıya göre benzerliği (1 = aynı yön) |
| Bi-encoder | Soru ve parça ayrı ayrı vektöre çevrilip karşılaştırılır; hızlı |
| Cross-encoder (reranker) | Soru ve parça birlikte okunur, tek bir alaka puanı çıkar; yavaş ama hassas |
| Grounding | LLM'in sadece verilen kaynakları kullanması |
| Faithfulness | Her iddiası kaynaklarda bulunan cevapların oranı |
| Recall@k / Hit@1 | Doğru madde ilk k sonuçta mı / ilk sırada mı? |
| MRR | Doğru maddenin sırasının tersi, ortalaması (1. = 1, 2. = 0,5, 3. = 0,33) |

## Sonuçlar

Test seti: 35 soru ([`test_questions.json`](test_questions.json)). 32 sorunun doğru maddesi belli (13 AI Act, 12 GDPR, 7 DORA), 3 soru kapsam dışı.

| Arama metriği | Sadece FAISS | FAISS + reranker |
|---|---|---|
| Hit@1 (doğru madde 1. sırada) | %53,1 (17/32) | **%59,4** (19/32) |
| Recall@5 (doğru madde LLM'e giden 5 parçada) | %84,4 (27/32) | **%93,8** (30/32) |
| Recall@20 (doğru madde 20 adayda) | %100 | %100 |
| MRR | 0,665 | **0,722** |

| Cevap metriği (reranker, ilk 5) | Sonuç |
|---|---|
| Faithfulness (her iddia kaynaklarda var, LLM hakem) | %87,5 (28/32) |
| Kapsam dışı sorularda doğru reddetme | 3/3 |

Not: `rag.py`'deki kaynak işareti temizliği (`[1†L1-L3]` → `[1]`) bu ölçümden sonra eklendi; bu yüzden `eval_results.csv` içindeki bazı cevaplarda eski işaretler görünür. Bu değişiklik sadece metnin görünüşünü etkiler, arama sonuçlarını ve puanları değiştirmez.

## Bulgular

- **Reranker en çok FAISS'in kötü sıraladığı sorularda işe yarıyor:** veri taşınabilirliği 20. → 1., otomatik kredi kararı 9. → 2., ICT risk sorumluluğu 12. → 2.
- **Bazı soruları ise kötüleştirdi:** yüksek risk sınıflandırması 1. → 5., hukuki işleme dayanakları 3. → 7. Muhtemel sebep: bu reranker hukuk metniyle değil, web arama verisiyle eğitilmiş.
- **Kaynağa sadık olmak, doğru olmakla aynı şey değil.** *"Yasaklı bir AI uygulamasının en yüksek cezası nedir?"* sorusunda doğru madde (AI Act Madde 99: 35 milyon € veya cironun %7'si) 7. sırada kaldı ve LLM'e hiç ulaşmadı. LLM, AB kurumlarına verilen cezalarla ilgili Madde 100'den "1,5 milyon €" cevabını verdi. Hakem cevabı kaynaklara uygun bulduğu için "sadık" saydı. Faithfulness kontrolü arama hatalarını yakalayamaz; bu yüzden hem arama hem cevap ölçümü gerekir.
- **Hakemin reddettiği cevaplar** kaynaklarda olmayan küçük eklemelerdi (örneğin rıza için "tarihli kayıt"). Bir cevap, sadece 5 kaynak verildiği halde `[7]` kaynağını gösterdi.
- **PDF temizliği önemliydi:** bir ekteki "... in accordance with Article 49" atfı satır kayması yüzünden tek başına bir satıra düştü ve önce yeni bir madde sanıldı (113 yerine 114 madde). Sıra numarası kontrolüyle düzeldi.

## Web API (FastAPI)

`api.py` sistemi bir web servisine çevirir. Modeller sunucu açılırken **bir kez** yüklenir; sonra her soru yaklaşık 1-3 saniye sürer.

```bash
uvicorn api:app --port 8000
# tarayıcıda denemek için http://127.0.0.1:8000/docs adresini aç
```

| Uç nokta | Ne yapar |
|---|---|
| `POST /ask` | `{"question": "...", "use_rerank": true}` → cevap, numaralı kaynaklar, süre |
| `GET /health` | Sunucu açık mı, modeller yüklü mü? |
| `GET /docs` | FastAPI'nin otomatik oluşturduğu test sayfası |

İstekler otomatik kontrol edilir (soru 3-500 karakter olmalı, değilse `422`); LLM çağrısı başarısız olursa (örneğin hız sınırı) API çökmek yerine `503` döner.

Aynı soru, reranker açık ve kapalı:

| | `use_rerank: true` | `use_rerank: false` |
|---|---|---|
| Kaynaklar | **GDPR Article 22**, GDPR Recital 71, AI Act ... | sadece AI Act gerekçeleri |
| Cevap | Sadece bir GDPR istisnası varsa, insan müdahalesi ve itiraz hakkıyla birlikte | "Evet, AI Act'e göre yüksek risklidir" (GDPR Madde 22'yi kaçırıyor) |
| Süre | ~3,0 sn | ~1,0 sn |

Reranker CPU'da yaklaşık 2 saniye ekliyor ama soruyu gerçekten cevaplayan maddeyi buluyor.

Repoda bir `Dockerfile` da var (sadece CPU'lu torch, modeller imaj oluşturulurken iniyor, port 7860).

## Çalıştırma

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

1. PDF'leri EUR-Lex'ten indirip `data/` klasörüne `ai_act.pdf`, `gdpr.pdf`, `dora.pdf` adlarıyla kaydet:
   [AI Act](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689) ·
   [GDPR](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679) ·
   [DORA](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32022R2554)
2. `GROQ_API_KEY=...` satırını içeren bir `.env` dosyası oluştur (asla GitHub'a yükleme).
3. Çalıştır:

```bash
python ingest.py
python build_index.py
python rag.py "Can a bank reject a loan using only an automated decision?"
python eval.py --retrieval   # sadece arama metrikleri, API çağrısı yok
python eval.py               # tam ölçüm (Groq ücretsiz katmanında ~10-15 dk)
```

`chunks.jsonl` ve `index.faiss` repoda var; `rag.py` ve `eval.py` ilk iki adım olmadan da çalışır.

## Sınırlamalar

- Küçük test seti (32 kapsam içi soru): tek bir soru Recall@5'i yaklaşık 3 puan değiştirir. Sonuçlar kesin bir sayı değil, bir yön gösterir.
- Test sorularını ve doğru maddeleri kendim eşleştirdim; bazı sorular başka maddelerle de cevaplanabilir.
- DORA'nın 106 gerekçesinden 104'ü bulundu; diğer ikisinin metni bir önceki gerekçeye karıştı.
- pypdf bazen kelimeleri bitiştiriyor (`withoutundue`).
- Belgeler ilk yayımlanan metinler; sonraki değişiklikler dahil değil.
- Sadece İngilizce. Bu bir bilgilendirmedir, hukuki tavsiye değildir.

## Sonraki adımlar

- Tam terimler ve sayılar (örneğin ceza miktarı) için hybrid arama (BM25 + vektör, Reciprocal Rank Fusion)
- Daha büyük veya hukuk alanına özel bir reranker (örneğin `bge-reranker-base`) ile karşılaştırma
- API'yi (Docker) herkese açık bir sunucuya yüklemek
- Her kaynak numarasının verilen kaynaklarda gerçekten var olup olmadığını kontrol etmek

Kaynak belgeler: © Avrupa Birliği, [EUR-Lex](https://eur-lex.europa.eu), Komisyon'un yeniden kullanım politikası kapsamında kullanıldı.