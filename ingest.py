"""
ingest.py - Step 1 of the RAG pipeline / RAG sürecinin 1. adımı

EN: Reads the 3 EU regulation PDFs in data/, removes page headers and footers,
    splits the text into Recitals, Articles and Annexes, cuts long parts into
    smaller chunks, and saves every chunk with its source info to chunks.jsonl.
TR: data/ içindeki 3 AB yönetmeliği PDF'ini okur, sayfa başlık ve altlıklarını
    atar, metni Recital (gerekçe), Article (madde) ve Annex (ek) olarak ayırır,
    uzun kısımları küçük parçalara böler ve her parçayı kaynak bilgisiyle
    chunks.jsonl dosyasına kaydeder.

Run / Çalıştır:   python ingest.py
Debug / Kontrol:  python ingest.py --debug gdpr   (shows raw lines after "Whereas:")
"""
import json                       # EN: write chunks as JSON lines  / TR: parçaları JSON satırı olarak yazmak için
import re                         # EN: regular expressions         / TR: metin kalıpları (regex) için
import sys                        # EN: read command line args      / TR: komut satırı argümanları için
from bisect import bisect_right   # EN: fast page lookup            / TR: hızlı sayfa bulma için
from pathlib import Path          # EN: file paths                  / TR: dosya yolları için

from pypdf import PdfReader       # EN: PDF text extraction         / TR: PDF'ten metin çıkarmak için

DATA_DIR = Path("data")           # EN: folder with the PDFs        / TR: PDF'lerin olduğu klasör
OUT_FILE = Path("chunks.jsonl")   # EN: output file                 / TR: çıktı dosyası

# EN: file name -> short name used in answers and citations
# TR: dosya adı -> cevaplarda ve kaynak gösterirken kullanılacak kısa ad
DOCS = {"ai_act.pdf": "AI Act", "gdpr.pdf": "GDPR", "dora.pdf": "DORA"}

MAX_CHARS = 1500   # EN: max length of one chunk (~250-300 words) / TR: bir parçanın en fazla uzunluğu (~250-300 kelime)
OVERLAP = 200      # EN: characters shared by two neighbour pieces / TR: komşu iki parçanın ortak karakter sayısı

# EN: Lines that are page decoration, not law text. We drop them.
# TR: Yasa metni olmayan, sayfa süsü olan satırlar. Bunları atıyoruz.
NOISE = [
    re.compile(r"^OJ L,? .*$"),                       # EN: AI Act page header      / TR: AI Act sayfa başlığı
    re.compile(r"^ELI: http\S*.*$"),                  # EN: AI Act page footer      / TR: AI Act sayfa altlığı
    re.compile(r"^.{0,40}Official Journal of the European Union.{0,40}$"),  # EN: GDPR/DORA header (any order) / TR: GDPR/DORA başlığı (her sırada)
    re.compile(r"^EN$"),                              # EN: lone language label     / TR: tek başına dil etiketi
    re.compile(r"^\d+/\d+$"),                         # EN: page number like 5/144  / TR: "5/144" gibi sayfa no
    re.compile(r"^(CHAPTER|SECTION) [IVXLC\d]+$"),    # EN: chapter/section labels  / TR: bölüm etiketleri
]
# EN: First line of a footnote, e.g. "(1) OJ C 517, 22.12.2021, p. 56." or
#     "(4) Directive 95/46/EC ... of 24 October 1995 on ...". Footnotes give the full
#     date of an act; recitals only name it ("Directive 95/46/EC ... (4) seeks to ...").
# TR: Bir dipnotun ilk satırı, örn. "(1) OJ C 517, 22.12.2021, p. 56." veya
#     "(4) Directive 95/46/EC ... of 24 October 1995 on ...". Dipnotlar yasanın tam
#     tarihini verir; recital'lar sadece adını anar ("Directive 95/46/EC ... (4) seeks to ...").
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
FOOTNOTE = re.compile(rf"^\(\s*\d+\s*\)\s*(OJ |Position of|.*\bof \d{{1,2}} ({MONTHS}) \d{{4}})")

# EN: A heading is a line that contains ONLY "Article 12" or "ANNEX III".
# TR: Başlık, içinde SADECE "Article 12" veya "ANNEX III" yazan satırdır.
HEADING = re.compile(r"^(Article (\d+)|ANNEX ([IVXLC]+))\s*$", re.MULTILINE)
# EN: A recital starts with "(12)" at the beginning of a line.
# TR: Bir recital satır başındaki "(12)" ile başlar.
RECITAL = re.compile(r"^\((\d+)\)\s*", re.MULTILINE)

ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def roman_to_int(s):
    """EN: 'XIII' -> 13.  TR: Roma rakamını sayıya çevirir."""
    total = 0
    for i, ch in enumerate(s):
        v = ROMAN[ch]
        # EN: a smaller value before a bigger one is subtracted (IV = 4)
        # TR: büyük rakamdan önce gelen küçük rakam çıkarılır (IV = 4)
        total += -v if i + 1 < len(s) and ROMAN[s[i + 1]] > v else v
    return total


def read_pdf(path):
    """EN: Return the cleaned full text and the start position of every page.
       TR: Temizlenmiş tüm metni ve her sayfanın metindeki başlangıç yerini döndürür."""
    reader = PdfReader(path)
    lines, page_starts, length = [], [], 0
    for page in reader.pages:
        page_starts.append(length)                       # EN: where this page begins / TR: bu sayfanın başladığı yer
        for line in (page.extract_text() or "").splitlines():
            line = line.strip()
            # EN: footnotes sit at the bottom of the page: from the first footnote
            #     line on, skip the rest of this page
            # TR: dipnotlar sayfanın en altındadır: ilk dipnot satırından
            #     itibaren sayfanın geri kalanını atla
            if FOOTNOTE.match(line):
                break
            # EN: skip empty lines and headers/footers
            # TR: boş satırları ve başlık/altlıkları atla
            if not line or any(p.match(line) for p in NOISE):
                continue
            lines.append(line)
            length += len(line) + 1                      # EN: +1 for "\n" / TR: araya eklenen "\n" için +1
    return "\n".join(lines), page_starts


def page_of(offset, page_starts):
    """EN: Page number (1-based) of a character position.
       TR: Bir karakter konumunun hangi sayfada olduğunu (1'den başlayarak) bulur."""
    return bisect_right(page_starts, offset)


def split_long(text):
    """EN: Cut a long text into pieces of at most MAX_CHARS, preferably at a sentence end.
       TR: Uzun metni en fazla MAX_CHARS uzunlukta parçalara böler, mümkünse cümle sonundan."""
    if len(text) <= MAX_CHARS:
        return [text]                                    # EN: short enough / TR: zaten kısa
    pieces, start = [], 0
    while start < len(text):
        end = min(start + MAX_CHARS, len(text))
        if end < len(text):
            cut = text.rfind(". ", start, end)           # EN: last sentence end in the window / TR: penceredeki son cümle sonu
            if cut > start + MAX_CHARS // 2:             # EN: use it only if not too early / TR: çok erken değilse kullan
                end = cut + 1
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        # EN: start the next piece OVERLAP chars earlier, at a word start
        # TR: sonraki parçayı OVERLAP kadar geriden, bir kelime başından başlat
        nxt = text.find(" ", end - OVERLAP)
        start = nxt + 1 if 0 <= nxt < end else end
    return pieces


def make_sections(text, page_starts):
    """EN: Split one regulation into (section, title, page, body) parts.
       TR: Bir yönetmeliği (bölüm, başlık, sayfa, metin) parçalarına ayırır."""
    sections = []
    # EN: Articles start after this sentence; recitals are before it.
    # TR: Maddeler bu cümleden sonra başlar; recital'lar bu cümleden öncedir.
    m = re.search(r"HAVE ADOPTED THIS REGULATION", text, re.IGNORECASE)
    split_at = m.start() if m else 0

    # --- 1) Recitals / Gerekçeler -------------------------------------------
    # EN: Numbers must go up (1, 2, 3 ...). We allow a gap of up to 2, so one
    #     badly read recital does not stop the whole list.
    # TR: Numaralar artarak gitmeli (1, 2, 3 ...). En fazla 2'lik boşluğa izin
    #     veriyoruz; böylece kötü okunan tek bir recital tüm listeyi durdurmaz.
    preamble = text[:split_at]
    w = preamble.find("Whereas:")
    expected, starts = 1, []
    for r in RECITAL.finditer(preamble, max(w, 0)):
        num = int(r.group(1))
        if expected <= num <= expected + 2:
            starts.append((r.start(), r.end(), num))
            expected = num + 1
    for i, (s, body_start, num) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else split_at
        sections.append((f"Recital {num}", "", page_of(s, page_starts), preamble[body_start:end]))

    # --- 2) Articles and Annexes / Maddeler ve Ekler -------------------------
    # EN: Same idea: Article numbers must go 1, 2, 3 ... and Annex numbers I, II, III ...
    #     A reference like "... in accordance with\nArticle 49" that happens to
    #     sit alone on a line is skipped because 49 is not the expected number.
    # TR: Aynı mantık: Madde numaraları 1, 2, 3 ... ve Ek numaraları I, II, III ...
    #     sırasıyla gitmeli. Satır kayması yüzünden tek başına kalan
    #     "Article 49" gibi bir atıf, beklenen numara olmadığı için atlanır.
    next_art, next_annex, heads = 1, 1, []
    for h in HEADING.finditer(text, split_at):
        if h.group(2) and int(h.group(2)) == next_art:
            heads.append(h)
            next_art += 1
        elif h.group(3) and roman_to_int(h.group(3)) == next_annex:
            heads.append(h)
            next_annex += 1
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        body = text[h.end():end].strip()
        # EN: the first line after the heading is the title, e.g. "Right to erasure"
        # TR: başlıktan sonraki ilk satır maddenin adıdır, örn. "Right to erasure"
        title, _, rest = body.partition("\n")
        sections.append((h.group(1), title.strip(), page_of(h.start(), page_starts), rest))
    return sections


def debug(name):
    """EN: Print raw lines after 'Whereas:' to see how recitals look in this PDF.
       TR: Bu PDF'te recital'ların nasıl göründüğünü görmek için 'Whereas:' sonrasını yazdırır."""
    file_name = next(f for f, d in DOCS.items() if f.startswith(name.lower()))
    text, _ = read_pdf(DATA_DIR / file_name)
    w = text.find("Whereas:")
    for line in text[w:].splitlines()[:80]:
        print(repr(line[:110]))


def main():
    chunks = []
    for file_name, doc in DOCS.items():
        text, page_starts = read_pdf(DATA_DIR / file_name)
        sections = make_sections(text, page_starts)
        n_before = len(chunks)
        for section, title, page, body in sections:
            body = re.sub(r"\s*\n\s*", " ", body).strip()   # EN: PDF line breaks -> spaces / TR: PDF satır sonları -> boşluk
            if not body:
                continue
            header = f"{doc} - {section}" + (f" - {title}" if title else "")
            for k, piece in enumerate(split_long(body)):
                chunks.append({
                    "id": f"{doc}|{section}|{k}",   # EN: unique id / TR: benzersiz kimlik
                    "doc": doc,                      # EN: AI Act / GDPR / DORA
                    "section": section,              # EN: e.g. "Article 17" / TR: örn. "Article 17"
                    "title": title,                  # EN: article title / TR: madde adı
                    "page": page,                    # EN: page in the PDF / TR: PDF'teki sayfa
                    # EN: the header line helps the search find the right article
                    # TR: baştaki başlık satırı, aramanın doğru maddeyi bulmasına yardım eder
                    "text": f"{header}\n{piece}",
                })
        # EN: Summary line so we can check the counts / TR: Sayıları kontrol etmek için özet satırı
        arts = [s for s in sections if s[0].startswith("Article")]
        last = arts[-1][0] if arts else "-"
        print(f"{doc:7s} pages={len(page_starts):3d}  recitals={sum(s[0].startswith('Recital') for s in sections):3d}  "
              f"articles={len(arts):3d} (last: {last})  annexes={sum(s[0].startswith('ANNEX') for s in sections):2d}  "
              f"chunks={len(chunks) - n_before}")

    with OUT_FILE.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(chunks)} chunks to {OUT_FILE}")

    # EN: Quick check: show one known article / TR: Hızlı kontrol: bilinen bir maddeyi göster
    sample = next((c for c in chunks if c["doc"] == "GDPR" and c["section"] == "Article 17"), None)
    if sample:
        print("\nExample chunk:\n" + sample["text"][:500] + " ...")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--debug":
        debug(sys.argv[2])
    else:
        main()