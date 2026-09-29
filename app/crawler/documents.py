"""Text extraction from PDF, DOCX, TXT and images (with Tesseract OCR)."""
from __future__ import annotations

import io
import logging
import re
import shutil
from dataclasses import dataclass, field
from functools import lru_cache

from app.config import get_settings

log = logging.getLogger(__name__)

# Invisible / control characters that can hide instructions inside documents.
_INVISIBLE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff\u00ad]")


@dataclass
class ExtractedDoc:
    text: str
    title: str = ""
    ocr_used: bool = False
    pages: int = 0
    links: list[str] = field(default_factory=list)  # URLs linked from inside the document


def sanitize_text(text: str) -> str:
    text = _INVISIBLE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ---------------------------------------------------------------- OCR

@lru_cache
def ocr_available() -> tuple[bool, str]:
    """(available, language string actually usable)."""
    try:
        import pytesseract
    except ImportError:
        return False, ""
    settings = get_settings()
    cmd = settings.tesseract_cmd or shutil.which("tesseract") or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    pytesseract.pytesseract.tesseract_cmd = cmd
    try:
        installed = set(pytesseract.get_languages(config=""))
    except Exception:  # binary missing
        log.warning("Tesseract not found — OCR disabled. Install it (see README) to read scanned notices.")
        return False, ""
    wanted = [l for l in settings.ocr_languages.split("+") if l in installed]
    missing = set(settings.ocr_languages.split("+")) - installed
    if missing:
        log.warning("Tesseract language packs missing: %s", ", ".join(sorted(missing)))
    return bool(wanted), "+".join(wanted or ["eng"])


def ocr_image(image) -> str:  # PIL.Image
    ok, langs = ocr_available()
    if not ok:
        return ""
    import pytesseract

    image = image.convert("L")
    # Upscale small images — Tesseract works best at ~300 DPI equivalent.
    if image.width < 1500:
        scale = 1500 / max(image.width, 1)
        image = image.resize((int(image.width * scale), int(image.height * scale)))
    try:
        return pytesseract.image_to_string(image, lang=langs, config="--psm 3")
    except Exception as e:
        log.warning("OCR failed: %s", e)
        return ""


def extract_image(data: bytes) -> ExtractedDoc:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as img:
        if img.width < 300 or img.height < 300:  # icons / thumbnails
            return ExtractedDoc(text="")
        text = ocr_image(img)
    text = sanitize_text(text)
    # Discard OCR noise from photos (few real words).
    words = re.findall(r"\w{3,}", text)
    if len(words) < 8:
        return ExtractedDoc(text="", ocr_used=True)
    return ExtractedDoc(text=text, ocr_used=True)


# ---------------------------------------------------------------- PDF

def _pdf_tables_text(page) -> tuple[str, list]:
    """Tables on a PDF page as linearised text, plus their bounding boxes."""
    try:
        tables = page.find_tables()
    except Exception:
        return "", []
    out, boxes = [], []
    for tab in tables.tables:
        rows = [[(c or "").replace("\n", " ").strip() for c in row] for row in tab.extract()]
        rows = [r for r in rows if any(r)]
        if len(rows) < 2:
            continue
        header = rows[0]
        lines = []
        for r in rows[1:]:
            pairs = [f"{header[i]}: {v}" if i < len(header) and header[i] else v for i, v in enumerate(r) if v]
            lines.append("; ".join(pairs))
        out.append("\n".join(lines))
        boxes.append(tab.bbox)
    return "\n\n".join(out), boxes


def extract_pdf(data: bytes) -> ExtractedDoc:
    import pymupdf as fitz

    settings = get_settings()
    doc = fitz.open(stream=data, filetype="pdf")
    parts: list[str] = []
    links: list[str] = []
    ocr_used = False
    ocr_pages = 0
    title = (doc.metadata or {}).get("title", "") or ""
    for page in doc:
        links += [lnk["uri"] for lnk in page.get_links() if lnk.get("uri")]
        tables_text, boxes = _pdf_tables_text(page)
        if boxes:
            # Text outside tables, then the linearised tables.
            blocks = page.get_text("blocks")
            body = "\n".join(
                b[4] for b in blocks
                if b[6] == 0 and not any(fitz.Rect(b[:4]).intersects(fitz.Rect(bx)) for bx in boxes)
            )
            text = body + "\n\n" + tables_text
        else:
            text = page.get_text("text")
        if len(text.strip()) < 30 and page.get_images() and ocr_pages < settings.crawl_ocr_max_pages:
            # Scanned page → render and OCR.
            from PIL import Image

            pix = page.get_pixmap(dpi=250)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            text = ocr_image(img)
            ocr_used = ocr_used or bool(text.strip())
            ocr_pages += 1
        parts.append(text)
    pages = doc.page_count
    doc.close()
    return ExtractedDoc(text=sanitize_text("\n\n".join(parts)), title=title.strip(), ocr_used=ocr_used, pages=pages,
                        links=list(dict.fromkeys(links)))


# ---------------------------------------------------------------- DOCX / TXT

def extract_docx(data: bytes) -> ExtractedDoc:
    import docx

    d = docx.Document(io.BytesIO(data))
    parts: list[str] = []
    for p in d.paragraphs:
        t = p.text.strip()
        if not t:
            continue
        style = (p.style.name or "").lower() if p.style is not None else ""
        if style.startswith("heading"):
            level = re.sub(r"\D", "", style) or "2"
            parts.append(f"{'#' * min(int(level), 6)} {t}")
        else:
            parts.append(t)
    for table in d.tables:
        rows = [[c.text.strip() for c in r.cells] for r in table.rows]
        if len(rows) >= 2:
            header = rows[0]
            for r in rows[1:]:
                parts.append("; ".join(f"{header[i]}: {v}" if header[i] else v for i, v in enumerate(r) if v))
    title = (d.core_properties.title or "").strip()
    return ExtractedDoc(text=sanitize_text("\n\n".join(parts)), title=title)


def extract_txt(data: bytes) -> ExtractedDoc:
    for enc in ("utf-8", "utf-16", "cp1252"):
        try:
            return ExtractedDoc(text=sanitize_text(data.decode(enc)))
        except UnicodeDecodeError:
            continue
    return ExtractedDoc(text=sanitize_text(data.decode("utf-8", errors="ignore")))


def sniff_type(data: bytes) -> str | None:
    """Real file type from magic bytes: 'pdf' | 'docx' | 'image' | None."""
    head = data[:8]
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"\x89PNG") or head.startswith(b"\xff\xd8\xff"):
        return "image"
    if head.startswith(b"PK\x03\x04") and b"word/" in data[:4000]:
        return "docx"
    return None


def extract_document(data: bytes, kind: str) -> ExtractedDoc:
    real = sniff_type(data) or kind
    if real == "pdf":
        return extract_pdf(data)
    if real == "docx":
        return extract_docx(data)
    if real == "image":
        return extract_image(data)
    return extract_txt(data)
