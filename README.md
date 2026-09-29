# Xavier's Assistant — SXCA College Chatbot

A retrieval-augmented (RAG) chatbot for **St. Xavier's College (Autonomous), Ahmedabad**. It answers
questions only from the college website (crawled automatically) and documents uploaded by an admin,
and says *"I don't have that information"* instead of guessing.

> **Status:** Phase 1 of 8. It has the crawler, document processing, indexing and a plain-text test chat page.
> Later phases add hybrid search and fact-checking, the admin panel, the branded widget, multilingual
> support and voice, faculty meeting booking, security hardening and deployment.

## Architecture (Phase 1)

```
sitemaps + WordPress REST API + link following
  → fetch (httpx; Playwright for JS pages) → extract (HTML / PDF / DOCX / OCR)
  → clean + hash (incremental) → chunk (heading-aware, ~500 tokens)
  → embed locally (bge-m3) → ChromaDB  +  metadata in SQLite

question → embed → vector search → similarity gate → LLM (Ollama | Gemini | Claude | OpenAI) → SSE stream
```

The server stores no conversations. The browser sends recent turns with each question.

## Hardware

| RAM | Recommended `OLLAMA_MODEL` |
|---|---|
| 8 GB | `qwen3.5:2b` |
| 16 GB | `qwen3.5:4b` (default) or `qwen3.5:9b` |
| 32 GB | `qwen3.5:35b-a3b` (MoE, fast on CPU) |

Everything runs CPU-only. Plan for about **10 GB of disk** for models (LLM 3.4 GB and bge-m3 2.3 GB, plus more in later phases).

## Local setup (Windows)

1. **Python 3.11**, from python.org. Check with `py -3.11 --version`.
2. **Ollama**, from https://ollama.com/download. Then pull the model:
   ```powershell
   ollama pull qwen3.5:4b
   ```
3. **Tesseract OCR** (needed for scanned PDFs and image notices):
   - Install the Windows build from https://github.com/UB-Mannheim/tesseract/wiki.
   - During setup, open *Additional language data* and tick **Hindi** and **Gujarati**.
   - The default path `C:\Program Files\Tesseract-OCR\tesseract.exe` is detected automatically. For any other location, set `TESSERACT_CMD` in `.env`.
4. **Project**:
   ```powershell
   cd XaviersBot
   py -3.11 -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   playwright install chromium
   copy .env.example .env
   ```
   `playwright install chromium` is optional: it lets the crawler render JavaScript-heavy pages.
   Edit `.env` if you need to change any settings.

## First crawl

| Command | What it does |
|---|---|
| `python -m scripts.crawl --max-pages 50` | Quick smoke test (a few minutes) |
| `python -m scripts.crawl` | Full deep crawl: crawl, clean, index (hours on CPU: ~3,000 PDFs) |
| `python -m scripts.crawl --no-index` | Crawl and clean only (network-bound, much faster). Index later. |
| `python -m scripts.crawl --index-only` | Index whatever is pending, e.g. after stopping a run with Ctrl+C |
| `python -m scripts.crawl --reindex` | Re-extract and re-embed everything (after changing extraction or chunk settings) |
| `python -m scripts.crawl --stats` | What is indexed, plus the latest errors |
| `python -m scripts.crawl --url https://sxca.edu.in/admissions/tuition-fees/` | Crawl one URL |

### Full crawl on Google Colab (free cloud GPU)

Embedding the whole site takes hours on a laptop CPU. It takes minutes on Colab's free T4 GPU, and Colab can also run OCR (Tesseract with Hindi and Gujarati).

1. Open `notebooks/colab_full_crawl.ipynb` in Colab (colab.research.google.com → File → Upload notebook).
2. Choose Runtime → Change runtime type → T4 GPU, then run all cells.
3. When it asks for a read-only GitHub token, paste one. The repository is private, and the token is not stored.
4. The finished knowledge base is saved to Google Drive as `MyDrive/xaviersbot/xaviersbot-data.zip`, with checkpoints after each stage. If Colab disconnects, run all cells again to resume.
5. On the laptop, stop the server, then run `ren data data-old` and `powershell Expand-Archive xaviersbot-data.zip -DestinationPath data`.

### How the deep crawl works

1. **Crawl.** URLs are discovered from:
   - the XML and HTML sitemaps and the WordPress REST API (pages, posts, and PDF/DOCX files in the media library)
   - every link on every page, recursively, up to `CRAWL_MAX_DEPTH`, including links in embedded PDF viewers, iframes, `data-href` attributes, `onclick` handlers and plain-text URLs
   - links inside PDFs
   - faculty profile pages (`/author/…`)

   Pages whose plain HTML has little text are rendered in headless Chromium. Accordion and tab panels are kept, while hidden text and mobile-only duplicate blocks are dropped. Emails and phone numbers stored in `data-emailid`, `mailto:` and `tel:` become readable text. Each PDF is titled from the link text on the page that links to it, and that page becomes its "section".
2. **Clean.** Lines repeated on at least 15% of pages ("Subscribe to Newsletter", quick-link blocks, contact strips) are removed from every page. They are kept once, in a "common site information" source.
3. **Index.** Empty pages and duplicate documents (the same PDF uploaded twice) are skipped. Everything else is chunked with its section path, for example "Admissions › UG Admissions › UG Science", embedded with bge-m3 in batches, and stored in ChromaDB.

Re-runs are incremental. Unchanged content is not downloaded or embedded again (checked by content hash, HTTP 304, and the WordPress modified date), and pages that return 404 are removed. Indexing progress is saved in SQLite, so an interrupted run resumes where it stopped.

- The first run downloads the bge-m3 embedding model (about 2.3 GB).
- Re-running is **incremental**. Unchanged pages and documents are not re-embedded, and pages that now return 404 are removed.
- Press **Ctrl+C** to stop safely. Anything already indexed is kept.

## Run the chat

```powershell
uvicorn app.main:app --reload
```

- **http://localhost:8000/test**: the test chat page.
- **http://localhost:8000/health**: the active LLM, whether it is reachable, and the number of indexed chunks.
- **http://localhost:8000/api/docs**: the API docs.

## Switching the LLM

Edit `.env` and restart. No re-indexing is needed.

| Provider | Settings in `.env` |
|---|---|
| Local (default) | `LLM_PROVIDER=ollama` |
| Google Gemini | `LLM_PROVIDER=gemini` and `GEMINI_API_KEY=...` |
| Claude | `LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY=...` and `ANTHROPIC_MODEL=...` |

## Tests

```powershell
pytest
```

## Crawl scope notes

- **Crawled:** `sxca.edu.in` (pages, news, events, and PDFs/DOCX uploaded since `CRAWL_MIN_UPLOAD_YEAR`), public Google Drive PDFs linked from college pages, and `library.sxca.edu.in`.
- **Never crawled:** the login portals (`lms.`, `portal.`, ERP).
- **Not crawled by default:** `admissions.sxca.edu.in`. Its robots.txt disallows all crawlers, and the crawler respects that. The admissions and fee pages on the main site are crawled.
