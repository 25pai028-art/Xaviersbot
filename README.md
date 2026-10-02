# Xavier's Assistant — SXCA College Chatbot

A retrieval-augmented (RAG) chatbot for **St. Xavier's College (Autonomous), Ahmedabad**. It answers
questions only from the college website (crawled automatically) and documents uploaded by an admin,
and says *"I don't have that information"* instead of guessing.

> **Status:** Phases 1–4 of 8 are done: the deep crawler, hybrid search with fact-checking, the admin panel
> and the branded chat widget. Still to come: multilingual support and voice, faculty meeting booking,
> security hardening and deployment.

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

- **http://localhost:8000/demo**: a page styled like sxca.edu.in with the chat widget on it.
- **http://localhost:8000/test**: the plain developer test chat page.
- **http://localhost:8000/health**: the active LLM, whether it is reachable, and the number of indexed chunks.
- **http://localhost:8000/api/docs**: the API docs.

## Chat widget on the college website

Add one line before `</body>` on sxca.edu.in (in WordPress, with a header/footer scripts plugin or Elementor's custom code):

```html
<script src="https://CHATBOT-SERVER/widget.js" defer></script>
```

- A crimson **Ask Xavier's Assistant** button appears at the bottom right of every page. It opens the chat full screen, on desktop and on phones.
- Any link to `#ask-xavier` on the site also opens the chat.
- Add the website's address to `CORS_ORIGINS` in `.env` (it already includes `https://sxca.edu.in`).
- The widget uses a Shadow DOM, so the website's styles and the widget's styles never affect each other.
- The conversation stays in that browser tab only, until it is closed or **New chat** is pressed. Nothing is saved on the server.

**Colours and text.** The colours come from sxca.edu.in (navy `#243A7B`, crimson `#B8354E`, slate `#37424E`). To change them without editing code, add CSS to the website:

```css
#xaviers-assistant { --xaviers-navy: #243a7b; --xaviers-crimson: #b8354e; }
```

Other variables: `--xaviers-slate`, `--xaviers-page`, `--xaviers-surface`, `--xaviers-bubble`, `--xaviers-display-font` and `--xaviers-body-font`. The welcome text, logo, languages shown and office contact details are set in `.env` (`WIDGET_*` and `COLLEGE_OFFICE_*`).

## Languages

Students can type in English, Hindi, Gujarati, Malayalam, Tamil, Telugu, Kannada, Marathi, Bengali, Punjabi, Odia or Urdu, or pick a language in the chat header.

- **How it works:** a question typed in an Indian script is answered in that language, and the selector switches to it. The question is translated to English, answered from the college website and fact-checked in English. The finished answer is then translated back.
- **Safety:** after translation, every number, email and link must still be there. If one is missing, the English answer is shown instead, with a note.
- **Speed:** a translated answer appears all at once after translation, not word by word.
- **Voice:** the mic and read-aloud buttons use the browser's own Indian-language voices. Chrome and Android phones have most of them; Windows has fewer. If a voice is missing, the button says so.

**Translation models (recommended).** The AI4Bharat IndicTrans2 models run locally for free, give much better wording than the chat LLM, and are faster. They are "gated", so download them once:

1. Create a free account at https://huggingface.co.
2. Open both pages and click **Agree and access repository**:
   - https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M
   - https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M
3. Create a **Read** token at https://huggingface.co/settings/tokens.
4. Run `.venv\Scripts\hf auth login`, paste the token, then run `python -m scripts.download_translation`.

Without these models, `TRANSLATION_PROVIDER=auto` uses the chat LLM to translate. That works, but the wording is weaker with the small local model.

## Admin panel

1. Create the first admin (super admin). It asks for a password, which must be 10+ characters with letters and numbers:
   ```powershell
   python -m scripts.create_admin --username admin
   ```
2. Start the server with `uvicorn app.main:app` and open **http://localhost:8000/admin**.

| Page | What it is for |
|---|---|
| Dashboard | Indexed pages, documents and chunks; questions asked; unanswered rate; thumbs up/down; active AI model; AI cost when a paid model is used; last and next crawl |
| Sources | Search everything the chatbot knows and read the extracted text. **Remove and block** takes a page out and keeps future crawls from adding it back; uploads can be deleted |
| Upload | Add PDF, DOCX, TXT, PNG or JPG files. They are checked for the real file type, stored outside the web folder, and indexed within a minute |
| Crawl | **Re-crawl now** with live progress and a stop button, or add or refresh a single college page. Also shows recent runs and errors. An automatic re-crawl runs every Sunday at 02:00 (`CRAWL_SCHEDULE_*`) |
| Official answers | The college's own Q&A, used **before** the website. A close match is answered word for word; a partial match becomes the AI's main source. Use the test box to check how a question matches |
| Unanswered / Feedback | Questions the bot couldn't answer, and thumbs-down answers. Only the question text is kept, and entries are deleted after 30 days (`RETENTION_DAYS`) |
| Users | Super admin only. Add editors, reset passwords, deactivate accounts |
| Account | Change your password; turn on two-step login with an authenticator app |
| Audit log | Who uploaded, deleted, edited or crawled what, and when |

**Security:**
- Passwords are hashed with Argon2.
- An account is locked for 15 minutes after 5 wrong passwords, and there is a limit per network.
- Session cookies are HttpOnly and SameSite=Strict; sessions end after 30 minutes of inactivity or 8 hours in total.
- Every form needs a CSRF token.
- Admin pages send strict security headers (no framing, no inline scripts).

Run the server with a single worker, because the scheduler runs inside it.

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
