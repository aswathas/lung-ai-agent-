# lung-ai-agent-

> **This is a student demonstration project. It is not a medical device and must not be
> used for diagnosis, treatment, or any clinical decision.** Everything it produces is
> unvalidated output from a general-purpose language model. Always consult a qualified
> healthcare professional.

A Streamlit chat app that answers lung-health and respiratory-diet questions using
retrieval-augmented generation over your own PDFs and web pages, with optional web-search
fallback, plus air-quality (AQI) lookups and medication-reminder emails.

## What the code actually does

Everything lives in [`AI Agents/`](AI%20Agents/), a single Streamlit script,
[`AI Agents/Report_Generator.py`](AI%20Agents/Report_Generator.py) (~690 lines), and an
earlier prototype, [`AI Agents/test.py`](AI%20Agents/test.py).

### RAG chat (`Report_Generator.py`)

1. **Query rewriting** — an `agno` agent (`gemini-exp-1206`) expands the user's question
   (e.g. `COPD` → `Chronic Obstructive Pulmonary Disease`) before retrieval.
2. **Retrieval** — uploaded PDFs and web pages are chunked with
   `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap), embedded with Google
   `text-embedding-004` (768-dim, cosine) through a small `GeminiEmbedder` wrapper, and
   stored in a **Qdrant** collection named `gemini-thinking-agent-agno`. Retrieval is
   `similarity_score_threshold` with `k=5` and a threshold slider (default `0.7`).
3. **Web-search fallback** — if no document clears the threshold (or you flip the 🌐 toggle),
   an `agno` agent calls **Exa** search (`include_domains` configurable, default
   `arxiv.org`, `wikipedia.org`, `github.com`, `medium.com`) and its summary becomes the
   context instead.
4. **Answer** — a `Pulmonology Specialist Agent` on `gemini-2.0-flash-thinking-exp-01-21`
   composes the answer from context plus your question.

Sidebar controls: API keys, chat-history clear, web-search toggle, similarity threshold,
AQI threshold, tablet-reminder toggle.

### Air-quality lookup

- `get_live_location()` — city from `ipinfo.io/json`, falling back to `shanghai`.
- `get_pollution_data()` — AQI and PM values from the **WAQI** feed API.

### Medication reminders (email)

Gmail SMTP sends reminders via `smtplib`, either from the **Air Pollution Alert** tab or
automatically from the **Medicine Schedule** tab: "Check Pollution & Send Reminders
Automatically" sends only when the current AQI is `>=` the sidebar threshold (default 100).
Medicines are entered as `name,time` lines in a textarea.

### Report generator (`test.py`)

The earlier prototype: upload a TXT/PDF patient document, and a Gemini Flash Thinking model
writes a report with observations, possible diagnoses, treatment and dietary
recommendations, and further-testing suggestions. Plus a plain chat tab. It is not
integrated with the RAG app above and reads a document from end to end with no chunking.

## Running it

Python 3.10+ recommended (the pinned deps target that era).

```bash
cd "AI Agents"
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run Report_Generator.py
```

`requirements.txt` is a full `pip freeze` (~200 pinned packages), so it will install far
more than this app uses. `requirement.txt` is an earlier snapshot of the same list.

### Configuration

Keys are read from environment variables, with `.env` loaded via `python-dotenv`:

| Variable | Used for |
| --- | --- |
| `GOOGLE_API_KEY` | Gemini models + embeddings |
| `QDRANT_API_KEY`, `QDRANT_URL` | Qdrant vector store |
| `EXA_API_KEY` | Web-search fallback |
| `POLLUTION_API_KEY` | WAQI AQI feed (defaults to `demo`) |
| `GMAIL_USER`, `GMAIL_APP_PASSWORD`, `RECIPIENT_EMAIL` | SMTP medication reminders |

The sidebar also accepts the Google, Qdrant and Exa keys at runtime; `test.py` reads its key
from the sidebar or `st.secrets`.

## Security warning

`AI Agents/.env` and `AI Agents/api_keys.txt` are **committed to this repository and contain
live credentials** — a Google API key, a Qdrant key and URL, an Exa key, a WAQI token, a
Gmail app password with the sending address, Twilio SID/token, and a recipient phone number.
`test.py` additionally hardcodes a Gemini API key as its fallback default.

Treat all of these as **public and compromised**: revoke and rotate them before doing
anything else. Deleting the files in a later commit does not un-publish them from git
history — the keys must be rotated, not just removed.

## Limitations

- Unvalidated output from general-purpose LLMs. No evaluation, no accuracy benchmark, no
  clinical validation exists in this repo. The agent is *instructed* to say "consult a
  healthcare professional", but that is a prompt string, not a safeguard.
- Medical reasoning is model recall plus whatever the retrieved chunks contain — there is no
  citation enforcement, and web results are summarised without provenance checks.
- Web scraping targets are filtered by CSS class names (`post-content`, `content`, `main`), so
  many pages will parse to almost nothing.
- Retrieval is single-shot against one collection; there is no reranking and no hybrid search.
- No tests, no CI, no logging, no persistence — chat history lives in Streamlit session state.

## License

None specified.