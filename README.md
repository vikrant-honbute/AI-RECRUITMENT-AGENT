# AI Recruitment Agent

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://huggingface.co/spaces/viki77/AI-Powered_Recruitment_Agent)

One-line: A Streamlit app that analyzes resumes, generates interview questions, answers resume-related queries (RAG), and suggests resume improvements. Uses Groq LLM + HuggingFace embeddings and a small FAISS store for retrieval.

---

## Table of contents

- [Demo](#demo)
- [Quick overview](#quick-overview)
- [Getting started](#getting-started)
- [How to use](#how-to-use)
- [Project structure](#project-structure)
- [Key implementation notes](#key-implementation-notes)
- [Deployment](#deployment)
- [Troubleshooting & limitations](#troubleshooting--limitations)
- [Contributing](#contributing)
- [License & credits](#license--credits)

---

## Demo

Live demo: https://huggingface.co/spaces/viki77/AI-Powered_Recruitment_Agent

---

## Quick overview

This repository contains a small production-style prototype for resume analysis and interview-prep:

- `app.py` — Streamlit entry: builds UI, wires `ui.py` to `agents.py`.
- `ui.py` — Streamlit components: sidebar, tabs, upload, reports, download buttons.
- `agents.py` — `ResumeAnalysisAgent`: PDF/text extraction, RAG vector store creation (FAISS), semantic skill scoring, weakness analysis, interview Q&A generation.
- `requirements.txt`, `Dockerfile`, `.streamlit/config.toml`, and a GitHub Actions workflow for pushing to a Hugging Face Space.

Core behavior: upload a resume (PDF/TXT), choose a role or upload a JD, click "Analyze Resume" — the agent builds embeddings, runs batched LLM analysis to score skills, produces missing-skill suggestions and can generate interview Q&A and improvement reports.

---

## Getting started

Prereqs

- Python 3.10+ (project was developed with Python 3.13)
- Git
- (Optional) Docker

Local (development)

```bash
git clone https://github.com/viki77/AI-Powered_Recruitment_Agent.git
cd AI_RECRUITMENT_AGENT
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 and enter your GROQ API key in the sidebar when prompted.

Notes

- The UI requires you to enter your Groq API key in the sidebar. The app does not automatically read a GROQ env var by default.
- Uploaded PDFs must contain selectable text; scanned PDFs (images) are not parsed by PyPDF2.

---

## How to use

1. Start the app and enter your GROQ API key in the sidebar.
2. Select a predefined role (from `ROLE_REQUIREMENTS` in `app.py`) or upload a Job Description (PDF/TXT).
3. Upload a candidate resume (PDF or TXT).
4. Click `ANALYZE RESUME` — the app will:
   - Extract text from the resume
   - Create chunked embeddings with HuggingFace `all-MiniLM-L6-v2` and store in FAISS
   - Run a batched Groq LLM call to score skills 0–10 and produce reasoning
   - Compute overall score and missing skills; perform weakness analysis (JSON)
5. Use the `Resume Q&A` tab to ask targeted questions — answers come from a RetrievalQA chain over the FAISS store.
6. Generate interview questions (JSON array) with ideal answers from the LLM.
7. Request improvement suggestions and download the reports.

Example analysis result keys (returned by `analyze_resume`):

- `overall_score` (int percent)
- `selected` (bool thresholded by `cutoff_score`)
- `skill_scores` (dict skill->0..10)
- `skill_reasoning` (dict skill->text)
- `missing_skills` (list)
- `detailed_weaknesses` (list of JSON objects with suggestions and examples)

---

## Project structure

```
.
├── app.py                      # Streamlit orchestration
├── agents.py                   # ResumeAnalysisAgent: extraction, RAG, LLM prompts
├── ui.py                       # Streamlit UI components and report builders
├── requirements.txt            # Python deps
├── Dockerfile                  # Image plus Streamlit flags for Spaces
├── .streamlit/config.toml      # Streamlit runtime config used for Spaces
├── .github/workflows/          # Optional: GH Action to push to HF Space
├── resume.txt                  # Example resume
├── jd.txt                      # Example job description
└── README.md
```

---

## Key implementation notes (developer-focused)

- LLM provider: `agents.py` uses `langchain_groq.ChatGroq` to call Groq's `llama-3.1-8b-instant`. To swap providers, replace `ChatGroq(...)` calls with another LangChain LLM wrapper.
- Embeddings: `HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")` — used to compute vectors for FAISS.
- Vector store: `FAISS.from_texts(chunks, embeddings)` — ephemeral in-memory store created per resume.
- JSON parsing: LLM outputs are parsed with regex and fallbacks to handle common formatting problems (single quotes, trailing commas).
- Rate limits: agent implements exponential backoff (`_retry_with_backoff`) for Groq API calls.
- PDF handling: `PyPDF2.PdfReader` — only extracts selectable text.

Developer tips

- To let the app read an API key automatically from environment, add something like `os.getenv("GROQ_API_KEY")` in `ui.setup_sidebar()` as the default value for the input box.
- To persist FAISS between runs, replace the in-memory FAISS store with a persisted store (Chroma/Chromadb or disk-backed FAISS).

---

## Deployment

Docker (local/container)

```bash
# build
docker build -t ai-recruitment-agent:latest .
# run
docker run -p 8501:8501 ai-recruitment-agent:latest
```

- The `Dockerfile` in the repo sets Streamlit server flags (`--server.enableXsrfProtection=false`, `--server.enableCORS=false`) which are useful for HF Spaces.

Hugging Face Spaces

- The repo includes `.streamlit/config.toml` configured for file uploads. Push the project to a Space and add the GROQ API key to the Space "Secrets" configuration so the app can use it from the environment or Secrets panel.
- A GitHub Action (`.github/workflows/hf-space-deploy.yml`) is included to push changes from `main` to your HF Space. Add `HF_TOKEN` as a repository secret for the action to succeed.

Security note: Store API keys in the platform's secret store (Hugging Face Secrets, GitHub Secrets for CI) — do not commit keys to source.

---

## Troubleshooting & limitations

- PDF extraction fails / returns empty: ensure the resume PDF includes selectable text. If scanned, run OCR (Tesseract or similar) first.
- Long resumes: the agent chunks text with overlap; extremely large documents may increase token usage. Consider truncating or increasing chunk sizes carefully.
- Rate limiting: if you hit provider rate limits, reduce frequency, lower batch sizes, or add more aggressive backoff.
- Model output validity: LLM-generated JSON may need post-processing; the agent attempts fixes but edge cases might still require manual inspection.

If you see errors during run, check the console logs for Streamlit and the container logs (Docker) for full stack traces.

---

## Contributing

Contributions are welcome. Suggested steps:

1. Fork the repo and make a branch for your change.
2. Run the app locally and verify behavior.
3. Open a PR with a clear description and tests/examples where appropriate.

If you want, I can add a `CONTRIBUTING.md` and an issue template.

---

## License & credits

- License: MIT (no license file included — add `LICENSE` before public distribution if needed).
- Credits: Groq, HuggingFace, LangChain, Streamlit.

---

If you'd like, I can now (pick one):

- Add `CONTRIBUTING.md` and `ISSUE_TEMPLATE.md` (recommended).
- Add a short code change so the app automatically reads `GROQ_API_KEY` from the environment when available.
- Insert example screenshots into README (provide images or allow me to add placeholders).

Reply with which next step you prefer and I'll implement it.
