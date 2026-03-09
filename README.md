---
title: AI-Powered Recruitment Agent
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# AI Recruitment Agent

Analyze resumes against job roles, ask questions about candidates, generate interview Q&A, and get improvement suggestions — powered by Groq LLM and RAG.

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/viki77/AI-Powered_Recruitment_Agent)

## What it does

Upload a resume (PDF or TXT), pick a target role or upload a custom job description, and the app will:

1. Extract text and build a FAISS vector store from resume chunks
2. Score each required skill 0–10 using Groq's `llama-3.1-8b-instant`
3. Compute an overall match score (cutoff: 50%) with strengths and gaps
4. Let you ask follow-up questions (RAG-backed Q&A)
5. Generate interview questions with answers shown inline
6. Suggest resume improvements with example bullet points

All tabs have downloadable `.txt` reports (analysis, interview Q&A, improvement suggestions).

## Setup

### Run locally

```bash
git clone https://github.com/viki77/AI-Powered_Recruitment_Agent.git
cd AI-Powered_Recruitment_Agent

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 — enter your [Groq API key](https://console.groq.com) in the sidebar.

### Run with Docker

```bash
docker build -t recruitment-agent .
docker run -p 8501:8501 recruitment-agent
```

### Deploy to Hugging Face Spaces

This repo is configured for Docker-based Spaces. The YAML frontmatter at the top of this README tells HF how to build the Space.

1. Push this repo to a Hugging Face Space (SDK: Docker)
2. Add your `GROQ_API_KEY` in Space Settings → Secrets
3. The included GitHub Action (`.github/workflows/hf-space-deploy.yml`) can auto-deploy on push to `main` — set `HF_TOKEN` in your GitHub repo secrets

## Project structure

```
app.py                  → Streamlit entry point, role definitions, session state
agents.py               → ResumeAnalysisAgent class (extraction, RAG, LLM calls)
ui.py                   → All Streamlit UI components (sidebar, tabs, reports)
requirements.txt        → Python dependencies
Dockerfile              → Container config with Streamlit server flags
.streamlit/config.toml  → Disables XSRF/CORS, sets max upload size (200 MB)
.github/workflows/      → GitHub Actions workflow for HF Space deployment
resume.txt              → Sample resume for testing
jd.txt                  → Sample job description for testing
```

## How the agent works

`agents.py` contains `ResumeAnalysisAgent` which does the following:

- **Text extraction** — `PyPDF2` for PDFs, plain read for `.txt`
- **Embeddings** — `sentence-transformers/all-MiniLM-L6-v2` via `langchain_huggingface`
- **Vector store** — In-memory FAISS store built per resume using `langchain_community`
- **Skill scoring** — All skills analyzed in a single batched LLM call to minimize API usage
- **Weakness analysis** — Missing skills analyzed in one LLM call, returns JSON with suggestions
- **Q&A** — `RetrievalQA` chain over the FAISS store
- **Interview questions** — Prompt-engineered generation with role context and difficulty levels
- **Rate limiting** — Exponential backoff on 429 errors (up to 5 retries)
- **JSON parsing** — Regex extraction with fallbacks for single quotes and trailing commas

## Supported roles

Defined in `ROLE_REQUIREMENTS` in `app.py`:

AI/ML Engineer · Data Scientist · Software Engineer · Backend Engineer · Frontend Engineer · Data Engineer · Product Manager · UX/UI Designer · Data Analyst · DevOps Engineer · Security Engineer · Cloud Architect

Edit `ROLE_REQUIREMENTS` in `app.py` to add or modify roles.

## Troubleshooting

| Problem                                          | Fix                                                                                            |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| HF Space shows "Missing configuration in README" | The YAML frontmatter (`---` block) must be at the very top of README.md — this is now included |
| 403 on file upload                               | `.streamlit/config.toml` disables XSRF/CORS — make sure this file is deployed                  |
| Empty text from PDF                              | PyPDF2 only extracts selectable text — scanned/image PDFs won't work                           |
| Groq rate limits / 429                           | The agent retries with backoff — reduce request frequency or use a paid tier                   |
| ModuleNotFoundError                              | Run `pip install -r requirements.txt` — langchain is pinned to `>=0.3,<1.0`                    |

## Contributing

1. Fork and clone
2. Run the app locally and test your changes
3. Open a PR with a clear description

## License

MIT
