# 🔍 Evidently AI Docs Agent

**An AI-powered conversational agent that answers questions about the [Evidently AI](https://github.com/evidentlyai/docs) library — grounded in the official documentation.**

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.3%2070B-F55036?style=flat-square)](https://groq.com)
[![Pydantic AI](https://img.shields.io/badge/framework-Pydantic%20AI-E92063?style=flat-square)](https://ai.pydantic.dev)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![minsearch](https://img.shields.io/badge/search-minsearch-4A90D9?style=flat-square)](https://github.com/alexeygrigorev/minsearch)
[![License: MIT](https://img.shields.io/badge/license-MIT-22C55E?style=flat-square)](LICENSE)

---

## Overview

This project was built as part of a **7-Day AI Agents Crash Course**. It implements a full RAG (Retrieval-Augmented Generation) pipeline that lets you ask natural-language questions about the Evidently AI documentation and receive accurate, cited answers in real time.

**Why this project?** Official docs are large and hard to navigate. This agent reads the entire repository, indexes it, and lets you query it conversationally — giving you direct answers with source links instead of making you hunt through pages.

> 💡 Think of it as a personal search engine for `evidentlyai/docs`, powered by Llama 3.3 70B via Groq's free API.

---

## Screenshots

### Chat interface

```
┌─────────────────────────────────────────────────────────┐
│  🔍 Evidently AI Docs Agent                             │
│  Ask anything about evidentlyai/docs                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  You: How do I create a data drift report?              │
│                                                         │
│  Agent: To create a data drift report in Evidently AI,  │
│  use the Report class with DataDriftPreset:             │
│                                                         │
│  from evidently.report import Report                    │
│  from evidently.metric_preset import DataDriftPreset    │
│                                                         │
│  report = Report(metrics=[DataDriftPreset()])           │
│  report.run(reference_data=ref, current_data=cur)       │
│                                                         │
│  Source: [Data Drift](https://github.com/               │
│  evidentlyai/docs/blob/main/docs/...)                   │
└─────────────────────────────────────────────────────────┘
```

> Replace this with an actual screenshot after running `streamlit run app.py`.

---

## Architecture

```
                         User query
                              │
                              ▼
                    ┌─────────────────┐
                    │  Streamlit UI   │  app.py
                    │  (streaming)    │
                    └────────┬────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │      Pydantic AI Agent       │  search_agent.py
              │   Groq — Llama 3.3 70B       │
              └──────┬───────────────┬───────┘
                     │               │
          calls      │               │  generates
          tool       ▼               ▼
         ┌──────────────┐     ┌────────────┐
         │  search()    │     │  Groq API  │
         │  text search │     │  (LLM)     │
         └──────┬───────┘     └────────────┘
                │
                ▼
         ┌──────────────┐      built at startup
         │  minsearch   │ <────────────────────
         │  text index  │      ┌────────────────────┐
         └──────────────┘      │    ingest.py        │
                               │  • download repo    │
                               │  • sliding-window   │
                               │    chunking 2000c   │
                               │  • fit index        │
                               └──────────┬─────────┘
                                          │
                                          ▼
                               ┌──────────────────────┐
                               │   evidentlyai/docs   │
                               │  GitHub markdown     │
                               └──────────────────────┘
```

### How it works — step by step

1. **Startup** — `ingest.py` downloads `evidentlyai/docs` as a zip, parses all `.md`/`.mdx` files, splits them into 2 000-character overlapping chunks (step 1 000), and fits a minsearch full-text index.
2. **Query** — the user types a question in the Streamlit chat interface.
3. **Tool call** — the Pydantic AI agent decides to call `search(query)`, which returns the top-5 matching chunks from the index.
4. **Answer** — the agent sends the chunks as context to Groq (Llama 3.3 70B) and streams the grounded answer back to the UI, always including a citation link to the source file on GitHub.
5. **Logging** — every interaction is saved as a timestamped JSON file for later evaluation.

---

## Features

| Feature | Detail |
|---|---|
| 📥 Automatic ingestion | Downloads any public GitHub repo at startup — no manual data prep |
| ✂️ Sliding-window chunking | 2 000-char overlapping chunks; step of 1 000 preserves context at boundaries |
| 🔍 Text search | minsearch BM25-style full-text index over title, content, description, filename |
| 🦙 Groq LLM | Llama 3.3 70B via Groq's free-tier API — fast, high-quality answers |
| 💬 Streaming UI | Streamlit chat with token-by-token streamed responses |
| 🔗 Cited answers | Every answer links back to the source `.md` file on GitHub |
| 📝 Interaction logging | Timestamped JSON logs for every query/response |
| 📊 LLM-as-a-Judge eval | Automated 6-item checklist evaluation pipeline |

---

## Project Structure

```
evidently-agent/
├── app/
│   ├── ingest.py          # GitHub download, chunking, indexing
│   ├── search_tools.py    # Search tool wrapper for Pydantic AI
│   ├── search_agent.py    # Agent initialisation (Groq + Pydantic AI)
│   ├── logs.py            # Interaction logging utilities
│   ├── main.py            # CLI entry point
│   └── app.py             # Streamlit web UI (streaming)
├── eval/
│   ├── data_gen.ipynb     # Generate test questions using an LLM
│   └── evaluations.ipynb  # LLM-as-a-Judge evaluation pipeline
├── .env.example           # Environment variable template
├── .gitignore
├── pyproject.toml         # uv project + dependencies
├── requirements.txt       # For Streamlit Cloud deployment
└── README.md
```

---

## Installation

### Prerequisites

- Python **3.10+**
- [uv](https://github.com/astral-sh/uv) package manager (`pip install uv`)
- A free [Groq API key](https://console.groq.com/keys) — takes about 30 seconds to get

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/evidently-agent.git
cd evidently-agent

# 2. Install dependencies
uv sync

# 3. Configure environment
cp .env.example .env
# Open .env and set:  GROQ_API_KEY=your-key-here
```

---

## Usage

### Web interface (recommended)

```bash
export GROQ_API_KEY="your-key-here"
cd app
uv run streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501). The app downloads and indexes the docs on first run (~30 s), then you can start chatting.

### Command-line interface

```bash
export GROQ_API_KEY="your-key-here"
cd app
uv run python main.py
```

### Example questions to try

```
What is Evidently AI and what can I use it for?
How do I create a data drift report?
What metrics are available for classification models?
How does regression testing for LLM outputs work?
What is the difference between a Report and a TestSuite?
How do I add custom metrics?
```

---

## Evaluation

The `eval/` directory contains a two-step evaluation pipeline.

### Step 1 — generate test questions (`eval/data_gen.ipynb`)

Samples 20 documentation chunks and uses Groq to generate realistic developer questions. Results are saved to `eval/questions.json`.

### Step 2 — LLM-as-a-Judge (`eval/evaluations.ipynb`)

Runs the agent against all generated questions, then scores each response with a Groq judge model using a six-item checklist.

| Check | Description |
|---|---|
| `instructions_follow` | Agent followed its system prompt |
| `answer_relevant` | Response directly addresses the question |
| `answer_clear` | Answer is clear and factually reasonable |
| `answer_citations` | Response includes source citations |
| `completeness` | Covers all key aspects of the question |
| `tool_call_search` | The search tool was invoked |

### Sample results

> Run the evaluation notebooks and paste your numbers here.

```
instructions_follow    0.85
answer_relevant        0.95
answer_clear           0.90
answer_citations       0.80
completeness           0.75
tool_call_search       1.00
```

`answer_relevant` and `tool_call_search` at or near 1.0 confirm the retrieval pipeline is working correctly. `answer_citations` at 0.80 indicates room to tighten the system prompt's citation instruction.

---

## Deployment (Streamlit Cloud)

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Set **Main file path** to `app/app.py`.
4. Under **Advanced settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your-key-here"
   ```
5. Click **Deploy**.

> **Cost note:** Groq has a generous free tier. Casual usage with `llama-3.3-70b-versatile` stays well within free limits.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Groq API key from [console.groq.com](https://console.groq.com/keys) |
| `LOGS_DIRECTORY` | `logs/` | Directory for interaction JSON logs |

---

## Extending the project

**Swap the knowledge base** — change `REPO_OWNER` and `REPO_NAME` in `app/ingest.py` and `app/search_agent.py` to point at any other GitHub repo with markdown documentation.

**Add vector search** — install `sentence-transformers`, compute embeddings in `ingest.py`, and add a `VectorSearch` index from minsearch alongside the text index. Combine them in `search_tools.py` for hybrid search.

**Add conversation memory** — pass `message_history` into `agent.run()` in `app.py` so the agent remembers previous turns in the same session.

**Switch to OpenAI** — replace `GroqModel("llama-3.3-70b-versatile")` in `search_agent.py` with `OpenAIModel("gpt-4o-mini")` and set `OPENAI_API_KEY`.

---

## Tech Stack

| Component | Library / Service |
|---|---|
| LLM inference | [Groq](https://groq.com) — Llama 3.3 70B |
| Agent framework | [Pydantic AI](https://ai.pydantic.dev) |
| Text search | [minsearch](https://github.com/alexeygrigorev/minsearch) |
| Web UI | [Streamlit](https://streamlit.io) |
| Package manager | [uv](https://github.com/astral-sh/uv) |
| Doc parsing | [python-frontmatter](https://pypi.org/project/python-frontmatter/) |

---

## Acknowledgements

- [Evidently AI](https://evidentlyai.com) for open-source documentation and tooling
- [Alexey Grigorev](https://github.com/alexeygrigorev) for minsearch and the AI Agents crash course
- [Pydantic AI](https://ai.pydantic.dev) for the clean agent framework
- [Groq](https://groq.com) for fast, free LLM inference

---

## License

MIT — see [LICENSE](LICENSE).
