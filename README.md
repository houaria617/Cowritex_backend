<h1 align="center">
  <span style="color: #7c3aed"></span> 
  <span style="color: #8b5cf6">C</span><span style="color: #7c3aed">o</span><span style="color: #6d28d9">W</span><span style="color: #5b21b6">r</span><span style="color: #4c1d95">i</span><span style="color: #6d28d9">t</span><span style="color: #7c3aed">e</span><span style="color: #8b5cf6">X</span>
  <span style="color: #a78bfa; font-size: 0.6em;"> Backend</span>
</h1>

<h3 align="center">
  <span style="color: #a78bfa">⚙️ Human-in-the-Loop AI Research Writing Assistant — Backend</span>
</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python&logoColor=white&labelColor=4c1d95&color=7c3aed" />
  <img src="https://img.shields.io/badge/FastAPI-backend-009688?style=for-the-badge&logo=fastapi&logoColor=white&labelColor=4c1d95&color=7c3aed" />
  <img src="https://img.shields.io/badge/LangGraph-orchestrator-8b5cf6?style=for-the-badge&labelColor=4c1d95&color=7c3aed" />
  <img src="https://img.shields.io/badge/ChromaDB-vector_store-7c3aed?style=for-the-badge&labelColor=4c1d95&color=a78bfa" />
  <img src="https://img.shields.io/badge/Groq-Llama_3-0d9488?style=for-the-badge&labelColor=4c1d95&color=7c3aed" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/ENSIA-National%20Higher%20School%20of%20AI-7c3aed?style=flat-square" />
  <img src="https://img.shields.io/badge/Department-Intelligent%20Systems%20Engineering-8b5cf6?style=flat-square" />
</p>

---

<p align="center">
  <b>Team:</b> Bouaziz Zineb • Djabir Houaria • Meriche Yasmine • Haddoud Mehdi<br/>
  <b>Supervisor:</b> Dr. HADJ AMEUR
</p>

---

## <span style="color: #7c3aed">📖</span> Overview

> *"Co — Collaborative Human–AI Partnership | Write — End-to-End Academic Writing | X — Extended Intelligence"*

This repo contains the **backend** of CoWriteX — the orchestrator, the AI agents, the FastAPI server, and the database layer that power the research writing assistant.

<table align="center">
  <tr>
    <td align="center"><span style="color: #a78bfa; font-size: 1.5em;"></span><br/><b>LangGraph</b><br/>orchestrator</td>
    <td align="center"><span style="color: #a78bfa; font-size: 1.5em;"></span><br/><b>Groq Llama 3</b><br/>LLM provider</td>
    <td align="center"><span style="color: #a78bfa; font-size: 1.5em;"></span><br/><b>ChromaDB</b><br/>vector store</td>
    <td align="center"><span style="color: #a78bfa; font-size: 1.5em;"></span><br/><b>FastAPI</b><br/>REST API</td>
  </tr>
</table>

---

## <span style="color: #7c3aed">📁</span> Project Structure

```
cowritex-backend/
├── 🧠 orchestrator/          # Coordinates everything — routes tasks between agents
├── ✍️ writing_agent/         # Responsible for generating and editing drafts
├── 📚 literature_agent/      # Responsible for searching and retrieving papers
├── 📊 visualization/         # Generates charts, tables, and exportable files
├── 🔌 api/                   # FastAPI backend — exposes endpoints to the frontend
├── 🗄️ database/              # Database schema and models
├── ⚙️ config/                # App settings and environment variables
└── 🧪 tests/                 # Tests for all modules
```

---

## <span style="color: #7c3aed">🧩</span> Tech Stack

| **Layer** | **Technology** |
|:---:|:---:|
|  Orchestrator | LangGraph |
|  LLM | Groq (Llama 3) |
|  Vector Store | ChromaDB |
|  Database | PostgreSQL (Supabase) |
|  Backend | FastAPI |
|  Package Manager | uv |

---

## <span style="color: #7c3aed">🚀</span> Getting Started

### <span style="color: #8b5cf6"> Prerequisites</span>

```bash
Python 3.11+
uv (pip install uv)
```

### <span style="color: #8b5cf6"> Installation</span>

```bash
# Clone the repository
git clone https://github.com/zinebbouaziz/FRONT.git
cd cowritex-backend

# Create virtual environment
uv venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

# Install dependencies
uv sync
```

### <span style="color: #8b5cf6"> Environment Variables</span>

```bash
cp .env.example .env
```

Fill in your `.env` file:

```env
GROQ_API_KEY=           # Get from console.groq.com (free)
TAVILY_API_KEY=         # Get from app.tavily.com (free)
LANGSMITH_API_KEY=      # Get from smith.langchain.com (free)
DATABASE_URL=           # Get from Supabase dashboard
```

### <span style="color: #8b5cf6"> Run Development Server</span>

```bash
uv run uvicorn api.main:app --reload
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) 

### <span style="color: #8b5cf6"> Run Tests</span>

```bash
uv run pytest tests/
```

---

## <span style="color: #7c3aed"></span> Git Rules — Read Before You Push

### <span style="color: #8b5cf6"> Branches</span>

```
main        → protected, never push directly
dev         → main working branch
feature/    → your working branch (branch from dev)
```

### <span style="color: #8b5cf6"> Workflow</span>

```bash
# 1. Always branch from dev
git checkout dev
git pull
git checkout -b feature/your-feature-name

# 2. Work on your feature

# 3. Push and open a PR into dev
git push origin feature/your-feature-name

# 4. Get at least one review before merging
```

### <span style="color: #8b5cf6"> Commit Format</span>

```
type(scope): short description

Examples:
 feat(orchestrator): add conditional routing logic
 fix(literature): handle empty search results
 refactor(state): update ResearchState fields
 docs(readme): update setup instructions
```

### <span style="color: #8b5cf6"> Golden Rules</span>

<table>
  <tr>
    <td></td>
    <td><b>Never push directly to main or dev</b></td>
  </tr>
  <tr>
    <td></td>
    <td><b>Never commit your <code>.env</code> file</b></td>
  </tr>
  <tr>
    <td></td>
    <td><b>Each person only works in their own folder</b></td>
  </tr>
  <tr>
    <td></td>
    <td><b>Update <code>.env.example</code> if you add a new environment variable</b></td>
  </tr>
  <tr>
    <td></td>
    <td><b>All cross-agent communication goes through the shared state — do not import from another agent's folder without discussion</b></td>
  </tr>
</table>

---

## <span style="color: #7c3aed">📐</span> Coding Conventions

<table>
  <tr>
    <td></td>
    <td>Use <b>snake_case</b> for files, folders, functions, and variables</td>
  </tr>
  <tr>
    <td></td>
    <td>Use <b>PascalCase</b> for class names</td>
  </tr>
  <tr>
    <td></td>
    <td>Use <b>UPPER_SNAKE_CASE</b> for constants</td>
  </tr>
  <tr>
    <td></td>
    <td>Every function must have a <b>docstring</b></td>
  </tr>
  <tr>
    <td></td>
    <td>Always use <b>type hints</b></td>
  </tr>
  <tr>
    <td></td>
    <td>Never hardcode API keys — always use <code>config/settings.py</code></td>
  </tr>
  <tr>
    <td></td>
    <td>No hardcoded model names — define them in <code>config/settings.py</code></td>
  </tr>
</table>

---

<p align="center">
  <br/>
  <span style="color: #8b5cf6; font-size: 1.2em;"><b> CoWriteX Backend — Orchestrating AI for Trustworthy Academic Writing ⚡</b></span>
  <br/><br/>
  <span style="color: #a78bfa">ENSIA • Intelligent Systems Engineering • 2025/2026</span>
</p>
