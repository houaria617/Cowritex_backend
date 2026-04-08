# CoWriteX Backend

Human-in-the-Loop AI Research Writing Assistant — Backend

---

## What is this?

CoWriteX is an AI-powered research writing assistant that helps researchers write, search, and cite academic papers. This repo contains the backend — the orchestrator, the agents, the API, and the database.

---

## Project Structure

```
cowritex-backend/
├── orchestrator/          # Coordinates everything — routes tasks between agents
├── writing_agent/         # Responsible for generating and editing drafts
├── literature_agent/      # Responsible for searching and retrieving papers
├── visualization/         # Generates charts, tables, and exportable files
├── api/                   # FastAPI backend — exposes endpoints to the frontend
├── database/              # Database schema and models
├── config/                # App settings and environment variables
└── tests/                 # Tests for all modules
```

---

## Team

| Name | Responsibility |
|------|---------------|
| [Your Name] | Orchestrator + FastAPI backend + Database |
| [Teammate 1] | Writing Agent |
| [Teammate 2] | Literature Agent |
| [Teammate 3] | Frontend (separate repo) |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestrator | LangGraph |
| LLM | Groq (Llama 3) |
| Vector Store | ChromaDB |
| Database | PostgreSQL (Supabase) |
| Backend | FastAPI |
| Package manager | uv |

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-org/cowritex-backend
cd cowritex-backend
```

### 2. Create and activate virtual environment
```bash
uv venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
uv sync
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Open .env and fill in your API keys
```

### 5. Run the API
```bash
uv run uvicorn api.main:app --reload
```

### 6. Open API docs
```
http://localhost:8000/docs
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in the values:

```bash
GROQ_API_KEY=           # Get from console.groq.com (free)
TAVILY_API_KEY=         # Get from app.tavily.com (free)
LANGSMITH_API_KEY=      # Get from smith.langchain.com (free)
DATABASE_URL=           # Get from Supabase dashboard
```

---

## Git Rules — Read Before You Push

### Branches
```
main        → protected, never push directly
dev         → main working branch
feature/    → your working branch (branch from dev)
```

### Workflow
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

### Commit format
```
type(scope): short description

Examples:
feat(orchestrator): add conditional routing logic
fix(literature): handle empty search results
refactor(state): update ResearchState fields
docs(readme): update setup instructions
```

### Golden rules
- **Never push directly to main or dev**
- **Never commit your `.env` file**
- **Each person only works in their own folder**
- **Update `.env.example` if you add a new environment variable**
- **All cross-agent communication goes through the shared state — do not import from another agent's folder without discussion**

---

## Coding Conventions

- Use **snake_case** for files, folders, functions, and variables
- Use **PascalCase** for class names
- Use **UPPER_SNAKE_CASE** for constants
- Every function must have a **docstring**
- Always use **type hints**
- Never hardcode API keys — always use `config/settings.py`
- No hardcoded model names — define them in `config/settings.py`

---

## Running Tests

```bash
uv run pytest tests/
```

---

## Questions?

Open an issue or message the team lead.