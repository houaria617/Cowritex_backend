# CoWriteX — ThunderClient Testing Workflow

A step-by-step guide to manually test the full system using ThunderClient
(VS Code extension). Follow the steps in order — each step uses IDs from
the previous response.

---

## 0. Setup

### Environment variables in ThunderClient
Go to **ThunderClient → Env → New Environment** named `CoWriteX Dev`.

| Variable | Value |
|---|---|
| `base_url` | `http://127.0.0.1:8000` |
| `token` | *(paste your Supabase JWT here — see below)* |
| `project_id` | *(filled after Step 2)* |
| `section_id` | *(filled after Step 3)* |
| `thread_id` | *(filled after Step 5)* |

### How to get your Supabase JWT for testing
1. Go to your Supabase dashboard → **Authentication → Users**
2. Create a test user or use an existing one
3. In your terminal:
```bash
curl -X POST https://YOUR_PROJECT.supabase.co/auth/v1/token?grant_type=password \
  -H "apikey: YOUR_ANON_KEY" \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"yourpassword"}'
```
4. Copy the `access_token` from the response → paste into `token` env variable.

### Auth header (add to every request)
**Header:** `Authorization: Bearer {{token}}`

---

## 1. Health Check

**GET** `{{base_url}}/health`

Expected response:
```json
{"status": "ok", "env": "development"}
```

---

## 2. Auth — Get My Profile

**GET** `{{base_url}}/auth/me`
Headers: `Authorization: Bearer {{token}}`

Expected:
```json
{
  "id": "...",
  "email": "test@example.com",
  "full_name": "...",
  "academic_position": null,
  "organization": null
}
```
✅ `password_hash` must NOT appear in response.

---

## 3. Projects — Create

**POST** `{{base_url}}/projects`
Headers: `Authorization: Bearer {{token}}`
Body (JSON):
```json
{
  "title": "My RAG Research Paper",
  "description": "Investigating retrieval-augmented generation for academic writing"
}
```

Expected: `201 Created`
```json
{
  "id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "title": "My RAG Research Paper",
  "status": "active",
  "progress": 0
}
```
📋 **Copy the `id` → paste into `project_id` env variable.**

---

## 4. Projects — Get Preferences

**GET** `{{base_url}}/projects/{{project_id}}/preferences`
Headers: `Authorization: Bearer {{token}}`

Expected:
```json
{
  "writing_style": "formal",
  "tone": "academic",
  "llm_provider": "groq",
  "grounded_only": false,
  "citation_style": "APA"
}
```

---

## 5. Projects — Update Preferences

**PATCH** `{{base_url}}/projects/{{project_id}}/preferences`
Headers: `Authorization: Bearer {{token}}`
Body:
```json
{
  "citation_style": "IEEE",
  "target_journal": "IEEE Transactions on Neural Networks",
  "grounded_only": false,
  "llm_provider": "groq"
}
```
Expected: `200 OK` with updated preferences.

---

## 6. Sections — Create

**POST** `{{base_url}}/projects/{{project_id}}/sections`
Headers: `Authorization: Bearer {{token}}`
Body:
```json
{
  "type": "introduction",
  "title": "Introduction",
  "position": 1
}
```
Expected: `201 Created`
📋 **Copy the `id` → paste into `section_id` env variable.**

Create a second section too (for surrounding-context testing):
```json
{
  "type": "methodology",
  "title": "Methodology",
  "position": 2
}
```

---

## 7. Sections — List

**GET** `{{base_url}}/projects/{{project_id}}/sections`
Headers: `Authorization: Bearer {{token}}`

Expected: array of 2 sections ordered by position.

---

## 8. Upload PDF (for Literature Agent)

**POST** `{{base_url}}/projects/{{project_id}}/upload/pdf`
Headers: `Authorization: Bearer {{token}}`
Body: **Form** (not JSON) → key: `file`, value: *(select a PDF file)*

Expected: `201 Created`
```json
{
  "filename": "my_paper.pdf",
  "path": "uploads/PROJECT_ID/my_paper.pdf",
  "project_id": "..."
}
```

---

## 9. Orchestrator — Run: CHAT intent

**POST** `{{base_url}}/projects/{{project_id}}/run`
Headers: `Authorization: Bearer {{token}}`
Body:
```json
{
  "user_message": "What is retrieval-augmented generation?",
  "section_id": null
}
```

Expected:
```json
{
  "thread_id": "...",
  "agent_output": "Retrieval-augmented generation (RAG) is...",
  "intents": ["chat"],
  "status": "pending_review"
}
```
> Chat intent goes directly to output (no HITL). Still returns `pending_review`
> at the API level but the output is ready.

---

## 10. Orchestrator — Run: WRITE intent

**POST** `{{base_url}}/projects/{{project_id}}/run`
Headers: `Authorization: Bearer {{token}}`
Body:
```json
{
  "user_message": "Write an introduction for a paper about RAG systems in academic writing",
  "section_id": "{{section_id}}"
}
```

Expected:
```json
{
  "thread_id": "xxxxxxxx-...",
  "agent_output": "In recent years, the proliferation of large language models...",
  "intents": ["write"],
  "status": "pending_review"
}
```
📋 **Copy `thread_id` → paste into `thread_id` env variable.**

---

## 11. Orchestrator — Status Check

**GET** `{{base_url}}/projects/{{project_id}}/run/{{thread_id}}/status`
Headers: `Authorization: Bearer {{token}}`

Expected:
```json
{
  "thread_id": "...",
  "status": "pending_review",
  "intents": ["write"],
  "last_agent": "writing",
  "error": null
}
```

---

## 12. Orchestrator — Resume: APPROVE

**POST** `{{base_url}}/projects/{{project_id}}/run/{{thread_id}}/resume`
Headers: `Authorization: Bearer {{token}}`
Body:
```json
{
  "hitl_action": "approve"
}
```

Expected:
```json
{
  "thread_id": "...",
  "agent_output": "✅ Output approved and saved.\n\n...",
  "status": "completed"
}
```
✅ Check Supabase → `document_versions` table — a new row should appear.

---

## 13. Orchestrator — Resume: EDIT

Run step 10 again to get a fresh thread_id, then:

**POST** `{{base_url}}/projects/{{project_id}}/run/{{thread_id}}/resume`
Body:
```json
{
  "hitl_action": "edit",
  "human_edited_text": "This is my manually edited version of the introduction."
}
```

Expected: `200` with `"status": "completed"`
✅ Check `document_versions` — `author_type` should be `"human"`.

---

## 14. Orchestrator — Resume: REGENERATE

Run step 10 again, then:

**POST** `{{base_url}}/projects/{{project_id}}/run/{{thread_id}}/resume`
Body:
```json
{
  "hitl_action": "regenerate",
  "hitl_feedback": "Make it more concise and focus on the technical contributions"
}
```

Expected: `"status": "pending_review"` — the graph loops back and generates again.
Then approve it with step 12.

---

## 15. Orchestrator — Run: SEARCH intent

**POST** `{{base_url}}/projects/{{project_id}}/run`
Body:
```json
{
  "user_message": "Find me the latest papers on RAG for question answering"
}
```

Expected: intents = `["search"]`, agent_output contains markdown table of papers.
Approve it. Then check:

**GET** `{{base_url}}/projects/{{project_id}}/sources`

Expected: list of saved papers with `relevance_score`.

---

## 16. Orchestrator — Run: LITERATURE intent

**POST** `{{base_url}}/projects/{{project_id}}/run`
Body:
```json
{
  "user_message": "Write a literature review on retrieval-augmented generation"
}
```

Expected: intents = `["literature"]`, long review with citations.
Approve it. Then check:

**GET** `{{base_url}}/projects/{{project_id}}/literature`

---

## 17. Orchestrator — Run: COMPOUND intent (write + search)

**POST** `{{base_url}}/projects/{{project_id}}/run`
Body:
```json
{
  "user_message": "Write the methodology section and find relevant papers on transformer fine-tuning",
  "section_id": "{{section_id}}"
}
```

Expected: intents = `["write", "search"]` (or similar compound),
agent_output contains both drafted text and search results merged.

---

## 18. Orchestrator — Run: VISUALIZE intent

**POST** `{{base_url}}/projects/{{project_id}}/run`
Body:
```json
{
  "user_message": "Create a bar chart showing accuracy: Baseline 78%, Proposed 91%, Ablation 85%"
}
```

Expected: intents = `["visualize"]`, agent_output with file path.
After approving:

**GET** `{{base_url}}/projects/{{project_id}}/visualizations`
Then download:
**GET** `{{base_url}}/visualizations/{viz_id}/download`

---

## 19. Suggestions — Inline Copilot

**POST** `{{base_url}}/sections/{{section_id}}/suggestions/inline`
Body:
```json
{
  "document": "The model works. It was tested on several datasets.",
  "target_text": "The model works."
}
```

Expected:
```json
{
  "original": "The model works.",
  "suggestion": "The proposed model demonstrates strong empirical performance.",
  "mode": "improve",
  "diff": [...]
}
```

---

## 20. Version History

**GET** `{{base_url}}/sections/{{section_id}}/versions`

Expected: list of versions ordered newest first, each with `version_number`,
`author_type`, `is_current`.

Restore a previous version:
**POST** `{{base_url}}/sections/{{section_id}}/versions/restore`
Body:
```json
{
  "version_id": "paste-old-version-id-here"
}
```

---

## 21. Chat History

**GET** `{{base_url}}/projects/{{project_id}}/chat`

Expected: chronological list of human + AI messages.

---

## 22. Error Cases to Verify

| Request | Expected |
|---|---|
| POST /projects with `{"title": ""}` | 422 |
| GET /projects/fake-uuid | 404 |
| GET /projects/OTHER_USER_PROJECT_ID | 403 |
| POST /run with `{"user_message": ""}` | 422 |
| POST /resume with `{"hitl_action": "edit"}` (no text) | 400 |
| POST /resume with `{"hitl_action": "maybe"}` | 422 |
| POST /upload/pdf with a .txt file | 400 |
| Any request without Authorization header | 403 |
| Any request with expired token | 401 |

---

## Suggested ThunderClient Collection Structure

```
CoWriteX/
├── 00_Health
│   └── GET /health
├── 01_Auth
│   ├── GET /auth/me
│   └── PATCH /auth/me
├── 02_Projects
│   ├── POST   /projects
│   ├── GET    /projects
│   ├── GET    /projects/{{project_id}}
│   ├── PATCH  /projects/{{project_id}}
│   ├── DELETE /projects/{{project_id}}
│   ├── GET    /projects/{{project_id}}/preferences
│   └── PATCH  /projects/{{project_id}}/preferences
├── 03_Sections
│   ├── POST   /projects/{{project_id}}/sections
│   ├── GET    /projects/{{project_id}}/sections
│   ├── GET    /sections/{{section_id}}
│   ├── PATCH  /sections/{{section_id}}
│   ├── DELETE /sections/{{section_id}}
│   ├── GET    /sections/{{section_id}}/versions
│   └── POST   /sections/{{section_id}}/versions/restore
├── 04_Orchestrator  ← test this most
│   ├── POST /projects/{{project_id}}/run          (chat)
│   ├── POST /projects/{{project_id}}/run          (write)
│   ├── POST /projects/{{project_id}}/run          (search)
│   ├── POST /projects/{{project_id}}/run          (literature)
│   ├── POST /projects/{{project_id}}/run          (visualize)
│   ├── POST .../run/{{thread_id}}/resume          (approve)
│   ├── POST .../run/{{thread_id}}/resume          (edit)
│   ├── POST .../run/{{thread_id}}/resume          (regenerate)
│   └── GET  .../run/{{thread_id}}/status
├── 05_Data
│   ├── GET    /projects/{{project_id}}/sources
│   ├── GET    /projects/{{project_id}}/literature
│   ├── GET    /literature/{{analysis_id}}
│   ├── GET    /sections/{{section_id}}/suggestions
│   ├── POST   /sections/{{section_id}}/suggestions/inline
│   ├── PATCH  /suggestions/{{suggestion_id}}
│   ├── GET    /projects/{{project_id}}/visualizations
│   ├── GET    /visualizations/{{viz_id}}/download
│   ├── GET    /projects/{{project_id}}/chat
│   └── DELETE /projects/{{project_id}}/chat
└── 06_Uploads
    ├── POST   /projects/{{project_id}}/upload/pdf
    ├── GET    /projects/{{project_id}}/upload/pdf
    └── DELETE /projects/{{project_id}}/upload/pdf/filename.pdf
```
