# CoWriteX Backend API Guide

This document is written for frontend developers who need to link the CoWriteX web app to the FastAPI backend.
It explains how the backend is organized, how authentication works, the main route files, and the important API endpoints.

---

## 1. Backend Overview

The backend is a FastAPI application that exposes REST endpoints for authentication, project management, sections, AI orchestration, data, and uploads.
It sits behind `api/main.py`, which registers all API routers and starts the FastAPI app.

Key backend layers:
- `api/main.py` — application entrypoint and router registration
- `api/routes/` — grouped endpoint handlers
- `api/dependencies.py` — shared authentication and authorization logic
- `api/schemas/requests.py` — request body validation models
- `database/repository.py` — database access helpers
- `database/client.py` — Supabase client initialization

---

## 2. Starting the Backend

### Prerequisites
- Python 3.11+
- Supabase account and project
- API keys for LLM providers (Groq, Google Gemini, etc.)

### Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # Or if using uv: uv sync
   ```

2. Set up environment variables in `.env`:
   ```bash
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_SERVICE_KEY=your-service-key
   SUPABASE_JWT_SECRET=your-jwt-secret  # Optional, for HS256 fallback
   GROQ_API_KEY=your-groq-key
   GOOGLE_API_KEY=your-gemini-key  # Optional
   LANGSMITH_API_KEY=your-langsmith-key  # Optional, for tracing
   ```

3. Run the FastAPI server:
   ```bash
   uvicorn api.main:app --reload --port 8000
   ```

4. Access the API docs at `http://localhost:8000/docs`.

---

## 4. How the backend works

### App startup

`api/main.py` creates a FastAPI app and includes these routers:
- `auth`  → `api/routes/auth.py`
- `projects` → `api/routes/projects.py`
- `sections` → `api/routes/sections.py`
- `orchestrator` → `api/routes/orchestrator.py`
- `data` → `api/routes/data.py`
- `uploads` → `api/routes/uploads.py`

The backend also configures CORS and exposes docs at `/docs`.

### Authentication flow

The backend uses Supabase-style JWT tokens.

- Protected routes depend on `get_current_user()` from `api/dependencies.py`.
- This function validates the bearer token, extracts the `sub` claim as `user_id`, and returns it.
- If the token is invalid or expired, the request fails with `401 Unauthorized`.

### Authorization flow

For routes that require project access, the dependency `verify_project_access()` is used.
It reads the project from the database and confirms that the current `user_id` owns it.

### Database access

All data read/write operations occur through `database/repository.py`.
The route functions are thin wrappers that validate input, authorize the user, and call repository helpers.

---

## 5. Important backend files

### `api/main.py`
- FastAPI app creation
- CORS middleware registration
- Router registration
- Root endpoint `/`
- Health check `/health`

### `api/routes/auth.py`
- `GET /auth/me` — returns the authenticated user profile
- `PATCH /auth/me` — updates the researcher profile
- Reads JWT claims when the profile row does not exist yet

### `api/routes/projects.py`
- `POST /projects` — create project
- `GET /projects` — list projects for current user
- `GET /projects/{project_id}` — get project details
- `PATCH /projects/{project_id}` — update project metadata
- `DELETE /projects/{project_id}` — delete project
- `GET /projects/{project_id}/preferences` — read project preferences
- `PATCH /projects/{project_id}/preferences` — update preferences

### `api/routes/sections.py`
- `POST /projects/{project_id}/sections` — create a section in a project
- `GET /projects/{project_id}/sections` — list project sections
- `GET /sections/{section_id}` — get section details and current content
- `PATCH /sections/{section_id}` — update section metadata
- `DELETE /sections/{section_id}` — delete a section
- `GET /sections/{section_id}/versions` — list saved versions for section
- `POST /sections/{section_id}/versions/restore` — restore a version

### `api/routes/orchestrator.py`
- `POST /projects/{project_id}/run` — start a graph run / AI workflow
- `POST /projects/{project_id}/run/{thread_id}/resume` — resume a paused run after human review
- `GET /projects/{project_id}/run/{thread_id}/status` — check run status

### `api/routes/data.py`
- Sources: `GET /projects/{project_id}/sources`, `DELETE /sources/{source_id}`
- Literature: `GET /projects/{project_id}/literature`, `GET /literature/{analysis_id}`
- Suggestions: `GET /sections/{section_id}/suggestions`, `PATCH /suggestions/{suggestion_id}`
- Inline suggestions: `POST /sections/{section_id}/suggestions/inline`
- Visualizations: `GET /projects/{project_id}/visualizations`, `GET /visualizations/{viz_id}`, `GET /visualizations/{viz_id}/download`, `DELETE /visualizations/{viz_id}`
- Chat history: `GET /projects/{project_id}/chat`, `DELETE /projects/{project_id}/chat`

### `api/routes/uploads.py`
- `POST /projects/{project_id}/upload/pdf` — upload a PDF
- `GET /projects/{project_id}/upload/pdf` — list uploaded PDFs
- `DELETE /projects/{project_id}/upload/pdf/{filename}` — delete a PDF

### `api/dependencies.py`
- `get_current_user()` — validates bearer JWT and returns `user_id`
- `verify_project_access()` — ensures a project belongs to current user
- Handles Supabase JWT HS256 / RS256 logic

### `api/schemas/requests.py`
- Defines request bodies with Pydantic models
- Ensures frontend sends correct payload shapes
- Useful models:
  - `UserProfileUpdate`
  - `ProjectCreate`, `ProjectUpdate`, `PreferencesUpdate`
  - `SectionCreate`, `SectionUpdate`, `VersionRestore`
  - `RunRequest`, `ResumeRequest`, `InlineSuggestionRequest`, `SuggestionUpdate`

---

---

## 6. Auth endpoints

**Note on Authentication:** Login and signup are handled directly by Supabase Auth. The frontend should use Supabase's client libraries to authenticate users. Once authenticated, the frontend receives a JWT token, which is sent in the `Authorization: Bearer <token>` header for all protected API requests. The backend validates this token and extracts the user ID.

### `GET /auth/me`
Returns the authenticated user profile.

Behavior:
- If an extended profile exists in the database, returns it.
- If no DB profile exists yet, returns a minimal profile built from JWT claims.
- Never returns `404` just because the user has no profile row.

Headers:
- `Authorization: Bearer <token>`

Example response:
```json
{
  "id": "user-id",
  "email": "user@example.com",
  "full_name": "User Name",
  "academic_position": null,
  "organization": null,
  "field_interests": null,
  "created_at": null,
  "updated_at": null,
  "_source": "jwt_claims"
}
```

### `PATCH /auth/me`
Updates the logged-in user's profile data.

Headers:
- `Authorization: Bearer <token>`

Body:
```json
{
  "full_name": "Updated Name",
  "academic_position": "Professor",
  "organization": "MIT",
  "field_interests": "NLP"
}
```

Response:
- `200 OK` with the updated profile object
- `400 Bad Request` if body is empty

---

## 7. Project endpoints

### `POST /projects`
Create a new project.

Body:
```json
{
  "title": "My Research Project",
  "description": "Project description"
}
```

### `GET /projects`
List all projects owned by the current user.

### `GET /projects/{project_id}`
Get project details.

### `PATCH /projects/{project_id}`
Update title, description, or status (`active` / `archived`).

### `DELETE /projects/{project_id}`
Delete the project.

### `GET /projects/{project_id}/preferences`
Read project preferences.

### `PATCH /projects/{project_id}/preferences`
Update project preferences.

Common preferences payload:
```json
{
  "writing_style": "concise",
  "tone": "formal",
  "target_journal": "Nature",
  "language": "English",
  "assistance_level": "moderate",
  "citation_style": "APA",
  "grounded_only": true,
  "llm_provider": "groq"
}
```

---

## 8. Section endpoints

### `POST /projects/{project_id}/sections`
Create a section in a project.

Body:
```json
{
  "type": "introduction",
  "title": "Introduction",
  "position": 1
}
```

### `GET /projects/{project_id}/sections`
List sections for a project.

### `GET /sections/{section_id}`
Get a section and its current content.

### `PATCH /sections/{section_id}`
Update section metadata.

### `DELETE /sections/{section_id}`
Delete a section.

### `GET /sections/{section_id}/versions`
List version history for a section.

### `POST /sections/{section_id}/versions/restore`
Restore a saved section version.

Body:
```json
{ "version_id": "..." }
```

---

## 9. Orchestrator endpoints

This is the backend workflow engine for AI-generated content.

### `POST /projects/{project_id}/run`
Start a new AI run for a project section.

Body:
```json
{
  "user_message": "Generate a summary for this section",
  "section_id": "..."
}
```

### `POST /projects/{project_id}/run/{thread_id}/resume`
Resume a paused run after the human reviewer makes a decision.

Body:
```json
{
  "hitl_action": "approve",
  "hitl_feedback": "Looks good"
}
```

### `GET /projects/{project_id}/run/{thread_id}/status`
Check the current status of the run.

---

## 10. Data endpoints

### Sources
- `GET /projects/{project_id}/sources`
- `DELETE /sources/{source_id}`

### Literature
- `GET /projects/{project_id}/literature`
- `GET /literature/{analysis_id}`

### Suggestions
- `GET /sections/{section_id}/suggestions`
- `PATCH /suggestions/{suggestion_id}`

### Inline suggestions
- `POST /sections/{section_id}/suggestions/inline`

Body:
```json
{
  "document": "Existing text",
  "target_text": "Write a better conclusion",
  "section_id": "..."
}
```

### Visualizations
- `GET /projects/{project_id}/visualizations`
- `GET /visualizations/{viz_id}`
- `GET /visualizations/{viz_id}/download`
- `DELETE /visualizations/{viz_id}`

### Chat history
- `GET /projects/{project_id}/chat`
- `DELETE /projects/{project_id}/chat`

---

## 11. Upload endpoints

### `POST /projects/{project_id}/upload/pdf`
Upload a PDF file.

Headers:
- `Content-Type: multipart/form-data`
- `Authorization: Bearer <token>`

### `GET /projects/{project_id}/upload/pdf`
List uploaded PDFs for the project.

### `DELETE /projects/{project_id}/upload/pdf/{filename}`
Delete a project PDF.

---

## 12. Frontend integration notes

### Authorization header
All protected routes require this header:
```http
Authorization: Bearer <token>
```

### Error handling
- `401` → authentication failed, invalid/expired token
- `403` → access denied
- `404` → resource not found
- `400` → invalid request body
- `204` → successful delete action with no response body

### Use `GET /auth/me` first
On app startup, call `GET /auth/me` to get the current user profile and determine if the profile is database-backed or JWT-backed.

### Project access
Routes under `/projects/{project_id}` only work when the authenticated user owns that project.

### Request validation
The backend uses Pydantic models for request bodies, so make sure the frontend sends the expected JSON fields and types.

---

## 13. Where to change backend behavior

- `api/routes/*.py` — add or update endpoints
- `api/dependencies.py` — change auth/permission logic
- `api/schemas/requests.py` — change request schemas and validation
- `database/repository.py` — change persistence and query logic
- `database/client.py` — change how the Supabase client is created

---

## 14. Useful links

- FastAPI docs: https://fastapi.tiangolo.com/
- API docs in development: `http://localhost:8000/docs`
- Backend entrypoint: `api/main.py`
- Auth route handler: `api/routes/auth.py`
- Project route handler: `api/routes/projects.py`
- Orchestrator route handler: `api/routes/orchestrator.py`
