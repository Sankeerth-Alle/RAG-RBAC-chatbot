# FinSolve RBAC-RAG Chatbot

A small academic demonstration of role-based retrieval with FastAPI, LangGraph, Chroma, Gemini, and Streamlit.

## Features

- JWT bearer-token login with bcrypt-hashed demo passwords
- Centralized RBAC enforced before generation
- Hybrid HR retrieval for exact employee questions
- Chroma metadata filtering plus a second authorization check
- Grounded structured answers with validated, deduplicated sources
- Streamlit chat history, loading state, clear-chat, access display, and logout

## Setup

Requirements: Python 3.12+ and UV (or pip).

```powershell
copy .env.example .env
# Edit .env and set GOOGLE_API_KEY and a private JWT_SECRET_KEY.

cd backend
uv sync
cd ..\frontend
uv sync
```

`.env` is ignored by Git. Never put real secrets in source files or frontend code.

## Build the Vector Database

On a new machine, or after changing source documents or the embedding model:

```powershell
cd backend
uv run python build_vector_db.py --reset
```

The script reads `training/data`, creates Markdown and CSV documents, attaches `access_level`, `source_file`, and `source` metadata, embeds them with `GEMINI_EMBEDDING_MODEL`, and persists the collection to `VECTOR_DB_PATH`.

The indexed embedding model is recorded in Chroma metadata. The backend refuses to query a collection whose recorded model is missing or differs from configuration. Rebuild with `--reset` after changing `GEMINI_EMBEDDING_MODEL`.

## Run

```powershell
# Terminal 1
cd backend
uv run python src/main.py

# Terminal 2
cd frontend
uv run streamlit run main.py
```

The backend binds to `0.0.0.0` and uses `BACKEND_PORT`. The frontend uses `BACKEND_URL`. CORS origins are configured with comma-separated `ALLOWED_ORIGINS`.

## Demo Accounts and RBAC

These credentials are intentionally created for local demonstration only:

| Role | Username | Password | Access |
| --- | --- | --- | --- |
| HR | Natasha | `hrpass123` | HR records and HR documents |
| Engineering | Tony | `password123` | Engineering documents |
| C-Level | Shashank | `password123` | All indexed document classes under the prototype executive policy |

The frontend displays the authenticated username and role returned by the backend. It never sends a role or access level. Authorization is derived from the signed token and enforced before documents reach the LLM.

## API

### Login

```http
POST /login
Content-Type: application/json

{"username": "Natasha", "password": "hrpass123"}
```

### Chat

```http
POST /chat
Authorization: Bearer <access_token>
Content-Type: application/json

{"message": "What is Aadhya Patel's salary?"}
```

Response:

```json
{
  "answer": "...",
  "sources": ["hr_data.csv"],
  "user": {"username": "Natasha", "role": "hr"}
}
```

Other endpoints: `GET /`, `GET /health`, and protected `GET /test`.

## Tests

The test suite does not require a live Gemini call:

```powershell
cd backend
uv run pytest tests -q
```

It covers login errors, JWT access, role policy, exact HR matching, unauthorized-document removal, clean no-context responses, and API contracts.

## Project Paths

Relative paths are resolved from the repository root. No OneDrive, Desktop, or developer-specific path is required. The main components are:

```text
backend/src/       FastAPI, auth, RBAC, LangGraph service
backend/build_vector_db.py
backend/tests/     Offline automated tests
frontend/main.py   Streamlit UI
training/data/     Source documents
```
