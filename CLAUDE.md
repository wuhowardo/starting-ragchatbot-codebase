# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

Requires a `.env` file in the repo root (copy from `.env.example`):
```
ANTHROPIC_API_KEY=your_key_here
```

Always use `uv` to run the server, Python files, and install packages. Never use `pip` directly.

```bash
# Run a Python file
uv run python some_script.py
```

```bash
# Install dependencies
uv sync

# Start the server (from repo root) — use Git Bash on Windows
./run.sh

# Or manually
cd backend && uv run uvicorn app:app --reload --port 8000
```

App runs at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

## Architecture

This is a full-stack RAG chatbot. The **backend** is a FastAPI app (`backend/`) served on port 8000, which also serves the **frontend** static files (`frontend/`) at the root path. There is no build step for the frontend.

### Request flow for a user query

1. `frontend/script.js` POSTs `{ query, session_id }` to `POST /api/query`
2. `backend/app.py` delegates to `RAGSystem.query()`
3. `RAGSystem` (`rag_system.py`) fetches session history, then calls `AIGenerator.generate_response()` with the query, history, and tool definitions
4. `AIGenerator` (`ai_generator.py`) makes a **first Claude API call** with `tool_choice: auto` and one registered tool: `search_course_content`
5. If Claude decides to search, `ToolManager` executes `CourseSearchTool.execute()`, which calls `VectorStore.search()` against ChromaDB
6. `VectorStore` optionally resolves a fuzzy course name via semantic search on the `course_catalog` collection, then retrieves top-5 chunks from `course_content` using `all-MiniLM-L6-v2` embeddings
7. Tool results are appended to the message history and a **second Claude API call** synthesizes the final answer
8. Sources, response, and session ID are returned to the frontend

### Key design decisions

- **Two ChromaDB collections**: `course_catalog` stores course-level metadata for fuzzy name resolution; `course_content` stores the actual text chunks for semantic search.
- **Session history is injected into the system prompt** (not as Claude message history) as a plain formatted string. Only the last 2 exchanges are kept (`MAX_HISTORY = 2`).
- **Documents are loaded at startup** (`app.py` `startup_event`). Already-indexed courses are skipped by title. To force a full re-index, call `vector_store.clear_all_data()` or pass `clear_existing=True` to `add_course_folder()`.
- **ChromaDB is persisted** to `backend/chroma_db/` on disk.

### Document format

Course `.txt` files in `docs/` must follow this structure:
```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 0: <title>
Lesson Link: <url>
<lesson content...>

Lesson 1: <title>
<lesson content...>
```
Files are chunked into ~800-character sentence-based chunks with 100-character overlap.
