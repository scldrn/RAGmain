# Open WebUI Integration

This integration keeps Open WebUI outside the core architecture. `agentic_rag` stays a standalone
CLI/API service, and Open WebUI talks to it through a thin Pipe adapter that lives under
`integrations/open_webui/`.

## Architecture

- `src/agentic_rag/` remains unaware of Open WebUI.
- Open WebUI imports `integrations/open_webui/agentic_rag_pipe.py` as a Pipe Function.
- The Pipe sends `POST /query` requests to the existing FastAPI runtime.
- Removing Open WebUI later only means deleting the Pipe from Open WebUI. The backend does not
  need to change.

## One-Command Local Launcher

If you want the whole stack ready with one command, use:

```bash
./scripts/isabella
```

`./scripts/isabella` starts the Agentic RAG API, boots an isolated Open WebUI runtime, signs in
programmatically against that runtime, registers or updates the Pipe function, applies the local
valves, and opens the UI. By default it serves Open WebUI at `http://127.0.0.1:8081` with
authentication disabled for that dedicated local runtime, so you can use it immediately without a
manual login step.

Useful flags:

- `./scripts/isabella --no-browser`
- `./scripts/isabella --webui-port 8090`
- `./scripts/isabella --api-port 8001`
- `./scripts/isabella --runtime-dir /custom/path`

To stop only the services started by Isabella for that runtime:

```bash
./scripts/isabella-stop
```

Or:

```bash
./scripts/isabella-stop --runtime-dir /custom/path
```

## What To Import Into Open WebUI

Use [`integrations/open_webui/agentic_rag_pipe.py`](../integrations/open_webui/agentic_rag_pipe.py).

The adapter implements an Open WebUI manifold pipe, so you can expose one or many collections as
selectable "models" inside the Open WebUI model selector.

## Start The Backend

Run the standalone API first:

```bash
source .venv/bin/activate
agentic-rag-api
```

By default the API listens on `http://127.0.0.1:8000`.

## Import In Open WebUI

1. Open the admin area in Open WebUI.
2. Go to `Workspace` -> `Functions`.
3. Import a new Pipe Function from file or paste the contents of
   `integrations/open_webui/agentic_rag_pipe.py`.
4. Enable the function.
5. Open the valves settings for the function and configure the backend URL.

## Recommended Valve Configuration

Minimal local setup:

```text
API_BASE_URL=http://host.docker.internal:8000
DEFAULT_COLLECTION_NAME=documents
COLLECTIONS=documents=Knowledge Base
```

Multi-collection example:

```text
API_BASE_URL=http://host.docker.internal:8000
DEFAULT_COLLECTION_NAME=documents
COLLECTIONS=documents=Knowledge Base
research-notes=Research Notes
team-a=Team A
MODEL_NAME_PREFIX=Agentic RAG
```

If your Agentic RAG API sits behind a reverse proxy with header-based auth, configure:

```text
AUTH_HEADER_NAME=X-API-Key
AUTH_HEADER_VALUE=your-shared-secret
```

## Notes

- The Pipe forwards the latest user text message to `POST /query`.
- The selected Open WebUI model determines the `collection_name`.
- `SHOW_STATUS=true` emits progress updates in the UI while the request is running.
- `SHOW_METADATA_FOOTER=true` appends collection and request metadata to the final answer.
- If Open WebUI runs in Docker, `host.docker.internal` is usually the simplest way to reach a
  backend running on the host. Adjust the URL to your network topology if you run both services
  elsewhere.
