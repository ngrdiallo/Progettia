from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import threading
import time

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from ddgs import DDGS
from tavily import TavilyClient

from router_core import LocalModelRouter, RouterError
from chat_storage import ChatStorage


class ChatStorageDep:
    def __init__(self):
        self._storage: ChatStorage | None = None

    def get(self) -> ChatStorage:
        if self._storage is None:
            self._storage = ChatStorage()
        return self._storage


chat_storage_dep = ChatStorageDep()


def get_chat_storage() -> ChatStorage:
    return chat_storage_dep.get()


class LlmRequest(BaseModel):
    prompt: str = Field(min_length=1)
    mode: str = Field(default="auto", pattern="^(auto|manual)$")
    model: str | None = None
    profile: str | None = None
    system_prompt: str | None = None
    allowed_directories: list[str] | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=8192)
    think: bool | str | None = None
    web_search_provider: str | None = None
    confirm_irreversible: bool = False


class FsCreateFileRequest(BaseModel):
    path: str = Field(min_length=1)
    content: str = ""
    overwrite: bool = False


class FsCopyFileRequest(BaseModel):
    source: str = Field(min_length=1)
    destination: str = Field(min_length=1)
    overwrite: bool = False


class PersistSystemPromptRequest(BaseModel):
    system_prompt: str = ""
    persist_to_config: bool = False


class ChatCreateRequest(BaseModel):
    title: str | None = None


class ChatUpdateRequest(BaseModel):
    title: str | None = None
    active_branch_id: str | None = None
    branch_seq: int | None = None


class MessageRequest(BaseModel):
    role: str = "user"
    text: str
    model: str | None = None
    intent: str | None = None
    at: str | None = None
    checkpoint_id: str | None = None
    parent_message_id: str | None = None
    branch_id: str | None = None
    thinking: str | None = None
    think_requested: bool | None = None
    think_applied: bool | str | None = None
    think_status: str | None = None
    warnings: list[str] | None = None
    streaming: bool | None = None


class MessageUpdateRequest(BaseModel):
    text: str | None = None
    model: str | None = None
    intent: str | None = None
    thinking: str | None = None
    think_status: str | None = None
    warnings: list[str] | None = None
    streaming: bool | None = None


class ImportLegacyRequest(BaseModel):
    sessions: list[dict] = Field(default_factory=list)


class FsToolGuard:
    def __init__(self, base_dir: Path, fs_cfg: dict | None = None) -> None:
        cfg = fs_cfg or {}
        self.base_dir = base_dir.resolve()
        allowlist = cfg.get("allowlist") or ["."]
        self.allow_roots = [self._resolve_allow_root(item) for item in allowlist]
        budget_cfg = cfg.get("budget") or {}
        self.max_calls = int(budget_cfg.get("max_calls", 60))
        self.window_seconds = int(budget_cfg.get("window_seconds", 60))
        audit_rel = cfg.get("audit_log") or "logs/fs_audit.jsonl"
        self.audit_path = (self.base_dir / audit_rel).resolve()
        self._calls: dict[str, list[float]] = {}
        self._lock = threading.RLock()

    def _resolve_allow_root(self, item: str) -> Path:
        candidate = (self.base_dir / item).resolve() if not Path(item).is_absolute() else Path(item).resolve()
        try:
            candidate.relative_to(self.base_dir)
        except ValueError as e:
            raise ValueError(f"allowlist path fuori workspace: {item}") from e
        return candidate

    def is_allowed(self, path: Path) -> bool:
        p = path.resolve()
        for root in self.allow_roots:
            try:
                p.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def enforce_budget(self, client_key: str) -> None:
        now_ts = time.time()
        with self._lock:
            bucket = [ts for ts in self._calls.get(client_key, []) if (now_ts - ts) <= self.window_seconds]
            if len(bucket) >= self.max_calls:
                raise HTTPException(
                    status_code=429,
                    detail=f"FS tool budget exceeded ({self.max_calls}/{self.window_seconds}s)",
                )
            bucket.append(now_ts)
            self._calls[client_key] = bucket

    def audit(self, payload: dict) -> None:
        try:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            row = json.dumps(payload, ensure_ascii=True)
            with self.audit_path.open("a", encoding="utf-8") as f:
                f.write(row + "\n")
        except Exception:
            # audit must never break runtime flow
            return


def create_app() -> FastAPI:
    app = FastAPI(title="Local Multi-Model Router", version="1.0.0")
    # Dev hardening: allow localhost cross-port calls (e.g., chat on :8001 -> API on :8000)
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://(127\.0\.0\.1|localhost)(:\d+)?$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    base_dir = Path(__file__).resolve().parent
    config_path = os.environ.get(
        "ROUTER_CONFIG",
        str(base_dir / "config.json"),
    )
    router = LocalModelRouter(config_path)
    fs_tools_cfg = {}
    try:
        fs_tools_cfg = router._config_snapshot().get("fs_tools", {})  # local snapshot only
    except Exception:
        fs_tools_cfg = {}
    fs_guard = FsToolGuard(base_dir=base_dir, fs_cfg=fs_tools_cfg)

    def _client_key(request: Request) -> str:
        client = request.client.host if request and request.client else "local"
        return str(client)

    def _audit_fs(request: Request, op: str, status: str, detail: str, **extra) -> None:
        fs_guard.audit(
            {
                "ts": time.time(),
                "client": _client_key(request),
                "op": op,
                "status": status,
                "detail": detail,
                **extra,
            }
        )

    def _resolve_within_root(user_path: str) -> Path:
        root = base_dir.resolve()
        candidate = (root / user_path).resolve() if not Path(user_path).is_absolute() else Path(user_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as e:
            raise HTTPException(status_code=400, detail="Path fuori dal workspace consentito") from e
        if not fs_guard.is_allowed(candidate):
            raise HTTPException(status_code=403, detail="Path fuori allowlist FS tools")
        return candidate

    @app.get("/")
    def home() -> FileResponse:
        return FileResponse(base_dir / "static" / "chat.html")

    @app.get("/chat")
    def chat() -> FileResponse:
        return FileResponse(base_dir / "static" / "chat.html")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/models")
    def models() -> dict[str, list[str]]:
        try:
            return {"models": router.available_models()}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/profiles")
    def profiles() -> dict:
        try:
            return router.available_profiles()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/reload")
    def reload_config() -> dict[str, str]:
        try:
            router.reload_config()
            return {"status": "reloaded"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/llm")
    def llm(req: LlmRequest) -> dict:
        try:
            return router.generate(
                prompt=req.prompt,
                mode=req.mode,
                manual_model=req.model,
                profile=req.profile,
                system_prompt=req.system_prompt,
                allowed_directories=req.allowed_directories,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                think=req.think,
                confirm_irreversible=req.confirm_irreversible,
            )
        except RouterError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/llm/stream")
    def llm_stream(req: LlmRequest) -> StreamingResponse:
        def stream_iter():
            try:
                for event in router.generate_stream(
                    prompt=req.prompt,
                    mode=req.mode,
                    manual_model=req.model,
                    profile=req.profile,
                    system_prompt=req.system_prompt,
                    allowed_directories=req.allowed_directories,
                    temperature=req.temperature,
                    max_tokens=req.max_tokens,
                    think=req.think,
                    confirm_irreversible=req.confirm_irreversible,
                ):
                    yield json.dumps(event, ensure_ascii=True) + "\n"
            except RouterError as e:
                yield json.dumps(
                    {
                        "type": "error",
                        "error": str(e),
                        "stream_status": "failed",
                        "interruption_stage": "pre_token",
                        "errors": [str(e)],
                    },
                    ensure_ascii=True,
                ) + "\n"
            except Exception as e:
                yield json.dumps(
                    {
                        "type": "error",
                        "error": str(e),
                        "stream_status": "failed",
                        "interruption_stage": "pre_token",
                        "errors": [str(e)],
                    },
                    ensure_ascii=True,
                ) + "\n"

        return StreamingResponse(stream_iter(), media_type="application/x-ndjson")

    @app.get("/fs/list")
    def fs_list(request: Request, path: str = Query(default=".")) -> dict:
        fs_guard.enforce_budget(_client_key(request))
        try:
            target = _resolve_within_root(path)
            if not target.exists() or not target.is_dir():
                raise HTTPException(status_code=404, detail="Directory non trovata")

            entries = []
            for item in sorted(target.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                entries.append(
                    {
                        "name": item.name,
                        "path": str(item.relative_to(base_dir.resolve())),
                        "kind": "dir" if item.is_dir() else "file",
                    }
                )
            _audit_fs(request, "fs_list", "ok", "listed", path=path, count=len(entries))
            return {
                "root": str(base_dir.resolve()),
                "path": str(target.relative_to(base_dir.resolve())),
                "entries": entries,
            }
        except HTTPException as e:
            _audit_fs(request, "fs_list", "error", str(e.detail), path=path, status_code=e.status_code)
            raise

    @app.post("/fs/create-file")
    def fs_create_file(request: Request, req: FsCreateFileRequest) -> dict:
        fs_guard.enforce_budget(_client_key(request))
        try:
            target = _resolve_within_root(req.path)
            if target.exists() and not req.overwrite:
                raise HTTPException(status_code=409, detail="File esistente. Usa overwrite=true")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(req.content, encoding="utf-8")
            rel = str(target.relative_to(base_dir.resolve()))
            _audit_fs(request, "fs_create_file", "ok", "created", path=rel, overwrite=req.overwrite)
            return {"status": "created", "path": rel}
        except HTTPException as e:
            _audit_fs(
                request,
                "fs_create_file",
                "error",
                str(e.detail),
                path=req.path,
                overwrite=req.overwrite,
                status_code=e.status_code,
            )
            raise

    @app.post("/fs/copy-file")
    def fs_copy_file(request: Request, req: FsCopyFileRequest) -> dict:
        fs_guard.enforce_budget(_client_key(request))
        try:
            source = _resolve_within_root(req.source)
            destination = _resolve_within_root(req.destination)
            if not source.exists() or not source.is_file():
                raise HTTPException(status_code=404, detail="File sorgente non trovato")
            if destination.exists() and not req.overwrite:
                raise HTTPException(status_code=409, detail="Destinazione esistente. Usa overwrite=true")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            src_rel = str(source.relative_to(base_dir.resolve()))
            dst_rel = str(destination.relative_to(base_dir.resolve()))
            _audit_fs(
                request,
                "fs_copy_file",
                "ok",
                "copied",
                source=src_rel,
                destination=dst_rel,
                overwrite=req.overwrite,
            )
            return {
                "status": "copied",
                "source": src_rel,
                "destination": dst_rel,
            }
        except HTTPException as e:
            _audit_fs(
                request,
                "fs_copy_file",
                "error",
                str(e.detail),
                source=req.source,
                destination=req.destination,
                overwrite=req.overwrite,
                status_code=e.status_code,
            )
            raise

    web_search_lock = threading.Lock()
    web_search_requests: dict[str, list[float]] = {}

    def _web_search_budget(client_key: str, max_calls: int = 10, window_seconds: int = 60) -> None:
        now = time.time()
        with web_search_lock:
            bucket = [ts for ts in web_search_requests.get(client_key, []) if (now - ts) <= window_seconds]
            if len(bucket) >= max_calls:
                raise HTTPException(status_code=429, detail=f"Web search budget exceeded ({max_calls}/{window_seconds}s)")
            bucket.append(now)
            web_search_requests[client_key] = bucket

    TAVILY_API_KEY = "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"

    @app.get("/web/search")
    def web_search(
        request: Request,
        q: str = Query(..., description="Search query"),
        max_results: int = Query(5, ge=1, le=20),
        provider: str = Query("duckduckgo", description="Provider: duckduckgo or tavily")
    ) -> dict:
        client = request.client.host if request.client else "local"
        _web_search_budget(client)

        try:
            if provider == "tavily":
                tavily = TavilyClient(api_key=TAVILY_API_KEY)
                response = tavily.search(query=q, max_results=max_results)
                results = response.get("results", [])
                return {
                    "query": q,
                    "provider": "tavily",
                    "results": [
                        {
                            "title": r.get("title"),
                            "url": r.get("url"),
                            "snippet": r.get("content")
                        }
                        for r in results
                    ],
                    "count": len(results)
                }
            else:
                with DDGS() as ddgs:
                    results = list(ddgs.text(q, max_results=max_results))
                return {
                    "query": q,
                    "provider": "duckduckgo",
                    "results": [
                        {
                            "title": r.get("title"),
                            "url": r.get("href"),
                            "snippet": r.get("body")
                        }
                        for r in results
                    ],
                    "count": len(results)
                }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Web search failed: {str(e)}")

    @app.post("/settings/system-prompt")
    def persist_system_prompt(req: PersistSystemPromptRequest) -> dict:
        if not req.persist_to_config:
            return {"status": "skipped"}

        cfg_path = Path(config_path)
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            prompting = cfg.setdefault("prompting", {})
            prompting["system_prompt"] = req.system_prompt
            cfg_path.write_text(json.dumps(cfg, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
            router.reload_config()
            return {"status": "persisted", "target": "prompting.system_prompt"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Persist failed: {e}") from e

    @app.get("/api/chat")
    def chat_list(
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> dict:
        storage = get_chat_storage()
        try:
            sessions = storage.list_sessions(limit=limit, offset=offset)
            return {"sessions": sessions, "limit": limit, "offset": offset}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/api/chat")
    def chat_create(req: ChatCreateRequest | None = None) -> dict:
        storage = get_chat_storage()
        try:
            title = req.title if req and req.title else "Nuova chat"
            session = storage.create_session(title=title)
            return session
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/api/chat/{session_id}")
    def chat_read(session_id: str) -> dict:
        storage = get_chat_storage()
        try:
            session = storage.get_session(session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            return session
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.put("/api/chat/{session_id}")
    def chat_update(session_id: str, req: ChatUpdateRequest) -> dict:
        storage = get_chat_storage()
        try:
            session = storage.update_session(
                session_id,
                title=req.title,
                active_branch_id=req.active_branch_id,
                branch_seq=req.branch_seq,
            )
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            return session
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.delete("/api/chat/{session_id}")
    def chat_delete(session_id: str) -> dict:
        storage = get_chat_storage()
        try:
            storage.delete_session(session_id)
            return {"status": "deleted", "session_id": session_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/api/chat/{session_id}/messages")
    def chat_add_message(session_id: str, req: MessageRequest) -> dict:
        storage = get_chat_storage()
        try:
            existing = storage.get_session(session_id)
            if not existing:
                raise HTTPException(status_code=404, detail="Session not found")
            msg = storage.add_message(session_id, req.model_dump(exclude_none=True))
            return msg
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.put("/api/chat/{session_id}/messages/{message_id}")
    def chat_update_message(session_id: str, message_id: str, req: MessageUpdateRequest) -> dict:
        storage = get_chat_storage()
        try:
            msg = storage.update_message(message_id, req.model_dump(exclude_none=True))
            if not msg:
                raise HTTPException(status_code=404, detail="Message not found")
            return msg
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/api/chat/{session_id}/messages")
    def chat_get_messages(session_id: str) -> dict:
        storage = get_chat_storage()
        try:
            session = storage.get_session(session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            return {"session_id": session_id, "messages": session.get("messages", [])}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/api/chat/import")
    def chat_import_legacy(req: ImportLegacyRequest) -> dict:
        storage = get_chat_storage()
        try:
            result = storage.import_from_localstorage(req.sessions)
            return {"status": "imported", **result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app


app = create_app()
