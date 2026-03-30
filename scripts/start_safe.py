from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def _resolve_python_executable(root: Path) -> list[str]:
    candidates = [
        root / ".venv" / "Scripts" / "python.exe",
        root.parent / ".venv" / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return [str(candidate)]
    if sys.executable:
        return [sys.executable]
    # Last-resort fallback on Windows launcher.
    return ["py", "-3"]


def _find_repo_root(start: Path) -> Path | None:
    cur = start.resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / "server.py").exists() and (candidate / "config.json").exists():
            return candidate
    return None


def _health(base_url: str) -> tuple[bool, int | None, str | None]:
    try:
        with urlopen(f"{base_url}/health", timeout=3) as resp:
            status = int(getattr(resp, "status", 0))
            body = resp.read().decode("utf-8", errors="ignore")
        return status == 200, status, body
    except URLError as e:
        return False, None, str(e)
    except Exception as e:
        return False, None, str(e)


def _port_free(host: str, port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Start uvicorn safely with optional automatic port fallback.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--fallback-next", action="store_true", help="If requested port is busy/unhealthy, try next free port.")
    parser.add_argument("--max-port", type=int, default=8010, help="Upper bound for fallback scan.")
    parser.add_argument("--no-reload", action="store_true")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    cwd = Path.cwd()
    root = _find_repo_root(cwd)
    if not root:
        print(json.dumps({
            "status": "fail",
            "reason": "Repo root non trovato (server.py/config.json assenti).",
            "cwd": str(cwd),
        }, ensure_ascii=True, indent=2))
        return 1

    healthy, status, body_or_err = _health(base_url=base_url)
    if healthy:
        print(json.dumps({
            "status": "ok",
            "reason": f"Server gia attivo su {args.host}:{args.port}",
            "health_status": status,
            "health_body": body_or_err,
            "action": "skip_start",
            "base_url": base_url,
        }, ensure_ascii=True, indent=2))
        return 0

    selected_port = args.port
    if not _port_free(host=args.host, port=args.port):
        if args.fallback_next:
            picked = None
            for p in range(args.port + 1, args.max_port + 1):
                url = f"http://{args.host}:{p}"
                ok, st, body = _health(base_url=url)
                if ok:
                    print(json.dumps({
                        "status": "ok",
                        "reason": f"Server gia attivo su {args.host}:{p}",
                        "health_status": st,
                        "health_body": body,
                        "action": "skip_start",
                        "base_url": url,
                    }, ensure_ascii=True, indent=2))
                    return 0
                if _port_free(host=args.host, port=p):
                    picked = p
                    break
            if picked is not None:
                selected_port = picked
            else:
                print(json.dumps({
                    "status": "conflict",
                    "reason": f"Nessuna porta libera nel range {args.port+1}-{args.max_port}.",
                    "health_error": body_or_err,
                    "action": "do_not_start",
                }, ensure_ascii=True, indent=2))
                return 1
        else:
            print(json.dumps({
                "status": "conflict",
                "reason": f"Porta {args.port} occupata ma health non risponde 200.",
                "health_error": body_or_err,
                "action": "do_not_start",
            }, ensure_ascii=True, indent=2))
            return 1

    if not _port_free(host=args.host, port=selected_port):
        print(json.dumps({
            "status": "conflict",
            "reason": f"Porta {selected_port} occupata ma health non risponde 200.",
            "health_error": body_or_err,
            "action": "do_not_start",
        }, ensure_ascii=True, indent=2))
        return 1

    py_cmd = _resolve_python_executable(root)
    cmd = [
        *py_cmd,
        "-m",
        "uvicorn",
        "server:app",
        "--host",
        args.host,
        "--port",
        str(selected_port),
    ]
    if not args.no_reload:
        cmd.append("--reload")
    print(json.dumps({
        "status": "ready",
        "reason": "Porta libera e nessun server attivo, avvio uvicorn.",
        "action": "start",
        "base_url": f"http://{args.host}:{selected_port}",
        "python": " ".join(py_cmd),
        "cmd": " ".join(cmd),
    }, ensure_ascii=True, indent=2))

    # Start in foreground so logs remain visible to the operator.
    return subprocess.call(cmd, cwd=str(root))


if __name__ == "__main__":
    sys.exit(main())
