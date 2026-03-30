from __future__ import annotations

import json
import socket
import errno
import sys
import argparse
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
from urllib.parse import urlparse


def _find_repo_root(start: Path) -> Path | None:
    cur = start.resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / "server.py").exists() and (candidate / "config.json").exists():
            return candidate
    return None


def _health(base_url: str = "http://127.0.0.1:8000") -> tuple[bool, int | None, str | None]:
    try:
        with urlopen(f"{base_url}/health", timeout=3) as resp:
            status = int(getattr(resp, "status", 0))
            body = resp.read().decode("utf-8", errors="ignore")
        return status == 200, status, body
    except URLError as e:
        return False, None, str(e)
    except Exception as e:
        return False, None, str(e)


def _port_bind_probe(host: str = "127.0.0.1", port: int = 8000) -> tuple[bool, int | None, str | None]:
    """Try a temporary bind to detect whether the TCP port is available.

    Returns:
        (is_free, errno_code, detail)
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        return True, None, None
    except OSError as e:
        return False, int(getattr(e, "errno", 0) or 0), str(e)
    finally:
        sock.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight runtime check for a target router base URL.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    args = parser.parse_args()
    parsed = urlparse(args.base_url)
    host = parsed.hostname or "127.0.0.1"
    port = int(parsed.port or (443 if parsed.scheme == "https" else 80))

    cwd = Path.cwd()
    root = _find_repo_root(cwd)

    result: dict[str, object] = {
        "cwd": str(cwd),
        "repo_root_found": str(root) if root else None,
        "inside_repo": root is not None,
        "recommended": [],
    }

    if not root:
        result["status"] = "fail"
        result["reason"] = "Repo root non trovato (server.py/config.json assenti nel path corrente e parent)."
        result["recommended"] = [
            "Vai nella cartella ai-stack e rilancia questo script.",
        ]
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 1

    server_ok, status, details = _health(base_url=args.base_url.rstrip("/"))
    result["health_ok"] = server_ok
    result["health_status"] = status
    result["base_url"] = args.base_url.rstrip("/")

    if server_ok:
        result["status"] = "ok"
        result["reason"] = f"Server gia attivo su {host}:{port}"
        result["recommended"] = [
            f"Non avviare una seconda istanza su porta {port}.",
            f"Esegui smoke: .venv\\Scripts\\python.exe scripts\\smoke_runtime.py --base-url {args.base_url.rstrip('/')}",
            "Oppure usa una porta alternativa se vuoi una seconda istanza.",
        ]
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0

    port_free, port_errno, port_detail = _port_bind_probe(host=host, port=port)
    result["port_free"] = port_free
    if port_errno is not None:
        result["port_probe_errno"] = port_errno
        result["port_probe_error"] = port_detail

    if not port_free:
        is_addr_in_use = port_errno in {errno.EADDRINUSE, 10048}
        result["status"] = "conflict"
        result["reason"] = (
            f"Porta {port} occupata da altro processo/servizio"
            if is_addr_in_use
            else f"Porta {port} non disponibile"
        )
        result["health_error"] = details
        result["recommended"] = [
            f"Verifica il processo in ascolto: Get-NetTCPConnection -LocalPort {port} -State Listen | Select-Object LocalAddress,LocalPort,OwningProcess",
            "Mostra dettagli processo: Get-Process -Id <PID>",
            "Se non vuoi fermarlo, avvia il router su porta alternativa.",
        ]
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 1

    result["status"] = "ready"
    result["reason"] = f"Nessun server risponde su {host}:{port}"
    result["health_error"] = details
    result["recommended"] = [
        f"cd {root.name}",
        f".venv\\Scripts\\python.exe -m uvicorn server:app --host {host} --port {port} --reload",
    ]
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
