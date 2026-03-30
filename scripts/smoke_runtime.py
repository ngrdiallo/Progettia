from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def _get_json(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        payload = resp.read().decode("utf-8")
    return json.loads(payload)


def _get_status(url: str, timeout: float) -> int:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return int(getattr(resp, "status", 0))


def _pick_model(models: list[str]) -> str:
    preferred = [
        "qwen3:latest",
        "qwen2.5:7b",
        "deepseek-r1:8b",
        "deepseek-coder:6.7b",
    ]
    for name in preferred:
        if name in models:
            return name
    if not models:
        raise RuntimeError("Nessun modello locale disponibile da /models")
    return models[0]


def _stream_smoke(base_url: str, model: str, timeout: float) -> dict:
    body = {
        "prompt": "Rispondi con una frase breve: smoke test stream.",
        "mode": "manual",
        "model": model,
        "think": True,
    }
    req = urllib.request.Request(
        f"{base_url}/llm/stream",
        data=json.dumps(body, ensure_ascii=True).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    counts: dict[str, int] = {}
    done_event: dict | None = None
    lines = 0

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            evt_type = str(evt.get("type", "?"))
            counts[evt_type] = counts.get(evt_type, 0) + 1
            lines += 1
            if evt_type == "done":
                done_event = evt
                break

    if done_event is None:
        raise RuntimeError("Stream non ha emesso evento done")

    if "response" not in done_event:
        raise RuntimeError("Evento done senza campo response")

    if "thinking" not in done_event:
        raise RuntimeError("Evento done senza campo thinking")

    errors = done_event.get("errors") or []
    return {
        "lines": lines,
        "counts": counts,
        "done_model": done_event.get("model_used"),
        "done_response_len": len((done_event.get("response") or "")),
        "done_thinking_len": len((done_event.get("thinking") or "")),
        "done_errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Runtime smoke test for Local Multi-Model Router")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Base URL del router")
    parser.add_argument("--timeout", type=float, default=120.0, help="Timeout secondi")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")

    try:
        health_status = _get_status(f"{base}/health", timeout=args.timeout)
        chat_status = _get_status(f"{base}/chat", timeout=args.timeout)
        models_payload = _get_json(f"{base}/models", timeout=args.timeout)
        models = models_payload.get("models") or []
        model = _pick_model(list(models))
        stream_result = _stream_smoke(base, model, timeout=args.timeout)
    except urllib.error.URLError as e:
        print(f"SMOKE_FAIL network={e}")
        return 1
    except Exception as e:
        print(f"SMOKE_FAIL error={e}")
        return 1

    if health_status != 200:
        print(f"SMOKE_FAIL health_status={health_status}")
        return 1

    if chat_status != 200:
        print(f"SMOKE_FAIL chat_status={chat_status}")
        return 1

    print(f"SMOKE_OK health={health_status} chat={chat_status} model={model}")
    print(f"SMOKE_STREAM lines={stream_result['lines']} counts={stream_result['counts']}")
    print(
        "SMOKE_DONE "
        f"model={stream_result['done_model']} "
        f"response_len={stream_result['done_response_len']} "
        f"thinking_len={stream_result['done_thinking_len']} "
        f"errors={stream_result['done_errors']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
