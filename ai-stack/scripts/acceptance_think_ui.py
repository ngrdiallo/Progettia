from __future__ import annotations

import sys
import threading
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from router_core import LocalModelRouter, RouteDecision


class StubClient:
    def __init__(self, nonstream_plan: list[object], stream_plan: list[object]) -> None:
        self._nonstream_plan = list(nonstream_plan)
        self._stream_plan = list(stream_plan)

    def _next_nonstream(self) -> object:
        if not self._nonstream_plan:
            return "ok"
        return self._nonstream_plan.pop(0)

    def _next_stream(self) -> object:
        if not self._stream_plan:
            return [{"response": "ok", "done": True}]
        return self._stream_plan.pop(0)

    def generate(self, **kwargs):
        step = self._next_nonstream()
        if isinstance(step, Exception):
            raise step
        return step

    def generate_stream(self, **kwargs):
        step = self._next_stream()
        if isinstance(step, Exception):
            raise step
        for item in step:
            yield item


def http_error(status_code: int, text: str) -> requests.HTTPError:
    err = requests.HTTPError(text)
    err.response = type("R", (), {"status_code": status_code})()
    return err


def make_router(nonstream_plan: list[object], stream_plan: list[object]) -> LocalModelRouter:
    router = LocalModelRouter("config.json")
    router._decide_model = lambda prompt, mode, manual_model, config=None: RouteDecision(
        mode="auto",
        intent="chat",
        selected_model="m1",
        override_from_prompt=False,
    )
    router._candidate_models = lambda d, config=None: ["m1"]
    router.available_models = lambda: ["m1"]
    router._effective_generation_controls = lambda profile_name, temperature, max_tokens, think, config=None: (
        temperature,
        max_tokens,
        think,
    )
    router._effective_stop_sequences = lambda profile_name, config=None: []
    router.client = StubClient(nonstream_plan=nonstream_plan, stream_plan=stream_plan)
    return router


def assert_eq(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected={expected!r} actual={actual!r}")


def run_nonstream_cases() -> None:
    r1 = make_router(nonstream_plan=["ok"], stream_plan=[])
    out1 = r1.generate(prompt="x", think=None)
    assert_eq(out1.get("think_status"), "unavailable", "nonstream unavailable")

    r2 = make_router(nonstream_plan=["ok"], stream_plan=[])
    out2 = r2.generate(prompt="x", think="high")
    assert_eq(out2.get("think_status"), "enabled", "nonstream enabled")

    r3 = make_router(nonstream_plan=[http_error(400, "reject think"), "ok"], stream_plan=[])
    out3 = r3.generate(prompt="x", think="high")
    assert_eq(out3.get("think_status"), "unsupported", "nonstream unsupported")

    r4 = make_router(nonstream_plan=[http_error(503, "transient"), "ok"], stream_plan=[])
    out4 = r4.generate(prompt="x", think="high")
    assert_eq(out4.get("think_status"), "downgraded", "nonstream downgraded")


def run_nonstream_400_stop_rejection_case() -> None:
    r = LocalModelRouter("config.json")
    r._decide_model = lambda prompt, mode, manual_model, config=None: RouteDecision(
        mode="auto",
        intent="chat",
        selected_model="m1",
        override_from_prompt=False,
    )
    r._candidate_models = lambda d, config=None: ["m1"]
    r.available_models = lambda: ["m1"]
    r._effective_generation_controls = lambda profile_name, temperature, max_tokens, think, config=None: (
        temperature,
        max_tokens,
        think,
    )
    r._effective_stop_sequences = lambda profile_name, config=None: ["STOP"]
    r._sleep_retry_backoff = lambda retry_index: None

    class StopRejectClient:
        def __init__(self):
            self.calls: list[tuple[bool, bool]] = []

        def generate(self, **kwargs):
            has_stop = kwargs.get("stop_sequences") is not None
            has_think = kwargs.get("think") is not None
            self.calls.append((has_stop, has_think))
            if has_stop:
                raise http_error(400, "stop unsupported")
            return "ok"

        def generate_stream(self, **kwargs):
            yield {"response": "ok", "done": True}

    client = StopRejectClient()
    r.client = client

    out = r.generate(prompt="x", think="high")
    assert_eq(out.get("response"), "ok", "nonstream 400-stop response")
    assert_eq(out.get("think_status"), "enabled", "nonstream 400-stop think_status")
    assert_eq(client.calls, [(True, True), (False, True)], "nonstream 400-stop retry sequence")
    warnings = out.get("warnings") or []
    if not any("retry without stop" in w for w in warnings):
        raise AssertionError("nonstream 400-stop missing retry without stop warning")


def run_stream_cases() -> None:
    r1 = make_router(nonstream_plan=[], stream_plan=[[{"response": "ok", "done": True}]])
    done1 = [e for e in r1.generate_stream(prompt="x", think=None) if e.get("type") == "done"][0]
    assert_eq(done1.get("think_status"), "unavailable", "stream unavailable")

    r2 = make_router(nonstream_plan=[], stream_plan=[[{"response": "ok", "done": True}]])
    done2 = [e for e in r2.generate_stream(prompt="x", think="high") if e.get("type") == "done"][0]
    assert_eq(done2.get("think_status"), "enabled", "stream enabled")

    r3 = make_router(nonstream_plan=[], stream_plan=[http_error(400, "reject think"), [{"response": "ok", "done": True}]])
    done3 = [e for e in r3.generate_stream(prompt="x", think="high") if e.get("type") == "done"][0]
    assert_eq(done3.get("think_status"), "unsupported", "stream unsupported")

    r4 = make_router(nonstream_plan=[], stream_plan=[http_error(503, "transient"), [{"response": "ok", "done": True}]])
    done4 = [e for e in r4.generate_stream(prompt="x", think="high") if e.get("type") == "done"][0]
    assert_eq(done4.get("think_status"), "downgraded", "stream downgraded")


def run_stream_meta_contract_case() -> None:
    r = make_router(nonstream_plan=[], stream_plan=[[{"response": "ok", "done": True}]])
    events = list(r.generate_stream(prompt="x", think="high", allowed_directories=["."]))
    meta = [e for e in events if e.get("type") == "meta"][0]
    done = [e for e in events if e.get("type") == "done"][0]
    nonstream = make_router(nonstream_plan=["ok"], stream_plan=[]).generate(
        prompt="x",
        think="high",
        allowed_directories=["."],
    )

    required_meta_keys = [
        "type",
        "model_used",
        "mode",
        "intent",
        "override_from_prompt",
        "fallback_used",
        "system_prompt_applied",
        "allowed_directories",
        "profile",
        "output_sanitized",
        "think",
        "think_requested",
        "think_applied",
        "think_status",
        "warnings",
        "errors",
    ]
    for key in required_meta_keys:
        if key not in meta:
            raise AssertionError(f"stream meta missing key: {key}")

    if meta.get("type") != "meta":
        raise AssertionError("stream meta type mismatch")
    if meta.get("errors") != []:
        raise AssertionError("stream meta errors must be empty on start")

    coherent_keys = [
        "model_used",
        "mode",
        "intent",
        "override_from_prompt",
        "fallback_used",
        "system_prompt_applied",
        "allowed_directories",
        "profile",
        "think_requested",
        "think_applied",
        "think_status",
    ]
    for key in coherent_keys:
        assert_eq(meta.get(key), done.get(key), f"stream meta/done coherence: {key}")
        assert_eq(meta.get(key), nonstream.get(key), f"stream meta/nonstream coherence: {key}")


def run_stream_interruption_semantics_case() -> None:
    # Post-token interruption must end with deterministic done payload, not generic error event.
    r = LocalModelRouter("config.json")
    r._decide_model = lambda prompt, mode, manual_model, config=None: RouteDecision(
        mode="auto",
        intent="chat",
        selected_model="m1",
        override_from_prompt=False,
    )
    r._candidate_models = lambda d, config=None: ["m1"]
    r.available_models = lambda: ["m1"]
    r._effective_generation_controls = lambda profile_name, temperature, max_tokens, think, config=None: (
        temperature,
        max_tokens,
        think,
    )
    r._effective_stop_sequences = lambda profile_name, config=None: []

    class PartialFailClient:
        def generate(self, **kwargs):
            return "unused"

        def generate_stream(self, **kwargs):
            yield {"response": "hello ", "done": False}
            raise requests.ConnectionError("socket closed")

    r.client = PartialFailClient()
    events = list(r.generate_stream(prompt="x", think="high"))
    done = [e for e in events if e.get("type") == "done"][0]

    assert_eq(done.get("stream_status"), "partial_interrupted", "post-token stream_status")
    assert_eq(done.get("interruption_stage"), "post_token", "post-token interruption_stage")
    assert_eq(done.get("response"), "hello ", "post-token partial response kept")
    if not done.get("errors"):
        raise AssertionError("post-token interruption should include technical error")

    # Pre-token interruption must fail explicitly with a deterministic marker.
    r2 = make_router(nonstream_plan=[], stream_plan=[requests.ConnectionError("connect failed")])
    try:
        list(r2.generate_stream(prompt="x", think=None))
        raise AssertionError("expected pre-token stream failure")
    except Exception as e:
        msg = str(e)
        if "stream_failed_pre_token:" not in msg:
            raise AssertionError(f"pre-token failure marker missing: {msg}")


def run_stream_retry_matrix_case() -> None:
    r = LocalModelRouter("config.json")
    r._decide_model = lambda prompt, mode, manual_model, config=None: RouteDecision(
        mode="auto",
        intent="chat",
        selected_model="m1",
        override_from_prompt=False,
    )
    r._candidate_models = lambda d, config=None: ["m1"]
    r.available_models = lambda: ["m1"]
    r._effective_generation_controls = lambda profile_name, temperature, max_tokens, think, config=None: (
        temperature,
        max_tokens,
        think,
    )
    r._effective_stop_sequences = lambda profile_name, config=None: ["STOP"]
    r._sleep_retry_backoff = lambda retry_index: None

    class MatrixClient:
        def __init__(self):
            self.plan = [
                ("high", ["STOP"], http_error(503, "transient-stop")),
                ("high", None, http_error(503, "transient-think")),
                (None, ["STOP"], [{"response": "ok", "done": True}]),
            ]

        def generate(self, **kwargs):
            return "unused"

        def generate_stream(self, **kwargs):
            if not self.plan:
                raise AssertionError("unexpected extra stream call")
            expected_think, expected_stop, step = self.plan.pop(0)
            assert_eq(kwargs.get("think"), expected_think, "retry-matrix think sequence")
            assert_eq(kwargs.get("stop_sequences"), expected_stop, "retry-matrix stop sequence")
            if isinstance(step, Exception):
                raise step
            for item in step:
                yield item

    client = MatrixClient()
    r.client = client

    events = list(r.generate_stream(prompt="x", think="high"))
    done = [e for e in events if e.get("type") == "done"][0]

    if client.plan:
        raise AssertionError("retry matrix plan not fully consumed")
    assert_eq(done.get("response"), "ok", "retry-matrix final response")
    assert_eq(done.get("think_applied"), None, "retry-matrix downgraded think_applied")
    assert_eq(done.get("think_status"), "downgraded", "retry-matrix think_status")

    warnings = done.get("warnings") or []
    expected_warnings = [
        "m1: transient 503 with stop sequences, retry without stop",
        "m1: transient 503, retry without think",
        "m1: retry without think after rejection",
    ]
    for expected in expected_warnings:
        if expected not in warnings:
            raise AssertionError(f"retry-matrix missing warning: {expected}")


def run_stream_malformed_chunk_case() -> None:
    r = make_router(
        nonstream_plan=[],
        stream_plan=[[
            "junk-line",
            {"unexpected": True},
            {"response": "ok", "done": True},
        ]],
    )
    events = list(r.generate_stream(prompt="x", think=None))
    done = [e for e in events if e.get("type") == "done"][0]
    assert_eq(done.get("response"), "ok", "malformed-chunk final response")
    assert_eq(done.get("errors"), [], "malformed-chunk errors")


def run_stream_retry_timeout_variant_case() -> None:
    # ReadTimeout before first token should trigger retry path and downgrade think.
    r = make_router(
        nonstream_plan=[],
        stream_plan=[requests.ReadTimeout("read timeout"), [{"response": "ok", "done": True}]],
    )
    done = [e for e in r.generate_stream(prompt="x", think="high") if e.get("type") == "done"][0]
    assert_eq(done.get("response"), "ok", "timeout-variant response")
    assert_eq(done.get("think_status"), "downgraded", "timeout-variant think_status")
    warnings = done.get("warnings") or []
    if not any("retry without think" in w for w in warnings):
        raise AssertionError("timeout-variant missing retry warning")


def run_fallback_nonstream_case() -> None:
    # Explicit fallback: first model fails, second succeeds.
    r = LocalModelRouter("config.json")
    r._decide_model = lambda prompt, mode, manual_model, config=None: RouteDecision(
        mode="auto",
        intent="chat",
        selected_model="m1",
        override_from_prompt=False,
    )
    r._candidate_models = lambda d, config=None: ["m1", "m2"]
    r.available_models = lambda: ["m1", "m2"]
    r._effective_generation_controls = lambda profile_name, temperature, max_tokens, think, config=None: (
        temperature,
        max_tokens,
        think,
    )
    r._effective_stop_sequences = lambda profile_name, config=None: []
    r._sleep_retry_backoff = lambda retry_index: None

    class FallbackClient:
        def generate(self, **kwargs):
            if kwargs.get("model") == "m1":
                raise requests.ConnectionError("m1 down")
            return "ok"

        def generate_stream(self, **kwargs):
            yield {"response": "ok", "done": True}

    r.client = FallbackClient()
    out = r.generate(prompt="x", think=None)
    assert_eq(out.get("model_used"), "m2", "fallback model used")
    assert_eq(out.get("fallback_used"), True, "fallback_used flag")


def run_guardrail_contract_cases() -> None:
    r1 = make_router(nonstream_plan=["unused"], stream_plan=[])
    r1._requires_irreversible_confirmation = lambda prompt, config=None: True
    out = r1.generate(prompt="danger", think="high", confirm_irreversible=False)
    for key in ["think_requested", "think_applied", "think_status", "warnings", "errors"]:
        if key not in out:
            raise AssertionError(f"nonstream guardrail missing key: {key}")
    assert_eq(out.get("think_status"), "unavailable", "nonstream guardrail think_status")

    r2 = make_router(nonstream_plan=[], stream_plan=[])
    r2._requires_irreversible_confirmation = lambda prompt, config=None: True
    done = [e for e in r2.generate_stream(prompt="danger", think="high", confirm_irreversible=False) if e.get("type") == "done"][0]
    for key in ["think_requested", "think_applied", "think_status", "warnings", "errors"]:
        if key not in done:
            raise AssertionError(f"stream guardrail missing key: {key}")
    assert_eq(done.get("think_status"), "unavailable", "stream guardrail think_status")

    # Regression: guardrail must honor profile-derived think defaults for think_requested.
    r3 = make_router(nonstream_plan=["unused"], stream_plan=[])
    r3._requires_irreversible_confirmation = lambda prompt, config=None: True
    r3._effective_generation_controls = LocalModelRouter._effective_generation_controls.__get__(
        r3, LocalModelRouter
    )
    r3._profile_config = lambda profile_name, config=None: {"think": "high"}
    out_profile = r3.generate(prompt="danger", think=None, confirm_irreversible=False)
    assert_eq(out_profile.get("think_requested"), True, "nonstream guardrail profile think_requested")

    r4 = make_router(nonstream_plan=[], stream_plan=[])
    r4._requires_irreversible_confirmation = lambda prompt, config=None: True
    r4._effective_generation_controls = LocalModelRouter._effective_generation_controls.__get__(
        r4, LocalModelRouter
    )
    r4._profile_config = lambda profile_name, config=None: {"think": "high"}
    done_profile = [
        e
        for e in r4.generate_stream(prompt="danger", think=None, confirm_irreversible=False)
        if e.get("type") == "done"
    ][0]
    assert_eq(done_profile.get("think_requested"), True, "stream guardrail profile think_requested")


def run_reload_concurrency_snapshot_case() -> None:
    r = LocalModelRouter("config.json")

    cfg_a = {
        "ollama": {
            "base_url": "http://127.0.0.1:11434",
            "timeout_seconds": 30,
            "stream_read_timeout_seconds": 900,
        },
        "intent_keywords": {"coding": [], "rag": [], "reasoning": []},
        "routing": {"short_prompt_chars": 5},
        "models": {"chat": "m1"},
        "fallback_order": ["m1"],
        "default_profile": "safe",
        "default_model": "m1",
        "intent_to_model": {"chat": "m1"},
        "profiles": {
            "safe": {
                "temperature": 0.1,
                "max_tokens": 64,
            }
        },
    }

    cfg_b = {
        "ollama": {
            "base_url": "http://127.0.0.1:11434",
            "timeout_seconds": 30,
            "stream_read_timeout_seconds": 900,
        },
        "intent_keywords": {"coding": [], "rag": [], "reasoning": []},
        "routing": {"short_prompt_chars": 5},
        "models": {"chat": "m2"},
        "fallback_order": ["m2"],
        "default_profile": "strict",
        "default_model": "m2",
        "intent_to_model": {"chat": "m2"},
        "profiles": {
            "strict": {
                "temperature": 0.2,
                "max_tokens": 64,
                "think": "high",
            }
        },
    }

    # Deterministic local model availability for both config variants.
    r.available_models = lambda: ["m1", "m2"]

    class ClientOk:
        def generate(self, **kwargs):
            return "ok"

        def generate_stream(self, **kwargs):
            yield {"response": "ok", "done": True}

    r.client = ClientOk()

    state = {"i": 0}

    def fake_load_config(_path):
        state["i"] += 1
        return cfg_a if state["i"] % 2 == 0 else cfg_b

    r._load_config = fake_load_config
    r._config = cfg_a

    outputs: list[tuple[str, str, bool]] = []
    lock = threading.Lock()
    errors: list[str] = []

    def reloader():
        try:
            for _ in range(200):
                r.reload_config()
        except Exception as e:
            errors.append(f"reload_error: {e}")

    def generator():
        try:
            for _ in range(200):
                out = r.generate(prompt="hello world", think=None)
                triple = (out.get("model_used"), out.get("profile"), bool(out.get("think_requested")))
                with lock:
                    outputs.append(triple)
        except Exception as e:
            errors.append(f"generate_error: {e}")

    t1 = threading.Thread(target=reloader)
    t2 = threading.Thread(target=generator)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    if errors:
        raise AssertionError(" | ".join(errors))

    allowed = {
        ("m1", "safe", False),
        ("m2", "strict", True),
    }
    for triple in outputs:
        if triple not in allowed:
            raise AssertionError(f"inconsistent snapshot triple: {triple}")


def run_profile_isolation_case() -> None:
    r = LocalModelRouter("config.json")
    cfg = {
        "ollama": {"base_url": "http://127.0.0.1:11434", "timeout_seconds": 30},
        "intent_keywords": {"coding": [], "rag": [], "reasoning": []},
        "routing": {"short_prompt_chars": 5},
        "models": {"chat": "test"},
        "fallback_order": ["test"],
        "default_profile": "safe",
        "default_model": "test",
        "intent_to_model": {"chat": "test"},
        "profiles": {
            "safe": {
                "system_prompt": "Sei sicuro. Rispondi con cautela.",
                "temperature": 0.1,
                "max_tokens": 64,
            },
            "analysis": {
                "system_prompt": "Sei analitico. Esplicita ragionamento.",
                "temperature": 0.2,
                "max_tokens": 128,
            },
        },
        "prompting": {"system_prompt": "Base prompt."},
    }
    r._config = cfg

    class ClientCapture:
        def __init__(self):
            self.last_system_prompt = None

        def generate(self, **kwargs):
            self.last_system_prompt = kwargs.get("system_prompt", "")
            return "ok"

        def generate_stream(self, **kwargs):
            self.last_system_prompt = kwargs.get("system_prompt", "")
            yield {"response": "ok", "done": True}

    client = ClientCapture()
    r.client = client

    out_safe = r.generate(prompt="test", profile="safe")
    assert out_safe.get("profile") == "safe", f"expected profile safe, got {out_safe.get('profile')}"
    assert "Sicuro" in client.last_system_prompt or "cautela" in client.last_system_prompt.lower(), \
        f"safe profile should use safe system prompt, got: {client.last_system_prompt}"

    out_analysis = r.generate(prompt="test", profile="analysis")
    assert out_analysis.get("profile") == "analysis", f"expected profile analysis, got {out_analysis.get('profile')}"
    assert "analitico" in client.last_system_prompt.lower() or "ragionamento" in client.last_system_prompt.lower(), \
        f"analysis profile should use analytical system prompt, got: {client.last_system_prompt}"

    assert client.last_system_prompt != r._config["profiles"]["safe"]["system_prompt"], \
        "profiles must produce different system prompts"


def main() -> None:
    run_nonstream_cases()
    run_nonstream_400_stop_rejection_case()
    run_stream_cases()
    run_stream_meta_contract_case()
    run_stream_interruption_semantics_case()
    run_stream_retry_matrix_case()
    run_stream_malformed_chunk_case()
    run_stream_retry_timeout_variant_case()
    run_fallback_nonstream_case()
    run_guardrail_contract_cases()
    run_reload_concurrency_snapshot_case()
    run_profile_isolation_case()
    print("ACCEPTANCE_THINK_UI_OK")


if __name__ == "__main__":
    main()
