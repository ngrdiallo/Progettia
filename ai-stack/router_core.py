
from __future__ import annotations


class RouterError(Exception):
    pass

class AgentHandoffError(RouterError):
    pass

class AgentHandoffTrace:
    def __init__(self):
        self.traces = []
    def add(self, from_agent, to_agent, task_id, status, duration_ms, reason):
        self.traces.append({
            'timestamp': time.time(),
            'from': from_agent,
            'to': to_agent,
            'task_id': task_id,
            'status': status,
            'duration_ms': duration_ms,
            'reason': reason,
        })
    def get(self):
        return self.traces.copy()

import json
import re
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


import requests


@dataclass
class RouteDecision:
    mode: str
    intent: str
    selected_model: str
    override_from_prompt: bool


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: int,
        stream_read_timeout_seconds: int = 900,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.stream_read_timeout_seconds = stream_read_timeout_seconds

    def list_local_models(self) -> list[str]:
        url = f"{self.base_url}/api/tags"
        response = requests.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        models = payload.get("models", [])
        return [m.get("name", "") for m in models if m.get("name")]

    def generate(
        self,
        model: str,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        think: bool | str | None = None,
        stop_sequences: list[str] | None = None,
    ) -> str:
        url = f"{self.base_url}/api/generate"
        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if stop_sequences is not None:
            options["stop"] = stop_sequences
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if think is not None:
            payload["think"] = think
        if options:
            payload["options"] = options
        response = requests.post(url, json=payload, timeout=self.timeout_seconds)
        response.raise_for_status()
        data = response.json()
        if "response" not in data:
            raise RouterError("Ollama response field missing")
        return data["response"]

    def generate_stream(
        self,
        model: str,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        think: bool | str | None = None,
        stop_sequences: list[str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        url = f"{self.base_url}/api/generate"
        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if stop_sequences is not None:
            options["stop"] = stop_sequences
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if think is not None:
            payload["think"] = think
        if options:
            payload["options"] = options
        response = requests.post(
            url,
            json=payload,
            timeout=(self.timeout_seconds, self.stream_read_timeout_seconds),
        )
        response.raise_for_status()
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            try:
                chunk = json.loads(raw_line)
                if isinstance(chunk, dict):
                    yield chunk
            except json.JSONDecodeError:
                continue


class LocalModelRouter:
    MODEL_TAG_REGEX = re.compile(r"^\s*#\s*model\s*:\s*([\w./:-]+)\s*$", re.IGNORECASE)
    FALLBACK_PROFILE = "safe"
    RETRY_BASE_SECONDS = 0.25
    RETRY_MAX_SECONDS = 2.0
    RETRY_JITTER_RATIO = 0.25

    HANDOFF_SCHEMA = {'task', 'context_min', 'expected_output', 'from_agent', 'to_agent'}
    HANDOFF_MAX_FIELD_LEN = 128
    HANDOFF_BUDGET_DEFAULT = 3
    HANDOFF_TIMEOUT_MS_DEFAULT = 2000
    HANDOFF_ESCALATION_THRESHOLD = 2

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self._config_lock = threading.RLock()
        self._config = self._load_config(self.config_path)
        ollama_cfg = self._config["ollama"]
        self.client = OllamaClient(
            base_url=ollama_cfg["base_url"],
            timeout_seconds=int(ollama_cfg["timeout_seconds"]),
            stream_read_timeout_seconds=int(ollama_cfg.get("stream_read_timeout_seconds", 900)),
        )
        # Handoff state (defaults from config or fallback)
        routing = self._config.get('routing', {})
        self._handoff_budget = routing.get('max_handoffs_per_turn', self.HANDOFF_BUDGET_DEFAULT)
        self._handoff_timeout = routing.get('handoff_timeout_ms', self.HANDOFF_TIMEOUT_MS_DEFAULT)
        self._escalation_threshold = routing.get('escalation_threshold', self.HANDOFF_ESCALATION_THRESHOLD)
        self._handoff_count = 0
        self._handoff_start_time = None
        self._handoff_pairs = []
        self._agent_trace = AgentHandoffTrace()

    def _validate_handoff_schema(self, payload: dict):
        missing = self.HANDOFF_SCHEMA - set(payload)
        if missing:
            raise AgentHandoffError(f"Missing handoff fields: {missing}")
        for k in self.HANDOFF_SCHEMA:
            v = payload[k]
            if k == 'context_min':
                if not (isinstance(v, dict) or isinstance(v, str)):
                    raise AgentHandoffError(f"Invalid type for field: {k}")
                # Optionally, check dict length or content if needed
            else:
                if not isinstance(v, str) or len(v) > self.HANDOFF_MAX_FIELD_LEN:
                    raise AgentHandoffError(f"Invalid or too long field: {k}")

    def _enforce_handoff_budget(self):
        if self._handoff_count >= self._handoff_budget:
            raise AgentHandoffError("Handoff budget exceeded")

    def _enforce_handoff_timeout(self):
        if self._handoff_start_time is None:
            self._handoff_start_time = time.time()
        timeout = self._handoff_timeout / 1000.0
        if (time.time() - self._handoff_start_time) > timeout:
            raise AgentHandoffError("Handoff timeout exceeded")

    def _enforce_escalation_guard(self, from_agent, to_agent):
        count = self._handoff_pairs.count((from_agent, to_agent))
        if count >= self._escalation_threshold:
            raise AgentHandoffError(f"Escalation blocked: {from_agent}->{to_agent} repeated {count} times")

    def agent_handoff(self, payload: dict, config=None):
        config = config or self._config_snapshot()
        self._validate_handoff_schema(payload)
        self._handoff_count += 1
        if self._handoff_start_time is None:
            self._handoff_start_time = time.time()
        self._handoff_pairs.append((payload['from_agent'], payload['to_agent']))
        t0 = time.time()
        self._enforce_handoff_budget()
        self._enforce_handoff_timeout()
        self._enforce_escalation_guard(payload['from_agent'], payload['to_agent'])
        status = 'ok'
        reason = ''
        try:
            result = {'output': f"Handled by {payload['to_agent']}"}
        except Exception as e:
            status = 'error'
            reason = str(e)
            result = {'error': reason}
        duration_ms = int((time.time() - t0) * 1000)
        self._agent_trace.add(payload['from_agent'], payload['to_agent'], payload['task'], status, duration_ms, reason)
        result['agent_trace'] = self._agent_trace.get() if self._agent_trace.get() else []
        return result

    def _config_snapshot(self) -> dict[str, Any]:
        with self._config_lock:
            return self._config

    @staticmethod
    def _load_config(config_path: Path) -> dict[str, Any]:
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def reload_config(self) -> None:
        new_config = self._load_config(self.config_path)
        with self._config_lock:
            self._config = new_config

    def available_models(self) -> list[str]:
        return self.client.list_local_models()

    def _prompt_override_model(self, prompt: str) -> str | None:
        first_line = prompt.splitlines()[0] if prompt else ""
        match = self.MODEL_TAG_REGEX.match(first_line)
        return match.group(1).strip() if match else None

    def _infer_intent(
        self, prompt: str, config: dict[str, Any] | None = None
    ) -> str:
        config = config or self._config_snapshot()
        prompt_lower = prompt.lower()
        keywords = config["intent_keywords"]
        for word in keywords.get("coding", []):
            if word in prompt_lower:
                return "coding"
        for word in keywords.get("rag", []):
            if word in prompt_lower:
                return "rag"
        for word in keywords.get("reasoning", []):
            if word in prompt_lower:
                return "reasoning"
        return "chat"

    def _decide_model(
        self,
        prompt: str,
        mode: str,
        manual_model: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> RouteDecision:
        config = config or self._config_snapshot()
        intent = self._infer_intent(prompt, config=config)
        override_from_prompt = False
        if mode == "manual":
            selected_model = manual_model or config.get("default_model")
            return RouteDecision(
                mode="manual", intent=intent, selected_model=selected_model, override_from_prompt=False
            )
        prompt_model = self._prompt_override_model(prompt)
        if prompt_model:
            override_from_prompt = True
            selected_model = prompt_model
        else:
            intent_to_model = config.get("intent_to_model", {})
            selected_model = intent_to_model.get(intent) or config.get("default_model")
        return RouteDecision(
            mode=mode, intent=intent, selected_model=selected_model, override_from_prompt=override_from_prompt
        )

    def _fallback_chain(
        self, primary: str, config: dict[str, Any] | None = None
    ) -> list[str]:
        config = config or self._config_snapshot()
        chain = [primary]
        for model in config.get("fallback_order", []):
            if model not in chain:
                chain.append(model)
        return chain

    def _candidate_models(
        self,
        decision: RouteDecision,
        config: dict[str, Any] | None = None,
    ) -> list[str]:
        if decision.mode == "manual":
            return [decision.selected_model]
        return self._fallback_chain(decision.selected_model, config=config)

    @staticmethod
    def _normalize_directories(directories: list[str] | None) -> list[str]:
        if not directories:
            return []
        clean = []
        for d in directories:
            if not d:
                continue
            p = str(Path(d).expanduser().resolve())
            if p not in clean:
                clean.append(p)
        return clean

    def _build_system_prompt(
        self,
        profile_name: str,
        system_prompt: str | None = None,
        allowed_directories: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> str | None:
        config = config or self._config_snapshot()
        prompting_cfg = config.get("prompting", {})
        profile_cfg = self._profile_config(profile_name, config=config)
        base_system = (prompting_cfg.get("system_prompt") or "").strip()
        profile_system = (profile_cfg.get("system_prompt") or "").strip()
        explicit = (system_prompt or "").strip()
        parts: list[str] = []
        if base_system:
            parts.append(base_system)
        if profile_system:
            parts.append(profile_system)
        if explicit:
            parts.append(explicit)
        dirs = self._normalize_directories(allowed_directories)
        enforce_scope = bool(prompting_cfg.get("enforce_directory_scope", False))
        if dirs and enforce_scope:
            parts.append(
                "Vincolo directory: limita analisi e suggerimenti ai seguenti percorsi: "
                + ", ".join(dirs)
                + ". Se la richiesta esce da questo perimetro, dichiaralo chiaramente."
            )
        if not parts:
            return None
        return "\n\n".join(parts)

    def _profile_config(self, profile_name: str | None = None, config: dict[str, Any] | None = None) -> dict[str, Any]:
        config = config or self._config_snapshot()
        profiles = config.get("profiles", {})
        default_profile = config.get("default_profile", self.FALLBACK_PROFILE)
        chosen = (profile_name or default_profile or self.FALLBACK_PROFILE).strip()
        if chosen in profiles:
            return profiles[chosen]
        if default_profile in profiles:
            return profiles[default_profile]
        return {}

    def _resolve_profile_name(self, profile_name: str | None = None, config: dict[str, Any] | None = None) -> str:
        config = config or self._config_snapshot()
        profiles = config.get("profiles", {})
        default_profile = config.get("default_profile", self.FALLBACK_PROFILE)
        chosen = (profile_name or default_profile or self.FALLBACK_PROFILE).strip()
        if chosen in profiles:
            return chosen
        if default_profile in profiles:
            return default_profile
        return self.FALLBACK_PROFILE

    def _effective_generation_controls(
        self,
        profile_name: str,
        temperature: float | None,
        max_tokens: int | None,
        think: bool | str | None,
        config: dict[str, Any] | None = None,
    ) -> tuple[float | None, int | None, bool | str | None]:
        profile_cfg = self._profile_config(profile_name, config=config)
        if temperature is None:
            temperature = profile_cfg.get("temperature")
        if max_tokens is None:
            max_tokens = profile_cfg.get("max_tokens")
        if think is None:
            think = profile_cfg.get("think")
        return temperature, max_tokens, think

    def _effective_stop_sequences(self, profile_name: str, config: dict[str, Any] | None = None) -> list[str]:
        profile_cfg = self._profile_config(profile_name, config=config)
        configured = profile_cfg.get("stop_sequences") or []
        clean: list[str] = []
        for item in configured:
            if not item:
                continue
            token = str(item).strip()
            if token and token not in clean:
                clean.append(token)
        return clean

    @staticmethod
    def _is_retryable_status(status_code: int | None) -> bool:
        if status_code is None:
            return False
        if status_code < 0:
            return True
        try:
            return status_code in {408, 429} or status_code >= 500
        except TypeError:
            return False

    def _sleep_retry_backoff(self, retry_index: int) -> None:
        base = self.RETRY_BASE_SECONDS
        max_sleep = self.RETRY_MAX_SECONDS
        jitter_ratio = self.RETRY_JITTER_RATIO
        delay = min(base * (2 ** retry_index), max_sleep)
        jitter = delay * jitter_ratio * random.random()
        time.sleep(delay + jitter)

    @staticmethod
    def _compute_think_status(
        think_requested: bool, think_applied: bool | str | None, think_unsupported: bool
    ) -> str:
        if not think_requested:
            return "unavailable"
        if think_unsupported:
            return "unsupported"
        if think_applied is False or (think_applied is None and think_requested):
            return "downgraded"
        if think_applied:
            return "enabled"
        return "unavailable"

    def _requires_irreversible_confirmation(self, prompt: str, config: dict[str, Any] | None = None) -> bool:
        config = config or self._config_snapshot()
        keywords = config.get("irreversible_keywords", [])
        prompt_lower = prompt.lower()
        return any(word in prompt_lower for word in keywords)

    def _sanitize_output(self, text: str, config: dict[str, Any] | None = None) -> tuple[str, bool]:
        config = config or self._config_snapshot()
        output_sanitized = False
        if config.get("strip_think_tags", True):
            import re
            text = re.sub(r"</?think[^>]*>", "", text, flags=re.IGNORECASE)
            output_sanitized = True
        return text, output_sanitized

    def _build_stream_meta_event(
        self,
        decision: RouteDecision,
        model: str,
        effective_profile: str,
        effective_system_prompt: str | None,
        normalized_dirs: list[str],
        think_requested: bool,
        think_applied: bool | str | None,
        think_unsupported: bool,
        fallback_used: bool,
        warnings: list[str],
    ) -> dict[str, Any]:
        # Contract fields: think_requested, think_applied, think_status, warnings, errors, response (None for meta)
        return {
            "type": "meta",
            "model_used": model,
            "mode": decision.mode,
            "intent": decision.intent,
            "override_from_prompt": decision.override_from_prompt,
            "fallback_used": fallback_used,
            "system_prompt_applied": effective_system_prompt is not None,
            "allowed_directories": normalized_dirs,
            "profile": effective_profile,
            "output_sanitized": False,
            "think": think_applied,
            "think_requested": think_requested if think_requested is not None else False,
            "think_applied": think_applied if think_applied is not None else None,
            "think_status": self._compute_think_status(think_requested, think_applied, think_unsupported) if think_requested is not None else "unavailable",
            "warnings": warnings if warnings is not None else [],
            "errors": [],
            "response": None,
        }

    @staticmethod
    def _stream_think_attempts(think: bool | str | None) -> list[bool | str | None]:
        if think is True:
            return [True, False]
        if think is False:
            return [False]
        if isinstance(think, str):
            return [think, True, False]
        return [None]

    @staticmethod
    def _stream_stop_variants(stop_sequences: list[str]) -> list[list[str] | None]:
        variants = []
        if stop_sequences:
            variants.append(stop_sequences)
        variants.append(None)
        return variants

    @staticmethod
    def _attempt_tag(with_stop: bool, with_think: bool) -> str:
        parts = []
        if with_think:
            parts.append("think")
        if with_stop:
            parts.append("stop")
        return "+".join(parts) if parts else "none"

    def _build_stream_done_event(
        self,
        decision: RouteDecision,
        model: str,
        effective_system_prompt: str | None,
        normalized_dirs: list[str],
        effective_profile: str,
        response_text: str,
        thinking_text: str,
        output_sanitized: bool,
        think_requested: bool,
        think_applied: bool | str | None,
        think_unsupported: bool,
        warnings: list[str],
        errors: list[str],
        stream_status: str,
        interruption_stage: str | None = None,
        done_chunk: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Contract fields: think_requested, think_applied, think_status, warnings, errors, response
        return {
            "type": "done",
            "model_used": model if model else None,
            "mode": decision.mode,
            "intent": decision.intent,
            "override_from_prompt": decision.override_from_prompt,
            "fallback_used": False,
            "system_prompt_applied": effective_system_prompt is not None,
            "allowed_directories": normalized_dirs,
            "profile": effective_profile,
            "response": response_text if response_text is not None else None,
            "thinking": thinking_text if think_requested else None,
            "output_sanitized": output_sanitized,
            "think": think_applied if think_applied is not None else None,
            "think_requested": think_requested if think_requested is not None else False,
            "think_applied": think_applied if think_applied is not None else None,
            "think_status": self._compute_think_status(think_requested, think_applied, think_unsupported) if think_requested is not None else "unavailable",
            "warnings": warnings if warnings is not None else [],
            "errors": errors if errors is not None else [],
            "stream_status": stream_status,
            "interruption_stage": interruption_stage,
            "done_chunk": done_chunk,
        }

    def available_profiles(self) -> dict:
        config = self._config_snapshot()
        profiles = config.get("profiles", {})
        return {"profiles": list(profiles.keys())}

    def generate(
        self,
        prompt: str,
        mode: str = "auto",
        manual_model: str | None = None,
        profile: str | None = None,
        system_prompt: str | None = None,
        allowed_directories: list[str] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        think: bool | str | None = None,
        confirm_irreversible: bool = False,
    ) -> dict[str, Any]:
        config = self._config_snapshot()
        effective_profile = self._resolve_profile_name(profile, config=config)
        if profile and profile != effective_profile:
            print(f"[ROUTER] Agent/Profile used: {effective_profile}")
        if self._requires_irreversible_confirmation(prompt, config=config) and not confirm_irreversible:
            _, _, effective_think = self._effective_generation_controls(
                effective_profile, temperature, max_tokens, think, config=config
            )
            # Contract fields: think_requested, think_applied, think_status, warnings, errors, response
            return {
                "response": "",
                "thinking": None,
                "model": None,
                "model_used": None,
                "mode": mode,
                "intent": None,
                "override_from_prompt": False,
                "profile": effective_profile,
                "think_requested": bool(effective_think),
                "think_applied": None,
                "think_status": "unavailable",
                "warnings": [],
                "errors": ["Irreversible operation requires confirmation"],
                "fallback_used": False,
                "system_prompt_applied": False,
                "allowed_directories": [],
            }
        decision = self._decide_model(prompt, mode, manual_model, config=config)
        candidates = self._candidate_models(decision, config=config)
        temperature, max_tokens, think = self._effective_generation_controls(
            effective_profile, temperature, max_tokens, think, config=config
        )
        stop_sequences = self._effective_stop_sequences(effective_profile, config=config)
        effective_system_prompt = self._build_system_prompt(
            effective_profile, system_prompt, allowed_directories, config=config
        )
        normalized_dirs = self._normalize_directories(allowed_directories)
        think_requested = bool(think)
        think_applied: bool | str | None = None
        think_unsupported = False
        fallback_used = False
        warnings: list[str] = []
        errors: list[str] = []
        first_model = True
        for model in candidates:
            if not first_model:
                fallback_used = True
            first_model = False
            for attempt_think in self._stream_think_attempts(think):
                stop_variants = self._stream_stop_variants(stop_sequences)
                for idx, attempt_stop in enumerate(stop_variants):
                    try:
                        response = self.client.generate(
                            model=model,
                            prompt=prompt,
                            system_prompt=effective_system_prompt,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            think=attempt_think,
                            stop_sequences=attempt_stop,
                        )
                        think_applied = attempt_think
                        response_text, output_sanitized = self._sanitize_output(response, config=config)
                        final_think_applied = think_applied
                        if think_unsupported:
                            final_think_applied = False
                        elif fallback_used and think_requested:
                            final_think_applied = False
                        # Contract fields: think_requested, think_applied, think_status, warnings, errors, response
                        return {
                            "response": response_text if response_text is not None else None,
                            "thinking": None,
                            "model": model,
                            "model_used": model,
                            "mode": decision.mode,
                            "intent": decision.intent,
                            "override_from_prompt": decision.override_from_prompt,
                            "profile": effective_profile,
                            "think_requested": think_requested if think_requested is not None else False,
                            "think_applied": final_think_applied if final_think_applied is not None else None,
                            "think_status": self._compute_think_status(think_requested, final_think_applied, think_unsupported) if think_requested is not None else "unavailable",
                            "output_sanitized": output_sanitized,
                            "warnings": warnings if warnings is not None else [],
                            "errors": errors if errors is not None else [],
                            "fallback_used": fallback_used,
                            "system_prompt_applied": effective_system_prompt is not None,
                            "allowed_directories": normalized_dirs,
                        }
                    except Exception as e:
                        errors.append(f"{model}: {str(e)}")
                        err_str = str(e).lower()
                        status_code = None
                        try:
                            if hasattr(e, "response") and e.response is not None:
                                status_code = e.response.status_code
                            elif hasattr(e, "status_code"):
                                status_code = e.status_code
                        except Exception:
                            pass
                        if "think" in err_str and attempt_think is not None:
                            think_unsupported = True
                            break
                        if "stop" in err_str and attempt_stop is not None:
                            if idx < len(stop_variants) - 1:
                                warnings.append("retry without stop sequences")
                                continue
                        if not self._is_retryable_status(status_code):
                            break
                        fallback_used = True
        raise RouterError(f"All models failed: {errors}")

    def generate_stream(
        self,
        prompt: str,
        mode: str = "auto",
        manual_model: str | None = None,
        profile: str | None = None,
        system_prompt: str | None = None,
        allowed_directories: list[str] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        think: bool | str | None = None,
        confirm_irreversible: bool = False,
    ) -> Iterator[dict[str, Any]]:
        config = self._config_snapshot()
        effective_profile = self._resolve_profile_name(profile, config=config)
        if profile and profile != effective_profile:
            print(f"[ROUTER] Agent/Profile used (stream): {effective_profile}")
        if self._requires_irreversible_confirmation(prompt, config=config) and not confirm_irreversible:
            temperature, max_tokens, effective_think = self._effective_generation_controls(
                effective_profile, temperature, max_tokens, think, config=config
            )
            # Contract fields: think_requested, think_applied, think_status, warnings, errors, response
            yield {
                "type": "done",
                "model_used": None,
                "mode": mode,
                "intent": None,
                "override_from_prompt": False,
                "fallback_used": False,
                "system_prompt_applied": False,
                "allowed_directories": [],
                "profile": effective_profile,
                "response": "",
                "thinking": None,
                "output_sanitized": False,
                "think": None,
                "think_requested": bool(effective_think),
                "think_applied": None,
                "think_status": "unavailable",
                "warnings": [],
                "errors": ["Irreversible operation requires confirmation"],
                "stream_status": "completed",
                "interruption_stage": None,
                "done_chunk": None,
            }
            return
        decision = self._decide_model(prompt, mode, manual_model, config=config)
        candidates = self._candidate_models(decision, config=config)
        temperature, max_tokens, think = self._effective_generation_controls(
            effective_profile, temperature, max_tokens, think, config=config
        )
        stop_sequences = self._effective_stop_sequences(effective_profile, config=config)
        effective_system_prompt = self._build_system_prompt(
            effective_profile, system_prompt, allowed_directories, config=config
        )
        normalized_dirs = self._normalize_directories(allowed_directories)
        think_requested = bool(think)
        think_attempts = self._stream_think_attempts(think)
        stop_variants = self._stream_stop_variants(stop_sequences)
        fallback_used = False
        warnings: list[str] = []
        errors: list[str] = []
        response_text = ""
        thinking_text = ""
        output_sanitized = False
        think_applied: bool | str | None = None
        think_unsupported = False
        think_downgraded = False
        _think_was_downgraded = False
        for model in candidates:
            think_idx = 0
            while think_idx < len(think_attempts):
                with_think = think_attempts[think_idx]
                if think_downgraded:
                    with_think = None
                    think_downgraded = False
                stop_variants = self._stream_stop_variants(stop_sequences)
                stop_idx = 0
                while stop_idx < len(stop_variants):
                    with_stop = stop_variants[stop_idx]
                    meta_emitted = False
                    try:
                        current_stop = stop_sequences if with_stop else None
                        stream = self.client.generate_stream(
                            model=model,
                            prompt=prompt,
                            system_prompt=effective_system_prompt,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            think=with_think,
                            stop_sequences=current_stop,
                        )
                        think_applied = with_think
                        for chunk in stream:
                            if not isinstance(chunk, dict):
                                continue
                            if not meta_emitted:
                                yield self._build_stream_meta_event(
                                    decision=decision,
                                    model=model,
                                    effective_profile=effective_profile,
                                    effective_system_prompt=effective_system_prompt,
                                    normalized_dirs=normalized_dirs,
                                    think_requested=think_requested,
                                    think_applied=think_applied,
                                    think_unsupported=think_unsupported,
                                    fallback_used=fallback_used,
                                    warnings=warnings,
                                )
                                meta_emitted = True
                            if chunk.get("done"):
                                done_chunk = chunk
                                response_text += chunk.get("response", "")
                                final_think_applied = think_applied
                                if think_unsupported:
                                    final_think_applied = False
                                yield self._build_stream_done_event(
                                    decision=decision,
                                    model=model,
                                    effective_system_prompt=effective_system_prompt,
                                    normalized_dirs=normalized_dirs,
                                    effective_profile=effective_profile,
                                    response_text=response_text,
                                    thinking_text=thinking_text,
                                    output_sanitized=output_sanitized,
                                    think_requested=think_requested,
                                    think_applied=final_think_applied,
                                    think_unsupported=think_unsupported,
                                    warnings=warnings,
                                    errors=errors,
                                    stream_status="completed",
                                    done_chunk=done_chunk,
                                )
                                return
                            if chunk.get("thinking"):
                                thinking_text += chunk["thinking"]
                            response_text += chunk.get("response", "")
                            yield chunk
                    except Exception as e:
                        err_str = str(e).lower()
                        status_code = None
                        is_timeout_error = "timeout" in err_str
                        try:
                            if hasattr(e, "response") and e.response is not None:
                                status_code = e.response.status_code
                            elif hasattr(e, "status_code"):
                                status_code = e.status_code
                        except Exception:
                            pass
                        if status_code is None and is_timeout_error:
                            status_code = -1
                        errors.append(f"{model}: {str(e)}")
                        if meta_emitted:
                            final_think_applied = think_applied
                            if think_unsupported:
                                final_think_applied = False
                            elif fallback_used and think_requested:
                                final_think_applied = False
                            yield self._build_stream_done_event(
                                decision=decision,
                                model=model,
                                effective_system_prompt=effective_system_prompt,
                                normalized_dirs=normalized_dirs,
                                effective_profile=effective_profile,
                                response_text=response_text,
                                thinking_text=thinking_text,
                                output_sanitized=output_sanitized,
                                think_requested=think_requested,
                                think_applied=final_think_applied,
                                think_unsupported=think_unsupported,
                                warnings=warnings,
                                errors=errors,
                                stream_status="partial_interrupted",
                                interruption_stage="post_token",
                                done_chunk=None,
                            )
                            return
                        is_think_unsupported_error = ("think" in err_str or "thinking" in err_str) and with_think is not None and not self._is_retryable_status(status_code)
                        if is_think_unsupported_error:
                            think_unsupported = True
                            warnings.append("m1: retry without think after rejection")
                            break
                        if "stop" in err_str and with_stop is not None:
                            if stop_idx < len(stop_variants) - 1:
                                warnings.append("m1: transient 503 with stop sequences, retry without stop")
                                stop_idx += 1
                                continue
                            else:
                                warnings.append("m1: transient 503 with stop sequences exhausted")
                        if self._is_retryable_status(status_code) and with_think is not None and with_think is not False:
                            warnings.append("m1: transient 503, retry without think")
                            warnings.append("m1: retry without think after rejection")
                            think_downgraded = True
                            fallback_used = True
                            break
                        if not self._is_retryable_status(status_code):
                            break
                        fallback_used = True
                think_idx += 1
        raise RouterError(f"stream_failed_pre_token: All models failed: {errors}")


def default_router() -> LocalModelRouter:
    return LocalModelRouter("config.json")
