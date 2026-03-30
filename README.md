# Local Multi-Model Router (Auto + Manual)

Router locale per Ollama con:

- auto-routing per intent (`coding`, `reasoning`, `rag`, `fast`, `chat`)
- override manuale (`/manual <model>` oppure tag `# model: ...`)
- API REST per integrazione VSCode/CLI
- fallback robusto su modelli locali

## Requisiti

- Ollama attivo su `http://localhost:11434`
- Python 3.10+

## Setup rapido

Usa i comandi della shell corretta.

Convenzione operativa (consigliata):

- Esegui i comandi da dentro `ai-stack`.
- Se sei nella cartella padre, fai `cd ai-stack` una sola volta all'inizio.

PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CMD:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In alternativa, senza attivazione (funziona in ogni shell):

```bat
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Avvio API REST

Preflight consigliato (evita errore porta 8000 gia in uso e path errato):

```powershell
.\.venv\Scripts\python.exe scripts\preflight_runtime.py --base-url http://127.0.0.1:8001
```

Avvio safe consigliato (workspace `vsc`, evita collisioni e prova fallback porta):

```powershell
.\.venv\Scripts\python.exe scripts\start_safe.py --port 8001 --fallback-next
```

Opzioni utili:

```powershell
.\.venv\Scripts\python.exe scripts\start_safe.py --host 127.0.0.1 --port 8001 --fallback-next --max-port 8010 --no-reload
```

Esiti preflight:

- `status: ok` -> router gia attivo su `127.0.0.1:8000`
- `status: ready` -> porta libera, puoi avviare uvicorn
- `status: conflict` -> porta occupata da altro processo/servizio

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

Per CMD:

```bat
.venv\Scripts\activate.bat
python -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

Endpoint principali:

- `GET /health`
- `GET /models`
- `GET /profiles`
- `POST /llm`
- `GET /chat` (interfaccia web chat)

Esempio chiamata:

```powershell
curl -X POST http://127.0.0.1:8000/llm `
  -H "Content-Type: application/json" `
  -d '{"prompt":"analizza questa query SQL", "mode":"auto"}'
```

Interfaccia web:

- `http://127.0.0.1:8000/chat`

## Smoke test rapido (runtime)

Con server gia avviato su `127.0.0.1:8000`, esegui un check automatico di:

- `GET /health`
- `GET /chat`
- stream `POST /llm/stream` con validazione evento `done` (`response` + `thinking`)

Nota: se sei gia dentro la cartella `ai-stack`, non ripetere `cd ai-stack`.

PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_runtime.py --base-url http://127.0.0.1:8001
```

CMD:

```bat
.venv\Scripts\python.exe scripts\smoke_runtime.py
```

## Acceptance test contratto UI-think

Test rapido e deterministico del contratto backend usato dalla UI per trasparenza thinking:

- `think_status`: `enabled|downgraded|unsupported|unavailable`
- campi richiesti: `think_requested`, `think_applied`, `warnings`, `errors`

PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\acceptance_think_ui.py
```

CMD:

```bat
.venv\Scripts\python.exe scripts\acceptance_think_ui.py
```

## Avvio CLI

```powershell
.\.venv\Scripts\Activate.ps1
python cli.py
```

Per CMD:

```bat
.venv\Scripts\activate.bat
python cli.py
```

Comandi CLI:

- `/models`
- `/profiles`
- `/profile <nome>`
- `/auto`
- `/manual <model>`
- `/confirm on|off`
- `/reload`
- `/exit`

## Configurazione

Modifica `config.json` per cambiare mappa intent -> modello, fallback e keyword.

Configurazione attuale tarata sui modelli gia presenti localmente:

- `deepseek-coder:6.7b`
- `deepseek-r1:8b`
- `qwen2.5:7b`
- `qwen3:latest`

## Installazione modelli opzionali (solo se necessari)

Esegui solo i pull che ti servono davvero:

```powershell
ollama pull phi3:3.8b
ollama pull qwen2.5-coder:7b
ollama pull qwen3-coder-next
```

Nota: alcuni modelli possono essere grandi o non disponibili in tutte le varianti/tag.

## Integrazione VSCode (REST)

Se vuoi inviare prompt al router via task/estensione, usa endpoint:

- `http://127.0.0.1:8000/llm`

Body JSON minimo:

```json
{
  "prompt": "fix this function",
  "mode": "auto"
}
```

Body JSON con system prompt e perimetro directory:

```json
{
  "prompt": "analizza il modulo e proponi refactor",
  "mode": "auto",
  "system_prompt": "Rispondi solo in italiano tecnico e con check-list operative.",
  "allowed_directories": [
    "C:/Users/A893apulia/Documents/Progettia/ai-stack",
    "C:/Users/A893apulia/Documents/Progettia/docs"
  ]
}
```

Body JSON con profilo safe e guardrail conferma:

```json
{
  "prompt": "esegui git reset --hard",
  "mode": "auto",
  "profile": "safe",
  "confirm_irreversible": false
}
```

In questo caso il router blocca la richiesta e chiede conferma esplicita.

Override manuale via body:

```json
{
  "prompt": "fix this function",
  "mode": "manual",
  "model": "deepseek-coder:6.7b"
}
```

## Note errori comuni

- Se compare `Impossibile trovare il file specificato` con `query_step1.py` o `ingest_json_pages.py`, quei file non fanno parte di questo progetto.
- Se compare `Defaulting to user installation`, la venv non e stata attivata correttamente.
- Non copiare i delimitatori markdown (` ``` `) dentro il terminale.

## Dove si imposta il system prompt

- Prompt globale di default: `config.json` -> `prompting.system_prompt`
- Prompt per profilo: `config.json` -> `profiles.<nome>.system_prompt`
- Prompt per singola richiesta: campo `system_prompt` nella `POST /llm`
- Da interfaccia web: pannello "system prompt (opzionale)" in `GET /chat`

Precedenza merge system prompt:

1. `prompting.system_prompt` (globale)
2. `profiles.<nome>.system_prompt` (profilo)
3. `system_prompt` nella request (override per richiesta)

## Posso indirizzare directory specifiche?

Si. Puoi passare una lista in `allowed_directories` nella `POST /llm` oppure dalla UI web nel campo dedicato.
Il router converte i percorsi in assoluti e li inserisce nel contesto del system prompt quando `prompting.enforce_directory_scope` e `true`.

## Guardrail operativi

- Il profilo di default e `safe` (`default_profile` in `config.json`).
- Se la richiesta sembra irreversibile (es. comandi distruttivi), il router chiede conferma.
- Conferma possibile via `confirm_irreversible=true` oppure includendo il token di conferma in prompt (`guardrails.confirmation_token`).
- L'output viene ripulito da token di controllo noti (`output_filters`) per ridurre leakage del template.
