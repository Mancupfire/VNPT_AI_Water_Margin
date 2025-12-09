# VNPT_AI_Water_Margin

Lightweight runner that queries a VNPT-hosted LLM to answer multiple-choice questions from JSON datasets and writes predictions to CSV/JSON.

Quickstart

1. Create a Python 3.11+ virtual environment and install dependencies:

```pwsh
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Create a `.env` file from `.env.example` and populate the VNPT credentials:

```pwsh
copy .env.example .env
# then edit .env and fill in VNPT_ACCESS_TOKEN, VNPT_TOKEN_ID, VNPT_TOKEN_KEY
```

3. Run the main runner:

```pwsh
python main.py
```

Notes

- Secrets should be provided via environment variables; do not commit credentials to version control.
- Adjust `SLEEP_TIME` in the environment to control request pacing (respect VNPT quota).
- Use `process_dataset(...)` in `main.py` to run on different input files.

Use different providers

- By default the project uses the `vnpt` provider. You can switch providers via the `PROVIDER` environment variable (supported: `vnpt`, `ollama`, `openai`).

Docker

Build the container:

```pwsh
docker build -t vnpt-ai-water-margin:latest .
```

Run container (example using env file):

```pwsh
docker run --rm --env-file .env vnpt-ai-water-margin:latest
```

Adapting providers

- Provider implementations are in `src/providers/`. To add a provider, implement a `create(config)` factory that returns an object with a `chat(messages, config)` method.

Examples

- To use a local Ollama instance, set `PROVIDER=ollama` and `OLLAMA_BASE` in your `.env`.
- To use OpenAI, set `PROVIDER=openai` and `OPENAI_API_KEY` (the OpenAI adapter requires the `openai` package).