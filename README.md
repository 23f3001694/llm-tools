# TDS Quiz Solver

A robust automated quiz-solving agent with multi-model LLM support, designed to handle data-related tasks within a 3-minute time limit.

## Features

- **Multi-Model Support**: Automatic failover between Gemini, Groq, and OpenRouter
- **Robust Error Handling**: Timeouts, retries, and graceful degradation
- **Task Isolation**: Each task runs in its own workspace directory
- **Sandboxed Code Execution**: Resource-limited Python execution
- **Structured Logging**: JSON logs with per-task context

## Quick Start

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .
```

### 2. Install Playwright

```bash
playwright install chromium
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys and credentials
```

### 4. Run the Server

```bash
python -m src.main
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Google Gemini API key | Required (at least one LLM) |
| `GROQ_API_KEY` | Groq API key | Optional fallback |
| `OPENROUTER_API_KEY` | OpenRouter API key | Optional fallback |
| `EMAIL` | Your email for quiz submissions | Required |
| `SECRET` | Your secret for authentication | Required |
| `TASK_TIMEOUT_SECONDS` | Max time per quiz chain | 170 |
| `MAX_RETRIES_PER_QUESTION` | Retry attempts per question | 3 |

## API Endpoints

### POST /solve

Start solving a quiz.

```json
{
  "email": "your@email.com",
  "secret": "your-secret",
  "url": "https://example.com/quiz-123"
}
```

### GET /healthz

Health check endpoint.

### GET /task/{task_id}

Get status of a running task.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   FastAPI   │────▶│  QuizAgent   │────▶│  LLM Client │
│   Server    │     │  (LangGraph) │     │  (Failover) │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │    Tools     │
                    ├──────────────┤
                    │ web_scraper  │
                    │ download_file│
                    │ run_code     │
                    │ post_request │
                    │ get_request  │
                    │ add_deps     │
                    └──────────────┘
```

## Docker

```bash
# Build
docker build -t tds-quiz-solver .

# Run
docker run -p 7860:7860 --env-file .env tds-quiz-solver
```

## Development

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/
```

## License

MIT
