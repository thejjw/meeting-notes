# Contributing to meeting-notes

Thank you for your interest in contributing!

## Development Setup

```bash
# Clone the repository
git clone <repo-url>
cd meeting-notes

# Install dependencies
uv sync --group dev

# Run tests
uv run pytest

# Run linter
uv run ruff check .

# Run type checker (with optional dependencies)
uv run --extra all pyright
```

## Project Structure

- `src/meeting_notes/` - Main package
- `tests/` - Unit and integration tests
- `config/` - Configuration files and profiles
- `docker/` - Docker configurations
- `prompts/` - Summarization prompts
- `schemas/` - JSON schemas

## Adding a New ASR Backend

1. Create `src/meeting_notes/asr/your_backend.py`
2. Implement the `ASRBackend` abstract class
3. Register in `src/meeting_notes/asr/registry.py`
4. Add tests in `tests/unit/`

## Adding a New Summarizer Adapter

1. Create adapter class in `src/meeting_notes/summarization/adapters.py`
2. Implement `SummarizerAdapter` abstract class
3. Register with `register_adapter()`
4. Add tests

## Code Style

- Python 3.12 (strictly `==3.12.*`)
- Type hints on all public functions
- ruff for linting
- pyright for type checking
- Conventional Commits for git messages

## Testing

```bash
# Unit tests only
uv run pytest tests/unit/

# All tests including integration
uv run pytest

# Specific test file
uv run pytest tests/unit/test_config.py
```

## Git Workflow

1. Create a feature branch
2. Make changes with conventional commits
3. Run tests and linter
4. Submit a pull request

## Questions?

Open an issue or start a discussion.
