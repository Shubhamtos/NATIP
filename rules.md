# NATIP Development Rules

You are developing NATIP (NSE Agentic Trading Intelligence Platform).

Always follow these rules:

## General
- Never rewrite completed modules unless requested.
- Preserve the existing project architecture.
- Use Python 3.12+.
- Use type hints everywhere.
- Use Google-style docstrings.
- Follow SOLID principles.
- Prefer composition over inheritance.
- Keep modules small and focused.

## Architecture
- Agents never communicate directly.
- All outputs go to the Evidence Store.
- The Rule Engine is the primary decision maker.
- AI only explains and challenges decisions.
- Do not hardcode trading rules.
- Every module should be independently testable.

## Code Quality
- Production-quality code only.
- No placeholder implementations.
- Include logging.
- Include error handling.
- Include unit tests.
- Keep imports clean.

## Important
Only implement the requested module.
Do not modify unrelated files unless absolutely necessary.