# Bandido Project State

## What This Is
A Python simulation of cooperative Bandido with local and API-backed player strategies.

## Current Entry Point
- Run: `python bandido.py`
- Benchmark: `python bandido.py --benchmark --games 25 --seed 1`
- MiniMax smoke test: `python bandido.py --strategy minimax-api --turn-limit 1 --api-options 4 --verbose`
- High-res image: `python bandido.py --png-scale 12 --output bandido_hires.png`

## Current State
- `bandido.py` contains the rules simulation, strategies, CLI, benchmark mode, and PNG rendering.
- Strategies are `random`, `greedy`, `compact`, `minimax-api`, and `nvidia-nim-api`.
- API strategy uses `.env` variables: `MINIMAX_API_KEY`, `MINIMAX_BASE_URL`, `MINIMAX_MODEL`, and `MINIMAX_TIMEOUT`.
- API choices are constrained to locally generated legal moves.
- Slow or failed API requests fall back to the best local ranked legal move.

## Verification
- Syntax: `python -m py_compile bandido.py`
- Local benchmark: `python bandido.py --benchmark --games 3 --seed 3`
