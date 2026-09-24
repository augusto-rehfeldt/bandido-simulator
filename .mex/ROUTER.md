# Bandido Project State

## What This Is
A Python simulation of cooperative Bandido with local and API-backed player strategies.
See README.md for usage, benchmark results and the card rules the simulator enforces.

## Current Entry Point
- Run: `python bandido.py --seed 37 --output game.png`
- Benchmark: `python bandido.py --benchmark --games 30 --seed 1`
- Catalogue check: `python bandido.py --validate`
- API smoke test: `python bandido.py --strategy minimax-api --turn-limit 1 --api-options 4 --verbose`

## Current State
- `bandido.py` contains the rules simulation, strategies, CLI, benchmark mode and PNG rendering
  (cropped to the tunnels unless `--full-board`).
- Strategies: `random`, `greedy`, `compact`, `minimax-api`, `nvidia-nim-api`.
- API strategies read `MINIMAX_*` or `NVIDIA_*` variables from `.env` (see `.env.example`).
- API choices are constrained to locally generated legal moves; invalid, failed or slow
  replies fall back to the best local ranked move.
- Card edge openings must sit on connector cells; `validate_piece_catalog` enforces it.

## Verification
- `python -B -m unittest -q test_bandido` (offline, test-first: add a failing test before changing behaviour)
