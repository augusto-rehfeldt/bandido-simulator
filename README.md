# Bandido simulator

A simulation of the cooperative tile game *Bandido*: players lay 3x6 tunnel cards
around a start card and try to close every open exit before the deck runs out.
Local strategies (random, greedy, compact) and optional LLM players choose
among the legal moves the simulator generates.

![A won game: every tunnel ends in a torch (red) or joins another tunnel](docs/won-game.png)

## Run

```powershell
pip install -r requirements.txt
python bandido.py --seed 37 --output game.png          # one game, rendered to PNG
python bandido.py --benchmark --games 30 --seed 1      # compare strategies
python bandido.py --validate                           # check the card catalogue
```

Images are cropped to the tunnels; `--full-board` renders the whole grid and
`--png-scale` sets the pixel size. Benchmark on 30 seeded games (2 players, 69 cards):

| strategy | wins | avg open exits left |
|---|---:|---:|
| random | 0/30 | 43.4 |
| greedy | 1/30 | 19.8 |
| compact | 8/30 | 9.5 |

## LLM players

`--strategy minimax-api` or `--strategy nvidia-nim-api` sends the ranked legal
moves to an OpenAI-compatible endpoint. Copy `.env.example` to `.env` and set
`MINIMAX_API_KEY` or `NVIDIA_API_KEY` (plus optional `*_MODEL`, `*_BASE_URL`,
`*_TIMEOUT`). An invalid, failed or slow reply falls back to the best-ranked move.

## Tests

```powershell
python -B -m unittest -q test_bandido
```

Offline: API calls are stubbed. The suite validates the catalogue (edge openings
only at connectors, torches as dead ends), placement, move projection,
strategies, seeded games and PNG rendering.

## License

CC0 1.0. See [LICENSE](LICENSE). Unofficial fan simulation; *Bandido* is a game by
Helvetiq, and this project is not affiliated with or endorsed by them.
