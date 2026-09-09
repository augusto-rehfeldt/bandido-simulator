import argparse
import json
import os
import random
import time
from dataclasses import dataclass

import numpy as np


EMPTY = -1
PATH = 0
WALL = 1
TORCH = 2


PORT_COORDS = {
    0: (0, 1),  # upper-left long-side connector
    1: (0, 4),  # upper-right long-side connector
    2: (1, 5),  # right short-side connector
    3: (2, 4),  # lower-right long-side connector
    4: (2, 1),  # lower-left long-side connector
    5: (1, 0),  # left short-side connector
}

BASIC_CARD_PORTS = [
    (0, 1, 2, 3),
    (0, 1, 3, 4),
    (0, 1, 2, 5),
    (0, 2, 3, 5),
    (1, 2, 4, 5),
    (0, 1, 2, 4),
    (0, 1, 3, 5),
    (0, 1, 4, 5),
    (0, 2, 3, 4),
    (0, 1, 4),
    (0, 1, 3),
    (0, 2, 4),
    (1, 2, 3),
    (0, 1, 2),
    (0, 2, 5),
    (1, 4, 5),
    (0, 3, 5),
    (1, 2, 4),
    (0, 1),
    (2, 5),
    (0, 2),
    (1, 2),
    (0, 3),
    (1, 4),
]

BASIC_CARD_NAMES = [
    "bed-left",
    "bed-right",
    "cross",
    "facebook",
    "twitter",
    "thorn-left",
    "thorn-right",
    "bench",
    "h",
    "momma-bear-left",
    "momma-bear-right",
    "fancy-j-left",
    "fancy-j-right",
    "t",
    "branch-left",
    "branch-right",
    "faucet-left",
    "faucet-right",
    "noodle",
    "straight",
    "l-curve",
    "j-curve",
    "c",
    "s-z",
]


def piece_from_ports(ports):
    shape = np.full((3, 6), WALL, dtype=np.int16)
    anchor_columns = []
    for port in ports:
        row, column = PORT_COORDS[port]
        shape[row, column] = PATH
        if row == 0:
            shape[1, column] = PATH
            anchor_columns.append(column)
        elif row == 2:
            shape[1, column] = PATH
            anchor_columns.append(column)
        elif column == 0:
            anchor_columns.append(1)
        elif column == shape.shape[1] - 1:
            anchor_columns.append(shape.shape[1] - 2)
        else:
            anchor_columns.append(column)
    if anchor_columns:
        shape[1, min(anchor_columns):max(anchor_columns) + 1] = PATH
    return shape.tolist()


def ender_shape(open_port, extra_cells, torch_cell):
    shape = np.full((3, 6), WALL, dtype=np.int16)
    shape[1, 1:5] = PATH
    row, column = PORT_COORDS[open_port]
    shape[row, column] = PATH
    if row == 0:
        shape[1, column] = PATH
    elif row == 2:
        shape[1, column] = PATH
    for row, column in extra_cells:
        shape[row, column] = PATH
    shape[torch_cell] = TORCH
    return shape.tolist()


ENDER_CARDS = [
    ("j-torch", ender_shape(4, [(1, 1), (1, 2), (1, 3)], (1, 4))),
    ("l-torch", ender_shape(3, [(1, 2), (1, 3), (1, 4)], (1, 1))),
    ("x-torch", ender_shape(5, [(1, 1), (0, 2), (2, 2), (1, 3)], (1, 4))),
    ("t-torch", ender_shape(2, [(1, 4), (0, 3), (2, 3)], (1, 1))),
    ("i-torch", ender_shape(0, [(1, 1), (1, 2)], (1, 4))),
    ("y-torch-left", ender_shape(5, [(1, 1), (0, 2), (1, 3)], (1, 4))),
    ("y-torch-right", ender_shape(2, [(1, 4), (0, 3), (1, 2)], (1, 1))),
]

CARD_NAMES = {
    **{str(index): name for index, name in enumerate(BASIC_CARD_NAMES, start=1)},
    **{str(index): name for index, (name, _shape) in enumerate(ENDER_CARDS, start=len(BASIC_CARD_PORTS) + 1)},
}

PIECES = {
    **{str(index): piece_from_ports(ports) for index, ports in enumerate(BASIC_CARD_PORTS, start=1)},
    **{str(index): shape for index, (_name, shape) in enumerate(ENDER_CARDS, start=len(BASIC_CARD_PORTS) + 1)},
}


def validate_piece_catalog():
    invalid_terminal_paths = []
    invalid_torches = []
    for shape_num, shape in PIECES.items():
        shape = np.array(shape, dtype=np.int16)
        for row, column in np.argwhere(shape == PATH):
            is_edge_cell = row in (0, shape.shape[0] - 1) or column in (0, shape.shape[1] - 1)
            path_neighbors = 0
            torch_neighbors = 0
            for next_row, next_column in ((row - 1, column), (row, column - 1), (row + 1, column), (row, column + 1)):
                if 0 <= next_row < shape.shape[0] and 0 <= next_column < shape.shape[1]:
                    path_neighbors += int(shape[next_row, next_column] == PATH)
                    torch_neighbors += int(shape[next_row, next_column] == TORCH)
            if path_neighbors + torch_neighbors <= 1 and not is_edge_cell:
                invalid_terminal_paths.append(shape_num)
                break

        for row, column in np.argwhere(shape == TORCH):
            path_neighbors = 0
            for next_row, next_column in ((row - 1, column), (row, column - 1), (row + 1, column), (row, column + 1)):
                if 0 <= next_row < shape.shape[0] and 0 <= next_column < shape.shape[1]:
                    path_neighbors += int(shape[next_row, next_column] == PATH)
            if path_neighbors != 1:
                invalid_torches.append(shape_num)
                break
    if invalid_terminal_paths:
        raise ValueError(f"Invalid cards with non-torch internal terminal paths: {', '.join(invalid_terminal_paths)}")
    if invalid_torches:
        raise ValueError(f"Invalid torch cards where the torch is not a terminal end: {', '.join(invalid_torches)}")


DEFAULT_DECK_MANIFEST = {
    **{str(index): 2 for index in range(1, len(BASIC_CARD_PORTS) + 1)},
    **{str(index): 3 for index in range(len(BASIC_CARD_PORTS) + 1, len(PIECES) + 1)},
}

START_PIECE = np.array(
    [[1, 0, 1, 1, 0, 1], [0, 0, 0, 0, 0, 0], [1, 0, 1, 1, 0, 1]],
    dtype=np.int16,
)


def adjacent_values(matrix, row, column):
    values = []
    for next_row, next_column in ((row - 1, column), (row, column - 1), (row + 1, column), (row, column + 1)):
        if 0 <= next_row < matrix.shape[0] and 0 <= next_column < matrix.shape[1]:
            values.append(matrix[next_row, next_column])
        else:
            values.append(EMPTY)
    return values


# ponytail: no python-dotenv dep in requirements; local parser stays.
def load_env(path=".env"):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def random_wall_code(used_codes):
    while True:
        code = random.randint(1000, 9999)
        if code % 10 != 0 and code not in used_codes:
            used_codes.add(code)
            return code


@dataclass(frozen=True)
class Piece:
    shape_num: str
    wall_code: int

    @property
    def base_shape(self):
        shape = np.array(PIECES[self.shape_num], dtype=np.int16)
        rendered = np.full(shape.shape, PATH, dtype=np.int16)
        rendered[shape == WALL] = self.wall_code
        rendered[shape == TORCH] = TORCH
        return rendered

    def rotations(self):
        seen = set()
        rotations = []
        for turns in range(4):
            shape = np.rot90(self.base_shape, turns)
            key = tuple(shape.ravel())
            if key not in seen:
                seen.add(key)
                rotations.append((turns * 90, shape.copy()))
        return rotations


@dataclass(frozen=True)
class Move:
    card: Piece
    coord: tuple
    shape: np.ndarray
    rotation: int
    score: int = 0
    features: dict = None
    rank: float = 0.0

    def label(self):
        coord = tuple(int(value) for value in self.coord)
        return (
            f"piece={self.card.shape_num}, rotation={int(self.rotation)}, "
            f"coord={coord}, score={int(self.score)}, rank={self.rank:.1f}"
        )


class Grid:
    def __init__(self, wh=150):
        self.wh = wh
        self.grid = np.full((wh, wh), EMPTY, dtype=np.int16)
        self.open_positions = set()

    def seed_start(self):
        shape = START_PIECE if random.choice([True, False]) else np.rot90(START_PIECE)
        top = self.wh // 2 - shape.shape[0] // 2
        left = self.wh // 2 - shape.shape[1] // 2
        top -= top % 3
        left -= left % 3
        self.insert((top, top + shape.shape[0], left, left + shape.shape[1]), shape)

    def insert(self, coord, shape):
        top, bottom, left, right = coord
        self.grid[top:bottom, left:right] = shape
        self.refresh_open_positions()

    def refresh_open_positions(self):
        self.open_positions.clear()
        for row, column in np.argwhere(self.grid == PATH):
            for next_row, next_column in ((row - 1, column), (row, column - 1), (row + 1, column), (row, column + 1)):
                if 0 <= next_row < self.wh and 0 <= next_column < self.wh:
                    if self.grid[next_row, next_column] == EMPTY:
                        self.open_positions.add((next_row, next_column))

    def is_won(self):
        return not self.open_positions

    def candidate_positions(self, shape):
        height, width = shape.shape
        candidates = set()
        for row, column in self.open_positions:
            for row_offset in range(height):
                top = row - row_offset
                if top % 3 != 0:
                    continue
                for column_offset in range(width):
                    left = column - column_offset
                    if left % 3 != 0:
                        continue
                    coord = (top, top + height, left, left + width)
                    if self.has_bounds(coord, margin=4):
                        candidates.add(coord)
        return list(candidates)

    def has_bounds(self, coord, margin=0):
        top, bottom, left, right = coord
        return top - margin >= 0 and left - margin >= 0 and bottom + margin <= self.wh and right + margin <= self.wh

    def check_place(self, coord, shape, wall_code):
        top, bottom, left, right = coord
        grid_portion = self.grid[top:bottom, left:right]
        if grid_portion.shape != shape.shape or not np.all(grid_portion == EMPTY):
            return False

        container = self.grid[top - 1:bottom + 1, left - 1:right + 1]
        if not self.check_incorrect_placement(container, shape, wall_code):
            return False

        area = self.grid[top - 4:bottom + 4, left - 4:right + 4].copy()
        area[4:shape.shape[0] + 4, 4:shape.shape[1] + 4] = shape
        return not self.creates_unreachable_hole(area)

    def count_open_after(self, coord, shape):
        return len(self.project_open_positions(coord, shape))

    def project_open_positions(self, coord, shape):
        top, _bottom, left, _right = coord
        occupied = {
            (top + row, left + column)
            for row in range(shape.shape[0])
            for column in range(shape.shape[1])
        }
        projected_open = self.open_positions - occupied

        for row, column in np.argwhere(shape == PATH):
            board_row = top + int(row)
            board_column = left + int(column)
            for next_row, next_column in (
                (board_row - 1, board_column),
                (board_row, board_column - 1),
                (board_row + 1, board_column),
                (board_row, board_column + 1),
            ):
                if (next_row, next_column) in occupied:
                    continue
                if 0 <= next_row < self.wh and 0 <= next_column < self.wh:
                    if self.grid[next_row, next_column] == EMPTY:
                        projected_open.add((next_row, next_column))

        return projected_open

    def move_features(self, coord, shape):
        top, bottom, left, right = coord
        occupied = {
            (top + row, left + column)
            for row in range(shape.shape[0])
            for column in range(shape.shape[1])
        }
        connected_exits = len(self.open_positions & occupied)
        projected_open = self.project_open_positions(coord, shape)
        open_after = len(projected_open)
        open_delta = len(self.open_positions) - open_after

        filled_rows, filled_columns = np.where(self.grid != EMPTY)
        if len(filled_rows):
            min_row = min(int(filled_rows.min()), top)
            max_row = max(int(filled_rows.max()), bottom - 1)
            min_column = min(int(filled_columns.min()), left)
            max_column = max(int(filled_columns.max()), right - 1)
        else:
            min_row, max_row, min_column, max_column = top, bottom - 1, left, right - 1
        board_width = max_column - min_column + 1
        board_height = max_row - min_row + 1

        branch_count = 0
        for row, column in np.argwhere(shape == PATH):
            board_row = top + int(row)
            board_column = left + int(column)
            path_neighbors = 0
            for next_row, next_column in (
                (board_row - 1, board_column),
                (board_row, board_column - 1),
                (board_row + 1, board_column),
                (board_row, board_column + 1),
            ):
                if (next_row, next_column) in occupied:
                    path_neighbors += int(shape[next_row - top, next_column - left] == PATH)
                elif 0 <= next_row < self.wh and 0 <= next_column < self.wh:
                    path_neighbors += int(self.grid[next_row, next_column] == PATH)
            branch_count += int(path_neighbors >= 3)

        nearest_distances = []
        for row, column in projected_open:
            distances = [abs(row - other_row) + abs(column - other_column) for other_row, other_column in projected_open if (other_row, other_column) != (row, column)]
            nearest_distances.append(min(distances) if distances else 0)
        isolated_exits = sum(distance > 9 for distance in nearest_distances)
        average_nearest_exit = sum(nearest_distances) / len(nearest_distances) if nearest_distances else 0

        return {
            "open_after": open_after,
            "open_delta": open_delta,
            "connected_exits": connected_exits,
            "new_exits": max(0, open_after - (len(self.open_positions) - connected_exits)),
            "branch_count": branch_count,
            "isolated_exits": isolated_exits,
            "average_nearest_exit": round(average_nearest_exit, 2),
            "board_width": board_width,
            "board_height": board_height,
            "board_area": board_width * board_height,
        }

    def check_incorrect_placement(self, container, shape, wall_code):
        amount_edge = 0
        leads_nowhere = 0
        placed = container.copy()
        height, width = shape.shape
        if placed.shape != (height + 2, width + 2):
            return False
        placed[1:height + 1, 1:width + 1] = shape

        for row, column in np.argwhere(placed == PATH):
            if not any(value == PATH for value in adjacent_values(placed, row, column)):
                return False

        for row in range(height):
            for column in range(width):
                value = shape[row, column]
                if row not in (0, height - 1) and column not in (0, width - 1):
                    continue

                adjacent = adjacent_values(placed, row + 1, column + 1)
                if value == PATH:
                    amount_edge += 1
                    if row == 0 and adjacent[0] == TORCH:
                        return False
                    if row == height - 1 and adjacent[2] == TORCH:
                        return False
                    if column == 0 and adjacent[1] == TORCH:
                        return False
                    if column == width - 1 and adjacent[3] == TORCH:
                        return False
                    if not all(item in (wall_code, PATH, TORCH, EMPTY) for item in adjacent):
                        return False
                    if not all(item in (wall_code, PATH, TORCH) for item in adjacent):
                        leads_nowhere += 1
                else:
                    if row == 0 and adjacent[0] == PATH:
                        return False
                    if row == height - 1 and adjacent[2] == PATH:
                        return False
                    if column == 0 and adjacent[1] == PATH:
                        return False
                    if column == width - 1 and adjacent[3] == PATH:
                        return False

        return leads_nowhere != amount_edge

    def creates_unreachable_hole(self, area):
        for top in range(0, area.shape[0] - 4, 3):
            for left in range(0, area.shape[1] - 4, 3):
                square = area[top:top + 5, left:left + 5]
                center = square[1:4, 1:4]
                if not np.all(center == EMPTY):
                    continue

                edges = (
                    square[0, 1:4],
                    square[4, 1:4],
                    np.array([square[1, 0], square[2, 0], square[3, 0]]),
                    np.array([square[1, 4], square[2, 4], square[3, 4]]),
                )
                if all(np.count_nonzero(edge == EMPTY) < 3 for edge in edges):
                    if any(np.any(edge == PATH) for edge in edges):
                        return True
        return False

    def board_summary(self):
        filled_rows, filled_columns = np.where(self.grid != EMPTY)
        if len(filled_rows) == 0:
            return {"open_paths": 0, "occupied_cells": 0, "bounds": None}
        return {
            "open_paths": len(self.open_positions),
            "occupied_cells": int(np.count_nonzero(self.grid != EMPTY)),
            "bounds": [int(filled_rows.min()), int(filled_rows.max()), int(filled_columns.min()), int(filled_columns.max())],
        }

    def to_image(self, scale=6):
        from PIL import Image

        image = np.where(self.grid[..., None] >= WALL, (139, 69, 19), (0, 0, 0)).astype(np.uint8)
        image[self.grid == EMPTY] = (245, 222, 179)
        image[self.grid == TORCH] = (230, 42, 30)
        data = Image.fromarray(image, "RGB")
        if scale > 1:
            data = data.resize((data.width * scale, data.height * scale), Image.Resampling.NEAREST)
        return data


def legal_moves(grid, hand):
    moves = []
    for card in hand:
        for rotation, shape in card.rotations():
            for coord in grid.candidate_positions(shape):
                if grid.check_place(coord, shape, card.wall_code):
                    features = grid.move_features(coord, shape)
                    score = features["open_delta"]
                    rank = strategic_rank(features)
                    moves.append(Move(card=card, coord=coord, shape=shape, rotation=rotation, score=score, features=features, rank=rank))
    return moves


def has_legal_move(grid, cards):
    for card in cards:
        for _rotation, shape in card.rotations():
            for coord in grid.candidate_positions(shape):
                if grid.check_place(coord, shape, card.wall_code):
                    return True
    return False


def strategic_rank(features):
    open_after = features["open_after"]
    open_delta = features["open_delta"]
    rank = open_delta * 120
    rank -= open_after * 3
    rank += features["connected_exits"] * 10
    rank -= features["new_exits"] * 18
    rank -= features["branch_count"] * 12
    rank -= features["isolated_exits"] * 16
    rank -= features["average_nearest_exit"] * 2
    rank -= features["board_area"] * 0.02

    if open_after <= 5:
        rank += open_delta * 40
        rank -= features["new_exits"] * 30
    return rank


def move_option(index, move):
    return {
        "id": index,
        "piece": move.card.shape_num,
        "rotation": int(move.rotation),
        "coord": [int(value) for value in move.coord],
        "rank": round(move.rank, 2),
        "features": move.features,
    }


class RandomStrategy:
    name = "random"

    def choose(self, grid, hand, moves, player_index):
        return random.choice(moves)


class GreedyStrategy:
    name = "greedy"

    def choose(self, grid, hand, moves, player_index):
        best_score = max(move.score for move in moves)
        best_moves = [move for move in moves if move.score == best_score]
        return random.choice(best_moves)


class CompactStrategy:
    name = "compact"

    def choose(self, grid, hand, moves, player_index):
        center = grid.wh / 2

        def ranking(move):
            top, bottom, left, right = move.coord
            move_row = (top + bottom) / 2
            move_column = (left + right) / 2
            distance = abs(move_row - center) + abs(move_column - center)
            area_growth = (bottom - top) * (right - left)
            return (move.rank, move.score, -distance, -area_growth)

        best_rank = max(ranking(move) for move in moves)
        best_moves = [move for move in moves if ranking(move) == best_rank]
        return random.choice(best_moves)


# ponytail: collapsed MinimaxApiStrategy + NvidiaNimApiStrategy; they differed only in env prefix + defaults.
class OpenAICompatibleApiStrategy:
    def __init__(self, name, prefix, default_base_url, default_model, default_timeout, max_options=12, extra_body_enabled=False):
        self.name = name
        self.prefix = prefix
        self.default_base_url = default_base_url
        self.default_model = default_model
        self.default_timeout = default_timeout
        self.max_options = max_options
        self.extra_body_enabled = extra_body_enabled
        self.client = None

    def get_client(self):
        if self.client is not None:
            return self.client
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(f"Install the OpenAI SDK with `pip install openai` to use --strategy {self.name}") from exc
        api_key = os.getenv(self.prefix + "API_KEY")
        if not api_key:
            raise RuntimeError(f"Set {self.prefix}API_KEY in .env or your environment to use --strategy {self.name}")
        self.client = OpenAI(
            base_url=os.getenv(self.prefix + "BASE_URL", self.default_base_url),
            api_key=api_key,
            timeout=float(os.getenv(self.prefix + "TIMEOUT", str(self.default_timeout))),
            max_retries=0,
        )
        return self.client

    def choose(self, grid, hand, moves, player_index):
        ranked = sorted(moves, key=lambda move: move.rank, reverse=True)[: self.max_options]
        options = [move_option(index, move) for index, move in enumerate(ranked)]
        prompt = {
            "role": "You are playing Bandido cooperatively, but players cannot communicate. Choose one legal move.",
            "goal": (
                "Close all open tunnel exits. Prefer moves that reduce open paths, keep exits clustered, "
                "avoid isolated exits, avoid branching, and avoid spreading the tunnel network."
            ),
            "player_index": player_index,
            "visible_hand": [card.shape_num for card in hand],
            "board_summary": grid.board_summary(),
            "legal_options": options,
            "required_response": "Return only JSON like {\"id\": 0}.",
        }
        client = self.get_client()
        create_kwargs = {
            "model": os.getenv(self.prefix + "MODEL", self.default_model),
            "messages": [{"role": "user", "content": json.dumps(prompt)}],
        }
        temperature = os.getenv(self.prefix + "TEMPERATURE")
        if temperature is not None:
            create_kwargs["temperature"] = float(temperature)
        top_p = os.getenv(self.prefix + "TOP_P")
        if top_p is not None:
            create_kwargs["top_p"] = float(top_p)
        max_tokens = os.getenv(self.prefix + "MAX_TOKENS")
        if max_tokens is not None:
            create_kwargs["max_tokens"] = int(max_tokens)
        if self.extra_body_enabled:
            create_kwargs["extra_body"] = {
                "chat_template_kwargs": {
                    "thinking": os.getenv(self.prefix + "THINKING", "false").lower() == "true",
                }
            }
        try:
            response = client.chat.completions.create(**create_kwargs)
            content = response.choices[0].message.content
        except Exception:
            return ranked[0]
        try:
            choice = json.loads(content)
            move_id = int(choice["id"])
            return ranked[move_id]
        except (ValueError, KeyError, TypeError, IndexError, json.JSONDecodeError):
            return ranked[0]


# ponytail: dict replaces build_strategy if-chain; strategies selected by name.
STRATEGIES = {
    "random": lambda _api_options: RandomStrategy(),
    "greedy": lambda _api_options: GreedyStrategy(),
    "compact": lambda _api_options: CompactStrategy(),
    "minimax-api": lambda api_options: OpenAICompatibleApiStrategy(
        name="minimax-api", prefix="MINIMAX_",
        default_base_url="https://api.minimax.io/v1",
        default_model="MiniMax-M2.7", default_timeout=10, max_options=api_options,
    ),
    "nvidia-nim-api": lambda api_options: OpenAICompatibleApiStrategy(
        name="nvidia-nim-api", prefix="NVIDIA_",
        default_base_url="https://integrate.api.nvidia.com/v1",
        default_model="deepseek-ai/deepseek-v4-pro", default_timeout=30,
        max_options=api_options, extra_body_enabled=True,
    ),
}


def build_strategy(name, api_options=12):
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}")
    return STRATEGIES[name](api_options)


def build_deck(pieces_amount, used_codes):
    shape_nums = []
    while len(shape_nums) < pieces_amount:
        for shape_num, amount in DEFAULT_DECK_MANIFEST.items():
            shape_nums.extend([shape_num] * amount)
            if len(shape_nums) >= pieces_amount:
                break
    shape_nums = shape_nums[:pieces_amount]
    random.shuffle(shape_nums)
    return [Piece(shape_num, random_wall_code(used_codes)) for shape_num in shape_nums]


class Player:
    def __init__(self, strategy, index=0):
        self.strategy = strategy
        self.index = index
        self.hand = []

    def draw_to(self, deck, hand_size):
        while len(self.hand) < hand_size and deck:
            self.hand.append(deck.pop(0))

    def exchange_hand(self, deck, hand_size):
        if not deck:
            return []
        discarded = self.hand[:]
        self.hand.clear()
        self.draw_to(deck, hand_size)
        deck.extend(discarded)
        random.shuffle(deck)
        return discarded

    def play(self, grid, verbose=False):
        moves = legal_moves(grid, self.hand)
        if not moves:
            return False
        move = self.strategy.choose(grid, self.hand, moves, self.index)
        grid.insert(move.coord, move.shape)
        self.hand.remove(move.card)
        if verbose:
            print(f"Player {self.index + 1} {self.strategy.name}: {move.label()}, open_paths={len(grid.open_positions)}")
        return True


class Game:
    def __init__(self, players, pieces_amount=69, grid_size=150, hand_size=3, seed=None):
        if seed is not None:
            random.seed(seed)
        self.players = players
        self.hand_size = hand_size
        used_codes = set()
        self.deck = build_deck(pieces_amount, used_codes)
        self.grid = Grid(grid_size)
        self.grid.seed_start()
        for player in self.players:
            player.draw_to(self.deck, self.hand_size)

    def remaining_cards(self):
        cards = list(self.deck)
        for player in self.players:
            cards.extend(player.hand)
        return cards

    def run(self, verbose=False, turn_limit=None):
        turns = 0
        passes = 0
        max_passes = max(1, len(self.players)) * 4

        while any(player.hand for player in self.players):
            for player in self.players:
                if turn_limit is not None and turns >= turn_limit:
                    return {"won": self.grid.is_won(), "turns": turns, "open_paths": len(self.grid.open_positions), "cards_left": len(self.deck)}
                if not player.hand:
                    continue

                turns += 1
                if player.play(self.grid, verbose=verbose):
                    player.draw_to(self.deck, self.hand_size)
                    passes = 0
                    if self.grid.is_won():
                        return {"won": True, "turns": turns, "open_paths": 0, "cards_left": len(self.deck)}
                else:
                    discarded = player.exchange_hand(self.deck, self.hand_size)
                    if verbose:
                        discarded_labels = [card.shape_num for card in discarded]
                        new_hand_labels = [card.shape_num for card in player.hand]
                        print(
                            f"Player {player.index + 1} {player.strategy.name}: no legal move, "
                            f"exchanged {discarded_labels} -> {new_hand_labels}, "
                            f"open_paths={len(self.grid.open_positions)}"
                        )
                    passes += 1
                    if passes >= max_passes:
                        if not has_legal_move(self.grid, self.remaining_cards()):
                            return {"won": False, "turns": turns, "open_paths": len(self.grid.open_positions), "cards_left": len(self.deck)}
                        passes = 0

        return {"won": self.grid.is_won(), "turns": turns, "open_paths": len(self.grid.open_positions), "cards_left": len(self.deck)}


def make_players(count, strategy_name, api_options=12):
    return [Player(build_strategy(strategy_name, api_options=api_options), index=index) for index in range(count)]


def play_one(args):
    started = time.perf_counter()
    game = Game(
        make_players(args.players, args.strategy, api_options=args.api_options),
        pieces_amount=args.pieces,
        grid_size=args.grid_size,
        hand_size=args.hand_size,
        seed=args.seed,
    )
    result = game.run(verbose=args.verbose, turn_limit=args.turn_limit)
    elapsed = time.perf_counter() - started

    if args.output:
        image = game.grid.to_image(scale=args.png_scale)
        image.save(args.output)
        if args.show:
            image.show()

    result["time"] = elapsed
    result["image"] = args.output
    return result


def benchmark(args):
    strategies = args.benchmark_strategies or ["random", "greedy", "compact"]
    for strategy in strategies:
        wins = 0
        open_paths = []
        turns = []
        started = time.perf_counter()
        for index in range(args.games):
            seed = args.seed + index if args.seed is not None else None
            game = Game(
                make_players(args.players, strategy, api_options=args.api_options),
                pieces_amount=args.pieces,
                grid_size=args.grid_size,
                hand_size=args.hand_size,
                seed=seed,
            )
            result = game.run(verbose=False, turn_limit=args.turn_limit)
            wins += int(result["won"])
            open_paths.append(result["open_paths"])
            turns.append(result["turns"])
        elapsed = time.perf_counter() - started
        print(
            f"{strategy}: wins={wins}/{args.games}, "
            f"avg_open_paths={sum(open_paths) / len(open_paths):.1f}, "
            f"avg_turns={sum(turns) / len(turns):.1f}, time={elapsed:.3f}s"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate and benchmark the cooperative Bandido board game.")
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--pieces", type=int, default=69)
    parser.add_argument("--grid-size", type=int, default=150)
    parser.add_argument("--hand-size", type=int, default=3)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--strategy",
        choices=["random", "greedy", "compact", "minimax-api", "nvidia-nim-api"],
        default="compact",
    )
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--turn-limit", type=int)
    parser.add_argument("--api-options", type=int, default=12)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument(
        "--benchmark-strategies",
        nargs="+",
        choices=["random", "greedy", "compact", "minimax-api", "nvidia-nim-api"],
    )
    parser.add_argument("--output", default="bandido.png")
    parser.add_argument("--png-scale", type=int, default=6)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--validate", action="store_true", help="run piece-catalog validation and exit")
    return parser.parse_args()


def main():
    load_env()
    args = parse_args()
    if args.validate:
        validate_piece_catalog()
        print("piece catalog OK")
        return
    if args.benchmark:
        benchmark(args)
        return

    result = play_one(args)
    status = "won" if result["won"] else "lost"
    print(
        f"Game {status} in {result['turns']} turns, "
        f"open paths: {result['open_paths']}, cards left: {result['cards_left']}, "
        f"time: {result['time']:.3f}s, image: {result['image']}"
    )


if __name__ == "__main__":
    main()
