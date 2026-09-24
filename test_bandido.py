"""Offline unit tests for the Bandido simulator (no API calls).

Run from this directory: python -B -m unittest -q test_bandido
"""
import os
import random
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import bandido as b


class CatalogTests(unittest.TestCase):
    def test_catalog_is_valid_and_named(self):
        b.validate_piece_catalog()
        self.assertEqual(set(b.PIECES), set(b.CARD_NAMES))
        self.assertEqual(set(b.PIECES), set(b.DEFAULT_DECK_MANIFEST))
        for shape in b.PIECES.values():
            self.assertEqual(np.array(shape).shape, (3, 6))

    def test_enders_have_one_torch(self):
        for index, (_name, shape) in enumerate(b.ENDER_CARDS, start=len(b.BASIC_CARD_PORTS) + 1):
            shape = np.array(shape)
            self.assertEqual(np.count_nonzero(shape == b.TORCH), 1, index)

    def test_invalid_torch_detected(self):
        bad = np.full((3, 6), b.WALL)
        bad[1, 1:5] = b.PATH
        bad[0, 0] = b.TORCH  # torch touching nothing
        with mock.patch.dict(b.PIECES, {"bad": bad.tolist()}):
            with self.assertRaisesRegex(ValueError, "torch"):
                b.validate_piece_catalog()

    def test_edge_paths_only_at_connectors(self):
        ports = set(b.PORT_COORDS.values())
        for key, shape in b.PIECES.items():
            shape = np.array(shape)
            for row, column in np.argwhere(shape == b.PATH):
                if row in (0, 2) or column in (0, 5):
                    self.assertIn((int(row), int(column)), ports, b.CARD_NAMES[key])

    def test_off_connector_edge_detected(self):
        bad = np.array(b.piece_from_ports((5, 2)))
        bad[0, 2] = b.PATH  # opening between connectors can never be matched
        with mock.patch.dict(b.PIECES, {"bad": bad.tolist()}):
            with self.assertRaisesRegex(ValueError, "connector"):
                b.validate_piece_catalog()

    def test_piece_from_ports_opens_requested_edges(self):
        shape = np.array(b.piece_from_ports((0, 2)))
        self.assertEqual(shape[0, 1], b.PATH)
        self.assertEqual(shape[1, 5], b.PATH)
        self.assertEqual(shape[2, 1], b.WALL)


class PieceTests(unittest.TestCase):
    def test_wall_code_rendering_and_rotations(self):
        piece = b.Piece("1", 1234)
        base = piece.base_shape
        self.assertTrue(set(np.unique(base)) <= {b.PATH, 1234})
        rotations = piece.rotations()
        self.assertEqual([r for r, _ in rotations], [0, 90, 180, 270][:len(rotations)])
        keys = {tuple(s.ravel()) for _, s in rotations}
        self.assertEqual(len(keys), len(rotations))

    def test_symmetric_piece_deduplicates_rotations(self):
        symmetric = b.Piece("1", 1111)
        with mock.patch.dict(b.PIECES, {"1": b.START_PIECE.tolist()}):
            self.assertEqual(len(symmetric.rotations()), 2)

    def test_wall_codes_unique_and_not_multiple_of_ten(self):
        random.seed(0)
        used = set()
        codes = [b.random_wall_code(used) for _ in range(300)]
        self.assertEqual(len(set(codes)), 300)
        self.assertTrue(all(c % 10 and 1000 <= c <= 9999 for c in codes))

    def test_build_deck_follows_manifest(self):
        random.seed(0)
        deck = b.build_deck(69, set())
        self.assertEqual(len(deck), 69)
        counts = {k: sum(p.shape_num == k for p in deck) for k in b.DEFAULT_DECK_MANIFEST}
        self.assertEqual(counts, b.DEFAULT_DECK_MANIFEST)
        self.assertEqual(len({p.wall_code for p in deck}), 69)
        self.assertEqual(len(b.build_deck(5, set())), 5)


class GridTests(unittest.TestCase):
    def setUp(self):
        random.seed(1)
        self.grid = b.Grid(60)
        self.grid.seed_start()

    def test_start_piece_opens_six_exits(self):
        self.assertEqual(len(self.grid.open_positions), 6)
        self.assertFalse(self.grid.is_won())
        summary = self.grid.board_summary()
        self.assertEqual(summary["occupied_cells"], 18)

    def test_empty_grid_summary_and_win(self):
        grid = b.Grid(12)
        self.assertTrue(grid.is_won())
        self.assertEqual(grid.board_summary(), {"open_paths": 0, "occupied_cells": 0, "bounds": None})

    def test_candidates_are_aligned_and_in_bounds(self):
        shape = b.Piece("1", 1234).base_shape
        for coord in self.grid.candidate_positions(shape):
            self.assertEqual((coord[0] % 3, coord[2] % 3), (0, 0))
            self.assertTrue(self.grid.has_bounds(coord, margin=4))

    def test_cannot_place_on_occupied_cells(self):
        rows, cols = np.where(self.grid.grid != b.EMPTY)
        top, left = int(rows.min()), int(cols.min())
        shape = b.Piece("1", 1234).base_shape
        coord = (top, top + shape.shape[0], left, left + shape.shape[1])
        self.assertFalse(self.grid.check_place(coord, shape, 1234))

    def test_legal_moves_connect_and_are_placeable(self):
        hand = [b.Piece(k, 1000 + int(k) * 7) for k in ("1", "7", "11")]
        moves = b.legal_moves(self.grid, hand)
        self.assertTrue(moves)
        self.assertTrue(b.has_legal_move(self.grid, hand))
        for move in moves:
            self.assertGreaterEqual(move.features["connected_exits"], 1)
            self.assertEqual(move.score, move.features["open_delta"])
            self.assertEqual(move.rank, b.strategic_rank(move.features))

    def test_torch_move_reduces_open_paths(self):
        hand = [b.Piece(str(k), 2001 + k) for k in range(7, 14)]
        moves = b.legal_moves(self.grid, hand)
        before = len(self.grid.open_positions)
        best = max(moves, key=lambda m: m.score)
        self.grid.insert(best.coord, best.shape)
        self.assertEqual(len(self.grid.open_positions), before - best.score)
        self.assertLess(len(self.grid.open_positions), before)

    def test_projection_matches_real_insert(self):
        moves = b.legal_moves(self.grid, [b.Piece("2", 1357)])
        move = moves[0]
        projected = self.grid.project_open_positions(move.coord, move.shape)
        self.grid.insert(move.coord, move.shape)
        self.assertEqual(projected, self.grid.open_positions)

    def test_image_render(self):
        image = self.grid.to_image(scale=2)
        self.assertEqual(image.size, (120, 120))

    def test_image_crop_to_board_with_margin(self):
        rows, cols = np.where(self.grid.grid != b.EMPTY)
        height = int(rows.max() - rows.min()) + 1 + 2 * 3
        width = int(cols.max() - cols.min()) + 1 + 2 * 3
        image = self.grid.to_image(scale=2, crop=3)
        self.assertEqual(image.size, (width * 2, height * 2))
        self.assertEqual(b.Grid(12).to_image(scale=1, crop=3).size, (12, 12))


class StrategyTests(unittest.TestCase):
    def moves(self):
        random.seed(2)
        grid = b.Grid(60)
        grid.seed_start()
        return grid, b.legal_moves(grid, [b.Piece(str(k), 3001 + k) for k in range(1, 14)])

    def test_greedy_and_compact_pick_best(self):
        grid, moves = self.moves()
        greedy = b.GreedyStrategy().choose(grid, [], moves, 0)
        self.assertEqual(greedy.score, max(m.score for m in moves))
        compact = b.CompactStrategy().choose(grid, [], moves, 0)
        self.assertEqual(compact.rank, max(m.rank for m in moves))

    def test_build_strategy(self):
        self.assertIsInstance(b.build_strategy("greedy"), b.GreedyStrategy)
        self.assertEqual(b.build_strategy("minimax-api", api_options=5).max_options, 5)
        with self.assertRaises(ValueError):
            b.build_strategy("nope")

    def test_api_strategy_parses_choice_and_falls_back(self):
        grid, moves = self.moves()
        ranked = sorted(moves, key=lambda m: m.rank, reverse=True)
        strategy = b.build_strategy("nvidia-nim-api", api_options=4)

        def reply(content):
            message = SimpleNamespace(content=content)
            create = mock.Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=message)]))
            strategy.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
            return create

        create = reply('{"id": 1}')
        self.assertIs(strategy.choose(grid, [], moves, 0), ranked[1])
        self.assertIn("extra_body", create.call_args.kwargs)
        for bad in ("not json", '{"id": 99}', '{"id": -1}', '{"x": 1}'):
            reply(bad)
            self.assertIs(strategy.choose(grid, [], moves, 0), ranked[0])
        strategy.client.chat.completions.create.side_effect = RuntimeError("down")
        self.assertIs(strategy.choose(grid, [], moves, 0), ranked[0])

    def test_api_strategy_requires_key(self):
        strategy = b.build_strategy("minimax-api")
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, "API_KEY"):
            strategy.get_client()


class GameTests(unittest.TestCase):
    def test_cli_image_is_cropped_unless_full_board(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            sizes = {}
            for full in (False, True):
                path = os.path.join(tmp, f"{full}.png")
                args = SimpleNamespace(players=2, strategy="compact", api_options=12, pieces=10,
                                       grid_size=60, hand_size=3, seed=3, verbose=False, turn_limit=5,
                                       output=path, png_scale=1, show=False, full_board=full)
                b.play_one(args)
                with Image.open(path) as image:
                    sizes[full] = image.size
        self.assertEqual(sizes[True], (60, 60))
        self.assertLess(sizes[False][0] * sizes[False][1], 60 * 60)

    def test_seeded_game_is_reproducible_and_consistent(self):
        def play():
            game = b.Game(b.make_players(2, "compact"), pieces_amount=20, grid_size=60, seed=5)
            return game.run(turn_limit=40), len(game.remaining_cards())
        (first, left), (second, _) = play(), play()
        self.assertEqual(first, second)
        self.assertLessEqual(first["turns"], 40)
        self.assertEqual(first["won"], first["open_paths"] == 0)

    def test_exchange_hand_keeps_cards(self):
        random.seed(0)
        deck = b.build_deck(10, set())
        player = b.Player(b.RandomStrategy())
        player.draw_to(deck, 3)
        before = {p.wall_code for p in deck} | {p.wall_code for p in player.hand}
        player.exchange_hand(deck, 3)
        after = {p.wall_code for p in deck} | {p.wall_code for p in player.hand}
        self.assertEqual(before, after)
        self.assertEqual(len(player.hand), 3)
        self.assertEqual(b.Player(b.RandomStrategy()).exchange_hand([], 3), [])


class EnvTests(unittest.TestCase):
    def test_load_env_does_not_override_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, ".env")
            with open(path, "w", encoding="utf-8") as f:
                f.write("# comment\nBANDIDO_A='one'\nBANDIDO_B=two\nnoequals\n")
            with mock.patch.dict(os.environ, {"BANDIDO_B": "kept"}):
                b.load_env(path)
                self.assertEqual(os.environ["BANDIDO_A"], "one")
                self.assertEqual(os.environ["BANDIDO_B"], "kept")
        b.load_env(os.path.join("missing", ".env"))


if __name__ == "__main__":
    unittest.main()
