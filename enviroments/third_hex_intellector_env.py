import numpy as np
import gymnasium
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from intellector.moves import POSSIBLE_MOVES
import intellector.pieces as pieces
import intellector.third_rewards as rewards_mod
import intellector.status as status


class ThirdHexIntellectorEnv(gymnasium.Env):

    metadata: dict = {
        "render_mode": "human",
    }

    def __init__(self, max_steps: int = 128, render_mode: str = "human") -> None:
        self.space_size = (
            POSSIBLE_MOVES["progressor"] * 5
            + POSSIBLE_MOVES["intellector"]
            + POSSIBLE_MOVES["agressor"] * 2
            + POSSIBLE_MOVES["dominator"] * 2
            + POSSIBLE_MOVES["defensor"] * 2
            + POSSIBLE_MOVES["liberator"] * 2
        )
        self.action_space = spaces.Discrete(self.space_size)
        self.grid_height = 7
        self.grid_width = 10
        self.hex_size = 1
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(self.grid_height, self.grid_width, 1), dtype=np.int8
        )

        self.board: np.ndarray = self.init_board()
        self.pieces: list[dict] = self.init_pieces()
        self.pieces_names: list[str] = self.get_pieces_names()
        self.turn: int = pieces.WHITE
        self.done: bool = False
        self.steps: int = 0
        self.checked: list[bool] = [False, False]
        self.max_steps: int = max_steps
        self.rewards: list[int] = [rewards_mod.DRAW, rewards_mod.DRAW]
        self.base_line_black = [(0, 0), (0, 2), (0, 4), (0, 6), (0, 8)]
        self.base_line_white = [(6, 0), (6, 2), (6, 4), (6, 6), (6, 8)]
        self.winner = status.NO_WINNER

        self.FIGURE_SYMBOLS = {
            pieces.PROGRESSOR: "p",
            pieces.INTELLECTOR: "i",
            pieces.LIBERATOR: "l",
            pieces.DOMINATOR: "do",
            pieces.AGRESSOR: "a",
            pieces.DEFENSOR: "de",
        }

        self.render_mode: str = render_mode

    @staticmethod
    def init_board() -> np.ndarray:
        board = np.zeros((2, 7, 10), dtype=np.uint8)

        board[pieces.WHITE, 1, [0, 2, 4, 6, 8]] = pieces.PROGRESSOR
        board[pieces.WHITE, 0, 4] = pieces.INTELLECTOR
        board[pieces.WHITE, 0, [1, 7]] = pieces.LIBERATOR
        board[pieces.WHITE, 0, [0, 8]] = pieces.DOMINATOR
        board[pieces.WHITE, 0, [2, 6]] = pieces.AGRESSOR
        board[pieces.WHITE, 0, [3, 5]] = pieces.DEFENSOR

        board[pieces.BLACK, 5, [0, 2, 4, 6, 8]] = pieces.PROGRESSOR
        board[pieces.BLACK, 6, 4] = pieces.INTELLECTOR
        board[pieces.BLACK, 5, [1, 7]] = pieces.LIBERATOR
        board[pieces.BLACK, 6, [0, 8]] = pieces.DOMINATOR
        board[pieces.BLACK, 6, [2, 6]] = pieces.AGRESSOR
        board[pieces.BLACK, 5, [3, 5]] = pieces.DEFENSOR

        return board

    def get_state(self, turn: int) -> np.ndarray:
        return self.board[turn].flatten()

    @staticmethod
    def init_pieces():
        pieces_white = {
            "progressor_1": (1, 0),
            "progressor_2": (1, 2),
            "progressor_3": (1, 4),
            "progressor_4": (1, 6),
            "progressor_5": (1, 8),
            "intellector": (0, 4),
            "dominator_1": (0, 0),
            "dominator_2": (0, 8),
            "agressor_1": (0, 2),
            "agressor_2": (0, 6),
            "defensor_1": (0, 3),
            "defensor_2": (0, 5),
            "liberator_1": (0, 1),
            "liberator_2": (0, 7),
        }

        pieces_black = {
            "progressor_1": (5, 0),
            "progressor_2": (5, 2),
            "progressor_3": (5, 4),
            "progressor_4": (5, 6),
            "progressor_5": (5, 8),
            "intellector": (6, 4),
            "dominator_1": (6, 0),
            "dominator_2": (6, 8),
            "agressor_1": (6, 2),
            "agressor_2": (6, 6),
            "defensor_1": (5, 3),
            "defensor_2": (5, 5),
            "liberator_1": (5, 1),
            "liberator_2": (5, 7),
        }

        return [pieces_white.copy(), pieces_black.copy()]

    def is_valid_move(self, next_pos):
        next_row, next_col = next_pos

        excluded_positions = [(6, 1), (6, 3), (6, 5), (6, 7)]
        if (
            0 <= next_row < self.grid_height
            and 0 <= next_col < self.grid_width - 1
            and (next_row, next_col) not in excluded_positions
        ):
            return True

        return False

    @staticmethod
    def adjacent_hexes(row, col):
        if col % 2 == 0:
            adjacent_hexes = [
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
                (row - 1, col),
                (row - 1, col + 1),
                (row - 1, col - 1),
            ]
        else:
            adjacent_hexes = [
                (row + 1, col),
                (row + 1, col - 1),
                (row + 1, col + 1),
                (row - 1, col),
                (row, col + 1),
                (row, col - 1),
            ]
        return adjacent_hexes

    def is_adjacent_move(self, current_pos, next_pos) -> bool:
        current_row, current_col = current_pos
        adjacent_positions = self.adjacent_hexes(current_row, current_col)
        return next_pos in adjacent_positions

    def get_pieces_names(self) -> list[str]:
        return list(self.pieces[0].keys())

    def is_game_done(self):
        white_intellector_present = np.any(
            self.board[pieces.WHITE] == pieces.INTELLECTOR
        )
        black_intellector_present = np.any(
            self.board[pieces.BLACK] == pieces.INTELLECTOR
        )

        if white_intellector_present:
            pos_white = tuple(
                np.argwhere(self.board[pieces.WHITE] == pieces.INTELLECTOR)[0]
            )
            if pos_white in self.base_line_white:
                self.done = True
                self.rewards = [rewards_mod.WIN_REWARD, rewards_mod.LOSE_REWARD]
                self.winner = status.WHITE_WINNER
                return self.done, self.rewards

        if black_intellector_present:
            pos_black = tuple(
                np.argwhere(self.board[pieces.BLACK] == pieces.INTELLECTOR)[0]
            )
            if pos_black in self.base_line_black:
                self.done = True
                self.rewards = [rewards_mod.LOSE_REWARD, rewards_mod.WIN_REWARD]
                self.winner = status.BLACK_WINNER
                return self.done, self.rewards

        _, _, mask_w = self.get_all_actions(pieces.WHITE)
        _, _, mask_b = self.get_all_actions(pieces.BLACK)

        if (not white_intellector_present) or (mask_w.sum() == 0):
            self.done = True
            self.rewards = [rewards_mod.LOSE_REWARD, rewards_mod.WIN_REWARD]
            self.winner = status.BLACK_WINNER
        elif (not black_intellector_present) or (mask_b.sum() == 0):
            self.done = True
            self.rewards = [rewards_mod.WIN_REWARD, rewards_mod.LOSE_REWARD]
            self.winner = status.WHITE_WINNER
        elif self.steps >= self.max_steps:
            self.done = True
            self.rewards = [rewards_mod.DRAW_REWARD, rewards_mod.DRAW_REWARD]
            self.winner = status.DRAW_WINNER

        return self.done, self.rewards

    def is_empty(self, pos, turn: int) -> bool:
        return self.board[turn, pos[0], pos[1]] == pieces.EMPTY

    def is_enemy(self, pos, turn: int) -> bool:
        return self.board[1 - turn, pos[0], pos[1]] != pieces.EMPTY

    def is_defensor(self, pos, turn: int) -> bool:
        return self.board[turn, pos[0], pos[1]] == pieces.DEFENSOR

    def is_intellector(self, pos, turn: int) -> bool:
        return self.board[turn, pos[0], pos[1]] == pieces.INTELLECTOR

    @staticmethod
    def get_size(name: str):
        size = POSSIBLE_MOVES[name]
        return size

    def get_empty_actions(self, name: str):
        size = self.get_size(name)
        possibles = np.zeros((size, 2), dtype=np.int32)
        actions_mask = np.zeros(size, dtype=np.int32)
        source_pos = np.zeros((size, 2), dtype=np.int32)

        return source_pos, possibles, actions_mask

    def is_in_range(self, pos) -> bool:
        row, col = pos
        excluded_positions = [(6, 1), (6, 3), (6, 5), (6, 7)]
        return (
            0 <= row < self.grid_height
            and 0 <= col < self.grid_width - 1
            and (row, col) not in excluded_positions
        )

    def get_actions_for_progressor(self, pos, turn: int):
        if pos is None:
            return self.get_empty_actions("progressor")

        row, col = pos
        source_pos, possibles, actions_mask = self.get_empty_actions("progressor")

        if turn == pieces.WHITE:
            if col % 2 == 0:
                progressor_moves = [(row + 1, col), (row, col - 1), (row, col + 1)]
            else:
                progressor_moves = [
                    (row + 1, col),
                    (row + 1, col - 1),
                    (row + 1, col + 1),
                ]
        else:
            if col % 2 == 0:
                progressor_moves = [
                    (row - 1, col),
                    (row - 1, col + 1),
                    (row - 1, col - 1),
                ]
            else:
                progressor_moves = [(row - 1, col), (row, col + 1), (row, col - 1)]

        for i, (r, c) in enumerate(progressor_moves):
            next_pos = (r, c)
            if not self.is_in_range(next_pos) or not self.is_empty(next_pos, turn):
                continue
            source_pos[i] = pos
            possibles[i] = next_pos
            actions_mask[i] = 1

        return source_pos, possibles, actions_mask

    def get_actions_for_agressor(self, pos, turn: int):
        source_pos, possibles, actions_mask = self.get_empty_actions("agressor")

        if pos is None:
            return source_pos, possibles, actions_mask

        row, col = pos
        index = 0

        # Horizontal moves to the right
        for i in range(col + 2, self.grid_width, 2):
            next_pos = (row, i)
            if not self.is_valid_move(next_pos):
                break
            if self.is_enemy(next_pos, turn):
                possibles[index] = next_pos
                source_pos[index] = pos
                actions_mask[index] = 1
                index += 1
                break
            elif self.is_empty(next_pos, turn):
                possibles[index] = next_pos
                source_pos[index] = pos
                actions_mask[index] = 1
                index += 1
            else:
                break

        # Horizontal moves to the left
        for i in range(col - 2, -1, -2):
            next_pos = (row, i)
            if not self.is_valid_move(next_pos):
                break
            if self.is_enemy(next_pos, turn):
                possibles[index] = next_pos
                source_pos[index] = pos
                actions_mask[index] = 1
                index += 1
                break
            elif self.is_empty(next_pos, turn):
                possibles[index] = next_pos
                source_pos[index] = pos
                actions_mask[index] = 1
                index += 1
            else:
                break

        # Diagonal moves down-left
        new_row, new_col = row, col
        while new_row < self.grid_height:
            if new_col % 2 == 0:
                new_col -= 1
                new_row += 1
            else:
                new_col -= 1
                new_row += 2
            next_pos = (new_row, new_col)
            if not self.is_valid_move(next_pos):
                break
            if self.is_enemy(next_pos, turn):
                possibles[index] = next_pos
                source_pos[index] = pos
                actions_mask[index] = 1
                index += 1
                break
            elif self.is_empty(next_pos, turn):
                possibles[index] = next_pos
                source_pos[index] = pos
                actions_mask[index] = 1
                index += 1
            else:
                break

        # Diagonal moves down-right
        new_row, new_col = row, col
        while new_row < self.grid_height:
            if new_col % 2 == 0:
                new_col += 1
                new_row += 1
            else:
                new_col += 1
                new_row += 2
            next_pos = (new_row, new_col)
            if not self.is_valid_move(next_pos):
                break
            if self.is_enemy(next_pos, turn):
                possibles[index] = next_pos
                source_pos[index] = pos
                actions_mask[index] = 1
                index += 1
                break
            elif self.is_empty(next_pos, turn):
                possibles[index] = next_pos
                source_pos[index] = pos
                actions_mask[index] = 1
                index += 1
            else:
                break

        # Diagonal moves up-left
        new_row, new_col = row, col
        while new_row > 0:
            if new_col % 2 == 0:
                new_col -= 1
                new_row -= 2
            else:
                new_col -= 1
                new_row -= 1
            next_pos = (new_row, new_col)
            if not self.is_valid_move(next_pos):
                break
            if self.is_enemy(next_pos, turn):
                possibles[index] = next_pos
                source_pos[index] = pos
                actions_mask[index] = 1
                index += 1
                break
            elif self.is_empty(next_pos, turn):
                possibles[index] = next_pos
                source_pos[index] = pos
                actions_mask[index] = 1
                index += 1
            else:
                break

        # Diagonal moves up-right
        new_row, new_col = row, col
        while new_row > 0:
            if new_col % 2 == 0:
                new_col += 1
                new_row -= 2
            else:
                new_col += 1
                new_row -= 1
            next_pos = (new_row, new_col)
            if not self.is_valid_move(next_pos):
                break
            if self.is_enemy(next_pos, turn):
                possibles[index] = next_pos
                source_pos[index] = pos
                actions_mask[index] = 1
                index += 1
                break
            elif self.is_empty(next_pos, turn):
                possibles[index] = next_pos
                source_pos[index] = pos
                actions_mask[index] = 1
                index += 1
            else:
                break

        return source_pos, possibles, actions_mask

    def get_actions_for_intellector(self, pos, turn: int):
        source_pos, possibles, actions_mask = self.get_empty_actions("intellector")

        if pos is None:
            return source_pos, possibles, actions_mask

        row, col = pos
        intellector_moves = self.adjacent_hexes(row, col)
        for i, (r, c) in enumerate(intellector_moves):
            next_pos = (r, c)
            if not self.is_valid_move(next_pos):
                continue
            if self.is_enemy(next_pos, turn):
                continue
            if self.is_defensor(next_pos, turn) or self.is_empty(next_pos, turn):
                possibles[i] = next_pos
                source_pos[i] = pos
                actions_mask[i] = 1

        return source_pos, possibles, actions_mask

    def get_actions_for_defensor(self, pos, turn: int):
        source_pos, possibles, actions_mask = self.get_empty_actions("defensor")

        if pos is None:
            return source_pos, possibles, actions_mask

        row, col = pos
        defensor_moves = self.adjacent_hexes(row, col)
        for i, (r, c) in enumerate(defensor_moves):
            next_pos = (r, c)
            if self.is_valid_move(next_pos):
                if self.is_intellector(next_pos, turn):
                    possibles[i] = next_pos
                    source_pos[i] = pos
                    actions_mask[i] = 1
                elif self.is_empty(next_pos, turn):
                    possibles[i] = next_pos
                    source_pos[i] = pos
                    actions_mask[i] = 1

        return source_pos, possibles, actions_mask

    def get_actions_for_liberator(self, pos, turn: int):
        source_pos, possibles, actions_mask = self.get_empty_actions("liberator")
        if pos is None:
            return source_pos, possibles, actions_mask

        row, col = pos
        liberator_moves = self.adjacent_hexes(row, col)
        for i, (r, c) in enumerate(liberator_moves):
            next_pos = (r, c)
            if (
                not self.is_valid_move(next_pos)
                or self.board[turn, r, c] != pieces.EMPTY
                or self.board[1 - turn, r, c] != pieces.EMPTY
            ):
                continue

            possibles[i] = next_pos
            source_pos[i] = pos
            actions_mask[i] = 1

        moves = [
            (row + 2, col),
            (row - 2, col),
            (row + 1, col + 2),
            (row - 1, col + 2),
            (row + 1, col - 2),
            (row - 1, col - 2),
        ]

        for i, (r, c) in enumerate(moves):
            next_pos = (r, c)
            if not self.is_valid_move(next_pos) or not self.is_empty(next_pos, turn):
                continue

            possibles[i + 6] = next_pos
            source_pos[i + 6] = pos
            actions_mask[i + 6] = 1

        return source_pos, possibles, actions_mask

    def get_actions_for_dominator(self, pos, turn: int):
        source_pos, possibles, actions_mask = self.get_empty_actions("dominator")
        if pos is None:
            return source_pos, possibles, actions_mask

        row, col = pos
        index = 0  # Индекс для отслеживания позиции в массивах possibles и actions_mask

        # Vertical moves downward
        for i in range(row + 1, self.grid_height):
            next_pos = (i, col)
            if not self.is_valid_move(next_pos):
                break
            if self.is_enemy(next_pos, turn):
                possibles[index] = next_pos
                source_pos[index] = pos
                actions_mask[index] = 1
                index += 1
                break
            elif self.is_empty(next_pos, turn):
                possibles[index] = next_pos
                source_pos[index] = pos
                actions_mask[index] = 1
                index += 1
            else:
                break

        # Vertical moves upward
        for i in range(row - 1, -1, -1):
            next_pos = (i, col)
            if not self.is_valid_move(next_pos):
                break
            if self.is_enemy(next_pos, turn):
                possibles[index] = next_pos
                source_pos[index] = pos
                actions_mask[index] = 1
                index += 1
                break
            elif self.is_empty(next_pos, turn):
                possibles[index] = next_pos
                source_pos[index] = pos
                actions_mask[index] = 1
                index += 1
            else:
                break

        # Diagonal moves to the left and down
        new_col = col
        new_row = row
        while new_col > 0:
            if new_col % 2 == 0:
                next_pos = (new_row, new_col - 1)
            else:
                next_pos = (new_row + 1, new_col - 1)

            if not self.is_valid_move(next_pos):
                break
            if self.is_enemy(next_pos, turn):
                possibles[index] = next_pos
                source_pos[index] = pos
                actions_mask[index] = 1
                index += 1
                break
            elif self.is_empty(next_pos, turn):
                possibles[index] = next_pos
                source_pos[index] = pos
                actions_mask[index] = 1
                index += 1
            else:
                break

            if new_col % 2 != 0:
                new_row += 1
            new_col -= 1

        # Diagonal moves to the left and up
        new_col = col
        new_row = row
        while new_col > 0:
            if new_col % 2 == 0:
                next_pos = (new_row - 1, new_col - 1)
            else:
                next_pos = (new_row, new_col - 1)

            if not self.is_valid_move(next_pos):
                break
            if self.is_enemy(next_pos, turn):
                possibles[index] = next_pos
                source_pos[index] = pos
                actions_mask[index] = 1
                index += 1
                break
            elif self.is_empty(next_pos, turn):
                possibles[index] = next_pos
                source_pos[index] = pos
                actions_mask[index] = 1
                index += 1
            else:
                break

            if new_col % 2 == 0:
                new_row -= 1
            new_col -= 1

        # Diagonal moves to the right and down
        new_col = col
        new_row = row
        while new_col < self.grid_width - 1:
            if new_col % 2 == 0:
                next_pos = (new_row, new_col + 1)
            else:
                next_pos = (new_row + 1, new_col + 1)

            if not self.is_valid_move(next_pos):
                break
            if self.is_enemy(next_pos, turn):
                possibles[index] = next_pos
                source_pos[index] = pos
                actions_mask[index] = 1
                index += 1
                break
            elif self.is_empty(next_pos, turn):
                possibles[index] = next_pos
                source_pos[index] = pos
                actions_mask[index] = 1
                index += 1
            else:
                break

            if new_col % 2 != 0:
                new_row += 1
            new_col += 1

        # Diagonal moves to the right and up
        new_col = col
        new_row = row
        while new_col < self.grid_width - 1:
            if new_col % 2 == 0:
                next_pos = (new_row - 1, new_col + 1)
            else:
                next_pos = (new_row, new_col + 1)

            if not self.is_valid_move(next_pos):
                break
            if self.is_enemy(next_pos, turn):
                possibles[index] = next_pos
                source_pos[index] = pos
                actions_mask[index] = 1
                index += 1
                break
            elif self.is_empty(next_pos, turn):
                possibles[index] = next_pos
                source_pos[index] = pos
                actions_mask[index] = 1
                index += 1
            else:
                break

            if new_col % 2 == 0:
                new_row -= 1
            new_col += 1

        return source_pos, possibles, actions_mask

    def reset(self, *, seed=None, options=None) -> np.ndarray:
        self.done = False
        self.turn = 0
        self.steps = 0
        self.board = self.init_board()
        self.pieces = self.init_pieces()
        self.checked = [False, False]
        self.rewards = [rewards_mod.DRAW, rewards_mod.DRAW]
        self.winner = status.NO_WINNER
        state = self.get_state(self.turn)
        return state

    @staticmethod
    def choose_color(row, col):
        color_peru = [
            (1, 1),
            (1, 3),
            (1, 5),
            (1, 7),
            (1, 9),
            (2, 2),
            (2, 4),
            (2, 6),
            (2, 8),
            (4, 1),
            (4, 3),
            (4, 5),
            (4, 7),
            (4, 9),
            (5, 2),
            (5, 4),
            (5, 6),
            (5, 8),
            (7, 1),
            (7, 3),
            (7, 5),
            (7, 7),
            (7, 9),
        ]
        return (row, col) in color_peru

    def get_source_pos(self, name: str, turn: int):
        cat = name.split("_")[0]
        pos = self.pieces[turn][name]
        if pos is None:
            pos = (0, 0)
        size = self.get_size(cat)
        return np.array([pos] * size)

    def render(self, mode="human"):
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        ax.set_aspect("equal")
        ax.axis("off")

        col_letters = self.get_col_letters()

        dx, dy = self.get_hex_offsets()

        exclude_indices = [(7, 2), (7, 4), (7, 6), (7, 8)]

        for col in range(1, self.grid_width):
            for original_row in range(self.grid_height, 0, -1):
                row = self.grid_height - original_row + 1
                if (row, col) in exclude_indices:
                    continue

                x, y = self.calculate_hex_position(dx, dy, col, original_row)

                color = "peru" if self.choose_color(row, col) else "moccasin"
                self.draw_hexagon(ax, (x, y), self.hex_size, color)

                self.draw_piece_from_board(ax, row, col, x, y, col_letters)

        self.set_plot_limits(ax, dx, dy)
        self.display_plot(mode)

    def get_col_letters(self):
        return {
            i: chr(65 + i - 1) for i in range(1, self.grid_width)
        }  # A = 65 in ASCII

    def get_hex_offsets(self):
        dx = 3 / 2 * self.hex_size
        dy = np.sqrt(3) * self.hex_size
        return dx, dy

    @staticmethod
    def calculate_hex_position(dx, dy, col, original_row):
        x = dx * col
        y = dy * original_row + (col % 2) * dy / 2
        return x, y

    @staticmethod
    def draw_hexagon(ax, center, size, color):
        hexagon = patches.RegularPolygon(
            center,
            numVertices=6,
            radius=size,
            orientation=np.radians(30),
            facecolor=color,
            edgecolor="sienna",
        )
        ax.add_patch(hexagon)

    def draw_piece_from_board(self, ax, row, col, x, y, col_letters):
        white_figure = self.board[0, row - 1, col - 1]
        black_figure = self.board[1, row - 1, col - 1]

        if white_figure > 0:
            color_prefix = "w"
            symbol = self.FIGURE_SYMBOLS[white_figure]
            ax.text(
                x,
                y,
                f"{row}{col_letters[col]}{color_prefix}{symbol}",
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )
        elif black_figure > 0:
            color_prefix = "b"
            symbol = self.FIGURE_SYMBOLS[black_figure]
            ax.text(
                x,
                y,
                f"{row}{col_letters[col]}{color_prefix}{symbol}",
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )
        else:
            ax.text(
                x,
                y,
                f"{row}{col_letters[col]}",
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )

    def set_plot_limits(self, ax, dx, dy):
        ax.set_xlim([0, dx * self.grid_width])
        ax.set_ylim([0, dy * (self.grid_height + 2)])
        ax.invert_yaxis()

    @staticmethod
    def display_plot(mode):
        if mode == "human":
            plt.show()

    def get_actions_for(self, name: str, turn: int):
        assert name in self.pieces_names, f"name not in {self.pieces_names}"
        piece_cat = name.split("_")[0]
        piece_pos = self.pieces[turn][name]

        if piece_cat == "intellector":
            src_poses, possibles, actions_mask = self.get_actions_for_intellector(
                piece_pos, turn
            )
            return src_poses, possibles, actions_mask

        if piece_cat == "progressor":
            src_poses, possibles, actions_mask = self.get_actions_for_progressor(
                piece_pos, turn
            )
            return src_poses, possibles, actions_mask

        if piece_cat == "dominator":
            src_poses, possibles, actions_mask = self.get_actions_for_dominator(
                piece_pos, turn
            )
            return src_poses, possibles, actions_mask

        if piece_cat == "agressor":
            src_poses, possibles, actions_mask = self.get_actions_for_agressor(
                piece_pos, turn
            )
            return src_poses, possibles, actions_mask

        if piece_cat == "defensor":
            src_poses, possibles, actions_mask = self.get_actions_for_defensor(
                piece_pos, turn
            )
            return src_poses, possibles, actions_mask

        if piece_cat == "liberator":
            src_poses, possibles, actions_mask = self.get_actions_for_liberator(
                piece_pos, turn
            )
            return src_poses, possibles, actions_mask

    def get_all_actions(self, turn: int):
        all_possibles = []
        all_source_pos = []
        all_actions_mask = []
        for name in self.pieces[turn].keys():
            src_pos, poss, actions = self.get_actions_for(name, turn)
            all_source_pos.extend(src_pos)
            all_possibles.extend(poss)
            all_actions_mask.extend(actions)

        return (
            np.array(all_source_pos),
            np.array(all_possibles),
            np.array(all_actions_mask),
        )

    def promote_progressor(self, pos, turn: int):
        row, col = pos
        if self.board[turn, pos[0], pos[1]] == pieces.PROGRESSOR and (
            (turn == pieces.WHITE and pos in self.base_line_white)
            or (turn == pieces.BLACK and pos in self.base_line_black)
        ):
            self.board[turn, row, col] = pieces.DOMINATOR

    def swap_intellector_defensor(self, current_pos, next_pos, turn: int):
        current_row, current_col = current_pos
        next_row, next_col = next_pos

        # Проверка, что фигуры действительно INTELLECTOR и DEFENSOR
        if (
            self.board[turn, current_row, current_col] == pieces.INTELLECTOR
            and self.board[turn, next_row, next_col] == pieces.DEFENSOR
        ) or (
            self.board[turn, current_row, current_col] == pieces.DEFENSOR
            and self.board[turn, next_row, next_col] == pieces.INTELLECTOR
        ):

            (
                self.board[turn, current_row, current_col],
                self.board[turn, next_row, next_col],
            ) = (
                self.board[turn, next_row, next_col],
                self.board[turn, current_row, current_col],
            )

            for key, value in self.pieces[turn].items():
                if isinstance(value, tuple):
                    if value[0] == current_row and value[1] == current_col:
                        self.pieces[turn][key] = next_pos
                    elif value[0] == next_row and value[1] == next_col:
                        self.pieces[turn][key] = current_pos

            rewards = [rewards_mod.DRAW, rewards_mod.DRAW]
            return True, rewards, [set(), set()]

        return False, None, None

    def move_piece(self, current_pos, next_pos, turn):
        curr_r, curr_c = int(current_pos[0]), int(current_pos[1])
        next_r, next_c = int(next_pos[0]), int(next_pos[1])
        current_pos = (curr_r, curr_c)
        next_pos = (next_r, next_c)

        swapped, step_rewards, infos = self.swap_intellector_defensor(
            current_pos, next_pos, turn
        )
        if swapped:
            return step_rewards, infos

        step_rewards = [rewards_mod.DRAW, rewards_mod.DRAW]

        target_val = self.board[1 - turn, next_r, next_c]
        if target_val == pieces.INTELLECTOR:
            step_rewards[turn] += rewards_mod.CAPTURE_REWARDS[pieces.INTELLECTOR]
            self.board[1 - turn, next_r, next_c] = pieces.EMPTY
            for key, pos in list(self.pieces[1 - turn].items()):
                if pos == next_pos:
                    self.pieces[1 - turn][key] = None
                    break

        self.board[turn, next_r, next_c] = self.board[turn, curr_r, curr_c]
        self.board[turn, curr_r, curr_c] = pieces.EMPTY

        self.promote_progressor(next_pos, turn)

        for key, pos in list(self.pieces[turn].items()):
            if pos == current_pos:
                self.pieces[turn][key] = next_pos
                break

        return step_rewards, [set(), set()]

    def step(self, action: int):
        if action >= self.space_size or action == 0:
            step_rewards = [rewards_mod.DRAW, rewards_mod.DRAW]
        else:
            src, poss, mask = self.get_all_actions(self.turn)
            if not mask[action]:
                step_rewards = [rewards_mod.DRAW, rewards_mod.DRAW]
            else:
                step_rewards, infos = self.move_piece(
                    src[action], poss[action], self.turn
                )

        step_rewards[self.turn] += rewards_mod.STEP_PENALTY

        self.steps += 1
        self.turn = 1 - self.turn

        done, final_rewards = self.is_game_done()

        total_rewards = [
            step_rewards[0] + final_rewards[0],
            step_rewards[1] + final_rewards[1],
        ]

        obs = self.get_state(self.turn)
        truncated = self.steps >= self.max_steps

        return obs, total_rewards, done, truncated, {"winner": self.winner}
