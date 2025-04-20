import numpy as np
import gymnasium
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from intellector.moves import POSSIBLE_MOVES
import intellector.pieces as pieces


class HexChessEnv(gymnasium.Env):

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

    def is_valid_move(self, next_pos, turn):
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
        rewards = [0, 0]
        white_intellector_present = np.any(
            self.board[pieces.WHITE, :, :] == pieces.INTELLECTOR
        )
        black_intellector_present = np.any(
            self.board[pieces.BLACK, :, :] == pieces.INTELLECTOR
        )

        if not white_intellector_present:
            self.done = True
            rewards = [-10000, 10000]  # Черные выигрывают, белые проигрывают
        elif not black_intellector_present:
            self.done = True
            rewards = [10000, -10000]  # Белые выигрывают, черные проигрывают
        else:
            _, _, actions_white = self.get_all_actions(pieces.WHITE)
            _, _, actions_black = self.get_all_actions(pieces.BLACK)
            if np.sum(actions_white) == 0:
                self.done = True
                rewards = [-100, 100]  # Белые не могут ходить, черные выигрывают
            elif np.sum(actions_black) == 0:
                self.done = True
                rewards = [100, -100]  # Черные не могут ходить, белые выигрывают
            elif self.steps >= self.max_steps:
                self.done = True
                rewards = [0, 0]  # Ничья

        return self.done, rewards

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
        return possibles, actions_mask

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
        possibles, actions_mask = self.get_empty_actions("progressor")

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
            possibles[i] = next_pos
            actions_mask[i] = 1

        return possibles, actions_mask

    def get_actions_for_agressor(self, pos, turn: int):
        possibles, actions_mask = self.get_empty_actions("agressor")

        if pos is None:
            return possibles, actions_mask

        row, col = pos

        index = 0

        # Horizontal moves to the right
        for i in range(col + 2, self.grid_width, 2):
            next_pos = (row, i)
            if not self.is_valid_move(next_pos, turn) or not self.is_empty(
                next_pos, turn
            ):
                break
            possibles[index] = next_pos
            actions_mask[index] = 1
            index += 1

        # Horizontal moves to the left
        for i in range(col - 2, -1, -2):
            next_pos = (row, i)
            if not self.is_valid_move(next_pos, turn) or not self.is_empty(
                next_pos, turn
            ):
                break
            possibles[index] = next_pos
            actions_mask[index] = 1
            index += 1

        # Diagonal moves to the left and down
        new_col = col
        new_row = row
        while new_row < self.grid_height:
            if new_col % 2 == 0:
                new_col -= 1
                new_row += 1
                next_pos = (new_row, new_col)
                if not self.is_valid_move(next_pos, turn) or not self.is_empty(
                    next_pos, turn
                ):
                    break
            else:
                new_col -= 1
                new_row += 2
                next_pos = (new_row, new_col)
                if not self.is_valid_move(next_pos, turn) or not self.is_empty(
                    next_pos, turn
                ):
                    break
            possibles[index] = next_pos
            actions_mask[index] = 1
            index += 1

        new_col = col
        new_row = row
        while new_row < self.grid_height:
            if new_col % 2 == 0:
                new_col += 1
                new_row += 1
                next_pos = (new_row, new_col)
                if not self.is_valid_move(next_pos, turn) or not self.is_empty(
                    next_pos, turn
                ):
                    break
            else:
                new_col += 1
                new_row += 2
                next_pos = (new_row, new_col)
                if not self.is_valid_move(next_pos, turn) or not self.is_empty(
                    next_pos, turn
                ):
                    break
            possibles[index] = next_pos
            actions_mask[index] = 1
            index += 1

        new_col = col
        new_row = row
        while new_row > 0:
            if new_col % 2 == 0:
                new_col -= 1
                new_row -= 2
                next_pos = (new_row, new_col)
                if not self.is_valid_move(next_pos, turn) or not self.is_empty(
                    next_pos, turn
                ):
                    break
            else:
                new_col -= 1
                new_row -= 1
                next_pos = (new_row, new_col)
                if not self.is_valid_move(next_pos, turn) or not self.is_empty(
                    next_pos, turn
                ):
                    break
            possibles[index] = next_pos
            actions_mask[index] = 1
            index += 1

        new_col = col
        new_row = row
        while new_row > 0:
            if new_col % 2 == 0:
                new_col += 1
                new_row -= 2
                next_pos = (new_row, new_col)
                if not self.is_valid_move(next_pos, turn) or not self.is_empty(
                    next_pos, turn
                ):
                    break
            else:
                new_col += 1
                new_row -= 1
                next_pos = (new_row, new_col)
                if not self.is_valid_move(next_pos, turn) or not self.is_empty(
                    next_pos, turn
                ):
                    break
            possibles[index] = next_pos
            actions_mask[index] = 1
            index += 1

        return possibles, actions_mask

    def get_actions_for_intellector(self, pos, turn: int):
        possibles, actions_mask = self.get_empty_actions("intellector")

        if pos is None:
            return possibles, actions_mask

        row, col = pos
        intellector_moves = self.adjacent_hexes(row, col)
        for i, (r, c) in enumerate(intellector_moves):
            next_pos = (r, c)
            if self.is_valid_move(next_pos, turn):
                if self.is_defensor(next_pos, turn):
                    possibles[i] = next_pos
                    actions_mask[i] = 1
                elif self.is_empty(next_pos, turn):
                    possibles[i] = next_pos
                    actions_mask[i] = 1

        return possibles, actions_mask

    def get_actions_for_defensor(self, pos, turn: int):
        possibles, actions_mask = self.get_empty_actions("defensor")

        if pos is None:
            return possibles, actions_mask

        row, col = pos
        defensor_moves = self.adjacent_hexes(row, col)
        for i, (r, c) in enumerate(defensor_moves):
            next_pos = (r, c)
            if self.is_valid_move(next_pos, turn):
                if self.is_intellector(next_pos, turn):
                    possibles[i] = next_pos
                    actions_mask[i] = 1
                elif self.is_empty(next_pos, turn):
                    possibles[i] = next_pos
                    actions_mask[i] = 1

        return possibles, actions_mask

    def get_actions_for_liberator(self, pos, turn: int):
        possibles, actions_mask = self.get_empty_actions("liberator")
        if pos is None:
            return possibles, actions_mask

        row, col = pos
        liberator_moves = self.adjacent_hexes(row, col)
        for i, (r, c) in enumerate(liberator_moves):
            next_pos = (r, c)
            if (
                not self.is_valid_move(next_pos, turn)
                or self.board[turn, r, c] != pieces.EMPTY
                or self.board[1 - turn, r, c] != pieces.EMPTY
            ):
                continue

            possibles[i] = next_pos
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
            if not self.is_valid_move(next_pos, turn) or not self.is_empty(
                next_pos, turn
            ):
                continue

            possibles[i + 6] = next_pos
            actions_mask[i + 6] = 1

        return possibles, actions_mask

    def get_actions_for_dominator(self, pos, turn: int):
        possibles, actions_mask = self.get_empty_actions("dominator")
        if pos is None:
            return possibles, actions_mask

        row, col = pos
        index = 0  # Индекс для отслеживания позиции в массивах possibles и actions_mask

        # Vertical moves downward
        for i in range(row + 1, self.grid_height):
            next_pos = (i, col)
            if not self.is_valid_move(next_pos, turn) or not self.is_empty(
                next_pos, turn
            ):
                break
            possibles[index] = next_pos
            actions_mask[index] = 1
            index += 1

        # Vertical moves upward
        for i in range(row - 1, -1, -1):
            next_pos = (i, col)
            if not self.is_valid_move(next_pos, turn) or not self.is_empty(
                next_pos, turn
            ):
                break
            if self.is_enemy(next_pos, turn):
                possibles[index] = next_pos
                actions_mask[index] = 1
                index += 1
                break
            possibles[index] = next_pos
            actions_mask[index] = 1
            index += 1

        # Diagonal moves to the left and down
        new_col = col
        new_row = row
        while new_col > 0:
            if new_col % 2 == 0:
                next_pos = (new_row, new_col - 1)
            else:
                next_pos = (new_row + 1, new_col - 1)

            if not self.is_valid_move(next_pos, turn) or not self.is_empty(
                next_pos, turn
            ):
                break
            possibles[index] = next_pos
            actions_mask[index] = 1
            index += 1

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

            if not self.is_valid_move(next_pos, turn) or not self.is_empty(
                next_pos, turn
            ):
                break
            possibles[index] = next_pos
            actions_mask[index] = 1
            index += 1

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

            if not self.is_valid_move(next_pos, turn) or not self.is_empty(
                next_pos, turn
            ):
                break
            possibles[index] = next_pos
            actions_mask[index] = 1
            index += 1

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

            if not self.is_valid_move(next_pos, turn) or not self.is_empty(
                next_pos, turn
            ):
                break
            possibles[index] = next_pos
            actions_mask[index] = 1
            index += 1

            if new_col % 2 == 0:
                new_row -= 1
            new_col += 1

        return possibles, actions_mask

    def reset(self, *, seed=None, options=None) -> np.ndarray:
        self.done = False
        self.turn = 0
        self.steps = 0
        self.board = self.init_board()
        self.pieces = self.init_pieces()
        self.checked = [False, False]
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
        self.display_plot(fig, mode)

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
        # Определение, какая фигура находится на позиции (учитываются оба массива)
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
    def display_plot(fig, mode):
        if mode == "human":
            plt.show()

    def get_actions_for(self, name: str, turn: int):
        assert name in self.pieces_names, f"name not in {self.pieces_names}"
        piece_cat = name.split("_")[0]
        piece_pos = self.pieces[turn][name]
        src_poses = self.get_source_pos(name, turn)

        if piece_cat == "intellector":
            possibles, actions_mask = self.get_actions_for_intellector(piece_pos, turn)
            return src_poses, possibles, actions_mask

        if piece_cat == "progressor":
            possibles, actions_mask = self.get_actions_for_progressor(piece_pos, turn)
            return src_poses, possibles, actions_mask

        if piece_cat == "dominator":
            possibles, actions_mask = self.get_actions_for_dominator(piece_pos, turn)
            return src_poses, possibles, actions_mask

        if piece_cat == "agressor":
            possibles, actions_mask = self.get_actions_for_agressor(piece_pos, turn)
            return src_poses, possibles, actions_mask

        if piece_cat == "defensor":
            possibles, actions_mask = self.get_actions_for_defensor(piece_pos, turn)
            return src_poses, possibles, actions_mask

        if piece_cat == "liberator":
            possibles, actions_mask = self.get_actions_for_liberator(piece_pos, turn)
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
        if (turn == pieces.WHITE and row == self.grid_height - 1) or (
            turn == pieces.BLACK and row == 0
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

            rewards = [0, 0]
            return True, rewards, [set(), set()]

        return False, None, None

    def move_piece(self, current_pos, next_pos, turn: int):
        next_row, next_col = next_pos
        current_row, current_col = current_pos

        swapped, rewards, infos = self.swap_intellector_defensor(
            current_pos, next_pos, turn
        )
        if swapped:
            return rewards, infos

        self.promote_progressor(next_pos, turn)

        # Удаление фигуры противника, если она есть на новой позиции
        if self.board[1 - turn, next_row, next_col] != pieces.EMPTY:
            self.board[1 - turn, next_row, next_col] = pieces.EMPTY
            to_remove = None
            for key, value in self.pieces[1 - turn].items():
                if (
                    isinstance(value, tuple)
                    and value[0] == next_row
                    and value[1] == next_col
                ):
                    to_remove = key
                    break
            if to_remove:
                self.pieces[1 - turn][to_remove] = None

        # Перемещение фигуры игрока на новую позицию
        self.board[turn, next_row, next_col] = self.board[
            turn, current_row, current_col
        ]
        self.board[turn, current_row, current_col] = pieces.EMPTY

        # Обновление позиции в списке фигур
        for key, value in self.pieces[turn].items():
            if (
                isinstance(value, tuple)
                and value[0] == current_row
                and value[1] == current_col
            ):
                self.pieces[turn][key] = next_pos

        rewards = [0, 0]
        return rewards, [set(), set()]

    def step(self, action: int):
        done, rewards = self.is_game_done()
        if done:
            observation = self.get_state(self.turn)
            return observation, rewards, done, self.steps >= self.max_steps, {}

        if action >= self.space_size:
            observation = self.get_state(self.turn)
            done, rewards = self.is_game_done()
            return observation, [0, 0], done, self.steps >= self.max_steps, {}

        # Пропустить действие, если action равно 0
        if action == 0:
            observation = self.get_state(self.turn)
            self.turn = 1 - self.turn
            self.steps += 1
            done, rewards = self.is_game_done()
            return observation, [0, 0], done, self.steps >= self.max_steps, {}

        source_pos, possibles, actions_mask = self.get_all_actions(self.turn)

        if not actions_mask[action]:
            observation = self.get_state(self.turn)
            done, rewards = self.is_game_done()
            return observation, [0, 0], done, self.steps >= self.max_steps, {}

        rewards, infos = self.move_piece(
            source_pos[action], possibles[action], self.turn
        )
        observation = self.get_state(self.turn)
        self.turn = 1 - self.turn
        self.steps += 1
        done, game_rewards = self.is_game_done()
        rewards = [
            sum(x) for x in zip(rewards, game_rewards)
        ]  # Суммировать награды текущего хода и конца игры
        return observation, rewards, done, self.steps >= self.max_steps, {}
