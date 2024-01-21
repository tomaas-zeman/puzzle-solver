from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


def fits_in_slice(slice: np.ndarray, var: np.ndarray):
    if slice.shape != var.shape:
        return False

    for row in range(slice.shape[0]):
        for col in range(slice.shape[1]):
            if slice[(row, col)] != 0 and var[(row, col)] != 0:
                return False
    return True


def find_valid_placement(grid: Grid, variation: np.ndarray):
    for row in range(grid.state.shape[0]):
        for col in range(grid.state.shape[1]):
            slice = grid.state[
                row : row + variation.shape[0], col : col + variation.shape[1]
            ]
            if fits_in_slice(slice, variation):
                return row, col

    return None


def create_variations(shape: list[list[int]]):
    variations = []

    current_map = np.array(shape)
    for _ in range(4):
        current_map = np.rot90(current_map)
        variations.extend([current_map, np.fliplr(current_map), np.flipud(current_map)])

    final_variations = []
    for v in variations:
        already_present = False
        for fv in final_variations:
            if np.array_equal(v, fv):
                already_present = True

        if not already_present:
            final_variations.append(v)

    return final_variations


class Shape:
    def __init__(self, id: int, shape: list[list[int]]):
        self.id = id
        self.shape = np.array(shape)
        self.variations = create_variations(shape)

    def still_fits_grid(self, grid: Grid):
        for variation in self.variations:
            if find_valid_placement(grid, variation) is not None:
                return True
        return False

    def __eq__(self, other: Shape):
        return np.array_equal(self.shape, other.shape)


@dataclass
class Grid:
    state: np.ndarray
    random_shape_mode: bool = False

    def place(self, shape: Shape, variation: np.ndarray):
        return (
            self._copy()._update_state(position, shape, variation)
            if (position := find_valid_placement(self, variation)) is not None
            else None
        )

    def can_cover_holes(self, shapes: list[Shape]):
        def fill_unused_space():
            state = np.copy(self.state)

            queue = deque([(state.shape[0] - 1, state.shape[1] - 1)])
            while queue:
                for neighbor in self._neighbors(queue.pop()):
                    if state[neighbor] == 0:
                        state[neighbor] = -1
                        queue.append(neighbor)

            return state

        if not shapes:
            return True

        sorted_shapes = sorted(shapes, key=lambda s: np.count_nonzero(s.shape == 1))
        state = fill_unused_space()

        for row in range(state.shape[0]):
            for col in range(state.shape[1]):
                if state[row, col] == 0:
                    hole = {(row, col)}
                    queue = deque([(row, col)])
                    while queue:
                        for neighbor in self._neighbors(queue.pop()):
                            if state[neighbor] == 0 and neighbor not in hole:
                                queue.append(neighbor)
                                hole.add(neighbor)

                    if len(hole) < np.count_nonzero(sorted_shapes[0].shape == 1):
                        return False

                    if self.random_shape_mode and len(hole) > np.count_nonzero(
                        sorted_shapes[-1].shape == 1
                    ):
                        return True

                    min_row = min([p[0] for p in hole])
                    max_row = max([p[0] for p in hole])
                    min_col = min([p[1] for p in hole])
                    max_col = max([p[1] for p in hole])
                    slice = state[min_row : max_row + 1, min_col : max_col + 1]
                    if all(
                        not fits_in_slice(slice, v) for v in sorted_shapes[0].variations
                    ):
                        return False

        return True

    def _neighbors(self, point: tuple[int, int]):
        row, col = point
        return [
            n
            for n in [
                (row, col - 1),
                (row, col + 1),
                (row - 1, col),
                (row + 1, col),
            ]
            if 0 <= n[0] < self.state.shape[0] and 0 <= n[1] < self.state.shape[1]
        ]

    def _copy(self):
        return Grid(np.copy(self.state), random_shape_mode=self.random_shape_mode)

    def _update_state(
        self, position: tuple[int, int], shape: Shape, variation: np.ndarray
    ):
        for row in range(variation.shape[0]):
            for col in range(variation.shape[1]):
                if variation[(row, col)] == 1:
                    self.state[(position[0] + row, position[1] + col)] = shape.id
        return self
