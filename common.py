from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def find_valid_placement(grid: Grid, variation: np.ndarray):
    def fits_in_slice(slice: np.ndarray, var: np.ndarray):
        for row in range(slice.shape[0]):
            for col in range(slice.shape[1]):
                if slice[(row, col)] != 0 and var[(row, col)] != 0:
                    return False
        return True

    for row in range(grid.state.shape[0]):
        for col in range(grid.state.shape[1]):
            slice = grid.state[
                row : row + variation.shape[0], col : col + variation.shape[1]
            ]
            if slice.shape == variation.shape and fits_in_slice(slice, variation):
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

    def place(self, shape: Shape, variation: np.ndarray):
        return (
            self._copy()._update_state(position, shape, variation)
            if (position := find_valid_placement(self, variation)) is not None
            else None
        )

    def _copy(self):
        return Grid(np.copy(self.state))

    def _update_state(
        self, position: tuple[int, int], shape: Shape, variation: np.ndarray
    ):
        for row in range(variation.shape[0]):
            for col in range(variation.shape[1]):
                if variation[(row, col)] == 1:
                    self.state[(position[0] + row, position[1] + col)] = shape.id
        return self
