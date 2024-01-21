import tasks
from common import Shape, Grid


def solve(shapes: list[Shape], grid: Grid):
    stack: list[tuple[list[Shape], Grid]] = [(shapes, grid)]
    visited = set()

    while stack:
        unused_shapes, current_grid = stack.pop()

        serialized = "".join(map(str, current_grid.state.flatten()))
        if serialized in visited:
            continue
        else:
            visited.add(serialized)

        if not unused_shapes:
            yield current_grid

        for shape in unused_shapes:
            for variation in shape.variations:
                if (next_grid := current_grid.place(shape, variation)) is not None:
                    next_shapes = [s for s in unused_shapes if s.id != shape.id]
                    if all(
                        next_shape.still_fits_grid(next_grid)
                        for next_shape in next_shapes
                    ) and next_grid.can_cover_holes(next_shapes):
                        stack.append((next_shapes, next_grid))


for solution in solve(*tasks.task_vojta_game_full()):
    print(solution.state, end="\n\n")
    with open("solutions/task_vojta_game_full.txt", "a") as file:
        file.write(str(solution.state))
        file.write("\n\n")
