from __future__ import annotations


def dtw_distance(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return float("inf")

    rows = len(left)
    cols = len(right)
    table = [[float("inf")] * (cols + 1) for _ in range(rows + 1)]
    table[0][0] = 0.0

    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            cost = abs(left[row - 1] - right[col - 1])
            table[row][col] = cost + min(
                table[row - 1][col],
                table[row][col - 1],
                table[row - 1][col - 1],
            )

    return float(table[rows][cols])


def dtw_similarity(left: list[float], right: list[float]) -> float:
    distance = dtw_distance(left, right)
    if distance == float("inf"):
        return 0.0
    return 1.0 / (1.0 + distance)
