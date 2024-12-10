import time

import numpy as np
import numpy.typing as npt
from numba import njit


def read_data():
    input_file = "input.txt"
    with open(input_file, "r") as file:
        data = list(map(lambda l: list(map(int, l.strip())), file.readlines()))
    return data


@njit
def get_neighbors(y, x, max_y, max_x):
    for dy, dx in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
        if 0 <= y + dy < max_y and 0 <= x + dx < max_x:
            yield y + dy, x + dx


@njit
def dijkstra_part1(data, y, x, visited):
    stack = [(y, x)]
    score = 0

    while len(stack) > 0:
        y, x = stack.pop(0)
        if visited[y, x]:
            continue

        if data[y, x] == 9:
            score += 1
            visited[y, x] = 1
            continue

        visited[y, x] = 1
        for ny, nx in get_neighbors(y, x, data.shape[0], data.shape[1]):
            if data[ny, nx] - data[y, x] == 1:
                stack.append((ny, nx))
    return score


def part1(data: npt.NDArray):
    starts = []
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            if data[y, x] == 0:
                starts.append((y, x))

    score = 0
    for y, x in starts:
        visited = np.zeros(data.shape, dtype=int)
        score += dijkstra_part1(data, y, x, visited)
    return score


@njit
def dijkstra_part2(data, starts, score, visited):
    stack = list(starts)
    for y, x in stack:
        score[y, x] = 1

    while len(stack) > 0:
        y, x = stack.pop(0)
        if visited[y, x]:
            continue
        visited[y, x] = 1

        for ny, nx in get_neighbors(y, x, data.shape[0], data.shape[1]):
            if visited[ny, nx]:
                continue
            if data[y, x] - data[ny, nx] == 1:
                score[ny, nx] += score[y, x]
                stack.append((ny, nx))

    result = 0
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            if data[y, x] == 0:
                result += score[y, x]
    return result


def part2(data: npt.NDArray):
    ends = []
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            if data[y, x] == 9:
                ends.append((y, x))

    score = np.zeros(data.shape, dtype=int)
    visited = np.zeros(data.shape, dtype=int)
    return dijkstra_part2(data, ends, score, visited)


def main():
    # прогрев
    t = time.time()
    part1(np.expand_dims(np.arange(10), 0))
    part2(np.expand_dims(np.arange(10), 0))
    print(round(time.time() - t, 6))

    t = time.time()
    # чтение
    data = read_data()
    data_np = np.array(data, dtype=int)

    # решение
    result1 = part1(data_np)
    result2 = part2(data_np)
    print(round(time.time() - t, 6))

    print(result1)
    print(result2)


if __name__ == '__main__':
    main()
