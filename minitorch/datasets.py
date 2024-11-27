"""Generates synthetic datasets for testing neural networks."""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """Generate N random points in the unit square.

    Args:
    ----
        N: int: The number of points to generate.

    Returns:
    -------
        List[Tuple[float, float]]: A list of N points, each represented as a tuple of two floats.

    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """For (x_1, x_2) in X, give the points identity 0 (blue) if x_1 < 0.5, else 1 (red). Intuitively, splits points down a vertical.

    Args:
    ----
        N: int: The number of points to generate.

    Returns:
    -------
        Graph: A graph object containing the points and their corresponding identities.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """For (x_1, x_2) in X, give the points identity 0 (blue) if x_1 + x_2 < 0.5, else 1 (red). Intuitively, splits points down a diagonal.

    Args:
    ----
        N: int: The number of points to generate.

    Returns:
    -------
        Graph: A graph object containing the points and their corresponding identities.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """For (x_1, x_2) in X, give the points identity 0 (blue) if x_1 < 0.2 or x_1 > 0.8, else 1 (red). Intuitively, splits points down two verticals, with blue points in between red points.

    Args:
    ----
        N: int: The number of points to generate.

    Returns:
    -------
        Graph: A graph object containing the points and their corresponding identities.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """For (x_1, x_2) in X, give the points identity 1 (red) if x_1 < 0.5 and x_2 > 0.5 or x_1 > 0.5 and x_2 < 0.5, else 0 (blue). Intuitively, splits points down two diagonals.

    Args:
    ----
        N: int: The number of points to generate.

    Returns:
    -------
        Graph: A graph object containing the points and their corresponding identities.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 and x_2 > 0.5 or x_1 > 0.5 and x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """For (x_1, x_2) in X, give the points identity 1 (red) if x_1^2 + x_2^2 > 0.1, else 0 (blue). Intuitively, splits points inside and outside a circle.

    Args:
    ----
        N: int: The number of points to generate.

    Returns:
    -------
        Graph: A graph object containing the points and their corresponding identities.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = x_1 - 0.5, x_2 - 0.5
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """Generates spiral shape with two arms, one red and the other blue.

    Args:
    ----
        N: int: The number of points to generate.

    Returns:
    -------
        Graph: A graph object containing the points and their corresponding identities.

    """

    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
