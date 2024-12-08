"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable, Optional

# ## Task 0.1


# Implement for Task 0.1.


def mul(a: float, b: float) -> float:
    """Multiplies two numbers together"""
    return a * b


def id(a: float) -> float:
    """Identity function: returns the input unchanged"""
    return a


def add(a: float, b: float) -> float:
    """Adds two numbers together"""
    return a + b


def neg(a: float) -> float:
    """Negates a number"""
    return -a


def lt(a: float, b: float) -> bool:
    """Checks if a is less than b"""
    return a < b


def eq(a: float, b: float) -> bool:
    """Checks if a is equal to b"""
    return a == b


def max(a: float, b: float) -> float:
    """Returns the larger of a and b"""
    return a if a > b else b


def is_close(a: float, b: float, abs_tol: float = 1e-2) -> bool:
    """Checks if a is 'close' to b"""
    return abs(a - b) < abs_tol


def sigmoid(x: float) -> float:
    """Sigmoid function"""
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def sigmoid_back(x: float, y: float) -> float:
    """Sigmoid backwards derivative"""
    if x >= 0.0:
        return (1.0 + math.exp(-x)) ** (-2.0) * math.exp(-x) * y
    else:
        return (
            math.exp(x)
            * (
                -math.exp(x) * (1.0 + math.exp(x)) ** (-2.0)
                + (1.0 + math.exp(x)) ** (-1.0)
            )
            * y
        )


def relu(x: float) -> float:
    """ReLU function"""
    return x if x > 0 else 0.0


def log(x: float) -> float:
    """Natural logarithm function"""
    return math.log(x)


def exp(x: float) -> float:
    """Exponential function"""
    return math.exp(x)


def inv(x: float) -> float:
    """Calculates the reciprocal of x"""
    return 1.0 / x


def log_back(x: float, y: float) -> float:
    """Computes the derivative of log of the first argument, x, times a second argument, y"""
    return inv(x) * y


def inv_back(x: float, y: float) -> float:
    """Computes the derivative of the reciprocal of the first argument, x, times a second argument, y"""
    return y * -(x ** (-2.0))


def relu_back(x: float, y: float) -> float:
    """Computes the derivative of the ReLU function of the first argument, x, times a second argument, y"""
    return 0.0 if x <= 0.0 else y


def permute(x: float, dims: Iterable[int]) -> float:
    """Permute the dimensions of a tensor."""
    return x


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# Note to self:

# fn: Callable[[float], float] -> means it takes in a float and returns a float
# fn: Callable[[float, float], float] -> means it takes in two floats and returns a float


# Implement for Task 0.3.


def map(fn: Callable[[float], float], l: Iterable[float]) -> Iterable[float]:
    """Applies a given function to each element of an input iterable.

    Args:
    ----
        fn: A function, specifically with one input and output
        l: An iterable of floats

    Returns:
    -------
        Mapped values in iterable

    """
    return [fn(x) for x in l]


def zipWith(
    fn: Callable[[float, float], float], a: Iterable[float], b: Iterable[float]
) -> Iterable[float]:
    """Applies a given function to pairs of elements from two input iterables.

    Args:
    ----
        fn: A function, specifically with two inputs and one output
        a: An iterable of floats
        b: An iterable of floats

    Returns:
    -------
        Mapped values in iterable

    """
    return [fn(x, y) for x, y in zip(a, b)]


def reduce(
    fn: Callable[[float, float], float],
    l: Iterable[float],
    start: Optional[float] = None,
) -> float:
    """Reduces an iterable to a single value by applying a given function cumulatively.

    Args:
    ----
        fn: A function, specifically with two inputs and one output.
        l: An iterable of floats.
        start: An optional initial value to start the reduction. If provided, it will be used as the initial accumulator value.

    Returns:
    -------
        The final reduced value. 0 if empty list to conform to python sum behavior

    """
    it = iter(l)

    if start is None:
        try:
            acc = next(it)
        except StopIteration:
            acc = 0
    else:
        acc = start

    for x in it:
        acc = fn(acc, x)
    return acc


def negList(l: Iterable[float]) -> Iterable[float]:
    """Negates each element in a list using map.


    Args:
    ----
        l: An iterable of floats

    Returns:
    -------
        A list of negated values

    """
    return map(neg, l)


def addLists(a: Iterable[float], b: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists using zipWith

    Args:
    ----
        a: Iterable of floats
        b: Iterable of floats

    Returns:
    -------
        An iterable of floats

    """
    return zipWith(add, a, b)


def sum(l: Iterable[float]) -> float:
    """Adds all elements of a list together, using reduce

    Args:
    ----
        l: Iterable of floats

    Returns:
    -------
        A single float

    """
    return reduce(add, l)


def prod(l: Iterable[float]) -> float:
    """Multiplies all elements of a list together, using reduce

    Args:
    ----
        l: Iterable of floats

    Returns:
    -------
        A single float

    """
    return reduce(mul, l)
