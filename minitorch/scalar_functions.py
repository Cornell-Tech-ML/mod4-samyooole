from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Applies the forward pass of the relevant ScalarFunction to the given values. Then, store in a context history.

        Args:
        ----
        *vals: the given values to apply the forward pass to.

        Returns:
        -------
        New scalar variable from the result with a new history.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for Add function"""
        return float(a + b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward pass for Add function"""
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for Log function"""
        ctx.save_for_backward(a)
        return float(operators.log(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for Log function"""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


class Mul(ScalarFunction):
    """Multiply function f(x, y) = x * y"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for Mul function"""
        ctx.save_for_backward(a, b)
        return float(operators.mul(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for Mul function"""
        (a, b) = ctx.saved_values
        return (b * d_output, a * d_output)


class Inv(ScalarFunction):
    """Inverse function: f(x) = 1/x"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for Inv function"""
        ctx.save_for_backward(a)
        return float(operators.inv(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for Inv function"""
        (a,) = ctx.saved_values

        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negative function: f(x) = -x"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for Neg function"""
        return float(operators.neg(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for Neg function"""
        return -d_output


class Sigmoid(ScalarFunction):
    """Sigmoid function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for Sigmoid function"""
        ctx.save_for_backward(a)

        return float(operators.sigmoid(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for Sigmoid function"""
        (a,) = ctx.saved_values

        return operators.sigmoid_back(a, d_output)


class ReLU(ScalarFunction):
    """ReLU function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for ReLU function"""
        ctx.save_for_backward(a)
        return float(operators.relu(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for ReLU function"""
        (a,) = ctx.saved_values

        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exponent function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for Exp function"""
        ctx.save_for_backward(a)
        return float(operators.exp(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for Exp function"""
        (a,) = ctx.saved_values

        return operators.exp(a) * d_output


class LT(ScalarFunction):
    """less than function"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for less than function"""
        return float(operators.lt(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for less than function"""
        return (0.0, 0.0)


class EQ(ScalarFunction):
    """equal function"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for equal function"""
        return float(operators.eq(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for equal function"""
        return (0.0, 0.0)
