from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling."""
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0, "Height must be divisible by kernel height"
    assert width % kw == 0, "Width must be divisible by kernel width"

    # TODO: Implement for Task 4.3.

    new_height = height // kh
    new_width = width // kw

    input = input.contiguous()
    tiled = input.view(batch, channel, new_height, kh, new_width, kw)
    tiled = tiled.permute(0, 1, 2, 4, 3, 5).contiguous()
    tiled = tiled.view(batch, channel, new_height, new_width, kh * kw)

    return tiled, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Average pooling 2D."""
    tiled, new_height, new_width = tile(input, kernel)

    # Compute the average across the kernel dimensions
    pooled = tiled.mean(dim=4)

    return pooled.view(input.shape[0], input.shape[1], new_height, new_width)


# Complete the following functions in minitorch/nn.py, and pass tests marked as task4_4.
# Add a property tests for the function in test/test_nn.py and ensure that you understand its gradient computation.
##minitorch.max
# minitorch.softmax
##minitorch.logsoftmax
# minitorch.maxpool2d
# minitorch.dropout

max_reduce = FastOps.reduce(operators.max, -1e9)


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for Max"""
        dim_val = int(dim.item())
        ctx.save_for_backward(input, dim_val)
        return max_reduce(input, dim_val)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for Max"""
        input, dim = ctx.saved_values
        arg_max = argmax(input, dim)
        return (grad_output * arg_max, 0.0)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor along a specified dimension.

    Args:
    ----
        input (Tensor): The input tensor.
        dim (int): The dimension along which to compute the argmax.

    Returns:
    -------
        Tensor: A tensor with the same shape as input, where the maximum values along the specified dimension are set to 1 and all other values are set to 0.

    """
    max_values = max_reduce(input, dim)

    return max_values == input


def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction along a specified dimension.

    Args:
    ----
        input (Tensor): The input tensor.
        dim (int): The dimension along which to compute the max.

    Returns:
    -------
        Tensor: A tensor with the maximum values along the specified dimension.

    """
    return Max.apply(input, input._ensure_tensor(dim))


# TODO: Implement for Task 4.3.


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax of a tensor along a specified dimension.

    Args:
    ----
        input (Tensor): The input tensor.
        dim (int): The dimension along which to compute the softmax.

    Returns:
    -------
        Tensor: A tensor with the softmax values along the specified dimension.

    """
    input_exp = input.exp()
    return input_exp / input_exp.sum(dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax of a tensor along a specified dimension.

    Args:
    ----
        input (Tensor): The input tensor.
        dim (int): The dimension along which to compute the log softmax.

    Returns:
    -------
        Tensor: A tensor with the log softmax values along the specified dimension.

    """
    max_input = max(input, dim)
    stable_input = input - max_input
    log_sum_exp = (stable_input.exp().sum(dim)).log()

    return stable_input - log_sum_exp


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply max pooling to a 2D input tensor.

    Args:
    ----
        input (Tensor): The input tensor.
        kernel (Tuple[int, int]): The size of the pooling kernel.

    Returns:
    -------
        Tensor: The result of applying max pooling to the input tensor.

    """
    batch, channel, _, _ = input.shape
    tiled, new_height, new_width = tile(input, kernel)
    reduced = max_reduce(tiled, -1).contiguous()
    return reduced.view(batch, channel, new_height, new_width)


def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Apply dropout to the input tensor.

    Args:
    ----
        input (Tensor): The input tensor.
        p (float): The probability of dropping out each element.
        ignore (bool): If True, dropout is not applied.

    Returns:
    -------
        Tensor: The result of applying dropout to the input tensor.

    """
    if ignore:
        return input
    if p == 1:
        return input.zeros(input.shape)
    if p == 0:
        return input
    else:
        mask = rand(input.shape) > p
        scale = 1.0 / (1.0 - p)
        return input * mask * scale
