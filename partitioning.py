import math

import numpy as np
import torch


class ZeroBlockMeanError(ValueError):
    pass


# When the period search finds no exploitable block structure it returns
# period==1, which forces per-element fp32 vmean/m_magnitude (~9 bytes/param,
# worse than AdamW). On modern architectures (SwiGLU MLP, GQA) the squared-grad
# E(p) curve is nearly flat -- no scale has lower within-block variance -- so the
# "biggest variance drop" detector has no signal and collapses to 1 by noise.
# Because the curve is flat, sharing vmean across a coarse block costs almost
# nothing in 2nd-moment fidelity, so instead of period==1 we fall back to a
# coarse shape-aligned block (the input-feature row for 2D weights, else a capped
# divisor) keeping state at ~1 byte/param. Set MEMORY_SAFE_FALLBACK=False to
# restore the original period==1 behavior.
MEMORY_SAFE_FALLBACK = True
FALLBACK_PERIOD_CAP = 2048


def _largest_divisor_at_most(n, cap):
    best = 1
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            if i <= cap and i > best:
                best = i
            j = n // i
            if j <= cap and j > best:
                best = j
    return best


def memory_safe_fallback_period(n, parameter_shape):
    if n < 8:
        return 1
    candidates = []
    # in_features (last dim) of a 2D weight divides n exactly and is the natural
    # per-output-row block; the diagnostic shows row-level sharing is as accurate
    # as any finer block on these flat-E tensors while giving ~1 byte/param.
    if parameter_shape is not None and len(parameter_shape) >= 2:
        in_features = int(parameter_shape[-1])
        if in_features >= 8 and n % in_features == 0:
            candidates.append(in_features)
    capped = _largest_divisor_at_most(n, FALLBACK_PERIOD_CAP)
    if capped >= 8:
        candidates.append(capped)
    if not candidates:
        return 1
    return max(candidates)


def divisors(n):

    small, large = [], []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            if i != n:
                small.append(i)
            if i != n // i and n // i != n:
                large.append(n // i)
    return small + large[::-1]


def _zero_block_mean_error(parameter_name, parameter_shape, period):
    return ZeroBlockMeanError(
        "Encountered a block with mean 0.0, cannot compute element-to-mean ratios. parameter_name={} parameter_shape={} period={}".format(
            parameter_name,
            parameter_shape,
            period,
        )
    )


def average_within_block_variance_cpu(
    list_of_values, period_to_check, parameter_name=None, parameter_shape=None
):

    list_of_values = np.asarray(list_of_values, dtype=float)
    n = len(list_of_values)

    if n % period_to_check != 0:
        raise ValueError(
            "Expected length {} to be divisible by period {}.".format(
                n, period_to_check
            )
        )

    blocks = list_of_values.reshape(-1, period_to_check)

    block_variances = blocks.var(axis=1)

    result = block_variances.mean()

    result = np.sqrt(result)

    return result


def average_within_block_variance_torch(
    values, period, parameter_name=None, parameter_shape=None
):
    if not isinstance(values, torch.Tensor):
        raise TypeError("Expected torch.Tensor, got {}".format(type(values).__name__))

    values = values.detach().reshape(-1).to(dtype=torch.float32)
    n = values.numel()

    if n % period != 0:
        raise ValueError(
            "Expected length {} to be divisible by period {}.".format(n, period)
        )

    blocks = values.view(-1, period)

    block_variances = blocks.var(dim=1, correction=0)

    result = block_variances.mean()

    result = torch.sqrt(result)
    return result.item()


def _find_period_by_block_variance_cpu(
    squared_grad_flattened_values,
    print_results=True,
    parameter_name=None,
    parameter_shape=None,
):
    squared_grad_flattened_values = np.asarray(
        squared_grad_flattened_values, dtype=float
    )
    n = len(squared_grad_flattened_values)

    list_period_and_error = []
    divs = divisors(n)

    for p in divs:
        try:
            err_average_variance_in_blocks = average_within_block_variance_cpu(
                squared_grad_flattened_values,
                p,
                parameter_name=parameter_name,
                parameter_shape=parameter_shape,
            )
        except ZeroBlockMeanError:
            continue
        list_period_and_error.append((p, err_average_variance_in_blocks))

    if len(list_period_and_error) == 0:
        raise ValueError(
            "Could not find any valid period candidates. parameter_name={} parameter_shape={}".format(
                parameter_name,
                parameter_shape,
            )
        )

    return _finalize_period_results(list_period_and_error, print_results)


def _find_period_by_block_variance_torch(
    squared_grad_flattened_values,
    print_results=True,
    parameter_name=None,
    parameter_shape=None,
):
    if not isinstance(squared_grad_flattened_values, torch.Tensor):
        raise TypeError(
            "Expected torch.Tensor for gpu backend, got {}".format(
                type(squared_grad_flattened_values).__name__
            )
        )
    if squared_grad_flattened_values.device.type != "cuda":
        raise ValueError(
            "Expected a CUDA tensor for gpu backend, got device {}".format(
                squared_grad_flattened_values.device
            )
        )

    values = squared_grad_flattened_values.detach().reshape(-1).to(dtype=torch.float32)
    n = values.numel()

    results = []
    divs = divisors(n)

    for p in divs:
        try:
            err = average_within_block_variance_torch(
                values,
                p,
                parameter_name=parameter_name,
                parameter_shape=parameter_shape,
            )
        except ZeroBlockMeanError:

            continue
        results.append((p, err))

    if len(results) == 0:
        raise ValueError(
            "Could not find any valid period candidates. parameter_name={} parameter_shape={}".format(
                parameter_name,
                parameter_shape,
            )
        )

    return _finalize_period_results(results, print_results)


def _find_period_by_block_variance_cuda_kernel(
    values,
    print_results=True,
    parameter_name=None,
    parameter_shape=None,
    input_is_squared=True,
):
    if not isinstance(values, torch.Tensor):
        raise TypeError(
            "Expected torch.Tensor for cuda_kernel backend, got {}".format(
                type(values).__name__
            )
        )
    if values.device.type != "cuda":
        raise ValueError(
            "Expected a CUDA tensor for cuda_kernel backend, got device {}".format(
                values.device
            )
        )

    flattened_values = values.detach().reshape(-1)
    n = flattened_values.numel()

    results = []
    divs = divisors(n)

    from gefen.kernels.period_variance import average_within_block_variance_cuda_kernel

    for p in divs:
        err = average_within_block_variance_cuda_kernel(
            flattened_values,
            p,
            input_is_squared=input_is_squared,
        )
        results.append((p, err))

    if len(results) == 0:
        raise ValueError(
            "Could not find any valid period candidates. parameter_name={} parameter_shape={}".format(
                parameter_name,
                parameter_shape,
            )
        )

    return _finalize_period_results(results, print_results)


def _finalize_period_results(list_periods_and_errors, print_results):
    if len(list_periods_and_errors) == 0:
        raise ValueError("Expected at least one valid period candidate.")

    scored_results = []
    previous_err = None

    for p, err in list_periods_and_errors:

        if previous_err is None:
            weighted_error = 1e-12
        else:

            weighted_error = err - previous_err
        previous_err = err

        scored_results.append((p, err, weighted_error))

    best_weighted_error = min(weighted_error for _, _, weighted_error in scored_results)

    if print_results:
        for p, err, weighted_error in scored_results:
            line = f"p={p} |  err={err:.15f} | weighted error = {weighted_error:.15f}"
            if weighted_error == best_weighted_error:
                print(f"\033[92m{line}\033[0m")
            elif weighted_error != best_weighted_error:
                print(line)

    _, best_period = min((weighted_error, p) for p, _, weighted_error in scored_results)

    if best_period < 8:

        best_period = 1

    return best_period


def find_period_by_block_variance(
    squared_grad_flattened_values,
    print_results=True,
    parameter_name=None,
    parameter_shape=None,
    backend="cpu",
    input_is_squared=True,
):
    if backend == "cpu":
        period = _find_period_by_block_variance_cpu(
            squared_grad_flattened_values,
            print_results=print_results,
            parameter_name=parameter_name,
            parameter_shape=parameter_shape,
        )
    elif backend == "gpu":
        period = _find_period_by_block_variance_torch(
            squared_grad_flattened_values,
            print_results=print_results,
            parameter_name=parameter_name,
            parameter_shape=parameter_shape,
        )
    elif backend == "cuda_kernel":
        period = _find_period_by_block_variance_cuda_kernel(
            squared_grad_flattened_values,
            print_results=print_results,
            parameter_name=parameter_name,
            parameter_shape=parameter_shape,
            input_is_squared=input_is_squared,
        )
    else:
        raise ValueError("Unexpected backend: {}".format(backend))

    if MEMORY_SAFE_FALLBACK and period == 1:
        try:
            n = int(squared_grad_flattened_values.numel())
        except AttributeError:
            n = int(len(squared_grad_flattened_values))
        return memory_safe_fallback_period(n, parameter_shape)
    return period
