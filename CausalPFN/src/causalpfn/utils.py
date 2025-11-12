from typing import Tuple

import torch


def sample_confidence_interval(samples: torch.Tensor, alphas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the (1 - alpha) confidence interval for an array of samples using quantiles.

    Args:
        samples (torch.Tensor): Tensor of shape (..., num_samples).
        alphas (torch.Tensor): Confidence level (e.g., 0.05 for a 95% confidence interval) with shape (num_alphas,).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (lower_bound, upper_bound), each of shape (num_alphas, ...)
    """
    threshold = 0.01

    # Compute quantiles
    lower_q = alphas / 2
    upper_q = 1 - alphas / 2

    orig_shape = samples.shape
    samples = samples.reshape(-1, orig_shape[-1])  # Flatten the last dimension

    lower_bound = torch.quantile(samples, lower_q.to(samples.dtype), dim=-1, interpolation="linear")
    upper_bound = torch.quantile(samples, upper_q.to(samples.dtype), dim=-1, interpolation="linear")

    # Handle array of small alphas
    zero_mask = alphas <= threshold
    if zero_mask.any():
        # First reshape to ensure proper indexing
        reshaped_lower = lower_bound.view(len(alphas), -1)
        reshaped_upper = upper_bound.view(len(alphas), -1)

        # Replace zeros with infinities (non-inplace to preserve gradients)
        # zero_mask has shape (len(alphas),), need to broadcast to (len(alphas), -1)
        inf_tensor_lower = torch.full_like(reshaped_lower, float("-inf"))
        inf_tensor_upper = torch.full_like(reshaped_upper, float("inf"))
        reshaped_lower = torch.where(zero_mask.unsqueeze(-1), inf_tensor_lower, reshaped_lower)
        reshaped_upper = torch.where(zero_mask.unsqueeze(-1), inf_tensor_upper, reshaped_upper)
        
        # Update lower_bound and upper_bound with the modified values
        lower_bound = reshaped_lower.reshape(lower_bound.shape)
        upper_bound = reshaped_upper.reshape(upper_bound.shape)

    lower_bound = lower_bound.reshape(len(alphas), *orig_shape[:-1])
    upper_bound = upper_bound.reshape(len(alphas), *orig_shape[:-1])

    return lower_bound, upper_bound
