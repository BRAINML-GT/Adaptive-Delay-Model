import torch

def SE_kernel_parallel(tau, kernel_params, num_dim):
    """
    Compute the multi-output squared exponential kernel for given tau values.

    Args:
        tau (torch.Tensor): Tensor of shape (lag+1, lag+1) representing time differences.
        kernel_params (tuple): Tuple containing kernel parameters (sigma, constant).
        num_dim (int): Number of dimensions (outputs).

    Returns:
        torch.Tensor: Kernel tensor of shape (lag+1, lag+1, num_dim, num_dim).
    """
    sigma = kernel_params

    if sigma.dim() == 1:
        sigma = torch.diag(sigma)

    # Compute tau_squared: Shape (lag+1, lag+1)
    tau_squared = torch.square(tau)  # Shape: (lag+1, lag+1)

    # Expand tau_squared to (lag+1, lag+1, 1, 1)
    tau_squared = tau_squared[..., None, None]  # Shape: (lag+1, lag+1, 1, 1)

    # Expand sigma and constant to (1, 1, num_dim, num_dim)
    sigma = sigma.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, num_dim, num_dim)

    # Compute exponent: Shape (lag+1, lag+1, num_dim, num_dim)
    exponent = -0.5 * sigma * tau_squared  # Broadcasting over tau and dimensions

    # Compute the kernel
    kcc = torch.exp(exponent)  # Shape: (lag+1, lag+1, num_dim, num_dim)

    return kcc  # Shape: (lag+1, lag+1, num_dim, num_dim)

def MOSE_kernel_parallel(tau: torch.Tensor,
                                  kernel_params: tuple[torch.Tensor, torch.Tensor],
                                  num_dim: int) -> torch.Tensor:
    """
    Compute the multi-output exponential kernel for given tau values, 
    WITHOUT a 'Trial' dimension.

    Args:
        tau (torch.Tensor): Shape (lag+1, lag+1) representing time differences.
        kernel_params (tuple): (sigmas, delays), each of shape (T, num_dim).
        num_dim (int): Number of dimensions (outputs).

    Returns:
        torch.Tensor: Kernel tensor of shape (T, lag+1, lag+1, num_dim, num_dim).
    """
    sigmas, delays = kernel_params  # Each shape: (T, num_dim)
    T, nd = delays.shape
    assert nd == num_dim, "num_dim mismatch with sigmas shape"

    lag = tau.shape[0] - 1  # tau is shape (lag+1, lag+1)

    tau_expanded = tau.view(1, lag+1, lag+1, 1, 1)   # (1, lag+1, lag+1, 1, 1)

    delay_i = delays.view(T, 1, 1, num_dim, 1)  # (T, 1, 1, num_dim, 1)
    delay_j = delays.view(T, 1, 1, 1, num_dim)  # (T, 1, 1, 1, num_dim)

    delta_t = tau_expanded + delay_j - delay_i

    deltaTsq = (delta_t ** 2) * sigmas

    amplitude = 1.0
    kernels = torch.exp(-0.5 * deltaTsq) * amplitude

    return kernels
