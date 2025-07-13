from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn, vmap
from typing import Tuple, List, Callable, Sequence
from adm.kernels import SE_kernel_parallel, MOSE_kernel_parallel

class SSM(nn.Module):
    """Generic lag-P linear state-space model with Gaussian noise."""

    def __init__(
        self,
        lag: int,
        num_dim: int,
        device: torch.device | str,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.lag: int = lag
        self.num_dim: int = num_dim  # latent dimensionality
        self.dtype: torch.dtype = dtype
        self.device = torch.device(device)

        # Auxiliary constant matrices -------------------------------------------------
        self.IO: Tensor = torch.hstack(
            (
                torch.eye((lag - 1) * num_dim, dtype=dtype),
                torch.zeros(((lag - 1) * num_dim, num_dim), dtype=dtype),
            )
        ).to(self.device)
        self.P: Tensor = self._build_permutation_matrix(num_dim, lag)
        self.O: Tensor = 1e-6 * torch.eye((lag - 1) * num_dim, dtype=dtype, device=self.device)


    def gaussian_kernel(self, kernel_size: int, sigma: float) -> Tensor:
        """Return a 1-D Gaussian kernel suitable for ``F.conv1d``."""
        half = kernel_size // 2
        x = torch.arange(-half, half + 1, dtype=self.dtype, device=self.device)
        kernel = torch.exp(-0.5 * sigma * x**2)
        kernel = (kernel / kernel.sum()).view(1, 1, -1)
        return kernel

    def _build_permutation_matrix(self, N: int, P: int) -> Tensor:
        """Construct permutation matrix that reshapes a Kronecker-style matrix."""
        size = (P + 1) * N
        A = torch.zeros((size, size), dtype=self.dtype, device=self.device)
        for i in range(P + 1):
            for r in range(N):
                A[r * (P + 1) + i, i * N + r] = 1.0
        return A

    @staticmethod
    def _batch_block_diag_loop_no_trial(matrices: torch.Tensor) -> torch.Tensor:
        """
        Construct block diagonal matrices from a batch (T) of (lag, num_dim, num_dim) slices,
        without a 'Trial' dimension.

        Args:
            matrices (torch.Tensor): Shape (T, lag, num_dim, num_dim)

        Returns:
            torch.Tensor: Block diagonal matrices of shape (T, lag*num_dim, lag*num_dim)
        """
        # Extract dimensions
        T, lag, num_dim, _ = matrices.shape
        device = matrices.device
        dtype = matrices.dtype

        # Prepare the output tensor
        # shape => (T, lag*num_dim, lag*num_dim)
        Q = torch.zeros(T, lag * num_dim, lag * num_dim, dtype=dtype, device=device)

        # Fill in each diagonal block
        for i in range(lag):
            start = i * num_dim
            end = (i + 1) * num_dim
            # matrices[:, i] has shape (T, num_dim, num_dim)
            Q[:, start:end, start:end] = matrices[:, i]

        return Q
    
    @staticmethod
    def _batch_block_diag_loop_no_trial_no_time(matrices: torch.Tensor) -> torch.Tensor:
        """
        Construct a single block-diagonal matrix from a stack of `lag` square matrices,
        *without* trial or time dimension.

        Args:
            matrices (torch.Tensor): Shape (lag, num_dim, num_dim)
                A sequence of `lag` matrices, each (num_dim x num_dim).

        Returns:
            torch.Tensor: One block diagonal matrix of shape (lag*num_dim, lag*num_dim),
                        where each block is matrices[i].
        """
        # matrices => (lag, num_dim, num_dim)
        lag, num_dim, _ = matrices.shape
        device = matrices.device
        dtype = matrices.dtype

        # Prepare the output tensor => (lag*num_dim, lag*num_dim)
        block_size = lag * num_dim
        block_diag = torch.zeros(block_size, block_size, dtype=dtype, device=device)

        # Fill each diagonal block
        for i in range(lag):
            start = i * num_dim
            end = (i + 1) * num_dim
            # matrices[i] => (num_dim, num_dim)
            block_diag[start:end, start:end] = matrices[i]

        return block_diag
    
    @staticmethod
    def _block_diag(matrices: Sequence[Tensor]) -> Tensor:
        """Batch-aware block-diag for list of (T, dᵢ, dᵢ) tensors."""
        T = matrices[0].size(0)
        d_total = sum(m.size(1) for m in matrices)
        out = matrices[0].new_zeros((T, d_total, d_total))
        idx = 0
        for m in matrices:
            d = m.size(1)
            out[:, idx : idx + d, idx : idx + d] = m
            idx += d
        return out

    def compute_adm(
        self,
        kernel_fn: Callable[[Tensor, Tuple[Tensor, Tensor], int], Tensor],
        params: Tuple[Tensor, Tensor],
        eps: float = 1e-4,
    ) -> Tuple[Tensor, Tensor]:
        """Return (Aₜ, Qₜ) for **time-varying** LDS (shape T × …)."""
        sigmas, delays = params  # each (T, num_dim)
        T, _ = delays.shape
        lag = self.lag
        d = self.num_dim

        # τ matrix (lag+1)×(lag+1)
        tau = torch.arange(0, lag + 1, device=self.device, dtype=self.dtype)
        tau = tau.unsqueeze(1) - tau.unsqueeze(0)

        # Kernel blocks: (T, lag+1, lag+1, d, d)
        D_blocks = kernel_fn(tau, (sigmas, delays), d)
        big = (
            D_blocks.permute(0, 1, 3, 2, 4)
            .reshape(T, (lag + 1) * d, (lag + 1) * d)
        )
        eye = torch.eye(big.size(-1), device=self.device, dtype=self.dtype).unsqueeze(0)
        L = torch.linalg.cholesky(big + eps * eye)
        R = L.transpose(-2, -1)

        n_p = d * lag
        R11, R12, R22 = R[:, :n_p, :n_p], R[:, :n_p, n_p:], R[:, n_p:, n_p:]
        inv_R11 = torch.linalg.solve_triangular(R11, torch.eye(n_p, device=self.device).expand(T, -1, -1), upper=True)
        B = (inv_R11 @ R12).transpose(-2, -1)  # (T, d, n_p)
        C = (R22.transpose(-2, -1) @ R22) / 2

        # Build A
        A_parts = B.view(T, d, lag, d)[:, :, torch.arange(lag - 1, -1, -1, device=self.device), :]
        A_flat = A_parts.reshape(T, d, lag * d)
        A_all = torch.cat([A_flat, self.IO.expand(T, -1, -1)], dim=1)

        # Build Q
        # Q_all = self._block_diag([C, self.O.expand(T, -1, -1)])

        C_expanded = C.unsqueeze(1).expand(-1, lag, d, d)
        Q_all = self._batch_block_diag_loop_no_trial(C_expanded)

        return A_all, Q_all

    def compute_adm_no_time(
        self,
        kernel_fn: Callable[[Tensor, Tensor, int], Tensor],
        params: Tensor,
        eps: float = 1e-4,
    ) -> Tuple[Tensor, Tensor]:
        """Return (A, Q) for **time-invariant** LDS (no batch dim)."""
        lag, d = self.lag, self.num_dim
        tau = torch.arange(0, lag + 1, dtype=self.dtype, device=self.device)
        tau = tau.unsqueeze(1) - tau.unsqueeze(0)

        D_blocks = kernel_fn(tau, params, d)  # (lag+1, lag+1, d, d)
        big = D_blocks.permute(0, 2, 1, 3).reshape((lag + 1) * d, (lag + 1) * d)
        eye = torch.eye(big.size(0), device=self.device, dtype=self.dtype)
        R = torch.linalg.cholesky(big + eps * eye).transpose(-2, -1)

        n_p = d * lag
        R11, R12, R22 = R[:n_p, :n_p], R[:n_p, n_p:], R[n_p:, n_p:]
        inv_R11 = torch.linalg.solve_triangular(R11, torch.eye(n_p, device=self.device), upper=True)
        B = (inv_R11 @ R12).transpose(-2, -1)
        C = (R22.transpose(-2, -1) @ R22) / 2

        A_flat = B.view(d, lag, d)[:, torch.arange(lag - 1, -1, -1, device=self.device), :].reshape(d, lag * d)
        A_all = torch.cat([A_flat, self.IO], dim=0)
        # Q_all = torch.block_diag(C, self.O)

        C_expanded = C.unsqueeze(0).expand(lag, d, d)
        Q_all = self._batch_block_diag_loop_no_trial_no_time(C_expanded)
        return A_all, Q_all


    # ----------------------------------------------------------------------
    #  Public helpers
    # ----------------------------------------------------------------------

    def sample(self, num_trial: int, T: int) -> Tensor:
        """Generate synthetic observations **y** of length *T* for *num_trial* runs."""
        if not hasattr(self, "A"):
            raise RuntimeError("Call forward() once before sampling to instantiate A,Q.")
        A, Q, C = self.A.cpu(), self.Q.cpu(), self.C.cpu()  # type: ignore[attr-defined]
        d_x, d_y = A.size(0), C.size(0)
        xs = torch.zeros((num_trial, T, d_x))
        ys = torch.zeros((num_trial, T, d_y))
        for k in range(num_trial):
            x_prev = torch.distributions.MultivariateNormal(torch.zeros(d_x), Q).sample()
            xs[k, 0] = x_prev
            ys[k, 0] = (C @ x_prev).squeeze()
            for t in range(1, T):
                q = torch.distributions.MultivariateNormal(torch.zeros(d_x), Q).sample()
                x_prev = A @ x_prev + q
                xs[k, t] = x_prev
                ys[k, t] = (C @ x_prev).squeeze()
        return ys

    # ----------------------------------------------------------------------
    #  Delay smoothing helper (used in MOSESSM)
    # ----------------------------------------------------------------------

    def _smooth_delay(self, delays: Tensor, kernel: Tensor) -> Tensor:
        k = kernel.size(-1)
        pad_right  = k - 1          # makes output length = T

        x = delays.T.unsqueeze(1)   # (N,1,T)

        x_pad = F.pad(x, (0, pad_right), mode="reflect")
        # x_pad = F.pad(x, (pad_left, pad_right), mode="constant", value=0.0)
        # print(x_pad.shape, kernel.repeat(x.size(0), 1, 1).shape, x.shape)
        y  = F.conv1d(x_pad,
                        kernel.repeat(x.size(1), 1, 1),
                        groups=x.size(1)
                        )

        y = y.squeeze(1).T       # (T, N)
        
        return y[:delays.size(0)]


# ---------------------------------------------------------------------------- #
#  Specific SSM variants (MOSES / SOSE)
# ---------------------------------------------------------------------------- #

class MOSESSM(SSM):
    """Multivariate latent with Off-diagonal Spatio-temporal Embedded Structure."""

    def __init__(
        self,
        lag: int,
        T: int,
        num_dim: int,
        eps: float,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(lag, num_dim, device, dtype=dtype)
        self.T: int = T
        self.eps = eps
        self.log_sigma = nn.Parameter(torch.log(torch.full((), 0.05, dtype=dtype, device=self.device)))
        self.delays = nn.Parameter(torch.zeros((T, num_dim - 1), dtype=dtype, device=self.device))

        self.gaussian_filter: Tensor = self.gaussian_kernel(6, 0.05)
        self.kernel_func: Callable[..., Tensor] = MOSE_kernel_parallel  # type: ignore[name-defined]

        assert num_dim >= 2, "MOSE model requires at least 2 latent dims."

    # ------------------------------------------------------------------
    #  Model forward: returns (A,Q)
    # ------------------------------------------------------------------

    def forward(self) -> Tuple[Tensor, Tensor]:
        sigma = torch.exp(self.log_sigma)
        smoothed_delays = self._smooth_delay(self.delays, self.gaussian_filter)
        full_delays = torch.cat(
            (
                torch.zeros((self.T, 1), dtype=self.dtype, device=self.device),
                smoothed_delays,
            ),
            dim=1,
        )
        params = sigma, full_delays
        return self.compute_adm(self.kernel_func, params, self.eps)

    # Convenience
    def get_delays(self) -> Tensor:
        return self._smooth_delay(self.delays, self.gaussian_filter)

class SOSESSM(SSM):
    """
    Scalar-output SSM with Squared-Exponential (SE) spatial kernel.

    """
    def __init__(
        self,
        lag: int,
        num_dim: int,
        eps: float,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if num_dim != 1:
            raise ValueError("SOSESSM supports only num_dim == 1 (scalar latent).")
        super().__init__(lag, num_dim, device, dtype=dtype)
        self.eps = eps
        # One log-σ parameter (learned) --------------------------------------
        self.log_sigma: nn.Parameter = nn.Parameter(
            torch.log(torch.full((), 0.05, dtype=dtype, device=self.device))
        )

        self.kernel_func: Callable[[Tensor, Tensor, int], Tensor] = SE_kernel_parallel  # type: ignore[name-defined]

    def forward(self) -> Tuple[Tensor, Tensor]:
        sigma = torch.exp(self.log_sigma)
        params = sigma
        return self.compute_adm_no_time(self.kernel_func, params, self.eps)