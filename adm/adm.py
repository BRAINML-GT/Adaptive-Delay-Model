
import torch
from torch import nn
import numpy as np
from torch.nn.functional import normalize
from adm.kalman_filter import KalmanFilter, GaussianState
from adm.dynamics import MOSESSM, SOSESSM
from typing import Sequence, Tuple, Optional

class ADM(nn.Module):
    """
    Adaptive Delay Model (ADM).

    Args
    ----
    lag:        Across-region lag (`lag_a`).
    xdima:      Latent dimension for across-region factors.
    xdimw:      Latent dimension for within-region factors.
    y_dims:     List of observed dimensions for each brain region.
    T:          Sequence length.
    device:     Torch device (default: "cpu").
    dtype:      Torch dtype   (default: torch.float32).
    """
    # ─────────────────────────────────────────────────────────────────── init ──
    def __init__(
        self,
        lag: int,
        xdima: int,
        xdimw: int,
        y_dims: Sequence[int],
        T: int,
        eps: float,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        # — constants ---------------------------------------------------------
        self.device = torch.device(device)
        self.dtype  = dtype
        self.lag_a  = lag        # across-region lag
        self.lag_w  = 2          # within-region lag (fixed)
        self.xdima      = xdima      # # across latent groups
        self.xdimw      = xdimw      # # within latent groups
        self.num_regions      = len(y_dims)
        self.T      = T

        # — latent dynamics modules ------------------------------------------
        self.across_dynamics = nn.ModuleList(
            MOSESSM(self.lag_a, T, self.num_regions, eps, self.device, self.dtype)
            for _ in range(self.xdima)
        )
        self.within_dynamics = nn.ModuleList(
            SOSESSM(self.lag_w, 1, eps, self.device, self.dtype)
            for _ in range(self.xdimw * self.num_regions)
        )

        # — latent geometry ---------------------------------------------------
        self.xdim = self.xdima * self.lag_a * self.num_regions + self.num_regions * self.xdimw * self.lag_w
        self.gdim = self.num_regions * (self.xdima + self.xdimw)

        # — index helpers -----------------------------------------------------
        cum = torch.cumsum(torch.tensor([0, *y_dims], device=self.device), 0)
        self.register_buffer("cum_y_dims", cum.to(torch.int64))

        self.register_buffer("x_shift_indices", self._build_x_shift_indices())
        x_indices, y_indices = self._build_xy_indices()

        H = torch.zeros(self.gdim, self.xdim, dtype=self.dtype, device=self.device)
        H[y_indices, x_indices] = 1.0
        self.register_buffer("H", H)                                       # (gdim × xdim)

        # # — observation model -------------------------------------------------
        self.Cs: list[torch.Tensor] = []
        for i, ydim in enumerate(y_dims):
            C_i = nn.init.xavier_uniform_(torch.randn(ydim, xdima + xdimw, dtype=dtype, device=device))
            setattr(self, f"C_{i}", C_i)
            self.Cs.append(C_i)

        self.diag_Rs = 0.1 * torch.ones(sum(y_dims), dtype=dtype, device=device)
        self.ds       = torch.zeros(sum(y_dims), dtype=dtype, device=device)

    # ───────────────────────────────────────────────────────────────── helpers ──
    def _build_xy_indices(self) -> torch.Tensor:
        """
        Return index mapping from global latent index (g) → x index (x).
        """
        start_within = self.xdima * self.lag_a * self.num_regions  # Starting index of within components in x_t

        y_indices = []
        x_indices = []

        # Build the index mapping from x_t to g_t
        for j in range(self.num_regions): 
            # Across components (a_j^i)
            for i in range(self.xdima):
                # Position of a_j^i in x_t
                x_idx = i * self.lag_a * self.num_regions + j 
                # Position in y_t
                y_idx = j * (self.xdima + self.xdimw) + i
                x_indices.append(x_idx)
                y_indices.append(y_idx)
            
            # Within components (c_j^i)
            for i in range(self.xdimw):
                # Position of c_j^i in x_t
                x_idx = start_within + (j * self.xdimw + i) * self.lag_w 
                # Position in y_t
                y_idx = j * (self.xdima + self.xdimw) + self.xdima + i
                x_indices.append(x_idx)
                y_indices.append(y_idx)

        # Convert indices to tensors
        y_indices = torch.tensor(y_indices, dtype=torch.long)
        x_indices = torch.tensor(x_indices, dtype=torch.long)

        return x_indices, y_indices
    
    def _build_x_shift_indices(self) -> torch.Tensor:
        """
        Return index to extract latent x.
        """
        x_indexes = torch.zeros((self.num_regions, self.xdima+self.xdimw)).to(torch.int)
        across_dims = self.num_regions*self.lag_a*self.xdima
        within_dims = self.num_regions*self.lag_w*self.xdimw
        within_indexes = torch.arange(across_dims, across_dims+within_dims, self.lag_w)
        for i in range(self.num_regions):
            if self.xdima > 0:
                indexes = torch.arange(i, across_dims, self.num_regions*self.lag_a)
            else:
                indexes = torch.tensor([], dtype=torch.int)
            for j in range(self.xdimw):
                tmp_within = within_indexes[i * self.xdimw + j]
                if len(tmp_within.shape) == 0:
                    tmp_within = tmp_within.unsqueeze(0)
                indexes = torch.cat((indexes, tmp_within))

            x_indexes[i] = indexes
        
        return x_indexes

    # ────────────────────────────────────────────────────────────── regression ──
    def fit_linear_regression(
        self,
        Xs: torch.Tensor,              # (Trials, N, p)
        ys: torch.Tensor,              # (Trials, N, d)
        weights: Optional[torch.Tensor] = None,
        *,
        fit_intercept: bool = True,
        expectations: Optional[Tuple[torch.Tensor, ...]] = None,
        prior_ExxT: Optional[torch.Tensor] = None,
        prior_ExyT: Optional[torch.Tensor] = None,
        nu0: float = 1.0,
        Psi0: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Bayesian linear regression y ~ 𝒩(Wx + b, Σ).

        Returns
        -------
        W … (d × p)      - weight matrix  
        b … (d)          - intercept        (zeros if `fit_intercept=False`)  
        diag Σ … (d)     - observation noise diagonal
        """
        Trials, N, p = Xs.shape
        _, _, d      = ys.shape
        device       = Xs.device
        dtype        = Xs.dtype
        x_dim        = p + int(fit_intercept)

        # — (E[xxᵀ], E[xyᵀ], …) ----------------------------------------------
        if expectations is None:
            ExxT = torch.zeros((x_dim, x_dim), device=device, dtype=dtype)
            ExyT = torch.zeros((x_dim, d),     device=device, dtype=dtype)
            EyyT = torch.zeros((d, d),         device=device, dtype=dtype)

            # priors
            if prior_ExxT is not None: ExxT += prior_ExxT
            if prior_ExyT is not None: ExyT += prior_ExyT

            X = Xs.reshape(-1, p)
            Y = ys.reshape(-1, d)

            w = weights.reshape(-1, 1) if weights is not None else torch.ones((Trials * N, 1), device=device, dtype=dtype)
            if fit_intercept:
                X = torch.cat((X, torch.ones_like(w)), dim=1)             # (TN × x_dim)

            weight_sum  = w.sum()
            Xw, Yw      = X * w, Y * w
            ExxT       += Xw.T @ X
            ExyT       += Xw.T @ Y
            EyyT       += Yw.T @ Y
        else:
            ExxT, ExyT, EyyT, weight_sum = expectations                # type: ignore[arg-type]

        # — MAP estimates -----------------------------------------------------
        W_full = torch.linalg.solve(ExxT, ExyT).T                       # (d × x_dim)
        W, b   = (W_full[:, :-1], W_full[:, -1]) if fit_intercept else (W_full, torch.zeros(d, device=device, dtype=dtype))

        err      = EyyT - W_full @ ExyT - ExyT.T @ W_full.T + W_full @ ExxT @ W_full.T
        nu_n     = nu0 + weight_sum
        Psi_n    = err + Psi0 * torch.eye(d, device=device, dtype=dtype)
        Sigma_di = Psi_n.diagonal() / (nu_n + d + 1)

        return W, b, Sigma_di

    def log_likelihood_observation(
        self,
        x: torch.Tensor,          # (B, T, D)
        y: torch.Tensor,          # (B, T, M)
        C: torch.Tensor,          # (M, D)   time-invariant
        R: torch.Tensor,          # (M, M)   time-invariant
    ) -> torch.Tensor:
        """
        log-likelihood for the observation

        """
        B, T, D = x.shape
        M = y.shape[-1]
        
        log_lik = 0.0
        inv_R = torch.linalg.inv(R)
        logdet_R = torch.logdet(R)

        for t in range(T):
            Cx_t = torch.einsum('md,bd->bm', C, x[:,t])
            diff_y = y[:,t] - Cx_t  # => (B,M)
            diff_y_3d = diff_y.unsqueeze(1)             # (B,1,M)
            inv_R_exp = inv_R.unsqueeze(0).expand(B,M,M)# (B,M,M)
            tmp_obs = torch.bmm(diff_y_3d, inv_R_exp)   # => (B,1,M)
            quad_obs_2d = torch.bmm(tmp_obs, diff_y.unsqueeze(-1))  # (B,1,1)
            quad_obs = quad_obs_2d.squeeze(-1).squeeze(-1)         # (B,)

            nll_obs_t_b = 0.5 * (
                quad_obs
                + logdet_R
                + M*np.log(2*np.pi)
            )
            ll_obs_t = - nll_obs_t_b.sum()
            log_lik += ll_obs_t

        return log_lik


    def expected_loglikelihood_xy(
        self,
        A: torch.Tensor,       # (T, D, D)
        Q: torch.Tensor,       # (T, D, D)
        C: torch.Tensor,       # (obs_dim, D)
        R_diag: torch.Tensor,  # (obs_dim,)     diagonal of R
        Ex: torch.Tensor,      # (T, NT, D, 1)   posterior means E[x_t]
        Exx: torch.Tensor,     # (T, NT, D, D)   posterior covariances E[x_t x_t^T]
        Exx_prev: torch.Tensor,# (T-1, NT, D, D) cross-covs E[x_t x_{t-1}^T]
        ys: torch.Tensor       # (NT, T, obs_dim)
    ) -> torch.Tensor:
        """
        Parallel computation of:
        1) The expected log-likelihood of x_t under x_t~N(A_t x_{t-1}, Q_t)
        2) The expected log-likelihood of y_t under y_t~N(C x_t, R)

        """
        ys = ys.permute(1, 0, 2)
        T, NT, D, _ = Ex.shape
        

        # Build residual for t=1..T-1
        A_exp = A.unsqueeze(1)   # (T, 1, D, D)
        A_T_exp= A.mT.unsqueeze(1)  # (T, 1, D, D)

        # residual_x => (T-1, NT, D, D)
        # Use the formula: Exx[t] - A[t]*Exx_prev[t-1]^T - Exx_prev[t-1]*A[t]^T + A[t]*Exx_prev[t-1]*A[t]^T
        residual_x = (
            Exx[1:] 
            - torch.matmul(A_exp[1:], Exx_prev.mT)
            - torch.matmul(Exx_prev, A_T_exp[1:])
            + torch.matmul(torch.matmul(A_exp[1:], Exx_prev), A_T_exp[1:])
        )  # => shape (T-1, NT, D, D)


        S_x = residual_x.sum(dim=1)   # shape => (T-1, D, D) # for sum LL

        Q_t = Q[1:]        # shape => (T-1, D, D)
        logdet_Q = torch.logdet(Q_t)  # batch of size (T-1)
        Q_inv = torch.linalg.inv(Q_t)
        QS_x = torch.matmul(Q_inv, S_x)
        trace_term_x = QS_x.diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1)  # => (T-1,)

        # negative log-likelihood for x:  0.5 [ sum_{t=1..T-1} NT * logdet(Q_t) + trace(Q_t^-1 S_x[t]) ]
        ll_x = -0.5 * (NT * logdet_Q.sum() + trace_term_x.sum()) # for sum LL

        # logdet(R) => sum(log(R_diag)), a scalar. => shape ()
        logdet_R = torch.log(R_diag).sum()  # float scalar

        X_t_2 = Ex.squeeze(-1)       # => (T,NT,D)
        obs_mean = torch.einsum('tnd,md->tnm', X_t_2, C)  # => (T,NT,obs_dim)
        diff_y = ys - obs_mean       # => (T,NT,obs_dim)

        cSigmaCt_diag = torch.einsum(
            'md, t i d d, dm -> t i m',
            C, Exx, C.T
        )  # => shape (T, NT, m)

        diff_sq = diff_y * diff_y   # => (T,NT,m)
        obs_res = diff_sq + cSigmaCt_diag

        obs_div = obs_res / R_diag.unsqueeze(0).unsqueeze(0)

        sum_obs = obs_div.sum(dim=(1,2)) # for sum LL

        ll_y = -0.5 * ( sum_obs.sum() + (T * NT) * logdet_R ) # for sum LL

        # Sum them up
        ll = ll_x + ll_y
        return ll

    def black_diagonal_batch(self, As, Qs):
        T, C, C = As[0].shape
        _, Cw, Cw = As[-1].shape
        # Create a batched zero tensor for off-diagonal blocks
        num_dims = self.xdima * C + self.xdimw * Cw * self.num_regions
        As_zeros_off_diag = torch.zeros(T, num_dims, num_dims).to(As[0].device).to(As[0].dtype)  # Shape: (T, nC, nC)
        Qs_zeros_off_diag = torch.zeros(T, num_dims, num_dims).to(As[0].device).to(As[0].dtype)  # Shape: (T, nC, nC)

        offset = 0
        for i, (A, Q) in enumerate(zip(As, Qs)):
            c = A.shape[1]
            As_zeros_off_diag[:, offset:offset + c, offset:offset + c] = A
            Qs_zeros_off_diag[:, offset:offset + c, offset:offset + c] = Q
            offset += c

        # The result is a tensor with block-diagonal structure
        return As_zeros_off_diag, Qs_zeros_off_diag

    def _block_diag_time_batch(
        self,
        As: list[torch.Tensor],
        Qs: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Assemble time–batched block-diagonal dynamics.

        Each entry in `As` / `Qs` is a tensor of shape **(T, Cᵢ, Cᵢ)** for a
        particular latent group *i*.  The routine returns *(A_blk, Q_blk)* with
        shapes **(T, Σ Cᵢ, Σ Cᵢ)** where the (Cᵢ x Cᵢ) blocks lie on the diagonal.

        Notes
        -----
        * Off-diagonal blocks are zero.
        * All tensors are created on the device / dtype of the first `As` entry.
        """
        T, C0, _ = As[0].shape
        num_dims = sum(A.shape[1] for A in As)          # Σ Cᵢ

        dev, dt = As[0].device, As[0].dtype
        A_blk   = torch.zeros(T, num_dims, num_dims, device=dev, dtype=dt)
        Q_blk   = torch.zeros_like(A_blk)

        offset = 0
        for A_i, Q_i in zip(As, Qs):
            c = A_i.shape[1]
            A_blk[:, offset : offset + c, offset : offset + c] = A_i
            Q_blk[:, offset : offset + c, offset : offset + c] = Q_i
            offset += c

        return A_blk, Q_blk


    def forward(self, y: torch.Tensor, *, parallel: bool = True):
        """
        Negative log-likelihood **nLL** and smoothed latent means given observation *y*.

        Parameters
        ----------
        y        : (trials, T, neurons) observation tensor.
        parallel : Use parallel Kalman filtering when *True*.

        Returns
        -------
        nLL      : Scalar negative log-likelihood.
        x_smooth : (trials, T, xdim) smoothed latent means.
        """
        trials, T = y.shape[0], y.shape[1]

        # — collect group-wise dynamics -----------------------------------------
        A_list, Q_list = [], []
        for dyn in self.across_dynamics:
            A_i, Q_i = dyn.forward()                    # (T, C, C)
            A_list.append(A_i)
            Q_list.append(Q_i)

        for dyn in self.within_dynamics:
            A_i, Q_i = dyn.forward()
            A_list.append(A_i.expand(T, -1, -1))
            Q_list.append(Q_i.expand(T, -1, -1))

        A_blk, Q_blk = self._block_diag_time_batch(A_list, Q_list)
        C_blk        = torch.block_diag(*self.Cs)       # (Σ y_dim × xdim)
        R_blk        = torch.diag(self.diag_Rs)

        # observation centering
        y_centered = y - self.ds

        C_eff = C_blk @ self.H                         # (Σ y_dim × xdim)
        kf    = KalmanFilter(A_blk, C_eff, Q_blk, R_blk)

        μ0 = torch.zeros(self.xdim, 1, dtype=self.dtype, device=self.device)
        Σ0 = torch.eye(self.xdim,     dtype=self.dtype, device=self.device)

        # if parallel is True, then use parallel scan based kalman filter
        if parallel:
            states = kf.filter_parallel(y_centered, A_blk, C_eff, Q_blk, R_blk, μ0, Σ0)
        # if parallel is false, then use naive kalman filter
        else:
            init = GaussianState(μ0, Σ0)
            states = kf.filter(init, y_centered.permute(1, 0, 2)[..., None],
                            update_first=True, return_all=True)

        states = GaussianState(
            states.mean,
            states.covariance[:, None].expand(-1, trials, -1, -1),
        )
        smoothed, cross_cov = kf.rts_smooth(states)

        LL  = self.expected_loglikelihood_xy(
            A_blk, Q_blk, C_eff, self.diag_Rs, smoothed.mean,
            smoothed.covariance, cross_cov, y_centered,
        )
        return -LL, smoothed.mean.transpose(1, 0)       # nLL, (trials × T × xdim)


    def update_C_and_R(self, ys: torch.Tensor, xs: torch.Tensor):
        """
        Re-estimate per-region emission matrices (Cᵢ), noise (Rᵢ), and offsets (dᵢ).

        Returns the updated observations reconstructed from the new parameters.
        """
        xs_norm  = normalize(xs, p=float("inf"), dim=1)
        C_list, Rdiag_list, d_list, xs_cat = [], [], [], []

        for r in range(self.num_regions):
            x_r = xs_norm[:, :, self.x_shift_indices[r], 0]                       # (trials, T, Kᵣ)
            y_r = ys[:, :, self.cum_y_dims[r] : self.cum_y_dims[r + 1]]

            C_r, d_r, Rdiag_r = self.fit_linear_regression(
                x_r, y_r,
                prior_ExxT = 1e-8 * torch.eye(x_r.shape[-1] + 1, device=x_r.device),
                prior_ExyT = torch.zeros(x_r.shape[-1] + 1, y_r.shape[-1], device=x_r.device),
                fit_intercept=True,
            )
            C_list.append(C_r)
            Rdiag_list.append(Rdiag_r)
            d_list.append(d_r)
            xs_cat.append(x_r)

        self.Cs       = C_list
        self.diag_Rs  = torch.cat(Rdiag_list)
        self.ds       = torch.cat(d_list)

        C_blk         = torch.block_diag(*C_list)
        xs_concat     = torch.cat(xs_cat, dim=2)
        y_recon       = (C_blk @ xs_concat[..., None]).squeeze(-1) + self.ds

        return y_recon, xs_concat


    def update_ys(self, xs: torch.Tensor):
        """
        Generate observations **ys** from latents **xs** using current `C`, `d`.

        Returns
        -------
        ys          : (trials, T, Σ y_dim) reconstructed observations
        xs_concat   : (trials, T, xdim_sel) concatenated latents
        """
        xs_norm, xs_raw = [], []
        for r in range(self.num_regions):
            xs_norm.append(normalize(xs[:, :, self.x_shift_indices[r], 0], p=float("inf"), dim=1))
            xs_raw.append(xs[:, :, self.x_shift_indices[r], 0])

        C_blk       = torch.block_diag(*self.Cs)
        xs_norm_cat = torch.cat(xs_norm, dim=2)
        xs_raw_cat  = torch.cat(xs_raw,  dim=2)

        ys = (C_blk @ xs_norm_cat[..., None]).squeeze(-1) + self.ds
        return ys, xs_raw_cat

    # def fit_linear_regression_new(self, Xs, ys,
    #                       weights=None,
    #                       fit_intercept=True,
    #                       expectations=None,
    #                       prior_ExxT=None,
    #                       prior_ExyT=None,
    #                       nu0=1,
    #                       Psi0=1):
    #     """
    #     Fit a linear regression y_i ~ N(Wx_i + b, Sigma) for W, b, Sigma.

    #     Parameters
    #     ----------
    #     Xs: Tensor
    #         Trials x N x D tensor, where Trials is the number of trials,
    #         N is the number of data points per trial,
    #         and D is the dimension of x_i.
    #     ys: Tensor
    #         Trials x N x P tensor, where P is the dimension of y_i.
    #     weights: Optional; Tensor
    #         Trials x N tensor of weights for each observation.
    #     fit_intercept: bool
    #         If False, the intercept term b is dropped.
    #     expectations: Optional; tuple
    #         Precomputed sufficient statistics (Exx, Exy, Eyy, weight_sum).
    #         If provided, Xs and ys are ignored.
    #     prior_ExxT: Optional; Tensor
    #         D x D prior matrix. Used when expectations is None.
    #     prior_ExyT: Optional; Tensor
    #         D x P prior matrix. Used when expectations is None.
    #     nu0: float
    #         Prior degrees of freedom for the covariance.
    #     Psi0: float
    #         Prior scale matrix for the covariance.

    #     Returns
    #     -------
    #     W: Tensor
    #         Weight matrix.
    #     b: Tensor (if fit_intercept is True)
    #         Intercept vector.
    #     Sigma: Tensor
    #         Covariance matrix.
    #     """
    #     # Check dimensions
    #     Trials, N, p = Xs.shape
    #     _, _, d = ys.shape
    #     assert ys.shape[0] == Trials and ys.shape[1] == N, "ys must have the same shape as Xs in first two dimensions."

    #     device = Xs.device
    #     dtype = Xs.dtype

    #     x_dim = p + int(fit_intercept)

    #     if expectations is None:
    #         ExxT = torch.zeros((x_dim, x_dim), device=device, dtype=dtype)
    #         ExyT = torch.zeros((x_dim, d), device=device, dtype=dtype)
    #         EyyT = torch.zeros((d, d), device=device, dtype=dtype)
    #         weight_sum = 0

    #         # Incorporate prior if provided
    #         if prior_ExxT is not None and prior_ExyT is not None:
    #             if prior_ExxT.shape != (x_dim, x_dim):
    #                 raise ValueError(f"prior_ExxT must have shape ({x_dim}, {x_dim}), but has shape {prior_ExxT.shape}")
    #             if prior_ExyT.shape != (x_dim, d):
    #                 raise ValueError(f"prior_ExyT must have shape ({x_dim}, {d}), but has shape {prior_ExyT.shape}")
    #             ExxT += prior_ExxT
    #             ExyT += prior_ExyT

    #         # Flatten Trials and N into one dimension
    #         Xs_all = Xs.reshape(-1, p)  # Shape: (Trials*N) x p
    #         ys_all = ys.reshape(-1, d)  # Shape: (Trials*N) x d

    #         # Default weights to ones if not provided
    #         if weights is not None:
    #             assert weights.shape == (Trials, N), "weights must have shape (Trials, N)"
    #             weights_all = weights.reshape(-1, 1)  # Shape: (Trials*N) x 1
    #         else:
    #             weights_all = torch.ones((Trials*N, 1), device=device, dtype=dtype)

    #         if fit_intercept:
    #             ones = torch.ones((Trials*N, 1), device=device, dtype=dtype)
    #             Xs_all = torch.cat((Xs_all, ones), dim=1)  # Shape: (Trials*N) x x_dim

    #         weight_sum = torch.sum(weights_all)
    #         weighted_x = Xs_all * weights_all  # Element-wise multiplication
    #         weighted_y = ys_all * weights_all

    #         ExxT += weighted_x.T @ Xs_all  # x_dim x x_dim
    #         ExyT += weighted_x.T @ ys_all   # x_dim x d
    #         EyyT += weighted_y.T @ ys_all   # d x d

    #     else:
    #         ExxT, ExyT, EyyT, weight_sum = expectations
    #         if ExxT.shape != (x_dim, x_dim):
    #             raise ValueError(f"ExxT must have shape ({x_dim}, {x_dim}), but has shape {ExxT.shape}")
    #         if ExyT.shape != (x_dim, d):
    #             raise ValueError(f"ExyT must have shape ({x_dim}, {d}), but has shape {ExyT.shape}")
    #         if EyyT.shape != (d, d):
    #             raise ValueError(f"EyyT must have shape ({d}, {d}), but has shape {EyyT.shape}")

    #     # Solve for the MAP estimate
    #     W_full = torch.linalg.solve(ExxT, ExyT).T  # Shape: d x x_dim
    #     if fit_intercept:
    #         W, b = W_full[:, :-1], W_full[:, -1]
    #     else:
    #         W = W_full
    #         b = torch.zeros(d, device=device, dtype=dtype)

    #     # Compute expected error for covariance matrix estimate
    #     # expected_err = EyyT - W_full @ ExyT - ExyT.T @ W_full.T + W_full @ ExxT @ W_full.T
    #     # nu = nu0 + weight_sum

    #     # Get MAP estimate of posterior covariance
    #     # Psi0_eye = Psi0 * torch.eye(d, device=device, dtype=dtype)
    #     # Sigma = (expected_err + Psi0_eye) / (nu + d + 1)
    #     expected_err = EyyT - W_full @ ExyT - ExyT.T @ W_full.T + W_full @ ExxT @ W_full.T
    #     nu_n = nu0 + weight_sum
    #     Psi_n = expected_err + Psi0 * torch.eye(d, device=device, dtype=dtype)
    #     Sigma = Psi_n / (nu_n + d + 1)
    #     # Sigma = expected_err / weight_sum

    #     if fit_intercept:
    #         return W, b, torch.diag(Sigma)
    #     else:
    #         return W, torch.diag(Sigma)

    # def fit_linear_regression(self, Xs, ys,
    #                       weights=None,
    #                       fit_intercept=True,
    #                       expectations=None,
    #                       prior_ExxT=None,
    #                       prior_ExyT=None,
    #                       nu0=1,
    #                       Psi0=1):
    #     """
    #     Fit a linear regression y_i ~ N(Wx_i + b, Sigma) for W, b, Sigma.

    #     Parameters
    #     ----------
    #     Xs: Tensor
    #         Trials x N x D tensor, where Trials is the number of trials,
    #         N is the number of data points per trial,
    #         and D is the dimension of x_i.
    #     ys: Tensor
    #         Trials x N x P tensor, where P is the dimension of y_i.
    #     weights: Optional; Tensor
    #         Trials x N tensor of weights for each observation.
    #     fit_intercept: bool
    #         If False, the intercept term b is dropped.
    #     expectations: Optional; tuple
    #         Precomputed sufficient statistics (Exx, Exy, Eyy, weight_sum).
    #         If provided, Xs and ys are ignored.
    #     prior_ExxT: Optional; Tensor
    #         D x D prior matrix. Used when expectations is None.
    #     prior_ExyT: Optional; Tensor
    #         D x P prior matrix. Used when expectations is None.
    #     nu0: float
    #         Prior degrees of freedom for the covariance.
    #     Psi0: float
    #         Prior scale matrix for the covariance.

    #     Returns
    #     -------
    #     W: Tensor
    #         Weight matrix.
    #     b: Tensor (if fit_intercept is True)
    #         Intercept vector.
    #     Sigma: Tensor
    #         Covariance matrix.
    #     """
    #     # Check dimensions
    #     Trials, N, p = Xs.shape
    #     _, _, d = ys.shape
    #     assert ys.shape[0] == Trials and ys.shape[1] == N, "ys must have the same shape as Xs in first two dimensions."

    #     device = Xs.device
    #     dtype = Xs.dtype

    #     x_dim = p + int(fit_intercept)

    #     if expectations is None:
    #         ExxT = torch.zeros((x_dim, x_dim), device=device, dtype=dtype)
    #         ExyT = torch.zeros((x_dim, d), device=device, dtype=dtype)
    #         EyyT = torch.zeros((d, d), device=device, dtype=dtype)
    #         weight_sum = 0

    #         # Incorporate prior if provided
    #         if prior_ExxT is not None and prior_ExyT is not None:
    #             if prior_ExxT.shape != (x_dim, x_dim):
    #                 raise ValueError(f"prior_ExxT must have shape ({x_dim}, {x_dim}), but has shape {prior_ExxT.shape}")
    #             if prior_ExyT.shape != (x_dim, d):
    #                 raise ValueError(f"prior_ExyT must have shape ({x_dim}, {d}), but has shape {prior_ExyT.shape}")
    #             ExxT += prior_ExxT
    #             ExyT += prior_ExyT

    #         # Flatten Trials and N into one dimension
    #         Xs_all = Xs.reshape(-1, p)  # Shape: (Trials*N) x p
    #         ys_all = ys.reshape(-1, d)  # Shape: (Trials*N) x d

    #         # Default weights to ones if not provided
    #         if weights is not None:
    #             assert weights.shape == (Trials, N), "weights must have shape (Trials, N)"
    #             weights_all = weights.reshape(-1, 1)  # Shape: (Trials*N) x 1
    #         else:
    #             weights_all = torch.ones((Trials*N, 1), device=device, dtype=dtype)

    #         if fit_intercept:
    #             ones = torch.ones((Trials*N, 1), device=device, dtype=dtype)
    #             Xs_all = torch.cat((Xs_all, ones), dim=1)  # Shape: (Trials*N) x x_dim

    #         weight_sum = torch.sum(weights_all)
    #         weighted_x = Xs_all * weights_all  # Element-wise multiplication
    #         weighted_y = ys_all * weights_all

    #         ExxT += weighted_x.T @ Xs_all  # x_dim x x_dim
    #         ExyT += weighted_x.T @ ys_all   # x_dim x d
    #         EyyT += weighted_y.T @ ys_all   # d x d

    #     else:
    #         ExxT, ExyT, EyyT, weight_sum = expectations
    #         if ExxT.shape != (x_dim, x_dim):
    #             raise ValueError(f"ExxT must have shape ({x_dim}, {x_dim}), but has shape {ExxT.shape}")
    #         if ExyT.shape != (x_dim, d):
    #             raise ValueError(f"ExyT must have shape ({x_dim}, {d}), but has shape {ExyT.shape}")
    #         if EyyT.shape != (d, d):
    #             raise ValueError(f"EyyT must have shape ({d}, {d}), but has shape {EyyT.shape}")

    #     # Solve for the MAP estimate
    #     W_full = torch.linalg.solve(ExxT, ExyT).T  # Shape: d x x_dim
    #     if fit_intercept:
    #         W, b = W_full[:, :-1], W_full[:, -1]
    #     else:
    #         W = W_full
    #         b = torch.zeros(d, device=device, dtype=dtype)

    #     # Compute expected error for covariance matrix estimate
    #     expected_err = EyyT - W_full @ ExyT - ExyT.T @ W_full.T + W_full @ ExxT @ W_full.T
    #     nu = nu0 + weight_sum

    #     # Get MAP estimate of posterior covariance
    #     Psi0_eye = Psi0 * torch.eye(d, device=device, dtype=dtype)
    #     Sigma = (expected_err + Psi0_eye) / (nu + d + 1)

    #     if fit_intercept:
    #         return W, b, torch.diag(Sigma)
    #     else:
    #         return W, torch.diag(Sigma)
    
    # def fit_linear_regression2(self, x, y):
    #     """
    #     Fit the linear regression model:
    #         y ~ N(W x + b, Sigma)
    #     where:
    #     - x has shape (batch, T, D)
    #     - y has shape (batch, T, N)
    #     The estimates are global (batch-free):
    #     - W: (N, D)
    #     - b: (N, 1)
    #     - Sigma: (N, N) (diagonal)
        
    #     This function computes the sufficient statistics by summing over the 
    #     batch and time dimensions rather than flattening the data.
        
    #     Parameters:
    #     x (torch.Tensor): Input tensor of shape (batch, T, D)
    #     y (torch.Tensor): Output tensor of shape (batch, T, N)
        
    #     Returns:
    #     W (torch.Tensor): Estimated weight matrix of shape (N, D)
    #     b (torch.Tensor): Estimated bias vector of shape (N, 1)
    #     Sigma (torch.Tensor): Estimated diagonal covariance matrix of shape (N, N)
    #     """
    #     batch, T, D = x.shape
    #     _, _, N = y.shape

    #     # Augment x with a column of ones to account for the bias term.
    #     ones = torch.ones((batch, T, 1), dtype=x.dtype, device=x.device)
    #     X_aug = torch.cat([x, ones], dim=2)  # Shape: (batch, T, D+1)

    #     # Compute X^T X without flattening by summing over batch and time dimensions.
    #     # XTX has shape (D+1, D+1)
    #     XTX = torch.einsum('bti,btj->ij', X_aug, X_aug)
        
    #     # Compute X^T y: shape (D+1, N)
    #     XTY = torch.einsum('bti,btn->in', X_aug, y)
        
    #     # Solve for beta = [W^T; b^T] with shape (D+1, N)
    #     beta = torch.linalg.solve(XTX, XTY)
        
    #     # Extract W and b from beta.
    #     # First D rows correspond to W^T, last row corresponds to b^T.
    #     W = beta[:-1, :].T         # Shape: (N, D)
    #     b = beta[-1, :].unsqueeze(1)  # Shape: (N, 1)

    #     # Compute predictions and residuals to estimate Sigma.
    #     # y_pred is computed as: y_pred = x @ W^T + b^T.
    #     y_pred = torch.matmul(x, W.T) + b.T  # Shape: (batch, T, N)
    #     residuals = y - y_pred               # Shape: (batch, T, N)

    #     # Estimate Sigma (diagonal) using the mean squared error for each output dimension.
    #     sigma_diag = (residuals ** 2).mean(dim=(0, 1))  # Shape: (N,)

    #     return W, b.squeeze(), sigma_diag

    # def log_likelihood_observation(
    #     self,
    #     x: torch.Tensor,          # (B, T, D)
    #     y: torch.Tensor,          # (B, T, M)
    #     C: torch.Tensor,          # (M, D)   time-invariant
    #     R: torch.Tensor,          # (M, M)   time-invariant
    # ) -> torch.Tensor:
    #     """
    #     Loop-based (over time dimension) standard (complete-data) log-likelihood 
    #     for a linear-Gaussian SSM with multiple trials in parallel.

    #     Model:
    #     x_{b,0} ~ N(mu0, Sigma0)
    #     For t in [1..T-1]:
    #         x_{b,t} ~ N(A[t] x_{b,t-1}, Q[t])
    #     For t in [0..T-1]:
    #         y_{b,t} ~ N(C x_{b,t}, R)

    #     Shapes:
    #     x: (B, T, D)
    #     y: (B, T, M)
    #     A[t]: (D, D), Q[t]: (D, D) for t in [0..T-1]
    #     C: (M, D), R: (M, M)  (time-invariant)
    #     mu0: (D,), Sigma0: (D,D)

    #     Returns:
    #     log_lik: scalar => sum of log p(x,y) across all B trials and T steps.
    #     """
    #     B, T, D = x.shape
    #     M = y.shape[-1]

    #     # 1) log p(x_{b,0}) for each b => sum
    #     # inv_Sigma0 = torch.linalg.inv(Sigma0)
    #     # logdet_Sigma0 = torch.logdet(Sigma0)

    #     # # x[:,0]: shape (B,D). We do (x0_b - mu0) for each b => shape (B,D)
    #     # diff0 = x[:,0] - mu0    # (B,D)
    #     # # Quadratic form => (diff0^T Sigma0^-1 diff0) for each b => shape (B,)
    #     # # We'll do bmm approach or 'einsum':
    #     # diff0_3d = diff0.unsqueeze(1)                 # (B,1,D)
    #     # inv_Sig0_exp = inv_Sigma0.unsqueeze(0).expand(B,D,D)  # (B,D,D)
    #     # tmp_init = torch.bmm(diff0_3d, inv_Sig0_exp)         # (B,1,D)
    #     # quad_init_2d = torch.bmm(tmp_init, diff0.unsqueeze(-1)) # => (B,1,1)
    #     # quad_init = quad_init_2d.squeeze(-1).squeeze(-1)     # (B,)

    #     # # dimension constant => D log(2 pi), plus logdet(Sigma0)
    #     # # sum => shape => ()
    #     # nll_init_b = 0.5 * (
    #     #     quad_init
    #     #     + logdet_Sigma0
    #     #     + D*np.log(2*np.pi)
    #     # )  # (B,)
    #     # # sum over b => shape => ()
    #     # ll_init = - nll_init_b.sum()

    #     # # We'll accumulate log_lik in a variable
    #     # log_lik = ll_init

    #     # 2) transitions => separate loop over t in [1..T-1], no loop over B
    #     # but we do a python 'for t in range(1, T): ...' to handle time dimension
    #     # Each step, we do:
    #     #   diff_x = x[:,t] - A[t] x[:,t-1]
    #     #   -0.5 [ diff_x^T Q[t]^-1 diff_x + logdet(Q[t]) + D ln(2 pi) ] sum over b
    #     # for t in range(1, T):
    #     #     # diff_x => shape (B,D)
    #     #     # A[t] => shape (D,D)
    #     #     # x[:,t-1] => shape (B,D)
    #     #     # We'll do a parallel matmul => x[:,t-1] => (B,D), A[t] => (D,D).
    #     #     # Let Ax => shape (B,D).
    #     #     #   We can do an einsum or (A[t] x_{b,t-1}).
    #     #     A_t = A[t]    # (D,D)
    #     #     Q_t = Q[t]    # (D,D)
    #     #     inv_Q_t = torch.linalg.inv(Q_t)
    #     #     logdet_Q_t = torch.logdet(Q_t)

    #     #     Ax = torch.einsum('ij,bj->bi', A_t, x[:,t-1])  # (B,D)
    #     #     diff_x = x[:,t] - Ax                           # (B,D)

    #     #     # quad => shape => (B,)
    #     #     diff_x_3d = diff_x.unsqueeze(1)         # (B,1,D)
    #     #     inv_Q_exp = inv_Q_t.unsqueeze(0).expand(B,D,D)  # (B,D,D)
    #     #     tmp_trans = torch.bmm(diff_x_3d, inv_Q_exp)     # (B,1,D)
    #     #     quad_trans_2d = torch.bmm(tmp_trans, diff_x.unsqueeze(-1))  # (B,1,1)
    #     #     quad_trans = quad_trans_2d.squeeze(-1).squeeze(-1)         # (B,)

    #     #     # partial => shape => (B,)
    #     #     # dimension => D log(2 pi)
    #     #     nll_trans_t_b = 0.5 * (
    #     #         quad_trans
    #     #         + logdet_Q_t
    #     #         + D*np.log(2*np.pi)
    #     #     )
    #     #     ll_trans_t = - nll_trans_t_b.sum()
    #     #     log_lik += ll_trans_t

    #     # # 3) observations => separate loop over t in [0..T-1]
    #     # # For each t, do diff_y = y[:,t] - C x[:,t], shape (B,M).
    #     # # then -0.5 [ diff_y^T R^-1 diff_y + logdet(R) + M log(2 pi ) ] sum over b
    #     log_lik = 0.0
    #     inv_R = torch.linalg.inv(R)
    #     logdet_R = torch.logdet(R)

    #     for t in range(T):
    #         # diff_y => shape (B,M)
    #         #   y[:,t] => (B,M)
    #         #   C => (M,D), x[:,t] => (B,D). We'll do an einsum or matmul => shape (B,M).
    #         #   C x[:,t] => (B,M)
    #         Cx_t = torch.einsum('md,bd->bm', C, x[:,t])
    #         diff_y = y[:,t] - Cx_t  # => (B,M)

    #         # quad => sum_{b} diff_y[b]^T R^-1 diff_y[b]
    #         # We'll do bmm approach => shape => (B,1,1)
    #         diff_y_3d = diff_y.unsqueeze(1)             # (B,1,M)
    #         inv_R_exp = inv_R.unsqueeze(0).expand(B,M,M)# (B,M,M)
    #         tmp_obs = torch.bmm(diff_y_3d, inv_R_exp)   # => (B,1,M)
    #         quad_obs_2d = torch.bmm(tmp_obs, diff_y.unsqueeze(-1))  # (B,1,1)
    #         quad_obs = quad_obs_2d.squeeze(-1).squeeze(-1)         # (B,)

    #         nll_obs_t_b = 0.5 * (
    #             quad_obs
    #             + logdet_R
    #             + M*np.log(2*np.pi)
    #         )
    #         ll_obs_t = - nll_obs_t_b.sum()
    #         log_lik += ll_obs_t

    #     return log_lik


    # def expected_loglikelihood_x(self, A, Q, Ex, Exx, Exx_prev):
    #     """
    #     Computes the negative log-likelihood of the latent variables x_t given A and Q.

    #     Parameters:
    #     - A: Tensor of shape (T, D, D), transition matrix.
    #     - Q: Tensor of shape (T, D, D), process noise covariance matrix.
    #     - Ex: Tensor of shape (T, NT, D, 1), filtered means of x_t.
    #     - Exx: Tensor of shape (T, NT, D, D), filtered covariances of x_t.
    #     - Exx_prev: Tensor of shape (T-1, NT, D, D), cross covariances E[x_t x_{t-1}^T].

    #     Returns:
    #     - neg_log_likelihood: Scalar tensor representing the negative log-likelihood.
    #     """
    #     # Extract dimensions
    #     T, NT, D, _ = Ex.shape

    #     # Ensure Exx_prev has the correct shape
    #     assert Exx_prev.shape == (T-1, NT, D, D), f"Exx_prev should have shape (T-1, NT, D, D), but got {Exx_prev.shape}"

    #     # Expand A and A transpose for batch matrix multiplication
    #     # A: (D, D) -> (1, 1, D, D)
    #     A_expanded = A.unsqueeze(1)  # (T, 1, D, D)
    #     A_T_expanded = A.mT.unsqueeze(1) # (T, 1, D, D)
    #     # A_expanded = A.unsqueeze(0).unsqueeze(0)  # (T, 1, D, D)
    #     # A_T_expanded = A.transpose(-2, -1).unsqueeze(0).unsqueeze(0)  # (T, 1, D, D)

    #     # Compute the residual: E[(x_t - A x_{t-1})(x_t - A x_{t-1})^T | data]
    #     # Formula:
    #     # residual = Exx[1:, :, :, :] - A @ Exx_prev.transpose(-2, -1) - Exx_prev @ A^T + A @ Exx_prev @ A^T
    #     residual = (
    #         Exx[1:, :, :, :] 
    #         - torch.matmul(A_expanded[1:], Exx_prev.mT) 
    #         - torch.matmul(Exx_prev, A_T_expanded[1:])
    #         + torch.matmul(torch.matmul(A_expanded[1:], Exx_prev), A_T_expanded[1:])
    #     )  # Shape: (T-1, NT, D, D)
    #     # residual = (
    #     #     Exx[1:, :, :, :] 
    #     #     - torch.matmul(A_expanded, Exx_prev.transpose(-2, -1)) 
    #     #     - torch.matmul(Exx_prev, A_T_expanded)
    #     #     + torch.matmul(torch.matmul(A_expanded, Exx_prev), A_T_expanded)
    #     # )  # Shape: (T-1, NT, D, D)

    #     # Sum residuals over all time steps and trials to get S
    #     # Shape: (D, D)
    #     S = residual.sum(dim=(1))  # mean over NT

    #     # Qnew = Q[1:] 
    #     Qnew = Q[1:]
    #     # Qnew = Q[1:] + 1e-8 * torch.eye(Q.shape[-1], device=Q.device).unsqueeze(0)

    #     # Compute log determinant of Q
    #     logdet_Q = torch.logdet(Qnew)  # Scalar

    #     # Compute trace(Q^{-1} @ S)
    #     Q_inv = torch.linalg.inv(Qnew)  # (D, D)
    #     QS = torch.matmul(Q_inv, S)  # (T-1, D, D)
    #     # Diagonal => (T-1, D), sum => (T-1,)
    #     trace_term = QS.diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1)
    #     # trace_term = torch.trace(torch.matmul(Q_inv, S))  # Scalar

    #     # Compute negative log-likelihood
    #     LL = -0.5 * (NT * logdet_Q.sum() + trace_term.sum())
    #     # LL = -0.5 * (logdet_Q.mean() + trace_term.mean())

    #     return LL

    # def expected_loglikelihood_xy(
    #     self,
    #     A: torch.Tensor,       # (T, D, D)
    #     Q: torch.Tensor,       # (T, D, D)
    #     C: torch.Tensor,       # (obs_dim, D)
    #     R_diag: torch.Tensor,  # (obs_dim,)     diagonal of R
    #     Ex: torch.Tensor,      # (T, NT, D, 1)   posterior means E[x_t]
    #     Exx: torch.Tensor,     # (T, NT, D, D)   posterior covariances E[x_t x_t^T]
    #     Exx_prev: torch.Tensor,# (T-1, NT, D, D) cross-covs E[x_t x_{t-1}^T]
    #     ys: torch.Tensor       # (NT, T, obs_dim)
    # ) -> torch.Tensor:
    #     """
    #     Parallel computation (no Python for-loops over T or NT) of:
    #     1) The negative log-likelihood of x_t under x_t~N(A_t x_{t-1}, Q_t)
    #     2) The negative log-likelihood of y_t under y_t~N(C x_t, R)

    #     All in a vectorized manner.
    #     """
    #     ys = ys.permute(1, 0, 2)
    #     T, NT, D, _ = Ex.shape
    #     obs_dim = R_diag.shape[0]

    #     # =============== 1) LATENT NEG-LL (x-part) ===============
    #     # We only have transitions for t=1..T-1
    #     # A[t], Q[t] used for step t, shape => (T, D, D).
    #     # So we do residual_x[t] = Exx[t] - A[t] Exx_prev[t-1]^T - ... + A[t] Exx_prev[t-1] A[t]^T
    #     # shape => (T-1, NT, D, D)
    #     # We'll define A_exp => shape => (T,1,D,D) to broadcast in matmul:

    #     # 1.1) Build residual for t=1..T-1 (all time steps in parallel)
    #     A_exp = A.unsqueeze(1)   # (T, 1, D, D)
    #     A_T_exp= A.mT.unsqueeze(1)  # (T, 1, D, D)

    #     # residual_x => (T-1, NT, D, D)
    #     # Use the formula: Exx[t] - A[t]*Exx_prev[t-1]^T - Exx_prev[t-1]*A[t]^T + A[t]*Exx_prev[t-1]*A[t]^T
    #     # We do .mT => short for transpose(-2,-1).
    #     residual_x = (
    #         Exx[1:] 
    #         - torch.matmul(A_exp[1:], Exx_prev.mT)
    #         - torch.matmul(Exx_prev, A_T_exp[1:])
    #         + torch.matmul(torch.matmul(A_exp[1:], Exx_prev), A_T_exp[1:])
    #     )  # => shape (T-1, NT, D, D)

    #     # sum over NT => shape => (T-1, D, D) 
    #     # we can do .sum(dim=1) => sum across trials

    #     S_x = residual_x.sum(dim=1)   # shape => (T-1, D, D) # for sum LL
    #     # S_x = residual_x.mean(dim=1) 

    #     # Q[1:] => shape => (T-1, D, D), for each t=1..T-1
    #     Q_t = Q[1:]        # shape => (T-1, D, D)
    #     # logdet => shape => (T-1,)
    #     logdet_Q = torch.logdet(Q_t)  # batch of size (T-1)
    #     # Q_inv => shape => (T-1, D, D)
    #     Q_inv = torch.linalg.inv(Q_t)
    #     # multiply => shape => (T-1, D, D)
    #     QS_x = torch.matmul(Q_inv, S_x)
    #     # diag => shape => (T-1, D)
    #     trace_term_x = QS_x.diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1)  # => (T-1,)

    #     # negative log-likelihood for x:  0.5 [ sum_{t=1..T-1} NT * logdet(Q_t) + trace(Q_t^-1 S_x[t]) ]

    #     ll_x = -0.5 * (NT * logdet_Q.sum() + trace_term_x.sum()) # for sum LL
    #     # ll_x = -0.5 * (logdet_Q.mean() + trace_term_x.mean())

    #     # 2) Observation part (y-part)
    #     # R is diagonal => R_diag => (obs_dim,)
    #     # logdet(R) = sum_{m=1..obs_dim} log(R_diag[m])
    #     # R^-1 => 1 / R_diag => shape (obs_dim,)
    #     # We'll do everything in parallel => no loops
    #     # ---------------------------
    #     # We define:
    #     # diff_y[t,i,m] = y[t,i,m] - sum_d( C[m,d] * E[x_{t,i},d] )
    #     # Then the second moment => C Exx[t,i] C^T => we only need the diagonal => cSigmaCt_diag[t,i,m]
    #     # => shape => (T,NT,obs_dim).
    #     #
    #     # Finally => sum_{t} sum_{i} sum_{m} [ diff_y^2 + cSigmaCt_diag ] / R_diag[m].
    #     # plus T * NT * sum_{m} log(R_diag[m]).

    #     # 2.1) logdet(R) => sum(log(R_diag)), a scalar. => shape ()
    #     # we do sum(log(R_diag))
    #     logdet_R = torch.log(R_diag).sum()  # float scalar

    #     # We'll define diff => shape => (T,NT,obs_dim).
    #     #   X_t => (T,NT,D,1). We'll define X_t_2 => (T,NT,D).
    #     #   We do an einsum => obs_mean[t,i,m] = sum_{d} C[m,d]* X_t_2[t,i,d].
    #     X_t_2 = Ex.squeeze(-1)       # => (T,NT,D)
    #     # obs_mean => (T,NT,obs_dim)
    #     obs_mean = torch.einsum('tnd,md->tnm', X_t_2, C)  # => (T,NT,obs_dim)
    #     diff_y = ys - obs_mean       # => (T,NT,obs_dim)

    #     # 2.2) cSigmaCt_diag: we only need the diagonal of C Exx[t,i] C^T.
    #     #   cSigmaCt_diag[t,i,m] = sum_{d1,d2} C[m,d1]* Exx[t,i,d1,d2]* C[m,d2]
    #     # => we can do an einsum => shape => (T,NT,m)
    #     # We'll define:
    #     #   't i d1 d2, m d1 d2 -> t i m' ??? Actually we can flatten m, or do:
    #     #   cSigmaCt_diag = torch.einsum('t i d1 d2, m d1, m d2-> t i m' ??? we must be careful to separate the 'm' index
    #     # simpler to do step by step:
    #     # 
    #     # approach: for each m, cSigmaCt_diag[t,i,m] = sum_{d1,d2} C[m,d1]*Exx[t,i,d1,d2]*C[m,d2].
    #     # We'll define => cSigmaCt_diag = (Exx[t,i] * something). We'll do a single e.g. 'tid1d2,md1,md2->tim' ?

    #     # let's do an expanded approach with a single 'einsum':
    #     # cSigmaCt_diag => shape (T,NT,m)
    #     # cSigmaCt_diag = torch.einsum('t i d1 d2, m d1, m d2-> t i m',
    #     #                              Exx, C, C)
    #     # and that yields a sum_{d1,d2} C[m,d1]*Exx[t,i,d1,d2]*C[m,d2], for each (t,i,m).
    #     cSigmaCt_diag = torch.einsum(
    #         'md, t i d d, dm -> t i m',
    #         C, Exx, C.T
    #     )  # => shape (T, NT, m)

    #     # Now we have diff_y => (T,NT,m), cSigmaCt_diag => (T,NT,m).
    #     # The sum of squares => diff_y^2 => (T,NT,m).
    #     # We'll do  ( diff_y^2 + cSigmaCt_diag ) => shape (T,NT,m).
    #     # Then we divide each [t,i,m] by R_diag[m], then sum over i,m => shape => (T,).
    #     # Then sum over t => shape => ().
    #     diff_sq = diff_y * diff_y   # => (T,NT,m)
    #     # sum => ( diff_sq + cSigmaCt_diag ) => shape => (T,NT,m)
    #     obs_res = diff_sq + cSigmaCt_diag

    #     # Now we do elementwise divide by R_diag => shape => (m,)
    #     # We'll broadcast => (T,NT,m).
    #     # R_diag => (m,), expand => (1,1,m). 
    #     # obs_div => obs_res / R_diag => (T,NT,m)
    #     obs_div = obs_res / R_diag.unsqueeze(0).unsqueeze(0)

    #     # sum over (NT,m) => shape => (T,)
    #     # sum => (T,) => then sum over T => shape => ()

    #     sum_obs = obs_div.sum(dim=(1,2)) # for sum LL
    #     # sum_obs = obs_div.mean(dim=(1)).sum(dim=(1,))

    #     # final => 0.5 [ sum_{t} sum_obs[t] + T * NT * sum_{m} log(R_diag[m]) ]
    #     # shape => float
    #     #   sum_{m} log(R_diag[m]) => logdet_R
    #     # Then T *NT => scalar
    #     # => partial nll
    #     ll_y = -0.5 * ( sum_obs.sum() + (T * NT) * logdet_R ) # for sum LL
    #     # ll_y = -0.5 * ( sum_obs.mean() + logdet_R )

    #     # =============== 3) Sum them up =================
    #     ll = ll_x + ll_y
    #     return ll

    # # def expected_loglikelihood_xy(
    # #     self,
    # #     A: torch.Tensor,       # (T, D, D)
    # #     Q: torch.Tensor,       # (T, D, D)
    # #     C: torch.Tensor,       # (obs_dim, D)
    # #     R: torch.Tensor,       # (obs_dim, obs_dim)
    # #     Ex: torch.Tensor,      # (T, NT, D, 1)   posterior means E[x_t]
    # #     Exx: torch.Tensor,     # (T, NT, D, D)   posterior covariances E[x_t x_t^T]
    # #     Exx_prev: torch.Tensor,# (T-1, NT, D, D) cross-covs E[x_t x_{t-1}^T]
    # #     ys: torch.Tensor       # (T, NT, obs_dim)
    # # ) -> torch.Tensor:
    # #     """
    # #     Parallel computation (no Python for-loops over T or NT) of:
    # #     1) The negative log-likelihood of x_t under x_t~N(A_t x_{t-1}, Q_t)
    # #     2) The negative log-likelihood of y_t under y_t~N(C x_t, R)

    # #     All in a vectorized manner.
    # #     """
    # #     device = Ex.device
    # #     dtype  = Ex.dtype
    # #     T, NT, D, _ = Ex.shape
    # #     obs_dim = R.shape[0]

    # #             # =============== 1) LATENT NEG-LL (x-part) ===============
    # #     # We only have transitions for t=1..T-1
    # #     # A[t], Q[t] used for step t, shape => (T, D, D).
    # #     # So we do residual_x[t] = Exx[t] - A[t] Exx_prev[t-1]^T - ... + A[t] Exx_prev[t-1] A[t]^T
    # #     # shape => (T-1, NT, D, D)
    # #     # We'll define A_exp => shape => (T,1,D,D) to broadcast in matmul:

    # #     # 1.1) Build residual for t=1..T-1 (all time steps in parallel)
    # #     A_exp = A.unsqueeze(1)   # (T, 1, D, D)
    # #     A_T_exp= A.mT.unsqueeze(1)  # (T, 1, D, D)

    # #     # residual_x => (T-1, NT, D, D)
    # #     # Use the formula: Exx[t] - A[t]*Exx_prev[t-1]^T - Exx_prev[t-1]*A[t]^T + A[t]*Exx_prev[t-1]*A[t]^T
    # #     # We do .mT => short for transpose(-2,-1).
    # #     residual_x = (
    # #         Exx[1:] 
    # #         - torch.matmul(A_exp[1:], Exx_prev.mT)
    # #         - torch.matmul(Exx_prev, A_T_exp[1:])
    # #         + torch.matmul(torch.matmul(A_exp[1:], Exx_prev), A_T_exp[1:])
    # #     )  # => shape (T-1, NT, D, D)

    # #     # sum over NT => shape => (T-1, D, D) 
    # #     # we can do .sum(dim=1) => sum across trials
    # #     S_x = residual_x.sum(dim=1)   # shape => (T-1, D, D)

    # #     # Q[1:] => shape => (T-1, D, D), for each t=1..T-1
    # #     Q_t = Q[1:]        # shape => (T-1, D, D)
    # #     # logdet => shape => (T-1,)
    # #     logdet_Q = torch.logdet(Q_t)  # batch of size (T-1)
    # #     # Q_inv => shape => (T-1, D, D)
    # #     Q_inv = torch.linalg.inv(Q_t)
    # #     # multiply => shape => (T-1, D, D)
    # #     QS_x = torch.matmul(Q_inv, S_x)
    # #     # diag => shape => (T-1, D)
    # #     trace_term_x = QS_x.diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1)  # => (T-1,)

    # #     # negative log-likelihood for x:  0.5 [ sum_{t=1..T-1} NT * logdet(Q_t) + trace(Q_t^-1 S_x[t]) ]
    # #     ll_x = -0.5 * (NT * logdet_Q.sum() + trace_term_x.sum())

    # #     # =============== 2) OBSERVATION NEG-LL (y-part) ===============
    # #     # For each t=0..T-1, each trial i=0..NT-1:
    # #     #  y_{t,i} ~ N(C x_{t,i}, R).
    # #     # We'll do everything in parallel over t and i.

    # #     # 2.1) Expand C => shape => (1,1, obs_dim,D) so we can broadcast
    # #     # Ex[t] => shape => (T,NT,D,1)
    # #     C_exp = C.view(1,1,obs_dim, D)    # shape => (1,1,obs_dim,D)

    # #     # 2.2) mean_{t,i} = y_{t,i} - C mu_{t,i}, 
    # #     #    with mu_{t,i} => shape => (D,1)
    # #     # We'll do a small bmm approach:
    # #     # step1 => C @ Ex[t], shape => (T,NT, obs_dim,1)
    # #     #    (C_exp => (1,1,obs_dim,D), Ex[t] => (T,NT,D,1) => broadcast => (T,NT, obs_dim,D) x (T,NT,D,1) => bmm
    # #     # We'll reshape a bit to do a direct bmm:

    # #     # let's shape => out => (T,NT,obs_dim,1)
    # #     # We'll do:
    # #     #   X_4d = Ex[t].squeeze(-1) => (T,NT,D)
    # #     #   we might prefer a single einsum approach:
    # #     X_t_2d = Ex.squeeze(-1)  # shape => (T,NT,D)
    # #     # diff => y[t] - C X_t => shape => (T,NT,obs_dim)
    # #     # We'll do an einsum:
    # #     #  obs_mean[t,i] = sum_d( C[k,d]* X_t_2d[t,i,d] ), shape => (T,NT,obs_dim)
    # #     obs_mean_y = torch.einsum('tkd,ad->tka', X_t_2d, C)


    # #     # Actually we want shape => (T,NT,obs_dim). Let's define obs_mean_y = (T,NT, obs_dim)
    # #     # We'll do: 'tkd,cd->tkc' => c is obs_dim. Let's rename properly:
    # #     # let's define: 'tkd,ad->tka' => a=obs_dim
    # #     obs_mean_y = torch.einsum('tkd,ad->tka', X_t_2d, C)   # => (T,NT,obs_dim)

    # #     # Then diff => shape => (T,NT,obs_dim)
    # #     diff_y = ys - obs_mean_y

    # #     # 2.3) second moment => C Exx[t] C^T => shape => (T,NT,obs_dim,obs_dim)
    # #     # We'll do a bmm approach or an einsum approach. Let's do bmm:
    # #     # We'll expand C => shape => (1,1,obs_dim,D). Then Exx[t] => (T,NT,D,D).
    # #     # step1: tmp => shape => (T,NT,obs_dim,D)
    # #     #   tmp[t,i] = C Exp @ Exx[t,i]
    # #     # We'll do a reshaping approach:
    # #     T2,NT2,D2,D3 = Exx.shape  # T2=T,NT2=NT,D2=D,D3=D
    # #     # expand C => (1,1,obs_dim,D) => broadcast => (T,NT,obs_dim,D)
    # #     C_4d = C_exp.expand(T,NT, -1, -1)  # => (T,NT, obs_dim,D)
    # #     # We'll do tmp = bmm( C_4d, Exx[t] ) => but we must reshape Exx => (T*NT, D, D).
    # #     Exx_2d = Exx.reshape(T*NT, D, D)             # (T*NT, D, D)
    # #     C_4d_rsh = C_4d.reshape(T*NT, obs_dim, D)    # (T*NT, obs_dim,D)
    # #     tmp_2d = torch.bmm(C_4d_rsh, Exx_2d)         # => (T*NT, obs_dim, D)
    # #     # then multiply by C^T => shape => (T*NT, obs_dim, obs_dim)
    # #     cSigmaCt_2d = torch.bmm(tmp_2d, C_4d_rsh.transpose(1,2))
    # #     cSigmaCt = cSigmaCt_2d.reshape(T, NT, obs_dim, obs_dim)

    # #     # 2.4) full "obs_res[t,i]" => diff_y[t,i] diff_y[t,i]^T + cSigmaCt[t,i]
    # #     # We'll do diff_y[t,i] => shape => (obs_dim), let's do a bmm approach:
    # #     # diff_y => (T,NT,obs_dim). We'll define diff_ => shape => (T*NT, obs_dim,1).
    # #     diff_y_2d = diff_y.reshape(T*NT, obs_dim)               # => (T*NT, obs_dim)
    # #     diff_y_3d = diff_y_2d.unsqueeze(-1)                     # => (T*NT, obs_dim,1)
    # #     diff_y_T_3d = diff_y_2d.unsqueeze(1)                    # => (T*NT, 1, obs_dim)
    # #     diff_diffT_2d = torch.bmm(diff_y_3d, diff_y_T_3d)        # => (T*NT, obs_dim, obs_dim)
    # #     # sum => cSigmaCt_2d => shape => (T*NT, obs_dim, obs_dim)
    # #     obs_res_2d = diff_diffT_2d + cSigmaCt_2d   # => (T*NT, obs_dim, obs_dim)
    # #     # sum over NT => shape => (T, obs_dim, obs_dim)
    # #     obs_res = obs_res_2d.reshape(T,NT, obs_dim,obs_dim).sum(dim=1)

    # #     # 2.5) each time step => 0.5 [ NT logdet(R) + trace(R^-1 * obs_res[t]) ]
    # #     # We'll define => obs_nll = 0.5 sum_{t=0..T-1} [ NT logdet(R) + trace(R_inv obs_res[t]) ]
    # #     # no python loop => do a single pass:
    # #     # shape => obs_res => (T, obs_dim, obs_dim)
    # #     # R_inv => (obs_dim, obs_dim)
    # #     R_inv = torch.linalg.inv(R)

    # #     # we do: trace_term => shape => (T,)
    # #     #   (R_inv @ obs_res[t]) => (obs_dim, obs_dim)
    # #     #   diag => (obs_dim,) => sum => scalar
    # #     # we'll do in batch:
    # #     # obs_res => reshape => (T, obs_dim, obs_dim)
    # #     # expand R_inv => (1, obs_dim,obs_dim) => do bmm => (T, obs_dim,obs_dim)
    # #     R_inv_exp = R_inv.unsqueeze(0).expand(T, obs_dim, obs_dim)
    # #     # We do rres = bmm(R_inv_exp, obs_res), shape => (T, obs_dim,obs_dim)
    # #     rres = torch.bmm(R_inv_exp, obs_res)
    # #     trace_term_y = rres.diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1)  # (T,)

    # #     # sum => shape => ()
    # #     ll_y = -0.5*( NT * torch.logdet(R)*T + trace_term_y.sum() )

    # #     # =============== 3) Sum them up =================
    # #     ll = ll_x + ll_y
    # #     return ll


    # def black_diagonal_batch(self, As, Qs):
    #     T, C, C = As[0].shape
    #     _, Cw, Cw = As[-1].shape
    #     # Step 1: Create a batched zero tensor for off-diagonal blocks
    #     num_dims = self.xdima * C + self.xdimw * Cw * self.num_regions
    #     As_zeros_off_diag = torch.zeros(T, num_dims, num_dims).to(As[0].device).to(As[0].dtype)  # Shape: (T, nC, nC)
    #     Qs_zeros_off_diag = torch.zeros(T, num_dims, num_dims).to(As[0].device).to(As[0].dtype)  # Shape: (T, nC, nC)

    #     offset = 0
    #     for i, (A, Q) in enumerate(zip(As, Qs)):
    #         c = A.shape[1]
    #         As_zeros_off_diag[:, offset:offset + c, offset:offset + c] = A
    #         Qs_zeros_off_diag[:, offset:offset + c, offset:offset + c] = Q
    #         offset += c

    #     # The result is a tensor with block-diagonal structure
    #     return As_zeros_off_diag, Qs_zeros_off_diag
    

    # def forward(self, y, epoch=-1):
    #     # y: Shape (trials, T, neurons)
    #     batch_size, T = y.shape[0], y.shape[1]

    #     Ats = []
    #     Qts = []
    #     for i in range(self.xdima):
    #         across_dy = self.across_dynamics[i]

    #         # log_sigma = self.log_sigma_mappings[i](y).mean(0)
    #         # delays = self.delay_mappings[i](y).mean(0)
    #         # params = delays

    #         As, Qs = across_dy.forward() # Shape: (T, C, C)

    #         Ats.append(As)
    #         Qts.append(Qs)
    #         # nLLs.append(nLL)
        
    #     for i in range(self.xdimw * self.num_regions):
    #         within_dy = self.within_dynamics[i]
    #         As, Qs = within_dy.forward()
    #         Ats.append(As[None, ...].expand(T, -1, -1))
    #         Qts.append(Qs[None, ...].expand(T, -1, -1))

    #     # Ats = self.black_diagonal_batch(Ats)
    #     # Qts = self.black_diagonal_batch(Qts)
    #     Ats, Qts = self.black_diagonal_batch(Ats, Qts)
    #     # Ats = torch.block_diag(*Ats)
    #     # Qts = torch.block_diag(*Qts)
    #     # if epoch == 0:
    #     #     Cs = torch.Tensor(self.true_C).to(self.device).to(self.dtype)
    #     # else:
    #     Cs = torch.block_diag(*self.Cs)
    #     # Cs = torch.Tensor(self.true_C).to(self.device).to(self.dtype)
    #     # 
    #     Rs = torch.diag(self.diag_Rs)

    #     y = y - self.ds

    #     ys_reshape = y.permute(1, 0, 2) # Shape (T, trials, neurons)

    #     new_Cs = Cs @ self.H
    #     kf = KalmanFilter(Ats, new_Cs, Qts, Rs)

    #     # init_mean = torch.zeros((batch_size, self.xdim, 1), dtype=self.dtype).to(self.device)
    #     # init_cov = torch.eye(self.xdim, dtype=self.dtype)[None].expand(batch_size, self.xdim, self.xdim).to(self.device)
    #     init_mean = torch.zeros((self.xdim, 1), dtype=self.dtype).to(self.device)
    #     init_cov = torch.eye(self.xdim, dtype=self.dtype).to(self.device)
    #     state = GaussianState(
    #                 init_mean,  # Shape (num_trials, xdim, 1)
    #                 init_cov,  # Shape (num_trials, xdim, xdim)
    #             )
    #     # states = kf.filter(state, ys_reshape[:,:,:,None], update_first=True, return_all=True)
    #     # states = GaussianState(states.mean, states.covariance[:, None, :, :].expand(-1, batch_size, -1, -1))

    #     filter_mean, filter_covariance = filter(y, Ats, new_Cs, Qts, Rs, init_mean, init_cov)
        
    #     states = GaussianState(filter_mean, filter_covariance[:, None, :, :].expand(-1, batch_size, -1, -1))

    #     smoothed, sigma_pair_smooth = kf.rts_smooth(states)

    #     smoothed_state_means = smoothed.mean

    #     smoothed_state_covariances = smoothed.covariance

    #     # filter_mean = normalize(smoothed_state_means, p=np.inf, dim=0)
    #     # filter_covariance = normalize(smoothed_state_covariances, p=np.inf, dim=0)
        
    #     # initial_state_mean = torch.zeros((Ats.shape[1],)).to(self.device)
    #     # initial_state_covariance = torch.zeros_like(Qts).to(self.device)

    #     # predicted_state_means, predicted_state_covariances, filtered_state_means, filtered_state_covariances = \
    #     #     kalman_filter(Ats, Cs @ self.Hs, Qts, Rs, initial_state_mean, initial_state_covariance, ys_reshape)
        
    #     # smoothed_state_means, smoothed_state_covariances, sigma_pair_smooth = \
    #     #     kalman_smooth(Ats, filtered_state_means, filtered_state_covariances, predicted_state_means, predicted_state_covariances)
        
    #     # smoothed_state_means = smoothed_state_means[:,:,:,None]

    #     # LL = self.expected_loglikelihood_x(Ats, Qts, smoothed_state_means, smoothed_state_covariances, sigma_pair_smooth)
    #     LL = self.expected_loglikelihood_xy(Ats, Qts, new_Cs, self.diag_Rs, smoothed_state_means, smoothed_state_covariances, sigma_pair_smooth, y)

    #     nLL = -1.0 * LL

    #     # data = loadmat("./data/sim_data.mat")
    #     # C = torch.tensor(data["C"]).to(self.device).to(self.dtype)

    #     # ys = (xs[:, :, None, :] @ C[None, None,:,:] ).squeeze()

    #     return nLL, smoothed_state_means.transpose(1, 0)

    # def update_C_and_R(self, ys, xs, C=None):

    #     xs = normalize(xs, p=np.inf, dim=1) # Shape: (B, T, C, 1)

    #     Cs = []
    #     diag_Rs = []
    #     ds = []

    #     xs_shift = []

    #     for i in range(self.num_regions):
    #         xs_i = xs[:,:, self.x_shift_indices[i], 0]
    #         y_i_true = ys[:, :, self.cum_y_dims[i]:self.cum_y_dims[i+1]]
    #         d   = y_i_true.shape[-1]   # output dimension
    #         S0  = 20.0                # your assumed noise variance per dimension
    #         nu0 = d + 2                # weak but proper IW prior

    #         Psi0 = S0 * (nu0 + d + 1)  # so that mode(Σ) = S0·I
    #         C_i, d_i, diag_R_i = self.fit_linear_regression_new(
    #                     xs_i,
    #                     y_i_true,
    #                     prior_ExxT=1e-8 * torch.eye(xs_i.shape[-1] + 1).to(xs_i.device), 
    #                     prior_ExyT=torch.zeros((xs_i.shape[-1] + 1, y_i_true.shape[-1])).to(xs_i.device),
    #                     # prior_ExxT=1e-4 * torch.eye(xs_i.shape[-1] ).to(xs_i.device), 
    #                     # prior_ExyT=torch.zeros((xs_i.shape[-1], y_i_true.shape[-1])).to(xs_i.device),
    #                     fit_intercept=True,
    #                     # nu0=nu0,
    #                     # Psi0=Psi0   
    #                 )
    #         # C_i, d_i, diag_R_i = self.fit_linear_regression(
    #         #             xs_i,
    #         #             y_i_true,
    #         #             prior_ExxT=1e-8 * torch.eye(xs_i.shape[-1] + 1).to(xs_i.device), 
    #         #             prior_ExyT=torch.zeros((xs_i.shape[-1] + 1, y_i_true.shape[-1])).to(xs_i.device),
    #         #             # prior_ExxT=1e-4 * torch.eye(xs_i.shape[-1] ).to(xs_i.device), 
    #         #             # prior_ExyT=torch.zeros((xs_i.shape[-1], y_i_true.shape[-1])).to(xs_i.device),
    #         #             fit_intercept=True
    #         #         )
    #         # C_i, d_i, diag_R_i = self.fit_linear_regression2(xs_i, y_i_true)
    #         xs_shift.append(xs_i)
    #         Cs.append(C_i)
    #         diag_Rs.append(diag_R_i)
    #         ds.append(d_i)
        
        
    #     C = torch.block_diag(*Cs)
    #     diag_R = torch.cat(diag_Rs)
    #     d = torch.cat(ds)

    #     # C = torch.block_diag(*self.Cs)

    #     # noise = torch.randn_like(ys) * diag_R
    #     xs_shift = torch.cat(xs_shift, dim=2)

    #     # ys = (C @ xs_shift[:, :, :, None]).squeeze(-1) + noise


    #     self.Cs = Cs
    #     self.diag_Rs = diag_R
    #     self.ds = d
        
    #     ys = (C @ xs_shift[:, :, :, None]).squeeze(-1) + self.ds

    #     # self.Cs = torch.Tensor(C.T).to(self.device)
    #     # self.diag_Rs = torch.Tensor([0.1]*C.shape[0]).to(self.device).to(self.dtype)

    #     return ys, xs_shift

    # def update_ys(self, xs):
    #     xs_norm = normalize(xs, p=np.inf, dim=1)
    #     # xs_norm = xs

    #     xs_shift_norm = []
    #     xs_shift = []
    #     for i in range(self.num_regions):
    #         xs_i = xs_norm[:,:, self.x_shift_indices[i], 0]
    #         xs_shift_norm.append(xs_i)
    #         xs_shift.append(xs[:,:, self.x_shift_indices[i], 0])
        
    #     C = torch.block_diag(*self.Cs)
    #     xs_shift_norm = torch.cat(xs_shift_norm, dim=2)
    #     xs_shift = torch.cat(xs_shift, dim=2)
    #     ys = (C @ xs_shift_norm[:, :, :, None]).squeeze(-1) + self.ds

    #     return ys, xs_shift
