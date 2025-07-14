
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
