import dataclasses
from typing import Optional, overload
import torch
import torch.linalg
from functools import partial
import torch
from torch import vmap, Tensor
from torch.utils._pytree import tree_flatten, tree_unflatten
from typing import Callable, Tuple

def safe_map(f, *args):
    args = list(map(list, args))
    n = len(args[0])
    for arg in args[1:]:
        assert len(arg) == n, f'length mismatch: {list(map(len, args))}'
    return list(map(f, *args))


def combine(tree, operator, a_flat, b_flat):
    # Lower `fn` to operate on flattened sequences of elems.
    a = tree_unflatten(a_flat, tree)
    b = tree_unflatten(b_flat, tree)
    c = operator(a, b)
    c_flat, _ = tree_flatten(c)
    return c_flat


def _interleave(a, b, axis: int):
    # https://stackoverflow.com/questions/60869537/how-can-i-interleave-5-pytorch-tensors
    b_trunc = (a.shape[axis] == b.shape[axis] + 1)
    if b_trunc:
        pad = [0, 0] * b.ndim
        pad[(b.ndim-axis-1)*2+1] = 1  # +1=always end of dim, pad-order is reversed so start is at end
        b = torch.nn.functional.pad(b, pad)

    stacked = torch.stack([a, b], dim=axis+1)
    interleaved = torch.flatten(stacked, start_dim=axis, end_dim=axis+1)
    if b_trunc:
        # TODO: find torch alternative for slice_along axis for torch.jit.script to work
        interleaved = torch.ops.aten.slice(interleaved, axis, 0, b.shape[axis]+a.shape[axis]-1)
    return interleaved

def _scan(tree, operator, elems, axis: int):
    """Perform scan on `elems`."""
    num_elems = elems[0].shape[axis]

    if num_elems < 2:
        return elems

    # Combine adjacent pairs of elements.
    reduced_elems = combine(tree, operator,
                            [torch.ops.aten.slice(elem, axis, 0, -1, 2) for elem in elems],
                            [torch.ops.aten.slice(elem, axis, 1, None, 2) for elem in elems])

    # Recursively compute scan for partially reduced tensors.
    odd_elems = _scan(tree, operator, reduced_elems, axis)

    if num_elems % 2 == 0:
        even_elems = combine(tree, operator,
                             [torch.ops.aten.slice(e, axis, 0, -1) for e in odd_elems],
                             [torch.ops.aten.slice(e, axis, 2, None, 2) for e in elems])
    else:
        even_elems = combine(tree, operator,
                             odd_elems,
                             [torch.ops.aten.slice(e, axis, 2, None, 2) for e in elems])

    # The first element of a scan is the same as the first element of the original `elems`.
    even_elems = [
        torch.cat([torch.ops.aten.slice(elem, axis, 0, 1), result], dim=axis)
        if result.shape.numel() > 0 and elem.shape[axis] > 0 else
        result if result.shape.numel() > 0 else
        torch.ops.aten.slice(elem, axis, 0, 1)
        for (elem, result) in zip(elems, even_elems)]

    return list(safe_map(partial(_interleave, axis=axis), even_elems, odd_elems))

# Pytorch impl. of jax.lax.associative_scan
# The offical Pytorch implementation is on the way: https://github.com/pytorch/pytorch/pull/139939
def associative_scan(operator: Callable, elems, dim: int = 0, reverse: bool = False):
    elems_flat, tree = tree_flatten(elems)

    if reverse:
        elems_flat = [torch.flip(elem, [dim]) for elem in elems_flat]

    assert dim >= 0 or dim < elems_flat[0].ndim, "Axis should be within bounds of input"
    num_elems = int(elems_flat[0].shape[dim])
    if not all(int(elem.shape[dim]) == num_elems for elem in elems_flat[1:]):
        raise ValueError('Array inputs to associative_scan must have the same '
                         'first dimension. (saw: {})'
                         .format([elem.shape for elem in elems_flat]))

    scans = _scan(tree, operator, elems_flat, dim)

    if reverse:
        scans = [torch.flip(scanned, [dim]) for scanned in scans]

    return tree_unflatten(scans, tree)

def first_filtering_element(
    A: Tensor,
    C: Tensor,
    Q: Tensor,
    R: Tensor,
    m0: Tensor,
    P0: Tensor,
    y: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Compute initial (t=0) filter element for scan.
    """
    D, N = A.size(0), y.size(-1)
    I_D = torch.eye(D, dtype=A.dtype, device=A.device)
    S = C @ Q @ C.T + R

    m1 = A @ m0 # state.mean
    P1 = A @ P0 @ A.T + Q # state.covariance
    S1 = C @ P1 @ C.T + R
 
    # S1 = S1 + 1e-6 * torch.eye(N, dtype=S1.dtype, device=S1.device)

    K1 = torch.linalg.solve(S1, (C @ P1)).mT
    # A1 = torch.zeros_like(A)
    A1 = (I_D - K1 @ C) @ A
    b = m1 + K1 @ (y[..., None] - C @ m1)

    factor = I_D - K1 @ C
    C1 = factor @ P1 @ factor.mT + K1 @ R @ K1.mT


    y_sol = torch.linalg.solve(S.unsqueeze(0).expand(y.shape[0], -1, -1), y)

    HF_sol = torch.linalg.solve(S, C @ A)
    eta = A.mT @ C.T @ y_sol[..., None]
    J = A.mT @ C.T @ HF_sol

    return A1, b, C1, J, eta


def generic_filtering_element(
    A: Tensor,
    C: Tensor,
    Q: Tensor,
    R: Tensor,
    y: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Generic element for all subsequent observations
    """

    D, N = A.size(0), y.size(-1)
    A = A.squeeze(0)
    Q = Q.squeeze(0)

    S = C @ Q @ C.T + R
    # S = S + 1e-6 * torch.eye(N, dtype=S.dtype, device=S.device)

    K = torch.linalg.solve(S, C @ Q).mT
    A1 = A - K @ C @ A
    b = K @ y[..., None]

    factor = torch.eye(Q.shape[0], dtype=Q.dtype, device=Q.device) - K @ C
    C1 = factor @ Q @ factor.mT + K @ R @ K.mT


    y_sol = torch.linalg.solve(S.unsqueeze(0).expand(y.shape[0], -1, -1), y)

    HF_sol = torch.linalg.solve(S, C @ A)
    eta = A.T @ C.mT @ y_sol[..., None]

    J = A.T @ C.mT @ HF_sol

    return A1, b, C1, J, eta



@torch.jit.script
def filtering_operator(
    elem1: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    elem2: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
    """
    Associative filtering operator, vectorized with vmap.
    """
    A1, b1, C1, J1, eta1 = elem1
    A2, b2, C2, J2, eta2 = elem2
    dim = A1.shape[-1]
    I = torch.eye(dim, dtype=A1.dtype, device=A1.device)

    I_C1J2 = I + C1 @ J2

    temp = torch.linalg.solve(I_C1J2.mT, A2.mT).mT


    A = temp @ A1
    b = temp[:, None, :, :] @ (b1 + C1[:,None,:,:] @ eta2) + b2
    C = temp @ C1 @ A2.mT + C2

    I_J2C1 = I + J2 @ C1
    temp2 = torch.linalg.solve(I_J2C1.mT, A1).mT

    eta = temp2[:,None,:,:] @ (eta2 - J2[:,None,:,:] @ b1) + eta1
    J = temp2 @ J2 @ A1 + J1

    return (A, b, C, J, eta)

# def first_filtering_element(A, C, Q, R, m0, P0, y):
#     """
#     Equivalent of the JAX version, but using PyTorch linalg.
#     """
#     S = C @ Q @ C.T + R
#     # We do a Cholesky factor (lower), then we might invert or solve.
#     # L = torch.linalg.cholesky(S)
#     # We'll replicate jsc.linalg.cho_factor(...) => in PyTorch, use L as is.

#     m1 = A @ m0 # state.mean
#     P1 = A @ P0 @ A.T + Q # state.covariance
#     S1 = C @ P1 @ C.T + R

#     # K1 = S1^{-1} (H P1)^T => K1 = torch.linalg.solve(S1, H@P1).T
#     K1 = torch.linalg.solve(S1, (C @ P1)).mT
#     A1 = torch.zeros_like(A)
#     b = m1 + K1 @ (y[..., None] - C @ m1)
#     # b = m1 + K1 @ (y - C @ m1)
    
#     # unstable version:
#     # C1 = P1 - K1 @ S1 @ K1.mT
#     # state version:
#     factor = torch.eye(P1.shape[0], dtype=P1.dtype, device=P1.device) - K1 @ C
#     C1 = factor @ P1 @ factor.mT + K1 @ R @ K1.mT

#     # For the pair (J, eta)
#     # J = F.T H.T S^{-1} H F,  eta = F.T H.T S^{-1} y
#     # S = L @ L.T => S^{-1} y => triangular_solve, etc.
#     # But simpler: solve y => we do triangular_solve on L then L.T, or just torch.linalg.solve(S, y).
#     # We'll do it directly:
#     y_sol = torch.linalg.solve(S.unsqueeze(0).expand(y.shape[0], -1, -1), y)
#     # y_sol = torch.linalg.solve(S, y)

#     HF_sol = torch.linalg.solve(S, C @ A)
#     eta = A.mT @ C.T @ y_sol[..., None]
#     # eta = A.mT @ C.T @ y_sol
#     J = A.mT @ C.T @ HF_sol

#     return A1, b, C1, J, eta


# def generic_filtering_element(A, C, Q, R, y):
#     """
#     Generic element for all subsequent observations
#     """

#     A = A.squeeze(0)
#     Q = Q.squeeze(0)

#     S = C @ Q @ C.T + R
#     # S = L L.T
#     # K = Q F^T H^T S^{-1}, etc. but let's replicate the formula from JAX code

#     # K = S^{-1} (H Q)^T => K = torch.linalg.solve(S, (H @ Q).T).T
#     K = torch.linalg.solve(S, C @ Q).mT
#     A1 = A - K @ C @ A
#     b = K @ y[..., None]
#     # b = K @ y
#     # unstable version:
#     # C1 = Q - K @ C @ Q
#     # stable version:
#     factor = torch.eye(Q.shape[0], dtype=Q.dtype, device=Q.device) - K @ C
#     C1 = factor @ Q @ factor.mT + K @ R @ K.mT

#     # For the pair (J, eta)

#     y_sol = torch.linalg.solve(S.unsqueeze(0).expand(y.shape[0], -1, -1), y)
#     # y_sol = torch.linalg.solve(S, y)
#     HF_sol = torch.linalg.solve(S, C @ A)
#     eta = A.T @ C.mT @ y_sol[..., None]
#     # eta = A.T @ C.mT @ y_sol
#     J = A.T @ C.mT @ HF_sol

#     return A1, b, C1, J, eta


# def make_associative_filtering_elements(As, C, Qs, R, m0, P0, observations):
#     """
#     Build the parallel filtering elements in a batched manner using vmap.
#     """
#     # Handle the first observation separately
#     first_elems = first_filtering_element(As[0], C, Qs[0], R, m0, P0, observations[:, 0])  # returns (A, b, C, J, eta)

#     # If there's only one observation, we won't have "generic" observations to vmap over.
#     # if observations.shape[1] > 1:
#     # We map `generic_filtering_element` over observations[1:]
#     A_batched, b_batched, C_batched, J_batched, eta_batched = vmap(
#         lambda A, Q, y: generic_filtering_element(A, C, Q, R, y), in_dims=1, out_dims=0
#     # )(As.expand(1, -1, -1, -1), Qs.expand(1, -1, -1, -1), observations)
#     )(As[1:].expand(1, -1, -1, -1), Qs[1:].expand(1, -1, -1, -1), observations[:, 1:])

#     # A_batched, b_batched, C_batched, J_batched, eta_batched = vmap(
#     #     lambda y: generic_filtering_element(A, C, Q, R, y), in_dims=0, out_dims=0
#     # )(observations[0, 1:])
#     # else:
#     #     # If T=1, create empty tensors that match the shapes of A, b, C, J, eta
#     #     A_batched  = torch.empty((0, *first_elems[0].shape), dtype=first_elems[0].dtype)
#     #     b_batched  = torch.empty((0, *first_elems[1].shape), dtype=first_elems[1].dtype)
#     #     C_batched  = torch.empty((0, *first_elems[2].shape), dtype=first_elems[2].dtype)
#     #     J_batched  = torch.empty((0, *first_elems[3].shape), dtype=first_elems[3].dtype)
#     #     eta_batched= torch.empty((0, *first_elems[4].shape), dtype=first_elems[4].dtype)

#     # Now we prepend the first element (unsqueezed) to the “generic” elements
#     # in order to have a batch dimension of size T.
#     A_total   = torch.cat([first_elems[0].unsqueeze(0), A_batched],   dim=0)
#     b_total   = torch.cat([first_elems[1].unsqueeze(0), b_batched],   dim=0)
#     C_total   = torch.cat([first_elems[2].unsqueeze(0), C_batched],   dim=0)
#     J_total   = torch.cat([first_elems[3].unsqueeze(0), J_batched],   dim=0)
#     eta_total = torch.cat([first_elems[4].unsqueeze(0), eta_batched], dim=0)

#     # A_total = A_batched
#     # b_total = b_batched
#     # C_total = C_batched
#     # J_total = J_batched
#     # eta_total = eta_batched

#     return (A_total, b_total, C_total, J_total, eta_total)

# @torch.jit.script
# def filtering_operator(
#     elem1: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
#     elem2: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
#     ):
#     """
#     Associative filtering operator, vectorized with vmap.
#     """
#     A1, b1, C1, J1, eta1 = elem1
#     A2, b2, C2, J2, eta2 = elem2
#     dim = A1.shape[-1]
#     I = torch.eye(dim, dtype=A1.dtype, device=A1.device)

#     # I + C1 @ J2
#     I_C1J2 = I + C1 @ J2
#     # Solve for [I_C1J2.T]^-1 * A2.T => triangular_solve in general, but let's do a generic solve
#     # NOTE: this is just a generic solve, not necessarily triangular.
#     # We'll do "temp = torch.linalg.solve(I_C1J2.T, A2.T).T"
#     # print(I_C1J2.shape, A2.shape)
#     temp = torch.linalg.solve(I_C1J2.mT, A2.mT).mT
#     # temp = _spd_solve(I_C1J2.mT, A2.mT).mT
#     # print("pass")

#     A = temp @ A1
#     b = temp[:, None, :, :] @ (b1 + C1[:,None,:,:] @ eta2) + b2
#     C = temp @ C1 @ A2.mT + C2

#     # I + J2 @ C1
#     I_J2C1 = I + J2 @ C1
#     temp2 = torch.linalg.solve(I_J2C1.mT, A1).mT
#     # temp2 = _spd_solve(I_J2C1.mT, A1).mT

#     eta = temp2[:,None,:,:] @ (eta2 - J2[:,None,:,:] @ b1) + eta1
#     J = temp2 @ J2 @ A1 + J1

#     return (A, b, C, J, eta)


# def filter(observations, A, C, Q, R, m0, P0):
#     """
#     Parallel Kalman Filter
#     """
#     # Build the initial elements
#     initial_elements = make_associative_filtering_elements(A, C, Q, R, m0, P0, observations)
#     # Apply associative_scan in forward direction
#     # final_elements = associative_scan(filtering_operator, initial_elements)

#     _, b, c, J_hist, eta_hist = associative_scan(filtering_operator, initial_elements, dim = 0, reverse=False)

    
#     # return final_elements[1], final_elements[2]  # (means, covariances)
#     # print(J_hist.shape, eta_hist.shape)
#     # P_hist = torch.linalg.inv(J_hist)              # (T, d, d)

#     # # --- broadcast P over the batch axis ----------------------------------
#     # B = eta_hist.size(1)                           # 100
#     # P_hist_b = P_hist.unsqueeze(1).expand(-1, B, -1, -1)   # (T, B, d, d)

#     # # filtered means  (T, B, d, 1)
#     # m_hist = torch.matmul(P_hist_b, eta_hist)

#     # # print(b.shape,c.shape, m_hist.shape, P_hist.shape)
#     # # return m_hist, P_hist                          # P_hist is (T, d, d)
#     return b, c


# The following code for sequential Kalmen Filter is adapted from https://github.com/raphaelreme/torch-kf

@dataclasses.dataclass
class GaussianState:
    """Gaussian state in Kalman Filter

    We emphasize that the mean is at least 2d (dim_x, 1).

    It also supports some of torch functionnality to clone, convert or slice both mean and covariance at once.

    Attributes:
        mean (torch.Tensor): Mean of the distribution
            Shape: (*, dim, 1)
        covariance (torch.Tensor): Covariance of the distribution
            Shape: (*, dim, dim)
        precision (Optional[torch.Tensor]): Optional inverse covariance matrix
            This may be useful for some computations (E.G mahalanobis distance, likelihood) after a predict step.
            Shape: (*, dim, dim)
    """

    mean: torch.Tensor
    covariance: torch.Tensor
    precision: Optional[torch.Tensor] = None

    def clone(self) -> "GaussianState":
        """Clone the Gaussian State using `torch.Tensor.clone`

        Returns:
            GaussianState: A copy of the Gaussian state
        """
        return GaussianState(
            self.mean.clone(), self.covariance.clone(), self.precision.clone() if self.precision is not None else None
        )

    def __getitem__(self, idx) -> "GaussianState":
        return GaussianState(
            self.mean[idx], self.covariance[idx], self.precision[idx] if self.precision is not None else None
        )

    def __setitem__(self, idx, value) -> None:
        if isinstance(value, GaussianState):
            self.mean[idx] = value.mean
            self.covariance[idx] = value.covariance
            if self.precision is not None and value.precision is not None:
                self.precision[idx] = value.precision

            return

        raise NotImplementedError()

    @overload
    def to(self, dtype: torch.dtype) -> "GaussianState": ...

    @overload
    def to(self, device: torch.device) -> "GaussianState": ...

    def to(self, fmt):
        """Convert a GaussianState to a specific device or dtype

        Args:
            fmt (torch.dtype | torch.device): Memory format to send the state to.

        Returns:
            GaussianState: The GaussianState with the right format
        """
        return GaussianState(
            self.mean.to(fmt),
            self.covariance.to(fmt),
            self.precision.to(fmt) if self.precision is not None else None,
        )

    def mahalanobis_squared(self, measure: torch.Tensor) -> torch.Tensor:
        """Computes the squared mahalanobis distance of given measure

        It supports batch computation: You can provide multiple measurements and have multiple states
        You just need to ensure that shapes are broadcastable.

        Args:
            measure (torch.Tensor): Points to consider
                Shape: (*, dim, 1)

        Returns:
            torch.Tensor: Squared mahalanobis distance for each measure/state
                Shape: (*)
        """
        diff = self.mean - measure  # You are responsible for broadcast
        if self.precision is None:
            # The inverse is transposed (back) to be contiguous: as it is symmetric
            # This is equivalent and faster to hold on the contiguous verison
            # But this may slightly increase floating errors.
            self.precision = self.covariance.inverse().mT

        return (diff.mT @ self.precision @ diff)[..., 0, 0]  # Delete trailing dimensions

    def mahalanobis(self, measure: torch.Tensor) -> torch.Tensor:
        """Computes the mahalanobis distance of given measure

        Computations of the sqrt can be slow. If you want to compare with a given threshold,
        you should rather compare the squared mahalanobis with the squared threshold.

        It supports batch computation: You can provide multiple measurements and have multiple states
        You just need to ensure that shapes are broadcastable.

        Args:
            measure (torch.Tensor): Points to consider
                Shape: (*, dim, 1)

        Returns:
            torch.Tensor: Mahalanobis distance for each measure/state
                Shape: (*)
        """
        return self.mahalanobis_squared(measure).sqrt()

    def log_likelihood(self, measure: torch.Tensor) -> torch.Tensor:
        """Computes the log-likelihood of given measure

        It supports batch computation: You can provide multiple measurements and have multiple states
        You just need to ensure that shapes are broadcastable.

        Args:
            measure (torch.Tensor): Points to consider
                Shape: (*, dim, 1)

        Returns:
            torch.Tensor: Log-likelihood for each measure/state
                Shape: (*, 1)
        """
        maha_2 = self.mahalanobis_squared(measure)
        log_det = torch.log(torch.det(self.covariance))

        return -0.5 * (self.covariance.shape[-1] * torch.log(2 * torch.tensor(torch.pi)) + log_det + maha_2)

    def likelihood(self, measure: torch.Tensor) -> torch.Tensor:
        """Computes the likelihood of given measure

        It supports batch computation: You can provide multiple measurements and have multiple states
        You just need to ensure that shapes are broadcastable.

        Args:
            measure (torch.Tensor): Points to consider
                Shape: (*, dim, 1)

        Returns:
            torch.Tensor: Likelihood for each measure/state
                Shape: (*, 1)
        """
        return self.log_likelihood(measure).exp()


class KalmanFilter:
    """Batch and fast Kalman filter implementation in PyTorch

    Kalman filtering optimally estimates the state x_k ~ N(mu_k, P_k) of a
    linear hidden markov model under Gaussian noise assumption. The model is:
    x_k = F x_{k-1} + N(0, Q)
    z_k = H x_k + N(0, R)

    where x_k is the unknown state of the system, F the state transition (or process) matrix,
    Q the process covariance, z_k the observed variables, H the measurement matrix and
    R the measurement covariance.

    .. note:

        In order to allow full flexibility on batch computation, the user has to be precise on the shape of its tensors
        1d vector should always be 2 dimensional and vertical. Check the documentation of each method.


    This is based on the numpy implementation of kalman filter: filterpy (https://filterpy.readthedocs.io/en/latest/)

    Attributes:
        process_matrix (torch.Tensor): State transition matrix (F)
            Shape: (*, dim_x, dim_x)
        measurement_matrix (torch.Tensor): Projection matrix (H)
            Shape: (*, dim_z, dim_x)
        process_noise (torch.Tensor): Uncertainty on the process (Q)
            Shape: (*, dim_x, dim_x)
        measurement_noise (torch.Tensor): Uncertainty on the measure (R)
            Shape: (*, dim_z, dim_z)

    """

    def __init__(
        self,
        process_matrix: torch.Tensor,
        measurement_matrix: torch.Tensor,
        process_noise: torch.Tensor,
        measurement_noise: torch.Tensor,
    ) -> None:
        # We do not check that any device/dtype/shape are shared (but they should be)
        self.process_matrix = process_matrix
        self.measurement_matrix = measurement_matrix
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self._alpha_sq = 1.0  # Memory fadding KF (As in filterpy)

    @property
    def state_dim(self) -> int:
        """Dimension of the state variable"""
        return self.process_matrix.shape[-1]

    @property
    def measure_dim(self) -> int:
        """Dimension of the measured variable"""
        return self.measurement_matrix.shape[0]

    @property
    def device(self) -> torch.device:
        """Device of the Kalman filter"""
        return self.process_matrix.device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the Kalman filter"""
        return self.process_matrix.dtype

    @overload
    def to(self, dtype: torch.dtype) -> "KalmanFilter": ...

    @overload
    def to(self, device: torch.device) -> "KalmanFilter": ...

    def to(self, fmt):
        """Convert a Kalman filter to a specific device or dtype

        Args:
            fmt (torch.dtype | torch.device): Memory format to send the filter to.

        Returns:
            KalmanFilter: The filter with the right format
        """
        return KalmanFilter(
            self.process_matrix.to(fmt),
            self.measurement_matrix.to(fmt),
            self.process_noise.to(fmt),
            self.measurement_noise.to(fmt),
        )

    def predict(
        self,
        state: GaussianState,
        t,
        *,
        process_matrix: Optional[torch.Tensor] = None,
        process_noise: Optional[torch.Tensor] = None,
    ) -> GaussianState:
        """Prediction from the given state

        Use the process model x_{k+1} = F x_k + N(0, Q) to compute the prior on the future state.
        Support batch computation: you can provide multiple models (F, Q) or/and multiple states.
        You just need to ensure that shapes are broadcastable.

        Example:
            # Initialize a random batch of gaussian state (5d)
            state = GaussianState(
                torch.randn(50, 5, 1),  # The last dimension is required.
                torch.randn(50, 5, 5),
            )

            # Use a single process model
            process_matrix = torch.randn(5, 5)  # Compatible with (50, 5, 5)
            process_noise = torch.randn(5, 5)

            predicted = kf.predict(state, process_matrix, process_noise)
            predicted.mean  # Shape: (50, 5, 1)  # Predictions for each state
            predicted.covariance  # Shape: (50, 5, 5)

            # Use several models
            process_matrix = torch.randn(10, 1, 5, 5)  # Compatible with (50, 5, 5)
            process_noise = torch.randn(1, 1, 5, 5)  # Let's use the same noise for each process matrix

            predicted = kf.predict(state, process_matrix, process_noise)
            predicted.mean  # Shape: (10, 50, 5, 1)  # Predictions for each model and state
            predicted.covariance  # Shape: (10, 50, 5, 5)

        Args:
            state (GaussianState): Current state estimation. Should have dim_x dimension.
            process_matrix (Optional[torch.Tensor]): Overwrite the default transition matrix
                Shape: (*, dim_x, dim_x)
            process_noise (Optional[torch.Tensor]): Overwrite the default process noise)
                Shape: (*, dim_x, dim_x)

        Returns:
            GaussianState: Prior on the next state. Will have dim_x dimension.

        """
        if process_matrix is None:
            process_matrix = self.process_matrix[t]
        if process_noise is None:
            process_noise = self.process_noise[t]

        mean = process_matrix @ state.mean
        covariance = self._alpha_sq * process_matrix @ state.covariance @ process_matrix.mT + process_noise

        return GaussianState(mean, covariance)

    def project(
        self,
        state: GaussianState,
        *,
        measurement_matrix: Optional[torch.Tensor] = None,
        measurement_noise: Optional[torch.Tensor] = None,
        precompute_precision=True,
    ) -> GaussianState:
        """Project the current state (usually the prior) onto the measurement space

        Use the measurement equation: z_k = H x_k + N(0, R).
        Support batch computation: You can provide multiple measurements, projections models (H, R)
        or/and multiple states. You just need to ensure that shapes are broadcastable.

        Example:
            # Initialize a random batch of gaussian state (5d)
            state = GaussianState(
                torch.randn(50, 5, 1),  # The last dimension is required.
                torch.randn(50, 5, 5),
            )

            # Use a single projection model
            measurement_matrix = torch.randn(3, 5)  # Compatible with (50, 5, 5)
            measurement_noise = torch.randn(3, 3)  # Broadcastable with (50, 3, 3)

            projection = kf.project(state, measurement_matrix, measurement_noise)
            projection.mean  # Shape: (50, 3, 1)  # projection for each state
            projection.covariance  # Shape: (50, 3, 3)

            # Use several models
            measurement_matrix = torch.randn(1, 1, 3, 5)  # Same measurement for each model, compatible with (50, 5, 5)
            measurement_noise = torch.randn(10, 1, 3, 3)  # Use different noises

            projection = kf.project(state, measurement_matrix, measurement_noise)
            projection.mean  # Shape: (1, 50, 3, 1)  # /!\\, the state will not be broadcasted to (10, 50, 5, 1).
            projection.covariance  # Shape: (10, 50, 3, 3)  # Projection cov for each model and each state

        Args:
            state (GaussianState): Current state estimation (Usually the results of `predict`)
            measurement_matrix (Optional[torch.Tensor]): Overwrite the default projection matrix
                Shape: (*, dim_z, dim_x)
            measurement_noise (Optional[torch.Tensor]): Overwrite the default projection noise)
                Shape: (*, dim_z, dim_z)
            precompute_precision (bool): Precompute precision matrix (inverse covariance)
                Done once to prevent more computations
                Default: True

        Returns:
            GaussianState: Prior on the next state

        """
        if measurement_matrix is None:
            measurement_matrix = self.measurement_matrix
        if measurement_noise is None:
            measurement_noise = self.measurement_noise

        mean = measurement_matrix @ state.mean
        covariance = measurement_matrix @ state.covariance @ measurement_matrix.mT + measurement_noise

        # covariance = covariance + 1e-8*torch.eye(covariance.shape[-1]).to(covariance.device)

        return GaussianState(
            mean,
            covariance,
            (
                # Cholesky inverse is usually slower with small dimensions
                # The inverse is transposed (back) to be contiguous: as it is symmetric
                # This is equivalent and faster to hold on the contiguous verison
                # But this may slightly increase floating errors.
                covariance.inverse().mT
                # torch.linalg.pinv(covariance)
                if precompute_precision
                else None
            ),
        )

    def update(
        self,
        state: GaussianState,
        measure: torch.Tensor,
        *,
        projection: Optional[GaussianState] = None,
        measurement_matrix: Optional[torch.Tensor] = None,
        measurement_noise: Optional[torch.Tensor] = None,
    ) -> GaussianState:
        """Compute the posterior estimation by integrating a new measure into the state

        Support batch computation: You can provide multiple measurements, projections models (H, R)
        or/and multiple states. You just need to ensure that shapes are broadcastable.

        Example:
            # Initialize a random batch of gaussian state (5d)
            state = GaussianState(
                torch.randn(50, 5, 1),  # The last dimension is required.
                torch.randn(50, 5, 5),
            )

            # Use a single projection model and a single measurement for each state
            measurement_matrix = torch.randn(3, 5)  # Compatible with (50, 5, 5)
            measurement_noise = torch.randn(3, 3)  # Broadcastable with (50, 3, 3)
            measure = torch.randn(50, 3, 3)

            new_state = kf.update(state, measure, None, measurement_matrix, measurement_noise)
            new_state.mean  # Shape: (50, 5, 1)  # Each state has been updated
            new_state.covariance  # Shape: (50, 5, 5)

            # Use several models and a single measurement for each state
            measurement_matrix = torch.randn(10, 1, 3, 5)  # Compatible with (50, 5, 5)
            measurement_noise = torch.randn(10, 1, 3, 3)  # Use different noises
            measure = torch.randn(50, 3, 1)  # The last unsqueezed dimension is required

            new_state = kf.update(state, measure, None, measurement_matrix, measurement_noise)
            new_state.mean  # Shape: (10, 50, 5, 1)  # Each state for each model has been updated
            new_state.covariance  # Shape: (10, 50, 5, 1)

            # Use several models and all measurements for each state
            measurement_matrix = torch.randn(10, 1, 3, 5)  # Compatible with (50, 5, 5)
            measurement_noise = torch.randn(10, 1, 3, 3)  # Use different noises
            # We have 50 measurements and we update each state/model with every measurements
            measure = torch.randn(50, 1, 1, 3, 1)

            new_state = kf.update(state, measure, None, measurement_matrix, measurement_noise)
            new_state.mean  # Shape: (50, 10, 50, 5, 1)  # Update for each measure, model and previous state
            new_state.covariance  # Shape: (10, 50, 5, 5)  # /!\\ The cov is not broadcasted to (50, 10, 50, 5, 5)

        Args:
            state (GaussianState): Current state estimation (Usually the results of `predict`)
            measure (torch.Tensor): State measure (z_k) (The last unsqueezed dimension is required)
                Shape: (*, dim_z, 1)
            projection (Optional[GaussianState]): Precomputed projection if any.
            measurement_matrix (Optional[torch.Tensor]): Overwrite the default projection matrix
                Shape: (*, dim_z, dim_x)
            measurement_noise (Optional[torch.Tensor]): Overwrite the default projection noise)
                Shape: (*, dim_z, dim_z)

        Returns:
            GaussianState: Prior on the next state

        """
        if measurement_matrix is None:
            measurement_matrix = self.measurement_matrix
        if measurement_noise is None:
            measurement_noise = self.measurement_noise
        if projection is None:
            projection = self.project(state, measurement_matrix=measurement_matrix, measurement_noise=measurement_noise, precompute_precision=True)

        residual = measure - projection.mean

        if projection.precision is None:  # Old version using cholesky and solve to prevent the inverse computation
            # Find K without inversing S but by solving the linear system SK^T = (PH^T)^T
            # May be slightly more robust but is usually slower in low dimension
            chol_decomposition, _ = torch.linalg.cholesky_ex(projection.covariance)  # pylint: disable=not-callable
            kalman_gain = torch.cholesky_solve(measurement_matrix @ state.covariance.mT, chol_decomposition).mT
        else:
            kalman_gain = state.covariance @ measurement_matrix.mT @ projection.precision

        mean = state.mean + kalman_gain @ residual

        # XXX: Did not use the more robust P = (I-KH)P(I-KH)' + KRK' from filterpy (as it is slower)
        # Again for robustness you should go with filterpy
        # covariance = state.covariance - kalman_gain @ measurement_matrix @ state.covariance
        factor = torch.eye(self.state_dim, dtype=self.dtype, device=self.device) - kalman_gain @ measurement_matrix
        covariance = factor @ state.covariance @ factor.mT + kalman_gain @ measurement_noise @ kalman_gain.mT

        return GaussianState(mean, covariance)


    def filter_parallel(
        self,
        observations: Tensor,   # (T, N)
        As: Tensor,             # (T, D, D)
        C: Tensor,              # (N, D)
        Qs: Tensor,             # (T, D, D)
        R: Tensor,              # (N, N)
        m0: Tensor,             # (D, 1)
        P0: Tensor,             # (D, D)
    ) -> Tuple[Tensor, Tensor]:
        """
        Parallel Scan Kalman Filter
        observations: (Trial, T, N)
        As: time-varying state transition matrices, (Trial, D, D)
        Qs: time-varying state noise matrices, (Trial, D, D)   
        C: time-invariant prjection matrix, (N, D)
        R: time-invariant observation noise diagonal matrix, (N, N)
        m0: initial state mean: (D, 1)
        P0: initial state covariance: (D, D)
        """
        # Build the initial elements
        first_elems = first_filtering_element(As[0], C, Qs[0], R, m0, P0, observations[:, 0])  # returns (A, b, C, J, eta)

        A_batched, b_batched, C_batched, J_batched, eta_batched = vmap(
            lambda A, Q, y: generic_filtering_element(A, C, Q, R, y), in_dims=1, out_dims=0
        )(As[1:].expand(1, -1, -1, -1), Qs[1:].expand(1, -1, -1, -1), observations[:, 1:])

        A_total   = torch.cat([first_elems[0].unsqueeze(0), A_batched],   dim=0)
        b_total   = torch.cat([first_elems[1].unsqueeze(0), b_batched],   dim=0)
        C_total   = torch.cat([first_elems[2].unsqueeze(0), C_batched],   dim=0)
        J_total   = torch.cat([first_elems[3].unsqueeze(0), J_batched],   dim=0)
        eta_total = torch.cat([first_elems[4].unsqueeze(0), eta_batched], dim=0)
        initial_elements = (A_total, b_total, C_total, J_total, eta_total)
        

        _, b_hist, C_hist, _, _ = associative_scan(filtering_operator, initial_elements, dim = 0, reverse=False)

        state = GaussianState(mean=b_hist, covariance=C_hist)

        return state

    def filter(
        self, state: GaussianState, measures: torch.Tensor, update_first=False, return_all=False
    ) -> GaussianState:
        """Filter signals with given measures

        It handles most of the default use-cases but it remains very standard, you probably will have to rewrite
        it for a specific problem. It supports nan values in measures. The states associated with a nan measure
        are not updated. For a multidimensional measure, a single nan value will invalidate all the measure
        (because the measurement matrix cannot be infered).

        Limitations examples:
        It only works if states and measures are already aligned (associated).
        It is memory intensive as it requires the input (and output if `return_all`) to be stored in a tensor.
        It does not support changing the Kalman model (F, Q, H, R) in time.

        Again all of this can be done manually using this function as a baseline for a more precise code.

        Args:
            state (GaussianState): Initial state to start filtering from
            measures (torch.Tensor): Measures in time
                Shape: (T, *, dim_z, 1)
            update_first (bool): Only update for the first timestep, then goes back to the predict / update cycle.
                Default: False
            return_all (bool): The state returns contains all states after an update step.
                To access predicted states, you either have to run again `predict` on the result, or do it manually.
                Default: False (Returns only the last state)

        Returns:
            GaussianState: The final updated state or all the update states (in a single GaussianState object)
        """
        # Convert state to the right dtype and device
        state = state.to(self.dtype).to(self.device)

        saver: GaussianState

        for t, measure in enumerate(measures):
            if t or not update_first:  # Do not predict on the first t
                state = self.predict(state, t)
                # state = self.predict(state)

            # Convert on the fly the measure to avoid to store them all in cuda memory
            # To avoid this overhead, the conversion can be done by the user before calling `batch_filter`
            measure = measure.to(self.dtype).to(self.device, non_blocking=True)

            # Support for nan measure: Do not update state associated with a nan measure
            mask = torch.isnan(measure[..., 0]).any(dim=-1)
            if mask.any():
                valid_state = GaussianState(state.mean[~mask], state.covariance[~mask])
                valid_state = self.update(valid_state, measure[~mask])  # Update states with valid measures
                state.mean[~mask] = valid_state.mean
                state.covariance[~mask] = valid_state.covariance
            else:
                state = self.update(state, measure)  # Update states

            if return_all:
                if t == 0:  # Create the saver now that we know the size of an updated state
                    # In this implementation, it cannot evolve in time, but it still supports
                    # to have a change from the initial_state shape to the first updated state (with the first measure)
                    saver = GaussianState(
                        torch.empty(
                            (measures.shape[0], *state.mean.shape), dtype=state.mean.dtype, device=state.mean.device
                        ),
                        torch.empty(
                            (measures.shape[0], *state.covariance.shape),
                            dtype=state.mean.dtype,
                            device=state.mean.device,
                        ),
                    )

                saver.mean[t] = state.mean
                saver.covariance[t] = state.covariance

        if return_all:
            return saver

        return state


    def rts_smooth(self, state: GaussianState) -> tuple[GaussianState, torch.Tensor]:
        out_mean = state.mean.clone()
        out_covariance = state.covariance.clone()

        # Keep process matrices at (T, dim, dim)
        process_matrix = self.process_matrix[:-1]
        process_noise = self.process_noise[:-1]
        
        # Using einsum for broadcasting over batch dimension
        # For (T,b,d1,d2) @ (T,d2,d3) -> (T,b,d1,d3)
        cov_at_process = torch.einsum('tbij,tjk->tbik', state.covariance[:-1], process_matrix.mT)
        predicted_covariance = torch.einsum('tij,tbjk->tbik', process_matrix, cov_at_process) + process_noise.unsqueeze(1)
        kalman_gains = torch.einsum('tbij,tbjk->tbik', cov_at_process, 
                                torch.linalg.inv(predicted_covariance))
        
        innovations = out_mean[1:] - torch.einsum('tij,tbjk->tbik', process_matrix, 
                                                state.mean[:-1])
        
        out_mean[:-1] = state.mean[:-1] + torch.einsum('tbij,tbjk->tbik', kalman_gains, 
                                                    innovations)
        
        covariance_diffs = out_covariance[1:] - predicted_covariance
        out_covariance[:-1] = state.covariance[:-1] + torch.einsum('tbij,tbjk,tbkl->tbil', 
                                                                kalman_gains, 
                                                                covariance_diffs, 
                                                                kalman_gains.mT)

        # pairwise_covs = (torch.einsum('tbij,tjk->tbik', out_covariance[:-1], process_matrix.mT) + 
        #                 torch.einsum('tbij,tbjk->tbik', kalman_gains, covariance_diffs))

        pairwise_covs = torch.einsum('tbij,tbjk->tbik', out_covariance[1:], kalman_gains.mT)

        return GaussianState(out_mean, out_covariance), pairwise_covs

