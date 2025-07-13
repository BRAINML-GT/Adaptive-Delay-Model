import numpy as np
import numpy.random as npr
from scipy.linalg import block_diag
from tqdm import trange
import torch
from cca_zoo_fixed import MCCA, CCA


def estimate_R_diag(Y_flat, prior_R, prior_weight):
    """
    Estimate posterior diagonal of R via a conjugate inverse-gamma / scaled-inv-chi2 update:
      Y_flat: (n_samples, N) flattened data
      prior_R: (N,) prior guess for var_i
      prior_weight: scalar pseudo-count (nu0)
    Returns:
      R_diag: (N,) posterior var estimate
    """
    # sample variance
    sample_var = np.var(Y_flat, axis=0, ddof=1)  # shape (N,)
    n = Y_flat.shape[0]
    # posterior mean-of-variance: (nu0*prior + n*sample)/(nu0 + n)
    R_diag = (prior_weight * prior_R + n * sample_var) / (prior_weight + n)
    return R_diag

def whiten_data(Y_flat, R_diag):
    """
    Whiten each channel by its noise std:
      Y_whiten[:, i] = Y_flat[:, i] / sqrt(R_diag[i])
    """
    return Y_flat / np.sqrt(R_diag)[None, :]

def init_C(y, num_groups, xdima, xdimw, ydims, prior_R, device, dtype):
    R_diag = prior_R

    if xdima != 0:
        cydims = np.cumsum(ydims)
        data_list = [y[0:cydims[0]].T]
        for i in range(len(ydims)-1):
            data_list.append(y[cydims[i]:cydims[i+1]].T)


        if len(ydims) == 2:
            cca = CCA(latent_dimensions = xdima).fit(data_list)
            C = cca.weights_
        else:
            cca = MCCA(latent_dimensions = xdima).fit(data_list)
            C = cca.weights_


        Cs = []
        for i in range(num_groups):
            C_across = C[i]
            if xdimw != 0:
                y_i = y[int(np.sum(ydims[0:i])): int(np.sum(ydims[0:i+1])), :]
                covY = np.cov(y_i)
                _, _, C_uncorr = np.linalg.svd(C_across.T @ covY)
                C_uncorr = C_uncorr[:, xdima:xdima + xdimw]
                C_all = np.concatenate((C_across, C_uncorr), axis=1)
                C_all = torch.tensor(C_all, device=device, dtype=dtype)
                Cs.append(C_all)
            else:
                Cs.append(torch.tensor(C_across, device=device, dtype=dtype))
    else:
        Cs = []
        for i in range(num_groups):
            if xdimw != 0:
                C_all = npr.randn(ydims[i], xdimw)
                C_all = torch.tensor(C_all, device=device, dtype=dtype)
                Cs.append(C_all)

    return Cs, torch.tensor(R_diag * np.ones((sum(ydims),)), device=device, dtype=dtype)