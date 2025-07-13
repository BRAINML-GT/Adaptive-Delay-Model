from typing import Iterable, Union

import numpy as np
from scipy.linalg import block_diag, eigh
from sklearn.decomposition import PCA

from cca_zoo._utils._checks import _process_parameter

import itertools
from abc import abstractmethod
from typing import Iterable, Union, List, Optional, Any

import numpy as np
from numpy.linalg import svd
from scipy.linalg import block_diag
from sklearn.base import BaseEstimator, MultiOutputMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array

from cca_zoo._utils._cross_correlation import cross_corrcoef
from cca_zoo._utils._cross_correlation import cross_cov

class _BaseModel(BaseEstimator, MultiOutputMixin, TransformerMixin):
    """
    A base class for multivariate latent variable linear.

    This class implements common methods and attributes for fitting and transforming
    multiple representations of data using latent variable linear. It inherits from scikit-learn's
    BaseEstimator, MultiOutputMixin and RegressorMixin classes.

    Parameters
    ----------
    latent_dimensions: int, optional
        Number of latent dimensions to fit. Default is 1.
    copy_data: bool, optional
        Whether to copy the data. Default is True.
    accept_sparse: bool, optional
        Whether to accept sparse data. Default is False.
    random_state: int, RandomState instance or None, optional (default=None)
        Pass an int for reproducible output across multiple function calls.

    Attributes
    ----------
    n_views_: int
        Number of representations.
    n_features_in_: list of int
        Number of features for each view.
    weights_: list of numpy arrays
        Weight vectors for each view.
    """

    weights_ = None

    def __init__(
        self,
        latent_dimensions: int = 1,
        copy_data=True,
        accept_sparse=False,
        random_state: Union[int, np.random.RandomState] = None,
    ):
        self.latent_dimensions = latent_dimensions
        self.copy_data = copy_data
        self.accept_sparse = accept_sparse
        self.random_state = random_state

    def _validate_data(self, views: Iterable[np.ndarray]):
        # if not self._get_tags().get("multiview", False) and len(views) > 2:
        # # if len(views) > 2:
        #     raise ValueError(
        #         f"Model can only be used with two representations, but {len(views)} were given. "
        #         "Use MCCA or GCCA instead for CCA or MPLS for PLS."
        #     )
        if self.copy_data:
            views = [
                check_array(
                    view,
                    copy=True,
                    accept_sparse=False,
                    accept_large_sparse=False,
                    ensure_min_samples=max(2, self.latent_dimensions),
                    ensure_min_features=self.latent_dimensions,
                )
                for view in views
            ]
        else:
            views = [
                check_array(
                    view,
                    copy=False,
                    accept_sparse=False,
                    accept_large_sparse=False,
                    ensure_min_samples=max(2, self.latent_dimensions),
                    ensure_min_features=self.latent_dimensions,
                )
                for view in views
            ]
        if not all(view.shape[0] == views[0].shape[0] for view in views):
            raise ValueError("All representations must have the same number of samples")
        if not all(view.ndim == 2 for view in views):
            raise ValueError("All representations must have 2 dimensions")
        self.n_views_ = len(views)
        self.n_features_in_ = [view.shape[1] for view in views]
        self.n_samples_ = views[0].shape[0]
        return views

    def _check_params(self):
        """
        Checks the parameters of the model.
        """
        pass

    @abstractmethod
    def fit(self, views: Iterable[np.ndarray], y=None, **kwargs):
        """
        Fits the model to the given data

        Parameters
        ----------
        views: list/tuple of numpy arrays or array likes with the same number of rows (samples)
        y: None
        kwargs: any additional keyword arguments required by the given model

        Returns
        -------
        self: object

        """
        return self

    def transform(
        self, views: Iterable[np.ndarray], *args, **kwargs
    ) -> List[np.ndarray]:
        """
        Transforms the given representations using the fitted model.

        Parameters
        ----------
        views: list/tuple of numpy arrays or array likes with the same number of rows (samples)
        kwargs: any additional keyword arguments required by the given model

        Returns
        -------
        representations: list of numpy arrays

        """
        check_is_fitted(self)
        views = [
            check_array(
                view,
                copy=True,
                accept_sparse=False,
                accept_large_sparse=False,
            )
            for view in views
        ]
        representations = []
        for i, view in enumerate(views):
            representation = view @ self.weights_[i]
            representations.append(representation)
        return representations

    def pairwise_correlations(
        self, views: Iterable[np.ndarray], **kwargs
    ) -> np.ndarray:
        """
        Calculate pairwise correlations between representations in each dimension.

        Parameters
        ----------
        views: list/tuple of numpy arrays or array-like objects with the same number of rows (samples)
        kwargs: any additional keyword arguments required by the given model

        Returns
        -------
        pairwise_correlations: numpy array of shape (n_views, n_views, latent_dimensions)
        """
        representations = self.transform(views, **kwargs)
        all_corrs = []
        for x, y in itertools.product(representations, repeat=2):
            all_corrs.append(np.diag(cross_corrcoef(x.T, y.T)))
        all_corrs = np.array(all_corrs).reshape(
            (self.n_views_, self.n_views_, self.latent_dimensions)
        )
        return all_corrs

    def average_pairwise_correlations(
        self, views: Iterable[np.ndarray], **kwargs
    ) -> np.ndarray:
        """
        Calculate the average pairwise correlations between representations in each dimension.

        Parameters
        ----------
        views: list/tuple of numpy arrays or array-like objects with the same number of rows (samples)
        kwargs: any additional keyword arguments required by the given model

        Returns
        -------
        average_pairwise_correlations: numpy array of shape (latent_dimensions, )
        """
        pair_corrs = self.pairwise_correlations(views, **kwargs)
        # Sum all the pairwise correlations for each dimension, subtract self-correlations, and divide by the number of representations
        dim_corrs = np.sum(pair_corrs, axis=(0, 1)) - pair_corrs.shape[0]
        # Number of pairs is n_views choose 2
        num_pairs = (self.n_views_ * (self.n_views_ - 1)) / 2
        dim_corrs = dim_corrs / (2 * num_pairs)
        return dim_corrs

    def score(
        self, views: Iterable[np.ndarray], y: Optional[Any] = None, **kwargs
    ) -> float:
        """
        Calculate the sum of average pairwise correlations between representations.

        Parameters
        ----------
        views: list/tuple of numpy arrays or array-like objects with the same number of rows (samples)
        y: None
        kwargs: any additional keyword arguments required by the given model

        Returns
        -------
        score: float
            Sum of average pairwise correlations between representations.
        """
        return self.average_pairwise_correlations(views, **kwargs).sum()

    def loadings_(self, views: Iterable[np.ndarray], **kwargs) -> List[np.ndarray]:
        """
        Calculate canonical loadings for each view.

        Canonical loadings represent the correlation between the original variables
        in a view and their respective canonical variates. Canonical variates are
        linear combinations of the original variables formed to maximize the correlation
        with canonical variates from another view.

        Mathematically, given two representations \(X_i\), canonical variates
        from the representations are:

            \(Z_i = w_i^T X_i\)

        The canonical loading for a variable in \(X_i\) is the correlation between
        that variable and \(Z_i\).

        Parameters
        ----------
        views: list/tuple of numpy arrays
            Each array corresponds to a view. All representations must have the same number of rows (observations).

        Returns
        -------
        loadings_: list of numpy arrays
            Canonical loadings for each view. High absolute values indicate that
            the respective original variables play a significant role in defining the canonical variate.

        """
        check_is_fitted(self, attributes=["weights_"])
        representations = self.transform(views, **kwargs)
        loadings = [
            cross_corrcoef(view, representation, rowvar=False)
            for view, representation in zip(views, representations)
        ]
        return loadings

    def explained_variance(self, views: Iterable[np.ndarray]) -> List[np.ndarray]:
        """
        Calculates variance captured by each latent dimension for each view.

        Returns
        -------
        variances_by_dimension: list of numpy arrays
        """
        check_is_fitted(self, attributes=["weights_"])

        normalized_weights_ = [
            weight / np.linalg.norm(weight, axis=0) for weight in self.weights_
        ]

        # Transform views using normalized weights
        transformed_views = [
            view @ weights for view, weights in zip(views, normalized_weights_)
        ]

        # Calculate variance for each latent dimension
        variances_by_dimension = [
            np.var(transformed_view, axis=0) for transformed_view in transformed_views
        ]
        return variances_by_dimension

    def explained_variance_ratio(self, views: Iterable[np.ndarray]) -> List[np.ndarray]:
        """
        Calculates variance ratio captured by each latent dimension for each view.

        Returns
        -------
        variance_ratios: list of numpy arrays
        """
        total_variances = [
            np.sum(s**2) / (view.shape[0] - 1)
            for view in views
            for _, s, _ in [svd(view)]
        ]

        variances_by_dimension = self.explained_variance(views)

        # Calculate variance ratio for each dimension
        variance_ratios = [
            var_by_dim / total_var
            for var_by_dim, total_var in zip(variances_by_dimension, total_variances)
        ]
        return variance_ratios

    def explained_variance_cumulative(
        self, views: Iterable[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Calculates cumulative explained variance ratio for each latent dimension.

        Returns
        -------
        cumulative_variance_ratios: list of numpy arrays
        """
        variance_ratios = self.explained_variance_ratio(views)
        cumulative_variance_ratios = [np.cumsum(ratio) for ratio in variance_ratios]
        return cumulative_variance_ratios

    def _compute_covariance(self, views: Iterable[np.ndarray]) -> np.ndarray:
        """
        Computes the covariance matrix for the given representations.

        Parameters
        ----------
        views: list/tuple of numpy arrays or array likes with the same number of rows (samples)

        Returns
        -------
        cov: numpy array
            Computed covariance matrix.
        """
        cov = np.cov(np.hstack(views), rowvar=False)
        cov -= block_diag(*[np.cov(view, rowvar=False) for view in views])
        return cov

    def explained_covariance(self, views: Iterable[np.ndarray]) -> np.ndarray:
        """
        Calculates the covariance matrix of the transformed components for each view.

        Parameters
        ----------
        views: list/tuple of numpy arrays or array likes with the same number of rows (samples)

        Returns
        -------
        explained_covariances: list of numpy arrays
            Covariance matrices for the transformed components of each view.
        """
        check_is_fitted(self, attributes=["weights_"])

        # Transform the representations using the loadings_
        representations = [
            view @ loading for view, loading in zip(views, self.loadings_(views))
        ]

        k = representations[0].shape[1]

        explained_covariances = np.zeros(k)

        # just take the kth column of each transformed view and _compute_covariance
        for i in range(k):
            representations_k = [view[:, i][:, None] for view in representations]
            cov_ = self._compute_covariance(representations_k)
            _, s_, _ = svd(cov_)
            explained_covariances[i] = s_[0]

        return explained_covariances

    def explained_covariance_ratio(self, views: Iterable[np.ndarray]) -> np.ndarray:
        # only works for 2 views
        check_is_fitted(self, attributes=["weights_"])
        assert len(views) == 2, "Only works for 2 views"
        minimum_dimension = min([view.shape[1] for view in views])
        cov = self._compute_covariance(views)
        _, S, _ = svd(cov)
        # select every other element starting from the first until the minimum dimension
        total_explained_covariance = S[::2][:minimum_dimension].sum()
        explained_covariances = self.explained_covariance(views)
        explained_covariance_ratios = explained_covariances / total_explained_covariance
        return explained_covariance_ratios

    def explained_covariance_cumulative(
        self, views: Iterable[np.ndarray]
    ) -> np.ndarray:
        """
        Calculates the cumulative explained covariance ratio for each latent dimension for each view.

        Returns
        -------
        cumulative_ratios: list of numpy arrays
        """
        ratios = self.explained_covariance_ratio(views)
        cumulative_ratios = [np.cumsum(ratio) for ratio in ratios]

        return cumulative_ratios



class MCCA(_BaseModel):
    r"""
    A class used to fit a Multiview Ridge CCA model.
    This model adds a regularization term to the MCCA objective function to avoid overfitting and improve stability.
    It uses PCA to perform the optimization efficiently for high dimensional data.

    where :math:`c_i` are the regularization parameters for each view.

    Parameters
    ----------
    latent_dimensions : int, optional
        Number of latent dimensions to use, by default 1
    copy_data : bool, optional
        Whether to copy the data, by default True
    random_state : int, optional
        Random state, by default None
    c : Union[Iterable[float], float], optional
        Regularisation parameter, by default None
    accept_sparse : Union[bool, str], optional
        Whether to accept sparse data, by default None

    Examples
    --------
    >>> import numpy as np
    >>> rng=np.random.RandomState(0)
    >>> X1 = rng.random((10,5))
    >>> X2 = rng.random((10,5))
    >>> model = MCCA()
    >>> model.fit((X1,X2)).score((X1,X2))

    References
    --------
    Vinod, Hrishikesh _B. "Canonical ridge and econometrics of joint production." Journal of econometrics 4.2 (1976): 147-166.
    """

    def __init__(
        self,
        latent_dimensions: int = 1,
        copy_data=True,
        random_state=None,
        c: Union[Iterable[float], float] = None,
        accept_sparse=None,
        eps: float = 1e-6,
        pca: bool = True,
    ):
        # Set the default value for accept_sparse
        if accept_sparse is None:
            accept_sparse = ["csc", "csr"]
        # Call the parent class constructor
        super().__init__(
            latent_dimensions=latent_dimensions,
            copy_data=copy_data,
            accept_sparse=accept_sparse,
            random_state=random_state,
        )
        # Store the c parameter
        self.c = c
        self.eps = eps
        self.pca = pca

    def _check_params(self):
        # Process the c parameter for each view
        self.c = _process_parameter("c", self.c, 0, self.n_views_)

    def fit(self, views: Iterable[np.ndarray], y=None, **kwargs):
        # Validate the input data
        views = self._validate_data(views)
        # Check the parameters
        self._check_params()
        views = self._process_data(views, **kwargs)
        eigvals, eigvecs = self._solve_gevp(views, y=y, **kwargs)
        # Compute the weights_ for each view
        self._weights(eigvals, eigvecs, views, **kwargs)
        return self

    def _process_data(self, views, **kwargs):
        if self.pca:
            views = self._apply_pca(views)
        return views

    def _solve_gevp(self, views: Iterable[np.ndarray], y=None, **kwargs):
        # Setup the eigenvalue problem
        A = self._A(views, **kwargs)
        B = self._B(views, **kwargs)
        self.splits = np.cumsum([view.shape[1] for view in views])
        # Solve the eigenvalue problem
        # Get the dimension of _A
        p = A.shape[0]
        # Solve the generalized eigenvalue problem Cx=lambda Dx using a subset of eigenvalues and eigenvectors
        [eigvals, eigvecs] = eigh(
            A,
            B,
            subset_by_index=[p - self.latent_dimensions, p - 1],
        )
        # Sort the eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigvals, axis=0)[::-1]
        if eigvals.shape[0] < self.latent_dimensions:
            [eigvals, eigvecs] = eigh(
                A,
                B,
            )
            # Sort the eigenvalues and eigenvectors in descending order
            idx = np.argsort(eigvals, axis=0)[::-1][: self.latent_dimensions]
        eigvecs = eigvecs[:, idx].real
        eigvals = eigvals[idx].real
        return eigvals, eigvecs

    def _weights(self, eigvals, eigvecs, views, **kwargs):
        # split eigvecs into weights_ for each view
        self.weights_ = np.split(eigvecs, self.splits[:-1], axis=0)
        if self.pca:
            # go from weights_ in PCA space to weights_ in original space
            self.weights_ = [
                pca.components_.T @ self.weights_[i]
                for i, pca in enumerate(self.pca_models)
            ]
            del self.pca_models

    def _apply_pca(self, views):
        """
        Do data driven PCA on each view
        """
        self.pca_models = [PCA() for _ in views]
        # Fit PCA on each view
        return [self.pca_models[i].fit_transform(view) for i, view in enumerate(views)]

    def _A(self, views, **kwargs):
        """
        Constructs the matrix A for the generalized eigenvalue problem in MCCA.

        Matrix A represents the between-view covariance matrix, capturing the
        covariance among all different views. It is calculated as the covariance
        matrix of the stacked views, from which the within-view covariance matrices
        are subtracted.

        Parameters
        ----------
        views : list of numpy.ndarray
            The input views, where each view is a numpy array.

        Returns
        -------
        numpy.ndarray
            The matrix A, representing the between-view covariance matrix.
        """
        all_views = np.hstack(views)
        A = np.cov(all_views, rowvar=False)
        A -= block_diag(*[np.cov(view, rowvar=False) for view in views])
        return A / len(views)

    def _B(self, views, **kwargs):
        """
        Constructs the matrix B for the generalized eigenvalue problem in MCCA.

        Matrix B represents the within-view covariance matrix with regularization.
        In the PCA case, it is constructed using the explained variance of each view
        with regularization terms added to the diagonal. Without PCA, it is the
        block diagonal matrix of each view's covariance matrix adjusted by the
        regularization parameter.

        The regularization improves numerical stability and controls overfitting.

        Parameters
        ----------
        views : list of numpy.ndarray
            The input views, where each view is a numpy array.

        Returns
        -------
        numpy.ndarray
            The matrix B, representing the regularized within-view covariance matrix.
        """
        if self.pca:
            B = block_diag(
                *[
                    np.diag((1 - self.c[i]) * pc.explained_variance_ + self.c[i])
                    for i, pc in enumerate(self.pca_models)
                ]
            )
        else:
            B = block_diag(
                *[
                    (1 - self.c[i]) * np.cov(view, rowvar=False)
                    + self.c[i] * np.eye(view.shape[1])
                    for i, view in enumerate(views)
                ]
            )
        B_smallest_eig = min(0, np.linalg.eigvalsh(B).min()) - self.eps
        B = B - B_smallest_eig * np.eye(B.shape[0])
        return B / len(views)

    def _more_tags(self):
        # Indicate that this class is for multiview data
        return {"multiview": True}


class rCCA(MCCA):
    r"""
    A class used to fit Regularised CCA (canonical ridge) model. This model adds a regularization term to the CCA objective function to avoid overfitting and improve stability. It uses PCA to perform the optimization efficiently for high dimensional data.

    The objective function of regularised CCA is:

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{ w_1^TX_1^TX_2w_2  \}\\

        \text{subject to:}

        (1-c_1)w_1^TX_1^TX_1w_1+c_1w_1^Tw_1=n

        (1-c_2)w_2^TX_2^TX_2w_2+c_2w_2^Tw_2=n

    where :math:`c_i` are the regularization parameters for each view.

    Examples
    --------
    >>> import numpy as np
    >>> rng=np.random.RandomState(0)
    >>> X1 = rng.random((10,5))
    >>> X2 = rng.random((10,5))
    >>> model = rCCA(c=0.1)
    >>> model.fit((X1,X2)).score((X1,X2))


    References
    --------
    Vinod, Hrishikesh _B. "Canonical ridge and econometrics of joint production." Journal of econometrics 4.2 (1976): 147-166.
    """

    def _A(self, views, **kwargs):
        if len(views) != 2:
            raise ValueError(
                f"Model can only be used with two representations, but {len(views)} were given. Use MCCA or GCCA instead for CCA or MPLS for PLS."
            )
        if self.pca:
            # Compute the B matrices for each view
            B = [
                (1 - self.c[i]) * pc.explained_variance_ + self.c[i]
                for i, pc in enumerate(self.pca_models)
            ]
            C = cross_cov(
                views[0] / np.sqrt(B[0]), views[1] / np.sqrt(B[1]), rowvar=False
            )
            self.primary_view = 0
            return C @ C.T
        else:
            # cholesky decomposition of views
            self.L0 = np.linalg.inv(
                np.linalg.cholesky(
                    (1 - self.c[0]) * np.cov(views[0], rowvar=False)
                    + (self.c[0] + self.eps) * np.eye(views[0].shape[1])
                )
            )
            self.L1 = np.linalg.inv(
                np.linalg.cholesky(
                    (1 - self.c[1]) * np.cov(views[1], rowvar=False)
                    + (self.c[1] + self.eps) * np.eye(views[1].shape[1])
                )
            )
            C = cross_cov(views[0], views[1], rowvar=False)
            if views[0].shape[1] <= views[1].shape[1]:
                self.primary_view = 0
                self.T = self.L0 @ C @ self.L1 @ self.L1.T @ C.T @ self.L0.T
                return self.T
            else:
                self.primary_view = 1
                self.T = self.L1 @ C.T @ self.L0 @ self.L0.T @ C @ self.L1.T
                return self.T

    def _B(self, views, **kwargs):
        return None

    def _weights(self, eigvals, eigvecs, views):
        self.weights_ = [None] * 2
        if self.pca:
            B = [
                (1 - self.c[i]) * pc.singular_values_**2 / self.n_samples_ + self.c[i]
                for i, pc in enumerate(self.pca_models)
            ]
            C = np.cov(
                views[self.primary_view], views[1 - self.primary_view], rowvar=False
            )[
                0 : views[self.primary_view].shape[1],
                views[self.primary_view].shape[1] :,
            ]
            # Compute the weight matrix for primary view
            self.weights_[1 - self.primary_view] = (
                # Project view 1 onto its principal components
                self.pca_models[1 - self.primary_view].components_.T
                # Scale by the inverse of B[0]
                @ np.diag(1 / B[1 - self.primary_view])
                # Multiply by the cross-covariance matrix
                @ C.T
                # Scale by the inverse of the square root of B[1]
                @ np.diag(1 / np.sqrt(B[self.primary_view]))
                # Multiply by the eigenvectors
                @ eigvecs
                # Scale by the inverse of the square root of eigenvalues
                / np.sqrt(eigvals)
            )

            # Compute the weight matrix for view 2
            self.weights_[self.primary_view] = (
                # Project view 2 onto its principal components
                self.pca_models[self.primary_view].components_.T
                # Scale by the inverse of the square root of B[1]
                @ np.diag(1 / np.sqrt(B[self.primary_view]))
                # Multiply by the eigenvectors
                @ eigvecs
            )
        else:
            if self.primary_view == 0:
                self.weights_[0] = self.L0.T @ eigvecs
                self.weights_[1] = (
                    (self.L1.T @ self.L1)
                    @ cross_cov(views[1], views[0], rowvar=False)
                    @ self.weights_[0]
                )
            else:
                self.weights_[1] = self.L1.T @ eigvecs
                self.weights_[0] = (
                    (self.L0.T @ self.L0)
                    @ cross_cov(views[0], views[1], rowvar=False)
                    @ self.weights_[1]
                )

    def _more_tags(self):
        # Inherit all tags from MCCA but override the multiview tag
        tags = super()._more_tags()
        tags["multiview"] = False
        return tags


class CCA(rCCA):
    r"""
    A class used to fit a simple CCA model. This model finds the linear projections of two representations that maximize their correlation.

    The objective function of CCA is:

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{ w_1^TX_1^TX_2w_2  \}\\

        \text{subject to:}

        w_1^TX_1^TX_1w_1=n

        w_2^TX_2^TX_2w_2=n

    Parameters
    ----------
    latent_dimensions : int, optional
        Number of latent dimensions to use, by default 1
    copy_data : bool, optional
        Whether to copy the data, by default True
    random_state : int, optional
        Random seed for reproducibility, by default None

    References
    --------

    Hotelling, Harold. "Relations between two sets of variates." Breakthroughs in statistics. Springer, New York, NY, 1992. 162-190.

    Example
    -------
    >>> import numpy as np
    >>> rng=np.random.RandomState(0)
    >>> X1 = rng.random((10,5))
    >>> X2 = rng.random((10,5))
    >>> model = CCA()
    >>> model.fit((X1,X2)).score((X1,X2))
    """

    def __init__(
        self,
        latent_dimensions: int = 1,
        copy_data=True,
        random_state=None,
        accept_sparse=None,
        eps: float = 1e-6,
        pca: bool = True,
    ):
        # Initialize the rCCA class with c set to 0
        super().__init__(
            latent_dimensions=latent_dimensions,
            copy_data=copy_data,
            random_state=random_state,
            c=0,  # Setting c to 0
            accept_sparse=accept_sparse,
            eps=eps,
            pca=pca,
        )
