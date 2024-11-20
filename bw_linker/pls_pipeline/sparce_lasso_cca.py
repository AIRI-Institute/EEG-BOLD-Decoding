import warnings
from copy import deepcopy
from typing import Optional

import numpy as np


class SparseLassoCCA:
    """
    Sparce Lasso CCA method which chooses important EEG channels. Note: this is our implementation of the algorithm as
    the original paper does not have open-source code available. For the original problem statement which is
    being solved by this algorithm please refer to the original paper
    Singer N, Poker G, Dunsky-Moran N, Nemni S, Reznik Balter S, Doron M, Baker T, Dagher A, Zatorre RJ, Hendler T.
    Development and validation of an fMRI-informed EEG model of reward-related ventral striatum activation.
    Neuroimage. 2023 Aug 1;276:120183. doi: 10.1016/j.neuroimage.2023.120183.

    Parameters
    ----------
    group_lasso_coef : float
        Coefficient of the group lasso penalty
    norm_coef : float
        Coefficient of the L2 penalty
    """
    def __init__(self, group_lasso_coef: float = 1, norm_coef: float = 1):
        self._lambda1 = group_lasso_coef
        self._lambda2 = norm_coef
        self._weights = None

    def fit_find_coef(self, covariance_matrix: np.ndarray, indices_per_channel: dict, desired_n_channels: int,
                      min_coef: float = 0, max_coef: float = 1e6, min_step: float = 1e-2, min_norm: float = 0.1,
                      print_convergence: bool = True):
        """
        Runs a binary search and looks for a group_lasso_coef which will get the desired number of channels

        Parameters
        ----------
        covariance_matrix : np.ndarray
            A covariance matrix of EEG features and fMRI signal
        indices_per_channel : dict
            A dictionary with channel names as keys and list of indices of EEG features which refer to this channel as
            values
        desired_n_channels : int
            A number of channels desired to be selected
        min_coef : float
            Minimal group_lasso_coef for the binary search. Default: 0
        max_coef : float
            Maximal group_lasso_coef for the binary search. Default: 1e6
        min_step : float
            Minimal step for binary search between neighbouring coefficients. Default: 1e-2
        min_norm : float
            Minimal norm for EEG channel coefficients to consider it important. Default: 0.1
        print_convergence : bool
            If True, prints message after convergence. Default: True
        """
        def get_n_channels(min_norm):
            return len(self.get_large_weights(min_norm=min_norm))

        converged = False
        min_coef = int(min_coef / min_step)
        max_coef = int(max_coef / min_step)
        iternum = 0
        non_zero_ch_weights = None

        while not converged:
            curr_coef = min_coef + (max_coef - min_coef) // 2
            if (curr_coef == min_coef) or (curr_coef == max_coef):
                break
            self._lambda1 = curr_coef * min_step
            self.fit(covariance_matrix=covariance_matrix, indices_per_channel=indices_per_channel)
            iternum += 1
            curr_n_channes = get_n_channels(min_norm=min_norm)
            if curr_n_channes > 0:
                non_zero_ch_weights = deepcopy(self.weights)
            if curr_n_channes == desired_n_channels:
                converged = True
                break
            elif curr_n_channes > desired_n_channels:
                min_coef = curr_coef
            elif curr_n_channes < desired_n_channels:
                max_coef = curr_coef

        if converged:
            if print_convergence:
                n_channels = get_n_channels(min_norm=min_norm)
                print(f'Converged. Coefficient: {self._lambda1}. Found {n_channels} channels in {iternum} iterations.')
        else:
            assert non_zero_ch_weights is not None, 'Found no weights for positive n_channels'
            self.reset_weights()
            self._weights = non_zero_ch_weights
            n_channels = get_n_channels(min_norm=min_norm)
            warnings.warn(
                f'Not converged. Final coefficient: {self._lambda1}. Found {n_channels} channels after {iternum}'
                f'iterations. Requested: {desired_n_channels} channels.'
            )

    def reset_weights(self):
        self._weights = None

    def normalize_weights(self):
        """
        Normalizes per channel weights
        """
        all_weights = []
        for weights in self.weights.values():
            all_weights.append(weights)
        all_weights = np.concatenate(all_weights)
        norm = np.linalg.norm(all_weights)
        if norm > 1e-6:
            for ch_name in self._weights.keys():
                self._weights[ch_name] = self._weights[ch_name] / norm

    def fit(self, covariance_matrix: np.ndarray, indices_per_channel: dict):
        """
        Fits a model

        Parameters
        ----------
        covariance_matrix : np.ndarray
            A covariance matrix of EEG features and fMRI signal
        indices_per_channel : dict
            A dictionary with channel names as keys and list of indices of EEG features which refer to this channel as
            values
        """
        self.reset_weights()
        weights = {}
        assert covariance_matrix.shape[0] == 1, 'This methods so far works for a single ROI'
        covariance_matrix = covariance_matrix[0, :]

        for ch_name, ch_indices in indices_per_channel.items():
            covar_submatrix = covariance_matrix[ch_indices]
            scale = max(1 - self._lambda1 / np.linalg.norm(covar_submatrix), 0) / (2 * self._lambda2)
            weights[ch_name] = scale * covar_submatrix

        self._weights = weights
        self.normalize_weights()

    @property
    def weights(self):
        return deepcopy(self._weights)

    @property
    def group_lasso_coef(self):
        return self._lambda1

    def get_large_weights(self, min_norm: float):
        """
        Returns dict of channels and weights which are above min_norm

        Parameters
        ----------
        min_norm : float
            Minimal norm for EEG channel coefficients to consider it important

        Returns
        -------
        important_channels : dict
            A dict of channels and weights which are above min_norm
        """
        return {ch: weights for ch, weights in self._weights.items() if np.linalg.norm(weights) > min_norm}


def get_covariance_matrix(fmri: np.ndarray, flattened_eeg: np.ndarray):
    """
    Returns a covariance matrix between fMRI and flattened EEG features

    Parameters
    ----------
    fmri : np.ndarray
        An array with BOLD signal of shape (n_times, )
    flattened_eeg : np.ndarray
        An array with EEG features flattened across feature dimensions, shape (n_features, n_times)

    Returns
    -------
    covariance_matrix : : np.ndarray
        A covariance matrix between fMRI and EEG features
    """
    return fmri[None, :] @ flattened_eeg.T


def find_best_channels(
        lasso_indices_per_channel: dict, desired_n_channels: int,
        covariance_matrix: Optional[np.ndarray] = None,
        fmri: Optional[np.ndarray] = None, flattened_eeg: Optional[np.ndarray] = None,
        min_coef: float = 0, max_coef: float = 1e6, min_step: float = 1e-2, min_norm: float = 1e-6,
        print_convergence: bool = False
):
    """
    Creates a covariance matrix (if needed) and runs a full search of the best channels

    Parameters
    ----------
    lasso_indices_per_channel : dict
        A dictionary with channel names as keys and list of indices of EEG features which refer to this channel as
        values
    desired_n_channels : int
        A desired number of best channels
    covariance_matrix : np.ndarray or None
        Covariance matrix between fMRI and EEG features. If None, will calculate from fmri and flattened_eeg.
        Default: None
    fmri : np.ndarray or None
        BOLD signal used to calculate covariance matrix, if it is not provided. Shape (n_times, ). Default: None
    flattened_eeg : np.ndarray or None
        EEG signal flattened across features dimensions used to calculate covariance matrix, if it is not provided.
        Shape (n_features, n_times). Default: None
    desired_n_channels : int
        A number of channels desired to be selected
    min_coef : float
        Minimal group_lasso_coef for the binary search. Default: 0
    max_coef : float
        Maximal group_lasso_coef for the binary search. Default: 1e6
    min_step : float
        Minimal step for binary search between neighbouring coefficients. Default: 1e-2
    min_norm : float
        Minimal norm for EEG channel coefficients to consider it important. Default: 0.1
    print_convergence : bool
        If True, prints message after convergence. Default: True

    Returns
    -------

    """
    if covariance_matrix is None:
        assert (fmri is not None) and (flattened_eeg is not None)
        covariance_matrix = get_covariance_matrix(fmri=fmri, flattened_eeg=flattened_eeg)
    else:
        assert (fmri is None) and (flattened_eeg is None)
    indices_per_channel = deepcopy(lasso_indices_per_channel)
    lasso_cca = SparseLassoCCA(group_lasso_coef=1, norm_coef=1)
    lasso_cca.fit_find_coef(covariance_matrix=covariance_matrix,
                            indices_per_channel=lasso_indices_per_channel,
                            desired_n_channels=desired_n_channels,
                            min_coef=min_coef, max_coef=max_coef, min_step=min_step, min_norm=min_norm,
                            print_convergence=print_convergence)

    curr_weights = lasso_cca.get_large_weights(min_norm=min_norm)

    # pick indices of high weight channels
    channel_indices = []
    for key in curr_weights.keys():
        channel_indices = channel_indices + indices_per_channel[key]

    channel_names = [(key, np.linalg.norm(value)) for key, value in curr_weights.items()]
    channel_names = sorted(channel_names, key=lambda x: x[1], reverse=True)
    channel_names = [name for name, _ in channel_names]

    return channel_indices, channel_names, lasso_cca
