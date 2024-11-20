import torch
import torch.nn as nn


class NegativeCorrelation(nn.Module):
    """
    An implementation of the Negative Correlation in PyTorch. Intended to be used as a loss function

    Parameters
    ----------
    dim : int
        A dimension over which the correlation is supposed to be computed. Expected input size to module is
        (batch, n_rois, n_times). Default: -1
    eps : float
        Small value to avoid division by 0. Default: 1e-6
    """
    def __init__(self, dim: int = -1, eps: float = 1e-6):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=dim, eps=eps)
        self.dim = dim

    def forward(self, predictions, targets):
        pearson = self.cos(predictions - predictions.mean(dim=self.dim, keepdim=True),
                           targets - targets.mean(dim=self.dim, keepdim=True))
        return -1. * torch.mean(pearson)


def build_criterion(criterion_name: str, criterion_kwargs: dict):
    """
    Initializes a criterion (loss function) and returns it

    Parameters
    ----------
    criterion_name : str
        A name of the required criterion
    criterion_kwargs : dict
        A dict with keyword arguments for a desired criterion

    Returns
    -------
    criterion : torch.nn.Module
        An initialized criterion
    """
    if criterion_name == 'MSELoss':
        criterion = nn.MSELoss(**criterion_kwargs)
    elif criterion_name == 'L1Loss':
        criterion = nn.MSELoss(**criterion_kwargs)
    elif criterion_name == 'NegativeCorrelation':
        criterion = NegativeCorrelation(**criterion_kwargs)
    else:
        raise NotImplementedError
    return criterion
