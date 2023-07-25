import torch


def dimensionalwise_normalize(matrix: torch.Tensor, iter=2) -> torch.Tensor:
    matrix = matrix.detach().clone()
    for _ in range(iter):
        for dim in range(matrix.dim()):
            torch.nn.functional.normalize(matrix, dim=dim, eps=1e-6)
    return matrix
