import torch
import torch.linalg
from ml.fit_data import FitData

DEVICE = 'cuda'

def make_pair_mask(group_sizes: list[int]) -> torch.Tensor:
    """
    Creates a boolean mask of shape (num_features, num_features), where:
      - mask[i,j] = True if features i and j are from different groups and i<j
      - mask[i,j] = False otherwise
    """

    group_indices = []
    i = 0
    for size in group_sizes:
        group_indices += [i] * size
        i += 1
    group_indices = torch.tensor(group_indices)

    i_less_than_j_mask = torch.ones(sum(group_sizes), sum(group_sizes), dtype=torch.bool).triu(diagonal=1)
    different_group_mask = group_indices.unsqueeze(1) != group_indices.unsqueeze(0)
    mask = i_less_than_j_mask & different_group_mask
    return mask

def make_triplet_mask(group_sizes: list[int]) -> torch.Tensor:
    """
    Creates a boolean mask of shape (num_features, num_features, num_features), where:
      - mask[i, j, k] = True if:
          * i < j < k
          * group_indices[i], group_indices[j], and group_indices[k] are all different
      - mask[i, j, k] = False otherwise
    """

    # 1) Assign each feature to a group
    group_indices = []
    g = 0
    for size in group_sizes:
        group_indices.extend([g]*size)
        g += 1
    group_indices = torch.tensor(group_indices)              # shape (F,)
    num_features = len(group_indices)

    # 2) Build index tensors i_indices, j_indices, k_indices
    #    i_indices shape = (F, 1, 1)
    #    j_indices shape = (1, F, 1)
    #    k_indices shape = (1, 1, F)
    i_indices = torch.arange(num_features).view(-1, 1, 1)
    j_indices = torch.arange(num_features).view(1, -1, 1)
    k_indices = torch.arange(num_features).view(1, 1, -1)

    # 3) Enforce i < j < k
    i_less_j_less_k_mask = (i_indices < j_indices) & (j_indices < k_indices)

    # 4) Ensure groups are distinct
    #    group_i, group_j, group_k will each broadcast to shape (F, F, F)
    group_i = group_indices[i_indices]
    group_j = group_indices[j_indices]
    group_k = group_indices[k_indices]

    diff_group_mask = (group_i != group_j) & (group_i != group_k) & (group_j != group_k)

    # 5) Combine conditions
    mask = i_less_j_less_k_mask & diff_group_mask
    return mask

def make_quad_mask(group_sizes: list[int]) -> torch.Tensor:
    """
    Creates a boolean mask of shape (num_features, num_features, num_features, num_features), where:
      - mask[i, j, k, l] = True if:
          * i < j < k < l
          * group_indices[i], group_indices[j], group_indices[k], group_indices[l] are all different
      - mask[i, j, k, l] = False otherwise
    """

    # 1) Build group_indices = [0,0,0,1,1,1,2,2,2,...] if group_sizes=[3,3,3,...]
    group_indices = []
    current_group = 0
    for size in group_sizes:
        group_indices.extend([current_group] * size)
        current_group += 1
    group_indices = torch.tensor(group_indices)                 # shape (F,)
    F = len(group_indices)

    # 2) Create i, j, k, l index tensors
    #    i_indices shape: (F,1,1,1)
    #    j_indices shape: (1,F,1,1)
    #    k_indices shape: (1,1,F,1)
    #    l_indices shape: (1,1,1,F)
    i_indices = torch.arange(F).view(-1, 1, 1, 1)
    j_indices = torch.arange(F).view(1, -1, 1, 1)
    k_indices = torch.arange(F).view(1, 1, -1, 1)
    l_indices = torch.arange(F).view(1, 1, 1, -1)

    # 3) Enforce i < j < k < l
    i_less_j_less_k_less_l = (
        (i_indices < j_indices) &
        (j_indices < k_indices) &
        (k_indices < l_indices)
    )

    # 4) Distinct group check
    #    group_i, group_j, group_k, group_l all broadcast to shape (F, F, F, F)
    group_i = group_indices[i_indices]
    group_j = group_indices[j_indices]
    group_k = group_indices[k_indices]
    group_l = group_indices[l_indices]

    distinct_group_mask = (
        (group_i != group_j) &
        (group_i != group_k) &
        (group_i != group_l) &
        (group_j != group_k) &
        (group_j != group_l) &
        (group_k != group_l)
    )

    # 5) Combine: True only when i<j<k<l AND all different groups
    quad_mask = i_less_j_less_k_less_l & distinct_group_mask
    return quad_mask

def expand_features(X: torch.Tensor, pair_indices: torch.Tensor, triplet_indices: torch.Tensor,
                    quad_indices: torch.Tensor) -> torch.Tensor:
    """
    Given an (N, F) input X (with F = sum(group_sizes)), returns an expanded
    design matrix A of shape (N, M), where:
      - The first F columns are onsite features (i.e., X itself)
      - Next columns are pairwise products x[i]*x[j] for valid pairs (i, j)
      - If triplet_mask is provided, further columns are x[i]*x[j]*x[k]
        for valid triplets (i, j, k).

    pair_mask: shape (F, F)
    triplet_mask: shape (F, F, F), or None if you are only using pairs

    Returns:
      A: shape (N, M). M = F + (#valid pairs) + (#valid triplets)
    """
    # 1) Onsite features, shape (N, F)
    A_list = [X]

    # 2) Pairwise features
    X_ij = X[:, pair_indices[:, 0]] * X[:, pair_indices[:, 1]]
    A_list.append(X_ij)

    # 3) Triplet features
    X_ijk = (
        X[:, triplet_indices[:, 0]]
        * X[:, triplet_indices[:, 1]]
        * X[:, triplet_indices[:, 2]]
    )
    A_list.append(X_ijk)

    # 4) Quad features
    X_ijkl = (
        X[:, quad_indices[:, 0]]
        * X[:, quad_indices[:, 1]]
        * X[:, quad_indices[:, 2]]
        * X[:, quad_indices[:, 3]]
    )
    A_list.append(X_ijkl)

    # Concatenate onsite, pairwise, and (optionally) triplet features along dim=1
    A = torch.cat(A_list, dim=1)
    return A


class ExactClusterExpansion:
    """
    Stores the onsite + pairwise coefficients and can apply them in forward().
    This is a linear model in expanded feature space.
    """
    def __init__(self, pair_indices: torch.Tensor, triplet_indices: torch.Tensor,
                 quad_indices: torch.Tensor, w: torch.Tensor):
        """
        w: shape (M,) - the solved linear coefficients.
        The first sum(group_sizes) entries correspond to onsite features,
        subsequent entries correspond to pairwise features defined by pair_mask.
        """
        self.pair_indices = pair_indices.cpu()
        self.triplet_indices = triplet_indices.cpu()
        self.quad_indices = quad_indices.cpu()
        self.w = w.cpu()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X shape: (N, F)
        Returns: (N,) energies
        """
        # Expand features with the same procedure
        A = expand_features(X, self.pair_indices, self.triplet_indices, self.quad_indices)
        return A @ self.w

def train_exact_ce(
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    pair_indices: torch.Tensor,
    triplet_indices: torch.Tensor,
    quad_indices: torch.Tensor,
    ridge_lambda: float,
) -> ExactClusterExpansion:
    """
    Constructs a design matrix, then performs a direct solve for the linear coefficients:
       w = (A^T A + ridge_lambda I)^(-1) A^T Y
    If ridge_lambda=0, it's a plain least squares.
    If ridge_lambda>0, it's ridge regression.
    """
    X_train = X_train.to(DEVICE)
    Y_train = Y_train.to(DEVICE)
    A = expand_features(X_train, pair_indices, triplet_indices, quad_indices)  # shape (N, M)

    ridge_lambda_I = ridge_lambda * torch.eye(A.shape[1], device=DEVICE, dtype=A.dtype)
    ATA = A.transpose(0,1) @ A + ridge_lambda_I # shape (M, M)
    ATy = A.transpose(0,1) @ Y_train  # shape (M,)
    w = torch.linalg.solve(ATA, ATy)  # shape (M,)
    return ExactClusterExpansion(pair_indices, triplet_indices, quad_indices, w)

def train_and_predict(
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_test: torch.Tensor | None,
    Y_test: torch.Tensor | None,
    pair_indices: torch.Tensor,
    triplet_indices: torch.Tensor,
    quad_indices: torch.Tensor,
    ridge_lambda: float,
) -> FitData:
    """
    Creates and trains a cluster expansion with a direct solve for the linear parameters,
    then returns predictions in a FitData object to integrate with your pipeline.
    """
    # Train
    model = train_exact_ce(X_train, Y_train, pair_indices, triplet_indices, quad_indices, ridge_lambda)

    # Predict on train
    train_mean = model.forward(X_train)
    train_std = torch.zeros_like(train_mean)

    # Predict on test (if provided)
    if X_test is None:
        Yhat_test_mean = None
        Yhat_test_std = None
        X_test_cpu = None
        Y_test_cpu = None
    else:
        Yhat_test_mean =  model.forward(X_test)
        Yhat_test_std = torch.zeros_like(Yhat_test_mean)
        X_test_cpu = X_test.cpu()
        Y_test_cpu = None if Y_test is None else Y_test.cpu()

    fit = FitData(
        X_train=X_train.cpu(),
        Y_train=Y_train.cpu(),
        Yhat_train_mean=train_mean,
        Yhat_train_std=train_std,
        X_test=X_test_cpu,
        Y_test=Y_test_cpu,
        Yhat_test_mean=Yhat_test_mean,
        Yhat_test_std=Yhat_test_std
    )
    return fit
