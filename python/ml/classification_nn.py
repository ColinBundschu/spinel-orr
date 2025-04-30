import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR

from ml.fit_data import FitData

DEVICE = "cuda"


class SimpleNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], output_dim: int):
        super().__init__()
        if len(hidden_dims) < 1:
            raise ValueError("hidden_dims must contain at least one layer.")

        self.fc0 = nn.Linear(input_dim, hidden_dims[0])
        self.fc = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.fc.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.fco = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning 'logits' of shape (N, output_dim).
        We'll chunk them for each step in our custom logic.
        """
        x = torch.relu(self.fc0(x))
        for layer in self.fc:
            x = torch.relu(layer(x))
        return self.fco(x)


def custom_loss_function(
    logits: torch.Tensor,
    Y: torch.Tensor,
    step_sizes: list[int]
) -> torch.Tensor:
    """
    Custom loss that treats each step independently:
      - For step j, we do a softmax over the chunk of logits for that step
        so probabilities sum to 1 for that step only.
      - Then we take the dot product (sum_c p_c * Y_c).
      - We then average across the batch for each step, then sum over the steps.
    """
    logits_chunks = torch.split(logits, step_sizes, dim=1)  # each (N, step_size_j)
    Y_chunks = torch.split(Y, step_sizes, dim=1)            # each (N, step_size_j)

    total_loss = 0.0
    for step_logits, step_y in zip(logits_chunks, Y_chunks):
        # shape (N, step_size_j)
        probs = torch.softmax(step_logits, dim=1)
        # Weighted sum for each sample => shape (N,)
        step_pred = (probs * step_y).sum(dim=1)
        # Mean over the batch
        total_loss += step_pred.mean()
    return total_loss


def train_nn(
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    Yenc: torch.Tensor | None,
    step_sizes: list[int],
    hidden_dims: tuple[int, ...],
    num_iterations: int,
    lr: float,
    decay: float,
    verbosity: int,
    gamma: float,
) -> SimpleNN:
    """
    Trains a model with the custom_loss_function.
    Each step's logits are softmaxed separately.
    """
    if Yenc is None:
        input_dim = X_train.shape[1]
        output_dim = sum(step_sizes)
    else:
        input_dim = X_train.shape[1] + Yenc.shape[1]
        output_dim = 1

    model = SimpleNN(input_dim, hidden_dims, output_dim).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=decay)
    scheduler = ExponentialLR(optimizer, gamma=gamma)

    for i in range(num_iterations):
        model.train()
        optimizer.zero_grad()

        logits = compute_logits(X_train, Yenc, model)
        loss = custom_loss_function(logits, Y_train, step_sizes)
        loss.backward()

        optimizer.step()
        scheduler.step()

        if verbosity >= 3 and i % 250 == 0:
            print(f"Step {i}: Loss = {loss.item():.4f}")

        # Early stop if LR too small
        current_lr = scheduler.get_last_lr()[0]
        if current_lr < 1e-6:
            if verbosity >= 2:
                print(f"Early exit with LR: {current_lr}")
            break

    return model

def compute_logits(X_train, Yenc, model):
    if Yenc is None:
        return model(X_train)

    logits = []
    for enc in Yenc:
        enc = enc.expand(X_train.shape[0], -1)
        X_train_i = torch.cat((X_train, enc), dim=1)
        logits.append(model(X_train_i))
    logits = torch.cat(logits, dim=1)
    return logits


def predict_barrier(
    logits: torch.Tensor,
    Y: torch.Tensor,
    step_sizes: list[int],
    N_samples: int,
    randomized: bool,
) -> torch.Tensor:
    """
    Vectorized version of predict_barrier.
    Returns shape (batch_size, 2): (pred_barrier, true_barrier).
    """
    # Create the chunked versions of shape:
    #   logit_chunks[j] => (batch_size, step_sizes[j])
    #   y_chunks[j]     => (batch_size, step_sizes[j])
    logit_chunks = torch.split(logits, step_sizes, dim=1)
    y_chunks = torch.split(Y, step_sizes, dim=1)

    # We'll accumulate the step-min predictions and step-min truths.
    # pred_mins[j] => a 1D tensor of shape (batch_size,)
    # true_mins[j] => a 1D tensor of shape (batch_size,)
    pred_mins = []
    true_mins = []

    for lc, yc in zip(logit_chunks, y_chunks):
        # lc: shape (batch_size, step_size_j)
        # yc: shape (batch_size, step_size_j)
        batch_size, step_size_j = yc.shape

        # True step-min is always just the minimum over each row (sample)
        step_min_true = yc.min(dim=1).values

        # If we need to consider all energies or more than exist, pick the entire row
        if N_samples == -1 or N_samples >= step_size_j:
            step_min_pred = step_min_true
        elif randomized:
            # We pick random indices for each sample (vectorized).
            # shape = (batch_size, N_samples)
            probs = torch.ones((batch_size, step_size_j), device=yc.device)
            random_idxs = torch.multinomial(probs, N_samples, replacement=False)
            # Gather energies for those positions
            chosen_energies = torch.gather(yc, 1, random_idxs)
            # Min across those N_samples
            step_min_pred = chosen_energies.min(dim=1).values
        else:
            # Non-random approach: pick top N_samples by logit-based softmax probability
            row_probs = torch.softmax(lc, dim=1)  # shape = (batch_size, step_size_j)
            probs_top, top_idxs = torch.topk(row_probs, k=N_samples, dim=1)
            # gather the corresponding Y energies
            chosen_energies = torch.gather(yc, 1, top_idxs)
            step_min_pred = chosen_energies.min(dim=1).values

        pred_mins.append(step_min_pred)
        true_mins.append(step_min_true)

    # Each list has len(step_sizes) tensors of shape (batch_size,).
    # Stack along dim=1 to get shape (batch_size, num_steps).
    pred_mins_t = torch.stack(pred_mins, dim=1)  # (batch_size, num_steps)
    true_mins_t = torch.stack(true_mins, dim=1)  # (batch_size, num_steps)

    # To replicate the “cyclic difference then max” approach:
    # roll(1, dims=1) shifts everything by 1 along each row. 
    # Then we do (current - previous).max() for each sample.
    # If you need a different form of “cyclic,” adjust the roll by -1, etc.
    barrier_pred = (pred_mins_t - pred_mins_t.roll(1, dims=1)).max(dim=1).values
    barrier_true = (true_mins_t - true_mins_t.roll(1, dims=1)).max(dim=1).values

    # Finally, we return shape (batch_size, 2).
    barriers = torch.stack([barrier_pred, barrier_true], dim=1)
    return barriers


def train_and_predict(
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_test: torch.Tensor | None,
    Y_test: torch.Tensor | None,
    step_sizes: list[int],
    Yenc: torch.Tensor | None = None,
    hidden_dims: tuple[int, ...] = (128,),
    decay: float = 0.01,
    verbosity: int = 3,
    lr: float = 0.001,
    gamma: float = 0.998,
    num_iterations: int = 8001,
    randomized: bool = False,
) -> list[FitData]:
    """
    1) If randomized=False, train a model with a custom multi-step logic 
       (one chunk of logits per step), then do top-N predictions:
         - For each step chunk, pick top N probabilities, gather Y energies, pick min -> step_min
         - barrier_eV = max(step_mins - step_mins.roll(1)).
    2) If randomized=True, skip training and simply pick random N_samples for each step 
       to form the predicted barrier.
    3) Do the same for test if provided.
    """
    N_max_samples = max(step_sizes)
    test_pred_barriers = []
    test_true_barriers = []
    X_test_cpus = []

    if randomized:
        # === Random baseline branch: skip training entirely ===
        if verbosity >= 2:
            print("Skipping training; using random subsets for predicted barrier.")

        for N_samples in range(1, N_max_samples + 1):
            # Predict on train
            train_barriers = predict_barrier(Y_train, Y_train, step_sizes, N_samples, randomized)
            train_pred_barrier = train_barriers[:, 0]
            train_true_barrier = train_barriers[:, 1]

            # Predict on test
            if X_test is not None and Y_test is not None:
                test_barriers = predict_barrier(Y_test, Y_test, step_sizes, N_samples, randomized)
                test_pred_barriers.append(test_barriers[:, 0])
                test_true_barriers.append(test_barriers[:, 1])
                X_test_cpus.append(X_test)  # or .cpu() if you want consistency
            else:
                test_pred_barriers.append(None)
                test_true_barriers.append(None)
                X_test_cpus.append(None)

    else:
        # === Normal branch: train model and do top-N prediction ===
        X_train_dev = X_train.to(DEVICE)
        Y_train_dev = Y_train.to(DEVICE)
        if Yenc is not None:
            Yenc = Yenc.to(device=DEVICE, dtype=torch.float32)

        # 1) Train
        model = train_nn(
            X_train_dev,
            Y_train_dev,
            Yenc,
            step_sizes,
            hidden_dims,
            num_iterations,
            lr,
            decay,
            verbosity,
            gamma,
        )

        # 2) Predict on train set
        model.eval()
        with torch.no_grad():
            # shape: (n_train, sum_of_steps)
            train_logits = compute_logits(X_train_dev, Yenc, model)

        for N_samples in range(1, N_max_samples + 1):
            train_barriers = predict_barrier(train_logits, Y_train, step_sizes, N_samples, randomized)
            train_pred_barrier = train_barriers[:, 0]
            train_true_barrier = train_barriers[:, 1]

            # 3) Do test predictions
            if X_test is not None and Y_test is not None:
                X_test_dev = X_test.to(DEVICE)
                with torch.no_grad():
                    test_logits = compute_logits(X_test_dev, Yenc, model)

                test_barriers = predict_barrier(test_logits, Y_test, step_sizes, N_samples, randomized)
                test_pred_barriers.append(test_barriers[:, 0])
                test_true_barriers.append(test_barriers[:, 1])
                X_test_cpus.append(X_test.cpu())
            else:
                test_pred_barriers.append(None)
                test_true_barriers.append(None)
                X_test_cpus.append(None)

    # 4) Build FitData
    fits = []
    for X_test_cpu, test_pred_barrier, test_true_barrier in zip(X_test_cpus, test_pred_barriers, test_true_barriers):
        if test_pred_barrier is not None:
            assert test_true_barrier is not None
            assert X_test_cpu is not None

        fits.append(FitData(
            X_train.cpu(),
            train_true_barrier,
            train_pred_barrier,
            torch.zeros_like(train_pred_barrier),
            X_test_cpu,
            test_true_barrier,
            test_pred_barrier,
            torch.zeros_like(test_pred_barrier) if test_pred_barrier is not None else None,
    ))
    return fits
