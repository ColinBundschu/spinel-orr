import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from ml.fit_data import FitData


DEVICE = 'cuda'

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dims: tuple[int, ...], output_dim):
        super(SimpleNN, self).__init__()
        if len(hidden_dims) < 1:
            raise ValueError("hidden_dim must contain at least one layer.")
        self.fc0 = nn.Linear(input_dim, hidden_dims[0])
        self.fc = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.fc.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        self.fco = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        x = torch.relu(self.fc0(x))
        for fci in self.fc:
            x = torch.relu(fci(x))
        return self.fco(x)

def train_nn(X_train, Y_train, hidden_dims, num_iterations, lr, decay, verbosity, gamma):
    input_dim = X_train.shape[1]
    output_dim = 1

    # Model definition
    model = SimpleNN(input_dim, hidden_dims, output_dim).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=decay)
    scheduler = ExponentialLR(optimizer, gamma=gamma)
    criterion = nn.L1Loss()

    for i in range(num_iterations):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train).squeeze()
        loss = criterion(outputs, Y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if verbosity >= 3 and i % 250 == 0:
            print(f"Step {i}: Loss = {loss.item()}")

        current_lr = scheduler.get_last_lr()[0]
        if current_lr < 1e-6:
            if verbosity >= 2:
                print(f'Early exit with LR: {current_lr}')
            break

    return model


def train_and_predict(X_train: torch.Tensor, Y_train: torch.Tensor, X_test: torch.Tensor | None,
                      Y_test: torch.Tensor | None, hidden_dims: tuple[int,...] = (128,), decay: float = 0.01,
                      verbosity: int = 3, lr: float = 0.001, gamma: float = 0.999, num_iterations: int = 3001
                      ) -> FitData:
    """
    Trains a standard neural network and computes predictions for training and test data.

    Args:
        X_train (torch.Tensor): Training input features (n_train_samples, n_features).
        Y_train (torch.Tensor): Training target values (n_train_samples,).
        X_test (torch.Tensor): Test input features (n_test_samples, n_features).
        Y_test (torch.Tensor): Test target values (n_test_samples,).
        hidden_dim (tuple[int,...]): Number of neurons in the hidden layer. Default is 50.
        num_iterations (int): Number of training iterations. Default is 5000.
        lr (float): Learning rate. Default is 0.01.
        device (str or torch.device): Device to use ('cuda', 'cpu', or None for automatic).

    Returns:
        FitData: Object containing predictions and standard deviations for training and test data.
    """
    model = train_nn(X_train, Y_train.squeeze(), hidden_dims=hidden_dims, num_iterations=num_iterations, lr=lr,
                     decay=decay, verbosity=verbosity, gamma=gamma)

    # Make predictions
    model.eval()
    with torch.no_grad():
        train_mean = model(X_train).squeeze().cpu()
    train_std = torch.zeros_like(train_mean)

    if X_test is None :
        Yhat_test_mean = None
        Yhat_test_std = None
    else:
        with torch.no_grad():
            Yhat_test_mean = model(X_test).squeeze(-1).cpu()
        Yhat_test_std = torch.zeros_like(Yhat_test_mean)
        X_test = X_test.cpu()
        Y_test = Y_test.cpu()

    # Extract posterior predictive means and standard deviations for test data
    fit = FitData(X_train.cpu(), Y_train.squeeze().cpu(), train_mean, train_std,
                  X_test, Y_test, Yhat_test_mean, Yhat_test_std)
    return fit
