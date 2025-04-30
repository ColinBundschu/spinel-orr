import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset

from ml.fit_data import FitData

DEVICE = 'cuda'

class DFTPredictorNN(nn.Module):
    def __init__(self, input_dim, hidden_dims: tuple[int, ...], output_dim, dropout: float = 0.0):
        super(DFTPredictorNN, self).__init__()
        if len(hidden_dims) < 1:
            raise ValueError("hidden_dims must contain at least one layer.")

        self.fc0 = nn.Linear(input_dim, hidden_dims[0])
        self.bn0 = nn.BatchNorm1d(hidden_dims[0])  # BatchNorm for first layer
        
        self.fc = nn.ModuleList()
        self.bn = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.fc.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.bn.append(nn.BatchNorm1d(hidden_dims[i + 1]))

        self.fco = nn.Linear(hidden_dims[-1], output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = torch.relu(self.bn0(self.fc0(x)))
        x = self.dropout(x)
        
        for fci, bni in zip(self.fc, self.bn):
            x = torch.relu(bni(fci(x)))
            x = self.dropout(x)
        
        return self.fco(x)

def train_nn(X_train, Y_train, hidden_dims, epochs, lr, decay, verbosity, gamma, batch_size, dropout=0.0):
    input_dim = X_train.shape[1]
    output_dim = 1

    # Model definition
    model = DFTPredictorNN(input_dim, hidden_dims, output_dim, dropout=dropout).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=decay)
    scheduler = ExponentialLR(optimizer, gamma=gamma)
    criterion = nn.SmoothL1Loss()

    # DataLoader setup
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for i in range(epochs):
        model.train()
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if verbosity >= 3:
            print(f"Epoch {i}: Loss = {loss.item()}")

        current_lr = scheduler.get_last_lr()[0]
        if current_lr < 1e-6:
            if verbosity >= 2:
                print(f'Early exit with LR: {current_lr}')
            break

    return model

def train_and_predict(X_train: torch.Tensor, Y_train: torch.Tensor, X_test: torch.Tensor | None, Y_test: torch.Tensor | None,
                      hidden_dims: tuple[int, ...] = (128,), decay: float = 0.01, verbosity: int = 0,
                      lr: float = 0.001, gamma: float = 0.999, num_iterations: int = 5000,
                      batch_size: int = 512, dropout: float = 0.0) -> FitData:
    """
    Trains a DFTPredictorNN and computes predictions for training and test data.

    Args:
        X_train (torch.Tensor): Training input features (n_train_samples, n_features).
        Y_train (torch.Tensor): Training target values (n_train_samples,).
        X_test (torch.Tensor): Test input features (n_test_samples, n_features).
        Y_test (torch.Tensor): Test target values (n_test_samples,).
        hidden_dims (tuple[int,...]): Number of neurons in each hidden layer.
        decay (float): Weight decay (L2 regularization).
        lr (float): Learning rate.
        gamma (float): Learning rate decay factor.
        num_iterations (int): Number of training iterations (epochs).
        batch_size (int): Batch size for DataLoader.
        dropout (float): Dropout rate.
        verbosity (int): Verbosity level.

    Returns:
        FitData: Object containing predictions and standard deviations for training and test data.
    """
    epochs = num_iterations * batch_size // X_train.shape[0]
    model = train_nn(X_train, Y_train[:,0].squeeze(), hidden_dims=hidden_dims, epochs=epochs, lr=lr,
                     decay=decay, verbosity=verbosity, gamma=gamma, batch_size=batch_size, dropout=dropout)

    # Make predictions
    model.eval()
    with torch.no_grad():
        train_mean = model(X_train).squeeze().cpu()
    train_std = torch.zeros_like(train_mean)

    if X_test is None:
        Yhat_test_mean = None
        Yhat_test_std = None
    else:
        with torch.no_grad():
            Yhat_test_mean = torch.cat([model(X_test), Y_test[:,1:]],dim=1).cpu()
        Yhat_test_std = torch.zeros_like(Yhat_test_mean)
        X_test = X_test.cpu()
        Y_test = Y_test.cpu()

    # Extract posterior predictive means and standard deviations for test data
    fit = FitData(X_train.cpu(), Y_train.squeeze().cpu(), train_mean, train_std, X_test, Y_test, Yhat_test_mean, Yhat_test_std)
    return fit
