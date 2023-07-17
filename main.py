from torch_geometric.datasets import GNNBenchmarkDataset
import torch
from torch_geometric.nn import GATConv
from model import GNNmodel
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

# Load data
data = GNNBenchmarkDataset(root='/data/MNIST', name='MNIST')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define model and optimizer
model = GNNmodel(hidden_channels=16, dataset=data)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

# Use a DataLoader to handle batching of data
loader = DataLoader(dataset=data, batch_size=32, shuffle=True)

def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        x, edge_index, batch_index = batch.x, batch.edge_index, batch.batch
        # Forward pass
        optimizer.zero_grad()
        out = model(x, edge_index)
        out = global_mean_pool(out, batch_index)
        # Compute the loss
        loss = torch.nn.functional.nll_loss(out, batch.y)
        # Backward pass
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        return total_loss / len(loader)

for epoch in range(1, 201):
    loss = train(model, loader, optimizer)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')