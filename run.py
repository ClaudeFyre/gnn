from torch_geometric.datasets import DBP15K
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch
from model import EntityAligner

# Load the dataset
path = '/tmp/DBP15K'
data = DBP15K(path, 'zh_en')

# Look at the attributes
print(data)

# Prepare data for your entity alignment model
data.train_y = data.train_y.to(torch.long)  # Convert train_y to long type
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EntityAligner(data.num_features, data.num_classes).to(device)
data = data.to(device)

# Define an optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

# Define a training function
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_y], data.train_y)
    loss.backward()
    optimizer.step()
    return loss

for epoch in range(1, 201):
    loss = train()
    print('Epoch: {:03d}, Loss: {:.5f}'.format(epoch, loss))
