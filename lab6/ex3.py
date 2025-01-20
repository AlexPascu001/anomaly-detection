import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.metrics import roc_auc_score
import scipy.io
import scipy.sparse as sp
import numpy as np

import matplotlib
matplotlib.use('TkAgg')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load the dataset
data = scipy.io.loadmat('ACM.mat')
attributes = torch.tensor(data["Attributes"].todense(), dtype=torch.float).to(device)
labels = torch.tensor(data["Label"].flatten(), dtype=torch.long).to(device)

# Convert adjacency matrix to edge index format
adj_matrix = data["Network"]
edge_index, _ = from_scipy_sparse_matrix(sp.csr_matrix(adj_matrix))
edge_index = edge_index.to(device)

class GraphEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim):
        super(GraphEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, latent_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.relu(x)

class AttributeDecoder(torch.nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(AttributeDecoder, self).__init__()
        self.conv1 = GCNConv(latent_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, z, edge_index):
        z = self.conv1(z, edge_index)
        z = F.relu(z)
        z = self.conv2(z, edge_index)
        return F.relu(z)

class StructureDecoder(torch.nn.Module):
    def __init__(self, latent_dim):
        super(StructureDecoder, self).__init__()
        self.conv = GCNConv(latent_dim, latent_dim)

    def forward(self, z, edge_index):
        z = self.conv(z, edge_index)
        z = F.relu(z)
        return torch.mm(z, z.t())  # Z × Zᵀ

class GraphAutoencoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim, output_dim):
        super(GraphAutoencoder, self).__init__()
        self.encoder = GraphEncoder(in_channels, hidden_dim, latent_dim)
        self.attr_decoder = AttributeDecoder(latent_dim, hidden_dim, output_dim)
        self.struct_decoder = StructureDecoder(latent_dim)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        attr_recon = self.attr_decoder(z, edge_index)
        struct_recon = self.struct_decoder(z, edge_index)
        return attr_recon, struct_recon, z

input_dim = attributes.shape[1]
hidden_dim = 128
latent_dim = 64
output_dim = input_dim  # Reconstruction target

model = GraphAutoencoder(input_dim, hidden_dim, latent_dim, output_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

adj_matrix = torch.tensor(data["Network"].toarray(), dtype=torch.float).to(device)
alpha = 0.8

def custom_loss(attr_recon, struct_recon, attributes, adj_matrix, alpha=0.8):
    """
    Custom loss function combining attribute and adjacency reconstruction losses.
    """
    attr_loss = F.mse_loss(attr_recon, attributes)  # || X - X_hat ||_F^2
    struct_loss = F.mse_loss(struct_recon, adj_matrix)  # || A - A_hat ||_F^2
    return alpha * attr_loss + (1 - alpha) * struct_loss


# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.004)

epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    attr_recon, struct_recon, _ = model(attributes, edge_index)

    loss = custom_loss(attr_recon, struct_recon, attributes, adj_matrix, alpha)

    # Backward pass
    loss.backward()
    optimizer.step()

    # Compute AUC Score every 5 epochs
    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            _, struct_recon, _ = model(attributes, edge_index)

        # Compute reconstruction error for adjacency matrix
        struct_error = torch.norm(adj_matrix - struct_recon, dim=1).cpu().numpy()

        # Assume binary labels (1 for anomalies, 0 for normal nodes)
        true_labels = labels.cpu().numpy()

        # Compute ROC AUC score
        roc_auc = roc_auc_score(true_labels, struct_error)

        print(f"Epoch {epoch}: Loss = {loss.item():.4f}, ROC AUC = {roc_auc:.4f}")

model.eval()
with torch.no_grad():
    _, struct_recon, z = model(attributes, edge_index)

reconstruction_error = torch.norm(adj_matrix - struct_recon, dim=1)

# Rank nodes by anomaly score
anomaly_scores = reconstruction_error.cpu().numpy()
anomalous_nodes = np.argsort(-anomaly_scores)  # Higher error -> more anomalous

print("Top anomalous nodes:", anomalous_nodes[:10])
