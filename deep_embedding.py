# pip install umap-learn
# pip uninstall torch-scatter torch-sparse torch-geometric torch-cluster  --y
# pip install torch-scatter -f https://data.pyg.org/whl/torch-{torch.__version__}.html
# pip install torch-sparse -f https://data.pyg.org/whl/torch-{torch.__version__}.html
# pip install torch-cluster -f https://data.pyg.org/whl/torch-{torch.__version__}.html
# pip install git+https://github.com/pyg-team/pytorch_geometric.git

import wget
import torch
import numpy as np
import pandas as pd
import networkx as nx
import umap.umap_ as umap
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
from torch_geometric.utils import train_test_split_edges, negative_sampling

# url = "https://snap.stanford.edu/data/flickrEdges.txt.gz"
# filename = wget.download(url)

filename = "flickrEdges.txt.gz"

data = pd.read_csv(filename, nrows=5000, comment='#', compression='gzip', header=None, sep=" ")
data.columns = ['source', 'target']
G = nx.from_pandas_edgelist(data, 'source', 'target')

data = from_networkx(G)

print(data)

data.train_mask = data.val_mask = data.test_mask = None
data = train_test_split_edges(data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
data = data.to(device)

class GNNModel(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super(GNNModel, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, embedding_dim)
        self.conv1 = GCNConv(embedding_dim, 64)
        self.conv2 = GCNConv(64, 32)

    def encode(self, edge_index):
        x = self.embedding.weight
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

    def decode(self, z, edge_index):
        return torch.sigmoid((z[edge_index[0]] * z[edge_index[1]]).sum(dim=1))

model = GNNModel(num_nodes=data.num_nodes, embedding_dim=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def compute_loss(z, pos_edge_index, neg_edge_index):
    pos_score = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
    neg_score = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)
    pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-15).mean()
    neg_loss = -torch.log(1 - torch.sigmoid(neg_score) + 1e-15).mean()
    return pos_loss + neg_loss

def train_gnn():
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.train_pos_edge_index)
    
    # Generate negative edges during training
    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1)
    )
    
    loss = compute_loss(z, data.train_pos_edge_index, neg_edge_index)
    loss.backward()
    optimizer.step()
    return loss.item()

for epoch in range(1, 101):
    loss = train_gnn()
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

# Get the embeddings
model.eval()
with torch.no_grad():
    z = model.encode(data.train_pos_edge_index)
    embeddings_gnn = z.cpu().numpy()

embeddings_gnn = z.cpu().numpy()

#Scatter Plot using PCA
pca = PCA(n_components=2)
z_pca = pca.fit_transform(embeddings_gnn)

plt.figure(figsize=(8, 8))
plt.scatter(z_pca[:, 0], z_pca[:, 1], s=2)
plt.title('GNN Embedding Visualized with PCA')
plt.savefig('gnn_pca.png')
plt.show()

#t-SNE Visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
z_tsne = tsne.fit_transform(embeddings_gnn)

plt.figure(figsize=(8, 8))
plt.scatter(z_tsne[:, 0], z_tsne[:, 1], s=2)
plt.title('GNN Embedding Visualized with t-SNE')
plt.savefig('gnn_tsne.png')
plt.show()

#UMAP Visualization
reducer = umap.UMAP(n_components=2, random_state=42)
z_umap = reducer.fit_transform(embeddings_gnn)

plt.figure(figsize=(8, 8))
plt.scatter(z_umap[:, 0], z_umap[:, 1], s=2)
plt.title('GNN Embedding Visualized with UMAP')
plt.savefig('gnn_umap.png')
plt.show()

# Display statistics
mean_z = np.mean(embeddings_gnn, axis=0)
std_z = np.std(embeddings_gnn, axis=0)
print('Mean of embeddings:', mean_z)
print('Standard deviation of embeddings:', std_z)

# Display model structure
print(model)