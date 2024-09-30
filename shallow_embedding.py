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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import from_networkx

# url = "https://snap.stanford.edu/data/flickrEdges.txt.gz"
# filename = wget.download(url)

filename = "flickrEdges.txt.gz"

data = pd.read_csv(filename, nrows=5000, comment='#', compression='gzip', header=None, sep=" ")
data.columns = ['source', 'target']
G = nx.from_pandas_edgelist(data, 'source', 'target')

data = from_networkx(G)

print(data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
data = data.to(device)

model = Node2Vec(
    data.edge_index, embedding_dim=128, walk_length=20,
    context_size=10, walks_per_node=10, num_negative_samples=1,
    p=1, q=1, sparse=True
).to(device)

loader = model.loader(batch_size=128, shuffle=True)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

def train_node2vec():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

for epoch in range(1, 101):
    loss = train_node2vec()
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

embedding = model.embedding.weight.data.cpu().numpy()

#Scatter Plot using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(embedding)

plt.figure(figsize=(8, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=2)
plt.title('Node2Vec Embedding Visualized with PCA')
plt.savefig('shallow_node2vec_pca.png')
plt.show()

#t-SNE Visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(embedding)

plt.figure(figsize=(8, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=2)
plt.title('Node2Vec Embedding Visualized with t-SNE')
plt.savefig('shallow_node2vec_tsne.png')
plt.show()

#UMAP Visualization
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(embedding)

plt.figure(figsize=(8, 8))
plt.scatter(X_umap[:, 0], X_umap[:, 1], s=2)
plt.title('Node2Vec Embedding Visualized with UMAP')
plt.savefig('shallow_node2vec_umap.png')
plt.show()

# Display statistics
mean_embedding = np.mean(embedding, axis=0)
std_embedding = np.std(embedding, axis=0)
print('Mean of embeddings:', mean_embedding)
print('Standard deviation of embeddings:', std_embedding)