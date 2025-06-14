import torch
torch_version = torch.__version__.split('+')[0]

# Import necessary libraries from PyTorch Geometric and others
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

# Load the Cora citation dataset with normalized features
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
data = dataset[0]  # The dataset contains a single graph

# Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        # First GCN layer: input to hidden representation
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        # Second GCN layer: hidden to output (class scores)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        # Apply first convolution + ReLU
        x = self.conv1(x, edge_index)
        x = x.relu()
        # Apply second convolution (logits)
        x = self.conv2(x, edge_index)
        return x

# Instantiate the model with n hidden channels
model = GCN(hidden_channels=16)

# Define the optimizer with learning rate and weight decay
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Training loop: forward pass, loss computation, backpropagation, optimizer step
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)  # Forward pass
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights
    return loss.item()

# Train for 200 epochs
for epoch in range(200):
    loss = train()

# Evaluation mode
model.eval()
out = model(data.x, data.edge_index)  # Forward pass
pred = out.argmax(dim=1)  # Predicted class = index with highest score
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()  # Count correct predictions
acc = int(correct) / int(data.test_mask.sum())  # Compute accuracy
print(f'Test Accuracy: {acc:.4f}')

# Reduce the learned node embeddings to 2D using t-SNE for visualization
z = TSNE(n_components=2).fit_transform(out.detach().numpy())

# Replace numeric labels with actual category names
label_names = [
    "Case_Based", "Genetic_Algorithms", "Neural_Networks", 
    "Probabilistic_Methods", "Reinforcement_Learning", 
    "Rule_Learning", "Theory"
]
node_labels_named = [label_names[y.item()] for y in data.y]

# Plot the embeddings colored by category
plt.figure(figsize=(10, 8))
sns.scatterplot(x=z[:, 0], y=z[:, 1], hue=node_labels_named, palette="tab10", legend='full')
plt.title("t-SNE Visualization of GCN Representations (by Category)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.show()
