import torch
import deepchem as dc
from rdkit import Chem
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader  
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error


#load data
rxrx19b_metadata = pd.read_csv('data/rxrx19b_metadata.csv', index_col=0)
rxrx19b_embeddings = pd.read_csv('data/rxrx19b_embeddings.csv', index_col=0)

X_df = rxrx19b_embeddings.copy()
X_df.index.name = 'site_id'
X_df.reset_index(inplace=True)

y_df = rxrx19b_metadata[['SMILES']].copy()
y_df.index.name = 'site_id'
y_df.reset_index(inplace=True)

data = pd.merge(X_df, y_df, on='site_id')
print(f"Total samples after merge: {len(data)}")

data = data.dropna(subset=['SMILES'])
print(f"Total samples after dropping NaNs: {len(data)}")

smiles_list = data['SMILES'].tolist()
featurizer = dc.feat.Mol2VecFingerprint()
features = featurizer.featurize(smiles_list)

data['mol2vec_features'] = features

expected_mol2vec_length = 300
mol2vec_lengths = [len(f) for f in data['mol2vec_features']]
assert all(length == expected_mol2vec_length for length in mol2vec_lengths), "Mol2Vec features have inconsistent lengths."

expected_embedding_length = data.drop(['site_id', 'SMILES', 'mol2vec_features'], axis=1).shape[1]
print(f"Expected embedding length: {expected_embedding_length}")

X_features = data.drop(['site_id', 'SMILES', 'mol2vec_features'], axis=1).values
Y_features = np.vstack(data['mol2vec_features'].values)


X_tensor = torch.from_numpy(X_features).float()
Y_tensor = torch.from_numpy(Y_features).float()

print(f"X_tensor shape: {X_tensor.shape}")
print(f"Y_tensor shape: {Y_tensor.shape}")

X_train, X_temp, Y_train, Y_temp = train_test_split(X_tensor, Y_tensor, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

class EmbeddingDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_dataset = EmbeddingDataset(X_train, Y_train)
val_dataset = EmbeddingDataset(X_val, Y_val)
test_dataset = EmbeddingDataset(X_test, Y_test)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

class ImageToMol2VecModel(nn.Module):
    def __init__(self, input_size=128, output_size=300):
        super(ImageToMol2VecModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_size)
        )
        
    def forward(self, x):
        return self.model(x)

model = ImageToMol2VecModel(input_size=expected_embedding_length, output_size=expected_mol2vec_length)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = model.to(device)

num_epochs = 50
best_val_loss = float('inf')
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_Y in train_loader:
        batch_X = batch_X.to(device)
        batch_Y = batch_Y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_X.size(0)
    epoch_train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_X, val_Y in val_loader:
            val_X = val_X.to(device)
            val_Y = val_Y.to(device)
            val_outputs = model(val_X)
            loss = criterion(val_outputs, val_Y)
            val_loss += loss.item() * val_X.size(0)
    epoch_val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_train_loss:.6f}, Validation Loss: {epoch_val_loss:.6f}")
    
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), 'best_model.pth')

model.load_state_dict(torch.load('best_model.pth'))
model = model.to(device)

model.eval()
test_loss = 0.0
cosine_sim = 0.0
num_batches = 0

all_test_outputs = []
all_test_targets = []

with torch.no_grad():
    for test_X, test_Y in test_loader:
        test_X = test_X.to(device)
        test_Y = test_Y.to(device)
        test_outputs = model(test_X)
        loss = criterion(test_outputs, test_Y)
        test_loss += loss.item() * test_X.size(0)
        
        test_outputs_cpu = test_outputs.cpu().numpy()
        test_Y_cpu = test_Y.cpu().numpy()
        
        all_test_outputs.append(test_outputs_cpu)
        all_test_targets.append(test_Y_cpu)
        
        cosine_sim += cosine_similarity(test_outputs_cpu, test_Y_cpu).mean()
        num_batches += 1

test_loss /= len(test_loader.dataset)
cosine_sim /= num_batches

print(f"Test Loss: {test_loss:.6f}")
print(f"Average Cosine Similarity: {cosine_sim:.6f}")

all_test_outputs = np.vstack(all_test_outputs)
all_test_targets = np.vstack(all_test_targets)

pca = PCA(n_components=2)
pca.fit(np.concatenate((all_test_outputs, all_test_targets), axis=0))

outputs_2d = pca.transform(all_test_outputs)
targets_2d = pca.transform(all_test_targets)

plt.figure(figsize=(10, 6))
plt.scatter(targets_2d[:, 0], targets_2d[:, 1], color='blue', alpha=0.5, label='Actual Mol2Vec Embeddings')
plt.scatter(outputs_2d[:, 0], outputs_2d[:, 1], color='red', alpha=0.5, label='Predicted Mol2Vec Embeddings')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of Actual vs. Predicted Mol2Vec Embeddings')
plt.legend()
plt.show()

mae = mean_absolute_error(all_test_targets, all_test_outputs)
print(f"Mean Absolute Error (MAE) on Test Set: {mae:.6f}")

errors = np.abs(all_test_targets - all_test_outputs)

plt.figure(figsize=(10, 6))
plt.hist(errors.flatten(), bins=50, color='purple', alpha=0.7)
plt.xlabel('Absolute Error')
plt.ylabel('Frequency')
plt.title('Histogram of Absolute Errors')
plt.show()

num_samples = 5
indices = np.random.choice(range(all_test_targets.shape[0]), num_samples, replace=False)

for idx in indices:
    actual_embedding = all_test_targets[idx]
    predicted_embedding = all_test_outputs[idx]
    cosine_sim_sample = cosine_similarity([actual_embedding], [predicted_embedding])[0][0]
    
    print(f"Sample {idx+1}:")
    print(f"Cosine Similarity: {cosine_sim_sample:.6f}")
    print(f"Actual Mol2Vec Embedding (first 5 values): {actual_embedding[:5]}")
    print(f"Predicted Mol2Vec Embedding (first 5 values): {predicted_embedding[:5]}")
    print("-" * 50)
