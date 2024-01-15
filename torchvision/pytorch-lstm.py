import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from talos import Scan
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pdb, random
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd
from sklearn.metrics import f1_score

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, num_layers=1, bidirectional=False):
        super(LSTMClassifier, self).__init__()

        self.lstm1 = nn.LSTM(input_size, hidden_size1, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        
        self.fc1 = nn.Linear(hidden_size2, 64)
        self.fc2 = nn.Linear(64, 64)
        
        self.output_layer = nn.Linear(64, output_size)
        self.softmax = nn.Softmax(dim=1)
        
        self.dropout = nn.Dropout(p=0.5)
        self.batch_norm = nn.BatchNorm1d(64)
        
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)
        nn.init.kaiming_uniform_(self.output_layer.weight)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        
        x = self.fc1(x[:, -1, :])
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = nn.functional.relu(x)
        
        x = self.fc2(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = nn.functional.relu(x)
        
        x = self.output_layer(x)
        x = self.softmax(x)
        
        return x

input_size = 2048
output_size = 2

hidden_size1 = 128
hidden_size2 = 64
num_layers = 1
bidirectional = False
learning_rate = 0.001

labels = np.loadtxt('/home/vegetabledogkingm/Desktop/model test/pytorch-i3d-master/list.txt')

data = []
loaded_data = np.load('/home/vegetabledogkingm/Desktop/model test/dtaset.npy')

X = np.array(loaded_data)
y = labels
X = torch.from_numpy(X)
y = torch.from_numpy(y)
y = y.type(torch.LongTensor)

patients=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 12, 12, 12, 13, 13, 14, 0, 1, 1, 1, 2, 3, 5, 6, 9, 10, 13, 15]
    
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True)
accuracies = []
f1_scores = []

for fold, (train_index, val_index) in enumerate(sgkf.split(X, y, patients)):
    print(f"Fold {fold + 1}")
    model = LSTMClassifier(input_size, hidden_size1, hidden_size2, output_size, num_layers, bidirectional)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    
    train_len = len(train_index)
    val_len = len(val_index)
    # Convert to DataLoader
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_index)  # 挑出這次要用的
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_index)    # 挑出這次要用的
    train_loader = torch.utils.data.DataLoader(X, sampler=train_subsampler, 
            batch_size=train_len, num_workers=50, drop_last=True) # 實際 load
    val_loader = torch.utils.data.DataLoader(X, sampler=val_subsampler,
            batch_size=val_len, num_workers=50, drop_last=True) # 實際 load

    # Training loop
    labels = y[train_index]
    for epoch in range(50):
        model.train()
        for inputs in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    model.eval()
    correct = 0
    total = 0
    all_predicted = []
    all_labels = []
    labels = y[val_index]
    with torch.no_grad():
        for inputs in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    accuracy = correct / total
    f1 = f1_score(all_labels, all_predicted, average='weighted')
    accuracies.append(accuracy)
    f1_scores.append(f1)

print("Average Accuracy:", np.mean(accuracies))
print("Average F1 Score:", np.mean(f1_scores))