import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from sklearn.model_selection import StratifiedGroupKFold
import pandas

from tqdm import tqdm
import pdb, random
import numpy as np

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
import pandas
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

#def train_one_fold(model, criterion, samevid_loss, samepat_loss, optimizer, data_loader, device) 

#def evaluate(model, data_loader, device)

def main(args):
    # Loading data
    whole_dataset = [] # 讀 dataset
    # Creating dataloaders
    data = pandas.read_csv("../labelled-cropped.csv")
    labels = data["label"].tolist() # 直接 assign label array: labels =  [0]*n + [1]*m 之類的
    patients = data["ulabel_pat"].tolist() # patient id 同上: patients = [0]*a + [1]*b + ...
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True)

    
    # start training
    print("Start training")
    allpred = []        # 最佳 val F1 的預測結果
    alllabel = []       # 最佳 val F1 的 label
    for fold, (train_ids, test_ids) in enumerate(sgkf.split(whole_dataset, labels, patients)):
        # create model (model = ...; optimizer = ...; ...)
        # Load fold
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)  # 挑出這次要用的
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)    # 挑出這次要用的
        train_loader = torch.utils.data.DataLoader(whole_dataset, sampler=train_subsampler, 
                batch_size=args.batch_size, num_workers=args.workers, drop_last=True) # 實際 load
        test_loader = torch.utils.data.DataLoader(whole_dataset, sampler=test_subsampler,
                batch_size=args.batch_size, num_workers=args.workers, drop_last=True) # 實際 load
        tmppred = []
        tmplabel = []
        for epoch in range(args.start_epoch, args.epochs):
            # train one fold: 等同原本的 epoch
            pred,label = train_one_fold(model, criterion, samevid_loss, samepat_loss, optimizer, train_loader, device)
            pred,label = evaluate(model, test_loader, device)
            # 算 val F1，如果 >= 前面的 epoch 就把預測結果 & label存到 tmp
        # 把 tmp 的串到 all
    
    # 用 all 的算 F1 score, acc 等

