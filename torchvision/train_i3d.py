import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str)
parser.add_argument('-root', type=str)

args = parser.parse_args()


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms


import numpy as np

from charades_dataset import Charades as Dataset


def run(init_lr=0.1, max_steps=64e3, mode='rgb', root='tmp', train_split='pytorch-i3d-master/charades.json', batch_size=1, save_model=''):
    # setup dataset
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(),
    ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(train_split, 'training', root, mode, train_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    val_dataset = Dataset(train_split, 'testing', root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)    

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}

    
    # setup the model
    model = torchvision.models.video.r2plus1d_18(pretrained=True)
    lr = init_lr
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    num_steps_per_update = 4 # accum gradient
    steps = 0
    i = 1
    torchvision_dataset = []
    # train it

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train(True)
        else:
            model.train(False)  # Set model to evaluate mode

        num_iter = 0
        optimizer.zero_grad()
        
        # Iterate over data.
        for data in dataloaders[phase]:
            num_iter += 1
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            per_frame_logits = model(inputs)
            array = per_frame_logits.data.cpu().numpy()
            torchvision_dataset.append(array)

            # # wrap them in Variable
            # inputs = Variable(inputs.cuda())
            # t = inputs.size(2)
            # labels = Variable(labels.cuda())

            # for data in dataloaders[phase]:
            #     per_frame_logits = model(inputs)
            #     array = per_frame_logits.data.cpu().numpy()
            #     torchvision_dataset.append(array)
    
    print(torchvision_dataset)
    np.save('dtaset', torchvision_dataset)
                    
if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode, root=args.root, save_model=args.save_model)
