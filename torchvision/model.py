import torch
import torchvision
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader
import videotransforms
import numpy as np
from charades_dataset import Charades as Dataset


# Assuming you have your own dataset class for train and validation
# Replace MyDatasetTrain and MyDatasetVal with your dataset classes
init_lr=0.1
max_steps=64e3
mode='rgb'
root='tmp'
train_split='pytorch-i3d-master/charades.json'
batch_size=1
save_model=''
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

# Define the r2plus1d_18 model
model = torchvision.models.video.r2plus1d_18(pretrained=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# Train the model
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    accuracy = total_correct / total_samples
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

# Save the trained model if needed
torch.save(model.state_dict(), 'r2plus1d_18_model.pth')
