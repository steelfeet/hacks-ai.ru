"""
https://www.kaggle.com/sharansmenon/herbarium-pytorch

"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split

import os
import json
import pathlib
from tqdm import tqdm


VALID_PART = 0.2
BATCH = 25
EPOCHS = 50
LR = 0.0001
IM_SIZE = 256

NUM_CL = 20

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


#Загружаем изображения
from PIL import Image

Transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    transforms.Resize(IM_SIZE),
    transforms.RandomCrop((IM_SIZE, IM_SIZE))
    ])


class GetData(Dataset):
    def __init__(self, Dir, FNames, Labels, Transform):
        self.dir = Dir
        self.fnames = FNames
        self.transform = Transform
        self.labels = Labels         
        
    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):       
        x = Image.open(self.fnames[index])
        x = x.convert('RGB')
    
        if "train" in self.dir:             
            return self.transform(x), self.labels[index]
        elif "test" in self.dir:            
            return self.transform(x), self.fnames[index]

data_path = str(pathlib.Path("D:\\", "_hack", "2021-22", "d-hacks-ai.ru", "758261-Irkutsk", "train", "2_employee_aug"))


MODEL_FILENAME = "mobilenet-employee.pth"

train_set = torchvision.datasets.ImageFolder(data_path, transform=Transform)
train_size = int((1 - VALID_PART) * len(train_set))
valid_size = len(train_set) - train_size
train_set, valid_set = random_split(train_set,[train_size,valid_size])

trainloader = DataLoader(train_set, batch_size=BATCH, shuffle=True)
validloader = DataLoader(valid_set, batch_size=BATCH, shuffle=True)

print()
print("trainloader shape: ")
print(next(iter(trainloader))[0].shape)


print()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("device: ", device)




model = torchvision.models.mobilenet_v3_large(pretrained=True)
# model = torchvision.models.efficientnet_b7(pretrained=True)

print()
#перенастраиваем модель под наши классы

"""
# efficient
print("prev features: ")
print(model.classifier[1].in_features) 
print(model.classifier[1].out_features)

for param in model.parameters():
    param.requires_grad = False
n_inputs = model.classifier[1].in_features
last_layer = nn.Linear(n_inputs, NUM_CL)
model.classifier[1] = last_layer
# print()
# print(model)
print("new features: ")
print(model.classifier[1].out_features)
"""

# mobilenet
for param in model.parameters():
    param.requires_grad = False

n_inputs = model.classifier[0].in_features
last_layer = nn.Linear(n_inputs, NUM_CL)
model.classifier = last_layer

# если есть - загружаем
MODEL_PATH = pathlib.Path(BASE_DIR, "models", MODEL_FILENAME)
if MODEL_PATH.is_file():
    print("file exist, load")
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

#настройки обучения
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters())

training_history = {'accuracy':[],'loss':[]}
validation_history = {'accuracy':[],'loss':[]}



# Training with Validation
min_valid_loss = np.inf

for e in range(EPOCHS):
    train_acc = 0.0
    train_loss = 0.0
    for data, labels in tqdm(trainloader):
        # Transfer Data to GPU if available
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()
        
        # Clear the gradients
        optimizer.zero_grad()
        # Forward Pass
        target = model(data)
        # Find the Loss
        loss = criterion(target,labels)
        # Calculate gradients
        loss.backward()
        # Update Weights
        optimizer.step()
        # Calculate Loss
        train_loss += loss.item()
        # Calculate Accuracy
        acc = ((target.argmax(dim=1) == labels).float().mean())
        train_acc += acc
    train_acc = train_acc / len(trainloader) * 100
    train_loss = train_loss / len(trainloader)        
    
    valid_acc = 0.0
    valid_loss = 0.0
    model.eval()     # Optional when not using Model Specific layer
    for data, labels in tqdm(validloader):
        # Transfer Data to GPU if available
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()
        
        # Forward Pass
        target = model(data)
        # Find the Loss
        loss = criterion(target,labels)
        # Calculate Loss
        valid_loss += loss.item()
        # Calculate Accuracy
        acc = ((target.argmax(dim=1) == labels).float().mean())
        valid_acc += acc
    valid_acc = valid_acc / len(validloader) * 100
    valid_loss = valid_loss / len(validloader)
    validation_history['accuracy'].append(valid_acc)
    validation_history['loss'].append(valid_loss)

    print(f'Epoch {e+1} | Train Acc: {train_acc:.6f} | Train Loss: {train_loss:.6f} | Valid Acc: {valid_acc:.6f} | Valid Loss: {valid_loss:.6f}')
    
    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        
        # Saving State Dict
        torch.save(model.state_dict(), str(pathlib.Path(BASE_DIR, "models", MODEL_FILENAME)))

    print()

print()
print("success")
print()
