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


TRAIN_DIR = pathlib.Path("D:\\", "_hack", "2021-22", "isshack", "_iss_1", "trainPmy")
MODEL_FILENAME = "v6.1-iss-mobilenet.pth"
VALID_PART = 0.1
BATCH = 128
EPOCHS = 50
LR = 0.01
IM_SIZE = 600



#формируем Датасет
X_Train = []
Y_Train = []
print("label names: ")
NUM_CL = 0
for label_name in sorted(TRAIN_DIR.glob('*/')):
    if label_name.is_dir():
        class_dir = pathlib.Path(TRAIN_DIR, label_name.name) #директория класса
        all_image_paths = list(class_dir.glob('*/'))
        for path in all_image_paths:
            X_Train.append(str(path))
            Y_Train.append(NUM_CL)
        print(label_name.name, " - ", str(NUM_CL))
        NUM_CL = NUM_CL + 1

print("load image success, X_Train count:", str(len(X_Train)))



#Загружаем изображения
from PIL import Image

Transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Resize((IM_SIZE, IM_SIZE)),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


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


train_set = GetData(str(TRAIN_DIR), X_Train, Y_Train, Transform)
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
print()
# print(model)
# print()
print("prev features: ")
print(model.classifier[0].in_features) 
print(model.classifier[0].out_features)


#перенастраиваем модель под наши классы
for param in model.parameters():
    param.requires_grad = False

n_inputs = model.classifier[0].in_features
last_layer = nn.Linear(n_inputs, NUM_CL)
model.classifier = last_layer
if torch.cuda.is_available():
    model.cuda()

# print()
# print(model)
print("new features: ")
print(model.classifier.out_features)


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
 
    print(f'Epoch {e+1} | Train Acc: {train_acc:.6f} | Train Loss: {train_loss:.6f} | Valid Acc: {valid_acc:.6f} | Valid Loss: {valid_loss:.6f}')
     
    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
         
        # Saving State Dict
        torch.save(model.state_dict(), str(pathlib.Path("models", MODEL_FILENAME)))

    print()

print()
print("success")
