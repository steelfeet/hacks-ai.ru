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

import os, sys
os.system('cls' if os.name == 'nt' else 'clear')
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

import json
import pathlib, datetime
from tqdm import tqdm


DATA_DIR = pathlib.Path("D:\\", "_hack", "2021-22", "d-hacks-ai.ru", "758261-Irkutsk")
TRAIN_DIR = pathlib.Path(DATA_DIR, "test")
# файлы, которые есть в csv
TEST_DIR = pathlib.Path(TRAIN_DIR, "v2")

MODEL_FILENAME_ACTION = "mobilenet-action-2.2.pth"
MODEL_FILENAME_EMPLOYEE = "mobilenet-employee-2.2.pth"
SUBMISSION_FILENAME = "submission-3.csv"

NUM_CL_ACTION = 20
NUM_CL_EMPLOYEE = 12

IM_SIZE = 450


# Формируем Датасет





# Загружаем изображения
from PIL import Image
# подготавливаем изображения
Transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Resize((IM_SIZE, IM_SIZE)),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])



# Загружаем модели
print()
print("Load models")
model_action = torchvision.models.mobilenet_v3_large()

# Перенастраиваем модель под наши классы
for param in model_action.parameters():
    param.requires_grad = False

n_inputs = model_action.classifier[0].in_features
last_layer = nn.Linear(n_inputs, NUM_CL_ACTION)
model_action.classifier = last_layer

# Загружаем веса модели
print()
print("Load state dict:")
model_action.load_state_dict(torch.load(str(pathlib.Path(BASE_DIR, "models", MODEL_FILENAME_ACTION)), map_location=torch.device('cpu')))
model_action.eval()


model_employee = torchvision.models.mobilenet_v3_large()

# Перенастраиваем модель под наши классы
for param in model_employee.parameters():
    param.requires_grad = False

n_inputs = model_employee.classifier[0].in_features
last_layer = nn.Linear(n_inputs, NUM_CL_EMPLOYEE)
model_employee.classifier = last_layer

# Загружаем веса модели
print()
print("Load state dict:")
model_employee.load_state_dict(torch.load(str(pathlib.Path(BASE_DIR, "models", MODEL_FILENAME_EMPLOYEE)), map_location=torch.device('cpu')))
model_employee.eval()



class GetData(Dataset):
    def __init__(self, Dir, FNames, Labels, Transform):
        self.dir = Dir
        self.fnames = FNames
        self.transform = Transform
        self.labels = Labels         
        
    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):       
        image = self.fnames[index]
        image_path = os.path.join(TEST_DIR, image)
        x = Image.open(image_path)
        x = x.convert('RGB')
    
        if "train" in str(self.dir):
            return self.transform(x), self.labels[index]
        elif "test" in str(self.dir):
            return self.transform(x), self.fnames[index]


# Подготавливаем загрузчик иображений
#testset = torchvision.datasets.ImageFolder(TEST_DIR, transform=Transform)
testset = GetData(TEST_DIR, os.listdir(TEST_DIR), None, Transform)

testloader = DataLoader(testset, batch_size=1, shuffle=False)
print()
print("success")
print("testset len:", str(len(testset)))


# Распознаем
print()
print("Start prediction")
preds_action = {}
preds_employee = {}
with torch.no_grad():
    model_action.eval()
    model_employee.eval()
    for image, fname in tqdm(testloader):

        logits = model_action(image)        
        ps = torch.exp(logits)        
        _, top_class = ps.topk(1, dim=1)
        
        for pred in top_class:
            preds_action[fname[0]] = int(pred.item())

        logits = model_employee(image)        
        ps = torch.exp(logits)        
        _, top_class = ps.topk(1, dim=1)
        
        for pred in top_class:
            preds_employee[fname[0]] = int(pred.item())


print(preds_action)

# преобразуем в формат 758261 (Иркутск), для отсутствующих файлов ставим 0;0
out_str = ""
import csv
with open(os.path.join(BASE_DIR, "test.csv"), "r") as file_name:
    reader = csv.reader(file_name)
    for row in reader: # each row is a list
        date = str(row[1])
        
        filename = str(date).replace(" ", "_")
        filename = filename.replace(":", "_") + ".jpg"
        
        try:
            pred_action = int(preds_action[filename])
            pred_employee = int(preds_employee[filename])
            out_str += f"{row[0]},{pred_employee},{pred_action}\n"
        except:
            out_str += f"{row[0]},0,0\n"



file = open(os.path.join(BASE_DIR, SUBMISSION_FILENAME), 'w')
file.write(out_str)
file.close()

print()
print("success")
