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
os.system('cls' if os.name == 'nt' else 'clear')

import json
import pathlib
from tqdm import tqdm


DATA_DIR = pathlib.Path("D:\\", "_hack", "2021-22", "isshack", "_iss_1")
TEST_DIR = pathlib.Path(DATA_DIR, "test-pitch")
MODEL_FILENAME = "v6.1-iss-mobilenet.pth" # + наш датасет
# MODEL_FILENAME = "v6-iss-mobilenet_v3_large.pth" # ТОЛЬКО! разметка fight_train.mp4
SUBMISSION_FILENAME = "iss-submission-pitch.txt"
mpg_file = str(pathlib.Path(DATA_DIR, "fight_test.mp4"))
mpg_time = 77  # в минутах

#для показа на чекпойнте 3
iss_start_time = 0
iss_end_time = 117



NUM_CL = 2
IM_SIZE = 600



# Декодируем видеофайл в изображения посекундно
import ffmpeg
print()
print("Decode mpg to img:")
if not os.path.isdir(TEST_DIR):
    os.mkdir(TEST_DIR)

"""
for t in tqdm(range(iss_start_time, iss_end_time)):
    out_filename = "%d.jpg" % t
    out_file = str(pathlib.Path(TEST_DIR, out_filename))
    stream = ffmpeg.input(mpg_file, ss=t)
    stream = ffmpeg.filter(stream, 'scale', 800, -1)
    stream = ffmpeg.output(stream, out_file, vframes=1, loglevel="error")
    ffmpeg.run(stream)
"""


from datetime import datetime
eval_start_time = datetime.now().timestamp()


# Формируем Датасет
print()
print("Load X_Test")
X_Test = []
for i in tqdm(range(iss_start_time, iss_end_time)):
    out_filename = "%d.jpg" % i
    out_file = str(pathlib.Path(TEST_DIR, out_filename))
    X_Test.append(out_file)
print("success")


# Загружаем изображения
from PIL import Image
# подготавливаем изображения
Transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Resize((IM_SIZE, IM_SIZE)),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


# Класс загрузчика изображений
class GetData(Dataset):
    def __init__(self, Dir, FNames, Labels, Transform):
        self.dir = str(Dir)
        self.fnames = FNames
        self.transform = Transform
        self.labels = Labels         
        
    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):       
        image_filename = self.fnames[index]
        x = Image.open(image_filename)
        x = x.convert('RGB')
    
        if "train" in self.dir:             
            return self.transform(x), self.labels[index]
        elif "test" in self.dir:            
            return self.transform(x), self.fnames[index]



# Загружаем модель
print()
print("Load model")
model = torchvision.models.mobilenet_v3_large()
# print(model)
# print()


# Перенастраиваем модель под наши классы
print()
print("prev features: ")
#print(model.classifier[0].in_features) 
print(model.classifier[0].out_features)

for param in model.parameters():
    param.requires_grad = False

n_inputs = model.classifier[0].in_features
last_layer = nn.Linear(n_inputs, NUM_CL)
model.classifier = last_layer
if torch.cuda.is_available():
    model.cuda()

print("new features: ")
print(model.classifier.out_features)



         
# Загружаем веса модели
print()
print("Load state dict:")
model.load_state_dict(torch.load(str(pathlib.Path("models", MODEL_FILENAME))))
model.eval()


# Подготавливаем загрузчик иображений
testset = GetData(TEST_DIR, X_Test, None, Transform)
testloader = DataLoader(testset, batch_size=1, shuffle=False)
print()
print("success")
print("testset len:", str(len(testset)))


# Распознаем
print()
print("Start prediction")
s_ls = {}
i = 0
with torch.no_grad():
    model.eval()
    for image, fname in tqdm(testloader):
        logits = model(image)        
        ps = torch.exp(logits)        
        _, top_class = ps.topk(1, dim=1)
        
        for pred in top_class:
            start_time = int(fname[0].split('\\')[-1][:-4])
            s_ls[start_time] = int(pred.item())

        i += 1



# преобразуем в формат ISS (обьединяем рядом стоящие секунды)
iss_data = []
current_label = 1
start_time = 0
for i in range(iss_start_time, iss_end_time):
    if (s_ls[i] != current_label):
        if (start_time == 0):
            iss_data.append([start_time, i-1, current_label])
        else:
            iss_data.append([start_time-1, i-1, current_label])
        current_label = s_ls[i]
        start_time = i


iss_data.append([start_time-1, i+1, current_label])



# сохраняем
sub = pd.DataFrame.from_records(iss_data)
sub.to_csv(SUBMISSION_FILENAME, index=False, sep=" ", header=False)


#время окончания работы распознавания
eval_end_time = datetime.now().timestamp()
eval_time = eval_end_time - eval_start_time
print()
print("Время работы:", str(eval_time))


#скорость в кадрах в сек
frames_n = iss_end_time - iss_start_time
print("Скорость: ", str(frames_n / eval_time), " к/сек")


print()
print("success")
