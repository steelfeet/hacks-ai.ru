"""
https://www.kaggle.com/sharansmenon/herbarium-pytorch

"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, sys
os.system('cls' if os.name == 'nt' else 'clear')
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

import json
import pathlib
from tqdm import tqdm

import time
import datetime

# загружам метки в словарь
labels = {}
import csv
with open(os.path.join(BASE_DIR, "train.csv"), "r") as file_name:
    reader = csv.reader(file_name)
    for row in reader: # each row is a list
        date = str(row[1])
        action = str(row[3]).split(".")
        labels[abs(hash(date))] = action[0]


DATA_DIR = pathlib.Path("D:\\", "_hack", "2021-22", "d-hacks-ai.ru", "758261-Irkutsk")
TRAIN_DIR = pathlib.Path(DATA_DIR, "train")
OUT_DIR = pathlib.Path(TRAIN_DIR, "imgs")

mpg_file = str(pathlib.Path(TRAIN_DIR, "train1.avi"))
start_date = "2022-05-24 08:08:33"
start_timestamp = int(time.mktime(datetime.datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S").timetuple())) + 3*3600 # часовой пояс?
# длительность 
end_time = 3 *3600 + 39 * 60 + 54


# Декодируем видеофайл в изображения посекундно
import ffmpeg
print()
print("Decode mpg to img:")
for t in tqdm(range(0, end_time)):
    
    ts = start_timestamp + t
    # переводим timestamp текущей секунды в читаемое имя
    out_filename = str(datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d_%H_%M_%S')) + ".jpg"
    # так, как в train
    out_date = str(datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))
    label = "0"
    try:
        label = labels[abs(hash(out_date))]
    except:
        label = "0"

    try:
        os.mkdir(pathlib.Path(OUT_DIR, label))
    except:
        pass

    out_file = str(pathlib.Path(OUT_DIR, label, out_filename))
    stream = ffmpeg.input(mpg_file, ss=t)
    stream = ffmpeg.filter(stream, 'scale', 800, -1)
    stream = ffmpeg.output(stream, out_file, vframes=1, loglevel="error")
    ffmpeg.run(stream)



