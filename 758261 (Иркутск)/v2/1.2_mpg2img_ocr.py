"""
https://www.kaggle.com/sharansmenon/herbarium-pytorch

"""
import cv2

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

from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

DATA_DIR = pathlib.Path("D:\\", "_hack", "2021-22", "d-hacks-ai.ru", "758261-Irkutsk")
TRAIN_DIR = pathlib.Path(DATA_DIR, "train")
# посекундные кадры ffmpeg
OUT_DIR = pathlib.Path(TRAIN_DIR, "imgs")
# файлы, которые есть в csv
REAL_DIR = pathlib.Path(TRAIN_DIR, "v2s1_action")
# REAL_DIR = pathlib.Path(TRAIN_DIR, "v2s1_employee")


# загружам метки в словарь
labels = {}
import csv
with open(os.path.join(BASE_DIR, "train.csv"), "r") as file_name:
    reader = csv.reader(file_name)
    for row in reader: # each row is a list
        date = str(row[1])
        action = str(row[3]).split(".")[0]
        
        filename = str(date).replace(" ", "_")
        filename = filename.replace(":", "_") + ".jpg"
        employee = str(row[2])
        labels[filename] = action
        # labels[filename] = employee

print(labels)


"""

mpg_file = str(pathlib.Path(TRAIN_DIR, "train1.avi"))
start_date = "2022-05-24 08:08:33"
start_timestamp = int(time.mktime(datetime.datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S").timetuple())) + 3*3600 # часовой пояс?
# длительность 
end_time = 3 *3600 + 39 * 60 + 54

top_left_x = 0
top_left_y = 30
bottom_right_x = 250
bottom_right_y = 60


"""
# второй файл
mpg_file = str(pathlib.Path(TRAIN_DIR, "train2.avi"))
start_date = "2022-05-26 08:02:47"
start_timestamp = int(time.mktime(datetime.datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S").timetuple())) + 3*3600 # часовой пояс?
# длительность 
end_time = 5 *3600 + 33 * 60 + 54

top_left_x = 0
top_left_y = 20
bottom_right_x = 250
bottom_right_y = 50


# Декодируем видеофайл в изображения посекундно
import ffmpeg
print()
print("Decode mpg to img:")
# Очищаем папку перед использованием
for f in os.listdir(OUT_DIR):
    os.remove(os.path.join(OUT_DIR, f))

# количество "прдсказанных" значений времени
predicted_n = 0
for t in tqdm(range(0, end_time)):
    
    ts = start_timestamp + t
    # переводим timestamp текущей секунды в читаемое имя
    out_filename = str(datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d_%H_%M_%S')) + ".jpg"
    # так, как в train
    out_date = str(datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))
    label = "None"

    out_file = str(pathlib.Path(OUT_DIR, out_filename))
    stream = ffmpeg.input(mpg_file, ss=t)
    stream = ffmpeg.filter(stream, 'scale', 800, -1)
    stream = ffmpeg.output(stream, out_file, vframes=1, loglevel="error")
    ffmpeg.run(stream)


    # подбор координат расположения даты и времени
    """
    image = cv2.imread(str(out_file))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    top_left_x = 0
    top_left_y = 30
    bottom_right_x = 250
    bottom_right_y = 60

    cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (255, 255, 0), 4)
    path = str(pathlib.Path(OUT_DIR, str(ts)+".jpg")) 
    cv2.imwrite(path, image)
    """

   
    # вырезаем прямоугольник с датой
    image = Image.open(out_file)
    image_ocr = image.crop((top_left_x, top_left_y, bottom_right_x, bottom_right_y))
    
    # для проверки
    path = str(pathlib.Path(OUT_DIR, str(ts)+".jpg")) 
    image_ocr.save(path, quality=95)

    #распознаем
    is_predicted = False
    ocr_date = str(pytesseract.image_to_string(image_ocr, lang='eng')).split()
    try:
        real_years = ocr_date[0]
        m, d, y = real_years.split("-")
        real_time = ocr_date[2]
        h, min, sec = real_time.split(":")


    except:
        # последний шаг, догадываемся по счету кадра
        try:
            is_predicted = True
            sec = str(sec).replace(",", "")
            sec = int(sec) + 1
            if sec == 60:
                min = int(min) + 1
                sec = 0
            min = int(min)
            if min > 59:
                h = int(h) + 1
                min = 0

            if sec < 10:
                sec = "0" + str(sec)
            if min < 10:
                min = "0" + str(min)
            if int(h) < 10:
                h = "0" + str(h)

            real_time = f"{h}:{min}:{sec}"
        except:
            pass

    real_years = f"{y}-{m}-{d}"
    real_date = f"{real_years} {real_time}"
    real_file = f"{real_years}_{real_time}.jpg"
    real_file = real_file.replace(":", "_")
    
    try:
        label = labels[real_file]
        del labels[real_file]
        if is_predicted:
            predicted_n = predicted_n + 1
    except:
        label = "None"
    


    try:
        os.mkdir(pathlib.Path(REAL_DIR, label))
    except:
        pass

    # вырезаем квадрат слева и перезаписываем
    width, height = image.size
    image_rec = image.crop((0, 0, height, height))
    
    real_path = str(pathlib.Path(REAL_DIR, label, real_file))
    try:
        image_rec.save(real_path, quality=95)
    except:
        pass


print(f"Не обнаружено {len(labels)} элементов")


