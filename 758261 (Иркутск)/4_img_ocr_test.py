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
import PIL.ImageOps
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# загружам метки в словарь
labels = []
import csv
with open(os.path.join(BASE_DIR, "test.csv"), "r") as file_name:
    reader = csv.reader(file_name)
    for row in reader: # each row is a list
        date = str(row[1])
        
        filename = str(date).replace(" ", "_")
        filename = filename.replace(":", "_") + ".jpg"
        # labels[filename] = action
        labels.append(filename)

print(labels)

DATA_DIR = pathlib.Path("D:\\", "_hack", "2021-22", "d-hacks-ai.ru", "758261-Irkutsk")
TRAIN_DIR = pathlib.Path(DATA_DIR, "test")
# посекундные кадры ffmpeg
OUT_DIR = pathlib.Path(TRAIN_DIR, "imgs")
# файлы, которые есть в csv
REAL_DIR = pathlib.Path(TRAIN_DIR, "imgs_real")


mpg_file = str(pathlib.Path(TRAIN_DIR, "test.avi"))
start_date = "2022-05-22 08:25:46"
start_timestamp = int(time.mktime(datetime.datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S").timetuple())) + 3*3600 # часовой пояс?
# длительность 
end_time = 3 *3600 + 27 * 60 + 39

top_left_x = 0
top_left_y = 20
bottom_right_x = 250
bottom_right_y = 50


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
"""

"""
# Декодируем видеофайл в изображения посекундно
import ffmpeg
print()
print("Decode mpg to img:")
# Очищаем папку перед использованием
for f in os.listdir(OUT_DIR):
    os.remove(os.path.join(OUT_DIR, f))
"""

# количество "предсказанных" значений времени
predicted_n = 0

for t in tqdm(range(0, end_time)):
    
    ts = start_timestamp + t
    # переводим timestamp текущей секунды в читаемое имя
    out_filename = str(datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d_%H_%M_%S')) + ".jpg"
    # так, как в train
    out_date = str(datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))

    out_file = str(pathlib.Path(OUT_DIR, out_filename))
    
    """
    stream = ffmpeg.input(mpg_file, ss=t)
    stream = ffmpeg.filter(stream, 'scale', 800, -1)
    stream = ffmpeg.output(stream, out_file, vframes=1, loglevel="error")
    ffmpeg.run(stream)
    """


    # подбор координат расположения даты и времени
    """
    image = cv2.imread(str(out_file))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (255, 255, 0), 4)
    path = str(pathlib.Path(OUT_DIR, str(ts)+".jpg")) 
    cv2.imwrite(path, image)
    """

   
    # вырезаем прямоугольник с датой
    image = Image.open(out_file)
    image_ocr = image.crop((top_left_x, top_left_y, bottom_right_x, bottom_right_y))
    

    
    #распознаем
    is_predicted = False
    ocr_date = str(pytesseract.image_to_string(image_ocr, lang='eng')).split()
    try:
        real_years = ocr_date[0]
        m, d, y = real_years.split("-")
        real_time = ocr_date[2]
        h, min, sec = real_time.split(":")

        current_d = int(d)
        current_h = int(h)
        current_min = int(min)
        current_sec = int(sec)
       

    except:
        """
        print()
        print("OCR Error!")
        print(ts)
        print(ocr_date)
        print(f"predicted_n: {predicted_n}")
        """

        # для проверки сохраняем вырезанную дату
        # path = str(pathlib.Path(OUT_DIR, str(ts)+".jpg")) 
        # image_ocr.save(path, quality=95)

        """
        # второй шаг, ночное время
        inverted_image_ocr = PIL.ImageOps.invert(image_ocr)
        ocr_date = str(pytesseract.image_to_string(inverted_image_ocr, lang='eng')).split()
        try:
            real_years = ocr_date[0]
            m, d, y = real_years.split("-")
            real_time = ocr_date[2]
            h, min, sec = real_time.split(":")
        except:
            print("Step II OCR Error!")
            print(ocr_date)
            print()
            path = str(pathlib.Path(OUT_DIR, str(ts)+"-inv.jpg")) 
            inverted_image_ocr.save(path, quality=95)
        """

        # последний шаг, догадываемся по счету кадра
        is_predicted = True
        current_sec = int(current_sec) + 1
        if current_sec > 59:
            current_min = int(current_min) + 1
            current_sec = 0
        if current_min > 59:
            current_h = int(current_h) + 1
            current_min = 0

        d = current_d
        h = current_h
        min = current_min
        sec = current_sec


    real_time = f"{h}:{min}:{sec}"
    real_years = f"2022-05-{d}"
    real_date = f"{real_years} {real_time}"
    real_file = f"{real_years}_{real_time}.jpg"
    real_file = real_file.replace(":", "_")
    

    real_path = str(pathlib.Path(REAL_DIR, real_file))
    if (real_file in labels):
        is_predicted = is_predicted + 1
        image.save(real_path, quality=95)
        labels.remove(real_file)


print(f"Не обнаружено {len(labels)} элементов")

with open(os.path.join(BASE_DIR, "non_labels.json"), "w", encoding="utf-8") as file:
    json.dump(labels, file, indent=4, ensure_ascii=False)

