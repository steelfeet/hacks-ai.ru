"""
https://www.kaggle.com/sharansmenon/herbarium-pytorch

"""
import cv2

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, sys, random, string
os.system('cls' if os.name == 'nt' else 'clear')
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

import json
import pathlib
from tqdm import tqdm

import time
import datetime

from PIL import Image, ImageDraw, ImageFont, ImageFilter


DATA_DIR = pathlib.Path("D:\\", "_hack", "2021-22", "d-hacks-ai.ru", "758261-Irkutsk")
TRAIN_DIR = pathlib.Path(DATA_DIR, "train")
"""
REAL_DIR = pathlib.Path(TRAIN_DIR, "employee_real")
"""
REAL_DIR = pathlib.Path(TRAIN_DIR, "action_real")
OUT_DIR = pathlib.Path(TRAIN_DIR, "2_action_aug")

MAX_COUNT = 500


def generate_random_string(length):
    letters = string.ascii_lowercase
    rand_string = ''.join(random.choice(letters) for i in range(length))
    return rand_string
    

def get_random_attributes(img):
    font_size = random.randint(8, 70) 
    r = random.randint(8, 250) 
    g = random.randint(8, 250) 
    b = random.randint(8, 250) 
    font_color=(r,g,b)

    width, height = img.size
    random_x = random.randint(20, width-20) 
    random_y = random.randint(20, height-20) 

    letters_count = random.randint(1, 3) 
    text = generate_random_string(letters_count)

    return font_size, random_x, random_y, text, font_color


for label in os.listdir(REAL_DIR):
    dir_in = os.path.join(REAL_DIR, label)
    dir_out = os.path.join(OUT_DIR, label)
    try:
        os.mkdir(dir_out)
    except:
        pass

    # сначала используем для аугментации все имеющиеся изображения
    for img_name in os.listdir(dir_in):
        img_path = os.path.join(dir_in, img_name)
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)

        letters_n = random.randint(2, 7) 
        for n in range(0, letters_n):
            font_size, random_x, random_y, text, font_color = get_random_attributes(img)
            unicode_font = ImageFont.truetype(os.path.join(BASE_DIR, "Arial.ttf"), font_size)
            draw.text((random_x,random_y), text, font=unicode_font, fill=font_color)

        img.save(os.path.join(dir_out, img_name))

    # добавляем случайные
    new_n = MAX_COUNT - len(os.listdir(dir_in))
    for i in range(0, new_n):
        img_name = random.choice(os.listdir(dir_in))
        img_path = os.path.join(dir_in, img_name)
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)

        letters_n = random.randint(2, 7) 
        for n in range(0, letters_n):
            font_size, random_x, random_y, text, font_color = get_random_attributes(img)
            unicode_font = ImageFont.truetype(os.path.join(BASE_DIR, "Arial.ttf"), font_size)
            draw.text((random_x,random_y), text, font=unicode_font, fill=font_color)

        new_img_name = str(i) + ".jpg"
        img.save(os.path.join(dir_out, new_img_name))




print("success")