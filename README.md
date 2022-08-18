# hacks-ai.ru

# Татарстан
## Разработка модели обнаружения для деперсонализации объектов на изображениях
до 19 августа

## Тизер 
Для решения задачи детекции необходимых обьектов использовался следующий подход:
В основе лежит baseline, предложенная организаторами конкурса
Для расширения датасета использовался следующий подход:
из изображений вырезаются все bbox, в результате остается только фон
вырезанные bbox располагаются в произвольных местах фоновых изображений и записываются в соответствующий labels

### Технические особенности:
Python, Yolo v5, PIL

### Уникальность:
Построенный алгоритм не встречается в научных публикациях, на основе проведенного анализа источников, 
это исключительно моя наработка.

### ScreenCast
https://drive.google.com/file/d/1vLbeJCuLhpNOvFc9FbYGH91nUkoRFonw/view?usp=sharing

norm_labels.ipynb - объединение меток из разных файлов в один, перевод меток в классы (были нули)  
sf_augmentation.ipynb - предложенный мной метод аугментации для детекции


# Статистика и результаты

## Yolo - основной
https://colab.research.google.com/drive/1QydzZF7FoXDNLMfpOpKcCAkvlFF1-UrJ

!python ./yolov5/train.py --img 416 --batch 16 --epochs 100 --data {path_to_data} --cfg ./yolov5/models/yolov5s.yaml --name yolov5s_results  --weights yolov5s.pt --cache  
sample_solution(5).csv  
Score = 0.458840  

my_dataset_yolov5l_600_e10  
Score = 0.512279  


## Yolo - Colab Pro
https://colab.research.google.com/drive/16S_s5LDDjcoBz_mrBFlhweTweWHm268Z  

!python ./yolov5/train.py --img 1024 --batch 4 --epochs 10 --data {path_to_data} --cfg ./yolov5/models/yolov5x.yaml --name yolov5x_results  --weights yolov5x.pt --cache  
sample_solution_de1.csv  
Score = 0.599295  

дообучение + 10 эпох  
sample_yolov5x_1024_e20.csv  
Score = 0.625110  


дообучение: мой датасет + 10 эпох  
!python ./yolov5/train.py --img 1024 --batch 4 --epochs 10 --data {path_to_data} --cfg ./yolov5/models/yolov5x.yaml --name yolov5x_results  --weights yolov5/runs/train/yolov5x_results6/weights/best.pt --cache   
my_yolov5x_1024_e20-10.csv  
---  
не хватает ОЗУ на кеш картинок  
- кеш  




