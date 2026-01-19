#TODO: Разделить Cross и Same на train, div и test. Соотношения 80:10:10
#Кол-во спикеров проверяем в Cross
import math
import os
import random
import shutil
from os import DirEntry

seed = 1234
same_path = "./Processed_DNS/Same/"
cross_path = "./Processed_DNS/Cross/"

def move_to_subset(subset_list: list[DirEntry], subset_path):
    dst = subset_path
    for el in subset_list:
        src = el.path
        shutil.move(src, dst)

#Same так закидываем (и так разделен по спикерам), Cross разделим сначала по типам
def divide_spkrs(root_path, spkr_list):
    train_percent = 0.8
    div_percent = 0.1
    #test_percent = 0.1

    train_path = os.path.join(root_path, "Train")
    div_path = os.path.join(root_path, "Div")
    test_path = os.path.join(root_path, "Test")

    # Создать папки
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(div_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    train_list = []
    div_list = []
    test_list = []

    #spkr_list = [spkr for spkr in os.scandir(root_path)
    #             if spkr.is_dir() and spkr.name != "Train"
    #             and spkr.name != "Div" and spkr.name != "Test"]

    if len(spkr_list) < 11:#Все в train
        train_list = spkr_list
    else:
        random.seed(seed)
        random.shuffle(spkr_list)

        train_len = math.ceil(len(spkr_list) * train_percent)
        div_len = (len(spkr_list) - train_len) // 2  #math.ceil(len(spkr_list) * div_percent)

        train_list = spkr_list[:train_len]
        div_list = spkr_list[train_len:(train_len+div_len)]
        test_list = spkr_list[(train_len+div_len):]

    # Переместить в train/div/test
    move_to_subset(train_list, train_path)
    move_to_subset(div_list, div_path)
    move_to_subset(test_list, test_path)



#Для Same
for el in os.scandir(same_path):
    spkr_list = [spkr for spkr in os.scandir(el.path)
                 if spkr.is_dir() and spkr.name != "Train"
                 and spkr.name != "Div" and spkr.name != "Test"]

    divide_spkrs(el.path, spkr_list)

#Для Cross
cross_dict = {}
for el in os.scandir(cross_path):
    key = el.name.split('_')[0]
    cross_dict.setdefault(key, []).append(el)

for el_list in cross_dict.values():
    divide_spkrs(cross_path, el_list)
