import os
import shutil
import wave
from audioop import cross
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional
import csv
import glob
import numpy as np
import time

import soundfile as sf
from soundfile import blocks

from tqdm import tqdm

'''
Подготовка данных для дальнейшего смешивания.
Итоговый объем данных в папке Processed_DNS ~107GB
'''

#TODO: Определять спикеров с помощью filelists_speakerphone

#TODO: Предлагается ограничить одного спикера 10000 фраз (или меньше - 5000 или 1000)
#TODO: Если спикеров на язык меньше 10, они все переносятся в cross
#TODO: Если (всего, у двух половин)фраз меньше 30, спикер удаляется

res_path = "./Processed_DNS/"
#Emotional speech - 1001_DFA_DIS_XX.wav, где 1001 - номер спикера

#Пути к filelists
german_wiki_csv_path = "./filelists_speakerphone/german_wikipedia.csv"


#Все пути к исходным датасетам
emotional_speech_path = Path("./datasets_fullband/clean_fullband/emotional_speech/crema_d")
french_speech_path = Path("./datasets_fullband/clean_fullband/french_speech/M-AILABS_Speech_Dataset/"
                      "fr_FR_190hrs_16k/")
german_speech_path = Path("./datasets_fullband/clean_fullband/german_speech/"
                      "M-AILABS_Speech_Dataset/de_DE_16k/")
italian_speech_path = Path("./datasets_fullband/clean_fullband/italian_speech/M-AILABS_Speech_Dataset/"
                           "it_IT_128hrs_16k/")
russian_speech_path = Path("./datasets_fullband/clean_fullband/russian_speech/M-AILABS_Speech_Dataset/"
                           "ru_RU_47hrs_48k/")
spanish_speech_path = Path("./datasets_fullband/clean_fullband/spanish_speech/M-AILABS_Speech_Dataset/"
                           "es_ES_16k/")
vctk_speech_path = Path("./datasets_fullband/clean_fullband/vctk_wav48_silence_trimmed/")





def get_list_from_csv(csv_file, pattern_col_index=0, encoding="utf-8"):
    index = defaultdict(list)  # key -> list of (filename, lineno, row)
    with open(csv_file, newline="", encoding=encoding) as f:
        reader = csv.reader(f)
        for lineno, row in enumerate(reader, start=1):
            if len(row) <= pattern_col_index or row[0] == "filename":
                continue
            key = row[pattern_col_index]
            index[key].append(row)
    return index

'''Проверяет, является ли аудиофайл пустым (все значения нулевые)'''
def check_zeroes(root_path, files_list):
    blocksize = 16000
    kept_list = []

    for name in files_list:
        skip = True
        path = os.path.join(root_path, name)
        chunk_bytes=1<<5
        with wave.open(path, 'rb') as wf:
            frame_bytes = wf.getsampwidth() * wf.getnchannels()
            frames_per_chunk = chunk_bytes // frame_bytes
            while True:
                data = wf.readframes(frames_per_chunk)
                if not data:
                    skip=True
                    break
                if data.count(b'\x00') != len(data):
                    skip=False
                    break
        if skip:
            #print(f"Skip file {name}: zeroes.")
            pass
        else:
            kept_list.append(name)

    files_list[:] = kept_list


def move_files_to_res(root_path, files_dict, data_type, check_for_zeroes=True):
    """
    Переносит файлы из словаря files_dict в папки Processed_DNS,
    сортируя всех спикеров по папкам.
    Same - фразы одного типа смешиваются только между собой
    Cross - смешиваются все фразы разных типов.
    У одного спикера не может быть фраз суммарно больше 5000
    (по 2500 на same и root),
    """
    same_path = res_path + 'Same/' + data_type + '/'
    cross_path = res_path + 'Cross/'
    #os.makedirs(same_path, exist_ok=True)
    #os.makedirs(cross_path, exist_ok=True)


    for key, el_list in files_dict.items():

        #if len(el_list) > 5000:
        #    el_list = el_list[:5000]
        if check_for_zeroes:
            check_zeroes(root_path, el_list)

        if len(el_list) <= 30:
            #print(f"SPK: {key}, i <= 30")
            continue

        mid = len(el_list) // 2
        same_part = el_list[:mid]
        cross_part = el_list[mid:]

        #same_path/Emotional_1001
        os.makedirs(same_path + data_type + '_' + key + '/', exist_ok=True)
        dst = os.path.join(same_path, data_type + '_' + key)
        i = len(os.listdir(dst))

        print(f"Process {data_type}_{key} ...")
        for el in same_part:
            src = os.path.join(root_path, el)
            if i >= 2500:
                #print(f"Same, SPK: {key}, i>2500")
                break
            shutil.move(src, dst)
            i += 1

        #cross_path/Emotional_1001/
        os.makedirs(cross_path + data_type + '_' + key + '/', exist_ok=True)
        dst = os.path.join(cross_path, data_type + '_' + key)
        i = len(os.listdir(dst))

        for el in cross_part:
            src = os.path.join(root_path, el)
            if i >= 2500:
                #print(f"Cross, SPK: {key}, i>2500")
                break
            shutil.move(src, dst)
            i += 1

def move_csv_to_res(csv_list: defaultdict, data_type, check_for_zeroes=True):
    root_path = Path("./datasets_fullband/clean_fullband/")
    same_path = os.path.join(res_path, 'Same', data_type)
    cross_path = os.path.join(res_path, 'Cross')

    print(f"Process {data_type}")
    for key in tqdm(csv_list.keys()):

        file_path_list = [el[0].replace("\\", "/") for el in csv_list[key]]

        file_path_list = [el for el in file_path_list
                          if os.path.exists(os.path.join(root_path, el))]

        if check_for_zeroes:
            check_zeroes(root_path, file_path_list)

        if len(file_path_list) > 30:
            os.makedirs(os.path.join(same_path, data_type + '_' + key), exist_ok=True)
            os.makedirs(os.path.join(cross_path, data_type + '_' + key), exist_ok=True)
        else:
            #print(f"Skip {key}: el={len(file_path_list)}")
            continue

        mid = len(file_path_list) // 2
        same_part = file_path_list[:mid]
        cross_part = file_path_list[mid:]

        for el in same_part:
            src = os.path.join(root_path, el)
            dst = os.path.join(same_path, data_type + '_' + key)
            shutil.move(src, dst)

        for el in cross_part:
            src = os.path.join(root_path, el)
            dst = os.path.join(cross_path, data_type + '_' + key)
            shutil.move(src, dst)

def process_ailabs(root_path, speech_type, check_for_zeroes=True):
    for dir in os.scandir(root_path):
        if dir.name == "mix":
            continue
        for spkr in os.scandir(dir.path):
            print(f"Process spkr {spkr.name}")
            for book_dir in os.scandir(spkr.path):
                speaker_dict = {}
                wavs_path = os.path.join(book_dir.path, 'wavs')
                for el in os.scandir(wavs_path):  # with os.scandir(wavs_path) as wavs:
                    # for el in wavs:
                    speaker_dict.setdefault(spkr.name, []).append(el.name)
                move_files_to_res(wavs_path, speaker_dict, speech_type, check_for_zeroes)

def process_SLR(root_path):
    speaker_dict = {}

    for el in os.scandir(root_path):
        if (el.name.split('_')[-2]) == 'seg':
            key = '_'.join(el.name.split('_')[0:-4])
        else:
            key = '_'.join(el.name.split('_')[0:-2])
        speaker_dict.setdefault(key, []).append(el.name)

    move_files_to_res(root_path, speaker_dict, "Spanish")

#Read_speech не трогаем - там больше 2х спикеров на аудио
def process_vctk(root_path):
    for spkr in os.scandir(root_path):
        speaker_dict = {}
        for el in os.scandir(spkr):
            if "_mic2.wav" in el.name:
                continue
            speaker_dict.setdefault(spkr.name, []).append(el.name)
        move_files_to_res(spkr.path, speaker_dict, "VCTK")


def process_emotional_speech(root_path):
    """
        Группирует файлы в директории по числовому префиксу до символа '_' в имени.
        - Не рекурсивно (только непосредственные элементы directory).
        - Возвращает словарь массивов с именами файлов (без полного пути).
        - Если нет разделителя "_" - попадает в группу None
    """
    # Получаем только файлы (не папки), без рекурсии
    files = os.listdir(root_path)

    result = {} #Dict[Optional[str], List[str]]

    for fname in files:
        if '_' in fname:
            key = fname.split('_')[0]
            #key = prefix if prefix.isdigit() else None
        else:
            key = None
        result.setdefault(key, []).append(fname)

    move_files_to_res(root_path, result, "Emotional")

    shutil.rmtree(root_path.parent)

def process_french_speech(root_path):

    process_ailabs(root_path, "French", check_for_zeroes=False)
    shutil.rmtree(root_path)

def process_german_speech(root_path):

    #Сначала обрабатываем Wiki
    csv_list = get_list_from_csv(german_wiki_csv_path, pattern_col_index=1)
    move_csv_to_res(csv_list, "German")

    process_ailabs(root_path, "German", check_for_zeroes=False)
    
    shutil.rmtree(root_path.parent.parent)

def process_italian_speech(root_path):

    process_ailabs(root_path, "Italian", check_for_zeroes=False)
    shutil.rmtree(root_path)

def process_russian_speech(root_path):

    process_ailabs(root_path, "Russian")
    shutil.rmtree(root_path)

def process_spanish_speech(root_path):

    process_ailabs(root_path, "Spanish", check_for_zeroes=False)

    slr_list = [slr for slr in os.scandir(root_path.parent.parent)
                if slr.is_dir() and slr.name[0:3] == "SLR"]
    for slr in slr_list:
        process_SLR(slr.path)

    shutil.rmtree(root_path.parent.parent)

def process_english_speech(root_path):
    process_vctk(root_path)

    shutil.rmtree(root_path)

def is_subset_dir(dir_str):
    if dir_str=="Train" or dir_str=="Dev" or dir_str=="Test":
        return True
    else:
        return False

#Сначала считаем фразы, потом спикеров
def utter_count(root_path):
    for spkr in os.scandir(root_path):
        if is_subset_dir(spkr.name):
            continue
        utt_list = [el for el in os.scandir(spkr.path)]

        if (len(utt_list)) < 15:
            shutil.rmtree(spkr.path)
            print(f"Spkr {spkr.name}, utt_list < 15 ({len(utt_list)})")
            continue
        if len(utt_list) > 2500:
            for el in utt_list[2500:]:
                print(f"Spkr {spkr.name}, utt_list > 2500 ({len(utt_list)})")
                shutil.rmtree(el.path)
#Фраз на спикера должно быть не меньше 30 и не больше 5000
def check_utter_count():
    same_path = "./Processed_DNS/Same/"
    cross_path = "./Processed_DNS/Cross/"

    for speech_type in os.scandir(same_path):
        utter_count(speech_type.path)

    utter_count(cross_path)

#Если в same-папке спикеров меньше 11, это очень мало. Переносим все в cross
def check_spkr_count():
    same_path = "./Processed_DNS/Same/"
    cross_path = "./Processed_DNS/Cross/"

    #Получаем список подтипов речей (языки, emotional, vctk)
    speech_types = [el for el in os.scandir(same_path)
            if el.is_dir()]

    for sp_type in speech_types:
        spkrs = [name for name in os.scandir(sp_type)
                 if name.is_dir() and not is_subset_dir(name.name)]
        if len(spkrs) == 0:
            continue

        if len(spkrs) < 11:
            #Перенести в соотвествующий cross
            print(f"Type {sp_type.name}: spkrs < 11 ({len(spkrs)})")
            for spkr in spkrs:
                for el in os.scandir(spkr):
                    shutil.move(el.path, os.path.join(cross_path, spkr.name))


            shutil.rmtree(sp_type.path)



'''
При желании можно загружать и обрабатывать данные частично.
Достаточно закомментировать ненужные команды
'''
def main():
    #Создаем папки, куда все будем записывать
    os.makedirs(res_path + '/Same', exist_ok=True)
    os.makedirs(res_path + '/Cross', exist_ok=True)
    process_emotional_speech(emotional_speech_path)
    process_french_speech(french_speech_path)
    process_german_speech(german_speech_path)
    process_italian_speech(italian_speech_path)
    process_russian_speech(russian_speech_path)
    process_spanish_speech(spanish_speech_path)
    process_english_speech(vctk_speech_path)

    check_utter_count()
    check_spkr_count()

if __name__ == "__main__":
    main()
