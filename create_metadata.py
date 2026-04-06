import argparse
import csv
import os
import pickle
import random
import warnings
from itertools import combinations, product
from pathlib import Path
from typing import Dict, List

import pyloudnorm as pyln
from tqdm import tqdm

from utils import *

import torch

MIN_LOUDNESS = -33
MAX_LOUDNESS = -25

SEED = 1234

MAX_AMP = 0.9

SAMPLERATE = 16000
REVERB = True
RIR_SR = 48000

#TODO: Максиальная длина смеси 10 секунд
#TODO: Убрать rng, сделать глобальный SEED

speech_type = {'Emotional': 2.4, 'French': 62, 'German': 319, 'Italian': 42, 'VCTK': 299+27,
               'Russian': 12, 'Spanish': 65}
train_len_same = {"Train": (50 * 60 * 60), # Длительность в секундах
                  "Dev": (5 * 60 * 60),
                  "Test": (2.5 * 60 * 60)}
train_len_cross = {"Train": (50 * 60 * 60),
                   "Dev": (5 * 60 * 60),
                   "Test": (2.5 * 60 * 60)}

same_path = Path("./Processed_DNS/Same/")
cross_path = Path("./Processed_DNS/Cross/")

wham_path = Path("./wham_noise")
rir_path = Path("./datasets_fullband/impulse_responses/SLR26/simulated_rirs_48k")

metadata_path = Path("./metadata")
rng_pkl_path = os.path.join(metadata_path, "rng_state.pkl")

SpeakersDict = Dict[str, List[str]]

def get_type_percents(sum, speech_type_dict):
    type_percent = {}
    for el in speech_type_dict:
        type_percent[el] = speech_type_dict[el] / sum * 100
    return type_percent

def get_audio_len(sum_len, percent_dict):
    audio_len_dict = {}
    for el in percent_dict:
        audio_len_dict[el] = percent_dict[el] * sum_len / 100
    return audio_len_dict


def _non_empty_sorted_speakers(speakers: SpeakersDict):
    return [(name, utts) for name, utts in sorted(speakers.items()) if utts]

def count_total_mixtures(speakers: SpeakersDict) -> int:
    """Подсчитать общее число уникальных смесей (len_i * len_j по всем неупорядоченным парам)."""
    non_empty = _non_empty_sorted_speakers(speakers)
    sizes = [len(utts) for _, utts in non_empty]
    total = 0
    n = len(sizes)
    for i in range(n):
        for j in range(i + 1, n):
            total += sizes[i] * sizes[j]
    return total

def iterate_pairwise_mixtures(speakers: SpeakersDict):
    """
    Генератор всех уникальных смесей двух фрагментов из разных спикеров.
    Возвращает (speaker_a, speaker_b, utt_a, utt_b) с speaker_a < speaker_b в сортировке имён.
    """
    non_empty = _non_empty_sorted_speakers(speakers)
    for (sa, utts_a), (sb, utts_b) in combinations(non_empty, 2):
        for ua, ub in product(utts_a, utts_b):
            yield (sa, sb, ua, ub)

def random_sample_mixtures(
    speakers: SpeakersDict,
    rng,
    sample_size = 100_000_000,
    max_attempts = None
):
    """
    Детерминированная случайная выборка уникальных комбинаций размера sample_size.
    Подходит для очень больших пространств, когда материализация всех комбинаций нежелательна.
    Если sample_size >= total — вернёт все комбинации в перемешанном порядке (память будет равна total в этом случае).
    """
    total = count_total_mixtures(speakers)
    if total == 0:
        return


    non_empty = _non_empty_sorted_speakers(speakers)
    m = len(non_empty)

    # Выбираем случайные позиции и проверяем, присутствуют ли они в
    # существующем наборе.
    used = set()
    attempts = 0
    max_attempts = max_attempts or (sample_size * 50)

    while len(used) < sample_size and attempts < max_attempts:
        i_non, j_non = sorted(rng.sample(range(m), 2))  # упорядочим сразу
        (spk_i, utts_i), (spk_j, utts_j) = non_empty[i_non], non_empty[j_non]

        ua = rng.choice(utts_i)
        ub = rng.choice(utts_j)
        key = (spk_i, spk_j, ua, ub)

        if key in used:
            attempts += 1
            continue

        used.add(key)
        attempts = 0
        yield spk_i, spk_j, ua, ub

    #if len(used) < sample_size:
    #    raise RuntimeError(
    #        f"Не удалось составить {sample_size} уникальных комбинаций за {attempts} попыток. "
    #        "Попробуйте увеличить max_attempts или уменьшить sample_size."
    #    )
    print("Комбинаций меньше итоговой длины")
    return

def get_random_rir(root_path, rng: random.Random):
    #Вероятность получить "чистый" сигнал без реверберации - 1/5
    if  rng.randrange(0, 5) == 0:
        return None, None, None

    room_list = []
    for room_size in os.scandir(root_path):
        for room in os.scandir(room_size.path):
            if not room.is_dir():
                continue
            room_list.append(room)

    rand_room = rng.choice(room_list)
    room_list = [el for el in os.scandir(rand_room.path)]
    rir1, rir2, rir3 = rng.sample(room_list, k=3)

    return rir1.path, rir2.path, rir3.path

def set_loudness(mix_list, mix_names, sr, rng):
    loudness_list = []
    sources_list_norm = []
    for i in range(len(mix_list)):
        meter = pyln.Meter(sr)
        #loudness_list.append(meter.integrated_loudness(mix_el))
        loudness = meter.integrated_loudness(mix_list[i])
        loudness_list.append(loudness)
        # Pick a random loudness

        # Если это шум, то он на 5 Дб тише
        if i == len(mix_list)-1:
            noise_max = MAX_LOUDNESS - 5
            noise_min = MIN_LOUDNESS - 5
            #mu = noise_max - (noise_max - noise_min) / 2
            #sigma = (noise_max - noise_min) / 5
            target_loudness = rng.uniform(noise_min, noise_max)
        else:
            #mu = MAX_LOUDNESS - (MAX_LOUDNESS - MIN_LOUDNESS)/2 #Матожидание
            #sigma = (MAX_LOUDNESS - MIN_LOUDNESS) / 3           #Дисперсия
            target_loudness = rng.uniform(MIN_LOUDNESS, MAX_LOUDNESS)

        with warnings.catch_warnings():
            # We don't want to pollute stdout, but we don't want to ignore
            # other warnings.
            warnings.simplefilter("ignore")
            src = pyln.normalize.loudness(mix_list[i], loudness,
                                          target_loudness)
        if loudness == float('-inf'):
            #print(f"Warning: -inf loudness for {mix_names[i]}!")
            raise Exception(f"Got -inf loudness in {mix_names[i]}")
        # If source clips, renormalize
        if np.max(np.abs(src) >= 1):
            src = src * 0.9 / np.max(np.abs(src))
            target_loudness = meter.integrated_loudness(src)

        sources_list_norm.append(src)



    return loudness_list, sources_list_norm
    #Для каждого LUFS считать дельту между сущ. и нужной громкостью

def get_wham_list(subset):
    subset_dir = {"Train": "tr", "Dev": "cv", "Test": "tt"}
    subset_out_list = []

    for el in os.scandir(os.path.join(wham_path, subset_dir[subset])):
        subset_out_list.append(el)

    return subset_out_list

def mix_sources(mix_list):
    """ Do the mixture for min mode and max mode """
    # Initialize mixture
    min_len = min(len(el) for el in mix_list)
    mix_list = [el[:min_len] for el in mix_list]

    mixture_max = np.zeros_like(mix_list[0])
    for i in range(len(mix_list)):
        mixture_max += mix_list[i]
    return mixture_max

def check_mix_clipping(mixture_data, sources_list_norm, sr):
    renormalize_loudness = []

    meter = pyln.Meter(sr)

    if np.max(np.abs(mixture_data)) > MAX_AMP:
        weight = MAX_AMP / np.max(np.abs(mixture_data))
    else:
        weight = 1

    for i in range(len(sources_list_norm)):
        new_loudness = meter.integrated_loudness(sources_list_norm[i] * weight)
        renormalize_loudness.append(new_loudness)
    return renormalize_loudness

def compute_gain(loudness, renormalize_loudness):
    gain = []
    for i in range(len(loudness)):
        delta_loudness = renormalize_loudness[i] - loudness[i]
        gain.append(np.power(10.0, delta_loudness / 20.0))
    return gain

def get_mix_list(sample_list, wham_el, rir_list, sr):
    s1 = read_audio(sample_list[2], RIR_SR)  # sf.read(el[2])
    s2 = read_audio(sample_list[3], RIR_SR)
    wham = read_audio(wham_el, RIR_SR)

    if not (rir_list[0] is None or rir_list[1] is None):
        s1, s2, wham = get_reverb_sources([s1, s2, wham], [rir_list[0], rir_list[1], rir_list[2]], RIR_SR)

    s1 = librosa.resample(s1, orig_sr=RIR_SR, target_sr=sr)
    s2 = librosa.resample(s2, orig_sr=RIR_SR, target_sr=sr)
    wham = librosa.resample(wham, orig_sr=RIR_SR, target_sr=sr)

    mix_list = [s1, s2, wham]

    return mix_list


def read_csv_list(filename):
    data_list = []
    if os.path.exists(filename):
        with open(filename, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for el in reader:
                data_list.append(el[0])
    return data_list

def process_spkrs(csv_path, sprks_path: str, subset, total_time, rng):
    sr = SAMPLERATE  # Потом перенести в параметры
    wham_list = get_wham_list(subset)
    speech_dir = {}
    mix_result_list = []

    time_sec = 0
    mix_count = 0

    srcs_csv_path = os.path.join(metadata_path, f"{subset}_sources.csv")
    wham_csv_path = os.path.join(metadata_path, f"{subset}_wham_sources.csv")
    speech_source_set = set()#set(read_csv_list(srcs_csv_path))
    wham_source_set = set(read_csv_list(wham_csv_path))

    for spkr in os.scandir(os.path.join(sprks_path, subset)):
        for audio in os.scandir(spkr.path):
            speech_dir.setdefault(spkr.name, []).append(audio.path)

    bar_format = '{l_bar}{bar} {n:0.2f}s/{total:0.2f}s [{elapsed}<{remaining}, {rate_fmt}]'
    with (tqdm(total=total_time, bar_format=bar_format) as pbar):
        for el in random_sample_mixtures(speech_dir, rng):
            try:

                if not has_speech_silero(el[2], sr=SAMPLERATE) or not has_speech_silero(el[3], sr=SAMPLERATE):
                    #print(f"File {el[2]} have no speech. Skip...")
                    continue

                if REVERB:
                    rir1, rir2, rir3 = get_random_rir(rir_path, rng)
                else:
                    rir1, rir2 , rir3 = None, None, None

                wham_el = rng.choice(wham_list)
                mix_list = get_mix_list(el, wham_el, [rir1, rir2, rir3], sr)#get_normalized_gain(el, wham_el, [rir1, rir2], sr, rng)

                initial_loudness, mod_sources_list = set_loudness(mix_list, [el[2], el[3], wham_el], sr, rng)
                mix_data = mix_sources(mod_sources_list)
                renormalize_loudness = check_mix_clipping(mix_data, mod_sources_list, sr)
                norm_gain = compute_gain(initial_loudness, renormalize_loudness)

                # Составляем список на вывод
                out_list = [el[2], norm_gain[0], el[3], norm_gain[1], wham_el.path, norm_gain[2],
                            rir1, rir2, rir3]

                # Добавляем источники в набор источников
                # (на будущее, чтобы не хранить неиспользуемые файлы)
                speech_source_set.add(el[2])
                speech_source_set.add(el[3])
                wham_source_set.add(wham_el.path)
                #print(f"Set len={len(speech_source_set)}")
 
                mix_count += 1

                mix_result_list.append(out_list)
                #TODO: csv_writer.writerow(out_list)

                time_step = len(mix_data) / sr
                if time_step > 10:
                    time_step = 10
                time_sec += time_step
                if time_sec > total_time:
                    pbar.update(pbar.total - pbar.n)
                    break
                pbar.update(time_step)

            except Exception as e:
                print("Exception")
                print(e)

    print(f"Mix count = {mix_count}")
    speech_source_set = sorted(speech_source_set)
    wham_source_set = sorted(wham_source_set)
    with (open(srcs_csv_path, "a", newline="", encoding="utf-8") as s,
          open(wham_csv_path, "w", newline="", encoding="utf-8") as wham,
          open(csv_path, "a", newline="", encoding="utf-8") as mix):
        s_writer = csv.writer(s)
        for el in speech_source_set:
            s_writer.writerow([el])

        noise_writer = csv.writer(wham)
        for el in wham_source_set:
            noise_writer.writerow([el])

        mix_writer = csv.writer(mix)
        if os.stat(csv_path).st_size == 0:
         head_row = ["source_1_path",  "source_1_gain", "source_2_path", "source_2_gain", "noise_path",
                    "noise_gain", "rir_1_path", "rir_2_path"]
         mix_writer.writerow(head_row)
        for el in mix_result_list:
            mix_writer.writerow(el)
    with open(rng_pkl_path, "wb") as f:
        pickle.dump(rng.getstate(), f)
    #print("Результат записан")

def is_csv_written(csv_path, speech_type):
    if not os.path.exists(csv_path):
        return False

    with open(csv_path, "r") as f:
        f.readline()
        for el in f:
            name = el.split(",")[0]
            if (speech_type[0] in name) and (speech_type[1] in name):
                return True
        return False




    #return out_list
#------------MAIN----------------------


def main(args):
    global speech_type

    rewrite = args.rewrite

    root_path = metadata_path

    os.makedirs(root_path, exist_ok=True)

    type_percent_dict = get_type_percents(sum(speech_type.values()), speech_type)

    rng = random.Random(SEED)
    if os.path.exists(rng_pkl_path):
        with open(rng_pkl_path, "rb") as f:
            rng.setstate(pickle.load(f))


    for subset in ["Train", "Dev", "Test"]:
        print(f"Обрабатываем {subset}...")

        same_audio_len_dict = get_audio_len(train_len_same[subset], type_percent_dict)

        csv_path = os.path.join(root_path, f"{subset}_mix.csv")
        sources_path = os.path.join(root_path, f"{subset}_sources.csv")
        wham_src_path = os.path.join(root_path, f"{subset}_wham_sources.csv")

        '''if (not rewrite and os.path.exists(csv_path)
                and os.path.exists(sources_path)
                and os.path.exists(wham_src_path)):
            print(f"Пропуск {subset}: уже существует")
            continue'''


        #head_row = ["source_1_path",  "source_1_gain", "source_2_path", "source_2_gain", "noise_path",
        #            "noise_gain", "rir_1_path", "rir_2_path"]
        #writer.writerow(head_row)

        #print("Process same...")
        for speech_type in os.scandir(same_path):
            if not speech_type.is_dir():
                continue

            key = speech_type.name
            time = same_audio_len_dict[key]


            if is_csv_written(csv_path, [subset, key]):
                print(f"{subset}/{key} уже записан. Пропуск...")
            else:
                print(f"Process {key}...")
                process_spkrs(csv_path, speech_type.path, subset, time, rng)

            same_audio_len_dict.pop(key)

        for el in same_audio_len_dict:
            print(f"add {el}")
        #Все неиспользованное время передается в cross
        for time in same_audio_len_dict.values():
            train_len_cross[subset] += time

        print("Process cross...")
        time = train_len_cross[subset]
        if is_csv_written(csv_path, [subset, "Cross"]):
            print(f"{subset}/Cross уже записан. Пропуск...")
        else:
            process_spkrs(csv_path, str(cross_path), subset, time, rng)

    os.remove(rng_pkl_path)
    os.system("sudo shutdown now")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rewrite', type=bool, default=False,
                        help='Перезаписать файлы')

    args = parser.parse_args()
    main(args)

