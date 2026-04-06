import argparse
import csv
import os
from pathlib import Path

from tqdm import tqdm

from utils import *

RIR_SR = 48000
SAMPLERATE = 16000

out_path = "./DNS2Mix"

def get_mix_sources_data(read_list):
    s1_path, s1_gain = read_list[0], read_list[1]
    s2_path, s2_gain = read_list[2], read_list[3]
    noise_path, noise_gain = read_list[4], read_list[5]
    rir1_path, rir2_path, rir3_path = read_list[6], read_list[7], read_list[8]

    #filename = Path(s1_path).stem + '_' + Path(s2_path).stem + '.wav'

    s1_data = read_audio(s1_path, RIR_SR)
    s2_data = read_audio(s2_path, RIR_SR)
    noise_data = read_audio(noise_path, RIR_SR)

    if not (rir1_path == "") or not (rir2_path == "") or not (rir3_path == ""):
        s1_data, s2_data, noise_data = get_reverb_sources([s1_data, s2_data, noise_data],
                                                          [rir1_path, rir2_path, rir3_path], RIR_SR)

    if RIR_SR != SAMPLERATE:
        s1_data = librosa.resample(s1_data, orig_sr=RIR_SR, target_sr=SAMPLERATE)
        s2_data = librosa.resample(s2_data, orig_sr=RIR_SR, target_sr=SAMPLERATE)
        noise_data = librosa.resample(noise_data, orig_sr=RIR_SR, target_sr=SAMPLERATE)

    s1_data = s1_data * float(s1_gain)
    s2_data = s2_data * float(s2_gain)
    noise_data = noise_data * float(noise_gain)

    mix_list = [s1_data, s2_data, noise_data]

    min_len = min(len(el) for el in mix_list)
    mix_list = [el[:min_len] for el in mix_list]

    mixture = np.zeros_like(mix_list[0])
    for i in range(len(mix_list)):
        mixture += mix_list[i]
    return [mixture] + mix_list

def create_mix(read_list, subset):
    filename = Path(read_list[0]).stem + '_' + Path(read_list[2]).stem + '.wav'
    dst_folder = os.path.join(out_path, subset)
    dest_dict = {0: "mix", 1: "s1", 2: "s2", 3: "noise"}

    if (os.path.exists(os.path.join(dst_folder, dest_dict[0], filename))
    and os.path.exists(os.path.join(dst_folder, dest_dict[1], filename))
    and os.path.exists(os.path.join(dst_folder, dest_dict[2], filename))):
        print(f"{filename} уже существует. Пропуск...")
        return

    data_list = get_mix_sources_data(read_list)
    for i in range(4):
        dst = os.path.join(dst_folder, dest_dict[i], filename)
        sf.write(dst, data_list[i], SAMPLERATE)



def main(args):
    metadata_path = "./metadata"

    subset_list = ["Train", "Dev", "Test"]
    type_list = ["mix", "s1", "s2", "noise"]



    #os.makedirs(, exist_ok=True)
    for subset in subset_list:
        for el in type_list:
            os.makedirs(os.path.join(out_path, subset, el), exist_ok = True)

        print(f"Process {subset}")

        csv_path = os.path.join(metadata_path, subset+"_mix.csv")

        with open(csv_path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f) - 1

        with open(csv_path, newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader) #Пропускаем заголовок
            for row in tqdm(reader, total=total_lines - 1):
                read_list = row
                create_mix(read_list, subset)

    os.system("sudo shutdown now")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
