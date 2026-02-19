import shutil
import subprocess

import librosa
import numpy as np
import soundfile as sf

import torch
import webrtcvad

from scipy.signal import fftconvolve



MIN_AMP = 0.05

def convert_to_mono(data):
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data

def check_for_clipping(data):
    max_val = np.max(np.abs(data))
    if max_val > 1:
        data = data / max_val
    return data

def read_soundfile(data_path, sr, mono=True):
    data, data_sr = sf.read(data_path)
    data = convert_to_mono(data)
    if data_sr != sr:
        data = librosa.resample(data, orig_sr=data_sr, target_sr=sr)
    return data

def read_ffmpeg(data_path, sr, mono=True):
    ffmpeg_command = [
        "ffmpeg", "-i", data_path,
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ar", str(sr),
    ]
    if mono:
        ffmpeg_command += ["-ac", "1"]
    ffmpeg_command += ["pipe:1"]
    out = subprocess.check_output(ffmpeg_command, stderr=subprocess.DEVNULL)
    audio = np.frombuffer(out, dtype=np.float32).astype(np.float64)
    return audio

def read_audio(data_path, sr=48000, mono=True):
    if shutil.which("ffmpeg"):
        data = read_ffmpeg(data_path, sr, mono)
    else:
        data = read_soundfile(data_path, sr, mono)
    return data

def reverb_audio(source, rir_path, sr):
    rir_data = read_audio(rir_path, sr)
    out_s = fftconvolve(source, rir_data, mode='full')
    # Если значения слишком малы, нужно увеличить значение для правильного измерения LUFS в дальнейшем
    if np.max(np.abs(out_s)) < MIN_AMP: #Если слишком
        out_s = out_s * MIN_AMP / np.max(np.abs(out_s))
    out_s = check_for_clipping(out_s)
    return out_s

def get_reverb_sources(sources_list, rir_path_list, sr):

    s1 = reverb_audio(sources_list[0], rir_path_list[0], sr)
    s2 = reverb_audio(sources_list[1], rir_path_list[1], sr)
    noise = reverb_audio(sources_list[2], rir_path_list[2], sr)

    return s1, s2, noise


model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    trust_repo=True
)
(get_speech_timestamps, _, _, _, _) = utils

def vad_timestamps_and_mask(wav_tensor, sr, model, threshold=0.5):
    """
    wav_tensor: 1D torch.Tensor, sr: int
    returns: speech_timestamps (list of dicts {'start','end'} in samples),
             mask: numpy array of 0/1 with length == wav length
    """
    speech_timestamps = get_speech_timestamps(wav_tensor, model, sampling_rate=sr, threshold=threshold)
    n = wav_tensor.size(0)
    mask = np.zeros(n, dtype=np.int16)
    for seg in speech_timestamps:
        start = int(seg['start'])   # sample index
        end = int(seg['end'])       # sample index (exclusive)
        # safety bounds
        start = max(0, min(n, start))
        end = max(0, min(n, end))
        mask[start:end] = 1
    return speech_timestamps, mask

def has_speech_silero(filename, sr=None):
    wav, sr = librosa.load(filename, sr=sr, mono=True)
    wav_torch = torch.from_numpy(wav)
    timestamps, _ = vad_timestamps_and_mask(wav_torch, sr, model)
    if len(timestamps) == 0:
        return False
    else:
        return True

def read_wav_to_pcm16(path, target_sr=16000):
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    pcm16 = (y * 32767).astype(np.int16)
    return pcm16.tobytes(), target_sr

def frames_from_pcm(pcm_bytes, sample_rate, frame_ms=30):
    bytes_per_sample = 2  # int16
    frame_bytes = int(sample_rate * (frame_ms / 1000.0) * bytes_per_sample)
    for start in range(0, len(pcm_bytes), frame_bytes):
        frame = pcm_bytes[start:start+frame_bytes]
        if len(frame) < frame_bytes:
            break
        yield frame

def has_speech_webrtc(path, aggressiveness=3, frame_ms=30, voiced_fraction_threshold=0.001, sr=16000):
    pcm_bytes, sr = read_wav_to_pcm16(path, target_sr=sr)
    vad = webrtcvad.Vad(aggressiveness)  # 0..3, более 3 — более строгий (меньше false positives)
    frames = list(frames_from_pcm(pcm_bytes, sr, frame_ms))
    if not frames:
        return False
    voiced = [1 if vad.is_speech(f, sr) else 0 for f in frames]
    voiced_frac = sum(voiced) / len(voiced)
    has_speech = voiced_frac >= voiced_fraction_threshold
    return has_speech