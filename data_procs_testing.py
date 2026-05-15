import librosa as lib
import numpy as np
import matplotlib.pyplot as plt

FRAMESIZE = 1024
HOPLENGTH = 512

def load_audio(file_path):
    audio, sr = lib.load(file_path, sr=None)
    return audio, sr
