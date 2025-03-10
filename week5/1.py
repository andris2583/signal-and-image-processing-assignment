import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, windows
import scipy.io.wavfile as wav

def spectrogram(file_path, window_size, stride, window_type='hann'):
    sample_rate, audio_data = wav.read(file_path)

    window = getattr(windows, window_type)(window_size)

    f, t, Zxx = stft(audio_data, fs=sample_rate, window=window, nperseg=window_size, noverlap=window_size - stride)

    magnitude = np.abs(Zxx)


    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, 20 * np.log10(magnitude), shading='gouraud', cmap='viridis')
    plt.title("Spectrogram")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.colorbar(label="Magnitude (dB)")
    plt.ylim(0, 1000)
    plt.show()


file_path = 'progression.wav'
window_size = 1024
stride = 512
spectrogram(file_path, window_size, stride)
