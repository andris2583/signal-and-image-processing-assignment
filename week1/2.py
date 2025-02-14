# -*- coding: utf-8 -*-
"""
Section 2
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import convolve


"""#### Task 2.1"""

soundbyte, soundbyte_samplerate = sf.read('laugh2.wav')

Left = soundbyte[:,0]
Right = soundbyte[:,1]

plt.figure(figsize=(20,5))
plt.title("Waveform of laugh2.wav")
time = np.linspace(0, len(Left) / soundbyte_samplerate, num=len(Left))
plt.plot(time, Left, label='Left')
plt.plot(time, Right, label='Right')
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitud")

plt.legend()

"""#### Task 2.2"""

from scipy import signal
from IPython.display import Audio, display



soundbyte, soundbyte_samplerate = sf.read('Spring_sound.wav')
reverb_1, reverb_1_samplerate = sf.read('Splash_impulse.wav')
reverb_2, reverb_2_samplerate = sf.read('Clap_impulse.wav')

# Resample Reverb 1
if reverb_1_samplerate != soundbyte_samplerate:
    num_samples = int(len(reverb_1) * soundbyte_samplerate / reverb_1_samplerate)
    reverb_1 = signal.resample(reverb_1, num_samples)

# Resample Reverb 2
if reverb_2_samplerate != soundbyte_samplerate:
    num_samples = int(len(reverb_2) * soundbyte_samplerate / reverb_2_samplerate)
    reverb_2 = signal.resample(reverb_2, num_samples)

mix_1 = [
    signal.convolve(reverb_1[:,0], soundbyte[:,0], mode='same'),
    signal.convolve(reverb_1[:,1], soundbyte[:,1], mode='same')
]

mix_2 = [
    signal.convolve(reverb_2[:,0], soundbyte[:,0], mode='same'),
    signal.convolve(reverb_2[:,1], soundbyte[:,1], mode='same')
]

# Original Sound
plt.figure(figsize=(20,5))
plt.title("Original Sound Waveform")
plt.plot(soundbyte[:,0], label='Left')
plt.plot(soundbyte[:,1], label='Right')
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()

# Reverb 1 (Impulse Response)
plt.figure(figsize=(20,5))
plt.title("Splash impulse Waveform")
plt.plot(reverb_1[:,0], label='Left')
plt.plot(reverb_1[:,1], label='Right')
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()

# Reverb 2 (Impulse Response)
plt.figure(figsize=(20,5))
plt.title("Clap impulse Waveform")
plt.plot(reverb_2[:,0], label='Left')
plt.plot(reverb_2[:,1], label='Right')
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()

# Mix 1 (Convolved Sound)
plt.figure(figsize=(20,5))
plt.title("Convolve Splash impulse Waveform")
plt.plot(mix_1[0], label='Left')
plt.plot(mix_1[1], label='Right')
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()

# Mix 2 (Convolved Sound)
plt.figure(figsize=(20,5))
plt.title("Convolve Clap impulse Waveform")
plt.plot(mix_2[0], label='Left')
plt.plot(mix_2[1], label='Right')
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()

#original sound
sound1 = Audio(soundbyte.T, rate=soundbyte_samplerate, autoplay=True)
sound1

#reverb 1
sound2 = Audio(reverb_1.T, rate=reverb_1_samplerate, autoplay=True)
sound2

#reverb 2
sound3 = Audio(reverb_2.T, rate=reverb_2_samplerate, autoplay=True)
sound3

#mix 1
sound4 = Audio(mix_1, rate=soundbyte_samplerate, autoplay=True)
sound4

#mix 2
sound4 = Audio(mix_2, rate=soundbyte_samplerate, autoplay=True)
sound4

