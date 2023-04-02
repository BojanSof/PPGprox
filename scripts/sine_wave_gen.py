import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft


def generate_sine_wave(n, f, amp, fs):
    return amp * np.sin(2*np.pi*f*np.arange(n)/fs)


def print_data_c(data, ncols=8):
    for i, d in enumerate(data):
        print(f"{d}, ", end="")
        if i > 0 and i % ncols == 0:
            print("")

if __name__ == "__main__":
    freqs = [1, 5, 10, 15]
    amps = [100, 20, 300, 1]
    sine = 0
    fs = 50
    n = 1024
    for (amp, freq) in zip(amps, freqs):
        sine += generate_sine_wave(n, freq, amp, fs)
    sine_fft = fft.fft(sine, n)
    sine_fft_freqs = fft.fftfreq(n, 1/fs)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5))
    ax1.plot(sine)
    ax1.set_title("Signal in time domain")
    ax2.plot(sine_fft_freqs[:n//2], np.abs(sine_fft)[:n//2])
    ax2.set_title("Signal in frequency domain")
    fig.tight_layout()
