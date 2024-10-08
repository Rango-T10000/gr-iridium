import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift

# 设置 agg.path.chunksize 以避免 OverflowError
plt.rcParams['agg.path.chunksize'] = 10000

#读取保存的中间文件.dat
file_path = './block_middle_output/after_burst_tagger_file.dat'
data = np.fromfile(file_path, dtype=np.complex64)

#画图
# 采样率
sample_rate = 2e6  # 2 MHz

# 计算FFT
n = len(data)
fft_data = fftshift(fft(data))  # 使用fftshift将零频率点移动到中心
frequencies = np.fft.fftfreq(n, d=1/sample_rate)
frequencies = fftshift(frequencies)  # 调整频率轴

# 计算频谱幅值（取绝对值）并转换为dB
magnitude_spectrum = 20 * np.log10(np.abs(fft_data))

# 画双边频谱图
plt.figure(figsize=(10, 6), dpi=600)
plt.plot(frequencies, magnitude_spectrum)
plt.title("Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid(True)
# plt.show()
plt.savefig('./block_middle_output/Frequency_Spectrum.jpg')

# 单边频谱，如果需要只看正频率部分可以解开注释
# one_sided_freqs = frequencies[n//2:]  # 取正频率部分
# one_sided_spectrum = magnitude_spectrum[n//2:]
# plt.figure(figsize=(10, 6))
# plt.plot(one_sided_freqs, one_sided_spectrum)
# plt.title("Single-Sided Frequency Spectrum")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Magnitude (dB)")
# plt.grid(True)
# plt.show()

# 画时域信号的幅值图
plt.figure(figsize=(10, 6), dpi=600)
plt.plot(np.abs(data))
plt.title("Time Domain Signal Amplitude")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)
# plt.show()
plt.savefig('./block_middle_output/Time_Domain.jpg')