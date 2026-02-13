Python 3.12.2 (tags/v3.12.2:6abddd9, Feb  6 2024, 21:26:36) [MSC v.1937 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
# Short-Time Fourier Transform (STFT) implementation

import numpy as np

# Parameters
sample_rate = 22050
n_fft = 2048
hop_length = 512
n_mels = 128
fmin = 0.0
frame_length = 2048  # number of samples in a single frame


def stft(audio):
    """
    Compute STFT of an audio signal.

    Parameters:
        audio (np.ndarray): 1D audio signal

    Returns:
        np.ndarray: STFT matrix (frequency_bins × time_frames)
    """

    # Hann window
    window = np.hanning(frame_length)

    # Calculate number of frames
...     n_frames = (len(audio) - frame_length) // hop_length + 1
... 
...     # Number of frequency bins from rFFT
...     n_freq_bins = frame_length // 2 + 1
... 
...     # Initialize STFT matrix
...     stft_matrix = np.zeros((n_freq_bins, n_frames), dtype=complex)
... 
...     # Process each frame
...     for i in range(n_frames):
...         start = i * hop_length
...         end = start + frame_length
... 
...         # Extract frame
...         frame = audio[start:end]
... 
...         # Apply window
...         frame = frame * window
... 
...         # Compute FFT
...         fft_frame = np.fft.rfft(frame, n=n_fft)
... 
...         # Store result
...         stft_matrix[:, i] = fft_frame
... 
...     return stft_matrix
... 
... 
... # Example usage
... if __name__ == "__main__":
...     # Generate dummy signal (1 second sine wave)
...     t = np.linspace(0, 1, sample_rate)
...     audio = np.sin(2 * np.pi * 440 * t)
... 
...     result = stft(audio)
...     print("STFT shape:", result.shape)
...     
... 
...         
