Python 3.12.2 (tags/v3.12.2:6abddd9, Feb  6 2024, 21:26:36) [MSC v.1937 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import numpy as np
import matplotlib.pyplot as plt

class MelSpectrogram:
    def __init__(self, sample_rate =22050, n_fft =2048, hop_length = 512, n_mels = 128, fmin=0.0, fmax = None):

        """
        sample_rate: Audio sample rate in Hz
        n_fft: FFT window size
        hop_length: Number ofsamples between
        n_mels: Number of mel frequency bins
        fmin: Minimum frequency (Hz)
        fmax: Maximum frequency (Hz), defaults to sample_rate/2
        """

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax if fmax else sample_rate/2.0

        #Create mel filter bank
        self.mel_filters = self._create_mel_filterbank()

    def _hz_to_mel(self, hz):
        return 2595*np.log10(1 + hz/700)

    def _mel_to_hz(self, mel):
        return 700* (10**(mel/2595) -1)
        
        
    def _create_mel_filterbank(self):
        #Number of frequency bins in STFT
        n_freqs = self.n_fft//2 + 1

        mel_min = self._hz_to_mel(self.fmin)
        mel_max = self._mel_to_hz(self.fmax)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)

        #Taking the fft bin numbers
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)

        #create filterbank
        filterbank = np.zeros((self.n_mels, n_freqs))

        for i in range(self.n_mels):

            left = bin_points[i]
            center = bin_points[i+1]
            right = bin_points[i+2]

            #Rising slope
            for j in range(left, center):
                filterbank[i,j] = (right -j) / (right - center)

            #falling slope
            for j in range(center, right):
                filterbank[i,j] = (right -j)/ (right - center)

        return filterbank

    def _stft(self, audio):
        """ Computing Short-Time Fourier Transform"""
        #Apply hanning window
        window = np.hanning(self.n_fft)

        #number of frames
        n_frames = 1+ (len(audio) - self.n_fft) // self.hop_length

        #Initialize STFT matrix
        for i in range(n_frames):
            start = i*self.hop_length
            end = start +self.n_fft

            #extract frame and apply window
            frame = audio[start:end] *window

            #compute FFT and take only postive frequencies
            fft_frame = np.fft.rfft(frame, n=self.n_fft)
            stft_matrix[:i] = fft_frame

        return stft_matrix
        

...     def transform(self, audio):
...         """
...         Compute mel spectrogram from audio signal.
... 
...         Args:
...             audio 1D numpy array of audio samples
... 
...         returns:
...             Mel Spectro (n_mels x n_frames)
...         """
...         stft = self._stft(audio)
... 
...         #Compute power spectrogram
...         poer_spec = np.abs(stft)**2
... 
...         #Apply mel filterbank
...         mel_spec = np.dot(sel.mel_filters, power_spec)
... 
...         return mel_spec
...     
...     
...     
...     def trsnform_db(self, audio, ref=1.0, amin=1e-10, top_db=80.0):
...         """
...         Compute mel spectrogram in dB scale.
... 
...         audio: 1D numpy array of audio samples
...         ref: Reference Power
...         amin: Minimum amplitude threshhold
...         top_db: Minimum db range
... 
...         the function will give you the Mel spectrogram in dB (n_mels x frames)
...         """
...         mel_spec = self.transform(audio)
... 
...         #convert to_dB
...         log_spec = 10*np.log10(np.maximum(amin, mel_spec) / ref)
...         log_spec = np.maximum(log_spec, log_spec.max() -top_db)
... 
...         return log_spec
... 
...         
... 
...         
...                 
... 
