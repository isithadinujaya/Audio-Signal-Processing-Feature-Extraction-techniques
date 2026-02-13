import numpy as np
import matplotlib.pyplot as plt

class MelSpectrogram:
    def __init__(self, sample_rate=22050, n_fft=2048, hop_length=512, 
                 n_mels=128, fmin=0.0, fmax=None):
        """
        Initialize Mel Spectrogram parameters.
        
        Args:
            sample_rate: Audio sample rate in Hz
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            n_mels: Number of mel frequency bins
            fmin: Minimum frequency (Hz)
            fmax: Maximum frequency (Hz), defaults to sample_rate/2
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax if fmax else sample_rate / 2.0
        
        # Create mel filterbank
        self.mel_filters = self._create_mel_filterbank()
    
    def _hz_to_mel(self, hz):
        """Convert Hz to mel scale."""
        return 2595 * np.log10(1 + hz / 700.0)
    
    def _mel_to_hz(self, mel):
        """Convert mel scale to Hz."""
        return 700 * (10**(mel / 2595.0) - 1)
    
    def _create_mel_filterbank(self):
        """Create mel filterbank matrix."""
        # Number of frequency bins in STFT
        n_freqs = self.n_fft // 2 + 1
        
        # Create mel-spaced frequency points
        mel_min = self._hz_to_mel(self.fmin)
        mel_max = self._hz_to_mel(self.fmax)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)
        
        # Convert Hz points to FFT bin numbers
        # FIXED: Use correct formula
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)
        
        # Ensure bins are within valid range
        bin_points = np.clip(bin_points, 0, n_freqs - 1)
        
        print(f"DEBUG: Mel filterbank creation")
        print(f"  n_freqs: {n_freqs}")
        print(f"  mel_min: {mel_min:.2f}, mel_max: {mel_max:.2f}")
        print(f"  hz range: {hz_points[0]:.2f} to {hz_points[-1]:.2f} Hz")
        print(f"  bin range: {bin_points[0]} to {bin_points[-1]}")
        print(f"  bin_points sample: {bin_points[:5]}")
        
        # Create filterbank
        filterbank = np.zeros((self.n_mels, n_freqs))
        
        for i in range(self.n_mels):
            # Left, center, right points for triangular filter
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]
            
            print(f"  Filter {i}: left={left}, center={center}, right={right}")
            
            # Skip if points are the same (shouldn't happen but just in case)
            if left == center == right:
                continue
            
            # Rising slope: from left to center
            if center > left:
                for j in range(left, center):
                    filterbank[i, j] = (j - left) / (center - left)
            
            # Falling slope: from center to right
            if right > center:
                for j in range(center, right):
                    filterbank[i, j] = (right - j) / (right - center)
        
        print(f"  Filterbank non-zero elements: {np.count_nonzero(filterbank)}")
        print(f"  Filterbank max: {filterbank.max():.6f}")
        
        return filterbank
    
    def _stft(self, audio):
        """Compute Short-Time Fourier Transform."""
        # Check if audio is long enough
        if len(audio) < self.n_fft:
            # Pad audio if too short
            audio = np.pad(audio, (0, self.n_fft - len(audio)), mode='constant')
        
        # Apply Hann window
        window = np.hanning(self.n_fft)
        
        # Calculate number of frames
        n_frames = max(1, 1 + (len(audio) - self.n_fft) // self.hop_length)
        
        # Initialize STFT matrix
        stft_matrix = np.zeros((self.n_fft // 2 + 1, n_frames), dtype=np.complex128)
        
        # Compute STFT
        for i in range(n_frames):
            start = i * self.hop_length
            end = start + self.n_fft
            
            # Handle edge case where we need padding
            if end > len(audio):
                frame = np.pad(audio[start:], (0, end - len(audio)), mode='constant')
            else:
                frame = audio[start:end]
            
            # Apply window
            frame = frame * window
            
            # Compute FFT and take only positive frequencies
            fft_frame = np.fft.rfft(frame, n=self.n_fft)
            stft_matrix[:, i] = fft_frame
        
        return stft_matrix
    
    def transform(self, audio):
        """
        Compute mel spectrogram from audio signal.
        
        Args:
            audio: 1D numpy array of audio samples
            
        Returns:
            Mel spectrogram (n_mels x n_frames)
        """
        # Ensure audio is 1D numpy array
        audio = np.asarray(audio).flatten()
        
        if len(audio) == 0:
            raise ValueError("Audio signal is empty")
        
        # Compute STFT
        stft = self._stft(audio)
        
        # Compute power spectrogram
        power_spec = np.abs(stft) ** 2
        
        # Apply mel filterbank
        mel_spec = np.dot(self.mel_filters, power_spec)
        
        return mel_spec
    
    def transform_db(self, audio, ref=None, amin=1e-10, top_db=80.0):
        """
        Compute mel spectrogram in dB scale.
        
        Args:
            audio: 1D numpy array of audio samples
            ref: Reference power (None = use max value)
            amin: Minimum amplitude threshold
            top_db: Maximum dB range
            
        Returns:
            Mel spectrogram in dB (n_mels x n_frames)
        """
        mel_spec = self.transform(audio)
        
        # Use max as reference if not provided
        if ref is None:
            ref = np.max(mel_spec)
            if ref == 0:
                ref = 1.0
        
        # Convert to dB
        log_spec = 10 * np.log10(np.maximum(amin, mel_spec) / ref)
        
        # Apply top_db limit
        if top_db is not None:
            log_spec = np.maximum(log_spec, log_spec.max() - top_db)
        
        return log_spec