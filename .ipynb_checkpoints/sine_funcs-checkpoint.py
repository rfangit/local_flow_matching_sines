import torch
import numpy as np
import matplotlib.pyplot as plt

# Function to create a dataset of sine waves of different phases
# Takes in a list of phases, the width of the dataset, and the frequency
# Create a tensor of shape [B, W] where B is number of phases, W is num_points
def create_phase_dataset(phases, num_points=1000, points_per_cycle=40, device = 'cpu'):
    dataset = torch.stack([generate_sine_wave(num_points, points_per_cycle, phase_offset=phase, device = device)
        for phase in phases])
    return dataset

def generate_sine_wave(num_points, points_per_cycle, amplitude=1.0, phase_offset=0, device='cpu'):
    t = np.arange(0, num_points)
    signal = np.sin(2 * np.pi * t/points_per_cycle + phase_offset)
    return torch.from_numpy(signal)

def window_signal(signal, window_size, stride=1):
    return signal.unfold(0, window_size, stride)

# Compute FFTs with phase
def compute_fft(signal):
    num_points = len(signal)
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(num_points)
    
    magnitude = np.abs(fft_result)
    phase_spectrum = np.angle(fft_result)
    
    # Only look at the positive half (real frequencies)
    half_n = len(frequencies) // 2
    positive_freqs = frequencies[:half_n]
    positive_magnitude = magnitude[:half_n]
    positive_phase = phase_spectrum[:half_n]
    
    # Find the frequency with the highest magnitude (excluding DC)
    peak_index = np.argmax(positive_magnitude[1:]) + 1
    peak_freq = positive_freqs[peak_index]
    peak_phase = positive_phase[peak_index]

    return positive_freqs, positive_magnitude, positive_phase, peak_freq, peak_phase

def analyze_fft_results(final_waveforms, title_suffix=""):
    """
    Analyze and plot FFT statistics of denoised waveforms.
    
    Args:
        final_waveforms: Tensor of shape [num_samples, signal_length] containing denoised waveforms
        title_suffix: Optional string to append to plot titles
    """
    # Convert to numpy if needed
    if isinstance(final_waveforms, torch.Tensor):
        final_waveforms = final_waveforms.detach().cpu().numpy()
    
    num_samples = final_waveforms.shape[0]
    
    # Initialize storage
    all_magnitudes = []
    peak_phases = []
    peak_frequencies = []
    
    # Process each waveform
    for waveform in final_waveforms:
        freqs, mag, phase, peak_freq, peak_phase = compute_fft(waveform)
        all_magnitudes.append(mag)
        peak_phases.append(peak_phase)
        peak_frequencies.append(peak_freq)
    
    # Convert to arrays
    all_magnitudes = np.array(all_magnitudes)
    peak_phases = np.array(peak_phases)
    peak_frequencies = np.array(peak_frequencies)
    
    # Calculate statistics
    average_magnitude = np.mean(all_magnitudes, axis=0)
    mean_phase = np.mean(peak_phases)
    mean_freq = np.mean(peak_frequencies)
    
    # Create plots
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Average magnitude spectrum
    plt.subplot(1, 3, 1)
    plt.plot(freqs, average_magnitude)
    plt.xlabel('Frequency')
    plt.ylabel('Average Magnitude')
    plt.title(f'Average FFT Magnitude Spectrum {title_suffix}')
    plt.grid(True)
    
    # Plot 2: Phase distribution (-π to π)
    plt.subplot(1, 3, 2)
    plt.hist(peak_phases, bins=20, range=(-np.pi, np.pi), edgecolor='black')
    plt.axvline(mean_phase, color='r', linestyle='--', label=f'Mean: {mean_phase:.2f}')
    plt.xlabel('Peak Phase (radians)')
    plt.ylabel('Count')
    plt.title(f'Peak Phase Distribution {title_suffix}')
    plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
               ['-π', '-π/2', '0', 'π/2', 'π'])
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Frequency distribution
    plt.subplot(1, 3, 3)
    plt.hist(peak_frequencies, bins=20, range=(freqs[0], freqs[-1]), edgecolor='black')
    plt.axvline(mean_freq, color='r', linestyle='--', label=f'Mean: {mean_freq:.2f}')
    plt.xlabel('Peak Frequency')
    plt.ylabel('Count')
    plt.title(f'Peak Frequency Distribution {title_suffix}')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\nFFT Analysis Summary ({num_samples} samples):")
    print("="*50)
    print(f"Average peak phase: {mean_phase:.3f} ± {np.std(peak_phases):.3f} rad")
    print(f"Average peak frequency: {mean_freq:.3f} ± {np.std(peak_frequencies):.3f}")