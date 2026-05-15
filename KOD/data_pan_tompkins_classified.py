import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wfdb
import os
import glob
from scipy import signal
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Standardowy podział inter-patient (AAMI EC57 / de Chazal et al.)
# Rekordy 102, 104, 107, 217 pominięte – dominują rytmy stymulowane (/).
# Rekordy 204 i 206 niedostępne w pobranym zbiorze MIT-BIH.
# ============================================================
DS1 = ['101','106','108','109','112','114','115','116',
       '118','119','122','124','201','203','205','207',
       '208','209','215','220','223','230']

DS2 = ['100','103','105','111','113','117','121','123',
       '200','202','210','212','213','214','219','221',
       '222','228','231','232','233','234']

class PanTompkinsQRS:
    def __init__(self, sampling_rate=360):
        self.fs = sampling_rate
        
    def bandpass_filter(self, data):
        
        nyquist = 0.5 * self.fs
        low = 5 / nyquist
        high = 15 / nyquist
        b, a = signal.butter(1, [low, high], btype='band')
        return signal.filtfilt(b, a, data)
    
    def derivative_filter(self, data):
        
        h = np.array([-1, -2, 0, 2, 1]) / 8
        return np.convolve(data, h, mode='same')
    
    def squaring(self, data):
        
        return data ** 2
    
    def moving_window_integration(self, data, window_size=None):

        if window_size is None:
            window_size = int(0.15 * self.fs)  
        
        window = np.ones(window_size) / window_size
        return np.convolve(data, window, mode='same')
    
    def find_peaks(self, data, min_distance=None):
        if min_distance is None:
            min_distance = int(0.2 * self.fs)  
        
        threshold = 0.125 * np.max(data)
        
        peaks = []
        i = 0
        while i < len(data):
            if data[i] > threshold:
                start = max(0, i - min_distance // 2)
                end = min(len(data), i + min_distance // 2)
                local_max_idx = start + np.argmax(data[start:end])
                
                if local_max_idx not in peaks:
                    peaks.append(local_max_idx)
                
                i = local_max_idx + min_distance
            else:
                i += 1
        
        return np.array(peaks)
    
    def detect_qrs(self, ecg_signal):
        
        filtered = self.bandpass_filter(ecg_signal)
        derivative = self.derivative_filter(filtered)
        squared = self.squaring(derivative)
        integrated = self.moving_window_integration(squared)
        qrs_peaks = self.find_peaks(integrated)
        
        return qrs_peaks, filtered, derivative, squared, integrated
    
    def plot_filtration_steps(self, ecg_signal, num_qrs=5, save_path='filtration_steps.png'):
        
        qrs_peaks, filtered, derivative, squared, integrated = self.detect_qrs(ecg_signal)
        
        if len(qrs_peaks) < num_qrs:
            print(f"Not enough QRS complexes detected, found {len(qrs_peaks)}")
            return
        
        start_idx = qrs_peaks[0] - int(0.2 * self.fs)
        end_idx = qrs_peaks[num_qrs-1] + int(0.4 * self.fs)
        
        start_idx = max(0, start_idx)
        end_idx = min(len(ecg_signal), end_idx)
        
        time = (np.arange(end_idx - start_idx)) / self.fs
        
        fig, axes = plt.subplots(5, 1, figsize=(15, 12))
        
        axes[0].plot(time, ecg_signal[start_idx:end_idx], 'darkblue', linewidth=0.8)
        axes[0].set_ylabel('Amplituda')
        axes[0].set_title('Surowy sygnał EKG')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(time, filtered[start_idx:end_idx], 'purple', linewidth=0.8)
        axes[1].set_ylabel('Amplituda')
        axes[1].set_title('Filtracja pasmowoprzepustowa (5-15 Hz)')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(time, derivative[start_idx:end_idx], 'mediumpurple', linewidth=0.8)
        axes[2].set_ylabel('Amplituda')
        axes[2].set_title('Filtracja różniczkowa')
        axes[2].grid(True, alpha=0.3)
        
        axes[3].plot(time, squared[start_idx:end_idx], 'maroon', linewidth=0.8)
        axes[3].set_ylabel('Amplituda')
        axes[3].set_title('Kwadratowanie')
        axes[3].grid(True, alpha=0.3)
        
        axes[4].plot(time, integrated[start_idx:end_idx], 'darkorange', linewidth=0.8)
        for peak in qrs_peaks[:num_qrs]:
            if start_idx <= peak < end_idx:
                axes[4].axvline((peak - start_idx)/self.fs, color='r', linestyle='--', alpha=0.5, linewidth=1)
                axes[4].plot((peak - start_idx)/self.fs, integrated[peak], 'ro', markersize=8)
        axes[4].set_ylabel('Amplituda')
        axes[4].set_title('Całkowanie ruchomym oknem')
        axes[4].set_xlabel('Czas (s)')
        axes[4].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Filtration steps plot saved to {save_path}")

class QRSBasedDataProcessor:
    def __init__(self, data_dir="mit-bih-data", window_size=360):
        self.data_dir = data_dir
        self.window_size = window_size
        self.half_window = window_size // 2
        self.qrs_detector = PanTompkinsQRS()
        self.label_encoder = LabelEncoder()
        
        self.annotation_mapping = {
            'N': 'Normal',
            'L': 'LBBB',
            'R': 'RBBB', 
            'A': 'APC',
            'a': 'APC',
            'J': 'NPC',
            'S': 'PVC',
            'V': 'PVC',
            'F': 'Fusion',
            'e': 'Atrial_escape',
            'j': 'Nodal_escape',
            'E': 'Ventricular_escape',
            '/': 'Paced',
            'f': 'Fusion_paced',
            'Q': 'Unknown',
            '?': 'Unknown',
            '[': 'Start',
            ']': 'End',
            'x': 'Non_conducted',
            '(': 'Waveform_onset',
            ')': 'Waveform_end',
            'p': 'Peak',
            't': 'T_wave',
            'u': 'U_wave',
            '`': 'Measure',
            '\'': 'Measure',
            '^': 'Non_conducted',
            '|': 'Isolated',
            '~': 'Change',
            '+': 'Rhythm_change',
            's': 'ST_change',
            'T': 'T_change',
            '*': 'Systole',
            'D': 'Diastole',
            '=': 'Measurement',
            '"': 'Comment',
            '@': 'Link'
        }

    def load_record(self, record_name):
        try:
            
            record_path = os.path.join(self.data_dir, record_name)
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
            
            if record.n_sig >= 2:
                ecg_signal_ch0 = record.p_signal[:, 0]  # Lead I
                ecg_signal_ch1 = record.p_signal[:, 1]  # Lead II
                ecg_signals = np.column_stack([ecg_signal_ch0, ecg_signal_ch1])
            else:
                ecg_signal = record.p_signal[:, 0]  
                ecg_signals = np.column_stack([ecg_signal, ecg_signal])  
            
            ann_samples = annotation.sample
            ann_symbols = annotation.symbol
            print(f"Symbols found in record {record_name}: {set(ann_symbols)}")
            return ecg_signals, ann_samples, ann_symbols
            
        except Exception as e:
            print(f"Error loading record {record_name}: {e}")
            return None, None, None

    def convert_to_frequency_domain(self, time_windows):

        freq_windows = []
        
        for window in time_windows:

            if window.shape[1] >= 2:
                lead_i = window[:, 0]
                lead_ii = window[:, 1]
            else:
                
                lead_i = window[:, 0]
                lead_ii = window[:, 0]
            
            channel_spectra = []
            for lead in [lead_i, lead_ii]:
                
                windowed_signal = lead * np.hanning(len(lead))
                
                fft_result = fft(windowed_signal)
                fft_magnitude = np.abs(fft_result)
                
                positive_freq_magnitude = fft_magnitude[:len(fft_magnitude)//2 + 1]
                channel_spectra.append(positive_freq_magnitude)
            
            freq_window = np.column_stack(channel_spectra)
            freq_windows.append(freq_window)
        
        return np.array(freq_windows)

    def extract_qrs_windows(self, ecg_signals, annotations, ann_symbols):

        detection_channel = ecg_signals[:, 0]
        qrs_peaks, filtered_signal, derivative, squared, integrated = self.qrs_detector.detect_qrs(detection_channel)
        
        windows = []
        labels = []
        
        for qrs_peak in qrs_peaks:

            start_idx = qrs_peak - self.half_window
            end_idx = qrs_peak + self.half_window
            
            if start_idx < 0 or end_idx >= len(ecg_signals):
                continue
            
            window = ecg_signals[start_idx:end_idx, :]
            
            annotations_in_window = []
            for i, ann_sample in enumerate(annotations):
                if start_idx <= ann_sample <= end_idx:
                    annotations_in_window.append(ann_symbols[i])
                 
            if len(annotations_in_window) > 0:
    
                primary_annotation = annotations_in_window[0]
                if primary_annotation in self.annotation_mapping:
                    label = self.annotation_mapping[primary_annotation]
                else:
                    label = 'Unknown'
            else:
    
                continue
            
            windows.append(window)
            labels.append(label)
        
        return np.array(windows), np.array(labels), qrs_peaks

    def process_all_records(self, record_list=None):
        """
        Przetwarza rekordy MIT-BIH i ekstrahuje okna QRS.

        Parameters
        ----------
        record_list : list[str] | None
            Lista numerów rekordów do przetworzenia (np. DS1 lub DS2).
            None = wszystkie rekordy w katalogu (stare zachowanie).
        """
        if record_list is not None:
            # Filtruj do rekordów, które faktycznie istnieją na dysku
            record_numbers = [r for r in record_list
                              if os.path.exists(os.path.join(self.data_dir, f"{r}.dat"))]
            missing = set(record_list) - set(record_numbers)
            if missing:
                print(f"UWAGA: brak plików dla rekordów: {sorted(missing)}")
        else:
            dat_files = glob.glob(os.path.join(self.data_dir, "*.dat"))
            record_numbers = [os.path.basename(f).replace('.dat', '') for f in dat_files]
            record_numbers.sort()

        print(f"Found {len(record_numbers)} records: {record_numbers}")
        
        all_windows = []
        all_labels = []
        record_stats = {}
        
        for record_number in record_numbers:
            print(f"\nProcessing record {record_number}")
            
            ecg_signals, annotations, ann_symbols = self.load_record(record_number)
            
            if ecg_signals is None:
                continue
            
            windows, labels, qrs_peaks = self.extract_qrs_windows(ecg_signals, annotations, ann_symbols)
            
            print(f"Record {record_number}: {len(qrs_peaks)} QRS detected, {len(windows)} windows extracted")
            
            unique_labels, counts = np.unique(labels, return_counts=True)
            record_stats[record_number] = {
                'total_qrs': len(qrs_peaks),
                'total_windows': len(windows),
                'labels': dict(zip(unique_labels, counts))
            }
            
            all_windows.extend(windows)
            all_labels.extend(labels)
        
        all_windows = np.array(all_windows)
        all_labels = np.array(all_labels)
        
        all_freq_windows = self.convert_to_frequency_domain(all_windows)
        
        print(f"Total QRS windows: {len(all_windows)}")
        print(f"Time domain shape: {all_windows[0].shape if len(all_windows) > 0 else 'None'}")
        print(f"Frequency domain shape: {all_freq_windows[0].shape if len(all_freq_windows) > 0 else 'None'}")
        
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        label_dist = dict(zip(unique_labels, counts))
        print(f"Label distribution: {label_dist}")
        
        return all_windows, all_freq_windows, all_labels, record_stats

    def create_summary_plots(self, all_windows, all_freq_windows, all_labels, record_stats):

        plt.figure(figsize=(18, 12))
        
        plt.subplot(2, 3, 1)
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        plt.bar(range(len(unique_labels)), counts)
        plt.xticks(range(len(unique_labels)), unique_labels, rotation=45, ha='right')
        plt.title('QRS-Based Dataset Label Distribution')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Sample windows for each class 
        plt.subplot(2, 3, 2)
        unique_labels_sample = unique_labels[:]  # Show first 6 classes
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels_sample)))
        
        for i, label in enumerate(unique_labels_sample):
            label_indices = np.where(all_labels == label)[0]
            if len(label_indices) > 0:
                sample_window = all_windows[label_indices[0]]
                if len(sample_window.shape) > 1:
                    signal_to_plot = sample_window[:, 0]  
                else:
                    signal_to_plot = sample_window
                plt.plot(signal_to_plot, color=colors[i], alpha=0.7, 
                        label=f'{label} (n={counts[list(unique_labels).index(label)]})')
        
        plt.title('Sample QRS Windows by Class (Lead I)')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 3)
        records = list(record_stats.keys())
        qrs_counts = [record_stats[r]['total_qrs'] for r in records]
        plt.bar(range(len(records)), qrs_counts)
        plt.xticks(range(len(records)), records, rotation=45, ha='right')
        plt.title('QRS Detections per Record')
        plt.ylabel('QRS Count')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Average frequency spectrum for top classes
        plt.subplot(2, 3, 4)
        freqs = np.linspace(0, 180, all_freq_windows.shape[1])  # 0 to 180 Hz
        top_classes = sorted(zip(unique_labels, counts), key=lambda x: x[1], reverse=True)[:]
        
        for label, count in top_classes:
            label_indices = np.where(all_labels == label)[0]
            if len(label_indices) > 10:
                class_freq_windows = all_freq_windows[label_indices[:50]]
                avg_spectrum = np.mean(class_freq_windows, axis=0)
                if len(avg_spectrum.shape) > 1:
                    spectrum_to_plot = avg_spectrum[:, 0]
                else:
                    spectrum_to_plot = avg_spectrum
                freq_mask = freqs <= 50
                plt.semilogy(freqs[freq_mask], spectrum_to_plot[freq_mask], 
                           alpha=0.8, linewidth=2, label=f'{label} (n={count})')
        
        plt.title('Average Frequency Spectrum by Class (0-50 Hz)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 5)
        
        for label, count in top_classes:
            label_indices = np.where(all_labels == label)[0]
            if len(label_indices) > 10:
                class_windows = all_windows[label_indices[:50]]
                if len(class_windows.shape) > 2:
                    avg_window = np.mean(class_windows[:, :, 0], axis=0)
                else:
                    avg_window = np.mean(class_windows, axis=0)
                plt.plot(avg_window, alpha=0.8, linewidth=2, label=f'{label} (n={count})')
        
        plt.title('Average QRS Morphology by Class (Lead I)')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Frequency domain comparison at key frequencies
        plt.subplot(2, 3, 6)
        freqs = np.linspace(0, 180, all_freq_windows.shape[1])
        
        # Extract magnitudes at key frequencies: 5Hz, 10Hz, 15Hz, 30Hz
        key_freq_indices = [int(f * all_freq_windows.shape[1] / 180) for f in [5, 10, 15, 30]]
        key_freq_names = ['5Hz', '10Hz', '15Hz', '30Hz']
        
        x_pos = np.arange(len(key_freq_names))
        width = 0.15
        
        for i, (label, count) in enumerate(top_classes[:5]):
            label_indices = np.where(all_labels == label)[0]
            if len(label_indices) > 5:
                avg_spectrum = np.mean(all_freq_windows[label_indices], axis=0)
                if len(avg_spectrum.shape) > 1:
                    spectrum_to_plot = avg_spectrum[:, 0]
                else:
                    spectrum_to_plot = avg_spectrum
                key_magnitudes = [spectrum_to_plot[idx] for idx in key_freq_indices]
                plt.bar(x_pos + i * width, key_magnitudes, width, 
                       label=f'{label}', alpha=0.8)
        
        plt.title('Average Magnitude at Key Frequencies')
        plt.xlabel('Frequency')
        plt.ylabel('Magnitude')
        plt.xticks(x_pos + width * 2, key_freq_names)
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('qrs_frequency_dataset_summary.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("Frequency domain summary plot saved: qrs_frequency_dataset_summary.png")

    def save_dataset(self, windows, freq_windows, labels, 
                     filename_time="qrs_classified_dataset.npz",
                     filename_freq="qrs_frequency_dataset.npz"):
        # Prepare arrays
        windows = np.array(windows)
        freq_windows = np.array(freq_windows)
        labels = np.array(labels)
        
        # Encode labels but keep original string labels for reference
        self.label_encoder.fit(labels)
        encoded_labels = self.label_encoder.transform(labels)
        
        # Save time-domain dataset (original)
        np.savez_compressed(filename_time, 
                          X=windows, 
                          y=encoded_labels,
                          y_strings=labels,  # Keep original string labels
                          label_classes=self.label_encoder.classes_)
        
        # Save frequency-domain dataset (FFT magnitude spectra)
        np.savez_compressed(filename_freq, 
                          X=freq_windows, 
                          y=encoded_labels,
                          y_strings=labels,  # Keep original string labels
                          label_classes=self.label_encoder.classes_)
        
        print(f"Time-domain dataset saved to {filename_time}")
        print(f"Frequency-domain dataset saved to {filename_freq}")
        print(f"Time domain shape: X={windows.shape}, y={encoded_labels.shape}")
        print(f"Frequency domain shape: X={freq_windows.shape}, y={encoded_labels.shape}")
        print(f"Classes: {self.label_encoder.classes_}")
        
        return windows, freq_windows, encoded_labels
"""
def plot_qrs_samples_from_npz(npz_file='qrs_classified_dataset.npz', num_samples=5, save_path='qrs_samples_from_dataset.png'):
    
    data = np.load(npz_file)
    X = data['X']
    y = data['y']
    label_classes = data['label_classes']
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 3*num_samples))
    
    for i in range(min(num_samples, len(X))):
        label_idx = y[i]
        label_name = label_classes[label_idx]
        
        axes[i, 0].plot(X[i, :, 0], 'b', linewidth=1)
        axes[i, 0].set_title(f'Sample {i}: {label_name} - Lead I')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].grid(True, alpha=0.3)
        
        axes[i, 1].plot(X[i, :, 1], 'r', linewidth=1)
        axes[i, 1].set_title(f'Sample {i}: {label_name} - Lead II')
        axes[i, 1].set_ylabel('Amplitude')
        axes[i, 1].grid(True, alpha=0.3)
    
    axes[-1, 0].set_xlabel('Sample')
    axes[-1, 1].set_xlabel('Sample')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"QRS samples from dataset plot saved to {save_path}")
"""
def main():
    print("=== QRS-Based ECG Classification Dataset Creator ===")
    print("Podział inter-patient zgodny z AAMI EC57 (de Chazal et al.)")

    processor = QRSBasedDataProcessor()

    print("\n=== Wizualizacja kroków filtracji (rekord 112 z DS1) ===")
    record = wfdb.rdrecord(os.path.join(processor.data_dir, '112'))
    ecg_signal = record.p_signal[:, 0]
    processor.qrs_detector.plot_filtration_steps(
        ecg_signal, num_qrs=5, save_path='pan_tompkins_filtration_steps.png')

    for split_name, record_list in [('DS1', DS1), ('DS2', DS2)]:
        print(f"\n{'='*60}")
        print(f"Przetwarzanie {split_name} ({len(record_list)} rekordów)...")
        print('='*60)

        windows, freq_windows, labels, stats = processor.process_all_records(record_list)

        if len(windows) == 0:
            print(f"Brak danych dla {split_name}!")
            continue

        time_path = f'qrs_classified_{split_name}.npz'
        freq_path = f'qrs_frequency_{split_name}.npz'
        processor.save_dataset(windows, freq_windows, labels,
                               filename_time=time_path,
                               filename_freq=freq_path)
        print(f"Zapisano: {time_path}, {freq_path}")

   # print("\n=== Przykładowe okna QRS z DS1 ===")
    #plot_qrs_samples_from_npz('qrs_classified_DS1.npz', num_samples=5,
                               #save_path='qrs_samples_from_dataset.png')

if __name__ == "__main__":
    main()