"""import json
import matplotlib.pyplot as plt
from kafka import KafkaConsumer # type: ignore
from collections import deque

# Konfiguracja
TOPIC = 'ecg-stream'
WINDOW_SIZE = 1000  # Ile próbek widocznych na ekranie

# Bufor na dane
data_buffer = deque(maxlen=WINDOW_SIZE)
consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

# Przygotowanie wykresu
plt.ion() # Tryb interaktywny
fig, ax = plt.subplots()
line, = ax.plot([], [], 'r-')
ax.set_ylim(-2, 2) # Amplituda EKG zazwyczaj mieści się w tym zakresie
ax.set_xlim(0, WINDOW_SIZE)
ax.set_title("ECG Stream z Kafki (Real-time)")

print("Waiting for data...")
for message in consumer:
    point = message.value
    data_buffer.append(point['value'])

    # Aktualizacja wykresu co 10 próbek (dla płynności)
    if point['sample_idx'] % 10 == 0:
        line.set_ydata(list(data_buffer))
        line.set_xdata(range(len(data_buffer)))
        plt.draw()
        plt.pause(0.001)"""

import json
import numpy as np
import tensorflow as tf
from kafka import KafkaConsumer
from collections import deque
import matplotlib.pyplot as plt

from data_pan_tompkins_classified import PanTompkinsQRS
from model_ml import create_ecg_model

# --- KONFIGURACJA ---
TOPIC = 'ecg-stream'
# MIT-BIH: 360 Hz – zgodnie z danymi treningowymi (DS1).
# Demo streamuje rekord z DS2 (niewidziany przez model podczas treningu).
FS = 360
WINDOW_SIZE = 360          # rozmiar okna modelu
HALF_WINDOW = WINDOW_SIZE // 2
VIEW_WIDTH = 1000           # próbki widoczne na wykresie
PT_BUFFER_SIZE = FS * 8    # 8 sekund sygnału dla Pan-Tompkinsa
PT_RUN_EVERY = 50          # co ile próbek uruchamiamy Pan-Tompkinsa
MIN_PEAK_DISTANCE = int(0.3 * FS)  # min. odstęp między pikami (~300 ms)

# Poprawna kolejność klas: LabelEncoder.fit(['N','S','V','F','Q'])
# sortuje ALFABETYCZNIE → F=0, N=1, Q=2, S=3, V=4
CLASS_MAP = ['F', 'N', 'Q', 'S', 'V']

# --- INICJALIZACJA MODELU ---
model = create_ecg_model(input_shape=(WINDOW_SIZE, 2), num_classes=5)
model(np.zeros((1, WINDOW_SIZE, 2), dtype=np.float32))  # build grafu
model.load_weights('models/ecg_model.keras')
stats = np.load('models/norm_stats.npz')
mean, std = stats['mean'], stats['std']  # shape (1, 1, 2) – statystyki z treningu

qrs_detector = PanTompkinsQRS(sampling_rate=FS)

# --- KAFKA ---
consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

# --- PRZYGOTOWANIE WYKRESU ---
plt.ion()
fig, ax = plt.subplots(figsize=(12, 6))
line, = ax.plot([], [], 'b-', lw=1, label='EKG (Lead I)')
ax.set_ylim(-2, 2)
ax.set_xlim(0, VIEW_WIDTH)
ax.set_xlabel("Próbki")
ax.set_ylabel("Amplituda (mV)")
ax.set_title("EKG w czasie rzeczywistym – klasyfikacja zespołów QRS")
ax.legend(loc='upper right')

# --- BUFORY ---
plot_buffer = deque(maxlen=VIEW_WIDTH)     # do wizualizacji (kanał 1)
pt_buf_ch0 = deque(maxlen=PT_BUFFER_SIZE)  # do Pan-Tompkinsa (kanał 1)
pt_buf_ch1 = deque(maxlen=PT_BUFFER_SIZE)  # do Pan-Tompkinsa (kanał 2)

# Zarządzanie adnotacjami na wykresie
# pending_annotations: lista (global_idx, label, prob) – dane do przerysowania
# current_artists: artyści matplotlib usuwani przed każdym odświeżeniem
pending_annotations = []
current_artists = []

last_classified_global = -MIN_PEAK_DISTANCE  # indeks ostatnio sklasyfikowanego piku
samples_since_pt = 0

print("Uruchamianie wizualizacji... (Ctrl+C aby zatrzymać)")

try:
    for message in consumer:
        data = message.value
        vals = data['values']
        s_idx = data['sample_idx']

        # Dodaj próbkę do buforów
        plot_buffer.append(vals[0])
        pt_buf_ch0.append(vals[0])
        pt_buf_ch1.append(vals[1])
        samples_since_pt += 1

        # --- AKTUALIZACJA WYKRESU (co 5 próbek) ---
        if s_idx % 5 == 0 and len(plot_buffer) > 1:
            # Usuń poprzednie adnotacje z wykresu
            for artist in current_artists:
                try:
                    artist.remove()
                except Exception:
                    pass
            current_artists.clear()

            # Rysuj sygnał
            line.set_xdata(np.arange(len(plot_buffer)))
            line.set_ydata(list(plot_buffer))

            # Przerysuj adnotacje z przeliczonymi pozycjami X
            buf_start_global = s_idx - len(plot_buffer) + 1
            still_visible = []
            for (g_idx, label, prob) in pending_annotations:
                x_pos = g_idx - buf_start_global
                if 0 <= x_pos < len(plot_buffer):
                    color = 'green' if label == 'N' else 'red'
                    vl = ax.axvline(x=x_pos, color=color, linestyle='--', alpha=0.4, lw=1)
                    tx = ax.text(x_pos, 1.6, f"{label}\n{prob:.2f}",
                                 color=color, fontweight='bold', ha='center', fontsize=8)
                    current_artists.extend([vl, tx])
                    still_visible.append((g_idx, label, prob))
            # Wyrzuć adnotacje, które przewinęły się poza widok
            pending_annotations = still_visible

            plt.draw()
            plt.pause(0.001)

        # --- PAN-TOMPKINS + KLASYFIKACJA (co PT_RUN_EVERY próbek) ---
        if samples_since_pt >= PT_RUN_EVERY and len(pt_buf_ch0) >= WINDOW_SIZE * 2:
            samples_since_pt = 0
            buf_ch0 = np.array(pt_buf_ch0)
            buf_ch1 = np.array(pt_buf_ch1)
            buf_len = len(buf_ch0)
            buf_start_global = s_idx - buf_len + 1

            # Wykryj piki QRS algorytmem Pan-Tompkinsa (tak jak przy treningu)
            qrs_peaks_local, *_ = qrs_detector.detect_qrs(buf_ch0)

            for peak_local in qrs_peaks_local:
                peak_global = buf_start_global + peak_local

                # Pomiń piki za blisko krawędzi bufora (brakuje próbek na pełne okno)
                if peak_local < HALF_WINDOW or peak_local >= buf_len - HALF_WINDOW:
                    continue

                # Pomiń piki w okresie refrakcji (już sklasyfikowany sąsiedni pik)
                if peak_global <= last_classified_global + MIN_PEAK_DISTANCE:
                    continue

                # Wytnij okno wycentrowane na piku R (identycznie jak przy treningu)
                window = np.column_stack([
                    buf_ch0[peak_local - HALF_WINDOW: peak_local + HALF_WINDOW],
                    buf_ch1[peak_local - HALF_WINDOW: peak_local + HALF_WINDOW],
                ])  # shape: (360, 2)

                # Normalizacja statystykami z treningu
                x_in = (window.astype(np.float32) - mean) / (std + 1e-8)
                x_in = x_in.reshape(1, WINDOW_SIZE, 2)

                prediction = model.predict(x_in, verbose=0)
                class_idx = int(np.argmax(prediction))
                label = CLASS_MAP[class_idx]
                prob = float(prediction[0][class_idx])

                pending_annotations.append((peak_global, label, prob))
                last_classified_global = peak_global

                print(f"[próbka {peak_global:7d}]  {label}  ({prob:.2f})")

except KeyboardInterrupt:
    print("\nZatrzymano.")
finally:
    plt.ioff()
    plt.show()
