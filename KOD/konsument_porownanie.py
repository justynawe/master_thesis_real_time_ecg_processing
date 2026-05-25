import json
import numpy as np
from kafka import KafkaConsumer
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data_pan_tompkins_classified import PanTompkinsQRS
from model_ml import create_ecg_model
from model_lstm import create_lstm_model

# --- KONFIGURACJA ---
TOPIC = 'ecg-stream'
FS = 360
WINDOW_SIZE = 360
HALF_WINDOW = WINDOW_SIZE // 2
VIEW_WIDTH = 1000
PT_BUFFER_SIZE = FS * 8
PT_RUN_EVERY = 50
MIN_PEAK_DISTANCE = int(0.3 * FS)
CLASS_MAP = ['F', 'N', 'Q', 'S', 'V']

# --- MODELE ---
print("Ładowanie modeli...")

resnet = create_ecg_model(input_shape=(WINDOW_SIZE, 2), num_classes=5)
resnet(np.zeros((1, WINDOW_SIZE, 2), dtype=np.float32))
resnet.load_weights('models/ecg_model.keras')
stats_r = np.load('models/norm_stats.npz')
mean_r, std_r = stats_r['mean'], stats_r['std']

lstm = create_lstm_model(input_shape=(WINDOW_SIZE, 2), num_classes=5)
lstm(np.zeros((1, WINDOW_SIZE, 2), dtype=np.float32))
lstm.load_weights('models/ecg_model_lstm.keras')
stats_l = np.load('models/norm_stats_lstm.npz')
mean_l, std_l = stats_l['mean'], stats_l['std']

print("Modele załadowane.")

qrs_detector = PanTompkinsQRS(sampling_rate=FS)

# --- KAFKA ---
consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

# --- WYKRES: dwa subploty z tym samym sygnałem ---
plt.ion()
fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(2, 1, hspace=0.45)

ax_r = fig.add_subplot(gs[0])
ax_l = fig.add_subplot(gs[1])

for ax, title in [(ax_r, '1D-ResNet'), (ax_l, 'CNN-BiLSTM')]:
    ax.set_xlim(0, VIEW_WIDTH)
    ax.set_ylim(-2, 2)
    ax.set_ylabel('Amplituda (mV)')
    ax.set_title(f'EKG – {title}', fontweight='bold')
    ax.grid(True, alpha=0.3)

ax_l.set_xlabel('Próbki')

line_r, = ax_r.plot([], [], 'b-', lw=1)
line_l, = ax_l.plot([], [], 'b-', lw=1)

# kolor adnotacji wg klasy
COLORS = {'N': 'green', 'V': 'red', 'S': 'darkorange', 'F': 'purple', 'Q': 'gray'}

# --- BUFORY ---
plot_buffer  = deque(maxlen=VIEW_WIDTH)
pt_buf_ch0   = deque(maxlen=PT_BUFFER_SIZE)
pt_buf_ch1   = deque(maxlen=PT_BUFFER_SIZE)

ann_r, ann_l         = [], []   # pending annotations per model
artists_r, artists_l = [], []   # matplotlib artists do usunięcia
last_r = last_l = -MIN_PEAK_DISTANCE
samples_since_pt = 0


def redraw_annotations(ax, annotations, artists, buf_start, buf_len, y_text):
    for a in artists:
        try:
            a.remove()
        except Exception:
            pass
    artists.clear()
    still_visible = []
    for (g_idx, label, prob) in annotations:
        x = g_idx - buf_start
        if 0 <= x < buf_len:
            c = COLORS.get(label, 'black')
            vl = ax.axvline(x=x, color=c, linestyle='--', alpha=0.4, lw=1)
            tx = ax.text(x, y_text, f"{label}\n{prob:.2f}",
                         color=c, fontweight='bold', ha='center', fontsize=8)
            artists.extend([vl, tx])
            still_visible.append((g_idx, label, prob))
    return still_visible


def classify(model, mean, std, buf_ch0, buf_ch1, peak_local):
    window = np.column_stack([
        buf_ch0[peak_local - HALF_WINDOW: peak_local + HALF_WINDOW],
        buf_ch1[peak_local - HALF_WINDOW: peak_local + HALF_WINDOW],
    ])
    x_in = ((window.astype(np.float32) - mean) / (std + 1e-8)).reshape(1, WINDOW_SIZE, 2)
    pred = model.predict(x_in, verbose=0)
    idx = int(np.argmax(pred))
    return CLASS_MAP[idx], float(pred[0][idx])


print("Uruchamianie wizualizacji... (Ctrl+C aby zatrzymać)")

try:
    for message in consumer:
        data  = message.value
        vals  = data['values']
        s_idx = data['sample_idx']

        plot_buffer.append(vals[0])
        pt_buf_ch0.append(vals[0])
        pt_buf_ch1.append(vals[1])
        samples_since_pt += 1

        # --- ODŚWIEŻANIE WYKRESU (co 5 próbek) ---
        if s_idx % 5 == 0 and len(plot_buffer) > 1:
            buf_start = s_idx - len(plot_buffer) + 1
            x_data    = np.arange(len(plot_buffer))
            y_data    = list(plot_buffer)

            line_r.set_xdata(x_data); line_r.set_ydata(y_data)
            line_l.set_xdata(x_data); line_l.set_ydata(y_data)

            ann_r[:] = redraw_annotations(ax_r, ann_r, artists_r, buf_start, len(plot_buffer), y_text=1.55)
            ann_l[:] = redraw_annotations(ax_l, ann_l, artists_l, buf_start, len(plot_buffer), y_text=1.55)

            plt.draw()
            plt.pause(0.001)

        # --- PAN-TOMPKINS + KLASYFIKACJA (co PT_RUN_EVERY próbek) ---
        if samples_since_pt >= PT_RUN_EVERY and len(pt_buf_ch0) >= WINDOW_SIZE * 2:
            samples_since_pt = 0
            buf_ch0   = np.array(pt_buf_ch0)
            buf_ch1   = np.array(pt_buf_ch1)
            buf_len   = len(buf_ch0)
            buf_start = s_idx - buf_len + 1

            peaks, *_ = qrs_detector.detect_qrs(buf_ch0)

            for peak_local in peaks:
                peak_global = buf_start + peak_local

                if peak_local < HALF_WINDOW or peak_local >= buf_len - HALF_WINDOW:
                    continue

                # ResNet
                if peak_global > last_r + MIN_PEAK_DISTANCE:
                    label_r, prob_r = classify(resnet, mean_r, std_r, buf_ch0, buf_ch1, peak_local)
                    ann_r.append((peak_global, label_r, prob_r))
                    last_r = peak_global
                    print(f"[{peak_global:7d}]  ResNet: {label_r} ({prob_r:.2f})")

                # BiLSTM
                if peak_global > last_l + MIN_PEAK_DISTANCE:
                    label_l, prob_l = classify(lstm, mean_l, std_l, buf_ch0, buf_ch1, peak_local)
                    ann_l.append((peak_global, label_l, prob_l))
                    last_l = peak_global
                    print(f"[{peak_global:7d}]  BiLSTM: {label_l} ({prob_l:.2f})")

except KeyboardInterrupt:
    print("\nZatrzymano.")
finally:
    plt.ioff()
    plt.show()
