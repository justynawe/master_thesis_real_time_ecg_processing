import wfdb
import time
import json
from kafka import KafkaProducer # type: ignore

# Konfiguracja
TOPIC = 'ecg-stream'
# Rekord z DS2 (zbiór testowy inter-patient)
# dowolny z DS2: 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234

RECORD_PATH = 'mit-bih-data/200'

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Wczytanie danych
record = wfdb.rdrecord(RECORD_PATH)
signals = record.p_signal   # shape (N, 2)
fs = record.fs              
interval = 1.0 / fs

print(f"Wysyłanie danych (kanały: {record.sig_name}, fs={fs} Hz)...")
for i, sample_pair in enumerate(signals):
    data = {
        'sample_idx': i,
        'values': [float(sample_pair[0]), float(sample_pair[1])],
        'fs': fs,
    }
    producer.send(TOPIC, value=data)

    # Symulacja rzeczywistego czasu 
    time.sleep(interval)

    if i % fs == 0:
        print(f"Wysłano {i // fs} sekund sygnału")
