"""
Generuje wizualizacje architektur obu modeli:
  - plot_model  → diagram warstw z kształtami tensorów (Keras + graphviz)
  - visualkeras → warstwowy widok 3D (pip install visualkeras)

Wyniki zapisywane do folderu models/plots/
"""

import os
import numpy as np
import tensorflow as tf

os.makedirs('models/plots', exist_ok=True)

from model_ml  import create_ecg_model
from model_lstm import create_lstm_model

models_cfg = [
    (create_ecg_model,  'ECG_ResNet1D', 'resnet'),
    (create_lstm_model, 'ECG_BiLSTM',   'bilstm'),
]

# --- 1. plot_model (wbudowane w Keras) ---
print("=== plot_model ===")
for factory, name, slug in models_cfg:
    model = factory(input_shape=(360, 2), num_classes=5)
    out_path = f'models/plots/{slug}_plot_model.png'
    try:
        tf.keras.utils.plot_model(
            model,
            to_file=out_path,
            show_shapes=True,
            show_layer_names=True,
            show_layer_activations=True,
            dpi=150,
            rankdir='TB',   # top-bottom
        )
        print(f"  Zapisano: {out_path}")
    except Exception as e:
        print(f"  [BŁĄD plot_model dla {name}]: {e}")
        print("  Upewnij się, że graphviz jest zainstalowany:")
        print("    pip install graphviz pydot")
        print("    https://graphviz.org/download/  (Windows: dodaj do PATH)")

# --- 2. visualkeras ---
print("\n=== visualkeras ===")
try:
    import visualkeras
    from PIL import ImageFont

    for factory, name, slug in models_cfg:
        model = factory(input_shape=(360, 2), num_classes=5)
        out_path = f'models/plots/{slug}_visualkeras.png'
        visualkeras.layered_view(
            model,
            to_file=out_path,
            legend=True,
            draw_volume=True,
            spacing=30,
            type_ignore=[tf.keras.layers.InputLayer],
        )
        print(f"  Zapisano: {out_path}")

except ImportError:
    print("  visualkeras nie jest zainstalowany.")
    print("  Zainstaluj: pip install visualkeras")

# --- 3. Podsumowanie parametrów w konsoli ---
print("\n=== Liczba parametrów ===")
for factory, name, slug in models_cfg:
    model = factory(input_shape=(360, 2), num_classes=5)
    total  = model.count_params()
    train  = sum(np.prod(w.shape) for w in model.trainable_weights)
    print(f"  {name:<20}  łącznie: {total:>10,}  trenowalnych: {train:>10,}")
