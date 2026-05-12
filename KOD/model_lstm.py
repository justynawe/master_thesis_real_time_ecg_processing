import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split

from model_ml import (
    AAMI_MAPPING, CLASS_NAMES,
    load_dataset, normalize_with_stats,
    evaluate_model,
)


# ============================================================
# Architecture: CNN stem + Bidirectional LSTM
# ============================================================

def create_lstm_model(input_shape=(360, 2), num_classes=5):
    """
    Hybrid CNN-BiLSTM for ECG beat classification.

    Parameters
    ----------
    input_shape : (timesteps, channels)
        Default (360, 2) – identyczny jak model ResNet.
    num_classes : int
        5 dla standardu AAMI (N, S, V, F, Q).

    Architecture
    ------------
    - CNN stem: dwie warstwy Conv1D+MaxPool redukują sekwencję 360→90
      i wyodrębniają lokalne cechy morfologiczne (kształt fali, nachylenie).
      Skraca sekwencję 4x przed LSTM, co znacząco przyspiesza trening.
    - 2x Bidirectional LSTM: odczytuje sekwencję w obu kierunkach –
      łączy kontekst przyczynowy (P→QRS→T) z retrospektywnym.
    - Dense + Dropout head: klasyfikacja z regularyzacją.
    """
    inputs = layers.Input(shape=input_shape, name='ecg_input')

    # CNN stem – lokalna ekstrakcja cech i redukcja długości sekwencji
    x = layers.Conv1D(32, kernel_size=7, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)   # 360 → 180

    x = layers.Conv1D(64, kernel_size=5, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)   # 180 → 90

    # BiLSTM – zależności czasowe w obu kierunkach
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.2)
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(32, return_sequences=False, dropout=0.2)
    )(x)

    # Głowa klasyfikacyjna
    x = layers.Dense(64, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation='softmax', name='class_probs')(x)

    model = models.Model(inputs, outputs, name='ECG_BiLSTM')
    return model


# ============================================================
# Trening
# ============================================================

def train_model(ds1_path='C:\\Users\\Justyna\\Documents\\SGH_PRACA_MAGISTERSKA\\KOD\\qrs_classified_DS1.npz',
                ds2_path='C:\\Users\\Justyna\\Documents\\SGH_PRACA_MAGISTERSKA\\KOD\\qrs_classified_DS2.npz',
                use_aami=True,
                epochs=60,
                batch_size=64,
                model_save_path='models/ecg_model_lstm.keras'):
    """
    Trenuje model BiLSTM z podziałem inter-patient (AAMI EC57):
      DS1 → trening (85%) + walidacja (15%)
      DS2 → test (żaden z tych rekordów nie był widziany podczas treningu)

    Taki sam pipeline jak model_ml.train_model() – normalizacja,
    class weights, early stopping, ModelCheckpoint.
    """
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    from sklearn.utils.class_weight import compute_class_weight

    # --- DS1: trening + walidacja ---
    print("Wczytywanie DS1 (trening/walidacja)...")
    X_ds1, y_ds1, class_names, le = load_dataset(ds1_path, use_aami)
    num_classes = len(class_names)

    X_train, X_val, y_train, y_val = train_test_split(
        X_ds1, y_ds1, test_size=0.15, random_state=42, stratify=y_ds1)

    # Normalizacja TYLKO na danych treningowych DS1
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std  = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_val   = (X_val   - mean) / std

    # --- DS2: test (niewidziany podczas treningu) ---
    print("Wczytywanie DS2 (test inter-patient)...")
    X_ds2, y_ds2, _, _ = load_dataset(ds2_path, use_aami)
    X_test = normalize_with_stats(X_ds2, mean, std)
    y_test = y_ds2

    # Class weights – kompensacja dominacji klasy N
    weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=y_train)
    class_weight_dict = dict(enumerate(weights))
    print(f"\nClass weights: { {class_names[k]: round(v, 2) for k, v in class_weight_dict.items()} }")

    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val_cat   = tf.keras.utils.to_categorical(y_val,   num_classes)

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_lstm_model(input_shape=input_shape, num_classes=num_classes)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    cb_list = [
        callbacks.EarlyStopping(patience=12, restore_best_weights=True, monitor='val_loss'),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, monitor='val_loss', verbose=1),
        callbacks.ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss', verbose=1),
    ]

    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=cb_list,
        verbose=1
    )

    # Zapisz statystyki normalizacji obok modelu
    np.savez('models/norm_stats_lstm.npz', mean=mean, std=std)

    _evaluate_and_save(model, X_test, y_test, class_names, history)

    return model, history, le, mean, std


def _evaluate_and_save(model, X_test, y_test, class_names, history=None):
    """Ewaluacja i zapis wykresów z przyrostkiem _lstm."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    print("\n=== Test set results (BiLSTM) ===")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix — BiLSTM — Test Set')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix_lstm.png', dpi=150)
    plt.close()
    print("Saved: confusion_matrix_lstm.png")

    if history is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history.history['accuracy'],     label='Train')
        ax1.plot(history.history['val_accuracy'], label='Validation')
        ax1.set_title('Accuracy — BiLSTM')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(history.history['loss'],     label='Train')
        ax2.plot(history.history['val_loss'], label='Validation')
        ax2.set_title('Loss — BiLSTM')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_curves_lstm.png', dpi=150)
        plt.close()
        print("Saved: training_curves_lstm.png")


# Inference helper – identyczny interfejs jak model_ml.predict_beat()

def predict_beat(model, window, mean, std, le):
    """
    Klasyfikuje pojedyncze okno pobudzenia EKG.

    Parameters
    ----------
    model  : tf.keras.Model  (BiLSTM)
    window : np.ndarray  shape (360, 2)
    mean, std : statystyki z treningu  shape (1, 1, 2)
    le     : LabelEncoder

    Returns
    -------
    class_name : str   np. 'N', 'V'
    confidence : float  0-1
    """
    x = (window.astype(np.float32) - mean) / (std + 1e-8)
    probs = model.predict(x[np.newaxis], verbose=0)[0]
    idx = np.argmax(probs)
    return le.classes_[idx], float(probs[idx])


if __name__ == '__main__':
    train_model(
        ds1_path='C:\\Users\\Justyna\\Documents\\SGH_PRACA_MAGISTERSKA\\KOD\\qrs_classified_DS1.npz',
        ds2_path='C:\\Users\\Justyna\\Documents\\SGH_PRACA_MAGISTERSKA\\KOD\\qrs_classified_DS2.npz',
        use_aami=True,
        epochs=60,
        batch_size=64,
        model_save_path='models/ecg_model_lstm.keras'
    )
