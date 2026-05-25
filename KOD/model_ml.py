import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

# AAMI standard: maps detailed labels from data_pan_tompkins_classified.py

AAMI_MAPPING = {
    'Normal':              'N',
    'LBBB':                'N',
    'RBBB':                'N',
    'Atrial_escape':       'N',
    'Nodal_escape':        'N',
    'APC':                 'S',
    'NPC':                 'S',
    'PVC':                 'V',
    'Ventricular_escape':  'V',
    'Fusion':              'F',
    'Paced':               'Q',
    'Fusion_paced':        'Q',
    'Unknown':             'Q',
}
CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']



# Architecture: 1D-CNN with residual blocks


def residual_block(x, filters, kernel_size=5, dilation_rate=1):
    """
    Two-layer residual block with BatchNormalization.
    Skip connection adjusts channel count with a 1x1 conv if needed.
    Dilation allows capturing longer-range temporal context without
    increasing kernel size (useful for P-wave / T-wave relationships).
    """
    shortcut = x

    x = layers.Conv1D(filters, kernel_size, padding='same',
                      dilation_rate=dilation_rate, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv1D(filters, kernel_size, padding='same',
                      dilation_rate=dilation_rate, use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # Match dimensions when filter count changes
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def create_ecg_model(input_shape=(360, 2), num_classes=5):
    """
    1D ResNet for ECG beat classification.

    Parameters
    ----------
    input_shape : (timesteps, channels)
        Default (360, 2) matches windows from data_pan_tompkins_classified.py
        (360 samples at 360 Hz ≈ 1-second window, Lead I + Lead II).
    num_classes : int
        5 for AAMI standard (N, S, V, F, Q).

    Architecture
    ------------
    - Initial wide conv to project raw signal to feature space
    - 4 residual blocks with progressively more filters
    - MaxPooling between blocks to reduce temporal resolution
    - GlobalAveragePooling replaces Flatten (fewer params, better regularisation)
    - Dense + Dropout classification head
    """
    inputs = layers.Input(shape=input_shape, name='ecg_input')

    # Initial feature extraction large kernel captures QRS complex span
    x = layers.Conv1D(32, kernel_size=7, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Block 1 fine-grained local features (P wave, Q dip)
    x = residual_block(x, filters=32, kernel_size=5)
    x = layers.MaxPooling1D(pool_size=2)(x)       # 360 -> 180

    # Block 2  R peak morphology
    x = residual_block(x, filters=64, kernel_size=5)
    x = layers.MaxPooling1D(pool_size=2)(x)       # 180 -> 90

    # Block 3 ST segment and T wave, dilated for wider receptive field
    x = residual_block(x, filters=128, kernel_size=3, dilation_rate=2)
    x = layers.MaxPooling1D(pool_size=2)(x)       # 90 -> 45

    # Block 4 global rhythm context
    x = residual_block(x, filters=256, kernel_size=3, dilation_rate=2)

    # Aggregate over time avoids overfitting vs. Flatten
    x = layers.GlobalAveragePooling1D()(x)

    # Classification head
    x = layers.Dense(128, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation='softmax', name='class_probs')(x)

    model = models.Model(inputs, outputs, name='ECG_ResNet1D')
    return model



# Data loading

def load_dataset(npz_path='qrs_classified_dataset.npz', use_aami=True):
    """
    Load the .npz dataset produced by data_pan_tompkins_classified.py.

    Parameters
    ----------
    npz_path : str
        Path to the time-domain dataset saved by QRSBasedDataProcessor.
    use_aami : bool
        If True, maps the detailed labels to the 5 AAMI classes.
        If False, uses all classes found in the data.

    Returns
    -------
    X : np.ndarray  shape (N, 360, 2)
    y : np.ndarray  integer class indices
    class_names : list[str]
    le : LabelEncoder fitted on class_names
    """
    data = np.load(npz_path)
    X = data['X'].astype(np.float32)   # (N, timesteps, channels)
    y_strings = data['y_strings']

    if use_aami:
        y_mapped = np.array([AAMI_MAPPING.get(lbl, None) for lbl in y_strings])
        valid_mask = np.isin(y_mapped, CLASS_NAMES)
        X = X[valid_mask]
        y_mapped = y_mapped[valid_mask]

        le = LabelEncoder()
        le.fit(CLASS_NAMES)            # fixed alphabetical order: F, N, Q, S, V
        y = le.transform(y_mapped)
        class_names = list(le.classes_)
    else:
        le = LabelEncoder()
        y = le.fit_transform(y_strings)
        class_names = list(le.classes_)

    print(f"Dataset: {X.shape}  |  classes: {class_names}")
    unique, counts = np.unique(y, return_counts=True)
    for idx, cnt in zip(unique, counts):
        print(f"  {class_names[idx]}: {cnt:>6} ({100*cnt/len(y):.1f}%)")

    return X, y, class_names, le


def normalize(X_train, X_val, X_test):
    """Z-score per channel computed from training data only."""
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std  = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std, mean, std


def normalize_with_stats(X, mean, std):
    """Normalizuje zbiór gotowymi statystykami (np. dla zbioru testowego DS2)."""
    return (X - mean) / (std + 1e-8)



# Training


def train_model(ds1_path='C:\\Users\\Justyna\\Documents\\SGH_PRACA_MAGISTERSKA\\KOD\\qrs_classified_DS1.npz',
                ds2_path='C:\\Users\\Justyna\\Documents\\SGH_PRACA_MAGISTERSKA\\KOD\\qrs_classified_DS2.npz',
                use_aami=True,
                epochs=60,
                batch_size=64,
                model_save_path='models/ecg_model.keras'):
    """
    Trenuje model z podziałem inter-patient (AAMI EC57):
      DS1 → trening (85%) + walidacja (15%)
      DS2 → test (żaden z tych rekordów nie był widziany podczas treningu)

    Parameters
    ----------
    ds1_path : str  – plik .npz z beatami rekordów DS1 (trening/walidacja)
    ds2_path : str  – plik .npz z beatami rekordów DS2 (test)
    """
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # --- DS1: trening + walidacja ---
    print("Wczytywanie DS1 (trening/walidacja)...")
    X_ds1, y_ds1, class_names, le = load_dataset(ds1_path, use_aami)
    num_classes = len(class_names)

    X_train, X_val, y_train, y_val = train_test_split(
        X_ds1, y_ds1, test_size=0.15, random_state=42, stratify=y_ds1)

    # Normalizacja liczona TYLKO na danych treningowych DS1
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std  = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_val   = (X_val   - mean) / std

    # --- DS2: test (niewidziany podczas treningu) ---
    print("Wczytywanie DS2 (test inter-patient)...")
    X_ds2, y_ds2, _, _ = load_dataset(ds2_path, use_aami)
    X_test = normalize_with_stats(X_ds2, mean, std)
    y_test = y_ds2

    # Class weights to counteract heavy Normal-beat dominance
    weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=y_train)
    class_weight_dict = dict(enumerate(weights))
    print(f"\nClass weights: { {class_names[k]: round(v, 2) for k, v in class_weight_dict.items()} }")

    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val_cat   = tf.keras.utils.to_categorical(y_val,   num_classes)

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_ecg_model(input_shape=input_shape, num_classes=num_classes)
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

    # Save normalisation stats alongside the model (needed for inference)
    np.savez('models/norm_stats.npz', mean=mean, std=std)

    evaluate_model(model, X_test, y_test, class_names, history)

    return model, history, le, mean, std


# Evaluation

def evaluate_model(model, X_test, y_test, class_names, history=None):
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    print("\n=== Test set results ===")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Macierz pomyłek zbioru testowego (model RestNet1D)')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.close()
    print("Saved: confusion_matrix.png")

    # Training curves
    if history is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history.history['accuracy'],     label='Train')
        ax1.plot(history.history['val_accuracy'], label='Validation')
        ax1.set_title('Dokładność (model RestNet1D)')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(history.history['loss'],     label='Train')
        ax2.plot(history.history['val_loss'], label='Validation')
        ax2.set_title('Strata (model RestNet1D)')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=150)
        plt.close()
        print("Saved: training_curves.png")


# Inference helper (for Kafka consumer integration)

def predict_beat(model, window, mean, std, le):
    """
    Classify a single pre-extracted ECG beat window.

    Parameters
    ----------
    model : tf.keras.Model
    window : np.ndarray  shape (360, 2) raw signal (mV)
    mean, std : normalisation stats from training (shape (1, 1, 2))
    le : fitted LabelEncoder

    Returns
    -------
    class_name : str  e.g. 'N', 'V'
    confidence : float  0-1
    """
    x = (window.astype(np.float32) - mean) / (std + 1e-8)
    probs = model.predict(x[np.newaxis], verbose=0)[0]
    idx = np.argmax(probs)
    return le.classes_[idx], float(probs[idx])




if __name__ == '__main__':
    model, history, le, mean, std = train_model(
        ds1_path='C:\\Users\\Justyna\\Documents\\SGH_PRACA_MAGISTERSKA\\KOD\\qrs_classified_DS1.npz',
        ds2_path='C:\\Users\\Justyna\\Documents\\SGH_PRACA_MAGISTERSKA\\KOD\\qrs_classified_DS2.npz',
        use_aami=True,
        epochs=60,
        batch_size=64,
        model_save_path='models/ecg_model.keras'
    )
