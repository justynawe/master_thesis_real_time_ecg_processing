"""
Porównanie modeli: 1D-ResNet vs CNN-BiLSTM
===========================================
Skrypt trenuje oba modele i generuje zbiorcze wykresy porównawcze.

Uruchomienie:
    python model_comparison.py

Wyniki (zapisywane w bieżącym katalogu):
    comparison_confusion_matrices.png  – macierze pomyłek obok siebie
    comparison_metrics_bar.png         – F1 / Precision / Recall per klasa
    comparison_training_curves.png     – krzywe treningu obu modeli
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from model_ml import (
    load_dataset, normalize_with_stats,
    create_ecg_model, CLASS_NAMES,
)
from model_lstm import create_lstm_model

# ------------------------------------------------------------------ #
# Parametry                                                           #
# ------------------------------------------------------------------ #
DS1_PATH  = 'C:\\Users\\Justyna\\Documents\\SGH_PRACA_MAGISTERSKA\\KOD\\qrs_classified_DS1.npz'
DS2_PATH  = 'C:\\Users\\Justyna\\Documents\\SGH_PRACA_MAGISTERSKA\\KOD\\qrs_classified_DS2.npz'
EPOCHS    = 60
BATCH     = 64


# ------------------------------------------------------------------ #
# Dane (wspólne dla obu modeli)                                       #
# ------------------------------------------------------------------ #

def prepare_data():
    print("=== Wczytywanie danych ===")
    X_ds1, y_ds1, class_names, le = load_dataset(DS1_PATH, use_aami=True)

    X_train, X_val, y_train, y_val = train_test_split(
        X_ds1, y_ds1, test_size=0.15, random_state=42, stratify=y_ds1)

    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std  = X_train.std(axis=(0, 1), keepdims=True) + 1e-8

    X_train = (X_train - mean) / std
    X_val   = (X_val   - mean) / std

    X_ds2, y_ds2, _, _ = load_dataset(DS2_PATH, use_aami=True)
    X_test = normalize_with_stats(X_ds2, mean, std)
    y_test = y_ds2

    return X_train, X_val, X_test, y_train, y_val, y_test, class_names, le


# ------------------------------------------------------------------ #
# Trening jednego modelu                                              #
# ------------------------------------------------------------------ #

def train_one(model, X_train, X_val, y_train, y_val, class_names, save_path):
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow.keras import callbacks

    num_classes = len(class_names)
    weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=y_train)
    class_weight_dict = dict(enumerate(weights))

    y_tr_cat  = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val_cat = tf.keras.utils.to_categorical(y_val,   num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    cb_list = [
        callbacks.EarlyStopping(patience=12, restore_best_weights=True, monitor='val_loss'),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6,
                                    monitor='val_loss', verbose=0),
        callbacks.ModelCheckpoint(save_path, save_best_only=True,
                                  monitor='val_loss', verbose=0),
    ]

    history = model.fit(
        X_train, y_tr_cat,
        validation_data=(X_val, y_val_cat),
        epochs=EPOCHS,
        batch_size=BATCH,
        class_weight=class_weight_dict,
        callbacks=cb_list,
        verbose=1,
    )
    return history


# ------------------------------------------------------------------ #
# Wykresy                                                             #
# ------------------------------------------------------------------ #

def plot_confusion_matrices(y_test, y_pred_resnet, y_pred_lstm, class_names):
    cm_r = confusion_matrix(y_test, y_pred_resnet)
    cm_l = confusion_matrix(y_test, y_pred_lstm)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, cm, title, cmap in zip(
        axes,
        [cm_r, cm_l],
        ['1D-ResNet', 'CNN-BiLSTM'],
        ['Blues', 'Oranges'],
    ):
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f'Macierz pomyłek — {title}', fontsize=13)
        ax.set_ylabel('Prawdziwa klasa')
        ax.set_xlabel('Przewidziana klasa')

    plt.suptitle('Porównanie modeli — zbiór testowy DS2 (inter-patient)', fontsize=14)
    plt.tight_layout()
    plt.savefig('comparison_confusion_matrices.png', dpi=150)
    plt.close()
    print("Saved: comparison_confusion_matrices.png")


def plot_metrics_bar(y_test, y_pred_resnet, y_pred_lstm, class_names):
    """F1-score per klasa dla obu modeli — wykres słupkowy."""
    metrics = {}
    for name, y_pred in [('1D-ResNet', y_pred_resnet), ('CNN-BiLSTM', y_pred_lstm)]:
        _, _, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=range(len(class_names)), zero_division=0)
        metrics[name] = f1

    x = np.arange(len(class_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, metrics['1D-ResNet'],  width, label='1D-ResNet',  color='steelblue')
    bars2 = ax.bar(x + width / 2, metrics['CNN-BiLSTM'], width, label='CNN-BiLSTM', color='darkorange')

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=12)
    ax.set_ylabel('F1-score')
    ax.set_ylim(0, 1.05)
    ax.set_title('F1-score per klasa AAMI — DS2 (inter-patient)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for bar in [*bars1, *bars2]:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f'{h:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('comparison_metrics_bar.png', dpi=150)
    plt.close()
    print("Saved: comparison_metrics_bar.png")


def plot_training_curves(hist_resnet, hist_lstm):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    pairs = [
        ('accuracy',     'val_accuracy', 'Accuracy',  axes[0]),
        ('loss',         'val_loss',     'Loss',       axes[1]),
    ]
    model_data = [
        ('1D-ResNet',  hist_resnet, 'steelblue'),
        ('CNN-BiLSTM', hist_lstm,   'darkorange'),
    ]

    for (train_key, val_key, ylabel, ax_row), (model_name, hist, color) in (
        (p, m) for p in pairs for m in model_data
    ):
        ax = ax_row[0] if model_name == '1D-ResNet' else ax_row[1]
        ax.plot(hist.history[train_key],  label='Train',      color=color)
        ax.plot(hist.history[val_key],    label='Validation',  color=color, linestyle='--')
        ax.set_title(f'{ylabel} — {model_name}')
        ax.set_xlabel('Epoka')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Krzywe uczenia — porównanie modeli', fontsize=14)
    plt.tight_layout()
    plt.savefig('comparison_training_curves.png', dpi=150)
    plt.close()
    print("Saved: comparison_training_curves.png")


def print_summary_table(y_test, y_pred_resnet, y_pred_lstm, class_names):
    print("\n" + "=" * 60)
    print("PODSUMOWANIE PORÓWNANIA MODELI — DS2 (inter-patient)")
    print("=" * 60)
    for model_name, y_pred in [('1D-ResNet ', y_pred_resnet), ('CNN-BiLSTM', y_pred_lstm)]:
        p, r, f1, sup = precision_recall_fscore_support(
            y_test, y_pred, labels=range(len(class_names)), zero_division=0)
        macro_f1 = f1.mean()
        acc = (y_test == y_pred).mean()
        print(f"\n{model_name}  |  Accuracy: {acc:.4f}  |  Macro-F1: {macro_f1:.4f}")
        header = f"  {'Klasa':>6}  {'Precyzja':>9}  {'Czułość':>8}  {'F1':>6}  {'Wsparcie':>9}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for i, cn in enumerate(class_names):
            print(f"  {cn:>6}  {p[i]:>9.4f}  {r[i]:>8.4f}  {f1[i]:>6.4f}  {sup[i]:>9d}")
    print("=" * 60)


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

def main():
    import os
    os.makedirs('models', exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test, class_names, le = prepare_data()
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes  = len(class_names)

    # --- 1D-ResNet ---
    print("\n" + "=" * 60)
    print("TRENING: 1D-ResNet")
    print("=" * 60)
    resnet = create_ecg_model(input_shape=input_shape, num_classes=num_classes)
    resnet.summary()
    hist_resnet = train_one(resnet, X_train, X_val, y_train, y_val,
                            class_names, save_path='models/ecg_model.keras')

    # --- CNN-BiLSTM ---
    print("\n" + "=" * 60)
    print("TRENING: CNN-BiLSTM")
    print("=" * 60)
    lstm = create_lstm_model(input_shape=input_shape, num_classes=num_classes)
    lstm.summary()
    hist_lstm = train_one(lstm, X_train, X_val, y_train, y_val,
                          class_names, save_path='models/ecg_model_lstm.keras')

    # --- Predykcje na DS2 ---
    y_pred_resnet = np.argmax(resnet.predict(X_test, verbose=0), axis=1)
    y_pred_lstm   = np.argmax(lstm.predict(X_test,   verbose=0), axis=1)

    # --- Wykresy i tabela ---
    plot_confusion_matrices(y_test, y_pred_resnet, y_pred_lstm, class_names)
    plot_metrics_bar(y_test, y_pred_resnet, y_pred_lstm, class_names)
    plot_training_curves(hist_resnet, hist_lstm)
    print_summary_table(y_test, y_pred_resnet, y_pred_lstm, class_names)


if __name__ == '__main__':
    main()
