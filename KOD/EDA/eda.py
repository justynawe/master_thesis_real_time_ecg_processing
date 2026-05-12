# -*- coding: utf-8 -*-
"""
EDA - Eksploracyjna analiza danych
Zbior MIT-BIH Arrhythmia Database (przetworzony algorytmem Pan-Tompkinsa)
Podzial inter-patient AAMI EC57: DS1 (trening), DS2 (test)
"""
import sys, io

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
from scipy import signal as scipy_signal
import os

# Styl wykresów 
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
})
OUTPUT_DIR = 'C:\\Users\\Justyna\\Documents\\SGH_PRACA_MAGISTERSKA\\KOD\\eda_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

FS = 360   # częstotliwość próbkowania MIT-BIH

# Mapowanie AAMI
AAMI_MAPPING = {
    'Normal':             'N',
    'LBBB':               'N',
    'RBBB':               'N',
    'Atrial_escape':      'N',
    'Nodal_escape':       'N',
    'APC':                'S',
    'NPC':                'S',
    'PVC':                'V',
    'Ventricular_escape': 'V',
    'Fusion':             'F',
    'Paced':              'Q',
    'Fusion_paced':       'Q',
    'Unknown':            'Q',
}
CLASS_NAMES  = ['N', 'S', 'V', 'F', 'Q']
CLASS_LABELS = {
    'N': 'N – Normalny',
    'S': 'S – Nadkomorowy',
    'V': 'V – Komorowy',
    'F': 'F – Fuzja',
    'Q': 'Q – Stymulowany/Nieznany',
}
CLASS_COLORS = {
    'N': '#2196F3',
    'S': '#FF9800',
    'V': '#F44336',
    'F': '#9C27B0',
    'Q': '#4CAF50',
}

# Ładowanie danych

def load_npz(path):
    d = np.load(path)
    X        = d['X'].astype(np.float32)   # (N, 360, 2)
    y_raw    = d['y_strings']
    y_aami   = np.array([AAMI_MAPPING.get(lbl, None) for lbl in y_raw])
    mask     = np.isin(y_aami, CLASS_NAMES)
    return X[mask], y_aami[mask], y_raw[mask]

print("Wczytywanie danych...")
X_ds1, y_ds1, y_raw_ds1 = load_npz('C:\\Users\\Justyna\\Documents\\SGH_PRACA_MAGISTERSKA\\KOD\\qrs_classified_DS1.npz')
X_ds2, y_ds2, y_raw_ds2 = load_npz('C:\\Users\\Justyna\\Documents\\SGH_PRACA_MAGISTERSKA\\KOD\\qrs_classified_DS2.npz')

X_all   = np.concatenate([X_ds1, X_ds2], axis=0)
y_all   = np.concatenate([y_ds1, y_ds2], axis=0)
y_raw_all = np.concatenate([y_raw_ds1, y_raw_ds2], axis=0)

print(f"DS1: {X_ds1.shape[0]:,} pobudzen  |  DS2: {X_ds2.shape[0]:,} pobudzen  |  Lacznie: {X_all.shape[0]:,}")



# 1. ROZKŁAD KLAS

def plot_class_distribution():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Rozkład klas pobudzeń (standard AAMI EC57)', fontsize=14, fontweight='bold')

    datasets = [
        ('DS1 – trening/walidacja', y_ds1),
        ('DS2 – test',              y_ds2),
        ('Łącznie',                 y_all),
    ]

    for ax, (title, y) in zip(axes, datasets):
        counts  = {c: int(np.sum(y == c)) for c in CLASS_NAMES}
        total   = sum(counts.values())
        labels  = [CLASS_LABELS[c] for c in CLASS_NAMES]
        values  = [counts[c] for c in CLASS_NAMES]
        colors  = [CLASS_COLORS[c] for c in CLASS_NAMES]

        bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=0.8)
        ax.set_title(f'{title}\n(n = {total:,})', fontweight='bold')
        ax.set_ylabel('Liczba pobudzeń')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha='right')
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, values):
            pct = 100 * val / total
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + total * 0.005,
                    f'{val:,}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/01_class_distribution.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Zapisano: {path}')



# 2. ROZKŁAD SZCZEGÓŁOWYCH ETYKIET AAMI

def plot_raw_label_distribution():
    """Pokazuje oryginalne etykiety adnotacji i jak mapują się na klasy AAMI."""
    from collections import Counter
    cnt = Counter(y_raw_all)
    # Posortuj malejąco, pokaż tylko te z mapowaniem (wyklucz artefakty)
    rows = []
    for raw_lbl, count in sorted(cnt.items(), key=lambda x: -x[1]):
        aami = AAMI_MAPPING.get(raw_lbl, '–')
        rows.append({'Oryginalna etykieta': raw_lbl, 'Klasa AAMI': aami, 'Liczba': count})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors_bar = [CLASS_COLORS.get(r['Klasa AAMI'], '#BDBDBD') for _, r in df.iterrows()]
    bars = ax.barh(df['Oryginalna etykieta'], df['Liczba'], color=colors_bar, edgecolor='white')
    ax.set_xlabel('Liczba pobudzeń')
    ax.set_title('Szczegółowe etykiety MIT-BIH i ich przynależność do klas AAMI', fontweight='bold')
    ax.invert_yaxis()

    # Legenda AAMI
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=CLASS_COLORS[c], label=CLASS_LABELS[c]) for c in CLASS_NAMES]
    legend_elements.append(Patch(facecolor='#BDBDBD', label='– (spoza AAMI, wykluczone)'))
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    ax.grid(axis='x', alpha=0.3)

    # Liczby na słupkach
    for bar, val in zip(bars, df['Liczba']):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2,
                f'{val:,}', va='center', fontsize=9)

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/02_raw_label_distribution.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Zapisano: {path}')



# 3. PRZYKŁADOWE PRZEBIEGI QRS DLA KAŻDEJ KLASY

def plot_sample_waveforms(n_samples=5):
    time_ms = np.arange(360) / FS * 1000  # oś czasu w milisekundach

    fig, axes = plt.subplots(len(CLASS_NAMES), n_samples,
                             figsize=(3 * n_samples, 2.5 * len(CLASS_NAMES)),
                             sharex=True)
    fig.suptitle('Przykładowe przebiegi QRS – Lead I (odprowadzenie I)', fontsize=14, fontweight='bold')

    for row, cls in enumerate(CLASS_NAMES):
        idx = np.where(y_all == cls)[0]
        np.random.seed(42)
        chosen = np.random.choice(idx, size=min(n_samples, len(idx)), replace=False)
        for col, i in enumerate(chosen):
            ax = axes[row, col]
            ax.plot(time_ms, X_all[i, :, 0], color=CLASS_COLORS[cls], lw=1.2)
            ax.set_ylim(-2.5, 2.5)
            ax.grid(True, alpha=0.25)
            if col == 0:
                ax.set_ylabel(CLASS_LABELS[cls], fontsize=9, fontweight='bold')
            if row == len(CLASS_NAMES) - 1:
                ax.set_xlabel('Czas (ms)', fontsize=8)

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/03_sample_waveforms.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Zapisano: {path}')


# 4. ŚREDNI PRZEBIEG QRS Z PRZEDZIAŁEM UFNOŚCI

def plot_mean_waveforms():
    time_ms = np.arange(360) / FS * 1000

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Uśredniony przebieg QRS ± 1 odchylenie standardowe', fontsize=14, fontweight='bold')

    for ax, ch, ch_name in zip(axes, [0, 1], ['Lead I', 'Lead II']):
        for cls in CLASS_NAMES:
            windows = X_all[y_all == cls, :, ch]
            mean = windows.mean(axis=0)
            std  = windows.std(axis=0)
            ax.plot(time_ms, mean, color=CLASS_COLORS[cls],
                    label=CLASS_LABELS[cls], lw=2)
            ax.fill_between(time_ms, mean - std, mean + std,
                            color=CLASS_COLORS[cls], alpha=0.12)
        ax.set_title(ch_name)
        ax.set_xlabel('Czas (ms)')
        ax.set_ylabel('Amplituda (mV)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axvline(500, color='gray', linestyle=':', lw=1, alpha=0.6, label='Centrum (R)')

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/04_mean_waveforms.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Zapisano: {path}')


# 5. ROZKŁAD AMPLITUD (BOXPLOT)

def plot_amplitude_distributions():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Rozkład amplitud sygnału EKG per klasa AAMI', fontsize=14, fontweight='bold')

    for ax, ch, ch_name in zip(axes, [0, 1], ['Lead I', 'Lead II']):
        data_per_class = []
        labels_bp = []
        colors_bp = []
        for cls in CLASS_NAMES:
            windows = X_all[y_all == cls, :, ch]
            # Max amplituda w oknie jako cecha reprezentacyjna
            data_per_class.append(windows.max(axis=1))
            labels_bp.append(f'{cls}\n(n={len(windows):,})')
            colors_bp.append(CLASS_COLORS[cls])

        bp = ax.boxplot(data_per_class, patch_artist=True, notch=False,
                        medianprops=dict(color='black', linewidth=2),
                        flierprops=dict(marker='o', markersize=2, alpha=0.3))
        for patch, color in zip(bp['boxes'], colors_bp):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticklabels(labels_bp)
        ax.set_title(ch_name)
        ax.set_ylabel('Maksymalna amplituda w oknie (mV)')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/05_amplitude_distributions.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Zapisano: {path}')


# 6. TABELA STATYSTYK OPISOWYCH

def print_and_save_stats():
    rows = []
    for split_name, X, y in [('DS1', X_ds1, y_ds1), ('DS2', X_ds2, y_ds2)]:
        total = len(y)
        for cls in CLASS_NAMES:
            windows = X[y == cls, :, 0]  # Lead I
            n = len(windows)
            if n == 0:
                continue
            flat = windows.flatten()
            rows.append({
                'Zbiór':         split_name,
                'Klasa (AAMI)':  cls,
                'Pełna nazwa':   CLASS_LABELS[cls],
                'n':             n,
                '% zbioru':      round(100 * n / total, 2),
                'Śr. amplituda': round(float(flat.mean()), 4),
                'Odch. std':     round(float(flat.std()),  4),
                'Min':           round(float(flat.min()),  4),
                'Max':           round(float(flat.max()),  4),
            })

    df = pd.DataFrame(rows)
    print('\n=== Statystyki opisowe per klasa AAMI (Lead I) ===')
    print(df.to_string(index=False))

    csv_path = f'{OUTPUT_DIR}/stats_per_class.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'Zapisano: {csv_path}')
    return df


# 7. WIDMO CZĘSTOTLIWOŚCIOWE PER KLASA

def plot_frequency_spectra():
    time_ms = np.arange(360) / FS * 1000
    freqs   = np.fft.rfftfreq(360, d=1.0 / FS)   # 0..180 Hz

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Uśrednione widmo amplitudowe QRS per klasa AAMI (FFT)', fontsize=14, fontweight='bold')

    for ax, ch, ch_name in zip(axes, [0, 1], ['Lead I', 'Lead II']):
        for cls in CLASS_NAMES:
            windows = X_all[y_all == cls, :, ch]
            # Uśrednione widmo FFT (próbkujemy max 2000 okien żeby było szybko)
            sample = windows[:2000]
            windowed = sample * np.hanning(360)
            spectra  = np.abs(np.fft.rfft(windowed, axis=1))
            mean_sp  = spectra.mean(axis=0)
            ax.semilogy(freqs, mean_sp, color=CLASS_COLORS[cls],
                        label=CLASS_LABELS[cls], lw=2)

        ax.set_xlim(0, 60)
        ax.set_title(ch_name)
        ax.set_xlabel('Częstotliwość (Hz)')
        ax.set_ylabel('Amplituda (skala log)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axvspan(5, 15, color='yellow', alpha=0.1, label='Pasmo QRS (5–15 Hz)')

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/06_frequency_spectra.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Zapisano: {path}')


# 8. KORELACJA MIĘDZY ODPROWADZENIAMI

def plot_lead_correlation():
    fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(4 * len(CLASS_NAMES), 4), sharey=True)
    fig.suptitle('Korelacja między Lead I a Lead II per klasa AAMI\n(próbka 500 pobudzeń)', fontsize=13, fontweight='bold')

    for ax, cls in zip(axes, CLASS_NAMES):
        idx = np.where(y_all == cls)[0]
        np.random.seed(0)
        sample_idx = np.random.choice(idx, size=min(500, len(idx)), replace=False)
        # Dla każdego okna: max amplituda L1 vs max amplituda L2
        l1 = X_all[sample_idx, :, 0].max(axis=1)
        l2 = X_all[sample_idx, :, 1].max(axis=1)
        corr = np.corrcoef(l1, l2)[0, 1]

        ax.scatter(l1, l2, color=CLASS_COLORS[cls], alpha=0.4, s=15, edgecolors='none')
        ax.set_title(f'{CLASS_LABELS[cls]}\nr = {corr:.3f}')
        ax.set_xlabel('Lead I max (mV)')
        if ax == axes[0]:
            ax.set_ylabel('Lead II max (mV)')
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/07_lead_correlation.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Zapisano: {path}')


# 9. NIERÓWNOWAGA KLAS – WIZUALIZACJA PROPORCJI 

def plot_class_imbalance_pie():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Proporcje klas AAMI – nierównowaga zbioru danych', fontsize=14, fontweight='bold')

    for ax, (title, y) in zip(axes, [('DS1 – trening/walidacja', y_ds1), ('DS2 – test', y_ds2)]):
        counts = [np.sum(y == c) for c in CLASS_NAMES]
        colors = [CLASS_COLORS[c] for c in CLASS_NAMES]
        labels = [f'{CLASS_LABELS[c]}\n({v:,})' for c, v in zip(CLASS_NAMES, counts)]
        wedges, texts, autotexts = ax.pie(
            counts, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=140,
            wedgeprops=dict(edgecolor='white', linewidth=1.5),
            pctdistance=0.75
        )
        for at in autotexts:
            at.set_fontsize(9)
        ax.set_title(title, fontweight='bold')

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/08_class_imbalance_pie.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Zapisano: {path}')




if __name__ == '__main__':
    print('\n=== EDA - MIT-BIH Arrhythmia Database ===\n')

    print('[1/8] Rozklad klas...')
    plot_class_distribution()

    print('[2/8] Szczegolowe etykiety -> AAMI...')
    plot_raw_label_distribution()

    print('[3/8] Przykladowe przebiegi QRS...')
    plot_sample_waveforms(n_samples=5)

    print('[4/8] Usrednione przebiegi...')
    plot_mean_waveforms()

    print('[5/8] Rozklad amplitud...')
    plot_amplitude_distributions()

    print('[6/8] Statystyki opisowe...')
    df_stats = print_and_save_stats()

    print('[7/8] Widmo czestotliwosciowe...')
    plot_frequency_spectra()

    print('[8/8] Korelacja odprowadzen...')
    plot_lead_correlation()

    print('[9/8] Nierownosc klas (pie)...')
    plot_class_imbalance_pie()

    print(f'\nGotowe! Wykresy zapisane w folderze: {OUTPUT_DIR}/')
