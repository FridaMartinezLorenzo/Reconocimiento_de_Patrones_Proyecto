#!/usr/bin/env python3
"""
================================================================================
WESAD (Wearable Stress and Affect Detection) - Pipeline Completo de Ciencia de Datos
================================================================================

Actividades:
1. Extracción de características de señales fisiológicas y etiquetado
2. Análisis Exploratorio de Datos (EDA)
3. Preprocesamiento de datos
4. Ranking de características con Factor de Fisher
5. Selección de 5 mejores características por Factor de Fisher
6. Entrenamiento, evaluación y comparación de modelos
7. Cálculo manual — Naive Bayes y Decision Tree

Salidas adicionales:
  - Tabla descripción del dataset original
  - Tabla descripción del dataset tras extracción de características
  - Grafos y tablas de resultados por etapa (un .jpg por figura)

Dataset: https://www.kaggle.com/datasets/orvile/wesad-wearable-stress-affect-detection-dataset
================================================================================
"""

import kagglehub
import numpy as np
import pandas as pd
import pickle
import os
import json
import copy
import warnings
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
path = kagglehub.dataset_download(
    "orvile/wesad-wearable-stress-affect-detection-dataset"
)
print("Path raíz:", path)

DATA_PATH   = os.path.join(path, "WESAD")
SUBJECT_IDS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
OUTPUT_DIR  = './resultados/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

FS_CHEST      = 700
FS_BVP        = 64
FS_EDA_WRIST  = 4
FS_TEMP_WRIST = 4
FS_ACC_WRIST  = 32

WINDOW_SIZE  = 60
WINDOW_SHIFT = 30
LABEL_MAP    = {1: 0, 2: 1, 3: 0}
RANDOM_STATE = 42
TEST_SIZE    = 0.30

# ── Helper global: guardar figura como .jpg ──────────────────────────────────
def save_jpg(fig, filename):
    fpath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(fpath, dpi=150, bbox_inches='tight',
                format='jpg', pil_kwargs={'quality': 95})
    plt.close(fig)
    print(f"  [JPG] {filename}")


# ============================================================================
# ACTIVIDAD 1: EXTRACCIÓN DE CARACTERÍSTICAS Y ETIQUETADO
# ============================================================================
print("=" * 80)
print("ACTIVIDAD 1: EXTRACCIÓN DE CARACTERÍSTICAS DE SEÑALES Y ETIQUETADO")
print("=" * 80)


def load_subject_data(data_path, subject_id):
    subject_str = f'S{subject_id}'
    pkl_path = os.path.join(data_path, subject_str, f'{subject_str}.pkl')
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data


def compute_statistical_features(signal_data, prefix):
    features = {}
    if len(signal_data) == 0:
        return features
    if signal_data.ndim > 1:
        signal_data = signal_data.flatten()
    features[f'{prefix}_mean']      = np.mean(signal_data)
    features[f'{prefix}_std']       = np.std(signal_data)
    features[f'{prefix}_min']       = np.min(signal_data)
    features[f'{prefix}_max']       = np.max(signal_data)
    features[f'{prefix}_median']    = np.median(signal_data)
    features[f'{prefix}_range']     = np.max(signal_data) - np.min(signal_data)
    features[f'{prefix}_kurtosis']  = stats.kurtosis(signal_data)
    features[f'{prefix}_skewness']  = stats.skew(signal_data)
    features[f'{prefix}_q25']       = np.percentile(signal_data, 25)
    features[f'{prefix}_q75']       = np.percentile(signal_data, 75)
    features[f'{prefix}_iqr']       = features[f'{prefix}_q75'] - features[f'{prefix}_q25']
    features[f'{prefix}_rms']       = np.sqrt(np.mean(signal_data ** 2))
    zc = np.sum(np.diff(np.sign(signal_data - np.mean(signal_data))) != 0)
    features[f'{prefix}_zcr']       = zc / len(signal_data)
    return features


def compute_frequency_features(signal_data, fs, prefix):
    features = {}
    if len(signal_data) < 4:
        return features
    if signal_data.ndim > 1:
        signal_data = signal_data.flatten()
    N  = len(signal_data)
    yf = np.abs(fft(signal_data - np.mean(signal_data)))[:N // 2]
    xf = fftfreq(N, 1 / fs)[:N // 2]
    if len(yf) > 0 and np.sum(yf) > 0:
        features[f'{prefix}_peak_freq']       = xf[np.argmax(yf)]
        features[f'{prefix}_spectral_energy'] = np.sum(yf ** 2) / N
        psd = yf ** 2 / np.sum(yf ** 2)
        psd = psd[psd > 0]
        features[f'{prefix}_spectral_entropy'] = -np.sum(psd * np.log2(psd))
    return features


def compute_eda_features(eda_signal, fs, prefix='EDA'):
    features = compute_statistical_features(eda_signal, prefix)
    features.update(compute_frequency_features(eda_signal, fs, prefix))
    if eda_signal.ndim > 1:
        eda_signal = eda_signal.flatten()
    if len(eda_signal) > 10:
        try:
            b, a  = signal.butter(2, 0.05 / (fs / 2), btype='low')
            scl   = signal.filtfilt(b, a, eda_signal)
            features[f'{prefix}_scl_mean'] = np.mean(scl)
            features[f'{prefix}_scl_std']  = np.std(scl)
            scr   = eda_signal - scl
            features[f'{prefix}_scr_mean'] = np.mean(scr)
            features[f'{prefix}_scr_std']  = np.std(scr)
            peaks, _ = signal.find_peaks(scr, height=np.std(scr) * 0.5)
            features[f'{prefix}_scr_num_peaks'] = len(peaks)
        except Exception:
            pass
    if len(eda_signal) > 1:
        d = np.diff(eda_signal)
        features[f'{prefix}_deriv_mean'] = np.mean(d)
        features[f'{prefix}_deriv_std']  = np.std(d)
    return features


def compute_ecg_features(ecg_signal, fs=700, prefix='ECG'):
    features = compute_statistical_features(ecg_signal, prefix)
    features.update(compute_frequency_features(ecg_signal, fs, prefix))
    if ecg_signal.ndim > 1:
        ecg_signal = ecg_signal.flatten()
    try:
        b, a = signal.butter(4, [0.5 / (fs / 2), 40 / (fs / 2)], btype='band')
        ecg_f = signal.filtfilt(b, a, ecg_signal)
        peaks, _ = signal.find_peaks(
            ecg_f,
            height=np.mean(ecg_f) + 0.5 * np.std(ecg_f),
            distance=int(0.5 * fs)
        )
        if len(peaks) > 2:
            rr = np.diff(peaks) / fs
            features[f'{prefix}_hr_mean'] = 60.0 / np.mean(rr)
            features[f'{prefix}_hr_std']  = np.std(60.0 / rr)
            features[f'{prefix}_rr_mean'] = np.mean(rr)
            features[f'{prefix}_rr_std']  = np.std(rr)
            features[f'{prefix}_rmssd']   = np.sqrt(np.mean(np.diff(rr) ** 2))
            nn50 = np.sum(np.abs(np.diff(rr)) > 0.05)
            features[f'{prefix}_pnn50']   = nn50 / len(rr)
        else:
            features[f'{prefix}_hr_mean'] = 0
            features[f'{prefix}_hr_std']  = 0
    except Exception:
        pass
    return features


def compute_acc_features(acc_data, fs, prefix='ACC'):
    features = {}
    if acc_data.ndim == 1:
        acc_data = acc_data.reshape(-1, 1)
    axes = ['x', 'y', 'z'] if acc_data.shape[1] >= 3 else [str(i) for i in range(acc_data.shape[1])]
    for i, axis in enumerate(axes[:acc_data.shape[1]]):
        features.update(compute_statistical_features(acc_data[:, i], f'{prefix}_{axis}'))
    if acc_data.shape[1] >= 3:
        mag = np.sqrt(np.sum(acc_data[:, :3] ** 2, axis=1))
        features.update(compute_statistical_features(mag, f'{prefix}_mag'))
        features.update(compute_frequency_features(mag, fs, f'{prefix}_mag'))
    return features


def compute_resp_features(resp_signal, fs=700, prefix='RESP'):
    features = compute_statistical_features(resp_signal, prefix)
    features.update(compute_frequency_features(resp_signal, fs, prefix))
    if resp_signal.ndim > 1:
        resp_signal = resp_signal.flatten()
    try:
        peaks, _ = signal.find_peaks(resp_signal, distance=int(fs * 1.5))
        if len(peaks) > 1:
            bi = np.diff(peaks) / fs
            features[f'{prefix}_rate_mean'] = 60.0 / np.mean(bi)
            features[f'{prefix}_rate_std']  = np.std(60.0 / bi)
            features[f'{prefix}_insp_time'] = np.mean(bi)
    except Exception:
        pass
    return features


def compute_temp_features(temp_signal, fs, prefix='TEMP'):
    features = compute_statistical_features(temp_signal, prefix)
    if temp_signal.ndim > 1:
        temp_signal = temp_signal.flatten()
    if len(temp_signal) > 1:
        x = np.arange(len(temp_signal))
        slope, *_ = stats.linregress(x, temp_signal)
        features[f'{prefix}_slope']      = slope
        td = np.gradient(temp_signal)
        features[f'{prefix}_deriv_mean'] = np.mean(td)
        features[f'{prefix}_deriv_std']  = np.std(td)
    return features


def compute_emg_features(emg_signal, fs=700, prefix='EMG'):
    features = compute_statistical_features(emg_signal, prefix)
    features.update(compute_frequency_features(emg_signal, fs, prefix))
    if emg_signal.ndim > 1:
        emg_signal = emg_signal.flatten()
    features[f'{prefix}_mav'] = np.mean(np.abs(emg_signal))
    features[f'{prefix}_var'] = np.var(emg_signal)
    if len(emg_signal) > 1:
        features[f'{prefix}_wl'] = np.sum(np.abs(np.diff(emg_signal)))
    return features


def extract_features_window(chest_data, wrist_data, labels_window):
    all_features = {}
    if 'ECG'  in chest_data and len(chest_data['ECG'])  > 0:
        all_features.update(compute_ecg_features(chest_data['ECG'],  FS_CHEST, 'c_ECG'))
    if 'EDA'  in chest_data and len(chest_data['EDA'])  > 0:
        all_features.update(compute_eda_features(chest_data['EDA'],  FS_CHEST, 'c_EDA'))
    if 'EMG'  in chest_data and len(chest_data['EMG'])  > 0:
        all_features.update(compute_emg_features(chest_data['EMG'],  FS_CHEST, 'c_EMG'))
    if 'Resp' in chest_data and len(chest_data['Resp']) > 0:
        all_features.update(compute_resp_features(chest_data['Resp'], FS_CHEST, 'c_RESP'))
    if 'Temp' in chest_data and len(chest_data['Temp']) > 0:
        all_features.update(compute_temp_features(chest_data['Temp'], FS_CHEST, 'c_TEMP'))
    if 'ACC'  in chest_data and len(chest_data['ACC'])  > 0:
        all_features.update(compute_acc_features(chest_data['ACC'],  FS_CHEST, 'c_ACC'))
    if 'BVP'  in wrist_data and len(wrist_data['BVP'])  > 0:
        all_features.update(compute_statistical_features(wrist_data['BVP'], 'w_BVP'))
        all_features.update(compute_frequency_features(wrist_data['BVP'], FS_BVP, 'w_BVP'))
    if 'EDA'  in wrist_data and len(wrist_data['EDA'])  > 0:
        all_features.update(compute_eda_features(wrist_data['EDA'],  FS_EDA_WRIST, 'w_EDA'))
    if 'TEMP' in wrist_data and len(wrist_data['TEMP']) > 0:
        all_features.update(compute_temp_features(wrist_data['TEMP'], FS_TEMP_WRIST, 'w_TEMP'))
    if 'ACC'  in wrist_data and len(wrist_data['ACC'])  > 0:
        all_features.update(compute_acc_features(wrist_data['ACC'],  FS_ACC_WRIST, 'w_ACC'))
    all_features['label'] = stats.mode(labels_window, keepdims=True)[0][0]
    return all_features


def extract_features_subject(data, subject_id):
    labels         = data['label'].flatten()
    chest_signals  = data['signal']['chest']
    wrist_signals  = data['signal']['wrist']
    chest_window   = WINDOW_SIZE  * FS_CHEST
    chest_shift    = WINDOW_SHIFT * FS_CHEST
    n_samples      = len(labels)
    all_features_list = []
    start = 0
    wc    = 0
    while start + chest_window <= n_samples:
        end   = start + chest_window
        lw    = labels[start:end]
        valid = lw[(lw == 1) | (lw == 2) | (lw == 3)]
        if len(valid) < 0.8 * len(lw):
            start += chest_shift
            continue
        cwd = {}
        for key in chest_signals:
            sig = chest_signals[key]
            cwd[key] = sig[start:end]
        t0, t1 = start / FS_CHEST, end / FS_CHEST
        wwd = {}
        for key in wrist_signals:
            sig = wrist_signals[key]
            if key == 'BVP':
                ws, we = int(t0 * FS_BVP), int(t1 * FS_BVP)
            elif key in ['EDA', 'TEMP']:
                ws, we = int(t0 * FS_EDA_WRIST), int(t1 * FS_EDA_WRIST)
            elif key == 'ACC':
                ws, we = int(t0 * FS_ACC_WRIST), int(t1 * FS_ACC_WRIST)
            else:
                ws, we = 0, 0
            wwd[key] = sig[ws:we] if we <= len(sig) else sig[ws:]
        feat = extract_features_window(cwd, wwd, lw)
        feat['subject_id'] = subject_id
        all_features_list.append(feat)
        wc    += 1
        start += chest_shift
    print(f"  Sujeto S{subject_id}: {wc} ventanas procesadas")
    return all_features_list


def generate_synthetic_wesad_data():
    print("\n*** MODO SINTÉTICO ***\n")
    np.random.seed(42)
    all_features = []
    for sid in SUBJECT_IDS:
        so = np.random.normal(0, 0.1)
        for condition, n_windows, label in [
            ('baseline',  np.random.randint(12, 18), 1),
            ('stress',    np.random.randint(8,  14), 2),
            ('amusement', np.random.randint(8,  12), 3),
        ]:
            for _ in range(n_windows):
                feat  = {}
                noise = np.random.normal(0, 0.05)
                hr_base  = 95 + np.random.normal(0, 10) if condition == 'stress' else 72 + np.random.normal(0, 8)
                hrv_base = 0.03 if condition == 'stress' else 0.06
                eda_base = 8.0 + np.random.normal(0, 2) if condition == 'stress' else 3.0 + np.random.normal(0, 1.5)
                scr_peaks = np.random.randint(5, 15) if condition == 'stress' else np.random.randint(0, 5)
                emg_base  = 0.05 + np.random.normal(0, 0.015) if condition == 'stress' else 0.02 + np.random.normal(0, 0.008)
                resp_rate = 22 + np.random.normal(0, 3) if condition == 'stress' else 16 + np.random.normal(0, 2)
                temp_base = 34.5 + np.random.normal(0, 0.3) if condition == 'stress' else 35.0 + np.random.normal(0, 0.3)

                # ECG
                feat.update({
                    'c_ECG_mean': 0.02+np.random.normal(0,0.005)+so*0.01,
                    'c_ECG_std': 0.15+np.random.normal(0,0.02),
                    'c_ECG_min': -0.5+np.random.normal(0,0.1),
                    'c_ECG_max': 1.0+np.random.normal(0,0.2),
                    'c_ECG_median': 0.01+np.random.normal(0,0.005),
                    'c_ECG_range': 1.5+np.random.normal(0,0.2),
                    'c_ECG_kurtosis': 3.0+np.random.normal(0,1),
                    'c_ECG_skewness': 0.5+np.random.normal(0,0.3),
                    'c_ECG_q25': -0.05+np.random.normal(0,0.01),
                    'c_ECG_q75': 0.08+np.random.normal(0,0.01),
                    'c_ECG_iqr': 0.13+np.random.normal(0,0.01),
                    'c_ECG_rms': 0.15+np.random.normal(0,0.02),
                    'c_ECG_zcr': 0.1+np.random.normal(0,0.02),
                    'c_ECG_peak_freq': 1.2+np.random.normal(0,0.2),
                    'c_ECG_spectral_energy': 0.005+np.random.normal(0,0.001),
                    'c_ECG_spectral_entropy': 5.0+np.random.normal(0,0.5),
                    'c_ECG_hr_mean': hr_base+so*5,
                    'c_ECG_hr_std': 5.0+np.random.normal(0,1.5),
                    'c_ECG_rr_mean': 60.0/hr_base,
                    'c_ECG_rr_std': hrv_base+np.random.normal(0,0.01),
                    'c_ECG_rmssd': hrv_base*1.2+np.random.normal(0,0.005),
                    'c_ECG_pnn50': (0.08+np.random.normal(0,0.03)) if condition=='stress' else (0.2+np.random.normal(0,0.05)),
                })
                # EDA chest
                feat.update({
                    'c_EDA_mean': eda_base+so,
                    'c_EDA_std': eda_base*0.2+np.random.normal(0,0.3),
                    'c_EDA_min': eda_base*0.5+np.random.normal(0,0.2),
                    'c_EDA_max': eda_base*1.5+np.random.normal(0,0.5),
                    'c_EDA_median': eda_base+np.random.normal(0,0.2),
                    'c_EDA_range': eda_base+np.random.normal(0,0.3),
                    'c_EDA_kurtosis': 2.5+np.random.normal(0,1),
                    'c_EDA_skewness': 0.3+np.random.normal(0,0.2),
                    'c_EDA_q25': eda_base*0.8+np.random.normal(0,0.2),
                    'c_EDA_q75': eda_base*1.2+np.random.normal(0,0.2),
                    'c_EDA_iqr': eda_base*0.4+np.random.normal(0,0.1),
                    'c_EDA_rms': eda_base*1.05+np.random.normal(0,0.3),
                    'c_EDA_zcr': 0.01+np.random.normal(0,0.005),
                    'c_EDA_peak_freq': 0.05+np.random.normal(0,0.02),
                    'c_EDA_spectral_energy': eda_base**2*0.01+np.random.normal(0,0.01),
                    'c_EDA_spectral_entropy': 4.0+np.random.normal(0,0.5),
                    'c_EDA_scl_mean': eda_base*0.9+np.random.normal(0,0.2),
                    'c_EDA_scl_std': 0.3+np.random.normal(0,0.1),
                    'c_EDA_scr_mean': 0.1+np.random.normal(0,0.05),
                    'c_EDA_scr_std': 0.5+np.random.normal(0,0.1),
                    'c_EDA_scr_num_peaks': scr_peaks,
                    'c_EDA_deriv_mean': 0.001+np.random.normal(0,0.0005),
                    'c_EDA_deriv_std': 0.01+np.random.normal(0,0.003),
                })
                # EMG
                feat.update({
                    'c_EMG_mean': emg_base+noise,
                    'c_EMG_std': emg_base*2.0+np.random.normal(0,0.01),
                    'c_EMG_min': -emg_base*5+np.random.normal(0,0.02),
                    'c_EMG_max': emg_base*5+np.random.normal(0,0.02),
                    'c_EMG_median': emg_base*0.1+np.random.normal(0,0.005),
                    'c_EMG_range': emg_base*10+np.random.normal(0,0.04),
                    'c_EMG_kurtosis': 5.0+np.random.normal(0,2),
                    'c_EMG_skewness': 0.1+np.random.normal(0,0.3),
                    'c_EMG_q25': -emg_base+np.random.normal(0,0.005),
                    'c_EMG_q75': emg_base+np.random.normal(0,0.005),
                    'c_EMG_iqr': emg_base*2+np.random.normal(0,0.01),
                    'c_EMG_rms': emg_base*1.5+np.random.normal(0,0.005),
                    'c_EMG_zcr': 0.4+np.random.normal(0,0.05),
                    'c_EMG_peak_freq': 50+np.random.normal(0,15),
                    'c_EMG_spectral_energy': emg_base**2+np.random.normal(0,0.001),
                    'c_EMG_spectral_entropy': 6.0+np.random.normal(0,0.5),
                    'c_EMG_mav': np.abs(emg_base)+np.random.normal(0,0.005),
                    'c_EMG_var': emg_base**2*3+np.random.normal(0,0.001),
                    'c_EMG_wl': emg_base*500+np.random.normal(0,20),
                })
                # RESP
                feat.update({
                    'c_RESP_mean': 0+np.random.normal(0,0.1),
                    'c_RESP_std': 200+np.random.normal(0,50),
                    'c_RESP_min': -500+np.random.normal(0,100),
                    'c_RESP_max': 500+np.random.normal(0,100),
                    'c_RESP_median': 0+np.random.normal(0,0.1),
                    'c_RESP_range': 1000+np.random.normal(0,150),
                    'c_RESP_kurtosis': 2.0+np.random.normal(0,0.5),
                    'c_RESP_skewness': 0.0+np.random.normal(0,0.2),
                    'c_RESP_q25': -150+np.random.normal(0,30),
                    'c_RESP_q75': 150+np.random.normal(0,30),
                    'c_RESP_iqr': 300+np.random.normal(0,50),
                    'c_RESP_rms': 200+np.random.normal(0,50),
                    'c_RESP_zcr': 0.005+np.random.normal(0,0.001),
                    'c_RESP_peak_freq': resp_rate/60+np.random.normal(0,0.02),
                    'c_RESP_spectral_energy': 10000+np.random.normal(0,2000),
                    'c_RESP_spectral_entropy': 4.5+np.random.normal(0,0.5),
                    'c_RESP_rate_mean': resp_rate,
                    'c_RESP_rate_std': 2.0+np.random.normal(0,0.5),
                    'c_RESP_insp_time': 60.0/resp_rate+np.random.normal(0,0.2),
                })
                # TEMP chest
                feat.update({
                    'c_TEMP_mean': temp_base+so*0.5,
                    'c_TEMP_std': 0.05+np.random.normal(0,0.01),
                    'c_TEMP_min': temp_base-0.1+np.random.normal(0,0.02),
                    'c_TEMP_max': temp_base+0.1+np.random.normal(0,0.02),
                    'c_TEMP_median': temp_base+np.random.normal(0,0.02),
                    'c_TEMP_range': 0.2+np.random.normal(0,0.03),
                    'c_TEMP_kurtosis': 2.0+np.random.normal(0,0.5),
                    'c_TEMP_skewness': 0.0+np.random.normal(0,0.1),
                    'c_TEMP_q25': temp_base-0.03+np.random.normal(0,0.01),
                    'c_TEMP_q75': temp_base+0.03+np.random.normal(0,0.01),
                    'c_TEMP_iqr': 0.06+np.random.normal(0,0.01),
                    'c_TEMP_rms': temp_base+np.random.normal(0,0.02),
                    'c_TEMP_zcr': 0.001+np.random.normal(0,0.0005),
                    'c_TEMP_slope': 0.0001+np.random.normal(0,0.00005),
                    'c_TEMP_deriv_mean': 0.0001+np.random.normal(0,0.0001),
                    'c_TEMP_deriv_std': 0.001+np.random.normal(0,0.0003),
                })
                # ACC chest
                feat.update({
                    'c_ACC_x_mean': 0.9+np.random.normal(0,0.05),
                    'c_ACC_x_std': 0.05+np.random.normal(0,0.01),
                    'c_ACC_y_mean': -0.2+np.random.normal(0,0.05),
                    'c_ACC_y_std': 0.04+np.random.normal(0,0.01),
                    'c_ACC_z_mean': -0.3+np.random.normal(0,0.05),
                    'c_ACC_z_std': 0.04+np.random.normal(0,0.01),
                    'c_ACC_mag_mean': 1.0+np.random.normal(0,0.03),
                    'c_ACC_mag_std': 0.05+np.random.normal(0,0.01),
                    'c_ACC_mag_peak_freq': 0.5+np.random.normal(0,0.2),
                    'c_ACC_mag_spectral_energy': 0.01+np.random.normal(0,0.003),
                    'c_ACC_mag_spectral_entropy': 3.0+np.random.normal(0,0.5),
                })
                # BVP wrist
                w_eda = eda_base*0.6+np.random.normal(0,0.3)
                w_tmp = temp_base-2+np.random.normal(0,0.3)
                feat.update({
                    'w_BVP_mean': 0.0+np.random.normal(0,0.5),
                    'w_BVP_std': 50+np.random.normal(0,15),
                    'w_BVP_min': -150+np.random.normal(0,40),
                    'w_BVP_max': 150+np.random.normal(0,40),
                    'w_BVP_median': 0+np.random.normal(0,0.5),
                    'w_BVP_range': 300+np.random.normal(0,60),
                    'w_BVP_kurtosis': 2.0+np.random.normal(0,0.8),
                    'w_BVP_skewness': 0.0+np.random.normal(0,0.3),
                    'w_BVP_q25': -30+np.random.normal(0,10),
                    'w_BVP_q75': 30+np.random.normal(0,10),
                    'w_BVP_iqr': 60+np.random.normal(0,15),
                    'w_BVP_rms': 50+np.random.normal(0,15),
                    'w_BVP_zcr': 0.1+np.random.normal(0,0.02),
                    'w_BVP_peak_freq': hr_base/60+np.random.normal(0,0.05),
                    'w_BVP_spectral_energy': 5000+np.random.normal(0,1500),
                    'w_BVP_spectral_entropy': 4.0+np.random.normal(0,0.5),
                })
                # EDA wrist
                feat.update({
                    'w_EDA_mean': w_eda,
                    'w_EDA_std': w_eda*0.15+np.random.normal(0,0.1),
                    'w_EDA_min': w_eda*0.6+np.random.normal(0,0.1),
                    'w_EDA_max': w_eda*1.4+np.random.normal(0,0.2),
                    'w_EDA_median': w_eda+np.random.normal(0,0.1),
                    'w_EDA_range': w_eda*0.8+np.random.normal(0,0.1),
                    'w_EDA_kurtosis': 2.5+np.random.normal(0,0.8),
                    'w_EDA_skewness': 0.3+np.random.normal(0,0.2),
                    'w_EDA_q25': w_eda*0.85+np.random.normal(0,0.1),
                    'w_EDA_q75': w_eda*1.15+np.random.normal(0,0.1),
                    'w_EDA_iqr': w_eda*0.3+np.random.normal(0,0.05),
                    'w_EDA_rms': w_eda*1.02+np.random.normal(0,0.1),
                    'w_EDA_zcr': 0.05+np.random.normal(0,0.02),
                    'w_EDA_peak_freq': 0.03+np.random.normal(0,0.01),
                    'w_EDA_spectral_energy': w_eda**2*0.01+np.random.normal(0,0.01),
                    'w_EDA_spectral_entropy': 3.5+np.random.normal(0,0.5),
                    'w_EDA_scl_mean': w_eda*0.95+np.random.normal(0,0.1),
                    'w_EDA_scl_std': 0.15+np.random.normal(0,0.05),
                    'w_EDA_scr_mean': 0.05+np.random.normal(0,0.02),
                    'w_EDA_scr_std': 0.2+np.random.normal(0,0.05),
                    'w_EDA_scr_num_peaks': max(0, scr_peaks-np.random.randint(0,3)),
                    'w_EDA_deriv_mean': 0.001+np.random.normal(0,0.0005),
                    'w_EDA_deriv_std': 0.005+np.random.normal(0,0.002),
                })
                # TEMP wrist
                feat.update({
                    'w_TEMP_mean': w_tmp,
                    'w_TEMP_std': 0.03+np.random.normal(0,0.01),
                    'w_TEMP_min': w_tmp-0.08+np.random.normal(0,0.02),
                    'w_TEMP_max': w_tmp+0.08+np.random.normal(0,0.02),
                    'w_TEMP_median': w_tmp+np.random.normal(0,0.02),
                    'w_TEMP_range': 0.16+np.random.normal(0,0.03),
                    'w_TEMP_kurtosis': 2.0+np.random.normal(0,0.5),
                    'w_TEMP_skewness': 0.0+np.random.normal(0,0.1),
                    'w_TEMP_q25': w_tmp-0.02+np.random.normal(0,0.01),
                    'w_TEMP_q75': w_tmp+0.02+np.random.normal(0,0.01),
                    'w_TEMP_iqr': 0.04+np.random.normal(0,0.01),
                    'w_TEMP_rms': w_tmp+np.random.normal(0,0.02),
                    'w_TEMP_zcr': 0.001+np.random.normal(0,0.0005),
                    'w_TEMP_slope': 0.0001+np.random.normal(0,0.00005),
                    'w_TEMP_deriv_mean': 0.0001+np.random.normal(0,0.0001),
                    'w_TEMP_deriv_std': 0.0005+np.random.normal(0,0.0002),
                })
                # ACC wrist
                feat.update({
                    'w_ACC_x_mean': 0.0+np.random.normal(0,0.1),
                    'w_ACC_x_std': 0.1+np.random.normal(0,0.03),
                    'w_ACC_y_mean': -0.5+np.random.normal(0,0.1),
                    'w_ACC_y_std': 0.1+np.random.normal(0,0.03),
                    'w_ACC_z_mean': -0.8+np.random.normal(0,0.1),
                    'w_ACC_z_std': 0.08+np.random.normal(0,0.02),
                    'w_ACC_mag_mean': 1.0+np.random.normal(0,0.05),
                    'w_ACC_mag_std': 0.1+np.random.normal(0,0.03),
                    'w_ACC_mag_peak_freq': 0.5+np.random.normal(0,0.2),
                    'w_ACC_mag_spectral_energy': 0.02+np.random.normal(0,0.005),
                    'w_ACC_mag_spectral_entropy': 3.0+np.random.normal(0,0.5),
                })
                feat['label']      = label
                feat['subject_id'] = sid
                all_features.append(feat)
    return pd.DataFrame(all_features)


# ── Cargar / generar datos ───────────────────────────────────────────────────
print("\nIntentando cargar el dataset WESAD...")
try:
    test_path = os.path.join(DATA_PATH, 'S2', 'S2.pkl')
    if not os.path.exists(test_path):
        raise FileNotFoundError
    print("Dataset WESAD encontrado. Procesando datos reales...")
    all_features_list = []
    for sid in SUBJECT_IDS:
        try:
            data = load_subject_data(DATA_PATH, sid)
            all_features_list.extend(extract_features_subject(data, sid))
        except Exception as e:
            print(f"  Error S{sid}: {e}")
    df_features = pd.DataFrame(all_features_list)
    USE_SYNTHETIC = False
except FileNotFoundError:
    df_features   = generate_synthetic_wesad_data()
    USE_SYNTHETIC = True

df_features['label_binary'] = df_features['label'].map(LABEL_MAP)
df_features = df_features.dropna(subset=['label_binary'])
df_features['label_binary'] = df_features['label_binary'].astype(int)
df_features.to_csv(os.path.join(OUTPUT_DIR, 'wesad_features.csv'), index=False)

feature_cols = [c for c in df_features.columns if c not in ['label', 'label_binary', 'subject_id']]
X = df_features[feature_cols].copy()
y = df_features['label_binary'].copy()

print(f"\nMuestras totales : {len(df_features)}")
print(f"Características  : {len(feature_cols)}")
print(f"Sujetos          : {df_features['subject_id'].nunique()}")


# ============================================================================
# TABLA DE RANGOS DE CARACTERÍSTICAS (post-extracción, pre-preprocesado)
# ============================================================================
print("\n" + "=" * 80)
print("TABLA DE RANGOS DE CARACTERÍSTICAS (dataset extraído sin preprocesar)")
print("=" * 80)

stress_X   = X[y == 1]
nostress_X = X[y == 0]

range_rows = []
for col in feature_cols:
    col_all = X[col].dropna()
    col_s   = stress_X[col].dropna()
    col_ns  = nostress_X[col].dropna()

    if   col.startswith('c_ECG') : sensor = 'ECG (chest)'
    elif col.startswith('c_EDA') : sensor = 'EDA (chest)'
    elif col.startswith('c_EMG') : sensor = 'EMG (chest)'
    elif col.startswith('c_RESP'): sensor = 'RESP (chest)'
    elif col.startswith('c_TEMP'): sensor = 'TEMP (chest)'
    elif col.startswith('c_ACC') : sensor = 'ACC (chest)'
    elif col.startswith('w_BVP') : sensor = 'BVP (wrist)'
    elif col.startswith('w_EDA') : sensor = 'EDA (wrist)'
    elif col.startswith('w_TEMP'): sensor = 'TEMP (wrist)'
    elif col.startswith('w_ACC') : sensor = 'ACC (wrist)'
    else                         : sensor = 'Otro'

    range_rows.append({
        'Sensor'         : sensor,
        'Característica' : col,
        'Min_global'     : round(float(col_all.min()),    6),
        'Max_global'     : round(float(col_all.max()),    6),
        'Media_global'   : round(float(col_all.mean()),   6),
        'Std_global'     : round(float(col_all.std()),    6),
        'Mediana_global' : round(float(col_all.median()), 6),
        'IQR_global'     : round(float(col_all.quantile(0.75) - col_all.quantile(0.25)), 6),
        'Min_NoStress'   : round(float(col_ns.min()),    6),
        'Max_NoStress'   : round(float(col_ns.max()),    6),
        'Media_NoStress' : round(float(col_ns.mean()),   6),
        'Std_NoStress'   : round(float(col_ns.std()),    6),
        'Min_Stress'     : round(float(col_s.min()),     6),
        'Max_Stress'     : round(float(col_s.max()),     6),
        'Media_Stress'   : round(float(col_s.mean()),    6),
        'Std_Stress'     : round(float(col_s.std()),     6),
    })

ranges_df = pd.DataFrame(range_rows)

# Guardar CSV
ranges_csv_path = os.path.join(OUTPUT_DIR, 'feature_ranges.csv')
ranges_df.to_csv(ranges_csv_path, index=False)
print(f"  CSV guardado: feature_ranges.csv  ({len(ranges_df)} características)")


def _darken(hex_color, factor=0.94):
    hex_color = hex_color.lstrip('#')
    r, g, b = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
    r = min(255, int(r * factor + 255 * (1 - factor)))
    g = min(255, int(g * factor + 255 * (1 - factor)))
    b = min(255, int(b * factor + 255 * (1 - factor)))
    return f'#{r:02x}{g:02x}{b:02x}'


def save_range_table_page(df_page, sensor_name, page_num, total_pages, filename):
    cols_display = [
        'Característica',
        'Min_global', 'Max_global', 'Media_global', 'Std_global',
        'Min_NoStress', 'Max_NoStress', 'Media_NoStress',
        'Min_Stress',   'Max_Stress',   'Media_Stress',
    ]
    col_headers_short = [
        'Característica',
        'Min', 'Max', 'Media', 'Std',
        'Min\nNo-S', 'Max\nNo-S', 'Media\nNo-S',
        'Min\nStress', 'Max\nStress', 'Media\nStress',
    ]
    n_rows = len(df_page)
    fig_h  = max(3.5, 0.38 * n_rows + 1.6)
    fig, ax = plt.subplots(figsize=(20, fig_h))
    ax.axis('off')

    cell_data = []
    for _, row in df_page.iterrows():
        cell_row = []
        for c in cols_display:
            v = row[c]
            if isinstance(v, float):
                cell_row.append(f'{v:.4e}' if abs(v) < 0.001 and v != 0 else f'{v:.4f}')
            else:
                cell_row.append(str(v))
        cell_data.append(cell_row)

    tbl = ax.table(cellText=cell_data, colLabels=col_headers_short,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.45)

    for j in range(len(col_headers_short)):
        tbl[(0, j)].set_facecolor('#2c3e50')
        tbl[(0, j)].set_text_props(color='white', fontweight='bold')

    col_groups = {
        0: '#fdfefe',
        1: '#eaf4fb', 2: '#eaf4fb', 3: '#eaf4fb', 4: '#eaf4fb',
        5: '#eafaf1', 6: '#eafaf1', 7: '#eafaf1',
        8: '#fdedec', 9: '#fdedec', 10: '#fdedec',
    }
    for i in range(1, n_rows + 1):
        for j in range(len(col_headers_short)):
            base = col_groups.get(j, '#fdfefe')
            tbl[(i, j)].set_facecolor(base if i % 2 == 1 else _darken(base))

    tbl.auto_set_column_width(list(range(len(col_headers_short))))
    page_label = f'  Página {page_num}/{total_pages}' if total_pages > 1 else ''
    ax.set_title(
        f'Rangos de Características — {sensor_name}{page_label}\n'
        f'[Global | No-Stress | Stress]  (Azul=Global, Verde=No-Stress, Rojo=Stress)',
        fontsize=12, fontweight='bold', pad=10
    )
    plt.tight_layout()
    save_jpg(fig, filename)


# Una figura JPG por sensor
print("\nGenerando tablas de rangos por sensor...")
sensor_order = [
    'ECG (chest)', 'EDA (chest)', 'EMG (chest)', 'RESP (chest)',
    'TEMP (chest)', 'ACC (chest)',
    'BVP (wrist)', 'EDA (wrist)', 'TEMP (wrist)', 'ACC (wrist)',
]
MAX_ROWS_PER_PAGE = 25

for sensor in sensor_order:
    df_sensor = ranges_df[ranges_df['Sensor'] == sensor].copy()
    if df_sensor.empty:
        continue
    pages       = [df_sensor.iloc[i:i + MAX_ROWS_PER_PAGE]
                   for i in range(0, len(df_sensor), MAX_ROWS_PER_PAGE)]
    total_pages = len(pages)
    slug        = sensor.replace(' ', '_').replace('(', '').replace(')', '')
    for pi, page_df in enumerate(pages, 1):
        suffix = f'_p{pi}' if total_pages > 1 else ''
        save_range_table_page(page_df, sensor, pi, total_pages,
                              f'tabla_rangos_{slug}{suffix}.jpg')

# Tabla resumen global (todas las features, paginada)
print("\nGenerando tabla resumen de rangos (todas las features)...")
df_sum = ranges_df[['Sensor', 'Característica',
                    'Min_global', 'Max_global', 'Media_global', 'Std_global',
                    'Media_NoStress', 'Media_Stress']].copy()

MAX_SUMMARY_ROWS = 35
pages_sum = [df_sum.iloc[i:i + MAX_SUMMARY_ROWS]
             for i in range(0, len(df_sum), MAX_SUMMARY_ROWS)]
total_sum = len(pages_sum)

sensor_palette = {
    'ECG (chest)' : '#d6eaf8', 'EDA (chest)' : '#d5f5e3',
    'EMG (chest)' : '#fef9e7', 'RESP (chest)': '#fdebd0',
    'TEMP (chest)': '#f9ebea', 'ACC (chest)' : '#e8daef',
    'BVP (wrist)' : '#d1f2eb', 'EDA (wrist)' : '#d6eaf8',
    'TEMP (wrist)': '#fdfefe', 'ACC (wrist)' : '#eafaf1',
}

for pi, page_df in enumerate(pages_sum, 1):
    n_rows  = len(page_df)
    fig_h   = max(4, 0.35 * n_rows + 1.5)
    fig, ax = plt.subplots(figsize=(18, fig_h))
    ax.axis('off')

    col_hdr   = ['Sensor', 'Característica',
                 'Min', 'Max', 'Media', 'Std',
                 'Media\nNo-Stress', 'Media\nStress']
    cell_data = []
    for _, row in page_df.iterrows():
        cell_row = [row['Sensor'], row['Característica']]
        for c in ['Min_global', 'Max_global', 'Media_global', 'Std_global',
                  'Media_NoStress', 'Media_Stress']:
            v = row[c]
            cell_row.append(f'{v:.4e}' if abs(v) < 0.001 and v != 0 else f'{v:.4f}')
        cell_data.append(cell_row)

    tbl = ax.table(cellText=cell_data, colLabels=col_hdr,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.4)

    for j in range(len(col_hdr)):
        tbl[(0, j)].set_facecolor('#2c3e50')
        tbl[(0, j)].set_text_props(color='white', fontweight='bold')

    prev_sensor = None
    alt = False
    for i in range(1, n_rows + 1):
        s = page_df.iloc[i - 1]['Sensor']
        if s != prev_sensor:
            alt = not alt
            prev_sensor = s
        base = sensor_palette.get(s, '#fdfefe')
        for j in range(len(col_hdr)):
            tbl[(i, j)].set_facecolor(base if alt else _darken(base, 0.97))
        # Resaltar si diferencia >50% entre clases
        v_ns = page_df.iloc[i - 1]['Media_NoStress']
        v_s  = page_df.iloc[i - 1]['Media_Stress']
        if abs(v_s - v_ns) > 0.5 * (abs(v_ns) + 1e-10):
            tbl[(i, 7)].set_facecolor('#f1948a')
            tbl[(i, 6)].set_facecolor('#aed6f1')

    suffix = f' — Parte {pi}/{total_sum}' if total_sum > 1 else ''
    ax.set_title(
        f'Tabla de Rangos de Características (post-extracción){suffix}\n'
        f'Color por sensor | Destacado: diferencia >50% entre clases (azul=No-Stress, rojo=Stress)',
        fontsize=12, fontweight='bold', pad=10
    )
    plt.tight_layout()
    save_jpg(fig, f'tabla_rangos_resumen_p{pi}.jpg')

print("  Tablas de rangos generadas correctamente.")


# ============================================================================
# TABLA 1 — DESCRIPCIÓN DEL DATASET ORIGINAL
# ============================================================================
print("\n" + "=" * 80)
print("TABLA 1: DESCRIPCIÓN DEL DATASET ORIGINAL (señales crudas)")
print("=" * 80)

dataset_original_info = [
    ['Sujetos',                  '15 (S2–S17, sin S1 ni S12)',  '—'],
    ['Dispositivo chest',        'RespiBAN Professional',        '700 Hz'],
    ['Dispositivo wrist',        'Empatica E4',                  'Variable por señal'],
    ['ECG (chest)',              '1 canal',                      '700 Hz'],
    ['EDA (chest)',              '1 canal',                      '700 Hz'],
    ['EMG (chest)',              '1 canal',                      '700 Hz'],
    ['Respiración (chest)',      '1 canal',                      '700 Hz'],
    ['Temperatura (chest)',      '1 canal',                      '700 Hz'],
    ['Acelerómetro (chest)',     '3 ejes (X, Y, Z)',             '700 Hz'],
    ['BVP (wrist)',              '1 canal',                      '64 Hz'],
    ['EDA (wrist)',              '1 canal',                      '4 Hz'],
    ['Temperatura (wrist)',      '1 canal',                      '4 Hz'],
    ['Acelerómetro (wrist)',     '3 ejes (X, Y, Z)',             '32 Hz'],
    ['Etiquetas',                '0=no def, 1=baseline, 2=stress, 3=amusement, 4=meditación', '700 Hz'],
    ['Duración aprox. por sujeto', '~2 horas',                  '—'],
    ['Protocolo TSST (estrés)',  'Tarea de aritmética + discurso público', '—'],
    ['Formato almacenamiento',   'Archivo .pkl por sujeto',      '—'],
]

fig, ax = plt.subplots(figsize=(16, 7))
ax.axis('off')
col_labels = ['Componente / Señal', 'Descripción', 'Frecuencia de muestreo']
tbl = ax.table(
    cellText=dataset_original_info,
    colLabels=col_labels,
    loc='center', cellLoc='left'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.55)
# Estilo cabecera
for j in range(len(col_labels)):
    tbl[(0, j)].set_facecolor('#2c3e50')
    tbl[(0, j)].set_text_props(color='white', fontweight='bold')
# Filas alternas
for i in range(1, len(dataset_original_info) + 1):
    for j in range(len(col_labels)):
        tbl[(i, j)].set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
ax.set_title('Tabla 1 — Descripción del Dataset Original WESAD',
             fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
save_jpg(fig, 'tabla1_descripcion_dataset_original.jpg')


# ============================================================================
# ACTIVIDAD 2: ANÁLISIS EXPLORATORIO DE DATOS
# ============================================================================
print("\n" + "=" * 80)
print("ACTIVIDAD 2: ANÁLISIS EXPLORATORIO DE DATOS")
print("=" * 80)

# Estadísticas descriptivas
print(f"\nShape: {df_features.shape}")
missing    = X.isnull().sum()
missing_pct = (missing / len(X)) * 100
inf_count  = np.isinf(X.select_dtypes(include=[np.number])).sum()
inf_cols   = inf_count[inf_count > 0]
class_counts = y.value_counts()
class_ratio  = class_counts.min() / class_counts.max()

print(f"\nDistribución binaria: {dict(class_counts)}")
print(f"Ratio minoría/mayoría: {class_ratio:.3f}")

# Diferencias por clase (t-test)
stress_data    = X[y == 1]
no_stress_data = X[y == 0]
class_diffs    = {}
for col in feature_cols:
    try:
        t_stat, p_val = stats.ttest_ind(
            stress_data[col].dropna(),
            no_stress_data[col].dropna()
        )
        class_diffs[col] = {
            'stress_mean'   : stress_data[col].mean(),
            'no_stress_mean': no_stress_data[col].mean(),
            'diff_pct'      : abs(stress_data[col].mean() - no_stress_data[col].mean()) /
                              (abs(no_stress_data[col].mean()) + 1e-10) * 100,
            't_stat'        : t_stat,
            'p_value'       : p_val
        }
    except Exception:
        pass

diff_df = pd.DataFrame(class_diffs).T.sort_values('p_value')

# ── EDA Figura 1: distribución de clases ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
label_names = {1: 'Baseline', 2: 'Stress', 3: 'Amusement'}
orig_counts = df_features['label'].value_counts()
colors_orig = ['#3498db', '#e74c3c', '#2ecc71']
axes[0].bar([label_names.get(x, str(x)) for x in orig_counts.index],
            orig_counts.values, color=colors_orig[:len(orig_counts)])
axes[0].set_title('Distribución de Clases Originales', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Número de muestras')
for i, v in enumerate(orig_counts.values):
    axes[0].text(i, v + 1, str(v), ha='center', fontweight='bold')

bin_counts = y.value_counts()
bin_names  = {0: 'No-Stress', 1: 'Stress'}
axes[1].bar([bin_names[x] for x in bin_counts.index],
            bin_counts.values, color=['#3498db', '#e74c3c'])
axes[1].set_title('Distribución Binaria', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Número de muestras')
for i, v in enumerate(bin_counts.values):
    axes[1].text(i, v + 1, str(v), ha='center', fontweight='bold')
plt.tight_layout()
save_jpg(fig, 'fig1_class_distribution.jpg')

# ── EDA Figura 2: boxplots top-6 ─────────────────────────────────────────────
top_features = diff_df.head(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for i, feat in enumerate(top_features):
    data_plot = pd.DataFrame({'value': X[feat], 'class': y.map({0:'No-Stress',1:'Stress'})})
    sns.boxplot(data=data_plot, x='class', y='value', ax=axes[i],
                palette=['#3498db','#e74c3c'])
    p_val = diff_df.loc[feat, 'p_value']
    axes[i].set_title(f'{feat}\n(p={p_val:.2e})', fontsize=9, fontweight='bold')
    axes[i].set_xlabel('')
plt.suptitle('Top 6 Características Más Discriminativas (Stress vs No-Stress)',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
save_jpg(fig, 'fig2_top_features_boxplot.jpg')

# ── EDA Figura 4: distribuciones por clase ───────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for i, feat in enumerate(top_features):
    for cls, color, lbl in [(0,'#3498db','No-Stress'),(1,'#e74c3c','Stress')]:
        axes[i].hist(X.loc[y==cls, feat].dropna(), bins=30, alpha=0.5,
                     color=color, label=lbl, density=True)
    axes[i].set_title(feat, fontsize=9, fontweight='bold')
    axes[i].legend(fontsize=8)
    axes[i].set_ylabel('Densidad')
plt.suptitle('Distribuciones de Top Features por Clase',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
save_jpg(fig, 'fig4_distributions_by_class.jpg')

# ── EDA Figura 5: valores faltantes ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
if X.isnull().sum().sum() > 0:
    sns.heatmap(X.isnull().astype(int).T, cbar=False, cmap='YlOrRd', ax=ax)
    ax.set_title('Mapa de Valores Faltantes', fontsize=13, fontweight='bold')
else:
    ax.text(0.5, 0.5, 'No hay valores faltantes en el dataset',
            ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.set_title('Mapa de Valores Faltantes', fontsize=13, fontweight='bold')
plt.tight_layout()
save_jpg(fig, 'fig5_missing_values.jpg')


# ============================================================================
# ACTIVIDAD 3: PREPROCESAMIENTO
# ============================================================================
print("\n" + "=" * 80)
print("ACTIVIDAD 3: PREPROCESAMIENTO DE DATOS")
print("=" * 80)

X_processed = X.copy()
preprocessing_log = {}

# 3.1 Valores faltantes
n_miss_before = X_processed.isnull().sum().sum()
for col in X_processed.columns:
    if X_processed[col].isnull().any():
        X_processed[col].fillna(X_processed[col].median(), inplace=True)
preprocessing_log['valores_faltantes_imputados'] = int(n_miss_before)

# 3.2 Valores infinitos
n_inf_before = np.isinf(X_processed.select_dtypes(include=[np.number])).sum().sum()
X_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
for col in X_processed.columns:
    if X_processed[col].isnull().any():
        X_processed[col].fillna(X_processed[col].median(), inplace=True)
preprocessing_log['valores_infinitos_reemplazados'] = int(n_inf_before)

# 3.3 Varianza cero
zero_var_cols = X_processed.columns[X_processed.var() < 1e-10].tolist()
X_processed.drop(columns=zero_var_cols, inplace=True)
preprocessing_log['cols_var_cero_eliminadas'] = len(zero_var_cols)

# 3.4 Alta correlación
corr_matrix_proc = X_processed.corr().abs()
upper_tri  = corr_matrix_proc.where(np.triu(np.ones(corr_matrix_proc.shape), k=1).astype(bool))
to_drop_corr = [c for c in upper_tri.columns if any(upper_tri[c] > 0.98)]
X_processed.drop(columns=to_drop_corr, inplace=True)
preprocessing_log['cols_alta_corr_eliminadas'] = len(to_drop_corr)

# 3.5 Winsorización
n_clipped = 0
for col in X_processed.columns:
    p1, p99 = X_processed[col].quantile(0.01), X_processed[col].quantile(0.99)
    n_clipped += ((X_processed[col] < p1) | (X_processed[col] > p99)).sum()
    X_processed[col] = X_processed[col].clip(p1, p99)
preprocessing_log['valores_winsorizados'] = int(n_clipped)

# 3.6 Estandarización
scaler   = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_processed),
                         columns=X_processed.columns, index=X_processed.index)
preprocessing_log['features_finales'] = len(X_processed.columns)

remaining_features = X_processed.columns.tolist()
print(f"Features originales : {len(feature_cols)}")
print(f"Features finales    : {len(remaining_features)}")
print(f"Muestras            : {len(X_scaled)}")


# ============================================================================
# TABLA 2 — DESCRIPCIÓN DEL DATASET TRAS EXTRACCIÓN DE CARACTERÍSTICAS
# ============================================================================
print("\n" + "=" * 80)
print("TABLA 2: DESCRIPCIÓN DEL DATASET TRAS EXTRACCIÓN DE CARACTERÍSTICAS")
print("=" * 80)

# Contar features por sensor
sensor_prefixes = {
    'ECG (chest)'   : 'c_ECG',
    'EDA (chest)'   : 'c_EDA',
    'EMG (chest)'   : 'c_EMG',
    'RESP (chest)'  : 'c_RESP',
    'TEMP (chest)'  : 'c_TEMP',
    'ACC (chest)'   : 'c_ACC',
    'BVP (wrist)'   : 'w_BVP',
    'EDA (wrist)'   : 'w_EDA',
    'TEMP (wrist)'  : 'w_TEMP',
    'ACC (wrist)'   : 'w_ACC',
}
feat_counts_orig   = {k: sum(1 for c in feature_cols        if c.startswith(v))
                      for k, v in sensor_prefixes.items()}
feat_counts_final  = {k: sum(1 for c in remaining_features  if c.startswith(v))
                      for k, v in sensor_prefixes.items()}

n_stress    = int((y == 1).sum())
n_no_stress = int((y == 0).sum())

desc_info = [
    ['Método de ventaneo',          f'{WINDOW_SIZE}s ventana, {WINDOW_SHIFT}s desplazamiento (50% overlap)'],
    ['Total de ventanas (muestras)', str(len(df_features))],
    ['Sujetos procesados',          str(df_features['subject_id'].nunique())],
    ['Etiquetado',                  'Voto mayoritario por ventana'],
    ['Muestras Stress (clase 1)',   f'{n_stress} ({n_stress/len(y)*100:.1f}%)'],
    ['Muestras No-Stress (clase 0)', f'{n_no_stress} ({n_no_stress/len(y)*100:.1f}%)'],
    ['Total features extraídas',    str(len(feature_cols))],
    ['Features tras preprocesado',  str(len(remaining_features))],
    ['Valores faltantes imputados', str(preprocessing_log['valores_faltantes_imputados'])],
    ['Valores infinitos tratados',  str(preprocessing_log['valores_infinitos_reemplazados'])],
    ['Cols. varianza ~0 eliminadas',str(preprocessing_log['cols_var_cero_eliminadas'])],
    ['Cols. alta corr. eliminadas', str(preprocessing_log['cols_alta_corr_eliminadas'])],
    ['Valores winsorizados (1-99%)', str(preprocessing_log['valores_winsorizados'])],
    ['Normalización',               'StandardScaler (media=0, std=1)'],
]
for sensor, cnt_orig in feat_counts_orig.items():
    cnt_fin = feat_counts_final[sensor]
    desc_info.append([f'Features {sensor}', f'{cnt_orig} extraídas → {cnt_fin} tras preprocesado'])

fig, ax = plt.subplots(figsize=(14, len(desc_info) * 0.52 + 1.2))
ax.axis('off')
col_labels_t2 = ['Aspecto', 'Valor / Descripción']
tbl2 = ax.table(
    cellText=desc_info,
    colLabels=col_labels_t2,
    loc='center', cellLoc='left'
)
tbl2.auto_set_font_size(False)
tbl2.set_fontsize(9)
tbl2.scale(1, 1.5)
for j in range(2):
    tbl2[(0, j)].set_facecolor('#2c3e50')
    tbl2[(0, j)].set_text_props(color='white', fontweight='bold')
for i in range(1, len(desc_info) + 1):
    for j in range(2):
        tbl2[(i, j)].set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
ax.set_title('Tabla 2 — Descripción del Dataset Tras Extracción de Características',
             fontsize=13, fontweight='bold', pad=10)
plt.tight_layout()
save_jpg(fig, 'tabla2_descripcion_dataset_features.jpg')


# ============================================================================
# ACTIVIDAD 4: RANKING FACTOR DE FISHER
# ============================================================================
print("\n" + "=" * 80)
print("ACTIVIDAD 4: RANKING DE CARACTERÍSTICAS — FACTOR DE FISHER")
print("=" * 80)


def fisher_score(X_data, y_data):
    """
    F(A_k) = Σ_j [ p_j * (μ_{k,j} − μ_k)² ]
             ──────────────────────────────────
             Σ_j [ p_j * σ_{k,j}² ]
    """
    X_arr = X_data.values if hasattr(X_data, 'values') else np.array(X_data)
    y_arr = y_data.values if hasattr(y_data, 'values') else np.array(y_data)
    classes    = np.unique(y_arr)
    N          = len(y_arr)
    n_features = X_arr.shape[1]
    scores     = np.zeros(n_features)
    for k in range(n_features):
        fv  = X_arr[:, k]
        mu  = np.mean(fv)
        num = den = 0.0
        for c in classes:
            mask = y_arr == c
            cv   = fv[mask]
            pj   = mask.sum() / N
            num += pj * (np.mean(cv) - mu) ** 2
            den += pj * np.var(cv)
        scores[k] = num / den if den > 1e-10 else 0.0
    return scores


fisher_scores = fisher_score(X_scaled, y)
fisher_df = pd.DataFrame({
    'Característica': X_scaled.columns,
    'Fisher_Score'  : fisher_scores
}).sort_values('Fisher_Score', ascending=False).reset_index(drop=True)
fisher_df['Rank'] = range(1, len(fisher_df) + 1)
fisher_df.to_csv(os.path.join(OUTPUT_DIR, 'fisher_ranking.csv'), index=False)

print(f"\nTop 10 características:")
print(fisher_df[['Rank','Característica','Fisher_Score']].head(10).to_string(index=False))

# Figura: ranking Fisher
fig, ax = plt.subplots(figsize=(13, 8))
top_n       = min(25, len(fisher_df))
top_fisher  = fisher_df.head(top_n)
colors_fish = plt.cm.RdYlGn_r(np.linspace(0, 1, top_n))
bars = ax.barh(range(top_n), top_fisher['Fisher_Score'].values, color=colors_fish)
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_fisher['Característica'].values, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('Fisher Score', fontsize=12)
ax.set_title(f'Top {top_n} Características — Factor de Fisher',
             fontsize=13, fontweight='bold')
mx = max(top_fisher['Fisher_Score'])
for bar, val in zip(bars, top_fisher['Fisher_Score'].values):
    ax.text(bar.get_width() + mx * 0.01, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=7)
plt.tight_layout()
save_jpg(fig, 'fig6_fisher_ranking.jpg')


# ============================================================================
# ACTIVIDAD 5: SELECCIÓN TOP-5 POR FISHER
# ============================================================================
print("\n" + "=" * 80)
print("ACTIVIDAD 5: SELECCIÓN TOP-5 POR FACTOR DE FISHER")
print("=" * 80)

N_SELECT         = 5
selected_features = fisher_df['Característica'].head(N_SELECT).tolist()

print(f"\n{'Rank':<6}{'Característica':<40}{'Fisher Score':<15}")
print("-" * 61)
for _, row in fisher_df.head(N_SELECT).iterrows():
    print(f"{int(row['Rank']):<6}{row['Característica']:<40}{row['Fisher_Score']:.6f}")

# Validación cruzada rápida: top-5 vs todas
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
for clf_name, clf in [('KNN (k=5)', KNeighborsClassifier(n_neighbors=5)),
                       ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE))]:
    sc_sel = cross_val_score(clf, X_scaled[selected_features], y, cv=cv, scoring='accuracy')
    sc_all = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')
    print(f"\n{clf_name}:")
    print(f"  Top-5  : {sc_sel.mean():.4f} ± {sc_sel.std():.4f}")
    print(f"  Todas  : {sc_all.mean():.4f} ± {sc_all.std():.4f}")

# Figura: top-5 seleccionadas
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

fisher_sel = fisher_df.head(N_SELECT)['Fisher_Score'].tolist()
colors_v   = plt.cm.viridis(np.linspace(0.2, 0.8, N_SELECT))
bars_s     = axes[0].barh(range(N_SELECT), fisher_sel, color=colors_v)
axes[0].set_yticks(range(N_SELECT))
axes[0].set_yticklabels(selected_features, fontsize=10)
axes[0].invert_yaxis()
axes[0].set_xlabel('Factor de Fisher', fontsize=12)
axes[0].set_title(f'Top {N_SELECT} Seleccionadas por Fisher', fontsize=13, fontweight='bold')
axes[0].grid(True, axis='x', alpha=0.3)
mx_f = max(fisher_sel) if max(fisher_sel) > 0 else 1
for bar, val in zip(bars_s, fisher_sel):
    axes[0].text(bar.get_width() + mx_f * 0.02, bar.get_y() + bar.get_height()/2,
                 f'{val:.4f}', va='center', fontsize=9)

top_n2 = min(25, len(fisher_df))
top_f2 = fisher_df.head(top_n2)
bar_colors2 = ['#e74c3c' if f in selected_features else '#7f8c8d'
               for f in top_f2['Característica']]
axes[1].barh(range(top_n2), top_f2['Fisher_Score'].values, color=bar_colors2)
axes[1].set_yticks(range(top_n2))
axes[1].set_yticklabels(top_f2['Característica'].values, fontsize=8)
axes[1].invert_yaxis()
axes[1].set_xlabel('Factor de Fisher', fontsize=12)
axes[1].set_title(f'Ranking Top-{top_n2}  (rojo = seleccionadas)',
                  fontsize=13, fontweight='bold')
axes[1].grid(True, axis='x', alpha=0.3)
plt.tight_layout()
save_jpg(fig, 'fig7_fisher_selection_results.jpg')

# Guardar dataset final
X_final       = X_scaled[selected_features].copy()
X_final['label'] = y.values
X_final.to_csv(os.path.join(OUTPUT_DIR, 'wesad_final_5features.csv'), index=False)


# ============================================================================
# MATRICES DE CORRELACIÓN — SEPARADAS
# ============================================================================
print("\nGenerando matrices de correlación...")

# F1: Top-20 discriminativas
top20_features = diff_df.head(20).index.tolist()
corr_top20     = X[top20_features].corr()
mask_top20     = np.triu(np.ones_like(corr_top20, dtype=bool))
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(corr_top20, mask=mask_top20, cmap='RdBu_r', center=0,
            annot=True, fmt='.2f', square=True, ax=ax,
            linewidths=0.5, cbar_kws={"shrink": 0.8}, annot_kws={"size": 7})
ax.set_title('Matriz de Correlaciones — Top 20 Features (ranking t-test)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
save_jpg(fig, 'fig3_corr_top20_discriminativas.jpg')

# F2: Top-5 Fisher
corr_top5 = X[selected_features].corr()
fig, ax   = plt.subplots(figsize=(7, 6))
sns.heatmap(corr_top5, cmap='RdBu_r', center=0, annot=True, fmt='.3f',
            square=True, ax=ax, linewidths=1.0, cbar_kws={"shrink": 0.8},
            annot_kws={"size": 10})
ax.set_title('Matriz de Correlaciones — Top-5 Features (Fisher)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
save_jpg(fig, 'fig3_corr_top5_fisher.jpg')

# F3: Por sensor
sensor_groups = {
    'ECG_chest'  : [c for c in X_scaled.columns if c.startswith('c_ECG')],
    'EDA_chest'  : [c for c in X_scaled.columns if c.startswith('c_EDA')],
    'EMG_chest'  : [c for c in X_scaled.columns if c.startswith('c_EMG')],
    'RESP_chest' : [c for c in X_scaled.columns if c.startswith('c_RESP')],
    'TEMP_chest' : [c for c in X_scaled.columns if c.startswith('c_TEMP')],
    'ACC_chest'  : [c for c in X_scaled.columns if c.startswith('c_ACC')],
    'BVP_wrist'  : [c for c in X_scaled.columns if c.startswith('w_BVP')],
    'EDA_wrist'  : [c for c in X_scaled.columns if c.startswith('w_EDA')],
    'TEMP_wrist' : [c for c in X_scaled.columns if c.startswith('w_TEMP')],
    'ACC_wrist'  : [c for c in X_scaled.columns if c.startswith('w_ACC')],
}
for sg_name, sg_cols in sensor_groups.items():
    if len(sg_cols) < 2:
        continue
    corr_sg   = X_scaled[sg_cols].corr()
    mask_sg   = np.triu(np.ones_like(corr_sg, dtype=bool))
    n         = len(sg_cols)
    fsz       = max(5, n * 0.7)
    fig, ax   = plt.subplots(figsize=(fsz, fsz * 0.9))
    sns.heatmap(corr_sg, mask=mask_sg, cmap='RdBu_r', center=0,
                annot=True, fmt='.2f', square=True, ax=ax,
                linewidths=0.5, cbar_kws={"shrink": 0.8},
                annot_kws={"size": max(6, 10 - n // 3)})
    ax.set_title(f'Correlaciones — {sg_name.replace("_"," ")}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    save_jpg(fig, f'fig3_corr_sensor_{sg_name}.jpg')

# F4: Todas las features
corr_all  = X_scaled.corr()
n_af      = len(X_scaled.columns)
fw = min(28, max(14, n_af * 0.25))
fh = min(24, max(12, n_af * 0.22))
fig, ax   = plt.subplots(figsize=(fw, fh))
mask_all  = np.triu(np.ones_like(corr_all, dtype=bool))
sns.heatmap(corr_all, mask=mask_all, cmap='RdBu_r', center=0,
            annot=False, square=False, ax=ax,
            linewidths=0, cbar_kws={"shrink": 0.6})
ax.set_title('Matriz de Correlaciones — Todas las Features Preprocesadas',
             fontsize=13, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=5, rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=5, rotation=0)
plt.tight_layout()
save_jpg(fig, 'fig3_corr_todas_features.jpg')


# ============================================================================
# ACTIVIDAD 6: ENTRENAMIENTO Y COMPARACIÓN DE MODELOS
# ============================================================================
print("\n" + "=" * 80)
print("ACTIVIDAD 6: ENTRENAMIENTO Y COMPARACIÓN DE MODELOS")
print("=" * 80)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer

# 6.1 Split
X_all  = X_scaled
X_top5 = X_scaled[selected_features]

X_all_train,  X_all_test,  y_train, y_test = train_test_split(
    X_all, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
X_top5_train, X_top5_test, _, _            = train_test_split(
    X_top5, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

# Imputación anti data leakage
for X_tr, X_te in [(X_all_train, X_all_test), (X_top5_train, X_top5_test)]:
    imp = SimpleImputer(strategy='mean')
    X_tr[:] = imp.fit_transform(X_tr)
    X_te[:] = imp.transform(X_te)

print(f"Train: {len(y_train)} | Test: {len(y_test)}")

# 6.2 Clasificadores
classifiers = {
    'Random Forest' : RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    'Naive Bayes'   : GaussianNB(),
    'Decision Tree' : DecisionTreeClassifier(random_state=RANDOM_STATE),
}


def evaluate_model(clf, X_tr, X_te, y_tr, y_te):
    clf.fit(X_tr, y_tr)
    y_pred  = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)
    return {
        'accuracy' : accuracy_score(y_te, y_pred),
        'precision': precision_score(y_te, y_pred, zero_division=0),
        'recall'   : recall_score(y_te, y_pred, zero_division=0),
        'f1'       : f1_score(y_te, y_pred, zero_division=0),
        'y_pred'   : y_pred,
        'y_proba'  : y_proba,
        'clf'      : clf,
    }


all_results = {}
for clf_name, clf_proto in classifiers.items():
    all_results[clf_name] = {
        'all' : evaluate_model(copy.deepcopy(clf_proto),
                               X_all_train,  X_all_test,  y_train, y_test),
        'top5': evaluate_model(copy.deepcopy(clf_proto),
                               X_top5_train, X_top5_test, y_train, y_test),
    }

# ── Consola: tabla resumen ───────────────────────────────────────────────────
rows = []
print(f"\n{'Modelo':<38} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
print("-" * 62)
for clf_name, results in all_results.items():
    for key, res in results.items():
        tag   = "Todas" if key == "all" else "Top-5"
        label = f"{clf_name} [{tag}]"
        rows.append({'label': label, **{k: res[k]
                     for k in ['accuracy','precision','recall','f1']}})
        print(f"  {label:<36} {res['accuracy']:>7.4f} {res['precision']:>7.4f} "
              f"{res['recall']:>7.4f} {res['f1']:>7.4f}")

y_test_arr = np.array(y_test)
class_names = ['No-Stress', 'Stress']
colors_cls  = ['#2ecc71', '#e74c3c']
metric_lbls = ['Precision', 'Recall', 'F1-Score']
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

# ── A) Matriz de confusión — una figura por modelo × configuración ───────────
print("\nGenerando matrices de confusión...")
for clf_name, results in all_results.items():
    clf_slug = clf_name.replace(' ', '_')
    for key, res in results.items():
        tag_label = "Todas" if key == "all" else "Top5"
        tag_title = "Todas las features" if key == "all" else "Top-5 Fisher"
        fig, ax   = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(y_test, res['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=0.5, cbar=False)
        ax.set_title(f"Matriz de Confusión\n{clf_name}  [{tag_title}]",
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicho', fontsize=10)
        ax.set_ylabel('Real', fontsize=10)
        plt.tight_layout()
        save_jpg(fig, f"fig8_confmat_{clf_slug}_{tag_label}.jpg")

# ── B) Barras comparativas (Todas vs Top-5) — una figura por modelo ──────────
print("\nGenerando barras comparativas...")
for clf_name, results in all_results.items():
    clf_slug = clf_name.replace(' ', '_')
    mv_all  = [results['all'][m]  for m in ['accuracy','precision','recall','f1']]
    mv_top5 = [results['top5'][m] for m in ['accuracy','precision','recall','f1']]
    fig, ax = plt.subplots(figsize=(7, 5))
    x     = np.arange(len(metrics_names))
    width = 0.35
    b1 = ax.bar(x - width/2, mv_all,  width, label='Todas',      color='#3498db', alpha=0.85)
    b2 = ax.bar(x + width/2, mv_top5, width, label='Top-5 Fisher', color='#e74c3c', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel('Valor', fontsize=11)
    ax.set_title(f"Comparativa de Métricas\n{clf_name}", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    save_jpg(fig, f"fig8_barras_{clf_slug}.jpg")

# ── C) Tabla de deltas — una figura por modelo ───────────────────────────────
print("\nGenerando tablas delta...")
for clf_name, results in all_results.items():
    clf_slug = clf_name.replace(' ', '_')
    mv_all  = [results['all'][m]  for m in ['accuracy','precision','recall','f1']]
    mv_top5 = [results['top5'][m] for m in ['accuracy','precision','recall','f1']]
    deltas  = [round(t - a, 4) for t, a in zip(mv_top5, mv_all)]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.axis('off')
    tbl = ax.table(
        cellText=[[m, f'{a:.4f}', f'{t:.4f}', f'{d:+.4f}']
                  for m, a, t, d in zip(metrics_names, mv_all, mv_top5, deltas)],
        colLabels=['Métrica', 'Todas', 'Top-5', 'Δ'],
        loc='center', cellLoc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.3, 1.8)
    for j in range(4):
        tbl[(0, j)].set_facecolor('#2c3e50')
        tbl[(0, j)].set_text_props(color='white', fontweight='bold')
    for i, d in enumerate(deltas):
        tbl[(i + 1, 3)].set_facecolor('#d5f5e3' if d >= 0 else '#fadbd8')
    ax.set_title(f"Δ Top-5 vs Todas\n{clf_name}", fontsize=11, fontweight='bold', pad=16)
    plt.tight_layout()
    save_jpg(fig, f"fig8_delta_{clf_slug}.jpg")

# ── D) Métricas por clase — una figura por modelo × configuración ─────────────
print("\nGenerando métricas por clase...")
for clf_name, results in all_results.items():
    clf_slug = clf_name.replace(' ', '_')
    for fkey, flabel in [('all', 'Todas las features'), ('top5', 'Top-5 Fisher')]:
        tag_label = "Todas" if fkey == "all" else "Top5"
        res  = results[fkey]
        y_pd = res['y_pred']
        pp   = precision_score(y_test, y_pd, average=None, zero_division=0)
        rp   = recall_score(y_test,    y_pd, average=None, zero_division=0)
        fp   = f1_score(y_test,        y_pd, average=None, zero_division=0)
        vbc  = [pp, rp, fp]
        fig, ax = plt.subplots(figsize=(7, 5))
        x     = np.arange(len(metric_lbls))
        width = 0.28
        for ci, (cname, ccolor) in enumerate(zip(class_names, colors_cls)):
            vals   = [vbc[mi][ci] for mi in range(3)]
            offset = (ci - 0.5) * width
            bars   = ax.bar(x + offset, vals, width, label=cname, color=ccolor, alpha=0.82)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_lbls, fontsize=11)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel('Valor', fontsize=10)
        ax.set_title(f"Métricas por Clase\n{clf_name}  [{flabel}]",
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        save_jpg(fig, f"fig8b_por_clase_{clf_slug}_{tag_label}.jpg")

# 6.5 Análisis del impacto
print(f"\n{'='*70}")
print("IMPACTO DE LA SELECCIÓN DE CARACTERÍSTICAS")
print(f"{'='*70}")
for clf_name, results in all_results.items():
    delta_f1  = results['top5']['f1'] - results['all']['f1']
    reduction = (1 - 5 / X_all.shape[1]) * 100
    verdict   = "SIN pérdida significativa" if abs(delta_f1) < 0.02 else \
                ("MEJORA" if delta_f1 > 0 else "PÉRDIDA")
    print(f"  {clf_name}: {X_all.shape[1]} → 5 features ({reduction:.1f}% menos) | "
          f"ΔF1={delta_f1:+.4f} → {verdict}")

# ── E) Probabilidades de salida — consola + figura por modelo × conf ─────────
np.random.seed(RANDOM_STATE)
sample_idx = np.sort(
    np.random.choice(len(y_test), size=min(5, len(y_test)), replace=False))

print(f"\n{'='*70}")
print("PROBABILIDADES DE SALIDA (5 instancias de ejemplo)")
print(f"{'='*70}")
for clf_name, results in all_results.items():
    print(f"\n  ── {clf_name} ──")
    for key, res in results.items():
        tag   = "Todas" if key == "all" else "Top-5 Fisher"
        proba = res['y_proba']
        pred  = res['y_pred']
        print(f"    [{tag}]")
        print(f"    {'Inst':>5}  {'Real':>9}  {'Pred':>9}  {'P(No-S)':>9}  {'P(S)':>8}  OK")
        for i in sample_idx:
            rl = 'Stress' if y_test_arr[i] == 1 else 'No-Stress'
            pl = 'Stress' if pred[i]        == 1 else 'No-Stress'
            ok = '✓' if y_test_arr[i] == pred[i] else '✗'
            print(f"    {i:>5}  {rl:>9}  {pl:>9}  {proba[i][0]:>9.4f}  {proba[i][1]:>8.4f}  {ok}")

print("\nGenerando gráficas de probabilidades...")
for clf_name, results in all_results.items():
    clf_slug = clf_name.replace(' ', '_')
    for fkey, flabel in [('all', 'Todas'), ('top5', 'Top-5 Fisher')]:
        tag_label = "Todas" if fkey == "all" else "Top5"
        res   = results[fkey]
        proba = res['y_proba']
        pred  = res['y_pred']
        p_no  = [proba[i][0] for i in sample_idx]
        p_yes = [proba[i][1] for i in sample_idx]
        reals = [y_test_arr[i] for i in sample_idx]
        x_i   = np.arange(len(sample_idx))
        fig, ax = plt.subplots(figsize=(8, 5))
        w = 0.35
        ax.bar(x_i - w/2, p_no,  w, label='P(No-Stress)', color='#3498db', alpha=0.85)
        ax.bar(x_i + w/2, p_yes, w, label='P(Stress)',    color='#e74c3c', alpha=0.85)
        for k, idx in enumerate(sample_idx):
            mk  = '✓' if y_test_arr[idx] == pred[idx] else '✗'
            col = 'green' if mk == '✓' else 'red'
            ax.text(k, 1.05, mk, ha='center', va='bottom',
                    fontsize=14, color=col, fontweight='bold')
        ax.set_xticks(x_i)
        ax.set_xticklabels([f"inst {i}\n({'S' if r_ == 1 else 'NS'})"
                            for i, r_ in zip(sample_idx, reals)], fontsize=9)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel('Probabilidad', fontsize=10)
        ax.set_title(f"Probabilidades de Salida — 5 Instancias\n"
                     f"{clf_name}  [{flabel}]", fontsize=10, fontweight='bold')
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, axis='y', alpha=0.25)
        plt.tight_layout()
        save_jpg(fig, f"fig9_proba_{clf_slug}_{tag_label}.jpg")

# Guardar CSV de métricas
metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'model_metrics_comparison.csv'), index=False)


# ============================================================================
# TABLA RESUMEN GENERAL DE MÉTRICAS (todas en un solo .jpg)
# ============================================================================
print("\nGenerando tabla resumen general de métricas...")

table_data = []
for clf_name, results in all_results.items():
    for key, res in results.items():
        tag = "Todas" if key == "all" else "Top-5"
        table_data.append([
            clf_name, tag,
            f"{res['accuracy']:.4f}",
            f"{res['precision']:.4f}",
            f"{res['recall']:.4f}",
            f"{res['f1']:.4f}",
        ])

fig, ax = plt.subplots(figsize=(13, 4.5))
ax.axis('off')
col_headers = ['Modelo', 'Features', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
tbl_res = ax.table(
    cellText=table_data,
    colLabels=col_headers,
    loc='center', cellLoc='center'
)
tbl_res.auto_set_font_size(False)
tbl_res.set_fontsize(10)
tbl_res.scale(1.2, 1.9)
for j in range(len(col_headers)):
    tbl_res[(0, j)].set_facecolor('#2c3e50')
    tbl_res[(0, j)].set_text_props(color='white', fontweight='bold')
row_colors = ['#eaf4fb', '#fdfefe', '#eafaf1', '#fdfefe', '#fef9e7', '#fdfefe']
for i in range(1, len(table_data) + 1):
    for j in range(len(col_headers)):
        tbl_res[(i, j)].set_facecolor(row_colors[(i - 1) % len(row_colors)])
ax.set_title('Tabla Resumen — Métricas de Rendimiento por Modelo y Conjunto de Features',
             fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
save_jpg(fig, 'tabla_resumen_metricas.jpg')


# ============================================================================
# ACTIVIDAD 7: CÁLCULO MANUAL — NAIVE BAYES Y DECISION TREE
# ============================================================================
print("\n" + "=" * 80)
print("ACTIVIDAD 7: CÁLCULO MANUAL — Naive Bayes y Decision Tree")
print("             Instancias de test: índices 33 y 78")
print("=" * 80)

MANUAL_IDX    = [33, 78]
X_manual      = X_top5_test
y_manual      = y_test_arr
nb_model      = all_results['Naive Bayes']['top5']['clf']
dt_model      = all_results['Decision Tree']['top5']['clf']
classes       = nb_model.classes_
class_labels  = {0: 'No-Stress', 1: 'Stress'}
feature_names = list(X_top5_train.columns)
X_train_arr   = X_top5_train.values
y_train_arr   = np.array(y_train)

# Estadísticas NB
nb_stats = {}
N_train  = len(y_train_arr)
for c in classes:
    mask = y_train_arr == c
    Xc   = X_train_arr[mask]
    nb_stats[c] = {
        'prior': mask.sum() / N_train,
        'mean' : Xc.mean(axis=0),
        'std'  : Xc.std(axis=0) + 1e-9,
    }


def gaussian_pdf(x, mean, std):
    return (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)


def trace_decision_tree(dt, x_row, feat_names):
    tree_   = dt.tree_
    node_id = 0
    path    = []
    while tree_.feature[node_id] != -2:
        fi   = tree_.feature[node_id]
        th   = tree_.threshold[node_id]
        fv   = x_row[fi]
        left = fv <= th
        path.append({'node': node_id, 'feature': feat_names[fi],
                     'threshold': th, 'value': fv,
                     'direction': "≤ izq" if left else "> der"})
        node_id = tree_.children_left[node_id] if left else tree_.children_right[node_id]
    lv   = tree_.value[node_id][0]
    lp   = lv / lv.sum()
    pred = int(dt.classes_[np.argmax(lp)])
    return path, node_id, lv, lp, pred


for inst_idx in MANUAL_IDX:
    x_row  = X_manual.iloc[inst_idx].values
    y_real = y_manual[inst_idx]

    print(f"\n{'─'*70}")
    print(f"  INSTANCIA idx={inst_idx}  | Etiqueta real: {class_labels[y_real]}")
    print(f"{'─'*70}")
    for fn, fv in zip(feature_names, x_row):
        print(f"  {fn:<38} = {fv:.6f}")

    # Naive Bayes manual
    print(f"\n  A) NAIVE BAYES GAUSSIANO")
    log_posteriors = {}
    for c in classes:
        prior = nb_stats[c]['prior']
        means = nb_stats[c]['mean']
        stds  = nb_stats[c]['std']
        print(f"\n  Clase {class_labels[c]} (prior={prior:.4f})")
        print(f"    {'Feature':<38} {'x':>10} {'μ':>10} {'σ':>10} {'f(x)':>14} {'log f':>10}")
        print(f"    {'-'*84}")
        ll = 0.0
        for fn, xk, mk, sk in zip(feature_names, x_row, means, stds):
            pv   = gaussian_pdf(xk, mk, sk)
            lpv  = np.log(pv + 1e-300)
            ll  += lpv
            print(f"    {fn:<38} {xk:>10.5f} {mk:>10.5f} {sk:>10.5f} {pv:>14.6e} {lpv:>10.4f}")
        lp = np.log(prior) + ll
        log_posteriors[c] = lp
        print(f"\n    log-posterior = log({prior:.4f}) + ({ll:.4f}) = {lp:.4f}")

    max_lp   = max(log_posteriors.values())
    exp_vals = {c: np.exp(lp - max_lp) for c, lp in log_posteriors.items()}
    total    = sum(exp_vals.values())
    post     = {c: v / total for c, v in exp_vals.items()}
    nb_pred_m = max(post, key=post.get)
    nb_pred_s = int(nb_model.predict(x_row.reshape(1, -1))[0])
    print(f"\n  Posterior normalizado:")
    for c in classes:
        print(f"    P({class_labels[c]:>10} | x) = {post[c]:.6f}")
    print(f"  ► Manual={class_labels[nb_pred_m]}  sklearn={class_labels[nb_pred_s]}  "
          f"Real={class_labels[y_real]}  "
          f"{'✓' if nb_pred_m==nb_pred_s else '✗'}")

    # Decision Tree manual
    print(f"\n  B) DECISION TREE (recorrido nodo a nodo)")
    path, leaf_node, lv, lp_dt, dt_pred_m = trace_decision_tree(dt_model, x_row, feature_names)
    dt_pred_s = int(dt_model.predict(x_row.reshape(1, -1))[0])
    print(f"  {'Paso':<5} {'Nodo':>5}  {'Feature':<38} {'x':>10} {'Umbral':>10}  {'Dir':<8}")
    print(f"  {'-'*76}")
    for si, step in enumerate(path, 1):
        print(f"  {si:<5} {step['node']:>5}  {step['feature']:<38} "
              f"{step['value']:>10.5f} {step['threshold']:>10.5f}  {step['direction']:<8}")
    print(f"\n  Hoja nodo {leaf_node}:")
    for c, cnt, p in zip(classes, lv, lp_dt):
        print(f"    {class_labels[c]:>10}: {int(cnt):>4} muestras → P={p:.4f}")
    print(f"  ► Manual={class_labels[dt_pred_m]}  sklearn={class_labels[dt_pred_s]}  "
          f"Real={class_labels[y_real]}  "
          f"{'✓' if dt_pred_m==dt_pred_s else '✗'}")


# ============================================================================
# ÁRBOL DE DECISIÓN — VISUALIZACIÓN DE LOS PRIMEROS 2 NIVELES
# ============================================================================
print("\nGenerando visualización de los primeros 2 niveles del árbol de decisión...")
print("  → Versión Top-5 Fisher features")
print("  → Versión Todas las features")

def draw_dt_levels(dt_clf, feat_names, class_lbl, max_depth=2,
                   subtitle='', filename='fig10_decision_tree_2levels.jpg'):
    """
    Dibuja manualmente los primeros `max_depth` niveles del árbol de decisión
    con todos sus valores internos: feature, umbral, gini/mse, muestras,
    distribución de clases y predicción en hoja.

    Se usa matplotlib puro (sin graphviz) para evitar dependencias externas.
    """
    tree_   = dt_clf.tree_
    n_feat  = len(feat_names)
    n_cls   = len(dt_clf.classes_)
    classes = dt_clf.classes_

    # ── Recolectar nodos hasta max_depth mediante BFS ────────────────────────
    # Cada entrada: (node_id, depth, x_center, x_left, x_right, parent_x, parent_y, side)
    # side: 'left' | 'right' | None (raíz)

    # Primero calculamos las posiciones con un BFS que asigna coordenadas x
    # basadas en el orden de posición hoja dentro del nivel.

    from collections import deque

    # Asignar posición horizontal a cada nodo usando el índice de hoja dentro del nivel
    node_info = {}   # node_id -> {'depth', 'pos': float 0..1, 'parent', 'side'}

    queue  = deque()
    queue.append((0, 0, 0.5, None, None))   # node_id, depth, pos, parent_id, side
    leaves_per_depth = {}  # depth -> count of positions already assigned

    def is_leaf(nid):
        return tree_.feature[nid] == -2

    all_bfs = []
    while queue:
        nid, depth, pos, parent, side = queue.popleft()
        node_info[nid] = {'depth': depth, 'pos': pos,
                          'parent': parent, 'side': side}
        all_bfs.append(nid)

        if depth < max_depth and not is_leaf(nid):
            # subdivide el espacio: hijo izquierdo en mitad izquierda, derecho en mitad derecha
            half = (2 ** depth)  # número de slots en este nivel
            slot_width = 1.0 / (2 ** (depth + 1))
            left_pos  = pos - slot_width / 2
            right_pos = pos + slot_width / 2
            # Clamp dentro de [0,1]
            left_pos  = max(slot_width / 2, left_pos)
            right_pos = min(1 - slot_width / 2, right_pos)

            l_child = tree_.children_left[nid]
            r_child = tree_.children_right[nid]
            queue.append((l_child, depth + 1, left_pos,  nid, 'left'))
            queue.append((r_child, depth + 1, right_pos, nid, 'right'))

    # ── Coordenadas y  (profundidad → y normalizada) ─────────────────────────
    y_levels = {d: 1.0 - d / (max_depth + 0.5) for d in range(max_depth + 1)}

    # ── Paleta de clases ──────────────────────────────────────────────────────
    cls_colors = {0: '#aed6f1', 1: '#f1948a'}   # No-Stress=azul claro, Stress=rojo claro

    # ── Calcular impureza Gini manualmente ────────────────────────────────────
    def gini(values):
        total = values.sum()
        if total == 0:
            return 0.0
        probs = values / total
        return 1.0 - np.sum(probs ** 2)

    # ── Figura ────────────────────────────────────────────────────────────────
    fig_w = max(18, 5 * (2 ** max_depth))
    fig_h = 4.5 * max_depth
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 1.08)
    ax.axis('off')

    node_xy = {}   # node_id -> (x, y)
    for nid in all_bfs:
        info = node_info[nid]
        d    = info['depth']
        x    = info['pos']
        y    = y_levels[d]
        node_xy[nid] = (x, y)

    # ── Dibujar aristas primero ───────────────────────────────────────────────
    for nid in all_bfs:
        info = node_info[nid]
        if info['parent'] is not None:
            px, py = node_xy[info['parent']]
            cx, cy = node_xy[nid]
            ax.plot([px, cx], [py, cy], color='#555555', linewidth=1.2, zorder=1)
            # Etiqueta True/False sobre la arista
            mid_x = (px + cx) / 2
            mid_y = (py + cy) / 2 + 0.01
            lbl   = 'True' if info['side'] == 'left' else 'False'
            color_lbl = '#1a5276' if lbl == 'True' else '#922b21'
            ax.text(mid_x, mid_y, lbl, ha='center', va='bottom',
                    fontsize=7, color=color_lbl, fontweight='bold')

    # ── Dibujar nodos ─────────────────────────────────────────────────────────
    BOX_W = 0.11
    BOX_H = 0.10

    for nid in all_bfs:
        info = node_info[nid]
        d    = info['depth']
        x, y = node_xy[nid]

        values   = tree_.value[nid][0]          # shape (n_classes,)
        total    = int(values.sum())
        g        = gini(values)
        majority = int(classes[np.argmax(values)])
        bg_color = cls_colors.get(majority, '#f9f9f9')

        leaf   = is_leaf(nid) or (d == max_depth)

        if leaf:
            # Nodo hoja o nivel máximo: mostrar distribución y predicción
            counts_str = ' / '.join([f'{class_lbl[c]}:{int(v)}'
                                     for c, v in zip(classes, values)])
            lines = [
                f"{'Hoja' if is_leaf(nid) else f'Nivel {d} (truncado)'}  [nodo {nid}]",
                f"Gini = {g:.3f}",
                f"Muestras = {total}",
                counts_str,
                f"Pred → {class_lbl[majority]}",
            ]
            lw = 2.0
        else:
            fi        = tree_.feature[nid]
            threshold = tree_.threshold[nid]
            fname     = feat_names[fi] if fi < len(feat_names) else f'feat_{fi}'
            # Acortar nombre si es muy largo
            fname_short = fname if len(fname) <= 22 else fname[:20] + '…'
            counts_str  = ' / '.join([f'{class_lbl[c]}:{int(v)}'
                                      for c, v in zip(classes, values)])
            lines = [
                f"[nodo {nid}]  {fname_short}",
                f"≤ {threshold:.4f}",
                f"Gini = {g:.3f}",
                f"Muestras = {total}",
                counts_str,
            ]
            lw = 1.2

        # Caja
        rect = plt.Rectangle(
            (x - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H,
            linewidth=lw, edgecolor='#2c3e50',
            facecolor=bg_color, zorder=2,
            transform=ax.transData
        )
        ax.add_patch(rect)

        # Texto dentro de la caja
        line_h = BOX_H / (len(lines) + 0.5)
        for li, line in enumerate(lines):
            ty = y + BOX_H / 2 - line_h * (li + 0.8)
            ax.text(x, ty, line, ha='center', va='center',
                    fontsize=max(5, 7 - d), zorder=3,
                    fontweight='bold' if li == 0 else 'normal',
                    color='#1a252f')

    # ── Leyenda ───────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(facecolor=cls_colors[0], edgecolor='#2c3e50',
                       label=f'Clase 0 — {class_lbl[0]}'),
        mpatches.Patch(facecolor=cls_colors[1], edgecolor='#2c3e50',
                       label=f'Clase 1 — {class_lbl[1]}'),
    ]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=9,
              framealpha=0.9, title='Predicción mayoritaria', title_fontsize=9)

    ax.set_title(
        f'Árbol de Decisión — Primeros {max_depth} Niveles\n'
        f'{subtitle}',
        fontsize=14, fontweight='bold', pad=10
    )

    plt.tight_layout()
    save_jpg(fig, filename)


# ── Top-5 Fisher features ────────────────────────────────────────────────────
draw_dt_levels(
    dt_clf    = all_results['Decision Tree']['top5']['clf'],
    feat_names = list(X_top5_train.columns),
    class_lbl  = {0: 'No-Stress', 1: 'Stress'},
    max_depth  = 2,
    subtitle   = 'Top-5 Features Fisher | entrenado con datos de train',
    filename   = 'fig10_decision_tree_2levels_top5.jpg'
)

# ── Todas las features preprocesadas ─────────────────────────────────────────
draw_dt_levels(
    dt_clf     = all_results['Decision Tree']['all']['clf'],
    feat_names = list(X_all_train.columns),
    class_lbl  = {0: 'No-Stress', 1: 'Stress'},
    max_depth  = 2,
    subtitle   = 'Todas las features preprocesadas | entrenado con datos de train',
    filename   = 'fig10_decision_tree_2levels_all.jpg'
)


# ============================================================================
# GUARDAR RESUMEN JSON Y CSV
# ============================================================================
results_summary = {
    'selected_features'        : selected_features,
    'selection_method'         : 'Fisher Score individual (filtrado)',
    'n_features_original'      : len(feature_cols),
    'n_features_preprocessed'  : len(remaining_features),
    'n_features_selected'      : N_SELECT,
    'n_samples'                : len(df_features),
    'n_subjects'               : int(df_features['subject_id'].nunique()),
    'class_distribution'       : dict(y.value_counts()),
    'fisher_ranking_top20'     : fisher_df.head(20).to_dict('records'),
}
with open(os.path.join(OUTPUT_DIR, 'results_summary.json'), 'w') as f:
    json.dump(results_summary, f, indent=2, default=str)


# ============================================================================
# RESUMEN FINAL
# ============================================================================
print(f"\n{'='*80}")
print("PIPELINE COMPLETADO EXITOSAMENTE")
print(f"{'='*80}")
print(f"\nArchivos generados en:  {OUTPUT_DIR}")
print("\n── TABLA DE RANGOS DE CARACTERÍSTICAS ───────────────────────────────────")
print("  feature_ranges.csv                   (CSV completo: 16 cols por feature)")
print("  tabla_rangos_resumen_p<N>.jpg        (tabla global paginada)")
print("  tabla_rangos_<sensor>.jpg            (una por sensor, paginada si >25 filas)")
print("\n── TABLAS DESCRIPTIVAS ──────────────────────────────────────────────────")
print("  tabla1_descripcion_dataset_original.jpg")
print("  tabla2_descripcion_dataset_features.jpg")
print("  tabla_resumen_metricas.jpg")
print("\n── ACTIVIDAD 1/2: EDA ───────────────────────────────────────────────────")
print("  fig1_class_distribution.jpg")
print("  fig2_top_features_boxplot.jpg")
print("  fig4_distributions_by_class.jpg")
print("  fig5_missing_values.jpg")
print("\n── ACTIVIDAD 3: CORRELACIONES ───────────────────────────────────────────")
print("  fig3_corr_top20_discriminativas.jpg")
print("  fig3_corr_top5_fisher.jpg")
print("  fig3_corr_todas_features.jpg")
print("  fig3_corr_sensor_<nombre>.jpg  (una por sensor)")
print("\n── ACTIVIDADES 4/5: FISHER ──────────────────────────────────────────────")
print("  fig6_fisher_ranking.jpg")
print("  fig7_fisher_selection_results.jpg")
print("\n── ACTIVIDAD 6: MODELOS (una figura por modelo/configuración) ───────────")
for clf_name in all_results:
    slug = clf_name.replace(' ', '_')
    print(f"  fig8_confmat_{slug}_Todas.jpg  /  _Top5.jpg")
    print(f"  fig8_barras_{slug}.jpg")
    print(f"  fig8_delta_{slug}.jpg")
    print(f"  fig8b_por_clase_{slug}_Todas.jpg  /  _Top5.jpg")
    print(f"  fig9_proba_{slug}_Todas.jpg  /  _Top5.jpg")
print("\n── ACTIVIDAD 7: ÁRBOL DE DECISIÓN ───────────────────────────────────────")
print("  fig10_decision_tree_2levels_top5.jpg  (2 niveles, Top-5 Fisher features)")
print("  fig10_decision_tree_2levels_all.jpg   (2 niveles, todas las features)")
print("\n── CSVs / JSON ──────────────────────────────────────────────────────────")
print("  wesad_features.csv")
print("  wesad_final_5features.csv")
print("  fisher_ranking.csv")
print("  model_metrics_comparison.csv")
print("  results_summary.json")

if USE_SYNTHETIC:
    print(f"\n⚠  DATOS SINTÉTICOS (dataset WESAD no encontrado en {DATA_PATH})")
    print("   Para datos reales: descarga de Kaggle y ajusta DATA_PATH.")