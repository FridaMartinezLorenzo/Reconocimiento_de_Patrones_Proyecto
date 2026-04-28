#!/usr/bin/env python3
"""
================================================================================
WESAD (Wearable Stress and Affect Detection) - Pipeline Completo de Ciencia de Datos
================================================================================

Actividades:
1. Extracción de características de señales fisiológicas y etiquetado
2. Análisis Exploratorio de Datos (EDA)
3. Preprocesamiento de datos
4. Ranking de características con Factor de Fisher (fórmula generalizada ponderada por p_j)
5. Selección de 5 mejores características por Factor de Fisher (método individual/filtrado)

Basado en: https://www.kaggle.com/code/apurvpanchal/wesad-stress-class
Dataset:  https://www.kaggle.com/datasets/orvile/wesad-wearable-stress-affect-detection-dataset

Estructura del dataset WESAD:
- 15 sujetos (S2-S17, sin S1 ni S12)
- Archivos .pkl por sujeto con señales de chest (RespiBAN) y wrist (Empatica E4)
- Etiquetas: 0=no definido, 1=baseline, 2=stress, 3=amusement, 4=meditation
- Señales chest (700 Hz): ACC(3), ECG(1), EDA(1), EMG(1), Resp(1), Temp(1)
- Señales wrist: BVP(64Hz), EDA(4Hz), TEMP(4Hz), ACC(32Hz)

Autores del dataset original:
Schmidt, Philip & Reiss, Attila et al. (2018). ICMI 2018.
================================================================================
"""

import kagglehub
import numpy as np
import pandas as pd
import pickle
import os
import warnings
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
# Cambiar esta ruta a donde tengas el dataset WESAD descargado de Kaggle
# Descargar dataset
path = kagglehub.dataset_download(
    "orvile/wesad-wearable-stress-affect-detection-dataset"
)

print("Path raíz:", path)
print("Contenido raíz:", os.listdir(path))

# Ruta correcta al dataset WESAD
DATA_PATH = os.path.join(path, "WESAD")
SUBJECT_IDS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
OUTPUT_DIR = './resultados/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Frecuencias de muestreo
FS_CHEST = 700   # Hz - RespiBAN
FS_BVP = 64      # Hz - Empatica E4 BVP
FS_EDA_WRIST = 4 # Hz - Empatica E4 EDA
FS_TEMP_WRIST = 4 # Hz - Empatica E4 TEMP
FS_ACC_WRIST = 32 # Hz - Empatica E4 ACC

# Ventana de extracción de características (en segundos)
WINDOW_SIZE = 60   # 60 segundos
WINDOW_SHIFT = 30  # 50% overlap

# Clases de interés (binario: stress vs no-stress)
# No-stress = baseline(1) + amusement(3), Stress = stress(2)
LABEL_MAP = {1: 0, 2: 1, 3: 0}  # 0=no-stress, 1=stress


# ============================================================================
# ACTIVIDAD 1: EXTRACCIÓN DE CARACTERÍSTICAS Y ETIQUETADO
# ============================================================================
print("=" * 80)
print("ACTIVIDAD 1: EXTRACCIÓN DE CARACTERÍSTICAS DE SEÑALES Y ETIQUETADO")
print("=" * 80)


def load_subject_data(data_path, subject_id):
    """
    Carga los datos de un sujeto desde el archivo .pkl de WESAD.
    
    Estructura del .pkl:
    - 'signal' -> 'chest' -> {'ACC','ECG','EDA','EMG','Resp','Temp'}
    - 'signal' -> 'wrist' -> {'ACC','BVP','EDA','TEMP'}
    - 'label' -> array de etiquetas (sampled at 700Hz)
    """
    subject_str = f'S{subject_id}'
    pkl_path = os.path.join(data_path, subject_str, f'{subject_str}.pkl')
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    return data


def compute_statistical_features(signal_data, prefix):
    """
    Calcula características estadísticas básicas de una señal.
    
    Parámetros:
    - signal_data: array numpy con los datos de la señal
    - prefix: prefijo para los nombres de las características
    
    Retorna:
    - dict con las características calculadas
    """
    features = {}
    
    if len(signal_data) == 0:
        return features
    
    # Aplanar si es necesario
    if signal_data.ndim > 1:
        signal_data = signal_data.flatten()
    
    features[f'{prefix}_mean'] = np.mean(signal_data)
    features[f'{prefix}_std'] = np.std(signal_data)
    features[f'{prefix}_min'] = np.min(signal_data)
    features[f'{prefix}_max'] = np.max(signal_data)
    features[f'{prefix}_median'] = np.median(signal_data)
    features[f'{prefix}_range'] = np.max(signal_data) - np.min(signal_data)
    features[f'{prefix}_kurtosis'] = stats.kurtosis(signal_data)
    features[f'{prefix}_skewness'] = stats.skew(signal_data)
    
    # Percentiles
    features[f'{prefix}_q25'] = np.percentile(signal_data, 25)
    features[f'{prefix}_q75'] = np.percentile(signal_data, 75)
    features[f'{prefix}_iqr'] = features[f'{prefix}_q75'] - features[f'{prefix}_q25']
    
    # Características derivadas
    features[f'{prefix}_rms'] = np.sqrt(np.mean(signal_data ** 2))
    
    # Tasa de cruces por cero
    zero_crossings = np.sum(np.diff(np.sign(signal_data - np.mean(signal_data))) != 0)
    features[f'{prefix}_zcr'] = zero_crossings / len(signal_data)
    
    return features


def compute_frequency_features(signal_data, fs, prefix):
    """
    Calcula características en el dominio de la frecuencia.
    """
    features = {}
    
    if len(signal_data) < 4:
        return features
    
    if signal_data.ndim > 1:
        signal_data = signal_data.flatten()
    
    # FFT
    N = len(signal_data)
    yf = np.abs(fft(signal_data - np.mean(signal_data)))[:N // 2]
    xf = fftfreq(N, 1 / fs)[:N // 2]
    
    if len(yf) > 0 and np.sum(yf) > 0:
        # Frecuencia dominante
        features[f'{prefix}_peak_freq'] = xf[np.argmax(yf)]
        
        # Energía espectral
        features[f'{prefix}_spectral_energy'] = np.sum(yf ** 2) / N
        
        # Entropía espectral
        psd = yf ** 2 / np.sum(yf ** 2)
        psd = psd[psd > 0]
        features[f'{prefix}_spectral_entropy'] = -np.sum(psd * np.log2(psd))
    
    return features


def compute_eda_features(eda_signal, fs, prefix='EDA'):
    """
    Calcula características específicas de EDA (Electrodermal Activity).
    Incluye separación tónica/fásica simplificada.
    """
    features = compute_statistical_features(eda_signal, prefix)
    features.update(compute_frequency_features(eda_signal, fs, prefix))
    
    if eda_signal.ndim > 1:
        eda_signal = eda_signal.flatten()
    
    # Componente tónica (SCL) - filtro paso bajo
    if len(eda_signal) > 10:
        try:
            b, a = signal.butter(2, 0.05 / (fs / 2), btype='low')
            scl = signal.filtfilt(b, a, eda_signal)
            features[f'{prefix}_scl_mean'] = np.mean(scl)
            features[f'{prefix}_scl_std'] = np.std(scl)
            
            # Componente fásica (SCR)
            scr = eda_signal - scl
            features[f'{prefix}_scr_mean'] = np.mean(scr)
            features[f'{prefix}_scr_std'] = np.std(scr)
            
            # Número de picos SCR
            peaks, _ = signal.find_peaks(scr, height=np.std(scr) * 0.5)
            features[f'{prefix}_scr_num_peaks'] = len(peaks)
        except Exception:
            pass
    
    # Derivada (tasa de cambio)
    if len(eda_signal) > 1:
        eda_deriv = np.diff(eda_signal)
        features[f'{prefix}_deriv_mean'] = np.mean(eda_deriv)
        features[f'{prefix}_deriv_std'] = np.std(eda_deriv)
    
    return features


def compute_ecg_features(ecg_signal, fs=700, prefix='ECG'):
    """
    Calcula características de ECG incluyendo HRV simplificado.
    """
    features = compute_statistical_features(ecg_signal, prefix)
    features.update(compute_frequency_features(ecg_signal, fs, prefix))
    
    if ecg_signal.ndim > 1:
        ecg_signal = ecg_signal.flatten()
    
    # Detección de picos R simplificada
    try:
        # Filtro paso banda para ECG (0.5 - 40 Hz)
        b, a = signal.butter(4, [0.5 / (fs / 2), 40 / (fs / 2)], btype='band')
        ecg_filtered = signal.filtfilt(b, a, ecg_signal)
        
        # Detección de picos R
        min_distance = int(0.5 * fs)  # mínimo 0.5 segundos entre latidos
        peaks, _ = signal.find_peaks(ecg_filtered, 
                                      height=np.mean(ecg_filtered) + 0.5 * np.std(ecg_filtered),
                                      distance=min_distance)
        
        if len(peaks) > 2:
            # Intervalos RR
            rr_intervals = np.diff(peaks) / fs  # en segundos
            
            # Frecuencia cardíaca
            features[f'{prefix}_hr_mean'] = 60.0 / np.mean(rr_intervals)
            features[f'{prefix}_hr_std'] = np.std(60.0 / rr_intervals)
            
            # HRV en dominio del tiempo
            features[f'{prefix}_rr_mean'] = np.mean(rr_intervals)
            features[f'{prefix}_rr_std'] = np.std(rr_intervals)  # SDNN
            features[f'{prefix}_rmssd'] = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
            
            # pNN50
            nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 0.05)
            features[f'{prefix}_pnn50'] = nn50 / len(rr_intervals)
        else:
            features[f'{prefix}_hr_mean'] = 0
            features[f'{prefix}_hr_std'] = 0
    except Exception:
        pass
    
    return features


def compute_acc_features(acc_data, fs, prefix='ACC'):
    """
    Calcula características del acelerómetro (3 ejes).
    """
    features = {}
    
    if acc_data.ndim == 1:
        acc_data = acc_data.reshape(-1, 1)
    
    axes = ['x', 'y', 'z'] if acc_data.shape[1] >= 3 else [str(i) for i in range(acc_data.shape[1])]
    
    for i, axis in enumerate(axes[:acc_data.shape[1]]):
        axis_data = acc_data[:, i]
        features.update(compute_statistical_features(axis_data, f'{prefix}_{axis}'))
    
    # Magnitud del vector
    if acc_data.shape[1] >= 3:
        magnitude = np.sqrt(np.sum(acc_data[:, :3] ** 2, axis=1))
        features.update(compute_statistical_features(magnitude, f'{prefix}_mag'))
        features.update(compute_frequency_features(magnitude, fs, f'{prefix}_mag'))
    
    return features


def compute_resp_features(resp_signal, fs=700, prefix='RESP'):
    """
    Calcula características de la señal de respiración.
    """
    features = compute_statistical_features(resp_signal, prefix)
    features.update(compute_frequency_features(resp_signal, fs, prefix))
    
    if resp_signal.ndim > 1:
        resp_signal = resp_signal.flatten()
    
    # Tasa respiratoria estimada
    try:
        peaks, _ = signal.find_peaks(resp_signal, distance=int(fs * 1.5))
        if len(peaks) > 1:
            breath_intervals = np.diff(peaks) / fs
            features[f'{prefix}_rate_mean'] = 60.0 / np.mean(breath_intervals)
            features[f'{prefix}_rate_std'] = np.std(60.0 / breath_intervals)
            features[f'{prefix}_insp_time'] = np.mean(breath_intervals)
    except Exception:
        pass
    
    return features


def compute_temp_features(temp_signal, fs, prefix='TEMP'):
    """
    Calcula características de temperatura.
    """
    features = compute_statistical_features(temp_signal, prefix)
    
    if temp_signal.ndim > 1:
        temp_signal = temp_signal.flatten()
    
    # Pendiente (slope) de la temperatura
    if len(temp_signal) > 1:
        x = np.arange(len(temp_signal))
        slope, _, _, _, _ = stats.linregress(x, temp_signal)
        features[f'{prefix}_slope'] = slope
    
    # Derivada
    if len(temp_signal) > 1:
        temp_deriv = np.gradient(temp_signal)
        features[f'{prefix}_deriv_mean'] = np.mean(temp_deriv)
        features[f'{prefix}_deriv_std'] = np.std(temp_deriv)
    
    return features


def compute_emg_features(emg_signal, fs=700, prefix='EMG'):
    """
    Calcula características de EMG (Electromyogram).
    """
    features = compute_statistical_features(emg_signal, prefix)
    features.update(compute_frequency_features(emg_signal, fs, prefix))
    
    if emg_signal.ndim > 1:
        emg_signal = emg_signal.flatten()
    
    # Amplitud media absoluta
    features[f'{prefix}_mav'] = np.mean(np.abs(emg_signal))
    
    # Varianza
    features[f'{prefix}_var'] = np.var(emg_signal)
    
    # Longitud de forma de onda
    if len(emg_signal) > 1:
        features[f'{prefix}_wl'] = np.sum(np.abs(np.diff(emg_signal)))
    
    return features


def extract_features_window(chest_data, wrist_data, labels_window):
    """
    Extrae todas las características de una ventana de tiempo.
    """
    all_features = {}
    
    # === SEÑALES DEL CHEST (RespiBAN - 700 Hz) ===
    # ECG
    if 'ECG' in chest_data and len(chest_data['ECG']) > 0:
        all_features.update(compute_ecg_features(chest_data['ECG'], FS_CHEST, 'c_ECG'))
    
    # EDA chest
    if 'EDA' in chest_data and len(chest_data['EDA']) > 0:
        all_features.update(compute_eda_features(chest_data['EDA'], FS_CHEST, 'c_EDA'))
    
    # EMG
    if 'EMG' in chest_data and len(chest_data['EMG']) > 0:
        all_features.update(compute_emg_features(chest_data['EMG'], FS_CHEST, 'c_EMG'))
    
    # Respiración
    if 'Resp' in chest_data and len(chest_data['Resp']) > 0:
        all_features.update(compute_resp_features(chest_data['Resp'], FS_CHEST, 'c_RESP'))
    
    # Temperatura chest
    if 'Temp' in chest_data and len(chest_data['Temp']) > 0:
        all_features.update(compute_temp_features(chest_data['Temp'], FS_CHEST, 'c_TEMP'))
    
    # ACC chest
    if 'ACC' in chest_data and len(chest_data['ACC']) > 0:
        all_features.update(compute_acc_features(chest_data['ACC'], FS_CHEST, 'c_ACC'))
    
    # === SEÑALES DEL WRIST (Empatica E4) ===
    # BVP
    if 'BVP' in wrist_data and len(wrist_data['BVP']) > 0:
        all_features.update(compute_statistical_features(wrist_data['BVP'], 'w_BVP'))
        all_features.update(compute_frequency_features(wrist_data['BVP'], FS_BVP, 'w_BVP'))
    
    # EDA wrist
    if 'EDA' in wrist_data and len(wrist_data['EDA']) > 0:
        all_features.update(compute_eda_features(wrist_data['EDA'], FS_EDA_WRIST, 'w_EDA'))
    
    # TEMP wrist
    if 'TEMP' in wrist_data and len(wrist_data['TEMP']) > 0:
        all_features.update(compute_temp_features(wrist_data['TEMP'], FS_TEMP_WRIST, 'w_TEMP'))
    
    # ACC wrist
    if 'ACC' in wrist_data and len(wrist_data['ACC']) > 0:
        all_features.update(compute_acc_features(wrist_data['ACC'], FS_ACC_WRIST, 'w_ACC'))
    
    # === ETIQUETA ===
    # Etiqueta por voto mayoritario en la ventana
    label_mode = stats.mode(labels_window, keepdims=True)[0][0]
    all_features['label'] = label_mode
    
    return all_features


def extract_features_subject(data, subject_id):
    """
    Extrae características de todas las ventanas de un sujeto.
    Solo procesa las etiquetas de interés (1=baseline, 2=stress, 3=amusement).
    """
    labels = data['label'].flatten()
    chest_signals = data['signal']['chest']
    wrist_signals = data['signal']['wrist']
    
    # Calcular tamaños de ventana en muestras
    chest_window = WINDOW_SIZE * FS_CHEST
    chest_shift = WINDOW_SHIFT * FS_CHEST
    
    n_samples_chest = len(labels)
    all_features_list = []
    
    # Iterar por ventanas
    start = 0
    window_count = 0
    while start + chest_window <= n_samples_chest:
        end = start + chest_window
        
        # Verificar que la ventana tenga etiquetas válidas
        labels_window = labels[start:end]
        valid_labels = labels_window[(labels_window == 1) | 
                                      (labels_window == 2) | 
                                      (labels_window == 3)]
        
        if len(valid_labels) < 0.8 * len(labels_window):
            start += chest_shift
            continue
        
        # Extraer señales chest para esta ventana
        chest_window_data = {}
        for key in chest_signals:
            sig = chest_signals[key]
            if sig.ndim == 1:
                chest_window_data[key] = sig[start:end]
            else:
                chest_window_data[key] = sig[start:end]
        
        # Calcular índices correspondientes para wrist
        time_start = start / FS_CHEST
        time_end = end / FS_CHEST
        
        wrist_window_data = {}
        for key in wrist_signals:
            sig = wrist_signals[key]
            if key == 'BVP':
                ws = int(time_start * FS_BVP)
                we = int(time_end * FS_BVP)
            elif key in ['EDA', 'TEMP']:
                ws = int(time_start * FS_EDA_WRIST)
                we = int(time_end * FS_EDA_WRIST)
            elif key == 'ACC':
                ws = int(time_start * FS_ACC_WRIST)
                we = int(time_end * FS_ACC_WRIST)
            else:
                ws, we = 0, 0
            
            if we <= len(sig):
                wrist_window_data[key] = sig[ws:we]
            else:
                wrist_window_data[key] = sig[ws:]
        
        # Extraer características
        features = extract_features_window(chest_window_data, wrist_window_data, labels_window)
        features['subject_id'] = subject_id
        all_features_list.append(features)
        
        window_count += 1
        start += chest_shift
    
    print(f"  Sujeto S{subject_id}: {window_count} ventanas procesadas")
    return all_features_list


def generate_synthetic_wesad_data():
    """
    Genera datos sintéticos que simulan la estructura del dataset WESAD
    para demostración cuando no se dispone del dataset real.
    
    Los valores están basados en rangos fisiológicos reales:
    - ECG: amplitud ~1mV, HR 60-100 bpm
    - EDA: 0.01-20 μS, estrés causa aumento
    - Temperatura: 30-37°C
    - BVP: señal pulsátil
    - EMG: señal mioeléctrica
    """
    print("\n*** MODO SINTÉTICO: Generando datos que simulan la estructura WESAD ***")
    print("*** Para usar datos reales, descarga el dataset de Kaggle y ajusta DATA_PATH ***\n")
    
    np.random.seed(42)
    all_features = []
    
    for sid in SUBJECT_IDS:
        # Variación inter-sujeto
        subject_offset = np.random.normal(0, 0.1)
        
        # Generar ~40 ventanas por sujeto con diferentes condiciones
        n_baseline = np.random.randint(12, 18)
        n_stress = np.random.randint(8, 14)
        n_amusement = np.random.randint(8, 12)
        
        for condition, n_windows, label in [('baseline', n_baseline, 1), 
                                              ('stress', n_stress, 2), 
                                              ('amusement', n_amusement, 3)]:
            for _ in range(n_windows):
                feat = {}
                noise = np.random.normal(0, 0.05)
                
                # --- ECG features ---
                if condition == 'stress':
                    hr_base = 95 + np.random.normal(0, 10)  # HR elevada en estrés
                    hrv_base = 0.03  # HRV reducida en estrés
                else:
                    hr_base = 72 + np.random.normal(0, 8)
                    hrv_base = 0.06
                
                feat['c_ECG_mean'] = 0.02 + np.random.normal(0, 0.005) + subject_offset * 0.01
                feat['c_ECG_std'] = 0.15 + np.random.normal(0, 0.02)
                feat['c_ECG_min'] = -0.5 + np.random.normal(0, 0.1)
                feat['c_ECG_max'] = 1.0 + np.random.normal(0, 0.2)
                feat['c_ECG_median'] = 0.01 + np.random.normal(0, 0.005)
                feat['c_ECG_range'] = feat['c_ECG_max'] - feat['c_ECG_min']
                feat['c_ECG_kurtosis'] = 3.0 + np.random.normal(0, 1)
                feat['c_ECG_skewness'] = 0.5 + np.random.normal(0, 0.3)
                feat['c_ECG_q25'] = -0.05 + np.random.normal(0, 0.01)
                feat['c_ECG_q75'] = 0.08 + np.random.normal(0, 0.01)
                feat['c_ECG_iqr'] = feat['c_ECG_q75'] - feat['c_ECG_q25']
                feat['c_ECG_rms'] = 0.15 + np.random.normal(0, 0.02)
                feat['c_ECG_zcr'] = 0.1 + np.random.normal(0, 0.02)
                feat['c_ECG_peak_freq'] = 1.2 + np.random.normal(0, 0.2)
                feat['c_ECG_spectral_energy'] = 0.005 + np.random.normal(0, 0.001)
                feat['c_ECG_spectral_entropy'] = 5.0 + np.random.normal(0, 0.5)
                feat['c_ECG_hr_mean'] = hr_base + subject_offset * 5
                feat['c_ECG_hr_std'] = 5.0 + np.random.normal(0, 1.5)
                feat['c_ECG_rr_mean'] = 60.0 / hr_base
                feat['c_ECG_rr_std'] = hrv_base + np.random.normal(0, 0.01)
                feat['c_ECG_rmssd'] = hrv_base * 1.2 + np.random.normal(0, 0.005)
                feat['c_ECG_pnn50'] = 0.2 + np.random.normal(0, 0.05) if condition != 'stress' else 0.08 + np.random.normal(0, 0.03)
                
                # --- EDA chest features ---
                if condition == 'stress':
                    eda_base = 8.0 + np.random.normal(0, 2)  # EDA elevada en estrés
                    scr_peaks = np.random.randint(5, 15)
                else:
                    eda_base = 3.0 + np.random.normal(0, 1.5)
                    scr_peaks = np.random.randint(0, 5)
                
                feat['c_EDA_mean'] = eda_base + subject_offset
                feat['c_EDA_std'] = eda_base * 0.2 + np.random.normal(0, 0.3)
                feat['c_EDA_min'] = eda_base * 0.5 + np.random.normal(0, 0.2)
                feat['c_EDA_max'] = eda_base * 1.5 + np.random.normal(0, 0.5)
                feat['c_EDA_median'] = eda_base + np.random.normal(0, 0.2)
                feat['c_EDA_range'] = feat['c_EDA_max'] - feat['c_EDA_min']
                feat['c_EDA_kurtosis'] = 2.5 + np.random.normal(0, 1)
                feat['c_EDA_skewness'] = 0.3 + np.random.normal(0, 0.2)
                feat['c_EDA_q25'] = eda_base * 0.8 + np.random.normal(0, 0.2)
                feat['c_EDA_q75'] = eda_base * 1.2 + np.random.normal(0, 0.2)
                feat['c_EDA_iqr'] = feat['c_EDA_q75'] - feat['c_EDA_q25']
                feat['c_EDA_rms'] = eda_base * 1.05 + np.random.normal(0, 0.3)
                feat['c_EDA_zcr'] = 0.01 + np.random.normal(0, 0.005)
                feat['c_EDA_peak_freq'] = 0.05 + np.random.normal(0, 0.02)
                feat['c_EDA_spectral_energy'] = eda_base ** 2 * 0.01 + np.random.normal(0, 0.01)
                feat['c_EDA_spectral_entropy'] = 4.0 + np.random.normal(0, 0.5)
                feat['c_EDA_scl_mean'] = eda_base * 0.9 + np.random.normal(0, 0.2)
                feat['c_EDA_scl_std'] = 0.3 + np.random.normal(0, 0.1)
                feat['c_EDA_scr_mean'] = 0.1 + np.random.normal(0, 0.05)
                feat['c_EDA_scr_std'] = 0.5 + np.random.normal(0, 0.1)
                feat['c_EDA_scr_num_peaks'] = scr_peaks
                feat['c_EDA_deriv_mean'] = 0.001 + np.random.normal(0, 0.0005)
                feat['c_EDA_deriv_std'] = 0.01 + np.random.normal(0, 0.003)
                
                # --- EMG features ---
                if condition == 'stress':
                    emg_base = 0.05 + np.random.normal(0, 0.015)
                else:
                    emg_base = 0.02 + np.random.normal(0, 0.008)
                
                feat['c_EMG_mean'] = emg_base + noise
                feat['c_EMG_std'] = emg_base * 2.0 + np.random.normal(0, 0.01)
                feat['c_EMG_min'] = -emg_base * 5 + np.random.normal(0, 0.02)
                feat['c_EMG_max'] = emg_base * 5 + np.random.normal(0, 0.02)
                feat['c_EMG_median'] = emg_base * 0.1 + np.random.normal(0, 0.005)
                feat['c_EMG_range'] = feat['c_EMG_max'] - feat['c_EMG_min']
                feat['c_EMG_kurtosis'] = 5.0 + np.random.normal(0, 2)
                feat['c_EMG_skewness'] = 0.1 + np.random.normal(0, 0.3)
                feat['c_EMG_q25'] = -emg_base + np.random.normal(0, 0.005)
                feat['c_EMG_q75'] = emg_base + np.random.normal(0, 0.005)
                feat['c_EMG_iqr'] = feat['c_EMG_q75'] - feat['c_EMG_q25']
                feat['c_EMG_rms'] = emg_base * 1.5 + np.random.normal(0, 0.005)
                feat['c_EMG_zcr'] = 0.4 + np.random.normal(0, 0.05)
                feat['c_EMG_peak_freq'] = 50 + np.random.normal(0, 15)
                feat['c_EMG_spectral_energy'] = emg_base ** 2 + np.random.normal(0, 0.001)
                feat['c_EMG_spectral_entropy'] = 6.0 + np.random.normal(0, 0.5)
                feat['c_EMG_mav'] = np.abs(emg_base) + np.random.normal(0, 0.005)
                feat['c_EMG_var'] = emg_base ** 2 * 3 + np.random.normal(0, 0.001)
                feat['c_EMG_wl'] = emg_base * 500 + np.random.normal(0, 20)
                
                # --- Respiración features ---
                if condition == 'stress':
                    resp_rate = 22 + np.random.normal(0, 3)  # Respiración rápida
                else:
                    resp_rate = 16 + np.random.normal(0, 2)
                
                feat['c_RESP_mean'] = 0 + np.random.normal(0, 0.1)
                feat['c_RESP_std'] = 200 + np.random.normal(0, 50)
                feat['c_RESP_min'] = -500 + np.random.normal(0, 100)
                feat['c_RESP_max'] = 500 + np.random.normal(0, 100)
                feat['c_RESP_median'] = 0 + np.random.normal(0, 0.1)
                feat['c_RESP_range'] = feat['c_RESP_max'] - feat['c_RESP_min']
                feat['c_RESP_kurtosis'] = 2.0 + np.random.normal(0, 0.5)
                feat['c_RESP_skewness'] = 0.0 + np.random.normal(0, 0.2)
                feat['c_RESP_q25'] = -150 + np.random.normal(0, 30)
                feat['c_RESP_q75'] = 150 + np.random.normal(0, 30)
                feat['c_RESP_iqr'] = feat['c_RESP_q75'] - feat['c_RESP_q25']
                feat['c_RESP_rms'] = 200 + np.random.normal(0, 50)
                feat['c_RESP_zcr'] = 0.005 + np.random.normal(0, 0.001)
                feat['c_RESP_peak_freq'] = resp_rate / 60 + np.random.normal(0, 0.02)
                feat['c_RESP_spectral_energy'] = 10000 + np.random.normal(0, 2000)
                feat['c_RESP_spectral_entropy'] = 4.5 + np.random.normal(0, 0.5)
                feat['c_RESP_rate_mean'] = resp_rate
                feat['c_RESP_rate_std'] = 2.0 + np.random.normal(0, 0.5)
                feat['c_RESP_insp_time'] = 60.0 / resp_rate + np.random.normal(0, 0.2)
                
                # --- Temperatura chest features ---
                if condition == 'stress':
                    temp_base = 34.5 + np.random.normal(0, 0.3)
                else:
                    temp_base = 35.0 + np.random.normal(0, 0.3)
                
                feat['c_TEMP_mean'] = temp_base + subject_offset * 0.5
                feat['c_TEMP_std'] = 0.05 + np.random.normal(0, 0.01)
                feat['c_TEMP_min'] = temp_base - 0.1 + np.random.normal(0, 0.02)
                feat['c_TEMP_max'] = temp_base + 0.1 + np.random.normal(0, 0.02)
                feat['c_TEMP_median'] = temp_base + np.random.normal(0, 0.02)
                feat['c_TEMP_range'] = feat['c_TEMP_max'] - feat['c_TEMP_min']
                feat['c_TEMP_kurtosis'] = 2.0 + np.random.normal(0, 0.5)
                feat['c_TEMP_skewness'] = 0.0 + np.random.normal(0, 0.1)
                feat['c_TEMP_q25'] = temp_base - 0.03 + np.random.normal(0, 0.01)
                feat['c_TEMP_q75'] = temp_base + 0.03 + np.random.normal(0, 0.01)
                feat['c_TEMP_iqr'] = feat['c_TEMP_q75'] - feat['c_TEMP_q25']
                feat['c_TEMP_rms'] = temp_base + np.random.normal(0, 0.02)
                feat['c_TEMP_zcr'] = 0.001 + np.random.normal(0, 0.0005)
                feat['c_TEMP_slope'] = 0.0001 + np.random.normal(0, 0.00005)
                feat['c_TEMP_deriv_mean'] = 0.0001 + np.random.normal(0, 0.0001)
                feat['c_TEMP_deriv_std'] = 0.001 + np.random.normal(0, 0.0003)
                
                # --- ACC chest features ---
                feat['c_ACC_x_mean'] = 0.9 + np.random.normal(0, 0.05)
                feat['c_ACC_x_std'] = 0.05 + np.random.normal(0, 0.01)
                feat['c_ACC_y_mean'] = -0.2 + np.random.normal(0, 0.05)
                feat['c_ACC_y_std'] = 0.04 + np.random.normal(0, 0.01)
                feat['c_ACC_z_mean'] = -0.3 + np.random.normal(0, 0.05)
                feat['c_ACC_z_std'] = 0.04 + np.random.normal(0, 0.01)
                feat['c_ACC_mag_mean'] = 1.0 + np.random.normal(0, 0.03)
                feat['c_ACC_mag_std'] = 0.05 + np.random.normal(0, 0.01)
                feat['c_ACC_mag_peak_freq'] = 0.5 + np.random.normal(0, 0.2)
                feat['c_ACC_mag_spectral_energy'] = 0.01 + np.random.normal(0, 0.003)
                feat['c_ACC_mag_spectral_entropy'] = 3.0 + np.random.normal(0, 0.5)
                
                # --- BVP wrist features ---
                feat['w_BVP_mean'] = 0.0 + np.random.normal(0, 0.5)
                feat['w_BVP_std'] = 50 + np.random.normal(0, 15)
                feat['w_BVP_min'] = -150 + np.random.normal(0, 40)
                feat['w_BVP_max'] = 150 + np.random.normal(0, 40)
                feat['w_BVP_median'] = 0 + np.random.normal(0, 0.5)
                feat['w_BVP_range'] = feat['w_BVP_max'] - feat['w_BVP_min']
                feat['w_BVP_kurtosis'] = 2.0 + np.random.normal(0, 0.8)
                feat['w_BVP_skewness'] = 0.0 + np.random.normal(0, 0.3)
                feat['w_BVP_q25'] = -30 + np.random.normal(0, 10)
                feat['w_BVP_q75'] = 30 + np.random.normal(0, 10)
                feat['w_BVP_iqr'] = feat['w_BVP_q75'] - feat['w_BVP_q25']
                feat['w_BVP_rms'] = 50 + np.random.normal(0, 15)
                feat['w_BVP_zcr'] = 0.1 + np.random.normal(0, 0.02)
                feat['w_BVP_peak_freq'] = hr_base / 60 + np.random.normal(0, 0.05)
                feat['w_BVP_spectral_energy'] = 5000 + np.random.normal(0, 1500)
                feat['w_BVP_spectral_entropy'] = 4.0 + np.random.normal(0, 0.5)
                
                # --- EDA wrist features ---
                w_eda_base = eda_base * 0.6 + np.random.normal(0, 0.3)
                feat['w_EDA_mean'] = w_eda_base
                feat['w_EDA_std'] = w_eda_base * 0.15 + np.random.normal(0, 0.1)
                feat['w_EDA_min'] = w_eda_base * 0.6 + np.random.normal(0, 0.1)
                feat['w_EDA_max'] = w_eda_base * 1.4 + np.random.normal(0, 0.2)
                feat['w_EDA_median'] = w_eda_base + np.random.normal(0, 0.1)
                feat['w_EDA_range'] = feat['w_EDA_max'] - feat['w_EDA_min']
                feat['w_EDA_kurtosis'] = 2.5 + np.random.normal(0, 0.8)
                feat['w_EDA_skewness'] = 0.3 + np.random.normal(0, 0.2)
                feat['w_EDA_q25'] = w_eda_base * 0.85 + np.random.normal(0, 0.1)
                feat['w_EDA_q75'] = w_eda_base * 1.15 + np.random.normal(0, 0.1)
                feat['w_EDA_iqr'] = feat['w_EDA_q75'] - feat['w_EDA_q25']
                feat['w_EDA_rms'] = w_eda_base * 1.02 + np.random.normal(0, 0.1)
                feat['w_EDA_zcr'] = 0.05 + np.random.normal(0, 0.02)
                feat['w_EDA_peak_freq'] = 0.03 + np.random.normal(0, 0.01)
                feat['w_EDA_spectral_energy'] = w_eda_base ** 2 * 0.01 + np.random.normal(0, 0.01)
                feat['w_EDA_spectral_entropy'] = 3.5 + np.random.normal(0, 0.5)
                feat['w_EDA_scl_mean'] = w_eda_base * 0.95 + np.random.normal(0, 0.1)
                feat['w_EDA_scl_std'] = 0.15 + np.random.normal(0, 0.05)
                feat['w_EDA_scr_mean'] = 0.05 + np.random.normal(0, 0.02)
                feat['w_EDA_scr_std'] = 0.2 + np.random.normal(0, 0.05)
                feat['w_EDA_scr_num_peaks'] = max(0, scr_peaks - np.random.randint(0, 3))
                feat['w_EDA_deriv_mean'] = 0.001 + np.random.normal(0, 0.0005)
                feat['w_EDA_deriv_std'] = 0.005 + np.random.normal(0, 0.002)
                
                # --- TEMP wrist features ---
                w_temp_base = temp_base - 2 + np.random.normal(0, 0.3)
                feat['w_TEMP_mean'] = w_temp_base
                feat['w_TEMP_std'] = 0.03 + np.random.normal(0, 0.01)
                feat['w_TEMP_min'] = w_temp_base - 0.08 + np.random.normal(0, 0.02)
                feat['w_TEMP_max'] = w_temp_base + 0.08 + np.random.normal(0, 0.02)
                feat['w_TEMP_median'] = w_temp_base + np.random.normal(0, 0.02)
                feat['w_TEMP_range'] = feat['w_TEMP_max'] - feat['w_TEMP_min']
                feat['w_TEMP_kurtosis'] = 2.0 + np.random.normal(0, 0.5)
                feat['w_TEMP_skewness'] = 0.0 + np.random.normal(0, 0.1)
                feat['w_TEMP_q25'] = w_temp_base - 0.02 + np.random.normal(0, 0.01)
                feat['w_TEMP_q75'] = w_temp_base + 0.02 + np.random.normal(0, 0.01)
                feat['w_TEMP_iqr'] = feat['w_TEMP_q75'] - feat['w_TEMP_q25']
                feat['w_TEMP_rms'] = w_temp_base + np.random.normal(0, 0.02)
                feat['w_TEMP_zcr'] = 0.001 + np.random.normal(0, 0.0005)
                feat['w_TEMP_slope'] = 0.0001 + np.random.normal(0, 0.00005)
                feat['w_TEMP_deriv_mean'] = 0.0001 + np.random.normal(0, 0.0001)
                feat['w_TEMP_deriv_std'] = 0.0005 + np.random.normal(0, 0.0002)
                
                # --- ACC wrist features ---
                feat['w_ACC_x_mean'] = 0.0 + np.random.normal(0, 0.1)
                feat['w_ACC_x_std'] = 0.1 + np.random.normal(0, 0.03)
                feat['w_ACC_y_mean'] = -0.5 + np.random.normal(0, 0.1)
                feat['w_ACC_y_std'] = 0.1 + np.random.normal(0, 0.03)
                feat['w_ACC_z_mean'] = -0.8 + np.random.normal(0, 0.1)
                feat['w_ACC_z_std'] = 0.08 + np.random.normal(0, 0.02)
                feat['w_ACC_mag_mean'] = 1.0 + np.random.normal(0, 0.05)
                feat['w_ACC_mag_std'] = 0.1 + np.random.normal(0, 0.03)
                feat['w_ACC_mag_peak_freq'] = 0.5 + np.random.normal(0, 0.2)
                feat['w_ACC_mag_spectral_energy'] = 0.02 + np.random.normal(0, 0.005)
                feat['w_ACC_mag_spectral_entropy'] = 3.0 + np.random.normal(0, 0.5)
                
                feat['label'] = label
                feat['subject_id'] = sid
                all_features.append(feat)
    
    return pd.DataFrame(all_features)


# === EJECUCIÓN DE LA EXTRACCIÓN ===
print("\nIntentando cargar el dataset WESAD...")

try:
    # Intentar cargar datos reales
    test_path = os.path.join(DATA_PATH, 'S2', 'S2.pkl')
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"No se encontró {test_path}")
    
    print("Dataset WESAD encontrado. Procesando datos reales...")
    all_features_list = []
    
    for sid in SUBJECT_IDS:
        print(f"\nProcesando sujeto S{sid}...")
        try:
            data = load_subject_data(DATA_PATH, sid)
            features = extract_features_subject(data, sid)
            all_features_list.extend(features)
        except Exception as e:
            print(f"  Error con sujeto S{sid}: {e}")
    
    df_features = pd.DataFrame(all_features_list)
    USE_SYNTHETIC = False

except FileNotFoundError:
    print("Dataset WESAD no encontrado localmente.")
    df_features = generate_synthetic_wesad_data()
    USE_SYNTHETIC = True

# Crear etiqueta binaria: stress(2) vs no-stress (baseline=1, amusement=3)
df_features['label_binary'] = df_features['label'].map(LABEL_MAP)
df_features = df_features.dropna(subset=['label_binary'])
df_features['label_binary'] = df_features['label_binary'].astype(int)

# Guardar dataset
df_features.to_csv(os.path.join(OUTPUT_DIR, 'wesad_features.csv'), index=False)

# Separar features y etiquetas
feature_cols = [c for c in df_features.columns if c not in ['label', 'label_binary', 'subject_id']]
X = df_features[feature_cols].copy()
y = df_features['label_binary'].copy()

print(f"\n{'='*60}")
print(f"RESUMEN DE LA EXTRACCIÓN DE CARACTERÍSTICAS")
print(f"{'='*60}")
print(f"Total de muestras (ventanas):  {len(df_features)}")
print(f"Total de características:       {len(feature_cols)}")
print(f"Sujetos procesados:            {df_features['subject_id'].nunique()}")
print(f"\nDistribución de clases originales:")
print(df_features['label'].value_counts().to_string())
print(f"\nDistribución binaria (0=no-stress, 1=stress):")
print(y.value_counts().to_string())
print(f"\nPrimeras 10 características: {feature_cols[:10]}")


# ============================================================================
# ACTIVIDAD 2: ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ============================================================================
print("\n\n" + "=" * 80)
print("ACTIVIDAD 2: ANÁLISIS EXPLORATORIO DE DATOS")
print("=" * 80)

# --- 2.1 Información general del dataset ---
print("\n--- 2.1 Información General ---")
print(f"\nShape del dataset: {df_features.shape}")
print(f"\nTipos de datos:")
print(X.dtypes.value_counts().to_string())

print(f"\nEstadísticas descriptivas (primeras 10 features):")
print(X[feature_cols[:10]].describe().round(4).to_string())

# --- 2.2 Valores faltantes ---
print("\n\n--- 2.2 Análisis de Valores Faltantes ---")
missing = X.isnull().sum()
missing_pct = (missing / len(X)) * 100
missing_summary = pd.DataFrame({'Faltantes': missing, 'Porcentaje': missing_pct})
missing_with_values = missing_summary[missing_summary['Faltantes'] > 0]

if len(missing_with_values) > 0:
    print(f"\nCaracterísticas con valores faltantes ({len(missing_with_values)}):")
    print(missing_with_values.sort_values('Porcentaje', ascending=False).head(20).to_string())
else:
    print("\nNo hay valores faltantes en el dataset.")

# --- 2.3 Valores infinitos ---
print("\n\n--- 2.3 Análisis de Valores Infinitos ---")
inf_count = np.isinf(X.select_dtypes(include=[np.number])).sum()
inf_cols = inf_count[inf_count > 0]
if len(inf_cols) > 0:
    print(f"\nCaracterísticas con valores infinitos:")
    print(inf_cols.to_string())
else:
    print("No hay valores infinitos en el dataset.")

# --- 2.4 Desbalance de clases ---
print("\n\n--- 2.4 Análisis de Desbalance de Clases ---")
class_counts = y.value_counts()
class_ratio = class_counts.min() / class_counts.max()
print(f"\nDistribución de clases:")
for cls, count in class_counts.items():
    label_name = "No-Stress" if cls == 0 else "Stress"
    print(f"  Clase {cls} ({label_name}): {count} ({count/len(y)*100:.1f}%)")
print(f"\nRatio minoría/mayoría: {class_ratio:.3f}")

if class_ratio < 0.5:
    print("⚠️  Dataset DESBALANCEADO - considerar técnicas de balanceo")
else:
    print("✓  Dataset relativamente balanceado")

# --- 2.5 Outliers ---
print("\n\n--- 2.5 Análisis de Outliers (IQR method) ---")
outlier_counts = {}
for col in feature_cols:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((X[col] < lower) | (X[col] > upper)).sum()
    if n_outliers > 0:
        outlier_counts[col] = n_outliers

outlier_df = pd.DataFrame.from_dict(outlier_counts, orient='index', columns=['n_outliers'])
outlier_df['pct'] = (outlier_df['n_outliers'] / len(X) * 100).round(2)
outlier_df = outlier_df.sort_values('n_outliers', ascending=False)
print(f"\nTop 15 características con más outliers:")
print(outlier_df.head(15).to_string())

# --- 2.6 Correlaciones ---
print("\n\n--- 2.6 Análisis de Correlaciones ---")
corr_matrix = X.corr()

# Características altamente correlacionadas (>0.95)
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_pairs = []
for col in upper_tri.columns:
    for idx in upper_tri.index:
        val = upper_tri.loc[idx, col]
        if abs(val) > 0.95:
            high_corr_pairs.append((idx, col, val))

print(f"\nPares de características con correlación > 0.95: {len(high_corr_pairs)}")
if len(high_corr_pairs) > 0:
    print("Primeros 15 pares:")
    for f1, f2, corr in high_corr_pairs[:15]:
        print(f"  {f1} <-> {f2}: {corr:.4f}")

# --- 2.7 Distribución por clase ---
print("\n\n--- 2.7 Diferencias entre clases (top features) ---")
stress_data = X[y == 1]
no_stress_data = X[y == 0]

class_diffs = {}
for col in feature_cols:
    stress_mean = stress_data[col].mean()
    no_stress_mean = no_stress_data[col].mean()
    
    # t-test
    try:
        t_stat, p_val = stats.ttest_ind(stress_data[col].dropna(), 
                                         no_stress_data[col].dropna())
        class_diffs[col] = {
            'stress_mean': stress_mean,
            'no_stress_mean': no_stress_mean,
            'diff_pct': abs(stress_mean - no_stress_mean) / (abs(no_stress_mean) + 1e-10) * 100,
            't_stat': t_stat,
            'p_value': p_val
        }
    except Exception:
        pass

diff_df = pd.DataFrame(class_diffs).T
diff_df = diff_df.sort_values('p_value')
print(f"\nTop 15 características más discriminativas (menor p-value en t-test):")
print(diff_df[['stress_mean', 'no_stress_mean', 'diff_pct', 't_stat', 'p_value']].head(15).round(6).to_string())


# ============================================================================
# GRÁFICAS DEL EDA
# ============================================================================
print("\n\nGenerando gráficas del EDA...")

# Gráfica 1: Distribución de clases
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1a: Clases originales
label_names = {1: 'Baseline', 2: 'Stress', 3: 'Amusement'}
orig_counts = df_features['label'].value_counts()
colors_orig = ['#3498db', '#e74c3c', '#2ecc71']
axes[0].bar([label_names.get(x, str(x)) for x in orig_counts.index], 
            orig_counts.values, color=colors_orig[:len(orig_counts)])
axes[0].set_title('Distribución de Clases Originales', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Número de muestras')
for i, v in enumerate(orig_counts.values):
    axes[0].text(i, v + 1, str(v), ha='center', fontweight='bold')

# 1b: Clases binarias
bin_counts = y.value_counts()
bin_names = {0: 'No-Stress', 1: 'Stress'}
colors_bin = ['#3498db', '#e74c3c']
axes[1].bar([bin_names[x] for x in bin_counts.index], 
            bin_counts.values, color=colors_bin)
axes[1].set_title('Distribución Binaria', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Número de muestras')
for i, v in enumerate(bin_counts.values):
    axes[1].text(i, v + 1, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_class_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()

# Gráfica 2: Top features más discriminativas - boxplots
top_features = diff_df.head(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, feat in enumerate(top_features):
    data_plot = pd.DataFrame({
        'value': X[feat],
        'class': y.map({0: 'No-Stress', 1: 'Stress'})
    })
    sns.boxplot(data=data_plot, x='class', y='value', ax=axes[i],
                palette=['#3498db', '#e74c3c'])
    p_val = diff_df.loc[feat, 'p_value']
    axes[i].set_title(f'{feat}\n(p={p_val:.2e})', fontsize=10, fontweight='bold')
    axes[i].set_xlabel('')

plt.suptitle('Top 6 Características Más Discriminativas (Stress vs No-Stress)', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_top_features_boxplot.png'), dpi=150, bbox_inches='tight')
plt.close()

# Gráfica 3: Mapa de correlaciones (subset)
top20_features = diff_df.head(20).index.tolist()
fig, ax = plt.subplots(figsize=(14, 12))
corr_sub = X[top20_features].corr()
mask = np.triu(np.ones_like(corr_sub, dtype=bool))
sns.heatmap(corr_sub, mask=mask, cmap='RdBu_r', center=0, 
            annot=True, fmt='.2f', square=True, ax=ax,
            linewidths=0.5, cbar_kws={"shrink": 0.8},
            annot_kws={"size": 7})
ax.set_title('Matriz de Correlaciones - Top 20 Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_correlation_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()

# Gráfica 4: Distribuciones por clase
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, feat in enumerate(top_features):
    for cls, color, label in [(0, '#3498db', 'No-Stress'), (1, '#e74c3c', 'Stress')]:
        data_cls = X.loc[y == cls, feat].dropna()
        axes[i].hist(data_cls, bins=30, alpha=0.5, color=color, label=label, density=True)
    axes[i].set_title(feat, fontsize=10, fontweight='bold')
    axes[i].legend(fontsize=8)
    axes[i].set_ylabel('Densidad')

plt.suptitle('Distribuciones de Top Features por Clase', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_distributions_by_class.png'), dpi=150, bbox_inches='tight')
plt.close()

# Gráfica 5: Missing values heatmap
fig, ax = plt.subplots(figsize=(16, 4))
missing_matrix = X.isnull().astype(int)
if missing_matrix.sum().sum() > 0:
    sns.heatmap(missing_matrix.T, cbar=False, cmap='YlOrRd', ax=ax)
    ax.set_title('Mapa de Valores Faltantes', fontsize=14, fontweight='bold')
else:
    ax.text(0.5, 0.5, 'No hay valores faltantes', ha='center', va='center', 
            fontsize=16, transform=ax.transAxes)
    ax.set_title('Mapa de Valores Faltantes', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_missing_values.png'), dpi=150, bbox_inches='tight')
plt.close()

print("Gráficas guardadas en:", OUTPUT_DIR)


# ============================================================================
# ACTIVIDAD 3: PREPROCESAMIENTO DE DATOS
# ============================================================================
print("\n\n" + "=" * 80)
print("ACTIVIDAD 3: PREPROCESAMIENTO DE DATOS")
print("=" * 80)

X_processed = X.copy()
preprocessing_log = []

# --- 3.1 Tratamiento de valores faltantes ---
print("\n--- 3.1 Tratamiento de Valores Faltantes ---")
n_missing_before = X_processed.isnull().sum().sum()
print(f"Valores faltantes antes: {n_missing_before}")

# Imputar con la mediana (robusto a outliers)
for col in X_processed.columns:
    if X_processed[col].isnull().any():
        median_val = X_processed[col].median()
        n_imputed = X_processed[col].isnull().sum()
        X_processed[col].fillna(median_val, inplace=True)
        preprocessing_log.append(f"Imputado {col}: {n_imputed} valores con mediana={median_val:.4f}")

n_missing_after = X_processed.isnull().sum().sum()
print(f"Valores faltantes después: {n_missing_after}")

# --- 3.2 Tratamiento de valores infinitos ---
print("\n--- 3.2 Tratamiento de Valores Infinitos ---")
n_inf_before = np.isinf(X_processed.select_dtypes(include=[np.number])).sum().sum()
print(f"Valores infinitos antes: {n_inf_before}")

X_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
for col in X_processed.columns:
    if X_processed[col].isnull().any():
        median_val = X_processed[col].median()
        X_processed[col].fillna(median_val, inplace=True)

n_inf_after = np.isinf(X_processed.select_dtypes(include=[np.number])).sum().sum()
print(f"Valores infinitos después: {n_inf_after}")

# --- 3.3 Eliminación de características con varianza cero ---
print("\n--- 3.3 Eliminación de Características con Varianza Cero ---")
zero_var_cols = X_processed.columns[X_processed.var() < 1e-10].tolist()
if zero_var_cols:
    print(f"Eliminando {len(zero_var_cols)} características con varianza ~0:")
    for col in zero_var_cols:
        print(f"  - {col}")
    X_processed.drop(columns=zero_var_cols, inplace=True)
    feature_cols = [c for c in feature_cols if c not in zero_var_cols]
else:
    print("Ninguna característica con varianza cero.")

# --- 3.4 Eliminación de características altamente correlacionadas ---
print("\n--- 3.4 Eliminación de Características Altamente Correlacionadas (>0.98) ---")
corr_matrix_proc = X_processed.corr().abs()
upper_tri = corr_matrix_proc.where(np.triu(np.ones(corr_matrix_proc.shape), k=1).astype(bool))
to_drop_corr = [column for column in upper_tri.columns if any(upper_tri[column] > 0.98)]

if to_drop_corr:
    print(f"Eliminando {len(to_drop_corr)} características altamente correlacionadas:")
    for col in to_drop_corr[:10]:
        print(f"  - {col}")
    if len(to_drop_corr) > 10:
        print(f"  ... y {len(to_drop_corr) - 10} más")
    X_processed.drop(columns=to_drop_corr, inplace=True)
else:
    print("Ninguna característica eliminada por alta correlación.")

# --- 3.5 Tratamiento de outliers (Winsorización) ---
print("\n--- 3.5 Tratamiento de Outliers (Winsorización al percentil 1-99) ---")
n_clipped = 0
for col in X_processed.columns:
    p1 = X_processed[col].quantile(0.01)
    p99 = X_processed[col].quantile(0.99)
    clipped = ((X_processed[col] < p1) | (X_processed[col] > p99)).sum()
    n_clipped += clipped
    X_processed[col] = X_processed[col].clip(p1, p99)

print(f"Total de valores winsorizados: {n_clipped}")

# --- 3.6 Estandarización ---
print("\n--- 3.6 Estandarización (StandardScaler) ---")
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X_processed),
    columns=X_processed.columns,
    index=X_processed.index
)

print(f"Media después de escalar (debe ser ~0): {X_scaled.mean().mean():.6f}")
print(f"Std después de escalar (debe ser ~1): {X_scaled.std().mean():.6f}")

# Resumen del preprocesamiento
remaining_features = X_processed.columns.tolist()
print(f"\n{'='*60}")
print(f"RESUMEN DEL PREPROCESAMIENTO")
print(f"{'='*60}")
print(f"Características originales:  {len(feature_cols)}")
print(f"Características eliminadas:  {len(feature_cols) - len(remaining_features)}")
print(f"Características finales:     {len(remaining_features)}")
print(f"Muestras:                    {len(X_scaled)}")


# ============================================================================
# ACTIVIDAD 4: RANKING CON FACTOR DE FISHER
# ============================================================================
print("\n\n" + "=" * 80)
print("ACTIVIDAD 4: RANKING DE CARACTERÍSTICAS - FACTOR DE FISHER")
print("=" * 80)


def fisher_score(X_data, y_data):
    """
    Calcula el Factor de Fisher tradicional para cada característica A_k.

    La fórmula implementada es la versión generalizada ponderada por proporciones
    de clase, válida tanto para clasificación binaria como multiclase:

        F(A_k) = Σ_j [ p_j * (μ_{k,j} - μ_k)² ]
                 ─────────────────────────────────
                 Σ_j [ p_j * σ_{k,j}² ]

    Donde:
    - p_j      : proporción de muestras de la clase c_j  (n_j / N)
    - μ_{k,j}  : media de la característica A_k para la clase c_j
    - σ_{k,j}  : desviación estándar de la característica A_k en la clase c_j
    - μ_k      : media global de la característica A_k

    Un valor mayor indica mayor capacidad discriminativa entre clases.
    """
    # Convertir a array numpy para operar de forma uniforme
    if hasattr(X_data, 'values'):
        X_arr = X_data.values
    else:
        X_arr = np.array(X_data)

    if hasattr(y_data, 'values'):
        y_arr = y_data.values
    else:
        y_arr = np.array(y_data)

    classes = np.unique(y_arr)
    N = len(y_arr)                          # Total de muestras
    n_features = X_arr.shape[1]
    fisher_scores = np.zeros(n_features)

    for k in range(n_features):
        feature_values = X_arr[:, k]

        # Media global de la característica A_k
        mu_k = np.mean(feature_values)

        numerator   = 0.0
        denominator = 0.0

        for c in classes:
            mask = y_arr == c
            class_values = feature_values[mask]
            n_j   = np.sum(mask)            # Número de muestras de la clase c_j
            p_j   = n_j / N                 # Proporción de la clase c_j

            mu_kj    = np.mean(class_values)          # Media de A_k en clase c_j
            sigma_kj = np.std(class_values, ddof=0)   # Desv. estándar de A_k en clase c_j

            numerator   += p_j * (mu_kj - mu_k) ** 2
            denominator += p_j * (sigma_kj ** 2)

        if denominator > 1e-10:
            fisher_scores[k] = numerator / denominator
        else:
            fisher_scores[k] = 0.0

    return fisher_scores


# Calcular Fisher Score para todas las características
print("\nCalculando Factor de Fisher para cada característica...")
fisher_scores = fisher_score(X_scaled, y)

# Crear DataFrame con ranking
fisher_df = pd.DataFrame({
    'Característica': X_scaled.columns,
    'Fisher_Score': fisher_scores
}).sort_values('Fisher_Score', ascending=False).reset_index(drop=True)

fisher_df['Rank'] = range(1, len(fisher_df) + 1)

print(f"\n{'='*70}")
print(f"RANKING COMPLETO DE CARACTERÍSTICAS POR FACTOR DE FISHER")
print(f"{'='*70}")
print(f"\nTop 20 características:")
print(fisher_df[['Rank', 'Característica', 'Fisher_Score']].head(20).to_string(index=False))

print(f"\nBottom 10 características:")
print(fisher_df[['Rank', 'Característica', 'Fisher_Score']].tail(10).to_string(index=False))

# Guardar ranking completo
fisher_df.to_csv(os.path.join(OUTPUT_DIR, 'fisher_ranking.csv'), index=False)

# Gráfica del ranking de Fisher
fig, ax = plt.subplots(figsize=(14, 8))
top_n = min(25, len(fisher_df))
top_fisher = fisher_df.head(top_n)

colors = plt.cm.RdYlGn_r(np.linspace(0, 1, top_n))
bars = ax.barh(range(top_n), top_fisher['Fisher_Score'].values, color=colors)
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_fisher['Característica'].values, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Fisher Score', fontsize=12)
ax.set_title(f'Top {top_n} Características por Factor de Fisher\n(Mayor = Más Discriminativa para Estrés)', 
             fontsize=14, fontweight='bold')

# Añadir valores
for i, (bar, val) in enumerate(zip(bars, top_fisher['Fisher_Score'].values)):
    ax.text(bar.get_width() + max(top_fisher['Fisher_Score']) * 0.01, bar.get_y() + bar.get_height()/2, 
            f'{val:.4f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig6_fisher_ranking.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\nGráfica del ranking guardada.")


# ============================================================================
# ACTIVIDAD 5: SELECCIÓN INDIVIDUAL POR FACTOR DE FISHER (TOP-5)
# ============================================================================
print("\n\n" + "=" * 80)
print("ACTIVIDAD 5: SELECCIÓN INDIVIDUAL - TOP 5 CARACTERÍSTICAS POR FACTOR DE FISHER")
print("=" * 80)

# ---------------------------------------------------------------------------
# La selección se realiza directamente a partir del ranking calculado en la
# Actividad 4. Se toman las 5 características con mayor Factor de Fisher,
# sin necesitar ningún clasificador ni validación cruzada interna
# (método de filtrado / selección individual).
# ---------------------------------------------------------------------------

N_SELECT = 5
selected_features = fisher_df['Característica'].head(N_SELECT).tolist()

print(f"\nSeleccionando las {N_SELECT} características con mayor Factor de Fisher...")
print(f"\n{'Rank':<6}{'Característica':<40}{'Fisher Score':<15}")
print("-" * 61)
for _, row in fisher_df.head(N_SELECT).iterrows():
    print(f"{int(row['Rank']):<6}{row['Característica']:<40}{row['Fisher_Score']:.6f}")

# Evaluación final con las 5 features seleccionadas vs todas las features
print(f"\n\n{'='*70}")
print(f"EVALUACIÓN COMPARATIVA")
print(f"{'='*70}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for clf_name, clf in [('KNN (k=5)', KNeighborsClassifier(n_neighbors=5)),
                       ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42))]:
    scores_selected = cross_val_score(clf, X_scaled[selected_features], y, cv=cv, scoring='accuracy')
    scores_all      = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')

    print(f"\n{clf_name}:")
    print(f"  Con {N_SELECT} features (Fisher top-{N_SELECT}): "
          f"{scores_selected.mean():.4f} ± {scores_selected.std():.4f}")
    print(f"  Con todas ({X_scaled.shape[1]}) features:         "
          f"{scores_all.mean():.4f} ± {scores_all.std():.4f}")

# Gráfica: comparativa del Fisher Score para las 5 seleccionadas
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel izquierdo – barras horizontales del Fisher Score de las top-5
fisher_selected = fisher_df.head(N_SELECT)['Fisher_Score'].tolist()
colors = plt.cm.viridis(np.linspace(0.2, 0.8, N_SELECT))
bars = axes[0].barh(range(N_SELECT), fisher_selected, color=colors)
axes[0].set_yticks(range(N_SELECT))
axes[0].set_yticklabels(selected_features, fontsize=10)
axes[0].invert_yaxis()
axes[0].set_xlabel('Factor de Fisher', fontsize=12)
axes[0].set_title(f'Top {N_SELECT} Características Seleccionadas\npor Factor de Fisher',
                  fontsize=14, fontweight='bold')
axes[0].grid(True, axis='x', alpha=0.3)
max_fisher = max(fisher_selected) if max(fisher_selected) > 0 else 1
for bar, val in zip(bars, fisher_selected):
    axes[0].text(bar.get_width() + max_fisher * 0.02,
                 bar.get_y() + bar.get_height() / 2,
                 f'{val:.4f}', va='center', fontsize=9)

# Panel derecho – ranking completo (top-25) con las 5 seleccionadas resaltadas
top_n = min(25, len(fisher_df))
top_fisher = fisher_df.head(top_n)
bar_colors = ['#e74c3c' if feat in selected_features else '#7f8c8d'
              for feat in top_fisher['Característica']]
axes[1].barh(range(top_n), top_fisher['Fisher_Score'].values, color=bar_colors)
axes[1].set_yticks(range(top_n))
axes[1].set_yticklabels(top_fisher['Característica'].values, fontsize=8)
axes[1].invert_yaxis()
axes[1].set_xlabel('Factor de Fisher', fontsize=12)
axes[1].set_title(f'Ranking Fisher – Top {top_n}\n(Rojo = seleccionadas)',
                  fontsize=14, fontweight='bold')
axes[1].grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig7_fisher_selection_results.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("\nGráfica de selección guardada.")

# Guardar resultados finales
results_summary = {
    'selected_features': selected_features,
    'selection_method': 'Fisher Score individual (filtrado)',
    'fisher_ranking_top20': fisher_df.head(20).to_dict('records')
}

import json
with open(os.path.join(OUTPUT_DIR, 'results_summary.json'), 'w') as f:
    json.dump(results_summary, f, indent=2)

# Guardar datos preprocesados
X_final = X_scaled[selected_features].copy()
X_final['label'] = y.values
X_final.to_csv(os.path.join(OUTPUT_DIR, 'wesad_final_5features.csv'), index=False)

print(f"\n\n{'='*80}")
print(f"PIPELINE COMPLETADO EXITOSAMENTE")
print(f"{'='*80}")
print(f"\nArchivos generados en {OUTPUT_DIR}:")
print(f"  - wesad_features.csv           (dataset completo con features)")
print(f"  - wesad_final_5features.csv    (5 mejores features + etiqueta)")
print(f"  - fisher_ranking.csv           (ranking completo de Fisher)")
print(f"  - results_summary.json         (resumen de resultados)")
print(f"  - fig1_class_distribution.png  (distribución de clases)")
print(f"  - fig2_top_features_boxplot.png (boxplots features discriminativas)")
print(f"  - fig3_correlation_matrix.png  (matriz de correlaciones)")
print(f"  - fig4_distributions_by_class.png (distribuciones por clase)")
print(f"  - fig5_missing_values.png      (valores faltantes)")
print(f"  - fig6_fisher_ranking.png             (ranking de Fisher)")
print(f"  - fig7_fisher_selection_results.png  (top-5 seleccionadas por Fisher)")

if USE_SYNTHETIC:
    print(f"\n⚠️  NOTA: Se usaron datos SINTÉTICOS porque el dataset WESAD no fue encontrado.")
    print(f"   Para usar datos reales:")
    print(f"   1. Descargar de: https://www.kaggle.com/datasets/orvile/wesad-wearable-stress-affect-detection-dataset")
    print(f"   2. Extraer en: {DATA_PATH}")
    print(f"   3. Asegurar estructura: {DATA_PATH}S2/S2.pkl, {DATA_PATH}S3/S3.pkl, etc.")
    print(f"   4. Ejecutar nuevamente este script.")