# ============================================================
# WESAD Stress Detection Pipeline (Final Stable Version)
# ============================================================

import os
import pickle
import kagglehub
import numpy as np
import pandas as pd
import warnings

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import butter, filtfilt, find_peaks
from scipy.integrate import trapezoid

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# ============================================================
# 1. Download Dataset
# ============================================================

path = kagglehub.dataset_download(
    "orvile/wesad-wearable-stress-affect-detection-dataset"
)

base_path = os.path.join(path, "WESAD")

subjects = sorted([s for s in os.listdir(base_path) if s.startswith("S")])

print("Subjects:", subjects)

# ============================================================
# 2. Sampling Rates
# ============================================================

FS_EDA = 4
FS_ECG = 700
FS_RESP = 700
FS_TEMP = 4
FS_ACC = 32

# ============================================================
# 3. Window Parameters
# ============================================================

WINDOW_SEC = 70
STEP_SEC = 60

WINDOW_EDA = WINDOW_SEC * FS_EDA
STEP_EDA = STEP_SEC * FS_EDA

# ============================================================
# 4. Load Subject
# ============================================================

def load_subject(subject):

    file_path = os.path.join(base_path, subject, subject + ".pkl")

    with open(file_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    chest = data["signal"]["chest"]

    signals = {
        "EDA": chest["EDA"].flatten(),
        "ECG": chest["ECG"].flatten(),
        "RESP": chest["Resp"].flatten(),
        "TEMP": chest["Temp"].flatten(),
        "ACC": chest["ACC"]
    }

    labels = data["label"]

    return signals, labels

# ============================================================
# 5. Filtering
# ============================================================

def butter_lowpass(signal, cutoff, fs):

    b, a = butter(4, cutoff/(fs/2), btype="low")

    return filtfilt(b, a, signal)

# ============================================================
# 6. EDA Features
# ============================================================

def tonic_phasic(eda):

    tonic = butter_lowpass(eda, 0.05, FS_EDA)

    phasic = eda - tonic

    return tonic, phasic


def extract_eda_features(eda):

    tonic, phasic = tonic_phasic(eda)

    peaks, _ = find_peaks(
        phasic,
        height=0.01,
        distance=FS_EDA
    )

    scr_count = len(peaks)

    phasic_auc = trapezoid(np.abs(phasic)) / FS_EDA

    tonic_mean = np.mean(tonic)

    return {
        "EDA_SCR_Count": scr_count,
        "EDA_Phasic_AUC": phasic_auc,
        "EDA_Tonic_Mean": tonic_mean
    }

# ============================================================
# 7. HRV Features
# ============================================================

def extract_hrv_features(ecg):

    if len(ecg) < 500:
        return None

    ecg_filtered = butter_lowpass(ecg, 40, FS_ECG)

    peaks, _ = find_peaks(
        ecg_filtered,
        distance=FS_ECG * 0.6
    )

    if len(peaks) < 3:
        return None

    rr = np.diff(peaks) / FS_ECG

    rmssd = np.sqrt(np.mean(np.square(np.diff(rr))))
    sdnn = np.std(rr)
    mean_rr = np.mean(rr)
    hr = 60 / mean_rr

    return {
        "HRV_RMSSD": rmssd,
        "HRV_SDNN": sdnn,
        "HRV_MeanRR": mean_rr,
        "HR": hr
    }

# ============================================================
# 8. Other Signals
# ============================================================

def extract_resp_features(resp):

    return {
        "RESP_Mean": np.mean(resp),
        "RESP_STD": np.std(resp)
    }


def extract_temp_features(temp):

    return {
        "TEMP_Mean": np.mean(temp),
        "TEMP_STD": np.std(temp)
    }


def extract_acc_features(acc):

    mag = np.linalg.norm(acc, axis=1)

    return {
        "ACC_Mean": np.mean(mag),
        "ACC_STD": np.std(mag)
    }

# ============================================================
# 9. Dataset Construction
# ============================================================

rows = []

for subject in subjects:

    print("Processing", subject)

    signals, labels = load_subject(subject)

    eda = signals["EDA"]
    ecg = signals["ECG"]
    resp = signals["RESP"]
    temp = signals["TEMP"]
    acc = signals["ACC"]

    for i in range(0, len(eda) - WINDOW_EDA, STEP_EDA):

        eda_seg = eda[i:i+WINDOW_EDA]
        temp_seg = temp[i:i+WINDOW_EDA]

        ecg_start = int(i * FS_ECG / FS_EDA)
        ecg_end = int((i + WINDOW_EDA) * FS_ECG / FS_EDA)

        resp_start = int(i * FS_RESP / FS_EDA)
        resp_end = int((i + WINDOW_EDA) * FS_RESP / FS_EDA)

        acc_start = int(i * FS_ACC / FS_EDA)
        acc_end = int((i + WINDOW_EDA) * FS_ACC / FS_EDA)

        ecg_seg = ecg[ecg_start:ecg_end]
        resp_seg = resp[resp_start:resp_end]
        acc_seg = acc[acc_start:acc_end]

        label_window = labels[i:i+WINDOW_EDA]

        if len(label_window) == 0:
            continue

        label = np.bincount(label_window).argmax()

        if label == 2:
            label = 1
        elif label == 1:
            label = 0
        else:
            continue

        row = {}

        eda_f = extract_eda_features(eda_seg)
        row.update(eda_f)

        hrv_f = extract_hrv_features(ecg_seg)
        if hrv_f is None:
            continue

        row.update(hrv_f)

        row.update(extract_resp_features(resp_seg))
        row.update(extract_temp_features(temp_seg))
        row.update(extract_acc_features(acc_seg))

        row["Label"] = label
        row["Subject"] = subject

        rows.append(row)

df = pd.DataFrame(rows)

print("Dataset shape:", df.shape)

print("\nSubjects distribution")
print(df["Subject"].value_counts())

print("\nLabel distribution")
print(df["Label"].value_counts())

# ============================================================
# 10. Preprocessing
# ============================================================

X = df.drop(["Label","Subject"], axis=1)
y = df["Label"]
groups = df["Subject"]

imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = pd.DataFrame(X, columns=df.drop(["Label","Subject"],axis=1).columns)

# ============================================================
# 11. Fisher Score
# ============================================================

def fisher_score(X,y):

    scores = {}

    for feature in X.columns:

        x = X[feature].values
        mean_total = np.mean(x)

        num = 0
        den = 0

        for c in np.unique(y):

            xc = x[y==c]

            num += len(xc)*(np.mean(xc)-mean_total)**2
            den += len(xc)*np.var(xc)

        scores[feature] = num/den if den != 0 else 0

    return pd.Series(scores).sort_values(ascending=False)

fisher = fisher_score(X,y)

print("\nFisher Ranking")
print(fisher)

# ============================================================
# 12. Mutual Information
# ============================================================

mi = mutual_info_classif(X,y)

mi_scores = pd.Series(mi,index=X.columns).sort_values(ascending=False)

print("\nMutual Information Ranking")
print(mi_scores)

# ============================================================
# 13. Sequential Feature Selection
# ============================================================

model = SVC(kernel="rbf")

sfs = SequentialFeatureSelector(
    model,
    n_features_to_select=5,
    direction="forward",
    cv=3
)

sfs.fit(X,y)

selected = X.columns[sfs.get_support()]

print("\nSelected Features")
print(selected)

# ============================================================
# 14. LOSO Evaluation
# ============================================================

logo = LeaveOneGroupOut()

accuracies = []

for train_idx, test_idx in logo.split(X,y,groups):

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = SVC(kernel="rbf")

    model.fit(X_train,y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test,pred)

    accuracies.append(acc)

print("\nLOSO Accuracy:", np.mean(accuracies))