# %% [code] {"execution":{"iopub.execute_input":"2026-03-11T16:40:07.082696Z","iopub.status.busy":"2026-03-11T16:40:07.082214Z","iopub.status.idle":"2026-03-11T16:40:09.668361Z","shell.execute_reply":"2026-03-11T16:40:09.667219Z"},"papermill":{"duration":2.593784,"end_time":"2026-03-11T16:40:09.670569","exception":false,"start_time":"2026-03-11T16:40:07.076785","status":"completed"},"tags":[]}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.execute_input":"2026-03-11T16:40:09.679382Z","iopub.status.busy":"2026-03-11T16:40:09.678220Z","iopub.status.idle":"2026-03-11T16:40:09.687004Z","shell.execute_reply":"2026-03-11T16:40:09.685884Z"},"papermill":{"duration":0.014968,"end_time":"2026-03-11T16:40:09.688977","exception":false,"start_time":"2026-03-11T16:40:09.674009","status":"completed"},"tags":[]}
# =========================
# CELDA 1: Imports + config
# =========================
import os
import re
import glob
import math
import json
import time
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

SEED = 123
random.seed(SEED)
np.random.seed(SEED)

# Rutas base (ajusta si cambia el dataset en Kaggle)
DB10_EMG = "/kaggle/input/datasets/jorgeoc/ninapro-db10/DB10_EXPORT_EMG"
NINAPRO_EMG = "/kaggle/input/datasets/jorgeoc/ninapro-datasets-db1-db2-db3/NINAPRO_EXPORT_EMG"

OUT_ROOT = Path("/kaggle/working/prepared_myoware_v2")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Parámetros
FS = 2000
WIN_MS = 200
STRIDE_MS = 100

WIN = int(FS * WIN_MS / 1000)       # 400
STRIDE = int(FS * STRIDE_MS / 1000) # 200

SHARD_SIZE = 20000
MAX_FILES = None   # pon un entero si quieres debug rápido

print("OUT_ROOT:", OUT_ROOT)
print("WIN:", WIN, "STRIDE:", STRIDE)

# %% [code]
# =========================
# CELDA 1B: Split por participante (FIX Revisor 2, puntos #3-4)
# =========================

"""
participant_split.py
=====================
Utilidades compartidas para resolver el problema de "data leakage" señalado
por el Revisor 2 (puntos #3-4): el split train/val/test actual se hace a
nivel de SHARD (archivo .npy con ventanas mezcladas de varios participantes),
no a nivel de PARTICIPANTE/GRABACIÓN.

Los archivos .npz de origen ya traen el ID de sujeto en el nombre, ej:
    S010_ex1.npz
    S012_ex2_motor.npz
    S105_ex2_stump.npz

Esto permite reconstruir el nivel de participante sin metadata adicional.

Pega este archivo como celda al inicio de los notebooks 01, 02, 03, 04 y 06
(o súbelo como Kaggle Dataset / Utility Script e impórtalo con
`from participant_split import *`).
"""

import re
import numpy as np


def extract_subject_id(filepath: str) -> str:
    """
    Extrae el ID de sujeto/participante desde la ruta de un archivo .npz.

    Ejemplos:
        ".../MDS1/S010_ex1.npz"          -> "S010"
        ".../MDS2/S105_ex2_stump.npz"    -> "S105"
        ".../DB1_s1/S1_E1_A1.npz"        -> "S1"     (fallback NinaPro DB1/DB3)

    Si no encuentra un patrón "S<numero>", usa el prefijo antes del primer
    "_" del nombre de archivo como fallback. Esto NUNCA debe devolver algo
    genérico compartido entre sujetos distintos (evita que todo caiga en un
    solo "grupo").
    """
    import os
    base = os.path.basename(filepath)

    # Patrón principal: S seguido de dígitos (NinaPro DB10 / DB1 / DB3)
    m = re.search(r"(S\d+)", base, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Fallback: prefijo antes del primer "_"
    stem = os.path.splitext(base)[0]
    prefix = stem.split("_")[0]
    if prefix:
        return prefix.upper()

    # Último recurso: nombre completo del archivo (garantiza no-colisión,
    # pero indica que el patrón no fue reconocido -> revisar manualmente)
    return f"UNKNOWN_{stem}"


def group_split_subjects(subject_ids, frac_val=0.15, frac_test=0.15, seed=123):
    """
    Divide una lista de subject_ids (con repeticiones, uno por ventana/muestra)
    en train/val/test garantizando que NINGÚN sujeto aparezca en más de una
    partición.

    Como los sujetos tienen distinto número de ventanas, las fracciones
    resultantes son aproximadas (se optimiza por conteo acumulado de
    ventanas, no por número de sujetos), pero la separación es exacta.

    Devuelve tres arrays booleanos (train_mask, val_mask, test_mask) del
    mismo largo que `subject_ids`.
    """
    subject_ids = np.asarray(subject_ids)
    unique_subjects, counts = np.unique(subject_ids, return_counts=True)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique_subjects))
    unique_subjects = unique_subjects[order]
    counts = counts[order]

    total = counts.sum()
    target_test = total * frac_test
    target_val = total * frac_val

    test_subjects, val_subjects, train_subjects = [], [], []
    acc_test, acc_val = 0, 0

    for subj, c in zip(unique_subjects, counts):
        if acc_test < target_test:
            test_subjects.append(subj)
            acc_test += c
        elif acc_val < target_val:
            val_subjects.append(subj)
            acc_val += c
        else:
            train_subjects.append(subj)

    test_subjects = set(test_subjects)
    val_subjects = set(val_subjects)
    train_subjects = set(train_subjects)

    # Sanity check: no debe haber overlap (por construcción no lo hay,
    # pero se valida explícitamente para detectar bugs futuros)
    assert not (test_subjects & val_subjects)
    assert not (test_subjects & train_subjects)
    assert not (val_subjects & train_subjects)

    train_mask = np.isin(subject_ids, list(train_subjects))
    val_mask = np.isin(subject_ids, list(val_subjects))
    test_mask = np.isin(subject_ids, list(test_subjects))

    print(f"[group_split_subjects] sujetos -> train={len(train_subjects)} "
          f"val={len(val_subjects)} test={len(test_subjects)}")
    print(f"[group_split_subjects] ventanas -> train={train_mask.sum()} "
          f"({train_mask.mean():.1%})  val={val_mask.sum()} "
          f"({val_mask.mean():.1%})  test={test_mask.sum()} "
          f"({test_mask.mean():.1%})")
    print(f"[group_split_subjects] train subjects: {sorted(train_subjects)}")
    print(f"[group_split_subjects] val subjects:   {sorted(val_subjects)}")
    print(f"[group_split_subjects] test subjects:  {sorted(test_subjects)}")

    return train_mask, val_mask, test_mask


def group_split_files(npz_files, frac_val=0.15, frac_test=0.15, seed=123):
    """
    Variante a nivel de ARCHIVO .npz (útil en la etapa de preparación del
    dataset, CELDA 3/4 de 01-preparar-dataset-modo-myoware.ipynb), antes de
    generar las ventanas. Devuelve tres listas de rutas: train, val, test.
    """
    subject_ids = np.array([extract_subject_id(fp) for fp in npz_files])
    train_mask, val_mask, test_mask = group_split_subjects(
        subject_ids, frac_val=frac_val, frac_test=frac_test, seed=seed
    )
    npz_files = np.asarray(npz_files)
    return (
        npz_files[train_mask].tolist(),
        npz_files[val_mask].tolist(),
        npz_files[test_mask].tolist(),
    )


FRAC_VAL = 0.15
FRAC_TEST = 0.15


# %% [code] {"execution":{"iopub.execute_input":"2026-03-11T16:40:09.698070Z","iopub.status.busy":"2026-03-11T16:40:09.697547Z","iopub.status.idle":"2026-03-11T16:40:09.720278Z","shell.execute_reply":"2026-03-11T16:40:09.718956Z"},"papermill":{"duration":0.029906,"end_time":"2026-03-11T16:40:09.722286","exception":false,"start_time":"2026-03-11T16:40:09.692380","status":"completed"},"tags":[]}
# =========================
# CELDA 2: Helpers
# =========================
def safe_name(s: str) -> str:
    return re.sub(r"[\\/:*?\"<>|]+", "_", s).strip()

def robust_norm_1d(x):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-6
    z = (x - med) / (1.4826 * mad)
    return np.clip(z, -10, 10).astype(np.float32)

def to_1ch_myoware(emg):
    """
    Convierte EMG multicanal a un canal "modo MyoWare".
    Usa mediana por canal para robustez.
    """
    x = np.asarray(emg)

    if x.ndim == 1:
        return robust_norm_1d(x)

    if x.ndim != 2:
        raise ValueError(f"ndim inesperado: {x.ndim}")

    # Heurística: si viene como (C,T) y C <= 32, trasponer
    if x.shape[0] <= 32 and x.shape[0] < x.shape[1]:
        x = x.T

    # x debería quedar (T, C)
    if x.shape[1] > 1:
        x1 = np.median(x.astype(np.float32), axis=1)
    else:
        x1 = x[:, 0].astype(np.float32)

    return robust_norm_1d(x1)

def pick_label(d):
    if "restimulus" in d:
        return np.asarray(d["restimulus"]).squeeze()
    if "stimulus" in d:
        return np.asarray(d["stimulus"]).squeeze()
    return None

def windowize_signal_and_label(x, y=None, win=400, stride=200):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    T = len(x)
    if T < win:
        return None, None

    idxs = np.arange(0, T - win + 1, stride)
    X = np.stack([x[i:i+win] for i in idxs], axis=0).astype(np.float32)

    if y is None:
        return X, None

    y = np.asarray(y).reshape(-1)
    if len(y) != T:
        return X, None

    Y = np.zeros((len(idxs),), dtype=np.int32)
    for j, i in enumerate(idxs):
        seg = y[i:i+win]
        vals, cnts = np.unique(seg, return_counts=True)
        Y[j] = int(vals[np.argmax(cnts)])
    return X, Y

def compute_quality_score(x_clean, x_corr):
    """
    Score 0..1. 1 = muy sana, 0 = muy dañada
    Basado en SNR relativo aproximado.
    """
    x_clean = np.asarray(x_clean, dtype=np.float32)
    x_corr  = np.asarray(x_corr, dtype=np.float32)

    noise = x_corr - x_clean
    p_sig = float(np.mean(x_clean**2) + 1e-8)
    p_noi = float(np.mean(noise**2) + 1e-8)
    snr_db = 10.0 * math.log10(p_sig / p_noi)

    # mapear aproximadamente [-5, 25] dB -> [0,1]
    q = (snr_db + 5.0) / 30.0
    q = float(np.clip(q, 0.0, 1.0))
    return q, snr_db

def corrupt_window(xw, rng):
    """
    Devuelve:
      x_corr: señal corrupta
      meta: dict con tipos de corrupción aplicados
    """
    x = xw.astype(np.float32).copy()
    applied = []

    # ruido gaussiano
    if rng.random() < 0.65:
        sigma = rng.uniform(0.05, 0.35)
        x += rng.normal(0, sigma, size=x.shape).astype(np.float32)
        applied.append(f"gaussian_sigma={sigma:.3f}")

    # drift
    if rng.random() < 0.45:
        amp = rng.uniform(0.2, 1.2)
        freq = rng.uniform(0.2, 1.5)
        t = np.linspace(0, 1, len(x), dtype=np.float32)
        x += (amp * np.sin(2*np.pi*freq*t)).astype(np.float32)
        applied.append(f"drift_amp={amp:.3f}_freq={freq:.3f}")

    # dropout
    if rng.random() < 0.25:
        n = int(rng.integers(10, 80))
        start = int(rng.integers(0, len(x)-n))
        x[start:start+n] = 0.0
        applied.append(f"dropout_len={n}")

    # clipping
    if rng.random() < 0.25:
        clipv = rng.uniform(2.0, 6.0)
        x = np.clip(x, -clipv, clipv)
        applied.append(f"clip={clipv:.3f}")

    # impulsos
    if rng.random() < 0.20:
        k = int(rng.integers(1, 5))
        for _ in range(k):
            pos = int(rng.integers(0, len(x)))
            x[pos] += float(rng.uniform(-8, 8))
        applied.append(f"impulses={k}")

    if not applied:
        applied.append("clean_like")

    return x.astype(np.float32), {"corruptions": applied}

def write_npy(folder, name, arr):
    folder = OUT_ROOT / folder
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / name, arr)

def shard_save(folder, prefix, arrays, shard_idx):
    folder = OUT_ROOT / folder
    folder.mkdir(parents=True, exist_ok=True)
    for k, v in arrays.items():
        np.save(folder / f"{k}_{shard_idx:04d}.npy", v)

# %% [code]
# =========================
# CELDA 3: Localizar .npz y separar por participante (FIX leakage)
# =========================
npz_files = sorted(glob.glob(os.path.join(DB10_EMG, "**", "*.npz"), recursive=True))
npz_files += sorted(glob.glob(os.path.join(NINAPRO_EMG, "**", "*.npz"), recursive=True))

if MAX_FILES is not None:
    npz_files = npz_files[:MAX_FILES]

print("Total .npz encontrados:", len(npz_files))

train_files, val_files, test_files = group_split_files(
    npz_files, frac_val=FRAC_VAL, frac_test=FRAC_TEST, seed=SEED
)

split_manifest = {
    "train_files": train_files,
    "val_files": val_files,
    "test_files": test_files,
    "seed": SEED,
    "frac_val": FRAC_VAL,
    "frac_test": FRAC_TEST,
}
with open(OUT_ROOT / "split_manifest.json", "w") as f:
    json.dump(split_manifest, f, indent=2)

print("Archivos -> train:", len(train_files), "val:", len(val_files), "test:", len(test_files))


# %% [code]
# =========================
# CELDA 4: Construcción de datasets (por split, sin mezclar participantes)
# =========================
def build_split(split_name, files_list, out_root):
    rng = np.random.default_rng(SEED)

    intent_X_buf, intent_Y_buf = [], []
    quality_X_buf, quality_Qscore_buf, quality_Qlabel_buf = [], [], []
    rest_Xcorr_buf, rest_Xclean_buf, rest_Qscore_buf = [], [], []
    quality_meta = []

    intent_shard = quality_shard = rest_shard = 0
    total_windows = 0
    total_intent1 = 0

    def shard_save_local(folder, arrays, shard_idx):
        folder = out_root / split_name / folder
        folder.mkdir(parents=True, exist_ok=True)
        for k, v in arrays.items():
            np.save(folder / f"{k}_{shard_idx:04d}.npy", v)

    for fi, fp in enumerate(files_list, start=1):
        try:
            d = np.load(fp, allow_pickle=True)
            emg_key = None
            for k in d.files:
                if "emg" in k.lower():
                    emg_key = k
                    break
            if emg_key is None:
                continue

            emg = d[emg_key]
            label = pick_label(d)
            x1 = to_1ch_myoware(emg)
            Xw, Yw = windowize_signal_and_label(x1, label, WIN, STRIDE)
            if Xw is None or Yw is None:
                continue

            Y_intent = (Yw != 0).astype(np.int32)
            total_windows += len(Xw)
            total_intent1 += int(Y_intent.sum())

            subj_id = extract_subject_id(fp)

            for j in range(len(Xw)):
                x_clean = Xw[j]
                y_bin = Y_intent[j]

                intent_X_buf.append(x_clean[None, :])
                intent_Y_buf.append(np.array([y_bin], dtype=np.float32))

                x_corr, meta = corrupt_window(x_clean, rng)
                q_score, snr_db = compute_quality_score(x_clean, x_corr)
                q_label = 1 if q_score >= 0.60 else 0

                quality_X_buf.append(x_corr[None, :])
                quality_Qscore_buf.append(np.array([q_score], dtype=np.float32))
                quality_Qlabel_buf.append(np.array([q_label], dtype=np.float32))
                quality_meta.append({
                    "file": fp, "subject_id": subj_id, "idx": j,
                    "intent": int(y_bin), "q_score": float(q_score),
                    "q_label": int(q_label), "snr_db": float(snr_db),
                    "corruptions": meta["corruptions"],
                })

                if y_bin == 1:
                    rest_Xcorr_buf.append(x_corr[None, :])
                    rest_Xclean_buf.append(x_clean[None, :])
                    rest_Qscore_buf.append(np.array([q_score], dtype=np.float32))

                if len(intent_X_buf) >= SHARD_SIZE:
                    shard_save_local("intent_dataset", {
                        "X": np.stack(intent_X_buf).astype(np.float32),
                        "Y": np.concatenate(intent_Y_buf).astype(np.float32),
                    }, intent_shard)
                    intent_X_buf, intent_Y_buf = [], []
                    intent_shard += 1

                if len(quality_X_buf) >= SHARD_SIZE:
                    shard_save_local("quality_dataset", {
                        "X": np.stack(quality_X_buf).astype(np.float32),
                        "Qscore": np.concatenate(quality_Qscore_buf).astype(np.float32),
                        "Qlabel": np.concatenate(quality_Qlabel_buf).astype(np.float32),
                    }, quality_shard)
                    meta_dir = out_root / split_name / "quality_dataset"
                    meta_dir.mkdir(parents=True, exist_ok=True)
                    with open(meta_dir / f"meta_{quality_shard:04d}.json", "w") as f:
                        json.dump(quality_meta, f)
                    quality_X_buf, quality_Qscore_buf, quality_Qlabel_buf, quality_meta = [], [], [], []
                    quality_shard += 1

                if len(rest_Xcorr_buf) >= SHARD_SIZE:
                    shard_save_local("restoration_dataset", {
                        "Xcorr": np.stack(rest_Xcorr_buf).astype(np.float32),
                        "Xclean": np.stack(rest_Xclean_buf).astype(np.float32),
                        "Qscore": np.concatenate(rest_Qscore_buf).astype(np.float32),
                    }, rest_shard)
                    rest_Xcorr_buf, rest_Xclean_buf, rest_Qscore_buf = [], [], []
                    rest_shard += 1

            if fi % 20 == 0:
                print(f"[{split_name}] [{fi}/{len(files_list)}] total_windows={total_windows}")

        except Exception as e:
            print("ERROR en", fp, "->", repr(e))

    if len(intent_X_buf):
        shard_save_local("intent_dataset", {
            "X": np.stack(intent_X_buf).astype(np.float32),
            "Y": np.concatenate(intent_Y_buf).astype(np.float32),
        }, intent_shard)
    if len(quality_X_buf):
        shard_save_local("quality_dataset", {
            "X": np.stack(quality_X_buf).astype(np.float32),
            "Qscore": np.concatenate(quality_Qscore_buf).astype(np.float32),
            "Qlabel": np.concatenate(quality_Qlabel_buf).astype(np.float32),
        }, quality_shard)
        meta_dir = out_root / split_name / "quality_dataset"
        meta_dir.mkdir(parents=True, exist_ok=True)
        with open(meta_dir / f"meta_{quality_shard:04d}.json", "w") as f:
            json.dump(quality_meta, f)
    if len(rest_Xcorr_buf):
        shard_save_local("restoration_dataset", {
            "Xcorr": np.stack(rest_Xcorr_buf).astype(np.float32),
            "Xclean": np.stack(rest_Xclean_buf).astype(np.float32),
            "Qscore": np.concatenate(rest_Qscore_buf).astype(np.float32),
        }, rest_shard)

    print(f"[{split_name}] Listo. Total windows: {total_windows}  intent=1: {total_intent1} "
          f"({total_intent1/max(total_windows,1):.2%})")


build_split("train", train_files, OUT_ROOT)
build_split("val",   val_files,   OUT_ROOT)
build_split("test",  test_files,  OUT_ROOT)


# %% [code] {"execution":{"iopub.execute_input":"2026-03-11T16:51:43.912357Z","iopub.status.busy":"2026-03-11T16:51:43.911148Z","iopub.status.idle":"2026-03-11T16:51:44.462374Z","shell.execute_reply":"2026-03-11T16:51:44.461289Z"},"papermill":{"duration":0.559201,"end_time":"2026-03-11T16:51:44.464846","exception":false,"start_time":"2026-03-11T16:51:43.905645","status":"completed"},"tags":[]}
# =========================
# CELDA 5 (corregida): Inspección rápida (por split: train/val/test)
# =========================
import glob, os, json
from collections import Counter

for split_name in ["train", "val", "test"]:
    intent_files = sorted(glob.glob(str(OUT_ROOT / split_name / "intent_dataset" / "X_*.npy")))
    quality_files = sorted(glob.glob(str(OUT_ROOT / split_name / "quality_dataset" / "X_*.npy")))
    rest_xcorr_files = sorted(glob.glob(str(OUT_ROOT / split_name / "restoration_dataset" / "Xcorr_*.npy")))

    print(f"--- {split_name} ---")
    print("  intent shards:", len(intent_files))
    print("  quality shards:", len(quality_files))
    print("  restoration shards:", len(rest_xcorr_files))

intent_files = sorted(glob.glob(str(OUT_ROOT / "train" / "intent_dataset" / "X_*.npy")))
quality_files = sorted(glob.glob(str(OUT_ROOT / "train" / "quality_dataset" / "X_*.npy")))

if len(intent_files) == 0 or len(quality_files) == 0:
    raise FileNotFoundError(
        "No se encontraron shards en train/. Revisa la ruta OUT_ROOT."
    )

Xi = np.load(intent_files[0], mmap_mode="r")
Yi = np.load(str(OUT_ROOT / "train" / "intent_dataset" / "Y_0000.npy"), mmap_mode="r")

Xq = np.load(quality_files[0], mmap_mode="r")
Qscore = np.load(str(OUT_ROOT / "train" / "quality_dataset" / "Qscore_0000.npy"), mmap_mode="r")
Qlabel = np.load(str(OUT_ROOT / "train" / "quality_dataset" / "Qlabel_0000.npy"), mmap_mode="r")

print("\nIntent X:", Xi.shape, Xi.dtype)
print("Intent Y:", Yi.shape, Yi.dtype, "pos frac:", Yi.mean())

print("Quality X:", Xq.shape, Xq.dtype)
print("Qscore:", Qscore.shape, Qscore.dtype, "min/max:", Qscore.min(), Qscore.max())
print("Qlabel balance:", Counter(Qlabel.astype(int).tolist()))

plt.figure(figsize=(8,4))
plt.hist(Qscore[:5000], bins=40)
plt.title("Distribución de quality score (train)")
plt.xlabel("q_score")
plt.ylabel("count")
plt.show()

plt.figure(figsize=(10,4))
plt.plot(Xi[0,0])
plt.title(f"Ejemplo señal intent_dataset (train) / label={Yi[0]}")
plt.show()
