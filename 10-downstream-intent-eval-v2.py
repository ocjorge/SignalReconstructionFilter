# %% [code]
# =========================
# CELDA 1: Imports + arquitecturas (copiadas de 06-validar-umbral-politica.ipynb)
# =========================
import os, glob, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=5, dilation=1, drop=0.1):
        super().__init__()
        pad = (k - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(drop)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        s = self.skip(x)
        x = self.drop(self.act(self.bn1(self.conv1(x))))
        x = self.drop(self.act(self.bn2(self.conv2(x))))
        return self.act(x + s)


class IntentTCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.tcn = nn.Sequential(
            TCNBlock(16, 16, dilation=1, drop=0.10),
            TCNBlock(16, 32, dilation=2, drop=0.10),
            TCNBlock(32, 32, dilation=4, drop=0.10),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.tcn(x)
        x = self.pool(x)
        return self.head(x)


class ConvAE1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 32, 5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.ConvTranspose1d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.ConvTranspose1d(16, 1, 4, stride=2, padding=1),
        )

    def forward(self, x):
        z = self.encoder(x)
        z = self.bottleneck(z)
        y = self.decoder(z)
        return y


# %% [code]
# =========================
# CELDA 2: Cargar modelos YA REENTRENADOS (con el dataset sin leakage)
# =========================
KAGGLE_MODELS_DIR = "/kaggle/input/models/jorgeoc/emg-modelos-v2/pytorch/default/1"
MODELS_DIR = "/kaggle/working/models_reconstructed"

import os, shutil

def ensure_pt_file(name):
    os.makedirs(MODELS_DIR, exist_ok=True)
    direct_path = os.path.join(KAGGLE_MODELS_DIR, f"{name}.pt")
    if os.path.isfile(direct_path):
        return direct_path

    exploded_outer = os.path.join(KAGGLE_MODELS_DIR, name)
    exploded_inner = os.path.join(exploded_outer, name)
    if os.path.isdir(exploded_inner) and os.path.isfile(os.path.join(exploded_inner, "data.pkl")):
        out_zip_noext = os.path.join(MODELS_DIR, name)
        shutil.make_archive(out_zip_noext, "zip", root_dir=exploded_outer, base_dir=name)
        out_pt = out_zip_noext + ".pt"
        os.replace(out_zip_noext + ".zip", out_pt)
        print(f"Reconstruido: {name}.pt <- {exploded_inner}")
        return out_pt

    raise FileNotFoundError(f"No se encontro {name}.pt ni como archivo ni como carpeta desempacada en {KAGGLE_MODELS_DIR}")


intent_path = ensure_pt_file("intent_binary_best")
restorer_path = ensure_pt_file("restorer_best")

intent_model = IntentTCN().to(device)
intent_model.load_state_dict(torch.load(intent_path, map_location=device))
intent_model.eval()

restorer_model = ConvAE1D().to(device)
restorer_model.load_state_dict(torch.load(restorer_path, map_location=device))
restorer_model.eval()

print("Modelos cargados.")


# %% [code]
# =========================
# CELDA 3: Cargar TEST (participantes nunca vistos, del notebook 01 parchado)
# =========================
ROOT = "/kaggle/input/datasets/jorgeoc/prepared-myoware-split-v2/prepared_myoware_v2"
RDIR_TEST = os.path.join(ROOT, "test", "restoration_dataset")

Xcorr_files = sorted(glob.glob(os.path.join(RDIR_TEST, "Xcorr_*.npy")))
Xclean_files = sorted(glob.glob(os.path.join(RDIR_TEST, "Xclean_*.npy")))

x_corr = np.concatenate([np.load(f) for f in Xcorr_files], axis=0)
x_clean = np.concatenate([np.load(f) for f in Xclean_files], axis=0)

print("x_corr:", x_corr.shape, "x_clean:", x_clean.shape)
# Recordatorio: todas estas ventanas tienen intencion verdadera = 1
# (asi se construyo restoration_dataset en el notebook 01)
y_true_intent = np.ones(len(x_corr), dtype=np.int32)


# %% [code]
# =========================
# CELDA 4: Generar señal restaurada con el autoencoder (batched)
# =========================
@torch.no_grad()
def restore_batch(x, batch_size=512):
    out = []
    for i in range(0, len(x), batch_size):
        xb = torch.tensor(x[i:i+batch_size], dtype=torch.float32, device=device)
        yb = restorer_model(xb).cpu().numpy()
        out.append(yb)
    return np.concatenate(out, axis=0).astype(np.float32)

x_restored = restore_batch(x_corr)
print("x_restored:", x_restored.shape)


# %% [code]
# =========================
# CELDA 5: Correr el clasificador de intencion sobre las 3 versiones
# =========================
@torch.no_grad()
def predict_intent_prob(x, batch_size=512):
    probs = []
    for i in range(0, len(x), batch_size):
        xb = torch.tensor(x[i:i+batch_size], dtype=torch.float32, device=device)
        logits = intent_model(xb)
        p = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        probs.append(p)
    return np.concatenate(probs)

p_corr = predict_intent_prob(x_corr)        # degradada, sin restaurar
p_restored = predict_intent_prob(x_restored) # restaurada (propuesta del paper)
p_clean = predict_intent_prob(x_clean)       # limpia (cota superior)


# %% [code]
# =========================
# CELDA 6: Metricas comparativas (a thr_intent=0.50)
# =========================
def detection_rate(p, thr=0.50):
    pred = (p >= thr).astype(np.int32)
    return float((pred == y_true_intent).mean())

THR = 0.50
rows = [
    {"signal": "degradada (sin restaurar)", "detection_rate": detection_rate(p_corr, THR), "mean_p_intent": float(p_corr.mean())},
    {"signal": "restaurada (autoencoder)",  "detection_rate": detection_rate(p_restored, THR), "mean_p_intent": float(p_restored.mean())},
    {"signal": "limpia (cota superior)",    "detection_rate": detection_rate(p_clean, THR), "mean_p_intent": float(p_clean.mean())},
]
result_df = pd.DataFrame(rows)
print(result_df.to_string(index=False))

missed_by_corr = (p_corr < THR)
recovered_by_restore = missed_by_corr & (p_restored >= THR)
print(f"\nVentanas no detectadas con senal degradada: {missed_by_corr.sum()} "
      f"({missed_by_corr.mean():.2%})")
print(f"De esas, recuperadas tras restauracion: {recovered_by_restore.sum()} "
      f"({recovered_by_restore.sum()/max(missed_by_corr.sum(),1):.2%})")

was_detected = (p_corr >= THR)
degraded_by_restore = was_detected & (p_restored < THR)
print(f"Ventanas antes detectadas y que la restauracion daña: {degraded_by_restore.sum()} "
      f"({degraded_by_restore.sum()/max(was_detected.sum(),1):.2%})")


# %% [code]
# =========================
# CELDA 7: Guardar resultados para el paper / cover letter
# =========================
result_df.to_csv("downstream_intent_eval_test.csv", index=False)
with open("downstream_intent_eval_summary.json", "w") as f:
    json.dump({
        "threshold": THR,
        "n_windows": int(len(x_corr)),
        "results": rows,
        "recovered_frac_of_missed": float(recovered_by_restore.sum() / max(missed_by_corr.sum(), 1)),
        "degraded_frac_of_detected": float(degraded_by_restore.sum() / max(was_detected.sum(), 1)),
    }, f, indent=2)
print("Guardado downstream_intent_eval_test.csv y .json")
