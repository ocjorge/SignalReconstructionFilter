# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T18:37:17.253845Z","iopub.status.busy":"2026-03-14T18:37:17.253487Z","iopub.status.idle":"2026-03-14T18:37:19.527699Z","shell.execute_reply":"2026-03-14T18:37:19.526672Z"},"papermill":{"duration":2.282658,"end_time":"2026-03-14T18:37:19.529825","exception":false,"start_time":"2026-03-14T18:37:17.247167","status":"completed"},"tags":[]}
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

# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T18:37:19.539293Z","iopub.status.busy":"2026-03-14T18:37:19.538444Z","iopub.status.idle":"2026-03-14T18:37:23.570361Z","shell.execute_reply":"2026-03-14T18:37:23.569354Z"},"papermill":{"duration":4.038814,"end_time":"2026-03-14T18:37:23.572535","exception":false,"start_time":"2026-03-14T18:37:19.533721","status":"completed"},"tags":[]}
# =========================
# CELDA 1: Imports
# =========================
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T18:37:23.582206Z","iopub.status.busy":"2026-03-14T18:37:23.581180Z","iopub.status.idle":"2026-03-14T18:37:23.600071Z","shell.execute_reply":"2026-03-14T18:37:23.599106Z"},"papermill":{"duration":0.025701,"end_time":"2026-03-14T18:37:23.601963","exception":false,"start_time":"2026-03-14T18:37:23.576262","status":"completed"},"tags":[]}
# =========================
# CELDA 2: Arquitecturas
# =========================
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

class QualityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.head_score = nn.Linear(32, 1)
        self.head_label = nn.Linear(32, 1)

    def forward(self, x):
        z = self.features(x)
        z = self.shared(z)
        q_score = torch.sigmoid(self.head_score(z))
        q_logit = self.head_label(z)
        return q_score, q_logit

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


# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T18:37:23.610866Z","iopub.status.busy":"2026-03-14T18:37:23.610550Z","iopub.status.idle":"2026-03-14T18:37:23.958852Z","shell.execute_reply":"2026-03-14T18:37:23.957832Z"},"papermill":{"duration":0.354898,"end_time":"2026-03-14T18:37:23.960665","exception":false,"start_time":"2026-03-14T18:37:23.605767","status":"completed"},"tags":[]}
# =========================
# CELDA 3: Cargar modelos
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

KAGGLE_MODELS_DIR = "/kaggle/input/models/jorgeoc/emg-modelos-v2/pytorch/default/1"
MODELS_DIR = "/kaggle/working/models_reconstructed"

# FIX: Kaggle a veces "desempaca" los .pt (que internamente son .zip) en
# carpetas al subirlos como Model, en vez de dejarlos como archivo unico.
# Esta funcion detecta ambos casos y reconstruye el .pt si hace falta.
import os, shutil

def ensure_pt_file(name):
    os.makedirs(MODELS_DIR, exist_ok=True)
    direct_path = os.path.join(KAGGLE_MODELS_DIR, f"{name}.pt")
    if os.path.isfile(direct_path):
        return direct_path

    # Buscar la carpeta "desempacada": .../<name>/<name>/{data.pkl,...}
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
quality_path = ensure_pt_file("quality_judge_best")
restorer_path = ensure_pt_file("restorer_best")

intent_model = IntentTCN().to(device)
intent_model.load_state_dict(torch.load(intent_path, map_location=device))
intent_model.eval()

quality_model = QualityNet().to(device)
quality_model.load_state_dict(torch.load(quality_path, map_location=device))
quality_model.eval()

restorer_model = ConvAE1D().to(device)
restorer_model.load_state_dict(torch.load(restorer_path, map_location=device))
restorer_model.eval()

print("Modelos cargados correctamente.")


# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T18:37:23.969733Z","iopub.status.busy":"2026-03-14T18:37:23.969404Z","iopub.status.idle":"2026-03-14T18:37:24.982982Z","shell.execute_reply":"2026-03-14T18:37:24.982089Z"},"papermill":{"duration":1.020189,"end_time":"2026-03-14T18:37:24.984768","exception":false,"start_time":"2026-03-14T18:37:23.964579","status":"completed"},"tags":[]}
# =========================
# CELDA 4: Utilidades + dataset
# =========================
def mae_np(a, b):
    return float(np.mean(np.abs(a - b)))

def mse_np(a, b):
    return float(np.mean((a - b) ** 2))

def snr_db(clean, noisy):
    noise = noisy - clean
    ps = np.mean(clean**2) + 1e-8
    pn = np.mean(noise**2) + 1e-8
    return float(10 * np.log10(ps / pn))

def corrcoef_1d(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

# FIX Revisor 2 #2: ya no se toma "el primer shard disponible" sin
# particion. Se usan las carpetas val/ y test/ generadas por el
# notebook 01 parchado (separadas por participante).
ROOT = "/kaggle/input/datasets/jorgeoc/prepared-myoware-split-v2/prepared_myoware_v2"

def load_full_split(split_name, max_n=None):
    rdir = os.path.join(ROOT, split_name, "restoration_dataset")
    xcorr_files = sorted(glob.glob(os.path.join(rdir, "Xcorr_*.npy")))
    xclean_files = sorted(glob.glob(os.path.join(rdir, "Xclean_*.npy")))
    if len(xcorr_files) == 0:
        raise FileNotFoundError(f"No se encontraron shards en {rdir}")
    xc = np.concatenate([np.load(f) for f in xcorr_files], axis=0)
    xl = np.concatenate([np.load(f) for f in xclean_files], axis=0)
    if max_n is not None and len(xc) > max_n:
        rng = np.random.default_rng(123)
        idx = rng.choice(len(xc), size=max_n, replace=False)
        xc, xl = xc[idx], xl[idx]
    return xc, xl

# VALIDACION: se usa para elegir los umbrales (barrido de grid)
x_corr_val, x_clean_val = load_full_split("val", max_n=6000)
print("VAL  -> x_corr:", x_corr_val.shape, "x_clean:", x_clean_val.shape)

# TEST: se toca UNA sola vez, al final, con el umbral ya elegido
x_corr_test, x_clean_test = load_full_split("test", max_n=6000)
print("TEST -> x_corr:", x_corr_test.shape, "x_clean:", x_clean_test.shape)

# Alias para compatibilidad con celdas de ejemplos mas abajo (10, 11, etc.)
x_corr, x_clean = x_corr_val, x_clean_val


# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T18:37:24.994906Z","iopub.status.busy":"2026-03-14T18:37:24.994600Z","iopub.status.idle":"2026-03-14T18:37:25.003640Z","shell.execute_reply":"2026-03-14T18:37:25.002727Z"},"papermill":{"duration":0.016548,"end_time":"2026-03-14T18:37:25.005413","exception":false,"start_time":"2026-03-14T18:37:24.988865","status":"completed"},"tags":[]}
# =========================
# CELDA 5: Política parametrizable
# =========================
@torch.no_grad()
def run_pipeline_on_batch(x_batch, thr_intent=0.50, thr_quality=0.50):
    xb = torch.tensor(x_batch, dtype=torch.float32, device=device)

    intent_logits = intent_model(xb)
    p_intent = torch.sigmoid(intent_logits).cpu().numpy().reshape(-1)

    q_score_pred, q_logit = quality_model(xb)
    q_score_pred = q_score_pred.cpu().numpy().reshape(-1)
    q_prob_good = torch.sigmoid(q_logit).cpu().numpy().reshape(-1)

    out = []
    restored = []

    for i in range(len(x_batch)):
        pi = float(p_intent[i])
        qg = float(q_prob_good[i])
        qs = float(q_score_pred[i])

        if pi < thr_intent:
            decision = "skip_restore"
            reason = "no_intent_detected"
        elif qg >= thr_quality:
            decision = "skip_restore"
            reason = "signal_quality_good_enough"
        else:
            decision = "restore"
            reason = "intent_detected_and_quality_bad"

        if decision == "restore":
            xr = restorer_model(xb[i:i+1]).cpu().numpy()[0]
        else:
            xr = x_batch[i].copy()

        restored.append(xr)
        out.append({
            "p_intent": pi,
            "q_score_pred": qs,
            "q_prob_good": qg,
            "decision": decision,
            "reason": reason
        })

    return out, np.stack(restored).astype(np.float32)


# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T18:37:25.014391Z","iopub.status.busy":"2026-03-14T18:37:25.014044Z","iopub.status.idle":"2026-03-14T18:37:28.172550Z","shell.execute_reply":"2026-03-14T18:37:28.171580Z"},"papermill":{"duration":3.164968,"end_time":"2026-03-14T18:37:28.174306","exception":false,"start_time":"2026-03-14T18:37:25.009338","status":"completed"},"tags":[]}
# =========================
# CELDA 6: Evaluar configuración
# =========================
def evaluate_pipeline_config(x_corr, x_clean, thr_intent=0.50, thr_quality=0.50):
    rows, x_out = run_pipeline_on_batch(x_corr, thr_intent=thr_intent, thr_quality=thr_quality)

    records = []
    for i, row in enumerate(rows):
        clean = x_clean[i, 0]
        corr  = x_corr[i, 0]
        out   = x_out[i, 0]

        records.append({
            **row,
            "mae_before": mae_np(clean, corr),
            "mae_after": mae_np(clean, out),
            "mse_before": mse_np(clean, corr),
            "mse_after": mse_np(clean, out),
            "snr_before": snr_db(clean, corr),
            "snr_after": snr_db(clean, out),
            "corr_before": corrcoef_1d(clean, corr),
            "corr_after": corrcoef_1d(clean, out),
            "improved_mae": mae_np(clean, out) < mae_np(clean, corr),
            "improved_snr": snr_db(clean, out) > snr_db(clean, corr),
            # FIX SNR: potencia de senal por ventana, para poder poolear
            # correctamente despues (todas las ventanas tienen el mismo
            # largo, asi que promediar sig_power y mse por separado y
            # luego dividir SI es valido; promediar SNR_db directamente NO).
            "sig_power": np.mean(clean ** 2),
        })

    df = pd.DataFrame(records)
    return df, x_out


def pooled_snr_db(df, mse_col):
    """SNR global correcto: pooled ps/pn, no promedio de dB por ventana."""
    ps = df["sig_power"].mean()
    pn = df[mse_col].mean() + 1e-8
    return 10 * np.log10(ps / pn)

df_base, x_out_base = evaluate_pipeline_config(x_corr, x_clean, thr_intent=0.50, thr_quality=0.50)
df_base.head()


# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T18:37:28.183840Z","iopub.status.busy":"2026-03-14T18:37:28.183533Z","iopub.status.idle":"2026-03-14T18:37:28.207034Z","shell.execute_reply":"2026-03-14T18:37:28.206103Z"},"papermill":{"duration":0.030322,"end_time":"2026-03-14T18:37:28.208751","exception":false,"start_time":"2026-03-14T18:37:28.178429","status":"completed"},"tags":[]}
# =========================
# CELDA 7: Resumen detallado
# =========================
print("Conteo de decisiones:")
print(df_base["decision"].value_counts(dropna=False))
print()

print("Razones:")
print(df_base["reason"].value_counts(dropna=False))
print()

print("Promedios globales:")
print(df_base[["mae_before", "mae_after", "snr_before", "snr_after", "corr_before", "corr_after"]].mean())
print()

dfr = df_base[df_base["decision"] == "restore"]
print("Solo restauradas:")
print(dfr[["mae_before", "mae_after", "snr_before", "snr_after", "corr_before", "corr_after"]].mean())
print("Frac mejora MAE:", dfr["improved_mae"].mean())
print("Frac mejora SNR:", dfr["improved_snr"].mean())
print()

dfs = df_base[df_base["decision"] == "skip_restore"]
print("No restauradas por razón:")
print(dfs["reason"].value_counts(normalize=True))


# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T18:37:28.219077Z","iopub.status.busy":"2026-03-14T18:37:28.218780Z","iopub.status.idle":"2026-03-14T18:37:28.824675Z","shell.execute_reply":"2026-03-14T18:37:28.823785Z"},"papermill":{"duration":0.613228,"end_time":"2026-03-14T18:37:28.826740","exception":false,"start_time":"2026-03-14T18:37:28.213512","status":"completed"},"tags":[]}
# =========================
# CELDA 8: No restauradas
# =========================
df_skip = df_base[df_base["decision"] == "skip_restore"].copy()

print("No restauradas totales:", len(df_skip))
print(df_skip["reason"].value_counts())

plt.figure(figsize=(8,4))
df_skip["reason"].value_counts().plot(kind="bar")
plt.title("Razones de no restauración")
plt.ylabel("count")
plt.show()

plt.figure(figsize=(8,4))
plt.hist(df_skip["p_intent"], bins=40)
plt.title("p_intent en no restauradas")
plt.show()

plt.figure(figsize=(8,4))
plt.hist(df_skip["q_prob_good"], bins=40)
plt.title("q_prob_good en no restauradas")
plt.show()


# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T18:37:28.840034Z","iopub.status.busy":"2026-03-14T18:37:28.839685Z","iopub.status.idle":"2026-03-14T18:37:29.484561Z","shell.execute_reply":"2026-03-14T18:37:29.483771Z"},"papermill":{"duration":0.653772,"end_time":"2026-03-14T18:37:29.486381","exception":false,"start_time":"2026-03-14T18:37:28.832609","status":"completed"},"tags":[]}
# =========================
# CELDA 9: Cuánto se restauran
# =========================
df_restore = df_base[df_base["decision"] == "restore"].copy()

df_restore["delta_mae"] = df_restore["mae_before"] - df_restore["mae_after"]
df_restore["delta_snr"] = df_restore["snr_after"] - df_restore["snr_before"]
df_restore["delta_corr"] = df_restore["corr_after"] - df_restore["corr_before"]

print("Restauradas totales:", len(df_restore))
print(df_restore[["delta_mae", "delta_snr", "delta_corr"]].describe())

plt.figure(figsize=(8,4))
plt.hist(df_restore["delta_mae"], bins=40)
plt.title("Cuánto mejora MAE al restaurar")
plt.show()

plt.figure(figsize=(8,4))
plt.hist(df_restore["delta_snr"], bins=40)
plt.title("Cuánto mejora SNR al restaurar")
plt.show()

plt.figure(figsize=(8,4))
plt.hist(df_restore["delta_corr"], bins=40)
plt.title("Cuánto mejora correlación al restaurar")
plt.show()


# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T18:37:29.499725Z","iopub.status.busy":"2026-03-14T18:37:29.499395Z","iopub.status.idle":"2026-03-14T18:37:30.231697Z","shell.execute_reply":"2026-03-14T18:37:30.230904Z"},"papermill":{"duration":0.741531,"end_time":"2026-03-14T18:37:30.234097","exception":false,"start_time":"2026-03-14T18:37:29.492566","status":"completed"},"tags":[]}
# =========================
# CELDA 10: Ejemplos restaurados
# =========================
restore_idx = df_base.index[df_base["decision"] == "restore"][:5]

for i in restore_idx:
    row = df_base.iloc[i]
    clean = x_clean[i, 0]
    corr = x_corr[i, 0]
    out = x_out_base[i, 0]

    print(f"idx={i}")
    print(f"reason={row['reason']}")
    print(f"p_intent={row['p_intent']:.3f} | q_score_pred={row['q_score_pred']:.3f} | q_prob_good={row['q_prob_good']:.3f}")
    print(f"MAE: {row['mae_before']:.4f} -> {row['mae_after']:.4f}")
    print(f"SNR: {row['snr_before']:.2f} -> {row['snr_after']:.2f}")
    print(f"Corr: {row['corr_before']:.3f} -> {row['corr_after']:.3f}")

    plt.figure(figsize=(12,3))
    plt.plot(corr, label="corrupta")
    plt.plot(out, label="restaurada")
    plt.plot(clean, label="limpia", alpha=0.8)
    plt.legend()
    plt.show()


# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T18:37:30.255998Z","iopub.status.busy":"2026-03-14T18:37:30.255382Z","iopub.status.idle":"2026-03-14T18:37:31.032143Z","shell.execute_reply":"2026-03-14T18:37:31.031444Z"},"papermill":{"duration":0.790682,"end_time":"2026-03-14T18:37:31.034669","exception":false,"start_time":"2026-03-14T18:37:30.243987","status":"completed"},"tags":[]}
# =========================
# CELDA 11: Ejemplos no restaurados
# =========================
skip_idx = df_base.index[df_base["decision"] == "skip_restore"][:5]

for i in skip_idx:
    row = df_base.iloc[i]
    clean = x_clean[i, 0]
    corr = x_corr[i, 0]
    out = x_out_base[i, 0]

    print(f"idx={i}")
    print(f"reason={row['reason']}")
    print(f"p_intent={row['p_intent']:.3f} | q_score_pred={row['q_score_pred']:.3f} | q_prob_good={row['q_prob_good']:.3f}")
    print(f"MAE: {row['mae_before']:.4f} -> {row['mae_after']:.4f}")
    print(f"SNR: {row['snr_before']:.2f} -> {row['snr_after']:.2f}")

    plt.figure(figsize=(12,3))
    plt.plot(corr, label="entrada")
    plt.plot(out, label="salida_pipeline")
    plt.plot(clean, label="limpia", alpha=0.8)
    plt.legend()
    plt.show()


# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T18:37:31.062021Z","iopub.status.busy":"2026-03-14T18:37:31.061698Z","iopub.status.idle":"2026-03-14T18:37:51.974896Z","shell.execute_reply":"2026-03-14T18:37:51.974163Z"},"papermill":{"duration":20.928209,"end_time":"2026-03-14T18:37:51.976516","exception":false,"start_time":"2026-03-14T18:37:31.048307","status":"completed"},"tags":[]}
# =========================
# CELDA 12: Barrido de umbrales
# =========================
intent_grid = [0.40, 0.50, 0.60]
quality_grid = [0.40, 0.50, 0.60]

grid_rows = []

for ti in intent_grid:
    for tq in quality_grid:
        # grid search SOLO sobre VALIDACION, nunca sobre test
        df_tmp, _ = evaluate_pipeline_config(x_corr_val, x_clean_val, thr_intent=ti, thr_quality=tq)

        dfr = df_tmp[df_tmp["decision"] == "restore"]
        restore_frac = len(dfr) / max(len(df_tmp), 1)

        row = {
            "thr_intent": ti,
            "thr_quality": tq,
            "restore_frac": restore_frac,
            "mae_before_global": df_tmp["mae_before"].mean(),
            "mae_after_global": df_tmp["mae_after"].mean(),
            # FIX SNR: usar pooled, no promedio de dB por ventana (sesgado
            # por ventanas de baja energia / reposo)
            "snr_before_global": pooled_snr_db(df_tmp, "mse_before"),
            "snr_after_global": pooled_snr_db(df_tmp, "mse_after"),
            "mae_gain_global": df_tmp["mae_before"].mean() - df_tmp["mae_after"].mean(),
            "snr_gain_global": pooled_snr_db(df_tmp, "mse_after") - pooled_snr_db(df_tmp, "mse_before"),
            "restore_mae_gain": dfr["mae_before"].mean() - dfr["mae_after"].mean() if len(dfr) else np.nan,
            "restore_snr_gain": (pooled_snr_db(dfr, "mse_after") - pooled_snr_db(dfr, "mse_before")) if len(dfr) else np.nan,
            "restore_mae_improve_frac": dfr["improved_mae"].mean() if len(dfr) else np.nan,
            "restore_snr_improve_frac": dfr["improved_snr"].mean() if len(dfr) else np.nan,
        }
        grid_rows.append(row)

grid_df = pd.DataFrame(grid_rows)
grid_df_sorted = grid_df.sort_values(["snr_gain_global", "mae_gain_global"], ascending=False)
print(grid_df_sorted)

best_row = grid_df_sorted.iloc[0]
BEST_THR_INTENT = float(best_row["thr_intent"])
BEST_THR_QUALITY = float(best_row["thr_quality"])
print(f"\nUmbral elegido en VALIDACION: thr_intent={BEST_THR_INTENT}, thr_quality={BEST_THR_QUALITY}")


# %% [code]
# =========================
# CELDA NUEVA: Evaluacion final, UNICA, sobre TEST (FIX Revisor 2 #2)
# =========================
df_test, x_out_test = evaluate_pipeline_config(
    x_corr_test, x_clean_test,
    thr_intent=BEST_THR_INTENT, thr_quality=BEST_THR_QUALITY
)

dfr_test = df_test[df_test["decision"] == "restore"]

print("=== Resultado final en TEST (participantes nunca vistos en train/val) ===")
print(f"thr_intent={BEST_THR_INTENT}  thr_quality={BEST_THR_QUALITY}")
print(f"restore_frac: {len(dfr_test)/max(len(df_test),1):.2%}")
mae_b = df_test["mae_before"].mean()
mae_a = df_test["mae_after"].mean()
# FIX SNR: pooled, no promedio de dB por ventana
snr_b = pooled_snr_db(df_test, "mse_before")
snr_a = pooled_snr_db(df_test, "mse_after")
print(f"MAE  antes -> despues: {mae_b:.4f} -> {mae_a:.4f}")
print(f"SNR  antes -> despues (pooled): {snr_b:.2f} dB -> {snr_a:.2f} dB  (delta={snr_a-snr_b:+.2f} dB)")


# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T18:37:52.002474Z","iopub.status.busy":"2026-03-14T18:37:52.002109Z","iopub.status.idle":"2026-03-14T18:37:52.290862Z","shell.execute_reply":"2026-03-14T18:37:52.290040Z"},"papermill":{"duration":0.303838,"end_time":"2026-03-14T18:37:52.292659","exception":false,"start_time":"2026-03-14T18:37:51.988821","status":"completed"},"tags":[]}
# =========================
# CELDA 13: Visualizar barrido
# =========================
print(grid_df.sort_values(["snr_gain_global", "mae_gain_global"], ascending=False))

pivot_snr = grid_df.pivot(index="thr_intent", columns="thr_quality", values="snr_gain_global")
pivot_mae = grid_df.pivot(index="thr_intent", columns="thr_quality", values="mae_gain_global")

plt.figure(figsize=(6,4))
plt.imshow(pivot_snr.values, aspect="auto")
plt.xticks(range(len(pivot_snr.columns)), pivot_snr.columns)
plt.yticks(range(len(pivot_snr.index)), pivot_snr.index)
plt.title("SNR gain global")
plt.xlabel("thr_quality")
plt.ylabel("thr_intent")
plt.colorbar()
plt.show()

plt.figure(figsize=(6,4))
plt.imshow(pivot_mae.values, aspect="auto")
plt.xticks(range(len(pivot_mae.columns)), pivot_mae.columns)
plt.yticks(range(len(pivot_mae.index)), pivot_mae.index)
plt.title("MAE gain global")
plt.xlabel("thr_quality")
plt.ylabel("thr_intent")
plt.colorbar()
plt.show()


# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T18:37:52.319808Z","iopub.status.busy":"2026-03-14T18:37:52.319498Z","iopub.status.idle":"2026-03-14T18:37:52.525435Z","shell.execute_reply":"2026-03-14T18:37:52.524654Z"},"papermill":{"duration":0.222747,"end_time":"2026-03-14T18:37:52.528089","exception":false,"start_time":"2026-03-14T18:37:52.305342","status":"completed"},"tags":[]}
# =========================
# CELDA 14: Señal sintética
# =========================
def make_synthetic_emg_window(T=400, seed=123):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, T)

    clean = np.zeros(T, dtype=np.float32)
    burst_center = 0.5
    env = np.exp(-0.5 * ((t - burst_center) / 0.12)**2)
    carrier = np.sin(2*np.pi*35*t) + 0.5*np.sin(2*np.pi*70*t)
    clean = (env * carrier).astype(np.float32)

    noisy = clean.copy()
    noisy += rng.normal(0, 0.25, size=T).astype(np.float32)
    noisy += (0.5 * np.sin(2*np.pi*1.2*t)).astype(np.float32)

    start = 140
    noisy[start:start+40] = 0.0

    return clean[None, None, :].astype(np.float32), noisy[None, None, :].astype(np.float32)

x_clean_syn, x_corr_syn = make_synthetic_emg_window()

rows_syn, x_out_syn = run_pipeline_on_batch(x_corr_syn, thr_intent=0.50, thr_quality=0.50)
row = rows_syn[0]

print(row)
print("MAE:", mae_np(x_clean_syn[0,0], x_corr_syn[0,0]), "->", mae_np(x_clean_syn[0,0], x_out_syn[0,0]))
print("SNR:", snr_db(x_clean_syn[0,0], x_corr_syn[0,0]), "->", snr_db(x_clean_syn[0,0], x_out_syn[0,0]))

plt.figure(figsize=(12,3))
plt.plot(x_corr_syn[0,0], label="sintética corrupta")
plt.plot(x_out_syn[0,0], label="salida pipeline")
plt.plot(x_clean_syn[0,0], label="sintética limpia", alpha=0.8)
plt.legend()
plt.show()


# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T18:37:52.558551Z","iopub.status.busy":"2026-03-14T18:37:52.558190Z","iopub.status.idle":"2026-03-14T18:37:52.568978Z","shell.execute_reply":"2026-03-14T18:37:52.568156Z"},"papermill":{"duration":0.028341,"end_time":"2026-03-14T18:37:52.570545","exception":false,"start_time":"2026-03-14T18:37:52.542204","status":"completed"},"tags":[]}
# =========================
# CELDA 15: Señal externa tipo MyoWare
# =========================
# Ejemplos de rutas:
# external_path = "/kaggle/input/mi-senal-myoware/senal.csv"
# external_path = "/kaggle/input/mi-senal-myoware/senal.npy"

external_path = None  # cámbialo cuando subas tu archivo

def robust_norm_1d(x):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-6
    z = (x - med) / (1.4826 * mad)
    return np.clip(z, -10, 10).astype(np.float32)

def load_external_signal(path):
    if path.endswith(".npy"):
        sig = np.load(path).astype(np.float32).reshape(-1)
    elif path.endswith(".csv"):
        sig = np.loadtxt(path, delimiter=",", dtype=np.float32).reshape(-1)
    elif path.endswith(".txt"):
        sig = np.loadtxt(path, dtype=np.float32).reshape(-1)
    else:
        raise ValueError("Formato no soportado. Usa .npy, .csv o .txt")
    return robust_norm_1d(sig)

def windows_1d(sig, win=400, stride=200):
    idxs = np.arange(0, len(sig) - win + 1, stride)
    return np.stack([sig[i:i+win] for i in idxs], axis=0)[:, None, :].astype(np.float32)

if external_path is not None:
    sig = load_external_signal(external_path)
    x_ext = windows_1d(sig, win=400, stride=200)

    rows_ext, x_out_ext = run_pipeline_on_batch(x_ext, thr_intent=0.50, thr_quality=0.50)
    df_ext = pd.DataFrame(rows_ext)

    print(df_ext["decision"].value_counts(dropna=False))
    print(df_ext["reason"].value_counts(dropna=False))

    i = 0
    print(df_ext.iloc[i].to_dict())

    plt.figure(figsize=(12,3))
    plt.plot(x_ext[i,0], label="entrada externa")
    plt.plot(x_out_ext[i,0], label="salida pipeline")
    plt.legend()
    plt.show()
else:
    print("Sube una señal externa y asigna external_path para probar MyoWare.")
