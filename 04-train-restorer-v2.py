# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T21:56:07.496737Z","iopub.status.busy":"2026-03-14T21:56:07.496093Z","iopub.status.idle":"2026-03-14T21:56:08.912949Z","shell.execute_reply":"2026-03-14T21:56:08.912218Z"},"papermill":{"duration":1.423925,"end_time":"2026-03-14T21:56:08.915309","exception":false,"start_time":"2026-03-14T21:56:07.491384","status":"completed"},"tags":[]}
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

# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T21:56:08.921866Z","iopub.status.busy":"2026-03-14T21:56:08.921374Z","iopub.status.idle":"2026-03-14T21:56:12.732887Z","shell.execute_reply":"2026-03-14T21:56:12.732041Z"},"papermill":{"duration":3.816194,"end_time":"2026-03-14T21:56:12.734450","exception":false,"start_time":"2026-03-14T21:56:08.918256","status":"completed"},"tags":[]}
# =========================
# CELDA 1: Dataset restauración
# =========================
import os
import glob
import time
import copy
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = "/kaggle/input/datasets/jorgeoc/prepared-myoware-split-v2/prepared_myoware_v2"
RDIR_TRAIN = os.path.join(ROOT, "train", "restoration_dataset")
RDIR_VAL = os.path.join(ROOT, "val", "restoration_dataset")

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("RDIR_TRAIN:", RDIR_TRAIN)
print("RDIR_VAL:", RDIR_VAL)

Xcorr_files_tr  = sorted(glob.glob(os.path.join(RDIR_TRAIN, "Xcorr_*.npy")))
Xclean_files_tr = sorted(glob.glob(os.path.join(RDIR_TRAIN, "Xclean_*.npy")))
Qscore_files_tr = sorted(glob.glob(os.path.join(RDIR_TRAIN, "Qscore_*.npy")))

Xcorr_files_val  = sorted(glob.glob(os.path.join(RDIR_VAL, "Xcorr_*.npy")))
Xclean_files_val = sorted(glob.glob(os.path.join(RDIR_VAL, "Xclean_*.npy")))
Qscore_files_val = sorted(glob.glob(os.path.join(RDIR_VAL, "Qscore_*.npy")))

print("train shards:", len(Xcorr_files_tr), " val shards:", len(Xcorr_files_val))

if len(Xcorr_files_tr) == 0:
    raise FileNotFoundError(f"No se encontraron shards de restoration en {RDIR_TRAIN}")

Xcorr_files, Xclean_files, Qscore_files = Xcorr_files_tr, Xclean_files_tr, Qscore_files_tr  # pruebas rapidas

# prueba rápida
xc0 = np.load(Xcorr_files[0], mmap_mode="r")
xy0 = np.load(Xclean_files[0], mmap_mode="r")
qs0 = np.load(Qscore_files[0], mmap_mode="r")

print("\nxc0 shape:", xc0.shape, xc0.dtype)
print("xy0 shape:", xy0.shape, xy0.dtype)
print("qs0 shape:", qs0.shape, qs0.dtype, "min/max:", float(qs0.min()), float(qs0.max()))

class ShardedRestorationDataset(Dataset):
    def __init__(self, Xcorr_files, Xclean_files, Qscore_files, max_per_shard=None, seed=123):
        self.samples = []
        rng = np.random.default_rng(seed)

        for xf, yf, qf in zip(Xcorr_files, Xclean_files, Qscore_files):
            X = np.load(xf, mmap_mode="r")
            n = len(X)
            idx = np.arange(n)

            if max_per_shard is not None and n > max_per_shard:
                idx = rng.choice(idx, size=max_per_shard, replace=False)

            for i in idx:
                self.samples.append((xf, yf, qf, int(i)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        xf, yf, qf, i = self.samples[idx]
        Xc = np.load(xf, mmap_mode="r")
        Xy = np.load(yf, mmap_mode="r")
        Qs = np.load(qf, mmap_mode="r")

        xcorr = torch.tensor(Xc[i], dtype=torch.float32)
        xclean = torch.tensor(Xy[i], dtype=torch.float32)
        qscore = torch.tensor([Qs[i]], dtype=torch.float32)
        return xcorr, xclean, qscore

def split_files(files, frac_val=0.15, seed=123):
    idx = np.arange(len(files))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = max(1, int(len(files) * frac_val))
    return idx[n_val:], idx[:n_val]

def mse_np(a, b):
    return float(np.mean((a - b) ** 2))

def mae_np(a, b):
    return float(np.mean(np.abs(a - b)))

def snr_db(clean, noisy):
    clean = np.asarray(clean)
    noisy = np.asarray(noisy)
    noise = noisy - clean
    ps = np.mean(clean**2) + 1e-8
    pn = np.mean(noise**2) + 1e-8
    return 10 * np.log10(ps / pn)

def corrcoef_1d(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T21:56:12.739945Z","iopub.status.busy":"2026-03-14T21:56:12.739561Z","iopub.status.idle":"2026-03-14T21:56:12.746050Z","shell.execute_reply":"2026-03-14T21:56:12.745492Z"},"papermill":{"duration":0.010674,"end_time":"2026-03-14T21:56:12.747292","exception":false,"start_time":"2026-03-14T21:56:12.736618","status":"completed"},"tags":[]}
# =========================
# CELDA 2: Modelo restaurador
# =========================
class ConvAE1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 5, stride=2, padding=2),  # 400 -> 200
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 32, 5, stride=2, padding=2), # 200 -> 100
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, 5, stride=2, padding=2), # 100 -> 50
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1), # 50 -> 100
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.ConvTranspose1d(32, 16, 4, stride=2, padding=1), # 100 -> 200
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.ConvTranspose1d(16, 1, 4, stride=2, padding=1),  # 200 -> 400
        )

    def forward(self, x):
        z = self.encoder(x)
        z = self.bottleneck(z)
        y = self.decoder(z)
        return y

# %% [code]
# =========================
# CELDA 2B: Cargar intent_model CONGELADO (guia para perdida task-aware)
# =========================
# FIX overnight: el restorer entrenado solo con MAE/MSE/SNR mejora
# metricas de senal pero empeora la deteccion de intencion downstream
# (ver notebook 10). Aqui se agrega un termino de perdida que usa el
# clasificador de intencion YA ENTRENADO (congelado, no se actualiza)
# como guia: la senal restaurada debe producir la MISMA prediccion de
# intencion que produce la senal limpia.
import torch.nn.functional as F


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


device_tmp = "cuda" if torch.cuda.is_available() else "cpu"

KAGGLE_MODELS_DIR = "/kaggle/input/models/jorgeoc/emg-modelos-v2/pytorch/default/1"
MODELS_SCRATCH_DIR = "/kaggle/working/models_reconstructed"

def ensure_pt_file(name):
    os.makedirs(MODELS_SCRATCH_DIR, exist_ok=True)
    direct_path = os.path.join(KAGGLE_MODELS_DIR, f"{name}.pt")
    if os.path.isfile(direct_path):
        return direct_path
    exploded_outer = os.path.join(KAGGLE_MODELS_DIR, name)
    exploded_inner = os.path.join(exploded_outer, name)
    if os.path.isdir(exploded_inner) and os.path.isfile(os.path.join(exploded_inner, "data.pkl")):
        import shutil
        out_zip_noext = os.path.join(MODELS_SCRATCH_DIR, name)
        shutil.make_archive(out_zip_noext, "zip", root_dir=exploded_outer, base_dir=name)
        out_pt = out_zip_noext + ".pt"
        os.replace(out_zip_noext + ".zip", out_pt)
        return out_pt
    raise FileNotFoundError(f"No se encontro {name}.pt en {KAGGLE_MODELS_DIR}")


intent_model = IntentTCN().to(device_tmp)
intent_model.load_state_dict(torch.load(ensure_pt_file("intent_binary_best"), map_location=device_tmp))
intent_model.eval()
for p in intent_model.parameters():
    p.requires_grad_(False)

print("intent_model (congelado) cargado para guiar el entrenamiento del restorer.")


# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T21:56:12.752600Z","iopub.status.busy":"2026-03-14T21:56:12.752377Z","iopub.status.idle":"2026-03-15T03:59:49.247555Z","shell.execute_reply":"2026-03-15T03:59:49.246864Z"},"papermill":{"duration":21816.503009,"end_time":"2026-03-15T03:59:49.252348","exception":false,"start_time":"2026-03-14T21:56:12.749339","status":"completed"},"tags":[]}
# =========================
# CELDA 3: Entrenamiento
# =========================

# FIX Revisor 2 #3-4: carpetas train/val ya separadas por participante
train_ds = ShardedRestorationDataset(
    Xcorr_files_tr, Xclean_files_tr, Qscore_files_tr,
    max_per_shard=4000,
    seed=1
)

val_ds = ShardedRestorationDataset(
    Xcorr_files_val, Xclean_files_val, Qscore_files_val,
    max_per_shard=4000,
    seed=2
)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

model = ConvAE1D().to(device)

l1 = nn.L1Loss()
mse = nn.MSELoss()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

LAMBDA_TASK = 2.0  # peso del termino task-aware, ajustable

def composite_loss(yhat, y):
    recon = 0.7 * l1(yhat, y) + 0.3 * mse(yhat, y)
    with torch.no_grad():
        p_clean = torch.sigmoid(intent_model(y))
    p_restored = torch.sigmoid(intent_model(yhat))
    task = F.mse_loss(p_restored, p_clean)
    total = recon + LAMBDA_TASK * task
    return total, recon.detach(), task.detach()

@torch.no_grad()
def evaluate(loader):
    model.eval()
    losses = []
    task_losses = []
    maes_before, maes_after = [], []
    mses_before, mses_after = [], []
    snr_before, snr_after = [], []
    corr_before, corr_after = [], []

    examples = []

    for xcorr, xclean, qscore in loader:
        xcorr = xcorr.to(device)
        xclean = xclean.to(device)

        xhat = model(xcorr)
        loss, recon_l, task_l = composite_loss(xhat, xclean)
        losses.append(loss.item() * xcorr.size(0))
        task_losses.append(task_l.item() * xcorr.size(0))

        xc = xclean.cpu().numpy()
        xn = xcorr.cpu().numpy()
        xr = xhat.cpu().numpy()

        for i in range(len(xc)):
            clean = xc[i,0]
            noisy = xn[i,0]
            rest  = xr[i,0]

            maes_before.append(mae_np(clean, noisy))
            maes_after.append(mae_np(clean, rest))
            mses_before.append(mse_np(clean, noisy))
            mses_after.append(mse_np(clean, rest))
            snr_before.append(snr_db(clean, noisy))
            snr_after.append(snr_db(clean, rest))
            corr_before.append(corrcoef_1d(clean, noisy))
            corr_after.append(corrcoef_1d(clean, rest))

        if len(examples) < 5:
            examples.append((xn[0,0], xr[0,0], xc[0,0]))

    mets = {
        "loss": np.sum(losses) / max(len(maes_before), 1),
        "task_loss": np.sum(task_losses) / max(len(maes_before), 1),
        "mae_before": float(np.mean(maes_before)),
        "mae_after": float(np.mean(maes_after)),
        "mse_before": float(np.mean(mses_before)),
        "mse_after": float(np.mean(mses_after)),
        "snr_before": float(np.mean(snr_before)),
        "snr_after": float(np.mean(snr_after)),
        "corr_before": float(np.mean(corr_before)),
        "corr_after": float(np.mean(corr_after)),
        "delta_mae": float(np.mean(maes_before) - np.mean(maes_after)),
        "delta_snr": float(np.mean(snr_after) - np.mean(snr_before)),
    }
    return mets, examples

history = {
    "train_loss": [], "val_loss": [],
    "mae_before": [], "mae_after": [],
    "snr_before": [], "snr_after": [],
    "corr_before": [], "corr_after": [],
}

best_state = None
best_score = -1e9
patience = 5
bad_epochs = 0
EPOCHS = 30

for ep in range(1, EPOCHS + 1):
    model.train()
    t0 = time.time()
    train_loss_sum, n_seen = 0.0, 0

    for xcorr, xclean, qscore in train_loader:
        xcorr = xcorr.to(device)
        xclean = xclean.to(device)

        opt.zero_grad()
        xhat = model(xcorr)
        loss, recon_l, task_l = composite_loss(xhat, xclean)
        loss.backward()
        opt.step()

        train_loss_sum += loss.item() * xcorr.size(0)
        n_seen += xcorr.size(0)

    train_loss = train_loss_sum / max(n_seen, 1)
    val_mets, examples = evaluate(val_loader)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_mets["loss"])
    history["mae_before"].append(val_mets["mae_before"])
    history["mae_after"].append(val_mets["mae_after"])
    history["snr_before"].append(val_mets["snr_before"])
    history["snr_after"].append(val_mets["snr_after"])
    history["corr_before"].append(val_mets["corr_before"])
    history["corr_after"].append(val_mets["corr_after"])

    score = (val_mets["delta_snr"] + val_mets["delta_mae"]
              + (val_mets["corr_after"] - val_mets["corr_before"])
              - 5.0 * val_mets["task_loss"])  # penaliza divergencia con la prediccion de intencion en senal limpia

    print(
        f"Epoch {ep:02d} | train_loss={train_loss:.4f} | val_loss={val_mets['loss']:.4f} | "
        f"task_loss={val_mets['task_loss']:.4f} | "
        f"MAE {val_mets['mae_before']:.4f}->{val_mets['mae_after']:.4f} | "
        f"SNR {val_mets['snr_before']:.2f}->{val_mets['snr_after']:.2f} | "
        f"Corr {val_mets['corr_before']:.4f}->{val_mets['corr_after']:.4f} | "
        f"time={time.time()-t0:.1f}s"
    )

    if score > best_score:
        best_score = score
        best_state = copy.deepcopy(model.state_dict())
        bad_epochs = 0
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            print("Early stopping.")
            break

model.load_state_dict(best_state)
torch.save(model.state_dict(), "/kaggle/working/restorer_taskaware_best.pt")
print("Guardado: /kaggle/working/restorer_taskaware_best.pt")
print("(nombre distinto a proposito, para no perder el restorer_best.pt original)")

# %% [code] {"execution":{"iopub.execute_input":"2026-03-15T03:59:49.259805Z","iopub.status.busy":"2026-03-15T03:59:49.259438Z","iopub.status.idle":"2026-03-15T04:01:19.130592Z","shell.execute_reply":"2026-03-15T04:01:19.129832Z"},"papermill":{"duration":89.877074,"end_time":"2026-03-15T04:01:19.132485","exception":false,"start_time":"2026-03-15T03:59:49.255411","status":"completed"},"tags":[]}
# =========================
# CELDA 4: Gráficas restauración
# =========================
val_mets, examples = evaluate(val_loader)
print("Métricas finales restaurador:", val_mets)

plt.figure(figsize=(8,4))
plt.plot(history["train_loss"], label="train_loss")
plt.plot(history["val_loss"], label="val_loss")
plt.title("Restaurador - loss")
plt.legend()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(history["mae_before"], label="mae_before")
plt.plot(history["mae_after"], label="mae_after")
plt.title("Restaurador - MAE")
plt.legend()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(history["snr_before"], label="snr_before")
plt.plot(history["snr_after"], label="snr_after")
plt.title("Restaurador - SNR")
plt.legend()
plt.show()

for i, (xn, xr, xc) in enumerate(examples):
    plt.figure(figsize=(12,3))
    plt.plot(xn, label="corrupta", alpha=0.8)
    plt.plot(xr, label="restaurada", alpha=0.8)
    plt.plot(xc, label="limpia", alpha=0.8)
    plt.title(f"Ejemplo {i}")
    plt.legend()
    plt.show()