# %% [code]
# %% [code] {"execution":{"iopub.execute_input":"2026-03-12T04:11:27.812183Z","iopub.status.busy":"2026-03-12T04:11:27.811863Z","iopub.status.idle":"2026-03-12T04:11:29.823932Z","shell.execute_reply":"2026-03-12T04:11:29.822949Z"},"papermill":{"duration":2.018845,"end_time":"2026-03-12T04:11:29.825871","exception":false,"start_time":"2026-03-12T04:11:27.807026","status":"completed"},"tags":[]}
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

# %% [code] {"execution":{"iopub.execute_input":"2026-03-12T04:11:29.832370Z","iopub.status.busy":"2026-03-12T04:11:29.831628Z","iopub.status.idle":"2026-03-12T04:11:29.840278Z","shell.execute_reply":"2026-03-12T04:11:29.839136Z"},"papermill":{"duration":0.013329,"end_time":"2026-03-12T04:11:29.841843","exception":false,"start_time":"2026-03-12T04:11:29.828514","status":"completed"},"tags":[]}
import os
import glob

root = "/kaggle/input/datasets/jorgeoc/prepared-myoware-v2/prepared_myoware_v2"

print("Subcarpetas:")
for p in glob.glob(os.path.join(root, "*")):
    print(p)

print("\nArchivos quality:")
for p in glob.glob(os.path.join(root, "quality_dataset", "*"))[:20]:
    print(p)

print("\nArchivos restoration:")
for p in glob.glob(os.path.join(root, "restoration_dataset", "*"))[:20]:
    print(p)

# %% [code] {"execution":{"iopub.execute_input":"2026-03-12T04:11:29.847477Z","iopub.status.busy":"2026-03-12T04:11:29.847250Z","iopub.status.idle":"2026-03-12T04:11:29.856780Z","shell.execute_reply":"2026-03-12T04:11:29.856108Z"},"papermill":{"duration":0.014184,"end_time":"2026-03-12T04:11:29.858447","exception":false,"start_time":"2026-03-12T04:11:29.844263","status":"completed"},"tags":[]}
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname)

# %% [code] {"execution":{"iopub.execute_input":"2026-03-12T04:11:29.864643Z","iopub.status.busy":"2026-03-12T04:11:29.864418Z","iopub.status.idle":"2026-03-12T04:11:33.725339Z","shell.execute_reply":"2026-03-12T04:11:33.724350Z"},"papermill":{"duration":3.866317,"end_time":"2026-03-12T04:11:33.727224","exception":false,"start_time":"2026-03-12T04:11:29.860907","status":"completed"},"tags":[]}

# =========================
# CELDA 1: Dataset calidad
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
QDIR_TRAIN = os.path.join(ROOT, "train", "quality_dataset")
QDIR_VAL = os.path.join(ROOT, "val", "quality_dataset")

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

X_files_tr = sorted(glob.glob(os.path.join(QDIR_TRAIN, "X_*.npy")))
Qscore_files_tr = sorted(glob.glob(os.path.join(QDIR_TRAIN, "Qscore_*.npy")))
Qlabel_files_tr = sorted(glob.glob(os.path.join(QDIR_TRAIN, "Qlabel_*.npy")))

X_files_val = sorted(glob.glob(os.path.join(QDIR_VAL, "X_*.npy")))
Qscore_files_val = sorted(glob.glob(os.path.join(QDIR_VAL, "Qscore_*.npy")))
Qlabel_files_val = sorted(glob.glob(os.path.join(QDIR_VAL, "Qlabel_*.npy")))

print("train shards:", len(X_files_tr), " val shards:", len(X_files_val))

if len(X_files_tr) == 0:
    raise FileNotFoundError(f"No se encontraron shards de quality en {QDIR_TRAIN}")

X_files, Qscore_files, Qlabel_files = X_files_tr, Qscore_files_tr, Qlabel_files_tr  # para la celda de prueba rapida

# prueba rápida
x0 = np.load(X_files[0], mmap_mode="r")
s0 = np.load(Qscore_files[0], mmap_mode="r")
l0 = np.load(Qlabel_files[0], mmap_mode="r")

print("x0 shape:", x0.shape, x0.dtype)
print("s0 shape:", s0.shape, s0.dtype, "min/max:", float(s0.min()), float(s0.max()))
print("l0 shape:", l0.shape, l0.dtype, "pos frac:", float(l0.mean()))

class ShardedQualityDataset(Dataset):
    def __init__(self, X_files, Qscore_files, Qlabel_files, max_per_shard=None, seed=123):
        self.samples = []
        rng = np.random.default_rng(seed)

        for xf, sf, lf in zip(X_files, Qscore_files, Qlabel_files):
            X = np.load(xf, mmap_mode="r")
            n = len(X)
            idx = np.arange(n)

            if max_per_shard is not None and n > max_per_shard:
                idx = rng.choice(idx, size=max_per_shard, replace=False)

            for i in idx:
                self.samples.append((xf, sf, lf, int(i)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        xf, sf, lf, i = self.samples[idx]
        X = np.load(xf, mmap_mode="r")
        S = np.load(sf, mmap_mode="r")
        L = np.load(lf, mmap_mode="r")

        x = torch.tensor(X[i], dtype=torch.float32)
        s = torch.tensor([S[i]], dtype=torch.float32)  # score continuo
        l = torch.tensor([L[i]], dtype=torch.float32)  # label binaria
        return x, s, l

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

def binary_metrics_from_probs(y_true, y_prob, thr=0.5):
    y_true = np.asarray(y_true).astype(np.int32)
    y_prob = np.asarray(y_prob).astype(np.float32)
    y_pred = (y_prob >= thr).astype(np.int32)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    acc = (tp + tn) / max(len(y_true), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return {
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }

   

# %% [code] {"execution":{"iopub.execute_input":"2026-03-12T04:11:33.734967Z","iopub.status.busy":"2026-03-12T04:11:33.733808Z","iopub.status.idle":"2026-03-12T04:11:33.741496Z","shell.execute_reply":"2026-03-12T04:11:33.740700Z"},"papermill":{"duration":0.012826,"end_time":"2026-03-12T04:11:33.743036","exception":false,"start_time":"2026-03-12T04:11:33.730210","status":"completed"},"tags":[]}
# =========================
# CELDA 2: Modelo calidad
# =========================
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

        self.head_score = nn.Linear(32, 1)  # regresión
        self.head_label = nn.Linear(32, 1)  # clasificación binaria

    def forward(self, x):
        z = self.features(x)
        z = self.shared(z)
        q_score = torch.sigmoid(self.head_score(z))
        q_logit = self.head_label(z)
        return q_score, q_logit

# %% [code] {"execution":{"iopub.execute_input":"2026-03-12T04:11:33.749857Z","iopub.status.busy":"2026-03-12T04:11:33.749619Z","iopub.status.idle":"2026-03-12T12:21:00.863361Z","shell.execute_reply":"2026-03-12T12:21:00.862501Z"},"papermill":{"duration":29367.122362,"end_time":"2026-03-12T12:21:00.868138","exception":false,"start_time":"2026-03-12T04:11:33.745776","status":"completed"},"tags":[]}
# =========================
# CELDA 3: Entrenamiento
# =========================
# FIX Revisor 2 #3-4: ya no se hace split por shard, se usan las
# carpetas train/ y val/ que ya vienen separadas por participante
# (generadas por el notebook 01 parchado)
train_ds = ShardedQualityDataset(
    X_files_tr, Qscore_files_tr, Qlabel_files_tr,
    max_per_shard=4000,
    seed=1
)

val_ds = ShardedQualityDataset(
    X_files_val, Qscore_files_val, Qlabel_files_val,
    max_per_shard=4000,
    seed=2
)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

model = QualityNet().to(device)

bce = nn.BCEWithLogitsLoss()
mse = nn.MSELoss()

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

@torch.no_grad()
def evaluate(loader):
    model.eval()
    loss_sum, n = 0.0, 0
    y_score_true, y_score_pred = [], []
    y_label_true, y_label_prob = [], []

    for x, s, l in loader:
        x = x.to(device)
        s = s.to(device)
        l = l.to(device)

        ps, qlogit = model(x)
        loss = 0.7 * mse(ps, s) + 0.3 * bce(qlogit, l)

        loss_sum += loss.item() * x.size(0)
        n += x.size(0)

        y_score_true.append(s.cpu().numpy().reshape(-1))
        y_score_pred.append(ps.cpu().numpy().reshape(-1))
        y_label_true.append(l.cpu().numpy().reshape(-1))
        y_label_prob.append(torch.sigmoid(qlogit).cpu().numpy().reshape(-1))

    y_score_true = np.concatenate(y_score_true)
    y_score_pred = np.concatenate(y_score_pred)
    y_label_true = np.concatenate(y_label_true)
    y_label_prob = np.concatenate(y_label_prob)

    mets_bin = binary_metrics_from_probs(y_label_true, y_label_prob, thr=0.5)
    mets = {
        "loss": loss_sum / max(n, 1),
        "score_mse": mse_np(y_score_true, y_score_pred),
        "score_mae": mae_np(y_score_true, y_score_pred),
        **mets_bin
    }
    return mets, y_score_true, y_score_pred, y_label_true, y_label_prob

history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": [], "val_score_mae": []}

best_state = None
best_score = -1
patience = 5
bad_epochs = 0
EPOCHS = 25

for ep in range(1, EPOCHS + 1):
    model.train()
    train_loss_sum = 0.0
    n_seen = 0
    t0 = time.time()

    for x, s, l in train_loader:
        x = x.to(device)
        s = s.to(device)
        l = l.to(device)

        opt.zero_grad()
        ps, qlogit = model(x)
        loss = 0.7 * mse(ps, s) + 0.3 * bce(qlogit, l)
        loss.backward()
        opt.step()

        train_loss_sum += loss.item() * x.size(0)
        n_seen += x.size(0)

    train_loss = train_loss_sum / max(n_seen, 1)
    val_mets, st, sp, lt, lp = evaluate(val_loader)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_mets["loss"])
    history["val_f1"].append(val_mets["f1"])
    history["val_acc"].append(val_mets["acc"])
    history["val_score_mae"].append(val_mets["score_mae"])

    score = 0.5 * val_mets["f1"] + 0.5 * (1 - val_mets["score_mae"])

    print(
        f"Epoch {ep:02d} | train_loss={train_loss:.4f} | val_loss={val_mets['loss']:.4f} | "
        f"f1={val_mets['f1']:.4f} | acc={val_mets['acc']:.4f} | "
        f"score_mae={val_mets['score_mae']:.4f} | score_mse={val_mets['score_mse']:.4f} | "
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
torch.save(model.state_dict(), "/kaggle/working/quality_judge_best.pt")
print("Guardado: /kaggle/working/quality_judge_best.pt")

# %% [code] {"execution":{"iopub.execute_input":"2026-03-12T12:21:00.876474Z","iopub.status.busy":"2026-03-12T12:21:00.876080Z","iopub.status.idle":"2026-03-12T12:24:03.525813Z","shell.execute_reply":"2026-03-12T12:24:03.525031Z"},"papermill":{"duration":182.656254,"end_time":"2026-03-12T12:24:03.527848","exception":false,"start_time":"2026-03-12T12:21:00.871594","status":"completed"},"tags":[]}
# =========================
# CELDA 4: Gráficas calidad
# =========================
val_mets, st, sp, lt, lp = evaluate(val_loader)
print("Métricas finales calidad:", val_mets)

plt.figure(figsize=(8,4))
plt.plot(history["train_loss"], label="train_loss")
plt.plot(history["val_loss"], label="val_loss")
plt.title("Quality - loss")
plt.legend()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(history["val_f1"], label="val_f1")
plt.plot(history["val_acc"], label="val_acc")
plt.title("Quality - clasificación")
plt.legend()
plt.show()

plt.figure(figsize=(6,6))
plt.scatter(st[:1500], sp[:1500], s=8)
plt.xlabel("q_score real")
plt.ylabel("q_score predicho")
plt.title("Quality score: real vs predicho")
plt.show()

plt.figure(figsize=(8,4))
plt.hist(lp[lt == 1], bins=30, alpha=0.7, label="sana")
plt.hist(lp[lt == 0], bins=30, alpha=0.7, label="mala")
plt.legend()
plt.title("Distribución de probabilidad del juez de calidad")
plt.show()