# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T21:54:59.301344Z","iopub.status.busy":"2026-03-14T21:54:59.301099Z","iopub.status.idle":"2026-03-14T21:55:01.317240Z","shell.execute_reply":"2026-03-14T21:55:01.316459Z"},"papermill":{"duration":2.022255,"end_time":"2026-03-14T21:55:01.319030","exception":false,"start_time":"2026-03-14T21:54:59.296775","status":"completed"},"tags":[]}
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

# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T21:55:01.325075Z","iopub.status.busy":"2026-03-14T21:55:01.324703Z","iopub.status.idle":"2026-03-14T21:55:04.833663Z","shell.execute_reply":"2026-03-14T21:55:04.832816Z"},"papermill":{"duration":3.513689,"end_time":"2026-03-14T21:55:04.835242","exception":false,"start_time":"2026-03-14T21:55:01.321553","status":"completed"},"tags":[]}
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

ROOT = "/kaggle/input/datasets/jorgeoc/prepared-myoware-v2/prepared_myoware_v2"
IDIR = os.path.join(ROOT, "intent_dataset")

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

X_files = sorted(glob.glob(os.path.join(IDIR, "X_*.npy")))
Y_files = sorted(glob.glob(os.path.join(IDIR, "Y_*.npy")))

print("IDIR:", IDIR)
print("shards X:", len(X_files))
print("shards Y:", len(Y_files))
print("Ejemplo X:", X_files[:3])
print("Ejemplo Y:", Y_files[:3])

if len(X_files) == 0 or len(Y_files) == 0:
    raise FileNotFoundError(f"No se encontraron shards en {IDIR}")

# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T21:55:04.841084Z","iopub.status.busy":"2026-03-14T21:55:04.840712Z","iopub.status.idle":"2026-03-14T21:55:04.866571Z","shell.execute_reply":"2026-03-14T21:55:04.865842Z"},"papermill":{"duration":0.030249,"end_time":"2026-03-14T21:55:04.867867","exception":false,"start_time":"2026-03-14T21:55:04.837618","status":"completed"},"tags":[]}
x0 = np.load(X_files[0], mmap_mode="r")
y0 = np.load(Y_files[0], mmap_mode="r")

print(x0.shape, x0.dtype)
print(y0.shape, y0.dtype)
print("pos frac shard0:", y0.mean())

# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T21:55:04.874139Z","iopub.status.busy":"2026-03-14T21:55:04.873583Z","iopub.status.idle":"2026-03-14T21:55:04.891180Z","shell.execute_reply":"2026-03-14T21:55:04.890502Z"},"papermill":{"duration":0.02219,"end_time":"2026-03-14T21:55:04.892516","exception":false,"start_time":"2026-03-14T21:55:04.870326","status":"completed"},"tags":[]}
# =========================
# CELDA 1: Dataset + utils
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
IDIR_TRAIN = os.path.join(ROOT, "train", "intent_dataset")
IDIR_VAL = os.path.join(ROOT, "val", "intent_dataset")

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

X_files_tr = sorted(glob.glob(os.path.join(IDIR_TRAIN, "X_*.npy")))
Y_files_tr = sorted(glob.glob(os.path.join(IDIR_TRAIN, "Y_*.npy")))

X_files_val = sorted(glob.glob(os.path.join(IDIR_VAL, "X_*.npy")))
Y_files_val = sorted(glob.glob(os.path.join(IDIR_VAL, "Y_*.npy")))

print("train shards:", len(X_files_tr), " val shards:", len(X_files_val))

if len(X_files_tr) == 0:
    raise FileNotFoundError(f"No se encontraron shards en {IDIR_TRAIN}")

X_files, Y_files = X_files_tr, Y_files_tr  # para pruebas rapidas mas abajo

class ShardedIntentDataset(Dataset):
    def __init__(self, X_files, Y_files, max_per_shard=None, seed=123):
        self.samples = []
        rng = np.random.default_rng(seed)

        for xf, yf in zip(X_files, Y_files):
            X = np.load(xf, mmap_mode="r")
            Y = np.load(yf, mmap_mode="r")
            n = len(X)

            idx = np.arange(n)
            if max_per_shard is not None and n > max_per_shard:
                idx = rng.choice(idx, size=max_per_shard, replace=False)

            for i in idx:
                self.samples.append((xf, yf, int(i)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        xf, yf, i = self.samples[idx]
        X = np.load(xf, mmap_mode="r")
        Y = np.load(yf, mmap_mode="r")
        x = torch.tensor(X[i], dtype=torch.float32)   # (1, T)
        y = torch.tensor([Y[i]], dtype=torch.float32) # (1,)
        return x, y

def split_files(files, frac_val=0.15, seed=123):
    idx = np.arange(len(files))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_val = max(1, int(len(files) * frac_val))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    return tr_idx, val_idx

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

def roc_auc_fast(y_true, y_score):
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float64)

    order = np.argsort(y_score)
    y_true = y_true[order]

    n1 = y_true.sum()
    n0 = len(y_true) - n1
    if n0 == 0 or n1 == 0:
        return float("nan")

    ranks = np.arange(1, len(y_true) + 1)
    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n1 * (n1 + 1) / 2) / (n0 * n1)
    return float(auc)

# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T21:55:04.898190Z","iopub.status.busy":"2026-03-14T21:55:04.897796Z","iopub.status.idle":"2026-03-14T21:55:04.905680Z","shell.execute_reply":"2026-03-14T21:55:04.905020Z"},"papermill":{"duration":0.01214,"end_time":"2026-03-14T21:55:04.906942","exception":false,"start_time":"2026-03-14T21:55:04.894802","status":"completed"},"tags":[]}
# =========================
# CELDA 2: Modelo TCN
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

# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T21:55:04.912599Z","iopub.status.busy":"2026-03-14T21:55:04.912360Z","iopub.status.idle":"2026-03-15T01:30:08.063291Z","shell.execute_reply":"2026-03-15T01:30:08.062453Z"},"papermill":{"duration":12903.158303,"end_time":"2026-03-15T01:30:08.067514","exception":false,"start_time":"2026-03-14T21:55:04.909211","status":"completed"},"tags":[]}
# =========================
# CELDA 3: Entrenamiento
# =========================
# FIX Revisor 2 #3-4: carpetas train/val ya separadas por participante
X_train, Y_train = X_files_tr, Y_files_tr
X_val, Y_val = X_files_val, Y_files_val

train_ds = ShardedIntentDataset(X_train, Y_train, max_per_shard=4000, seed=1)
val_ds   = ShardedIntentDataset(X_val, Y_val, max_per_shard=4000, seed=2)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

model = IntentTCN().to(device)


# manejar desbalance
y_pos_frac = []
for yf in Y_train:
    y_pos_frac.append(np.load(yf, mmap_mode="r").mean())

approx_pos = float(np.mean(y_pos_frac))
pos_weight_value = (1.0 - approx_pos) / max(approx_pos, 1e-6)
pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

print("approx_pos:", approx_pos, "pos_weight:", pos_weight.item())

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

@torch.no_grad()
def evaluate(loader):
    model.eval()
    losses, ys, ps = [], [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)
        p = torch.sigmoid(logits)

        losses.append(loss.item() * x.size(0))
        ys.append(y.cpu().numpy().reshape(-1))
        ps.append(p.cpu().numpy().reshape(-1))

    ys = np.concatenate(ys)
    ps = np.concatenate(ps)
    loss = np.sum(losses) / max(len(ys), 1)
    mets = binary_metrics_from_probs(ys, ps, thr=0.5)
    mets["auc"] = roc_auc_fast(ys, ps)
    mets["loss"] = loss
    return mets, ys, ps

history = {
    "train_loss": [], "val_loss": [],
    "val_acc": [], "val_f1": [], "val_auc": []
}

best_state = None
best_score = -1
patience = 5
bad_epochs = 0
EPOCHS = 25

for ep in range(1, EPOCHS + 1):
    model.train()
    t0 = time.time()
    train_loss_sum = 0.0
    n_seen = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * x.size(0)
        n_seen += x.size(0)

    train_loss = train_loss_sum / max(n_seen, 1)
    val_mets, yv, pv = evaluate(val_loader)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_mets["loss"])
    history["val_acc"].append(val_mets["acc"])
    history["val_f1"].append(val_mets["f1"])
    history["val_auc"].append(val_mets["auc"])

    score = 0.6 * val_mets["f1"] + 0.4 * val_mets["auc"]

    print(
        f"Epoch {ep:02d} | "
        f"train_loss={train_loss:.4f} | val_loss={val_mets['loss']:.4f} | "
        f"acc={val_mets['acc']:.4f} | f1={val_mets['f1']:.4f} | auc={val_mets['auc']:.4f} | "
        f"prec={val_mets['precision']:.4f} | rec={val_mets['recall']:.4f} | "
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
torch.save(model.state_dict(), "/kaggle/working/intent_binary_best.pt")
print("Guardado: /kaggle/working/intent_binary_best.pt")

# %% [code] {"execution":{"iopub.execute_input":"2026-03-15T01:30:08.075565Z","iopub.status.busy":"2026-03-15T01:30:08.074782Z","iopub.status.idle":"2026-03-15T01:34:07.513901Z","shell.execute_reply":"2026-03-15T01:34:07.513119Z"},"papermill":{"duration":239.445141,"end_time":"2026-03-15T01:34:07.515587","exception":false,"start_time":"2026-03-15T01:30:08.070446","status":"completed"},"tags":[]}
# =========================
# CELDA 4: Gráficas
# =========================
val_mets, yv, pv = evaluate(val_loader)
print("Métricas finales val:", val_mets)

plt.figure(figsize=(8,4))
plt.plot(history["train_loss"], label="train_loss")
plt.plot(history["val_loss"], label="val_loss")
plt.title("Intent - loss")
plt.legend()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(history["val_f1"], label="val_f1")
plt.plot(history["val_auc"], label="val_auc")
plt.plot(history["val_acc"], label="val_acc")
plt.title("Intent - métricas")
plt.legend()
plt.show()

cm = val_mets["tp"], val_mets["tn"], val_mets["fp"], val_mets["fn"]
print("TP, TN, FP, FN =", cm)

yhat = (pv >= 0.5).astype(int)
idx_pos = np.where(yhat == 1)[0][:3]
idx_neg = np.where(yhat == 0)[0][:3]

# visualizar ejemplos
all_x = []
for x, y in val_loader:
    all_x.append(x.numpy())
all_x = np.concatenate(all_x, axis=0)

for i in list(idx_pos) + list(idx_neg):
    plt.figure(figsize=(10,3))
    plt.plot(all_x[i,0])
    plt.title(f"Ejemplo val idx={i} | y_true={yv[i]} | p={pv[i]:.3f}")
    plt.show()