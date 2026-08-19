# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T21:42:47.105035Z","iopub.status.busy":"2026-03-14T21:42:47.104752Z","iopub.status.idle":"2026-03-14T21:42:48.909806Z","shell.execute_reply":"2026-03-14T21:42:48.909095Z"},"papermill":{"duration":1.812797,"end_time":"2026-03-14T21:42:48.912111","exception":false,"start_time":"2026-03-14T21:42:47.099314","status":"completed"},"tags":[]}
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

# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T21:42:48.920320Z","iopub.status.busy":"2026-03-14T21:42:48.919966Z","iopub.status.idle":"2026-03-14T21:42:54.754391Z","shell.execute_reply":"2026-03-14T21:42:54.753569Z"},"papermill":{"duration":5.839852,"end_time":"2026-03-14T21:42:54.756124","exception":false,"start_time":"2026-03-14T21:42:48.916272","status":"completed"},"tags":[]}
# =========================
# CELDA 1: Imports
# =========================
import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T21:42:54.763829Z","iopub.status.busy":"2026-03-14T21:42:54.763194Z","iopub.status.idle":"2026-03-14T21:42:54.777957Z","shell.execute_reply":"2026-03-14T21:42:54.777445Z"},"papermill":{"duration":0.019911,"end_time":"2026-03-14T21:42:54.779218","exception":false,"start_time":"2026-03-14T21:42:54.759307","status":"completed"},"tags":[]}
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


# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T21:42:54.785615Z","iopub.status.busy":"2026-03-14T21:42:54.785350Z","iopub.status.idle":"2026-03-14T21:42:55.366729Z","shell.execute_reply":"2026-03-14T21:42:55.365867Z"},"papermill":{"duration":0.586288,"end_time":"2026-03-14T21:42:55.368203","exception":false,"start_time":"2026-03-14T21:42:54.781915","status":"completed"},"tags":[]}
# =========================
# CELDA 3: Cargar modelos (FIX: robusto a multi-version y .pt desempacado)
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

import shutil as _shutil

MODEL_BASE_GLOB = "/kaggle/input/models/jorgeoc/emg-modelos-v2/pytorch/default/*"
MODELS_DIR = "/kaggle/working/models_reconstructed"

def ensure_pt_file(name):
    os.makedirs(MODELS_DIR, exist_ok=True)
    version_dirs = sorted(glob.glob(MODEL_BASE_GLOB), reverse=True)
    if not version_dirs:
        raise FileNotFoundError(f"No se encontro ninguna version del modelo en {MODEL_BASE_GLOB}")

    for vdir in version_dirs:
        direct_path = os.path.join(vdir, f"{name}.pt")
        if os.path.isfile(direct_path):
            print(f"Encontrado {name}.pt en {vdir}")
            return direct_path

        exploded_outer = os.path.join(vdir, name)
        exploded_inner = os.path.join(exploded_outer, name)
        if os.path.isdir(exploded_inner) and os.path.isfile(os.path.join(exploded_inner, "data.pkl")):
            out_zip_noext = os.path.join(MODELS_DIR, name)
            _shutil.make_archive(out_zip_noext, "zip", root_dir=exploded_outer, base_dir=name)
            out_pt = out_zip_noext + ".pt"
            if os.path.exists(out_pt):
                os.remove(out_pt)
            os.replace(out_zip_noext + ".zip", out_pt)
            print(f"Reconstruido: {name}.pt <- {exploded_inner}")
            return out_pt

    raise FileNotFoundError(f"No se encontro {name}.pt en ninguna version bajo {MODEL_BASE_GLOB}")


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


# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T21:42:55.375694Z","iopub.status.busy":"2026-03-14T21:42:55.375441Z","iopub.status.idle":"2026-03-14T21:42:55.387114Z","shell.execute_reply":"2026-03-14T21:42:55.386409Z"},"papermill":{"duration":0.017292,"end_time":"2026-03-14T21:42:55.388583","exception":false,"start_time":"2026-03-14T21:42:55.371291","status":"completed"},"tags":[]}
# =========================
# CELDA 4: Utilidades
# =========================
THR_INTENT = 0.50
THR_QUALITY = 0.50
WIN = 400
STRIDE = 200

def robust_norm_1d(x):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-6
    z = (x - med) / (1.4826 * mad)
    return np.clip(z, -10, 10).astype(np.float32)

def windows_1d(sig, win=400, stride=200):
    idxs = np.arange(0, len(sig) - win + 1, stride)
    return np.stack([sig[i:i+win] for i in idxs], axis=0)[:, None, :].astype(np.float32)

def mae_np(a, b):
    return float(np.mean(np.abs(a - b)))

def snr_db(clean, noisy):
    noise = noisy - clean
    ps = np.mean(clean**2) + 1e-8
    pn = np.mean(noise**2) + 1e-8
    return float(10 * np.log10(ps / pn))

@torch.no_grad()
def infer_one_window(x_window, thr_intent=0.50, thr_quality=0.50):
    """
    x_window: np.array (1, 400) o (1,1,400)
    """
    if x_window.ndim == 2:
        x_window = x_window[None, :, :]
    xb = torch.tensor(x_window, dtype=torch.float32, device=device)

    t0 = time.perf_counter()

    intent_logits = intent_model(xb)
    p_intent = float(torch.sigmoid(intent_logits).cpu().numpy().reshape(-1)[0])

    q_score_pred, q_logit = quality_model(xb)
    q_score_pred = float(q_score_pred.cpu().numpy().reshape(-1)[0])
    q_prob_good = float(torch.sigmoid(q_logit).cpu().numpy().reshape(-1)[0])

    if p_intent < thr_intent:
        decision = "skip_restore"
        reason = "no_intent_detected"
        out = x_window[0].copy()
    elif q_prob_good >= thr_quality:
        decision = "skip_restore"
        reason = "signal_quality_good_enough"
        out = x_window[0].copy()
    else:
        decision = "restore"
        reason = "intent_detected_and_quality_bad"
        out = restorer_model(xb).cpu().numpy()[0]

    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0

    return {
        "p_intent": p_intent,
        "q_score_pred": q_score_pred,
        "q_prob_good": q_prob_good,
        "decision": decision,
        "reason": reason,
        "latency_ms": latency_ms,
        "output_window": out.astype(np.float32),
    }

def infer_batch(x_batch, thr_intent=0.50, thr_quality=0.50):
    rows = []
    x_out = []
    latencies = []

    for i in range(len(x_batch)):
        res = infer_one_window(x_batch[i], thr_intent=thr_intent, thr_quality=thr_quality)
        rows.append({
            "idx": i,
            "p_intent": res["p_intent"],
            "q_score_pred": res["q_score_pred"],
            "q_prob_good": res["q_prob_good"],
            "decision": res["decision"],
            "reason": res["reason"],
            "latency_ms": res["latency_ms"],
        })
        x_out.append(res["output_window"])
        latencies.append(res["latency_ms"])

    return pd.DataFrame(rows), np.stack(x_out).astype(np.float32), np.array(latencies, dtype=np.float32)


# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T21:42:55.395490Z","iopub.status.busy":"2026-03-14T21:42:55.395224Z","iopub.status.idle":"2026-03-14T21:42:55.398810Z","shell.execute_reply":"2026-03-14T21:42:55.398147Z"},"papermill":{"duration":0.008618,"end_time":"2026-03-14T21:42:55.400094","exception":false,"start_time":"2026-03-14T21:42:55.391476","status":"completed"},"tags":[]}
# =========================
# CELDA 5: Elegir fuente
# =========================
SOURCE_MODE = "dataset"
# opciones:
# "dataset"
# "synthetic"
# "external_file"

external_path = None
# ejemplo:
# external_path = "/kaggle/input/mi-senal/senal.npy"
# external_path = "/kaggle/input/mi-senal/senal.csv"
# external_path = "/kaggle/input/mi-senal/senal.txt"


# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T21:42:55.407344Z","iopub.status.busy":"2026-03-14T21:42:55.406618Z","iopub.status.idle":"2026-03-14T21:42:56.147207Z","shell.execute_reply":"2026-03-14T21:42:56.146455Z"},"papermill":{"duration":0.745917,"end_time":"2026-03-14T21:42:56.148894","exception":false,"start_time":"2026-03-14T21:42:55.402977","status":"completed"},"tags":[]}
# =========================
# CELDA 6: Dataset real
# =========================
x_in = None
x_clean_ref = None

if SOURCE_MODE == "dataset":
    candidate_dirs = []
    for p in glob.glob("/kaggle/input/**/restoration_dataset", recursive=True):
        if os.path.isdir(p):
            xcorr = glob.glob(os.path.join(p, "Xcorr_*.npy"))
            xclean = glob.glob(os.path.join(p, "Xclean_*.npy"))
            if len(xcorr) > 0 and len(xclean) > 0:
                candidate_dirs.append(p)

    if not candidate_dirs:
        raise FileNotFoundError("No encontré restoration_dataset.")

    RDIR = candidate_dirs[0]
    Xcorr_files = sorted(glob.glob(os.path.join(RDIR, "Xcorr_*.npy")))
    Xclean_files = sorted(glob.glob(os.path.join(RDIR, "Xclean_*.npy")))

    x_in = np.load(Xcorr_files[0])[:500]
    x_clean_ref = np.load(Xclean_files[0])[:500]

    print("Usando dataset real")
    print("x_in:", x_in.shape)
    print("x_clean_ref:", x_clean_ref.shape)


# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T21:42:56.156699Z","iopub.status.busy":"2026-03-14T21:42:56.156148Z","iopub.status.idle":"2026-03-14T21:42:56.163573Z","shell.execute_reply":"2026-03-14T21:42:56.162885Z"},"papermill":{"duration":0.012643,"end_time":"2026-03-14T21:42:56.164937","exception":false,"start_time":"2026-03-14T21:42:56.152294","status":"completed"},"tags":[]}
# =========================
# CELDA 7: Señales sintéticas
# =========================
def make_synthetic_sequence(num_windows=40, T=400, seed=123):
    rng = np.random.default_rng(seed)
    x_clean_list = []
    x_corr_list = []

    for i in range(num_windows):
        t = np.linspace(0, 1, T)

        if i % 4 == 0:
            clean = np.zeros(T, dtype=np.float32)
            clean += 0.02 * rng.normal(size=T).astype(np.float32)
        else:
            env = np.exp(-0.5 * ((t - 0.5) / 0.12)**2)
            carrier = np.sin(2*np.pi*35*t) + 0.5*np.sin(2*np.pi*70*t)
            clean = (1.8 * env * carrier).astype(np.float32)

        noisy = clean.copy()
        noisy += rng.normal(0, 0.25, size=T).astype(np.float32)

        if i % 3 == 0:
            noisy += (0.5 * np.sin(2*np.pi*1.2*t)).astype(np.float32)

        if i % 5 == 0:
            start = 120
            noisy[start:start+50] = 0.0

        x_clean_list.append(clean[None, :])
        x_corr_list.append(noisy[None, :])

    x_clean = np.stack(x_clean_list).astype(np.float32)
    x_corr = np.stack(x_corr_list).astype(np.float32)
    return x_clean, x_corr

if SOURCE_MODE == "synthetic":
    x_clean_ref, x_in = make_synthetic_sequence(num_windows=40)
    print("Usando señal sintética")
    print("x_in:", x_in.shape)
    print("x_clean_ref:", x_clean_ref.shape)


# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T21:42:56.171880Z","iopub.status.busy":"2026-03-14T21:42:56.171658Z","iopub.status.idle":"2026-03-14T21:42:56.176681Z","shell.execute_reply":"2026-03-14T21:42:56.176010Z"},"papermill":{"duration":0.010046,"end_time":"2026-03-14T21:42:56.178022","exception":false,"start_time":"2026-03-14T21:42:56.167976","status":"completed"},"tags":[]}
# =========================
# CELDA 8: Archivo externo
# =========================
def load_external_signal(path):
    if path is None:
        raise ValueError("Asigna external_path.")
    if path.endswith(".npy"):
        sig = np.load(path).astype(np.float32).reshape(-1)
    elif path.endswith(".csv"):
        sig = np.loadtxt(path, delimiter=",", dtype=np.float32).reshape(-1)
    elif path.endswith(".txt"):
        sig = np.loadtxt(path, dtype=np.float32).reshape(-1)
    else:
        raise ValueError("Formato no soportado. Usa .npy, .csv o .txt")
    return robust_norm_1d(sig)

if SOURCE_MODE == "external_file":
    sig_ext = load_external_signal(external_path)
    x_in = windows_1d(sig_ext, win=WIN, stride=STRIDE)
    x_clean_ref = None

    print("Usando archivo externo")
    print("x_in:", x_in.shape)


# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T21:42:56.185478Z","iopub.status.busy":"2026-03-14T21:42:56.184913Z","iopub.status.idle":"2026-03-14T21:42:58.570905Z","shell.execute_reply":"2026-03-14T21:42:58.570251Z"},"papermill":{"duration":2.391692,"end_time":"2026-03-14T21:42:58.572693","exception":false,"start_time":"2026-03-14T21:42:56.181001","status":"completed"},"tags":[]}
# =========================
# CELDA 9: Inferencia
# =========================
if x_in is None:
    raise ValueError("No se cargó ninguna fuente de entrada. Revisa SOURCE_MODE.")

df_inf, x_out, lat_ms = infer_batch(
    x_in,
    thr_intent=THR_INTENT,
    thr_quality=THR_QUALITY
)

print(df_inf.head())
print()

# FIX Revisor 2 #12: reportar distribucion completa, no solo mediana
lat = np.asarray(lat_ms, dtype=np.float64)

latency_stats = {
    "n": int(len(lat)),
    "mean_ms": float(np.mean(lat)),
    "std_ms": float(np.std(lat)),
    "median_ms": float(np.median(lat)),
    "p50_ms": float(np.percentile(lat, 50)),
    "p90_ms": float(np.percentile(lat, 90)),
    "p95_ms": float(np.percentile(lat, 95)),
    "p99_ms": float(np.percentile(lat, 99)),
    "min_ms": float(np.min(lat)),
    "max_ms": float(np.max(lat)),
}

print("=== Distribucion completa de latencia (pipeline completo) ===")
for k, v in latency_stats.items():
    print(f"{k:>10}: {v:.3f}" if k != "n" else f"{k:>10}: {v}")

for decision_type in df_inf["decision"].unique():
    sub = df_inf.loc[df_inf["decision"] == decision_type, "latency_ms"].to_numpy(dtype=np.float64)
    print(f"\n-- rama \'{decision_type}\' (n={len(sub)}) --")
    print(f"  mean={sub.mean():.3f} ms  median={np.median(sub):.3f} ms  "
          f"p95={np.percentile(sub,95):.3f} ms  p99={np.percentile(sub,99):.3f} ms")

import json
with open("latency_report.json", "w") as f:
    json.dump(latency_stats, f, indent=2)
print("\nGuardado latency_report.json")


# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T21:42:58.580365Z","iopub.status.busy":"2026-03-14T21:42:58.580138Z","iopub.status.idle":"2026-03-14T21:42:58.615042Z","shell.execute_reply":"2026-03-14T21:42:58.614090Z"},"papermill":{"duration":0.040316,"end_time":"2026-03-14T21:42:58.616501","exception":false,"start_time":"2026-03-14T21:42:58.576185","status":"completed"},"tags":[]}
# =========================
# CELDA 10: Resumen
# =========================
print("Conteo de decisiones:")
print(df_inf["decision"].value_counts(dropna=False))
print()

print("Razones:")
print(df_inf["reason"].value_counts(dropna=False))
print()

print("Latencia por decisión:")
print(df_inf.groupby("decision")["latency_ms"].describe())


# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T21:42:58.624466Z","iopub.status.busy":"2026-03-14T21:42:58.623997Z","iopub.status.idle":"2026-03-14T21:42:58.750863Z","shell.execute_reply":"2026-03-14T21:42:58.749886Z"},"papermill":{"duration":0.13248,"end_time":"2026-03-14T21:42:58.752270","exception":false,"start_time":"2026-03-14T21:42:58.619790","status":"completed"},"tags":[]}
# =========================
# CELDA 11: Métricas con referencia
# =========================
if x_clean_ref is not None:
    records = []
    for i in range(len(x_in)):
        clean = x_clean_ref[i, 0]
        corr = x_in[i, 0]
        out = x_out[i, 0]

        records.append({
            "idx": i,
            "decision": df_inf.iloc[i]["decision"],
            "reason": df_inf.iloc[i]["reason"],
            "mae_before": mae_np(clean, corr),
            "mae_after": mae_np(clean, out),
            "snr_before": snr_db(clean, corr),
            "snr_after": snr_db(clean, out),
            "improved_mae": mae_np(clean, out) < mae_np(clean, corr),
            "improved_snr": snr_db(clean, out) > snr_db(clean, corr),
        })

    df_metrics = pd.DataFrame(records)

    print("Promedios globales:")
    print(df_metrics[["mae_before", "mae_after", "snr_before", "snr_after"]].mean())
    print()

    dfr = df_metrics[df_metrics["decision"] == "restore"]
    if len(dfr):
        print("Solo restauradas:")
        print(dfr[["mae_before", "mae_after", "snr_before", "snr_after"]].mean())
        print("Frac mejora MAE:", dfr["improved_mae"].mean())
        print("Frac mejora SNR:", dfr["improved_snr"].mean())
else:
    print("No hay señal limpia de referencia; se omiten métricas antes/después.")


# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T21:42:58.760186Z","iopub.status.busy":"2026-03-14T21:42:58.759753Z","iopub.status.idle":"2026-03-14T21:42:59.476353Z","shell.execute_reply":"2026-03-14T21:42:59.475669Z"},"papermill":{"duration":0.722201,"end_time":"2026-03-14T21:42:59.477854","exception":false,"start_time":"2026-03-14T21:42:58.755653","status":"completed"},"tags":[]}
# =========================
# CELDA 12: Gráficas
# =========================
plt.figure(figsize=(8,4))
df_inf["reason"].value_counts().plot(kind="bar")
plt.title("Razones del pipeline")
plt.ylabel("count")
plt.show()

plt.figure(figsize=(8,4))
plt.hist(df_inf["latency_ms"], bins=40)
plt.title("Distribución de latencia de inferencia (ms)")
plt.show()

plt.figure(figsize=(8,4))
plt.hist(df_inf["p_intent"], bins=40)
plt.title("Distribución p_intent")
plt.show()

plt.figure(figsize=(8,4))
plt.hist(df_inf["q_prob_good"], bins=40)
plt.title("Distribución q_prob_good")
plt.show()


# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T21:42:59.488529Z","iopub.status.busy":"2026-03-14T21:42:59.487812Z","iopub.status.idle":"2026-03-14T21:43:00.216054Z","shell.execute_reply":"2026-03-14T21:43:00.215456Z"},"papermill":{"duration":0.735127,"end_time":"2026-03-14T21:43:00.217905","exception":false,"start_time":"2026-03-14T21:42:59.482778","status":"completed"},"tags":[]}
# =========================
# CELDA 13: Ejemplos
# =========================
num_examples = min(6, len(x_in))

for i in range(num_examples):
    row = df_inf.iloc[i]
    print(f"idx={i}")
    print(f"decision={row['decision']} | reason={row['reason']}")
    print(f"p_intent={row['p_intent']:.3f} | q_score_pred={row['q_score_pred']:.3f} | q_prob_good={row['q_prob_good']:.3f}")
    print(f"latency_ms={row['latency_ms']:.3f}")
    print()

    plt.figure(figsize=(12,3))
    plt.plot(x_in[i,0], label="entrada")
    plt.plot(x_out[i,0], label="salida_pipeline")
    if x_clean_ref is not None:
        plt.plot(x_clean_ref[i,0], label="referencia_limpia", alpha=0.8)
    plt.legend()
    plt.show()


# %% [code] {"execution":{"iopub.execute_input":"2026-03-14T21:43:00.235483Z","iopub.status.busy":"2026-03-14T21:43:00.235200Z","iopub.status.idle":"2026-03-14T21:43:00.240934Z","shell.execute_reply":"2026-03-14T21:43:00.240341Z"},"papermill":{"duration":0.015766,"end_time":"2026-03-14T21:43:00.242455","exception":false,"start_time":"2026-03-14T21:43:00.226689","status":"completed"},"tags":[]}
# =========================
# CELDA 14: Envío serial opcional
# =========================
USE_SERIAL = False
SERIAL_PORT = "COM3"   # cambia esto en Windows
SERIAL_BAUD = 115200

if USE_SERIAL:
    import serial
    import time as pytime

    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
    pytime.sleep(2)

    def map_decision_to_msg(row):
        if row["decision"] == "restore":
            return "RESTORE"
        elif row["reason"] == "signal_quality_good_enough":
            return "OK"
        else:
            return "REJECT"

    for i in range(len(df_inf)):
        msg = map_decision_to_msg(df_inf.iloc[i])
        ser.write((msg + "\n").encode("utf-8"))
        print("Enviado:", msg)
        pytime.sleep(0.2)

    ser.close()
else:
    print("USE_SERIAL=False, no se envía nada al Arduino.")
