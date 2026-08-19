"""
generar_figuras.py
Genera tres figuras para el artículo:
  - figura_A_tabla_sujetos.png  → tabla de sujetos sanos y amputados
  - figura_B_decisiones.png     → distribución de decisiones por grupo
  - figura_C_metricas.png       → métricas comparativas por sujeto

Uso:
  python generar_figuras.py --carpeta ./sMEG --models_dir ./models
"""

import os
import csv
import time
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr

import torch
import torch.nn as nn

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
WIN         = 400
STRIDE      = 200
THR_INTENT  = 0.50
THR_QUALITY = 0.50
MAX_WINDOWS = 300

# Sujetos sanos: 18 archivos (sin P28)
SANOS_IDS = [
    "emgP14Rcsv", "emgP15Rcsv",
    "emgP16Rcsv", "emgP17Rcsv",
    "emgP18Rcsv", "emgP19Rcsv",
    "emgP20Rcsv", "emgP21Rcsv",
    "emgP22Rcsv", "emgP23Rcsv",
    "emgP24Rcsv", "emgP25Rcsv",
    "emgP26Rcsv", "emgP27Rcsv",
    "emgP29Rcsv", "emgP30Rcsv",
    "emgP31Rcsv", "emgP32Rcsv",
]

# Metadatos sanos del Excel
# Metadatos sanos del Excel
# FIX (auditoría de datos, agosto 2026): las filas P29-P32 estaban mal
# emparejadas (le faltaba Flor por completo, Tania/Nancy quedaban
# corridas). Fuente de verdad: 1-registro_de_captura_de_sen_ales.xlsx
SANOS_META = {
    "emgP14Rcsv": {"sujeto": "S1", "nombre": "Oscar",       "edad": 54, "sexo": "M", "miembro": "Derecho"},
    "emgP15Rcsv": {"sujeto": "S1", "nombre": "Oscar",       "edad": 54, "sexo": "M", "miembro": "Izquierdo"},
    "emgP16Rcsv": {"sujeto": "S2", "nombre": "A. Cisniega", "edad": 28, "sexo": "M", "miembro": "Derecho"},
    "emgP17Rcsv": {"sujeto": "S2", "nombre": "A. Cisniega", "edad": 28, "sexo": "M", "miembro": "Izquierdo"},
    "emgP18Rcsv": {"sujeto": "S3", "nombre": "Jesús",       "edad": 29, "sexo": "M", "miembro": "Derecho"},
    "emgP19Rcsv": {"sujeto": "S3", "nombre": "Jesús",       "edad": 29, "sexo": "M", "miembro": "Izquierdo"},
    "emgP20Rcsv": {"sujeto": "S4", "nombre": "Arturo",      "edad": 27, "sexo": "M", "miembro": "Derecho"},
    "emgP21Rcsv": {"sujeto": "S4", "nombre": "Arturo",      "edad": 27, "sexo": "M", "miembro": "Izquierdo"},
    "emgP22Rcsv": {"sujeto": "S5", "nombre": "A. Valdivieso","edad": 23, "sexo": "F", "miembro": "Derecho"},
    "emgP23Rcsv": {"sujeto": "S5", "nombre": "A. Valdivieso","edad": 23, "sexo": "F", "miembro": "Izquierdo"},
    "emgP24Rcsv": {"sujeto": "S6", "nombre": "Diana",       "edad": 32, "sexo": "F", "miembro": "Derecho"},
    "emgP25Rcsv": {"sujeto": "S6", "nombre": "Diana",       "edad": 32, "sexo": "F", "miembro": "Izquierdo"},
    "emgP26Rcsv": {"sujeto": "S7", "nombre": "Dulce",       "edad": 26, "sexo": "F", "miembro": "Derecho"},
    "emgP27Rcsv": {"sujeto": "S7", "nombre": "Dulce",       "edad": 26, "sexo": "F", "miembro": "Izquierdo"},
    "emgP29Rcsv": {"sujeto": "S8", "nombre": "Flor",        "edad": 50, "sexo": "F", "miembro": "Izquierdo"},  # falta el derecho (P28, perdido)
    "emgP30Rcsv": {"sujeto": "S9", "nombre": "Tania",       "edad": 24, "sexo": "F", "miembro": "Derecho"},
    "emgP31Rcsv": {"sujeto": "S9", "nombre": "Tania",       "edad": 24, "sexo": "F", "miembro": "Izquierdo"},
    "emgP32Rcsv": {"sujeto": "S10","nombre": "Nancy",       "edad": 20, "sexo": "F", "miembro": "Derecho"},  # falta el izquierdo (P33 sin sufijo, perdido)
}

# Amputados
# Amputados
# FIX (auditoría de datos, agosto 2026): "mujer" y "hombre" estaban
# completamente cruzados (sexo, años de amputación y archivos no
# correspondían entre sí). Confirmado con Jorge:
#   P1 = mujer, 31 años, 5 años post-amputación, 3 sesiones (P33/P34/P35)
#   P2 = hombre, 28 años, 8 años post-amputación, 4 sesiones (Q1-Q4)
AMPUTADOS_META = {
    "mujer": {
        "label": "P1 (Mujer)",
        "edad": 31, "sexo": "F",
        "anos_amputacion": 5,
        "sesiones": ["emgP33RcsvAMPUTADA1", "emgP34RcsvAMPUTADA2", "emgP35RcsvAMPUTADA3"],
        "semanas": ["S1", "S2", "S3"],
    },
    "hombre": {
        "label": "P2 (Hombre)",
        "edad": 28, "sexo": "M",
        "anos_amputacion": 8,
        "sesiones": ["emgQ1csv", "emgQ2csv", "emgQ3csv", "emgQ4csv"],
        "semanas": ["S1", "S2", "S3", "S4"],
    },
}

# Colores artículo (fondo blanco)
C_OK      = "#2ca02c"
C_RESTORE = "#ff7f0e"
C_REJECT  = "#d62728"
C_SANO    = "#1f77b4"
C_AMP     = "#9467bd"


# ──────────────────────────────────────────────
# MODELOS
# ──────────────────────────────────────────────
class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=5, dilation=1, drop=0.1):
        super().__init__()
        pad = (k - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=dilation)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.act   = nn.ReLU()
        self.drop  = nn.Dropout(drop)
        self.skip  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        s = self.skip(x)
        x = self.drop(self.act(self.bn1(self.conv1(x))))
        x = self.drop(self.act(self.bn2(self.conv2(x))))
        return self.act(x + s)


class IntentTCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU())
        self.tcn = nn.Sequential(
            TCNBlock(16, 16, dilation=1, drop=0.10),
            TCNBlock(16, 32, dilation=2, drop=0.10),
            TCNBlock(32, 32, dilation=4, drop=0.10))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(32, 32), nn.ReLU(),
            nn.Dropout(0.10), nn.Linear(32, 1))

    def forward(self, x):
        return self.head(self.pool(self.tcn(self.stem(x))))


class QualityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.shared = nn.Sequential(
            nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1))
        self.head_score = nn.Linear(32, 1)
        self.head_label = nn.Linear(32, 1)

    def forward(self, x):
        z = self.shared(self.features(x))
        return torch.sigmoid(self.head_score(z)), self.head_label(z)


class ConvAE1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 5, stride=2, padding=2), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, 5, stride=2, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.BatchNorm1d(64), nn.ReLU())
        self.bottleneck = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.ConvTranspose1d(32, 16, 4, stride=2, padding=1), nn.BatchNorm1d(16), nn.ReLU(),
            nn.ConvTranspose1d(16, 1,  4, stride=2, padding=1))

    def forward(self, x):
        return self.decoder(self.bottleneck(self.encoder(x)))


def _find_pt_file(models_dir, name):
    """
    FIX: kagglehub.model_download() a veces entrega los .pt "desempacados"
    en carpetas (un .pt es internamente un .zip). Esta funcion detecta
    ambos casos y reconstruye el .pt si hace falta, buscando tambien en
    subcarpetas (kagglehub puede anidar por version/framework).
    """
    import glob as _glob
    import shutil as _shutil

    # Caso 1: archivo directo en cualquier nivel
    direct_hits = _glob.glob(os.path.join(models_dir, "**", f"{name}.pt"), recursive=True)
    if direct_hits:
        return direct_hits[0]

    # Caso 2: carpeta "desempacada" .../<name>/<name>/{data.pkl,...}
    exploded_hits = _glob.glob(os.path.join(models_dir, "**", name, name), recursive=True)
    for exploded_inner in exploded_hits:
        if os.path.isfile(os.path.join(exploded_inner, "data.pkl")):
            exploded_outer = os.path.dirname(exploded_inner)
            scratch_dir = os.path.join(os.getcwd(), "_models_reconstructed")
            os.makedirs(scratch_dir, exist_ok=True)
            out_zip_noext = os.path.join(scratch_dir, name)
            _shutil.make_archive(out_zip_noext, "zip", root_dir=exploded_outer, base_dir=name)
            out_pt = out_zip_noext + ".pt"
            if os.path.exists(out_pt):
                os.remove(out_pt)
            os.replace(out_zip_noext + ".zip", out_pt)
            print(f"  Reconstruido: {name}.pt <- {exploded_inner}")
            return out_pt

    raise FileNotFoundError(
        f"No se encontro {name}.pt ni como archivo ni como carpeta "
        f"desempacada bajo {models_dir}. Contenido encontrado: "
        f"{os.listdir(models_dir) if os.path.isdir(models_dir) else 'CARPETA NO EXISTE'}"
    )


def load_models(models_dir, device):
    intent_path = _find_pt_file(models_dir, "intent_binary_best")
    quality_path = _find_pt_file(models_dir, "quality_judge_best")
    restorer_path = _find_pt_file(models_dir, "restorer_best")

    intent = IntentTCN().to(device)
    intent.load_state_dict(torch.load(intent_path, map_location=device))
    intent.eval()

    quality = QualityNet().to(device)
    quality.load_state_dict(torch.load(quality_path, map_location=device))
    quality.eval()

    restorer = ConvAE1D().to(device)
    restorer.load_state_dict(torch.load(restorer_path, map_location=device))
    restorer.eval()
    return intent, quality, restorer


# ──────────────────────────────────────────────
# UTILIDADES
# ──────────────────────────────────────────────
def robust_norm(x):
    x   = np.asarray(x, dtype=np.float32).reshape(-1)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-6
    return np.clip((x - med) / (1.4826 * mad), -10, 10).astype(np.float32)


def load_csv_signal(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            vals = []
            for v in row:
                try:
                    vals.append(float(str(v).strip()))
                except ValueError:
                    pass
            if len(vals) >= 2:
                rows.append(vals[1])
            elif len(vals) == 1:
                rows.append(vals[0])
    if not rows:
        raise ValueError(f"Sin datos: {path}")
    return np.array(rows, dtype=np.float32)


@torch.no_grad()
def infer_window(norm_w, intent_m, quality_m, restorer_m, device):
    xb = torch.tensor(norm_w[None, None, :], dtype=torch.float32, device=device)
    t0 = time.perf_counter()
    p_intent = float(torch.sigmoid(intent_m(xb)).cpu().item())
    q_score_t, q_logit = quality_m(xb)
    q_score = float(q_score_t.cpu().item())
    q_good  = float(torch.sigmoid(q_logit).cpu().item())

    if p_intent < THR_INTENT:
        decision = "REJECT"
        recon    = norm_w.copy()
    elif q_good >= THR_QUALITY:
        decision = "OK"
        recon    = norm_w.copy()
    else:
        decision = "RESTORE"
        recon    = restorer_m(xb).cpu().numpy()[0, 0].astype(np.float32)

    latency = (time.perf_counter() - t0) * 1000.0
    corr = 0.0
    mse  = 0.0
    if decision == "RESTORE":
        mse = float(np.mean((norm_w - recon) ** 2))
        if np.std(norm_w) > 1e-9 and np.std(recon) > 1e-9:
            corr, _ = pearsonr(norm_w, recon)

    return {
        "decision": decision,
        "p_intent": p_intent,
        "q_score":  q_score,
        "q_good":   q_good,
        "latency":  latency,
        "mse":      mse,
        "corr":     corr,
    }


def process_file(path, intent_m, quality_m, restorer_m, device):
    signal = load_csv_signal(path)
    n = len(signal)
    n_wins = min(1 + max(0, (n - WIN) // STRIDE), MAX_WINDOWS)

    results = []
    for i in range(n_wins):
        s, e = i * STRIDE, i * STRIDE + WIN
        raw_w = signal[s:e]
        if len(raw_w) < WIN:
            break
        norm_w = robust_norm(raw_w)
        results.append(infer_window(norm_w, intent_m, quality_m, restorer_m, device))
    return results


def aggregate(results):
    if not results:
        return {}
    total = len(results)
    dc = {k: sum(1 for r in results if r["decision"] == k)
          for k in ("OK", "RESTORE", "REJECT")}
    q_scores  = [r["q_score"]  for r in results]
    latencies = [r["latency"]  for r in results]
    mse_list  = [r["mse"]  for r in results if r["decision"] == "RESTORE"]
    corr_list = [r["corr"] for r in results if r["decision"] == "RESTORE"]
    return {
        "total":      total,
        "ok_pct":     100 * dc["OK"]      / total,
        "restore_pct":100 * dc["RESTORE"] / total,
        "reject_pct": 100 * dc["REJECT"]  / total,
        "q_score_mean": float(np.mean(q_scores)),
        "lat_mean":     float(np.mean(latencies)),
        "lat_p50":      float(np.percentile(latencies, 50)),
        "lat_p95":      float(np.percentile(latencies, 95)),
        "mse_mean":     float(np.mean(mse_list))  if mse_list  else 0.0,
        "corr_mean":    float(np.mean(corr_list)) if corr_list else 0.0,
    }


# ──────────────────────────────────────────────
# FIGURA A — TABLA DE SUJETOS
# ──────────────────────────────────────────────
def figura_A(out_path):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.patch.set_facecolor("white")
    fig.suptitle("Tabla de sujetos del estudio", fontsize=14,
                 fontweight="bold", y=0.98)

    # ── Tabla sanos ──
    ax = axes[0]
    ax.axis("off")
    ax.set_title("Sujetos sanos (n = 10, 18 registros)", fontsize=11,
                 fontweight="bold", loc="left", pad=6)

    cols_s = ["ID sujeto", "Nombre", "Edad", "Sexo",
              "Músculo", "Tarea", "Miembro D", "Miembro I"]

    # FIX (auditoría de datos, agosto 2026): esta tabla estaba hardcodeada
    # de forma independiente a SANOS_META, con el mismo bug (le faltaba
    # Flor, Tania/Nancy corridas). Ahora se construye directamente desde
    # SANOS_META, ya corregido, para que nunca se puedan desincronizar.
    _sanos_por_sujeto = {}
    for _fname, _meta in SANOS_META.items():
        _key = _meta["sujeto"]
        _sanos_por_sujeto.setdefault(_key, {"nombre": _meta["nombre"], "edad": _meta["edad"],
                                             "sexo": _meta["sexo"], "D": "—", "I": "—"})
        if _meta["miembro"] == "Derecho":
            _sanos_por_sujeto[_key]["D"] = _fname.replace("emg", "").replace("Rcsv", "")
        else:
            _sanos_por_sujeto[_key]["I"] = _fname.replace("emg", "").replace("Rcsv", "")

    def _sujeto_num(k):
        return int(k.replace("S", ""))

    sanos_unicos = []
    for _key in sorted(_sanos_por_sujeto.keys(), key=_sujeto_num):
        _d = _sanos_por_sujeto[_key]
        sanos_unicos.append((
            _key, _d["nombre"], _d["edad"], _d["sexo"],
            "Flexor carpi radialis", "Abrir/cerrar mano", _d["D"], _d["I"]
        ))

    t = ax.table(
        cellText=sanos_unicos,
        colLabels=cols_s,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    t.auto_set_font_size(False)
    t.set_fontsize(8.5)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor("#1f77b4")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f0f4ff" if r % 2 == 0 else "white")

    # ── Tabla amputados ──
    ax2 = axes[1]
    ax2.axis("off")
    ax2.set_title("Participantes con amputación transradial (n = 2)", fontsize=11,
                  fontweight="bold", loc="left", pad=6)

    cols_a = ["ID", "Sexo", "Edad", "Años desde\namputación",
              "Extremidad", "Sesiones", "Semanas registradas", "Tarea"]

    # FIX: igual que la tabla de sanos, esto estaba hardcodeado por
    # separado con el bug de sexo/edad/años cruzados. Ahora se construye
    # desde AMPUTADOS_META, ya corregido.
    _extremidad = {"mujer": "Miembro residual (izquierdo)", "hombre": "Miembro residual (derecho)"}
    amp_rows = []
    for _key in ("mujer", "hombre"):
        _m = AMPUTADOS_META[_key]
        _pid = "P1" if _key == "mujer" else "P2"
        amp_rows.append((
            _pid, _m["sexo"], _m["edad"], _m["anos_amputacion"],
            _extremidad[_key], len(_m["sesiones"]),
            ", ".join(_m["semanas"]), "Activación muscular residual"
        ))

    t2 = ax2.table(
        cellText=amp_rows,
        colLabels=cols_a,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    t2.auto_set_font_size(False)
    t2.set_fontsize(8.5)
    for (r, c), cell in t2.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor("#9467bd")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f5f0ff" if r % 2 == 0 else "white")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Figura A guardada: {out_path}")


# ──────────────────────────────────────────────
# FIGURA B — DISTRIBUCIÓN DE DECISIONES
# ──────────────────────────────────────────────
def figura_B(sanos_agg, amp_agg, out_path):
    etiquetas = (
        [f"{SANOS_META[sid]['sujeto']}\n{SANOS_META[sid]['miembro'][:1]}"
         for sid in SANOS_IDS]
        + ["P1\n(Mujer)", "P2\n(Hombre)"]
    )
    ok_vals      = [d["ok_pct"]      for d in sanos_agg] + \
                   [amp_agg["mujer"]["ok_pct"], amp_agg["hombre"]["ok_pct"]]
    restore_vals = [d["restore_pct"] for d in sanos_agg] + \
                   [amp_agg["mujer"]["restore_pct"], amp_agg["hombre"]["restore_pct"]]
    reject_vals  = [d["reject_pct"]  for d in sanos_agg] + \
                   [amp_agg["mujer"]["reject_pct"], amp_agg["hombre"]["reject_pct"]]

    n   = len(etiquetas)
    x   = np.arange(n)
    fig, ax = plt.subplots(figsize=(16, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    b1 = ax.bar(x, ok_vals,      color=C_OK,      alpha=0.88, label="Aceptado",    width=0.65)
    b2 = ax.bar(x, restore_vals, color=C_RESTORE,  alpha=0.88, label="Restaurado",
                bottom=ok_vals, width=0.65)
    b3 = ax.bar(x, reject_vals,  color=C_REJECT,   alpha=0.88, label="Descartado",
                bottom=[o + r for o, r in zip(ok_vals, restore_vals)], width=0.65)

    # línea divisoria sanos / amputados
    ax.axvline(x=n - 2.5, color="black", linestyle="--", linewidth=1.2, alpha=0.5)
    ax.text(n - 2.5 - 0.3, 103, "Sanos", ha="right",   fontsize=9, color="gray")
    ax.text(n - 2.5 + 0.1, 103, "Amputados", ha="left", fontsize=9, color="gray")

    # colores de borde en amputados
    for bar in list(b1) + list(b2) + list(b3):
        bar.set_edgecolor("none")
    idx_amp = [n - 2, n - 1]
    for bars in [b1, b2, b3]:
        for i in idx_amp:
            bars[i].set_edgecolor("#333333")
            bars[i].set_linewidth(1.4)

    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas, fontsize=7.5)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Porcentaje de ventanas (%)", fontsize=10)
    ax.set_title(
        "Distribución de decisiones del flujo por sujeto\n"
        "(Aceptado / Restaurado / Descartado)",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=9, loc="upper left",
              framealpha=0.4, edgecolor="#cccccc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="gray")
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Figura B guardada: {out_path}")


# ──────────────────────────────────────────────
# FIGURA C — MÉTRICAS COMPARATIVAS
# ──────────────────────────────────────────────
def figura_C(sanos_agg, amp_agg, out_path):
    n_s = len(sanos_agg)
    x_s = np.arange(n_s)

    metricas = [
        ("q_score_mean", "Puntuación de calidad promedio",      [0, 1],    True),
        ("corr_mean",    "Correlación de reconstrucción (RECONSTRUIDO)", [0, 1], True),
        ("mse_mean",     "Error cuadrático medio (RECONSTRUIDO)",    None,       False),
        ("lat_mean",     "Latencia de inferencia promedio (ms)", None,      False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Métricas comparativas por sujeto: sanos vs. amputados",
        fontsize=12, fontweight="bold", y=1.01
    )

    for ax, (key, titulo, ylim, add_umbral) in zip(axes.flat, metricas):
        ax.set_facecolor("white")

        # puntos sanos
        vals_s = [d[key] for d in sanos_agg]
        ax.scatter(x_s, vals_s, color=C_SANO, s=55, zorder=5,
                   label="Sanos", alpha=0.85)

        # líneas de referencia amputados
        v_mujer  = amp_agg["mujer"][key]
        v_hombre = amp_agg["hombre"][key]
        ax.axhline(v_mujer,  color=C_AMP,   linestyle="--", linewidth=1.5,
                   label=f"P1 Mujer ({v_mujer:.3f})", alpha=0.9)
        ax.axhline(v_hombre, color="#8c564b", linestyle=":",  linewidth=1.5,
                   label=f"P2 Hombre ({v_hombre:.3f})", alpha=0.9)

        # umbral de calidad
        if add_umbral and key == "q_score_mean":
            ax.axhline(THR_QUALITY, color="red", linestyle="-.", linewidth=1,
                       alpha=0.5, label=f"Umbral ({THR_QUALITY})")

        ax.set_title(titulo, fontsize=9, fontweight="bold")
        ax.set_xlabel("Registro sano (índice)", fontsize=8)
        ax.set_xticks(x_s)
        ax.set_xticklabels(
            [f"{SANOS_META[sid]['sujeto']}\n{SANOS_META[sid]['miembro'][:1]}"
             for sid in SANOS_IDS],
            fontsize=6.5
        )
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=7.5, framealpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.35, color="gray")
        ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Figura C guardada: {out_path}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--carpeta",    default="./sMEG",   help="Carpeta con los CSV")
    p.add_argument("--models_dir", default=None,
                   help="Carpeta con los .pt. Si se omite, se descarga "
                        "automaticamente con kagglehub (jorgeoc/emg-modelos-v2).")
    p.add_argument("--out_dir",    default="./figuras_articulo",
                   help="Carpeta de salida para las figuras")
    return p.parse_args()


def resolve_models_dir(models_dir_arg):
    if models_dir_arg is not None:
        return models_dir_arg
    import kagglehub
    path = kagglehub.model_download("jorgeoc/emg-modelos-v2/pyTorch/default")
    print("Modelos descargados via kagglehub en:", path)
    return path


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Figura A (no requiere modelos) ──
    figura_A(os.path.join(args.out_dir, "figura_A_tabla_sujetos.png"))

    # ── Cargar modelos ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDispositivo: {device}")
    print("Cargando modelos...")
    intent_m, quality_m, restorer_m = load_models(resolve_models_dir(args.models_dir), device)
    print("Modelos cargados.\n")

    # ── Procesar sanos ──
    print("Procesando sujetos sanos...")
    sanos_agg = []
    for sid in SANOS_IDS:
        path = os.path.join(args.carpeta, sid + ".csv")
        if not os.path.exists(path):
            print(f"  ⚠ No encontrado: {path} — omitido")
            sanos_agg.append({k: 0.0 for k in
                ["ok_pct","restore_pct","reject_pct",
                 "q_score_mean","lat_mean","lat_p50","lat_p95",
                 "mse_mean","corr_mean","total"]})
            continue
        results = process_file(path, intent_m, quality_m, restorer_m, device)
        agg     = aggregate(results)
        sanos_agg.append(agg)
        print(f"  {sid}: OK={agg['ok_pct']:.1f}%  RESTORE={agg['restore_pct']:.1f}%  "
              f"REJECT={agg['reject_pct']:.1f}%  q={agg['q_score_mean']:.3f}  "
              f"lat={agg['lat_mean']:.2f}ms")

    # ── Procesar amputados (+ desglose por semana) ──
    print("\nProcesando amputados...")
    amp_agg = {}
    amp_weekly_rows = []
    for key, meta in AMPUTADOS_META.items():
        all_results = []
        print(f"  {meta['label']} — desglose por semana:")
        for wi, ses in enumerate(meta["sesiones"]):
            week_label = meta["semanas"][wi] if wi < len(meta["semanas"]) else f"S{wi+1}"
            path = os.path.join(args.carpeta, ses + ".csv")
            if not os.path.exists(path):
                print(f"    ⚠ No encontrado: {path} — omitido")
                continue
            res_week = process_file(path, intent_m, quality_m, restorer_m, device)
            agg_week = aggregate(res_week)
            if agg_week:
                print(f"    {week_label} ({ses}): OK={agg_week['ok_pct']:.1f}%  "
                      f"RESTORE={agg_week['restore_pct']:.1f}%  "
                      f"REJECT={agg_week['reject_pct']:.1f}%  "
                      f"q={agg_week['q_score_mean']:.3f}  "
                      f"corr={agg_week['corr_mean']:.3f}")
                amp_weekly_rows.append({
                    "participante": meta["label"], "semana": week_label,
                    "sesion": ses, **agg_week,
                })
            all_results.extend(res_week)
        agg = aggregate(all_results)
        amp_agg[key] = agg
        print(f"  {meta['label']} (TOTAL): OK={agg['ok_pct']:.1f}%  "
              f"RESTORE={agg['restore_pct']:.1f}%  "
              f"REJECT={agg['reject_pct']:.1f}%  q={agg['q_score_mean']:.3f}  "
              f"lat={agg['lat_mean']:.2f}ms")

    # Guardar CSV con el desglose semanal
    if amp_weekly_rows:
        weekly_path = os.path.join(args.out_dir, "amputados_por_semana.csv")
        with open(weekly_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(amp_weekly_rows[0].keys()))
            writer.writeheader()
            writer.writerows(amp_weekly_rows)
        print(f"\nDesglose semanal guardado: {weekly_path}")

        # Grafica de tendencia (calidad y % restaurado por semana)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        fig.patch.set_facecolor("white")
        for key, meta in AMPUTADOS_META.items():
            rows = [r for r in amp_weekly_rows if r["participante"] == meta["label"]]
            if not rows:
                continue
            xs = [r["semana"] for r in rows]
            q_vals = [r["q_score_mean"] for r in rows]
            restore_vals = [r["restore_pct"] for r in rows]
            axes[0].plot(xs, q_vals, marker="o", label=meta["label"], linewidth=2)
            axes[1].plot(xs, restore_vals, marker="o", label=meta["label"], linewidth=2)

        axes[0].axhline(0.5, color="gray", linestyle="--", alpha=0.6, label="Umbral (0.5)")
        axes[0].set_title("Calidad promedio por semana")
        axes[0].set_ylabel("q_score_mean")
        axes[0].legend(fontsize=8)
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)

        axes[1].set_title("% de ventanas restauradas por semana")
        axes[1].set_ylabel("Porcentaje (%)")
        axes[1].legend(fontsize=8)
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)

        plt.tight_layout()
        trend_path = os.path.join(args.out_dir, "figura_D_tendencia_semanal.png")
        fig.savefig(trend_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Figura de tendencia guardada: {trend_path}")

    # ── Figuras B y C ──
    figura_B(sanos_agg, amp_agg,
             os.path.join(args.out_dir, "figura_B_decisiones.png"))
    figura_C(sanos_agg, amp_agg,
             os.path.join(args.out_dir, "figura_C_metricas.png"))

    # ── CSV resumen ──
    resumen_path = os.path.join(args.out_dir, "resumen_resultados.csv")
    rows = []
    for i, sid in enumerate(SANOS_IDS):
        m = SANOS_META[sid]
        a = sanos_agg[i]
        rows.append({
            "grupo": "Sano", "id": sid,
            "sujeto": m["sujeto"], "edad": m["edad"],
            "sexo": m["sexo"], "miembro": m["miembro"],
            **{k: round(a[k], 4) for k in
               ["ok_pct","restore_pct","reject_pct",
                "q_score_mean","lat_mean","lat_p50","lat_p95",
                "mse_mean","corr_mean"]}
        })
    for key, meta in AMPUTADOS_META.items():
        a = amp_agg[key]
        rows.append({
            "grupo": "Amputado", "id": key,
            "sujeto": meta["label"], "edad": meta["edad"],
            "sexo": meta["sexo"], "miembro": "Muñón residual",
            **{k: round(a[k], 4) for k in
               ["ok_pct","restore_pct","reject_pct",
                "q_score_mean","lat_mean","lat_p50","lat_p95",
                "mse_mean","corr_mean"]}
        })
    pd.DataFrame(rows).to_csv(resumen_path, index=False)
    print(f"\nResumen CSV: {resumen_path}")
    print("\n¡Listo! Figuras generadas en:", args.out_dir)


if __name__ == "__main__":
    main()
