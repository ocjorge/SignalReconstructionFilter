"""
main.py — Inferencia de señal real/simulada + reconstrucción + reporte PDF
          + Arduino UNO R4 WiFi
          + análisis comparativo semanal Q1–Q4
          + HTML interactivo

Modos:
  python main.py --mode csv --csv_path signal.csv
  python main.py --mode sim
  python main.py --mode csv --csv_path signal.csv --no_arduino
  python main.py --mode csv --csv_path signal.csv --save_reconstructed --pdf_path reporte.pdf

  python main.py --mode weekly
  python main.py --mode weekly --weekly_csvs emgQ1csv.csv emgQ2csv.csv emgQ3csv.csv emgQ4csv.csv
  python main.py --mode weekly --no_html
  python main.py --mode weekly --models_dir ./models

Notas:
- Para CSV real tipo tiempo,señal, el flujo csv usa SOLO la segunda columna.
- Los modelos deben estar en ./models o en la ruta indicada por --models_dir:
    intent_binary_best.pt
    quality_judge_best.pt
    restorer_best.pt
"""

import os
import time
import csv
import glob
import json
import argparse
from typing import Iterator, Tuple, Optional, List

import numpy as np

import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

# Arduino / serial
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except Exception:
    serial = None
    SERIAL_AVAILABLE = False

# Correlación
try:
    from scipy.stats import pearsonr
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# =========================================================
# CONFIG
# =========================================================
SERIAL_BAUD       = 115200
SERIAL_TIMEOUT    = 3.0
SERIAL_BOOT_WAIT  = 3.5

THR_INTENT        = 0.50
THR_QUALITY       = 0.50

WIN               = 400
SAVE_LOG          = True
LOG_PATH          = "stream_log.csv"
REPORT_EVERY      = 25
WARMUP_ITERS      = 3

# Weekly defaults
# DEFAULT_WEEKLY_CSVS = ["emgQ1csv.csv", "emgQ2csv.csv", "emgQ3csv.csv", "emgQ4csv.csv"]
DEFAULT_WEEKLY_CSVS = ["emgP33RcsvAMPUTADA1.csv", "emgP34RcsvAMPUTADA2.csv", "emgP35RcsvAMPUTADA3.csv", "emgP35RcsvAMPUTADA3.csv"]
WEEKS               = ["Semana 1 (Q1)", "Semana 2 (Q2)", "Semana 3 (Q3)", "Semana 4 (Q4)"]

# Colores
DECISION_COLORS = {
    "OK": "#2ecc71",
    "RESTORE": "#e67e22",
    "REJECT": "#e74c3c",
}

WEEK_COLORS = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71"]
BG     = "#0f0f1a"
PANEL  = "#1a1a2e"
PANEL2 = "#16213e"
WHITE  = "white"
GRAY   = "#aaaacc"


# =========================================================
# ARGUMENTOS
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Inferencia de señal con reconstrucción, PDF, Arduino y análisis semanal"
    )

    parser.add_argument(
        "--mode",
        choices=["csv", "sim", "weekly"],
        default=None,
        help="Fuente de datos: 'csv', 'sim' o 'weekly'"
    )

    # flujo operativo
    parser.add_argument(
        "--csv_path",
        type=str,
        default="emgQ3csv.csv",
        help="Ruta al archivo CSV (solo si --mode csv)"
    )
    parser.add_argument(
        "--sim_windows",
        type=int,
        default=50,
        help="Número de ventanas simuladas (default: 50)"
    )
    parser.add_argument(
        "--no_arduino",
        action="store_true",
        help="No conectar al Arduino (solo consola)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Pausa en segundos entre ventanas (default: 0.5)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=200,
        help="Stride para señal continua (default: 200)"
    )
    parser.add_argument(
        "--max_windows",
        type=int,
        default=200,
        help="Máximo de ventanas a procesar (default: 200)"
    )
    parser.add_argument(
        "--save_reconstructed",
        action="store_true",
        help="Guardar ventanas reconstruidas en CSV"
    )
    parser.add_argument(
        "--reconstructed_csv",
        type=str,
        default="reconstructed_windows.csv",
        help="CSV de salida para ventanas reconstruidas"
    )
    parser.add_argument(
        "--pdf_path",
        type=str,
        default="reconstruction_report.pdf",
        help="Ruta del PDF operativo de salida"
    )
    parser.add_argument(
        "--pdf_max_examples",
        type=int,
        default=30,
        help="Máximo de ejemplos incluidos en el PDF operativo"
    )

    # modelos
    parser.add_argument(
        "--models_dir",
        type=str,
        default=None,
        help="Ruta opcional a carpeta de modelos"
    )

    # weekly
    parser.add_argument(
        "--weekly_csvs",
        nargs=4,
        default=DEFAULT_WEEKLY_CSVS,
        help="Cuatro archivos CSV para weekly: Q1 Q2 Q3 Q4"
    )
    parser.add_argument(
        "--weekly_pdf_path",
        type=str,
        default="emg_reporte_semanal.pdf",
        help="Ruta del PDF semanal"
    )
    parser.add_argument(
        "--html_path",
        type=str,
        default="emg_comparativo_interactivo.html",
        help="Ruta del HTML semanal"
    )
    parser.add_argument(
        "--no_pdf",
        action="store_true",
        help="Omitir generación de PDF semanal"
    )
    parser.add_argument(
        "--no_html",
        action="store_true",
        help="Omitir generación de HTML semanal"
    )

    return parser.parse_args()


# =========================================================
# MODELOS
# =========================================================
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
            nn.Dropout(0.1),
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
            nn.ReLU(),
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
        return self.decoder(z)


# =========================================================
# UTILIDADES
# =========================================================
def robust_norm_1d(x: np.ndarray) -> np.ndarray:
    x   = np.asarray(x, dtype=np.float32).reshape(-1)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-6
    z   = (x - med) / (1.4826 * mad)
    return np.clip(z, -10, 10).astype(np.float32)


def percentiles_ms(values: List[float]):
    if not values:
        return {"p50": float("nan"), "p90": float("nan"), "p95": float("nan")}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def compute_mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def compute_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    if SCIPY_AVAILABLE:
        r, _ = pearsonr(a, b)
        return float(r)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 else 0.0


def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL2)
    ax.tick_params(colors=GRAY, labelsize=8)
    ax.xaxis.label.set_color(GRAY)
    ax.yaxis.label.set_color(GRAY)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333355")
    if title:
        ax.set_title(title, color=WHITE, fontsize=9, pad=4)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, alpha=0.15, color="#444466")


# =========================================================
# CARGA DE MODELOS
# =========================================================
def find_models_dir(override: Optional[str] = None) -> str:
    candidates = []
    if override:
        candidates.append(os.path.abspath(override))
    candidates.append(os.path.abspath("./models"))

    for p in glob.glob("/kaggle/input/**/intent_binary_best.pt", recursive=True):
        candidates.append(os.path.dirname(p))

    required = ["intent_binary_best.pt", "quality_judge_best.pt", "restorer_best.pt"]
    for d in candidates:
        if all(os.path.exists(os.path.join(d, f)) for f in required):
            return d

    raise FileNotFoundError(
        "\n[ERROR] No se encontraron los modelos.\n"
        "Coloca los tres archivos .pt en ./models/ :\n"
        "    intent_binary_best.pt\n"
        "    quality_judge_best.pt\n"
        "    restorer_best.pt\n"
        "O usa --models_dir /ruta/a/tus/modelos"
    )


def load_models(models_dir: str, device: str):
    print(f"Modelos en: {models_dir}")

    intent = IntentTCN().to(device)
    intent.load_state_dict(
        torch.load(os.path.join(models_dir, "intent_binary_best.pt"), map_location=device)
    )
    intent.eval()

    quality = QualityNet().to(device)
    quality.load_state_dict(
        torch.load(os.path.join(models_dir, "quality_judge_best.pt"), map_location=device)
    )
    quality.eval()

    restorer = ConvAE1D().to(device)
    restorer.load_state_dict(
        torch.load(os.path.join(models_dir, "restorer_best.pt"), map_location=device)
    )
    restorer.eval()

    print("Modelos cargados.\n")
    return intent, quality, restorer


# =========================================================
# ARDUINO
# =========================================================
def find_serial_port() -> str:
    if not SERIAL_AVAILABLE:
        raise RuntimeError("pyserial no está instalado.")

    ports = list(serial.tools.list_ports.comports())
    if not ports:
        raise RuntimeError("No se detectaron puertos seriales.")

    if len(ports) == 1:
        print(f"Puerto detectado: {ports[0].device} — {ports[0].description}")
        return ports[0].device

    print("Puertos disponibles:")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p.device} — {p.description}")

    while True:
        try:
            idx = int(input("Elige el número del puerto: "))
            if 0 <= idx < len(ports):
                return ports[idx].device
        except ValueError:
            pass
        print(f"Elige un número entre 0 y {len(ports) - 1}.")


def connect_arduino():
    port = find_serial_port()
    try:
        ser = serial.Serial(port, SERIAL_BAUD, timeout=SERIAL_TIMEOUT)
        print(f"  Esperando boot del Arduino ({SERIAL_BOOT_WAIT}s)...")
        time.sleep(SERIAL_BOOT_WAIT)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        print(f"Arduino conectado en {port} @ {SERIAL_BAUD}\n")
        return ser
    except Exception as e:
        print(f"ERROR al abrir {port}: {e}")
        print("Verifica que el Arduino esté conectado y que ningún otro programa use el puerto.")
        raise


def send_to_arduino(ser, decision: str):
    ser.reset_input_buffer()
    ser.write((decision + "\n").encode("utf-8"))
    ser.flush()

    wait = 1.0 if decision == "RESTORE" else 0.5
    time.sleep(wait)

    ack = ser.readline().decode("utf-8", errors="ignore").strip()
    if ack:
        print(f"  Arduino ACK: {ack}")
    else:
        print(f"  Arduino ACK: (sin respuesta — decisión enviada: {decision})")


# =========================================================
# LECTURA DE CSV
# =========================================================
def load_signal_csv_second_column(path: str) -> np.ndarray:
    """
    Lee CSV general.
    - Si hay >= 2 columnas numéricas, usa la segunda.
    - Si hay 1 columna numérica, usa esa.
    """
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            vals = []
            for v in row:
                vv = str(v).strip()
                if not vv:
                    continue
                try:
                    vals.append(float(vv))
                except ValueError:
                    pass
            if vals:
                rows.append(vals)

    if not rows:
        raise ValueError(f"CSV sin datos numéricos válidos: {path}")

    if max(len(r) for r in rows) >= 2:
        data = [r[1] for r in rows if len(r) >= 2]
    else:
        data = [r[0] for r in rows]

    arr = np.asarray(data, dtype=np.float32).reshape(-1)
    if len(arr) < WIN:
        raise ValueError(f"La señal tiene {len(arr)} muestras, pero WIN={WIN}.")
    return arr


def load_csv_time_signal(path: str):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            try:
                vals = [float(v) for v in row if str(v).strip()]
            except ValueError:
                continue
            if len(vals) >= 2:
                rows.append(vals)

    if not rows:
        raise ValueError(f"CSV sin filas válidas de dos columnas: {path}")

    times  = np.array([r[0] for r in rows], dtype=np.float32)
    signal = np.array([r[1] for r in rows], dtype=np.float32)
    return times, signal


# =========================================================
# FUENTES DE DATOS
# =========================================================
def source_csv(
    csv_path: str,
    win: int,
    stride: int,
    max_windows: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    signal = load_signal_csv_second_column(csv_path)
    n = len(signal)
    n_wins = min(1 + max(0, (n - win) // stride), max_windows)

    for i in range(n_wins):
        s = i * stride
        e = s + win
        raw_w = signal[s:e]
        if len(raw_w) < win:
            break
        norm_w = robust_norm_1d(raw_w)
        t_w = np.arange(s, e, dtype=np.float32)
        yield raw_w.astype(np.float32), norm_w, t_w


def source_sim(sim_windows: int, win: int) -> Iterator[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    rng = np.random.default_rng(12345)

    for i in range(sim_windows):
        x = np.linspace(0, 2 * np.pi, win, dtype=np.float32)

        signal = (
            0.9 * np.sin(2.5 * x + rng.uniform(0, np.pi))
            + 0.4 * np.sin(5.5 * x + rng.uniform(0, np.pi))
            + 0.2 * rng.normal(size=win)
        ).astype(np.float32)

        # A veces degradamos más para forzar RESTORE
        if i % 3 == 0:
            signal += (0.7 * rng.normal(size=win)).astype(np.float32)
        if i % 5 == 0:
            signal *= np.float32(rng.uniform(0.2, 0.6))

        raw_w = signal.astype(np.float32)
        norm_w = robust_norm_1d(raw_w)
        t_w = np.arange(win, dtype=np.float32)
        yield raw_w, norm_w, t_w


# =========================================================
# INFERENCIA
# =========================================================
@torch.no_grad()
def infer_window(
    window_norm: np.ndarray,
    intent_model,
    quality_model,
    restorer_model,
    device,
) -> dict:
    xb = torch.tensor(window_norm[None, None, :], dtype=torch.float32, device=device)
    t0 = time.perf_counter()

    p_intent = float(torch.sigmoid(intent_model(xb)).cpu().reshape(-1)[0])

    q_score_t, q_logit = quality_model(xb)
    q_score_pred = float(q_score_t.cpu().reshape(-1)[0])
    q_prob_good  = float(torch.sigmoid(q_logit).cpu().reshape(-1)[0])

    if p_intent < THR_INTENT:
        decision      = "REJECT"
        reason        = "no_intent_detected"
        reconstructed = window_norm.copy()

    elif q_prob_good >= THR_QUALITY:
        decision      = "OK"
        reason        = "signal_quality_good_enough"
        reconstructed = window_norm.copy()

    else:
        decision      = "RESTORE"
        reason        = "intent_detected_quality_bad"
        reconstructed = restorer_model(xb).cpu().numpy()[0, 0].astype(np.float32)

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "decision":             decision,
        "reason":               reason,
        "p_intent":             p_intent,
        "q_prob_good":          q_prob_good,
        "q_score_pred":         q_score_pred,
        "latency_ms":           latency_ms,
        "reconstructed_window": reconstructed,
        # aliases para weekly
        "q_score":              q_score_pred,
        "reconstructed":        reconstructed,
    }


def warmup(intent_model, quality_model, restorer_model, device):
    dummy = np.zeros(WIN, dtype=np.float32)
    for _ in range(WARMUP_ITERS):
        infer_window(dummy, intent_model, quality_model, restorer_model, device)
    print(f"Warmup completado ({WARMUP_ITERS} iteraciones).\n")


# =========================================================
# LOG CSV
# =========================================================
def init_log(path: str):
    if not SAVE_LOG:
        return None, None
    f = open(path, "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow([
        "timestamp", "idx", "decision", "reason",
        "p_intent", "q_prob_good", "q_score_pred", "latency_ms",
    ])
    return f, writer


def init_reconstructed_csv(path: str):
    f = open(path, "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(["idx", "decision", "reason"] + [f"v{i}" for i in range(WIN)])
    return f, writer


# =========================================================
# PDF OPERATIVO
# =========================================================
def save_pdf_report(pdf_path: str, examples: List[dict], summary: dict):
    print(f"\nGenerando PDF en: {pdf_path} ...")

    with PdfPages(pdf_path) as pdf:
        # portada
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor("#1a1a2e")
        fig.suptitle(
            "Reporte de Reconstrucción de Señales",
            fontsize=22, fontweight="bold", color="white", y=0.93
        )
        plt.axis("off")

        dc = summary["decision_counts"]
        total = summary["num_valid"] or 1

        lines = [
            f"Ventanas procesadas  : {summary['num_valid']}",
            f"  OK      : {dc.get('OK', 0):>4}  ({100*dc.get('OK',0)/total:.1f}%)",
            f"  RESTORE : {dc.get('RESTORE', 0):>4}  ({100*dc.get('RESTORE',0)/total:.1f}%)",
            f"  REJECT  : {dc.get('REJECT', 0):>4}  ({100*dc.get('REJECT',0)/total:.1f}%)",
            "",
            f"Umbrales             : THR_INTENT={THR_INTENT}  |  THR_QUALITY={THR_QUALITY}",
            f"Ventana              : {WIN} muestras",
        ]
        if summary["latencies"]:
            p = percentiles_ms(summary["latencies"])
            lines += [
                "",
                f"Latencia (ms)  mean : {np.mean(summary['latencies']):.3f}",
                f"               p50  : {p['p50']:.3f}",
                f"               p90  : {p['p90']:.3f}",
                f"               p95  : {p['p95']:.3f}",
            ]

        y = 0.80
        for line in lines:
            fig.text(0.08, y, line, fontsize=13, color="white", fontfamily="monospace")
            y -= 0.055

        counts = [dc.get(k, 0) for k in ("OK", "RESTORE", "REJECT")]
        labels = ["OK", "RESTORE", "REJECT"]
        colors = [DECISION_COLORS[k] for k in labels]
        non_zero = [(c, l, col) for c, l, col in zip(counts, labels, colors) if c > 0]

        if non_zero:
            ax_pie = fig.add_axes([0.60, 0.25, 0.35, 0.55])
            ax_pie.set_facecolor("#1a1a2e")
            nz_counts, nz_labels, nz_colors = zip(*non_zero)
            wedges, texts, autotexts = ax_pie.pie(
                nz_counts,
                labels=nz_labels,
                colors=nz_colors,
                autopct="%1.1f%%",
                startangle=90,
                textprops={"color": "white", "fontsize": 11},
            )
            for at in autotexts:
                at.set_color("white")
            ax_pie.set_title("Distribución de decisiones", color="white", fontsize=12, pad=10)

        pdf.savefig(fig, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        # páginas por ejemplo
        for ex in examples:
            fig, axs = plt.subplots(2, 1, figsize=(11.69, 8.27), sharex=False)
            fig.patch.set_facecolor("white")

            idx       = ex["idx"]
            decision  = ex["decision"]
            reason    = ex["reason"]
            raw_w     = ex["raw_window"]
            norm_w    = ex["norm_window"]
            recon_w   = ex["reconstructed_window"]
            p_intent  = ex["p_intent"]
            q_good    = ex["q_prob_good"]
            q_score   = ex["q_score_pred"]
            latency   = ex["latency_ms"]
            t_w       = ex.get("time_window", None)

            x = t_w if t_w is not None else np.arange(len(raw_w))

            axs[0].plot(x, raw_w, linewidth=1.2)
            axs[0].set_title(f"Ventana #{idx} — Señal original")
            axs[0].set_ylabel("Amplitud")
            axs[0].grid(True, alpha=0.3)

            axs[1].plot(np.arange(len(norm_w)), norm_w, label="Normalizada", linewidth=1.2)
            axs[1].plot(np.arange(len(recon_w)), recon_w, label="Reconstruida/Passthrough", linewidth=1.2)
            axs[1].set_title("Entrada al modelo vs salida")
            axs[1].set_xlabel("Muestra")
            axs[1].set_ylabel("Valor")
            axs[1].grid(True, alpha=0.3)
            axs[1].legend()

            fig.suptitle(
                f"Decisión: {decision} | razón={reason} | "
                f"p_intent={p_intent:.3f} | q_good={q_good:.3f} | "
                f"q_score={q_score:.3f} | {latency:.2f} ms",
                fontsize=12,
                fontweight="bold",
                color=DECISION_COLORS.get(decision, "black"),
                y=0.98
            )

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print("PDF generado correctamente.")


# =========================================================
# WEEKLY
# =========================================================
def process_week(
    week_idx: int,
    csv_path: str,
    intent_model,
    quality_model,
    restorer_model,
    device: str,
    stride: int,
    max_windows: int,
) -> dict:
    week_name = WEEKS[week_idx]

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"\n[ERROR] No se encontró {csv_path}\n"
            "Asegúrate de que los CSV estén en la ruta correcta."
        )

    print(f"  Leyendo {csv_path} ...")
    times, raw_signal = load_csv_time_signal(csv_path)
    n = len(raw_signal)
    n_wins = min(1 + max(0, (n - WIN) // stride), max_windows)
    print(f"  {n} muestras  →  {n_wins} ventanas  (win={WIN}, stride={stride})")

    windows_raw, windows_norm, windows_recon, windows_time = [], [], [], []
    decisions, reasons = [], []
    p_intents, q_scores, q_prob_goods, latencies = [], [], [], []

    for i in range(n_wins):
        s = i * stride
        e = s + WIN
        raw_w = raw_signal[s:e]
        t_w   = times[s:e]

        if len(raw_w) < WIN:
            break

        norm_w = robust_norm_1d(raw_w)
        res = infer_window(norm_w, intent_model, quality_model, restorer_model, device)

        windows_raw.append(raw_w)
        windows_norm.append(norm_w)
        windows_recon.append(res["reconstructed"])
        windows_time.append(t_w)
        decisions.append(res["decision"])
        reasons.append(res["reason"])
        p_intents.append(res["p_intent"])
        q_scores.append(res["q_score"])
        q_prob_goods.append(res["q_prob_good"])
        latencies.append(res["latency_ms"])

    dec_counts = {k: decisions.count(k) for k in ("OK", "RESTORE", "REJECT")}

    mse_list, corr_list = [], []
    for i, d in enumerate(decisions):
        if d == "RESTORE":
            mse_list.append(compute_mse(windows_norm[i], windows_recon[i]))
            corr_list.append(compute_corr(windows_norm[i], windows_recon[i]))

    q_ok      = [q_scores[i] for i, d in enumerate(decisions) if d == "OK"]
    q_restore = [q_scores[i] for i, d in enumerate(decisions) if d == "RESTORE"]
    q_reject  = [q_scores[i] for i, d in enumerate(decisions) if d == "REJECT"]

    examples = {}
    for dec in ("OK", "RESTORE", "REJECT"):
        idxs = [i for i, d in enumerate(decisions) if d == dec]
        if idxs:
            pick = idxs[len(idxs) // 2]
            examples[dec] = {
                "idx":        pick,
                "raw":        windows_raw[pick],
                "norm":       windows_norm[pick],
                "recon":      windows_recon[pick],
                "time":       windows_time[pick],
                "p_intent":   p_intents[pick],
                "q_score":    q_scores[pick],
                "q_prob_good": q_prob_goods[pick],
                "latency_ms": latencies[pick],
            }

    return {
        "week_name":    week_name,
        "week_idx":     week_idx,
        "csv_path":     csv_path,
        "n_windows":    len(decisions),
        "dec_counts":   dec_counts,
        "p_intents":    p_intents,
        "q_scores":     q_scores,
        "q_prob_goods": q_prob_goods,
        "latencies":    latencies,
        "mse_list":     mse_list,
        "corr_list":    corr_list,
        "q_ok":         q_ok,
        "q_restore":    q_restore,
        "q_reject":     q_reject,
        "examples":     examples,
        "raw_signal":   raw_signal,
        "times":        times,
    }


def make_cover(pdf: PdfPages, all_weeks: list):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor(BG)

    fig.text(
        0.5, 0.93,
        "Análisis Comparativo de Señales sEMG — Rehabilitación de Prótesis",
        ha="center", fontsize=20, fontweight="bold", color=WHITE
    )
    fig.text(
        0.5, 0.87,
        f"Pipeline real: IntentTCN · QualityNet · ConvAE1D | "
        f"THR_INTENT={THR_INTENT}  THR_QUALITY={THR_QUALITY} | WIN={WIN}",
        ha="center", fontsize=11, color=GRAY
    )

    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        left=0.07, right=0.96, top=0.82, bottom=0.07,
        hspace=0.42, wspace=0.32
    )

    totals  = [sum(w["dec_counts"].values()) or 1 for w in all_weeks]
    ok_pcts = [100 * w["dec_counts"].get("OK",0)      / t for w, t in zip(all_weeks, totals)]
    re_pcts = [100 * w["dec_counts"].get("RESTORE",0) / t for w, t in zip(all_weeks, totals)]
    rj_pcts = [100 * w["dec_counts"].get("REJECT",0)  / t for w, t in zip(all_weeks, totals)]

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(np.arange(4), ok_pcts, color=DECISION_COLORS["OK"], alpha=0.85, label="OK")
    ax1.bar(np.arange(4), re_pcts, color=DECISION_COLORS["RESTORE"], alpha=0.85,
            label="RESTORE", bottom=ok_pcts)
    ax1.bar(
        np.arange(4), rj_pcts, color=DECISION_COLORS["REJECT"], alpha=0.85,
        label="REJECT", bottom=[o+r for o, r in zip(ok_pcts, re_pcts)]
    )
    ax1.set_xticks(np.arange(4))
    ax1.set_xticklabels([f"Q{i+1}" for i in range(4)], color=GRAY, fontsize=8)
    ax1.set_ylim(0, 100)
    style_ax(ax1, "Distribución de Decisiones (%)", ylabel="% ventanas")
    ax1.legend(fontsize=7, facecolor=PANEL, labelcolor=WHITE, framealpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    means_q = [float(np.mean(w["q_scores"])) if w["q_scores"] else 0.0 for w in all_weeks]
    ax2.plot(np.arange(4), means_q, marker="o", linewidth=2)
    ax2.set_xticks(np.arange(4))
    ax2.set_xticklabels([f"Q{i+1}" for i in range(4)], color=GRAY, fontsize=8)
    style_ax(ax2, "q_score promedio por semana", ylabel="q_score")

    ax3 = fig.add_subplot(gs[1, 0])
    lat_means = [float(np.mean(w["latencies"])) if w["latencies"] else 0.0 for w in all_weeks]
    ax3.bar(np.arange(4), lat_means, color=WEEK_COLORS, alpha=0.9)
    ax3.set_xticks(np.arange(4))
    ax3.set_xticklabels([f"Q{i+1}" for i in range(4)], color=GRAY, fontsize=8)
    style_ax(ax3, "Latencia promedio", ylabel="ms")

    ax4 = fig.add_subplot(gs[1, 1])
    mse_means = [float(np.mean(w["mse_list"])) if w["mse_list"] else 0.0 for w in all_weeks]
    corr_means = [float(np.mean(w["corr_list"])) if w["corr_list"] else 0.0 for w in all_weeks]
    ax4.plot(np.arange(4), mse_means, marker="o", label="MSE RESTORE")
    ax4_t = ax4.twinx()
    ax4_t.plot(np.arange(4), corr_means, marker="s", linestyle="--", label="Corr RESTORE")
    style_ax(ax4, "Reconstrucción (solo RESTORE)", ylabel="MSE")
    ax4_t.tick_params(colors=GRAY, labelsize=8)
    ax4_t.spines["right"].set_edgecolor("#333355")
    ax4_t.set_ylabel("Correlación", color=GRAY, fontsize=8)

    fig.text(
        0.07, 0.02,
        "Resumen global del análisis semanal. Cada semana se procesa con la misma lógica "
        "de inferencia del flujo operativo.",
        color=GRAY, fontsize=9
    )

    pdf.savefig(fig, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def make_week_page(pdf: PdfPages, wdata: dict):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(2, 2, figure=fig, left=0.06, right=0.97, top=0.90, bottom=0.08,
                           hspace=0.35, wspace=0.28)

    week_name = wdata["week_name"]
    fig.text(0.5, 0.95, week_name, ha="center", fontsize=18, fontweight="bold", color=WHITE)

    dc = wdata["dec_counts"]
    total = sum(dc.values()) or 1

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.pie(
        [dc["OK"], dc["RESTORE"], dc["REJECT"]],
        labels=["OK", "RESTORE", "REJECT"],
        colors=[DECISION_COLORS["OK"], DECISION_COLORS["RESTORE"], DECISION_COLORS["REJECT"]],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"color": WHITE, "fontsize": 9}
    )
    ax1.set_facecolor(PANEL2)
    ax1.set_title("Distribución de decisiones", color=WHITE)

    ax2 = fig.add_subplot(gs[0, 1])
    qs = np.asarray(wdata["q_scores"], dtype=np.float32) if wdata["q_scores"] else np.array([0], dtype=np.float32)
    ax2.hist(qs, bins=20, alpha=0.9)
    style_ax(ax2, "Histograma de q_score", xlabel="q_score", ylabel="Frecuencia")

    ax3 = fig.add_subplot(gs[1, 0])
    lats = np.asarray(wdata["latencies"], dtype=np.float32) if wdata["latencies"] else np.array([0], dtype=np.float32)
    ax3.plot(lats, linewidth=1.2)
    style_ax(ax3, "Latencia por ventana", xlabel="Ventana", ylabel="ms")

    ax4 = fig.add_subplot(gs[1, 1])
    txt = [
        f"CSV              : {wdata['csv_path']}",
        f"Ventanas         : {wdata['n_windows']}",
        f"OK               : {dc['OK']} ({100*dc['OK']/total:.1f}%)",
        f"RESTORE          : {dc['RESTORE']} ({100*dc['RESTORE']/total:.1f}%)",
        f"REJECT           : {dc['REJECT']} ({100*dc['REJECT']/total:.1f}%)",
        "",
        f"q_score promedio : {np.mean(qs):.4f}",
        f"Latencia media   : {np.mean(lats):.4f} ms",
        f"MSE promedio     : {np.mean(wdata['mse_list']) if wdata['mse_list'] else 0.0:.6f}",
        f"Corr promedio    : {np.mean(wdata['corr_list']) if wdata['corr_list'] else 0.0:.6f}",
    ]
    ax4.set_facecolor(PANEL2)
    ax4.axis("off")
    y = 0.92
    for line in txt:
        ax4.text(0.03, y, line, color=WHITE, fontsize=11, family="monospace")
        y -= 0.085

    pdf.savefig(fig, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def make_examples_page(pdf: PdfPages, wdata: dict):
    examples = wdata["examples"]
    if not examples:
        return

    fig, axs = plt.subplots(3, 2, figsize=(11.69, 8.27))
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"Ejemplos representativos — {wdata['week_name']}", color=WHITE, fontsize=18, y=0.97)

    order = ["OK", "RESTORE", "REJECT"]
    row = 0

    for dec in order:
        if dec not in examples:
            axs[row, 0].axis("off")
            axs[row, 1].axis("off")
            row += 1
            continue

        ex = examples[dec]
        t = ex["time"]
        raw = ex["raw"]
        norm = ex["norm"]
        recon = ex["recon"]

        axL = axs[row, 0]
        axR = axs[row, 1]

        axL.plot(t, raw, linewidth=1.2)
        style_ax(axL, f"{dec} — señal original", xlabel="Tiempo", ylabel="Amplitud")

        axR.plot(np.arange(len(norm)), norm, label="Normalizada", linewidth=1.2)
        axR.plot(np.arange(len(recon)), recon, label="Recon/Passthrough", linewidth=1.2)
        style_ax(
            axR,
            f"{dec} — salida",
            xlabel="Muestra",
            ylabel="Valor"
        )
        axR.legend(fontsize=7, facecolor=PANEL, labelcolor=WHITE, framealpha=0.3)

        row += 1

    pdf.savefig(fig, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def make_progression_page(pdf: PdfPages, all_weeks: list):
    fig, axs = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Comparativa Q1 → Q4", color=WHITE, fontsize=18, y=0.97)

    x = np.arange(4)

    ok_p = []
    re_p = []
    rj_p = []
    q_m  = []
    lat_m = []
    mse_m = []
    corr_m = []

    for w in all_weeks:
        dc = w["dec_counts"]
        total = sum(dc.values()) or 1
        ok_p.append(100 * dc["OK"] / total)
        re_p.append(100 * dc["RESTORE"] / total)
        rj_p.append(100 * dc["REJECT"] / total)
        q_m.append(float(np.mean(w["q_scores"])) if w["q_scores"] else 0.0)
        lat_m.append(float(np.mean(w["latencies"])) if w["latencies"] else 0.0)
        mse_m.append(float(np.mean(w["mse_list"])) if w["mse_list"] else 0.0)
        corr_m.append(float(np.mean(w["corr_list"])) if w["corr_list"] else 0.0)

    axs[0, 0].plot(x, ok_p, marker="o", label="OK")
    axs[0, 0].plot(x, re_p, marker="s", label="RESTORE")
    axs[0, 0].plot(x, rj_p, marker="^", label="REJECT")
    style_ax(axs[0, 0], "Evolución de decisiones", xlabel="Semana", ylabel="%")
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(["Q1", "Q2", "Q3", "Q4"], color=GRAY)
    axs[0, 0].legend(fontsize=7, facecolor=PANEL, labelcolor=WHITE, framealpha=0.3)

    axs[0, 1].plot(x, q_m, marker="o")
    style_ax(axs[0, 1], "q_score promedio", xlabel="Semana", ylabel="q_score")
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(["Q1", "Q2", "Q3", "Q4"], color=GRAY)

    axs[1, 0].bar(x, lat_m, color=WEEK_COLORS)
    style_ax(axs[1, 0], "Latencia promedio", xlabel="Semana", ylabel="ms")
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(["Q1", "Q2", "Q3", "Q4"], color=GRAY)

    axs[1, 1].plot(x, mse_m, marker="o", label="MSE")
    ax_t = axs[1, 1].twinx()
    ax_t.plot(x, corr_m, marker="s", linestyle="--", label="Corr")
    style_ax(axs[1, 1], "Reconstrucción en RESTORE", xlabel="Semana", ylabel="MSE")
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(["Q1", "Q2", "Q3", "Q4"], color=GRAY)
    ax_t.tick_params(colors=GRAY, labelsize=8)
    ax_t.spines["right"].set_edgecolor("#333355")
    ax_t.set_ylabel("Correlación", color=GRAY, fontsize=8)

    pdf.savefig(fig, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def generate_weekly_pdf(all_weeks: list, pdf_path: str):
    print(f"\nGenerando PDF semanal en: {pdf_path} ...")
    with PdfPages(pdf_path) as pdf:
        make_cover(pdf, all_weeks)
        for w in all_weeks:
            make_week_page(pdf, w)
            make_examples_page(pdf, w)
        make_progression_page(pdf, all_weeks)
    print("PDF semanal generado correctamente.")


def generate_html(all_weeks: list, html_path: str):
    print(f"Generando HTML interactivo en: {html_path} ...")

    data = []
    for w in all_weeks:
        dc = w["dec_counts"]
        total = sum(dc.values()) or 1
        data.append({
            "week_name": w["week_name"],
            "csv_path": w["csv_path"],
            "n_windows": w["n_windows"],
            "ok_pct": 100 * dc["OK"] / total,
            "restore_pct": 100 * dc["RESTORE"] / total,
            "reject_pct": 100 * dc["REJECT"] / total,
            "q_score_mean": float(np.mean(w["q_scores"])) if w["q_scores"] else 0.0,
            "lat_mean": float(np.mean(w["latencies"])) if w["latencies"] else 0.0,
            "mse_mean": float(np.mean(w["mse_list"])) if w["mse_list"] else 0.0,
            "corr_mean": float(np.mean(w["corr_list"])) if w["corr_list"] else 0.0,
        })

    html = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Comparativo sEMG Q1-Q4</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body {{
  font-family: Arial, sans-serif;
  margin: 0;
  background: #0f0f1a;
  color: white;
}}
.container {{
  max-width: 1100px;
  margin: 0 auto;
  padding: 24px;
}}
.card {{
  background: #1a1a2e;
  border-radius: 14px;
  padding: 18px;
  margin-bottom: 18px;
  box-shadow: 0 8px 22px rgba(0,0,0,.25);
}}
h1, h2 {{
  margin-top: 0;
}}
table {{
  width: 100%;
  border-collapse: collapse;
}}
th, td {{
  padding: 10px;
  border-bottom: 1px solid #2d2d50;
  text-align: left;
}}
.bar {{
  height: 14px;
  border-radius: 7px;
  background: linear-gradient(90deg, #2ecc71, #f1c40f, #e67e22, #e74c3c);
}}
.small {{
  color: #bbb;
  font-size: 13px;
}}
pre {{
  background: #16213e;
  padding: 14px;
  border-radius: 10px;
  overflow: auto;
}}
</style>
</head>
<body>
<div class="container">
  <div class="card">
    <h1>Análisis Comparativo sEMG — Q1 a Q4</h1>
    <div class="small">
      Pipeline: IntentTCN · QualityNet · ConvAE1D |
      THR_INTENT={THR_INTENT} |
      THR_QUALITY={THR_QUALITY} |
      WIN={WIN}
    </div>
  </div>

  <div class="card">
    <h2>Resumen tabular</h2>
    <table>
      <thead>
        <tr>
          <th>Semana</th>
          <th>Ventanas</th>
          <th>OK %</th>
          <th>RESTORE %</th>
          <th>REJECT %</th>
          <th>q_score medio</th>
          <th>Latencia media (ms)</th>
          <th>MSE medio</th>
          <th>Corr media</th>
        </tr>
      </thead>
      <tbody>
"""

    for row in data:
        html += f"""
        <tr>
          <td>{row['week_name']}</td>
          <td>{row['n_windows']}</td>
          <td>{row['ok_pct']:.2f}</td>
          <td>{row['restore_pct']:.2f}</td>
          <td>{row['reject_pct']:.2f}</td>
          <td>{row['q_score_mean']:.4f}</td>
          <td>{row['lat_mean']:.4f}</td>
          <td>{row['mse_mean']:.6f}</td>
          <td>{row['corr_mean']:.6f}</td>
        </tr>
"""

    html += """
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2>Datos JSON</h2>
    <pre>""" + json.dumps(data, indent=2, ensure_ascii=False) + """</pre>
  </div>
</div>
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print("HTML generado correctamente.")


# =========================================================
# FLUJO OPERATIVO
# =========================================================
def run_operational_mode(args, intent_model, quality_model, restorer_model, device):
    ser = None
    if not args.no_arduino:
        try:
            ser = connect_arduino()
        except Exception:
            print("⚠️  Continuando sin Arduino (solo consola).\n")

    if args.mode == "csv":
        print(f"Leyendo señal desde: {args.csv_path}\n")
        data_source = source_csv(
            args.csv_path,
            win=WIN,
            stride=args.stride,
            max_windows=args.max_windows,
        )
    else:
        print(f"Generando {args.sim_windows} ventanas simuladas.\n")
        data_source = source_sim(args.sim_windows, WIN)

    log_file, log_writer = init_log(LOG_PATH)

    reconstructed_file = None
    reconstructed_writer = None
    if args.save_reconstructed:
        reconstructed_file, reconstructed_writer = init_reconstructed_csv(args.reconstructed_csv)

    latencies = []
    num_valid = 0
    decision_counts = {"OK": 0, "REJECT": 0, "RESTORE": 0}
    pdf_examples: List[dict] = []

    print("=" * 92)
    print(
        f"{'#':>5}  {'DECISION':<8}  {'p_intent':>8}  "
        f"{'q_good':>8}  {'q_score':>8}  {'ms':>8}  REASON"
    )
    print("=" * 92)

    try:
        for raw_window, norm_window, t_window in data_source:
            result = infer_window(norm_window, intent_model, quality_model, restorer_model, device)

            decision          = result["decision"]
            reason            = result["reason"]
            p_intent          = result["p_intent"]
            q_prob_good       = result["q_prob_good"]
            q_score_pred      = result["q_score_pred"]
            latency_ms        = result["latency_ms"]
            reconstructed_win = result["reconstructed_window"]

            num_valid += 1
            decision_counts[decision] += 1
            latencies.append(latency_ms)

            print(
                f"{num_valid:>5}  {decision:<8}  {p_intent:>8.3f}  "
                f"{q_prob_good:>8.3f}  {q_score_pred:>8.3f}  "
                f"{latency_ms:>8.2f}  {reason}"
            )

            if ser is not None:
                send_to_arduino(ser, decision)

            if log_writer is not None:
                log_writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    num_valid,
                    decision,
                    reason,
                    f"{p_intent:.6f}",
                    f"{q_prob_good:.6f}",
                    f"{q_score_pred:.6f}",
                    f"{latency_ms:.6f}",
                ])

            if reconstructed_writer is not None:
                reconstructed_writer.writerow(
                    [num_valid, decision, reason] +
                    [float(v) for v in reconstructed_win]
                )

            if len(pdf_examples) < args.pdf_max_examples:
                pdf_examples.append({
                    "idx": num_valid,
                    "decision": decision,
                    "reason": reason,
                    "p_intent": p_intent,
                    "q_prob_good": q_prob_good,
                    "q_score_pred": q_score_pred,
                    "latency_ms": latency_ms,
                    "raw_window": raw_window.copy(),
                    "norm_window": norm_window.copy(),
                    "reconstructed_window": reconstructed_win.copy(),
                    "time_window": None if t_window is None else np.asarray(t_window).copy(),
                })

            if num_valid % REPORT_EVERY == 0:
                print("-" * 92)
                print(
                    f"Procesadas={num_valid} | "
                    f"OK={decision_counts['OK']} | "
                    f"RESTORE={decision_counts['RESTORE']} | "
                    f"REJECT={decision_counts['REJECT']} | "
                    f"lat_media={np.mean(latencies):.3f} ms"
                )
                print("-" * 92)

            if args.delay > 0:
                time.sleep(args.delay)

    finally:
        if log_file is not None:
            log_file.close()
        if reconstructed_file is not None:
            reconstructed_file.close()
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass

    summary = {
        "num_valid": num_valid,
        "decision_counts": decision_counts,
        "latencies": latencies,
    }

    save_pdf_report(args.pdf_path, pdf_examples, summary)

    print("\n" + "=" * 92)
    print("RESUMEN FINAL")
    print(f"Ventanas procesadas : {num_valid}")
    print(f"OK                 : {decision_counts['OK']}")
    print(f"RESTORE            : {decision_counts['RESTORE']}")
    print(f"REJECT             : {decision_counts['REJECT']}")
    if latencies:
        p = percentiles_ms(latencies)
        print(f"Latencia media (ms): {np.mean(latencies):.4f}")
        print(f"p50={p['p50']:.4f} | p90={p['p90']:.4f} | p95={p['p95']:.4f}")
    print(f"PDF                : {args.pdf_path}")
    if args.save_reconstructed:
        print(f"Reconstrucciones   : {args.reconstructed_csv}")
    print("=" * 92)


# =========================================================
# FLUJO WEEKLY
# =========================================================
def run_weekly_mode(args, intent_model, quality_model, restorer_model, device):
    print("=" * 70)
    print("  Análisis Comparativo sEMG — Rehabilitación de Prótesis Q1–Q4")
    print(f"  Dispositivo: {device}")
    print("=" * 70)

    all_weeks = []
    for wi, (name, csv_path) in enumerate(zip(WEEKS, args.weekly_csvs)):
        print(f"[{name}]")
        wdata = process_week(
            wi, csv_path,
            intent_model, quality_model, restorer_model,
            device, args.stride, args.max_windows,
        )
        dc  = wdata["dec_counts"]
        tot = sum(dc.values()) or 1
        print(
            f"  OK={dc['OK']} ({100*dc['OK']/tot:.1f}%)  "
            f"RESTORE={dc['RESTORE']} ({100*dc['RESTORE']/tot:.1f}%)  "
            f"REJECT={dc['REJECT']} ({100*dc['REJECT']/tot:.1f}%)"
        )
        print(
            f"  q_score medio = {np.mean(wdata['q_scores']) if wdata['q_scores'] else 0.0:.4f}  "
            f"|  latencia media = {np.mean(wdata['latencies']) if wdata['latencies'] else 0.0:.3f} ms\n"
        )
        all_weeks.append(wdata)

    if not args.no_pdf:
        generate_weekly_pdf(all_weeks, args.weekly_pdf_path)
    if not args.no_html:
        generate_html(all_weeks, args.html_path)

    print("\n" + "=" * 70)
    print("  SALIDAS GENERADAS")
    if not args.no_pdf:
        print(f"  PDF  → {args.weekly_pdf_path}")
    if not args.no_html:
        print(f"  HTML → {args.html_path}")
    print("=" * 70)


# =========================================================
# MAIN
# =========================================================
def main():
    args = parse_args()

    if args.mode is None:
        print("¿Fuente de datos?")
        print("  [1] Archivo CSV")
        print("  [2] Simulación")
        print("  [3] Análisis semanal Q1–Q4")
        choice = input("Elige 1, 2 o 3: ").strip()
        if choice == "1":
            args.mode = "csv"
        elif choice == "2":
            args.mode = "sim"
        else:
            args.mode = "weekly"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDispositivo: {device}")

    models_dir = find_models_dir(args.models_dir)
    intent_model, quality_model, restorer_model = load_models(models_dir, device)
    warmup(intent_model, quality_model, restorer_model, device)

    if args.mode in ("csv", "sim"):
        run_operational_mode(args, intent_model, quality_model, restorer_model, device)
    else:
        run_weekly_mode(args, intent_model, quality_model, restorer_model, device)


if __name__ == "__main__":
    main()