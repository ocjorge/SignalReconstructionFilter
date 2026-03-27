"""
main.py — Inferencia de señal real/simulada + reconstrucción + reporte PDF + display en Arduino UNO R4 WiFi

Flujo:
  [CSV real o simulación] → [Python + modelos] → [decisión + reconstrucción] → [Arduino LED matriz]
                                                                              → [PDF comparativo]

Uso:
  python main.py --mode csv --csv_path signal.csv
  python main.py --mode sim
  python main.py --mode csv --csv_path signal.csv --no_arduino
  python main.py --mode csv --csv_path signal.csv --save_reconstructed --pdf_path reporte.pdf

Notas:
- Para CSV real tipo tiempo,señal, usa SOLO la segunda columna.
- Los modelos deben estar en ./models:
    intent_binary_best.pt
    quality_judge_best.pt
    restorer_best.pt
- El PDF se genera siempre al finalizar (usa --pdf_path para cambiar el nombre).
"""

import os
import time
import csv
import glob
import argparse
from typing import Iterator, Tuple, Optional, List

import numpy as np
import serial
import serial.tools.list_ports
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")           # sin GUI; funciona en servidores y entornos sin pantalla
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# =========================================================
# ARGUMENTOS
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Inferencia de señal con reconstrucción, PDF y display Arduino"
    )
    parser.add_argument(
        "--mode",
        choices=["csv", "sim"],
        default=None,
        help="Fuente de datos: 'csv' para archivo, 'sim' para simulación"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="signal.csv",
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
        help="CSV de salida para ventanas reconstruidas (default: reconstructed_windows.csv)"
    )
    parser.add_argument(
        "--pdf_path",
        type=str,
        default="reconstruction_report.pdf",
        help="Ruta del PDF de salida (default: reconstruction_report.pdf)"
    )
    parser.add_argument(
        "--pdf_max_examples",
        type=int,
        default=30,
        help="Número máximo de ventanas incluidas en el PDF (default: 30)"
    )
    return parser.parse_args()


# =========================================================
# CONFIG
# =========================================================
SERIAL_BAUD    = 115200
SERIAL_TIMEOUT = 3.0       # debe ser > animateRestore (~360 ms) + latencia modelo + margen
SERIAL_BOOT_WAIT = 3.5     # el UNO R4 WiFi puede tardar más que el UNO clásico en reiniciar

THR_INTENT  = 0.50
THR_QUALITY = 0.50

WIN          = 400
SAVE_LOG     = True
LOG_PATH     = "stream_log.csv"
REPORT_EVERY = 25
WARMUP_ITERS = 3


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


def find_models_dir() -> str:
    # Buscar primero en ./models local
    local = os.path.abspath("./models")
    if (
        os.path.exists(os.path.join(local, "intent_binary_best.pt"))
        and os.path.exists(os.path.join(local, "quality_judge_best.pt"))
        and os.path.exists(os.path.join(local, "restorer_best.pt"))
    ):
        return local

    # Fallback: entorno Kaggle
    for p in glob.glob("/kaggle/input/**/intent_binary_best.pt", recursive=True):
        model_dir = os.path.dirname(p)
        if (
            os.path.exists(os.path.join(model_dir, "quality_judge_best.pt"))
            and os.path.exists(os.path.join(model_dir, "restorer_best.pt"))
        ):
            return model_dir

    raise FileNotFoundError(
        "No encontré los modelos.\n"
        "Coloca intent_binary_best.pt, quality_judge_best.pt y restorer_best.pt en ./models/"
    )


# =========================================================
# REPORTE PDF
# =========================================================
DECISION_COLORS = {
    "OK":      "#2ecc71",   # verde
    "RESTORE": "#e67e22",   # naranja
    "REJECT":  "#e74c3c",   # rojo
}

def save_pdf_report(pdf_path: str, examples: List[dict], summary: dict):
    """
    Genera un PDF con:
      - Página de portada / resumen estadístico
      - Una página por ventana con:
          · Fila superior: señal original (raw) con eje de tiempo si está disponible
          · Fila inferior: señal normalizada (entrada al modelo) vs señal reconstruida/passthrough
    """
    print(f"\nGenerando PDF en: {pdf_path} ...")

    with PdfPages(pdf_path) as pdf:
        # ── Portada ──────────────────────────────────────────────────
        fig = plt.figure(figsize=(11.69, 8.27))   # A4 horizontal
        fig.patch.set_facecolor("#1a1a2e")
        fig.suptitle(
            "Reporte de Reconstrucción de Señales",
            fontsize=22, fontweight="bold", color="white", y=0.93
        )
        plt.axis("off")

        dc = summary["decision_counts"]
        total = summary["num_valid"] or 1

        # Estadísticas de texto
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
            fig.text(0.08, y, line, fontsize=13, color="white",
                     fontfamily="monospace")
            y -= 0.055

        # Mini gráfico de torta con distribución de decisiones
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
            ax_pie.set_title("Distribución de decisiones",
                             color="white", fontsize=12, pad=10)

        pdf.savefig(fig, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        # ── Una página por ventana ────────────────────────────────────
        for ex in examples:
            decision = ex["decision"]
            color    = DECISION_COLORS.get(decision, "#888888")

            fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27))
            fig.patch.set_facecolor("#1a1a2e")
            for ax in axes:
                ax.set_facecolor("#16213e")
                ax.tick_params(colors="white")
                ax.xaxis.label.set_color("white")
                ax.yaxis.label.set_color("white")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#444466")

            # Título
            title = (
                f"Ventana #{ex['idx']}   |   Decisión: {decision}   |   Razón: {ex['reason']}\n"
                f"p_intent={ex['p_intent']:.3f}   q_good={ex['q_prob_good']:.3f}   "
                f"q_score={ex['q_score_pred']:.3f}   latency={ex['latency_ms']:.2f} ms"
            )
            fig.suptitle(title, fontsize=11, color=color, fontweight="bold", y=0.98)

            # ── Subplot superior: señal RAW ──
            ax0 = axes[0]
            if ex["time_window"] is not None:
                x_axis = ex["time_window"]
                ax0.set_xlabel("Tiempo (s)")
            else:
                x_axis = np.arange(len(ex["raw_window"]))
                ax0.set_xlabel("Muestra")

            ax0.plot(x_axis, ex["raw_window"],
                     color="#74b9ff", linewidth=1.2, label="Señal original (raw)")
            ax0.set_ylabel("Amplitud")
            ax0.set_title("Señal original", color="white")
            ax0.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
            ax0.grid(True, alpha=0.2, color="#555577")

            # ── Subplot inferior: normalizada vs reconstruida ──
            ax1 = axes[1]
            samples = np.arange(WIN)
            ax1.plot(samples, ex["normalized_window"],
                     color="#a29bfe", linewidth=1.2, alpha=0.85,
                     label="Entrada normalizada")
            ax1.plot(samples, ex["reconstructed_window"],
                     color=color, linewidth=1.5, alpha=0.95,
                     label="Reconstruida / passthrough")

            # Rellenar diferencia solo en ventanas RESTORE para destacarla
            if decision == "RESTORE":
                ax1.fill_between(
                    samples,
                    ex["normalized_window"],
                    ex["reconstructed_window"],
                    alpha=0.15, color=color
                )

            ax1.set_xlabel("Muestra")
            ax1.set_ylabel("Amplitud normalizada")
            ax1.set_title("Entrada normalizada vs Salida del modelo", color="white")
            ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
            ax1.grid(True, alpha=0.2, color="#555577")

            plt.tight_layout(rect=[0, 0, 1, 0.94])
            pdf.savefig(fig, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)

    print(f"PDF guardado en: {pdf_path}")


# =========================================================
# FUENTES DE DATOS
# =========================================================
def source_csv(
    csv_path: str,
    win: int = 400,
    stride: int = 200,
    max_windows: int = 200,
) -> Iterator[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """
    Lee CSV en tres modos:
      A) Cada fila es una ventana completa de WIN valores.
      B) Una sola columna continua  → ventanas con stride.
      C) Dos columnas: tiempo,señal → usa SOLO la segunda columna.

    Retorna: (raw_window, normalized_window, time_window_or_None)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No encontré el archivo: {csv_path}")

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                vals = [float(v) for v in row if str(v).strip() != ""]
            except ValueError:
                continue   # saltar encabezados o filas no numéricas
            if vals:
                rows.append(vals)

    if not rows:
        raise ValueError("El CSV no tiene filas válidas.")

    first_len = len(rows[0])

    # ── Modo A ───────────────────────────────────────────────────────
    if first_len == win:
        print(f"CSV modo A: {len(rows)} ventanas de {win} valores cada una.")
        for count, row in enumerate(rows):
            if count >= max_windows:
                break
            raw  = np.array(row, dtype=np.float32)
            norm = robust_norm_1d(raw)
            yield raw, norm, None
        return

    # ── Modo C ───────────────────────────────────────────────────────
    if first_len >= 2:
        times  = np.array([row[0] for row in rows], dtype=np.float32)
        signal = np.array([row[1] for row in rows], dtype=np.float32)
        n_wins = min(1 + max(0, (len(signal) - win) // stride), max_windows)
        print(
            f"CSV modo C (tiempo,señal): {len(signal)} muestras "
            f"→ {n_wins} ventanas (win={win}, stride={stride})."
        )
        for i in range(n_wins):
            start = i * stride
            end   = start + win
            raw   = signal[start:end]
            t_win = times[start:end]
            if len(raw) < win:
                break
            yield raw.astype(np.float32), robust_norm_1d(raw), t_win.astype(np.float32)
        return

    # ── Modo B ───────────────────────────────────────────────────────
    all_vals = np.array([row[0] for row in rows], dtype=np.float32)
    n_wins   = min(1 + max(0, (len(all_vals) - win) // stride), max_windows)
    print(
        f"CSV modo B: {len(all_vals)} muestras "
        f"→ {n_wins} ventanas (win={win}, stride={stride})."
    )
    for i in range(n_wins):
        start = i * stride
        end   = start + win
        raw   = all_vals[start:end]
        if len(raw) < win:
            break
        yield raw.astype(np.float32), robust_norm_1d(raw), None


def source_sim(
    n_windows: int = 50,
    win: int = 400,
) -> Iterator[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """
    Ventanas simuladas: reposo, señal limpia, ruidosa y con dropout.
    Retorna: (raw_window, normalized_window, None)
    """
    print(f"Simulación: generando {n_windows} ventanas de {win} valores.")
    rng = np.random.default_rng(42)
    t   = np.linspace(0, 1, win, dtype=np.float32)

    for _ in range(n_windows):
        mode = rng.choice(["rest", "clean", "noisy", "dropout"])

        if mode == "rest":
            sig = 0.02 * rng.standard_normal(win).astype(np.float32)
        else:
            env     = np.exp(-0.5 * ((t - 0.5) / 0.12) ** 2).astype(np.float32)
            carrier = (
                np.sin(2 * np.pi * 35 * t) + 0.5 * np.sin(2 * np.pi * 70 * t)
            ).astype(np.float32)
            sig = (1.8 * env * carrier).astype(np.float32)

            if mode in ("noisy", "dropout"):
                sig += rng.normal(0, 0.25, size=win).astype(np.float32)

            if mode == "dropout":
                s = rng.integers(80, 220)
                sig[s: s + 40] = 0.0

        raw  = sig.astype(np.float32)
        norm = robust_norm_1d(raw)
        yield raw, norm, None


# =========================================================
# PUERTO SERIAL
# =========================================================
def find_serial_port() -> str:
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        raise RuntimeError("No se encontró ningún puerto serial.")

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


def connect_arduino() -> serial.Serial:
    port = find_serial_port()
    try:
        ser = serial.Serial(port, SERIAL_BAUD, timeout=SERIAL_TIMEOUT)
        # El UNO R4 WiFi hace reset al abrir el puerto serial; hay que esperar
        # a que el bootloader termine antes de mandar cualquier dato.
        print(f"  Esperando boot del Arduino ({SERIAL_BOOT_WAIT}s)...")
        time.sleep(SERIAL_BOOT_WAIT)
        ser.reset_input_buffer()   # descartar basura del bootloader
        ser.reset_output_buffer()
        print(f"Arduino conectado en {port} @ {SERIAL_BAUD}\n")
        return ser
    except serial.SerialException as e:
        print(f"ERROR al abrir {port}: {e}")
        print("Verifica que el Arduino esté conectado y que ningún otro programa use el puerto.")
        raise


def send_to_arduino(ser: serial.Serial, decision: str):
    """
    Envía la decisión al Arduino y espera el ACK.

    Por qué limpiamos el buffer antes:
      Si una lectura anterior expiró sin recibir ACK, puede haber bytes
      residuales que confundirían el siguiente readline().

    Por qué ajustamos el timeout para RESTORE:
      animateRestore() en el Arduino tarda ~360 ms (4 × 90 ms).
      Con latencia del modelo y overhead serial, el ACK puede llegar
      hasta ~500 ms después; el timeout normal de 3 s cubre esto,
      pero lo hacemos explícito para RESTORE por claridad.
    """
    ser.reset_input_buffer()                        # descartar ACKs rezagados
    ser.write((decision + "\n").encode("utf-8"))
    ser.flush()                                     # asegurar que los bytes salgan

    # Para RESTORE la animación bloquea el Arduino ~360 ms antes de responder
    wait = 1.0 if decision == "RESTORE" else 0.5
    time.sleep(wait)

    ack = ser.readline().decode("utf-8", errors="ignore").strip()
    if ack:
        print(f"  Arduino ACK: {ack}")
    else:
        print(f"  Arduino ACK: (sin respuesta — decisión enviada: {decision})")


# =========================================================
# CARGA DE MODELOS
# =========================================================
def load_models(device: str):
    models_dir = find_models_dir()
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
    """
    window_norm : ventana ya normalizada (WIN,)
    Devuelve dict con decisión, métricas y ventana de salida:
      - OK / REJECT → passthrough (señal normalizada tal cual)
      - RESTORE     → salida del autoencoder
    """
    xb = torch.tensor(window_norm[None, None, :], dtype=torch.float32, device=device)
    t0 = time.perf_counter()

    p_intent = float(torch.sigmoid(intent_model(xb)).cpu().reshape(-1)[0])

    q_score_t, q_logit = quality_model(xb)
    q_score_pred = float(q_score_t.cpu().reshape(-1)[0])
    q_prob_good  = float(torch.sigmoid(q_logit).cpu().reshape(-1)[0])

    if p_intent < THR_INTENT:
        decision     = "REJECT"
        reason       = "no_intent_detected"
        reconstructed = window_norm.copy()

    elif q_prob_good >= THR_QUALITY:
        decision     = "OK"
        reason       = "signal_quality_good_enough"
        reconstructed = window_norm.copy()

    else:
        decision     = "RESTORE"
        reason       = "intent_detected_quality_bad"
        reconstructed = restorer_model(xb).cpu().numpy()[0, 0].astype(np.float32)

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "decision":          decision,
        "reason":            reason,
        "p_intent":          p_intent,
        "q_prob_good":       q_prob_good,
        "q_score_pred":      q_score_pred,
        "latency_ms":        latency_ms,
        "reconstructed_window": reconstructed,
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
    f      = open(path, "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow([
        "timestamp", "idx", "decision", "reason",
        "p_intent", "q_prob_good", "q_score_pred", "latency_ms",
    ])
    return f, writer


def init_reconstructed_csv(path: str):
    f      = open(path, "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(["idx", "decision", "reason"] + [f"v{i}" for i in range(WIN)])
    return f, writer


# =========================================================
# MAIN
# =========================================================
def main():
    args = parse_args()

    # ── Elegir modo interactivamente si no se pasó por argumento ──
    if args.mode is None:
        print("¿Fuente de datos?")
        print("  [1] Archivo CSV")
        print("  [2] Simulación")
        choice = input("Elige 1 o 2: ").strip()
        args.mode = "csv" if choice == "1" else "sim"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDispositivo: {device}")

    intent_model, quality_model, restorer_model = load_models(device)
    warmup(intent_model, quality_model, restorer_model, device)

    # ── Arduino (opcional) ──
    ser = None
    if not args.no_arduino:
        try:
            ser = connect_arduino()
        except Exception:
            print("⚠️  Continuando sin Arduino (solo consola).\n")

    # ── Fuente de datos ──
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

    # ── Logs de salida ──
    log_file, log_writer = init_log(LOG_PATH)

    reconstructed_file   = None
    reconstructed_writer = None
    if args.save_reconstructed:
        reconstructed_file, reconstructed_writer = init_reconstructed_csv(
            args.reconstructed_csv
        )

    # ── Acumuladores ──
    latencies       = []
    num_valid       = 0
    decision_counts = {"OK": 0, "REJECT": 0, "RESTORE": 0}
    pdf_examples: List[dict] = []   # se llenarán para el PDF final

    print("=" * 92)
    print(
        f"{'#':>5}  {'DECISION':<8}  {'p_intent':>8}  "
        f"{'q_good':>8}  {'q_score':>8}  {'ms':>8}  REASON"
    )
    print("=" * 92)

    try:
        for raw_window, norm_window, t_window in data_source:
            result = infer_window(
                norm_window, intent_model, quality_model, restorer_model, device
            )

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

            # ── Enviar al Arduino ──
            if ser is not None:
                send_to_arduino(ser, decision)

            # ── Log CSV principal ──
            if log_writer is not None:
                log_writer.writerow([
                    f"{time.time():.3f}", num_valid, decision, reason,
                    f"{p_intent:.6f}", f"{q_prob_good:.6f}",
                    f"{q_score_pred:.6f}", f"{latency_ms:.6f}",
                ])

            # ── Log reconstrucciones ──
            if reconstructed_writer is not None:
                reconstructed_writer.writerow(
                    [num_valid, decision, reason] + reconstructed_win.tolist()
                )

            # ── Acumular para PDF ──
            # Siempre guardar las primeras N; también guardar todas las RESTORE
            if len(pdf_examples) < args.pdf_max_examples or decision == "RESTORE":
                # Evitar duplicados si ya se añadió por el límite y es RESTORE
                if len(pdf_examples) < args.pdf_max_examples:
                    pdf_examples.append({
                        "idx":                num_valid,
                        "decision":           decision,
                        "reason":             reason,
                        "p_intent":           p_intent,
                        "q_prob_good":        q_prob_good,
                        "q_score_pred":       q_score_pred,
                        "latency_ms":         latency_ms,
                        "raw_window":         raw_window.copy(),
                        "normalized_window":  norm_window.copy(),
                        "reconstructed_window": reconstructed_win.copy(),
                        "time_window":        t_window.copy() if t_window is not None else None,
                    })

            # ── Reporte periódico ──
            if num_valid % REPORT_EVERY == 0:
                p = percentiles_ms(latencies)
                print("-" * 92)
                print(f"  Ventanas procesadas: {num_valid} | Decisiones: {decision_counts}")
                print(
                    f"  Latencia ms — mean={np.mean(latencies):.2f}  "
                    f"p50={p['p50']:.2f}  p90={p['p90']:.2f}  p95={p['p95']:.2f}"
                )
                print("-" * 92)

            time.sleep(args.delay)

    except KeyboardInterrupt:
        print("\n\nInterrumpido por el usuario.")

    # ── Resumen final en consola ──
    print("\n" + "=" * 92)
    print("RESUMEN FINAL")
    print(f"  Ventanas procesadas : {num_valid}")
    print(f"  Decisiones          : {decision_counts}")
    if latencies:
        p = percentiles_ms(latencies)
        print(
            f"  Latencia ms         : mean={np.mean(latencies):.2f}  "
            f"p50={p['p50']:.2f}  p90={p['p90']:.2f}  p95={p['p95']:.2f}"
        )
    print("=" * 92)

    # ── Cerrar logs ──
    if log_file is not None:
        log_file.close()
        print(f"\nLog guardado en: {LOG_PATH}")

    if reconstructed_file is not None:
        reconstructed_file.close()
        print(f"Reconstrucciones guardadas en: {args.reconstructed_csv}")

    # ── Generar PDF ──
    if pdf_examples:
        summary = {
            "num_valid":      num_valid,
            "decision_counts": decision_counts,
            "latencies":      latencies,
            "lat_mean":       float(np.mean(latencies)) if latencies else float("nan"),
            "lat_p50":        percentiles_ms(latencies)["p50"],
            "lat_p90":        percentiles_ms(latencies)["p90"],
            "lat_p95":        percentiles_ms(latencies)["p95"],
        }
        save_pdf_report(args.pdf_path, pdf_examples, summary)
    else:
        print("No hay ejemplos para incluir en el PDF.")

    # ── Cerrar puerto serial ──
    if ser is not None:
        ser.close()
        print("Puerto serial cerrado.")


if __name__ == "__main__":
    main()