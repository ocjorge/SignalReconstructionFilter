"""
main.py — Inferencia de señal real/simulada + reconstrucción + display en Arduino UNO R4 WiFi

Flujo:
  [CSV real o simulación] → [Python + modelos] → [decisión + reconstrucción] → [Arduino LED matriz]

Uso:
  python main.py --mode csv --csv_path signal.csv
  python main.py --mode sim
  python main.py --mode csv --csv_path signal.csv --no_arduino

Notas:
- Para CSV real tipo tiempo,señal, usa SOLO la segunda columna.
- Los modelos deben estar en ./models:
    intent_binary_best.pt
    quality_judge_best.pt
    restorer_best.pt
"""

import os
import time
import csv
import glob
import argparse
from typing import Iterator, Tuple, Optional

import numpy as np
import serial
import serial.tools.list_ports
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# =========================================================
# ARGUMENTOS
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Inferencia de señal con reconstrucción y display Arduino"
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
        help="Número de ventanas simuladas a generar (default: 50)"
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
        help="Segundos de pausa entre ventanas (default: 0.5)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=200,
        help="Stride para particionar señal continua (default: 200)"
    )
    parser.add_argument(
        "--max_windows",
        type=int,
        default=200,
        help="Máximo de ventanas a procesar en modo CSV (default: 200)"
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        help="Guardar gráficas de ejemplos"
    )
    parser.add_argument(
        "--plots_dir",
        type=str,
        default="plots_out",
        help="Carpeta para guardar gráficas"
    )
    parser.add_argument(
        "--save_reconstructed",
        action="store_true",
        help="Guardar reconstrucciones en CSV"
    )
    parser.add_argument(
        "--reconstructed_csv",
        type=str,
        default="reconstructed_windows.csv",
        help="Archivo CSV para guardar ventanas reconstruidas"
    )
    return parser.parse_args()


# =========================================================
# CONFIG
# =========================================================
SERIAL_BAUD = 115200
SERIAL_TIMEOUT = 2.0

THR_INTENT = 0.50
THR_QUALITY = 0.50

WIN = 400
SAVE_LOG = True
LOG_PATH = "stream_log.csv"
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
        return self.decoder(z)


# =========================================================
# UTILIDADES
# =========================================================
def robust_norm_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-6
    z = (x - med) / (1.4826 * mad)
    return np.clip(z, -10, 10).astype(np.float32)


def percentiles_ms(values):
    if not values:
        return {"p50": float("nan"), "p90": float("nan"), "p95": float("nan")}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def find_models_dir() -> str:
    # local primero
    local = os.path.abspath("./models")
    if os.path.exists(os.path.join(local, "intent_binary_best.pt")):
        if (
            os.path.exists(os.path.join(local, "quality_judge_best.pt"))
            and os.path.exists(os.path.join(local, "restorer_best.pt"))
        ):
            return local

    # kaggle si existiera
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


def save_plot_window(idx: int,
                     raw_window: np.ndarray,
                     used_input_window: np.ndarray,
                     reconstructed_window: np.ndarray,
                     decision: str,
                     reason: str,
                     out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(12, 4))
    plt.plot(raw_window, label="raw_window", alpha=0.7)
    plt.plot(used_input_window, label="input_normalized", alpha=0.9)
    plt.plot(reconstructed_window, label="reconstructed_or_passthrough", alpha=0.9)
    plt.title(f"idx={idx} | {decision} | {reason}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"window_{idx:04d}_{decision}.png"), dpi=150)
    plt.close()


# =========================================================
# FUENTES DE DATOS
# =========================================================
def source_csv(csv_path: str,
               win: int = 400,
               stride: int = 200,
               max_windows: int = 200) -> Iterator[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """
    Lee CSV en tres modos:
      A) Cada fila es una ventana completa de 400 valores
      B) Una sola columna continua
      C) Dos columnas: tiempo,señal  -> usa SOLO la segunda columna

    Y retorna:
      raw_window, normalized_window, time_window_or_None
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
                continue
            if vals:
                rows.append(vals)

    if not rows:
        raise ValueError("El CSV no tiene filas válidas.")

    first_len = len(rows[0])

    # ---- Modo A: cada fila ya es una ventana completa
    if first_len == win:
        print(f"CSV modo A: {len(rows)} ventanas de {win} valores cada una.")
        count = 0
        for row in rows:
            raw = np.array(row, dtype=np.float32)
            norm = robust_norm_1d(raw)
            yield raw, norm, None
            count += 1
            if count >= max_windows:
                break
        return

    # ---- Modo C: dos columnas, usar segunda como señal
    if first_len >= 2:
        times = np.array([row[0] for row in rows], dtype=np.float32)
        signal = np.array([row[1] for row in rows], dtype=np.float32)

        n_windows = 1 + max(0, (len(signal) - win) // stride)
        n_windows = min(n_windows, max_windows)
        print(f"CSV modo C (tiempo,señal): {len(signal)} muestras → {n_windows} ventanas (win={win}, stride={stride}).")

        for i in range(n_windows):
            start = i * stride
            end = start + win
            raw = signal[start:end]
            t_win = times[start:end]
            if len(raw) < win:
                break
            norm = robust_norm_1d(raw)
            yield raw.astype(np.float32), norm.astype(np.float32), t_win.astype(np.float32)
        return

    # ---- Modo B: una sola columna continua
    all_vals = np.array([row[0] for row in rows], dtype=np.float32)
    n_windows = 1 + max(0, (len(all_vals) - win) // stride)
    n_windows = min(n_windows, max_windows)
    print(f"CSV modo B: {len(all_vals)} muestras → {n_windows} ventanas (win={win}, stride={stride}).")

    for i in range(n_windows):
        start = i * stride
        end = start + win
        raw = all_vals[start:end]
        if len(raw) < win:
            break
        norm = robust_norm_1d(raw)
        yield raw.astype(np.float32), norm.astype(np.float32), None


def source_sim(n_windows: int = 50, win: int = 400) -> Iterator[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """
    Genera ventanas simuladas con casos:
      - casi reposo
      - señal útil limpia
      - señal útil ruidosa
      - señal con dropout
    """
    print(f"Simulación: generando {n_windows} ventanas de {win} valores.")
    rng = np.random.default_rng(42)
    t = np.linspace(0, 1, win, dtype=np.float32)

    for i in range(n_windows):
        mode = rng.choice(["rest", "clean", "noisy", "dropout"])

        if mode == "rest":
            sig = 0.02 * rng.standard_normal(win).astype(np.float32)

        else:
            env = np.exp(-0.5 * ((t - 0.5) / 0.12) ** 2).astype(np.float32)
            carrier = (np.sin(2 * np.pi * 35 * t) + 0.5 * np.sin(2 * np.pi * 70 * t)).astype(np.float32)
            sig = (1.8 * env * carrier).astype(np.float32)

            if mode in ["noisy", "dropout"]:
                sig += rng.normal(0, 0.25, size=win).astype(np.float32)

            if mode == "dropout":
                start = rng.integers(80, 220)
                sig[start:start + 40] = 0.0

        raw = sig.astype(np.float32)
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
        time.sleep(2)
        ser.reset_input_buffer()
        print(f"Arduino conectado en {port} @ {SERIAL_BAUD}\n")
        return ser
    except serial.SerialException as e:
        print(f"ERROR al abrir {port}: {e}")
        print("Verifica que el Arduino esté conectado y que ningún otro programa use el puerto.")
        raise


def send_to_arduino(ser: serial.Serial, decision: str):
    msg = decision + "\n"
    ser.write(msg.encode("utf-8"))
    ack = ser.readline().decode("utf-8", errors="ignore").strip()
    if ack:
        print(f"  Arduino ACK: {ack}")
    else:
        print("  Arduino ACK: (sin respuesta)")


# =========================================================
# CARGA DE MODELOS
# =========================================================
def load_models(device: str):
    models_dir = find_models_dir()
    print(f"Modelos en: {models_dir}")

    intent = IntentTCN().to(device)
    intent.load_state_dict(torch.load(
        os.path.join(models_dir, "intent_binary_best.pt"), map_location=device))
    intent.eval()

    quality = QualityNet().to(device)
    quality.load_state_dict(torch.load(
        os.path.join(models_dir, "quality_judge_best.pt"), map_location=device))
    quality.eval()

    restorer = ConvAE1D().to(device)
    restorer.load_state_dict(torch.load(
        os.path.join(models_dir, "restorer_best.pt"), map_location=device))
    restorer.eval()

    print("Modelos cargados.\n")
    return intent, quality, restorer


# =========================================================
# INFERENCIA
# =========================================================
@torch.no_grad()
def infer_window(window_norm: np.ndarray,
                 intent_model,
                 quality_model,
                 restorer_model,
                 device):
    """
    window_norm: ventana ya normalizada (400,)
    Devuelve también la ventana de salida:
      - passthrough si OK/REJECT
      - reconstruida si RESTORE
    """
    xb = torch.tensor(window_norm[None, None, :], dtype=torch.float32, device=device)

    t0 = time.perf_counter()

    p_intent = float(torch.sigmoid(intent_model(xb)).cpu().reshape(-1)[0])

    q_score_t, q_logit = quality_model(xb)
    q_score_pred = float(q_score_t.cpu().reshape(-1)[0])
    q_prob_good = float(torch.sigmoid(q_logit).cpu().reshape(-1)[0])

    if p_intent < THR_INTENT:
        decision = "REJECT"
        reason = "no_intent_detected"
        reconstructed = xb.cpu().numpy()[0, 0].astype(np.float32)

    elif q_prob_good >= THR_QUALITY:
        decision = "OK"
        reason = "signal_quality_good_enough"
        reconstructed = xb.cpu().numpy()[0, 0].astype(np.float32)

    else:
        decision = "RESTORE"
        reason = "intent_detected_quality_bad"
        reconstructed = restorer_model(xb).cpu().numpy()[0, 0].astype(np.float32)

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "decision": decision,
        "reason": reason,
        "p_intent": p_intent,
        "q_prob_good": q_prob_good,
        "q_score_pred": q_score_pred,
        "latency_ms": latency_ms,
        "reconstructed_window": reconstructed,
    }


def warmup(intent_model, quality_model, restorer_model, device):
    dummy = np.zeros(WIN, dtype=np.float32)
    for _ in range(WARMUP_ITERS):
        _ = infer_window(dummy, intent_model, quality_model, restorer_model, device)
    print(f"Warmup completado ({WARMUP_ITERS} iteraciones).\n")


# =========================================================
# LOG
# =========================================================
def init_log(path: str):
    if not SAVE_LOG:
        return None, None
    f = open(path, "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow([
        "timestamp", "idx", "decision", "reason",
        "p_intent", "q_prob_good", "q_score_pred", "latency_ms"
    ])
    return f, writer


def init_reconstructed_csv(path: str):
    f = open(path, "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    header = ["idx", "decision", "reason"] + [f"v{i}" for i in range(WIN)]
    writer.writerow(header)
    return f, writer


# =========================================================
# MAIN
# =========================================================
def main():
    args = parse_args()

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
            max_windows=args.max_windows
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

    print("=" * 92)
    print(f"{'#':>5}  {'DECISION':<8}  {'p_intent':>8}  {'q_good':>8}  {'q_score':>8}  {'ms':>8}  REASON")
    print("=" * 92)

    try:
        for raw_window, norm_window, t_window in data_source:
            result = infer_window(norm_window, intent_model, quality_model, restorer_model, device)

            decision = result["decision"]
            reason = result["reason"]
            p_intent = result["p_intent"]
            q_prob_good = result["q_prob_good"]
            q_score_pred = result["q_score_pred"]
            latency_ms = result["latency_ms"]
            reconstructed_window = result["reconstructed_window"]

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
                    f"{time.time():.3f}",
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
                    [num_valid, decision, reason] + reconstructed_window.tolist()
                )

            if args.save_plots and (num_valid <= 10 or decision == "RESTORE"):
                save_plot_window(
                    idx=num_valid,
                    raw_window=raw_window,
                    used_input_window=norm_window,
                    reconstructed_window=reconstructed_window,
                    decision=decision,
                    reason=reason,
                    out_dir=args.plots_dir
                )

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

    if log_file is not None:
        log_file.close()
        print(f"\nLog guardado en: {LOG_PATH}")

    if reconstructed_file is not None:
        reconstructed_file.close()
        print(f"Reconstrucciones guardadas en: {args.reconstructed_csv}")

    if args.save_plots:
        print(f"Gráficas guardadas en: {args.plots_dir}")

    if ser is not None:
        ser.close()
        print("Puerto serial cerrado.")


if __name__ == "__main__":
    main()