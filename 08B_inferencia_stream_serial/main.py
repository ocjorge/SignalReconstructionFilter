"""
main.py — Inferencia de señal + display en Arduino UNO WiFi R4

Flujo:
  [Archivo CSV o Simulación] → [Python + modelos] → [Arduino LED matriz]

Uso:
  python main.py --mode csv --csv_path datos.csv
  python main.py --mode sim
  python main.py --mode csv --csv_path datos.csv --no_arduino
"""

import os
import time
import csv
import glob
import argparse
import numpy as np
import serial
import serial.tools.list_ports
import torch
import torch.nn as nn


# =========================================================
# ARGUMENTOS
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Inferencia de señal con display Arduino")
    parser.add_argument(
        "--mode",
        choices=["csv", "sim"],
        default=None,
        help="Fuente de datos: 'csv' para archivo, 'sim' para simulación"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="datos.csv",
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
    return parser.parse_args()


# =========================================================
# CONFIG
# =========================================================
SERIAL_BAUD    = 115200
SERIAL_TIMEOUT = 2.0

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


def find_models_dir() -> str:
    for p in glob.glob("/kaggle/input/**/intent_binary_best.pt", recursive=True):
        model_dir = os.path.dirname(p)
        if (os.path.exists(os.path.join(model_dir, "quality_judge_best.pt")) and
                os.path.exists(os.path.join(model_dir, "restorer_best.pt"))):
            return model_dir

    local = os.path.abspath("./models")
    if os.path.exists(os.path.join(local, "intent_binary_best.pt")):
        return local

    raise FileNotFoundError(
        "No encontré los modelos.\n"
        "Coloca intent_binary_best.pt, quality_judge_best.pt y restorer_best.pt en ./models/"
    )


def percentiles_ms(values):
    if not values:
        return {"p50": float("nan"), "p90": float("nan"), "p95": float("nan")}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


# =========================================================
# FUENTES DE DATOS
# =========================================================
def source_csv(csv_path: str, win: int = 400):
    """
    Genera ventanas desde un CSV.

    Formatos soportados:
      A) Cada fila es una ventana completa (400 columnas).
      B) Una sola columna de valores; se parte en bloques de 400.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No encontré el archivo: {csv_path}")

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = []
        for row in reader:
            try:
                vals = [float(v) for v in row if v.strip() != ""]
            except ValueError:
                continue  # saltar encabezados o filas inválidas
            if vals:
                rows.append(vals)

    if not rows:
        raise ValueError("El CSV no tiene filas válidas.")

    # Detectar formato
    if len(rows[0]) == win:
        # Formato A: cada fila es una ventana
        print(f"CSV modo A: {len(rows)} ventanas de {win} valores cada una.")
        for row in rows:
            yield robust_norm_1d(np.array(row, dtype=np.float32))

    else:
        # Formato B: columna única, partir en bloques
        all_vals = np.array([v for row in rows for v in row], dtype=np.float32)
        n_windows = len(all_vals) // win
        print(f"CSV modo B: {len(all_vals)} valores totales → {n_windows} ventanas de {win}.")
        for i in range(n_windows):
            chunk = all_vals[i * win:(i + 1) * win]
            yield robust_norm_1d(chunk)


def source_sim(n_windows: int = 50, win: int = 400):
    """
    Genera ventanas simuladas mezclando señal limpia y ruidosa aleatoriamente.
    """
    print(f"Simulación: generando {n_windows} ventanas de {win} valores.")
    rng = np.random.default_rng(42)
    t = np.linspace(0, 2 * np.pi, win)

    for i in range(n_windows):
        signal_type = rng.choice(["clean", "noisy", "noise_only"])

        if signal_type == "clean":
            freq = rng.uniform(1, 5)
            sig  = np.sin(freq * t) + 0.1 * rng.standard_normal(win)
        elif signal_type == "noisy":
            freq = rng.uniform(1, 5)
            sig  = np.sin(freq * t) + rng.standard_normal(win)
        else:
            sig = rng.standard_normal(win)

        yield robust_norm_1d(sig.astype(np.float32))


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
        time.sleep(2)   # esperar reset del Arduino
        ser.reset_input_buffer()
        print(f"Arduino conectado en {port} @ {SERIAL_BAUD}\n")
        return ser
    except serial.SerialException as e:
        print(f"ERROR al abrir {port}: {e}")
        print("Verifica que el Arduino esté conectado y ningún otro programa use el puerto.")
        raise


def send_to_arduino(ser: serial.Serial, decision: str):
    """Manda la decisión al Arduino y espera el ACK."""
    msg = decision + "\n"
    ser.write(msg.encode("utf-8"))

    # Leer ACK con timeout
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
def infer_window(window: np.ndarray, intent_model, quality_model, restorer_model, device):
    xb = torch.tensor(window[None, None, :], dtype=torch.float32, device=device)
    t0 = time.perf_counter()

    p_intent = float(torch.sigmoid(intent_model(xb)).cpu().reshape(-1)[0])

    q_score_t, q_logit = quality_model(xb)
    q_score_pred = float(q_score_t.cpu().reshape(-1)[0])
    q_prob_good  = float(torch.sigmoid(q_logit).cpu().reshape(-1)[0])

    if p_intent < THR_INTENT:
        decision = "REJECT"
        reason   = "no_intent_detected"
    elif q_prob_good >= THR_QUALITY:
        decision = "OK"
        reason   = "signal_quality_good_enough"
    else:
        decision = "RESTORE"
        reason   = "intent_detected_quality_bad"
        _ = restorer_model(xb)

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "decision":     decision,
        "reason":       reason,
        "p_intent":     p_intent,
        "q_prob_good":  q_prob_good,
        "q_score_pred": q_score_pred,
        "latency_ms":   latency_ms,
    }


def warmup(intent_model, quality_model, restorer_model, device):
    dummy = np.zeros(WIN, dtype=np.float32)
    for _ in range(WARMUP_ITERS):
        infer_window(dummy, intent_model, quality_model, restorer_model, device)
    print(f"Warmup completado ({WARMUP_ITERS} iteraciones).\n")


# =========================================================
# LOG
# =========================================================
def init_log(path: str):
    if not SAVE_LOG:
        return None, None
    f      = open(path, "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow([
        "timestamp", "decision", "reason",
        "p_intent", "q_prob_good", "q_score_pred", "latency_ms"
    ])
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

    # ── Dispositivo ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDispositivo: {device}")

    # ── Modelos ──
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
        print(f"Leyendo ventanas desde: {args.csv_path}\n")
        data_source = source_csv(args.csv_path, WIN)
    else:
        print(f"Generando {args.sim_windows} ventanas simuladas.\n")
        data_source = source_sim(args.sim_windows, WIN)

    # ── Log ──
    log_file, log_writer = init_log(LOG_PATH)

    # ── Estadísticas ──
    latencies       = []
    num_valid       = 0
    decision_counts = {"OK": 0, "REJECT": 0, "RESTORE": 0}

    print("=" * 72)
    print(f"{'#':>5}  {'DECISION':<8}  {'p_intent':>8}  {'q_good':>6}  {'q_score':>7}  {'ms':>7}  REASON")
    print("=" * 72)

    try:
        for window in data_source:
            result = infer_window(window, intent_model, quality_model, restorer_model, device)

            decision     = result["decision"]
            reason       = result["reason"]
            p_intent     = result["p_intent"]
            q_prob_good  = result["q_prob_good"]
            q_score_pred = result["q_score_pred"]
            latency_ms   = result["latency_ms"]

            num_valid += 1
            decision_counts[decision] += 1
            latencies.append(latency_ms)

            # Consola
            print(
                f"{num_valid:>5}  {decision:<8}  {p_intent:>8.3f}  "
                f"{q_prob_good:>6.3f}  {q_score_pred:>7.3f}  "
                f"{latency_ms:>7.2f}  {reason}"
            )

            # Arduino
            if ser is not None:
                send_to_arduino(ser, decision)

            # Log CSV
            if log_writer is not None:
                log_writer.writerow([
                    f"{time.time():.3f}", decision, reason,
                    f"{p_intent:.6f}", f"{q_prob_good:.6f}",
                    f"{q_score_pred:.6f}", f"{latency_ms:.6f}",
                ])

            # Reporte periódico
            if num_valid % REPORT_EVERY == 0:
                p = percentiles_ms(latencies)
                print("-" * 72)
                print(f"  Ventanas procesadas: {num_valid} | Decisiones: {decision_counts}")
                print(
                    f"  Latencia ms — mean={np.mean(latencies):.2f}  "
                    f"p50={p['p50']:.2f}  p90={p['p90']:.2f}  p95={p['p95']:.2f}"
                )
                print("-" * 72)

            time.sleep(args.delay)

    except KeyboardInterrupt:
        print("\n\nInterrumpido por el usuario.")

    # ── Resumen final ──
    print("\n" + "=" * 72)
    print("RESUMEN FINAL")
    print(f"  Ventanas procesadas : {num_valid}")
    print(f"  Decisiones          : {decision_counts}")
    if latencies:
        p = percentiles_ms(latencies)
        print(
            f"  Latencia ms         : mean={np.mean(latencies):.2f}  "
            f"p50={p['p50']:.2f}  p90={p['p90']:.2f}  p95={p['p95']:.2f}"
        )
    print("=" * 72)

    # ── Cierre ──
    if log_file is not None:
        log_file.close()
        print(f"\nLog guardado en: {LOG_PATH}")

    if ser is not None:
        ser.close()
        print("Puerto serial cerrado.")


if __name__ == "__main__":
    main()