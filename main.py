import os
import time
import csv
import glob
import numpy as np
import serial
import torch
import torch.nn as nn


# =========================================================
# CONFIG
# =========================================================
SERIAL_PORT = "/dev/cu.usbmodemB08184983A842"      # Cambiar esto en Windows, por ejemplo COM3 o COM5
SERIAL_BAUD = 115200
SERIAL_TIMEOUT = 1.0

THR_INTENT = 0.50
THR_QUALITY = 0.50

WIN = 400
SAVE_LOG = True
LOG_PATH = "stream_log.csv"

REPORT_EVERY = 25            # imprimir resumen cada N ventanas válidas
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
        y = self.decoder(z)
        return y


# =========================================================
# UTILIDADES
# =========================================================
def robust_norm_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-6
    z = (x - med) / (1.4826 * mad)
    return np.clip(z, -10, 10).astype(np.float32)


def find_models_dir() -> str:
    candidate_dirs = []
    for p in glob.glob("/kaggle/input/**/intent_binary_best.pt", recursive=True):
        model_dir = os.path.dirname(p)
        q_path = os.path.join(model_dir, "quality_judge_best.pt")
        r_path = os.path.join(model_dir, "restorer_best.pt")
        if os.path.exists(q_path) and os.path.exists(r_path):
            candidate_dirs.append(model_dir)

    if candidate_dirs:
        return candidate_dirs[0]

    # ruta manual alternativa fuera de Kaggle
    local_candidate = os.path.abspath("./models")
    if os.path.exists(os.path.join(local_candidate, "intent_binary_best.pt")):
        return local_candidate

    raise FileNotFoundError(
        "No encontré directorio de modelos válido. "
        "Coloca los .pt en /kaggle/input/... o en ./models/"
    )


def parse_window_line(line: str, win: int = 400):
    """
    Espera una línea tipo:
    v1,v2,v3,...,v400
    """
    try:
        vals = [float(v) for v in line.strip().split(",") if v.strip() != ""]
    except ValueError:
        return None

    if len(vals) != win:
        return None

    arr = np.array(vals, dtype=np.float32)
    return robust_norm_1d(arr)


def percentiles_ms(values):
    if len(values) == 0:
        return {"p50": np.nan, "p90": np.nan, "p95": np.nan}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


# =========================================================
# CARGA MODELOS
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

MODELS_DIR = find_models_dir()
print("MODELS_DIR:", MODELS_DIR)

intent_model = IntentTCN().to(device)
intent_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "intent_binary_best.pt"), map_location=device))
intent_model.eval()

quality_model = QualityNet().to(device)
quality_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "quality_judge_best.pt"), map_location=device))
quality_model.eval()

restorer_model = ConvAE1D().to(device)
restorer_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "restorer_best.pt"), map_location=device))
restorer_model.eval()

print("Modelos cargados correctamente.")


# =========================================================
# INFERENCIA
# =========================================================
@torch.no_grad()
def infer_window(window_1d: np.ndarray):
    """
    window_1d: np.array shape (400,)
    Devuelve decisión, scores y latencia.
    """
    xb = torch.tensor(window_1d[None, None, :], dtype=torch.float32, device=device)

    t0 = time.perf_counter()

    # intent
    intent_logits = intent_model(xb)
    p_intent = float(torch.sigmoid(intent_logits).cpu().numpy().reshape(-1)[0])

    # quality
    q_score_pred, q_logit = quality_model(xb)
    q_score_pred = float(q_score_pred.cpu().numpy().reshape(-1)[0])
    q_prob_good = float(torch.sigmoid(q_logit).cpu().numpy().reshape(-1)[0])

    # decisión
    if p_intent < THR_INTENT:
        decision = "REJECT"
        reason = "no_intent_detected"
    elif q_prob_good >= THR_QUALITY:
        decision = "OK"
        reason = "signal_quality_good_enough"
    else:
        decision = "RESTORE"
        reason = "intent_detected_and_quality_bad"
        _ = restorer_model(xb)

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "decision": decision,
        "reason": reason,
        "p_intent": p_intent,
        "q_prob_good": q_prob_good,
        "q_score_pred": q_score_pred,
        "latency_ms": latency_ms,
    }


def warmup():
    dummy = np.zeros(WIN, dtype=np.float32)
    for _ in range(WARMUP_ITERS):
        _ = infer_window(dummy)
    print(f"Warmup completado con {WARMUP_ITERS} iteraciones.")


# =========================================================
# LOG
# =========================================================
def init_log(path: str):
    if not SAVE_LOG:
        return None

    f = open(path, "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow([
        "timestamp",
        "decision",
        "reason",
        "p_intent",
        "q_prob_good",
        "q_score_pred",
        "latency_ms",
    ])
    return f, writer


# =========================================================
# MAIN
# =========================================================
def main():
    warmup()

    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=SERIAL_TIMEOUT)
    time.sleep(2)
    print(f"Conectado a {SERIAL_PORT} @ {SERIAL_BAUD}")

    log_handle = init_log(LOG_PATH)
    log_file, log_writer = (log_handle if log_handle is not None else (None, None))

    latencies = []
    num_valid = 0
    num_invalid = 0

    decision_counts = {
        "OK": 0,
        "REJECT": 0,
        "RESTORE": 0,
    }

    try:
        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            window = parse_window_line(line, win=WIN)
            if window is None:
                num_invalid += 1
                err_msg = "ERR,BAD_WINDOW\n"
                ser.write(err_msg.encode("utf-8"))
                continue

            result = infer_window(window)

            decision = result["decision"]
            reason = result["reason"]
            p_intent = result["p_intent"]
            q_prob_good = result["q_prob_good"]
            q_score_pred = result["q_score_pred"]
            latency_ms = result["latency_ms"]

            # Respuesta serial compacta
            # Formato:
            # DECISION,p_intent,q_prob_good,q_score_pred,latency_ms
            msg = f"{decision},{p_intent:.3f},{q_prob_good:.3f},{q_score_pred:.3f},{latency_ms:.3f}\n"
            ser.write(msg.encode("utf-8"))

            num_valid += 1
            decision_counts[decision] += 1
            latencies.append(latency_ms)

            if log_writer is not None:
                log_writer.writerow([
                    time.time(),
                    decision,
                    reason,
                    f"{p_intent:.6f}",
                    f"{q_prob_good:.6f}",
                    f"{q_score_pred:.6f}",
                    f"{latency_ms:.6f}",
                ])

            print(
                f"[{num_valid}] decision={decision} | reason={reason} | "
                f"p_intent={p_intent:.3f} | q_good={q_prob_good:.3f} | "
                f"q_score={q_score_pred:.3f} | latency_ms={latency_ms:.3f}"
            )

            if num_valid % REPORT_EVERY == 0:
                p = percentiles_ms(latencies[1:] if len(latencies) > 1 else latencies)
                print("-" * 72)
                print(f"Ventanas válidas: {num_valid} | inválidas: {num_invalid}")
                print("Conteo decisiones:", decision_counts)
                print(
                    f"Latencia ms -> mean={float(np.mean(latencies)):.3f}, "
                    f"median={float(np.median(latencies)):.3f}, "
                    f"p50={p['p50']:.3f}, p90={p['p90']:.3f}, p95={p['p95']:.3f}"
                )
                print("-" * 72)

    except KeyboardInterrupt:
        print("\nFinalizado por usuario.")

    finally:
        if log_file is not None:
            log_file.close()
            print(f"Log guardado en: {LOG_PATH}")
        ser.close()
        print("Puerto serial cerrado.")


if __name__ == "__main__":
    main()
