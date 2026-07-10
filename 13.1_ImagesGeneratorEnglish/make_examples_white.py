"""
make_examples_white.py
Generates representative example figures with a white background, in English.

Usage:
  python make_examples_white.py --folder ./sMEG --models_dir ./models --out_dir ./figuras_articulo

Generates:
  figura_D_ejemplos_P1_S1.png   -- P1 Week 1 (OK + Reconstructed + Discarded)
  figura_E_comparativa_P1P2.png -- one reconstructed window for P1 and one for P2
"""

import os, csv, time, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from scipy.stats import pearsonr

WIN        = 400
STRIDE     = 200
MAX_WIN    = 300
THR_INTENT = 0.50
THR_QUALITY= 0.50

# ── article colors, white background ──
C_NORM  = "#1f77b4"   # blue — normalized signal
C_RECON = "#d62728"   # red — reconstructed / direct pass
C_OK     = "#2ca02c"
C_REST   = "#ff7f0e"
C_REJ    = "#d62728"

DEC_EN   = {"OK": "Accepted", "RESTORE": "Reconstructed", "REJECT": "Discarded"}
DEC_COL  = {"OK": C_OK, "RESTORE": C_REST, "REJECT": C_REJ}

# ── models ──────────────────────────────────────────
class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=5, dilation=1, drop=0.1):
        super().__init__()
        pad = (k-1)*dilation//2
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
        self.stem = nn.Sequential(nn.Conv1d(1,16,7,padding=3), nn.BatchNorm1d(16), nn.ReLU())
        self.tcn  = nn.Sequential(TCNBlock(16,16,dilation=1,drop=.10),
                                   TCNBlock(16,32,dilation=2,drop=.10),
                                   TCNBlock(32,32,dilation=4,drop=.10))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32,32), nn.ReLU(),
                                   nn.Dropout(.10), nn.Linear(32,1))
    def forward(self, x): return self.head(self.pool(self.tcn(self.stem(x))))

class QualityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1,16,7,padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16,32,5,padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32,64,5,padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.shared = nn.Sequential(nn.Flatten(), nn.Linear(64,32), nn.ReLU(), nn.Dropout(.1))
        self.head_score = nn.Linear(32,1)
        self.head_label = nn.Linear(32,1)
    def forward(self, x):
        z = self.shared(self.features(x))
        return torch.sigmoid(self.head_score(z)), self.head_label(z)

class ConvAE1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1,16,5,stride=2,padding=2), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16,32,5,stride=2,padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32,64,5,stride=2,padding=2), nn.BatchNorm1d(64), nn.ReLU())
        self.bottleneck = nn.Sequential(nn.Conv1d(64,64,3,padding=1), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64,32,4,stride=2,padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.ConvTranspose1d(32,16,4,stride=2,padding=1), nn.BatchNorm1d(16), nn.ReLU(),
            nn.ConvTranspose1d(16,1,4,stride=2,padding=1))
    def forward(self, x): return self.decoder(self.bottleneck(self.encoder(x)))

def load_models(models_dir, device):
    im = IntentTCN().to(device)
    im.load_state_dict(torch.load(os.path.join(models_dir,"intent_binary_best.pt"), map_location=device))
    im.eval()
    qm = QualityNet().to(device)
    qm.load_state_dict(torch.load(os.path.join(models_dir,"quality_judge_best.pt"), map_location=device))
    qm.eval()
    rm = ConvAE1D().to(device)
    rm.load_state_dict(torch.load(os.path.join(models_dir,"restorer_best.pt"), map_location=device))
    rm.eval()
    return im, qm, rm

# ── utilities ───────────────────────────────────────
def robust_norm(x):
    x   = np.asarray(x, dtype=np.float32).reshape(-1)
    med = np.median(x)
    mad = np.median(np.abs(x-med))+1e-6
    return np.clip((x-med)/(1.4826*mad), -10, 10).astype(np.float32)

def load_csv(path):
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
        raise ValueError(f"No data: {path}")
    times  = np.array([r[0] for r in rows], dtype=np.float32)
    signal = np.array([r[1] for r in rows], dtype=np.float32)
    return times, signal

@torch.no_grad()
def infer(norm_w, im, qm, rm, device):
    xb = torch.tensor(norm_w[None,None,:], dtype=torch.float32, device=device)
    t0 = time.perf_counter()
    p  = float(torch.sigmoid(im(xb)).cpu().item())
    qs, ql = qm(xb)
    q_score = float(qs.cpu().item())
    q_good  = float(torch.sigmoid(ql).cpu().item())
    if p < THR_INTENT:
        dec = "REJECT"; recon = norm_w.copy()
    elif q_good >= THR_QUALITY:
        dec = "OK";     recon = norm_w.copy()
    else:
        dec = "RESTORE"; recon = rm(xb).cpu().numpy()[0,0].astype(np.float32)
    lat = (time.perf_counter()-t0)*1000
    mse = float(np.mean((norm_w-recon)**2)) if dec=="RESTORE" else 0.0
    corr = float(pearsonr(norm_w,recon)[0]) if dec=="RESTORE" and np.std(norm_w)>1e-9 else 0.0
    return {"dec":dec,"p":p,"q_score":q_score,"q_good":q_good,
            "lat":lat,"norm":norm_w,"recon":recon,"mse":mse,"corr":corr}

def process_file(path, im, qm, rm, device):
    times, sig = load_csv(path)
    n = len(sig)
    n_wins = min(1+max(0,(n-WIN)//STRIDE), MAX_WIN)
    results = []
    for i in range(n_wins):
        s,e = i*STRIDE, i*STRIDE+WIN
        raw = sig[s:e]
        if len(raw)<WIN: break
        norm = robust_norm(raw)
        res  = infer(norm, im, qm, rm, device)
        res["raw"] = raw
        res["time"] = times[s:e]
        results.append(res)
    return results

def pick_example(results, decision):
    idxs = [i for i,r in enumerate(results) if r["dec"]==decision]
    if not idxs: return None
    return results[idxs[len(idxs)//2]]

# ── Figure D — P1 Week 1 examples ─────────────────
def figura_D(results, participant, week, out_path):
    order = ["OK","RESTORE","REJECT"]
    examples = {d: pick_example(results, d) for d in order}
    n_rows = sum(1 for v in examples.values() if v is not None)

    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 3.8*n_rows))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Representative decision examples — {participant}, {week}",
        fontsize=12, fontweight="bold", y=1.01
    )

    row = 0
    for dec in order:
        ex = examples[dec]
        if ex is None: continue
        axL = axes[row, 0] if n_rows > 1 else axes[0]
        axR = axes[row, 1] if n_rows > 1 else axes[1]

        color_dec = DEC_COL[dec]
        label_dec = DEC_EN[dec]

        # original signal
        axL.plot(ex["time"], ex["raw"], color="#333333", linewidth=1.1)
        axL.set_title(f"{label_dec} — original signal", fontsize=9,
                      fontweight="bold", color=color_dec)
        axL.set_xlabel("Time (s)", fontsize=8)
        axL.set_ylabel("Amplitude", fontsize=8)
        axL.spines["top"].set_visible(False)
        axL.spines["right"].set_visible(False)
        axL.yaxis.grid(True, linestyle="--", alpha=0.35)
        axL.set_axisbelow(True)

        # normalized vs reconstructed / direct pass
        axR.plot(np.arange(WIN), ex["norm"],  color=C_NORM,  linewidth=1.1,
                 label="Normalized", alpha=0.8)
        axR.plot(np.arange(WIN), ex["recon"], color=C_RECON, linewidth=1.3,
                 label="Reconstructed / direct pass")
        axR.set_title(f"{label_dec} — pipeline output", fontsize=9,
                      fontweight="bold", color=color_dec)
        axR.set_xlabel("Sample", fontsize=8)
        axR.set_ylabel("Normalized value", fontsize=8)
        axR.legend(fontsize=7.5, framealpha=0.4)
        axR.spines["top"].set_visible(False)
        axR.spines["right"].set_visible(False)
        axR.yaxis.grid(True, linestyle="--", alpha=0.35)
        axR.set_axisbelow(True)

        # metrics annotation
        info = (f"p_intent={ex['p']:.2f}  "
                f"quality={ex['q_score']:.2f}  "
                f"lat.={ex['lat']:.1f} ms")
        if dec == "RESTORE":
            info += f"  MSE={ex['mse']:.3f}  r={ex['corr']:.3f}"
        fig.text(0.5, axes[row,0].get_position().y0 - 0.03,
                 info, ha="center", fontsize=7.5, color="#555555",
                 transform=fig.transFigure)

        row += 1

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figure D saved: {out_path}")

# ── Figure E — P1 vs P2 reconstructed-window comparison ─
def figura_E(res_p1, res_p2, out_path):
    ex1 = pick_example(res_p1, "RESTORE")
    ex2 = pick_example(res_p2, "RESTORE")

    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Example of a reconstructed window: P1 (Female) vs P2 (Male)",
        fontsize=12, fontweight="bold"
    )

    for row, (ex, label) in enumerate([(ex1,"P1 — Female"), (ex2,"P2 — Male")]):
        if ex is None:
            axes[row,0].axis("off"); axes[row,1].axis("off"); continue

        # original signal
        axes[row,0].plot(ex["time"], ex["raw"], color="#333333", linewidth=1.1)
        axes[row,0].set_title(f"{label} — original signal (Reconstructed)",
                               fontsize=9, fontweight="bold", color=C_REST)
        axes[row,0].set_xlabel("Time (s)", fontsize=8)
        axes[row,0].set_ylabel("Amplitude", fontsize=8)
        axes[row,0].spines["top"].set_visible(False)
        axes[row,0].spines["right"].set_visible(False)
        axes[row,0].yaxis.grid(True, linestyle="--", alpha=0.35)
        axes[row,0].set_axisbelow(True)

        # normalized vs reconstructed
        axes[row,1].plot(np.arange(WIN), ex["norm"],  color=C_NORM,  linewidth=1.1,
                         label="Normalized", alpha=0.8)
        axes[row,1].plot(np.arange(WIN), ex["recon"], color=C_RECON, linewidth=1.3,
                         label="Reconstructed")
        axes[row,1].set_title(f"{label} — autoencoder output",
                               fontsize=9, fontweight="bold", color=C_REST)
        axes[row,1].set_xlabel("Sample", fontsize=8)
        axes[row,1].set_ylabel("Normalized value", fontsize=8)
        axes[row,1].legend(fontsize=7.5, framealpha=0.4)
        axes[row,1].spines["top"].set_visible(False)
        axes[row,1].spines["right"].set_visible(False)
        axes[row,1].yaxis.grid(True, linestyle="--", alpha=0.35)
        axes[row,1].set_axisbelow(True)

        # metrics
        fig.text(0.5, axes[row,0].get_position().y0 - 0.025,
                 f"MSE={ex['mse']:.3f}   r={ex['corr']:.3f}   "
                 f"p_intent={ex['p']:.2f}   quality={ex['q_score']:.2f}   lat.={ex['lat']:.1f} ms",
                 ha="center", fontsize=7.5, color="#555555",
                 transform=fig.transFigure)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figure E saved: {out_path}")

# ── main ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--folder",     default="./sMEG")
    p.add_argument("--models_dir", default="./models")
    p.add_argument("--out_dir",    default="./figuras_articulo")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("Loading models...")
    im, qm, rm = load_models(args.models_dir, device)
    print("Models loaded.\n")

    # P1 (Female) — Week 1
    p1_s1 = os.path.join(args.folder, "emgQ1csv.csv")
    print(f"Processing P1 W1: {p1_s1}")
    res_p1_s1 = process_file(p1_s1, im, qm, rm, device)
    figura_D(res_p1_s1, "P1 (Female)", "Week 1",
             os.path.join(args.out_dir, "figura_D_ejemplos_P1_S1.png"))

    # P2 (Male) — Week 1 (first available)
    p2_s1 = os.path.join(args.folder, "emgP33RcsvAMPUTADA1.csv")
    print(f"Processing P2 W1: {p2_s1}")
    res_p2_s1 = process_file(p2_s1, im, qm, rm, device)

    # Figure E — reconstructed-window comparison
    figura_E(res_p1_s1, res_p2_s1,
             os.path.join(args.out_dir, "figura_E_reconstruida_P1_P2.png"))

    print("\nDone!")

if __name__ == "__main__":
    main()