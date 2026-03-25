import time
import serial
import numpy as np

SERIAL_PORT = "/dev/cu.usbmodemB08184983A842"
BAUD = 115200
WIN = 400

ser = serial.Serial(SERIAL_PORT, BAUD, timeout=1)
time.sleep(2)

rng = np.random.default_rng(123)

for k in range(20):
    t = np.linspace(0, 1, WIN)

    if k % 3 == 0:
        sig = 0.02 * rng.normal(size=WIN)
    else:
        env = np.exp(-0.5 * ((t - 0.5) / 0.12)**2)
        carrier = np.sin(2*np.pi*35*t) + 0.5*np.sin(2*np.pi*70*t)
        sig = 1.8 * env * carrier + 0.2 * rng.normal(size=WIN)

    line = ",".join(f"{x:.6f}" for x in sig) + "\n"
    ser.write(line.encode("utf-8"))

    resp = ser.readline().decode("utf-8", errors="ignore").strip()
    print(f"[{k}] respuesta:", resp)

    time.sleep(0.3)

ser.close()