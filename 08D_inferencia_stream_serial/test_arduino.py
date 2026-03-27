"""
test_arduino.py — Prueba directa de comunicación con el Arduino
Uso: python test_arduino.py
"""
import time
import serial

PORT    = "/dev/tty.usbmodemB08184983A842"
BAUD    = 115200
TIMEOUT = 3.0

print(f"Conectando a {PORT} @ {BAUD}...")
ser = serial.Serial(PORT, BAUD, timeout=TIMEOUT)

print("Esperando boot del Arduino (3.5s)...")
time.sleep(3.5)
ser.reset_input_buffer()
ser.reset_output_buffer()
print("Listo.\n")

for cmd in ["OK", "REJECT", "RESTORE", "OK"]:
    print(f"→ Enviando: {cmd}")
    ser.reset_input_buffer()
    ser.write((cmd + "\n").encode("utf-8"))
    ser.flush()

    wait = 1.0 if cmd == "RESTORE" else 0.5
    time.sleep(wait)

    ack = ser.readline().decode("utf-8", errors="ignore").strip()
    print(f"← ACK: '{ack}'" if ack else "← ACK: (sin respuesta)")
    print()
    time.sleep(1.0)

ser.close()
print("Puerto cerrado.")