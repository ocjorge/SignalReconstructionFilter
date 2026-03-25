import time
import serial

SERIAL_PORT = "/dev/cu.usbmodemB08184983A842"
BAUD = 9600

ser = serial.Serial(SERIAL_PORT, BAUD, timeout=1)
time.sleep(2)

msgs = ["OK\n", "REJECT\n", "RESTORE\n", "OK\n", "RESTORE\n"]

for msg in msgs:
    print("Enviando:", msg.strip())
    ser.write(msg.encode("utf-8"))
    time.sleep(2)

ser.close