# inference_serial.py
print("Starting Serial Inference...")

import serial
import time
import torch
import numpy as np
from env import norm
from train import DQNAgent  # Your agent definition

# ---------------------- CONFIG ----------------------
SERIAL_PORT = 'COM4'      # Change to your Arduino COM port
BAUD_RATE = 9600
DEVICE = "cpu"            # CPU mode
READ_TIMEOUT = 1          # Serial read timeout (seconds)
SEND_INTERVAL = 2         # Send decision every 2 seconds

# ---------------------- LOAD MODEL ----------------------
agent = DQNAgent(device=DEVICE)
agent.q.load_state_dict(torch.load("dqn_model.pth", map_location=DEVICE))
agent.q.eval()
print("[INFO] Model loaded successfully.")

# ---------------------- SERIAL SETUP ----------------------
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=READ_TIMEOUT)
    time.sleep(2)  # Wait for Arduino to initialize
    print(f"[INFO] Connected to Arduino on {SERIAL_PORT}")
except Exception as e:
    print(f"[ERROR] Could not open serial port: {e}")
    exit(1)

# ---------------------- HELPER ----------------------
def parse_sensor_data(line):
    """
    Expects line format: "Soil=XXX | Temp=XX.XC | Hum=XX.X% | Pump=ON | Mode=Auto"
    Returns: [temp, humidity, soil]
    """
    try:
        parts = line.strip().split("|")
        soil = float(parts[0].split("=")[1])
        temp = float(parts[1].split("=")[1].replace("C", ""))
        hum  = float(parts[2].split("=")[1].replace("%", ""))
        return np.array([temp, hum, soil], dtype=np.float32)
    except Exception as e:
        print(f"[WARN] Failed to parse line: {line} | Error: {e}")
        return None

# ---------------------- MAIN LOOP ----------------------
print("[INFO] Starting live inference loop...")
try:
    while True:
        # 1️⃣ Read line from Arduino
        if ser.in_waiting > 0:
            raw_line = ser.readline().decode('utf-8').strip()
            state_raw = parse_sensor_data(raw_line)
            
            if state_raw is None:
                continue  # skip if parsing failed

            # 2️⃣ Normalize state
            state = norm(state_raw)

            # 3️⃣ RL model predicts action
            with torch.no_grad():
                action = agent.act(state, greedy=True)  # 0 = OFF, 1 = ON

            # 4️⃣ Send decision back to Arduino
            ser.write(f"{action}\n".encode())

            # 5️⃣ Debug print
            print(f"Temp={state_raw[0]:5.1f}C | Hum={state_raw[1]:5.1f}% | Soil={state_raw[2]:4.0f} | Action={'ON' if action==1 else 'OFF'}")

        time.sleep(SEND_INTERVAL)

except KeyboardInterrupt:
    print("\n[INFO] Exiting serial inference.")
    ser.close()
except Exception as e:
    print(f"[ERROR] {e}")
    ser.close()
