# env.py
print("start")

import random
import numpy as np
import time


class SmartFarmEnv:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(
            seed if seed is not None else int(time.time())
        )

        self.temp_range = (15.0, 35.0)
        self.humidity_range = (30.0, 90.0)
        self.soil_range = (0.0, 1023.0)

        self.step_count = 0
        self.max_steps = 200

    def reset(self):
        self.temp = self.rng.uniform(*self.temp_range)
        self.humidity = self.rng.uniform(*self.humidity_range)
        self.soil = self.rng.uniform(*self.soil_range)

        self.step_count = 0
        return self._get_state()

    def _get_state(self):
        return np.array(
            [self.temp, self.humidity, self.soil],
            dtype=np.float32
        )

    def step(self, action):
        self.step_count += 1

        # Environmental noise
        self.temp += self.np_rng.normal(0, 0.1)
        self.humidity += self.np_rng.normal(0, 0.2)

        # Pump dynamics
        # action = 1 → Pump ON → soil wetter → value decreases
        # action = 0 → Pump OFF → soil dries → value increases
        if action == 1:
            self.soil -= self.rng.uniform(15.0, 30.0)
        else:
            self.soil += self.rng.uniform(3.0, 8.0)

        # Clamp values
        self.temp = float(np.clip(self.temp, *self.temp_range))
        self.humidity = float(np.clip(self.humidity, *self.humidity_range))
        self.soil = float(np.clip(self.soil, *self.soil_range))

        # ================= REWARD FUNCTION =================
        raw_reward = 0.0

        DRY = 700
        WET = 400

        # ---- SOIL MOISTURE REWARD ----
        if self.soil > DRY:  # Dry soil
            if action == 1:
                raw_reward += 15
            else:
                raw_reward -= 20

        elif WET <= self.soil <= DRY:  # Optimal soil
            if action == 0:
                raw_reward += 8
            else:
                raw_reward -= 10

        else:  # Wet soil
            if action == 0:
                raw_reward += 10
            else:
                raw_reward -= 25

        # ---- TEMPERATURE REWARD ----
        if self.temp < 20:
            raw_reward -= (20 - self.temp) * 0.5
        elif self.temp > 30:
            raw_reward -= (self.temp - 30) * 0.5
        else:
            raw_reward += 2

        # ---- HUMIDITY REWARD ----
        if self.humidity < 40:
            raw_reward -= (40 - self.humidity) * 0.2
        elif self.humidity > 70:
            raw_reward -= (self.humidity - 70) * 0.2
        else:
            raw_reward += 1

        # Small step penalty
        raw_reward -= 0.01

        # 🔥 NORMALIZATION (KEY FIX)
        reward = raw_reward / 25.0
        reward = np.clip(reward, -1.0, 1.0)

        done = self.step_count >= self.max_steps

        return self._get_state(), reward, done, {"step": self.step_count}


# ================= STATE NORMALIZATION =================
def norm(s):
    temp_norm = (s[0] - 15.0) / (35.0 - 15.0)
    humidity_norm = (s[1] - 30.0) / (90.0 - 30.0)
    soil_norm = s[2] / 1023.0

    return np.array(
        [temp_norm, humidity_norm, soil_norm],
        dtype=np.float32
    )
