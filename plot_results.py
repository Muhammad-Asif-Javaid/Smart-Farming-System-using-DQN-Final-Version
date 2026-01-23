print("Start")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --------- LOAD DATA ----------
train = pd.read_csv("training_data.csv")
evald = pd.read_csv("evaluation_data.csv")

STEPS_PER_EPISODE = 200


# --------- HELPER FUNCTION ----------
def moving_average(data, window=50):
    return np.convolve(data, np.ones(window)/window, mode='valid')


# =====================================================
# GRAPH 1: TOTAL EPISODE REWARD (RAW)
# =====================================================
ep_total_reward = train.groupby(train.index // STEPS_PER_EPISODE)["Reward"].sum()

plt.figure()
plt.plot(ep_total_reward)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Performance (Episode Total Reward)")
plt.show()


# =====================================================
# GRAPH 2: SMOOTHED EPISODE REWARD (FIXED NOISE)
# =====================================================
plt.figure()
plt.plot(moving_average(ep_total_reward, window=30))
plt.xlabel("Episode")
plt.ylabel("Smoothed Reward")
plt.title("Training Performance (Moving Average)")
plt.show()


# =====================================================
# GRAPH 3: AVERAGE REWARD PER STEP (BEST GRAPH)
# =====================================================
ep_avg_reward = ep_total_reward / STEPS_PER_EPISODE

plt.figure()
plt.plot(ep_avg_reward)
plt.xlabel("Episode")
plt.ylabel("Avg Reward per Step")
plt.title("Average Reward per Step ([-1, 1] Range)")
plt.show()


# =====================================================
# GRAPH 4: SINGLE STEP REWARD (STEP LEVEL)
# =====================================================
plt.figure()
plt.plot(train["Reward"][:2000])  # first 2000 steps for clarity
plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("Single-Step Reward (Zoomed View)")
plt.show()


# =====================================================
# GRAPH 5: CUMULATIVE REWARD TREND
# =====================================================
plt.figure()
plt.plot(np.cumsum(train["Reward"]))
plt.xlabel("Step")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Learning Progress")
plt.show()


# =====================================================
# GRAPH 6: SOIL MOISTURE vs PUMP ACTION
# =====================================================
plt.figure()
plt.scatter(evald["Soil"], evald["Action"], alpha=0.5)
plt.xlabel("Soil Moisture")
plt.ylabel("Pump Action (0 = OFF, 1 = ON)")
plt.title("Soil Moisture vs Pump Decision")
plt.show()


# =====================================================
# GRAPH 7: SOIL MOISTURE DISTRIBUTION PER ACTION
# =====================================================
plt.figure()
evald[evald["Action"] == 0]["Soil"].plot(kind="hist", alpha=0.6, bins=30, label="Pump OFF")
evald[evald["Action"] == 1]["Soil"].plot(kind="hist", alpha=0.6, bins=30, label="Pump ON")
plt.xlabel("Soil Moisture")
plt.ylabel("Frequency")
plt.title("Soil Moisture Distribution by Action")
plt.legend()
plt.show()







# --------- GRAPH 3: Action Distribution (BAR) ----------
action_counts = evald["Action"].value_counts()

plt.figure()
action_counts.plot(kind="bar")
plt.xticks([0,1], ["PUMP OFF (0)", "PUMP ON (1)"], rotation=0)
plt.ylabel("Count")
plt.title("Action Distribution")
plt.show()

# --------- GRAPH 4: Action Distribution (PIE) ----------
plt.figure()
action_counts.plot(kind="pie", autopct="%1.1f%%", startangle=90)
plt.ylabel("")
plt.title("Pump ON vs OFF Percentage")
plt.show()