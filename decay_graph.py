import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------
# Load training hyperparameters CSV
# ----------------------------------
df = pd.read_csv("hyperparameters.csv")

episodes = df["Episode"]
epsilon = df["Epsilon"]

# ----------------------------------
# 1️⃣ Epsilon vs Episodes
# ----------------------------------
plt.figure()
plt.plot(episodes, epsilon)
plt.xlabel("Episodes")
plt.ylabel("Epsilon (ε)")
plt.title("Epsilon vs Episodes")
plt.grid(True)
plt.savefig("epsilon_vs_episodes.png")
plt.show()

# ----------------------------------
# 2️⃣ Epsilon Decay Curve
# ----------------------------------
plt.figure()
plt.plot(episodes, epsilon)
plt.xlabel("Episodes")
plt.ylabel("Decayed Epsilon")
plt.title("Epsilon Decay Curve (Exploration → Exploitation)")
plt.grid(True)
plt.savefig("epsilon_decay_curve.png")
plt.show()
