# evaluate.py
print("start")
import csv
import torch
from env import SmartFarmEnv, norm
from train import QNetwork, DQNAgent

def action_name(a):
    return "PUMP ON" if a == 1 else "PUMP OFF"

def evaluate(agent, episodes=10):
    env = SmartFarmEnv(999)

    with open("evaluation_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Episode","Step",
            "Temperature","Humidity","Soil",
            "Action","Reward"
        ])

        for ep in range(1, episodes + 1):
            s_raw = env.reset()
            s = norm(s_raw)
            done = False
            total_reward = 0

            print(f"\n========== EVALUATION EPISODE {ep} ==========")

            while not done:
                a = agent.act(s, greedy=True)
                ns_raw, r, done, info = env.step(a)
                ns = norm(ns_raw)

                temp, hum, soil = s_raw

                # ---- TERMINAL PRINT (VIVA FRIENDLY) ----
                print(
                    f"Step {info['step']:3d} | "
                    f"T={temp:5.1f}°C | "
                    f"H={hum:5.1f}% | "
                    f"Soil={soil:4.0f} | "
                    f"Action={action_name(a):8s} | "
                    f"Reward={r:6.2f}"
                )

                # ---- CSV LOG ----
                writer.writerow([
                    ep, info["step"],
                    temp, hum, soil,
                    action_name(a), r
                ])

                s = ns
                s_raw = ns_raw
                total_reward += r

            print(f"[EVAL RESULT] Episode {ep} Total Reward = {total_reward:.2f}")

    print("\nEvaluation Complete. CSV saved as evaluation_data.csv")

if __name__ == "__main__":
    device = "cpu"
    agent = DQNAgent(device)
    agent.q.load_state_dict(
        torch.load("dqn_model.pth", map_location=device)
    )
    evaluate(agent, episodes=20)
