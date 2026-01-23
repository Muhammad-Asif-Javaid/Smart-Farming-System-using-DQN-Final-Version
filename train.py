# train.py
print("Start")
import csv
import torch
import random
import math
import numpy as np
from dataclasses import dataclass
import torch.nn as nn
import torch.optim as optim
from env import SmartFarmEnv, norm

# ----------------------- Replay Buffer -----------------------
@dataclass
class ReplayBuffer:
    capacity: int
    buffer: None = None

    def __post_init__(self):
        self.buffer = []

    def push(self, *transition):
        self.buffer.append(transition)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buffer)

# ----------------------- DQN -----------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, device="cpu", lr=1e-3, gamma=0.99, batch_size=64, epsilon_init=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.device = device
        self.q = QNetwork(3,2).to(device)
        self.target = QNetwork(3,2).to(device)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(10000)
        self.gamma = gamma
        self.batch = batch_size
        self.steps = 0
        self.epsilon = epsilon_init
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def act(self, state, greedy=False):
        if not greedy and random.random() < max(self.epsilon_min, self.epsilon):
            self.steps += 1
            action = random.randint(0,1)
        else:
            with torch.no_grad():
                s = torch.tensor(state).float().unsqueeze(0).to(self.device)
                action = self.q(s).argmax().item()
        self.epsilon *= self.epsilon_decay
        return action

    def train_step(self):
        if len(self.buffer) < self.batch:
            return
        s,a,r,ns,d = self.buffer.sample(self.batch)
        s = torch.tensor(s).float()
        a = torch.tensor(a).long().unsqueeze(1)
        r = torch.tensor(r).float().unsqueeze(1)
        ns = torch.tensor(ns).float()
        d = torch.tensor(d).float().unsqueeze(1)

        qv = self.q(s).gather(1,a)
        with torch.no_grad():
            tq = r + (1 - d) * self.gamma * self.target(ns).max(1, keepdim=True)[0]

        loss = nn.MSELoss()(qv,tq)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

# ----------------------- Training -----------------------
def train_agent():
    env = SmartFarmEnv(123)
    agent = DQNAgent()

    # ------------------- CSV Setup -------------------
    step_file = "training_data.csv"
    hyper_file = "hyperparameters.csv"

    with open(step_file, "w", newline="") as f_step, open(hyper_file, "w", newline="") as f_hyper:
        writer_step = csv.writer(f_step)
        writer_hyper = csv.writer(f_hyper)

        # Step-wise header
        writer_step.writerow(["Episode","Step","Temp","Humidity","Soil","Action","Reward"])

        # Episode-summary / hyperparameters header
        writer_hyper.writerow([
            "Episode", "Total_Reward", "Cumulative_Reward",
            "Epsilon", "Learning_Rate", "Gamma", "Batch_Size",
            "Epsilon_Decay", "Steps_Per_Episode", "ON_Percent", "OFF_Percent"
        ])

        cumulative_reward_total = 0

        for ep in range(1,1001):
            s_raw = env.reset()
            s = norm(s_raw)
            done = False
            total_reward = 0
            step_count = 0
            action_counts = {0:0,1:0}

            while not done:
                a = agent.act(s)
                ns_raw,r,done,info = env.step(a)
                ns = norm(ns_raw)
                agent.buffer.push(s,a,r,ns,done)
                agent.train_step()

                step_count += 1
                total_reward += r
                action_counts[a] += 1
                cumulative_reward_total += r

                # Write step-wise data
                writer_step.writerow([ep, info["step"], *s_raw, a, r])

                s = ns
                s_raw = ns_raw

            # Episode summary -> hyperparameters CSV
            on_percent = action_counts[1]/step_count*100
            off_percent = action_counts[0]/step_count*100

            writer_hyper.writerow([
                ep, total_reward, cumulative_reward_total,
                agent.epsilon, agent.opt.param_groups[0]['lr'], agent.gamma, agent.batch,
                agent.epsilon_decay, step_count,  # <-- added columns
                on_percent, off_percent
            ])

            # Update target network
            if ep % 10 == 0:
                agent.target.load_state_dict(agent.q.state_dict())
                print(f"Episode {ep} Reward {total_reward:.2f} | Epsilon {agent.epsilon:.3f}")

    # Save model
    torch.save(agent.q.state_dict(), "dqn_model.pth")
    print("Training Complete")
    return agent

if __name__=="__main__":
    train_agent()
