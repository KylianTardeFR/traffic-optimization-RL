import glob, re
import pandas as pd
import matplotlib.pyplot as plt
from fixed_time import fixed_time
from max_pressure import max_pressure

SCENARIO = "outputs/cologne1"
MODELS = {"DQN": "DQN", "PPO": "ppo"}


def load_episodes(model_folder, prefix="train"):
    rows = []
    for f in glob.glob(f"{SCENARIO}/{model_folder}/{prefix}_conn*_ep*.csv"):
        ep = int(re.search(r"ep(\d+)", f).group(1))
        df = pd.read_csv(f)
        if len(df) < 3:
            continue
        rows.append({
            "episode": ep,
            "wait": df["system_mean_waiting_time"].mean(),
            "speed": df["system_mean_speed"].mean(),
            "stopped": df["system_total_stopped"].mean(),
        })
    return pd.DataFrame(rows).sort_values("episode").reset_index(drop=True)


# ── Run baselines ──
print("Running Fixed Time baseline...")
ft = fixed_time()
print("Running Max Pressure baseline...")
mp = max_pressure()

BASELINES = {
    "Fixed Time": {"wait": ft["mean_waiting_time"], "speed": ft["mean_speed"], "stopped": ft["mean_stopped"]},
    "Max Pressure": {"wait": mp["mean_waiting_time"], "speed": mp["mean_speed"], "stopped": mp["mean_stopped"]},
}

# ── Load RL episode data ──
data = {}
for name, folder in MODELS.items():
    data[name] = {
        "train": load_episodes(folder, "train"),
        "eval":  load_episodes(folder, "eval"),
    }

# ── Print summary ──
print(f"\n{'Model':<14} {'Train Wait':>12} {'Train Speed':>13} {'Eval Wait':>12} {'Eval Speed':>13} {'Spikes':>8}")
print("-" * 80)
for name in MODELS:
    tr = data[name]["train"]
    ev = data[name]["eval"]
    tr_stable = tr[tr["wait"] < 50].tail(20)
    ev_last = ev.tail(5)
    spikes = len(tr[tr["wait"] > 50])
    print(f"{name:<14} {tr_stable['wait'].mean():>10.2f} s {tr_stable['speed'].mean():>11.2f} m/s"
          f" {ev_last['wait'].mean():>10.2f} s {ev_last['speed'].mean():>11.2f} m/s {spikes:>7d}")

print()
for name, b in BASELINES.items():
    print(f"{name:<14} {b['wait']:>10.2f} s {b['speed']:>11.2f} m/s {'(baseline — single run)':>34}")

# ── Plot ──
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
colors = {"DQN": "#e8772e", "PPO": "#0ea5e9"}
baseline_colors = {"Fixed Time": "#6b7280", "Max Pressure": "#8b5cf6"}
metrics = [("wait", "Mean Waiting Time (s)"),
           ("speed", "Mean Speed (m/s)"),
           ("stopped", "Avg Stopped Vehicles")]

for ax, (col, label) in zip(axes, metrics):
    for name in MODELS:
        df = data[name]["train"]
        clip = 50 if col in ("wait", "stopped") else float("inf")
        df_clean = df[df[col] < clip]
        smooth = df_clean.set_index("episode")[col].rolling(10, min_periods=1).mean()
        ax.plot(smooth.index, smooth.values, label=name, color=colors[name], linewidth=2)
    for name, b in BASELINES.items():
        ax.axhline(b[col], label=name, color=baseline_colors[name], linewidth=1.5, linestyle="--")
    ax.set_xlabel("Episode")
    ax.set_ylabel(label)
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle("DQN vs PPO vs Baselines — Training on Cologne1 (10-ep rolling avg)", fontweight="bold")
fig.tight_layout()
fig.savefig("outputs/cologne1/comparison.png", dpi=150)
print(f"\nSaved → outputs/cologne1/comparison.png\n")
plt.show()
