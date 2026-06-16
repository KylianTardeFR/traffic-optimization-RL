"""Evaluate a trained model on multiple SUMO seeds.

This is the bug fix for the eval problem: SB3's EvalCallback only logs
rewards to evaluations.npz — sumo-rl's traffic-metric CSVs aren't
written during its eval loop. So we just do the eval ourselves after
training, capturing the metrics step-by-step.

Import `evaluate` from a notebook:
    from evaluate import evaluate
    from stable_baselines3 import PPO
    model = PPO.load("checkpoints/cologne1/ppo/seed42/final.zip")
    df = evaluate(model)
"""
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import sumo_rl


def evaluate(model, eval_seeds=(100, 200, 300, 400, 500),
             reward_fn="diff-waiting-time", scenario="cologne1"):
    """Run `model` on one episode per SUMO seed.

    Returns a DataFrame with one row per eval seed and columns:
        seed, mean_wait, mean_speed, mean_stopped, p95_wait
    """
    sumo_path = Path(sumo_rl.__file__).parent
    scenario_path = sumo_path / "nets" / "RESCO" / scenario
    net_file = scenario_path / f"{scenario}.net.xml"
    route_file = scenario_path / f"{scenario}.rou.xml"

    rows = []
    for seed in eval_seeds:
        print(f"  eval seed {seed}...", flush=True)
        env = gym.make(
            "sumo-rl-v0",
            net_file=net_file,
            route_file=route_file,
            single_agent=True,
            use_gui=False,
            num_seconds=5400,
            begin_time=25200,
            delta_time=5,
            yellow_time=2,
            min_green=5,
            max_green=60,
            sumo_seed=seed,
            reward_fn=reward_fn,
            add_system_info=True,
            sumo_warnings=False,
        )

        obs, _ = env.reset()
        waits, speeds, stopped = [], [], []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            if "system_mean_waiting_time" in info:
                waits.append(info["system_mean_waiting_time"])
            if "system_mean_speed" in info:
                speeds.append(info["system_mean_speed"])
            if "system_total_stopped" in info:
                stopped.append(info["system_total_stopped"])
        env.close()

        rows.append({
            "seed": seed,
            "mean_wait":    float(np.mean(waits))   if waits   else float("nan"),
            "mean_speed":   float(np.mean(speeds))  if speeds  else float("nan"),
            "mean_stopped": float(np.mean(stopped)) if stopped else float("nan"),
            "p95_wait":     float(np.percentile(waits, 95)) if waits else float("nan"),
        })

    return pd.DataFrame(rows)
