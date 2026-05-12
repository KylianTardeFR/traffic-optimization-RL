import gymnasium as gym
from pathlib import Path
import sumo_rl
import numpy as np


def main():
    scenario = "cologne1"
    model_name = "fixed_time"
    seed = 42

    output_dir = Path("outputs") / scenario / model_name

    sumo_path = Path(sumo_rl.__file__).parent
    scenario_path = sumo_path / "nets" / "RESCO" / scenario
    net_file = scenario_path / f"{scenario}.net.xml"
    route_file= scenario_path / f"{scenario}.rou.xml"

    env = gym.make(
                "sumo-rl-v0",
                net_file=net_file,
                route_file=route_file,
                out_csv_name=str(output_dir),
                single_agent=True,
                use_gui=False,
                num_seconds=5400,
                begin_time=25200,
                delta_time=5,
                fixed_ts=True,
                sumo_seed=seed,
                add_system_info=True,
                sumo_warnings=False,
            )
    
    env.reset()
    terminated = truncated = False
    waits, speeds, stopped = [], [], []

    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(0)
        if "system_mean_waiting_time" in info:
            waits.append(info["system_mean_waiting_time"])
        if "system_mean_speed" in info:
            speeds.append(info["system_mean_speed"])
        if "system_total_stopped" in info:
            stopped.append(info["system_total_stopped"])

    env.close()
    return {
        "seed": seed,
        "mean_waiting_time": float(np.mean(waits)) if waits else float("nan"),
        "mean_speed": float(np.mean(speeds)) if speeds else float("nan"),
        "mean_stopped": float(np.mean(stopped)) if stopped else float("nan"),
    }


if __name__ == "__main__":
    main()