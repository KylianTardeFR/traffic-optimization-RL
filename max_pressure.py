import gymnasium as gym
from pathlib import Path
import sumo_rl
import numpy as np


def max_pressure_action(ts: sumo_rl.TrafficSignal) -> int:
    if ts.time_since_last_phase_change < ts.min_green:
        return ts.green_phase

    controlled_links = ts.sumo.trafficlight.getControlledLinks(ts.id)

    best_phase, best_pressure = ts.green_phase, -float("inf")
    for phase_idx, phase in enumerate(ts.green_phases):
        seen_in = set()
        incoming_q = 0
        for link_idx, link_group in enumerate(controlled_links):
            if link_idx < len(phase.state) and phase.state[link_idx] in "Gg":
                lane = link_group[0][0]
                if lane not in seen_in:
                    incoming_q += ts.sumo.lane.getLastStepHaltingNumber(lane)
                    seen_in.add(lane)
        outgoing_q = sum(
            ts.sumo.lane.getLastStepHaltingNumber(l) for l in ts.out_lanes
        )
        pressure = incoming_q - outgoing_q
        if pressure > best_pressure:
            best_pressure = pressure
            best_phase = phase_idx

    return best_phase


def max_pressure():
    scenario = "cologne1"
    model_name = "max_pressure"
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
        yellow_time=2,
        min_green=5,
        max_green=60,
        fixed_ts=False,
        sumo_seed=seed,
        add_system_info=True,
        sumo_warnings=False,
        reward_fn="queue",
    )

    env.reset()
    terminated = truncated = False
    waits, speeds, stopped = [], [], []
    raw_env = env.unwrapped

    while not (terminated or truncated):
        ts_id = next(iter(raw_env.traffic_signals))
        ts = raw_env.traffic_signals[ts_id]
        action = max_pressure_action(ts)

        _, _, terminated, truncated, info = env.step(action)
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
    print(max_pressure())