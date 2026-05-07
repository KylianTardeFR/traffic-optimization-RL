from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
import gymnasium as gym
from pathlib import Path
import sumo_rl
from Utils.callbacks import TSCMetricsCallback


def main():
    scenario = "cologne1"
    model_name = "ppo"
    timesteps = 200_000
    seed = 42
    eval_seed = 43
    reward_fn = "diff-waiting-time"

    tb_dir = Path("tb") / scenario / model_name
    output_dir = Path("outputs") / scenario / model_name
    checkpoint_dir = Path("checkpoints") / scenario / model_name

    sumo_path = Path(sumo_rl.__file__).parent
    print(sumo_path)
    scenario_path = sumo_path / "nets" / "RESCO" / scenario
    net_file = scenario_path / f"{scenario}.net.xml"
    route_file= scenario_path / f"{scenario}.rou.xml"

    env = gym.make(
            "sumo-rl-v0",
            net_file=net_file,
            route_file=route_file,
            out_csv_name=str(output_dir / "train"),
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

    eval_env = gym.make(
            "sumo-rl-v0",
            net_file=net_file,
            route_file=route_file,
            out_csv_name=str(output_dir / "eval"),
            single_agent=True,
            use_gui=False,
            num_seconds=5400,
            begin_time=25200,
            delta_time=5,
            yellow_time=2,
            min_green=5,
            max_green=60,
            sumo_seed=eval_seed,
            reward_fn=reward_fn,
            add_system_info=True,
            sumo_warnings=False,
        )
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        tensorboard_log=str(tb_dir),
        seed=seed,
        verbose=1,
        learning_rate=3.0e-4,
        n_steps=1024,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5
    )

    callbacks=[
        TSCMetricsCallback(),
        CheckpointCallback(
            save_freq=50000,
            save_path=checkpoint_dir,
            name_prefix="ppo",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(checkpoint_dir / "best"),
            log_path=str(checkpoint_dir / "eval_logs"),
            eval_freq=20000,
            n_eval_episodes=3,
            deterministic=True,
        ),
    ]

    try:
        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            tb_log_name=scenario,
            progress_bar=True
        )
    finally:
        model.save(str(checkpoint_dir / "final"))
        env.close()
        eval_env.close()
        print(f"[train] saved final model to {checkpoint_dir / 'final'}")


if __name__ == "__main__":
    main()