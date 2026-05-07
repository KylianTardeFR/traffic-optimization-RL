import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TSCMetricsCallback(BaseCallback):
    TRACKED_KEYS = (
        "system_mean_waiting_time",
        "system_mean_speed",
        "system_total_stopped",
        "system_total_running",
        "system_mean_travel_time",
    )

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._running: dict[str, list[float]] = {k: [] for k in self.TRACKED_KEYS}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if not isinstance(info, dict):
                continue
            for key in self.TRACKED_KEYS:
                if key in info:
                    self._running[key].append(float(info[key]))
        return True

    def _on_rollout_end(self) -> None:
        """Aggregate and log at the end of each rollout."""
        for key, vals in self._running.items():
            if vals:
                tb_key = "traffic/" + key.replace("system_", "")
                self.logger.record(tb_key, float(np.mean(vals)))
                self._running[key] = []