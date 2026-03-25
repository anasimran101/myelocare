import os

#disable wandb temporraily (login required which is annoying)
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"   # also prevents auto-login safely
os.environ["WANDB_SILENT"] = "true"     # hides any remaining wandb output

import io
import json
import time
from logging import INFO
from pathlib import Path
from typing import Callable, Optional, Iterable

import wandb
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.common import log, logger
from flwr.serverapp import Grid
from flwr.serverapp.strategy import Result
from flwr.serverapp.strategy.strategy_utils import log_strategy_start_info

from flwr.serverapp.strategy import FedAvg, FedProx
from yolosimulation.task import get_model

PROJECT_NAME = "FLOWER-YOLO-SIMU"


class CustomStrategyBase:
    """Reusable base class that injects logging, checkpointing,
    and metric saving into any Flower strategy."""

    # ---------------------------
    # Setup helpers
    # ---------------------------
    def set_save_path_and_run_dir(self, path: Path, run_dir: str):
        self.save_path = path
        self.run_dir = run_dir

    def _update_best_acc(self, current_round: int, accuracy: float, arrays: ArrayRecord):
        if accuracy > self.best_acc_so_far:
            self.best_acc_so_far = accuracy
            logger.log(INFO, "New best global model found: %f", accuracy)

            model = get_model()
            model.model.load_state_dict(arrays.to_torch_state_dict())

            model_path = self.save_path / f"best_{accuracy:.4f}_round_{current_round}.pt"
            model.save(model_path)

            logger.log(INFO, "Saved best model: %s", model_path)

    def save_metrics_as_json(self, current_round: int, result: Result):
        path = self.save_path / "results.json"

        if path.exists():
            try:
                results = json.loads(path.read_text())
            except json.JSONDecodeError:
                results = []
        else:
            results = []

        round_results = {
            "round": current_round,
            "train_metrics": dict(result.train_metrics_clientapp.get(current_round, {})),
            "evaluate_metrics_clientapp": dict(
                result.evaluate_metrics_clientapp.get(current_round, {})
            ),
            "evaluate_metrics_serverapp": dict(
                result.evaluate_metrics_serverapp.get(current_round, {})
            ),
        }

        results.append(round_results)
        path.write_text(json.dumps(results, indent=2))

    # ---------------------------
    # Shared training loop
    def _common_start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: Optional[ConfigRecord] = None,
        evaluate_config: Optional[ConfigRecord] = None,
        evaluate_fn: Optional[
            Callable[[int, ArrayRecord], Optional[MetricRecord]]
        ] = None,
    ) -> Result:

        wandb.init(project=PROJECT_NAME, name=f"{self.run_dir}-ServerApp")

        self.best_acc_so_far = 0.0

        log(INFO, "Starting %s strategy:", self.__class__.__name__)
        log_strategy_start_info(num_rounds, initial_arrays, train_config, evaluate_config)

        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config

        result = Result()
        arrays = initial_arrays

        t_start = time.time()

        # Initial evaluation
        if evaluate_fn:
            res = evaluate_fn(0, arrays)
            if res:
                result.evaluate_metrics_serverapp[0] = res

        for current_round in range(1, num_rounds + 1):
            log(INFO, f"\n[ROUND {current_round}/{num_rounds}]")

            # ---------------- TRAIN ----------------
            train_replies = grid.send_and_receive(
                messages=self.configure_train(current_round, arrays, train_config, grid),
                timeout=timeout,
            )

            agg_arrays, agg_train_metrics = self.aggregate_train(
                current_round, train_replies
            )

            if agg_arrays is not None:
                arrays = agg_arrays
                result.arrays = agg_arrays

            if agg_train_metrics:
                result.train_metrics_clientapp[current_round] = agg_train_metrics
                wandb.log(dict(agg_train_metrics), step=current_round)

            # ---------------- EVALUATE CLIENT ----------------
            eval_replies = grid.send_and_receive(
                messages=self.configure_evaluate(
                    current_round, arrays, evaluate_config, grid
                ),
                timeout=timeout,
            )

            agg_eval_metrics = self.aggregate_evaluate(current_round, eval_replies)

            if agg_eval_metrics:
                result.evaluate_metrics_clientapp[current_round] = agg_eval_metrics
                wandb.log(dict(agg_eval_metrics), step=current_round)

            # ---------------- EVALUATE SERVER ----------------
            if evaluate_fn:
                res = evaluate_fn(current_round, arrays)
                if res:
                    result.evaluate_metrics_serverapp[current_round] = res
                    self._update_best_acc(
                        current_round,
                        res["metrics/mAP50-95(B)"],
                        arrays,
                    )

                    wandb.log(dict(res), step=current_round)

            # Save JSON
            self.save_metrics_as_json(current_round, result)

        log(INFO, "Finished in %.2fs", time.time() - t_start)
        return result


class CustomFedAvg(CustomStrategyBase, FedAvg):
    def start(self, *args, **kwargs):
        return self._common_start(*args, **kwargs)


class CustomFedProx(CustomStrategyBase, FedProx):
    def start(self, *args, **kwargs):
        return self._common_start(*args, **kwargs)
    

STRATEGY_REGISTRY = {
    "FedAvg": CustomFedAvg,
    "FedProx": CustomFedProx
}

def load_strategy(strategy_name: str, **kwargs):
    try:
        strategy_cls = STRATEGY_REGISTRY[strategy_name]
    except KeyError:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    return strategy_cls(**kwargs)