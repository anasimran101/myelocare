import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from yolosimulation.strategy import CustomFedAvg

from yolosimulation.task import get_model, test, create_run_dir, make_project_name
from yolosimulation.data_partition import create_partitions
from yolosimulation.config import *



app = ServerApp()




def gen_run_dir():
    import os
    from pathlib import Path
    import json 
    from datetime import datetime
    BASE = Path("runs")
    today = datetime.now().strftime("%Y%m%d")
    day_dir = BASE / today
    day_dir.mkdir(parents=True, exist_ok=True)

    # Find next available run ID
    existing = sorted(day_dir.glob("run*"))
    if not existing:
        RUN_ID = "run001"
    else:
        last = existing[-1].name
        num = int(last.replace("run", ""))
        RUN_ID = f"run{num+1:03d}"

    # Stable folder for this run
    RUN_DIR = day_dir / RUN_ID
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    # Save this run info in a config file that all clients can read
    run_info_file = BASE / today / "current_run.json"
    with open(run_info_file, "w") as f:
        json.dump({
            "RUN_ID": RUN_ID,
            "RUN_DIR": str(RUN_DIR.resolve())
        }, f, indent=2)

    # Optionally, also set environment variables so clients started
    # from this server process inherit them
    os.environ["RUN_ID"] = RUN_ID
    os.environ["RUN_DIR"] = str(RUN_DIR.resolve())

@app.main()
def main(grid: Grid, context: Context) -> None:


    fraction_evaluate = context.run_config.get("fraction-evaluate", 0.5)
    num_rounds = context.run_config.get("num-server-rounds", 1)
    lr = context.run_config.get("learning-rate", 0.001)


    if(RUN_PARTITIONER):
        create_partitions(MAIN_DATASET_PATH, DATASET_PATH,NUM_CLIENTS, CLASSES, [ 1/NUM_CLIENTS for i in range(0, NUM_CLIENTS) ])

    # Load initial YOLOv8 model
    global_model = get_model()

    # FIX: correct conversion to Flower arrays
    arrays = ArrayRecord(global_model.model.state_dict())

    strategy = CustomFedAvg(
        fraction_evaluate=fraction_evaluate
    )
    gen_run_dir()
    save_path, run_dir = create_run_dir(config=context.run_config)
    strategy.set_save_path_and_run_dir(save_path, run_dir)

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({
            "lr": lr,
            "localepochs": LOCAL_EPOCHS,
            "dataset-base": DATASET_PATH
        }),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # Save model
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:

    model = get_model()

    model.model.load_state_dict(arrays.to_torch_state_dict())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.model.to(device)


    project, name = make_project_name("val", f"server", server_round)
    metrics = test(model, project=project, name=name, data_path=VAL_YAML_FILE, device=device)
    return MetricRecord(metrics)
