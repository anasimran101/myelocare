import torch
import os
from flwr.clientapp import ClientApp
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from yolosimulation.task import get_model, train as train_fn, test as test_fn, make_project_name
from yolosimulation.config import *


# Initialize Flower ClientApp
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    """Train the YOLOv8 model on local data."""
    
    # Load YOLOv8 model
    model = get_model()

    # Load server weights
    state_dict = msg.content["arrays"].to_torch_state_dict()
    model.model.load_state_dict(state_dict)  # underlying nn.Module
    
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.model.to(device)

    # Training config
    node_partition_id = context.node_config.get("partition-id", -1)
    local_epochs = msg.content["config"]["localepochs"]
    lr = msg.content["config"]["lr"]
    dataset_base = msg.content["config"]["dataset-base"]
    
    round_id = -1
    round_id = msg.content["config"]["server-round"]

    # Create consistent log folders for eval too
    project, name = make_project_name("train", f"client_{node_partition_id}", round_id)

    print(context.node_config)
    print(context.run_config)
    print("Configs: ", msg.content["config"])
    
    yaml_path = os.path.join(DATASET_PATH, f"client_{node_partition_id}.yaml")
    print(f"Train YAML file for client node {node_partition_id}, file path {yaml_path}")

    # Local training
    metrics = train_fn(model,project=project, name=name, epochs=local_epochs, lr=lr, data_path=yaml_path)
    # Return updated weights and metrics
    model_record = ArrayRecord(model.model.state_dict())
    metric_record = MetricRecord(metrics)

    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)

@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the YOLOv8 model on local validation data."""

    # Load model
    model = get_model()

    # Load server weights
    node_partition_id = context.node_config.get("partition-id", -1)
    state_dict = msg.content["arrays"].to_torch_state_dict()
    model.model.load_state_dict(state_dict)
    dataset_base = context.run_config.get("dataset-base")
    
    yaml_path = os.path.join(DATASET_PATH, f"client_{node_partition_id}.yaml")
    print(f"Eval YAML file for client node {node_partition_id}, file path {yaml_path}")

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.model.to(device)

    # Evaluate
    round_id = -1
    round_id = msg.content["config"]["server-round"]
    project, name = make_project_name("val", f"client_{node_partition_id}", round_id)
    metrics = test_fn(model,data_path=yaml_path,project=project, name=name, )  # return dict, int
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
