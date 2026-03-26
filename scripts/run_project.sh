#!/bin/bash

# Activate virtual environment
source ../myelocare_env/bin/activate

# Parameters
EPOCHS_LIST=(5)
ROUNDS_LIST=(50)
LR_LIST=(0.001)
STRATEGIES=("FedAvg")
PARTITIONER="noniid"

# Create logs directory once
mkdir -p logs

# disable dup logs in flwr
#export RAY_DEDUP_LOGS=0

# Loop over parameters
for SR in "${STRATEGIES[@]}"; do
  for E in "${EPOCHS_LIST[@]}"; do
    for R in "${ROUNDS_LIST[@]}"; do
      for LR in "${LR_LIST[@]}"; do

        LOG="logs/${SR}_e${E}_r${R}_lr${LR}_$(date +%Y%m%d_%H%M%S).log"
        RUN_CONFIG="num-server-rounds=${R} local-epochs=${E} learning-rate=${LR} strategy=\"${SR}\" partitioner=\"${PARTITIONER}\" batch-size=1 num-clients=3 fraction-evaluate=0.2"
        echo "flwr run . local-simulation-gpu --run-config=\"$RUN_CONFIG\" > \"$LOG\" 2>&1"

        flwr run . local-simulation-gpu --run-config="$RUN_CONFIG" > "$LOG" 2>&1
          

        echo "Completed: $LOG"
      done
    done
  done
done

# Deactivate virtual environment
deactivate