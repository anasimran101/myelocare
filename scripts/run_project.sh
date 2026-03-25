#!/bin/bash

# Activate virtual environment
source /home/lab/code/myelocare_env/bin/activate

# Parameters
EPOCHS_LIST=(1 5 10 25 50 75)
ROUNDS_LIST=(5 10 15 20)
LR_LIST=(0.001)
STRATEGIES=("FedProx")
PARTITIONER="noniid"

# Create logs directory once
mkdir -p logs

# Loop over parameters
for SR in "${STRATEGIES[@]}"; do
  for E in "${EPOCHS_LIST[@]}"; do
    for R in "${ROUNDS_LIST[@]}"; do
      for LR in "${LR_LIST[@]}"; do

        LOG="logs/${SR}_e${E}_r${R}_lr${LR}.log"

        echo "Running STRATEGY=$SR E=$E R=$R LR=$LR ..."

        flwr run . local-simulation-gpu \
          --run-config="num-server-rounds=${R} local-epochs=${E} learning-rate=${LR} strategy=${SR} partitioner=${PARTITIONER} batch-size=1 num-clients=3 fraction-evaluate=0.2" \
          > "$LOG" 2>&1

        echo "Completed: $LOG"
      done
    done
  done
done

# Deactivate virtual environment
deactivate