#!/bin/bash

# Activate virtual environment
source /home/anas/python_venvs/pytorch-gpu/bin/activate

# Parameters
EPOCHS_LIST=(1 5 10 25 50 75)
ROUNDS_LIST=(5 10 15 20)
LR_LIST=(0.001)

# Create logs directory once
mkdir -p logs

# Loop over parameters
for E in "${EPOCHS_LIST[@]}"; do
  for R in "${ROUNDS_LIST[@]}"; do
    for LR in "${LR_LIST[@]}"; do
      
      if [ $E -lte 10 ] && [ $R -lte 10 ]; then
        continue
      fi

      LOG="logs/e${E}_r${R}_lr${LR}.log"

      echo "Running E=$E R=$R LR=$LR ..."
      flwr run . local-simulation-gpu \
        --run-config="num-server-rounds=${R} local-epochs=${E} learning-rate=${LR} batch-size=1 num-clients=3 fraction-evaluate=0.2 run-partitioner=0" \
        > "$LOG" 2>&1

      echo "Completed: $LOG"
    done
  done
done

# Deactivate virtual environment
deactivate