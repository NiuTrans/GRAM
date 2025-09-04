#!/bin/bash

MODEL_PATH={model_path}

GPU_IDS=(0 1 2 3 4 5 6 7)
PORTS=(8000 8001 8002 8003 8004 8005 8006 8007)

NUM_INSTANCES=${#GPU_IDS[@]}

for ((i=0; i<NUM_INSTANCES; i++)); do
    GPU_ID=${GPU_IDS[$i]}
    PORT=${PORTS[$i]}

    PID=$(lsof -ti tcp:$PORT)
    if [ -n "$PID" ]; then
	    echo "Killing process $PID on port $PORT"
	    kill -9 $PID
    else
	    echo "Port $PORT is free."
    fi

    echo "Launching instance on GPU $GPU_ID with port $PORT..."

    CUDA_VISIBLE_DEVICES=$GPU_ID \
    nohup vllm serve $MODEL_PATH \
            --port $PORT \
            --max_model_len 8192 > vllm_log_${GPU_ID}.log &
done

echo "All VLLM instances launched."
