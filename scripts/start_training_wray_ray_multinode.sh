#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# set -x
# --- Parse arguments ---
RESTART_RAY=false
for arg in "$@"; do
    if [[ "$arg" == "--restart-ray" ]]; then
        RESTART_RAY=true
    fi
done

# --- Get node info ---
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
last_node=${nodes_array[-1]}
last_node_ip=$(ping $last_node -c 1 -q 2>&1 | grep -Po "(\d{1,3}\.){3}\d{1,3}")

USER_CPUS_PER_TASK=8
NUM_RAY_NODES=$(($SLURM_JOB_NUM_NODES-1))
echo NUM_RAY_NODES: $NUM_RAY_NODES
# if ! (( $NUM_RAY_NODES <= $SLURM_JOB_NUM_NODES - 1 )); then
#     echo "Assertion failed: NUM_RAY_NODES is not less than SLURM_JOB_NUM_NODES - 1"
#     exit 1  # Exit with an error code if the assertion fails
# fi

worker_total_num=$((SLURM_JOB_NUM_NODES - 1))
worker_start_num=$((SLURM_JOB_NUM_NODES - $NUM_RAY_NODES))

# --- Optionally kill all Ray processes ---
if $RESTART_RAY; then
    echo "Stopping all Ray nodes..."
    # Only stop on nodes where Ray is started
    echo "Stopping Ray on $last_node"
    ssh $last_node 'pkill -9 -f "ray start"'
    for ((i = worker_total_num-1; i >= worker_start_num; i--)); do
        node_i=${nodes_array[$i]}
        echo "Stopping Ray on $node_i"
        ssh $node 'pkill -9 -f "ray start"'
    done
    sleep 5
fi

# --- Check if Ray head is running ---
RAY_PORT=10001
port=58947
ray_head_running=false
if timeout 10 ray status --address=$last_node:$port > /dev/null 2>&1; then
    ray_head_running=true
fi
if $ray_head_running; then
    echo "Ray head node is already running on $last_node ($last_node_ip:$RAY_PORT)."
else
    # Find an available port
    #port=$(comm -23 <(seq 49152 65535) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | grep "[0-9]\{1,5\}" | sort | uniq) | shuf | head -n 1)
    ip_head=$last_node_ip:$port
    export ip_head
    echo "IP Head: $ip_head"
    echo "Starting ray HEAD at $last_node"
    srun --nodes=1 --ntasks=1 -w "$last_node" \
        ray start --head --node-ip-address="$last_node_ip" --port=$port \
        --num-cpus "${USER_CPUS_PER_TASK}" --num-gpus ${SLURM_GPUS_PER_NODE} --block &
    sleep 15
    for ((i = worker_total_num-1; i >= worker_start_num; i--)); do
        node_i=${nodes_array[$i]}
        echo "Starting ray WORKER $i at $node_i"
        srun --nodes=1 --ntasks=1 --cpus-per-task="${USER_CPUS_PER_TASK}" -w "$node_i" \
            ray start --address "$ip_head" \
            --num-cpus "${USER_CPUS_PER_TASK}" --num-gpus ${SLURM_GPUS_PER_NODE} --block &
        sleep 15
    done
fi

NUM_TRAINING_PROCS=$SLURM_NTASKS_PER_NODE

let NUM_TRAINING_NODES=$SLURM_NNODES-$NUM_RAY_NODES
echo "NUM_TRAINING_NODES: $NUM_TRAINING_NODES"
echo "NUM_TRAINING_PROCS: $NUM_TRAINING_PROCS"


CONFIG_PATH=$1
SAVE_BASE_PATH=${2:-"/checkpoint/"}
if [[ "${SAVE_BASE_PATH}" != */ ]]; then
    SAVE_BASE_PATH="${SAVE_BASE_PATH}/"
fi
DESCRIPTOR=${3:-"advgame"}
CONFIG_NAME="${CONFIG_PATH##*/}"
CONFIG_NAME="${CONFIG_NAME%.*}"
DATE="$(date +%Y%m%d-%H%M%S)"
RUN_NAME="${CONFIG_NAME}/${DATE}_${DESCRIPTOR}"

srun --nodes $NUM_TRAINING_NODES --ntasks-per-node $NUM_TRAINING_PROCS fairseq2 lm online_finetune_game ${SAVE_BASE_PATH}${RUN_NAME} --no-sweep-dir --config-file ${CONFIG_PATH} --config vllm.ray_cluster_ip_address=$last_node_ip common.metric_recorders.wandb.run_name=${RUN_NAME}
