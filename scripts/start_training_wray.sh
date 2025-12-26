#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

### SBATCH directives to specify job configuration
## %j is the job id, %u is the user id. These files are where output is printed.
#SBATCH --output=/checkpoint/ram/kulikov/slurm_logs/%j.out
#SBATCH --error=/checkpoint/ram/kulikov/slurm_logs/%j.err
#SBATCH --job-name=grpo
#SBATCH --nodes=5
#SBATCH --mem=0G
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=8
#SBATCH --account=ram
#SBATCH --qos=ram_high
#SBATCH --time 24:00:00

# Get the list of nodes and convert it to an array
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
# Get the last node and its IP address
last_node=${nodes_array[-1]}
last_node_ip=$(ping $last_node -c 1 -q 2>&1 | grep -Po "(\d{1,3}\.){3}\d{1,3}")
# Check if Ray is already running on the last node
ray_running=$(python check_ray.py $last_node_ip 1> /dev/null 2> /dev/null)

# --- Check if Ray is already running using the python script ---
# Call the script and check its exit code OR capture output
RAY_PORT=10001
if python check_ray.py "$last_node_ip" "$RAY_PORT" > /dev/null 2>&1; then
    # The script exited with 0 (success)
    echo "Ray head node is already running on $last_node ($last_node_ip:$RAY_PORT)."
    ray_running=true
else
    # The script exited with non-zero (failure)
    echo "Ray head node not found or not responding on $last_node ($last_node_ip:$RAY_PORT)."
    ray_running=false
fi

if [ "$ray_running" = false  ]; then
    # Find an available port
    port=$(comm -23 <(seq 49152 65535) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | grep "[0-9]\{1,5\}" | sort | uniq) | shuf | head -n 1)
    # Define the number of CPUs per task
    USER_CPUS_PER_TASK=8
    # Start the Ray cluster on the last node
    srun --nodes=1 --ntasks=1 --cpus-per-task="${USER_CPUS_PER_TASK}" -w "$last_node" ray start --head --node-ip-address="$last_node_ip" --port=$port --num-cpus "${USER_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &

    sleep 15
else
    echo "Ray is running on $last_node"
fi

let NUM_TRAINING_NODES=$SLURM_NNODES-1
NUM_TRAINING_PROCS=8

# srun --nodes $NUM_TRAINING_NODES --ntasks-per-node $NUM_TRAINING_PROCS fairseq2 lm online_finetune /checkpoint/ram/kulikov/online_dpo_frombase_test --no-sweep-dir --config-file ./online_dpo_config.yaml --config criterion.config.vllm_model.ray_cluster_ip_address=$last_node_ip

CONFIG_PATH=$1
SAVE_BASE_PATH=${2:-"/checkpoint/"}
if [[ "${SAVE_BASE_PATH}" != */ ]]; then
    SAVE_BASE_PATH="${SAVE_BASE_PATH}/"
fi
CONFIG_NAME="${CONFIG_PATH##*/}"
CONFIG_NAME="${CONFIG_NAME%.*}"
DATE="$(date +%Y%m%d-%H%M%S)"
RUN_NAME="${CONFIG_NAME}_${DATE}"

srun --nodes $NUM_TRAINING_NODES --ntasks-per-node $NUM_TRAINING_PROCS fairseq2 lm online_finetune_game ${SAVE_BASE_PATH}${RUN_NAME} --no-sweep-dir --config-file ${CONFIG_PATH} --config vllm.ray_cluster_ip_address=$last_node_ip common.metric_recorders.wandb.run_name=$RUN_NAME
