# Safety Alignment of LMs via Non-cooperative Games
[Anselm Paulus](https://scholar.google.com/citations?user=njZL5CQAAAAJ),
[Ilia Kulikov](https://scholar.google.com/citations?user=fN7fYXIAAAAJ),
[Brandon Amos](https://bamos.github.io/),
[Rémi Munos](https://scholar.google.com/citations?hl=en&user=OvKEnVwAAAAJ),
[Ivan Evtimov](https://ivanevtimov.eu/),
[Kamalika Chaudhuri](https://cseweb.ucsd.edu/~kamalika)
[Arman Zharmagambetov](https://arman-z.github.io),


This repo is an official implementation of the **AdvGame** ([arxiv:2512.20806](https://arxiv.org/abs/2512.20806)).

tl;dr: We train Attacker LM and Defender LM to play against each others. This leads to a Defender with much better utility-safety tradeoff, and an Attacker that is quite useful for downstream red-teaming tasks.

## Install
Clone the repository:
```
git clone git@github.com:facebookresearch/advgame.git
cd advgame
```

Run the installation script:
```
bash install_env.sh
```

## Data and Models

The following instructions assume you are using [Slurm](https://slurm.schedmd.com/documentation.html) for job scheduling and resource management. If you're not using Slurm, adapt the commands to your scheduler or environment.

### Models
Activate env if not done yet:
```
source ./env/bin/activate
```

Run the following command to download Qwen2.5 models under /scratch/models:
```
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash scripts/download_qwen_models.sh
```
Alternatively, see [scripts/download_llama3_models.sh](scripts/download_llama3_models.sh) for downloading Llama3 models and [scripts/parallel_rsync_copy.sh](scripts/parallel_rsync_copy.sh) to copy your local models under /scratch/models.

### Data

Dataset is already pre-generated and available under [data/wildjailbreak_alpaca](data/wildjailbreak_alpaca/). We also provide the original data generation scripts available under [scripts/data_and_model_processing](scripts/data_and_model_processing/).


## Training
The following instructions assume you are using [Slurm](https://slurm.schedmd.com/documentation.html) for job scheduling and resource management. If you're not using Slurm, adapt the commands to your scheduler or environment.

First allocate resources on Slurm. Below is for training on 2 nodes (= 16 H100/H200 GPUs):
```
salloc --nodes 2 --tasks-per-node 8 --cpus-per-task 24 -t 72:00:00 --gpus-per-node=8 --mem=0 --account=ACCOUNT --job-name advgame
```

Activate env if not done yet:
```
source ./env/bin/activate
```

Run the training script:
```
bash scripts/start_training_wray_ray_multinode.sh PATH_TO_CONFIG FULL_PATH_TO_DATASET PATH_TO_DUMP_CHECKPOINTS
```

You might want to check [configs/paper](configs/paper/) to see/modify training configurations and hyperparameters. For example, the following will launch [AdvGame-DPO on Qwen2.5](configs/paper/qwen25/paper_qwen25_dpo_pairwise_offpolicy.yaml).
```
bash scripts/start_training_wray_ray_multinode.sh configs/paper/qwen25/paper_qwen25_dpo_pairwise_offpolicy.yaml /home/advgame/data/wildjailbreak_alpaca /checkpoint/advgame
```

## Evaluations

See the [eval/](eval/) directory for detailed instructions on running evaluations after training completes.

## License
The majority of AdvGame is licensed under [CC-BY-NC 4.0 license](./LICENSE), however portions of the project are available under separate license terms: fairseq2 is licensed under the MIT license (see [fairseq2/LICENSE](./fairseq2/LICENSE)); safety-eval-fork is licensed under [Apache-2.0](eval/safety-eval-fork/LICENSE.md);
