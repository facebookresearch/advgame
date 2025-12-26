# sp_advgame

## Install
Clone the relevant repositories
```
cd ~/Projects
git clone git@github.com:fairinternal/sp_advgame.git
git clone git@github.com:fairinternal/fairseq2-ext.git
git clone git@github.com:facebookresearch/fairseq2.git
```

Obtain a compute on which all proceeding install steps are performed e.g. via
```
srun --ntasks=1 --ntasks-per-node=1 --cpus-per-task 16  --mem-per-cpu=16G -t 24:00:00 --gpus-per-node=1 --account=memorization --qos=h200_memorization_high --pty zsh
```

Create the uv venv
```
export ENV_PATH=/storage/home/apaulus/envs/advgame
uv venv $ENV_PATH
source $ENV_PATH/bin/activate
```

Install dependencies
```
cd ~/Projects/fairseq2-ext
uv sync --extra fs2v05-pt271-cu126 --active
```

Add sp_advgame as remote in fairseq2 inside env
```
cd ~/Projects/fairseq2
git remote add sp_advgame git@github.com:fairinternal/sp_advgame.git
git fetch sp_advgame
git checkout fairseq2/gdpo
```

Replace installed fairse2 with editable install **on our fairseq2/gdpo branch**
```
cd ~/Projects/fairseq2
pip uninstall fairseq2
pip install --no-deps -e .
```
If a warning shows that fairseq2 is not installed, try deactivating and reactivating the uv venv

Install remaining deps
```
uv pip install polars retrying pandas xxhash
```

## Training
First allocate resources:
- Testing:
```
salloc --nodes 2 --tasks-per-node 4 --cpus-per-task 8 --mem-per-cpu=16G -t 24:00:00 --gpus-per-node=4 --account=memorization --qos=h200_dev --job-name advgame_test
```
- Full training:
```
salloc --nodes 2 --tasks-per-node 8 --cpus-per-task 24 -t 72:00:00 --gpus-per-node=8 --mem=0G --account=memorization --qos=h200_alignment_vip --job-name advgame
```

After it is allocated, change directory and activate env
```
cd ~/Project/sp_advgame/scipts
source ~/envs/YOUR_ENV_NAME/bin/activate
```
For testing run
```
bash start_training_wray_ray_multinode.sh ../configs/online_dpo_game_pairwise_llama3_2_1b_test.yaml "test"
```
Or to run full training, first sync models to scratch for fast loading
```
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash parallel_rsync_copy.sh /checkpoint/memorization/apaulus/models /scratch
```
and run
```
bash start_training_wray_ray_multinode.sh ../configs/online_dpo_game_pairwise_llama3_1_8b_abliterated_llama3_1_8b_llama3_3_70b_abliterated.yaml "your_job_description"
```
