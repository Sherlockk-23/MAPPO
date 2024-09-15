#!/bin/sh
env="Overcooked"
layout="asymmetric_advantages"
algo="rmappo"
exp="asymmetric_915"
seed_max=1
num_agents=2


echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python ../train/train_overcooked.py --save_gifs --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --num_agents ${num_agents} --seed ${seed} --layout_name ${layout}\
    --n_training_threads 1 --n_rollout_threads 20 --episode_length 100 --use_render True --use_wandb True
done
