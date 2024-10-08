#!/bin/sh
env="Overcooked"
layout="forced_coordination"
num_landmarks=3
algo="rmappo"
exp="9_14_late_night"
seed_max=1
num_agents=2


echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python train/train_overcooked.py --save_gifs --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --num_agents ${num_agents} --seed ${seed} --layout_name ${layout}\
    --n_training_threads 1 --n_rollout_threads 100 --episode_length 100 --use_render True
done
