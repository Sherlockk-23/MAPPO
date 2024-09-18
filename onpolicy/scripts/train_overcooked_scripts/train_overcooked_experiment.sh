#!/bin/sh
env="Overcooked"
layout="cramped_room"
algo="rmappo"
exp="para_debug_usefulsoup"
seed_max=10
num_agents=2


echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python ../train/train_overcooked.py --save_gifs --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --num_agents ${num_agents} --seed ${seed} --layout_name ${layout}\
    --n_training_threads 1 --n_rollout_threads 20 --episode_length 150 --use_render False --cuda False\
    --reward_shaping_type 1
done
