#! /bin/bash

# basic configs
gpu_ids=7

# dataset configs
dataset_mode=ycb
data_dir=/mnt/blob/data/DexYCB_sub
params_dir=params
images_dir=images

# model configs
model=trainer
gen_name=generator_spade_attn
load_path=checkpoints/net_epoch_30_id_G.pth
output_dir=results
eval_pairs=assets/eval_pairs.pkl

# eval configs
python3 eval.py         --gpu_ids       ${gpu_ids}          \
                        --gen_name      ${gen_name}         \
                        --model         ${model}            \
                        --data_dir      ${data_dir}         \
                        --params_dir    ${params_dir}       \
                        --images_dir    ${images_dir}       \
                        --dataset_mode  ${dataset_mode}     \
                        --load_path     ${load_path}        \
                        --output_dir    ${output_dir}       \
                        --eval_pairs    ${eval_pairs}       \
                        --save_res      --use_spade