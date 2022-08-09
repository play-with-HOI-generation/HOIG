#! /bin/bash

# basic configs
gpu_ids=0,1,2,3,4,5,6,7

# dataset configs
dataset_mode=ycb
data_dir=/mnt/blob/data/DexYCB_sub
params_dir=params
images_dir=images

# saving configs
checkpoints_dir=checkpoints
name=exp_ycb

# model configs
model=trainer
gen_name=generator_spade_attn
image_size=256

# training configs
load_path="None"
batch_size=4
lambda_rec=10.0
lambda_tsf=10.0
lambda_mask=1.0
lambda_mask_smooth=1.0
nepochs_no_decay=15  # fixing learning rate when epoch ranges in [0, 5]
nepochs_decay=15    # decreasing the learning rate when epoch ranges in [6, 25+5]

python -m torch.distributed.launch --nproc_per_node 8 \
        train_ddp.py \
        --gpu_ids                 ${gpu_ids}           \
        --data_dir                ${data_dir}          \
        --params_dir              ${params_dir}        \
        --images_dir              ${images_dir}        \
        --checkpoints_dir         ${checkpoints_dir}   \
        --load_path               ${load_path}         \
        --model                   ${model}             \
        --gen_name                ${gen_name}          \
        --name                    ${name}              \
        --dataset_mode            ${dataset_mode}      \
        --image_size              ${image_size}        \
        --batch_size              ${batch_size}        \
        --lambda_tsf              ${lambda_tsf}        \
        --lambda_rec              ${lambda_rec}        \
        --lambda_mask             ${lambda_mask}       \
        --lambda_mask_smooth      ${lambda_mask_smooth}\
        --nepochs_no_decay        ${nepochs_no_decay}  \
        --nepochs_decay           ${nepochs_decay}     \
        --num_repeats             10                  \
         --mask_bce     --use_vgg    --use_spade