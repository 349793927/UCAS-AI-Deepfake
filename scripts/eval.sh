#!/bin/bash

# 只使用本机两个GPU
GPU_NUM=8

# 设置 NCCL 和 CUDA 环境变量，避免通信问题
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1

# 运行命令
torchrun \
    --nproc_per_node=$GPU_NUM \
    --master_port=29512 \
    main_finetune.py \
    --model AIDE \
    --eval True\
    --resume weight/progan_train.pth \
    --batch_size 64 \
    --num_workers 16 \
    --blr 1e-4 \
    --epochs 5 \
    --data_path /data/liu.ruiqi/Chameleon \
    --eval_data_path /data/liu.ruiqi/Chameleon \
    ${@:1}
