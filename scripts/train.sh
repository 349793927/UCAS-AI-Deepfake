#!/bin/bash

# 只使用本机两个GPU
GPU_NUM=8

# 设置 NCCL 和 CUDA 环境变量，避免通信问题
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
# export NCCL_SOCKET_IFNAME=lo

# 运行命令
torchrun \
    --nproc_per_node=$GPU_NUM \
    --master_port=29513 \
    main_finetune.py \
    --model AIDE \
    --batch_size 128 \
    --blr 1e-3 \
    --epochs 10 \
    --data_path /data/liu.ruiqi/UCAS/train \
    --eval_data_path /data/liu.ruiqi/UCAS/val \
    # unifd: ADM 80 stable_diffusion_v_1_4 79.7 wokong 79.72
    # unifd: ADM 56 stable_diffusion_v_1_4 79.7 wokong 50.21%
    # --data_path /data/liu.ruiqi/Genimage/VQDM/train \
    # --eval_data_path /data/liu.ruiqi/AIGC_bm \
    ${@:1}
