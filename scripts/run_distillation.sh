#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1  # Example for multiple GPUs

python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=29500 \
    -m src.run_distillation
