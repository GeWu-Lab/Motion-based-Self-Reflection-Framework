#!/bin/sh
export CURRENT_DIR=$(pwd)
torchrun --nproc_per_node=1 --standalone multi_train.py demonstration_num=500 dataset=idx_speed_chunking language_condition=speed_adjust_llava_motion  epoch=201 task_name=speed_adjust_llava_motion data_path=all_dataset_mapping.json num_diffusion_iters=5 have_ego=false use_language_idx=true language_codebook_size=37 