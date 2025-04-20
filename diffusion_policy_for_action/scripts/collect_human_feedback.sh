#!/bin/sh
export CURRENT_DIR=$(pwd)
python inference_with_human_feedback.py weight_name=coffee_d0 task_name=motion_conditioned exp_name=motion_conditioned_human_annotation rollout_time=5 model_path=./llava_checkpoints/llava-v1.5-7b_lora_mpm use_temporal_ensemble=false have_ego=false language_codebook_size=37 action_length=4 seed=20