# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# nnodes determines the number of GPU nodes to utilize (usually 1 for an 8 GPU node)
# nproc_per_node indicates the number of GPUs per node to employ.
cd /local/mnt/workspace/wanqi/8bit_optimizer/remote/SpinQuant
export CUDA_VISIBLE_DEVICES=0
pip install torch==2.1.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
export USE_NVFP4=1
# export QUANT_MODE="Dynamic_Double"
export WANDB_MODE="disabled"
# torchrun --standalone --nproc_per_node=gpu optimize_rotation.py \
#     --input_model  /local/mnt/workspace/wanqi/llama/LLM-Research/Meta-Llama-3.1-8B \
#     --output_rotation_path "/local/mnt/workspace/wanqi/8bit_optimizer/remote/SpinQuant/output/llama3/your_path" \
#     --output_dir "/local/mnt/workspace/wanqi/8bit_optimizer/remote/SpinQuant/output/llama3/your_output_path/" \
#     --logging_dir "/local/mnt/workspace/wanqi/8bit_optimizer/remote/SpinQuant/output/llama3/your_log_path/" \
#     --model_max_length 2048 \
#     --fp16 False \
#     --bf16 True \
#     --log_on_each_node False \
#     --per_device_train_batch_size 1 \
#     --logging_steps 1 \
#     --learning_rate 1.5 \
#     --weight_decay 0. \
#     --lr_scheduler_type "cosine" \
#     --gradient_checkpointing True \
#     --save_safetensors False \
#     --max_steps 100 \
#     --w_bits 16 \
#     --a_bits 4 \
#     --k_bits 4 \
#     --v_bits 4 \
#     --w_clip \
#     --a_asym \
#     --k_asym \
#     --v_asym \
#     --k_groupsize 128 \
#     --v_groupsize 128 > /local/mnt/workspace/wanqi/8bit_optimizer/remote/SpinQuant/output/llama3/optimize.log 2>&1


# torchrun --nnodes=1 --nproc_per_node=1 ptq.py \
#     --input_model /local/mnt/workspace/wanqi/llama/LLM-Research/Meta-Llama-3.1-8B \
#     --do_train False \
#     --do_eval True \
#     --per_device_eval_batch_size 4 \
#     --model_max_length 2048 \
#     --fp16 False \
#     --bf16 True \
#     --save_safetensors False \
#     --w_bits 4 \
#     --a_bits 4 \
#     --k_bits 16 \
#     --v_bits 16 \
#     --w_clip \
#     --w_rtn \
#     --a_asym \
#     --k_asym \
#     --v_asym \
#     --k_groupsize 128 \
#     --v_groupsize 128 \
#     --rotate \
#     --optimized_rotation_path "/local/mnt/workspace/wanqi/8bit_optimizer/remote/SpinQuant/output/llama3/your_path/R.bin" > /local/mnt/workspace/wanqi/8bit_optimizer/remote/SpinQuant/output/llama3/ptq_try.log 2>&1


# torchrun --standalone --nproc_per_node=gpu analysis.py \
#     --input_model /local/mnt/workspace/wanqi/8bit_optimizer/remote/SpinQuant/llama3_gln/checkpoint-500 \
#     --do_train False \
#     --do_eval True \
#     --per_device_eval_batch_size 4 \
#     --model_max_length 256 \
#     --fp16 False \
#     --bf16 True \
#     --save_safetensors False \
#     --w_bits 16 \
#     --a_bits 16 \
#     --k_bits 16 \
#     --v_bits 16 \
#     --w_clip \
#     --w_rtn \
#     --a_asym \
#     --k_asym \
#     --v_asym \
#     --k_groupsize 128 \
#     --v_groupsize 128 \
#     --rotate \
#     --optimized_rotation_path "/local/mnt/workspace/wanqi/8bit_optimizer/remote/SpinQuant/output/llama3/your_path/R.bin" 


# torchrun --standalone --nproc_per_node=gpu ptq.py \
#     --input_model /local/mnt/workspace/wanqi/8bit_optimizer/remote/SpinQuant/llama3_gln/checkpoint-500 \
#     --do_train False \
#     --do_eval True \
#     --per_device_eval_batch_size 4 \
#     --model_max_length 256 \
#     --fp16 False \
#     --bf16 True \
#     --save_safetensors False \
#     --w_bits 16 \
#     --a_bits 16 \
#     --k_bits 16 \
#     --v_bits 16 \
#     --w_clip \
#     --w_rtn \
#     --a_asym \
#     --k_asym \
#     --v_asym \
#     --k_groupsize 128 \
#     --v_groupsize 128 \
#     --groupnorm \
#     --optimized_rotation_path "/local/mnt/workspace/wanqi/8bit_optimizer/remote/SpinQuant/output/llama3/your_path/R.bin" 