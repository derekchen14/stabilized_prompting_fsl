#!/bin/bash
#
set -xue
# export CUDA_VISIBLE_DEVICES=3
# ________ Fine-tuned Model Training ________
# Training with all available data
# python main.py --dataset mwoz --task fine_tune --style dataset --do-train --do-save \
#       --model t5 --size small --num-shots full --maximum-len 1024 --prompt-style none \
#       --context-len 4 --batch-size 8 --log-interval 1200 --learning-rate 1e-4 --n-epochs 7 #--ignore-cache
# python main.py --dataset abcd --task fine_tune --n-epochs 7 --do-train --debug \
#       --style dataset --model gpt --size small --num-shots full --batch-size 6 \
#       --learning-rate 1e-4  --maximum-len 512 --prompt-style naive --context-len 5
# python main.py --dataset abcd --task fine_tune --n-epochs 7 --do-train --do-save \
#       --style dataset --model gpt --size small --num-shots full --batch-size 8 \
#       --learning-rate 1e-4  --maximum-len 512 --prompt-style naive
# python main.py --dataset sgd --task in_context --n-epochs 7 --do-eval --do-save \
#       --style dataset --model gpt --size small --num-shots full --batch-size 8 \
#       --learning-rate 1e-4  --maximum-len 512 --prompt-style naive --debug --ignore-cache

# python main.py --dataset sgd --task fine_tune --n-epochs 7 --do-eval --do-save \
#       --style dataset --model gpt --size small --num-shots full --batch-size 2 \
#       --learning-rate 1e-4  --maximum-len 512 --prompt-style naive #--ignore-cache

# for cl in 1 3 5 7; do
#       for style in 'naive' 'none'; do
#             # python main.py --dataset sgd --task fine_tune --n-epochs 10 --do-train --do-save \
#             #       --style dataset --model gpt --size small --num-shots full --batch-size 8 --prune-keep 3 \
#             #       --learning-rate 1e-4  --maximum-len 512 --prompt-style ${style} --ignore-cache --context-length ${cl}

#             # python main.py --dataset sgd --task fine_tune --n-epochs 7 --do-eval --do-save \
#             #       --style dataset --model gpt --size small --num-shots full --batch-size 16 \
#             #       --learning-rate 1e-4  --maximum-len 512 --prompt-style ${style} --context-length ${cl} --quantify --ignore-cache 

#       done
# done


# python main.py --dataset sgd --task fine_tune --n-epochs 7 --do-eval \
#       --style dataset --model gpt --size small --num-shots full --batch-size 16 \
#       --learning-rate 1e-4  --maximum-len 512 --prompt-style none --context-length 1 \
#       --quantify --ignore-cache --debug

# python main.py --dataset sgd --task fine_tune --n-epochs 10 --do-train --do-save\
#       --style dataset --model gpt --size small --num-shots full --batch-size 16 \
#       --learning-rate 1e-4  --maximum-len 512 --prompt-style none --context-length 2 \
#       --ignore-cache

# python main.py --dataset sgd --task fine_tune --n-epochs 10 --do-eval\
#       --style dataset --model gpt --size small --num-shots full --batch-size 16 \
#       --learning-rate 1e-4  --maximum-len 512 --prompt-style none --context-length 1 \
#       --ignore-cache --quantify

# for lr in 5e-4 1e-4 5e-5 1e-5; do
#       for style in 'naive' 'none'; do
#             python main.py --dataset sgd --task fine_tune --n-epochs 10 --do-train --do-save \
#                   --style dataset --model gpt --size small --num-shots full --batch-size 16 --prune-keep 1 \
#                   --learning-rate ${lr}  --maximum-len 512 --prompt-style ${style} --ignore-cache --context-length 1
#       done
# done


# export CUDA_VISIBLE_DEVICES=0,1


# python main.py --dataset sgd --task fine_tune --style dataset --n-epochs 20 --do-train --do-save \
#       --model gpt --size small --num-shots full --prompt-style none \
#       --learning-rate 1e-4 --verbose --context-length 2 --batch-size 16 --ignore-cache --prune-keep 2

# python main.py --dataset dstc --task fine_tune --style dataset --n-epochs 10 --do-train \
#       --model gpt --size large --num-shots full --maximum-length 512 --prompt-style none \
#       --learning-rate 3e-5 --verbose --context-length 2 --batch-size 16 --ignore-cache --output-dir finetuned --parallel

output_dir="/local2/data/qkun/stabilized_prompting_fsl/"

# # # # # # # 1labm
export CUDA_VISIBLE_DEVICES=0,1,2,3
python main.py --output-dir ${output_dir} \
      --dataset mwoz --task fine_tune --n-epochs 10 --do-train \
      --style dataset --model t5 --size large --num-shots full --batch-size 8 --grad-accum-steps 16\
      --learning-rate 3e-2  --maximum-len 512 --prompt-style naive \
      --verbose --prune-keep 2 --fp16 --ignore-cache --parallel  # quarter
# # # export CUDA_VISIBLE_DEVICES=2,3
# # # python main.py --output-dir ${output_dir} \
# # #       --dataset gsim --task fine_tune --n-epochs 10 --do-train \
# # #       --style dataset --model gpt --size small --num-shots full --batch-size 16 \
# # #       --learning-rate 3e-5  --maximum-len 512 --prompt-style naive \
# # #       --verbose --prune-keep 2 --trainer --ignore-cache #--parallel  # quarter

# export CUDA_VISIBLE_DEVICES=6
# python main.py --output-dir ${output_dir} \
#       --dataset dstc --task fine_tune --n-epochs 10 --do-train \
#       --style dataset --model gpt --size small --num-shots full \
#       --batch-size 16 --grad-accum-steps 1\
#       --learning-rate 3e-5  --maximum-len 512 --prompt-style naive \
#       --verbose --prune-keep 2 --trainer --ignore-cache #--parallel  # quarter

# export CUDA_VISIBLE_DEVICES=7
# python main.py --output-dir ${output_dir} \
#       --dataset mwoz --task fine_tune --n-epochs 10 --do-train \
#       --style dataset --model t5 --size small --num-shots full \
#       --batch-size 16 --grad-accum-steps 1\
#       --learning-rate 3e-5  --maximum-len 512 --prompt-style naive \
#       --verbose --prune-keep 2 --ignore-cache #--parallel  # quarter

export CUDA_VISIBLE_DEVICES=5,6
python main.py --output-dir ${output_dir} \
      --dataset mwoz --task fine_tune --n-epochs 10 --do-train \
      --style dataset --model gpt --size small --num-shots full \
      --batch-size 16 --grad-accum-steps 1\
      --learning-rate 3e-5  --maximum-len 512 --prompt-style naive \
      --verbose --prune-keep 2 --ignore-cache --parallel  # quarter





# export CUDA_VISIBLE_DEVICES=6,7
# export CUDA_LAUNCH_BLOCKING=1
# python main.py --output-dir ${output_dir} \
#       --dataset gsim --task fine_tune --n-epochs 10 --do-train \
#       --style dataset --model gpt --size small --num-shots full \
#       --batch-size 16 --grad-accum-steps 2\
#       --learning-rate 3e-5  --maximum-len 512 --prompt-style naive \
#       --verbose --prune-keep 2 --fp16 --trainer --parallel  # quarter

# export CUDA_VISIBLE_DEVICES=6
# deepspeed main.py --output-dir ${output_dir} \
#       --dataset gsim --task fine_tune --n-epochs 10 --do-train \
#       --style dataset --model gpt --size small --num-shots full --batch-size 16 \
#       --learning-rate 3e-5  --maximum-len 512 --prompt-style naive \
#       --verbose --prune-keep 2 --fp16 --trainer \
#       --deepspeed ds_config.json


# # # # # lab
# output_dir='results'
# export CUDA_VISIBLE_DEVICES=1,6
# python main.py --output-dir ${output_dir} \
#       --dataset mwoz --task fine_tune --n-epochs 10 --do-train \
#       --style dataset --model gpt --size medium --num-shots full \
#       --batch-size 4 --grad-accum-steps 8\
#       --learning-rate 3e-5  --maximum-len 512 --prompt-style naive \
#       --verbose --prune-keep 2 --fp16 --trainer --ignore-cache  --parallel #--ignore-cache # --parallel  # quarter

# # export CUDA_VISIBLE_DEVICES=7
# # python main.py --output-dir ${output_dir} \
# #       --dataset mwoz --task fine_tune --n-epochs 10 --do-train \
# #       --style dataset --model gpt --size small --num-shots full --batch-size 16 \
# #       --learning-rate 3e-5  --maximum-len 512 --prompt-style naive \
# #       --verbose --prune-keep 2 --fp16 --trainer #--ignore-cache  # quarter







# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
# python main.py \
#       --dataset mwoz --task fine_tune --n-epochs 10 --do-train \
#       --style dataset --model gpt --size large --num-shots full --batch-size 4 \
#       --learning-rate 1e-5  --maximum-len 512 --prompt-style naive \
#       --verbose --prune-keep 2 --fp16 --fp16_backend 'amp' --ignore-cache --parallel

# # # deepspeed test.py
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# deepspeed main.py \
#       --dataset mwoz --task fine_tune --n-epochs 10 --do-train \
#       --style dataset --model gpt --size medium --num-shots full --batch-size 16 \
#       --learning-rate 3e-6  --maximum-len 512 --prompt-style naive \
#       --verbose --prune-keep 2 --fp16 --fp16_backend 'amp' \
#       --deepspeed ds_config.json

# # print(args.world_size)


# deepspeed main.py \
#       --dataset sgd --task fine_tune --n-epochs 10 --do-train --do-save \
#       --style dataset --model gpt --size small --num-shots full --batch-size 16 \
#       --learning-rate 3e-5  --maximum-len 512 --prompt-style naive \
#       --verbose --prune-keep 2 \
#       --deepspeed ds_config.json

# # # # lab
# export CUDA_VISIBLE_DEVICES=5,6
# # python main.py \
# #       --dataset sgd --task fine_tune --n-epochs 10 --do-train --do-save \
# #       --style dataset --model gpt --size small --num-shots full \
# #       --learning-rate 3e-5  --maximum-len 512 --prompt-style naive --batch-size 4 \
# #       --verbose --prune-keep 2 --parallel

# deepspeed main.py \
#       --dataset sgd --task fine_tune --n-epochs 10 --do-train --do-save \
#       --style dataset --model gpt --size medium --num-shots full \
#       --learning-rate 3e-5  --maximum-len 512 --prompt-style naive \
#       --verbose --prune-keep 2 --fp16 --fp16_backend 'amp' \
#       --deepspeed ds_config.json

# python main.py \
#       --dataset sgd --task fine_tune --n-epochs 10 --do-train --do-save \
#       --style dataset --model gpt --size large --num-shots full --batch-size 16 \
#       --learning-rate 3e-5  --maximum-len 512 --prompt-style naive \
#       --verbose --prune-keep 2 --parallel  # quarter --ignore-cache

# export CUDA_VISIBLE_DEVICES=3
# deepspeed main.py --num_gpus=1 --fp16 \
#       --dataset sgd --task fine_tune --n-epochs 10 --do-train --do-save \
#       --style dataset --model gpt --size medium --num-shots full --batch-size 16 \
#       --learning-rate 3e-5  --maximum-len 512 --prompt-style naive \
#       --verbose --prune-keep 2 \
#       --deepspeed ds_config_gptj.json




# python main.py --dataset sgd --task fine_tune --style dataset --n-epochs 10 --do-eval --qualify \
#       --model gpt --size small --num-shots full --maximum-length 512 --prompt-style none \
#       --learning-rate 1e-4 --verbose --context-length 2 --batch-size 16


# # # # in_context
# # ['abcd', 'dstc', 'gsim', 'mwoz', 'sgd', 'tt']
# python main.py --dataset dstc --task in_context --style dataset --do-eval --seed 14 \
#       --model gpt --size small --num-shots full --maximum-length 512 --prompt-style naive \
#       --temperature 0.8 --verbose --context-length 2 --batch-size 4 --debug --ignore-cache
