#!/bin/bash
set -xue
# ________ Fine-tuned Model Training ________
# Training with all available data
# python main.py --dataset dstc --task fine_tune --style dataset --do-train --do-save \
#       --model t5 --size small --num-shots full --maximum-len 512 --prompt-style naive \
#       --prune-keep 3 --log-interval 800 --batch-size 16 --n-epochs 7 --learning-rate 1e-4 \
#       --grad-accum-steps 4 --eval-interval half --verbose --filter --ignore-cache #  --percent 0.8
# python main.py --dataset sgd --task fine_tune --n-epochs 7 --do-train --debug \
#       --style dataset --model gpt --size small --num-shots full --batch-size 9 \
#       --learning-rate 1e-4  --maximum-len 512 --prompt-style naive # --ignore-cache
# python main.py --dataset mwoz --task fine_tune --n-epochs 7 --do-train \
#      --style dataset --model t5 --size large --num-shots full --batch-size 12 \
#      --learning-rate 1e-3  --maximum-len 512 --prompt-style schema \
#      --log-interval 1800 --prune-keep 2  --grad-accum-steps 10 --parallel
# python main.py --dataset abcd --task fine_tune --n-epochs 7 --do-train --do-save --verbose \
#       --model t5 --size small --num-shots full --batch-size 4 --context-len 2 --qualify \
#       --learning-rate 1e-5  --maximum-len 512 --prompt-style naive --seed 12 \
#       --log-interval 800 --eval-interval half # quarter

# Finetune the Sentence Transformers model from SBERT
# python contrast.py --batch-size 64 --kappa 20 --n-epochs 21 --num-shots five --seed 14 \
#        --log-interval 200 --checkpoint-interval 1400 --loss-function cosine --do-save \
#        --learning-rate 1e-5 --qualify --do-train
# python contrast.py --learning-rate 1e-5 --kappa 30 --n-epochs 21 --num-shots five \
#        --batch-size 64 --log-interval 120 --loss-function contrast --do-train \
#        --checkpoint-interval 700 --do-save --qualify --seed 12
# python contrast.py --learning-rate 3e-5 --kappa 20 --n-epochs 14 --num-shots five \
#        --batch-size 64 --log-interval 200 --loss-function custom --do-train \
#        --checkpoint-interval 1200 --qualify --seed 16 --do-save
# python contrast.py --batch-size 128 --kappa 20 --num-shots five --seed 15 \
#       --loss-function contrast --do-eval --learning-rate 3e-5 --qualify

# Leveraging Slot Descriptions for Zero-Shot Cross-Domain DST (domain held out for testing)
# python main.py --dataset mwoz --task fine_tune --n-epochs 7 --do-train --debug \
#       --style domain --left-out hotel --model trade --size small --num-shots few \
#       --learning-rate 1e-4  --maximum-len 1024 --prompt-style naive --context-len 9 \
#       --batch-size 64 --log-interval 1200 --prune-keep 2
# Zero-Shot DST via Cross-Task Transfer (dataset is held out for testing)
# python main.py --dataset mwoz --task fine_tune --style dataset --do-train --debug \
#       --model t5 --size small --num-shots percent --threshold 0.01 --prompt-style naive \
#       --maximum-len 512 --temperature 0.8 --threshold 1.4 --context-len 8

# ________ In-context Learning, without Back-propogation ________
# >> ICL Baseline
# python main.py --dataset sgd --task in_context --style domain --do-eval --seed 15 \
#       --model gpt --size large --num-shots full --qualify --maximum-len 1020 --context-len 14 \
#       --threshold 1.4   --temperature 0.8 --prompt-style statement
# python main.py --dataset sgd --task in_context --style dataset --do-eval --seed 14 \
#       --model gpt --size small --num-shots full --maximum-length 512 --prompt-style naive \
#       --temperature 0.8 --verbose --context-length 2 --ignore-cache --batch-size 4
# python main.py --dataset mwoz --task in_context --style dataset --do-eval --seed 15 \
#       --model gpt --size small --num-shots five --maximum-len 1024 --prompt-style naive \
#       --temperature 0.8 --threshold 1.4 --context-len 2 --left-out mwoz --ignore-cache \
#       --batch-size 3 --search cosine --quantify --debug --eval-interval quarter

# ________ Meta-Stabilize Pre-training Mode ___________
# >> Our System
# python main.py --dataset mwoz --task meta_learn --n-epochs 8 --do-train --do-save \
#       --style domain --left-out hotel --model gpt --size medium --num-shots one \
#       --learning-rate 1e-5 --batch-size 6 --prompt-style schema  --seed 15
# python main.py --dataset mwoz --task meta_learn --n-epochs 5 --do-train --do-save \
#      --left-out mwoz --model t5 --size small --num-shots five --chunk-per-epoch 8 \
#      --learning-rate 3e-4 --prompt-style schema --batch-size 4 --log-interval 1600 \
#      --eval-interval half --do-leave --qualify --grad-accum-steps 8 --filter \
#      --do-leave --kappa 30 --search cosine --use-tuned # parallel
# python main.py --dataset mwoz --task meta_learn --n-epochs 7 --do-train --do-save \
#      --left-out mwoz --model t5 --size small --num-shots five --prompt-style statement \
#      --learning-rate 3e-4 --batch-size 16 --grad-accum-steps 8 --log-interval 1200 \
#      --eval-interval half --checkpoint-interval 4000  --do-leave
# python main.py --dataset sgd --task meta_learn --n-epochs 7 --do-train --debug \
#       --style dataset --left-out sgd --model gpt --size small --num-shots five \
#       --learning-rate 3e-5 --prompt-style naive --batch-size 4 --log-interval 1200 \
#       --percent 0.7 --verbose --qualify --eval-interval quarter
# >> Pre-train without ICL
python main.py --dataset mwoz --task pre_train --n-epochs 5 --do-train --do-save \
     --left-out mwoz --model t5 --size small --num-shots five --chunk-per-epoch 8 \
     --learning-rate 3e-4 --prompt-style schema --batch-size 4 --log-interval 1600 \
     --eval-interval half --do-leave --qualify --grad-accum-steps 8 --filter \
     --do-leave --kappa 30 --search cosine --use-tuned # parallel

# ______________ Special Modes ________________
# >> Interactive Mode
# python main.py --dataset mwoz --task generate --style domain --do-interact --seed 30 \
#       --model gpt --size medium --batch-size 7 --num-shots zero --threshold 1.4 \
#       --temperature 1.2
# >> Evaluation Mode  # notice how prompt style matches the checkpoint
# python main.py --dataset mwoz --task meta_learn --do-eval --num-shots five --context-len 2  \
#       --model gpt --size small --maximum-len 1024 --batch-size 8 --left-out mwoz --verbose \
#       --prompt-style statement --quantify --checkpoint statement_lr1e-05_clen2_epoch7_acc883.pt
# CUDA_VISIBLE_DEVICES=1  python main.py --dataset mwoz --task meta_learn --do-eval --num-shots five \
#       --model t5 --size medium --maximum-len 512 --batch-size 8 --left-out mwoz --verbose \
#       --prompt-style schema --quantify --checkpoint naive_lr1e-05_clen2_epoch6_acc746.pt \
#       --kappa 30 --learning-rate 3e-5 --use-tuned --loss-function cosine --filter
# python main.py --dataset mwoz --task meta_learn --do-eval --num-shots five --filter --parallel \
#       --model t5 --size large --maximum-len 512 --batch-size 8 --left-out mwoz --verbose \
#       --prompt-style naive --quantify --checkpoint naive_lr0.0003_filter_epoch1_acc828.pt \
#       --eval-interval tenth --kappa 30 --learning-rate 3e-5 --use-tuned --loss-function cosine
# python main.py --dataset mwoz --task fine_tune --do-eval --context-len 2 --batch-size 16 \
#       --model gpt --size small --maximum-len 512 --prompt-style naive --eval-interval quarter \
#       --quantify --qualify --verbose  --checkpoint naive_epoch10_lr1e-05_clen2_acc522.pt
