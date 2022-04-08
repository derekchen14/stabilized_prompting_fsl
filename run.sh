# ________ Fine-tuned Model Training ________
# Training with all available data
# python main.py --dataset mwoz --task fine_tune --style dataset --do-train --do-save \
#       --model t5 --size small --num-shots full --maximum-len 1024 --prompt-style none \
#       --context-len 4 --batch-size 8 --log-interval 1200 --learning-rate 1e-4 --n-epochs 7 #--ignore-cache
# python main.py --dataset abcd --task fine_tune --n-epochs 7 --do-train --debug \
#       --style dataset --model gpt --size small --num-shots full --batch-size 6 \
#       --learning-rate 1e-4  --maximum-len 512 --prompt-style naive --context-len 5
python main.py --dataset tt --task fine_tune --n-epochs 7 --do-train --debug \
      --style dataset --model gpt --size small --num-shots full --batch-size 8 \
      --learning-rate 1e-4  --maximum-len 512 --prompt-style naive --ignore-cache
# python main.py --dataset dstc --task fine_tune --n-epochs 7 --do-train --do-save \
#       --style dataset --model gpt --size small --num-shots full --batch-size 8 \
#       --learning-rate 1e-4  --maximum-len 512 --prompt-style naive
# python main.py --dataset mwoz --task fine_tune --n-epochs 100 --do-train --debug \
#       --style dataset --model trade --num-shots full \
#       --learning-rate 1e-4  --prompt-style naive --context-len 9 \
#       --batch-size 64 --log-interval 1200 --prune-keep 2 --ignore-cache

# Leveraging Slot Descriptions for Zero-Shot Cross-Domain DST (domain held out for testing)
# python main.py --dataset mwoz22 --task fine_tune --style domain --do-train --do-save \
#       --model t5 --size small --num-shots zero --maximum-len 512 --prompt-style human \
#       --temperature 0.8 --threshold 1.4 --context-len 8
# python main.py --dataset mwoz --task fine_tune --n-epochs 7 --do-train --debug \
#       --style domain --left-out hotel --model trade --size small --num-shots few \
#       --learning-rate 1e-4  --maximum-len 1024 --prompt-style naive --context-len 9 \
#       --batch-size 64 --log-interval 1200 --prune-keep 2
# python main.py --dataset mwoz22 --task fine_tune --n-epochs 7 --do-train --debug \
#       --style domain --left-out hotel --model bart --size small --num-shots zero \
#       --learning-rate 1e-4  --maximum-len 1024 --prompt-style naive --context-len 9 \
#       --batch-size 8 --log-interval 1200 --prune-keep 2
# python main.py --dataset mwoz22 --task fine_tune --style domain --do-train --do-save \
#       --model t5 --size small --num-shots zero --maximum-len 512 --prompt-style slotval \
#       --temperature 0.8 --threshold 1.4 --context-len 8
# python main.py --dataset mwoz22 --task fine_tune --style domain --do-train --do-save \
#       --model t5 --size small --num-shots zero --maximum-len 512 --prompt-style question \
#       --temperature 0.8 --threshold 1.4 --context-len 8

# Zero-Shot DST via Cross-Task Transfer (dataset is held out for testing)
# python main.py --dataset mwoz --task fine_tune --style dataset --do-train --debug \
#       --model t5 --size small --num-shots percent --threshold 0.01 --prompt-style naive \
#       --maximum-len 512 --temperature 0.8 --threshold 1.4 --context-len 8
# python main.py --dataset mwoz --task fine_tune --style dataset --do-train --debug \
#       --model t5 --size small --num-shots percent --threshold 0.05 --prompt-style naive \
#       --maximum-len 512 --temperature 0.8 --threshold 1.4 --context-len 8
# python main.py --dataset mwoz --task fine_tune --style dataset --do-train --debug \
#       --model t5 --size small --num-shots percent --threshold 0.10 --prompt-style naive \
#       --maximum-len 512 --temperature 0.8 --threshold 1.4 --context-len 8

# ________ In-context Learning, without Back-propogation ________
# >> ICL Baseline
# python main.py --dataset sgd --task in_context --style domain --do-eval --seed 15 \
#       --model gpt --size large --num-shots full --qualify --maximum-len 1020 --context-len 14 \
#       --threshold 1.4   --temperature 0.8 --prompt-style statement
# python main.py --dataset sgd --task in_context --style dataset --do-eval --seed 14 \
#       --model gpt --size small --num-shots full --maximum-length 512 --prompt-style naive \
#       --temperature 0.8 --verbose --context-length 2 --ignore-cache --batch-size 4
# python main.py --dataset mwoz --task in_context --style domain --do-eval --seed 15 \
#       --model gpt --size small --num-shots full --maximum-len 512 --prompt-style schema \
#       --temperature 0.8 --threshold 1.4 --context-len 3 --left-out hotel

# ________ Meta-Stabilize Pre-training Mode ___________
# >> Our System
# python main.py --dataset mwoz --task meta_learn --style domain --do-train --seed 15 \
#       --model gpt --size large --num-shots percent --maximum-len 1020 --prompt-style schema
# python main.py --dataset sgd --task meta_learn --style dataset --do-train --debug \
#       --n-epochs 3 --learning-rate 1e-5 --model roberta --prune-keep 3 --batch-size 4
# python main.py --dataset mwoz --task meta_learn --n-epochs 3 --do-train --debug \
#       --style domain --left-out hotel --model gpt --size small --num-shots 0 \
#       --learning-rate 1e-5  --prune-keep 3 --batch-size 4 --log-interval 800
# python main.py --dataset mwoz --task meta_learn --n-epochs 3 --do-train --debug \
#       --style dataset --left-out mwoz --model gpt --size small --num-shots 0 \
#       --learning-rate 1e-5  --prune-keep 3 --batch-size 4 --log-interval 800

# ______________ Special Modes ________________
# >> Interactive Mode
# python main.py --dataset mwoz --task generate --style domain --do-interact --seed 30 \
#       --model gpt --size medium --batch-size 7 --num-shots zero --threshold 1.4 \
#       --temperature 1.2
# >> Evaluation Mode
# python main.py --dataset sgd --task track --style dataset --do-eval \
#       --model t5 --size small --num-shots few --prompt-style naive \
#       --maximum-len 512 --temperature 0.8 --threshold 1.4 --context-len 8
