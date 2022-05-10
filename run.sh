# ________ Fine-tuned Model Training ________
# Training with all available data
# python main.py --dataset mwoz --task fine_tune --style dataset --do-train --do-save \
#       --model gpt --size small --num-shots full --maximum-len 512 --prompt-style naive \
#       --prune-keep 3 --log-interval 900 --context-len 2 --batch-size 12 --n-epochs 7 \
#       --learning-rate 3e-5 --seed 15 # --ignore-cache # --percent 0.4 --eval-interval quarter --qualify 
      # --ignore-cache # --verbose
# python main.py --dataset sgd --task fine_tune --n-epochs 7 --do-train --debug \
#       --style dataset --model gpt --size small --num-shots full --batch-size 9 \
#       --learning-rate 1e-4  --maximum-len 512 --prompt-style naive # --ignore-cache
python main.py --dataset dstc --task fine_tune --n-epochs 7 --do-train --do-save \
      --style dataset --model gpt --size small --num-shots full --batch-size 18 \
      --learning-rate 2e-5  --maximum-len 512 --prompt-style naive --verbose \
      --log-interval 700 --prune-keep 3 --seed 11 --ignore-cache # --eval-interval half
# python main.py --dataset abcd --task fine_tune --n-epochs 7 --do-train --do-save \
#       --style dataset --model gpt --size small --parallel --num-shots full --batch-size 12 \
#       --learning-rate 3e-5  --maximum-len 512 --prompt-style naive --log-interval 900
# python main.py --dataset tt --task fine_tune --n-epochs 7 --do-train --debug \
#       --style dataset --model gpt --size small --num-shots full --batch-size 12 \
#       --learning-rate 1e-4  --maximum-len 512 --prompt-style naive --ignore-cache

# Finetune the Sentence Transformers model from SBERT
# python contrast.py --learning-rate 3e-5 --kappa 10 --finetune icdst --num-shots five \
#       --batch-size 32 --n-epochs 7 --seed 21 --log-interval 900

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
#       --model gpt --size small --num-shots five --maximum-len 1024 --prompt-style statement \
#       --temperature 0.8 --threshold 1.4 --context-len 3 --left-out mwoz \
#       --batch-size 3 --search cosine --quantify --debug # --parallel  # --ignore-cache

# ________ Meta-Stabilize Pre-training Mode ___________
# >> Our System
# python main.py --dataset mwoz --task meta_learn --style domain --do-train --seed 15 \
#       --model gpt --size large --num-shots one --maximum-len 1020 --prompt-style schema
# python main.py --dataset mwoz --task meta_learn --n-epochs 3 --do-train --debug \
#       --style domain --left-out hotel --model gpt --size small --num-shots ten \
#       --learning-rate 1e-5  --prune-keep 3 --batch-size 4 --log-interval 800
# python main.py --dataset mwoz --task meta_learn --n-epochs 3 --do-train --debug \
#       --style dataset --left-out mwoz --model gpt --size small --num-shots five \
#       --learning-rate 1e-5  --prompt-style naive --batch-size 4 --context-len 3

# ______________ Special Modes ________________
# >> Interactive Mode
# python main.py --dataset mwoz --task generate --style domain --do-interact --seed 30 \
#       --model gpt --size medium --batch-size 7 --num-shots zero --threshold 1.4 \
#       --temperature 1.2
# >> Evaluation Mode
# python main.py --dataset dstc --task fine_tune --do-eval --context-len 2 --batch-size 16 \
#       --model gpt --size small --num-shots full --maximum-len 512 --prompt-style naive \
#       --checkpoint naive_epoch10_lr1e-05_clen2_acc522.pt --quantify

