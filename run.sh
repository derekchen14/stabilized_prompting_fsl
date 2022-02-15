# ________ Fine-tuned Model Training ________
# Leveraging Slot Descriptions for Zero-Shot Cross-Domain DST (domain held out for testing)
# python main.py --dataset mwoz22 --task fine_tune --style domain --do-train --do-save \
#       --model t5 --size small --num-shots zero --max-len 512 --prompt-style human \
#       --temperature 0.8 --threshold 1.4 --context-len 8
python main.py --dataset mwoz22 --task fine_tune --style domain --do-train --do-save \
      --model t5 --size small --num-shots zero --max-len 1024 --prompt-style naive \
      --context-len 9 --batch-size 8 --log-interval 1200 --learning-rate 1e-4 --n-epochs 7
# python main.py --dataset mwoz22 --task fine_tune --style domain --do-train --do-save \
#       --model t5 --size small --num-shots zero --max-len 512 --prompt-style slotval \
#       --temperature 0.8 --threshold 1.4 --context-len 8
# python main.py --dataset mwoz22 --task fine_tune --style domain --do-train --do-save \
#       --model t5 --size small --num-shots zero --max-len 512 --prompt-style question \
#       --temperature 0.8 --threshold 1.4 --context-len 8

# Zero-Shot DST via Cross-Task Transfer (dataset is held out for testing)
# python main.py --dataset mwoz22 --task fine_tune --style dataset --do-train --debug \
#       --model t5 --size small --num-shots percent --threshold 0.01 --prompt-style naive \
#       --max-len 512 --temperature 0.8 --threshold 1.4 --context-len 8
# python main.py --dataset mwoz22 --task fine_tune --style dataset --do-train --debug \
#       --model t5 --size small --num-shots percent --threshold 0.05 --prompt-style naive \
#       --max-len 512 --temperature 0.8 --threshold 1.4 --context-len 8
# python main.py --dataset mwoz22 --task fine_tune --style dataset --do-train --debug \
#       --model t5 --size small --num-shots percent --threshold 0.10 --prompt-style naive \
#       --max-len 512 --temperature 0.8 --threshold 1.4 --context-len 8

# ________ In-context Learning, without Back-propogation ________
# >> ICL Baseline
# python main.py --dataset sgd --task in_context --style domain --do-eval --seed 15 \
#       --model gpt --size large --num-shots few --qualify --max-len 1020 --context-len 4 \
#       --threshold 1.4   --temperature 0.8 --prompt-style statement
# python main.py --dataset mwoz --task in_context --style domain --do-eval --seed 15 \
#       --model gpt --size small --num-shots few --max-len 512 --prompt-style schema \
#       --temperature 0.8 --threshold 1.4 --context-len 8

# ________ Meta-Stabilize Pre-training Mode ___________
# >> Our System
# python main.py --dataset mwoz --task meta_learn --style domain --do-train --seed 15 \
#       --model gpt --size large --num-shots percent --max-len 1020 --prompt-style schema
# python main.py --dataset abcd --task meta_learn --style subflows --do-train --debug \
#       --n-epochs 3 --learning-rate 1e-5 --model roberta --prune-keep 3 --batch-size 4

# ______________ Special Modes ________________
# >> Interactive Mode
# python main.py --dataset mwoz --task generate --style domain --do-interact --seed 30 \
#       --model gpt --size medium --batch-size 7 --num-shots zero --threshold 1.4 \
#       --temperature 1.2
# >> Evaluation Mode
# python main.py --dataset sgd --task track --style dataset --do-eval \
#       --model t5 --size small --num-shots few --prompt-style naive \
#       --max-len 512 --temperature 0.8 --threshold 1.4 --context-len 8
