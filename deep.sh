deepspeed main.py --dataset mwoz --task fine_tune --n-epochs 7 --do-train \
      --style dataset --model gpt --size medium --num-shots full --batch-size 4 \
      --learning-rate 1e-5  --maximum-len 512 --prompt-style naive --seed 27 \
      --verbose --prune-keep 2 --fp16 --fp16_backend 'amp' --deepspeed ds_config.json
