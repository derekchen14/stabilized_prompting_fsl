# ________ BASELINE TRAINING ACROSS TASKS ________

# Intent Classification (classify)
# python main.py --dataset dd --task clc --style topics --do-train --do-save \
#       --n-epochs 7 --learning-rate 1e-5 --model roberta --prune-keep 3 --batch-size 4
# python main.py --dataset ed --task clc --style emotions --do-train --debug \
#       --n-epochs 3 --learning-rate 1e-5 --model roberta --prune-keep 3 --batch-size 4
# python main.py --dataset abcd --task clc --style subflows --do-train --debug \
#       --n-epochs 3 --learning-rate 1e-5 --model roberta --prune-keep 3 --batch-size 4
# python main.py --dataset cdgc --task clc --style movies --do-train --debug \
#       --n-epochs 7 --learning-rate 1e-5 --model roberta --prune-keep 3 --batch-size 4

# Dialogue State Tracking (track)
python main.py --dataset sgd --task dst --style intents --do-train --debug \
      --n-epochs 3 --learning-rate 1e-5 --model bart --prune-keep 3 --batch-size 4
# python main.py --dataset gsim --task dst --style user_acts --ignore-cache --do-train --do-save \
#       --n-epochs 7 --learning-rate 1e-5 --model roberta --prune-keep 3 --batch-size 4
# python main.py --dataset mwoz --task dst --style apis --do-train --do-save \
#       --n-epochs 7 --learning-rate 1e-5 --model roberta --prune-keep 3 --batch-size 4
# python main.py --dataset tt --task dst --style apis --do-train --do-save \
#       --n-epochs 7 --learning-rate 1e-5 --model roberta --prune-keep 3 --batch-size 4

# Response Generation (generate)
# python main.py --dataset dd --task rg --style topics --do-train --seed 12 \
#       --n-epochs 7 --learning-rate 1e-5 --model gpt --prune-keep 2 --batch-size 2 --do-save
# python main.py --dataset ed --task rg --style emotions --ignore-cache --do-train --do-save \
#       --n-epochs 7 --learning-rate 1e-5 --model gpt --prune-keep 3 --batch-size 4
# python main.py --dataset midas --task rg --style dialog_acts --do-train --do-save \
#       --n-epochs 7 --learning-rate 1e-5 --model gpt --prune-keep 3 --batch-size 4
# python main.py --dataset mwoz --task rg --style slots --do-train --do-save \
#       --n-epochs 7 --learning-rate 1e-5 --model gpt --prune-keep 3 --batch-size 4


# ______________ Model Variations ________________
# python main.py --style dd --model gpt --style clc --do-train --debug \  # --do-save
#       --n-epochs 3 --learning-rate 1e-5
# python main.py --style dd --model bart --style clc --do-train --debug \  # --do-save
#       --n-epochs 3 --learning-rate 1e-5

# ______ Automatic Evaluation ______
# python main.py --style dd --model roberta --style topics --do-eval --quantify --task clc
# python main.py --domain airline --model roberta --do-eval --quantify --do-augment \
# 	--mixing single --task eda
# python main.py --domain airline --model roberta --do-eval --quantify --do-augment \
#	--mixing single --task translate
# python main.py --domain telco --model roberta --do-eval --quantify --do-augment \
# 		--mixing single --task decode
# (Evaluation of mixing styles)
# python main.py --domain airline --do-eval --quantify --do-augment --mixing top --task none
# python main.py --domain airline --do-eval --quantify --do-augment --mixing cat --task none
# python main.py --domain airline --do-eval --quantify --do-augment --mixing all --task none
# python main.py --domain airline --do-eval --quantify --do-augment --mixing hs --task none
# (Generate CSV for Human Evaluation)
# python extract.py  # change the domain in the file 
