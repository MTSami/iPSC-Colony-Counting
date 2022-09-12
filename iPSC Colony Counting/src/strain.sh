# module load Anaconda3
# module load cuda10.0/toolkit
# conda activate object-locator

python -m object-locator.train \
       --train-dir "./data/512_512/images/train" \
       --val-dir "./data/512_512/images/val" \
       --save "./saved_model.ckpt" \
       --optimizer "sgd" \
       --batch-size 8 \
       --epochs 500 \
       --val-freq 25
