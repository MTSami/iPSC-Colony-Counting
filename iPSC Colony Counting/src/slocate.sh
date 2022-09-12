# module load Anaconda3
# module load cuda10.0/toolkit
# conda activate object-locator

python -m object-locator.locate \
       --dataset "./data/512_512/images/test" \
       --out "./output" \
       --model "./saved_model_400.ckpt" \
       --taus "-1"\
       --evaluate
