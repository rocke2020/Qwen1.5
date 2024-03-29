# 
gpu=$1
if [ -z $gpu ]; then
    gpu=0
fi
# export CUDA_VISIBLE_DEVICES=$gpu

GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
DIR=`pwd`
echo "DIR: $DIR"

file=app/quick_start/a1.py
# nohup python $file \
#     > $file.log 2>&1 &
# python $file \
#     2>&1  </dev/null | tee $file.log