# 
gpu=$1
if [ -z $gpu ]; then
    gpu=0
fi
export CUDA_VISIBLE_DEVICES=$gpu

file=app/quick_start/a1.py
# nohup python $file \
#     > $file.log 2>&1 &
python $file \
    2>&1  </dev/null | tee $file.log