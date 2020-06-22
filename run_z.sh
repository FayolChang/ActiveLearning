#!/bin/sh

cd /root/workspace/ActiveLearning
export PYTHONPATH="."
export ROOT_DIR="root"

/root/anaconda3/envs/py36/bin/python vaal/main.py --per_gpu_batch_size 8

/root/anaconda3/envs/py36/bin/python batchbald_bx/debug_tst_data.py  --epoch_num 500
/root/anaconda3/envs/py36/bin/python batchbald_bx/tst_data.py

