#!/bin/sh

cd /root/workspace/ActiveLearning
export PYTHONPATH="."
export ROOT_DIR="root"

/root/anaconda3/envs/py36/bin/python batchbald_bx/debug_tst_data.py
/root/anaconda3/envs/py36/bin/python batchbald_bx/tst_data.py

