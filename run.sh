#!/bin/sh

export PYTHONPATH="."

#/home/lwty/anaconda3/envs/py36/bin/python active_learning/infer.py  --per_gpu_batch_size 8
/home/lwty/anaconda3/envs/py36/bin/python active_learning/train.py  --per_gpu_batch_size 8

