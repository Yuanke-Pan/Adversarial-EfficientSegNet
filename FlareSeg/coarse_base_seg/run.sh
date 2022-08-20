# !/bin/bash -e
 export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
 export CUDA_VISIBLE_DEVICES=3,4
 nohup python run.py>diceloss.out &