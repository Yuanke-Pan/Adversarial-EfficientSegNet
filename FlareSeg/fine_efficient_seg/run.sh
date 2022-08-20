# !/bin/bash -e
 export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
 export CUDA_VISIBLE_DEVICES=0,1,2,3
 nohup python run.py>diceloss.out&