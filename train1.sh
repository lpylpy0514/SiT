#!/bin/bash
# preprocess
source /root/anaconda3/bin/activate seqtrackv2
pip install wandb
pip install thop
apt-get install libturbojpeg
cd /18353470163/lpy/SiT
# about your tracker
script="ostrack"
config="s0_s0_resolution_16_4"
#config="spd_test"
num_gpus=1
num_thread=8

# training
python tracking/train.py --script $script --config $config --save_dir ./output --mode multiple --nproc_per_node $num_gpus --use_wandb 0

