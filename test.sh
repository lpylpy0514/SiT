# about your tracker
script="ostrack"
config="s0_s0_resolution_16_16"
num_gpus=2
num_thread=8

# evaluation for got10k
python tracking/test.py $script $config --dataset got10k_test --threads $num_thread --num_gpus $num_gpus
python lib/test/utils/transform_got10k.py --tracker_name $script --cfg_name $config
# evaluation for trackingnet
python tracking/test.py $script $config --dataset trackingnet --threads $num_thread --num_gpus $num_gpus
python lib/test/utils/transform_trackingnet.py --tracker_name $script --cfg_name $config
# evaluation for lasot
python tracking/test.py $script $config --dataset lasot --threads $num_thread --num_gpus $num_gpus
python tracking/analysis_results.py