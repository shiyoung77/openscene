#!/bin/sh
set -x

#exp_dir=$1
#config=$2
#feature_type=$3

exp_dir="out/scannet_lseg"
config="config/scannet/ours_lseg_pretrained.yaml"
feature_type="ensemble"

mkdir -p ${exp_dir}
result_dir=${exp_dir}/result_eval

#export PYTHONPATH=.
#python -u run/evaluate.py \
#  --config=${config} \
#  feature_type ${feature_type} \
#  save_folder ${result_dir} \
#  2>&1 | tee -a ${exp_dir}/eval-$(date +"%Y%m%d_%H%M").log


export PYTHONPATH=.
python -u run/eval_instance_retrieval.py \
  --config=${config} \
  feature_type ${feature_type} \
  save_folder ${result_dir} \
  2>&1 | tee -a ${exp_dir}/eval-$(date +"%Y%m%d_%H%M").log
