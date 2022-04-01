#!/bin/bash

source ~/.bashrc
conda activate shas

export SHAS_ROOT=~/repos/SHAS

test_set=$1 #IWSLT.tst2020
inference_times=$2
len_low=$3
len_high=$4
mult=$5     # 0 or 1

if [[ $mult == 0 ]]; then
    path_to_checkpoint=/home/usuaris/scratch/ioannis.tsiamas/SHAS_checkpoints/en_sfc_model_epoch-6.pt
    path_to_yamls=${IWSLT_DATA}/${test_set}/yamls
else
    path_to_checkpoint=/home/usuaris/scratch/ioannis.tsiamas/SHAS_checkpoints/mult_sfc_model_epoch-4.pt
    path_to_yamls=${IWSLT_DATA}/${test_set}/yamls_mult
fi

path_to_wavs=${IWSLT_DATA}/${test_set}/wavs

for max_segment_length in $(seq $len_low $len_high); do
    path_to_custom_segmentation_yaml=${path_to_yamls}/${max_segment_length}_${inference_times}.yaml

    if [ ! -f $path_to_custom_segmentation_yaml ]; then

        echo "------------------->"$path_to_custom_segmentation_yaml

        python ${SHAS_ROOT}/src/supervised_hybrid/segment.py \
        -wavs $path_to_wavs \
        -ckpt $path_to_checkpoint \
        -yaml $path_to_custom_segmentation_yaml \
        -max $max_segment_length \
        -n $inference_times
    fi
done