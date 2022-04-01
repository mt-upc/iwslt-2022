#!/bin/bash

test_set=$1     # IWSLT.tst2020, IWSLT.tst2019, ...
tgt_lang=$2
inference_times=$3
max_segment_length=$4
mult=$5

if [ $tgt_lang == "de" ]; then
  tokenizer=13a
elif [ $tgt_lang == "ja" ]; then
  tokenizer=ja-mecab
elif [ $tgt_lang == "zh" ]; then
  tokenizer=zh
fi

while read -r path_to_checkpoint; do

    echo $path_to_checkpoint 

    if [[ $test_set == *"IWSLT"* ]]; then

        bash ${IWSLT_ROOT}/scripts/segmentation/eval.sh \
            $path_to_checkpoint \
            $test_set \
            en \
            $tgt_lang \
            $inference_times \
            $max_segment_length \
            $max_segment_length \
            $mult \
            0

    else

        exp_name=$(basename $(dirname $(dirname $path_to_checkpoint)))
        ckpt_name=$(basename $path_to_checkpoint .pt)

        echo "Generating translations for "${test_set}
        fairseq-generate ${DATA_ROOT}/en-${tgt_lang} \
            --path $path_to_checkpoint \
            --task speech_to_text \
            --gen-subset ${test_set}_mustc \
            --seed 48151623 \
            --prefix-size 1 \
            --scoring sacrebleu \
            --max-target-positions 1024 \
            --max-tokens 1_920_000 \
            --max-source-positions 1_500_000 \
            --beam 5 \
            --sacrebleu-tokenizer $tokenizer > ${SAVE_DIR}/final_checkpoints/${exp_name}_${ckpt_name}_${test_set}.out 2>&1

    fi

done <${IWSLT_ROOT}/scripts/generations/best_checkpoints_list.txt 
