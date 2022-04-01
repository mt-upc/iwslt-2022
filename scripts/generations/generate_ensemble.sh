#!/bin/bash

test_set=$1     # IWSLT.tst2020, IWSLT.tst2019
tgt_lang=$2
inference_times=$3
max_segment_length=$4
mult=$5
ckpts=$6
name=$7
device_id=$8

if [ $tgt_lang == "de" ]; then
  tokenizer=13a
elif [ $tgt_lang == "ja" ]; then
  tokenizer=ja-mecab
elif [ $tgt_lang == "zh" ]; then
  tokenizer=zh
fi

if [[ $test_set == *"IWSLT"* || $test_set == tst-COMMON-${tgt_lang} ]]; then

    CUDA_VISIBLE_DEVICES=$device_id \
    bash ${IWSLT_ROOT}/scripts/segmentation/eval.sh \
        $ckpts \
        $test_set \
        en \
        $tgt_lang \
        $inference_times \
        $max_segment_length \
        $max_segment_length \
        $mult \
        $name \
        $device_id

else

    echo "Generating translations for "${test_set}
    CUDA_VISIBLE_DEVICES=$device_id \
    fairseq-generate ${DATA_ROOT}/en-${tgt_lang} \
        --path $ckpts \
        --task speech_to_text \
        --gen-subset ${test_set}_mustc \
        --seed 48151623 \
        --prefix-size 1 \
        --scoring sacrebleu \
        --max-target-positions 1024 \
        --max-tokens 1_200_000 \
        --max-source-positions 1_200_000 \
        --beam 5 \
        --sacrebleu-tokenizer $tokenizer > ${SAVE_DIR}/final_checkpoints/${name}.out 2>&1

fi
