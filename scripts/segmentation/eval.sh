#!/bin/bash

ckpt_file=$1
test_set=$2     # IWSLT.tst2020, IWSLT.tst2019
src_lang=$3
tgt_lang=$4     # de, ja, zh
inference_times=$5
len_low=$6
len_high=$7
mult=$8     # 0 or 1
ensemble=$9  # 0 or name

if [[ $ensemble == 0 ]]; then
    exp_name=$(basename $(dirname $(dirname $ckpt_file)))
    ckpt_name=$(basename $ckpt_file .pt)
else
    exp_name=$ensemble
    ckpt_name=ckpt
fi

path_to_wavs=${IWSLT_DATA}/${test_set}/wavs

if [[ $mult == 0 ]]; then
    path_to_custom_segmentations=${IWSLT_DATA}/${test_set}/yamls
    path_to_custom_datasets=${IWSLT_DATA}/${test_set}/tsvs
    path_to_translations=${IWSLT_DATA}/${test_set}/translations/${exp_name}/${ckpt_name}
    path_to_scores=${IWSLT_DATA}/${test_set}/bleu/${exp_name}/${ckpt_name}
else
    path_to_custom_segmentations=${IWSLT_DATA}/${test_set}/yamls_mult
    path_to_custom_datasets=${IWSLT_DATA}/${test_set}/tsvs_mult
    path_to_translations=${IWSLT_DATA}/${test_set}/translations_mult/${exp_name}/${ckpt_name}
    path_to_scores=${IWSLT_DATA}/${test_set}/bleu_mult/${exp_name}/${ckpt_name}
fi

mkdir -p $path_to_custom_datasets $path_to_translations $path_to_scores

if [[ $tgt_lang == "de" ]]; then
  tokenizer=13a
elif [[ $tgt_lang == "ja" ]]; then
  tokenizer=ja-mecab
elif [[ $tgt_lang == "zh" ]]; then
  tokenizer=zh
fi

if [[ $test_set == IWSLT.tst2022 ]]; then
    file_order_name=FILE_ORDER_${tgt_lang}
else
    file_order_name=FILE_ORDER
fi

for max_segment_length in $(seq $len_low $len_high); do

    segm_yaml=${path_to_custom_segmentations}/${max_segment_length}_${inference_times}.yaml
    segm_params=${max_segment_length}_${inference_times}

    if [ ! -f $segm_yaml ]; then
        continue
    fi

    if [ -f ${path_to_scores}/${segm_params}.json ]; then
        continue
    fi

    echo "--------------------->"${segm_yaml}

    path_to_tsv=${path_to_custom_datasets}/${segm_params}.tsv
    # if [ ! -f $path_to_tsv ]; then
    if [ ! -f ${path_to_custom_datasets}/flac_${segm_params}.zip ]; then
        echo "Preparing custom dataset tsv"
        python ${IWSLT_ROOT}/scripts/segmentation/prep_custom_data.py \
            -yaml $segm_yaml \
            -wav $path_to_wavs \
            -order ${IWSLT_DATA}/${test_set}/${file_order_name} \
            -tsv $path_to_tsv \
            -lang $tgt_lang
    fi

    translations_dir=${path_to_translations}/${segm_params}
    mkdir -p $translations_dir

    out_file=${translations_dir}/fairseq_generate.out

    echo "Generating translations"
    fairseq-generate $path_to_custom_datasets \
        --path $ckpt_file \
        --data-config-yaml ${DATA_ROOT}/${src_lang}-${tgt_lang}/config.yaml \
        --task speech_to_text \
        --gen-subset $segm_params \
        --seed 48151623 \
        --prefix-size 1 \
        --scoring sacrebleu \
        --max-target-positions 1024 \
        --max-tokens 1_920_000 \
        --max-source-positions 1_500_000 \
        --beam 5 \
        --sacrebleu-tokenizer $tokenizer > $out_file 2>&1

    rm -r ${path_to_custom_datasets}/flac_${segm_params}.zip

    echo "Formatting translations"
    python ${IWSLT_ROOT}/scripts/segmentation/format_gen_output.py \
        -i $out_file \
        -o ${translations_dir}/raw_translations.txt

    if [[ $test_set == IWSLT.tst2021 || $test_set == IWSLT.tst2022 ]]; then
        continue
    fi

    # activate the secondary python2 env
    eval "$(conda shell.bash hook)"
    conda activate snakes27

    _set=$(echo $test_set| cut -d'.' -f 2)

    # align the hypotheses with the references
    echo "Aligning translations"
    bash ${MWERSEGMENTER_ROOT}/segmentBasedOnMWER.sh \
        ${IWSLT_DATA}/${test_set}/IWSLT.TED.${_set}.${src_lang}-${tgt_lang}.${src_lang}.xml \
        ${IWSLT_DATA}/${test_set}/IWSLT.TED.${_set}.${src_lang}-${tgt_lang}.${tgt_lang}.xml \
        ${translations_dir}/raw_translations.txt \
        ${exp_name}-${ckpt_name} \
        $tgt_lang \
        ${translations_dir}/aligned_translations.xml \
        normalize \
        1

    # re-activate main environment
    eval "$(conda shell.bash hook)"
    conda activate iwslt22

    python ${IWSLT_ROOT}/scripts/segmentation/score_translation.py \
        -ref ${translations_dir}/__mreference \
        -trans ${translations_dir}/__segments \
        -s ${path_to_scores}/${segm_params}.json \
        -l $tgt_lang
done
