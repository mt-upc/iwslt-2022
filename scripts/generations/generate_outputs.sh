#!/bin/sh

EXP_NAME=$1
SUBSET="dev_mustc"
TGT_LANG=$2

CKPTS_DIR=${SAVE_DIR}/${EXP_NAME}/ckpts

if [ ! -d "${CKPTS_DIR}" ]; then
  printf "Unexisting directory: ${CKPTS_DIR}\n"
  exit 0
fi

if [ $TGT_LANG == "de" ]; then
  tokenizer=13a
elif [ $TGT_LANG == "ja" ]; then
  tokenizer=ja-mecab
elif [ $TGT_LANG == "zh" ]; then
  tokenizer=zh
fi

for ckpt_file in $(ls -t ${CKPTS_DIR}/checkpoint*.pt); do
  if [ -f $ckpt_file ]; then
    out_file="${ckpt_file%.pt}__${SUBSET}.out"
    if [ ! -f "$out_file" ]; then
      printf "Generating outputs for: ${ckpt_file}\n"
      fairseq-generate ${DATA_ROOT}/en-${TGT_LANG}/ \
        --path ${ckpt_file} \
        --task speech_to_text \
        --gen-subset $SUBSET \
        --seed 48151623 \
        --prefix-size 1 \
        --scoring sacrebleu \
        --max-source-positions 400_000 \
        --max-target-positions 1024 \
        --max-tokens 1_920_000 \
        --beam 5 \
        --sacrebleu-tokenizer $tokenizer > $out_file 2>&1 && \
      printf "Generated outputs saved at: ${out_file}\n"
    fi
  fi
done