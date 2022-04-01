## Data

```bash
export MUSTC_ROOT=...
export COVOST_ROOT=...
export EUROPARL_ROOT=...
```

### MUST-C

Download (add later)

```bash
...
```

Prepare data (without vocabulary)

(around 200 wav files are not found for chinese)

```bash
python examples/speech_to_text/prep_mustc_data.py \
--data-root /home/usuaris/veussd/DATABASES/speech_translation/MUSTC_v2.0_wav_16k \
--task st --use-audio-input --only-manifest --append-lang-id
```

### CoVoST

Download commonvoice version 8 and extract

```bash
wget -bqc https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-8.0-2022-01-19/cv-corpus-8.0-2022-01-19-en.tar.gz
```

Download covost tsvs

```bash
wget https://dl.fbaipublicfiles.com/covost/covost_v2.en_de.tsv.tar.gz
wget https://dl.fbaipublicfiles.com/covost/covost_v2.en_zh-CN.tsv.tar.gz
wget https://dl.fbaipublicfiles.com/covost/covost_v2.en_ja.tsv.tar.gz
```

Convert to 16khz, mono, .wav

```bash
export COVOST_ROOT=/home/usuaris/iaonnis.tsiamas/datasets/speech_translation/cv-corpus-8.0-2022-01-19/en
num_processors=24
mkdir -p ${COVOST_ROOT}/clips_mono_16k
for split in {train,dev,test}; do
    cat ${COVOST_ROOT}/${split}.tsv | cut -f2 | parallel -j $num_processors ffmpeg -i ${COVOST_ROOT}/clips/{} -ac 1 -ar 16000 -hide_banner -loglevel error ${COVOST_ROOT}/clips_mono_16k/{.}.wav
done
```

Prepare covost tsvs in fairseq format

```bash
for tgt_lang in {de,zh-CH,ja}; do
    python examples/iwslt22/scripts/prepare_covost_tsv.py \
    -d $COVOST_ROOT -s en -t $tgt_lang --append-lang-id
done
```

### Europarl-ST

Download Europarl-ST and extract

```bash

```

Prepare europarl tsvs in fairseq format
(TODO) adapt script to work with both wav and the original mp3/flac format

```bash
python examples/iwslt22/scripts/prepare_europarl.py -d ${EUROPARL_ROOT} --lang-pair en-de --task st --use-audio-input --only-manifest --append-lang-id
```

## Filtering

### ASR inference on the "train" sets

```bash
for tgt_lang in {de,ja,zh}; do
    python examples/iwslt22/scripts/filtering/asr_inference.py --tsv_path ~/datasets/speech_translation/MUSTC_v2.0_wav_16k/en-${tgt_lang}/train_asr.tsv -o ~/datasets/st_filtering/MUSTC_v2.0/en
done

for split in {train,dev,test}; do
    python examples/iwslt22/scripts/filtering/asr_inference.py --tsv_path /home/usuaris/veussd/ioannis.tsiamas/datasets/speech_translation/EuroparlST_wav_16k/en/en-de_dev_${split}.tsv -o ~/datasets/st_filtering/EuroparlST/en
done

for split in {train,dev,test}; do
    for tgt_lang in {de,ja,zh}; do
        python examples/iwslt22/scripts/filtering/asr_inference.py --tsv_path ~/datasets/speech_translation/cv-corpus-8.0-2022-01-19/en/CoVoST/en-${tgt_lang}/${split}_asr.tsv -o ~/datasets/st_filtering/CoVoST/en
    done
done
```

### create filtered tsvs

```bash
export DATASET_ROOT=/home/usuaris/veussd/DATABASES/speech_translation/IWSLT22/datasets
export ASR_INFER_ROOT=/home/usuaris/veussd/DATABASES/speech_translation/IWSLT22/asr_inference
export KD_ROOT=/home/usuaris/veussd/DATABASES/speech_translation/IWSLT22/knowledge_distillation
export MUSTC_ROOT=/home/usuaris/veussd/DATABASES/speech_translation/MUSTC_v2.0_wav_16k
export EUROPARL_ROOT=/home/usuaris/veussd/DATABASES/speech_translation/EuroparlST_wav_16k
export COVOST_ROOT=/home/usuaris/veussd/DATABASES/speech_translation/cv-corpus-8.0-2022-01-19/en/CoVoST
```

```bash
export DATA_ROOT=/home/usuaris/iaonnis.tsiamas/datasets/speech_translation/IWSLT22/data
export ASR_INFER_ROOT=/home/usuaris/iaonnis.tsiamas/datasets/speech_translation/IWSLT22/asr_inference
export KD_ROOT=/home/usuaris/iaonnis.tsiamas/datasets/speech_translation/IWSLT22/knowledge_distillation
export MUSTC_ROOT=/home/usuaris/iaonnis.tsiamas/datasets/speech_translation/MUSTC_v2.0_wav_16k
export EUROPARL_ROOT=/home/usuaris/iaonnis.tsiamas/datasets/speech_translation/EuroparlST_wav_16k
export COVOST_ROOT=/home/usuaris/iaonnis.tsiamas/datasets/speech_translation/cv-corpus-8.0-2022-01-19_old/en/CoVoST
```

```bash
for tgt_lang in {de,ja,zh}; do
    python examples/iwslt22/scripts/filtering/filter_tsv.py \
    -tsv ${MUSTC_ROOT}/en-${tgt_lang}/train_st.tsv \
    -p ${ASR_INFER_ROOT}/MUSTC_v2.0/en/train_asr_wer_results.json \
    -o ${MUSTC_ROOT}/en-${tgt_lang} \
    -par -wer 0.75
done

for split in {train,dev,test}; do
    python examples/iwslt22/scripts/filtering/filter_tsv.py \
    -tsv ${EUROPARL_ROOT}/en/en-de_${split}_st.tsv \
    -p ${ASR_INFER_ROOT}/EuroparlST/en/en-de_${split}_asr_wer_results.json \
    -o ${EUROPARL_ROOT}/en \
    -par -wer 0.75
done

for tgt_lang in {de,ja,zh}; do
    for split in {train,dev,test}; do
        python examples/iwslt22/scripts/filtering/filter_tsv.py \
        -tsv ${COVOST_ROOT}/en-${tgt_lang}/${split}_st.tsv \
        -p ${ASR_INFER_ROOT}/CoVoST/en/${split}_asr_wer_results.json \
        -o ${COVOST_ROOT}/en-${tgt_lang} \
        -par -wer 0.75
    done
done
```

### Make symbolic links

```bash
for tgt_lang in {ja,zh}; do
    ln -s ${MUSTC_ROOT}/en-${tgt_lang}/train_st_filtered.tsv ${DATA_ROOT}/en-${tgt_lang}/train_mustc.tsv
    ln -s ${MUSTC_ROOT}/en-${tgt_lang}/dev_st.tsv ${DATA_ROOT}/en-${tgt_lang}/dev_mustc.tsv
    ln -s ${MUSTC_ROOT}/en-${tgt_lang}/tst-COMMON_st.tsv ${DATA_ROOT}/en-${tgt_lang}/tst-COMMON_mustc.tsv
done

for tgt_lang in {ja,zh}; do
    for split in {train,dev,test}; do
        if [[ $split != train ]]; then
            ln -s ${COVOST_ROOT}/en-${tgt_lang}/${split}_st_filtered.tsv ${DATA_ROOT}/en-${tgt_lang}/train_${split}_covost.tsv
        else
            ln -s ${COVOST_ROOT}/en-${tgt_lang}/${split}_st_filtered.tsv ${DATA_ROOT}/en-${tgt_lang}/${split}_covost.tsv
        fi
    done
done

for split in {train,dev,test}; do
    if [[ $split != train ]]; then
        ln -s ${EUROPARL_ROOT}/en/en-de_${split}_st_filtered.tsv ${DATA_ROOT}/en-de/train_${split}_europarl.tsv
    else
        ln -s ${EUROPARL_ROOT}/en/en-de_${split}_st_filtered.tsv ${DATA_ROOT}/en-de/${split}_europarl.tsv
    fi
done
```

## Knowledge Distillation

```bash
for tgt_lang in {de,ja,zh}; do
    python examples/iwslt22/scripts/knowledge_distillation/prepare_corpus_for_kd.py \
    -d ${MUSTC_ROOT}/en-${tgt_lang} \
    -asr train_asr_filtered.tsv \
    -st train_st_filtered.tsv \
    -o ${KD_ROOT}/en-${tgt_lang}/mustc_train_st_f
done
#this is done

for split in {train,dev,test}; do
    python examples/iwslt22/scripts/knowledge_distillation/prepare_corpus_for_kd.py \
    -d ${EUROPARL_ROOT}/en \
    -asr en-de_${split}_asr_filtered.tsv \
    -st en-de_${split}_st_filtered.tsv \
    -o ${KD_ROOT}/en-de/europarl_${split}_st_f
done

for tgt_lang in {de,ja,zh}; do
    for split in {train,dev,test}; do
        python examples/iwslt22/scripts/knowledge_distillation/prepare_corpus_for_kd.py \
        -d ${COVOST_ROOT}/en-${tgt_lang} \
        -asr ${split}_asr_filtered.tsv \
        -st ${split}_st_filtered.tsv \
        -o ${KD_ROOT}/en-${tgt_lang}/covost_${split}_st_f
    done
done

```


## IWSLT DATA

```bash
for tgt_lang in {de,ja,zh}; do
    wget http://i13pc106.ira.uka.de/~jniehues/IWSLT-SLT/data/eval/en-${tgt_lang}/IWSLT-SLT.tst2022.en-${tgt_lang}.tgz
    tar -xvf IWSLT-SLT.tst2022.en-${tgt_lang}.tgz
    rm IWSLT-SLT.tst2022.en-${tgt_lang}.tgz
    cut -d' ' -f1 IWSLT.tst2022/IWSLT.TED.tst2022.en-${tgt_lang}.en.video_url > IWSLT.tst2022/FILER_ORDER.en-${tgt_lang}
done


```


conda activate iwslt22
export TOKENIZERS_PARALLELISM=true
cd ~/repos/fairseq-internal-iwslt22/ 
python examples/iwslt22/scripts/knowledge_distillation/prepare_corpus_for_kd.py -d ~/datasets/speech_translation/MUSTC_v2.0_wav_16k/en-de/ -asr train_asr.tsv -st train_st.tsv -o ~/datasets/knowledge_distillation/MUSTC_v2.0/de_train.json

fairseq-train ${DATASETS_ROOT}/en-de --train-subset train_mustc_st_f --valid-subset dev_mustc_st --save-dir /home/usuaris/veussd/DATABASES/speech_translation/MUSTC_v2.0_wav_16k/en-de --num-workers 1 --max-tokens 480000 --max-update 100000 --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy --arch s2t_transformer_s --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 1

/home/usuaris/veussd/gerard.ion.gallego/pretrained_models/mbart50.ft.1n/sentence.bpe.model
/home/usuaris/veussd/gerard.ion.gallego/pretrained_models/mbart50.ft.1n/dict.de_DE.txt

for f in ./*_asr*.tsv; do
    sed -e '1 s/$/\ttgt_lang/' -e '2,$ s/$/\ten/' < $f > ${f}.new && \
    mv ${f}.new $f
    echo $f
done

TGT_LANG=zh
for f in ./*_st*.tsv; do
    sed -e '1 s/$/\ttgt_lang/' -e '2,$ s/$/\t'"${TGT_LANG}"'/' < $f > ${f}.new && \
    mv ${f}.new $f
    echo $f
done

```bash
dir_path=/home/usuaris/iaonnis.tsiamas/datasets/speech_translation/MUSTC_v2.0_wav_16k/*/*.tsv
sed -i 's+/home/usuaris/veussd/DATABASES/speech_translation/MUSTC_v2.0+/home/usuaris/iaonnis.tsiamas/datasets/speech_translation/MUSTC_v2.0_wav_16k+g' $dir_path
```

```bash
dir_path=/home/usuaris/scratch/ioannis.tsiamas/datasets/speech_translation/cv-corpus-8.0-2022-01-19/en/CoVoST/en-/*.tsv
sed -i 's+/home/usuaris/veussd/DATABASES/speech_translation/cv-corpus-8.0-2022-01-19/en/clips_16k_mono+/home/usuaris/scratch/ioannis.tsiamas/datasets/speech_translation/cv-corpus-8.0-2022-01-19/en/clips_16k_mono+g' $dir_path
```

```bash
dir_path=/home/usuaris/iaonnis.tsiamas/datasets/speech_translation/cv-corpus-8.0-2022-01-19/en/CoVoST/en-de/*.tsv
sed -i 's+/home/usuaris/iaonnis.tsiamas/datasets/speech_translation/cv-corpus-8.0-2022-01-19/en/CoVoST/en-de/clips_16k_mono+/home/usuaris/iaonnis.tsiamas/datasets/speech_translation/cv-corpus-8.0-2022-01-19/en/clips_16k_mono+g' $dir_path
```

```bash
dir_path=/home/usuaris/scratch/ioannis.tsiamas/datasets/speech_translation/MUSTC_v2.0_wav_16k/en-zh/*.tsv
sed -i 's+/home/usuaris/veussd/DATABASES/speech_translation/MUSTC_v2.0_wav_16k/en-zh+/home/usuaris/scratch/ioannis.tsiamas/datasets/speech_translation/MUSTC_v2.0_wav_16k/en-zh+g' $dir_path
```


/home/usuaris/veussd/DATABASES/speech_translation/cv-corpus-8.0-2022-01-19/en/clips_16k_mono/

max_tokens: 480000
max_source: 400000
batch_size: 18
==> 2136

max_tokens: 480000
max_source: 400000
batch_size: 24
==> 2129

max_tokens: 600000
max_source: 400000
batch_size: 24
==> 1635

exp_name=de_lna_old_adapters_hubert_2.5e-4_largeGPU
ckpts=${SAVE_DIR}/${exp_name}/ckpts/checkpoint_
inputs="${ckpts}8_15000.pt ${ckpts}8_15500.pt ${ckpts}8_16000.pt ${ckpts}9_16500.pt ${ckpts}9_17000.pt"
python ${FAIRSEQ_ROOT}/scripts/average_checkpoints.py \
  --inputs $inputs --output ${ckpts}avg_5_around_16000.pt

exp_name=de_lna_adapters_hubert_2.5e-4_largeGPU
ckpts=${SAVE_DIR}/${exp_name}/ckpts/checkpoint_
inputs="${ckpts}6_11000.pt ${ckpts}6_11500.pt ${ckpts}6_12000.pt ${ckpts}7_12500.pt ${ckpts}7_13000.pt"

python ${FAIRSEQ_ROOT}/scripts/average_checkpoints.py \
  --inputs $inputs --output ${ckpts}avg_5_around_12000.pt

exp_name=de_lna_hubert_2.5e-4_largeGPU
python ${FAIRSEQ_ROOT}/scripts/average_checkpoints.py \
--inputs ${SAVE_DIR}/${exp_name}/ckpts/ --num-update-checkpoint 10 --output ${SAVE_DIR}/${exp_name}/ckpts/avg_10_last.pt


exp_name=de_lna_adapters_kd_1_hubert_2.5e-4_largeGPU
ckpts=${SAVE_DIR}/${exp_name}/ckpts/checkpoint_
inputs="${ckpts}12_23500.pt ${ckpts}12_24000.pt ${ckpts}13_24500.pt ${ckpts}13_25000.pt ${ckpts}13_25500.pt"
python ${FAIRSEQ_ROOT}/scripts/average_checkpoints.py \
  --inputs $inputs --output ${ckpts}avg_5_around_24500.pt


exp_name=de_lna_adapters_kd_0.5_hubert_2.5e-4_largeGPU
ckpts=${SAVE_DIR}/${exp_name}/ckpts/checkpoint_
inputs="${ckpts}11_22000.pt ${ckpts}12_22500.pt ${ckpts}12_23000.pt ${ckpts}12_23500.pt ${ckpts}12_24000.pt"
python ${FAIRSEQ_ROOT}/scripts/average_checkpoints.py \
  --inputs $inputs --output ${ckpts}avg_5_around_23000.pt

python ${FAIRSEQ_ROOT}/scripts/average_checkpoints.py \
--inputs ${SAVE_DIR}/${exp_name}/ckpts/ --num-update-checkpoint 10 --output ${SAVE_DIR}/${exp_name}/ckpts/avg_10_last.pt


/home/usuaris/veu/gerard.ion.gallego/iwslt22-outputs/de_lna_old_adapters_hubert_2.5e-4_largeGPU/ckpts/checkpoint_8_16000.pt
/home/usuaris/veu/ioannis.tsiamas/IWSLT22/save_dir/de_lna_old_adapters_hubert_2.5e-4_largeGPU/ckpts/checkpoint_avg_5_around_16000.pt

conda activate cu115
bash repos/iwslt-2022/scripts/generations/generate_ensemble.sh IWSLT.tst2020 de 3 13 1 /home/usuaris/veu/gerard.ion.gallego/iwslt22-outputs/de_lna_adapters_kd_0.5_hubert_2.5e-4_largeGPU_ft_largeGPU/ckpts/checkpoint_avg_5_around_2250.pt:/home/usuaris/veu/ioannis.tsiamas/IWSLT22/save_dir/de_lna_adapters_hubert_2.5e-4_largeGPU/ckpts/checkpoint_avg_5_around_12000.pt:/home/usuaris/veu/gerard.ion.gallego/iwslt22-outputs/de_lna_hubert_2.5e-4_largeGPU/ckpts/checkpoint_11_22000.pt:/home/usuaris/veu/ioannis.tsiamas/IWSLT22/save_dir/de_lna_wav2vec_2.5e-4_smallGPU/ckpts/checkpoint_8_15000.pt ensemble5 0


bash repos/iwslt-2022/scripts/segmentation/eval.sh /home/usuaris/veu/gerard.ion.gallego/iwslt22-outputs/de_lna_adapters_kd_0.5_hubert_2.5e-4_largeGPU_ft_largeGPU/ckpts/checkpoint_avg_5_around_2250.pt IWSLT.tst2021 en de 3 16 16 1 0








bash repos/iwslt-2022/scripts/generations/generate_ensemble.sh tst-COMMON-ja ja 3 10 1 /home/usuaris/veu/ioannis.tsiamas/IWSLT22/save_dir/ja_lna_hubert_2.5e-4_smallGPU/ckpts/checkpoint_avg_5_around_23000.pt:/home/usuaris/veu/gerard.ion.gallego/iwslt22-outputs/ja_lna_adapters_hubert_2.5e-4_largeGPU/ckpts/checkpoint_avg_5_around_14000.pt ensemble 0

bash repos/iwslt-2022/scripts/generations/generate_ensemble.sh tst-COMMON-zh zh 3 12 1 /home/usuaris/veu/ioannis.tsiamas/IWSLT22/save_dir/zh_lna_hubert_2.5e-4_smallGPU/ckpts/checkpoint_avg_5_around_23000.pt:/home/usuaris/veu/gerard.ion.gallego/iwslt22-outputs/zh_lna_adapters_hubert_2.5e-4_largeGPU/ckpts/checkpoint_avg_5_around_16000.pt ensemble 0



bash repos/iwslt-2022/scripts/segmentation/segment.sh tst-COMMON-ja 3 12 13 1


exp_name=de_lna_wav2vec_2.5e-4_smallGPU
ckpts=${SAVE_DIR}/${exp_name}/ckpts/checkpoint_
inputs="${ckpts}7_14000.pt ${ckpts}7_14500.pt ${ckpts}8_15000.pt ${ckpts}8_15500.pt ${ckpts}8_16000.pt"
python ${FAIRSEQ_ROOT}/scripts/average_checkpoints.py \
  --inputs $inputs --output ${ckpts}avg_5_around_15000.pt