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
num_processors=...

mkdir -p ${COVOST_ROOT}/en/clips_converted
for split in {train,dev,test}; do
    cat ${COVOST_ROOT}/en/${split}.tsv | cut -f2 | parallel -j $num_processors ffmpeg -i ${COVOST_ROOT}/en/clips/{} -ac 1 -ar 16000 -hide_banner -loglevel error ${COVOST_ROOT}/en/clips_converted/{.}.wav
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
for tgt_lang in {de,ja,zh}; do
    ln -s ${MUSTC_ROOT}/en-${tgt_lang}/train_st_filtered.tsv ${DATASET_ROOT}/en-${tgt_lang}/mustc_train_st_f.tsv
    ln -s ${MUSTC_ROOT}/en-${tgt_lang}/dev_st.tsv ${DATASET_ROOT}/en-${tgt_lang}/mustc_dev_st.tsv
    ln -s ${MUSTC_ROOT}/en-${tgt_lang}/tst-COMMON_st.tsv ${DATASET_ROOT}/en-${tgt_lang}/mustc_tst-COMMON_st.tsv
    done
done

for tgt_lang in {de,ja,zh}; do
    for split in {train,dev,test}; do
        ln -s ${COVOST_ROOT}/en-${tgt_lang}/${split}_st_filtered.tsv ${DATASET_ROOT}/en-${tgt_lang}/covost_${split}_st_f.tsv
    done
done

for split in {train,dev,test}; do
    ln -s ${EUROPARL_ROOT}/en/en-de_${split}_st_filtered.tsv ${DATASET_ROOT}/en-de/europarl_${split}_st_f.tsv
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