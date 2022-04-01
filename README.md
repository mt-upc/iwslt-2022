# UPC's Speech Translation System for IWSLT 2022

System submitted to the [IWSLT 2022](https://iwslt.org/2022/) [offline speech translation task](https://iwslt.org/2022/offline) by the [UPC Machine Translation group](https://mt.cs.upc.edu).

The pre-print of the submission is available on [arxiv](insert_link).

<em>Abstract ...</em>

## Setting up the environment

Set the environment variables:

```bash
export IWSLT_ROOT=...
export FAIRSEQ_ROOT=...
export MWERSEGMENTER_ROOT=...
export SHAS_ROOT=...

export MUSTC_ROOT=...
export CV_ROOT=...
export COVOST_ROOT=${CV_ROOT}/en/CoVoST
export EUROPARL_ROOT=...
export IWSLT_TST_ROOT=...
export DATA_ROOT=...

export MODELS_ROOT=...

export KD_ROOT=...
export FILTER_ROOT=...
```

Clone this repository to `$IWSLT_ROOT`:

```bash
git clone https://github.com/mt-upc/iwslt-2022.git ${IWSLT_ROOT} 
```

(TODO: FIX ENV)
Create a conda environment using the `environment.yml` file and activate it:

```bash
conda env create -f ${IWSLT_ROOT}/environment.yml && \
conda activate iwslt22
```

Clone our Fairseq fork and install it:

```bash
git clone -b iwslt22 https://github.com/mt-upc/fairseq-internal.git ${FAIRSEQ_ROOT} && \
pip install --editable ${FAIRSEQ_ROOT}
```

Install NVIDIA's [apex](https://github.com/NVIDIA/apex) library for faster training with fp16 precision:

```bash
git clone https://github.com/NVIDIA/apex && cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
--global-option="--deprecated_fused_adam" --global-option="--xentropy" \
--global-option="--fast_multihead_attn" ./
cd .. && rm -rf apex
```

## Pre-trained models

In this project we use pre-trained speech encoders and text decoders.\
Download HuBERT, wav2vec2.0 and mBART models to `$MODELS_ROOT`:

```bash
mkdir -p ${MODELS_ROOT}/{wav2vec,hubert,mbart}
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_960h_pl.pt -P ${MODELS_ROOT}/wav2vec
wget https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k.pt -P ${MODELS_ROOT}/hubert
wget https://dl.fbaipublicfiles.com/fairseq/models/mbart50/mbart50.ft.1n.tar.gz -O - | \
tar -xz --strip-components 1 -C ${MODELS_ROOT}/mbart
```

## DATA

### Download

Download MuST-C v2 en-de, en-ja and en-zh to `$MUSTC_ROOT`:\
The dataset is available [here](https://ict.fbk.eu/must-c/). Press the bottom ”click here to download the corpus”, and select version V2.

Download the Common Voice version 8 and the CoVoST tsvs (en-de, en-ja, en-zh) to `$CV_ROOT`:

```bash
mkdir -p ${CV_ROOT}/en/CoVoST/{en-de,en-ja,en-zh}
wget https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-8.0-2022-01-19/cv-corpus-8.0-2022-01-19-en.tar.gz -P ${COVOST_ROOT}
wget https://dl.fbaipublicfiles.com/covost/covost_v2.en_de.tsv.tar.gz -P ${CoVoST_ROOT}/en-de
wget https://dl.fbaipublicfiles.com/covost/covost_v2.en_zh-CN.tsv.tar.gz -P ${CoVoST_ROOT}/en-zh
wget https://dl.fbaipublicfiles.com/covost/covost_v2.en_ja.tsv.tar.gz -P ${CoVoST_ROOT}/en-ja
```

Download Europarl-ST v1.1 to `$EUROPARL_ROOT`:

```bash
mkdir -p ${EUROPARL_ROOT}
wget https://www.mllp.upv.es/europarl-st/v1.1.tar.gz -O - | tar -xz --strip-components 1 -C ${EUROPARL_ROOT}
```

Download the IWLST data (tst.2019,tst.2020,tst.2021,tst.2022):

```bash
mkdir -p $IWSLT_TST_ROOT
for year in {2019,2020,2021}; do
    wget http://i13pc106.ira.uka.de/~jniehues/IWSLT-SLT/data/eval/en-de/IWSLT-SLT.tst2019.en-de.tgz
    tar -xvf IWSLT-SLT.tst2022.en-${tgt_lang}.tgz -C ${IWSLT_TST_ROOT}
    rm IWSLT-SLT.tst2022.en-${tgt_lang}.tgz
done
for tgt_lang in {de,ja,zh}; do
    wget http://i13pc106.ira.uka.de/~jniehues/IWSLT-SLT/data/eval/en-${tgt_lang}/IWSLT-SLT.tst2022.en-${tgt_lang}.tgz
    tar -xvf IWSLT-SLT.tst2022.en-${tgt_lang}.tgz -C ${IWSLT_TST_ROOT}
    rm IWSLT-SLT.tst2022.en-${tgt_lang}.tgz
    cut -d' ' -f1 ${IWSLT_TST_ROOT}/IWSLT.tst2022/IWSLT.TED.tst2022.en-${tgt_lang}.en.video_url > ${IWSLT_TST_ROOT}/IWSLT.tst2022/FILER_ORDER.en-${tgt_lang}
done
```

### Data Preparation

Prepare the MuST-C data: \
(We use a modified version of the fairseq script that does not create a vocabulary)

```bash
for task in {asr,st}; do
    python ${IWSLT_ROOT}scripts/data_prep/prep_mustc_data.py \
    --data-root ${MUSTC_ROOT} --task $task --use-audio-input --only-manifest --append-lang-id
done
```

Convert the Common Voice clips to 16kHz and mono: \
(We only need to convert the ones in the train, dev and test splits)

```bash
mkdir -p ${CV_ROOT}/en/clips_mono_16k
for split in {train,dev,test}; do
    cat ${COVOST_ROOT}/${split}.tsv | cut -f2 | parallel -$(eval nproc) $num_processors ffmpeg -i ${CV_ROOT}/en//clips/{} \
    -ac 1 -ar 16000 -hide_banner -loglevel error ${CV_ROOT}/en/clips_mono_16k/{.}.wav
done
```

Prepare CoVoST data in fairseq format:

```bash
for tgt_lang in {de,zh-CH,ja}; do
    for task in {asr,st}; do
        python ${IWSLT_ROOT}/scripts/data_prep/prep_covost_data.py \
        -d $COVOST_ROOT -s en -t $tgt_lang --append-lang-id
    done
done
```

Prepare the Europarl-ST data in fairseq format
(TODO) adapt script to work with both wav and the original mp3/flac format

```bash
for task in {asr,st}; do
    python ${IWSLT_ROOT}/scripts/data_prep/prep_europarl_data.py \
    -d ${EUROPARL_ROOT} --lang-pair en-de --task st --use-audio-input --only-manifest --append-lang-id
done
```

### Data Filtering

Do ASR inference on the "train" sets using a pre-trained wav2vec 2.0 model:

```bash
for tgt_lang in {de,ja,zh}; do
    python ${IWSLT_ROOT}/scripts/filtering/asr_inference.py \
    --tsv_path ${MUSTC_ROOT}/en-${tgt_lang}/train_asr.tsv -o ${FILTERING_ROOT}/MUSTC_v2.0/en
done

for split in {train,dev,test}; do
    python examples/iwslt22/scripts/filtering/asr_inference.py \
    --tsv_path ${EUROPARL_ROOT}/en/en-de_${split}_asr.tsv -o ${FILTERING_ROOT}/EuroparlST/en
done

for split in {train,dev,test}; do
    for tgt_lang in {de,ja,zh}; do
        python examples/iwslt22/scripts/filtering/asr_inference.py \
        --tsv_path ${COVOST_ROOT}/en-${tgt_lang}/${split}_asr.tsv -o ${FILTERING_ROOT}/CoVoST/en
    done
done
```

Apply ASR-based and text-based filtering to create clean versions of the train sets:

```bash
for tgt_lang in {de,ja,zh}; do
    python ${IWSLT_ROOT}/filtering/filter_tsv.py \
    -tsv ${MUSTC_ROOT}/en-${tgt_lang}/train_st.tsv \
    -p ${FILTERING_ROOT}/MUSTC_v2.0/en/train_asr_wer_results.json \
    -o ${MUSTC_ROOT}/en-${tgt_lang} \
    -par -wer 0.75
done

for split in {train,dev,test}; do
    python ${IWSLT_ROOT}/filtering/filter_tsv.py \
    -tsv ${EUROPARL_ROOT}/en/en-de_${split}_st.tsv \
    -p ${FILTERING_ROOT}/EuroparlST/en/en-de_${split}_asr_wer_results.json \
    -o ${EUROPARL_ROOT}/en \
    -par -wer 0.75
done

for tgt_lang in {de,ja,zh}; do
    for split in {train,dev,test}; do
        python ${IWSLT_ROOT}/filtering/filter_tsv.py \
        -tsv ${COVOST_ROOT}/en-${tgt_lang}/${split}_st.tsv \
        -p ${FILTERING_ROOT}/CoVoST/en/${split}_asr_wer_results.json \
        -o ${COVOST_ROOT}/en-${tgt_lang} \
        -par -wer 0.75
    done
done
```

### Combine the different datasets into en-de, en-ja and en-zh directories

Make symbolink links:

```bash
mkdir -p ${DATA_ROOT}/{en-de,en-zh,en-ja}

for tgt_lang in {de,ja,zh}; do
    for task in {asr,st}; do
        ln -s ${MUSTC_ROOT}/en-${tgt_lang}/train_${task}_filtered.tsv ${DATA_ROOT}/en-${tgt_lang}/train_${task}_mustc.tsv
        ln -s ${MUSTC_ROOT}/en-${tgt_lang}/dev_${task}.tsv ${DATA_ROOT}/en-${tgt_lang}/dev_${task}_mustc.tsv
        ln -s ${MUSTC_ROOT}/en-${tgt_lang}/tst-COMMON_${task}.tsv ${DATA_ROOT}/en-${tgt_lang}/tst-COMMON_${task}_mustc.tsv
    done
done

for tgt_lang in {de,ja,zh}; do
    for split in {train,dev,test}; do
        for task in {asr,st}; do
            if [[ $split != train ]]; then
                ln -s ${COVOST_ROOT}/en-${tgt_lang}/${split}_${task}_filtered.tsv ${DATA_ROOT}/en-${tgt_lang}/train_${split}_${task}_covost.tsv
            else
                ln -s ${COVOST_ROOT}/en-${tgt_lang}/${split}_${task}_filtered.tsv ${DATA_ROOT}/en-${tgt_lang}/${split}_${task}_covost.tsv
            fi
        done
    done
done

for split in {train,dev,test}; do
    for task in {asr,st}; do
        if [[ $split != train ]]; then
            ln -s ${EUROPARL_ROOT}/en/en-de_${split}_${task}_filtered.tsv ${DATA_ROOT}/en-de/train_${split}_${task}_europarl.tsv
        else
            ln -s ${EUROPARL_ROOT}/en/en-de_${split}_${task}_filtered.tsv ${DATA_ROOT}/en-de/${split}_${task}_europarl.tsv
        fi
    done
done
```

### Knowledge Distillation

We are using knowledge distillation with mBART50 as the teacher. \
We are extracting the top-k probabilities offline before training:

```bash
for tgt_lang in {de,ja,zh}; do
    for asr_tsv_file in ${DATA_ROOT}/en-${tgt_lang}/train*asr*.tsv; do
        st_tsv_file=$(echo $asr_tsv_file | sed "s/_asr_/_st_/g")
        kd_subdir=$(basename "$st_tsv_file" .tsv)
        python ${IWSLT_ROOT}knowledge_distillation/extract_topk_logits.py \
        -asr $asr_tsv_file -st $st_tsv_file -o ${KD_ROOT}/en-${tgt_lang}/${kd_subdir}
done
```
