# UPC's Speech Translation System for IWSLT 2022

System submitted to the [IWSLT 2022](https://iwslt.org/2022/) [offline speech translation task](https://iwslt.org/2022/offline) by the [UPC Machine Translation group](https://mt.cs.upc.edu).

The paper is available [here](https://aclanthology.org/2022.iwslt-1.23/).

## Abstract

<em>This paper describes the submissions of the UPC Machine Translation group to the IWSLT 2022 Offline Speech Translation and Speech-to-Speech Translation tracks. The offline task involves translating English speech to German, Japanese and Chinese text. Our Speech Translation systems are trained end-to-end and are based on large pretrained speech and text models. We use an efficient fine-tuning technique that trains only specific layers of our system, and explore the use of adapter modules for the non-trainable layers. We further investigate the suitability of different speech encoders (wav2vec 2.0, HuBERT) for our models and the impact of knowledge distillation from the Machine Translation model that we use for the decoder (mBART). For segmenting the IWSLT test sets we fine-tune a pretrained audio segmentation model and achieve improvements of 5 BLEU compared to the given segmentation. Our best single model uses HuBERT and parallel adapters and achieves 29.42 BLEU at English-German MuST-C tst-COMMON and 26.77 at IWSLT 2020 test. By ensembling many models, we further increase translation quality to 30.83 BLEU and 27.78 accordingly. Furthermore, our submission for English-Japanese achieves 15.85 and English-Chinese obtains 25.63 BLEU on the MuST-C tst-COMMON sets. Finally, we extend our system to perform English-German Speech-to-Speech Translation with a pretrained Text-to-Speech model.</em>

## Citation
```
@inproceedings{tsiamas-etal-2022-pretrained,
    title = "Pretrained Speech Encoders and Efficient Fine-tuning Methods for Speech Translation: {UPC} at {IWSLT} 2022",
    author = "Tsiamas, Ioannis  and
      G{\'a}llego, Gerard I.  and
      Escolano, Carlos  and
      Fonollosa, Jos{\'e}  and
      Costa-juss{\`a}, Marta R.",
    booktitle = "Proceedings of the 19th International Conference on Spoken Language Translation (IWSLT 2022)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland (in-person and online)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.iwslt-1.23",
    pages = "265--276",
}
```

## Contents

- [Environment Setup](#environment-setup)
- [Pretrained Models](#pretrained-models)
- [Data](#data)
- [Knowledge Distillation](#knowledge-distillation)
- [Training](#training)
- [MuST-C Evaluation](#evaluation-on-must-c-known-segmentation)
- [IWSLT Evaluation](#evaluation-on-iwslttst20xx-unknown-segmentation)

## Environment Setup

Set the environment variables:

```bash
export IWSLT_ROOT=...                          # where to clone this repo
export FAIRSEQ_ROOT=...                        # where to clone fairseq
```

Clone this repository to `$IWSLT_ROOT`:

```bash
git clone --recursive https://github.com/mt-upc/iwslt-2022.git ${IWSLT_ROOT}
```

Create a conda environment using the `environment.yml` file, activate it and install Fairseq:

```bash
conda env create -f ${IWSLT_ROOT}/environment.yml && \
conda activate iwslt22 && \
pip install --editable ${IWSLT_ROOT}/fairseq/
```

Install NVIDIA's [apex](https://github.com/NVIDIA/apex) library for faster training with fp16 precision:

```bash
git clone https://github.com/NVIDIA/apex && cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
--global-option="--deprecated_fused_adam" --global-option="--xentropy" \
--global-option="--fast_multihead_attn" ./
```

## Pretrained models

In this project we use pre-trained speech encoders and text decoders.\
Download HuBERT, wav2vec2.0 and mBART models to `$MODELS_ROOT`:

```bash
export MODELS_ROOT=...

mkdir -p ${MODELS_ROOT}/{wav2vec,hubert,mbart}
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_960h_pl.pt -P ${MODELS_ROOT}/wav2vec
wget https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k.pt -P ${MODELS_ROOT}/hubert
wget https://dl.fbaipublicfiles.com/fairseq/models/mbart50/mbart50.ft.1n.tar.gz -O - | \
tar -xz --strip-components 1 -C ${MODELS_ROOT}/mbart
```

## Data

### Download

Set the data environment variables:

```bash
export MUSTC_ROOT=...           # where to download MuST-C v2                  
export CV_ROOT=...              # where to download the CommonVoice corpus 8.0
export EUROPARL_ROOT=...        # where to download Europarl-ST
export IWSLT_TST_ROOT=...       # where to download the IWSLT test sets
```

Download MuST-C v2 en-de, en-ja and en-zh to `$MUSTC_ROOT`:\
The dataset is available [here](https://ict.fbk.eu/must-c/). Press the bottom ”click here to download the corpus”, and select version V2.

Download the Common Voice version 8 and the CoVoST tsvs (en-de, en-ja, en-zh) to `$CV_ROOT`:

```bash
export COVOST_ROOT=${CV_ROOT}/en/CoVoST
mkdir -p ${COVOST_ROOT}/{en-de,en-ja,en-zh}
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
    # get the file order for this pair
    cut -d' ' -f1 ${IWSLT_TST_ROOT}/IWSLT.tst2022/IWSLT.TED.tst2022.en-${tgt_lang}.en.video_url > ${IWSLT_TST_ROOT}/IWSLT.tst2022/FILER_ORDER.en-${tgt_lang}
done
```

### Data Preparation

Convert the Common Voice clips to 16kHz and mono: \
(We only need to convert the ones in the train, dev and test splits)

```bash
mkdir -p ${CV_ROOT}/en/clips_mono_16k
for split in {train,dev,test}; do
    cat ${COVOST_ROOT}/${split}.tsv | cut -f2 | parallel -j $(eval nproc) ffmpeg -i ${CV_ROOT}/en//clips/{} \
    -ac 1 -ar 16000 -hide_banner -loglevel error ${CV_ROOT}/en/clips_mono_16k/{.}.wav
done
```

Prepare the tsvs for the MuST-C, Europarl-ST and CoVoST data: \
We do this process for both the ASR and ST tasks and for all language pairs. \
We only prepare the tsvs and do not learn a vocabulary since we will reuse the one from mBART50.

```bash
# MuST-C (en-de,en-zh,en-ja)
for task in {asr,st}; do
    python ${IWSLT_ROOT}scripts/data_prep/prep_mustc_data.py \
    --data-root ${MUSTC_ROOT} --task $task --use-audio-input --only-manifest --append-lang-id
done
# Europarl-ST (en-de)
for task in {asr,st}; do
    python ${IWSLT_ROOT}/scripts/data_prep/prep_europarl_data.py \
    -d ${EUROPARL_ROOT} --lang-pair en-de --task st --use-audio-input --only-manifest --append-lang-id
done
# CoVoST (en-de,en-zh,en-ja)
for tgt_lang in {de,zh-CH,ja}; do
    for task in {asr,st}; do
        python ${IWSLT_ROOT}/scripts/data_prep/prep_covost_data.py \
        -d $COVOST_ROOT -s en -t $tgt_lang --append-lang-id
    done
done
```

### Data Filtering

Do ASR inference on the "train" sets using a pre-trained wav2vec 2.0 model and save the results at `$FILTER_ROOT`:

```bash
export FILTER_ROOT=...
# MuST-C
for tgt_lang in {de,ja,zh}; do
    python ${IWSLT_ROOT}/scripts/filtering/asr_inference.py \
    --tsv_path ${MUSTC_ROOT}/en-${tgt_lang}/train_asr.tsv -o ${FILTERING_ROOT}/MUSTC_v2.0/en
done
# Europarl-ST
for split in {train,dev,test}; do
    python ${IWSLT_ROOT}/scripts/filtering/asr_inference.py \
    --tsv_path ${EUROPARL_ROOT}/en/en-de_${split}_asr.tsv -o ${FILTERING_ROOT}/EuroparlST/en
done
# CoVoST
for split in {train,dev,test}; do
    for tgt_lang in {de,ja,zh}; do
        python ${IWSLT_ROOT}/scripts/filtering/asr_inference.py \
        --tsv_path ${COVOST_ROOT}/en-${tgt_lang}/${split}_asr.tsv -o ${FILTERING_ROOT}/CoVoST/en
    done
done
```

Apply ASR-based and text-based filtering to create clean versions of the train sets:

```bash
# MuST-C
for tgt_lang in {de,ja,zh}; do
    python ${IWSLT_ROOT}/filtering/filter_tsv.py \
    -tsv ${MUSTC_ROOT}/en-${tgt_lang}/train_st.tsv \
    -p ${FILTERING_ROOT}/MUSTC_v2.0/en/train_asr_wer_results.json \
    -o ${MUSTC_ROOT}/en-${tgt_lang} \
    -par -wer 0.75
done
# Europarl-ST
for split in {train,dev,test}; do
    python ${IWSLT_ROOT}/filtering/filter_tsv.py \
    -tsv ${EUROPARL_ROOT}/en/en-de_${split}_st.tsv \
    -p ${FILTERING_ROOT}/EuroparlST/en/en-de_${split}_asr_wer_results.json \
    -o ${EUROPARL_ROOT}/en \
    -par -wer 0.75
done
# CoVoST
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

Set up the path:

```bash
export DATA_ROOT=...
mkdir -p ${DATA_ROOT}/{en-de,en-zh,en-ja}
```

Make symbolink links:

```bash
# from MuST-C
for tgt_lang in {de,ja,zh}; do
    for task in {asr,st}; do
        ln -s ${MUSTC_ROOT}/en-${tgt_lang}/train_${task}_filtered.tsv ${DATA_ROOT}/en-${tgt_lang}/train_${task}_mustc.tsv
        ln -s ${MUSTC_ROOT}/en-${tgt_lang}/dev_${task}.tsv ${DATA_ROOT}/en-${tgt_lang}/dev_${task}_mustc.tsv
        ln -s ${MUSTC_ROOT}/en-${tgt_lang}/tst-COMMON_${task}.tsv ${DATA_ROOT}/en-${tgt_lang}/tst-COMMON_${task}_mustc.tsv
    done
done
# from Europarl-ST
for split in {train,dev,test}; do
    for task in {asr,st}; do
        if [[ $split != train ]]; then
            ln -s ${EUROPARL_ROOT}/en/en-de_${split}_${task}_filtered.tsv ${DATA_ROOT}/en-de/train_${split}_${task}_europarl.tsv
        else
            ln -s ${EUROPARL_ROOT}/en/en-de_${split}_${task}_filtered.tsv ${DATA_ROOT}/en-de/${split}_${task}_europarl.tsv
        fi
    done
done
# from CoVoST
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
```

## Knowledge Distillation

We are using knowledge distillation for en-de with mBART50 as the teacher. \
Extract the top-k probabilities offline before training and save them at `$KD_ROOT`:

```bash
export KD_ROOT=...
for asr_tsv_file in ${DATA_ROOT}/en-de/train*asr*.tsv; do
    st_tsv_file=$(echo $asr_tsv_file | sed "s/_asr_/_st_/g")
    kd_subdir=$(basename "$st_tsv_file" .tsv)
    python ${IWSLT_ROOT}knowledge_distillation/extract_topk_logits.py \
    -asr $asr_tsv_file -st $st_tsv_file -o ${KD_ROOT}/en-de/${kd_subdir}
done
```

## Training

Set up the path to save the training outputs:

```bash
export SAVE_DIR=...
```

All our experiments can be found at `${IWSLT_ROOT}/config`.\
To train an experiment called `EXP_NAME`, run the following command:

```bash
EXP_NAME=...     # one of the available experiments

# to adjust the update_freq according to the number of available GPUs
base_update_freq=24
n_gpus=$(nvidia-smi --list-gpus | wc -l)

fairseq-hydra-train \
    --config-dir ${IWSLT_ROOT}/config/ \
    --config-name ${EXP_NAME}.yaml \
    dataset.num_workers=$(($(eval nproc) / 2)) \
    optimization.update_freq=[$(( $base_update_freq / $n_gpus ))]
```

## Evaluation on MuST-C (known segmentation)

To generate the translations for the MuST-C dev or tst-COMMON sets run the following command:

```bash
EXP_NAME=...    # one of the trained experiments
CKPT_NAME=...   # the name of a .pt file
SUBSET=...      # dev_mustc or tst-COMMON_mustc
TGT_LANG=...    # de, zh or ja

${IWSLT_ROOT}/scripts/generate.sh $EXP_NAME $CKPT_NAME $SUBSET $TGT_LANG 
```

## Evaluation on IWSLT.tst20xx (unknown segmentation)

To generate translations for the IWSLT test sets, we first have to segment the audio files.

We are using [SHAS](https://arxiv.org/abs/2202.04774). Clone the SHAS repo at `$SHAS_ROOT`:

```bash
git clone https://github.com/mt-upc/SHAS.git ${SHAS_ROOT}
```

Create an environment for the segmentation:

```bash
conda env create -f ${SHAS_ROOT}/environment.yml
```

Download the Multilingual [checkpoint](https://drive.google.com/u/0/uc?export=download&confirm=x9hB&id=1GzwhzbHBFtwDmQPKoDOdAfESvWBrv_wB) for the Segmentation Frame Classifier at `$SHAS_ROOT/mult_sfc_model_epoch-4.pt`.

Segment the wav files of the IWSLT test sets with the multilingual classifier and the pDAC algorithm with max-segment-length of 16 and inference-times of 3, which were found to be optimal. Save the segmentation yaml at `$path_to_custom_segmentation_yaml`:

```bash
conda activate shas
SUBSET=...   # IWSLT.tst2019, IWSLT.tst2020, IWSLT.tst2021 or IWSLT.tst2022
python ${SHAS_ROOT}/src/supervised_hybrid/segment.py \
  -wavs ${IWSLT_TST_ROOT}/${SUBSET}/wavs \
  -ckpt ${SHAS_ROOT}/mult_sfc_model_epoch-4.pt \
  -yaml $path_to_custom_segmentation_yaml \
  -max 16 -n 3
```

To evaluate translations from a custom segmentation, we are using to mwerSegmenter to align the hypotheses with the references.

Download mwerSegmenter at `${MWERSEGMENTER_ROOT}` and follow the instructions in `${MWERSEGMENTER_ROOT}/README` to install it:

```bash
export MWERSEGMENTER_ROOT=...
mkdir -p $MWERSEGMENTER_ROOT
wget https://www-i6.informatik.rwth-aachen.de/web/Software/mwerSegmenter.tar.gz
tar -zxvf mwerSegmenter.tar.gz -C ${MWERSEGMENTER_ROOT}
rm -r mwerSegmenter.tar.gz
```

We also need a python2 environment to run it:

```bash
conda create -n snakes27 python=2.7
```

Generate translations on the created segmentation and calculate the BLEU scores if the `$SUBSET` is IWSLT.tst2019 or IWSLT.tst2020:

```bash
${IWSLT_ROOT}/scripts/segmentation/eval.sh \
    ${SAVE_DIR}/${EXP_NAME}/ckpts/${CKPT_NAME} \
    $SUBSET \
    $TGT_LANG \
    $path_to_custom_segmentation_yaml
```
