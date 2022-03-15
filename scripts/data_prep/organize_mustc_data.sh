#!/bin/bash

path_to_mustc=~/datasets/speech_translation/MUSTC_v2.0

mkdir -p ${path_to_mustc}/all_wav

for lang_pair in {en-de,en-ja,en-zh}; do
    for split in {train,dev,tst-COMMON,tst-HE}; do

        wav_target_dir=${path_to_mustc}/all_wav/${split}
        mkdir -p $wav_target_dir

        lang_pair_wav_dir=${path_to_mustc}/${lang_pair}/data/${split}/wav

        if [[ ! -d $lang_pair_wav_dir ]]; then
            echo skipping ${lang_pair}/${split}
            continue
        fi

        echo processing ${lang_pair}/${split}

        for wav_file in ${lang_pair_wav_dir}/*.wav; do

            if [[ ! -s $wav_file ]]; then
                continue
            fi

            wav_file_name="$(basename -- $wav_file)"
            wav_file_target=${wav_target_dir}/${wav_file_name}
            if [[ ! -f $wav_file_target ]]; then
                cp -r $wav_file $wav_file_target
            fi
        done

    done
done