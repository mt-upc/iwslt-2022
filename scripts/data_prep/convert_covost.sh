#!/bin/bash

path=$1
num=$2

for i in $(seq 0 9); do
    for j in $(seq 0 9); do
        for k in $(seq 0 9); do
            echo ${i}${j}${k}
            ls -1U ${path}/*${i}${j}${k}.mp3 | parallel -j $num ffmpeg -i {} -ac 1 -ar 16000 -hide_banner -loglevel error {.}.wav
        done
    done
done