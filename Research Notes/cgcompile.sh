#!/usr/bin/env bash

for i in *.cpp
do
    make all NAME="${i%%.*}"
    ./${i%%.*} image.jpg
    cat trace.out
    ./readtracelog.sh ${i%%.*} trace.out > ${i%%.*}.log
done
