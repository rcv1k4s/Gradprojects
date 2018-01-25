#!/usr/bin/env bash

for i in *.cpp
do
    make clean NAME="${i%%.*}"
    #./${i%%.*} image.jpg
    #gprof ${i%%.*} gmon.out > analysis${i%%.*}.txt
    rm analysis*
done
