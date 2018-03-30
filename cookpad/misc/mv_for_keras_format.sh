#!/bin/sh

set -eu

class_list=`cat input/train_master.tsv | sed -e '1d' | awk '{print $2}' | sort |uniq`

file_names=(`cat input/train_master.tsv | sed -e '1d' | awk '{print $1}' | xargs`)
class_names=(`cat input/train_master.tsv | sed -e '1d' | awk '{print $2}' | xargs`)
for class in $class_list
do
    mkdir -p input/images/$class
done

for i in `seq 1 ${#file_names[@]}`
do
    file_name=${file_names[i]}
    class_name=${class_names[i]}
    mv input/images/$file_name input/images/$class_name/$file_name
done

