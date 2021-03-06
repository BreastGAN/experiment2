#!/bin/bash

set -e

height="${1:-512}"
width="${2:-408}"

size="${height}x${width}"
transformed="./data_in/transformed"
transformed_out="$transformed/small_all_$size"
transformed_all="$transformed/small_all_${size}_final"

datasets="$(ls "$transformed_out" | grep -Eo 'dataset_[^\.]*' | sort | uniq | grep -Eo '[a-zA-Z0-9]*$')"
for dataset in $datasets; do
    echo "$dataset"
    dataset_folder="$transformed/${dataset}_${size}"
    mkdir -p "$dataset_folder"
    files="$(ls "$transformed_out" | grep "$dataset" | sed -E "s@^@$transformed_out/@" | tr '\n' ' ')"
    cp $files $dataset_folder
    ./local/merge_images.sh "$dataset_folder" "$dataset_folder"
done

# All:
./local/merge_images.sh "$transformed_out" "$transformed_all"

