#!/bin/bash

set -e

datasets=( inbreast cbis bcdr01 bcdr02 zrh )
cancers=( True False )
for dataset in "${datasets[@]}"; do
    for cancer in "${cancers[@]}"; do
       qsub convert_images.sh "$dataset" "$cancer"
    done
done
