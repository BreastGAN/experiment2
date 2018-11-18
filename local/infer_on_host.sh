#!/bin/bash

set -e

PROJECT_HOME="."
VIRTUAL_ENV_PATH=$PROJECT_HOME/venv

checkpoint_path="${1}"
masks="${2:-True}"
icnr="${3:-False}"
upsample_method="${4:-conv2d_transpose}"
loss_identity_lambda="${5:-0.0}"
spectral_norm="${6:-False}"
output_dir="${7}"

# activate virtual environment
source $VIRTUAL_ENV_PATH/bin/activate

## EXECUTION OF PYTHON CODE:
echo "Run model:"
cd $PROJECT_HOME
base_path="$PROJECT_HOME/data_in/transformed/small_all_512x408_final"
python -m models.breast_cycle_gan.inference --height 512 --width 408 --image_source "$base_path"/cancer.eval.tfrecord --model="C2H" --include_masks="$masks" --checkpoint_path="$checkpoint_path" --use_icnr="$icnr" --upsample_method="$upsample_method" --loss_identity_lambda="$loss_identity_lambda" --generated_dir="$output_dir" --use_spectral_norm="$spectral_norm"
python -m models.breast_cycle_gan.inference --height 512 --width 408 --image_source "$base_path"/healthy.eval.tfrecord --model="H2C" --include_masks="$masks" --checkpoint_path="$checkpoint_path" --use_icnr="$icnr" --upsample_method="$upsample_method" --loss_identity_lambda="$loss_identity_lambda" --generated_dir="$output_dir" --use_spectral_norm="$spectral_norm"
