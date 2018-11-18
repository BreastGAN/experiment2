#!/bin/bash

set -e

# Replace docker args with the script args
args="$(echo $@ | sed 's/^.*-- //')"
echo "Args: $args" >&2
set -- $args

arg="$1"
if [ -z "$arg" ]; then
    echo 'No run config passed. Can either be "model", "modelboard" ,"jupyter", or "lab".'
    exit 1
fi

set -- ${@:2}

# Activate virtual env
source venv/bin/activate

if ./setup/check_venv.sh
then
    echo "Not in venv, please activate it first."
    exit 1
fi

# Run jupyter notebook environment
if [ "$arg" == 'jupyter' ]; then
    exec jupyter notebook --allow-root --ip 0.0.0.0 --no-browser $@
fi

# Run jupyter notebook environment
if [ "$arg" == 'lab' ]; then
    exec jupyter lab --allow-root --ip 0.0.0.0 --no-browser $@
fi

# Run the model
if [ "$arg" == 'model' ]; then
    exec ./local/run_on_host.sh $@ # Pass arg and all successive arguments
fi

# Run the model + TensorBoard
if [ "$arg" == 'modelboard' ]; then
    tensorboard --logdir data_out >/dev/null & # Run tensorboard in the background
    exec ./local/run_on_host.sh $@ # Pass arg and all successive arguments
fi

echo "Run config $arg unknown!"' Can either be "model", "modelboard", "jupyter", or "lab".'
exit 1
