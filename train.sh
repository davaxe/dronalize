python train.py --add-name Test --dry-run $1 --use-cuda 1 --num-workers 4 --store-model 0

## Apptainer
# apptainer run --nv build/dronalize.sif python train.py -an Test -dr $1 -uc $2 -nw 4

## Docker
# docker run --gpus all -v "$(pwd)":/app -w /app dronalize python train.py -an Test -dr $1 -uc $2 -nw 4
