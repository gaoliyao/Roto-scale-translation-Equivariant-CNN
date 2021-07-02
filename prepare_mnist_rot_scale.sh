# !/bin/bash
# MIT License. Copyright (c) 2020 Ivan Sosnovik, Michał Szmaja

MNIST_DIR="${MNIST_DIR:-./datasets}"
MNIST_SCALE_DIR="${MNIST_SCALE_DIR:-./datasets}"


echo "Preparing datasets..."
for i in {0..0}
do 
    echo ""
    echo "Dataset [$((i+1))/6]"

    python prepare_datasets.py --source "$MNIST_DIR" --dest "$MNIST_SCALE_DIR" --min_rot -120 --max_rot 120 --min_scale 0.3 --download --seed $i
    
done
