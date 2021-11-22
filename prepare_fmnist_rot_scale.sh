# !/bin/bash
# MIT License. Copyright (c) 2020 Ivan Sosnovik, Michał Szmaja

FMNIST_DIR="${FMNIST_DIR:-./datasets}"
FMNIST_SCALE_DIR="${FMNIST_SCALE_DIR:-./datasets}"


echo "Preparing datasets..."
for i in {0..0}
do 
    echo ""
    echo "Dataset [$((i+1))/6]"

    python prepare_datasets_fmnist.py --source "$FMNIST_DIR" --dest "$FMNIST_SCALE_DIR" --min_rot -180 --max_rot 180 --min_scale 0.3 --download --seed $i
    
done
