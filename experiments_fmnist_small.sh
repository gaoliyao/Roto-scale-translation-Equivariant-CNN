# !/bin/bash
# MIT License. Copyright (c) 2020 Ivan Sosnovik, Michał Szmaja

FMNIST_SCALE_DIR="${FMNIST_SCALE_DIR:-./datasets}"


function train_scale_fmnist() {
    # 1 model_name 
    # 2 extra_scaling
    # for seed in {0..5}
    for seed in {0..0}
    do 
        data_dir="$FMNIST_SCALE_DIR/FMNIST_scale/seed_$seed/scale_0.3_1.0"
        python train_scale_fmnist.py \
            --batch_size 32 \
            --epochs 60 \
            --optim adam \
            --lr 0.01 \
            --lr_steps 30 \
            --model $1 \
            --basis "A1"\
            --save_model_path "./saved_models/fmnist/$1_extra_scaling_$2.pt" \
            --cuda \
            --extra_scaling $2 \
            --tag "sesn_experiments" \
            --data_dir="$data_dir" \

    done               
}


#model_list=(
#    "mnist_ses_scalar_28"   # MNIST (28x28) 
#    "mnist_ses_vector_28"   # MNIST (28x28)
#    "mnist_ses_scalar_28p"  # MNIST (28x28) +
#    "mnist_ses_vector_28p"  # MNIST (28x28) +
#)

model_list=(    
    "mnist_res_scalar_56_rot_1"              # scalar, 8 rotations                   
    "mnist_res_scalar_56_rot_1"              # scalar, 8 rotations                   
    "mnist_res_scalar_56_rot_1"              # scalar, 8 rotations                   s
)

for model_name in "${model_list[@]}"
do
    for extra_scaling in 1.0
    do 
        train_scale_fmnist "$model_name" "$extra_scaling" 
    done
done
