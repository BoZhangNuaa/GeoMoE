CHECKPOINT_DIR=save/GeoMoE

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port=16915 train.py \
    --batch_size 320 \
    --ngpus 8 \
    --model GeoMoE \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --build_ratio 0.25 \
    --epochs 400 \
    --blr 1.5e-4 \
    --save_interval 10 \
    --warmup_ratio 0.025 \
    --min_lr_ratio 1e-3 \
    --log_dir ${CHECKPOINT_DIR} \
    --datasets OpticalRS \
    --checkpoints ${CHECKPOINT_DIR} \
    --log_interval 100 \
    --num_workers 2 \
    --dataset_/datasets/OpticalRS-4M