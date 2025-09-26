CHECKPOINT_DIR=MoE
PRETRAIN_CHKPT=/MoE.pth
DATAPATH=/datasets/NWPU
CUDA_VISIBLE_DEVICES=5 accelerate launch --main_process_port=16900 classify.py \
    --batch_size 64 \
    --ngpus 1 \
    --model MoE \
    --save_interval 401\
    --warmup_ratio 0.025 \
    --min_lr_ratio 0.001 \
    --dataset NWPU \
    --root_dir ${DATAPATH} \
    --checkpoint ${CHECKPOINT_DIR} \
    --epoch 200 \
    --log_interval 100 \
    --log_dir ${CHECKPOINT_DIR} \
    --blr 1e-3 \
    --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --finetune ${PRETRAIN_CHKPT} \
    --layer_decay 0.85 \
    --TR 19_0.json \
    --eval_epoch 190 \
    --eval_interval 10 \
    --img_size 320 \
    --lrd moe_lrd
    #--eval


CUDA_VISIBLE_DEVICES=5 accelerate launch --main_process_port=16900 classify.py \
    --batch_size 64 \
    --ngpus 1 \
    --model MoE \
    --save_interval 401\
    --warmup_ratio 0.025 \
    --min_lr_ratio 0.001 \
    --dataset NWPU \
    --root_dir ${DATAPATH} \
    --checkpoint ${CHECKPOINT_DIR} \
    --epoch 200 \
    --log_interval 100 \
    --log_dir ${CHECKPOINT_DIR} \
    --blr 1e-3 \
    --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --finetune ${PRETRAIN_CHKPT} \
    --layer_decay 0.85 \
    --TR 28_0.json \
    --eval_epoch 190 \
    --eval_interval 10 \
    --img_size 320 \
    --lrd moe_lrd
    #--eval


