CHECKPOINT_DIR=GeoMoE
PRETRAIN_CHKPT=/GeoMoE.pth
DATAPATH=/datasets/AID
CUDA_VISIBLE_DEVICES=7 accelerate launch --main_process_port=16902 classify.py \
    --batch_size 64 \
    --ngpus 1 \
    --model GeoMoE \
    --save_interval 401\
    --warmup_ratio 0.025 \
    --min_lr_ratio 0.001 \
    --dataset AID \
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
    --lrd geo_lrd
    #--eval


CUDA_VISIBLE_DEVICES=7 accelerate launch --main_process_port=16902 classify.py \
    --batch_size 64 \
    --ngpus 1 \
    --model GeoMoE \
    --save_interval 401\
    --warmup_ratio 0.025 \
    --min_lr_ratio 0.001 \
    --dataset AID \
    --root_dir ${DATAPATH} \
    --checkpoint ${CHECKPOINT_DIR} \
    --epoch 200 \
    --log_interval 100 \
    --log_dir ${CHECKPOINT_DIR} \
    --blr 1e-3 \
    --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --finetune ${PRETRAIN_CHKPT} \
    --layer_decay 0.85 \
    --TR 55_0.json \
    --eval_epoch 190 \
    --eval_interval 10 \
    --img_size 320 \
    --lrd geo_lrd
    #--eval


