############################### default runtime #################################
custom_imports = dict(
    imports=['config.OptimConstructor', 'config.MoE',
             'config.EncoderDecoder_AUX', 'config.loveda'],
    allow_failed_imports=False)
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

############################### dataset #################################

dataset_type = 'LoveDADataset_'
data_root = '/data1/bozhang/datasets/LoveDA'
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=(512, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='Train/images_png', seg_map_path='Train/masks_png'),
        pipeline=train_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='Test/images_png',
                         seg_map_path='Test/masks_png'),
        pipeline=test_pipeline))

test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], format_only=True)

############################### running schedule #################################

# optimizer

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.05),
    constructor='MoELayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        layer_decay_rate=0.85,
    )
)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=400),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0,
        T_max=19600,
        begin=400,
        end=20000,
        by_epoch=False,
    )
]

# training schedule for 80k
train_cfg = dict(type='IterBasedTrainLoop',
                 max_iters=20000, val_interval=20000)
# val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=10000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

############################### model #################################

norm_cfg = dict(type='SyncBN', requires_grad=True)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder_AUX',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        depth=12,
        drop_rate=0.2,
        embed_dim=768,
        img_size=512,
        mlp_ratio=4,
        moe_mlp_ratio=0.75,
        num_heads=12,
        patch_size=16,
        fpn_layers=[3, 5, 7, 11],
        pretrained='/data1/bozhang/ICASSP/MoE.pth',
        type='MoEDet'),
    decode_head=dict(
        type='UPerHead',
        in_channels=[768, 768, 768, 768],
        num_classes=7,
        ignore_index=255,
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', stride=(384, 384), crop_size=(512, 512)))
