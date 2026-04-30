# 操作流程

## 环境

本节主要完成基础运行环境的搭建。在确认为 `Linux` 系统、已配置 `Miniconda` 且 `CUDA` 版本为 `11.8` 的基础上，我们将创建一个独立的虚拟环境，以避免与其他项目产生依赖冲突。随后，依次安装 `PyTorch` 框架及其实际所需的核心依赖库（包含 `OpenMMLab` 基础组件与 `timm` 等）。

```shell
conda create -n mmdet python=3.10

conda activate mmdet

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

pip install "mmengine==0.10.5"

pip install -U openmim

mim install "mmcv==2.1.0"

pip install timm

pip install "numpy==1.26.4"

pip install ftfy

pip install regex
```

## 代码复现

模型训练需要依赖预训练权重作为初始化参数。请先自主从 🤗[`HuggingFace`](https://huggingface.co/BoZhangNuaa/GeoMoE) 下载权重文件  [`GeoMoE.pth`](https://huggingface.co/BoZhangNuaa/GeoMoE/blob/main/GeoMoE.pth) 至本地目录。随后，将官方代码仓库克隆至本地工作区，为后续的任务构建完整的项目目录结构。

```shell
mkdir ~/geomoe

cd ~/geomoe

git clone https://github.com/BoZhangNuaa/GeoMoE.git
```

### 目标检测

目标检测任务基于 `MMDetection` 框架实现。请先激活项目虚拟环境并安装该框架对应的版本。

```shell
conda activate mmdet

pip install "mmdet==3.3.0"
```

接下来进行 `DIOR` 数据集的部署。为确保训练脚本能准确读取数据，需将预先下载好的 `DIOR` 数据集压缩包 [[百度网盘]](https://pan.baidu.com/s/1QbEh2ifywGgavawjN-rLFg?pwd=4mvx) 上传至服务器，并解压至规范的数据存放路径中。

```
cd "PATH/OF/DIOR.tar.gz"

mkdir -p ~/datasets/ && tar -xzvf DIOR.tar.gz -C ~/datasets/
```

至此，`DIOR` 数据准备完成。在此基础上，需将 `GeoMoE` 仓库中定制化的检测模型组件及配置文件迁移到 `MMDetection` 对应的目录层级下，以便框架能够调用这些自定义结构。

```shell
cd ~/geomoe

git clone https://github.com/open-mmlab/mmdetection.git

mkdir ~/geomoe/mmdetection/tools/configs

cp ~/geomoe/GeoMoE/Detection/dior/mmdetection/GeoMoE.py ~/geomoe/mmdetection/tools/configs

cp ~/geomoe/GeoMoE/Detection/dior/mmdetection/OptimConstructor.py ~/geomoe/mmdetection/tools/configs

cp ~/geomoe/GeoMoE/Detection/dior/mmdetection/FasterRCNN_AUX.py ~/geomoe/mmdetection/tools/configs
```

由于原配置文件中存在较多绝对路径，为避免逐一修改带来的遗漏或引发路径报错，建议直接重建训练配置文件。我们将清空旧文件并创建一个新的配置文件。

```shell
rm ~/geomoe/GeoMoE/Detection/dior/config/GeoMoE.py

nano ~/geomoe/GeoMoE/Detection/dior/config/GeoMoE.py
```

在写入配置前，可通过以下命令获取当前用户所在主目录的绝对路径，以便后续替换配置中的路径占位符。

```shell
cd ~

pwd
```

上述 `nano` 命令打开了一个文件写入界面。请将以下配置内容粘贴进去，**并务必将代码中的 `~` 符号替换为您上一步获取到的实际绝对目录路径**，确保路径连贯正确后保存。

```python
auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
crop_size = (
    800,
    800,
)
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'configs.OptimConstructor',
        'configs.GeoMoE',
        'configs.FasterRCNN_AUX',
    ])
data_root = '~/datasets/DIOR'

default_hooks = dict(
    checkpoint=dict(interval=3, type='CheckpointHook'),
    logger=dict(interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'pytorch'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
metainfo = dict(
    classes=(
        'baseballfield',
        'storagetank',
        'airport',
        'trainstation',
        'overpass',
        'ship',
        'Expressway-toll-station',
        'vehicle',
        'Expressway-Service-area',
        'airplane',
        'dam',
        'bridge',
        'groundtrackfield',
        'harbor',
        'windmill',
        'tenniscourt',
        'basketballcourt',
        'stadium',
        'golffield',
        'chimney',
    ))
model = dict(
    backbone=dict(
        depth=[
            2,
            2,
            11,
        ],
        drop_rate=0.1,
        embed_dim=[
            256,
            384,
            768,
        ],
        img_size=[
            800,
            200,
            100,
        ],
        mlp_ratio=[
            4,
            4,
            4,
        ],
        moe_mlp_ratio=0.75,
        num_heads=12,
        patch_size=[
            4,
            2,
            2,
        ],
        pretrained=
        '~/geomoe/GeoMoE.pth',
        type='GeoMoEDet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        in_channels=[
            256,
            384,
            768,
            768,
        ],
        num_outs=5,
        out_channels=256,
        type='FPN'),
    roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                ],
                type='DeltaXYWHBBoxCoder'),
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
            loss_cls=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            num_classes=20,
            reg_class_agnostic=False,
            roi_feat_size=7,
            type='Shared2FCBBoxHead'),
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        type='StandardRoIHead'),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales=[
                8,
            ],
            strides=[
                4,
                8,
                16,
                32,
                64,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
        loss_cls=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        type='RPNHead'),
    test_cfg=dict(
        rcnn=dict(
            max_per_img=100,
            nms=dict(iou_threshold=0.5, type='nms'),
            score_thr=0.05),
        rpn=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=1000)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=False,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.5,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.25,
                type='RandomSampler')),
        rpn=dict(
            allowed_border=-1,
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type='RandomSampler')),
        rpn_proposal=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=2000)),
    type='FasterRCNN_AUX')
model_wrapper = dict(
    detect_anomalous_params=False,
    find_unused_parameters=False,
    type='MMDistributedDataParallel')
optim_wrapper = dict(
    constructor='GeoMoELayerDecayOptimizerConstructor',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=8e-5, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(layer_decay_rate=0.8),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=1e-06, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=12,
        gamma=0.1,
        milestones=[
            8,
            11,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/test.json',
        backend_args=None,
        data_prefix=dict(img='images'),
        data_root='~/datasets/DIOR',
        metainfo=dict(
            classes=(
                'baseballfield',
                'storagetank',
                'airport',
                'trainstation',
                'overpass',
                'ship',
                'Expressway-toll-station',
                'vehicle',
                'Expressway-Service-area',
                'airplane',
                'dam',
                'bridge',
                'groundtrackfield',
                'harbor',
                'windmill',
                'tenniscourt',
                'basketballcourt',
                'stadium',
                'golffield',
                'chimney',
            )),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                800,
                800,
            ), type='Resize'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='~/datasets/DIOR/annotations/test.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        800,
        800,
    ), type='Resize'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=12, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=1,
    dataset=dict(
        ann_file='annotations/trainval.json',
        backend_args=None,
        data_prefix=dict(img='images'),
        data_root='~/datasets/DIOR',
        filter_cfg=dict(filter_empty_gt=True, min_size=4),
        metainfo=dict(
            classes=(
                'baseballfield',
                'storagetank',
                'airport',
                'trainstation',
                'overpass',
                'ship',
                'Expressway-toll-station',
                'vehicle',
                'Expressway-Service-area',
                'airplane',
                'dam',
                'bridge',
                'groundtrackfield',
                'harbor',
                'windmill',
                'tenniscourt',
                'basketballcourt',
                'stadium',
                'golffield',
                'chimney',
            )),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale=(
                800,
                800,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        800,
        800,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/test.json',
        backend_args=None,
        data_prefix=dict(img='images'),
        data_root='~/datasets/DIOR',
        metainfo=dict(
            classes=(
                'baseballfield',
                'storagetank',
                'airport',
                'trainstation',
                'overpass',
                'ship',
                'Expressway-toll-station',
                'vehicle',
                'Expressway-Service-area',
                'airplane',
                'dam',
                'bridge',
                'groundtrackfield',
                'harbor',
                'windmill',
                'tenniscourt',
                'basketballcourt',
                'stadium',
                'golffield',
                'chimney',
            )),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                800,
                800,
            ), type='Resize'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='~/datasets/DIOR/annotations/test.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
```

至此，DIOR 目标检测的运行环境与参数已全部配置就绪。最后，调用分布式训练脚本启动模型的微调过程。

```shell
cd ~/geomoe/mmdetection

bash ./tools/dist_train.sh ~/geomoe/GeoMoE/Detection/dior/config/GeoMoE.py 4
```

### 语义分割

语义分割任务则基于 `MMSegmentation` 框架。同样地，需在激活的环境下安装该任务所需的框架版本。

```
conda activate mmdet

pip install "mmsegmentation==1.2.2"
```

针对语义分割，我们使用 `LoveDA` 数据集 [[百度网盘]](https://pan.baidu.com/s/1qQooVIDLufmAfR3rjYMzvw?pwd=sz2r)。请按照与目标检测相同的逻辑，将数据集压缩包上传至服务器并解压至统一的数据存储目录中。

```shell
cd "PATH/OF/LoveDA.tar.gz"

mkdir -p ~/datasets/ && tar -xzvf LoveDA.tar.gz -C ~/datasets/
```

至此，`LoveDA` 数据准备完成。随后，克隆 `MMSegmentation` 官方仓库，并将 `GeoMoE` 项目中关于 `LoveDA` 任务的定制化网络组件及配置脚本复制到该框架的特定目录内。

```shell
cd ~/geomoe

git clone https://github.com/open-mmlab/mmsegmentation.git

mkdir ~/geomoe/mmsegmentation/tools/configs

cp ~/geomoe/GeoMoE/Segmentation/Loveda/mmsegmentation/GeoMoE.py ~/geomoe/mmsegmentation/tools/configs/

cp ~/geomoe/GeoMoE/Segmentation/Loveda/mmsegmentation/OptimConstructor.py ~/geomoe/mmsegmentation/tools/configs/

cp ~/geomoe/GeoMoE/Segmentation/Loveda/mmsegmentation/EncoderDecoder_AUX.py ~/geomoe/mmsegmentation/tools/configs/

cp ~/geomoe/GeoMoE/Segmentation/Loveda/mmsegmentation/loveda.py ~/geomoe/mmsegmentation/tools/configs/
```

为统一管理数据路径并确保训练顺利执行，我们需要重新编写该任务的训练配置。

```shell
nano ~/geomoe/GeoMoE/Segmentation/Loveda/config/GeoMoE.py
```

同样地，写入前需要先确认当前用户的主目录路径。

```shell
cd ~

pwd
```

在打开的文件编辑器中粘贴以下整合后的配置代码，并仔细核对，将所有的 `~` 符号替换为您个人的真实绝对路径后保存。

```python
############################### default runtime #################################
custom_imports = dict(
    imports=['configs.OptimConstructor', 'configs.GeoMoE',
             'configs.EncoderDecoder_AUX', 'configs.loveda'],
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
data_root = '~/datasets/LoveDA'
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
    constructor='GeoMoELayerDecayOptimizerConstructor',
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
        T_max=19500,
        begin=500,
        end=20000,
        by_epoch=False,
    )
]

# training schedule for 80k
train_cfg = dict(type='IterBasedTrainLoop',
                 max_iters=20000, val_interval=20000)
#val_cfg = dict(type='ValLoop')
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
        type='GeoMoEDet',
        img_size=[512, 128, 64],
        patch_size=[4, 2, 2],
        embed_dim=[256, 384, 768],
        depth=[2, 2, 11],
        num_heads=12,
        drop_rate=0.2,
        moe_mlp_ratio=0.75,
        mlp_ratio=[4, 4, 4],
        pretrained='~/geomoe/GeoMoE.pth',
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[256, 384, 768, 768],
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
```

配置文件保存后，`LoveDA` 语义分割任务的准备工作即告完成。请执行以下指令，利用分布式脚本开启多卡微调训练。

```shell
cd ~/geomoe/mmsegmentation/

bash ./tools/dist_train.sh ~/geomoe/GeoMoE/Segmentation/Loveda/config/GeoMoE.py 8
```
