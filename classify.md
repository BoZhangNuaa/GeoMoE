# 操作流程

## 环境

本节主要完成基础运行环境的搭建。在确认为 `Linux` 系统、已配置 `Miniconda` 且 `CUDA` 版本为 `11.8` 的基础上，我们将创建一个独立的虚拟环境，以避免与其他项目产生依赖冲突。随后，依次安装 `PyTorch` 框架及其实际所需的核心依赖库（例如 `accelerate`、`transformers` 以及 `timm` 等）。

```shell
conda create -n classify python=3.10

conda activate classify

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

pip install accelerate 

pip install transformers

pip install timm
```

### 场景分类

在正式开始场景分类任务的训练前，需要先初始化 `accelerate` 的配置。请运行以下命令启动配置流程，在交互选项中选择不进行分布式训练，其余选项均保持默认即可。

```
accelerate config
```

- **AID**

  接下来进行 `AID` 数据集的部署。为确保训练脚本能准确读取数据，需将预先从网盘下载好的 `AID` 数据集压缩包 [[百度网盘]](https://pan.baidu.com/s/18rSj5d2kaJ_ro5St4cFyMw?pwd=xh6x) 上传至服务器，并解压至规范的数据存放路径中（执行时请将路径替换为您的实际压缩包路径）。

  ```shell
  cd "PATH/OF/AID.tar.gz"
  
  mkdir -p ~/datasets/ && tar -xzvf AID.tar.gz -C ~/datasets/
  ```

  至此，AID 数据准备完成。随后，切换到场景分类任务对应的工作目录。

  ```shell
  cd ~/geomoe/GeoMoE/Classify
  ```

  配置就绪后，通过以下命令进行模型微调。下方代码分别展示了基于两种不同数据划分比例的训练启动指令，均指定使用单张显卡完成训练。

  ```shell
  CHECKPOINT_DIR=GeoMoE
  PRETRAIN_CHKPT=~/geomoe/GeoMoE.pth
  DATAPATH=~/datasets/AID
  CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port=16902 classify.py \
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
  
  CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port=16902 classify.py \
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
  ```

- **NWPU**

  接下来进行 `NWPU` 数据集的部署。与 `AID` 数据集的操作同理，为确保训练脚本能准确读取数据，需将预先下载好的 `NWPU` 数据集压缩包  [[百度网盘]](https://pan.baidu.com/s/1CoiX9-NmGTMP_skXVfjwxg?pwd=grtj)  上传至服务器，并将其解压至指定的数据存放路径中。

  ```shell
  cd "PATH/OF/NWPU.tar.gz"
  
  mkdir -p ~/datasets/ && tar -xzvf NWPU.tar.gz -C ~/datasets/
  ```

  至此，`NWPU` 数据准备完成。接着，重新进入分类任务的工作目录准备运行脚本。

  ```shell
  cd ~/geomoe/GeoMoE/Classify
  ```

  最后，通过以下命令启动 `NWPU` 数据集上的微调任务。此处同样提供了两种划分的独立执行代码，各项学习率和正则化参数已在脚本中预设完毕。

  ```shell
  CHECKPOINT_DIR=GeoMoE
  PRETRAIN_CHKPT=~/geomoe/GeoMoE.pth
  DATAPATH=~/datasets/NWPU
  CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port=16903 classify.py \
      --batch_size 64 \
      --ngpus 1 \
      --model GeoMoE \
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
      --lrd geo_lrd
      #--eval
  
  
  CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port=16903 classify.py \
      --batch_size 64 \
      --ngpus 1 \
      --model GeoMoE \
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
      --lrd geo_lrd
      #--eval
  ```

  