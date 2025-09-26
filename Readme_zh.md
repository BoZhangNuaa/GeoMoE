  <h2 align="center"><strong>	Efficient Self-Supervised Learning for Remote Sensing via <\br>Sparse Convolutional Mixture-of-Experts</strong></h2>

  <p align="center">
    Bo Zhang<sup>12</sup>&nbsp;&nbsp;&nbsp;
    Renzhi Wang<sup>12</sup>&nbsp;&nbsp;&nbsp;
    Xiao Ling<sup>3</sup>&nbsp;&nbsp;&nbsp;
    Piji Li<sup>12*</sup></br>
    </br>
  <sup>1</sup> College of Artificial Intelligence, Nanjing University of Aeronautics and Astronautics&nbsp;&nbsp;&nbsp;</br>
  <sup>2</sup>The Key Laboratory of Brain-Machine Intelligence Technology, Ministry of Education &nbsp;&nbsp;&nbsp;</br>
  <sup>3</sup>College of Astronautics, Nanjing University of Aeronautics and Astronautics&nbsp;&nbsp;
<div align='center' style="font-size: larger; "><strong>Patent pending</strong></div>
  <p align="center">
    📃 <a href="" target="_blank">Paper (Patent pending)</a> |
    🤗 <a href="https://huggingface.co/BoZhangNuaa/GeoMoE" target="_blank">Models</a> |
    📃 <a href="https://github.com/BoZhangNuaa/GeoMoE/blob/main/Readme.md" target="_blank">en</a>
  </p>


## Intruduction

GeoMoE是一个在OpticalRS-4M上进行预训练的遥感基础模型，以超稀疏的混合专家结构和卷积前端取得了更快的训练速度和收敛速度，并在下游任务上有好的评测得分。

## News

- `2025.09`：创建仓库
- `2025.09`：公布了目标检测的相关内容
- `2025.09`：公布了语义分割的相关内容

## TodoList

- [x] 公布GeoMoE和MoE的训练权重🤗[HuggingFace](https://huggingface.co/BoZhangNuaa/GeoMoE)
- [x] 公布下游任务的微调代码、参数、日志
  - [x] 场景分类
  - [x] 目标检测
  - [x] 语义分割
- [ ] 公布预训练代码

## Experiments

实验结果和日志如表格所示：

| Downstream Tasks      | Datasets      | MoE                                                 | GeoMoE                                                    |
| --------------------- | ------------- | --------------------------------------------------- | --------------------------------------------------------- |
| Scene Classification  | AID 20%       | 96.77([log](./Classify/MoE/AID/MoE_AID_20.log))     | 97.01([log](./Classify/GeoMoE/AID/GeoMoE_AID_20.log))     |
|                       | AID 50%       | 98.18([log](./Classify/MoE/AID/MoE_AID_50.log))     | 98.40([log](./Classify/GeoMoE/AID/GeoMoE_AID_50.log))     |
| Scene Classification  | RESISC-45 10% | 93.80([log](./Classify/MoE/NWPU/MoE_RESISC_10.log)) | 94.47([log](./Classify/GeoMoE/NWPU/GeoMoE_RESISC_10.log)) |
|                       | RESISC-45 20% | 95.59([log](./Classify/MoE/NWPU/MoE_RESISC_20.log)) | 96.04([log](./Classify/GeoMoE/NWPU/GeoMoE_RESISC_20.log)) |
| Object Detection      | DIOR          | 76.40([log](./Detection/dior/MoE.log))              | 79.30([log](./Detection/dior/GeoMoE.log))                 |
| Object Detection      | DIOR-R        | 69.70([log](./Detection/dior-r/MoE.log))            | 71.82([log](./Detection/dior-r/GeoMoE.log))               |
| Semantic Segmentation | LoveDA        | 53.48([log](./Segmentation/Loveda/MoE.zip))         | 54.76([log](./Segmentation/Loveda/GeoMoE.zip))            |
| Semantic Segmentation | SpaceNetv1    | 86.46([log](./Segmentation/Spacenet/MoE.log))       | 86.62([log](./Segmentation/Spacenet/GeoMoE.log))          |

### Sence Classification

为便于处理，我们将RESISC和AID数据集转换为COCO格式，其标注文件分别保留在[RESISC](./Classify/RESISC)和[AID](./Classify/AID)目录下。

使用以下命令运行：

```shell
cd Classify
bash GeoMoE.sh
bash GeoMoE_AID.sh
bash MoE.sh
bash MoE_AID.sh
```

### Object Detection

为了便于处理，我们将DIOR转化为了COCO格式，标注文件保留在[annotation.zip](./Detection/dior/annotation.zip)。

该下游任务分别基于 [mmdetection](https://github.com/open-mmlab/mmdetection) 和 [mmrotate](https://github.com/open-mmlab/mmrotate/tree/1.x) 实现。

### Semantic Segmentation

我们将SpaceNetv1的前5000个样本用于训练，其余样本用于测试，该分割比例与[SelectiveMAE](https://github.com/MiliLab/SelectiveMAE)一致。

LovaDA为在线评测，因此我们将公布日志替换为公布模型输出。

该下游任务基于 [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) 实现。

## Reference

本项目的代码构建参考了[ConvMAE](https://github.com/Alpha-VL/ConvMAE)和[SelectiveMAE](https://github.com/MiliLab/SelectiveMAE)。





