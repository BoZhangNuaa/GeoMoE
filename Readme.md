## Intruduction

GeoMoE是一个在OpticalRS-4M上进行预训练的遥感基础模型，以超稀疏的混合专家结构和卷积前端取得了更快的训练速度和收敛速度，并在下游任务上有好的评测得分。

## News

- `2025.09`：创建仓库

## TodoList

- [x] 开源GeoMoE和MoE的训练权重🤗[HuggingFace](https://huggingface.co/BoZhangNuaa/GeoMoE)
- [ ] 开源下游任务的微调代码、参数、日志
  - [ ] 场景分类
  - [ ] 目标检测
  - [ ] 语义分割
- [ ] 开源预训练代码

## Experiments

我们再次复现了论文中的实验结果，如表格所示：

| Downstream Tasks      | Datasets | MoE                                    | GeoMoE                                    |
| --------------------- | -------- | -------------------------------------- | ----------------------------------------- |
| Sence Classification  |          |                                        |                                           |
|                       |          |                                        |                                           |
| Object Detection      | DIOR     | 76.40([log](./Detection/dior/MoE.log)) | 79.30([log](./Detection/dior/GeoMoE.log)) |
|                       |          |                                        |                                           |
| Semantic Segmentation |          |                                        |                                           |
|                       |          |                                        |                                           |

- [ ] ### Sence Classification

- [ ] ### Object Detection

- [ ] ### Semantic Segmentation

## Reference

本项目的代码构建参考了[ConvMAE](https://github.com/Alpha-VL/ConvMAE)和[SelectiveMAE](https://github.com/MiliLab/SelectiveMAE)。





