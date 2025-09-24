  <h2 align="center"><strong>	Efficient Self-Supervised Learning for Remote Sensing via</br> Sparse Convolutional Mixture-of-Experts</strong></h2>

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
    📃 <a href="https://github.com/BoZhangNuaa/GeoMoE/blob/main/Readme_zh.md" target="_blank">zh</a>
  </p>




## Intruduction

GeoMoE is a remote sensing foundation model pretrained on OpticalRS-4M, achieving faster training and convergence speeds with an ultra-sparse mixture-of-experts architecture and convolutional front-end, while also delivering strong evaluation scores on downstream tasks.

## News

- `2025.09`：Create repository.
- `2025.09`：Announced content related to object detection.
- `2025.09`：Announced the relevant content of semantic segmentation.

## TodoList

- [x] Open-source training weights for `GeoMoE` and `MoE` 🤗[HuggingFace](https://huggingface.co/BoZhangNuaa/GeoMoE)
- [ ] Finetuning code, parameters, and logs for open-source downstream tasks.
  - [ ] Scene classification
  - [x] Object detection
  - [x] Semantic segmentation
- [ ] Open-source pretrained code

## Experiments

Experimental results and logs are shown in the table:

| Downstream Tasks      | Datasets   | MoE                                           | GeoMoE                                           |
| --------------------- | ---------- | --------------------------------------------- | ------------------------------------------------ |
| Scene Classification  | AID        |                                               |                                                  |
| Scene Classification  | RESISC-45  |                                               |                                                  |
| Object Detection      | DIOR       | 76.40([log](./Detection/dior/MoE.log))        | 79.30([log](./Detection/dior/GeoMoE.log))        |
| Object Detection      | DIOR-R     | 69.70([log](./Detection/dior-r/MoE.log))      | 71.82([log](./Detection/dior-r/GeoMoE.log))      |
| Semantic Segmentation | LoveDA     | 53.48([log](./Segmentation/Loveda/MoE.zip))   | 54.76([log](./Segmentation/Loveda/GeoMoE.zip))   |
| Semantic Segmentation | SpaceNetv1 | 86.46([log](./Segmentation/Spacenet/MoE.log)) | 86.62([log](./Segmentation/Spacenet/GeoMoE.log)) |

### Scene Classification

### Object Detection

For ease of processing, we converted DIOR to COCO format, with the annotation files retained in [annotation.zip](./Detection/dior/annotation.zip)。

### Semantic Segmentation

We used the first 5000 samples of SpaceNetv1 for training and the remaining samples for testing, with this split ratio consistent with [SelectiveMAE](https://github.com/MiliLab/SelectiveMAE).

`LovaDA` is an online evaluation, so we are replacing the publication of logs with the publication of model outputs.

## Reference

The code for this project was developed with reference to [ConvMAE](https://github.com/Alpha-VL/ConvMAE) and [SelectiveMAE](https://github.com/MiliLab/SelectiveMAE).





