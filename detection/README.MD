# COCO Object detection with FasterNet
 
## Dependency Setup

After setting up dependency for [Image Classification](https://github.com/JierunChen/FasterNet), install the following packages
```
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
pip install mmdet==2.25.0
```

## Data preparation

Prepare COCO according to the guidelines in [MMDetection v2.25.0](https://mmdetection.readthedocs.io/en/v2.25.0/).

## Results and models on COCO

| Method     | Backbone | Pretrain    | Lr schd | Aug | box AP | mask AP | Config                                               | Download |
|------------|----------|-------------|:-------:|:---:|:------:|:-------:|------------------------------------------------------|----------|
| Mask R-CNN | FasterNet-S | ImageNet-1K |    1x   |  No |  39.9  |   36.9  | [config](configs/fasternet/mask_rcnn_fasternet_s_fpn_1x_coco.py) | [log](https://github.com/JierunChen/FasterNet/releases/download/v1.0/mask_rcnn_fasternet_s_fpn_1x_coco_20221111_063428.log) & [model](https://github.com/JierunChen/FasterNet/releases/download/v1.0/mask_rcnn_fasternet_s_fpn_1x_coco_20221111_063419.pth) |
| Mask R-CNN | FasterNet-M | ImageNet-1K |    1x   |  No |  43.0  |   39.1  | [config](configs/fasternet/mask_rcnn_fasternet_m_fpn_1x_coco.py) | [log](https://github.com/JierunChen/FasterNet/releases/download/v1.0/mask_rcnn_fasternet_m_fpn_1x_coco_20221107_124415.log) & [model](https://github.com/JierunChen/FasterNet/releases/download/v1.0/mask_rcnn_fasternet_m_fpn_1x_coco_20221107_124408.pth) |
| Mask R-CNN | FasterNet-L | ImageNet-1K |    1x   |  No |  44.0  |   39.9  | [config](configs/fasternet/mask_rcnn_fasternet_l_fpn_1x_coco.py) | [log](https://github.com/JierunChen/FasterNet/releases/download/v1.0/mask_rcnn_fasternet_l_fpn_1x_coco_20221107_004515.log) & [model](https://github.com/JierunChen/FasterNet/releases/download/v1.0/mask_rcnn_fasternet_l_fpn_1x_coco_20221107_004433.pth) |

## Evaluation

To evaluate FasterNet-T0 + Mask R-CNN on COCO val2017 on a single node with 8 GPUs, run
```
bash ./dist_test.sh configs/fasternet/mask_rcnn_fasternet_s_fpn_1x_coco.py \
 ckpts/mask_rcnn_fasternet_s_fpn_1x_coco_20221111_063419.pth 8 --eval bbox segm
```

To measure its FLOPs, run
```
python get_flops.py configs/fasternet/mask_rcnn_fasternet_s_fpn_1x_coco.py
```

To measure its throughput on a GPU, run
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 \
benchmark.py configs/fasternet/mask_rcnn_fasternet_s_fpn_1x_coco.py \
ckpts/mask_rcnn_fasternet_s_fpn_1x_coco_20221111_063419.pth --launcher pytorch --fuse-conv-bn
```

**Note**: There are an issue related to throughput measurement in the [paper v1](https://arxiv.org/abs/2303.03667). 
Although it do not affect the conclusion that FasterNet achieves higher accuracy-latency efficiency, we clarify that:

- PConv and FasterNet use `"slicing"` type for faster inference and throughput measurement. 
However, it implicitly modifies the shortcut, making a computation inconsistency to using `"split_cat"`. 
To fix that, we may 
    - clone the input via `x = x.clone()` before applying partial convolution, but it introduces additional latency and can defeat the benefits of using `"slicing"` over `"split_cat"`.
    - move the shortcut after the PConv operator, which resolves the issue and is likely to maintain the effectiveness. Models modified are under retraining and will be released once finished.        


## Training

To train FasterNet-T0 + Mask R-CNN on COCO train2017 on a single node with 8 GPUs for 12 epochs, run
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./dist_train.sh \
configs/fasternet/mask_rcnn_fasternet_s_fpn_1x_coco.py 8 \
--work-dir work_dirs/mask_rcnn_fasternet_s_fpn_1x_coco/$(date +'%Y%m%d_%H%M%S') \
--cfg-options model.pretrained=../model_ckpt/fasternet_t0-epoch=281-val_acc1=71.9180.pth 
```

## Acknowledgement
This repository is mainly based on [mmdetection](https://github.com/open-mmlab/mmdetection) library.

## Citation
If you find this repository helpful, please consider citing:
```
@article{chen2023run,
  title={Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks},
  author={Chen, Jierun and Kao, Shiu-hong and He, Hao and Zhuo, Weipeng and Wen, Song and Lee, Chul-Ho and Chan, S-H Gary},
  journal={arXiv preprint arXiv:2303.03667},
  year={2023}
}
```
