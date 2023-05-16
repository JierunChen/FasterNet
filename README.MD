# Run, Don’t Walk: Chasing Higher FLOPS for Faster Neural Networks [CVPR 2023]

This is the official Pytorch/PytorchLightning implementation of the paper: <br/>
> [**Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks**](https://arxiv.org/abs/2303.03667)      
> Jierun Chen, Shiu-hong Kao, Hao He, Weipeng Zhuo, Song Wen, Chul-Ho Lee, S.-H. Gary Chan        
> *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023*
> 

--- 
We propose a simple yet fast and effective partial convolution (**PConv**), as well as a latency-efficient family of architectures called **FasterNet**.

<p align="center">
<img src="https://user-images.githubusercontent.com/42027432/225219182-572e9a35-2e8e-4a93-8fb0-e17973263082.png" width=100% 
class="center">
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/42027432/230023468-5617e203-b65d-4269-a67d-cda36a4af6b8.png" width=100% 
class="center">
</p>

## Image Classification
### 1. Dependency Setup
Create an new conda virtual environment
```
conda create -n fasternet python=3.9.12 -y
conda activate fasternet
```
Clone this repo and install required packages:
```
git clone https://github.com/JierunChen/FasterNet
cd FasterNet/
pip install -r requirements.txt
```

### 2. Dataset Preparation

Download the [ImageNet-1K](http://image-net.org/) classification dataset and structure the data as follows:
```
/path/to/imagenet-1k/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```

### 3. Pre-trained Models

| name | resolution | acc | #params | FLOPs | model |
|:---:|:---:|:---:|:---:| :---:|:---:|
| FasterNet-T0 | 224x224 | 71.9 | 3.9M | 0.34G | [model](https://github.com/JierunChen/FasterNet/releases/download/v1.0/fasternet_t0-epoch.281-val_acc1.71.9180.pth) |
| FasterNet-T1 | 224x224 | 76.2 | 7.6M | 0.85G | [model](https://github.com/JierunChen/FasterNet/releases/download/v1.0/fasternet_t1-epoch.291-val_acc1.76.2180.pth) |
| FasterNet-T2 | 224x224 | 78.9 | 15.0M | 1.90G | [model](https://github.com/JierunChen/FasterNet/releases/download/v1.0/fasternet_t2-epoch.289-val_acc1.78.8860.pth) |
| FasterNet-S | 224x224 | 81.3 | 31.1M | 4.55G | [model](https://github.com/JierunChen/FasterNet/releases/download/v1.0/fasternet_s-epoch.299-val_acc1.81.2840.pth) |
| FasterNet-M | 224x224 | 83.0 | 53.5M | 8.72G | [model](https://github.com/JierunChen/FasterNet/releases/download/v1.0/fasternet_m-epoch.291-val_acc1.82.9620.pth) |
| FasterNet-L | 224x224 | 83.5 | 93.4M | 15.49G | [model](https://github.com/JierunChen/FasterNet/releases/download/v1.0/fasternet_l-epoch.299-val_acc1.83.5060.pth) |

### 4. Evaluation

We give an example evaluation command for a ImageNet-1K pre-trained FasterNet-T0 on a single GPU:
```
python train_test.py -c cfg/fasternet_t0.yaml \
--checkpoint_path model_ckpt/fasternet_t0-epoch=281-val_acc1=71.9180.pth \
--data_dir ../../data/imagenet --test_phase -g 1 -e 125
```

- For evaluating other model variants, change `-c`, `--checkpoint_path` accordingly. You can get the pre-trained models from the tables above. 
- For multi-GPU evaluation, change `-g` to a larger number or a list, e.g., `8` or `0,1,2,3,4,5,6,7`. Note that the batch size for evaluation 
should be changed accordingly, e.g., change `-e` from `125` to `1000`.

To measure the latency on CPU/ARM and throughput on GPU (if any), run
```
python train_test.py -c cfg/fasternet_t0.yaml \
--checkpoint_path model_ckpt/fasternet_t0-epoch=281-val_acc1=71.9180.pth \
--data_dir ../../data/imagenet --test_phase -g 1 -e 32  --measure_latency --fuse_conv_bn
```
 
- `-e` controls the batch size of input on GPU while the batch size of input is fixed internally to 1 on CPU/ARM.  


### 5. Training

FasterNet-T0 training on ImageNet-1K with a 8-GPU node:
```
python train_test.py -g 0,1,2,3,4,5,6,7 --num_nodes 1 -n 4 -b 4096 -e 2000 \
--data_dir ../../data/imagenet --pin_memory --wandb_project_name fasternet \
--model_ckpt_dir ./model_ckpt/$(date +'%Y%m%d_%H%M%S') --cfg cfg/fasternet_t0.yaml
```
To train other FasterNet variants, `--cfg` need to be changed. You may also want to change the training batch size `-b`.       
  

## Acknowledgement
This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) , [poolformer](https://github.com/sail-sg/poolformer), [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) and [mmdetection](https://github.com/open-mmlab/mmdetection) repositories.

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
