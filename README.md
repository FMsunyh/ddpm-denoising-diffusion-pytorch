# mini-stable-diffusion

## Denoising Diffusion Probabilistic Model, in Pytorch

Implementation of <a href="https://arxiv.org/abs/2006.11239">Denoising Diffusion Probabilistic Model</a> in Pytorch

## python环境
- torch >=2.0.1
- python 3.10
- 
## 训练数据
- celeba数据集 | https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
 
 百度云盘下载: CelebA/Img/img_align_celeba.zip

```
cp img_align_celeba.zip ./data/celebA/
cd ./data/celebA/
unzip img_align_celeba.zip
```

## 开启训练
- MNIST
```
python train.py --dataset mnist --epochs 6 --channels 1
```

- celebA
```
python train.py --dataset CelebA --epochs 100 --channels 3
```

## 输出路径
可以查看每一轮的预测结果
```
./outputs
```

## Citations

```bibtex
@misc{ho2020denoising,
    title={Denoising Diffusion Probabilistic Models},
    author={Jonathan Ho and Ajay Jain and Pieter Abbeel},
    year={2020},
    eprint={2006.11239},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
