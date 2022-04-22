# LDRN

This repository is implementation of the [ **Learning Local Distribution for Extremely Efficient Single-Image Super-Resolution** ]()

![image-20220421205419523](C:\Users\XW\AppData\Roaming\Typora\typora-user-images\image-20220421205419523.png)

## Requirements

- PyTorch 1.10.0
- opencv-python 4.5.2.52
- Numpy 1.20.1
- torchvision 0.11.1
- tqdm 4.62.2

## Train

The DIV2K, Set5, Set14, B100, Urban100, Manga109 dataset can be downloaded from the links below.

- [DIV2K](http://www.vision.ee.ethz.ch/%7Etimofter/publications/Agustsson-CVPRW-2017.pdf)
- [Set5](https://www.kaggle.com/datasets/msahebi/super-resolution)
- [Set14](https://www.kaggle.com/datasets/msahebi/super-resolution)
- [B100](https://www.kaggle.com/datasets/msahebi/super-resolution)
- [Urban100](https://www.kaggle.com/datasets/msahebi/super-resolution)
- [Manga109](https://www.kaggle.com/datasets/msahebi/super-resolution)

Otherwise,you can use `dataProcess.py`  to create custom dataset and use `dataProcess2.py`  to create GaussianBlur  dataset.  Get start with the following command

```bash
python train.py --line_weight 12 1 --scale 2 --cuda
```

The pre-training weight file is saved in the weight directory, and you can also start with the following command

```bash
python train.py --line_weight 12 1 --scale 2 --pre_train "weight/LDRN_X2_bestpsnr_37.35.pth" --cuda
```

## Results

### set5(Bicubic)

| Eval.Mat  | Scale |     LDRN     |
| :-------: | :---: | :----------: |
| PSNR/SSIM |   2   | 37.35/0.9812 |
| PSNR/SSIM |   3   | 33.86/0.9666 |
| PSNR/SSIM |   4   | 30.82/0.9349 |

### set5(Gaussian)

| Eval.Mat  | Scale |     LDRN     |
| :-------: | :---: | :----------: |
| PSNR/SSIM |   2   | 37.24/0.9809 |
| PSNR/SSIM |   3   | 33.69/0.9658 |
| PSNR/SSIM |   4   | 30.72/0.9345 |

### x3

![image-20220422215211392](C:\Users\XW\AppData\Roaming\Typora\typora-user-images\image-20220422215211392.png)

### x4

![image-20220422215238259](C:\Users\XW\AppData\Roaming\Typora\typora-user-images\image-20220422215238259.png)

