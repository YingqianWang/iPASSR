#### PyTorch implementation of "Symmetric Parallax Attention for Stereo Image Super-Resolution", *arXiv 2020*.

## Overview
<img src="https://raw.github.com/YingqianWang/iPASSR/master/Figs/Network.png" width="800"><br>
#### Fig. 1: An overview of our network.

## Requirement
* **PyTorch 1.3.0, torchvision 0.4.1. The code is tested with python=3.7, cuda=9.0.**
* **Matlab (For training/test data generation and performance evaluation)**

## Train
**Will be available soon.**

## Test
* **Download the test sets and unzip them to `./data`. Here, we provide the full test set on [Baidu Drive](https://pan.baidu.com/s/1LfopBUF_aKvfvUhSqA2Cew) (Key: NUDT) which is used in our paper.** 
* **Download our pretrained models (<a href="https://wyqdatabase.s3-us-west-1.amazonaws.com/log.zip">log.zip</a>) and unzip them to `./log`.**
* **Run `test.py` to perform a demo inference. Results (`.png` files) will be saved to `./results`.**
* **Run `evaluation.m` to calculate PSNR and SSIM scores.**

## Quantitative Results
<img src="https://raw.github.com/YingqianWang/iPASSR/master/Figs/Quantitative.png" width="1000"><br>
<br>
## Qualitative Results
<img src="https://raw.github.com/YingqianWang/iPASSR/master/Figs/2xSR.jpg" width="1000"><br>
<img src="https://raw.github.com/YingqianWang/iPASSR/master/Figs/4xSR.jpg" width="1000"><br>
<img src="https://raw.github.com/YingqianWang/iPASSR/master/Figs/RealSR.jpg" width="1000"><br>


## Citiation
**If you find this work helpful, please consider citing the following paper:**
```
@artical{iPASSR,
  author    = {Wang, Yingqian and Ying, Xinyi and Wang, Longguang and Yang, Jungang and An, Wei and Guo, Yulan},
  title     = {Symmetric Parallax Attention for Stereo Image Super-Resolution},
  journal = {arXiv Preprint},
  year      = {2020},
}
```

## Contact
**Any question regarding this work can be addressed to wangyingqian16@nudt.edu.cn.**
