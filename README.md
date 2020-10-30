# Symmetric Parallax Attention for Stereo Image Super-Resolution
#### *Yingqian Wang, Xinyi Ying, Longguang Wang, Jungang Yang, Wei An, Yulan Guo*
<br>

## Overview
<img src="https://raw.github.com/YingqianWang/iPASSR/main/Figs/Network.png" width="800"><br>
<br>

## Download the Results
**We share the quantitative and qualitative results achieved by our iPASSR on all the test sets for both 2xSR and 4xSR. Then, researchers can compare their algorithms to our method without performing inference. Results are available at [Baidu Drive](https://pan.baidu.com/s/12R9ZxB78QL4suy8ZuJ88Uw) (Key: NUDT).**
<br>

## PyTorch Implementation

### Requirement
* **PyTorch 1.3.0, torchvision 0.4.1. The code is tested with python=3.7, cuda=9.0.**
* **Matlab (For training/test data generation and performance evaluation)**

### Train
* **Download the training sets from [Baidu Drive](https://pan.baidu.com/s/173UGmmN0rtOUghIT40oy8w) (Key: NUDT) and unzip them to `./data/train/`.** 
* **Run `./data/train/GenerateTrainingPatches.m` to generate training patches.**
* **Run `train.py` to perform training. Checkpoint will be saved to  `./log/`.**

### Test
* **Download the test sets and unzip them to `./data`. Here, we provide the full test sets used in our paper on [Baidu Drive](https://pan.baidu.com/s/1SIYGcMBEDDZ0wYrkxL9bnQ) (Key: NUDT).** 
* **Run `test.py` to perform a demo inference. Results (`.png` files) will be saved to `./results`.**
* **Run `evaluation.m` to calculate PSNR and SSIM scores.**
<br>

## Quantitative Results
<img src="https://raw.github.com/YingqianWang/iPASSR/main/Figs/Quantitative.png" width="1000"><br>
<br>

## Qualitative Results [demo video](https://wyqdatabase.s3-us-west-1.amazonaws.com/iPASSR_visual_comparison.mp4)
<img src="https://raw.github.com/YingqianWang/iPASSR/main/Figs/2xSR.png" width="1000"><br>
<img src="https://raw.github.com/YingqianWang/iPASSR/main/Figs/4xSR.png" width="1000"><br>
<img src="https://raw.github.com/YingqianWang/iPASSR/main/Figs/RealSR.png" width="1000"><br>
<br>

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
<br>

## Contact
**Any question regarding this work can be addressed to [wangyingqian16@nudt.edu.cn](wangyingqian16@nudt.edu.cn).**
