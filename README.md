# iPASSR
**PyTorch implementation of "*Symmetric Parallax Attention for Stereo Image Super-Resolution*", NTIRE workshop at CVPR 2021 (oral). [<a href="https://arxiv.org/pdf/2011.03802.pdf">pdf</a>], [<a href="https://wyqdatabase.s3-us-west-1.amazonaws.com/iPASSR_visual_comparison.mp4">demo video</a>].**<br><br>

## *Highlights:*
#### 1. *We develop a Siamese network equipped with a bi-directional PAM to super-resolve both left and right images.*
  <p align="center"> <img src="https://raw.github.com/YingqianWang/iPASSR/master/Figs/Network.png" width="100%"></p>
  
#### 2. *We propose an inline occlusion handling scheme to deduce occlusions from parallax attention maps.*
  <p align="center"><img src="https://raw.github.com/YingqianWang/iPASSR/master/Figs/OcclusionDeduce.png" width="40%"><img src="https://raw.github.com/YingqianWang/iPASSR/master/Figs/OcclusionMask.png" width="55%"></p>
  
#### 3. *We design several illuminance-robust losses to enhance stereo consistency.* [[demo video](https://wyqdatabase.s3-us-west-1.amazonaws.com/iPASSR_illuminance_change.mp4)]
  <p align="center"> <img src="https://raw.github.com/YingqianWang/iPASSR/master/Figs/ResLoss.png" width="100%"></p>
  
#### 4. *Our iPASSR significantly outperforms PASSRnet with a comparable model size.*
  <p align="center"> <img src="https://raw.github.com/YingqianWang/iPASSR/master/Figs/Quantitative.png" width="100%"></p>
  <p align="center"> <img src="https://raw.github.com/YingqianWang/iPASSR/master/Figs/2xSR.png" width="100%"></p>
  <p align="center"> <img src="https://raw.github.com/YingqianWang/iPASSR/master/Figs/4xSR.png" width="100%"></p>
  <p align="center"> <img src="https://raw.github.com/YingqianWang/iPASSR/master/Figs/RealSR.png" width="100%"></p>

## Download the Results
**We share the quantitative and qualitative results achieved by our iPASSR on all the test sets for both 2xSR and 4xSR. Then, researchers can compare their algorithms to our method without performing inference. Results are available at [Google Drive](https://drive.google.com/file/d/1LKe86Cet8LFRG-mgG9oYAJlyiJljDJb0/view?usp=sharing) and [Baidu Drive](https://pan.baidu.com/s/1w8RtQau2RoY89jsFvMCStw) (Key: NUDT).**
<br>

## Requirement
* **PyTorch 1.3.0, torchvision 0.4.1. The code is tested with python=3.7, cuda=9.0.**
* **Matlab (For training/test data generation and performance evaluation)**

## Train
* **Download the training sets from [Baidu Drive](https://pan.baidu.com/s/173UGmmN0rtOUghIT40oy8w) (Key: NUDT) and unzip them to `./data/train/`.** 
* **Run `./data/train/GenerateTrainingPatches.m` to generate training patches.**
* **Run `train.py` to perform training. Checkpoint will be saved to  `./log/`.**

## Test
* **Download the test sets and unzip them to `./data`. Here, we provide the full test sets used in our paper on [Google Drive](https://drive.google.com/file/d/1LQDUclNtNZWTT41NndISLGvjvuBbxeUs/view?usp=sharing) and [Baidu Drive](https://pan.baidu.com/s/1SIYGcMBEDDZ0wYrkxL9bnQ) (Key: NUDT).** 
* **Run `test.py` to perform a demo inference. Results (`.png` files) will be saved to `./results`.**
* **Run `evaluation.m` to calculate PSNR and SSIM scores.**

## Some Useful Recources
* **The 2x/4x models of EDSR/RDN/RCAN retrained on stereo image datasets. [Google Drive](https://drive.google.com/drive/folders/1ovN_O34qTToI6jiaL69T1_vlHv2jwu0f?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1GrKi8taYnEColKz_wa5f4w) (Key: NUDT).**
* **The 2x/4x results of StereoSR. [Google Drive](https://drive.google.com/file/d/1nt6uBhyze0Cx0VdI3O7PsxN51qZNqIKd/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1E_w0BHeaNzjQEyc22To5FQ) (Key: NUDT).**
* **The 4x results of SRRes+SAM. [Google Drive](https://drive.google.com/file/d/1uOM-LUDandK2lX9p2mknc_J1C81Zbn8e/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1KIr0v1XfBPPQR-MEMTUbwg) (Key: NUDT).**

## Citiation
**We hope this work can facilitate the future research in stereo image SR. If you find this work helpful, please consider citing:**
```
@InProceedings{iPASSR,
    author    = {Wang, Yingqian and Ying, Xinyi and Wang, Longguang and Yang, Jungang and An, Wei and Guo, Yulan},
    title     = {Symmetric Parallax Attention for Stereo Image Super-Resolution},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2021},
    pages     = {766-775}
}

```

## Contact
**Any question regarding this work can be addressed to [wangyingqian16@nudt.edu.cn](wangyingqian16@nudt.edu.cn).**
