# Graph Geometry-Preserving Autoencoders
The official repository for \<Graph Geometry-Preserving Autoencoders\> (Lim, Kim, Lee, Jang, and Park, ICML 2024)

> This paper proposes Graph Geometry-Preserving Autoencoder (GGAE), a regularized autoencoder trained by minimizing the *reconstruction error + distortion measure of graph geometry*. It produces latent representation that preserves shortest-path distances along a graph connecting data points by semantic distances or similarity.


## Preview
### 1. Swiss Roll
<center>
<div class="imgCollage">
<span style="width: 31.8%"><img src="./figure/swissroll_ideal.png" width="250" height="250"/></span>
<span style="width: 31.8%"><img src="./figure/swissroll_ae.png" width="250" height="250"/> </span>
<span style="width: 31.8%"><img src="./figure/swissroll_ggae.png" width="250" height="250"/> </span>
</div>
  <I>Figure 1: <b>(Left)</b> An ideal latent representation, <b>(Middle)</b> distorted representation obtained by AE, and <b>(Right)</b> graph geometry-preserving representation obtained by GGAE. </I>
</center>
<br>

### 2. Rotating MNIST
<center>
<div class="imgCollage">
<span style="width: 31.8%"><img src="./figure/rotatingmnist_ideal.png" width="250" height="250"/></span>
<span style="width: 31.8%"><img src="./figure/rotatingmnist_ae_video.gif" width="250" height="250"/> </span>
<span style="width: 31.8%"><img src="./figure/rotatingmnist_ggae_video.gif" width="250" height="250"/> </span>
</div>
  <I>Figure 2: <b>(Left)</b> An ideal latent representation, <b>(Middle)</b> distorted representation obtained by AE, and <b>(Right)</b> graph geometry-preserving representation obtained by GGAE. </I>
</center>
<br>

### 3. dSprites
<center>
<div class="imgCollage">
<span style="width: 31.8%"><img src="./figure/dsprites_ideal.png" width="250" height="250"/></span>
<span style="width: 31.8%"><img src="./figure/dsprites_ae.png" width="250" height="250"/> </span>
<span style="width: 31.8%"><img src="./figure/dsprites_ggae.png" width="250" height="250"/> </span>
</div>
  <I>Figure 3: <b>(Left)</b> An ideal latent representation, <b>(Middle)</b> distorted representation obtained by AE, and <b>(Right)</b> graph geometry-preserving representation obtained by GGAE. </I>
</center>
<br>

## Environment

The project is developed under a standard PyTorch environment.
- python 3.7
- numpy
- matplotlib
- scikit-learn
- tqdm
- argparse
- omegaconf
- tensorboardX
- torch 1.13.1

## Preparing the Datasets
### Rotating MNIST
Run `./notebook/01_RotatingMNIST_dataset_generation.ipynb`.
### dSprites
Run `./notebook/02_dSprites_dataset_generation.ipynb`.

## Running
### 1. Train
#### 1.1 AE
```js
python train.py --config configs/swissroll/swissroll_ae_z2.yml
python train.py --config configs/rotatingmnist/rotatingmnist_ae_z2.yml
python train.py --config configs/dsprites/dsprites_ae_z3.yml
```
#### 1.2 GGAE
```js
python train.py --config configs/swissroll/swissroll_ggae_z2.yml
python train.py --config configs/rotatingmnist/rotatingmnist_ggae_z2.yml
python train.py --config configs/dsprites/dsprites_ggae_z3.yml
```
- The results will be saved in './results' directory.

### 2. Tensorboard 
```js
tensorboard --logdir results/
```

## Citation
If you found this library useful in your research, please consider citing:
```
@inproceedings{limgraph,
  title={Graph Geometry-Preserving Autoencoders},
  author={Lim, Jungbin and Kim, Jihwan and Lee, Yonghyeon and Jang, Cheongjae and Park, Frank C},
  booktitle={Forty-first International Conference on Machine Learning}
}
```