# Anime_GAN
This repository records my GAN models with Anime. 

# Dependencies
* [Python 3.6+](https://www.continuum.io/downloads)
* [PyTorch 0.4.0](http://pytorch.org/)
* [numpy 1.14.1, matplotlib 2.2.2, scipy 1.1.0](https://www.scipy.org/install.html)
* [imageio 2.3.0](https://pypi.org/project/imageio/)
* [tqdm 4.24.0](https://pypi.org/project/tqdm/)
* [keras 2.1.2](https://keras.io/#installation)(for WGAN_keras)
* [tensorflow 1.2.1](https://www.tensorflow.org/install/)(for WGAN_keras)

# DCGAN
you need to download the [dataset](https://1drv.ms/u/s!AgBYzHhocQD4g0_Fr-mC-DYfWahJ) named **faces.zip**, then execute the first step——Cloning the repository, and extract and move it in `Anime_GAN-master/DCGAN/`.
## 1. Cloning the repository
```bash
$ git clone https://github.com/FangYang970206/Anime_GAN.git
$ cd Anime_GAN-master/DCGAN/
```
## 2. run the code
```bash
$ python main.py 
```
## 3. 500 epochs result
![](result/DCGAN_500.png)
---
# WGAN
The WGAN implemented by pytorch has bugs. If anyone can find them, i will appreciate you so much! 
I implement a keras version Anime_WGAN. Like DCGAN, you should download the [dataset](https://1drv.ms/u/s!AgBYzHhocQD4g0_Fr-mC-DYfWahJ) named **faces.zip**, then execute the first step——Cloning the repository, and extract and move it in `Anime_GAN-master/WGAN_keras/faces/`.
## 1. Cloning the repository
```bash
$ git clone https://github.com/FangYang970206/Anime_GAN.git
$ cd Anime_GAN-master/WGAN_keras/
```
## 2. run the code
```bash
$ python wgan_keras.py 
```
## 3. 100000 steps result
![](result/wgan_keras_result.png)
