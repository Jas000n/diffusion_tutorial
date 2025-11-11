# 🧠 Diffusion Tutorials: MNIST & CIFAR-10


<img src="pics/demo.gif" width="100">

> A minimal, reproducible tutorial for learning diffusion policies — with two runnable examples:  
> 
> - ✨ **MNIST** (Lightweight) — Runs on entry-level GPUs or CPUs, using a minimal UNet architecture for grayscale digit generation.
> - 💥 **CIFAR-10 (Full)** — Runs on GPUs, using a deeper UNet architecture for color image generation.

A glimpse of the two tutorials:
* MNIST: ![alt text](./pics/MNIST.png)
* CIFAR-10:![alt text](./pics/cifar10.png)
    
## 🚀 1. Introduction

This tutorial demonstrates how to build a Diffusion Policy from scratch using PyTorch.
We start with a simple grayscale example on MNIST, then extend it to a larger UNet and CIFAR-10.

You’ll learn:

* The basics of diffusion models (forward noise / reverse denoising)

* How to implement a UNet backbone for diffusion

* How to train and sample from diffusion models on MNIST and CIFAR-10

## ⚙️ 2. Setup
```
git clone https://github.com/yourname/diffusion-tutorial.git
cd diffusion-tutorial
conda create -n diffusion python=3.10
conda activate diffusion
pip install -r requirements.txt
```
## 🧩 3. About Diffusion
> Every block of stone has a statue inside it, and it is the task of the sculptor to discover it.       - Michelangelo
![alt text](./pics/Michelangelo.png)
### 3.1 How do we dig out image from noise?
In the same vein, consider a blurry or noisy image as an uncarved stone, from which we gradually dig out the true image step by step.
### 3.2 Forward Pass
In order to *"dig out"* the image, we first have to learn how to rebuild the uncarved stone.

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?q(x_t%20%7C%20x_{t-1})%20%3D%20%5Cmathcal%7BN%7D%5C!%5Cleft(x_t%3B%20(1%20-%20%5Cbeta_t)x_{t-1}%2C%20%5Cbeta_t%20I%5Cright)" />
</p>

- <img src="https://latex.codecogs.com/svg.image?x_0" />: Original (clean) data sample  
- <img src="https://latex.codecogs.com/svg.image?x_t" />: Noisy sample at step *t*  
- <img src="https://latex.codecogs.com/svg.image?%5Cbeta_t" />: Variance (noise level) at step *t*  
- <img src="https://latex.codecogs.com/svg.image?I" />: Identity matrix  

Because the forward process is Gaussian and Markov, all the noise additions are linear and independent.  
- Linear combination of Gaussian (Normal distribution) is still Gaussian.  
- A Markov process means that the next state depends only on the current state, not on any earlier ones.

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?q(x_t%20%7C%20x_0)%20%3D%20%5Cmathcal%7BN%7D%5C!%5Cleft(x_t%3B%20%5Csqrt%7B%5Cbar%7B%5Calpha%7D_t%7D%20x_0%2C%20(1%20-%20%5Cbar%7B%5Calpha%7D_t)I%5Cright)" />
</p>

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?%5Ctext%7Bwhere%7D%20%5Cquad%20%5Cbar%7B%5Calpha%7D_t%20%3D%20%5Cprod_%7Bi%3D1%7D%5Et%20(1%20-%20%5Cbeta_i)" />
</p>

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?x_t%20%3D%20%5Csqrt%7B%5Cbar%7B%5Calpha%7D_t%7D%20x_0%20%2B%20%5Csqrt%7B1%20-%20%5Cbar%7B%5Calpha%7D_t%7D%20%5Cepsilon%2C%20%5Cquad%20%5Cepsilon%20%5Csim%20%5Cmathcal%7BN%7D(0%2C%20I)" />
</p>

![alt text](pics/forward_diffusion_cifar10.png)
### 3.3 Reverse Pass
We will have the model learning this process, but reversely.
![alt text](pics/reverse_diffusion.drawio.svg)
### 3.4 UNet(But not necessarily has to be)
![alt text](pics/unet.png)

But in general, all generative model could work, e.g. DiT.
## 💪 4. Get your hands dirty
#### 4.1 MNIST Tutorial (Lightweight)
> 💡 For users with entry level GPU or without a GPU, or if you just want to understand the diffusion core idea.
```
cd MNIST
python main.py
```
See the diffision process:
```
python visual.py
```


#### 4.2 CIFAR-10 Tutorial (Full Version)
> ⚡️ For users with a high end GPU — this version trains a larger UNet on color images.
```
cd CIFAR10
python main.py
```
See the diffision process:
```
python visual.py
```

## Last but not least

Everything is a Distribution! Diffusion models are not limited to generating images, they learn **distributions**.  
That means: *anything that can be represented as data can be modeled as a diffusion process.*
### 🤖 Robot Motion Generation
### 🏠 Interior Design Generation
### 🎶 Music or Audio Generation
### 🧬 Molecular & Protein Design
