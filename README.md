# Deep ProMPs learning for strawberry picking


<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Set Up](#Set-Up)
   * [Dataset](#Dataset)
   * [Code](#Code)
* [Usage](#usage)
* [References](#references)
* [Contact](#contact)


## About The Project

The project is about Probabilistic Movement Primitives prediction through Deep models. Everything is applied to the Agri-robotics framework, in particular to the problem of strawberry picking.

The particular task considered is the Reach to Pick task so the simple action of approaching a target ripe strawberry with a robotic arm. This task should be performed using as input simply the RGB image of the strawberry cluster from the home position.

The movement is predicted using movement primitives as encoding method so that only a bunch of weights needs to be predicted by the deep neural network taking as input the image from the home position.

In particular the predicted policy is not deterministic but stochastic since the model is trained on a distribution of trajectories provided as demonstrations.

Two baselines have been followed to predict the mean and covariance of the ProMps weights describing the trajectory of to reach a target berry.

The first one is: Auto Encoder + Multi Layer Perceptron model.

![AE](img/AE+MLP.png)

The second one is: Variational Auto Encoder + Multi Layer Perceptron model.

![VAE](img/VAE+MLP.png)

## Build With
All the models are trained using [Tensorflow 2.7](https://pypi.org/project/tensorflow/) .

## Getting Started

### Dataset
The complete dataset is contained in the folder ```dataset/```. It is made by RGB images and the annotations of the ProMs weights of the collected demonstrations in  ```dataset/annotations```. The RGB images are separated in 3 folders: 
- ```dataset/rgb_segmented_white``` contains 252 RGB images with the annotated bounding box extracted with Detectron2 for which the trajectory has been collected.
- ```dataset/additional_segmented``` contains 505 addtional RGB images with the annotated bounding box extracted with Detectron2 for which the trajectory has not been collected.
- ```dataset/rgb_tot_white``` merges the 2 previous folders.

### Code
In the ```code/``` folder there are the scripts to treìain the Autoencoder and Variational Autoencoder models
- ```code/Autoencoder```
- ```code/VAE```

and the scripts to train the 7 models (one for each joint) using repectively the AE or VAE

- ```code/Models_AE```
- ```code/Models_VAE```


## Usage
To train or evaluate the AE or VAE model run:

```
python code/Autoencoder/train_test.py
```
or

```
python code/VAE/train_test.py
```

To train or evaluate the models for ProMPs prediciton of the first joint based on AE or VAE run:

```
python code/Models_AE/J1/train_test.py
```
or

```
python code/Models_VAE/J1/train_test.py
```


## References

#### ProMPs:

http://eprints.lincoln.ac.uk/id/eprint/25785/1/5177-probabilistic-movement-primitives.pdf

## Contact

For any issues please contact Alessandra Tafuro at taffi98.at@gmail.com
