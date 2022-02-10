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
    * [Joint Space](#Joint-Space)
    * [Task Space](#Task-Space)
* [Results](#results)
    * [Joint Space](#Joint-Space)
    * [Task Space](#Task-Space)
* [References](#references)
* [Contact](#contact)


## About The Project

The project is about Probabilistic Movement Primitives prediction through Deep models. Everything is applied to the Agri-robotics framework, in particular to the problem of strawberry picking.

The particular task considered is the Reach to Pick task so the simple action of approaching a target ripe strawberry with a robotic arm. This task should be performed using as input simply the RGB image of the strawberry cluster from the home position.

The movement is predicted using movement primitives as encoding method so that only a bunch of weights needs to be predicted by the deep neural network taking as input the image from the home position.

In particular the predicted policy is not deterministic but stochastic since the model is trained on a distribution of trajectories provided as demonstrations.

Three baselines have been followed to predict the mean and covariance of the ProMps weights describing the trajectory of to reach a target berry.

The first one is: Auto Encoder + Multi Layer Perceptron model.

![AE](img/AE+MLP.png)

The second one is: Variational Auto Encoder + Multi Layer Perceptron model.

![VAE](img/VAE+MLP.png)

The last one is: Variational Auto Encoder + Multi Layer Perceptron model conditioned with the x,y pixel coordinates of the enter of the bounding box of the target barry.

![VAE](img/CVAE+MLP.png)

All the baselines have been tested both on joint and in task space.

## Build With

All the models are trained using [Tensorflow 2.7](https://pypi.org/project/tensorflow/) .

## Getting Started

### Dataset

The complete dataset is contained in the folder ```dataset/```. It is made by RGB images and the annotations of the ProMs weights of the collected demonstrations in  ```dataset/annotations```. The RGB images are separated in 3 folders: 
- ```dataset/rgb_segmented_white``` contains 252 RGB images with the annotated bounding box extracted with Detectron2 for which the trajectory has been collected.
- ```dataset/additional_segmented``` contains 505 addtional RGB images with the annotated bounding box extracted with Detectron2 for which the trajectory has not been collected.
- ```dataset/rgb_tot_white``` merges the 2 previous folders.

The complete dataset can be downloaded at this [link](https://drive.google.com/drive/folders/1k7QSSXEPNbt2EQn0ZCS2IrFKZuag864N?usp=sharing)

In the ```preprocessing/``` folder there are all the codes for the preprocessing of the collected images (segmentation) and trajectories.

### Code

In the ```code/``` folder there are the scripts to train the Autoencoder and Variational Autoencoder models
- ```code/Autoencoder```
- ```code/VAE```

and the scripts to train the 7 models (one for each joint) using repectively the AE or VAE or cVAE

- ```code/Models_AE```
- ```code/Models_VAE```
- ```code/Models_cVAE```


## Usage

To train or evaluate the AE or VAE model run:

```
python code/Autoencoder/train_test.py
```
or

```
python code/VAE/train_test.py
```
### Joint space

To train or evaluate the models for ProMPs prediciton of the first joint based on AE or VAE or cVAE run:

```
python code/Models_AE/J1/train_test.py
```
or

```
python code/Models_VAE/J1/train_test.py
```

or 
```
python code/Models_cVAE/J1/train_test.py
```
The same applies for the other joints.

### Task space

To train or evaluate the models for ProMPs prediciton of the coordinate X based on AE or VAE or cVAE run:

```
python code/Models_AE_task/X/train_test.py
```
or

```
python code/Models_VAE_task/X/train_test.py
```

or 
```
python code/Models_cVAE_task/X/train_test.py
```
The same applies for the other coordinates and quaternions components.

### Results

#### Joint space

Comparing the three architectures the performances increase going from AE to VAE to cVAE models!

RMSE(mean_traj)+ RMSE(cov_traj)

| Joints      | Autoencoder     | Variational Autoencoder     | Error reduciton     | Conditional Variational Autoencoder     | Error reduciton     |
| ----------- | --------------- | --------------------------- |-------------------- | --------------------------------------- | ------------------- | 
| J1          | 0.000544        | 0.000304                    | - 44%               | 0.000132                                | - 56%               | 
| J2          | 0.004899        | 0.001612                    | - 66%               | 0.001085                                | - 37.5%             | 
| J3          | 0.009301        | 0.000456                    | - 95%               | 0.000146                                | - 67.9%             | 
| J4          | 0.003711        | 0.002957                    | - 21%               | 0.000920                                | - 68.8%             | 
| J5          | 0.002424        | 0.002302                    | - 4%                | 0.002284                                | - 0.9%              | 
| J6          | 0.002901        | 0.002660                    | - 10%               | 0.001745                                | - 35%               | 
| J7          | 0.002140        | 0.000824                    | - 61%               | 0.000687                                | - 16.6%             | 

#### Task space

Comparing the three architectures the performances increase going from AE to VAE to cVAE models!

RMSE(mean_traj)+ RMSE(cov_traj)

| Joints      | Autoencoder     | Variational Autoencoder     | Error reduciton     | Conditional Variational Autoencoder     | Error reduciton     |
| ----------- | --------------- | --------------------------- |-------------------- | --------------------------------------- | ------------------- | 
| X           | 0.000127        | 0.000114                    | - 10%               | 0.000132                                | - 56%               | 
| Y           | 0.001531        | 0.000297                    | - 80%               | 0.001085                                | - 37.5%             | 
| Z           | 0.000092        | 0.000018                    | - 80%               | 0.000146                                | - 67.9%             | 
| Q1          | 0.000153        | 0.000103                    | - 21%               | 0.000920                                | - 68.8%             | 
| Q2          | 0.002424        | 0.000050                    | - 97%               | 0.002284                                | - 0.9%              | 
| Q3          | 0.000115        | 0.000022                    | - 80%               | 0.001745                                | - 35%               | 
| Q4          | 0.000078        | 0.000060                    | - 23%               | 0.000687                                | - 16.6%             | 


## References

#### ProMPs:

http://eprints.lincoln.ac.uk/id/eprint/25785/1/5177-probabilistic-movement-primitives.pdf

## Contact

For any issues please contact Alessandra Tafuro at taffi98.at@gmail.com
