# Deep ProMPs learning for strawberry picking


<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Set Up](#Set-Up)
* [Usage](#usage)
  *  [ProMPs training](#ProMPs-training)
* [References](#references)
* [Contact](#contact)


## About The Project

The project is about Probabilistic Movement Primitives prediction through Deep models. Everything is applied to the Agri-robotics framework, in particular to the problem of strawberry picking.

The particular task considered is the Reach to Pick task so the simple action of approaching a target ripe strawberry with a robotic arm. This task should be performed using as input simply the RGB image of the strawberry cluster from the home position.

The movement is predicted using movement primitives as encoding method so that only a bunch of weights needs to be predicted by the deep neural network taking as input the image from the home position.

In particular the predicted policy is not deterministic but stochastic since the model is trained on a distribution of trajectories provided as demonstrations.


## Build With

## Getting Started

### Prerequisites

## Usage


## Contact

For any issues please contact Alessandra Tafuro at taffi98.at@gmail.com
