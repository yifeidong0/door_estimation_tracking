# ROS Package for Door state estimation (developing)

[![Python](https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7-blue.svg?style=flat-square)](https://www.python.org/)
[![ROS Version](https://img.shields.io/badge/ROS-noetic-green?style=flat-square)](https://wiki.ros.org)

A ROS package for detecting objects using [YOLOv5](https://github.com/facebookresearch/detectron2).



The information about supported models are available [here](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md).

**Maintainer:** Yifei Dong
**Affiliation:** Robotic Systems Lab, ETH Zurich   
**Contact:** yifdong@ethz.ch   

## Table of Contents

- [ROS Package for Detectron2](#ros-package-for-detectron2)
  - [Table of Contents](#table-of-contents)
  - [Setup Instructions](#setup-instructions)
  - [Build Instructions](#build-instructions)
  - [Running Instructions](#running-instructions)
    - [Demo on RGBD Freiburg Office dataset](#demo-on-rgbd-freiburg-office-dataset)
    - [Example Usage](#example-usage)
  - [Acknowledgment](#acknowledgment)

## Setup Instructions

First, clone the project repository to the `src` directory in your catkin workspace:

```bash
git clone git@bitbucket.org:leggedrobotics/detectron2_ros.git
```

To setup the python dependencies, run the bash script:

```bash
./install.sh
```

__Note:__
The above script installs [PyTorch 1.9](https://pytorch.org/get-started/locally/) with CUDA 11.1, [detectron2 v0.5](https://github.com/facebookresearch/detectron2) and [other python modules](requirements.txt).

## Build Instructions

```bash
cd /PATH/TO/catkin_ws
catkin config -DCMAKE_BUILD_TYPE=Release
catkin build
```

## Running Instructions

The pre-trained models are downloaded by the [`detectron2.model_zoo`](https://github.com/facebookresearch/detectron2/blob/master/detectron2/model_zoo/model_zoo.py#L83) submodule so one does not need to manually download them. The very first time the model is run it downloads the checkpoint from the internet.

### Demo on RGBD Freiburg Office dataset

- To run the demo, first download the bag file using the bash script [here](bags/download_example_bag.sh):

```bash
bash ./bags/download_example_bag.sh
```

- Run the launch file which runs the nodes for playing the bag file, detecting objects and displaying the results:

```bash
roslaunch detectron2_ros demo_image_segmentation.launch
```

The ouput should be similar to as follows:

![Results on Freiburg dataset](docs/images/example_output.gif)

### Example Usage

For more generic usage, run the following launch file for Mask R-CNN with ResNet-101 head and Fast Proposal Network trained on COCO instance segmentation dataset:

```bash
roslaunch detectron2_ros image_segment.launch cfg_filename:=COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml
```

- To view the results using [`image_view`](http://wiki.ros.org/image_view):

```bash
rosrun image_view image_view image:=/detectron2/image
```

- To change the model, you just need to load a different configuration file. The argument `cfg_filename`
is set relative to the [`configs`](https://github.com/facebookresearch/detectron2/tree/master/configs) directory in detectron2 module.

The output with the model trained only on LVIS dataset should look as follows:

![Results on Gazebo simulation](docs/images/results_gazebo.png)

## Acknowledgment

The repository is inspired from the code from the following:

- <https://github.com/DavidFernandezChaves/Detectron2_ros>
