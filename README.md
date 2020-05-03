# Deep embedded Music
Implementation of an embedding space for audio using only unsupervised machine learning. The approach is to adapt
 triplet loss to an unsupervised setting, using the distributional hypothesis from natural language, which states
  that words appearing in similar contexts tend to have similar meanings. The idea is based on 
  [Tile2Vec](https://arxiv.org/abs/1805.02855), which applies this same hypothesis to geographical tiles. The
   resulting embedding space is evaluated on two datasets, the DCASE 2018 task 5 dataset and a music dataset.

## Requirements
- [TensorFlow 2.1](https://github.com/tensorflow/tensorflow) 
- [pandas](https://pandas.pydata.org/) 
- [numpy](https://numpy.org/) 
- [scipy](https://www.scipy.org/) 
- [scikit-learn](https://scikit-learn.org/stable/) 
- [librosa](https://github.com/librosa/librosa)
- [matplotlib](https://matplotlib.org/) 
- [seaborn](https://seaborn.pydata.org/)

## Table Of Contents
-  [Introduction](#introduction)
-  [Usage](#usage)

## Introduction
The last few years have seen significant advances in non-speech audio processing, as popular deep learning architectures, developed in the speech and image processing communities, have been ported to this relatively understudied domain. However, these data-hungry neural networks are not always matched to the available training data in the audio domain. While unlabeled audio is easy to collect, manually labelling data for each new sound application remains notoriously costly and time-consuming. Therefore a considerable amount of research in recent years focuses on the application of unsupervised machine learning in the audio domain.

Another prominent problem in the audio domain is distinguishing between pieces of music, which is for humans from easy to difficult. It is mostly simple to distinguish music genres, like classical music or techno. However, the difficulty increases when songs within the same genre will need to be compared. It is further relatively hard to find some similarity measure to compare sounds or to sort a bunch of songs by their similarity.

This research seeks to alleviate this incongruity by developing alternative learning strategies that exploit basic semantic properties of sound that are not grounded by an explicit label. This alternative will then be applied on a noise detection dataset and a music dataset to show the applicability and usefulness of the approach. Further, it attempts to show that the representation learnt, provides a similarity measure for both of the datasets, which can then be used to compare specific audios or sort them by similarity.

## Usage

### Train triplet loss
The following docker command can be executed using the `onstart.sh` script, to train the triplet loss architecture on a GPU. The container will start, and the training procedure begins right away.
```shell script
docker run -it -v ${PWD}:/tf/ -w /tf --name deep-embedded-music-0 \
--gpus all --privileged=true \
tensorflow/tensorflow:2.1.0-gpu-py3 /bin/bash ./onstart.sh
```

>Take note of your Docker version with docker -v. Versions earlier than 19.03 require nvidia-docker2 and the --runtime=nvidia flag. On versions including and after 19.03, you will use the nvidia-container-toolkit package and the --gpus all flag. Both options are documented on the page linked above.

If you want to start the training without the `onstart.sh` script, you can start the docker container and then
 executing the python script `train_triplet_loss.py` manually.
 ```shell script
cd deep-embedded-music
python -m src.train_triplet_loss
```
If the tensorboard profiler wants to be used, by setting the parameter `use_profiler = 1`, the tensorflow nightly
 build `tensorflow/tensorflow:nightly-gpu-py3` has to be used to start the docker container as well as the option
  `--privileged=true`. This is due to NVIDIA CUPTI libary API change: 
  [https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm-cupti](https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm-cupti).

### Train classifier
A separate python script has to be called named `train_classifier.py`, to train the classifier on top of the
 embedding architecture. The `--model_to_load` argument specifies which embedding model should be used to train the
  classifier, the classifier will be trained using the same dataset.
 ```shell script
cd deep-embedded-music
python -m src.train_classifier --model_to_load results/path_to_model/
```

### Run tensorboard
The tensorboard can be used, to examine the results from the trained models. The tensorboard can be started by
 executing the docker command below. The tensorboard is then available on [http://localhost:6006](http://localhost:6006). The `--logdir` specifies the location of the results to visualise.
```shell script
docker run -it -p 6006:6006 --rm -v ${PWD}:/tf/ --name deep-embedded-music-tensorboard \
tensorflow/tensorflow:2.1.0-gpu-py3 \
tensorboard --bind_all --logdir tf/experiments/DCASE/results/
```
