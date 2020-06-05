# Deep embedded Music
Implementation of an embedding space for audio using only unsupervised machine learning. The approach is to adapt
 triplet loss to an unsupervised setting, using the distributional hypothesis from natural language, which states
  that words appearing in similar contexts tend to have similar meanings. The idea is based on 
  [Tile2Vec](https://arxiv.org/abs/1805.02855), which applies this same hypothesis to geographical tiles. The
   resulting embedding space is evaluated on two datasets, the DCASE 2018 task 5 dataset and a music dataset.
   
## DCASE embedding space

![DCASE embedding space](http://fabiangroeger.com/wp-content/uploads/2020/06/deep_embedded_music_dcase_space.gif)

## Music embedding space

![Music embedding space](http://fabiangroeger.com/wp-content/uploads/2020/06/deep_embedded_music_music_space.gif)

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
-  [Abstract](#abstract)
-  [Usage](#usage)

## Abstract
For humans to distinguish between music genres is from easy to difficult, depending on how similar the given genres are. It is rather simple to distinguish music songs, for example, of classical music and techno or country and hip-hop rap. However, the difficulty increases if songs within the same genre are compared. When it comes to finding the similarity between songs or to sort a list of songs by their similarity, it is hard, even for humans. In this thesis, an alternative learning strategy is developed that exploits basic semantic properties of sound that are not grounded by an explicit label. This thesis intends to adapt Tile2Vec, an image embedding method, to audio streams and evaluate its performance on the "SINS, DCASE 2018: task 5" noise detection dataset as well as on a music dataset, consisting of songs of seven sub-genres. Tile2Vec is an adaption of triplet loss to an unsupervised setting using the distributional hypothesis from natural language, which states that words appearing in similar contexts tend to have similar meanings.

For this work, log Mel spectrograms were used to represent the audios in a more compact form. The selection of triplets is made by using the temporal proximity, and as the network, a state-of-the-art ResNet18 architecture is used. The embedding space is created by training a ResNet18 model utilising the triplet loss and using the log Mel spectrogram as input. This creates a lower-dimensional space, where the distances within, represent the similarities between the data points.
    
The first experiments with the noise detection dataset confirmed that the embedding space could learn semantic properties even without any label. The conducted error analysis revealed misclassified segments and microphones malfunctions in the dataset. Moreover, the embedding space showed clusters of similar-sounding sounds, regardless of its label. A simple linear logistic classifier trained on the resulting embeddings is compared to the results from the DCASE challenge. The best classifier of this thesis reached an F1 score of 62.19\%, while most of the models submitted to the challenge accomplished results 80\% and higher.
    
The same architecture is used to train an embedding space for music, which showed similar significant results. It revealed clear clusters when applying k-Means to it. Moreover, each cluster showed a significantly higher amount of segments of a specific class, which demonstrates that the model succeeded in building clusters for each genre. An expert affirmed, that the neighbours of each segment, even if from a different category, build a neighbourhood, which sounds similar and is therefore correct. When training the simple classifier on the music embedding space, it achieved an astonishing F1 score of approximately 84\% on unseen test data. The expert, who performed the qualitative analysis, was thrilled and impressed with the results.
    
To conclude, the learnt embedding spaces from both datasets successfully learnt to find similarities regardless of its ground truth even if the categories are very similar to each other.

## Usage
First you will need to clone the repository:
```shell script
git clone https://github.com/FabianGroeger96/deep-embedded-music.git
```

### Train embedding space
The following docker command can be executed using the `onstart.sh` script, to train the embedding space using triplet loss on a GPU. The container will start, and the training procedure begins right away.
```shell script
docker run -it -v ${PWD}:/tf/ -w /tf --name deep-embedded-music \
--gpus all --privileged=true \
tensorflow/tensorflow:2.1.0-gpu-py3 /bin/bash ./onstart.sh
```

>Take note of your Docker version with docker -v. Versions earlier than 19.03 require nvidia-docker2 and the --runtime=nvidia flag. On versions including and after 19.03, you will use the nvidia-container-toolkit package and the --gpus all flag.

 The training process can be started manually without the `onstart.sh` script, by starting the docker container and then executing the python script `train_triplet_loss.py` manually.
 ```shell script
cd deep-embedded-music
python -m src.train_triplet_loss
```
If the tensorboard profiler wants to be used, by setting the parameter `use_profiler = 1`, the tensorflow nightly
 build `tensorflow/tensorflow:nightly-gpu-py3` has to be used to start the docker container as well as the option
  `--privileged=true`. This is due to NVIDIA CUPTI libary API change: 
  [https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm-cupti](https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm-cupti)

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
tensorflow/tensorflow:2.1.0-py3 \
tensorboard --bind_all --logdir tf/experiments/DCASE/results/
```
