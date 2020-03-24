# Deep Embedded Music
Tile2Vec embedding applied to audio

## Docker

### Start the container

Personal:
```shell script
docker run -it -v ${PWD}:/tf/ -v /Volumes/Magneto/Datasets/:/tf/datasets/ --rm --name deep-embedded-music tensorflow/tensorflow:2.1.0-py3
```

GPU:
```shell script
docker run -it -v ${PWD}:/tf/ --rm --name deep-embedded-music tensorflow/tensorflow:2.1.0--gpu-py3 /bin/bash
```

### Run tensorboard

Personal:
```shell script
docker run -p 6006:6006 --rm -v ${PWD}:/tf/ -v /Volumes/Magneto/Datasets/:/tf/datasets/ tensorflow/tensorflow:2.1.0-py3 tensorboard --bind_all --logdir tf/experiments/DCASE/results/
```

GPU:
```shell script
docker run -p 6006:6006 --rm -v ${PWD}:/tf/ tensorflow/tensorflow:2.1.0-gpu-py3 tensorboard --bind_all --logdir tf/experiments/DCASE/results/
```

### Run script

GPU:
```shell script
python -m src.train_triplet_loss
```
