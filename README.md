# Deep Embedded Music
Tile2Vec embedding applied to audio

## Docker

### Start the container

```shell script
docker run -it -p 8888:8888 -p 6006:6006 -v /Users/fabiangroger/Documents/_git/GitHub/deep-embedded-music:/tf/ -v /Volumes/Magneto/Datasets/:/tf/datasets/ --rm --name deep-embedded-music deep-embedded-music
```

### Run tensorboard

```shell script
docker run -p 6006:6006 --rm -v /Users/fabiangroger/Documents/_git/GitHub/deep-embedded-music:/tf/ -v /Volumes/Magneto/Datasets/:/tf/datasets/ tensorflow/tensorflow:2.1.0-py3 tensorboard --bind_all --logdir tf/experiments/DCASE/results/
```
