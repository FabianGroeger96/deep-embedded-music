# Deep Embedded Music
Tile2Vec embedding applied to audio

## Docker

### Start the container

Personal:
```shell script
docker run -it -v $(PWD):/tf/ -v /Volumes/Magneto/Datasets/:/tf/data/ -w /tf -d --rm --name deep-embedded-music \
tensorflow/tensorflow:2.1.0-py3 /bin/bash ./onstart.sh
```

GPU:
```shell script
docker run -it -v ${PWD}:/tf/ -w /tf --rm --name deep-embedded-music-1 -e NVIDIA_VISIBLE_DEVICES=1 \
tensorflow/tensorflow:2.1.0-gpu-py3 /bin/bash ./onstart.sh
```

### Run tensorboard

Personal:
```shell script
docker run -it -p 6006:6006 --rm -v $(PWD):/tf/ --name deep-embedded-music-tensorboard \
tensorflow/tensorflow:2.1.0-py3 \
tensorboard --bind_all --logdir tf/experiments/DCASE/results/
```

GPU:
```shell script
docker run -it -p 6006:6006 --rm -v ${PWD}:/tf/ --name deep-embedded-music-tensorboard \
tensorflow/tensorflow:2.1.0-gpu-py3 \
tensorboard --bind_all --logdir tf/experiments/DCASE/results/
```
Detach container:
ctrl+p + ctrl+q

Browser:
http://gpu04:6006

### Run script

```shell script
cd deep-embedded-music
python -m src.train_triplet_loss
```

### Delete folders

```shell script
docker run -v $(pwd):/app -it ubuntu /bin/bash
rm -R to_be_deleted/
```
