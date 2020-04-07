apt-get -y install git
apt-get -y install zip
apt-get -y install unzip
apt-get -y install libsndfile1
apt-get -y install ffmpeg
apt -y install git
pip install --upgrade pip
pip install -r requirements.txt

pip install -q -U -q tensorboard_plugin_profile

python -m src.train_triplet_loss