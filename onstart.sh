apt-get update

apt-get -y install git
apt-get -y install libav-tools
apt-get -y install libsndfile1-dev
apt-get -y install libsndfile1

pip install --upgrade pip
pip install -r requirements.txt

pip install -q -U -q tensorboard_plugin_profile

python -m src.train_triplet_loss