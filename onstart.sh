apt-get -y install git
apt-get -y install zip
apt-get -y install unzip
apt-get -y install libsndfile1
apt-get -y install ffmpeg
apt -y install git
pip install --upgrade pip
pip install -r requirements.txt

!pip uninstall -y -q tensorflow tensorboard
!pip uninstall -y -q tensorflow tensorboard
!pip install -q -U -q tf-nightly tb-nightly tensorboard_plugin_profile

python -m src.train_triplet_loss