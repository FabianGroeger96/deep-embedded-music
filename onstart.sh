apt-get -y update && apt-get -y upgrade
apt-get -y install git
apt-get -y install zip
apt-get -y install unzip
apt-get -y install libsndfile1
apt -y install git
pip install --upgrade pip
pip install -r requirements.txt

python -m src.train_triplet_loss