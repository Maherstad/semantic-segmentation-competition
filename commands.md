conda create --name venv 
source activate venv
sudo apt-get update
sudo apt-get install gdal-bin
sudo apt-get install zip
sudo apt install git-all



pip install -r requirements.txt


mkdir raw_dataset
mv test.zip raw_dataset
mv train.zip raw_dataset
cd raw_dataset
unzip test.zip
unzip train.zip
unzip metadata.zip
cd ..
rm test.zip
rm train.zip
rm metadata.zip 
