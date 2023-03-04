if [ -e interface/datasets/coco/annotations ] 
then
    echo "Already setup"
else
    wget https://ia600904.us.archive.org/21/items/MSCoco2014/annotations_trainval2014.zip
    mkdir -p interface/datasets/coco
    mv annotations_trainval2014.zip interface/datasets/coco
    cd interface/datasets/coco
    unzip annotations_trainval2014.zip
fi

#npm install pad-left
#node convert-flir.mj

pip3 install torch torchvision torchaudio --upgrade

pip3 install -qr https://raw.githubusercontent.com/hustvl/YOLOP/main/requirements.txt
pip3 install -qr https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt  # install dependencies

if [ -e data/attention-data ] 
then
    cd data 
    git clone git@github.com:irh-ca-team-car/attention-data
    cd ..
fi