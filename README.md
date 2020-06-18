Pytorch-Video-Classification

Make video classification on UCF101 using RESNET pretrained and GRU(RNN) with Pytorch framework.

**Machine Windows 10

How to Set the Environment 

1) conda create -n [Name]

2) activate [Name] 

3) conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

4) conda install pytorch-cpu torchvision-cpu -c pytorch

5) pip install pandas scikit-learn tqdm opencv-python


##prepare datasets (video to images)

Download the UCF 101 datasets https://www.crcv.ucf.edu/data/UCF101/UCF101.rar 

extract the rar files. and rename the path to UCF-101 to UCF 

put the 6.5 GB files to ./data folder 

run cmd 

Write python make_train_test.py and run the command. Grab a Coffee it will take 1-2 hours. 

## train the datasets

RUN  python train.py 