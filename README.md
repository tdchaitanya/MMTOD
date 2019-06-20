# Borrow from Anywhere: Pseudo Multimodal Object Detection in Thermal Domain.

![Framework](./images/framework.png)

This repository is a modified fork of [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch)

We make use of [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/) and [mingyuliutw/UNIT](https://github.com/mingyuliutw/UNIT) for training CycleGAN and  UNIT models. Trained Thermal-to-RGB model weights are provided for both UNIT and CycleGAN, so you don't have to train them from scratch.

## Preparation:

Clone the repository: 
```
https://github.com/tdchaitanya/MMTOD.git

```
Create a folder:

```
mkdir data models
```

### prerequisites

* Python 3.6
* Pytorch 1.0
* CUDA 8.0 or higher

### Data Preparation

* **FLIR ADAS**: Download the FLIR ADAS dataset from [here](https://www.flir.in/oem/adas/adas-dataset-form/) and arrange it in PASCAL-VOC format. 

```
data
├── coco
├── pretrained_model --> resnet101_caffe.pth
└── VOCdevkit2007
    ├── VOC2007
        ├── Annotations
        ├── ImageSets
        │   └── Main --> trainval.txt, test.txt
        └── JPEGImages
```

FLIR annoations are in COCO format, to convert them into PASCAL VOC format use the scripts in `generate_annotations` folder

* **KAIST**: KAIST dataset can be downloaded [here](https://sites.google.com/site/pedestrianbenchmark/). As mentioned for FLIR dataset, KAIST dataset should also be arranged in PASCAL-VOC format.

KAIST annotations are in `txt` format, to convert them into PASCAL VOC format use the scripts in `generate_annotations` folder.

`trainval.txt` and `test.txt` for FLIR, FLIR-(1/2) and FLIR-(1/4), KAIST datasets will be provided in the google drive folder linked below.

### Pretrained Model: 
We'll be using pretrained Resnet-101 model for the Faster-RCNN base. You can download the weights from:  

* ResNet101: [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0), [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth)

Download them and put them into the data/pretrained_model/ directory.

### Compilation

Install all the python dependencies using pip:
```
pip install -r requirements.txt
```

Compile the cuda dependencies using following simple commands:

```
cd lib
python setup.py build develop
```

**As pointed out in this [issue](https://github.com/jwyang/faster-rcnn.pytorch/issues/16), if you encounter some error during the compilation, you might miss to export the CUDA paths to your environment.**

## Train

Pick the `trainval.txt` for FLIR dataset along with the common `test.txt` and place them in `./data/VOCdevkit2007/VOC2007/ImageSets/Main/` folder. `trainval.txt` is different for each of FLIR, FLIR-1/2 and FLIR-1/4.

**Single mode Faster-RCNN on FLIR ADAS:**

Exceute the following command to start training: 

```
python trainval_net.py --dataset pascal_voc --net res101_thermal --bs 8 --nw 4 --epochs 15 --cuda --use_tfb 
```

**MMTOD-UNIT with MS-COCO as the RGB branch:**

For Thermal-to-RGB translation you need the RGB to Thermal UNIT weigths, these can be downloaded from `unit/models` folder in google drive . Place these weigths in   `lib/model/unit/models` Run the following command to see the results: 

Along with `rgb2thermal.pt` you'll also need VGG16 weights, dowload it from this [link](https://www.dropbox.com/s/76l3rt4kyi3s8x7/vgg16.t7?dl=1) and place it in same folder.

Since we initialise the RGB and Thermal branches with pre-trained weights you'll need the pre-trained weigths for both the branches. 

Pre-trained weigths for RGB branch can be found in the `MS-COCO/res101_coco` and pre-trained weigths for thermal branch cane be found in `FLIR/res101_thermal` in this [drive folder](https://drive.google.com/drive/folders/1Hz6h3WS_rX6wvaGr4duC3ctd90R_l0eS?usp=sharing). Place `res101_coco` and `res101_thermal` folders in  the `models` directory. 

To start the training run the following command: 

```
python trainval_unit_update_coco.py --dataset pascal_voc --net res101_unit_update_coco --bs 1 --nw 4 --epochs 15 --cuda
```

**MMTOD-UNIT with PASCAL-VOC as the RGB branch:**

As mentioned above, you need the RGB-to-Thermal and other weights, follow the instructions mentioned for the `MS-COCO` case above and place the weight files in appropriate folder. 

Pre-trained weigths for RGB branch can be found in the `PASCAL-VOC/res101_pascal` and pre-trained weigths for thermal branch can be found in `FLIR/res101_thermal` in this [drive folder](https://drive.google.com/drive/folders/1Hz6h3WS_rX6wvaGr4duC3ctd90R_l0eS?usp=sharing). Place `res101_pascal` and `res101_thermal` folders in  the `models` directory. 

To start the training run the following command: 

```
python trainval_unit_update.py --dataset pascal_voc --net res101_unit_update --bs 1 --nw 4 --epochs 15 --cuda --use_tfb
```

**MMTOD-CycleGAN with MS-COCO as the RGB branch:**

For Thermal-to-RGB translation you need the RGB to Thermal CycleGAN weights, these can be downloaded from `cgan/checkpoints/rg2thermal_flir` folder in [drive folder](https://drive.google.com/drive/folders/1Hz6h3WS_rX6wvaGr4duC3ctd90R_l0eS?usp=sharing) . Place these weigths in   `lib/model/cgan/checkpoints` Run the following command to see the results: 

Follow the instructions mentioned for MMTO-UNIT for downloading the pre-trained RGB (MS-COCO), Thermal weights.

To start the training execute the following command:

```
python trainval_cgan_update_coco.py --dataset pascal_voc --net res101_cgan_update_coco --bs 4 --nw 4 --epochs 15 --cuda --name rgb2thermal_flir --use_tfb
```
**MMTOD-CycleGAN with PASCAL-VOC as the RGB branch:**

As mentioned above, you need the RGB-to-Thermal and other weights, follow the instructions mentioned for the `MS-COCO` case above and place the weight files in appropriate folder. 

Follow the instructions mentioned for MMTO-UNIT for downloading the pre-trained RGB (PASCAL-VOC), Thermal weights.

To start the training execute the following command:


```
python trainval_cgan_update.py --dataset pascal_voc --net res101_cgan_update --bs 4 --nw 4 --epochs 15 --cuda --name rgb2thermal_flir --use_tfb
```

## Training on FLIR-1/2 and FLIR-1/4
For training on FLIR-1/2 and FLIR-1/4 you just need to change the `trainval.txt` file, replace it with the one from the corresponding folder in this [drive](https://drive.google.com/drive/folders/1Hz6h3WS_rX6wvaGr4duC3ctd90R_l0eS?usp=sharing) and use the same training commands, procedure. 


## Reproducing results in the paper.

All the weight files can be found in the google drive folder located [here](https://drive.google.com/drive/folders/1Hz6h3WS_rX6wvaGr4duC3ctd90R_l0eS?usp=sharing)


## Reproducing the results on FLIR thermal dataset(Table 1):

Pick the `trainval.txt` for FLIR dataset (full) along with the common `test.txt` and place them in `./data/VOCdevkit2007/VOC2007/ImageSets/Main/` folder

**Baseline:** Weights for the baseline are located in `res101_thermal` folder of the google drive. Place the folder as it is in `models`directory and run the following command: 

```
python test_net.py --dataset pascal_voc --net res101_thermal --checksession 1 --checkepoch 15 --checkpoint 1963 --cuda

```

**MMTOD-UNIT** 

1). **MS-COCO as RGB Branch**

Weights are located in `res101_unit_update_coco` folder

For Thermal-to-RGB translation you need the RGB to Thermal UNIT weigths, these can be downloaded from `unit/models` folder in google drive . Place these weigths in   `lib/model/unit/models` Run the following command to see the results: 

Along with `rgb2thermal.pt` you'll also need VGG16 weights, dowload it from this [link](https://www.dropbox.com/s/76l3rt4kyi3s8x7/vgg16.t7?dl=1) and place it in same folder.


```
python test_net_unit_update.py --dataset pascal_voc --net res101_unit_update_coco --checksession 1 --checkepoch 15 --checkpoint 15717 --cuda 
```

2). **PASCAL-VOC as RGB Branch**

Weights are located in `res101_unit_update` folder. 

As mentioned for the MS-COCO above, make sure to downlaod the Thermal-to-RGB weight files and place them in the appropriate directory.

```
python test_net_unit_update.py --dataset pascal_voc --net res101_unit_update --checksession 1 --checkepoch 15 --checkpoint 15717 --cuda 
```

**MMTOD-CGAN**

1). **MS-COCO as the RGB Branch:**

Weights are located in `res101_cgan_update_coco` folder. 

For Thermal-to-RGB translation you need the RGB to Thermal CycleGAN weights, these can be downloaded from `cgan/checkpoints/rg2thermal_flir` folder in google drive . Place these weigths in   `lib/model/cgan/checkpoints` Run the following command to see the results: 

```
python test_net_cgan_update.py --dataset pascal_voc --net res101_cgan_update_coco --checksession 1 --checkepoch 15 --checkpoint 3928 --cuda --name rgb2thermal_flir
```

2). **PASCAL-VOC as RGB Branch**

Weights are located in `res101_cgan_update` folder. 

As mentioned for the MS-COCO above, make sure to download the Thermal-to-RGB weight files and place them in the appropriate directory.

```
python test_net_cgan_update.py --dataset pascal_voc --net res101_cgan_update --checksession 1 --checkepoch 15 --checkpoint 3928 --cuda --name rgb2thermal_flir

```
